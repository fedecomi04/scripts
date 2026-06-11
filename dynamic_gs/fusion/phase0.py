"""Pure-function lift of Phase 0a (SAM3 + Fast-SAM3D generation) and
Phase 0b (CPD/TEASER++ fusion) from the legacy pipeline.

Lifted verbatim from ``dynamic_gs_pipeline.py`` (methods
``_run_sam3_and_sam3d_generation``, ``_fuse_sam3d_objects_into_scene``,
``_backproject_mask_to_world``, ``_save_sam3_debug_plots``) and
refactored so every ``self.*`` access becomes an explicit parameter.
The pipeline now calls these as thin shims; the new static-gs pipeline
calls them directly.

Note on FoundationPose: the original Phase 0b also constructed an FP
tracker per instance and stashed it on ``self._fp_trackers_by_instance``.
FP is dead in the rewrite — XFeat + LighterGlue is the only tracker —
so no FP construction happens here. Per-instance manifest entries no
longer carry ``mesh_path`` / ``mesh_to_world_4x4`` fields either.

The post-fusion cache save is NOT done here. Callers decide whether
and where to persist (static-gs always saves; dynamic-gs conditionally
on the live + feedforward flags via the legacy shim).
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nerfstudio.utils.rich_utils import CONSOLE

from ..utils import (
    load_sam3d_gaussian_ply,
    load_sam3d_rotation_wxyz,
    register_and_fuse_sam3d_object,
)
from ..utils.sam3_segmentation import load_sam3_masks, run_sam3_subprocess
from ..utils.fastsam_segmentation import run_fastsam_subprocess
from ..utils import timing_ledger as _tl
from ..utils.sam3d import (
    get_sam3d_output_paths,
    resolve_sam3d_pose_path,
    run_sam3d_multi_object_subprocess,
    sam3d_pose_has_rotation,
)


# ============================================================================
# Helpers (module-level pure functions, lifted from pipeline)
# ============================================================================


def backproject_mask_to_world(
    mask_bool_np: np.ndarray,
    depth_image: torch.Tensor,
    rgb_image: torch.Tensor,
    camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an image-plane mask through a depth image into world 3D points.

    Args:
        mask_bool_np: (H_mask, W_mask) boolean array on CPU.
        depth_image: (H, W) depth in meters (CPU tensor).
        rgb_image: (H, W, 3) uint8 or float RGB (CPU or GPU tensor).
        camera: ``Cameras`` with at least one element (we use index 0).

    Returns:
        ``(points_np, colors_np)`` where ``points_np`` is ``(N, 3)`` float32
        in world coordinates and ``colors_np`` is ``(N, 3)`` float32 in [0, 1].
        Points with missing/zero depth are filtered out.
    """
    H, W = int(depth_image.shape[0]), int(depth_image.shape[1])

    if mask_bool_np.shape != (H, W):
        mask_resized = np.array(
            Image.fromarray(mask_bool_np.astype(np.uint8) * 255).resize((W, H), Image.NEAREST),
            dtype=np.uint8,
        ) > 127
    else:
        mask_resized = mask_bool_np

    depth_np = depth_image.detach().cpu().numpy().astype(np.float32)

    if hasattr(rgb_image, "detach"):
        rgb_cpu = rgb_image.detach().cpu()
    else:
        rgb_cpu = rgb_image
    rgb_np = rgb_cpu.numpy() if hasattr(rgb_cpu, "numpy") else np.asarray(rgb_cpu)
    if rgb_np.dtype == np.uint8:
        rgb_np = rgb_np.astype(np.float32) / 255.0
    else:
        rgb_np = rgb_np.astype(np.float32)
    if rgb_np.shape[:2] != (H, W):
        rgb_np = np.array(
            Image.fromarray((rgb_np * 255).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    ys, xs = np.where(mask_resized & (depth_np > 1e-4))
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    z = depth_np[ys, xs]

    # MAD-based depth outlier scrub. Mask-boundary pixels frequently hit
    # the background/table behind the object (silhouette-edge depth bleed);
    # without this scrub CPD's similarity fit is pulled toward a smaller
    # scale and a shifted centroid. 5.0 × 1.4826 ≈ 7.4 MAD is intentionally
    # permissive so legitimate object depth extent is preserved.
    if z.size >= 10:
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med))) + 1e-6
        keep = np.abs(z - med) < 5.0 * 1.4826 * mad
        if keep.sum() >= 3:
            ys = ys[keep]
            xs = xs[keep]
            z = z[keep]

    fx = float(camera.fx[0].item())
    fy = float(camera.fy[0].item())
    cx = float(camera.cx[0].item())
    cy = float(camera.cy[0].item())
    c2w = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)

    src_H = int(camera.height[0].item()) if hasattr(camera.height[0], "item") else int(camera.height[0])
    src_W = int(camera.width[0].item()) if hasattr(camera.width[0], "item") else int(camera.width[0])
    if (H, W) != (src_H, src_W):
        sx = W / float(src_W)
        sy = H / float(src_H)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    # Back-project in Nerfstudio/OpenGL camera frame (x right, y up, z back).
    x_cam = (xs.astype(np.float32) - cx) / fx * z
    y_cam = -(ys.astype(np.float32) - cy) / fy * z
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = pts_cam @ R.T + t[None, :]
    colors = rgb_np[ys, xs]
    return pts_world.astype(np.float32), colors.astype(np.float32)


def cull_points_in_front(
    points_world: np.ndarray,
    target_points_world: np.ndarray,
    camera,
    render_hw: tuple[int, int],
    band_m: float = 0.0,
    radius_px: int = 2,
) -> np.ndarray:
    """Boolean keep-mask: drop ``points_world`` that lie in FRONT of the trusted
    real surface from the camera viewpoint (between the camera and the surface).

    Builds a front-surface depth buffer by projecting ``target_points_world``
    (the back-projected real/GT depth) into the image, then removes any inserted
    point whose forward-depth is closer than that surface by more than ``band_m``.
    Points with no surface along their ray (outside the silhouette) are kept.
    Inverse of :func:`backproject_mask_to_world` (Nerfstudio/OpenGL camera frame).
    Mirrors the tuned cull in scripts/experiments/nonrigid_bench/.
    """
    from scipy.ndimage import minimum_filter

    H, W = int(render_hw[0]), int(render_hw[1])
    fx = float(camera.fx[0].item())
    fy = float(camera.fy[0].item())
    cx = float(camera.cx[0].item())
    cy = float(camera.cy[0].item())
    c2w = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float64)
    src_H = int(camera.height[0].item()) if hasattr(camera.height[0], "item") else int(camera.height[0])
    src_W = int(camera.width[0].item()) if hasattr(camera.width[0], "item") else int(camera.width[0])
    if (H, W) != (src_H, src_W):
        fx *= W / float(src_W)
        fy *= H / float(src_H)
        cx *= W / float(src_W)
        cy *= H / float(src_H)
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    def _project(pts: np.ndarray):
        cam = (pts.astype(np.float64) - t) @ R
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx + fx * cam[:, 0] / z
            v = cy - fy * cam[:, 1] / z
        return u, v, z

    tu, tv, tz = _project(target_points_world)
    tui, tvi = np.round(tu).astype(int), np.round(tv).astype(int)
    tvalid = np.isfinite(tz) & (tz > 0) & (tui >= 0) & (tui < W) & (tvi >= 0) & (tvi < H)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    np.minimum.at(depth_buf, (tvi[tvalid], tui[tvalid]), tz[tvalid])
    if radius_px > 0:
        depth_buf = minimum_filter(depth_buf, size=2 * int(radius_px) + 1)

    u, v, z = _project(points_world)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    in_img = np.isfinite(z) & (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    d_surf = np.full(points_world.shape[0], np.inf, dtype=np.float64)
    d_surf[in_img] = depth_buf[vi[in_img], ui[in_img]]
    has_surface = np.isfinite(d_surf)
    in_front = has_surface & (z < d_surf - float(band_m))
    return ~in_front


def save_sam3_debug_plots(
    rgb_path: Path,
    sam3_objects: list,
    out_dir: Path,
    prefix: str = "static0",
) -> None:
    """Overview + per-object overlay PNGs for SAM3 segmentation review."""
    rgb_pil = Image.open(rgb_path).convert("RGB")
    rgb_np = np.array(rgb_pil, dtype=np.uint8)
    H, W = rgb_np.shape[:2]

    palette = np.array(
        [
            (255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 128, 0),
            (200, 0, 200), (0, 200, 200), (255, 255, 0), (128, 64, 255),
        ],
        dtype=np.uint8,
    )

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    overview = rgb_np.astype(np.float32)
    for i, obj in enumerate(sam3_objects):
        mask_path = obj.get("mask_path")
        if not mask_path:
            continue
        m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if m.shape != (H, W):
            m = np.array(
                Image.fromarray(m).resize((W, H), Image.NEAREST),
                dtype=np.uint8,
            )
        mask_bool = m > 127
        color = palette[i % len(palette)].astype(np.float32)
        alpha = 0.5
        overview[mask_bool] = overview[mask_bool] * (1 - alpha) + color * alpha

    overview_img = Image.fromarray(overview.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overview_img)
    for i, obj in enumerate(sam3_objects):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        color = tuple(int(c) for c in palette[i % len(palette)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"#{i} s={obj.get('score', 0.0):.2f}"
        text_xy = (x0 + 2, max(0, y0 - 22))
        try:
            tb = draw.textbbox(text_xy, label, font=font)
            draw.rectangle(tb, fill=(0, 0, 0))
        except Exception:
            pass
        draw.text(text_xy, label, fill=color, font=font)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_img.save(out_dir / f"{prefix}_sam3_overview.png")

    for i, obj in enumerate(sam3_objects):
        mask_path = obj.get("mask_path")
        if not mask_path:
            continue
        m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if m.shape != (H, W):
            m = np.array(
                Image.fromarray(m).resize((W, H), Image.NEAREST),
                dtype=np.uint8,
            )
        mask_bool = m > 127
        per_img = rgb_np.astype(np.float32).copy()
        per_img[mask_bool] = per_img[mask_bool] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
        per_pil = Image.fromarray(per_img.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(per_pil)
        bbox = obj.get("bbox")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
            label = f"obj_{i:02d} score={obj.get('score', 0.0):.3f} area={obj.get('mask_area', 0)}"
            text_xy = (x0 + 2, max(0, y0 - 22))
            try:
                tb = draw.textbbox(text_xy, label, font=font)
                draw.rectangle(tb, fill=(0, 0, 0))
            except Exception:
                pass
            draw.text(text_xy, label, fill=(255, 255, 0), font=font)
        per_pil.save(out_dir / f"{prefix}_obj_{i:02d}_overlay.png")

    CONSOLE.log(
        f"[phase-0] saved SAM3 debug plots: {prefix}_sam3_overview.png + "
        f"{len(sam3_objects)} per-object overlays in {out_dir}"
    )


# ============================================================================
# Phase 0a — SAM3 segmentation + Fast-SAM3D 3D object generation
# ============================================================================


def run_phase0a_sam3_and_sam3d(
    *,
    model,
    datamanager,
    timing: Optional[dict] = None,
) -> Optional[dict]:
    """Pre-static: run SAM3 mask discovery + Fast-SAM3D 3D object generation.

    Writes per-object Gaussian PLYs and pose sidecars under
    ``initialization_artifacts/`` but does NOT mutate the Gaussian scene.
    Fusion (insertion + instance-id propagation) happens later via
    :func:`run_phase0b_fusion` at the static→dynamic transition (in
    legacy dynamic-gs) or at end-of-training (in static-gs).

    Args:
        model: SplatfactoModel-derived model whose ``config`` carries the
            ``sam3_*`` and ``sam3d_*`` knobs. Moved to CPU around the
            SAM3D subprocess to free GPU memory on small cards.
        datamanager: ``DynamicGSDataManager`` exposing ``static_manager``,
            ``dynamic_manager``, ``get_initialization_debug_dir()``, and
            ``get_initialization_artifact_dir()``.
        timing: optional dict accumulating per-step durations (keys
            ``S0.1_sam3_segmentation``, ``S0.2_sam3d_multi_generation``,
            ``S0.4a_generation_total``). Pass ``None`` to skip timing.

    Returns:
        ``{"sam3_objects": [...], "sam3d_results": [...]}`` for the caller
        to stash and pass to :func:`run_phase0b_fusion` later, or ``None``
        if SAM3 found 0 candidate objects.
    """
    if timing is None:
        timing = {}

    t_total = time.time()
    model_cfg = model.config
    debug_dir = datamanager.get_initialization_debug_dir()
    artifact_dir = datamanager.get_initialization_artifact_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Segment the LAST static frame, not the first. The operator sweeps the
    # arm camera TOWARD the object and ends centered on it, so the final
    # keyframe has the closest, most head-on view — the best mask for SAM3 +
    # the best input for SAM3D (a far/glancing first-frame view yields a thin
    # sliver mask that can collapse SAM3D's crop below its 2x2 minimum).
    # Mirrors adaptive_downsample, which already uses the last camera pose as
    # its reference viewpoint. ``frame_idx_0`` (kept name for the downstream
    # depth/intrinsics lookups) is read from the chosen batch's own
    # ``image_idx`` so everything stays consistent.
    batch = datamanager.static_manager.cached_train[-1]
    static_image = batch["image"]

    static_ds = datamanager.static_manager.train_dataset
    depth_filenames = static_ds.metadata.get("depth_filenames")
    depth_scale = float(static_ds.metadata.get("depth_unit_scale_factor", 1.0))
    frame_idx_0 = int(batch.get("image_idx", len(datamanager.static_manager.cached_train) - 1))
    static_depth_m = None
    if depth_filenames is not None and frame_idx_0 < len(depth_filenames):
        try:
            depth_pil = Image.open(Path(depth_filenames[frame_idx_0]))
            depth_np = np.array(depth_pil).astype(np.float32) * depth_scale
            static_depth_m = torch.from_numpy(depth_np)
        except Exception as exc:
            CONSOLE.log(f"[phase-0] warning: failed to load static depth ({exc})")

    static_intrinsics = {
        "fx": float(static_ds.cameras.fx[frame_idx_0].item()),
        "fy": float(static_ds.cameras.fy[frame_idx_0].item()),
        "cx": float(static_ds.cameras.cx[frame_idx_0].item()),
        "cy": float(static_ds.cameras.cy[frame_idx_0].item()),
    }

    static_image_path = debug_dir / "static0_rgb.png"
    img_cpu = static_image.cpu()
    if img_cpu.dtype == torch.uint8:
        static_np = img_cpu.numpy()
    else:
        static_np = (img_cpu.numpy() * 255).clip(0, 255).astype(np.uint8)
    gripper_mask_t = batch.get("mask")
    if gripper_mask_t is not None:
        m = gripper_mask_t.detach().cpu()
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        keep = (m > 0.5).numpy()
        if keep.shape != static_np.shape[:2]:
            resized = Image.fromarray(keep.astype(np.uint8) * 255).resize(
                (static_np.shape[1], static_np.shape[0]), Image.NEAREST
            )
            keep = np.array(resized) > 127
        static_np = static_np.copy()
        static_np[~keep] = 0
    Image.fromarray(static_np).save(static_image_path)

    results_json = debug_dir / "static0_sam3_results.json"
    sam3_cached = model_cfg.sam3_reuse_cached and results_json.exists()
    # Invalidate the cache when the configured backend differs from the one that
    # produced it (the summary tags "segmentation_backend"; SAM3's legacy JSON
    # omits it → treated as "sam3"). Prevents a SAM3 cache from being served to
    # a FastSAM run and vice versa after a backend switch.
    if sam3_cached:
        try:
            _cached_backend = json.loads(results_json.read_text()).get("segmentation_backend", "sam3")
        except Exception:
            _cached_backend = "sam3"
        if _cached_backend != getattr(model_cfg, "segmentation_backend", "sam3"):
            CONSOLE.log(
                f"[phase-0] cached SAM3 results are from backend '{_cached_backend}' but config "
                f"requests '{getattr(model_cfg, 'segmentation_backend', 'sam3')}' — re-running segmentation"
            )
            sam3_cached = False

    run_device = torch.device(model.means.device)
    if not sam3_cached:
        model.to("cpu")
        for mgr in [datamanager.static_manager, datamanager.dynamic_manager]:
            if hasattr(mgr, "cached_train"):
                for batch_iter in mgr.cached_train:
                    for k in list(batch_iter.keys()):
                        if hasattr(batch_iter[k], "cpu") and hasattr(batch_iter[k], "device"):
                            if str(batch_iter[k].device).startswith("cuda"):
                                batch_iter[k] = batch_iter[k].cpu()
            if hasattr(mgr, "cached_eval"):
                for batch_iter in mgr.cached_eval:
                    for k in list(batch_iter.keys()):
                        if hasattr(batch_iter[k], "cpu") and hasattr(batch_iter[k], "device"):
                            if str(batch_iter[k].device).startswith("cuda"):
                                batch_iter[k] = batch_iter[k].cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        t_sam3 = time.time()
        if sam3_cached:
            sam3_objects = load_sam3_masks(results_json)
            CONSOLE.log(f"[phase-0] reusing cached SAM3 results: {len(sam3_objects)} objects")
        else:
            backend = getattr(model_cfg, "segmentation_backend", "sam3")
            if backend == "fastsam":
                CONSOLE.log("[phase-0] segmentation backend = FastSAM+CLIP")
                sam3_objects = run_fastsam_subprocess(
                    image_path=static_image_path,
                    text_prompt=model_cfg.sam3_prompt_text,
                    output_dir=debug_dir,
                    output_stem="static0",
                    sam3_conda_env=model_cfg.sam3_conda_env_name,
                    min_area_ratio=model_cfg.sam3_candidate_min_area_ratio,
                    max_area_ratio=model_cfg.sam3_candidate_max_area_ratio,
                    dedup_iou=model_cfg.sam3_candidate_dedup_iou,
                    max_objects=model_cfg.sam3_candidate_max_objects,
                    min_score=model_cfg.sam3_min_score,
                    fastsam_conf=getattr(model_cfg, "fastsam_conf", 0.4),
                    fastsam_iou=getattr(model_cfg, "fastsam_iou", 0.9),
                    clip_model=getattr(model_cfg, "fastsam_clip_model", "ViT-B-32-quickgelu"),
                    clip_pretrained=getattr(model_cfg, "fastsam_clip_pretrained", "openai"),
                    fastsam_weights=getattr(model_cfg, "fastsam_weights", "FastSAM-x.pt"),
                )
            else:
                sam3_objects = run_sam3_subprocess(
                    image_path=static_image_path,
                    text_prompt=model_cfg.sam3_prompt_text,
                    output_dir=debug_dir,
                    output_stem="static0",
                    sam3_conda_env=model_cfg.sam3_conda_env_name,
                    min_area_ratio=model_cfg.sam3_candidate_min_area_ratio,
                    max_area_ratio=model_cfg.sam3_candidate_max_area_ratio,
                    dedup_iou=model_cfg.sam3_candidate_dedup_iou,
                    max_objects=model_cfg.sam3_candidate_max_objects,
                    confidence_threshold=model_cfg.sam3_confidence_threshold,
                    min_score=model_cfg.sam3_min_score,
                )
        timing.setdefault("S0.1_sam3_segmentation", []).append(time.time() - t_sam3)

        # In live mode the real SAM3 subprocess ran in a separate helper
        # before the pipeline was constructed; the inline timer above only
        # measures the cached re-read. Pull the true subprocess wall-clock
        # from the live-session sidecar when present.
        try:
            live_timings_path = artifact_dir / "live_sam3_timings.json"
            if live_timings_path.is_file():
                live_timings = json.loads(live_timings_path.read_text())
                if "S0.1_sam3_segmentation" in live_timings:
                    timing["S0.1_sam3_segmentation"] = [
                        float(live_timings["S0.1_sam3_segmentation"]),
                    ]
        except Exception as _exc:
            CONSOLE.log(f"[phase-0] could not read live SAM3 timing sidecar: {_exc}")

        # --- timing ledger: segmentation load/infer split (recorded path) ---
        _data_root = Path(getattr(datamanager.config, "data", debug_dir.parent.parent))
        if not sam3_cached:
            _seg_name = "FastSAM" if backend == "fastsam" else "SAM3"
            _sc = debug_dir / ("_fastsam_timing.json" if backend == "fastsam" else "_sam3_timing.json")
            try:
                if _sc.is_file():
                    _d = json.loads(_sc.read_text())
                    _ld, _inf = float(_d.get("load", 0.0)), float(_d.get("infer", 0.0))
                    _tl.record(_data_root, "segmentation", _seg_name, "load", t_sam3, t_sam3 + _ld)
                    _tl.record(_data_root, "segmentation", _seg_name, "infer", t_sam3 + _ld, t_sam3 + _ld + _inf)
                else:
                    # no split sidecar (e.g. SAM3) — record the whole call as infer
                    _tl.record(_data_root, "segmentation", _seg_name, "infer",
                               t_sam3, t_sam3 + float(timing["S0.1_sam3_segmentation"][-1]))
            except Exception:
                pass

        if not sam3_objects:
            CONSOLE.log("[phase-0] SAM3 found 0 objects; skipping Phase 0 prefusion")
            return None

        CONSOLE.log(f"[phase-0] SAM3 discovered {len(sam3_objects)} objects")
        save_sam3_debug_plots(
            rgb_path=static_image_path,
            sam3_objects=sam3_objects,
            out_dir=debug_dir,
            prefix="static0",
        )

        # SAM3D multi-object generation. Full image + metric pointmap from
        # the static depth, one model load + per-mask sequential inference.
        t_sam3d = time.time()
        output_stems = [f"static0_obj_{i:02d}_sam3d" for i in range(len(sam3_objects))]
        sam3d_results: list = [None] * len(sam3_objects)

        to_run_indices: list[int] = []
        to_run_mask_paths: list[Path] = []
        to_run_stems: list[str] = []
        for obj_i, sam3_obj in enumerate(sam3_objects):
            stem = output_stems[obj_i]
            paths = get_sam3d_output_paths(artifact_dir, stem, image_dir=debug_dir)
            pose_path_resolved = resolve_sam3d_pose_path(paths["ply_path"], paths["pose_path"])
            if model_cfg.sam3_reuse_cached and paths["ply_path"].exists() and sam3d_pose_has_rotation(pose_path_resolved):
                if pose_path_resolved is not None:
                    paths["pose_path"] = pose_path_resolved
                sam3d_results[obj_i] = paths
                CONSOLE.log(f"[phase-0] object {obj_i}: reusing cached SAM3D output")
            else:
                to_run_indices.append(obj_i)
                to_run_mask_paths.append(Path(sam3_obj["mask_path"]))
                to_run_stems.append(stem)

        full_depth_path = None
        full_intrinsics_path = None
        if static_depth_m is not None and to_run_indices:
            H_img, W_img = int(static_np.shape[0]), int(static_np.shape[1])
            full_depth_path = artifact_dir / "static0_full_depth_meters.tiff"
            Image.fromarray(static_depth_m.cpu().numpy().astype(np.float32)).save(full_depth_path)
            full_intrinsics_path = artifact_dir / "static0_full_intrinsics.json"
            full_intrinsics_path.write_text(
                json.dumps(
                    {
                        **static_intrinsics,
                        "width": W_img,
                        "height": H_img,
                    },
                    indent=2,
                )
                + "\n"
            )

        if to_run_indices:
            try:
                multi_results = run_sam3d_multi_object_subprocess(
                    render_image_path=static_image_path,
                    object_mask_paths=to_run_mask_paths,
                    output_dir=artifact_dir,
                    output_stems=to_run_stems,
                    image_dir=debug_dir,
                    max_side=518,
                    depth_path=full_depth_path,
                    intrinsics_path=full_intrinsics_path,
                )
            except Exception as exc:
                CONSOLE.log(f"[phase-0] SAM3D multi-object subprocess failed: {exc}")
                multi_results = [{} for _ in to_run_indices]

            for idx, result in zip(to_run_indices, multi_results):
                if result:
                    sam3d_results[idx] = result
                    CONSOLE.log(f"[phase-0] object {idx}: SAM3D generation complete")
                else:
                    sam3d_results[idx] = {}
                    CONSOLE.log(f"[phase-0] object {idx}: SAM3D failed (empty result)")

        sam3d_results = [r if r else {} for r in sam3d_results]
        timing.setdefault("S0.2_sam3d_multi_generation", []).append(time.time() - t_sam3d)
        # --- timing ledger: SAM3D import/load/infer split (from the sidecar) ---
        try:
            _sd_sc = artifact_dir / "_sam3d_timing.json"
            if _sd_sc.is_file() and _sd_sc.stat().st_mtime >= t_sam3d:
                _d = json.loads(_sd_sc.read_text())
                _imp = float(_d.get("import_config", 0.0))
                _mld = float(_d.get("model_load", 0.0))
                _inf = float(_d.get("infer_total", 0.0))
                _b = t_sam3d
                _tl.record(_data_root, "object_3d_gen", "SAM3D import", "load", _b, _b + _imp); _b += _imp
                _tl.record(_data_root, "object_3d_gen", "SAM3D model", "load", _b, _b + _mld); _b += _mld
                _tl.record(_data_root, "object_3d_gen", "SAM3D", "infer", _b, _b + _inf)
        except Exception:
            pass
    finally:
        if not sam3_cached:
            model.to(run_device)
            for mgr in [datamanager.static_manager, datamanager.dynamic_manager]:
                if hasattr(mgr, "cached_train"):
                    for batch_iter in mgr.cached_train:
                        for k in list(batch_iter.keys()):
                            if hasattr(batch_iter[k], "to") and hasattr(batch_iter[k], "device"):
                                if not str(batch_iter[k].device).startswith("cuda"):
                                    batch_iter[k] = batch_iter[k].to(run_device)
                if hasattr(mgr, "cached_eval"):
                    for batch_iter in mgr.cached_eval:
                        for k in list(batch_iter.keys()):
                            if hasattr(batch_iter[k], "to") and hasattr(batch_iter[k], "device"):
                                if not str(batch_iter[k].device).startswith("cuda"):
                                    batch_iter[k] = batch_iter[k].to(run_device)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    timing.setdefault("S0.4a_generation_total", []).append(time.time() - t_total)
    CONSOLE.log(
        f"[phase-0] generation complete: {len(sam3_objects)} masks, "
        f"{sum(1 for r in sam3d_results if r)} SAM3D PLYs ready; "
        f"fusion deferred to caller"
    )
    return {
        "sam3_objects": sam3_objects,
        "sam3d_results": sam3d_results,
    }


# ============================================================================
# Phase 0b — register + insert + propagate instance IDs
# ============================================================================


def run_phase0b_fusion(
    *,
    model,
    datamanager,
    generation_outputs: dict,
    device,
    timing: Optional[dict] = None,
) -> dict:
    """Post-static: take the pre-generated SAM3D PLYs and fuse them into
    the trained scene. Runs at end-of-static (static-gs) or at the
    static→dynamic transition (legacy dynamic-gs).

    After static optimization the rendered depth used by
    ``_get_existing_object_subset`` is meaningful (vs being garbage at
    init time when only SfM seeds exist), so instance-id propagation
    gets a denser set of seed Gaussians. SAM3D's back-side Gaussians
    also never see static photometric optimization, so they're not
    opacity-eroded before downstream tracking.

    Args:
        model: SplatfactoModel-derived model. Must expose
            ``_get_existing_object_subset``, ``_get_object_mask_slab_indices``,
            ``_estimate_spacing``, ``insert_object_gaussians``,
            ``object_instance_ids``, ``means``, and ``camera_optimizer``.
        datamanager: provides ``static_manager.cached_train[0]``,
            ``.train_dataset.cameras``, and the two
            ``get_initialization_*_dir`` methods.
        generation_outputs: ``{"sam3_objects", "sam3d_results"}``
            returned by :func:`run_phase0a_sam3_and_sam3d`; pass ``{}``
            to skip fusion (returns empty manifest).
        device: torch device the fusion camera should live on (typically
            ``model.device``).
        timing: optional dict accumulating per-object + total durations.

    Returns:
        Manifest dict ``{instance_id: {...stats...}}`` describing what
        was fused per object. Also written to
        ``initialization_artifacts/phase0_manifest.json``.
    """
    if timing is None:
        timing = {}

    t_total = time.time()
    sam3_objects = generation_outputs.get("sam3_objects") or []
    sam3d_results = generation_outputs.get("sam3d_results") or []
    if not sam3_objects:
        return {}

    debug_dir = datamanager.get_initialization_debug_dir()
    artifact_dir = datamanager.get_initialization_artifact_dir()

    # MUST match the frame Phase 0a segmented (the LAST static frame) — Phase 0b
    # reprojects the SAM3D-generated object against this camera, so a mismatch
    # would misalign the fusion. See the note in run_phase0a_sam3_and_sam3d.
    batch = datamanager.static_manager.cached_train[-1]
    static_image = batch["image"]
    frame_idx_0 = int(batch.get("image_idx", len(datamanager.static_manager.cached_train) - 1))
    camera = datamanager.static_manager.train_dataset.cameras[
        frame_idx_0 : frame_idx_0 + 1
    ].to(device)

    # Apply the post-static camera-optimizer offset to the fusion camera so
    # ``get_outputs``, the back-projection, and ``register_and_fuse_sam3d_object``
    # all see the optimized pose.
    camera.metadata = dict(camera.metadata or {})
    camera.metadata["cam_idx"] = frame_idx_0
    if (
        model.camera_optimizer.config.mode != "off"
        and 0 <= frame_idx_0 < model.camera_optimizer.num_cameras
    ):
        optimized_c2w = model.camera_optimizer.apply_to_camera(camera).detach()
        if optimized_c2w.shape == camera.camera_to_worlds.shape:
            camera.camera_to_worlds = optimized_c2w
            CONSOLE.log(
                f"[phase-0] using post-static optimized pose for cam_idx={frame_idx_0}"
            )

    static_ds = datamanager.static_manager.train_dataset
    depth_filenames = static_ds.metadata.get("depth_filenames")
    depth_scale = float(static_ds.metadata.get("depth_unit_scale_factor", 1.0))
    static_depth_m = None
    if depth_filenames is not None and frame_idx_0 < len(depth_filenames):
        try:
            depth_pil = Image.open(Path(depth_filenames[frame_idx_0]))
            depth_np = np.array(depth_pil).astype(np.float32) * depth_scale
            static_depth_m = torch.from_numpy(depth_np)
        except Exception as exc:
            CONSOLE.log(f"[phase-0] warning: failed to load static depth ({exc}); falling back to SfM targets")

    manifest: dict = {}
    n_objs = len(sam3_objects)
    print(
        f"\n==> [phase-0b] fusing {n_objs} SAM3D object(s) into the scene. "
        f"Per-object CPD/TEASER++ registration + insertion + flag propagation. "
        f"Total can be several minutes; per-object progress lines follow.\n",
        flush=True,
    )
    for obj_idx, (sam3_obj, sam3d_out) in enumerate(zip(sam3_objects, sam3d_results)):
        instance_id = obj_idx + 1
        print(f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: starting", flush=True)
        if not sam3d_out:
            CONSOLE.log(f"[phase-0] skipping object {obj_idx}: SAM3D failed or empty")
            continue

        t_fusion = time.time()

        # Re-render every iteration: ``insert_object_gaussians`` mutates
        # the Gaussian count, so ``model.info`` from a prior render goes
        # stale and ``extract_projected_centers_and_radii`` would raise
        # on a length mismatch.
        with torch.no_grad():
            outputs = model.get_outputs(camera)
        render_h, render_w = outputs["rgb"].shape[:2]

        ply_path = sam3d_out["ply_path"]
        pose_path = sam3d_out["pose_path"]
        try:
            source_points, source_colors = load_sam3d_gaussian_ply(ply_path)
            source_rotation_wxyz = load_sam3d_rotation_wxyz(pose_path)
        except Exception as exc:
            CONSOLE.log(f"[phase-0] skipping object {obj_idx}: {exc}")
            continue

        obj_mask_np = np.array(Image.open(sam3_obj["mask_path"]).convert("L"))
        obj_mask_tensor = torch.from_numpy((obj_mask_np > 127).astype(np.float32))
        obj_mask_tensor = obj_mask_tensor[..., None].to(device)
        if obj_mask_tensor.shape[0] != render_h or obj_mask_tensor.shape[1] != render_w:
            obj_mask_tensor = torch.nn.functional.interpolate(
                obj_mask_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(render_h, render_w),
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)

        existing_indices, existing_means, existing_colors = model._get_existing_object_subset(
            obj_mask_tensor,
            outputs["depth"],
        )
        existing_means_np = existing_means.detach().cpu().numpy()
        existing_colors_np = existing_colors.detach().cpu().numpy()

        # Dense registration target via back-projection through the static
        # depth (Gazebo GT). Falls back to SfM seeds when depth missing.
        target_points_np = existing_means_np
        target_colors_np = existing_colors_np
        if static_depth_m is not None:
            target_points_np, target_colors_np = backproject_mask_to_world(
                obj_mask_tensor.squeeze(-1).cpu().numpy() > 0.5,
                static_depth_m,
                static_image,
                camera,
            )
        if target_points_np.shape[0] < 3:
            CONSOLE.log(
                f"[phase-0] skipping object {obj_idx}: only {target_points_np.shape[0]} target points for registration"
            )
            continue

        c2w_rotation = camera.camera_to_worlds[0, :3, :3].detach().cpu().numpy().astype(np.float32)

        backend = model.config.sam3d_registration_backend
        t_cpd = time.time()
        print(
            f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: running {backend.upper()} registration "
            f"(source={len(source_points)} pts, target={len(target_points_np)} pts)",
            flush=True,
        )
        insertion_result = register_and_fuse_sam3d_object(
            source_points=source_points,
            source_colors=source_colors,
            target_points=target_points_np,
            target_colors=target_colors_np,
            source_rotation_wxyz=source_rotation_wxyz,
            camera_to_world_rotation=c2w_rotation,
            debug_dir=debug_dir,
            artifact_dir=artifact_dir,
            output_stem=f"static0_obj_{obj_idx:02d}_sam3d",
            registration_backend=backend,
            teaser_params={
                "noise_bound": model.config.sam3d_teaser_noise_bound,
                "max_correspondences": model.config.sam3d_teaser_max_correspondences,
                "normal_radius_mult": model.config.sam3d_teaser_fpfh_normal_radius_mult,
                "feature_radius_mult": model.config.sam3d_teaser_fpfh_feature_radius_mult,
                "color_weight": model.config.sam3d_teaser_color_weight,
                "fpfh_max_nn": model.config.sam3d_teaser_fpfh_max_nn,
                "normal_max_nn": model.config.sam3d_teaser_normal_max_nn,
                "enable_reproject": model.config.sam3d_teaser_enable_reproject,
                "reproject_max_corr_mult": model.config.sam3d_teaser_reproject_max_corr_mult,
                "reproject_noise_bound": model.config.sam3d_teaser_reproject_noise_bound,
                "enable_post_icp": model.config.sam3d_teaser_enable_post_icp,
                "post_icp_max_corr_mult": model.config.sam3d_teaser_post_icp_max_corr_mult,
                "post_icp_iterations": model.config.sam3d_teaser_post_icp_iterations,
            },
        )
        print(
            f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: {backend.upper()} done in {time.time() - t_cpd:.1f}s "
            f"(kept {insertion_result.kept_point_count} pts)",
            flush=True,
        )
        # Surface the TEASER multi-stage breakdown (the generic timing report
        # only records the S0.3 per-object total; the per-stage detail lives in
        # insertion_result.timing and is otherwise dropped on the Phase-0b path).
        if backend == "teaser":
            _it = insertion_result.timing or {}
            _tm = _it.get("D0.3b3_teaser_meta") or {}
            _rm = _it.get("D0.3b3_reproject_meta")
            _im = _it.get("D0.3b3_icp_meta")
            _parts = [
                f"fpfh_corr={_tm.get('fpfh_correspondences', '?')} "
                f"used={_tm.get('used_correspondences', '?')} "
                f"scale={_tm.get('scale', float('nan')):.3f} "
                f"({_it.get('D0.3b3_refinement', 0.0):.2f}s)"
            ]
            if _rm is not None:
                _parts.append(
                    f"reproject geom_corr={_rm.get('geom_correspondences', '?')} "
                    f"used={_rm.get('used_correspondences', '?')} "
                    f"delta_scale={_rm.get('delta_scale', float('nan')):.3f} "
                    f"({_it.get('D0.3b3_reproject', 0.0):.2f}s)"
                )
            if _im is not None:
                _parts.append(
                    f"icp fitness={_im.get('fitness', float('nan')):.3f} "
                    f"rmse={_im.get('inlier_rmse', float('nan')) * 1000.0:.2f}mm "
                    f"({_it.get('D0.3b3_icp', 0.0):.2f}s)"
                )
            print(f"    [phase-0b] TEASER stages: " + " | ".join(_parts), flush=True)

        # Cull / flag tunables — see legacy pipeline comments for what
        # each one does. Bumping CULL_STRENGTH or TAU_FLOOR_M removes more
        # SAM3D points on the camera-visible side; bumping CULL_DEPTH_TOL_M
        # makes E denser so tau is estimated on a richer set.
        CULL_STRENGTH = 1.3
        TAU_FLOOR_M = 0.003
        CULL_DEPTH_TOL_M = 0.015
        FLAG_DEPTH_TOL_M = 0.02

        e_indices_cull = model._get_object_mask_slab_indices(
            obj_mask_tensor, outputs["depth"], depth_tol_m=CULL_DEPTH_TOL_M
        )
        e_indices_flag = model._get_object_mask_slab_indices(
            obj_mask_tensor, outputs["depth"], depth_tol_m=FLAG_DEPTH_TOL_M
        )

        cull_pts_np = insertion_result.kept_points.astype(np.float32)
        cull_colors_np = insertion_result.kept_colors.astype(np.float32)
        n_culled_sam3d = 0
        tau = 0.0
        if cull_pts_np.shape[0] > 0 and e_indices_cull.numel() >= 2:
            e_pts_np = (
                model.means[e_indices_cull].detach().cpu().numpy().astype(np.float32)
            )
            tau = max(
                model._estimate_spacing(e_pts_np) * CULL_STRENGTH,
                TAU_FLOOR_M,
            )

            from sklearn.neighbors import NearestNeighbors as _CullNN

            e_nn = _CullNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(e_pts_np)
            sam3d_d, _ = e_nn.kneighbors(cull_pts_np)
            keep_mask = ~(np.isfinite(sam3d_d[:, 0]) & (sam3d_d[:, 0] <= tau))
            n_culled_sam3d = int((~keep_mask).sum())
            cull_pts_np = cull_pts_np[keep_mask]
            cull_colors_np = cull_colors_np[keep_mask]

        # In-front (occlusion) cull: drop any surviving inserted point closer to
        # the camera than the trusted real front surface (the back-projected
        # target). band=0 => remove everything strictly in front; the occluded
        # back (incl. thin parts) is kept. Complements the proximity de-dup above.
        IN_FRONT_BAND_M = 0.0
        n_culled_in_front = 0
        if cull_pts_np.shape[0] > 0 and target_points_np.shape[0] >= 3:
            keep_front = cull_points_in_front(
                cull_pts_np, target_points_np, camera, (render_h, render_w),
                band_m=IN_FRONT_BAND_M, radius_px=2,
            )
            n_culled_in_front = int((~keep_front).sum())
            cull_pts_np = cull_pts_np[keep_front]
            cull_colors_np = cull_colors_np[keep_front]

        if cull_pts_np.shape[0] > 0:
            inserted_indices = model.insert_object_gaussians(
                torch.from_numpy(cull_pts_np),
                torch.from_numpy(cull_colors_np),
                object_flag=False,
                instance_id=instance_id,
            )
        else:
            inserted_indices = torch.zeros((0,), dtype=torch.long, device=model.means.device)

        MAX_RADIUS_M = 0.02
        n_flagged_existing = 0
        match_indices = torch.zeros((0,), dtype=torch.long, device=model.means.device)
        if e_indices_flag.numel() > 0 and insertion_result.kept_point_count > 0:
            mean_device = model.means.device
            instance_ids_flat = model.object_instance_ids.squeeze(-1)
            slab_owners = instance_ids_flat[e_indices_flag]
            eligible_mask = (slab_owners == 0) | (slab_owners == instance_id)
            candidate_indices = e_indices_flag.to(device=mean_device)[eligible_mask]

            if candidate_indices.numel() > 0:
                from sklearn.neighbors import NearestNeighbors as _MatchNN

                candidate_pts_np = (
                    model.means[candidate_indices].detach().cpu().numpy().astype(np.float32)
                )

                proxy_points_np = insertion_result.kept_points.astype(np.float32)
                proxy_spacing = model._estimate_spacing(proxy_points_np)
                proxy_radius = min(MAX_RADIUS_M, max(0.003, 1.5 * proxy_spacing))
                proxy_nn = _MatchNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(proxy_points_np)
                proxy_d, _ = proxy_nn.kneighbors(candidate_pts_np)
                near_proxy_np = np.isfinite(proxy_d[:, 0]) & (proxy_d[:, 0] <= proxy_radius)

                target_pts_np = existing_means_np.astype(np.float32)
                near_target_np = np.zeros((len(candidate_pts_np),), dtype=bool)
                if len(target_pts_np) > 0:
                    target_spacing = model._estimate_spacing(target_pts_np)
                    target_radius = min(MAX_RADIUS_M, max(0.002, 6.0 * target_spacing))
                    target_nn = _MatchNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(target_pts_np)
                    target_d, _ = target_nn.kneighbors(candidate_pts_np)
                    near_target_np = np.isfinite(target_d[:, 0]) & (target_d[:, 0] <= target_radius)

                match_mask_np = near_proxy_np | near_target_np
                if match_mask_np.any():
                    match_indices = candidate_indices[
                        torch.from_numpy(match_mask_np).to(device=mean_device)
                    ]
                    model.object_instance_ids[match_indices] = instance_id
                    n_flagged_existing = int(match_indices.numel())

        obj_total = time.time() - t_fusion
        timing.setdefault(f"S0.3_fusion_obj_{obj_idx}", []).append(obj_total)
        print(
            f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: done in {obj_total:.1f}s",
            flush=True,
        )

        instance_count = int(
            (model.object_instance_ids.squeeze(-1) == instance_id).sum().item()
        )

        manifest[instance_id] = {
            "object_index": obj_idx,
            "mask_path": str(sam3_obj["mask_path"]),
            "ply_path": str(ply_path),
            "score": sam3_obj.get("score", 0.0),
            "existing_gaussians": int(existing_indices.numel()),
            "sam3d_pre_cull_count": int(insertion_result.kept_point_count),
            "sam3d_culled": int(n_culled_sam3d),
            "sam3d_cull_rate": (
                float(n_culled_sam3d) / float(insertion_result.kept_point_count)
                if insertion_result.kept_point_count > 0
                else 0.0
            ),
            "cull_tau_m": float(tau),
            "sam3d_culled_in_front": int(n_culled_in_front),
            "registration_backend": str(backend),
            "inserted_gaussians": int(inserted_indices.numel()),
            "flagged_existing_gaussians": int(n_flagged_existing),
            "instance_count": instance_count,
            "kept_points": insertion_result.kept_point_count,
            "chosen_scale": insertion_result.chosen_scale,
            "source_spacing": float(insertion_result.source_spacing),
        }
        CONSOLE.log(
            f"[phase-0] object {obj_idx} (instance_id={instance_id}): "
            f"existing={existing_indices.numel()}, "
            f"sam3d={insertion_result.kept_point_count}->{inserted_indices.numel()} "
            f"(proximity_culled={n_culled_sam3d} tau={tau * 1000:.1f}mm, in_front_culled={n_culled_in_front}), "
            f"flagged_existing={n_flagged_existing}, "
            f"instance_total={instance_count}, "
            f"scale={insertion_result.chosen_scale:.4f}"
        )

    manifest_path = artifact_dir / "phase0_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    fusion_time = time.time() - t_total
    timing.setdefault("S0.4b_fusion_total", []).append(fusion_time)
    try:
        _tl.record(Path(getattr(datamanager.config, "data", debug_dir.parent.parent)),
                   "object_fusion", "NDP register+fuse", "fusion", t_total, time.time(),
                   n=len(manifest))
    except Exception:
        pass
    num_prefused = int((model.object_instance_ids > 0).any(dim=-1).sum().item())
    CONSOLE.log(
        f"[phase-0] fusion complete: {len(manifest)} objects fused, "
        f"{num_prefused} Gaussians with instance IDs, "
        f"fusion time={fusion_time:.2f}s"
    )
    return manifest
