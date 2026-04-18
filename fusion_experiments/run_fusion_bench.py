#!/usr/bin/env python3
"""Offline SAM3D fusion benchmark with lots of debug images.

This script intentionally runs *outside* the training loop:
- Loads a saved static-phase checkpoint once
- Reuses the cached D0 artifacts (render/masks + SAM3D raw output)
- Runs multiple registration variants and saves:
  - alignment point clouds (PLY)
  - before/after renders
  - point projections overlaid on the rendered image
  - summary metrics (JSON + CSV)

It also benchmarks the current in-pipeline implementation (baseline).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import torch

from nerfstudio.utils.eval_utils import eval_setup

from dynamic_gs.utils import sam3d_fusion as sf
from dynamic_gs.utils.sam3d_fusion import Sam3DInsertionResult, load_sam3d_gaussian_ply, save_point_cloud


try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    from probreg import cpd
except ImportError:  # pragma: no cover
    cpd = None


def _require_open3d():
    if o3d is None:
        raise ImportError("open3d is required for this benchmark.")
    return o3d


def _load_binary_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    if rgb.ndim == 4:
        rgb = rgb[0]
    arr = rgb.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _overlay_mask(rgb_u8: np.ndarray, mask_u8: np.ndarray, color=(255, 0, 0), alpha=0.35) -> np.ndarray:
    out = rgb_u8.astype(np.float32)
    mask = mask_u8 > 0
    if mask.ndim == 3:
        mask = mask[..., 0] > 0
    if np.any(mask):
        out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _abs_diff(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    a = a_u8.astype(np.int16)
    b = b_u8.astype(np.int16)
    return np.clip(np.abs(a - b), 0, 255).astype(np.uint8)


def _load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    o3d_mod = _require_open3d()
    pcd = o3d_mod.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = None
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
    return pts, cols


def _nn_stats(source: np.ndarray, target: np.ndarray, max_points: int = 50_000) -> dict[str, float]:
    o3d_mod = _require_open3d()
    if len(source) == 0 or len(target) == 0:
        return {"mean": float("nan"), "median": float("nan")}
    if len(source) > max_points:
        rng = np.random.default_rng(42)
        source = source[rng.choice(len(source), size=max_points, replace=False)]
    if len(target) > max_points:
        rng = np.random.default_rng(43)
        target = target[rng.choice(len(target), size=max_points, replace=False)]
    src = o3d_mod.geometry.PointCloud()
    src.points = o3d_mod.utility.Vector3dVector(source.astype(np.float64))
    tgt = o3d_mod.geometry.PointCloud()
    tgt.points = o3d_mod.utility.Vector3dVector(target.astype(np.float64))
    d = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float32)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(d)), "median": float(np.median(d))}


def _draw_projected_points(
    rgb_u8: np.ndarray,
    points_world: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    color=(0, 255, 255),
    radius: int = 1,
    max_points: int = 40_000,
) -> np.ndarray:
    if len(points_world) == 0:
        return rgb_u8
    points = points_world
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    pixels, depths, valid = sf._project_points(points, viewmat, intrinsics, width, height)
    valid_idx = np.flatnonzero(valid & np.isfinite(depths))
    if len(valid_idx) == 0:
        return rgb_u8
    img = Image.fromarray(rgb_u8)
    draw = ImageDraw.Draw(img)
    for idx in valid_idx:
        u, v = int(pixels[idx, 0]), int(pixels[idx, 1])
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
    return np.array(img)


def _make_contact_sheet(paths: list[Path], labels: list[str], out_path: Path, cols: int = 2) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x0, y0 = c * w, r * h
        sheet.paste(im.resize((w, h), resample=Image.BILINEAR), (x0, y0))
        draw.text((x0 + 8, y0 + 8), labels[i], fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


@dataclass
class GaussianSnapshot:
    num_points: int
    gauss_params: dict[str, torch.Tensor]
    object_flags: torch.Tensor
    current_active_mask: torch.Tensor
    sam3d_init_target_flags: torch.Tensor
    persistent_ready: bool

    @staticmethod
    def capture(model) -> "GaussianSnapshot":
        names = ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]
        gauss_params = {name: model.gauss_params[name].detach().clone() for name in names}
        return GaussianSnapshot(
            num_points=int(model.num_points),
            gauss_params=gauss_params,
            object_flags=model.object_flags.detach().clone(),
            current_active_mask=model.current_active_mask.detach().clone(),
            sam3d_init_target_flags=model.sam3d_init_target_flags.detach().clone(),
            persistent_ready=bool(getattr(model, "_persistent_object_membership_ready", False)),
        )

    def restore(self, model) -> None:
        for name, tensor in self.gauss_params.items():
            model.gauss_params[name] = torch.nn.Parameter(tensor.detach().clone())
        model._resize_dynamic_buffers(self.num_points)
        model.object_flags.copy_(self.object_flags)
        model.current_active_mask.copy_(self.current_active_mask)
        model.sam3d_init_target_flags.copy_(self.sam3d_init_target_flags)
        model._persistent_object_membership_ready = bool(self.persistent_ready)


def _strategy_baseline_current(
    *,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    render_object_mask: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    debug_dir: Path,
    output_stem: str,
) -> Sam3DInsertionResult:
    return sf.register_and_fuse_sam3d_object(
        source_points=source_points,
        source_colors=source_colors,
        target_points=target_points,
        target_colors=target_colors,
        render_object_mask=render_object_mask,
        viewmat=viewmat,
        intrinsics=intrinsics,
        width=width,
        height=height,
        debug_dir=debug_dir,
        output_stem=output_stem,
    )


def _make_transform_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (float(scale) * np.asarray(rotation, dtype=np.float32)).astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _strategy_fgr_probreg_icp(
    *,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    render_object_mask: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    debug_dir: Path,
    output_stem: str,
) -> dict[str, Any]:
    if cpd is None:
        return {"status": "skipped", "reason": "probreg not installed"}
    if len(source_points) == 0 or len(target_points) < 3:
        return {"status": "rejected", "reason": "empty source/target"}

    # Match baseline preprocessing: isotropic scale from bbox diagonals + centroid translate.
    source_diag = sf._bbox_diagonal(source_points)
    target_diag = sf._bbox_diagonal(target_points)
    base_scale = target_diag / max(source_diag, 1e-6)
    source_centroid = sf._centroid(source_points)
    target_centroid = sf._centroid(target_points)
    scaled_source = target_centroid[None, :] + base_scale * (source_points - source_centroid[None, :])
    scaled_source_colors = source_colors.astype(np.float32)

    target_spacing = sf._median_nn_distance(target_points)
    source_spacing = sf._median_nn_distance(scaled_source)
    voxel_size = max(1.5 * max(target_spacing, source_spacing), 1e-3)
    source_down_points, source_down_colors = sf._voxel_downsample(scaled_source, scaled_source_colors, voxel_size)
    target_down_points, target_down_colors = sf._voxel_downsample(target_points, target_colors, voxel_size)

    fgr_result, _ = sf._run_fgr(
        source_down_points,
        source_down_colors,
        target_down_points,
        target_down_colors,
        voxel_size,
    )
    fgr_transform = np.asarray(fgr_result.transformation, dtype=np.float32)

    fgr_aligned = sf._transform_points(source_down_points, fgr_transform)
    visible_idx = sf._visible_source_indices(
        fgr_aligned,
        render_object_mask,
        viewmat,
        intrinsics,
        width,
        height,
    )
    icp_source_points = source_down_points
    icp_source_colors = source_down_colors
    if len(visible_idx) >= 32:
        icp_source_points = source_down_points[visible_idx]
        icp_source_colors = source_down_colors[visible_idx]

    source_init = sf._transform_points(icp_source_points, fgr_transform)

    probreg_voxel = max(2.0 * voxel_size, 1e-3)
    source_probreg, _ = sf._voxel_downsample(source_init, icp_source_colors, probreg_voxel)
    target_probreg, _ = sf._voxel_downsample(target_down_points, target_down_colors, probreg_voxel)

    result = cpd.registration_cpd(
        source_probreg,
        target_probreg,
        tf_type_name="rigid",
        update_scale=True,
        maxiter=80,
        tol=1e-6,
        w=0.0,
    )
    probreg_tf = result.transformation
    probreg_transform = _make_transform_matrix(probreg_tf.scale, probreg_tf.rot, probreg_tf.t)
    init_transform = probreg_transform @ fgr_transform

    icp_result, _ = sf._run_icp(
        icp_source_points,
        icp_source_colors,
        target_down_points,
        target_down_colors,
        init_transform,
        voxel_size,
    )
    icp_transform = np.asarray(icp_result.transformation, dtype=np.float32)

    aligned_points = sf._transform_points(scaled_source, icp_transform)
    aligned_colors = scaled_source_colors.astype(np.float32)
    final_scale = float(base_scale * sf._extract_isotropic_scale(icp_transform))

    dedup_threshold = 1.5 * target_spacing
    target_pcd = sf._to_pcd(target_points, target_colors)
    distances = np.asarray(sf._to_pcd(aligned_points).compute_point_cloud_distance(target_pcd), dtype=np.float32)
    keep_mask = np.isfinite(distances) & (distances >= dedup_threshold)
    kept_points = aligned_points[keep_mask].astype(np.float32)
    kept_colors = aligned_colors[keep_mask].astype(np.float32)

    # Minimal artifacts
    save_point_cloud(debug_dir / f"{output_stem}_probreg_aligned_output.ply", aligned_points, aligned_colors)
    save_point_cloud(debug_dir / f"{output_stem}_probreg_kept_points.ply", kept_points, kept_colors)

    return {
        "status": "ok",
        "chosen_scale": final_scale,
        "voxel_size": float(voxel_size),
        "dedup_threshold": float(dedup_threshold),
        "source_point_count": int(len(source_points)),
        "target_point_count": int(len(target_points)),
        "visible_source_point_count": int(len(visible_idx)),
        "registration_source_point_count": int(len(icp_source_points)),
        "kept_point_count": int(len(kept_points)),
        "fgr_transformation": fgr_transform,
        "probreg_transformation": probreg_transform,
        "icp_transformation": icp_transform,
        "icp_fitness": float(icp_result.fitness),
        "icp_rmse": float(icp_result.inlier_rmse),
        "aligned_points": aligned_points,
        "aligned_colors": aligned_colors,
        "kept_points": kept_points,
        "kept_colors": kept_colors,
    }


def _limit_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(123)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="all | baseline_current | fgr_probreg_icp",
    )
    parser.add_argument(
        "--max-insert-points",
        type=int,
        default=100_000,
        help="0 means insert all kept points (can be slow).",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle.expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    dataset_root = Path(manifest["dataset_root"])
    frame_name = manifest["frame_name"]
    dynamic_frame_idx = int(manifest["dynamic_frame_idx"])

    out_dir = args.output
    if out_dir is None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = dataset_root / "fusion_bench" / "runs" / f"{frame_name}_{stamp}"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint once
    _, pipeline, checkpoint_path, step = eval_setup(args.capture_config, test_mode="val")
    datamanager = pipeline.datamanager
    datamanager.set_phase("dynamic")
    datamanager.set_dynamic_frame_idx(dynamic_frame_idx)
    camera, batch = datamanager.get_current_dynamic_train_batch()

    model = pipeline.model
    base_snapshot = GaussianSnapshot.capture(model)

    # Reference images
    outputs_before = model.get_outputs(camera)
    before_u8 = _to_uint8_rgb(outputs_before["rgb"])
    _save_uint8(out_dir / "before_render.png", before_u8)
    live_path = bundle_dir / "live_rgb.png"
    live_u8 = np.array(Image.open(live_path).convert("RGB")) if live_path.exists() else None

    # Load cached bundle assets
    render_object_mask = _load_binary_mask(bundle_dir / "render_object_mask.png")
    viewmat = np.load(bundle_dir / "ref_viewmat.npy").astype(np.float32)
    intrinsics = np.load(bundle_dir / "ref_intrinsics.npy").astype(np.float32)
    width = int(manifest["ref_width"])
    height = int(manifest["ref_height"])

    source_points, source_colors = load_sam3d_gaussian_ply(bundle_dir / "sam3d_raw_output.ply")
    target_points, target_colors = _load_point_cloud(bundle_dir / "target_subset.ply")
    if target_colors is None or len(target_colors) != len(target_points):
        target_colors = np.full((len(target_points), 3), 0.5, dtype=np.float32)

    strategies = ["baseline_current", "fgr_probreg_icp"]
    if args.strategy != "all":
        if args.strategy not in strategies:
            raise ValueError(f"Unknown strategy {args.strategy}. Options: all, {', '.join(strategies)}")
        strategies = [args.strategy]

    rows: list[dict[str, Any]] = []

    for strategy in strategies:
        # Reset model to baseline state before each run
        base_snapshot.restore(model)

        strat_dir = out_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)

        if strategy == "baseline_current":
            result = _strategy_baseline_current(
                source_points=source_points,
                source_colors=source_colors,
                target_points=target_points,
                target_colors=target_colors,
                render_object_mask=render_object_mask.astype(bool),
                viewmat=viewmat,
                intrinsics=intrinsics,
                width=width,
                height=height,
                debug_dir=strat_dir,
                output_stem=f"{frame_name}_{strategy}",
            )
            kept_points = result.kept_points
            kept_colors = result.kept_colors
            aligned_points = result.aligned_points
            icp_fitness = result.icp_fitness
            chosen_scale = result.chosen_scale
            status = "ok"
            extra: dict[str, Any] = {
                "icp_rmse": result.icp_rmse,
                "kept_point_count": result.kept_point_count,
                "visible_source_point_count": result.visible_source_point_count,
                "registration_source_point_count": result.registration_source_point_count,
            }
        elif strategy == "fgr_probreg_icp":
            probreg_out = _strategy_fgr_probreg_icp(
                source_points=source_points,
                source_colors=source_colors,
                target_points=target_points,
                target_colors=target_colors,
                render_object_mask=render_object_mask.astype(bool),
                viewmat=viewmat,
                intrinsics=intrinsics,
                width=width,
                height=height,
                debug_dir=strat_dir,
                output_stem=f"{frame_name}_{strategy}",
            )
            status = probreg_out["status"]
            if status != "ok":
                (strat_dir / "metrics.json").write_text(json.dumps(probreg_out, indent=2) + "\n")
                rows.append({"strategy": strategy, "status": status, "reason": probreg_out.get("reason", "")})
                continue
            kept_points = probreg_out["kept_points"]
            kept_colors = probreg_out["kept_colors"]
            aligned_points = probreg_out["aligned_points"]
            icp_fitness = float(probreg_out["icp_fitness"])
            chosen_scale = float(probreg_out["chosen_scale"])
            extra = {
                "icp_rmse": float(probreg_out["icp_rmse"]),
                "kept_point_count": int(probreg_out["kept_point_count"]),
                "visible_source_point_count": int(probreg_out["visible_source_point_count"]),
                "registration_source_point_count": int(probreg_out["registration_source_point_count"]),
            }
        else:  # pragma: no cover
            raise AssertionError(strategy)

        # Save alignment PLYs (full aligned + kept)
        save_point_cloud(strat_dir / "aligned_points.ply", aligned_points, None)
        save_point_cloud(strat_dir / "kept_points.ply", kept_points, None)

        # Render after insertion for qualitative judgement
        insert_points, insert_colors = _limit_points(kept_points, kept_colors, args.max_insert_points)
        inserted_idx = model.insert_object_gaussians(
            torch.from_numpy(insert_points),
            torch.from_numpy(insert_colors),
            object_flag=True,
        )
        del inserted_idx

        outputs_after = model.get_outputs(camera)
        after_u8 = _to_uint8_rgb(outputs_after["rgb"])
        _save_uint8(strat_dir / "after_render.png", after_u8)

        # Save object mask render (post insertion)
        try:
            obj_mask = model.render_object_mask(camera)
            obj_mask_np = obj_mask.detach().float().cpu().numpy()
            if obj_mask_np.ndim == 3:
                obj_mask_np = obj_mask_np[..., 0]
            _save_uint8(strat_dir / "after_object_mask.png", (obj_mask_np > 0.5).astype(np.uint8) * 255)
        except Exception:
            pass

        # Debug overlays
        after_overlay = _overlay_mask(after_u8, render_object_mask, color=(255, 0, 0), alpha=0.35)
        _save_uint8(strat_dir / "after_overlay_render_object_mask.png", after_overlay)
        proj_kept = _draw_projected_points(after_overlay, kept_points, viewmat, intrinsics, width, height, color=(0, 255, 255))
        _save_uint8(strat_dir / "after_overlay_projected_kept_points.png", proj_kept)
        _save_uint8(strat_dir / "before_after_absdiff.png", _abs_diff(before_u8, after_u8))

        if live_u8 is not None:
            _save_uint8(strat_dir / "live_rgb.png", live_u8)
            _save_uint8(strat_dir / "after_live_absdiff.png", _abs_diff(after_u8, live_u8))

        # Contact sheet
        contact_paths = [out_dir / "before_render.png", strat_dir / "after_render.png"]
        contact_labels = ["before", f"after ({strategy})"]
        if live_u8 is not None:
            contact_paths.append(strat_dir / "live_rgb.png")
            contact_labels.append("live")
            contact_paths.append(strat_dir / "after_live_absdiff.png")
            contact_labels.append("|after-live|")
        _make_contact_sheet(contact_paths, contact_labels, strat_dir / "contact_sheet.png", cols=2)

        # Metrics
        nn_aligned_to_target = _nn_stats(aligned_points, target_points)
        nn_kept_to_target = _nn_stats(kept_points, target_points)
        metrics = {
            "strategy": strategy,
            "status": status,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": int(step),
            "frame_name": frame_name,
            "chosen_scale": float(chosen_scale),
            "icp_fitness": float(icp_fitness),
            "nn_aligned_to_target": nn_aligned_to_target,
            "nn_kept_to_target": nn_kept_to_target,
            "kept_points": int(len(kept_points)),
            "inserted_points": int(len(insert_points)),
            "max_insert_points": int(args.max_insert_points),
            **extra,
        }
        (strat_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        rows.append(
            {
                "strategy": strategy,
                "status": status,
                "chosen_scale": float(chosen_scale),
                "icp_fitness": float(icp_fitness),
                "nn_aligned_median": float(nn_aligned_to_target["median"]),
                "nn_kept_median": float(nn_kept_to_target["median"]),
                "kept_points": int(len(kept_points)),
                "inserted_points": int(len(insert_points)),
            }
        )

    # Write summary
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "timestamp": dt.datetime.now().isoformat(),
                "dataset_root": str(dataset_root),
                "bundle_dir": str(bundle_dir),
                "capture_config": str(args.capture_config.resolve()),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": int(step),
                "frame_name": frame_name,
                "strategies": strategies,
                "summary_csv": str(summary_path),
            },
            indent=2,
        )
        + "\n"
    )

    print(out_dir)


if __name__ == "__main__":
    main()

