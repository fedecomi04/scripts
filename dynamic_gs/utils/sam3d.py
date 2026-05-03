from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np
from PIL import Image
import torch

SAM3D_REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sam-3d-objects"
SAM3D_CONFIG_PATH = SAM3D_REPO_ROOT / "checkpoints" / "hf" / "pipeline.yaml"
SAM3D_RUNTIME_CONFIG_PATH = SAM3D_REPO_ROOT / "checkpoints" / "hf" / "pipeline_runtime_small.yaml"


def get_sam3d_output_paths(
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    return {
        "ply_path": output_dir / f"{output_stem}_raw_output.ply",
        "pose_path": output_dir / f"{output_stem}_pose.json",
        "preview_path": image_dir / f"{output_stem}_preview.png",
        "run_info_path": output_dir / f"{output_stem}_run_info.txt",
        "glb_path": output_dir / f"{output_stem}_mesh.glb",
        "mesh_ply_path": output_dir / f"{output_stem}_mesh.ply",
    }


def resolve_sam3d_pose_path(raw_ply_path: Path, fallback_pose_path: Path | None = None) -> Path | None:
    raw_ply_path = Path(raw_ply_path)
    candidates: list[Path] = []
    if raw_ply_path.name.endswith("_raw_output.ply"):
        candidates.append(raw_ply_path.with_name(raw_ply_path.name[: -len("_raw_output.ply")] + "_pose.json"))
    if fallback_pose_path is not None:
        candidates.append(Path(fallback_pose_path))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def load_sam3d_pose(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    pose: Dict[str, np.ndarray] = {}
    for key in ("translation", "rotation", "scale"):
        value = payload.get(key)
        if value is None:
            continue
        pose[key] = np.asarray(value, dtype=np.float32).reshape(-1)
    return pose


def sam3d_pose_has_rotation(path: Path | None) -> bool:
    if path is None or not Path(path).exists():
        return False
    try:
        pose = load_sam3d_pose(Path(path))
    except Exception:
        return False
    rotation = pose.get("rotation")
    return rotation is not None and rotation.size == 4 and np.isfinite(rotation).all()


def _load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, resample=Image.NEAREST)
    return (np.array(mask_image) > 127).astype(np.uint8)


def _install_kaolin_stub() -> None:
    if "kaolin" in sys.modules:
        return

    kaolin = ModuleType("kaolin")
    visualize = ModuleType("kaolin.visualize")
    render = ModuleType("kaolin.render")
    camera = ModuleType("kaolin.render.camera")
    utils = ModuleType("kaolin.utils")
    testing = ModuleType("kaolin.utils.testing")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    visualize.IpyTurntableVisualizer = _Dummy
    camera.Camera = _Dummy
    camera.CameraExtrinsics = _Dummy
    camera.PinholeIntrinsics = _Dummy
    testing.check_tensor = lambda *args, **kwargs: True
    render.camera = camera
    utils.testing = testing
    kaolin.visualize = visualize
    kaolin.render = render
    kaolin.utils = utils

    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.visualize"] = visualize
    sys.modules["kaolin.render"] = render
    sys.modules["kaolin.render.camera"] = camera
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing


def _import_official_api():
    notebook_dir = SAM3D_REPO_ROOT / "notebook"
    _install_kaolin_stub()
    for path in (str(SAM3D_REPO_ROOT), str(notebook_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ["LIDRA_SKIP_INIT"] = "true"
    from inference import Inference  # type: ignore

    return Inference


def _write_runtime_config() -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(SAM3D_CONFIG_PATH)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.dtype = "float16"
    config.depth_model.device = "cpu"
    # SAM3D mesh decoder: FoundationPose needs a triangle mesh, but the
    # mesh decoder's 256^3 FlexiCubes grid (~740 MB) plus the rest of the
    # SAM3D pipeline pushes peak GPU usage past 8 GiB. The user's fork
    # already does aggressive CPU offload for the diffusion stack, but the
    # mesh decoder's grid is locked on GPU at __init__. Until a separate
    # mesh-only post-pass is implemented (or a larger GPU is available),
    # decode gaussian only. The FP tracker will warn and skip when no
    # mesh PLY is found in `phase0_manifest.json`.
    config.decode_formats = ["gaussian"]
    config.slat_decoder_mesh_config_path = None
    config.slat_decoder_mesh_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    OmegaConf.save(config, SAM3D_RUNTIME_CONFIG_PATH)
    return SAM3D_RUNTIME_CONFIG_PATH


def _save_preview(mask: np.ndarray, image_rgb: np.ndarray, preview_path: Path) -> None:
    overlay = image_rgb.copy().astype(np.float32)
    overlay[mask > 0] = 0.65 * overlay[mask > 0] + 0.35 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(preview_path)


def prepare_cropped_sam3d_inputs(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    padding: int = 32,
    min_crop_side: int = 300,
    depth_path: Path | None = None,
    depth_scale: float = 1.0,
    camera_intrinsics: dict | None = None,
) -> Dict[str, Path]:
    """Crop the SAM3D inputs tightly around the object mask for lighter inference.

    ``min_crop_side`` ensures the object doesn't fill the entire crop, which
    would cause SAM3D to generate an extremely dense sparse structure that
    OOMs on 8 GB GPUs.  A value of 300 keeps sparse coord counts manageable.

    When ``depth_path`` is given, also crop the metric depth image and write
    out an intrinsics sidecar (focal length + principal point shifted by the
    crop).  This lets the SAM3D subprocess build a metric-scale pointmap and
    feed it to the pipeline as ``pointmap=`` (bypassing MoGe's scale/shift
    invariant depth estimate).
    """

    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_pil = Image.open(render_image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    mask = _load_binary_mask(object_mask_path, image_pil.size)
    if int(mask.sum()) == 0:
        raise ValueError("SAM3D crop input mask is empty.")

    ys, xs = np.nonzero(mask > 0)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    side = max(y1 - y0, x1 - x0) + 2 * int(padding)
    side = max(side, 32, int(min_crop_side))

    crop_y0 = max(0, int(round(center_y - side / 2)))
    crop_x0 = max(0, int(round(center_x - side / 2)))
    crop_y1 = min(image_rgb.shape[0], crop_y0 + side)
    crop_x1 = min(image_rgb.shape[1], crop_x0 + side)
    crop_y0 = max(0, crop_y1 - side)
    crop_x0 = max(0, crop_x1 - side)

    cropped_image = image_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
    cropped_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

    cropped_render_path = image_dir / f"{output_stem}_crop_render.png"
    cropped_mask_path = image_dir / f"{output_stem}_crop_mask.png"
    Image.fromarray(cropped_image).save(cropped_render_path)
    Image.fromarray((cropped_mask > 0).astype(np.uint8) * 255).save(cropped_mask_path)

    result = {
        "render_image_path": cropped_render_path,
        "object_mask_path": cropped_mask_path,
    }

    # Optional: crop the metric depth image and write the crop-shifted intrinsics
    # sidecar, so the SAM3D worker can build a metric pointmap for this crop.
    if depth_path is not None and camera_intrinsics is not None:
        depth_path = Path(depth_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_raw = np.array(Image.open(depth_path))
        depth_m = depth_raw.astype(np.float32) * float(depth_scale)
        if depth_m.shape[:2] != image_rgb.shape[:2]:
            # Resize depth to match image resolution (nearest to preserve holes)
            depth_pil = Image.fromarray(depth_m)
            depth_pil = depth_pil.resize(
                (image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST
            )
            depth_m = np.array(depth_pil, dtype=np.float32)
        cropped_depth = depth_m[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32)

        cropped_depth_path = image_dir / f"{output_stem}_crop_depth.tiff"
        Image.fromarray(cropped_depth).save(cropped_depth_path)

        cropped_intrinsics = {
            "fx": float(camera_intrinsics["fx"]),
            "fy": float(camera_intrinsics["fy"]),
            # cx/cy shift because the crop origin is (crop_x0, crop_y0)
            "cx": float(camera_intrinsics["cx"]) - float(crop_x0),
            "cy": float(camera_intrinsics["cy"]) - float(crop_y0),
            "width": int(cropped_image.shape[1]),
            "height": int(cropped_image.shape[0]),
            "crop_origin": [int(crop_x0), int(crop_y0)],
            "orig_width": int(image_rgb.shape[1]),
            "orig_height": int(image_rgb.shape[0]),
        }
        cropped_intrinsics_path = image_dir / f"{output_stem}_crop_intrinsics.json"
        cropped_intrinsics_path.write_text(json.dumps(cropped_intrinsics, indent=2) + "\n")

        result["depth_path"] = cropped_depth_path
        result["intrinsics_path"] = cropped_intrinsics_path

    return result


def _resize_image_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    if max(height, width) <= max_side:
        return image_rgb, mask

    scale = max_side / float(max(height, width))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image_resized = np.array(Image.fromarray(image_rgb).resize(new_size, resample=Image.BILINEAR))
    mask_resized = np.array(
        Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(new_size, resample=Image.NEAREST)
    )
    return image_resized, (mask_resized > 127).astype(np.uint8)


def _build_pytorch3d_pointmap(
    depth_m: np.ndarray,
    intrinsics: dict,
) -> np.ndarray:
    """Build a ``(H, W, 3)`` pytorch3d-convention pointmap from metric depth + intrinsics.

    PyTorch3D camera convention (from SAM3D's ``camera_to_pytorch3d_camera``,
    which is ``look_at_view_transform(eye=[0,0,-1], at=[0,0,0], up=[0,-1,0])``):
    x points LEFT, y points UP, z points FORWARD.  Standard CV backprojection
    gives (x-right, y-down, z-forward), so we flip x and y.

    Pixels with invalid depth (``<=0``) are left as ``NaN`` so SAM3D's
    preprocessor (which calls ``_clip_pointmap``) can treat them as holes.
    """
    H, W = depth_m.shape[:2]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    z = depth_m.astype(np.float32)
    valid = z > 0.0

    # Standard CV backprojection with pytorch3d axis flips (x→-x, y→-y)
    x_cam = -(uu - cx) / fx * z
    y_cam = -(vv - cy) / fy * z
    z_cam = z

    pointmap = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)
    pointmap[~valid] = np.nan
    return pointmap


def run_sam3d_single_object(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    max_side: int = 518,
    depth_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> Dict[str, Path]:
    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_paths = get_sam3d_output_paths(output_dir, output_stem, image_dir=image_dir)
    ply_path = output_paths["ply_path"]
    glb_path = output_paths["glb_path"]
    pose_path = output_paths["pose_path"]
    preview_path = output_paths["preview_path"]
    run_info_path = output_paths["run_info_path"]

    if not render_image_path.exists():
        raise FileNotFoundError(render_image_path)
    if not object_mask_path.exists():
        raise FileNotFoundError(object_mask_path)
    if not SAM3D_REPO_ROOT.exists():
        raise FileNotFoundError(SAM3D_REPO_ROOT)
    if not SAM3D_CONFIG_PATH.exists():
        raise FileNotFoundError(SAM3D_CONFIG_PATH)

    image_pil = Image.open(render_image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    mask = _load_binary_mask(object_mask_path, image_pil.size)
    if image_rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"SAM3D image/mask shape mismatch: {image_rgb.shape} vs {mask.shape}")
    if int(mask.sum()) == 0:
        raise ValueError("SAM3D input mask is empty.")

    # Optional: load metric depth + intrinsics to build a pytorch3d pointmap.
    # When present, SAM3D uses this instead of its internal MoGe monocular
    # depth estimator, giving the pose decoder (ScaleShiftInvariant) a metric
    # reference.
    pointmap_full = None
    if depth_path is not None and intrinsics_path is not None:
        depth_path = Path(depth_path)
        intrinsics_path = Path(intrinsics_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Intrinsics sidecar not found: {intrinsics_path}")
        depth_m = np.array(Image.open(depth_path)).astype(np.float32)
        intrinsics = json.loads(intrinsics_path.read_text())
        if depth_m.shape[:2] != image_rgb.shape[:2]:
            depth_pil = Image.fromarray(depth_m)
            depth_pil = depth_pil.resize((image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
            depth_m = np.array(depth_pil, dtype=np.float32)
        pointmap_full = _build_pytorch3d_pointmap(depth_m, intrinsics)  # (H, W, 3) float32

    attempted_sizes = []
    candidate_sizes = []
    for size in [max_side, 112, 96, 80, 64, 48]:
        size = min(int(size), int(max_side))
        if size not in candidate_sizes:
            candidate_sizes.append(size)

    runtime_config_path = _write_runtime_config()
    Inference = _import_official_api()
    used_shape = None

    output = None
    for candidate_size in candidate_sizes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        resized_image_rgb, resized_mask = _resize_image_and_mask(image_rgb, mask, max_side=candidate_size)
        attempted_sizes.append((candidate_size, tuple(resized_image_rgb.shape)))
        _save_preview(resized_mask, resized_image_rgb, preview_path)

        # Resize pointmap to match the resized image using nearest (preserves NaN holes).
        resized_pointmap = None
        if pointmap_full is not None:
            ph, pw = pointmap_full.shape[:2]
            th, tw = resized_image_rgb.shape[:2]
            if (ph, pw) == (th, tw):
                resized_pointmap = pointmap_full
            else:
                pm_t = torch.from_numpy(pointmap_full).permute(2, 0, 1).unsqueeze(0)
                pm_t = torch.nn.functional.interpolate(
                    pm_t, size=(th, tw), mode="nearest",
                )
                resized_pointmap = pm_t.squeeze(0).permute(1, 2, 0).contiguous()

        inference = None
        try:
            inference = Inference(str(runtime_config_path), compile=False)
            if resized_pointmap is not None:
                pm = resized_pointmap
                if not isinstance(pm, torch.Tensor):
                    pm = torch.from_numpy(pm)
                output = inference(resized_image_rgb, resized_mask, seed=42, pointmap=pm)
            else:
                output = inference(resized_image_rgb, resized_mask, seed=42)
            used_shape = tuple(resized_image_rgb.shape)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output = None
            continue
        finally:
            if inference is not None:
                del inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if output is None:
        raise RuntimeError(f"SAM3D failed with CUDA OOM for all candidate sizes: {attempted_sizes}")

    if "gs" not in output:
        raise RuntimeError("SAM3D did not return a gaussian output.")
    output["gs"].save_ply(str(ply_path))

    pose_data = {}
    for key in ("translation", "rotation", "scale"):
        value = output.get(key)
        if value is not None:
            pose_data[key] = torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
    if "rotation" not in pose_data or len(pose_data["rotation"]) != 4:
        raise RuntimeError("SAM3D did not return a valid object rotation pose.")
    if pose_data:
        pose_path.write_text(json.dumps(pose_data, indent=2) + "\n")

    run_info: List[str] = [
        "SAM 3D Objects dynamic-gs integration run",
        f"Rendered image: {render_image_path}",
        f"Object mask: {object_mask_path}",
        f"Original image shape: {tuple(image_rgb.shape)}",
        f"Used inference image shape: {used_shape}",
        f"Attempted sizes: {attempted_sizes}",
        f"Repo root: {SAM3D_REPO_ROOT}",
        f"Config path: {SAM3D_CONFIG_PATH}",
        f"Saved runtime config: {runtime_config_path}",
        f"Saved gaussian splat: {ply_path}",
        f"Saved pose sidecar: {pose_path}",
        f"Saved preview: {preview_path}",
    ]

    glb = output.get("glb")
    if glb is None:
        run_info.append("No GLB object was returned by SAM 3D Objects.")
    else:
        glb.export(str(glb_path))
        run_info.append(f"Saved mesh/glb: {glb_path}")

    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_info_path.write_text("\n".join(run_info) + "\n")
    return output_paths


def run_sam3d_multi_object(
    render_image_path: Path,
    object_mask_paths: list[Path],
    output_dir: Path,
    output_stems: list[str],
    image_dir: Path | None = None,
    max_side: int = 518,
    depth_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> list[Dict[str, Path]]:
    """Run SAM3D on multiple object masks with a single model load.

    This is an intentional behavior change from the single-object path:
    unlike ``run_sam3d_single_object`` which crops inputs via
    ``prepare_cropped_sam3d_inputs()`` for lighter inference, this function
    operates on the full (resized) image because SAM3D predicts object
    pose/layout relative to the full image context.  Cropping would alter
    the estimated rotation, translation, and scale.

    The efficiency comes from reusing one ``Inference(...)`` model load
    across all masks, not from a native batched API — each mask still gets
    its own sequential inference call.

    When ``depth_path`` and ``intrinsics_path`` are both given, the worker
    builds a full-image metric pytorch3d pointmap and passes it to SAM3D
    via ``inference(..., pointmap=...)``, bypassing the internal MoGe
    monocular depth estimator. The depth file is expected to already be
    in metres (float32 TIFF), and the intrinsics JSON to match the full
    image resolution (no crop shift).
    """
    render_image_path = Path(render_image_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if len(object_mask_paths) != len(output_stems):
        raise ValueError(
            f"mask count ({len(object_mask_paths)}) != stem count ({len(output_stems)})"
        )

    if not render_image_path.exists():
        raise FileNotFoundError(render_image_path)
    if not SAM3D_REPO_ROOT.exists():
        raise FileNotFoundError(SAM3D_REPO_ROOT)
    if not SAM3D_CONFIG_PATH.exists():
        raise FileNotFoundError(SAM3D_CONFIG_PATH)

    image_pil = Image.open(render_image_path).convert("RGB")
    image_rgb = np.array(image_pil)

    # Load all masks
    masks = []
    for mask_path in object_mask_paths:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        m = _load_binary_mask(mask_path, image_pil.size)
        if image_rgb.shape[:2] != m.shape[:2]:
            raise ValueError(f"SAM3D image/mask shape mismatch: {image_rgb.shape} vs {m.shape}")
        masks.append(m)

    # Optional: build a full-image metric pointmap (pytorch3d convention)
    # to bypass MoGe. The depth file is already in metres; the intrinsics
    # JSON matches the full image resolution.
    pointmap_full = None
    if depth_path is not None and intrinsics_path is not None:
        depth_path = Path(depth_path)
        intrinsics_path = Path(intrinsics_path)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Intrinsics sidecar not found: {intrinsics_path}")
        depth_m = np.array(Image.open(depth_path)).astype(np.float32)
        intrinsics = json.loads(intrinsics_path.read_text())
        if depth_m.shape[:2] != image_rgb.shape[:2]:
            depth_pil = Image.fromarray(depth_m)
            depth_pil = depth_pil.resize((image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
            depth_m = np.array(depth_pil, dtype=np.float32)
        pointmap_full = _build_pytorch3d_pointmap(depth_m, intrinsics)  # (H, W, 3) float32

    runtime_config_path = _write_runtime_config()
    Inference = _import_official_api()

    # Load model once
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inference = None
    try:
        inference = Inference(str(runtime_config_path), compile=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load SAM3D model: {exc}") from exc

    all_results: list[Dict[str, Path]] = []
    try:
        for i, (mask, stem) in enumerate(zip(masks, output_stems)):
            output_paths = get_sam3d_output_paths(output_dir, stem, image_dir=image_dir)
            ply_path = output_paths["ply_path"]
            pose_path = output_paths["pose_path"]
            preview_path = output_paths["preview_path"]
            run_info_path = output_paths["run_info_path"]
            mesh_ply_path = output_paths["mesh_ply_path"]

            if int(mask.sum()) == 0:
                print(f"[sam3d-multi] skipping mask {i} ({stem}): empty mask", file=sys.stderr)
                all_results.append({})
                continue

            # Resize image + mask for inference
            resized_image_rgb, resized_mask = _resize_image_and_mask(image_rgb, mask, max_side=max_side)
            _save_preview(resized_mask, resized_image_rgb, preview_path)

            output = None
            used_shape = None
            # Try with requested size, then progressively smaller on OOM
            candidate_sizes = []
            for size in [max_side, 112, 96, 80, 64, 48]:
                size = min(int(size), int(max_side))
                if size not in candidate_sizes:
                    candidate_sizes.append(size)

            attempted_sizes = []
            for candidate_size in candidate_sizes:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cur_image, cur_mask = _resize_image_and_mask(image_rgb, mask, max_side=candidate_size)
                attempted_sizes.append((candidate_size, tuple(cur_image.shape)))

                # Resize the full pointmap to match the resized image
                # (nearest preserves NaN holes for invalid depth pixels).
                resized_pointmap = None
                if pointmap_full is not None:
                    ph, pw = pointmap_full.shape[:2]
                    th, tw = cur_image.shape[:2]
                    if (ph, pw) == (th, tw):
                        resized_pointmap = torch.from_numpy(pointmap_full)
                    else:
                        pm_t = torch.from_numpy(pointmap_full).permute(2, 0, 1).unsqueeze(0)
                        pm_t = torch.nn.functional.interpolate(pm_t, size=(th, tw), mode="nearest")
                        resized_pointmap = pm_t.squeeze(0).permute(1, 2, 0).contiguous()
                try:
                    if resized_pointmap is not None:
                        output = inference(cur_image, cur_mask, seed=42, pointmap=resized_pointmap)
                    else:
                        output = inference(cur_image, cur_mask, seed=42)
                    used_shape = tuple(cur_image.shape)
                    break
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    output = None
                    continue

            if output is None:
                print(
                    f"[sam3d-multi] mask {i} ({stem}) failed with OOM for all sizes: {attempted_sizes}",
                    file=sys.stderr,
                )
                all_results.append({})
                continue

            if "gs" not in output:
                print(f"[sam3d-multi] mask {i} ({stem}) did not return a gaussian output", file=sys.stderr)
                all_results.append({})
                continue

            output["gs"].save_ply(str(ply_path))

            # Export the SAM3D triangle mesh (FoundationPose input). The mesh
            # decoder produces a `MeshExtractResult` with `.vertices` and
            # `.faces` torch tensors in the same canonical mesh frame as the
            # Gaussian splat above.
            mesh_saved = False
            mesh_list = output.get("mesh")
            if mesh_list is not None and len(mesh_list) > 0:
                mesh_result = mesh_list[0]
                if getattr(mesh_result, "success", True):
                    try:
                        import trimesh
                        verts = mesh_result.vertices.detach().cpu().numpy()
                        faces = mesh_result.faces.detach().cpu().numpy()
                        if verts.shape[0] > 0 and faces.shape[0] > 0:
                            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                            tm.export(str(mesh_ply_path))
                            mesh_saved = True
                    except Exception as exc:
                        print(
                            f"[sam3d-multi] mask {i} ({stem}): mesh export failed: {exc}",
                            file=sys.stderr,
                        )

            pose_data = {}
            for key in ("translation", "rotation", "scale"):
                value = output.get(key)
                if value is not None:
                    pose_data[key] = torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
            if "rotation" not in pose_data or len(pose_data["rotation"]) != 4:
                print(f"[sam3d-multi] mask {i} ({stem}): no valid rotation in SAM3D output", file=sys.stderr)
                all_results.append({})
                continue
            pose_path.write_text(json.dumps(pose_data, indent=2) + "\n")

            run_info: List[str] = [
                f"SAM3D multi-object: mask {i} ({stem})",
                f"Rendered image: {render_image_path}",
                f"Object mask: {object_mask_paths[i]}",
                f"Original image shape: {tuple(image_rgb.shape)}",
                f"Used inference image shape: {used_shape}",
                f"Attempted sizes: {attempted_sizes}",
                f"Saved gaussian splat: {ply_path}",
                f"Saved pose sidecar: {pose_path}",
                f"Saved triangle mesh: {mesh_ply_path if mesh_saved else 'NOT SAVED'}",
            ]
            run_info_path.write_text("\n".join(run_info) + "\n")
            del output
            all_results.append(output_paths)
    finally:
        if inference is not None:
            del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def run_sam3d_multi_object_subprocess(
    render_image_path: Path,
    object_mask_paths: list[Path],
    output_dir: Path,
    output_stems: list[str],
    image_dir: Path | None = None,
    max_side: int = 518,
    depth_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> list[Dict[str, Path]]:
    """Run multi-object SAM3D generation in a single subprocess.

    Uses ``sys.executable`` (same as the current SAM3D subprocess, since
    SAM3D runs in ``radiance_ros``).

    When ``depth_path`` and ``intrinsics_path`` are both given, they are
    forwarded to the worker so SAM3D receives a metric pointmap instead
    of relying on its internal MoGe estimator. They must describe the
    full image (no crop offset).
    """
    render_image_path = Path(render_image_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    multi_mask_data = {
        "mask_paths": [str(Path(p).resolve()) for p in object_mask_paths],
        "output_stems": list(output_stems),
    }

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-image", str(render_image_path),
        "--output-dir", str(output_dir),
        "--image-dir", str(image_dir),
        "--max-side", str(max_side),
        "--multi-mask-json", json.dumps(multi_mask_data),
    ]
    if depth_path is not None:
        command.extend(["--depth", str(depth_path)])
    if intrinsics_path is not None:
        command.extend(["--intrinsics", str(intrinsics_path)])
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SAM3D multi-object subprocess failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    # Rebuild output paths from stems
    all_results: list[Dict[str, Path]] = []
    for stem in output_stems:
        output_paths = get_sam3d_output_paths(output_dir, stem, image_dir=image_dir)
        resolved_pose_path = resolve_sam3d_pose_path(output_paths["ply_path"], output_paths["pose_path"])
        if output_paths["ply_path"].exists() and sam3d_pose_has_rotation(resolved_pose_path):
            if resolved_pose_path is not None:
                output_paths["pose_path"] = resolved_pose_path
            all_results.append(output_paths)
        else:
            all_results.append({})
    return all_results


def run_sam3d_single_object_subprocess(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    max_side: int = 518,
    depth_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> Dict[str, Path]:
    """Run the working SAM3D generation path in a fresh Python process.

    This keeps the heavy SAM3D CUDA state separate from the main `ns-train`
    process, which is much more memory-stable on the 8 GB GPU.

    When ``depth_path`` and ``intrinsics_path`` are both given, the worker
    builds a metric pytorch3d pointmap from them and passes it to SAM3D via
    ``inference(..., pointmap=...)``, bypassing the internal MoGe depth
    estimator.
    """

    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-image",
        str(render_image_path),
        "--object-mask",
        str(object_mask_path),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        output_stem,
        "--image-dir",
        str(image_dir),
        "--max-side",
        str(max_side),
    ]
    if depth_path is not None:
        command.extend(["--depth", str(depth_path)])
    if intrinsics_path is not None:
        command.extend(["--intrinsics", str(intrinsics_path)])
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SAM3D subprocess failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    output_paths = get_sam3d_output_paths(output_dir, output_stem, image_dir=image_dir)
    resolved_pose_path = resolve_sam3d_pose_path(output_paths["ply_path"], output_paths["pose_path"])
    if not sam3d_pose_has_rotation(resolved_pose_path):
        raise RuntimeError(
            f"SAM3D subprocess produced `{output_paths['ply_path']}` but no valid rotation pose sidecar was found."
        )
    output_paths["pose_path"] = resolved_pose_path if resolved_pose_path is not None else output_paths["pose_path"]
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-image", type=Path, required=True)
    parser.add_argument("--object-mask", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=518)
    parser.add_argument(
        "--multi-mask-json", type=str, default=None,
        help='JSON string: {"mask_paths": [...], "output_stems": [...]}',
    )
    parser.add_argument(
        "--depth", type=Path, default=None,
        help="Optional metric depth image matching --render-image (same crop).",
    )
    parser.add_argument(
        "--intrinsics", type=Path, default=None,
        help="Optional JSON sidecar with {fx, fy, cx, cy} matching --depth.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    try:
        if args.multi_mask_json is not None:
            data = json.loads(args.multi_mask_json)
            mask_paths = [Path(p) for p in data["mask_paths"]]
            output_stems = data["output_stems"]
            run_sam3d_multi_object(
                render_image_path=args.render_image,
                object_mask_paths=mask_paths,
                output_dir=args.output_dir,
                output_stems=output_stems,
                image_dir=args.image_dir,
                max_side=args.max_side,
                depth_path=args.depth,
                intrinsics_path=args.intrinsics,
            )
        else:
            if args.object_mask is None or args.output_stem is None:
                print("Single-object mode requires --object-mask and --output-stem", file=sys.stderr)
                return 1
            run_sam3d_single_object(
                render_image_path=args.render_image,
                object_mask_path=args.object_mask,
                output_dir=args.output_dir,
                output_stem=args.output_stem,
                image_dir=args.image_dir,
                max_side=args.max_side,
                depth_path=args.depth,
                intrinsics_path=args.intrinsics,
            )
        return 0
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"SAM3D worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI path
    raise SystemExit(_main())
