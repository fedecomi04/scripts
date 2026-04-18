#!/usr/bin/env python3
"""Prepare a reusable D0 reference bundle for offline SAM3D fusion experiments.

This script is designed to be cheap and fully reusable:
- Reuses the already-saved D0 artifacts in `dynamic_scene/render_masks_esam/`
  (render, masks, SAM3D raw output, pose.json).
- Loads a static-phase checkpoint and re-renders once at the D0 camera pose to
  cache the rendered RGB/depth and the current object target subset extracted
  from the checkpoint Gaussians.

Example:
  python scripts/fusion_experiments/prepare_d0_bundle.py \
    --dataset-root /path/to/dynamic_gs_test_xxx \
    --capture-config /path/to/config.yml \
    --frame arm_05460 \
    --output /path/to/dynamic_gs_test_xxx/fusion_bench/d0_bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from nerfstudio.utils.eval_utils import eval_setup

from dynamic_gs.utils.sam3d_fusion import save_point_cloud


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _load_binary_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def _save_rgb(path: Path, rgb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb_np = rgb.detach().float().clamp(0.0, 1.0).cpu().numpy()
    Image.fromarray((rgb_np * 255.0 + 0.5).astype(np.uint8)).save(path)


def _save_depth_npy_and_vis(depth_npy_path: Path, depth_vis_path: Path, depth: torch.Tensor) -> None:
    depth_npy_path.parent.mkdir(parents=True, exist_ok=True)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_np = depth.detach().float().cpu().numpy().astype(np.float32)
    np.save(depth_npy_path, depth_np)

    finite = np.isfinite(depth_np) & (depth_np > 0)
    if not np.any(finite):
        vis = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        lo = float(np.percentile(depth_np[finite], 1.0))
        hi = float(np.percentile(depth_np[finite], 99.0))
        if hi <= lo:
            hi = lo + 1e-3
        vis = np.clip((depth_np - lo) / (hi - lo), 0.0, 1.0)
        vis = (vis * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(vis).save(depth_vis_path)


def _find_dynamic_frame_idx(datamanager, frame_name: str) -> int:
    num = datamanager.get_num_dynamic_frames()
    for idx in range(num):
        if datamanager.get_dynamic_frame_name(idx) == frame_name:
            return idx
    raise ValueError(f"Frame {frame_name} not found in dynamic frames (count={num})")


def _resolve_sam3d_raw_output(artifacts_dir: Path, frame: str) -> Path:
    preferred = artifacts_dir / f"{frame}_d0_true_sam3d_raw_output.ply"
    if preferred.exists():
        return preferred
    legacy = artifacts_dir / f"{frame}_sam3d_raw_output.ply"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(preferred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True, help="Path to the saved static-capture config.yml")
    parser.add_argument("--frame", type=str, default="arm_05460")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/dynamic_scene/render_masks_esam",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/fusion_bench/d0_bundle",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    artifacts_dir = args.artifacts_dir or (dataset_root / "dynamic_scene" / "render_masks_esam")
    output_dir = (args.output or (dataset_root / "fusion_bench" / "d0_bundle")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = args.frame
    required = {
        "render.png": artifacts_dir / f"{frame}_render.png",
        "change_mask.png": artifacts_dir / f"{frame}_change_mask.png",
        "render_object_mask.png": artifacts_dir / f"{frame}_render_object_mask.png",
        "live_object_mask.png": artifacts_dir / f"{frame}_live_object_mask.png",
        "sam3d_raw_output.ply": _resolve_sam3d_raw_output(artifacts_dir, frame),
        "sam3d_pose.json": artifacts_dir / f"{frame}_sam3d_pose.json",
    }
    for dst_name, src in required.items():
        if not src.exists():
            raise FileNotFoundError(src)
        _link_or_copy(src, output_dir / dst_name)

    live_rgb_path = dataset_root / "dynamic_scene" / "rgb" / f"{frame}.png"
    if live_rgb_path.exists():
        _link_or_copy(live_rgb_path, output_dir / "live_rgb.png")

    # Load checkpoint and render once at the D0 camera pose
    config, pipeline, checkpoint_path, step = eval_setup(args.capture_config, test_mode="val")
    datamanager = pipeline.datamanager
    datamanager.set_phase("dynamic")
    frame_idx = _find_dynamic_frame_idx(datamanager, frame)
    datamanager.set_dynamic_frame_idx(frame_idx)
    camera, batch = datamanager.get_current_dynamic_train_batch()

    model = pipeline.model
    outputs = model.get_outputs(camera)
    if outputs.get("rgb") is None or outputs.get("depth") is None:
        raise RuntimeError("Render did not produce rgb/depth outputs.")

    _save_rgb(output_dir / "reference_render_rgb.png", outputs["rgb"])
    _save_depth_npy_and_vis(
        output_dir / "reference_render_depth.npy",
        output_dir / "reference_render_depth_vis.png",
        outputs["depth"],
    )

    # Cache projection params for downstream use
    viewmat, intrinsics, width, height = model._get_render_projection_params(camera.to(model.device))
    np.save(output_dir / "ref_viewmat.npy", viewmat.astype(np.float32))
    np.save(output_dir / "ref_intrinsics.npy", intrinsics.astype(np.float32))

    # Extract the current visible object subset from the checkpoint Gaussians
    render_object_mask = _load_binary_mask(output_dir / "render_object_mask.png")
    mask_tensor = torch.from_numpy(render_object_mask.astype(np.float32))
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor[..., None]
    mask_tensor = mask_tensor.to(device=model.device, dtype=outputs["depth"].dtype)
    indices, means, colors = model._get_existing_object_subset(mask_tensor, outputs["depth"])
    means_np = means.detach().cpu().numpy().astype(np.float32)
    colors_np = colors.detach().cpu().numpy().astype(np.float32)
    save_point_cloud(output_dir / "target_subset.ply", means_np, colors_np)
    np.save(output_dir / "target_subset_indices.npy", indices.detach().cpu().numpy().astype(np.int64))

    manifest = {
        "dataset_root": str(dataset_root),
        "capture_config": str(args.capture_config.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(step),
        "frame_name": frame,
        "dynamic_frame_idx": int(frame_idx),
        "ref_width": int(width),
        "ref_height": int(height),
        "paths": {k: str((output_dir / k).resolve()) for k in required.keys()},
        "cached": {
            "reference_render_rgb": str((output_dir / "reference_render_rgb.png").resolve()),
            "reference_render_depth_npy": str((output_dir / "reference_render_depth.npy").resolve()),
            "ref_viewmat_npy": str((output_dir / "ref_viewmat.npy").resolve()),
            "ref_intrinsics_npy": str((output_dir / "ref_intrinsics.npy").resolve()),
            "target_subset_ply": str((output_dir / "target_subset.ply").resolve()),
            "target_subset_indices_npy": str((output_dir / "target_subset_indices.npy").resolve()),
        },
        "counts": {
            "target_subset_points": int(means_np.shape[0]),
        },
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output_dir)


if __name__ == "__main__":
    main()
