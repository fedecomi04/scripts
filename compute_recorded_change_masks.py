#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dynamic_gs.utils.active_mask import build_change_mask


DATASET_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynaarm_gs_depth_mask_01"
)
OUTPUT_DIRNAME = "change_masks_pipeline_prev_frame"
SUMMARY_FILENAME = "change_mask_summary.csv"


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read RGB image from {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb.astype(np.float32) / 255.0)


def _load_depth_tensor(path: Path) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image from {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth.astype(np.float32))


def _load_mask_tensor(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask from {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy((mask > 0).astype(np.float32))[..., None]


def _save_mask_png(path: Path, mask: torch.Tensor) -> None:
    array = (mask.detach().cpu().numpy()[..., 0] > 0.5).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), array):
        raise RuntimeError(f"Failed to save change mask to {path}")


def _load_saved_change_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read saved change mask from {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy((mask > 0).astype(np.float32))[..., None]


def _get_last_two_recordings(root: Path) -> list[Path]:
    recordings = sorted([path for path in root.iterdir() if path.is_dir()])
    if len(recordings) < 2:
        raise RuntimeError(f"Need at least two recordings under {root}, found {len(recordings)}")
    return recordings[-2:]


def _compute_dataset_change_masks(dataset_dir: Path) -> None:
    transforms_path = dataset_dir / "transforms.json"
    transforms = json.loads(transforms_path.read_text())
    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames found in {transforms_path}")

    output_dir = dataset_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME

    rows: list[dict[str, str]] = []

    previous_rgb = None
    previous_depth = None

    for frame_idx, frame in enumerate(frames):
        rgb_path = (dataset_dir / frame["file_path"]).resolve()
        depth_path = (dataset_dir / frame["depth_file_path"]).resolve()
        mask_path = (dataset_dir / frame["mask_path"]).resolve()
        frame_name = Path(frame["file_path"]).name

        current_rgb = _load_rgb_tensor(rgb_path)
        current_depth = _load_depth_tensor(depth_path)
        current_mask = _load_mask_tensor(mask_path)

        if previous_rgb is None or previous_depth is None:
            change_mask = torch.zeros_like(current_mask)
            prev_name = ""
        else:
            prev_name = Path(frames[frame_idx - 1]["file_path"]).name
            output_path = output_dir / frame_name
            if output_path.exists():
                change_mask = _load_saved_change_mask(output_path)
            else:
                # Reuse the exact same pipeline change-mask function, but apply it
                # frame-to-frame so the saved masks indicate when motion occurs.
                change_mask = build_change_mask(
                    previous_depth,
                    current_depth,
                    pred_rgb=previous_rgb,
                    gt_rgb=current_rgb,
                    valid_mask=current_mask,
                )

        output_path = output_dir / frame_name
        if not output_path.exists():
            _save_mask_png(output_path, change_mask)

        changed_pixels = int((change_mask[..., 0] > 0.5).sum().item())
        total_valid_pixels = int((current_mask[..., 0] > 0.5).sum().item())
        changed_fraction = float(changed_pixels / max(total_valid_pixels, 1))
        rows.append(
            {
                "frame_index": str(frame_idx),
                "frame_name": frame_name,
                "previous_frame_name": prev_name,
                "changed_pixels": str(changed_pixels),
                "valid_pixels": str(total_valid_pixels),
                "changed_fraction": f"{changed_fraction:.8f}",
            }
        )

        previous_rgb = current_rgb
        previous_depth = current_depth

        if frame_idx % 25 == 0 or frame_idx == len(frames) - 1:
            print(
                f"[change-mask] {dataset_dir.name}: frame {frame_idx + 1}/{len(frames)} "
                f"({frame_name}) changed_pixels={changed_pixels}",
                flush=True,
            )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "frame_name",
                "previous_frame_name",
                "changed_pixels",
                "valid_pixels",
                "changed_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for dataset_dir in _get_last_two_recordings(DATASET_ROOT):
        print(f"[change-mask] processing {dataset_dir}")
        _compute_dataset_change_masks(dataset_dir)


if __name__ == "__main__":
    main()
