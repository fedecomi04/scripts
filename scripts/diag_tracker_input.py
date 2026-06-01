"""Dump the exact image XFeat sees for the per-tick extract on a given
dynamic frame, to verify whether the input is degenerate (uniform / black /
gripper-everywhere).

Usage:
    python scripts/diag_tracker_input.py <dataset_root> [--frame N] [--out-dir DIR]

Writes 3 PNGs side-by-side:
  - raw_rgb.png         : original on-disk RGB
  - dataset_mask.png    : gripper-keep mask (1=keep, 0=gripper)
  - tracking_rgb.png    : `_build_tracking_rgb` composite (rgb + gazebo-blue
                          where mask=0)  ← this is exactly what XFeat sees
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/diag_tracker"))
    args = ap.parse_args()

    transforms_path = args.dataset / "dynamic_scene" / "transforms.json"
    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"]
    frame = frames[args.frame]

    # Resolve relative paths against the dataset root.
    rgb_rel = frame.get("file_path") or frame.get("image_path")
    mask_rel = frame.get("mask_path")
    rgb_path = (args.dataset / "dynamic_scene" / rgb_rel).resolve()
    mask_path = (args.dataset / "dynamic_scene" / mask_rel).resolve() if mask_rel else None

    print(f"frame {args.frame}: rgb={rgb_path.name}  mask={mask_path.name if mask_path else 'none'}")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)  # BGR
    print(f"rgb shape={rgb.shape}  range=[{rgb.min()}..{rgb.max()}]  mean BGR={rgb.mean(axis=(0,1))}")

    mask = cv2.imread(str(mask_path), 0) if mask_path else None
    if mask is not None:
        keep_frac = (mask > 0).mean()
        print(f"mask shape={mask.shape}  keep_fraction={keep_frac:.4f}  drop_fraction={(1-keep_frac):.4f}")

    # Reproduce `_build_tracking_rgb`: rgb where mask=1, gazebo blue where mask=0.
    gazebo_bgr = np.array([0.86 * 255, 0.92 * 255, 1.0 * 255], dtype=np.float32)[::-1]  # gazebo is (0.86,0.92,1.0) in RGB → BGR reversed
    tracking_bgr = rgb.astype(np.float32).copy()
    if mask is not None:
        gripper = (mask == 0)
        tracking_bgr[gripper] = gazebo_bgr
    tracking_u8 = np.clip(tracking_bgr, 0, 255).astype(np.uint8)
    print(f"tracking_rgb stats: range=[{tracking_u8.min()}..{tracking_u8.max()}]  mean BGR={tracking_u8.mean(axis=(0,1))}  std={tracking_u8.std(axis=(0,1))}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / "raw_rgb.png"), rgb)
    if mask is not None:
        cv2.imwrite(str(args.out_dir / "dataset_mask.png"), mask)
    cv2.imwrite(str(args.out_dir / "tracking_rgb.png"), tracking_u8)
    print(f"\nwrote 3 PNGs to {args.out_dir}/")
    print(f"  raw_rgb.png       — on-disk RGB ({rgb_path.name})")
    if mask is not None:
        print(f"  dataset_mask.png  — gripper-keep mask")
    print(f"  tracking_rgb.png  — what XFeat actually sees")


if __name__ == "__main__":
    main()
