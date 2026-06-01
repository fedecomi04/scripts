"""Standalone test: feed the actual tracking RGB straight to XFeat and
report how many keypoints come out. Confirms whether XFeat itself is
failing on this image or whether something in the pipeline path is.

Usage:
    python scripts/diag_xfeat_direct.py /path/to/tracking_rgb.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Path to tracking_rgb.png")
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--detection-threshold", type=float, default=0.05)
    args = ap.parse_args()

    # Match the pipeline's import path for XFeat.
    repo = Path(__file__).resolve().parents[1] / "third_party" / "xfeat"
    sys.path.insert(0, str(repo))
    from modules.xfeat import XFeat  # type: ignore

    img_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"could not load {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"image shape={img_rgb.shape}  dtype={img_rgb.dtype}  range=[{img_rgb.min()}..{img_rgb.max()}]")

    # Build the exact tensor `_extract` would see: (1, 3, H, W), float32, 0..255 range.
    inp = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda()
    print(f"tensor input shape={inp.shape}  range=[{inp.min().item():.1f}..{inp.max().item():.1f}]")

    weights = repo / "weights" / "xfeat.pt"
    print(f"weights: {weights}  exists={weights.exists()}")
    xfeat = XFeat(weights=str(weights), top_k=args.top_k,
                  detection_threshold=args.detection_threshold)
    print(f"XFeat on device {xfeat.dev}, top_k={args.top_k}, detection_threshold={args.detection_threshold}")

    # Warm-up + actual call, matching _extract.
    with torch.inference_mode():
        for i in range(3):
            out_list = xfeat.detectAndCompute(inp, top_k=args.top_k)
            out = out_list[0]
            kp = out["keypoints"]
            desc = out["descriptors"]
            print(f"call {i}: keypoints shape={tuple(kp.shape)}  descriptors shape={tuple(desc.shape)}")
            if kp.shape[0] > 0:
                print(f"  kp range x=[{kp[:,0].min().item():.1f}..{kp[:,0].max().item():.1f}]"
                      f"  y=[{kp[:,1].min().item():.1f}..{kp[:,1].max().item():.1f}]")
                print(f"  desc norm range=[{desc.norm(dim=-1).min().item():.4f}..{desc.norm(dim=-1).max().item():.4f}]")

    # Sanity: try a tiny detection threshold to see if anything was being filtered out.
    if args.detection_threshold > 0:
        print()
        print("retry with detection_threshold=0 (no score filtering):")
        xfeat2 = XFeat(weights=str(weights), top_k=args.top_k, detection_threshold=0.0)
        with torch.inference_mode():
            out_list = xfeat2.detectAndCompute(inp, top_k=args.top_k)
            kp = out_list[0]["keypoints"]
            print(f"  keypoints shape={tuple(kp.shape)}")


if __name__ == "__main__":
    main()
