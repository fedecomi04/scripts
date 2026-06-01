"""Test LighterGlue directly: extract XFeat features on two consecutive
frames of the fidget dataset and run the matcher. See if it returns any
matches at all.
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
    ap.add_argument("img_a", type=Path)
    ap.add_argument("img_b", type=Path)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--min-conf", type=float, default=0.1)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1] / "third_party" / "xfeat"
    sys.path.insert(0, str(repo))
    from modules.xfeat import XFeat  # type: ignore
    from modules.lighterglue import LighterGlue  # type: ignore

    def _extract(img_path, xfeat):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).cuda()
        H, W = rgb.shape[:2]
        with torch.inference_mode():
            for _ in range(2):
                _ = xfeat.detectAndCompute(inp, top_k=args.top_k)
            out = xfeat.detectAndCompute(inp, top_k=args.top_k)[0]
        return out["keypoints"], out["descriptors"], (W, H)

    weights = repo / "weights" / "xfeat.pt"
    xfeat = XFeat(weights=str(weights), top_k=args.top_k, detection_threshold=0.05)
    lg_weights = str(repo / "weights" / "xfeat-lighterglue.pt")
    lg = LighterGlue(weights=lg_weights)

    kp_a, desc_a, size_a = _extract(args.img_a, xfeat)
    kp_b, desc_b, size_b = _extract(args.img_b, xfeat)
    print(f"A: {kp_a.shape[0]} keypoints, descriptors shape {tuple(desc_a.shape)}, size {size_a}")
    print(f"B: {kp_b.shape[0]} keypoints, descriptors shape {tuple(desc_b.shape)}, size {size_b}")

    dev = lg.dev
    data = {
        "keypoints0": kp_a.to(dev).float()[None],
        "keypoints1": kp_b.to(dev).float()[None],
        "descriptors0": desc_a.to(dev).float()[None],
        "descriptors1": desc_b.to(dev).float()[None],
        "image_size0": torch.tensor([size_a[0], size_a[1]], device=dev, dtype=torch.float32)[None],
        "image_size1": torch.tensor([size_b[0], size_b[1]], device=dev, dtype=torch.float32)[None],
    }
    with torch.inference_mode():
        out = lg(data, min_conf=args.min_conf)
    matches = out.get("matches", [None])
    if matches and matches[0] is not None:
        m = matches[0].cpu().numpy()
        print(f"LighterGlue matches (min_conf={args.min_conf}): {m.shape[0]} pairs")
    else:
        print(f"LighterGlue returned no matches (min_conf={args.min_conf})")

    # Try with very low confidence too.
    if args.min_conf > 0.01:
        with torch.inference_mode():
            out_lo = lg(data, min_conf=0.0)
        m_lo = out_lo.get("matches", [None])[0]
        if m_lo is not None:
            print(f"LighterGlue matches (min_conf=0.0): {m_lo.shape[0]} pairs")


if __name__ == "__main__":
    main()
