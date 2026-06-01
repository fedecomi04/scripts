"""Simulate the per-tick XFeat extract on the D0 live image:
- Extract top_k keypoints on the FULL image (no mask) — same as per-tick path
- Count how many land INSIDE the rendered object mask
- Save a visualization with green dots = inside-mask, red dots = outside

Usage:
    python scripts/diag_xfeat_pertick.py <live_rgb.png> <rendered_obj_mask.png> [--top-k 300] [--out DIR]
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
    ap.add_argument("live_rgb", type=Path)
    ap.add_argument("rendered_obj_mask", type=Path)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--detection-threshold", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("/tmp/diag_xfeat_pertick"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1] / "third_party" / "xfeat"
    sys.path.insert(0, str(repo))
    from modules.xfeat import XFeat  # type: ignore

    img_bgr = cv2.imread(str(args.live_rgb), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(args.rendered_obj_mask), 0)
    if mask.shape != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 127

    inp = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda()
    weights = repo / "weights" / "xfeat.pt"
    xfeat = XFeat(weights=str(weights), top_k=args.top_k, detection_threshold=args.detection_threshold)

    with torch.inference_mode():
        for _ in range(2):  # warmup
            _ = xfeat.detectAndCompute(inp, top_k=args.top_k)
        out = xfeat.detectAndCompute(inp, top_k=args.top_k)[0]
    kp = out["keypoints"].cpu().numpy().astype(np.float32)
    N = kp.shape[0]
    print(f"XFeat extracted {N} keypoints on full {img_rgb.shape[:2]} image (top_k={args.top_k})")

    xs = np.clip(kp[:, 0].round().astype(np.int64), 0, mask.shape[1] - 1)
    ys = np.clip(kp[:, 1].round().astype(np.int64), 0, mask.shape[0] - 1)
    inside = mask_bool[ys, xs]
    n_inside = int(inside.sum())
    print(f"keypoints INSIDE rendered_obj_mask: {n_inside}/{N}  ({100.0*n_inside/max(N,1):.1f}%)")

    args.out.mkdir(parents=True, exist_ok=True)
    overlay = img_bgr.copy()
    for i in range(N):
        color = (0, 255, 0) if inside[i] else (0, 0, 255)
        cv2.circle(overlay, (xs[i], ys[i]), 3, color, -1)
    # Outline the mask in cyan.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    cv2.imwrite(str(args.out / "xfeat_pertick_overlay.png"), overlay)
    cv2.imwrite(str(args.out / "rendered_obj_mask.png"), mask)
    print(f"saved overlay (green=in-mask, red=out-of-mask, cyan=mask contour) → {args.out / 'xfeat_pertick_overlay.png'}")


if __name__ == "__main__":
    main()
