#!/usr/bin/env python3
"""Measure the pixel shift between the saved mask and the true object location
in the depth (object = pixels near the mask-region's median depth). Reports
dx,dy in pixels + as a fraction of width/height. Constant px => crop-origin
bug; proportional => scale/letterbox bug.

Usage: diag_mask_shift.py <dataset_dir> [<dataset_dir> ...]
"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def measure(ddir):
    ddir = Path(ddir)
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    H, W = mask.shape

    mys, mxs = np.where(mask)
    mcx, mcy = mxs.mean(), mys.mean()

    # search region: mask bbox padded generously
    pad = 120
    y0, y1 = max(0, mys.min() - pad), min(H, mys.max() + pad)
    x0, x1 = max(0, mxs.min() - pad), min(W, mxs.max() + pad)
    reg = depth[y0:y1, x0:x1]
    valid = np.isfinite(reg) & (reg > 0)
    om = np.median(reg[valid])
    near = valid & (np.abs(reg - om) < 0.04)
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    # largest connected component = the object in depth
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(near)
    if nlab <= 1:
        return None
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ocx = cent[big][0] + x0
    ocy = cent[big][1] + y0

    dx, dy = mcx - ocx, mcy - ocy
    return dict(name=ddir.name, W=W, H=H, mask_c=(mcx, mcy), obj_c=(ocx, ocy),
                dx=dx, dy=dy, fx=dx / W, fy=dy / H, obj_med=om)


def main():
    print(f"{'dataset':28s} {'mask_cx,cy':>16} {'depthobj_cx,cy':>16} "
          f"{'dx,dy px':>14} {'dx%W,dy%H':>14}")
    for d in sys.argv[1:]:
        r = measure(d)
        if r is None:
            print(f"{Path(d).name:28s}  (no depth object found)"); continue
        print(f"{r['name']:28s} {r['mask_c'][0]:7.0f},{r['mask_c'][1]:6.0f} "
              f"{r['obj_c'][0]:7.0f},{r['obj_c'][1]:6.0f} "
              f"{r['dx']:+7.0f},{r['dy']:+5.0f} "
              f"{100*r['fx']:+6.1f}%,{100*r['fy']:+5.1f}%   (obj@{r['obj_med']:.2f}m)")


if __name__ == "__main__":
    main()
