#!/usr/bin/env python3
"""Is the RGB/mask spatially aligned with the depth? Crop around the object and
render RGB, depth (colormapped), and the mask contour over BOTH. If the depth
silhouette of the object is shifted from the RGB mask contour, the mask samples
neighbouring (table) depth -> target contamination -> insert offset.

Usage: diag_rgbd_align.py <dataset_dir>
"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    fr = tj["frames"][-1]
    stem = Path(fr["file_path"]).stem
    rgb = np.array(Image.open(ss / "rgb" / f"{stem}.png").convert("RGB"))
    depth = np.array(Image.open(ss / "depth" / f"{stem}.tiff")).astype(np.float32) * 1e-3
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127

    ys, xs = np.where(mask)
    pad = 80
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)

    rgb_c = rgb[y0:y1, x0:x1].copy()
    depth_c = depth[y0:y1, x0:x1].copy()
    mask_c = mask[y0:y1, x0:x1].astype(np.uint8)

    # colormap depth (valid range)
    v = depth_c.copy(); v[~np.isfinite(v)] = 0
    lo, hi = np.percentile(v[v > 0], [2, 98]) if (v > 0).any() else (0, 1)
    vn = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    depth_rgb = cv2.applyColorMap((vn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
    depth_rgb[v <= 0] = 0

    # mask contour
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for img in (rgb_c, depth_rgb):
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    # depth silhouette: threshold near the object's median depth, contour it
    obj_med = np.median(depth_c[(mask_c > 0) & (depth_c > 0)])
    near = ((np.abs(depth_c - obj_med) < 0.06) & (depth_c > 0)).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dcnts, _ = cv2.findContours(near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dcnts = [c for c in dcnts if cv2.contourArea(c) > 200]
    cv2.drawContours(rgb_c, dcnts, -1, (255, 0, 255), 2)   # depth silhouette = magenta

    panel = np.concatenate([rgb_c, depth_rgb], axis=1)
    out = dbg / "DIAG_rgbd_align_GREENmaskRGB__leftRGB_rightDEPTH__MAGENTAdepthSil.png"
    Image.fromarray(panel).save(out)
    print(f"object median depth = {obj_med:.3f} m")
    print(f"crop = rows {y0}:{y1} cols {x0}:{x1}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
