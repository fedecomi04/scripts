#!/usr/bin/env python3
"""Overlay the mask on the ANCHOR's OWN rgb (static0_rgb.png) and depth
(static0_full_depth_meters.tiff). If the mask hugs the object in the RGB but is
shifted from the object in the depth -> RGB/depth are mutually misaligned.

Usage: diag_anchor_align.py <dataset_dir>
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def main():
    ddir = Path(sys.argv[1])
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"

    rgb = np.array(Image.open(dbg / "static0_rgb.png").convert("RGB"))
    depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    print(f"rgb {rgb.shape[:2]}  depth {depth.shape}  mask {mask.shape}")

    ys, xs = np.where(mask)
    pad = 90
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    rgb_c = rgb[y0:y1, x0:x1].copy()
    depth_c = depth[y0:y1, x0:x1].copy()
    mask_c = mask[y0:y1, x0:x1].astype(np.uint8)

    v = depth_c.copy(); v[~np.isfinite(v)] = 0
    val = v[v > 0]
    lo, hi = np.percentile(val, [2, 98])
    vn = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    drgb = cv2.cvtColor(cv2.applyColorMap((vn * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    drgb[v <= 0] = 0

    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for img in (rgb_c, drgb):
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    # depth silhouette near object median
    om = np.median(depth_c[(mask_c > 0) & np.isfinite(depth_c) & (depth_c > 0)])
    near = ((np.abs(depth_c - om) < 0.05) & np.isfinite(depth_c) & (depth_c > 0)).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dc, _ = cv2.findContours(near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(drgb, [c for c in dc if cv2.contourArea(c) > 150], -1, (255, 255, 255), 2)

    panel = np.concatenate([rgb_c, drgb], 1)
    out = dbg / "DIAG_anchor_align_LEFTrgb_RIGHTdepth_GREENmask_WHITEdepthSil.png"
    Image.fromarray(panel).save(out)
    print(f"object median depth = {om:.3f} m   wrote {out}")


if __name__ == "__main__":
    main()
