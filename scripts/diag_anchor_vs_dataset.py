#!/usr/bin/env python3
"""Does the mask align with the ANCHOR depth (the frame it was segmented on,
static0_full_depth_meters.tiff) but NOT with the dataset frame phase-0b uses
(depth_filenames[cached_train[-1].image_idx])?

Back-project the mask through each depth in CAMERA frame and report compactness
(a clean object => small lateral extent + tight depth; a misaligned pairing =>
table bleed => large extent). Also find which dataset frame best matches the
anchor depth, and how far off the best match is.

Usage: diag_anchor_vs_dataset.py <dataset_dir>
"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image


def backproj_cam(depth_m, mask, fx, fy, cx, cy, mad=True):
    ys, xs = np.where(mask & np.isfinite(depth_m) & (depth_m > 1e-4))
    z = depth_m[ys, xs]
    if mad and z.size >= 10:
        med = np.median(z); m = np.median(np.abs(z - med)) + 1e-6
        k = np.abs(z - med) < 5.0 * 1.4826 * m
        ys, xs, z = ys[k], xs[k], z[k]
    x = (xs - cx) / fx * z; y = (ys - cy) / fy * z
    return np.stack([x, y, z], -1)


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127

    anchor = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    ai = json.load(open(art / "static0_full_intrinsics.json"))

    def report(name, depth, fx_, fy_, cx_, cy_):
        p = backproj_cam(depth, mask, fx_, fy_, cx_, cy_)
        if len(p) == 0:
            print(f"  {name:28s}: no valid depth under mask"); return
        ext = (p.max(0) - p.min(0)) * 100
        zr = np.percentile(p[:, 2], [5, 95])
        print(f"  {name:28s}: N={len(p):>6}  extent(cm)={np.round(ext,1)}  "
              f"z5-95={zr[0]:.3f}-{zr[1]:.3f}m  zspread={(zr[1]-zr[0])*100:.1f}cm")

    print("back-project mask in CAMERA frame (compact = aligned, big = table bleed):")
    report("ANCHOR depth (mask's frame)", anchor, ai["fx"], ai["fy"], ai["cx"], ai["cy"])

    # dataset frames: which matches anchor best, and the last frame phase-0b likely used
    best, berr = None, 1e18
    for i, fr in enumerate(tj["frames"]):
        dp = ss / "depth" / f"{Path(fr['file_path']).stem}.tiff"
        if not dp.exists():
            continue
        d = np.array(Image.open(dp)).astype(np.float32) * 1e-3
        m = np.isfinite(anchor) & (anchor > 0) & (d > 0)
        if m.sum() < 1000:
            continue
        e = float(np.mean(np.abs(d[m] - anchor[m])))
        if e < berr:
            berr, best = e, i
    print(f"\nbest dataset frame vs anchor depth: idx={best} "
          f"({Path(tj['frames'][best]['file_path']).stem})  mean|Δ|={berr*100:.2f} cm "
          f"(0 => identical; large => anchor is NOT a dataset frame)")

    last = len(tj["frames"]) - 1
    for i in sorted(set([best, last])):
        d = np.array(Image.open(ss / "depth" / f"{Path(tj['frames'][i]['file_path']).stem}.tiff")).astype(np.float32) * 1e-3
        report(f"dataset frame {i} depth", d, fx, fy, cx, cy)


if __name__ == "__main__":
    main()
