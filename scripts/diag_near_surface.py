#!/usr/bin/env python3
"""Test the LOCAL NEAR-SURFACE filter for mask-border table leakage:
drop mask pixels whose depth exceeds the local-window MINIMUM depth by > delta.
The object's near surface ~= local min (kept); table behind/around it at the
border is > local_min + delta (dropped). Scale-invariant, keeps object interior
and thin parts. Writes kept(green)/dropped(red) PLY for the user to judge.

Usage: diag_near_surface.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.ndimage import minimum_filter


def backproj(mask, depth_m, fx, fy, cx, cy):
    ys, xs = np.where(mask & np.isfinite(depth_m) & (depth_m > 1e-4))
    z = depth_m[ys, xs]
    x = (xs - cx) / fx * z; y = (ys - cy) / fy * z
    return np.stack([x, y, z], -1), ys, xs


def near_surface_keep(mask, depth_m, win_px, delta_m):
    valid = np.isfinite(depth_m) & (depth_m > 1e-4)
    dfar = np.where(valid, depth_m, np.inf).astype(np.float32)
    local_min = minimum_filter(dfar, size=win_px, mode="nearest")
    keep = mask & valid & ((depth_m - local_min) <= delta_m)
    return keep


def main():
    ddir = Path(sys.argv[1])
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    ai = json.load(open(art / "static0_full_intrinsics.json"))
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    fx, fy, cx, cy = ai["fx"], ai["fy"], ai["cx"], ai["cy"]
    H, W = depth.shape
    win = max(11, int(round(0.02 * min(H, W))) | 1)  # ~2% short side, odd

    raw, _, _ = backproj(mask, depth, fx, fy, cx, cy)
    ext_raw = (raw.max(0) - raw.min(0)) * 100
    print(f"{ddir.name}: win={win}px")
    print(f"  RAW            N={len(raw):>6}  extent(cm)={np.round(ext_raw,1)}")
    for delta in (0.010, 0.015, 0.025):
        keep = near_surface_keep(mask, depth, win, delta)
        p, _, _ = backproj(keep, depth, fx, fy, cx, cy)
        ext = (p.max(0) - p.min(0)) * 100
        print(f"  delta={int(delta*1000):>2}mm   N={len(p):>6}  extent(cm)={np.round(ext,1)}  "
              f"dropped={len(raw)-len(p):>5} ({100*(len(raw)-len(p))/len(raw):.0f}%)")

    # PLY at delta=15mm for judging
    keep = near_surface_keep(mask, depth, win, 0.015)
    dropped = mask & ~keep & np.isfinite(depth) & (depth > 1e-4)
    pk, _, _ = backproj(keep, depth, fx, fy, cx, cy)
    pd, _, _ = backproj(dropped, depth, fx, fy, cx, cy)
    def col(a, c): return np.tile(np.array(c, float) / 255.0, (len(a), 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.concatenate([pk, pd]))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate([col(pk, (0, 220, 0)), col(pd, (255, 0, 0))]))
    out = dbg / "DIAG_nearsurf_keptGREEN_droppedRED_delta15mm.ply"
    o3d.io.write_point_cloud(str(out), pc)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
