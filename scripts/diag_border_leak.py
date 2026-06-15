#!/usr/bin/env python3
"""Quantify mask-border table leakage in the registration target (now that the
anchor frame is consistent), and test a GENERAL, scale-invariant fix:
depth-discontinuity rejection — drop mask pixels whose local depth window-range
exceeds a threshold (the object->table step). Compare to plain erosion.

Usage: diag_border_leak.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter, binary_erosion


def backproj(mask, depth_m, fx, fy, cx, cy):
    ys, xs = np.where(mask & np.isfinite(depth_m) & (depth_m > 1e-4))
    z = depth_m[ys, xs]
    x = (xs - cx) / fx * z; y = (ys - cy) / fy * z
    return np.stack([x, y, z], -1)


def table_fraction(p):
    if len(p) < 50:
        return 0.0, len(p)
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(p.astype(np.float64))
    _, inl = pc.segment_plane(0.006, 3, 400)
    return len(inl) / len(p), len(p)


def depth_discontinuity_keep(mask, depth_m, win=7, max_range_m=0.015):
    """True where the pixel is mask & NOT near a depth step. Robust to NaN:
    fill invalid with the local median before computing the window range."""
    valid = np.isfinite(depth_m) & (depth_m > 1e-4)
    d = np.where(valid, depth_m, np.nan).astype(np.float32)
    # local range on valid depth: max-min over the window, NaN-aware via filling
    filled = np.where(valid, d, 0.0)
    big = np.where(valid, d, -1e9)
    small = np.where(valid, d, 1e9)
    lo = minimum_filter(small, size=win)
    hi = maximum_filter(big, size=win)
    rng = hi - lo
    # also require the pixel itself + enough valid neighbours
    near_edge = rng > max_range_m
    keep = mask & valid & (~near_edge)
    return keep


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    ai = json.load(open(art / "static0_full_intrinsics.json"))
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    fx, fy, cx, cy = ai["fx"], ai["fy"], ai["cx"], ai["cy"]

    raw = backproj(mask, depth, fx, fy, cx, cy)
    tf_raw, n_raw = table_fraction(raw)
    print(f"RAW target (anchor mask+depth):       N={n_raw:>6}  table-plane%={100*tf_raw:.0f}")

    # Fix candidates
    for win, thr in [(5, 0.010), (7, 0.015), (9, 0.020)]:
        keep = depth_discontinuity_keep(mask, depth, win=win, max_range_m=thr)
        p = backproj(keep, depth, fx, fy, cx, cy)
        tf, n = table_fraction(p)
        print(f"  depth-disc win={win} thr={int(thr*1000)}mm:  N={n:>6}  table%={100*tf:>3.0f}  "
              f"kept={100*n/max(n_raw,1):.0f}%")

    for er in (3, 6, 10):
        em = binary_erosion(mask, iterations=er)
        p = backproj(em, depth, fx, fy, cx, cy)
        tf, n = table_fraction(p)
        print(f"  erosion {er}px:                    N={n:>6}  table%={100*tf:>3.0f}  kept={100*n/max(n_raw,1):.0f}%")

    # write PLY for the recommended setting (depth-disc win7 15mm) for the user to judge
    keep = depth_discontinuity_keep(mask, depth, win=7, max_range_m=0.015)
    dropped = mask & ~keep & np.isfinite(depth) & (depth > 1e-4)
    pk = backproj(keep, depth, fx, fy, cx, cy)
    pd = backproj(dropped, depth, fx, fy, cx, cy)
    def col(a, c): return np.tile(np.array(c, float) / 255.0, (len(a), 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.concatenate([pk, pd]))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate([col(pk, (0, 220, 0)), col(pd, (255, 0, 0))]))
    out = dbg / "DIAG_borderleak_keptGREEN_droppedRED.ply"
    o3d.io.write_point_cloud(str(out), pc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
