#!/usr/bin/env python3
"""Reproduce phase-0's masked target and pin down the frame/depth used.

1. Which dataset frame's depth matches the saved anchor depth
   (static0_full_depth_meters.tiff)?  -> the true anchor frame.
2. Back-project the saved mask through that frame's depth+pose (= exact
   backproject_mask_to_world) and compare its centroid to the saved
   target_reg_ref.ply and to the seed near that location.

Usage: diag_target_repro.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    frames = tj["frames"]

    anchor_depth = np.array(Image.open(art / "static0_full_depth_meters.tiff")).astype(np.float32)
    a_valid = np.isfinite(anchor_depth) & (anchor_depth > 0)
    print(f"anchor depth (static0_full_depth_meters.tiff): shape={anchor_depth.shape} "
          f"valid_px={a_valid.sum()} range=[{anchor_depth[a_valid].min():.3f},{anchor_depth[a_valid].max():.3f}] m")

    # Which dataset depth file matches the anchor depth? (compare on commonly-valid px)
    best, best_err = None, 1e18
    for i, fr in enumerate(frames):
        stem = Path(fr["file_path"]).stem
        dp = ss / "depth" / f"{stem}.tiff"
        if not dp.exists():
            continue
        d = np.array(Image.open(dp)).astype(np.float32) * 1e-3
        if d.shape != anchor_depth.shape:
            continue
        m = a_valid & np.isfinite(d) & (d > 0)
        if m.sum() < 1000:
            continue
        err = float(np.mean(np.abs(d[m] - anchor_depth[m])))
        if err < best_err:
            best_err, best = err, i
    if best is None:
        print("no matching dataset depth frame found"); return
    print(f"closest dataset frame to anchor depth: idx={best} "
          f"({Path(frames[best]['file_path']).stem})  mean|Δ|={best_err:.5f} m\n")

    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127
    print(f"mask: shape={mask.shape}  px={mask.sum()}")

    def backproj(depth_m, c2w, msk):
        H, W = depth_m.shape
        if msk.shape != (H, W):
            msk = np.array(Image.fromarray(msk.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        ys, xs = np.where(msk & (depth_m > 1e-4))
        z = depth_m[ys, xs]
        # MAD scrub (same as backproject_mask_to_world)
        if z.size >= 10:
            med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
            keep = np.abs(z - med) < 5.0 * 1.4826 * mad
            ys, xs, z = ys[keep], xs[keep], z[keep]
        x = (xs - cx) / fx * z
        y = -(ys - cy) / fy * z
        p = np.stack([x, y, -z], -1)
        R, t = c2w[:3, :3], c2w[:3, 3]
        return p @ R.T + t

    tgt = np.asarray(o3d.io.read_point_cloud(str(art / "static0_obj_00_sam3d_target_reg_ref.ply")).points)
    seed = o3d.io.read_point_cloud(str(ss / "depth_camera_init_points.ply"))
    seed_pts = np.asarray(seed.points)

    print(f"\n{'frame':>6} {'centroid (masked backproj)':>34}  {'vs target_reg_ref':>18}")
    for i in sorted(set([best, len(frames) - 1, len(frames) - 2, 0])):
        stem = Path(frames[i]["file_path"]).stem
        dp = ss / "depth" / f"{stem}.tiff"
        if not dp.exists():
            continue
        d = np.array(Image.open(dp)).astype(np.float32) * 1e-3
        c2w = np.array(frames[i]["transform_matrix"], np.float64)
        p = backproj(d, c2w, mask)
        c = p.mean(0)
        dvst = np.linalg.norm(c - tgt.mean(0)) * 100
        print(f"{i:>6} {str(np.round(c,4)):>34}  {dvst:>14.2f} cm")

    # Reference: seed near the target.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(tgt)
    d_seed, _ = nn.kneighbors(seed_pts)
    seed_near = seed_pts[d_seed[:, 0] < 0.08]
    print(f"\n target_reg_ref centroid : {np.round(tgt.mean(0),4)}  (N={len(tgt)})")
    print(f" seed<8cm of target      : {np.round(seed_near.mean(0),4)}  (N={len(seed_near)})")
    print(f" target vs seed<8cm      : {np.linalg.norm(tgt.mean(0)-seed_near.mean(0))*100:.2f} cm")


if __name__ == "__main__":
    main()
