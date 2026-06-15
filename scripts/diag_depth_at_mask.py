#!/usr/bin/env python3
"""The 2D mask is clean — so why is the back-projected target half 'table'?
Inspect the depth distribution at the mask pixels and the geometry of the
RANSAC plane (is it really a horizontal table below the object, or a false
slice of the curved object?).

Usage: diag_depth_at_mask.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    fr = tj["frames"][-1]
    c2w = np.array(fr["transform_matrix"], np.float64)
    depth = np.array(Image.open(ss / "depth" / f"{Path(fr['file_path']).stem}.tiff")).astype(np.float32) * 1e-3
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127

    ys, xs = np.where(mask & (depth > 1e-4))
    z = depth[ys, xs]
    print(f"mask valid-depth px = {z.size}  (mask px total = {mask.sum()})")
    print(f"depth at mask: min={z.min():.3f} max={z.max():.3f} median={np.median(z):.3f} m")
    # histogram
    hist, edges = np.histogram(z, bins=20)
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  {e0:.3f}-{e1:.3f} m | {'#'*int(60*h/hist.max()):60s} {h}")

    # world points + RANSAC plane
    x = (xs - cx) / fx * z; y = -(ys - cy) / fy * z
    P = (np.stack([x, y, -z], -1)) @ c2w[:3, :3].T + c2w[:3, 3]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    plane, inl = pc.segment_plane(0.008, 3, 400)
    inl = np.asarray(inl)
    n = np.array(plane[:3]); n /= np.linalg.norm(n)
    print(f"\nRANSAC plane: normal={np.round(n,3)} (|z-up component|={abs(n[2]):.2f})  "
          f"inliers={len(inl)} ({100*len(inl)/len(P):.0f}%)")
    is_p = np.zeros(len(P), bool); is_p[inl] = True
    plane_pts, obj_pts = P[is_p], P[~is_p]
    # vertical (world-Z) separation between 'plane' group and 'object' group
    print(f"  plane group  world-Z mean={plane_pts[:,2].mean():.3f}  extent(cm)={np.round((plane_pts.max(0)-plane_pts.min(0))*100,1)}")
    print(f"  object group world-Z mean={obj_pts[:,2].mean():.3f}  extent(cm)={np.round((obj_pts.max(0)-obj_pts.min(0))*100,1)}")
    # Is the world up-axis Z or Y? check overall scene orientation via gravity-ish:
    print(f"  (note: world axis with table-normal = the axis where the two groups separate)")
    for ax, nm in enumerate("XYZ"):
        sep = abs(plane_pts[:,ax].mean() - obj_pts[:,ax].mean())
        print(f"    axis {nm}: plane_mean-obj_mean = {sep*100:.2f} cm")


if __name__ == "__main__":
    main()
