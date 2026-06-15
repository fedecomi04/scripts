#!/usr/bin/env python3
"""Test: is the phase-0 target offset because transforms.json holds RAW FK poses
while the seed was built with online_fusion's per-frame ICP-refined poses?

For a few frames, back-project the full depth with the RAW transforms.json pose,
ICP-align that cloud to the seed PLY, and report the residual translation. If the
correction grows toward the LAST frame (which phase-0 uses) and reaches ~3 cm,
the offset is a raw-FK-vs-ICP pose-frame mismatch, not intrinsics.

Usage: diag_pose_drift.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def backproject_full(depth_m, fx, fy, cx, cy, c2w):
    H, W = depth_m.shape
    ys, xs = np.where(depth_m > 1e-4)
    z = depth_m[ys, xs]
    # OpenGL convention (matches backproject_mask_to_world).
    x = (xs - cx) / fx * z
    y = -(ys - cy) / fy * z
    zc = -z
    p = np.stack([x, y, zc], -1)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return p @ R.T + t


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    tj = json.load(open(ss / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    frames = tj["frames"]
    n = len(frames)

    seed = o3d.io.read_point_cloud(str(ss / "depth_camera_init_points.ply"))
    seed = seed.voxel_down_sample(0.004)
    print(f"seed: {len(seed.points)} pts (4mm voxel)   n_frames={n}\n")
    print(f"{'frame':>6} {'file':>16} {'icp_fit':>8} {'|t| cm':>8}  t_xyz (cm)")

    probe = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]))
    for i in probe:
        fp = frames[i]["file_path"]
        stem = Path(fp).stem
        dpath = ss / "depth" / f"{stem}.tiff"
        if not dpath.exists():
            print(f"{i:>6} {stem:>16}  (no depth)")
            continue
        depth = np.array(Image.open(dpath)).astype(np.float32) * 1e-3   # mm->m
        c2w = np.array(frames[i]["transform_matrix"], dtype=np.float64)
        pts = backproject_full(depth, fx, fy, cx, cy, c2w)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc = pc.voxel_down_sample(0.004)
        # crop seed to this frame's neighborhood so ICP locks onto the overlap
        reg = o3d.pipelines.registration.registration_icp(
            pc, seed, 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        t = reg.transformation[:3, 3]
        print(f"{i:>6} {stem:>16} {reg.fitness:>8.3f} {np.linalg.norm(t)*100:>8.2f}  {np.round(t*100,2)}")


if __name__ == "__main__":
    main()
