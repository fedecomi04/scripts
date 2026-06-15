#!/usr/bin/env python3
"""Confirm: the seed is in online_fusion's per-frame ICP-refined frame, but
phase-0 back-projects the mask with the RAW transforms.json pose. The ICP
correction is mostly a small ROTATION, which barely moves the full-cloud
centroid but swings the OFF-CENTER object several cm.

For the anchor frame (last): ICP-align its full raw-pose cloud to the seed,
report rotation+translation of the correction, then apply that correction to
the masked target and measure how much the seed gap closes.

Usage: diag_confirm_icp.py <dataset_dir>
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
    frames = tj["frames"]
    i = len(frames) - 1
    stem = Path(frames[i]["file_path"]).stem
    c2w = np.array(frames[i]["transform_matrix"], np.float64)

    depth = np.array(Image.open(ss / "depth" / f"{stem}.tiff")).astype(np.float32) * 1e-3
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127

    def backproj(msk):
        H, W = depth.shape
        m = msk
        ys, xs = np.where(m & (depth > 1e-4))
        z = depth[ys, xs]
        x = (xs - cx) / fx * z; y = -(ys - cy) / fy * z
        p = np.stack([x, y, -z], -1)
        return p @ c2w[:3, :3].T + c2w[:3, 3]

    full = backproj(np.ones_like(mask, bool))
    tgt = backproj(mask)

    seed = o3d.io.read_point_cloud(str(ss / "depth_camera_init_points.ply"))
    seed_pts = np.asarray(seed.points)
    seed_ds = seed.voxel_down_sample(0.004)

    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(full)
    pc = pc.voxel_down_sample(0.004)
    reg = o3d.pipelines.registration.registration_icp(
        pc, seed_ds, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
    T = reg.transformation
    R = T[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"anchor frame {stem}: ICP(full raw cloud -> seed) fitness={reg.fitness:.3f}")
    print(f"  correction rotation = {ang:.2f} deg   translation = {np.linalg.norm(T[:3,3])*100:.2f} cm\n")

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(seed_pts)

    def gap(p):
        d, _ = nn.kneighbors(p)
        return float(np.median(d[:, 0])) * 100   # median dist to seed, cm

    tgt_h = np.c_[tgt, np.ones(len(tgt))]
    tgt_corr = (tgt_h @ T.T)[:, :3]
    print(f"  masked target -> seed   median dist  BEFORE correction = {gap(tgt):.2f} cm")
    print(f"  masked target -> seed   median dist  AFTER  correction = {gap(tgt_corr):.2f} cm")
    print(f"  centroid shift from applying correction = {np.linalg.norm(tgt_corr.mean(0)-tgt.mean(0))*100:.2f} cm")


if __name__ == "__main__":
    main()
