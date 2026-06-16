"""Confirm the ~4mm error is a recoverable per-frame rigid misalignment.

For several frame PAIRS that overlap, run point-to-plane ICP from the VIO-pose
relative transform. If ICP finds a non-trivial correction (~mm) that meaningfully
TIGHTENS the overlap (fitness up, inlier RMSE down), then per-frame ICP refinement
is the right fix and the residual is recoverable. Also report whether the recovered
corrections are CONSISTENT in direction (-> static extrinsic) or random (-> per-frame).

CPU only.
"""
import json
import os
import sys
import numpy as np
import cv2
import open3d as o3d

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/ZED/zed_validate2"
Z_MAX, Z_MIN = 2.0, 0.05


def load():
    j = json.load(open(os.path.join(DATA, "transforms.json")))
    return j["fl_x"], j["fl_y"], j["cx"], j["cy"], j["frames"]


def cloud_cam(frame, fx, fy, cx, cy, voxel=0.005):
    """Pointcloud in CAMERA frame (OpenGL), masked, voxel-downsampled, with normals."""
    depth = cv2.imread(os.path.join(DATA, frame["depth_file_path"]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = depth.astype(np.float32) / 1000.0
    valid = (depth > Z_MIN) & (depth < Z_MAX)
    mpath = os.path.join(DATA, frame.get("mask_path", ""))
    if os.path.exists(mpath):
        m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if m is not None and m.shape == depth.shape:
            valid &= m > 0
    vs, us = np.where(valid)
    z = depth[vs, us]
    x = (us - cx) / fx * z
    y = -(vs - cy) / fy * z
    pts = np.stack([x, y, -z], axis=1).astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return pc


def main():
    fx, fy, cx, cy, frames = load()
    n = len(frames)
    cams = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])

    # pick pairs separated by a moderate baseline (enough overlap, real parallax)
    pairs = []
    for a in range(0, n - 20, max(1, n // 8)):
        # find b ahead of a with ~3-8 cm baseline
        for b in range(a + 5, min(n, a + 40)):
            d = np.linalg.norm(cams[b] - cams[a])
            if 0.03 <= d <= 0.10:
                pairs.append((a, b))
                break
    print(f"[diag] testing {len(pairs)} overlapping pairs\n")

    print(f"{'pair':>10} | {'init RMSE(mm)':>12} | {'icp RMSE(mm)':>12} | "
          f"{'init fit':>8} | {'icp fit':>8} | {'corr trans(mm)':>14} | {'corr rot(deg)':>12}")
    print("-" * 100)

    corr_trans = []
    for a, b in pairs:
        pa = cloud_cam(frames[a], fx, fy, cx, cy)
        pb = cloud_cam(frames[b], fx, fy, cx, cy)
        if pa is None or pb is None or len(pa.points) < 500 or len(pb.points) < 500:
            continue
        Ta = np.array(frames[a]["transform_matrix"], dtype=np.float64)
        Tb = np.array(frames[b]["transform_matrix"], dtype=np.float64)
        # transform that maps b-cam points into a-cam frame, per VIO: inv(Ta) @ Tb
        T_init = np.linalg.inv(Ta) @ Tb
        # eval initial alignment
        ev0 = o3d.pipelines.registration.evaluate_registration(pb, pa, 0.02, T_init)
        reg = o3d.pipelines.registration.registration_icp(
            pb, pa, 0.02, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        # the correction ICP applied on top of VIO:
        dT = reg.transformation @ np.linalg.inv(T_init)
        dt_mm = np.linalg.norm(dT[:3, 3]) * 1000
        ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
        corr_trans.append(dT[:3, 3])
        print(f"{f'{a}->{b}':>10} | {ev0.inlier_rmse*1000:>12.2f} | {reg.inlier_rmse*1000:>12.2f} | "
              f"{ev0.fitness:>8.3f} | {reg.fitness:>8.3f} | {dt_mm:>14.2f} | {ang:>12.3f}")

    if corr_trans:
        ct = np.array(corr_trans)
        mean_dir = ct.mean(0)
        consistency = np.linalg.norm(mean_dir) / (np.linalg.norm(ct, axis=1).mean() + 1e-9)
        print(f"\n[diag] correction direction consistency = {consistency:.2f} "
              f"(1.0 = all same direction -> STATIC extrinsic; ~0 = random -> per-frame)")
        print(f"       mean correction vector (mm): "
              f"[{mean_dir[0]*1000:.2f}, {mean_dir[1]*1000:.2f}, {mean_dir[2]*1000:.2f}]")
    print("\n[interpret] icp RMSE << init RMSE and fitness up -> per-frame ICP refinement recovers it.")


if __name__ == "__main__":
    main()
