"""Final discriminator: how much of the ~4mm sheet thickness is a STATIC extrinsic
(rigidly fixable) vs an irreducible viewpoint-dependent depth-bias FLOOR.

Two tests on a flat table patch (sheet thickness = RMS signed point-to-plane dist):

  TEST 1 (static-segment): fuse only a near-ZERO-motion run of frames.
    - If still ~4mm  -> motion-independent (extrinsic/convention/depth-bias), NOT sync.
    - If ~noise floor -> would have meant time-sync (already ruled out by speed test).

  TEST 2 (single global T_fix on the camera->pose extrinsic): grid/optimize one small
    rigid right-multiply applied to EVERY c2w, minimizing sheet thickness.
    - If it knocks thickness down a lot -> that share is a static extrinsic (apply T_fix, done).
    - If it barely moves -> residual is non-rigid depth bias (accept / TSDF-average it).

CPU only.
"""
import json
import os
import sys
import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import minimize

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/ZED/zed_validate2"
Z_MAX, Z_MIN = 2.0, 0.05


def load():
    j = json.load(open(os.path.join(DATA, "transforms.json")))
    return j["fl_x"], j["fl_y"], j["cx"], j["cy"], j["frames"]


def bp_cam(frame, fx, fy, cx, cy, sub=20000):
    """masked points in CAMERA (OpenGL) frame."""
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
    pts = np.stack([x, y, -z, np.ones_like(z)], 1).astype(np.float64)
    if len(pts) > sub:
        pts = pts[np.random.default_rng(0).choice(len(pts), sub, replace=False)]
    return pts


def se3(p):
    """6-vec (rx,ry,rz, tx,ty,tz) -> 4x4. small-angle ok, use Rodrigues for safety."""
    R, _ = cv2.Rodrigues(np.array(p[:3], dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p[3:]
    return T


def fuse_world(frames_idx, frames, fx, fy, cx, cy, Tfix=np.eye(4)):
    P = []
    for i in frames_idx:
        pc = bp_cam(frames[i], fx, fy, cx, cy)
        if pc is None:
            continue
        c2w = np.array(frames[i]["transform_matrix"], dtype=np.float64) @ Tfix
        P.append((c2w @ pc.T).T[:, :3])
    return np.concatenate(P, 0) if P else None


def plane_thickness(pts):
    """fit dominant plane via Open3D RANSAC, return RMS signed dist of inliers (m)."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    plane, inl = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=400)
    a, b, c, d = plane
    nrm = np.array([a, b, c])
    nn = np.linalg.norm(nrm)
    sd = (pts[inl] @ nrm + d) / nn
    return np.sqrt((sd ** 2).mean()), len(inl)


def main():
    fx, fy, cx, cy, frames = load()
    n = len(frames)
    cams = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    speed = np.array([np.linalg.norm(cams[min(n-1,i+1)] - cams[max(0,i-1)]) /
                      max(1, min(n-1,i+1)-max(0,i-1)) for i in range(n)])

    # ---- TEST 1: static-segment ----
    # find the longest run of >=8 consecutive frames whose speed is in the slowest 20%
    thr = np.quantile(speed, 0.20)
    slow = speed <= thr
    best, cur = [], []
    for i in range(n):
        if slow[i]:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    static_idx = best if len(best) >= 6 else list(np.where(slow)[0])
    print(f"[test1] static segment: {len(static_idx)} frames, "
          f"median speed {np.median(speed[static_idx])*1000:.2f} mm/frame "
          f"(vs whole-traj median {np.median(speed)*1000:.2f})")
    Ps = fuse_world(static_idx, frames, fx, fy, cx, cy)
    th_s, ni_s = plane_thickness(Ps)
    print(f"[test1] STATIC-CLIP table thickness = {th_s*1000:.2f} mm ({ni_s} inliers)")
    print("        -> ~4mm here means motion-INDEPENDENT (extrinsic/convention/depth-bias).\n")

    # baseline on a broad sample for the global-fix test
    sample = list(range(0, n, 3))
    Pb = fuse_world(sample, frames, fx, fy, cx, cy)
    th0, ni0 = plane_thickness(Pb)
    print(f"[test2] baseline thickness (every-3rd frame) = {th0*1000:.2f} mm ({ni0} inliers)")

    # ---- TEST 2: single global T_fix (right-multiply every c2w) ----
    # cache camera-frame clouds for the sample once (the optimize only rotates/translates them)
    cache = []
    for i in sample:
        pc = bp_cam(frames[i], fx, fy, cx, cy, sub=8000)
        if pc is not None:
            cache.append((np.array(frames[i]["transform_matrix"], dtype=np.float64), pc))

    def thickness_for(p):
        Tf = se3(p)
        P = np.concatenate([(c2w @ Tf @ pc.T).T[:, :3] for c2w, pc in cache], 0)
        th, _ = plane_thickness(P)
        return th

    res = minimize(lambda p: thickness_for(p) * 1000.0, x0=np.zeros(6),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 300})
    p = res.x
    print(f"[test2] best global T_fix: rot(deg)=[{np.degrees(p[0]):.3f},{np.degrees(p[1]):.3f},"
          f"{np.degrees(p[2]):.3f}] trans(mm)=[{p[3]*1000:.2f},{p[4]*1000:.2f},{p[5]*1000:.2f}]")
    print(f"[test2] thickness  before {th0*1000:.2f} mm  ->  after {res.fun:.2f} mm "
          f"(reduced {(th0*1000-res.fun):.2f} mm, {100*(1-res.fun/(th0*1000)):.0f}%)")
    print("\n[verdict] big reduction -> static extrinsic, apply T_fix. "
          "small reduction -> residual is non-rigid depth bias (TSDF-average / accept).")


if __name__ == "__main__":
    main()
