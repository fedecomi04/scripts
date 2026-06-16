"""Distinguish a TIME-SYNC lag from a STATIC extrinsic for the ~4mm constant pose error.

Key discriminator:
  - A time-sync lag (depth stamp vs pose stamp) produces a reprojection offset that
    scales with CAMERA SPEED: offset ~= v * dt. Fast-moving frames smear MORE.
  - A static extrinsic / convention error produces a FIXED rigid offset independent of
    speed: fast and slow frames smear the same.

Method: bin frames by inter-frame camera translation speed (from consecutive VIO
poses). For each speed bin, fuse only those frames and measure voxel RMS spread.
  - spread RISES with speed -> time-sync lag dominates (correctable by a dt shift).
  - spread FLAT vs speed     -> static extrinsic/convention (correctable by one rigid T).

CPU only.
"""
import json
import os
import sys
import numpy as np
import cv2

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/ZED/zed_validate2"
Z_MAX = 2.0
Z_MIN = 0.05


def load():
    j = json.load(open(os.path.join(DATA, "transforms.json")))
    return j["fl_x"], j["fl_y"], j["cx"], j["cy"], j["frames"]


def backproject(frame, fx, fy, cx, cy):
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
    pts_cam = np.stack([x, y, -z, np.ones_like(z)], axis=1)
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    pts_w = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)
    if len(pts_w) > 40000:
        idx = np.random.default_rng(0).choice(len(pts_w), 40000, replace=False)
        pts_w = pts_w[idx]
    return pts_w


def voxel_spread(points, frame_ids, voxel_m=0.01, min_frames=3):
    keys = np.floor(points / voxel_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s, pts_s, fid_s = keys[order], points[order], frame_ids[order]
    diff = np.any(np.diff(keys_s, axis=0) != 0, axis=1)
    bounds = np.concatenate([[0], np.where(diff)[0] + 1, [len(keys_s)]])
    sp = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 4:
            continue
        if len(np.unique(fid_s[a:b])) < min_frames:
            continue
        p = pts_s[a:b]
        sp.append(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
    sp = np.array(sp)
    return (np.median(sp) if len(sp) else np.nan), len(sp)


def main():
    fx, fy, cx, cy, frames = load()
    n = len(frames)
    cams = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    # inter-frame speed (m per frame) — central difference
    speed = np.zeros(n)
    for i in range(n):
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        speed[i] = np.linalg.norm(cams[b] - cams[a]) / max(1, b - a)
    print(f"[diag] {n} frames. inter-frame translation (mm/frame): "
          f"min {speed.min()*1000:.2f} med {np.median(speed)*1000:.2f} max {speed.max()*1000:.2f}\n")

    # bin frames into slow / medium / fast thirds by speed
    qs = np.quantile(speed, [1/3, 2/3])
    bins = {
        "SLOW  (bottom third)": np.where(speed <= qs[0])[0],
        "MED   (middle third)": np.where((speed > qs[0]) & (speed <= qs[1]))[0],
        "FAST  (top third)":    np.where(speed > qs[1])[0],
    }

    # cache back-projections
    bp = {}
    for i in range(n):
        p = backproject(frames[i], fx, fy, cx, cy)
        if p is not None and len(p):
            bp[i] = p

    print(f"{'speed bin':>22} | {'med inter-frame mm':>18} | {'voxel RMS spread (mm)':>21} | {'n vox':>7}")
    print("-" * 80)
    for name, idxs in bins.items():
        idxs = [i for i in idxs if i in bp]
        if len(idxs) < 4:
            continue
        P = np.concatenate([bp[i] for i in idxs], 0)
        F = np.concatenate([np.full(len(bp[i]), i) for i in idxs], 0)
        spread, nv = voxel_spread(P, F)
        print(f"{name:>22} | {np.median(speed[idxs])*1000:>18.2f} | "
              f"{spread*1000:>21.2f} | {nv:>7}")

    print("\n[interpret] spread RISES with speed -> TIME-SYNC lag (correct by dt shift).")
    print("            spread FLAT vs speed     -> STATIC extrinsic/convention (one rigid T).")


if __name__ == "__main__":
    main()
