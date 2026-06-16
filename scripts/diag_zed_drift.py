"""Diagnose whether zed_validate2 cloud smear is POSE DRIFT or per-frame GEOMETRY noise.

Idea: back-project every frame's masked depth into world via its VIO pose, then ask
how thick a flat surface is when reconstructed from a SHORT trajectory window vs the
WHOLE trajectory.

  - If thickness GROWS with window length  -> accumulating POSE drift (VIO error).
  - If thickness is FLAT regardless of window -> per-frame geometry / depth noise / no-parallax.

We measure "thickness" two ways, both pose-independent of any single reference frame:
  (A) Single-frame self-consistency: a frame's own depth vs a tiny local plane fit
      (already known sub-mm) — the noise floor.
  (B) Cross-frame agreement: for a fixed world voxel, the std of points landing in it,
      computed over increasing numbers of contributing frames. This is the drift signal.

CPU-only (no GPU contention with the running ns-train).
"""
import json
import os
import sys
import numpy as np
import cv2

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/ZED/zed_validate2"
Z_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
Z_MIN = 0.05


def load():
    j = json.load(open(os.path.join(DATA, "transforms.json")))
    fx, fy = j["fl_x"], j["fl_y"]
    cx, cy = j["cx"], j["cy"]
    frames = j["frames"]
    return fx, fy, cx, cy, frames


def backproject(frame, fx, fy, cx, cy):
    """Return (N,3) world points (OpenGL c2w) from masked depth, plus the pose."""
    dpath = os.path.join(DATA, frame["depth_file_path"])
    mpath = os.path.join(DATA, frame.get("mask_path", ""))
    depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None, None
    depth = depth.astype(np.float32) / 1000.0  # uint16 mm -> m
    h, w = depth.shape
    valid = (depth > Z_MIN) & (depth < Z_MAX)
    if os.path.exists(mpath):
        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.shape == depth.shape:
            valid &= mask > 0
    vs, us = np.where(valid)
    z = depth[vs, us]
    # OpenGL camera convention (matches zed_svo_to_dataset back-projection):
    #   x=(u-cx)/fx*z, y=-(v-cy)/fy*z, z_cam=-z
    x = (us - cx) / fx * z
    y = -(vs - cy) / fy * z
    zc = -z
    pts_cam = np.stack([x, y, zc, np.ones_like(z)], axis=1)  # (N,4)
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    pts_w = (c2w @ pts_cam.T).T[:, :3]
    return pts_w.astype(np.float32), c2w


def voxel_thickness(points, frame_ids, voxel_m, min_frames):
    """For voxels touched by >= min_frames distinct frames, measure spread of points
    within the voxel along the local normal proxy: use per-voxel 3D std magnitude.
    Returns median per-voxel RMS spread (m) and how many voxels qualified."""
    keys = np.floor(points / voxel_m).astype(np.int64)
    # hash voxel keys
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    pts_s = points[order]
    fid_s = frame_ids[order]
    # find voxel boundaries
    diff = np.any(np.diff(keys_s, axis=0) != 0, axis=1)
    bounds = np.concatenate([[0], np.where(diff)[0] + 1, [len(keys_s)]])
    spreads = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 4:
            continue
        nfr = len(np.unique(fid_s[a:b]))
        if nfr < min_frames:
            continue
        p = pts_s[a:b]
        c = p.mean(axis=0)
        # RMS distance to centroid = isotropic spread; thin surface -> dominated by
        # the across-surface axis, which is exactly what drift inflates.
        rms = np.sqrt(((p - c) ** 2).sum(axis=1).mean())
        spreads.append(rms)
    spreads = np.array(spreads)
    return (np.median(spreads) if len(spreads) else np.nan), len(spreads)


def main():
    fx, fy, cx, cy, frames = load()
    n = len(frames)
    print(f"[diag] {n} frames, z-cap {Z_MAX} m\n")

    # Accumulate points + frame ids over the whole trajectory, but ALSO snapshot at
    # increasing window lengths to see if thickness grows with trajectory length.
    windows = [10, 25, 50, 100, 237]
    windows = [w for w in windows if w <= n] + ([n] if n not in windows else [])
    windows = sorted(set(windows))

    # subsample frames evenly for speed but keep order
    step = 1
    sel = list(range(0, n, step))

    all_pts = []
    all_fid = []
    for k, fi in enumerate(sel):
        pts, c2w = backproject(frames[fi], fx, fy, cx, cy)
        if pts is None or len(pts) == 0:
            continue
        # subsample points per frame to keep memory sane
        if len(pts) > 40000:
            idx = np.random.default_rng(fi).choice(len(pts), 40000, replace=False)
            pts = pts[idx]
        all_pts.append(pts)
        all_fid.append(np.full(len(pts), fi, dtype=np.int64))

    print(f"[diag] back-projected {len(all_pts)} frames\n")
    print(f"{'window(frames)':>14} | {'median voxel RMS spread (mm)':>30} | {'n voxels':>9}")
    print("-" * 62)
    for w in windows:
        # frames 0..w
        keep = [i for i, fi in enumerate(sel) if fi < w]
        if not keep:
            continue
        P = np.concatenate([all_pts[i] for i in keep], axis=0)
        F = np.concatenate([all_fid[i] for i in keep], axis=0)
        spread, nv = voxel_thickness(P, F, voxel_m=0.01, min_frames=3)
        print(f"{w:>14} | {spread * 1000:>30.2f} | {nv:>9}")

    print("\n[interpret] spread GROWS with window -> pose drift dominates.")
    print("            spread FLAT across windows -> per-frame geometry/noise dominates.")


if __name__ == "__main__":
    main()
