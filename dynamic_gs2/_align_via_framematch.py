#!/usr/bin/env python
"""Compare recording_15fps tracker estimate vs bag GT, using IMAGE CONTENT to
establish per-frame correspondence (robust to the 'same-scene vs same-motion'
question).

  1. NCC-match every recording RGB frame -> its nearest bag RGB frame.
  2. bag frame -> GT screwdriver pose (bag gt_object_trajectory.csv). So each
     recording frame gets a GT object position on the bag's clock.
  3. Recording tracker estimate: object_track_poses.jsonl, tick k -> recording
     frame (k-1); object world centroid = R_k @ centroid_world + t_k.
  4. Restrict to frames where GT actually moves, Umeyama-align estimate->GT
     (absorbs frame + anchor offset), report mean/max/RMS position error.

If the aligned error is small & consistent -> same teleop, valid number.
If it's large/structured -> different execution; the image match only shared
the static scene.
"""
import glob
import json
import os
import sys

import numpy as np
import cv2

REC = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/recording_15fps_2026-06-11_115107_screwdriver_good_recording"
BAGDS = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver_from_bag_20260611"


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def sig(f, s=200):
    im = cv2.imread(f, 0)
    im = cv2.resize(im, (s, s)).astype(np.float32)
    return ((im - im.mean()) / (im.std() + 1e-6)).ravel()


def main():
    rec_files = sorted(glob.glob(os.path.join(REC, "dynamic_scene/rgb/*.png")))
    bag_files = sorted(glob.glob(os.path.join(BAGDS, "dynamic_scene/rgb/*.png")))
    gt = np.loadtxt(os.path.join(BAGDS, "gt_object_trajectory.csv"), delimiter=",")
    gt_xyz = gt[:, 1:4]
    print(f"recording frames={len(rec_files)}  bag frames={len(bag_files)}  gt rows={len(gt)}")

    # ---- NCC frame match: rec -> bag (matrix) ----
    print("building signatures + matching (image content)...")
    bag_sig = np.array([sig(f) for f in bag_files])          # Nb x D
    rec_sig = np.array([sig(f) for f in rec_files])          # Nr x D
    cc = rec_sig @ bag_sig.T / rec_sig.shape[1]              # Nr x Nb
    match = cc.argmax(1)                                     # rec i -> bag frame
    mscore = cc.max(1)
    # enforce monotonic-ish mapping via median filter (kills outlier matches)
    from scipy.ndimage import median_filter
    match_s = median_filter(match, size=7).astype(int)
    print(f"  match score mean={mscore.mean():.3f}  monotonic frac="
          f"{np.mean(np.diff(match_s) >= 0):.2f}")

    gt_per_rec = gt_xyz[np.clip(match_s, 0, len(gt_xyz) - 1)]   # GT per recording frame

    # ---- estimate per recording frame ----
    anchor = None
    est = {}
    seg = 0
    for ln in open(os.path.join(REC, "object_track_poses.jsonl")):
        o = json.loads(ln)
        if o.get("type") == "anchor":
            seg = o.get("segment", 0)
            if anchor is None:
                anchor = np.array(o["centroid_world"], float)
        elif o.get("type") == "pose" and seg == 0 and o.get("ok", True):
            R = np.array(o["R"], float).reshape(3, 3)
            t = np.array(o["t"], float)
            est[o["tick"] - 1] = R @ anchor + t              # tick k -> rec frame k-1
    fr = sorted(f for f in est if 0 <= f < len(rec_files))
    E = np.array([est[f] for f in fr])
    Ggt = gt_per_rec[fr]
    print(f"matched estimate frames: {len(fr)}")

    # ---- restrict to frames where GT moves ----
    d = np.linalg.norm(np.diff(gt_per_rec, axis=0), axis=1)
    sm = np.convolve(d, np.ones(5) / 5, mode="same")
    moving_frames = set(np.where(sm > 0.1 * sm.max())[0])
    mv = np.array([i for i, f in enumerate(fr) if f in moving_frames])
    print(f"moving matched frames: {len(mv)}  "
          f"GT net over moving {np.linalg.norm(Ggt[mv[-1]]-Ggt[mv[0]])*1000:.1f}mm" if len(mv) else "no moving")

    Em, Gm = E[mv], Ggt[mv]
    R, T = umeyama(Em, Gm)
    err = np.linalg.norm((R @ Em.T).T + T - Gm, axis=1) * 1000

    print("\n==== TRACKING ERROR (image-anchored per-frame, rigid-aligned) ====")
    print(f"  MEAN   = {err.mean():7.2f} mm")
    print(f"  MAX    = {err.max():7.2f} mm")
    print(f"  RMS    = {np.sqrt((err**2).mean()):7.2f} mm")
    print(f"  median = {np.median(err):7.2f} mm")
    print(f"  p95    = {np.percentile(err,95):7.2f} mm")
    extent = np.linalg.norm(Gm.max(0) - Gm.min(0)) * 1000
    print(f"  (GT spatial extent {extent:.0f} mm; error/extent = {err.mean()/extent:.2f})")
    print("  => small & consistent = same teleop; large = image match only shared the scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
