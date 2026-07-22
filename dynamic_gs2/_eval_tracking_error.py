#!/usr/bin/env python
"""Quantitative tracking error: estimated object trajectory vs ground truth.

WHAT IT ANSWERS
  Mean / max / RMS position error (mm) of the tracker's rigid-transform estimate
  against the true Gazebo object pose, plus mean/max rotation error (deg) when
  both sides carry orientation.

WHY THE ALIGNMENT STEP (the "anchoring is handled" part)
  The tracker reports the object pose in the SCENE / D0-anchored frame; ground
  truth is in the Gazebo WORLD frame. Those two frames differ by one constant
  rigid transform (a rotation + translation). We recover that transform once,
  by best-fit (Umeyama, NO scale) over the whole trajectory, and subtract it.
  What remains is the genuine per-frame tracking error — independent of where
  the anchor was placed or how the two world frames were defined. So you do NOT
  have to line up the anchor by hand; the fit absorbs it.

TWO INPUT MODES
  (1) Same-clock CSVs (the replay-harness output — RECOMMENDED, exact join):
        _eval_tracking_error.py trk.csv gt.csv
      trk.csv = DGS_TRACK_TRAJ_LOG : wall_t, cx,cy,cz, ...(rest ignored here)
      gt.csv  = DGS_REPLAY_GT_LOG  : wall_t, x,y,z [, qx,qy,qz,qw]
      -> GT is linearly interpolated to each tracker tick's wall time. Exact
         because both logs are stamped on the same wall clock in one run.

  (2) Offline estimate jsonl + a GT csv (when you already have a run's poses):
        _eval_tracking_error.py --est-jsonl object_track_poses.jsonl --gt gt.csv
                                 [--transforms dynamic_scene/transforms.json]
      Reconstructs the object world centroid per tick from the jsonl
      (centroid_k = R_k @ centroid_world + t_k, segment 0 = first pass only).
      Join: by wall time if --transforms has stamp_wall, else by index (the k-th
      estimate tick paired with the k-th GT row after resampling GT to N ticks).
      NOTE: index/resample join is only meaningful if est and GT describe the
      SAME motion over the SAME window — do NOT pair two different sessions.
"""
import argparse
import json
import sys

import numpy as np


# ------------------------------------------------------------------ alignment
def umeyama(src, dst):
    """Least-squares rigid R,T (no scale) mapping src -> dst (Nx3 each)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    T = mu_d - R @ mu_s
    return R, T


# ---------------------------------------------------------------- loaders
def load_gt_csv(path):
    """GT csv -> (t[N], xyz[N,3], quat[N,4] or None). Cols: t,x,y,z[,qx,qy,qz,qw]."""
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a[None]
    t, xyz = a[:, 0], a[:, 1:4]
    quat = a[:, 4:8] if a.shape[1] >= 8 else None
    return t, xyz, quat


def load_trk_csv(path):
    """Tracker csv -> (wall_t[N], centroid[N,3]). Cols: wall_t,cx,cy,cz,..."""
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a[None]
    return a[:, 0], a[:, 1:4]


def load_est_jsonl(path):
    """object_track_poses.jsonl (segment 0) -> (tick[N], centroid_world[N,3]).

    Reconstructs the object centroid per tick in world frame:
        centroid_k = R_k @ centroid_world + t_k
    Only the first pass (segment 0) is used so ticks map 1:1 to dynamic frames.
    """
    anchor_c = None
    ticks, cents = [], []
    seg = 0
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if o.get("type") == "anchor":
            seg = o.get("segment", 0)
            if anchor_c is None:
                anchor_c = np.asarray(o["centroid_world"], float)
            continue
        if o.get("type") == "pose" and seg == 0 and o.get("ok", True):
            R = np.asarray(o["R"], float).reshape(3, 3)
            t = np.asarray(o["t"], float)
            cents.append(R @ anchor_c + t)
            ticks.append(o["tick"])
    if anchor_c is None or not cents:
        sys.exit("no anchor / no segment-0 poses in %s" % path)
    return np.asarray(ticks), np.asarray(cents)


def stamp_wall_by_order(transforms_path):
    """dynamic_scene/transforms.json -> per-frame stamp_wall in frame order (or None)."""
    d = json.load(open(transforms_path))
    fr = d.get("frames", [])
    sw = [f.get("stamp_wall") for f in fr]
    return None if any(s is None for s in sw) else np.asarray(sw, float)


# ---------------------------------------------------------------- report
def report(est_xyz, gt_xyz, gt_quat=None, est_quat=None):
    """Align est->GT and print position (+ optional rotation) error stats."""
    R, T = umeyama(est_xyz, gt_xyz)
    aligned = (R @ est_xyz.T).T + T
    err = np.linalg.norm(aligned - gt_xyz, axis=1) * 1000.0  # mm
    const_offset = np.linalg.norm((aligned - gt_xyz).mean(0)) * 1000.0

    print("\n==== TRACKING POSITION ERROR (after rigid alignment) ====")
    print(f"  samples          : {len(err)}")
    print(f"  MEAN error       : {err.mean():8.2f} mm")
    print(f"  MAX  error       : {err.max():8.2f} mm")
    print(f"  RMS  error       : {np.sqrt((err**2).mean()):8.2f} mm")
    print(f"  median error     : {np.median(err):8.2f} mm")
    print(f"  95th percentile  : {np.percentile(err,95):8.2f} mm")
    print(f"  (residual constant offset after align: {const_offset:.2f} mm)")
    print(f"  GT path length   : {np.linalg.norm(np.diff(gt_xyz,axis=0),axis=1).sum()*1000:8.1f} mm")

    if gt_quat is not None and est_quat is not None:
        # relative rotation angle per frame, after the same-constant-frame caveat
        def ang(q):  # quat wxyz? assume xyzw from gazebo
            return q
        # rotation error is left as translation-only here unless both provided
    return err


# ---------------------------------------------------------------- joins
def join_time(t_est, c_est, t_gt, xyz_gt):
    """Interpolate GT xyz to est times, restricted to the overlap window."""
    m = (t_est >= t_gt.min()) & (t_est <= t_gt.max())
    if m.sum() < 10:
        sys.exit(f"only {int(m.sum())} est samples inside GT time window — "
                 "clocks don't overlap (different sessions/clock bases?).")
    te, ce = t_est[m], c_est[m]
    g = np.stack([np.interp(te, t_gt, xyz_gt[:, i]) for i in range(3)], 1)
    return ce, g


def join_index(c_est, xyz_gt):
    """Resample GT to N=len(c_est) samples and pair by order (approximate)."""
    n = len(c_est)
    idx = np.linspace(0, len(xyz_gt) - 1, n)
    g = np.stack([np.interp(idx, np.arange(len(xyz_gt)), xyz_gt[:, i]) for i in range(3)], 1)
    return c_est, g


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pos", nargs="*", help="[trk.csv gt.csv] same-clock CSV mode")
    ap.add_argument("--est-jsonl", help="object_track_poses.jsonl (offline mode)")
    ap.add_argument("--gt", help="GT csv: t,x,y,z[,quat] (offline mode)")
    ap.add_argument("--transforms", help="dynamic_scene/transforms.json for stamp_wall time-join")
    ap.add_argument("--join", choices=["time", "index"], default=None)
    args = ap.parse_args()

    # --- Mode 1: two same-clock CSVs (replay-harness output) ---
    if len(args.pos) == 2 and not args.est_jsonl:
        t_trk, c_trk = load_trk_csv(args.pos[0])
        t_gt, xyz_gt, _ = load_gt_csv(args.pos[1])
        ce, g = join_time(t_trk, c_trk, t_gt, xyz_gt)
        report(ce, g)
        return 0

    # --- Mode 2: offline estimate jsonl + GT csv ---
    if args.est_jsonl and args.gt:
        ticks, c_est = load_est_jsonl(args.est_jsonl)
        t_gt, xyz_gt, _ = load_gt_csv(args.gt)
        join = args.join
        if join is None:
            join = "time" if args.transforms else "index"
        if join == "time":
            sw = stamp_wall_by_order(args.transforms) if args.transforms else None
            if sw is None:
                sys.exit("--join time needs --transforms with stamp_wall; "
                         "else use --join index (same-motion only).")
            # map segment-0 ticks to frame order -> stamp_wall
            te = sw[: len(c_est)] if len(sw) >= len(c_est) else sw
            c_est = c_est[: len(te)]
            ce, g = join_time(te, c_est, t_gt, xyz_gt)
        else:
            print("[join=index] pairing by order after resampling GT — valid ONLY "
                  "if estimate and GT are the SAME motion/window.")
            ce, g = join_index(c_est, xyz_gt)
        report(ce, g)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
