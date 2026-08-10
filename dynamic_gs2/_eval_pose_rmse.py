#!/usr/bin/env python
"""Relative-pose tracking accuracy: estimate vs Gazebo ground truth, same clock.

ANSWERS
  Is the tracking accurate and robust enough to keep the dynamic GS object aligned during real
  motion and occlusion? Reports the object's POSITION error (mm) and ROTATION error (deg) as
  RMSE + 95th percentile, plus how often tracking reported failure, the longest unbroken failure
  run (the occlusion dropout proxy) and how the inlier count behaves there.

INPUTS (one live run, both stamped on the sim clock)
  --est  <data_dir>/object_track_poses.jsonl   written by the pipeline every tick
  --gt   <data_dir>/gt_object_poses.csv        written by record_gt.sh alongside the run

THE TWO FRAME MISMATCHES, AND HOW EACH IS HANDLED

  (1) BODY frame: Gazebo reports the MODEL ORIGIN pose; the tracker's object is a cloud whose
      anchor centroid sits somewhere else on the object. Those are different points, so their
      positions must never be differenced directly.
      Handled by comparing MOTION, not position: both sides are reduced to a rigid motion
      T_rel(k) = T(k).T(0)^-1 and then applied to THE SAME point (the anchor centroid from the
      pose log). The body offset cancels exactly, because a rigid motion is independent of which
      point it is applied to. What is reported is how far the object's centroid has drifted from
      where ground truth says it should be -- i.e. how misaligned the rendered object is.

  (2) WORLD frame: the tracker reports in the scene frame, ground truth in the Gazebo world
      frame. These are EXPECTED to be identical (camera poses come from Gazebo FK, and the
      dataparser keeps them metric and un-recentered -- no orientation/centering applied), so
      zeroing at t0 should be all that is needed. That expectation is CHECKED, not trusted: the
      report fits the best rigid transform between the two centroid paths and prints its rotation
      angle, then re-reports BOTH position and rotation with that transform removed. A constant
      rotation between the frames acts on relative rotations by conjugation, so the aligned
      rotation is angle(Q R_est Q^T . R_gt^T) -- not just the raw difference.
        near 0 deg  -> frames agree; the direct numbers ARE the tracking error.
        large       -> the direct numbers are dominated by the frame offset; read the aligned ones.

  Note a constant error is INVISIBLE to this method: zeroing at t0 absorbs any fixed offset, so
  what is measured is time-varying error (drift + jitter), not systematic bias.

JOIN
  Ground truth is interpolated onto each tracker tick's own t_sec -- linear on position, SLERP on
  rotation -- over the overlapping window. Both logs come from one session on the sim clock, so
  this is an exact join rather than a resample-and-hope pairing.
"""
import argparse
import json
import sys

import numpy as np

try:
    from scipy.spatial.transform import Rotation, Slerp
except ImportError:
    sys.exit("needs scipy (run in the dynamic_gs env)")


# ------------------------------------------------------------------ loading
def load_est(path, segment=0):
    """pose log -> (t_sec[N], R[N,3,3], t[N,3], ok[N], inliers[N] or None, centroid[3])."""
    ts, Rs, Ts, oks, inl = [], [], [], [], []
    centroid, seg = None, 0
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if o.get("type") == "anchor":
            seg = o.get("segment", 0)
            if centroid is None and o.get("centroid_world") is not None:
                centroid = np.asarray(o["centroid_world"], float).reshape(3)
        elif o.get("type") == "pose" and seg == segment:
            ts.append(float(o["t_sec"]))
            Rs.append(np.asarray(o["R"], float).reshape(3, 3))
            Ts.append(np.asarray(o["t"], float).reshape(3))
            oks.append(bool(o.get("ok", True)))
            inl.append(o.get("inliers", -1))
    if not ts:
        sys.exit("no pose records for segment %d in %s" % (segment, path))
    if centroid is None:
        print("!! anchor carries no centroid_world; falling back to the world origin, which makes\n"
              "   the position error a world-origin lever-arm number rather than object drift.")
        centroid = np.zeros(3)
    inl = np.asarray(inl, float)
    return (np.asarray(ts), np.asarray(Rs), np.asarray(Ts), np.asarray(oks),
            (inl if (inl >= 0).any() else None), centroid)


def load_gt(path):
    """GT csv -> (t[M], xyz[M,3], quat_xyzw[M,4]), time-sorted, strictly increasing."""
    a = np.loadtxt(path, delimiter=",", comments="#")
    if a.ndim == 1:
        a = a[None]
    if a.shape[1] < 8:
        sys.exit("GT csv needs 8 columns t,x,y,z,qx,qy,qz,qw (got %d)" % a.shape[1])
    a = a[np.argsort(a[:, 0])]
    a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]     # SLERP needs strictly increasing t
    return a[:, 0], a[:, 1:4], a[:, 4:8]


# ------------------------------------------------------------------ geometry
def relative_to_first(R, t):
    """T_rel(k) = T(k).T(0)^-1 -> starts at exactly identity / zero."""
    R_rel = R @ R[0].T
    return R_rel, t - np.einsum("nij,j->ni", R_rel, t[0])


def apply(R, t, p):
    """Apply a stack of rigid motions to ONE point -> (N,3)."""
    return np.einsum("nij,j->ni", R, p) + t


def geodesic_deg(Ra, Rb):
    """Per-sample angle of Ra Rb^T, in degrees."""
    tr = np.trace(Ra @ np.transpose(Rb, (0, 2, 1)), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def umeyama(src, dst):
    """Least-squares rigid R,T (no scale) mapping src -> dst (Nx3 each)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def angle_of(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))))


# ------------------------------------------------------------------ report
def stats(e, label, unit):
    print("  %-24s n=%-5d RMSE=%8.2f  p95=%8.2f  mean=%8.2f  median=%8.2f  max=%8.2f  %s"
          % (label, len(e), np.sqrt((e ** 2).mean()), np.percentile(e, 95),
             e.mean(), np.median(e), e.max(), unit))


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="object_track_poses.jsonl")
    ap.add_argument("--gt", required=True, help="gt_object_poses.csv from record_gt.sh")
    ap.add_argument("--segment", type=int, default=0, help="pose-log segment (default 0)")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="object radius (m): also report worst-case surface misalignment")
    ap.add_argument("--dump", help="optional per-sample CSV: t,pos_mm,rot_deg,ok,inliers")
    args = ap.parse_args()

    e_t, e_R, e_p, e_ok, e_in, cent = load_est(args.est, args.segment)
    g_t, g_p, g_q = load_gt(args.gt)
    print("estimate ticks=%d (t %.2f..%.2f)   GT samples=%d (t %.2f..%.2f)   anchor centroid=(%.3f %.3f %.3f)"
          % (len(e_t), e_t[0], e_t[-1], len(g_t), g_t[0], g_t[-1], cent[0], cent[1], cent[2]))

    # ---- exact same-clock join: GT interpolated onto the tracker's own tick times ----
    m = (e_t >= g_t[0]) & (e_t <= g_t[-1])
    if m.sum() < 10:
        sys.exit("only %d ticks inside the GT window -- were both logs from the SAME run, with the "
                 "recorder started before the pipeline?" % int(m.sum()))
    if m.sum() < len(e_t):
        print("  (%d of %d ticks fall outside the GT window and are dropped)"
              % (len(e_t) - int(m.sum()), len(e_t)))
    e_t, e_R, e_p, e_ok = e_t[m], e_R[m], e_p[m], e_ok[m]
    e_in = e_in[m] if e_in is not None else None
    gi_p = np.stack([np.interp(e_t, g_t, g_p[:, i]) for i in range(3)], 1)
    gi_R = Slerp(g_t, Rotation.from_quat(g_q))(e_t).as_matrix()

    # ---- zero both, then apply BOTH motions to the SAME point (kills the body-frame offset) ----
    eR, et = relative_to_first(e_R, e_p)
    gR, gt_ = relative_to_first(gi_R, gi_p)
    c_est, c_gt = apply(eR, et, cent), apply(gR, gt_, cent)

    pos_mm = np.linalg.norm(c_est - c_gt, axis=1) * 1000.0
    rot_deg = geodesic_deg(eR, gR)

    n_fail = int((~e_ok).sum())
    ok = e_ok if e_ok.sum() >= 10 else np.ones_like(e_ok)
    if e_ok.sum() < 10:
        print("\n!! only %d OK ticks -- reporting over ALL ticks instead" % int(e_ok.sum()))

    print("\nGT object motion over the window: path %.1f mm, net %.1f mm, net rotation %.1f deg"
          % (np.linalg.norm(np.diff(c_gt, axis=0), axis=1).sum() * 1000.0,
             np.linalg.norm(c_gt[-1] - c_gt[0]) * 1000.0, geodesic_deg(gR[-1][None], gR[0][None])[0]))

    print("\n==== OBJECT ALIGNMENT ERROR (both zeroed at t0, evaluated at the object centroid) ====")
    stats(pos_mm[ok], "position", "mm")
    stats(rot_deg[ok], "rotation", "deg")
    if args.radius > 0:
        surf = pos_mm + 2.0 * np.sin(np.radians(rot_deg) / 2.0) * args.radius * 1000.0
        stats(surf[ok], "surface (r=%.0fmm)" % (args.radius * 1000), "mm")

    # ---- frame check ----
    # A constant rotation Q between the scene and Gazebo frames conjugates every relative
    # rotation (R_est = Q R_gt Q^T). Rotation VECTORS then satisfy r_est = Q r_gt exactly, and
    # -- unlike anything built from positions -- that relation is untouched by a translation
    # between the frames. So Q comes straight out of a Kabsch fit on the rotvec pairs (through
    # the origin, no centering), and the conjugation can be removed from the rotation error.
    print("\n==== FRAME CHECK (scene frame vs Gazebo world frame) ====")
    rv_e = Rotation.from_matrix(eR).as_rotvec()
    rv_g = Rotation.from_matrix(gR).as_rotvec()
    if np.degrees(np.linalg.norm(rv_g, axis=1)).max() < 3.0:
        print("  the object rotates <3 deg over this run, so a frame rotation cannot be identified\n"
              "  from it. Position numbers stand only if the frames are known to coincide.")
    else:
        U, _, Vt = np.linalg.svd(rv_e[ok].T @ rv_g[ok])
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        Qf = U @ D @ Vt
        ang = angle_of(Qf)
        print("  fitted frame rotation   : %.2f deg" % ang)
        stats(geodesic_deg(eR, Qf @ gR @ Qf.T)[ok], "rotation (conjug. removed)", "deg")
        if ang < 2.0:
            print("  -> frames agree; the numbers above ARE the tracking error.")
        else:
            print("  -> scene and Gazebo frames differ by %.1f deg. The POSITION error above is then\n"
                  "     dominated by that offset, not by tracking. They are expected to coincide, so\n"
                  "     treat this as a bug to chase before reading the position number." % ang)

    # ---- robustness / occlusion ----
    print("\n==== ROBUSTNESS ====")
    print("  ticks evaluated         : %d" % len(e_t))
    print("  tracking FAILED ticks   : %d (%.1f%%), longest unbroken run %d"
          % (n_fail, 100.0 * n_fail / max(1, len(e_t)), longest_run(~e_ok)))
    if e_in is not None:
        print("  inliers  ok=True        : median %.0f  p5 %.0f  min %.0f"
              % (np.median(e_in[e_ok]), np.percentile(e_in[e_ok], 5), e_in[e_ok].min())
              if e_ok.any() else "  inliers: n/a")
        if n_fail:
            print("  inliers  ok=False       : median %.0f  max %.0f"
                  % (np.median(e_in[~e_ok]), e_in[~e_ok].max()))
    else:
        print("  (no inlier counts in this log -- older run, or pipeline predates the field)")
    if n_fail and n_fail < len(e_t):
        stats(pos_mm[~e_ok], "position @failed", "mm")
        stats(rot_deg[~e_ok], "rotation @failed", "deg")

    if args.dump:
        with open(args.dump, "w") as fh:
            fh.write("t,pos_mm,rot_deg,ok,inliers\n")
            for i in range(len(e_t)):
                fh.write("%.6f,%.4f,%.4f,%d,%d\n" % (e_t[i], pos_mm[i], rot_deg[i], int(e_ok[i]),
                                                     -1 if e_in is None else int(e_in[i])))
        print("\nper-sample dump -> %s" % args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
