#!/usr/bin/env python
"""Diagnostic: is the reported 'tracking error' actually a clock/latency offset?

Reuses the loaders from _eval_pose_rmse.py, then:
  (1) TIME-OFFSET SWEEP: shift the estimate's t_sec by delta before joining GT, and
      recompute the same both-zeroed-at-t0 position/rotation RMSE. If RMSE has a sharp
      minimum at delta != 0, the two logs are not synchronous and the headline error is a
      lag artifact, not tracking drift.
  (2) SPEED vs ERROR: correlate per-tick |GT velocity| with the position error. A time
      offset makes error ~ |v|*delta, so error should ride the speed. Occlusion/tracking
      failure instead rides low inlier counts.
"""
import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "_eval_pose_rmse.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402


def eval_at_offset(e_t, e_R, e_p, e_ok, g_t, g_p, g_q, cent, delta):
    """Return (pos_mm, rot_deg, ok, gt_speed_mm_per_s) after shifting est clock by delta."""
    t = e_t + delta
    m = (t >= g_t[0]) & (t <= g_t[-1])
    if m.sum() < 10:
        return None
    t, R, p, ok = t[m], e_R[m], e_p[m], e_ok[m]
    gi_p = np.stack([np.interp(t, g_t, g_p[:, i]) for i in range(3)], 1)
    gi_R = Slerp(g_t, Rotation.from_quat(g_q))(t).as_matrix()
    eR, et = ev.relative_to_first(R, p)
    gR, gt_ = ev.relative_to_first(gi_R, gi_p)
    c_est, c_gt = ev.apply(eR, et, cent), ev.apply(gR, gt_, cent)
    pos_mm = np.linalg.norm(c_est - c_gt, axis=1) * 1000.0
    rot_deg = ev.geodesic_deg(eR, gR)
    # GT speed at each tick (central difference on the interpolated GT centroid path)
    dt = np.gradient(t)
    v = np.gradient(c_gt, axis=0) / dt[:, None]
    speed = np.linalg.norm(v, axis=1) * 1000.0
    return pos_mm, rot_deg, ok, speed


def rmse(x):
    return float(np.sqrt((x ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--span", type=float, default=1.0, help="sweep +/- this many seconds")
    ap.add_argument("--step", type=float, default=0.02, help="sweep step (s)")
    args = ap.parse_args()

    e_t, e_R, e_p, e_ok, e_in, cent = ev.load_est(args.est, args.segment)
    g_t, g_p, g_q = ev.load_gt(args.gt)

    # use only OK ticks for the sync fit so failures don't pollute the minimum
    base = eval_at_offset(e_t, e_R, e_p, e_ok, g_t, g_p, g_q, cent, 0.0)
    pos0, rot0, ok0, speed0 = base
    use0 = ok0 if ok0.sum() >= 10 else np.ones_like(ok0)

    print("==== TIME-OFFSET SWEEP (est clock shifted by delta, OK ticks only) ====")
    print("  delta(s)   pos_RMSE(mm)   rot_RMSE(deg)   n")
    deltas = np.arange(-args.span, args.span + 1e-9, args.step)
    best = None
    rows = []
    for d in deltas:
        r = eval_at_offset(e_t, e_R, e_p, e_ok, g_t, g_p, g_q, cent, d)
        if r is None:
            continue
        pos_mm, rot_deg, ok, _ = r
        use = ok if ok.sum() >= 10 else np.ones_like(ok)
        pr, rr = rmse(pos_mm[use]), rmse(rot_deg[use])
        rows.append((d, pr, rr, int(use.sum())))
        if best is None or pr < best[1]:
            best = (d, pr, rr, int(use.sum()))
    # print a coarse grid (every ~0.1s) plus the neighbourhood of the best
    for d, pr, rr, n in rows:
        near_best = abs(d - best[0]) <= 2 * args.step + 1e-9
        if abs(round(d / 0.1) * 0.1 - d) < 1e-9 or near_best:
            mark = "  <-- MIN" if abs(d - best[0]) < 1e-9 else ""
            print("  %+7.3f   %10.2f     %10.3f   %4d%s" % (d, pr, rr, n, mark))

    print("\n  best delta = %+.3f s  ->  pos_RMSE %.2f mm (was %.2f at 0), rot_RMSE %.3f deg (was %.3f)"
          % (best[0], best[1], rmse(pos0[use0]), best[2], rmse(rot0[use0])))
    if abs(best[0]) < args.step + 1e-9:
        print("  => minimum is at ~0: clocks are synchronous, t0 alignment is fine.")
    else:
        print("  => minimum is OFF ZERO by %.0f ms and RMSE drops %.0f%%: the logs are NOT synchronous."
              % (best[0] * 1000.0, 100.0 * (1 - best[1] / max(1e-9, rmse(pos0[use0])))))
        print("     The headline error is (partly) a latency/clock artifact, not tracking drift.")

    # ---- how much of the position error is the constant frame offset vs real tracking ----
    # Recompute the zeroed paths at delta=0, fit the frame rotation Qf on rotvec pairs (as the
    # eval does), then rotate the GT relative path by Qf about the shared t0 anchor and see how
    # much position RMSE drops. Big drop => frame offset. Small drop => genuine tracking error.
    t = e_t
    m = (t >= g_t[0]) & (t <= g_t[-1])
    t, R, p, ok = t[m], e_R[m], e_p[m], e_ok[m]
    use = ok if ok.sum() >= 10 else np.ones_like(ok)
    gi_p = np.stack([np.interp(t, g_t, g_p[:, i]) for i in range(3)], 1)
    gi_R = Slerp(g_t, Rotation.from_quat(g_q))(t).as_matrix()
    eR, et = ev.relative_to_first(R, p)
    gR, gt_ = ev.relative_to_first(gi_R, gi_p)
    c_est, c_gt = ev.apply(eR, et, cent), ev.apply(gR, gt_, cent)
    rv_e = Rotation.from_matrix(eR).as_rotvec()
    rv_g = Rotation.from_matrix(gR).as_rotvec()
    U, _, Vt = np.linalg.svd(rv_e[use].T @ rv_g[use])
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Qf = U @ D @ Vt
    c_gt_al = (Qf @ (c_gt - cent).T).T + cent          # rotate GT path by Qf about the t0 anchor
    pos_raw = np.linalg.norm(c_est - c_gt, axis=1) * 1000.0
    pos_al = np.linalg.norm(c_est - c_gt_al, axis=1) * 1000.0
    print("\n==== POSITION: FRAME OFFSET vs TRACKING (at delta=0, OK ticks) ====")
    print("  fitted frame rotation            : %.2f deg" % ev.angle_of(Qf))
    print("  pos_RMSE raw                     : %6.2f mm" % rmse(pos_raw[use]))
    print("  pos_RMSE with frame rot removed  : %6.2f mm   (residual = genuine tracking)" % rmse(pos_al[use]))

    print("\n==== SPEED vs POSITION ERROR (at delta=0, OK ticks) ====")
    p, s = pos0[use0], speed0[use0]
    if s.std() > 0:
        c = float(np.corrcoef(p, s)[0, 1])
        print("  corr(|GT velocity|, pos_err) = %.2f   (high => error rides object speed => a lag)" % c)
    hi = s >= np.percentile(s, 75)
    lo = s <= np.percentile(s, 25)
    print("  pos_err RMSE in fastest 25%% of ticks: %6.2f mm   (median speed %6.0f mm/s)"
          % (rmse(p[hi]), np.median(s[hi])))
    print("  pos_err RMSE in slowest 25%% of ticks: %6.2f mm   (median speed %6.0f mm/s)"
          % (rmse(p[lo]), np.median(s[lo])))


if __name__ == "__main__":
    raise SystemExit(main())
