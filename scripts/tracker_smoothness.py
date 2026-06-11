#!/usr/bin/env python
"""Quantify tracker trajectory smoothness vs ground truth.

Inputs:
  tracker CSV (DGS_TRACK_TRAJ_LOG): wall_t, cx,cy,cz, rvx,rvy,rvz, inliers, corr
  GT CSV     (DGS_REPLAY_GT_LOG):    wall_t, x,y,z   (forced object pose)

Method: interpolate GT to the tracker tick times, rigid-align (Umeyama, no scale)
GT->tracker over the whole path (this ABSORBS a constant offset — which the user
says is fine), then measure the high-pass residual RMS = continuous oscillation
(the thing to minimize). Also reports the constant offset (informational) + a
filter-free jerk RMS.

Usage: tracker_smoothness.py <tracker.csv> <gt.csv> [smooth_window]
"""
import sys
import numpy as np


def umeyama(src, dst):
    """Rigid R,T (no scale) mapping src->dst, least squares."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    T = mu_d - R @ mu_s
    return R, T


def movavg(x, w):
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], axis=1)


def main():
    trk = np.loadtxt(sys.argv[1], delimiter=",")
    gt = np.loadtxt(sys.argv[2], delimiter=",")
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    if trk.ndim == 1:
        trk = trk[None]
    tt, tc, trv, inl = trk[:, 0], trk[:, 1:4], trk[:, 4:7], trk[:, 7]
    gtt, gp = gt[:, 0], gt[:, 1:4]

    # overlap window only
    m = (tt >= gtt.min()) & (tt <= gtt.max())
    tt, tc, trv, inl = tt[m], tc[m], trv[m], inl[m]
    if len(tt) < 20:
        print(f"too few overlapping ticks ({len(tt)}) — check clocks / object motion")
        return 1
    gpi = np.stack([np.interp(tt, gtt, gp[:, k]) for k in range(3)], axis=1)

    R, T = umeyama(gpi, tc)               # GT -> tracker frame
    aligned = (R @ gpi.T).T + T
    resid = tc - aligned                   # per-tick error (offset + jitter)

    offset = np.linalg.norm(resid.mean(0))
    hp = resid - movavg(resid, W)          # remove slow component -> oscillation
    osc_rms = float(np.sqrt((hp ** 2).sum(1).mean()))

    dt = float(np.median(np.diff(tt)))
    vel = np.diff(tc, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_rms = float(np.sqrt((jerk ** 2).sum(1).mean()))
    # rotational oscillation: high-pass of the rotvec
    rhp = trv - movavg(trv, W)
    rosc = float(np.sqrt((rhp ** 2).sum(1).mean())) * 180 / np.pi

    gt_path_len = float(np.linalg.norm(np.diff(gpi, axis=0), axis=1).sum())
    print(f"n={len(tt)} dt={dt*1000:.0f}ms inliers_mean={inl.mean():.0f} "
          f"gt_path_len={gt_path_len*1000:.0f}mm")
    print(f"constant_offset = {offset*1000:7.2f} mm   (OK if along whole path)")
    print(f"OSCILLATION_rms = {osc_rms*1000:7.3f} mm   (high-pass residual; LOWER=smoother)")
    print(f"rot_osc_rms     = {rosc:7.3f} deg")
    print(f"jerk_rms        = {jerk_rms:7.1f} m/s^3   (filter-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
