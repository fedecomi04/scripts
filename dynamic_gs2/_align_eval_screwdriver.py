#!/usr/bin/env python
"""Screwdriver tracking error: estimated object trajectory vs Gazebo GT.

Pipeline (matches the method explained to the operator):
  1. Reconstruct the estimated object world position per tick from
     object_track_poses.jsonl (segment 0):  c_k = R_k @ centroid_world + t_k.
  2. Load GT (screwdriver /gazebo/model_states) and find its MOVING window
     (speed > threshold) — the object is tracked before it moves, so the
     stationary head/tail are separated out.
  3. Temporal sync: detect the motion onset in BOTH signals, take each moving
     window, resample both to a common N by normalized time (fps/clock-agnostic).
  4. Umeyama (rigid, no scale) fit on the MOVING samples ONLY -> well-conditioned
     R,T (a stationary blob can't constrain rotation). That R,T is the frame
     alignment; it absorbs the D0 anchor + the scene<->gazebo frame offset.
  5. Report error split: MOVING (tracking under motion) vs STATIONARY (jitter
     floor, GT perfectly still), constant bias vs high-frequency jitter, and a
     same-session sanity check (large structured residual => different teleop).
"""
import argparse
import json
import sys

import numpy as np


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def load_est(path):
    anchor = None
    cent, tsec, tick = [], [], []
    seg = 0
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if o.get("type") == "anchor":
            seg = o.get("segment", 0)
            if anchor is None:
                anchor = np.asarray(o["centroid_world"], float)
        elif o.get("type") == "pose" and seg == 0 and o.get("ok", True):
            R = np.asarray(o["R"], float).reshape(3, 3)
            t = np.asarray(o["t"], float)
            cent.append(R @ anchor + t)
            tsec.append(o.get("t_sec", len(tsec)))
            tick.append(o.get("tick", len(tick)))
    return np.asarray(cent), np.asarray(tsec)


def speed(xyz, t=None):
    if t is None:
        t = np.arange(len(xyz), dtype=float)
    v = np.gradient(xyz, t, axis=0)
    return np.linalg.norm(v, axis=1)


def moving_window(xyz, t=None, frac=0.10):
    """Indices [lo,hi) where smoothed speed exceeds frac*max (the manipulation)."""
    s = speed(xyz, t)
    k = max(3, len(s) // 50)
    sm = np.convolve(s, np.ones(k) / k, mode="same")
    thr = frac * sm.max()
    idx = np.where(sm > thr)[0]
    if len(idx) < 5:
        return 0, len(xyz), s
    return idx[0], idx[-1] + 1, s


def resample(xyz, n):
    idx = np.linspace(0, len(xyz) - 1, n)
    return np.stack([np.interp(idx, np.arange(len(xyz)), xyz[:, i]) for i in range(3)], 1)


def resample_arclen(xyz, n):
    """Resample to n points EQUALLY SPACED ALONG THE PATH (arc-length).
    Correspondence-free: matches spatial points regardless of velocity/dwell/fps."""
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return resample(xyz, n)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, xyz[:, i]) for i in range(3)], 1)


def icp(src, dst, iters=25):
    """Rigid ICP (no scale): refine R,T mapping src->dst by NN correspondence."""
    from scipy.spatial import cKDTree
    R, T = np.eye(3), np.zeros(3)
    cur = src.copy()
    tree = cKDTree(dst)
    for _ in range(iters):
        _, j = tree.query(cur)
        dR, dT = umeyama(cur, dst[j])
        cur = (dR @ cur.T).T + dT
        R, T = dR @ R, dR @ T + dT
    _, j = tree.query(cur)
    resid = np.linalg.norm(cur - dst[j], axis=1)
    return R, T, resid


def stats(err, label):
    print(f"  {label:12s} mean={err.mean():6.2f}  max={err.max():6.2f}  "
          f"rms={np.sqrt((err**2).mean()):6.2f}  median={np.median(err):6.2f}  "
          f"p95={np.percentile(err,95):6.2f}  (mm, n={len(err)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("-n", type=int, default=150, help="common resample count")
    args = ap.parse_args()

    E, e_t = load_est(args.est)
    g = np.loadtxt(args.gt, delimiter=",")
    G, g_t = g[:, 1:4], g[:, 0]
    print(f"estimate poses={len(E)}  GT rows={len(G)}")

    # index-based motion detection for BOTH (model_states has duplicate
    # timestamps -> a time-derivative divides by zero; position deltas are safe)
    e_lo, e_hi, e_s = moving_window(E)
    g_lo, g_hi, g_s = moving_window(G)
    Em, Gm = E[e_lo:e_hi], G[g_lo:g_hi]
    print(f"estimate motion window: idx [{e_lo},{e_hi}) len={len(Em)}  "
          f"net {np.linalg.norm(Em[-1]-Em[0])*1000:.1f}mm path {np.linalg.norm(np.diff(Em,axis=0),axis=1).sum()*1000:.1f}mm")
    print(f"GT motion window:       len={len(Gm)}  "
          f"net {np.linalg.norm(Gm[-1]-Gm[0])*1000:.1f}mm path {np.linalg.norm(np.diff(Gm,axis=0),axis=1).sum()*1000:.1f}mm  "
          f"(bag t {g_t[g_lo]:.1f}->{g_t[g_hi-1]:.1f}s)")

    # ---- arc-length resample both paths (velocity/fps-agnostic) ----
    N = min(args.n, len(Em), len(Gm))
    Er = resample_arclen(Em, N)

    # speed-profile correlation (same-session indicator), forward vs reversed
    def speed_prof(a):
        s = np.linalg.norm(np.diff(a, axis=0), axis=1)
        return (s - s.mean()) / (s.std() + 1e-9)
    es = speed_prof(resample(Em, N))
    corr_f = float((es * speed_prof(resample(Gm, N))).mean())
    corr_r = float((es * speed_prof(resample(Gm[::-1], N))).mean())

    # ---- ICP path match, forward + reversed, keep best ----
    best = None
    for tag, Gcand in (("forward", Gm), ("reversed", Gm[::-1])):
        Gr = resample_arclen(Gcand, N)
        R0, T0 = umeyama(Er, Gr)                       # arc-length init
        Ei = (R0 @ Er.T).T + T0
        Rr, Tr, resid = icp(Ei, Gr)                    # correspondence-free refine
        R, T = Rr @ R0, Rr @ T0 + Tr
        res = resid * 1000
        if best is None or res.mean() < best[0]:
            best = (res.mean(), tag, R, T, Gr, res)
    mean_mv, direction, R, T, Gr, res_mv = best

    gt_extent = np.linalg.norm(Gr.max(0) - Gr.min(0)) * 1000
    print(f"\n[path match] arc-length + ICP; best GT direction = {direction}, N={N}")
    print(f"[same-session] speed-profile corr: forward={corr_f:+.2f} reversed={corr_r:+.2f} "
          f"(near +1 => same teleop)")

    print("\n==== SAME-SESSION CHECK ====")
    print(f"  ICP path-deviation residual mean={mean_mv:.2f} mm  "
          f"(GT spatial extent {gt_extent:.0f} mm, ratio {mean_mv/gt_extent:.2f})")
    same = (max(corr_f, corr_r) > 0.5) and (mean_mv < 0.15 * gt_extent)
    print(f"  verdict: {'SAME teleop (valid comparison)' if same else 'DIFFERENT teleop / weak match — number is NOT reliable'}")

    print("\n==== TRACKING ERROR (estimate path vs GT path, rigid-aligned) ====")
    stats(res_mv, "PATH DEV")

    # stationary head: pre-motion estimate vs GT start, under the SAME R,T
    if e_lo > 3 and g_lo > 3:
        Es = resample(E[:e_lo], min(60, e_lo))
        gt_start = G[:g_lo].mean(0)
        err_s = np.linalg.norm((R @ Es.T).T + T - gt_start, axis=1) * 1000
        stats(err_s, "STATIONARY")
        print("   (STATIONARY = tracker jitter floor: GT is perfectly still here)")

    # bias vs high-frequency jitter on the moving window
    W = max(3, N // 15)
    k = np.ones(W) / W
    aligned = (R @ Er.T).T + T
    d = aligned - Gr
    bias = np.linalg.norm(d.mean(0)) * 1000
    hp = d - np.stack([np.convolve(d[:, i], k, mode="same") for i in range(3)], 1)
    s = W // 2
    jit = np.sqrt((hp[s:-s] ** 2).sum(1).mean()) * 1000
    print(f"\n  constant BIAS (systematic)   : {bias:6.2f} mm")
    print(f"  high-freq JITTER (jiggle RMS): {jit:6.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
