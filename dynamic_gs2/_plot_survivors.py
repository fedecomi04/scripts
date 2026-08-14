#!/usr/bin/env python
"""Plots for the 3 surviving good runs in S1/.

Figure 1  plot_3d_trajectories.png : per object, real (GT) vs tracked centroid path in 3D,
          both zeroed at t0 (so they start together and the gap = tracking drift), with small
          orientation triads sampled along each path (solid = GT, dashed = tracked).
Figure 2  plot_error_vs_time.png    : position error (mm, log) and rotation error (deg) vs time,
          all three runs overlaid.

Trajectories/errors are computed identically to _eval_pose_rmse.py (relative motion applied to
the anchor centroid), with each run's untrackable tail cut by --tmax.
"""
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "_eval_pose_rmse.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

S1 = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/S1"
RUNS = [
    # label,            folder,                    tmax,     colour
    # tmax also trims the terminal breakdown spike (last ~2 ticks where the object is yanked away
    # at 300-600 mm/s and the tracker loses it), so the plots show only clean tracking.
    ("banana",          "225559_banana",           200.55,  "#e0a11b"),
    ("fidget spinner",  "002736_fidget_spinner",   220.45,  "#1f77b4"),
    ("screwdriver",     "000021_screwdriver",      235.0,   "#8c3fbf"),
]


def gt_glitch_mask(c_est, c_gt, win=41, gt_thr=0.040, est_thr=0.020):
    """A GT (Gazebo) glitch = the GROUND TRUTH jumps off a smooth path while the ESTIMATE stays
    smooth there (the tracker didn't follow because the jump wasn't real). Real fast motion moves
    BOTH, so it is not flagged. Returns a boolean mask of ticks to drop from the plots."""
    if len(c_gt) < win:
        return np.zeros(len(c_gt), bool)
    res_gt = np.linalg.norm(c_gt - np.stack([medfilt(c_gt[:, i], win) for i in range(3)], 1), axis=1)
    res_est = np.linalg.norm(c_est - np.stack([medfilt(c_est[:, i], win) for i in range(3)], 1), axis=1)
    return (res_gt > gt_thr) & (res_est < est_thr)


def compute(folder, tmax):
    est = os.path.join(S1, folder, "object_track_poses.jsonl")
    gt = os.path.join(S1, folder, "gt_object_poses.csv")
    e_t, e_R, e_p, e_ok, _, cent = ev.load_est(est)
    g_t, g_p, g_q = ev.load_gt(gt)
    m = (e_t >= g_t[0]) & (e_t <= g_t[-1]) & (e_t <= tmax) & e_ok
    e_t, e_R, e_p = e_t[m], e_R[m], e_p[m]
    gi_p = np.stack([np.interp(e_t, g_t, g_p[:, i]) for i in range(3)], 1)
    gi_R = Slerp(g_t, Rotation.from_quat(g_q))(e_t).as_matrix()
    eR, et = ev.relative_to_first(e_R, e_p)
    gR, gt_ = ev.relative_to_first(gi_R, gi_p)
    c_est = ev.apply(eR, et, cent)
    c_gt = ev.apply(gR, gt_, cent)
    pos_mm = np.linalg.norm(c_est - c_gt, axis=1) * 1000.0
    rot_deg = ev.geodesic_deg(eR, gR)
    glitch = gt_glitch_mask(c_est, c_gt)
    # motion onset: first tick where the GT centroid has moved >20 mm from its start
    gdisp = np.linalg.norm(c_gt - c_gt[0], axis=1) * 1000.0
    mv = np.where((gdisp > 20) & ~glitch)[0]
    t_move = float((e_t - e_t[0])[mv[0]]) if len(mv) else None
    # displacement from the shared start point, in cm
    d_est = (c_est - cent) * 100.0
    d_gt = (c_gt - cent) * 100.0
    return dict(t=e_t - e_t[0], pos=pos_mm, rot=rot_deg,
                d_est=d_est, d_gt=d_gt, eR=eR, gR=gR, glitch=glitch, t_move=t_move)


def nan_gap(a, mask):
    """Copy of array with masked rows set to NaN so plotted lines break (not bridge) there."""
    b = np.array(a, float)
    b[mask] = np.nan
    return b


def triad(ax, origin, R, L, ls, alpha):
    for k, col in enumerate(("#d62728", "#2ca02c", "#1f77b4")):   # x,y,z
        v = R[:, k] * L
        ax.plot([origin[0], origin[0] + v[0]], [origin[1], origin[1] + v[1]],
                [origin[2], origin[2] + v[2]], color=col, lw=1.3, ls=ls, alpha=alpha)


def set_equal_3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2.0
    r = max((hi - lo).max(), 5.0) / 2.0 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def main():
    data = {lbl: compute(f, tm) for lbl, f, tm, _ in RUNS}

    # ---------- Figure 1: 3D trajectories (moving phase only) ----------
    fig = plt.figure(figsize=(16, 5.4))
    for j, (lbl, f, tm, col) in enumerate(RUNS):
        d = data[lbl]
        g = d["glitch"]
        mv = d["t"] >= (d["t_move"] or 0.0)        # only the part where the object is moving
        keep = mv & ~g
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        gtl = nan_gap(d["d_gt"], ~mv | g)          # blank out static + glitch, keep the line breaking
        estl = nan_gap(d["d_est"], ~mv | g)
        ax.plot(*gtl.T, color="0.25", lw=2.2, label="real (GT)")
        ax.plot(*estl.T, color=col, lw=2.0, ls="--", label="tracked")
        allpts = np.vstack([d["d_gt"][keep], d["d_est"][keep]])
        L = max(allpts.max(0).max() - allpts.min(0).min(), 5.0) * 0.10
        mv_idx = np.where(mv)[0]
        for i in mv_idx[np.linspace(0, len(mv_idx) - 1, 5).astype(int)]:
            if g[i]:
                continue
            triad(ax, d["d_gt"][i], d["gR"][i], L, "-", 0.9)
            triad(ax, d["d_est"][i], d["eR"][i], L, "--", 0.9)
        ax.scatter(*d["d_gt"][keep][0], color="green", s=45, label="motion start")
        ax.scatter(*d["d_gt"][keep][-1], color="0.25", s=45, marker="s")
        ax.scatter(*d["d_est"][keep][-1], color=col, s=55, marker="X")
        set_equal_3d(ax, allpts)
        ax.set_title("%s\nwhile moving: pos med %.0f mm · rot med %.1f°"
                     % (lbl, np.median(d["pos"][keep]), np.median(d["rot"][keep])), fontsize=11)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Object centroid trajectory while MOVING — real (solid, grey) vs tracked (dashed) · "
                 "triads = orientation (RGB = xyz axes)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(S1, "plot_3d_trajectories.png")
    fig.savefig(p1, dpi=150); print("wrote", p1)

    # ---------- Figure 2: error vs time (moving phase only, linear) ----------
    fig2, (axp, axr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for lbl, f, tm, col in RUNS:
        d = data[lbl]
        g = d["glitch"]
        tm0 = d["t_move"] or 0.0
        mv = d["t"] >= tm0
        x = d["t"][mv] - tm0
        axp.plot(x, nan_gap(d["pos"], g)[mv], color=col, lw=1.5, label=lbl)
        axr.plot(x, nan_gap(d["rot"], g)[mv], color=col, lw=1.5, label=lbl)
    axp.set_ylabel("position error (mm)")
    axp.grid(True, alpha=0.3); axp.legend(fontsize=10, title="object")
    axp.set_title("Tracking error while the object is MOVING (static phase removed)")
    axr.set_ylabel("rotation error (deg)"); axr.set_xlabel("time since motion onset (s)")
    axr.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(S1, "plot_error_vs_time.png")
    fig2.savefig(p2, dpi=150); print("wrote", p2)


if __name__ == "__main__":
    raise SystemExit(main())
