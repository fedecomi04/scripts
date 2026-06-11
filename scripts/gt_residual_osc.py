#!/usr/bin/env python
"""GT-referenced oscillation: tracker centroid vs interpolated GT object position,
joined per-frame via the fixture's stamp_wall. Umeyama-aligns (absorbs constant
offset), then reports high-pass residual RMS — valid in MOVING segments because
true motion is subtracted by the GT, unlike trajectory high-pass.

Usage: gt_residual_osc.py <fixture_dir> label1=trk1.csv label2=trk2.csv ...
Optionally set a common frame set across all runs (always on).
"""
import sys
import json
from pathlib import Path

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


def main():
    fix = Path(sys.argv[1])
    meta = json.loads((fix / "dynamic_scene" / "transforms.json").read_text())
    stamp = {Path(fr["file_path"]).stem: fr.get("stamp_wall") for fr in meta["frames"]}
    gt = np.loadtxt(fix / "gt_object_trajectory.csv", delimiter=",")
    gtt, gp = gt[:, 0], gt[:, 1:4]
    gspd = np.linalg.norm(np.gradient(gp, gtt, axis=0), axis=1)  # GT speed m/s

    runs = {}
    for a in sys.argv[2:]:
        lab, p = a.split("=", 1)
        rows = {}
        for ln in open(p):
            f = ln.strip().split(",")
            if len(f) >= 10 and stamp.get(f[9]) is not None:
                rows[f[9]] = (stamp[f[9]], np.array([float(x) for x in f[1:4]]))
        runs[lab] = rows
    common = sorted(set.intersection(*[set(r) for r in runs.values()]))
    # restrict to frames inside the GT window
    common = [f for f in common if gtt.min() <= runs[list(runs)[0]][f][0] <= gtt.max()]
    print(f"common frames in GT window: {len(common)}")
    if len(common) < 30:
        print("too few — aborting")
        return 1

    W = 9
    k = np.ones(W) / W
    for lab, rows in runs.items():
        t = np.array([rows[f][0] for f in common])
        c = np.stack([rows[f][1] for f in common])
        g = np.stack([np.interp(t, gtt, gp[:, i]) for i in range(3)], 1)
        spd = np.interp(t, gtt, gspd)
        R, T = umeyama(g, c)
        resid = c - ((R @ g.T).T + T)
        hp = resid - np.stack([np.convolve(resid[:, i], k, mode="same") for i in range(3)], 1)
        s = W // 2
        hp = hp[s:-s]; sp = spd[s:-s]; rs = resid[s:-s]
        stat = sp < 0.005   # GT slower than 5 mm/s = stationary
        def rms(x):
            return float(np.sqrt((x ** 2).sum(1).mean())) * 1000 if len(x) else float("nan")
        print(f"  {lab:10s} osc_all={rms(hp):6.2f}mm  osc_stationary={rms(hp[stat]):6.2f}mm "
              f"osc_moving={rms(hp[~stat]):6.2f}mm  |resid|med={np.median(np.linalg.norm(rs,axis=1))*1000:6.1f}mm "
              f"(n={len(hp)}, stat={int(stat.sum())})")


if __name__ == "__main__":
    raise SystemExit(main())
