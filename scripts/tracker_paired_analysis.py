#!/usr/bin/env python
"""Paired analysis of offline tracker runs on IDENTICAL frames.

Each CSV (DGS_TRACK_TRAJ_LOG): wall_t,cx,cy,cz,rvx,rvy,rvz,inliers,corr,frame_name
Runs are joined per-frame (same input frames), restricted to the COMMON set of
successfully-tracked frames (kills survivorship bias). Reports:
  - per-run high-pass jitter (translation mm + rotation deg) on the common set
  - paired per-frame |pose_a - pose_b| between consecutive settings
Usage: tracker_paired_analysis.py label1=csv1 label2=csv2 ...
"""
import sys
import numpy as np


def load(p):
    rows = {}
    for ln in open(p):
        f = ln.strip().split(",")
        if len(f) < 10:
            continue
        rows[f[9]] = np.array([float(x) for x in f[1:7]])  # frame -> [c(3), rv(3)]
    return rows


def hp_jitter(arr, W=9):
    k = np.ones(W) / W
    sm = np.stack([np.convolve(arr[:, i], k, mode="same") for i in range(arr.shape[1])], 1)
    s = W // 2
    return (arr - sm)[s:-s]


def main():
    runs = {}
    for a in sys.argv[1:]:
        lab, p = a.split("=", 1)
        runs[lab] = load(p)
    common = None
    for r in runs.values():
        common = set(r) if common is None else common & set(r)
    common = sorted(common)
    print(f"common successfully-tracked frames: {len(common)} "
          f"(per-run: {', '.join(f'{l}={len(r)}' for l, r in runs.items())})")
    mats = {l: np.stack([r[f] for f in common]) for l, r in runs.items()}

    print("\nper-run high-pass jitter on COMMON frames:")
    for l, m in mats.items():
        tj = hp_jitter(m[:, :3])
        rj = hp_jitter(m[:, 3:6])
        print(f"  {l:14s} trans={np.sqrt((tj**2).sum(1).mean())*1000:6.2f}mm  "
              f"rot={np.sqrt((rj**2).sum(1).mean())*180/np.pi:5.2f}deg")

    labs = list(mats)
    print("\npaired per-frame differences (median |Δ| over common frames):")
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            d = mats[labs[i]] - mats[labs[j]]
            dt = np.linalg.norm(d[:, :3], axis=1)
            dr = np.linalg.norm(d[:, 3:6], axis=1) * 180 / np.pi
            print(f"  {labs[i]} vs {labs[j]:14s} trans med={np.median(dt)*1000:6.2f}mm "
                  f"p95={np.percentile(dt,95)*1000:7.2f}mm | rot med={np.median(dr):5.2f}deg")


if __name__ == "__main__":
    main()
