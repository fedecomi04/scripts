#!/usr/bin/env python3
"""Prototype: compare depth-denoise filters on noisy ZED-sim depth by
back-projecting each variant to a colored point cloud.

For N random static frames, writes 4 PLYs each into <static>/reproj_check/:
  <stem>_0_raw.ply           raw noisy depth (baseline)
  <stem>_1_median.ply        masked median only
  <stem>_2_bilateral.ply     masked bilateral only
  <stem>_3_med_bilateral.ply masked median -> masked bilateral (the cascade)

Both filters are VALIDITY-MASKED: invalid (0) pixels are excluded from every
weighted average / median, so holes never bleed surface toward zero. Prints a
local-roughness metric per variant (mean |d - mean(valid 3x3 neighbours)| over
valid px, in mm) so the denoise is measured, not just eyeballed.

Usage:
  python scripts/reproj_denoise_compare.py <data_dir> [n] [seed]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# --- filter params (light, matched to the ZED quadratic noise: sub-mm..~2mm) ---
MEDIAN_RADIUS = 2          # 5x5 masked median (outlier / flying-pixel removal)
BILAT_RADIUS = 3           # 7x7 masked bilateral
BILAT_SIGMA_S_PX = 2.0     # spatial Gaussian (px) — small, just a local patch
BILAT_SIGMA_R_M = 0.005    # range/depth Gaussian (m) — averages within-surface,
                           # below a typical object/table step (>1cm) so it won't bridge it


def _windows(arr, radius, fill):
    """Yield (dy, dx, shifted_view) over a (2r+1)^2 neighbourhood, zero/fill-padded."""
    H, W = arr.shape
    pad = np.full((H + 2 * radius, W + 2 * radius), fill, dtype=arr.dtype)
    pad[radius:radius + H, radius:radius + W] = arr
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yield dy, dx, pad[radius + dy:radius + dy + H, radius + dx:radius + dx + W]


def masked_median(depth, valid, radius=MEDIAN_RADIUS):
    """Median over VALID neighbours only (invalid -> NaN -> ignored)."""
    d = np.where(valid, depth, np.nan).astype(np.float32)
    stack = [w for _, _, w in _windows(d, radius, np.nan)]
    with np.errstate(all="ignore"):
        out = np.nanmedian(np.stack(stack, axis=0), axis=0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def masked_bilateral(depth, valid, radius=BILAT_RADIUS,
                     sigma_s=BILAT_SIGMA_S_PX, sigma_r=BILAT_SIGMA_R_M):
    """Edge-preserving bilateral using ONLY valid neighbours (normalized conv)."""
    dc = depth.astype(np.float32)
    vc = valid.astype(np.float32)
    num = np.zeros_like(dc)
    den = np.zeros_like(dc)
    vpad_iter = list(_windows(vc, radius, 0.0))
    dpad_iter = list(_windows(dc, radius, 0.0))
    for (dy, dx, vn), (_, _, dn) in zip(vpad_iter, dpad_iter):
        ws = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s))
        diff = dn - dc
        wr = np.exp(-(diff * diff) / (2.0 * sigma_r * sigma_r))
        w = ws * wr * vn          # zero weight where neighbour invalid
        num += w * dn
        den += w
    out = np.where((den > 0) & (vc > 0), num / np.maximum(den, 1e-12), 0.0)
    return out.astype(np.float32)


def roughness_mm(depth, valid):
    """Mean |d - mean(valid 3x3 neighbours)| over valid px, in mm. Lower = smoother."""
    d = np.where(valid, depth, 0.0).astype(np.float32)
    v = valid.astype(np.float32)
    # 3x3 valid-neighbour mean (exclude the centre).
    nsum = np.zeros_like(d); wsum = np.zeros_like(d)
    dwins = list(_windows(d, 1, 0.0)); vwins = list(_windows(v, 1, 0.0))
    for (dy, dx, dn), (_, _, vn) in zip(dwins, vwins):
        if dy == 0 and dx == 0:
            continue
        nsum += dn; wsum += vn
    neigh_mean = np.where(wsum > 0, nsum / np.maximum(wsum, 1e-12), 0.0)
    m = valid & (wsum > 0)
    return float(np.mean(np.abs(depth[m] - neigh_mean[m])) * 1000.0)


def write_ply(path, xyz, rgb):
    m = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {m}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    v = np.empty(m, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); f.write(v.tobytes())


def backproject(depth_m, rgb, fx, fy, cx, cy):
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_m > 0.01
    z = depth_m[valid]
    x = (uu[valid] - cx) / fx * z
    y = (vv[valid] - cy) / fy * z
    return np.stack([x, y, z], 1), rgb[valid]


def main():
    data_dir = Path(sys.argv[1]).resolve()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    static = data_dir / "static_scene"
    tf = json.loads((static / "transforms.json").read_text())
    fx, fy = float(tf["fl_x"]), float(tf["fl_y"])
    cx, cy = float(tf["cx"]), float(tf["cy"])
    frames = tf["frames"]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(frames), size=min(n, len(frames)), replace=False).tolist())
    out_dir = static / "reproj_check"; out_dir.mkdir(exist_ok=True)
    print(f"[denoise] {data_dir.name}: {len(idx)} frames -> {out_dir}")
    print(f"[denoise] median r={MEDIAN_RADIUS} (5x5); bilateral r={BILAT_RADIUS} "
          f"sigma_s={BILAT_SIGMA_S_PX}px sigma_r={BILAT_SIGMA_R_M*1000:.0f}mm")
    print(f"{'frame':>10} {'raw':>8} {'median':>8} {'bilat':>8} {'med+bil':>8}   (roughness mm)")

    for i in idx:
        fr = frames[i]
        rgb = cv2.cvtColor(cv2.imread(str(static / fr["file_path"].replace("./", ""))), cv2.COLOR_BGR2RGB)
        dmm = cv2.imread(str(static / fr["depth_file_path"].replace("./", "")), cv2.IMREAD_UNCHANGED)
        depth = dmm.astype(np.float32) / 1000.0
        valid0 = depth > 0.01

        d_med = masked_median(depth, valid0)
        d_bil = masked_bilateral(depth, valid0)
        d_mb = masked_bilateral(d_med, d_med > 0.01)

        variants = {
            "0_raw": depth, "1_median": d_med, "2_bilateral": d_bil, "3_med_bilateral": d_mb,
        }
        rough = {}
        stem = Path(fr["file_path"]).stem
        for name, dv in variants.items():
            vv = dv > 0.01
            rough[name] = roughness_mm(dv, vv)
            xyz, col = backproject(dv, rgb, fx, fy, cx, cy)
            write_ply(out_dir / f"{stem}_{name}.ply", xyz, col)
        print(f"{stem:>10} {rough['0_raw']:>8.2f} {rough['1_median']:>8.2f} "
              f"{rough['2_bilateral']:>8.2f} {rough['3_med_bilateral']:>8.2f}")


if __name__ == "__main__":
    main()
