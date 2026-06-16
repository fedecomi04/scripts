#!/usr/bin/env python3
"""Re-fuse a static TSDF seed from BILATERAL-FILTERED depth, non-destructively.

Same OnlineFusion (ICP + TSDF + adaptive downsample) as the real seed build, but
each depth frame is masked-bilateral filtered before integration. Writes a
SEPARATE PLY (depth_camera_init_points_bilateral.ply) so it can be compared to
the original depth_camera_init_points.ply — the original is never touched.

Usage:
  python scripts/fuse_bilateral_experiment.py <static_dir> [sigma_r_mm] [radius] [sigma_s_px]
    sigma_r_mm  range/depth Gaussian (mm), default 5
    radius      bilateral window radius (px), default 3  (7x7)
    sigma_s_px  spatial Gaussian (px), default 2

Set DGS_TSDF_VOXEL_M (default 0.003 = 3mm for 1200p GPU) as needed.
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from dynamic_gs.utils.online_fusion import (
    OnlineFusion, adaptive_downsample, NEAR_RADIUS_M, FAR_VOXEL_M, WITH_COLOR,
)
import open3d as o3d


def _windows(arr, radius, fill):
    H, W = arr.shape
    pad = np.full((H + 2 * radius, W + 2 * radius), fill, dtype=arr.dtype)
    pad[radius:radius + H, radius:radius + W] = arr
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yield dy, dx, pad[radius + dy:radius + dy + H, radius + dx:radius + dx + W]


def masked_bilateral(depth_m, valid, radius, sigma_s, sigma_r):
    """Edge-preserving bilateral over VALID neighbours only (normalized conv)."""
    dc = depth_m.astype(np.float32)
    vc = valid.astype(np.float32)
    num = np.zeros_like(dc); den = np.zeros_like(dc)
    vwins = list(_windows(vc, radius, 0.0))
    dwins = list(_windows(dc, radius, 0.0))
    for (dy, dx, vn), (_, _, dn) in zip(vwins, dwins):
        ws = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s))
        diff = dn - dc
        wr = np.exp(-(diff * diff) / (2.0 * sigma_r * sigma_r))
        w = ws * wr * vn
        num += w * dn; den += w
    return np.where((den > 0) & (vc > 0), num / np.maximum(den, 1e-12), 0.0).astype(np.float32)


def main():
    static = Path(sys.argv[1]).resolve()
    sigma_r_m = (float(sys.argv[2]) if len(sys.argv) > 2 else 5.0) / 1000.0
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    sigma_s = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    if static.name != "static_scene":
        static = static / "static_scene"
    meta = json.loads((static / "transforms.json").read_text())
    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    W, H = int(meta["w"]), int(meta["h"])
    frames = sorted(meta["frames"],
                    key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
    print(f"[bilat-fuse] {static} | {len(frames)} frames | bilateral r={radius} "
          f"sigma_s={sigma_s}px sigma_r={sigma_r_m*1000:.1f}mm")

    fuser = OnlineFusion(fx, fy, cx, cy, W, H)
    for fr in frames:
        dpath = static / fr["depth_file_path"].lstrip("./")
        mpath = static / fr["mask_path"].lstrip("./") if fr.get("mask_path") else None
        rpath = static / fr["file_path"].lstrip("./")
        depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
        if mpath and mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            depth[m == 0] = 0  # zero gripper BEFORE filtering so it can't bleed in
        # --- bilateral on metric depth, validity-masked (skipped if sigma_r<=0:
        #     plain re-fuse with the current fusion code, no filtering) ---
        if sigma_r_m > 0.0:
            d_m = depth.astype(np.float32) / 1000.0
            d_m = masked_bilateral(d_m, d_m > 0.0, radius, sigma_s, sigma_r_m)
            depth = np.clip(d_m * 1000.0, 0, 65535).astype(np.uint16)
        rgb = cv2.imread(str(rpath), cv2.IMREAD_COLOR)[:, :, ::-1].copy() if WITH_COLOR else None
        fuser.add_frame(depth, np.asarray(fr["transform_matrix"], dtype=np.float64), rgb)

    pc = fuser.finalize()
    n_full = len(pc.points)
    if n_full > 0:
        last_cam = np.asarray(frames[-1]["transform_matrix"], dtype=np.float64)[:3, 3]
        pc = adaptive_downsample(pc, last_cam)
    suffix = "bilateral" if sigma_r_m > 0.0 else "REFUSE"
    out = static / f"depth_camera_init_points_{suffix}.ply"
    o3d.io.write_point_cloud(str(out), pc)
    print(f"[bilat-fuse] filter={'bilateral' if sigma_r_m>0 else 'NONE (plain re-fuse)'} | "
          f"{n_full:,} -> {len(pc.points):,} pts | wrote {out}")


if __name__ == "__main__":
    main()
