"""Prototype: CPU TSDF integrate-during-sweep. Mirrors online_fusion.fuse_recorded_dataset but
forces the CPU path and TIMES per-frame add_frame, so we can decide whether CPU TSDF can keep up
with a ~10 Hz live sweep (100 ms/frame budget) -> kill the 12 s cold-subprocess GPU build by
building the seed incrementally as static frames arrive (then it's ready at trigger, ~0 s).

Two modes:
  icp    (default) — full add_frame (per-frame multi-scale ICP + integrate), as production does.
  noicp           — integrate-only at the FK pose (skip ICP). The recorded transforms.json is
                    already ICP-refined (invariant #3), so for a fresh live capture the question is
                    whether we even need per-frame ICP in the seed.

Run: python -m dynamic_gs2.verify._probe_cpu_tsdf_sweep [dataset_dir] [icp|noicp]
"""
import sys, time, json, re
from pathlib import Path
import numpy as np, cv2

import dynamic_gs2.online_fusion as OF

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_162315_live").resolve()
MODE = sys.argv[2] if len(sys.argv) > 2 else "icp"
static_dir = DS / "static_scene"
meta = json.loads((static_dir / "transforms.json").read_text())
fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
cx, cy = float(meta["cx"]), float(meta["cy"])
W, H = int(meta["w"]), int(meta["h"])
frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
print(f"dataset={DS.name} frames={len(frames)} res={W}x{H} mode={MODE} "
      f"voxel={OF.TSDF_VOXEL_M*1000:.0f}mm trunc={OF.TSDF_TRUNC_M*1000:.0f}mm")

import os
os.environ["DGS_FUSION_DEVICE"] = "cpu"          # force the CPU path
fuser = OF.OnlineFusion(fx, fy, cx, cy, W, H)

def load(fr):
    depth = cv2.imread(str(static_dir / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
    mp = fr.get("mask_path") or fr.get("mask_file_path")
    if mp:
        m = cv2.imread(str(static_dir / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE)
        if m is not None: depth[m == 0] = 0
    return depth, np.asarray(fr["transform_matrix"], dtype=np.float64)

per = []
t_all0 = time.time()
for k, fr in enumerate(frames):
    depth, c2w_gl = load(fr)
    t0 = time.time()
    if MODE == "noicp":
        # integrate-only at the FK pose (bypass the ICP in add_frame)
        c2w_cv = OF.OnlineFusion._cv_c2w(c2w_gl)
        fuser._impl._integrate(depth, None, c2w_cv)
        fuser._impl.idx += 1
    else:
        fuser.add_frame(depth, c2w_gl, None)
    dt = time.time() - t0
    per.append(dt)
    if k < 3 or k == len(frames) - 1:
        print(f"  frame {k:3d}: {dt*1000:7.1f} ms")
t_int = time.time() - t_all0

t0 = time.time()
pc = fuser.finalize()
n_full = len(pc.points)
if n_full > 0:
    last_cam = np.asarray(frames[-1]["transform_matrix"], dtype=np.float64)[:3, 3]
    pc = OF.adaptive_downsample(pc, last_cam)
t_fin = time.time() - t0

per = np.array(per)
budget = 0.100   # 10 Hz
print(f"\n--- CPU TSDF sweep ({MODE}) ---")
print(f"per-frame add: median={np.median(per)*1000:.1f} ms  mean={per.mean()*1000:.1f} ms  "
      f"max={per.max()*1000:.1f} ms")
print(f"keeps up @10Hz (<100ms)? {'YES' if np.median(per) < budget else 'NO'}  "
      f"({(per < budget).sum()}/{len(per)} frames under budget)")
print(f"total integrate (all {len(frames)} frames) = {t_int:.2f} s   finalize+downsample = {t_fin:.2f} s")
print(f"seed points: {n_full:,} -> {len(pc.points):,} after downsample")
print(f">>> If incremental-during-sweep: trigger-time cost = ONLY finalize = {t_fin:.2f} s (vs ~12 s GPU cold subprocess)")
print(f">>> If batch-at-trigger (CPU): {t_int + t_fin:.2f} s total at trigger")
