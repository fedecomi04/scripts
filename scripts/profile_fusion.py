"""Instrument OnlineFusion.add_frame and report where the time goes.

Loads validate_run_1, runs fusion on 30 frames (warm-up skip 5), prints
per-substep timing breakdown so we know what to optimize.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dynamic_gs.utils import online_fusion as of  # noqa: E402

DATA = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/validate_run_1/static_scene"


def main() -> None:
    import json
    import re

    meta = json.loads(open(f"{DATA}/transforms.json").read())
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
    fx, fy = meta["fl_x"], meta["fl_y"]
    cx, cy = meta["cx"], meta["cy"]
    W, H = int(meta["w"]), int(meta["h"])

    # Wrap each substep with timers by monkey-patching the OnlineFusion methods.
    fuser = of.OnlineFusion(fx, fy, cx, cy, W, H)
    timings: dict = {"src_cloud": [], "icp_total": [], "integrate": [], "model_refresh": [], "total": []}

    # Wrap _src_cloud
    _orig_src = fuser._src_cloud
    def _timed_src(depth, c2w):
        t = time.time()
        r = _orig_src(depth, c2w)
        timings["src_cloud"].append(1000 * (time.time() - t))
        return r
    fuser._src_cloud = _timed_src  # type: ignore

    # Wrap _integrate
    _orig_int = fuser._integrate
    def _timed_int(depth, rgb, c2w):
        t = time.time()
        r = _orig_int(depth, rgb, c2w)
        timings["integrate"].append(1000 * (time.time() - t))
        return r
    fuser._integrate = _timed_int  # type: ignore

    # Time the ICP block + model refresh manually by overriding add_frame.
    _orig_add = fuser.add_frame
    def _timed_add(depth, c2w_opengl, rgb=None):
        n_before = fuser.idx
        n_pending_before = len(fuser._pend)
        t0 = time.time()
        c2w_cv = fuser._cv_c2w(np.asarray(c2w_opengl, np.float64))
        src = fuser._src_cloud(depth, c2w_cv)
        if fuser.model is None:
            fuser.model = src
            fuser._integrate(depth, rgb, c2w_cv)
            fuser.idx += 1
            timings["icp_total"].append(0.0)
            timings["model_refresh"].append(0.0)
            timings["total"].append(1000 * (time.time() - t0))
            return c2w_cv
        T = np.eye(4)
        t_icp = time.time()
        for dist, iters in of.ICP_STAGES:
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            reg = o3d.pipelines.registration.registration_icp(src, fuser.model, dist, T, fuser.estim, crit)
            T = reg.transformation
        timings["icp_total"].append(1000 * (time.time() - t_icp))
        refined = T @ c2w_cv if reg.fitness >= of.ICP_FITNESS_MIN else c2w_cv
        fuser._integrate(depth, rgb, refined)
        src.transform(refined @ np.linalg.inv(c2w_cv))
        fuser._pend.append(src)
        fuser.idx += 1
        t_ref = time.time()
        if fuser.idx % of.MODEL_REFRESH_EVERY == 0:
            for s in fuser._pend:
                fuser.model += s
            fuser.model = fuser.model.voxel_down_sample(of.ICP_VOXEL_M)
            fuser.model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=of.NORMAL_RADIUS_M, max_nn=30))
            fuser._pend = []
            timings["model_refresh"].append(1000 * (time.time() - t_ref))
        else:
            timings["model_refresh"].append(0.0)
        timings["total"].append(1000 * (time.time() - t0))
        return refined
    fuser.add_frame = _timed_add  # type: ignore

    N = min(30, len(frames))
    print(f"Profiling {N} frames; warmup skip = 5")
    for i in range(N):
        fr = frames[i]
        depth = cv2.imread(f"{DATA}/{fr['depth_file_path'].lstrip('./')}", cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
        m = cv2.imread(f"{DATA}/{fr['mask_path'].lstrip('./')}", cv2.IMREAD_GRAYSCALE)
        depth[m == 0] = 0
        rgb = cv2.imread(f"{DATA}/{fr['file_path'].lstrip('./')}", cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        fuser.add_frame(depth, np.asarray(fr["transform_matrix"], np.float64), rgb)
        if i % 5 == 0 or i == N - 1:
            print(f"  frame {i:>3d}: total {timings['total'][-1]:>6.1f} ms  icp {timings['icp_total'][-1]:>6.1f}  src {timings['src_cloud'][-1]:>6.1f}  int {timings['integrate'][-1]:>6.1f}  refresh {timings['model_refresh'][-1]:>6.1f}")

    print("\n=== Profile summary (skip first 5 = warm-up) ===")
    for k in ("src_cloud", "icp_total", "integrate", "model_refresh", "total"):
        arr = np.array(timings[k][5:])
        if arr.size == 0:
            continue
        print(f"  {k:<14s}  mean {arr.mean():>6.1f}  p50 {np.percentile(arr,50):>6.1f}  p90 {np.percentile(arr,90):>6.1f}  max {arr.max():>6.1f}  ms  (n={arr.size})")


if __name__ == "__main__":
    main()
