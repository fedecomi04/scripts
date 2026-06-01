"""Benchmark CPU vs GPU TSDF + ICP on validate_run_1.

Two implementations, same dataset, same 30 frames (warm-up skip 5):

  CPU baseline (current production)
    - o3d.geometry.PointCloud (legacy) for ICP source + model
    - o3d.pipelines.integration.ScalableTSDFVolume for fusion
    - o3d.pipelines.registration.registration_icp (CPU)

  GPU path
    - o3d.t.geometry.PointCloud on CUDA for ICP source
    - o3d.t.pipelines.registration.icp (CUDA, multi-scale)
    - o3d.t.pipelines.slam.Model on CUDA for TSDF (VoxelBlockGrid)

Both paths run on the same 30 frames; per-substep timings printed.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dynamic_gs.utils import online_fusion as of  # noqa: E402

DATA = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/validate_run_1/static_scene"
N_FRAMES = 30
WARMUP_SKIP = 5


# ----------------------------------------------------------------------------
# Common loader
# ----------------------------------------------------------------------------

def load_frames():
    meta = json.loads(open(f"{DATA}/transforms.json").read())
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
    intr = dict(
        fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
        W=int(meta["w"]), H=int(meta["h"]),
    )
    out = []
    for fr in frames[:N_FRAMES]:
        depth = cv2.imread(f"{DATA}/{fr['depth_file_path'].lstrip('./')}", cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
        m = cv2.imread(f"{DATA}/{fr['mask_path'].lstrip('./')}", cv2.IMREAD_GRAYSCALE)
        depth[m == 0] = 0
        rgb = cv2.imread(f"{DATA}/{fr['file_path'].lstrip('./')}", cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        c2w = np.asarray(fr["transform_matrix"], np.float64)
        out.append((depth, rgb, c2w))
    return out, intr


def opengl_to_cv(c2w_opengl):
    return c2w_opengl @ np.diag([1.0, -1.0, -1.0, 1.0])


def stats(name, arr_ms):
    arr = np.asarray(arr_ms[WARMUP_SKIP:])
    if arr.size == 0:
        print(f"  {name:<14s} (no data)")
        return None
    print(f"  {name:<14s}  mean {arr.mean():>6.1f}  p50 {np.percentile(arr,50):>6.1f}  "
          f"p90 {np.percentile(arr,90):>6.1f}  max {arr.max():>6.1f}  ms  (n={arr.size})")
    return arr.mean()


# ----------------------------------------------------------------------------
# CPU baseline — mirror OnlineFusion exactly so timings are comparable
# ----------------------------------------------------------------------------

def run_cpu(frames, intr):
    fuser = of.OnlineFusion(intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["W"], intr["H"])
    timings = {"src": [], "icp": [], "int": [], "ref": [], "total": []}
    for i, (depth, rgb, c2w_opengl) in enumerate(frames):
        t0 = time.time()
        c2w_cv = opengl_to_cv(np.asarray(c2w_opengl, np.float64))
        t = time.time(); src = fuser._src_cloud(depth, c2w_cv); timings["src"].append(1000*(time.time()-t))
        if fuser.model is None:
            fuser.model = src
            t = time.time(); fuser._integrate(depth, rgb, c2w_cv); timings["int"].append(1000*(time.time()-t))
            timings["icp"].append(0.0); timings["ref"].append(0.0); fuser.idx += 1
            timings["total"].append(1000*(time.time()-t0))
            continue
        T = np.eye(4)
        t = time.time()
        for dist, iters in of.ICP_STAGES:
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            reg = o3d.pipelines.registration.registration_icp(src, fuser.model, dist, T, fuser.estim, crit)
            T = reg.transformation
        timings["icp"].append(1000*(time.time()-t))
        refined = T @ c2w_cv if reg.fitness >= of.ICP_FITNESS_MIN else c2w_cv
        t = time.time(); fuser._integrate(depth, rgb, refined); timings["int"].append(1000*(time.time()-t))
        src.transform(refined @ np.linalg.inv(c2w_cv))
        fuser._pend.append(src); fuser.idx += 1
        t = time.time()
        if fuser.idx % of.MODEL_REFRESH_EVERY == 0:
            for s in fuser._pend: fuser.model += s
            fuser.model = fuser.model.voxel_down_sample(of.ICP_VOXEL_M)
            fuser.model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=of.NORMAL_RADIUS_M, max_nn=30))
            fuser._pend = []
        timings["ref"].append(1000*(time.time()-t))
        timings["total"].append(1000*(time.time()-t0))
        if i % 5 == 0 or i == N_FRAMES-1:
            print(f"  [cpu] frame {i:>3d}: total {timings['total'][-1]:>6.1f} ms")
    return timings


# ----------------------------------------------------------------------------
# GPU path
# ----------------------------------------------------------------------------

def run_gpu(frames, intr):
    dev = o3c.Device("CUDA:0")
    intrinsic_tensor = o3c.Tensor(
        [[intr["fx"], 0, intr["cx"]],
         [0, intr["fy"], intr["cy"]],
         [0, 0, 1]],
        dtype=o3c.Dtype.Float64,
    )
    # Build a SLAM model: TSDF voxel block grid on CUDA.
    voxel_size = of.TSDF_VOXEL_M
    block_resolution = 16
    block_count = 40000
    slam_model = o3d.t.pipelines.slam.Model(
        voxel_size, block_resolution, block_count,
        o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
        dev,
    )
    # ICP stages mirror the CPU path.
    voxel_sizes = o3d.utility.DoubleVector([of.ICP_VOXEL_M*2, of.ICP_VOXEL_M])
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=of.ICP_STAGES[0][1]),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=of.ICP_STAGES[1][1]),
    ]
    max_correspondence_distances = o3d.utility.DoubleVector([of.ICP_STAGES[0][0], of.ICP_STAGES[1][0]])
    estim = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

    model_pcd = None  # accumulator on GPU
    pending: List[o3d.t.geometry.PointCloud] = []
    idx = 0
    timings = {"depth_upload": [], "src": [], "icp": [], "int": [], "ref": [], "total": [], "sync": []}

    def _sync():
        # cuda is async; cuda.synchronize forces completion
        o3c.cuda.synchronize()

    def _src_cloud_gpu(depth_u16, c2w_cv):
        # Upload depth to GPU, back-project, voxel-down, estimate normals.
        t = time.time()
        depth_t = o3c.Tensor(depth_u16.astype(np.uint16), device=dev)
        depth_img = o3d.t.geometry.Image(depth_t)
        timings["depth_upload"].append(1000*(time.time()-t))
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth=depth_img,
            intrinsics=intrinsic_tensor,
            extrinsics=o3c.Tensor(np.linalg.inv(c2w_cv).astype(np.float64), o3c.Dtype.Float64),
            depth_scale=of.DEPTH_SCALE,
            depth_max=of.DEPTH_MAX_M,
            stride=of.ICP_SRC_STRIDE,
        )
        # voxel-down + normals
        pcd = pcd.voxel_down_sample(of.ICP_VOXEL_M)
        pcd.estimate_normals(max_nn=30, radius=of.NORMAL_RADIUS_M)
        return pcd

    for i, (depth, rgb, c2w_opengl) in enumerate(frames):
        t0 = time.time()
        c2w_cv = opengl_to_cv(np.asarray(c2w_opengl, np.float64))

        t = time.time(); src = _src_cloud_gpu(depth, c2w_cv); _sync(); timings["src"].append(1000*(time.time()-t))

        if model_pcd is None:
            model_pcd = src.clone()
            # TSDF integrate first frame
            t = time.time()
            depth_t = o3c.Tensor(depth.astype(np.uint16), device=dev)
            depth_img = o3d.t.geometry.Image(depth_t)
            rgb_t = o3c.Tensor(np.ascontiguousarray(rgb), device=dev)
            rgb_img = o3d.t.geometry.Image(rgb_t)
            frustum = slam_model.voxel_grid.compute_unique_block_coordinates(
                depth_img, intrinsic_tensor, o3c.Tensor(np.linalg.inv(c2w_cv), o3c.Dtype.Float64),
                of.DEPTH_SCALE, of.DEPTH_MAX_M,
            )
            slam_model.voxel_grid.integrate(
                frustum, depth_img, rgb_img,
                intrinsic_tensor, intrinsic_tensor,
                o3c.Tensor(np.linalg.inv(c2w_cv), o3c.Dtype.Float64),
                of.DEPTH_SCALE, of.DEPTH_MAX_M,
            )
            _sync(); timings["int"].append(1000*(time.time()-t))
            timings["icp"].append(0.0); timings["ref"].append(0.0)
            idx += 1
            timings["total"].append(1000*(time.time()-t0))
            continue

        # multi-scale ICP source vs model
        t = time.time()
        reg = o3d.t.pipelines.registration.multi_scale_icp(
            source=src,
            target=model_pcd,
            voxel_sizes=voxel_sizes,
            criteria_list=criteria_list,
            max_correspondence_distances=max_correspondence_distances,
            init_source_to_target=o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
            estimation_method=estim,
        )
        T = reg.transformation.cpu().numpy()
        fitness = float(reg.fitness)
        _sync(); timings["icp"].append(1000*(time.time()-t))

        refined = T @ c2w_cv if fitness >= of.ICP_FITNESS_MIN else c2w_cv

        # TSDF integrate at refined pose
        t = time.time()
        depth_t = o3c.Tensor(depth.astype(np.uint16), device=dev)
        depth_img = o3d.t.geometry.Image(depth_t)
        rgb_t = o3c.Tensor(np.ascontiguousarray(rgb), device=dev)
        rgb_img = o3d.t.geometry.Image(rgb_t)
        frustum = slam_model.voxel_grid.compute_unique_block_coordinates(
            depth_img, intrinsic_tensor, o3c.Tensor(np.linalg.inv(refined), o3c.Dtype.Float64),
            of.DEPTH_SCALE, of.DEPTH_MAX_M,
        )
        slam_model.voxel_grid.integrate(
            frustum, depth_img, rgb_img,
            intrinsic_tensor, intrinsic_tensor,
            o3c.Tensor(np.linalg.inv(refined), o3c.Dtype.Float64),
            of.DEPTH_SCALE, of.DEPTH_MAX_M,
        )
        _sync(); timings["int"].append(1000*(time.time()-t))

        # Update model accumulator (same strategy as CPU: append + periodic refresh)
        src.transform(o3c.Tensor((refined @ np.linalg.inv(c2w_cv)).astype(np.float64), o3c.Dtype.Float64))
        pending.append(src)
        idx += 1
        t = time.time()
        if idx % of.MODEL_REFRESH_EVERY == 0:
            for s in pending:
                model_pcd = model_pcd.append(s)
            model_pcd = model_pcd.voxel_down_sample(of.ICP_VOXEL_M)
            model_pcd.estimate_normals(max_nn=30, radius=of.NORMAL_RADIUS_M)
            pending = []
            _sync()
        timings["ref"].append(1000*(time.time()-t))
        timings["total"].append(1000*(time.time()-t0))
        if i % 5 == 0 or i == N_FRAMES-1:
            print(f"  [gpu] frame {i:>3d}: total {timings['total'][-1]:>6.1f} ms")
    return timings, slam_model


# ----------------------------------------------------------------------------
def main():
    frames, intr = load_frames()
    print(f"Loaded {len(frames)} frames @ {intr['W']}x{intr['H']}")
    print(f"TSDF voxel: {of.TSDF_VOXEL_M*1000:.1f} mm  ICP stages: {of.ICP_STAGES}\n")

    print("=== CPU baseline ===")
    cpu_t = run_cpu(frames, intr)
    print("\n--- CPU summary (skip first 5 = warmup) ---")
    cpu_means = {k: stats(k, cpu_t[k]) for k in ("src", "icp", "int", "ref", "total")}

    print("\n=== GPU path ===")
    gpu_t, _ = run_gpu(frames, intr)
    print("\n--- GPU summary (skip first 5 = warmup) ---")
    gpu_means = {k: stats(k, gpu_t[k]) for k in ("src", "icp", "int", "ref", "total")}

    print("\n=== Speedup (CPU / GPU, mean) ===")
    for k in ("src", "icp", "int", "ref", "total"):
        if cpu_means[k] and gpu_means[k]:
            print(f"  {k:<14s}  {cpu_means[k]:>6.1f} → {gpu_means[k]:>6.1f}  ms  "
                  f"({cpu_means[k]/max(gpu_means[k],1e-3):>5.1f}× faster)")


if __name__ == "__main__":
    main()
