"""Sweep TSDF voxel size on the GPU path; measure per-frame timings,
final point count, and approximate VRAM use.

Same dataset / N_FRAMES / WARMUP_SKIP as bench_gpu_fusion.py. For each
voxel size, run the full GPU fusion over 30 frames, then extract the
fused cloud and print:
  - per-frame integrate / total ms (mean / p90 / max)
  - final point count
  - VRAM consumed by the TSDF voxel block grid (peak)

Voxel sizes tested: 2.5, 1.5, 1.0, 0.5 mm.

Truncation distance is scaled to ~4× voxel (matches OnlineFusion
convention). Block count is enlarged for finer voxels so we don't OOM
on hash-table allocation.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

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


def vram_mb():
    """Peak VRAM in MB allocated by the current process via nvidia-smi."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return float(out[0])
    except Exception:
        return -1.0


def run_gpu_for_voxel(frames, intr, voxel_size, block_count):
    dev = o3c.Device("CUDA:0")
    intrinsic_tensor = o3c.Tensor(
        [[intr["fx"], 0, intr["cx"]],
         [0, intr["fy"], intr["cy"]],
         [0, 0, 1]],
        dtype=o3c.Dtype.Float64,
    )
    trunc = max(voxel_size * 4.0, 0.005)
    slam_model = o3d.t.pipelines.slam.Model(
        voxel_size, 16, block_count,
        o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
        dev,
    )
    voxel_sizes_icp = o3d.utility.DoubleVector([of.ICP_VOXEL_M*2, of.ICP_VOXEL_M])
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=of.ICP_STAGES[0][1]),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=of.ICP_STAGES[1][1]),
    ]
    max_corr = o3d.utility.DoubleVector([of.ICP_STAGES[0][0], of.ICP_STAGES[1][0]])
    estim = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    model_pcd = None
    pending = []
    idx = 0
    t_int, t_total = [], []

    def _sync():
        o3c.cuda.synchronize()

    def _src(depth_u16, c2w_cv):
        depth_t = o3c.Tensor(depth_u16.astype(np.uint16), device=dev)
        depth_img = o3d.t.geometry.Image(depth_t)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth=depth_img, intrinsics=intrinsic_tensor,
            extrinsics=o3c.Tensor(np.linalg.inv(c2w_cv).astype(np.float64), o3c.Dtype.Float64),
            depth_scale=of.DEPTH_SCALE, depth_max=of.DEPTH_MAX_M,
            stride=of.ICP_SRC_STRIDE,
        )
        pcd = pcd.voxel_down_sample(of.ICP_VOXEL_M)
        pcd.estimate_normals(max_nn=30, radius=of.NORMAL_RADIUS_M)
        return pcd

    def _integrate(depth, rgb, c2w_cv):
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

    for i, (depth, rgb, c2w_opengl) in enumerate(frames):
        t0 = time.time()
        c2w_cv = opengl_to_cv(np.asarray(c2w_opengl, np.float64))
        src = _src(depth, c2w_cv)
        if model_pcd is None:
            model_pcd = src.clone()
            t = time.time(); _integrate(depth, rgb, c2w_cv); _sync(); t_int.append(1000*(time.time()-t))
            idx += 1
            t_total.append(1000*(time.time()-t0))
            continue
        reg = o3d.t.pipelines.registration.multi_scale_icp(
            source=src, target=model_pcd,
            voxel_sizes=voxel_sizes_icp, criteria_list=criteria_list,
            max_correspondence_distances=max_corr,
            init_source_to_target=o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
            estimation_method=estim,
        )
        T = reg.transformation.cpu().numpy()
        fitness = float(reg.fitness)
        refined = T @ c2w_cv if fitness >= of.ICP_FITNESS_MIN else c2w_cv
        t = time.time(); _integrate(depth, rgb, refined); _sync(); t_int.append(1000*(time.time()-t))
        src.transform(o3c.Tensor((refined @ np.linalg.inv(c2w_cv)).astype(np.float64), o3c.Dtype.Float64))
        pending.append(src)
        idx += 1
        if idx % of.MODEL_REFRESH_EVERY == 0:
            for s in pending:
                model_pcd = model_pcd.append(s)
            model_pcd = model_pcd.voxel_down_sample(of.ICP_VOXEL_M)
            model_pcd.estimate_normals(max_nn=30, radius=of.NORMAL_RADIUS_M)
            pending = []
            _sync()
        t_total.append(1000*(time.time()-t0))

    # Extract final cloud.
    _sync()
    t = time.time()
    pc = slam_model.voxel_grid.extract_point_cloud()
    _sync()
    extract_s = time.time() - t
    n_pts = int(pc.point.positions.shape[0])

    # Save PLY into a sibling tmp dir so the source seed isn't touched.
    out_dir = Path(DATA).parent / "_voxel_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"voxel_{int(round(voxel_size*1000*10)):03d}_x0p1mm.ply"
    # Convert tensor pointcloud → legacy for write_point_cloud color support
    pc_legacy = pc.to_legacy()
    o3d.io.write_point_cloud(str(ply_path), pc_legacy)
    print(f"  wrote {ply_path}")

    return {
        "int_ms": np.array(t_int[WARMUP_SKIP:]),
        "total_ms": np.array(t_total[WARMUP_SKIP:]),
        "n_pts": n_pts,
        "extract_s": extract_s,
        "trunc_m": trunc,
        "ply_path": str(ply_path),
    }


def main():
    frames, intr = load_frames()
    print(f"Loaded {len(frames)} frames @ {intr['W']}x{intr['H']}")
    print(f"ICP stages: {of.ICP_STAGES}\n")

    # Each block stores 16^3 = 4096 voxels. Per-block VRAM (TSDF + weight + RGB):
    # tsdf(f32) + weight(f32) + rgb(u8x3) = 11 B per voxel = 45 KB per block.
    # Try modest block counts; the SLAM model only allocates the hash table up
    # front and grows lazily, so over-provisioning is safe up to GPU VRAM.
    # block_count is the INITIAL hashmap capacity; Open3D rehashes when it
    # grows past load factor. The constructor allocates per-block VRAM
    # (TSDF + weight + RGB + free-list bookkeeping) up-front, so we want
    # the smallest initial count that still avoids constant rehashing.
    # Voxel halving doesn't 8x the block count because most blocks stay
    # sparse — empirically each block holds ~surface-area-passes worth
    # of voxels. Picking a conservative ramp:
    sweeps = [
        # (voxel_m, block_count)
        (0.0025, 8_000),
        (0.0015, 16_000),
        (0.0010, 32_000),
        (0.0005, 80_000),
    ]
    only = None
    if len(sys.argv) > 1:
        only = float(sys.argv[1])
    if only is not None:
        sweeps = [(s[0], s[1]) for s in sweeps if abs(s[0] * 1000 - only) < 0.01]
    results = []
    for voxel, blocks in sweeps:
        import gc as _gc
        _gc.collect()
        o3c.cuda.synchronize()
        vram_before = vram_mb()
        print(f"--- voxel {voxel*1000:.1f} mm  (block_count={blocks}) ---")
        try:
            r = run_gpu_for_voxel(frames, intr, voxel, blocks)
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            continue
        vram_after = vram_mb()
        r["vram_delta_mb"] = vram_after - vram_before
        r["voxel_mm"] = voxel * 1000
        r["trunc_mm"] = r["trunc_m"] * 1000
        results.append(r)
        print(f"  integrate (per-frame, skip {WARMUP_SKIP}):  mean {r['int_ms'].mean():>6.1f}  p90 {np.percentile(r['int_ms'],90):>6.1f}  max {r['int_ms'].max():>6.1f}  ms")
        print(f"  total     (per-frame, skip {WARMUP_SKIP}):  mean {r['total_ms'].mean():>6.1f}  p90 {np.percentile(r['total_ms'],90):>6.1f}  max {r['total_ms'].max():>6.1f}  ms")
        print(f"  final points: {r['n_pts']:>10,}    extract: {r['extract_s']:.2f}s    vram Δ: {r['vram_delta_mb']:.0f} MB\n")

    print("=== Summary ===")
    print(f"{'voxel(mm)':>10s}  {'trunc(mm)':>9s}  {'int mean':>9s}  {'int p90':>8s}  {'total mean':>10s}  {'points':>12s}  {'vramΔ':>7s}")
    for r in results:
        print(f"{r['voxel_mm']:>10.1f}  {r['trunc_mm']:>9.1f}  {r['int_ms'].mean():>8.1f}  {np.percentile(r['int_ms'],90):>7.1f}  {r['total_ms'].mean():>9.1f}  {r['n_pts']:>12,}  {r['vram_delta_mb']:>6.0f}")


if __name__ == "__main__":
    main()
