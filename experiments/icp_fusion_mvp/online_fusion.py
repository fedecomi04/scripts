#!/usr/bin/env python3
# =============================================================================
# online_fusion.py  --  FINAL online RGB-D fusion (dense + precise point cloud).
#
# For a Gaussian-splat seed whose POSITIONS are frozen (no geometry optimization
# in training), so the cloud must be metrically accurate. Runs ONLINE: each frame
# is processed as it streams, so when capture stops the cloud is ready (only the
# final extract is left). Self-contained and SAM-FREE -- needs only RGB + depth +
# per-frame pose (transforms.json) + a robot/gripper mask.
#
# PER FRAME (must fit the inter-frame gap; ~242 ms measured here at 2.5 mm):
#   1. decimate depth (stride) -> back-project gripper-free points for ICP
#   2. point-to-plane ICP, init from the FK pose, against a GLOBAL voxel model
#      (1 cm; bounded by scene extent, not #frames; normals refreshed every K)
#   3. TSDF-integrate the FULL-RES depth at the refined pose
#   (decimation is ONLY for ICP; the TSDF always sees full depth -> precision kept)
# AT END: extract_point_cloud (~0.6 s).
#
# Measured (136-frame tabletop orbit, CPU): 242 ms/frame mean (100% < 300 ms),
# ~1.0 M points, 0.73 mm table thickness -- matches the 40 s batch ICP.
#
# Why these choices (all measured): point-to-plane (point-to-point was 3.3 mm vs
# 0.73 mm); ICP kept (raw-pose TSDF is 5.4 mm vs 0.9 mm); decimate ICP source not
# TSDF depth; CPU (Open3D GPU TSDF is broken on sm_120 -- ran in 108 s).
#
# RUN:  conda activate dynamic_gs
#       python experiments/icp_fusion_mvp/online_fusion.py
# =============================================================================

import os
import re
import sys
import json
import time

import numpy as np
import cv2
import open3d as o3d

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATASET_DIR = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "new_env/static_scene"
)
OUTPUT_PLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "online_seed.ply")

TSDF_VOXEL_M = 0.0015        # 1.5 mm  -> matches current pipeline (dynamic_gs/utils/online_fusion.py)
TSDF_TRUNC_M = 0.006         # ~4 x voxel -> matches current pipeline
DEPTH_SCALE = 1000.0         # uint16 mm -> m
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0

ICP_SRC_STRIDE = 4           # decimate depth for the ICP source (TSDF still gets full depth)
ICP_VOXEL_M = 0.01           # ICP source + model resolution
MODEL_REFRESH_EVERY = 5      # re-voxel/re-normal the global model every N frames
NORMAL_RADIUS_M = 0.03
ICP_STAGES = ((0.05, 6), (0.02, 12))   # (max_corr_dist, iterations) coarse->fine
ICP_FITNESS_MIN = 0.30       # below this, trust the FK pose
WITH_COLOR = True            # fuse real RGB (set False for geometry-only -> a bit faster)


# ----------------------------------------------------------------------------
# Online fusion engine. Mirrors the CURRENT PIPELINE (dynamic_gs/utils/online_fusion.py):
#   - GPU path: o3d.t.pipelines.slam.Model (VoxelBlockGrid TSDF) + multi_scale_icp on CUDA
#   - CPU path: legacy ScalableTSDFVolume + registration_icp (fallback)
#   - OnlineFusion auto-selects GPU when o3d.core.cuda.is_available();
#     DGS_FUSION_DEVICE=cpu forces the fallback.
# Same ICP stages / TSDF voxel / fitness gate / dedup as production.
# ----------------------------------------------------------------------------
class _CpuOnlineFusion:
    """Legacy ScalableTSDFVolume + registration_icp (CPU fallback)."""

    def __init__(self, fx, fy, cx, cy, W, H):
        self.fx, self.fy, self.cx, self.cy, self.W, self.H = fx, fy, cx, cy, W, H
        self.intr_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        ctype = (o3d.pipelines.integration.TSDFVolumeColorType.RGB8 if WITH_COLOR
                 else o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(TSDF_VOXEL_M, TSDF_TRUNC_M, ctype)
        self.estim = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        self.model = None
        self._pend = []
        self.idx = 0
        self._dummy = o3d.geometry.Image(np.zeros((H, W, 3), np.uint8))

    def _src_cloud(self, depth_u16, c2w_cv):
        v = (depth_u16 > DEPTH_MIN_M * DEPTH_SCALE) & (depth_u16 < DEPTH_MAX_M * DEPTH_SCALE)
        vv, uu = np.where(v)
        vv, uu = vv[::ICP_SRC_STRIDE], uu[::ICP_SRC_STRIDE]
        zz = depth_u16[vv, uu] / DEPTH_SCALE
        x = (uu - self.cx) * zz / self.fx
        y = (vv - self.cy) * zz / self.fy
        cam = np.stack([x, y, zz, np.ones_like(zz)], 1)
        world = (c2w_cv @ cam.T).T[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(world)
        pc = pc.voxel_down_sample(ICP_VOXEL_M)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
        return pc

    def _integrate(self, depth_u16, rgb_u8, c2w_cv):
        color = o3d.geometry.Image(np.ascontiguousarray(rgb_u8)) if (WITH_COLOR and rgb_u8 is not None) else self._dummy
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(np.ascontiguousarray(depth_u16)),
            depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_MAX_M, convert_rgb_to_intensity=False)
        self.vol.integrate(rgbd, self.intr_o3d, np.linalg.inv(c2w_cv))

    def add_frame(self, depth_u16, c2w_cv, rgb_u8=None):
        src = self._src_cloud(depth_u16, c2w_cv)
        if self.model is None:
            self.model = src
            self._integrate(depth_u16, rgb_u8, c2w_cv)
            self.idx += 1
            return c2w_cv
        T = np.eye(4)
        for dist, iters in ICP_STAGES:
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            reg = o3d.pipelines.registration.registration_icp(src, self.model, dist, T, self.estim, crit)
            T = reg.transformation
        refined = T @ c2w_cv if reg.fitness >= ICP_FITNESS_MIN else c2w_cv
        self._integrate(depth_u16, rgb_u8, refined)
        src.transform(refined @ np.linalg.inv(c2w_cv))
        self._pend.append(src)
        self.idx += 1
        if self.idx % MODEL_REFRESH_EVERY == 0:
            for s in self._pend:
                self.model += s
            self.model = self.model.voxel_down_sample(ICP_VOXEL_M)
            self.model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
            self._pend = []
        return refined

    def finalize(self):
        return self.vol.extract_point_cloud()


class _GpuOnlineFusion:
    """GPU: o3d.t.pipelines.slam.Model (VoxelBlockGrid TSDF) + multi_scale_icp on CUDA.
    Mirrors the CPU semantics frame-for-frame (same ICP stages, fitness gate, dedup)."""

    def __init__(self, fx, fy, cx, cy, W, H):
        import open3d.core as o3c
        self.fx, self.fy, self.cx, self.cy, self.W, self.H = float(fx), float(fy), float(cx), float(cy), int(W), int(H)
        self._dev = o3c.Device("CUDA:0")
        self._intrinsic_t = o3c.Tensor([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
                                       dtype=o3c.Dtype.Float64)
        self._slam = o3d.t.pipelines.slam.Model(TSDF_VOXEL_M, 16, 8000,
                                                o3c.Tensor(np.eye(4), o3c.Dtype.Float64), self._dev)
        self._voxel_sizes = o3d.utility.DoubleVector([ICP_VOXEL_M * 2, ICP_VOXEL_M])
        self._criteria = [o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_STAGES[0][1]),
                          o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_STAGES[1][1])]
        self._max_corr_dists = o3d.utility.DoubleVector([ICP_STAGES[0][0], ICP_STAGES[1][0]])
        self._estim = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        self._model_pcd = None
        self._pend = []
        self.idx = 0

    def _src_cloud(self, depth_u16, c2w_cv):
        import open3d.core as o3c
        depth_img = o3d.t.geometry.Image(o3c.Tensor(depth_u16.astype(np.uint16), device=self._dev))
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth=depth_img, intrinsics=self._intrinsic_t,
            extrinsics=o3c.Tensor(np.linalg.inv(c2w_cv).astype(np.float64), o3c.Dtype.Float64),
            depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX_M, stride=ICP_SRC_STRIDE)
        pcd = pcd.voxel_down_sample(ICP_VOXEL_M)
        pcd.estimate_normals(max_nn=30, radius=NORMAL_RADIUS_M)
        return pcd

    def _integrate(self, depth_u16, rgb_u8, c2w_cv):
        import open3d.core as o3c
        depth_img = o3d.t.geometry.Image(o3c.Tensor(depth_u16.astype(np.uint16), device=self._dev))
        if WITH_COLOR and rgb_u8 is not None:
            rgb_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb_u8), device=self._dev))
        else:
            rgb_img = o3d.t.geometry.Image(o3c.Tensor(np.zeros((self.H, self.W, 3), np.uint8), device=self._dev))
        extr_t = o3c.Tensor(np.linalg.inv(c2w_cv), o3c.Dtype.Float64)
        frustum = self._slam.voxel_grid.compute_unique_block_coordinates(
            depth_img, self._intrinsic_t, extr_t, DEPTH_SCALE, DEPTH_MAX_M)
        self._slam.voxel_grid.integrate(frustum, depth_img, rgb_img, self._intrinsic_t, self._intrinsic_t,
                                        extr_t, DEPTH_SCALE, DEPTH_MAX_M)

    def add_frame(self, depth_u16, c2w_cv, rgb_u8=None):
        import open3d.core as o3c
        src = self._src_cloud(depth_u16, c2w_cv)
        if self._model_pcd is None:
            self._model_pcd = src.clone()
            self._integrate(depth_u16, rgb_u8, c2w_cv)
            self.idx += 1
            return c2w_cv
        reg = o3d.t.pipelines.registration.multi_scale_icp(
            source=src, target=self._model_pcd, voxel_sizes=self._voxel_sizes,
            criteria_list=self._criteria, max_correspondence_distances=self._max_corr_dists,
            init_source_to_target=o3c.Tensor(np.eye(4), o3c.Dtype.Float64), estimation_method=self._estim)
        T = reg.transformation.cpu().numpy()
        refined = T @ c2w_cv if float(reg.fitness) >= ICP_FITNESS_MIN else c2w_cv
        self._integrate(depth_u16, rgb_u8, refined)
        src.transform(o3c.Tensor((refined @ np.linalg.inv(c2w_cv)).astype(np.float64), o3c.Dtype.Float64))
        self._pend.append(src)
        self.idx += 1
        if self.idx % MODEL_REFRESH_EVERY == 0:
            for s in self._pend:
                self._model_pcd = self._model_pcd.append(s)
            self._model_pcd = self._model_pcd.voxel_down_sample(ICP_VOXEL_M)
            self._model_pcd.estimate_normals(max_nn=30, radius=NORMAL_RADIUS_M)
            self._pend = []
        return refined

    def finalize(self):
        return self._slam.voxel_grid.extract_point_cloud().to_legacy()


class OnlineFusion:
    """Auto-selects GPU (Open3D tensor SLAM pipeline) when CUDA is available;
    falls back to CPU. DGS_FUSION_DEVICE=cpu forces the CPU path."""

    def __init__(self, fx, fy, cx, cy, W, H):
        force = os.environ.get("DGS_FUSION_DEVICE", "auto").lower()
        use_gpu = force == "gpu" or (force == "auto" and o3d.core.cuda.is_available())
        if use_gpu:
            try:
                self._impl = _GpuOnlineFusion(fx, fy, cx, cy, W, H)
                self.device = "gpu"
            except Exception as exc:
                if force == "gpu":
                    raise
                print(f"[OnlineFusion] GPU init failed ({exc}); CPU fallback", flush=True)
                self._impl = _CpuOnlineFusion(fx, fy, cx, cy, W, H)
                self.device = "cpu"
        else:
            self._impl = _CpuOnlineFusion(fx, fy, cx, cy, W, H)
            self.device = "cpu"

    @staticmethod
    def _cv_c2w(c2w_opengl):
        return np.asarray(c2w_opengl, np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])

    @property
    def idx(self):
        return self._impl.idx

    def add_frame(self, depth_u16, c2w_opengl, rgb_u8=None):
        """Process one streamed frame. depth_u16: gripper already zeroed (uint16 mm).
        Returns the ICP-refined camera-to-world (OpenCV convention)."""
        return self._impl.add_frame(depth_u16, self._cv_c2w(c2w_opengl), rgb_u8)

    def finalize(self):
        return self._impl.finalize()


# ----------------------------------------------------------------------------
# Minimal SAM-free dataset loaders
# ----------------------------------------------------------------------------
def load_dataset(root):
    meta = json.load(open(os.path.join(root, "transforms.json")))
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=int(meta["w"]), h=int(meta["h"]))
    key = lambda fr: int(re.findall(r"\d+", os.path.basename(fr["file_path"]))[-1])
    return sorted(meta["frames"], key=key), intr


def _ap(root, rel):
    return os.path.join(root, rel.lstrip("./"))


def load_depth_gripper_zeroed(root, fr):
    z = cv2.imread(_ap(root, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
    if fr.get("mask_path"):
        m = cv2.imread(_ap(root, fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
        z[m == 0] = 0                                            # robot/gripper excluded
    return z


def load_rgb(root, fr):
    return cv2.imread(_ap(root, fr["file_path"]), cv2.IMREAD_COLOR)[:, :, ::-1].copy()


# ----------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)
    frames, intr = load_dataset(DATASET_DIR)
    N = len(frames)
    print(f"[online_fusion] {N} frames | TSDF {TSDF_VOXEL_M*1000:.1f} mm | color={WITH_COLOR}")
    fuser = OnlineFusion(intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["w"], intr["h"])

    per_frame = []
    for i, fr in enumerate(frames):
        depth = load_depth_gripper_zeroed(DATASET_DIR, fr)       # (live: arrives from the camera)
        rgb = load_rgb(DATASET_DIR, fr) if WITH_COLOR else None
        t = time.time()
        fuser.add_frame(depth, fr["transform_matrix"], rgb)      # the per-frame online work
        per_frame.append(time.time() - t)
        if i % 20 == 0 or i == N - 1:
            print(f"  frame {i+1}/{N}  add_frame {1000*per_frame[-1]:.0f} ms")

    t = time.time()
    pc = fuser.finalize()
    o3d.io.write_point_cloud(OUTPUT_PLY, pc)
    t_ext = time.time() - t

    pf = np.array(per_frame[5:]) * 1000                          # skip warm-up
    print(f"\n[online_fusion] per-frame add_frame: mean {pf.mean():.0f} ms  p90 {np.percentile(pf,90):.0f} ms  max {pf.max():.0f} ms")
    print(f"                fits 300 ms: {(pf<300).mean()*100:.0f}% of frames")
    print(f"                end-of-capture extract: {t_ext:.1f} s")
    print(f"[out] {len(pc.points):,} points -> {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
