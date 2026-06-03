"""Online RGB-D fusion for the static-GS init seed.

Verbatim port of ``experiments/icp_fusion_mvp/online_fusion.py`` into a
reusable utility. Algorithm (proven, measured, SAM-free):

  Per frame
    1. Decimate depth (``ICP_SRC_STRIDE``) → back-project gripper-free
       points for ICP source.
    2. Point-to-plane ICP, init from the FK pose, against a GLOBAL
       voxel model (1 cm; bounded by scene extent, not #frames; normals
       refreshed every ``MODEL_REFRESH_EVERY`` frames).
    3. TSDF-integrate the FULL-RES depth at the refined pose.
    (Decimation is ONLY for ICP; the TSDF always sees full depth →
     precision kept.)

  At end
    ``finalize()`` → ``extract_point_cloud`` (~0.6 s).

  Measured (136-frame tabletop orbit, CPU):
    add_frame: mean 242 ms (100 % < 300 ms)
    finalize: ~0.6 s
    output: ~1.0 M points, 0.73 mm table thickness — matches the 40 s
    batch ICP.

Conventions (validated upstream — keep exactly):
* Pose is OpenGL c2w; ``OnlineFusion`` converts internally
  (``c2w @ diag(1, -1, -1, 1)`` → OpenCV).
* Depth is uint16 mm; robot/gripper pixels must be **pre-zeroed** by
  the caller (mask==0 → depth==0 → excluded from both ICP and TSDF).
* World frame is unrecentered. Splatfacto's dataparser is set to
  ``orientation_method="none"``, ``center_method="none"``,
  ``auto_scale_poses=False`` so the seed PLY lands in the same frame
  as the recorded camera poses.

Tuning knobs (module constants, raise for speed, lower for density):
* ``TSDF_VOXEL_M`` — 2.5 mm default. 2.0 denser/slower; 3.0 faster.
* ``ICP_SRC_STRIDE`` — 4 default. Raise to cut per-frame ICP cost.
* ``WITH_COLOR`` — True by default. False = geometry-only, faster.
* ``MODEL_REFRESH_EVERY`` — 5 default.

API:
    f = OnlineFusion(fx, fy, cx, cy, W, H)
    refined_cv_c2w = f.add_frame(depth_u16_gripper_zeroed, c2w_opengl, rgb_u8=None)
    cloud_o3d = f.finalize()                 # open3d.geometry.PointCloud
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------------
# Tunables (mirror experiments/icp_fusion_mvp/online_fusion.py exactly).
# ----------------------------------------------------------------------------
TSDF_VOXEL_M = 0.002         # 2 mm — coarser, fewer seed points, lower VRAM
TSDF_TRUNC_M = 0.008         # ~4× voxel
DEPTH_SCALE = 1000.0         # uint16 mm → m
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0

ICP_SRC_STRIDE = 4           # decimate depth for ICP source (TSDF still full)
ICP_VOXEL_M = 0.01           # ICP source + model voxel
MODEL_REFRESH_EVERY = 5      # re-voxel/re-normal the global model every N frames
NORMAL_RADIUS_M = 0.03
ICP_STAGES: Tuple[Tuple[float, int], ...] = ((0.05, 6), (0.02, 12))  # coarse→fine
ICP_FITNESS_MIN = 0.30       # below this: trust the FK pose
WITH_COLOR = True            # fuse real RGB (False = geometry-only, faster)

# ---------------------------------------------------------------------------
# Adaptive near/far downsample (applied to every finalized seed cloud).
# Strategy: keep points within NEAR_RADIUS_M of the LAST camera pose at native
# TSDF density; voxel-downsample the rest to FAR_VOXEL_M.
# Validated 2026-06-01 on validate_run_1: 10.15M → 1.14M (8.9× reduction).
# ---------------------------------------------------------------------------
NEAR_RADIUS_M = 1.0
FAR_VOXEL_M = 0.01


def _adaptive_downsample_gpu(
    pc: o3d.geometry.PointCloud,
    last_cam_world_xyz: np.ndarray,
    near_radius_m: float,
    far_voxel_m: float,
) -> o3d.geometry.PointCloud:
    """GPU path: convert to ``o3d.t.geometry.PointCloud`` on CUDA, split
    near/far via tensor mask, voxel-downsample the far region on GPU, then
    concatenate back to a legacy ``PointCloud`` on CPU."""
    device = o3d.core.Device("CUDA:0")
    pts_np = np.asarray(pc.points)
    has_color = pc.has_colors()
    has_normal = pc.has_normals()

    tpc = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(pts_np.astype(np.float32), device=device)
    )
    if has_color:
        tpc.point["colors"] = o3d.core.Tensor(
            np.asarray(pc.colors).astype(np.float32), device=device
        )
    if has_normal:
        tpc.point["normals"] = o3d.core.Tensor(
            np.asarray(pc.normals).astype(np.float32), device=device
        )

    cam_t = o3d.core.Tensor(
        np.asarray(last_cam_world_xyz, dtype=np.float32).reshape(1, 3),
        device=device,
    )
    diff = tpc.point["positions"] - cam_t
    sq = diff * diff
    # Manual elementwise sum: Open3D 0.19's CUDA reduction kernel rejects
    # ``sum(dim=1)`` on large Float32 tensors ("Unsupported data type"),
    # but the 3-term add lowers to plain elementwise ops which are safe.
    dist2 = sq[:, 0] + sq[:, 1] + sq[:, 2]
    # M4: 10 M-pt clouds keep ~600 MB of CUDA tensors alive (diff, sq, dist2).
    # Drop diff + sq as soon as dist2 is computed so they don't co-exist with
    # near_t / far_t / merged below.
    del diff, sq
    near_mask = dist2 <= float(near_radius_m * near_radius_m)
    far_mask = near_mask.logical_not()
    n_far = int(far_mask.to(o3d.core.Dtype.Int64).sum(0).item())
    if n_far == 0:
        del dist2, near_mask, far_mask
        return pc

    near_t = tpc.select_by_mask(near_mask)
    far_t = tpc.select_by_mask(far_mask).voxel_down_sample(float(far_voxel_m))
    # M4: tpc was the full-res GPU copy of the input cloud (~360 MB at 10 M pts);
    # we now have near_t + far_t which together carry the same data. Drop tpc
    # and the masks/dist2 so peak VRAM doesn't include both.
    del tpc, dist2, near_mask, far_mask
    merged = near_t.append(far_t)
    del near_t, far_t

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(
        merged.point["positions"].cpu().numpy().astype(np.float64)
    )
    if "colors" in merged.point:
        out.colors = o3d.utility.Vector3dVector(
            merged.point["colors"].cpu().numpy().astype(np.float64)
        )
    if "normals" in merged.point:
        out.normals = o3d.utility.Vector3dVector(
            merged.point["normals"].cpu().numpy().astype(np.float64)
        )
    return out


def _adaptive_downsample_cpu(
    pc: o3d.geometry.PointCloud,
    last_cam_world_xyz: np.ndarray,
    near_radius_m: float,
    far_voxel_m: float,
) -> o3d.geometry.PointCloud:
    """CPU fallback path: legacy Open3D ``select_by_index`` + ``voxel_down_sample``."""
    pts = np.asarray(pc.points)
    cam = np.asarray(last_cam_world_xyz, dtype=np.float64).reshape(3)
    depth = np.linalg.norm(pts - cam, axis=1)
    near_mask = depth <= near_radius_m
    near_idx = np.where(near_mask)[0]
    far_idx = np.where(~near_mask)[0]
    if far_idx.size == 0:
        return pc
    near_pc = pc.select_by_index(near_idx)
    far_pc = pc.select_by_index(far_idx)
    far_down = far_pc.voxel_down_sample(far_voxel_m)
    return near_pc + far_down


def adaptive_downsample(
    pc: o3d.geometry.PointCloud,
    last_cam_world_xyz: np.ndarray,
    near_radius_m: float = NEAR_RADIUS_M,
    far_voxel_m: float = FAR_VOXEL_M,
) -> o3d.geometry.PointCloud:
    """Near/far adaptive downsample of a TSDF-fused cloud.

    Points within ``near_radius_m`` of the last camera position are kept at
    native density; the rest are voxel-downsampled to ``far_voxel_m``.

    Uses the Open3D tensor pipeline on CUDA when available (~10× faster on
    the 13.77 M-point validate_run_1 cloud); falls back to legacy CPU.

    Args:
        pc: the TSDF-fused cloud (output of ``OnlineFusion.finalize()``).
        last_cam_world_xyz: (3,) world-frame position of the LAST captured
            camera (the operator's final viewpoint). NOT the first.
        near_radius_m: keep-as-is radius around that camera (default 1.0 m).
        far_voxel_m: voxel size for the far region (default 5 mm).

    Returns:
        A new ``PointCloud`` (near-full ⊕ far-downsampled). Colors + normals
        preserved when present on the input.
    """
    pts = np.asarray(pc.points)
    if pts.size == 0:
        return pc
    if o3d.core.cuda.is_available():
        try:
            return _adaptive_downsample_gpu(pc, last_cam_world_xyz, near_radius_m, far_voxel_m)
        except Exception as exc:
            print(f"[adaptive_downsample] GPU path failed ({exc}); falling back to CPU", flush=True)
    return _adaptive_downsample_cpu(pc, last_cam_world_xyz, near_radius_m, far_voxel_m)


# ----------------------------------------------------------------------------
# CPU implementation — legacy Open3D (kept as fallback when CUDA unavailable).
# ----------------------------------------------------------------------------


class _CpuOnlineFusion:
    """CPU implementation: legacy ``ScalableTSDFVolume`` + legacy
    ``registration_icp``. ~630 ms/frame at 800×800 on validate_run_1."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, W: int, H: int):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.W, self.H = int(W), int(H)
        self.intr_o3d = o3d.camera.PinholeCameraIntrinsic(self.W, self.H, self.fx, self.fy, self.cx, self.cy)
        ctype = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if WITH_COLOR
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(TSDF_VOXEL_M, TSDF_TRUNC_M, ctype)
        self.estim = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        self.model: Optional[o3d.geometry.PointCloud] = None
        self._pend: List[o3d.geometry.PointCloud] = []
        self.idx = 0
        # A dummy 0-image is needed when WITH_COLOR=False; create_from_color_and_depth
        # requires a color arg even if the integrator ignores it.
        self._dummy = o3d.geometry.Image(np.zeros((self.H, self.W, 3), np.uint8))

    def _src_cloud(self, depth_u16: np.ndarray, c2w_cv: np.ndarray) -> o3d.geometry.PointCloud:
        v = (depth_u16 > DEPTH_MIN_M * DEPTH_SCALE) & (depth_u16 < DEPTH_MAX_M * DEPTH_SCALE)
        vv, uu = np.where(v)
        vv, uu = vv[::ICP_SRC_STRIDE], uu[::ICP_SRC_STRIDE]
        zz = depth_u16[vv, uu] / DEPTH_SCALE
        x = (uu - self.cx) * zz / self.fx
        y = (vv - self.cy) * zz / self.fy
        cam = np.stack([x, y, zz, np.ones_like(zz)], axis=1)
        world = (c2w_cv @ cam.T).T[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(world)
        pc = pc.voxel_down_sample(ICP_VOXEL_M)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
        return pc

    def _integrate(self, depth_u16: np.ndarray, rgb_u8: Optional[np.ndarray], c2w_cv: np.ndarray) -> None:
        if WITH_COLOR and rgb_u8 is not None:
            color = o3d.geometry.Image(np.ascontiguousarray(rgb_u8))
        else:
            color = self._dummy
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            o3d.geometry.Image(np.ascontiguousarray(depth_u16)),
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_MAX_M,
            convert_rgb_to_intensity=False,
        )
        self.vol.integrate(rgbd, self.intr_o3d, np.linalg.inv(c2w_cv))

    def add_frame(
        self,
        depth_u16: np.ndarray,
        c2w_cv: np.ndarray,
        rgb_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        src = self._src_cloud(depth_u16, c2w_cv)
        if self.model is None:
            self.model = src
            self._integrate(depth_u16, rgb_u8, c2w_cv)
            self.idx += 1
            return c2w_cv
        T = np.eye(4)
        reg = None
        for dist, iters in ICP_STAGES:
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            reg = o3d.pipelines.registration.registration_icp(src, self.model, dist, T, self.estim, crit)
            T = reg.transformation
        assert reg is not None
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

    def finalize(self) -> o3d.geometry.PointCloud:
        return self.vol.extract_point_cloud()


# ----------------------------------------------------------------------------
# GPU implementation — Open3D 0.19 tensor pipeline on CUDA. ~16 ms/frame
# at 800×800 on validate_run_1 (sm_120). Proven by ``scripts/bench_gpu_fusion.py``.
# ----------------------------------------------------------------------------


class _GpuOnlineFusion:
    """GPU implementation: ``o3d.t.pipelines.slam.Model`` (VoxelBlockGrid TSDF)
    + ``o3d.t.pipelines.registration.multi_scale_icp``. Mirrors the CPU
    semantics frame-for-frame (same dedup, same ICP stages, same fitness gate)."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, W: int, H: int):
        import open3d.core as o3c  # local alias to keep CPU path import-safe

        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.W, self.H = int(W), int(H)
        self._dev = o3c.Device("CUDA:0")
        self._intrinsic_t = o3c.Tensor(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=o3c.Dtype.Float64,
        )
        # SLAM model: TSDF VoxelBlockGrid on CUDA.
        # 1.5 mm voxel, 16³ block. Initial 8k blocks → grows via Open3D's
        # internal hashmap rehash. Starting low avoids pre-allocating the
        # ~700 MB block buffer that triggers OOM on multi-stream GPUs.
        self._slam = o3d.t.pipelines.slam.Model(
            TSDF_VOXEL_M,
            16,
            8000,
            o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
            self._dev,
        )
        # ICP stages mirror CPU. Coarse-to-fine voxel sizes match ICP_STAGES.
        self._voxel_sizes = o3d.utility.DoubleVector([ICP_VOXEL_M * 2, ICP_VOXEL_M])
        self._criteria = [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_STAGES[0][1]),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_STAGES[1][1]),
        ]
        self._max_corr_dists = o3d.utility.DoubleVector([ICP_STAGES[0][0], ICP_STAGES[1][0]])
        self._estim = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        self._model_pcd: Optional[o3d.t.geometry.PointCloud] = None
        self._pend: List[o3d.t.geometry.PointCloud] = []
        self.idx = 0

    def _sync(self) -> None:
        import open3d.core as o3c
        o3c.cuda.synchronize()

    def _src_cloud(self, depth_u16: np.ndarray, c2w_cv: np.ndarray) -> o3d.t.geometry.PointCloud:
        import open3d.core as o3c
        depth_t = o3c.Tensor(depth_u16.astype(np.uint16), device=self._dev)
        depth_img = o3d.t.geometry.Image(depth_t)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth=depth_img,
            intrinsics=self._intrinsic_t,
            extrinsics=o3c.Tensor(np.linalg.inv(c2w_cv).astype(np.float64), o3c.Dtype.Float64),
            depth_scale=DEPTH_SCALE,
            depth_max=DEPTH_MAX_M,
            stride=ICP_SRC_STRIDE,
        )
        pcd = pcd.voxel_down_sample(ICP_VOXEL_M)
        pcd.estimate_normals(max_nn=30, radius=NORMAL_RADIUS_M)
        return pcd

    def _integrate(
        self,
        depth_u16: np.ndarray,
        rgb_u8: Optional[np.ndarray],
        c2w_cv: np.ndarray,
    ) -> None:
        import open3d.core as o3c
        depth_t = o3c.Tensor(depth_u16.astype(np.uint16), device=self._dev)
        depth_img = o3d.t.geometry.Image(depth_t)
        if WITH_COLOR and rgb_u8 is not None:
            rgb_t = o3c.Tensor(np.ascontiguousarray(rgb_u8), device=self._dev)
            rgb_img = o3d.t.geometry.Image(rgb_t)
        else:
            # Zero RGB image keeps the VoxelBlockGrid happy when WITH_COLOR=False.
            rgb_t = o3c.Tensor(np.zeros((self.H, self.W, 3), np.uint8), device=self._dev)
            rgb_img = o3d.t.geometry.Image(rgb_t)
        extr_t = o3c.Tensor(np.linalg.inv(c2w_cv), o3c.Dtype.Float64)
        frustum = self._slam.voxel_grid.compute_unique_block_coordinates(
            depth_img, self._intrinsic_t, extr_t, DEPTH_SCALE, DEPTH_MAX_M,
        )
        self._slam.voxel_grid.integrate(
            frustum, depth_img, rgb_img,
            self._intrinsic_t, self._intrinsic_t,
            extr_t, DEPTH_SCALE, DEPTH_MAX_M,
        )

    def add_frame(
        self,
        depth_u16: np.ndarray,
        c2w_cv: np.ndarray,
        rgb_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        import open3d.core as o3c
        src = self._src_cloud(depth_u16, c2w_cv)
        if self._model_pcd is None:
            self._model_pcd = src.clone()
            self._integrate(depth_u16, rgb_u8, c2w_cv)
            self.idx += 1
            return c2w_cv
        reg = o3d.t.pipelines.registration.multi_scale_icp(
            source=src,
            target=self._model_pcd,
            voxel_sizes=self._voxel_sizes,
            criteria_list=self._criteria,
            max_correspondence_distances=self._max_corr_dists,
            init_source_to_target=o3c.Tensor(np.eye(4), o3c.Dtype.Float64),
            estimation_method=self._estim,
        )
        T = reg.transformation.cpu().numpy()
        fitness = float(reg.fitness)
        refined = T @ c2w_cv if fitness >= ICP_FITNESS_MIN else c2w_cv
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

    def finalize(self) -> o3d.geometry.PointCloud:
        """Extract the TSDF and convert tensor → legacy ``PointCloud`` so
        downstream (``adaptive_downsample``, ``write_point_cloud``) is
        unchanged.

        M4: free the VoxelBlockGrid + ICP-model-cloud + pending buffers right
        after extraction so adaptive_downsample's peak VRAM doesn't have to
        co-exist with the steady-state ~2.5 GB VBG. Net savings on a 10 M-pt
        validate_run_1 finalize: ~1 GB observed.
        """
        tpc = self._slam.voxel_grid.extract_point_cloud()
        legacy = tpc.to_legacy()
        # Drop the GPU-resident tensor copy of the extracted cloud.
        del tpc
        # Drop the SLAM model (which owns the VoxelBlockGrid) + the ICP
        # model cloud + pending merge buffers. After finalize the caller
        # never calls add_frame() again, so these are pure dead weight.
        self._slam = None  # type: ignore[assignment]
        self._model_pcd = None
        self._pend = []
        import gc as _gc
        _gc.collect()
        try:
            import open3d.core as _o3c
            if _o3c.cuda.is_available():
                _o3c.cuda.release_cache()
        except Exception:
            pass
        return legacy


# ----------------------------------------------------------------------------
# Public dispatcher.
# ----------------------------------------------------------------------------

class OnlineFusion:
    """Stream-friendly RGB-D fusion. Call ``add_frame`` per arriving frame,
    ``finalize`` once at the end.

    Auto-selects GPU (Open3D tensor SLAM pipeline) when CUDA is available;
    falls back to CPU (legacy Open3D) otherwise. Set ``DGS_FUSION_DEVICE=cpu``
    to force the CPU path (useful for debugging or sm_*-incompatible cards)."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, W: int, H: int):
        force = os.environ.get("DGS_FUSION_DEVICE", "auto").lower()
        use_gpu = force == "gpu" or (force == "auto" and o3d.core.cuda.is_available())
        if use_gpu:
            try:
                self._impl: object = _GpuOnlineFusion(fx, fy, cx, cy, W, H)
                self.device = "gpu"
            except Exception as exc:
                if force == "gpu":
                    raise
                print(f"[OnlineFusion] GPU init failed ({exc}); falling back to CPU", flush=True)
                self._impl = _CpuOnlineFusion(fx, fy, cx, cy, W, H)
                self.device = "cpu"
        else:
            self._impl = _CpuOnlineFusion(fx, fy, cx, cy, W, H)
            self.device = "cpu"

    @staticmethod
    def _cv_c2w(c2w_opengl: np.ndarray) -> np.ndarray:
        """OpenGL (+y up, +z back) → OpenCV (+y down, +z forward)."""
        return np.asarray(c2w_opengl, dtype=np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])

    @property
    def idx(self) -> int:
        return self._impl.idx  # type: ignore[attr-defined]

    def add_frame(
        self,
        depth_u16: np.ndarray,
        c2w_opengl: np.ndarray,
        rgb_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one streamed frame.

        Args:
            depth_u16: (H, W) uint16, mm; gripper already zeroed by the
                caller (mask==0 → depth==0).
            c2w_opengl: (4, 4) OpenGL c2w from FK / transforms.json.
            rgb_u8: (H, W, 3) uint8 RGB (NOT BGR). Optional; ignored
                when ``WITH_COLOR=False``.

        Returns:
            (4, 4) OpenCV c2w after ICP refinement (or the input FK pose
            unchanged when ICP fitness < ``ICP_FITNESS_MIN``).
        """
        c2w_cv = self._cv_c2w(c2w_opengl)
        return self._impl.add_frame(depth_u16, c2w_cv, rgb_u8)  # type: ignore[attr-defined]

    def finalize(self) -> o3d.geometry.PointCloud:
        """Extract the fused cloud as a legacy ``PointCloud``. Safe to call
        once after all ``add_frame`` calls; do not ``add_frame`` after."""
        return self._impl.finalize()  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# Convenience: drive the fusion offline over an existing dataset.
# ----------------------------------------------------------------------------

def fuse_recorded_dataset(static_dir: Path) -> Path:
    """Run ``OnlineFusion`` over every frame in
    ``<static_dir>/transforms.json`` and write
    ``<static_dir>/depth_camera_init_points.ply``.

    Returns the written PLY path. Updates the transforms.json's
    ``ply_file_path`` entry so the dataparser picks it up unchanged.

    Use this when the dataset was captured via the publisher's
    ``start_recording`` (`live_session.py`, `bootstrap_live.sh`) — the
    publisher doesn't expose a per-frame hook for online fusion, so
    we run the same algorithm as a one-shot post-capture pass.
    """
    import json
    import re

    import cv2

    static_dir = Path(static_dir)
    meta_path = static_dir / "transforms.json"
    meta = json.loads(meta_path.read_text())
    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    W, H = int(meta["w"]), int(meta["h"])
    # Sort frames by numerical suffix so we process them in capture order.
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))

    fuser = OnlineFusion(fx, fy, cx, cy, W, H)
    for fr in frames:
        depth_path = static_dir / fr["depth_file_path"].lstrip("./")
        mask_path = static_dir / fr["mask_path"].lstrip("./") if fr.get("mask_path") else None
        rgb_path = static_dir / fr["file_path"].lstrip("./")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            depth[m == 0] = 0
        rgb = None
        if WITH_COLOR:
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        fuser.add_frame(depth, np.asarray(fr["transform_matrix"], dtype=np.float64), rgb)

    pc = fuser.finalize()
    n_full = len(pc.points)
    # Adaptive near/far downsample: keep <NEAR_RADIUS_M of the LAST camera pose
    # at native density, voxel-downsample the rest to FAR_VOXEL_M.
    if n_full > 0:
        last_cam = np.asarray(frames[-1]["transform_matrix"], dtype=np.float64)[:3, 3]
        pc = adaptive_downsample(pc, last_cam)
        print(
            f"[fuse_recorded] adaptive downsample: {n_full:,} → {len(pc.points):,} "
            f"(near<{NEAR_RADIUS_M:.1f}m full, far→{FAR_VOXEL_M*1000:.1f}mm voxel)",
            flush=True,
        )
    ply_path = static_dir / "depth_camera_init_points.ply"
    o3d.io.write_point_cloud(str(ply_path), pc)

    meta["ply_file_path"] = "depth_camera_init_points.ply"
    tmp = meta_path.with_name(f".{meta_path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    import os as _os
    _os.replace(tmp, meta_path)
    return ply_path
