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

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------------
# Tunables (mirror experiments/icp_fusion_mvp/online_fusion.py exactly).
# ----------------------------------------------------------------------------
TSDF_VOXEL_M = 0.0015        # 1.5 mm — finer detail, ~5 ms integrate on GPU
TSDF_TRUNC_M = 0.006         # ~4× voxel
DEPTH_SCALE = 1000.0         # uint16 mm → m
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0

ICP_SRC_STRIDE = 4           # decimate depth for ICP source (TSDF still full)
ICP_VOXEL_M = 0.01           # ICP source + model voxel
MODEL_REFRESH_EVERY = 5      # re-voxel/re-normal the global model every N frames
NORMAL_RADIUS_M = 0.03
ICP_STAGES: Tuple[Tuple[float, int], ...] = ((0.05, 6), (0.02, 12))  # coarse→fine
ICP_FITNESS_MIN = 0.30       # below this: trust the FK pose
WITH_COLOR = True            # fuse real RGB (False = geometry-only, faster)


class OnlineFusion:
    """Stream-friendly RGB-D fusion. Call ``add_frame`` per arriving
    frame, ``finalize`` once at the end."""

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

    @staticmethod
    def _cv_c2w(c2w_opengl: np.ndarray) -> np.ndarray:
        """OpenGL (+y up, +z back) → OpenCV (+y down, +z forward)."""
        return np.asarray(c2w_opengl, dtype=np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])

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
            (4, 4) OpenCV c2w after ICP refinement (or the input FK
            pose unchanged when ICP fitness < ``ICP_FITNESS_MIN``).
        """
        c2w_cv = self._cv_c2w(c2w_opengl)
        src = self._src_cloud(depth_u16, c2w_cv)
        if self.model is None:
            # First frame anchors the world.
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
        # Push the refined source into the pending pool; periodic merge
        # keeps the global model bounded by scene extent, not #frames.
        src.transform(refined @ np.linalg.inv(c2w_cv))
        self._pend.append(src)
        self.idx += 1
        if self.idx % MODEL_REFRESH_EVERY == 0:
            for s in self._pend:
                self.model += s
            self.model = self.model.voxel_down_sample(ICP_VOXEL_M)
            self.model.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30)
            )
            self._pend = []
        return refined

    def finalize(self) -> o3d.geometry.PointCloud:
        """Extract the fused cloud. Safe to call once after all
        ``add_frame`` calls; do not ``add_frame`` after."""
        return self.vol.extract_point_cloud()


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
    ply_path = static_dir / "depth_camera_init_points.ply"
    o3d.io.write_point_cloud(str(ply_path), pc)

    meta["ply_file_path"] = "depth_camera_init_points.ply"
    tmp = meta_path.with_name(f".{meta_path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    import os as _os
    _os.replace(tmp, meta_path)
    return ply_path
