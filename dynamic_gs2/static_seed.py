"""static_seed.py — GPU TSDF seed build (ICP + integrate), wrapping the proven OnlineFusion.

The seed PLY (static_scene/depth_camera_init_points.ply) is what splatfacto inits from.
WRAPS dynamic_gs2.online_fusion.OnlineFusion (GPU VoxelBlockGrid + multi-scale ICP);
no fusion math is reimplemented. Consumes the SAME Frame stream as the tracker (depth_m
metres -> uint16 mm + OpenGL c2w, gripper pre-zeroed via mask_keep).

Per static_phase.md §2a: add_frame does ICP-THEN-integrate atomically (~22 ms/frame @1200p,
~8.1 GB resident) and is NOT separable, and the whole seed is only ~1.5 s — so the canonical
policy is to feed the recorded sweep frames AFTER SAM3D unloads (GPU then free), not during.
The orchestrator times `after.tsdf_integrate` around build()+finalize().
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .frame import Frame, Intrinsics


class StaticSeedBuilder:
    """Accumulate swept frames into a TSDF, then extract + write the seed PLY. One builder
    per static run. The GPU VoxelBlockGrid is held only between build() and finalize()."""

    def __init__(self, intr: Intrinsics, *, depth_max_m: float = 2.0,
                 voxel_m: float = 0.002, device: str = "auto"):
        import os
        # OnlineFusion reads these from the env (DGS_*), so set them for this build only.
        os.environ["DGS_TSDF_DEPTH_MAX_M"] = str(float(depth_max_m))
        os.environ["DGS_TSDF_VOXEL_M"] = str(float(voxel_m))
        if device != "auto":
            os.environ["DGS_FUSION_DEVICE"] = device
        from .online_fusion import OnlineFusion
        self._fusion = OnlineFusion(intr.fx, intr.fy, intr.cx, intr.cy,
                                    int(intr.width), int(intr.height))
        self._n = 0
        self._last_cam_xyz: Optional[np.ndarray] = None

    def add_frame(self, frame: Frame) -> None:
        """ICP-refine + integrate ONE swept frame. depth_m (metres) -> uint16 mm with the
        robot zeroed by mask_keep (so the gripper never fuses into the seed)."""
        depth_mm = np.clip(frame.depth_m * 1000.0, 0, 65535).astype(np.uint16)
        keep = frame.mask_keep
        keep = keep[..., 0] if keep.ndim == 3 else keep
        depth_mm[keep == 0] = 0
        c2w = np.asarray(frame.c2w_4x4, dtype=np.float64)
        self._fusion.add_frame(depth_mm, c2w, frame.rgb_bgr)
        self._last_cam_xyz = c2w[:3, 3].copy()           # last viewpoint -> adaptive downsample anchor
        self._n += 1

    def add_frames(self, frames: List[Frame]) -> int:
        for f in frames:
            self.add_frame(f)
        return self._n

    @property
    def num_frames(self) -> int:
        return self._n

    def finalize(self, data_dir) -> Path:
        """Extract the fused cloud, adaptive-downsample (near-full + far-voxel), write the
        seed PLY, and return its path. Frees the GPU grid."""
        from .online_fusion import adaptive_downsample
        import open3d as o3d
        pc = self._fusion.finalize()
        if self._last_cam_xyz is not None and len(pc.points) > 0:
            pc = adaptive_downsample(pc, self._last_cam_xyz)
        out = Path(data_dir) / "static_scene" / "depth_camera_init_points.ply"
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out), pc)
        self._seed_points = int(len(pc.points))
        return out

    @property
    def seed_points(self) -> int:
        return int(getattr(self, "_seed_points", 0))
