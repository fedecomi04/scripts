"""FoundationPose-based 6D object tracker.

Replaces the previous CoTracker3 + Kabsch-RANSAC pipeline. The FP estimator
takes RGB-D + a triangle mesh and returns the mesh-to-camera 4x4 pose.
This tracker wraps that, keeps it stateful across frames, and exposes a
public API that returns the **world-frame absolute** rigid transform
``(R, t)`` from the captured D0 reference pose to the current pose, in the
exact convention that
``DynamicGSModel.apply_rigid_object_transform_from_reference`` expects.

The tracker skips ``est.register`` on frame 0: the initial mesh-to-world
transform comes from the SAM3D fusion result (persisted in
``phase0_manifest.json``), so we already know where the object is. We
push that pose directly into ``est.pose_last`` (in FP's centered-mesh
frame) and call ``track_one`` for refinement. A fallback ``register``
path is provided for legacy datasets without ``mesh_to_world_4x4``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

# Make the third_party FoundationPose package importable. The repo lives at
# scripts/third_party/FoundationPose; estimater.py lives at its root and
# uses unqualified imports (``from Utils import *``), so its directory must
# be first on sys.path.
_FP_REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "FoundationPose"


def _ensure_fp_on_path() -> None:
    repo = str(_FP_REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    learning = str(_FP_REPO_ROOT / "learning")
    if learning not in sys.path:
        sys.path.insert(0, learning)


class FoundationPoseTracker:
    """Per-object 6D tracker. One instance tracks one mesh."""

    def __init__(
        self,
        mesh_path: str | Path,
        mesh_to_world: np.ndarray,
        mesh_unit_scale: float = 1.0,
        debug_dir: Optional[str | Path] = None,
    ) -> None:
        _ensure_fp_on_path()
        # Imported lazily so importing this module costs nothing when FP isn't used.
        import nvdiffrast.torch as dr  # type: ignore
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor  # type: ignore

        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"FoundationPose mesh not found: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
            raise ValueError(
                f"FoundationPose requires a triangle mesh; got {type(mesh).__name__} from {mesh_path}"
                " (no faces). SAM3D's gaussian-splat .ply will not work — use the mesh decoder output."
            )

        if mesh_unit_scale != 1.0:
            mesh.apply_scale(float(mesh_unit_scale))

        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        extents = bbox_max - bbox_min
        logging.info(
            f"[FP] mesh={mesh_path.name} verts={len(mesh.vertices)} faces={len(mesh.faces)} "
            f"bbox_extents_m=({extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f})"
        )

        mesh_to_world = np.asarray(mesh_to_world, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(mesh_to_world).all():
            raise ValueError("mesh_to_world must be a finite 4x4 matrix")

        debug_dir_path = Path(debug_dir) if debug_dir is not None else (_FP_REPO_ROOT / "debug")
        debug_dir_path.mkdir(parents=True, exist_ok=True)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        self._est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=self._glctx,
            debug=0,
            debug_dir=str(debug_dir_path),
        )
        self._mesh = mesh
        self._mesh_to_world_init: np.ndarray = mesh_to_world.astype(np.float64)
        self._mesh_to_world_init_inv: np.ndarray = np.linalg.inv(self._mesh_to_world_init)
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # FP-frame conversions
    # ------------------------------------------------------------------
    def _tf_to_centered_np(self) -> np.ndarray:
        """4x4 numpy view of FP's ``get_tf_to_centered_mesh``."""
        T = self._est.get_tf_to_centered_mesh()
        return T.detach().cpu().numpy().astype(np.float64).reshape(4, 4)

    def _world_to_centered_pose(self, mesh_to_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
        """Convert mesh-to-world (orig mesh frame) into FP's pose_last (centered-mesh-to-camera)."""
        mesh_to_camera = np.linalg.inv(camera_to_world) @ mesh_to_world
        return mesh_to_camera @ np.linalg.inv(self._tf_to_centered_np())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize_from_known_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        camera_to_world: np.ndarray,
        refine_iterations: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Skip ``register``; seed pose_last from the known scene pose.

        With ``refine_iterations=0`` the seed is trusted as-is and the returned
        ``(R, t)`` is identity in world frame. With ``refine_iterations>0`` the
        FP refiner runs ``track_one`` on the D0 observation; this is risky here
        because the Poisson-reconstructed mesh has a slightly different bbox
        centroid than the SAM3D Gaussian centers used to compute
        ``mesh_to_world_4x4``, so the refiner can converge far from the seed.
        Default is 0 — refinement happens normally on frame 1 onwards.

        Returns (R, t) in world frame, absolute from the D0 reference pose.
        """
        camera_to_world = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
        depth_np = np.ascontiguousarray(depth, dtype=np.float32)

        pose_centered = self._world_to_centered_pose(self._mesh_to_world_init, camera_to_world)
        self._est.pose_last = torch.as_tensor(
            pose_centered, dtype=torch.float, device="cuda"
        )
        logging.info("[FP] initialized from known scene pose (skipping register)")

        if int(refine_iterations) <= 0:
            self._initialized = True
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t

        pose_in_camera = self._est.track_one(
            rgb=rgb, depth=depth_np, K=K_np, iteration=int(refine_iterations)
        )
        self._initialized = True
        return self._delta_world_from_pose_in_camera(pose_in_camera, camera_to_world)

    def fallback_register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        ob_mask: np.ndarray,
        camera_to_world: np.ndarray,
        refine_iterations: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard FP register path (requires a 2D mask). Use only when no known pose is available."""
        camera_to_world = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
        depth_np = np.ascontiguousarray(depth, dtype=np.float32)
        mask_bool = np.asarray(ob_mask).astype(bool)

        logging.warning("[FP] WARNING: fell back to register() with mask")
        pose_in_camera = self._est.register(
            K=K_np, rgb=rgb, depth=depth_np, ob_mask=mask_bool, iteration=int(refine_iterations)
        )
        # FP's register set self.pose_last to the centered-mesh pose; reproduce
        # what pose_in_camera implies to keep _mesh_to_world_init consistent
        # with what the model expects (the register pose becomes our D0 reference).
        self._mesh_to_world_init = camera_to_world @ np.asarray(pose_in_camera, dtype=np.float64).reshape(4, 4)
        self._mesh_to_world_init_inv = np.linalg.inv(self._mesh_to_world_init)
        self._initialized = True
        return self._delta_world_from_pose_in_camera(pose_in_camera, camera_to_world)

    def track_one(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        camera_to_world: np.ndarray,
        iterations: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame tracking. Returns (R, t) in world frame, absolute from D0 reference pose."""
        if not self._initialized:
            raise RuntimeError("FoundationPoseTracker.track_one called before initialize_from_known_pose")
        camera_to_world = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
        depth_np = np.ascontiguousarray(depth, dtype=np.float32)

        pose_in_camera = self._est.track_one(
            rgb=rgb, depth=depth_np, K=K_np, iteration=int(iterations)
        )
        return self._delta_world_from_pose_in_camera(pose_in_camera, camera_to_world)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _delta_world_from_pose_in_camera(
        self, pose_in_camera: np.ndarray, camera_to_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        T_cam_orig = np.asarray(pose_in_camera, dtype=np.float64).reshape(4, 4)
        T_world_orig_now = camera_to_world @ T_cam_orig
        delta_world = T_world_orig_now @ self._mesh_to_world_init_inv
        R = delta_world[:3, :3].astype(np.float32)
        t = delta_world[:3, 3].astype(np.float32)
        return R, t
