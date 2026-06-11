"""Shared tracker primitives.

Extracted from the legacy ``cotracker_motion`` module so that the active
tracker (``xfeat_motion``) can depend on a neutral module instead of a
sibling tracker implementation. Legacy tracker files now live under
``dynamic_gs/utils/_legacy_trackers/`` and are not loaded by the
runtime; only this module survives in the import graph.

Contents:
  * ``MotionEstimate`` — the result dataclass returned by every tracker
    backend (formerly ``CoTrackerMotionEstimate``). Kept verbatim — the
    pipeline reads many of these fields by name.
  * Static image / camera / depth / mask helpers (``prepare_depth_image``,
    ``extract_intrinsics``, ``extract_camera_to_world``,
    ``prepare_tracking_rgb``, ``prepare_tracking_rgb_gpu``, ``resize_mask``,
    ``sample_mask_points``, ``filter_points_in_image``,
    ``filter_points_by_mask_array``, ``sample_depth_bilinear``,
    ``backproject_to_world``).
  * Kabsch + RANSAC (``estimate_rigid_transform``,
    ``estimate_rigid_transform_ransac``).

The pipeline still imports ``CoTrackerMotionEstimate`` from a couple of
debug-viz call sites by name; the alias at the bottom of this file
keeps those working without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nerfstudio.cameras.cameras import Cameras


@dataclass
class MotionEstimate:
    """Result dataclass returned by every tracker backend's
    ``estimate_and_advance``.

    Field names are referenced verbatim by ``dynamic_gs_pipeline``
    (``motion_estimate.inlier_count`` etc.) — do not rename without
    matching the pipeline call sites.
    """

    success: bool
    ready: bool
    rotation: np.ndarray
    translation: np.ndarray
    correspondence_count: int
    inlier_count: int
    track_count_before: int
    track_count_after: int
    raw_visible_count: int
    mask_visible_count: int
    depth_valid_count: int
    used_mask_fallback: bool
    mean_residual: float
    median_residual: float
    # Debug: pixel coordinates of tracked points (None if not ready)
    previous_points_xy: Optional[np.ndarray] = None
    current_points_xy: Optional[np.ndarray] = None
    tracked_inlier_mask: Optional[np.ndarray] = None
    previous_rgb: Optional[Tensor] = None
    current_rgb: Optional[Tensor] = None
    previous_mask: Optional[object] = None
    current_mask: Optional[object] = None
    # Per-call sub-step timings in seconds (see pipeline for keys read).
    timings: dict = field(default_factory=dict)


# Back-compat alias. The legacy name is referenced in a few places that
# still import ``CoTrackerMotionEstimate`` directly.
CoTrackerMotionEstimate = MotionEstimate


def prepare_tracking_rgb(image: Tensor) -> Tensor:
    """CPU-resident [0, 255] float HWC. Kept for compatibility with the
    init paths that don't have a GPU device handy. Per-tick paths should
    prefer ``prepare_tracking_rgb_gpu`` to avoid a forced .cpu() sync."""
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image tensor, got shape {tuple(image.shape)}")
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = image.detach().float().cpu()
    if image.max().item() <= 1.0 + 1e-6:
        image = image * 255.0
    return image.clamp(0.0, 255.0)


def prepare_tracking_rgb_gpu(image: Tensor, device: torch.device) -> Tensor:
    """GPU-resident HWC float in whatever range the caller passed.

    Pipeline contract: input is ``get_live_rgb`` output (float on GPU,
    ``[0, 1]``, HWC). No host sync. Each tracker's preprocess
    re-normalises to whatever the model expects.
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(
            f"Expected HxWxC image tensor for tracking, got shape {tuple(image.shape)}"
        )
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.detach().to(device, dtype=torch.float32, non_blocking=True)


def prepare_depth_image(depth: Tensor) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth.detach().float().cpu().numpy().astype(np.float32)


def extract_intrinsics(camera: "Cameras") -> np.ndarray:
    return camera.get_intrinsics_matrices()[0].detach().cpu().numpy().astype(np.float32)


def extract_camera_to_world(camera: "Cameras") -> np.ndarray:
    camera_to_world = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)
    if camera_to_world.shape == (4, 4):
        camera_to_world = camera_to_world[:3, :]
    return camera_to_world


def resize_mask(mask: Tensor, output_shape: Optional[tuple[int, int]]) -> Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if output_shape is None or tuple(mask.shape[:2]) == tuple(output_shape):
        return mask
    mask_4d = mask[None, None].float()
    resized = F.interpolate(mask_4d, size=output_shape, mode="nearest")
    return resized[0, 0]


def _subsample_points(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    if len(points_xy) <= max_points:
        return points_xy.astype(np.float32)
    keep = np.linspace(0, len(points_xy) - 1, num=max_points)
    keep = np.unique(np.round(keep).astype(np.int64))
    return points_xy[keep].astype(np.float32)


def _shrink_mask_for_sampling(mask_np: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return mask_np
    side = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    margin_px = max(1, int(round(0.025 * side)))
    kernel_size = 2 * margin_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inner_mask = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1) > 0
    return inner_mask if np.any(inner_mask) else mask_np


def sample_mask_points(
    mask: Tensor,
    max_points: int,
    rgb: Optional[Tensor] = None,
    output_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Sample up to ``max_points`` 2D pixel coords from inside ``mask``.

    If ``rgb`` is provided, use FAST keypoints (texture-aware) inside
    the mask; otherwise fall back to evenly-spaced mask pixels.
    """
    mask = resize_mask(mask, output_shape)
    mask_np = (mask.detach().float().cpu().numpy() > 0.5)
    sample_mask_np = _shrink_mask_for_sampling(mask_np)
    ys, xs = np.nonzero(sample_mask_np)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if rgb is not None:
        rgb_np = rgb.detach().float().cpu().numpy()
        if rgb_np.shape[:2] != mask_np.shape:
            rgb_np = cv2.resize(rgb_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] >= 3:
            gray = cv2.cvtColor(rgb_np[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_np[..., 0] if rgb_np.ndim == 3 else rgb_np
        gray = gray.copy()
        gray[~sample_mask_np] = 0

        detector = cv2.FastFeatureDetector_create(threshold=28, nonmaxSuppression=True)
        keypoints = detector.detect(gray, None)
        if keypoints:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
            fast_points: list[list[float]] = []
            for kp in keypoints:
                x = int(round(kp.pt[0]))
                y = int(round(kp.pt[1]))
                if x < 0 or x >= sample_mask_np.shape[1] or y < 0 or y >= sample_mask_np.shape[0]:
                    continue
                if not sample_mask_np[y, x]:
                    continue
                fast_points.append([float(x), float(y)])
            if fast_points:
                return _subsample_points(np.asarray(fast_points, dtype=np.float32), max_points)

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    return _subsample_points(coords, max_points)


def filter_points_in_image(
    points_xy: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points_xy).all(axis=1)
    valid &= points_xy[:, 0] >= 0.0
    valid &= points_xy[:, 0] <= max(width - 1, 0)
    valid &= points_xy[:, 1] >= 0.0
    valid &= points_xy[:, 1] <= max(height - 1, 0)
    return points_xy, visibility & valid


def filter_points_by_mask_array(
    points_xy: np.ndarray,
    visibility: np.ndarray,
    mask: Tensor,
    output_shape: tuple[int, int],
) -> np.ndarray:
    resized = resize_mask(mask, output_shape)
    mask_np = (resized.detach().float().cpu().numpy() > 0.5)
    if mask_np.size == 0:
        return np.zeros_like(visibility, dtype=bool)
    xs = np.clip(np.round(points_xy[:, 0]).astype(np.int64), 0, mask_np.shape[1] - 1)
    ys = np.clip(np.round(points_xy[:, 1]).astype(np.int64), 0, mask_np.shape[0] - 1)
    return visibility & mask_np[ys, xs]


def sample_depth_bilinear(depth: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    height, width = depth.shape
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= max(width - 1, 0))
        & (y >= 0.0)
        & (y <= max(height - 1, 0))
    )

    x0 = np.clip(np.floor(x).astype(np.int64), 0, max(width - 1, 0))
    y0 = np.clip(np.floor(y).astype(np.int64), 0, max(height - 1, 0))
    x1 = np.clip(x0 + 1, 0, max(width - 1, 0))
    y1 = np.clip(y0 + 1, 0, max(height - 1, 0))

    d00 = depth[y0, x0]
    d01 = depth[y0, x1]
    d10 = depth[y1, x0]
    d11 = depth[y1, x1]

    local_valid = (
        np.isfinite(d00) & np.isfinite(d01) & np.isfinite(d10) & np.isfinite(d11)
        & (d00 > 0.0) & (d01 > 0.0) & (d10 > 0.0) & (d11 > 0.0)
    )
    valid &= local_valid

    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    depth_values = (
        d00 * (1.0 - wx) * (1.0 - wy)
        + d01 * wx * (1.0 - wy)
        + d10 * (1.0 - wx) * wy
        + d11 * wx * wy
    ).astype(np.float32)
    valid &= np.isfinite(depth_values) & (depth_values > 0.0)
    depth_values[~valid] = 0.0
    return depth_values, valid


def backproject_to_world(
    points_xy: np.ndarray,
    depth_values: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Pixel + depth → world. Nerfstudio camera_to_worlds use OpenGL
    camera coords (+X right, +Y up, +Z back, looking along -Z)."""
    if len(points_xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    z = depth_values

    camera_points = np.stack(
        [
            (x - cx) * z / max(fx, 1e-8),
            -(y - cy) * z / max(fy, 1e-8),
            -z,
        ],
        axis=1,
    ).astype(np.float32)

    rotation = camera_to_world[:, :3]
    translation = camera_to_world[:, 3]
    return (camera_points @ rotation.T + translation[None, :]).astype(np.float32)


def _so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) -> rotation matrix (3, 3)."""
    R, _ = cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> rotation vector (3,)."""
    w, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return w.flatten()


class PoseKalmanFilter:
    """Constant-velocity error-state Kalman filter on SE(3).

    Smooths the per-tick (R, t) pose measurements coming out of
    RANSAC+Kabsch. The dominant noise source is per-tick match-set
    variance (different LighterGlue match subsets -> slightly different
    Kabsch solutions), which is approximately zero-mean — exactly what a
    KF removes. A constant-velocity motion model keeps lag low during
    real object motion while strongly attenuating stationary jitter.

    State (12,): [position (3), velocity (3), rotation-error (3),
    angular velocity (3)]. Rotation is kept as a nominal matrix
    ``_R_nom`` with a small-angle error state that is injected and
    zeroed after every update (standard ESKF). Angular velocity is
    expressed in the world frame and composed on the left.

    All units are metres / radians / seconds. Cost is a handful of
    12x12 numpy ops per tick (<0.05 ms).
    """

    def __init__(
        self,
        accel_sigma: float = 0.05,
        alpha_sigma: float = 0.25,
        meas_trans_sigma: float = 0.003,
        meas_rot_sigma: float = 0.009,
        snap_trans_m: float = 0.05,
        snap_rot_rad: float = 0.1745,  # 10 deg
    ) -> None:
        # Process noise: white translational acceleration (m/s^2) and
        # white angular acceleration (rad/s^2). Larger = trusts the
        # measurements more = less smoothing, less lag.
        self._accel_sigma = float(accel_sigma)
        self._alpha_sigma = float(alpha_sigma)
        # Measurement noise: 1-sigma of the RANSAC pose estimate
        # (metres, radians). Larger = smoother output, more lag.
        self._meas_trans_var = float(meas_trans_sigma) ** 2
        self._meas_rot_var = float(meas_rot_sigma) ** 2
        # Innovation gate: a per-tick pose jump larger than this cannot
        # come from continuous motion (at 20+ Hz it would mean >1 m/s /
        # >200 deg/s) — it is a reacquisition after tracking loss or an
        # anchor-pool discontinuity. Smoothing through it causes
        # overshoot (the step kicks the velocity state), so snap-reset
        # to the measurement instead.
        self._snap_trans_m = float(snap_trans_m)
        self._snap_rot_rad = float(snap_rot_rad)
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._t_nom = np.zeros(3, dtype=np.float64)
        self._R_nom = np.eye(3, dtype=np.float64)
        self._x = np.zeros(12, dtype=np.float64)   # error/velocity state
        self._P = np.eye(12, dtype=np.float64)
        self._last_time: float = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def current(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest filtered pose (R (3,3), t (3,)) as float32."""
        return (
            self._R_nom.astype(np.float32),
            self._t_nom.astype(np.float32),
        )

    def filter(
        self, R_meas: np.ndarray, t_meas: np.ndarray, timestamp: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict + update with one RANSAC pose measurement.

        Returns the filtered (R, t). The first call initializes the
        state to the measurement and returns it unchanged.
        """
        R_meas64 = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
        t_meas64 = np.asarray(t_meas, dtype=np.float64).reshape(3)

        if not self._initialized:
            self._t_nom = t_meas64.copy()
            self._R_nom = R_meas64.copy()
            self._x[:] = 0.0
            # Position/rotation start at measurement uncertainty;
            # velocities start uncertain so the first few ticks adapt fast.
            self._P = np.diag(
                [self._meas_trans_var] * 3 + [1.0] * 3
                + [self._meas_rot_var] * 3 + [10.0] * 3
            )
            self._last_time = float(timestamp)
            self._initialized = True
            return self.current()

        dt = float(timestamp) - self._last_time
        self._last_time = float(timestamp)
        if not (0.0 < dt < 0.5):
            dt = 1.0 / 20.0  # tick-rate fallback for clock hiccups

        # --- Predict (constant velocity) ---
        self._t_nom = self._t_nom + self._x[3:6] * dt
        self._R_nom = _so3_exp(self._x[9:12] * dt) @ self._R_nom
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt
        F[6:9, 9:12] = np.eye(3) * dt
        # Piecewise-constant white-acceleration noise per axis.
        q_p = self._accel_sigma ** 2
        q_r = self._alpha_sigma ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = np.zeros((12, 12))
        for i in range(3):
            Q[i, i] = q_p * dt4 / 4.0
            Q[i, i + 3] = Q[i + 3, i] = q_p * dt3 / 2.0
            Q[i + 3, i + 3] = q_p * dt2
            j = i + 6
            Q[j, j] = q_r * dt4 / 4.0
            Q[j, j + 3] = Q[j + 3, j] = q_r * dt3 / 2.0
            Q[j + 3, j + 3] = q_r * dt2
        self._P = F @ self._P @ F.T + Q

        # --- Update ---
        # Innovation: translation residual + rotation residual on the
        # SO(3) tangent (world frame, left convention).
        y = np.concatenate([
            t_meas64 - self._t_nom,
            _so3_log(R_meas64 @ self._R_nom.T),
        ])

        # Innovation gate: discontinuity (reacquisition after the object
        # left the view, or an anchor-pool jump) — snap to the
        # measurement and restart the velocity estimate instead of
        # smoothing through it (which would overshoot).
        if (
            np.linalg.norm(y[0:3]) > self._snap_trans_m
            or np.linalg.norm(y[3:6]) > self._snap_rot_rad
        ):
            self._t_nom = t_meas64.copy()
            self._R_nom = R_meas64.copy()
            self._x[:] = 0.0
            self._P = np.diag(
                [self._meas_trans_var] * 3 + [1.0] * 3
                + [self._meas_rot_var] * 3 + [10.0] * 3
            )
            return self.current()

        H = np.zeros((6, 12))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        R_noise = np.diag([self._meas_trans_var] * 3 + [self._meas_rot_var] * 3)
        S = H @ self._P @ H.T + R_noise
        K = self._P @ H.T @ np.linalg.solve(S, np.eye(6))
        dx = K @ y
        self._P = (np.eye(12) - K @ H) @ self._P

        # Inject: position error folds into the nominal directly (the
        # position block of x doubles as the error state pre-injection).
        self._t_nom = self._t_nom + dx[0:3]
        self._x[3:6] += dx[3:6]
        self._R_nom = _so3_exp(dx[6:9]) @ self._R_nom
        self._x[9:12] += dx[9:12]
        # Error states are zeroed after injection (x[0:3]/x[6:9] never
        # accumulate — they live only inside dx).

        return self.current()


def estimate_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Closed-form Kabsch SVD for source→target rigid alignment."""
    if source_points.shape != target_points.shape or source_points.shape[0] < 3:
        return None

    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_zero = source_points - source_center[None, :]
    target_zero = target_points - target_center[None, :]

    try:
        u, _, vh = np.linalg.svd(source_zero.T @ target_zero, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh = vh.copy()
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    return rotation.astype(np.float32), translation.astype(np.float32)


def estimate_rigid_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float,
    iterations: int,
    min_inliers: int,
) -> Optional[dict]:
    """3-point RANSAC over Kabsch.

    Returns ``None`` only when no RANSAC trial produced ≥3 inliers.
    When trials produced inliers but fewer than ``min_inliers``, returns
    the best-trial pose AND residual stats over ALL pairs so the caller
    can diagnose whether the threshold was just tight (residuals close
    to ``threshold``) or matches are garbage (residuals in metres).
    """
    if source_points.shape != target_points.shape or source_points.shape[0] < max(min_inliers, 3):
        return None

    # Fixed seed per call → RANSAC is deterministic for a given match set (the
    # only tick-to-tick stochasticity is the match set itself). DGS_RANSAC_SEED
    # overrides for seed-ensemble experiments (measuring sampling variance).
    rng = np.random.default_rng(int(os.environ.get("DGS_RANSAC_SEED", "12345")))
    best_rotation = None
    best_translation = None
    best_inlier_mask = None
    best_inlier_count = 0
    best_mean_residual = float("inf")

    all_indices = np.arange(source_points.shape[0])
    for _ in range(iterations):
        sample_indices = rng.choice(all_indices, size=3, replace=False)
        transform = estimate_rigid_transform(source_points[sample_indices], target_points[sample_indices])
        if transform is None:
            continue
        rotation, translation = transform
        residuals = np.linalg.norm(source_points @ rotation.T + translation[None, :] - target_points, axis=1)
        inlier_mask = np.isfinite(residuals) & (residuals <= threshold)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 3:
            continue
        mean_residual = float(residuals[inlier_mask].mean()) if inlier_count > 0 else float("inf")
        if inlier_count > best_inlier_count or (
            inlier_count == best_inlier_count and mean_residual < best_mean_residual
        ):
            best_rotation = rotation
            best_translation = translation
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_mean_residual = mean_residual

    if best_inlier_mask is None:
        return None

    residuals_all = np.linalg.norm(
        source_points @ best_rotation.T + best_translation[None, :] - target_points,
        axis=1,
    )
    if best_inlier_count < min_inliers:
        return {
            "rotation": best_rotation.astype(np.float32),
            "translation": best_translation.astype(np.float32),
            "inlier_mask": best_inlier_mask,
            "mean_residual": float(np.mean(residuals_all)),
            "median_residual": float(np.median(residuals_all)),
        }

    # Refit on the inlier set for sub-pixel precision.
    refined = estimate_rigid_transform(source_points[best_inlier_mask], target_points[best_inlier_mask])
    if refined is None:
        return None
    rotation, translation = refined
    residuals = np.linalg.norm(source_points @ rotation.T + translation[None, :] - target_points, axis=1)
    inlier_mask = np.isfinite(residuals) & (residuals <= threshold)
    if int(inlier_mask.sum()) < min_inliers:
        return {
            "rotation": best_rotation.astype(np.float32),
            "translation": best_translation.astype(np.float32),
            "inlier_mask": best_inlier_mask,
            "mean_residual": float(np.mean(residuals_all)),
            "median_residual": float(np.median(residuals_all)),
        }

    return {
        "rotation": rotation.astype(np.float32),
        "translation": translation.astype(np.float32),
        "inlier_mask": inlier_mask,
        "mean_residual": float(residuals[inlier_mask].mean()),
        "median_residual": float(np.median(residuals[inlier_mask])),
    }
