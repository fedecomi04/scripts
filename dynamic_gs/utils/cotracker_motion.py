from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nerfstudio.cameras.cameras import Cameras


@dataclass
class CoTrackerMotionEstimate:
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
    previous_points_xy: Optional[np.ndarray] = None  # (K, 2) on previous frame
    current_points_xy: Optional[np.ndarray] = None    # (K, 2) on current frame
    tracked_inlier_mask: Optional[np.ndarray] = None  # (K,) True only for RANSAC inliers
    previous_rgb: Optional[Tensor] = None             # previous frame image
    current_rgb: Optional[Tensor] = None              # current frame image


class CoTrackerMotionEstimator:
    """Offline pairwise CoTracker motion estimation for a rigid dynamic object."""

    def __init__(
        self,
        device: torch.device | str,
        query_point_count: int,
        min_track_points: int,
        ransac_iterations: int,
        ransac_inlier_threshold: float,
        point_refresh_min_distance: float,
        checkpoint_path: str = "",
        hub_repo: str = "facebookresearch/co-tracker",
        hub_model: str = "cotracker3_offline",
    ) -> None:
        self.device = torch.device(device)
        self.query_point_count = max(int(query_point_count), 3)
        self.min_track_points = max(int(min_track_points), 3)
        self.ransac_iterations = max(int(ransac_iterations), 1)
        self.ransac_inlier_threshold = float(ransac_inlier_threshold)
        self.point_refresh_min_distance = float(point_refresh_min_distance)
        self.checkpoint_path = checkpoint_path.strip()
        self.hub_repo = hub_repo
        self.hub_model = hub_model

        self._predictor = None
        self._previous_rgb: Optional[Tensor] = None
        self._previous_depth: Optional[np.ndarray] = None
        self._previous_intrinsics: Optional[np.ndarray] = None
        self._previous_camera_to_world: Optional[np.ndarray] = None
        self._reference_mask: Optional[Tensor] = None
        self._reference_world_points: Optional[np.ndarray] = None
        self._current_points_xy: Optional[np.ndarray] = None
        self.last_init_fast_point_count = 0
        self.last_init_sampled_count = 0
        self.last_init_depth_valid_count = 0
        self.last_init_used_dense_fallback = False

    @property
    def ready(self) -> bool:
        return (
            self._previous_rgb is not None
            and self._previous_depth is not None
            and self._previous_intrinsics is not None
            and self._previous_camera_to_world is not None
            and self._reference_world_points is not None
            and self._current_points_xy is not None
            and len(self._current_points_xy) >= self.min_track_points
        )

    @property
    def current_track_count(self) -> int:
        if self._current_points_xy is None:
            return 0
        return int(len(self._current_points_xy))

    def initialize(self, rgb: Tensor, depth: Tensor, camera: Cameras, mask: Tensor) -> int:
        self._previous_rgb = self._prepare_tracking_rgb(rgb)
        self._previous_depth = self._prepare_depth_image(depth)
        self._previous_intrinsics = self._extract_intrinsics(camera)
        self._previous_camera_to_world = self._extract_camera_to_world(camera)
        self._reference_mask = self._resize_mask(mask, self._previous_rgb.shape[:2]).detach().clone()
        if self._previous_depth.shape != self._previous_rgb.shape[:2]:
            raise RuntimeError(
                "CoTracker initialization requires RGB and depth at the same resolution, "
                f"got rgb={tuple(self._previous_rgb.shape[:2])} depth={tuple(self._previous_depth.shape)}."
            )
        sampled_points = self._sample_mask_points(
            mask,
            max_points=self.query_point_count,
            rgb=self._previous_rgb,
            output_shape=self._previous_rgb.shape[:2],
        )
        self.last_init_fast_point_count = int(len(sampled_points))
        self.last_init_used_dense_fallback = False
        if len(sampled_points) < self.min_track_points:
            sampled_points = self._sample_mask_points(
                mask,
                max_points=self.query_point_count,
                rgb=None,
                output_shape=self._previous_rgb.shape[:2],
            )
            self.last_init_used_dense_fallback = True
        self.last_init_sampled_count = int(len(sampled_points))
        reference_depth_values, reference_depth_valid = self._sample_depth_bilinear(self._previous_depth, sampled_points)
        self.last_init_depth_valid_count = int(reference_depth_valid.sum())
        sampled_points = sampled_points[reference_depth_valid]
        reference_depth_values = reference_depth_values[reference_depth_valid]
        self._current_points_xy = sampled_points.astype(np.float32)
        self._reference_world_points = self._backproject_to_world(
            self._current_points_xy,
            reference_depth_values,
            self._previous_intrinsics,
            self._previous_camera_to_world,
        )
        return int(len(self._current_points_xy))

    def replace_tracking_points(self, mask: Tensor) -> int:
        """Reference-query mode keeps the original sampled points fixed."""
        del mask
        return 0

    def filter_points_by_mask(self, mask: Tensor) -> int:
        """Remove tracked points that fall outside the given mask. Returns count removed."""
        if self._current_points_xy is None or len(self._current_points_xy) == 0:
            return 0
        # Convert mask to numpy binary at the resolution matching the tracked points
        m = mask.detach().float()
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0.5).cpu().numpy()
        # Scale points to mask resolution if needed
        if self._previous_rgb is not None:
            pts_h, pts_w = self._previous_rgb.shape[:2]
            mask_h, mask_w = m.shape
            scale_x = mask_w / pts_w if pts_w > 0 else 1.0
            scale_y = mask_h / pts_h if pts_h > 0 else 1.0
        else:
            scale_x, scale_y = 1.0, 1.0
        keep = []
        for pt in self._current_points_xy:
            mx = int(round(pt[0] * scale_x))
            my = int(round(pt[1] * scale_y))
            mx = max(0, min(mx, m.shape[1] - 1))
            my = max(0, min(my, m.shape[0] - 1))
            keep.append(m[my, mx])
        keep = np.array(keep, dtype=bool)
        removed = int((~keep).sum())
        if removed > 0 and keep.any():
            self._current_points_xy = self._current_points_xy[keep]
        return removed

    def refresh_tracking_points(self, mask: Tensor) -> int:
        del mask
        return 0

    def estimate_and_advance(
        self,
        current_rgb: Tensor,
        current_depth: Tensor,
        current_camera: Cameras,
        current_mask: Tensor | None = None,
    ) -> CoTrackerMotionEstimate:
        identity = np.eye(3, dtype=np.float32)
        zero = np.zeros((3,), dtype=np.float32)
        track_count_before = self.current_track_count

        current_rgb_prepared = self._prepare_tracking_rgb(current_rgb)
        current_depth_prepared = self._prepare_depth_image(current_depth)
        current_intrinsics = self._extract_intrinsics(current_camera)
        current_camera_to_world = self._extract_camera_to_world(current_camera)
        if current_depth_prepared.shape != current_rgb_prepared.shape[:2]:
            raise RuntimeError(
                "CoTracker motion estimation requires RGB and depth at the same resolution, "
                f"got rgb={tuple(current_rgb_prepared.shape[:2])} depth={tuple(current_depth_prepared.shape)}."
            )

        if not self.ready:
            return CoTrackerMotionEstimate(
                success=False,
                ready=False,
                rotation=identity,
                translation=zero,
                correspondence_count=0,
                inlier_count=0,
                track_count_before=track_count_before,
                track_count_after=self.current_track_count,
                raw_visible_count=0,
                mask_visible_count=0,
                depth_valid_count=0,
                used_mask_fallback=False,
                mean_residual=float("inf"),
                median_residual=float("inf"),
            )

        predictor = self._get_predictor()
        debug_prev_points = self._current_points_xy.copy()
        debug_prev_rgb = self._previous_rgb.clone()
        query_points = torch.from_numpy(self._current_points_xy.astype(np.float32)).to(self.device)
        query_frames = torch.zeros((query_points.shape[0], 1), dtype=query_points.dtype, device=self.device)
        queries = torch.cat([query_frames, query_points], dim=-1).unsqueeze(0)

        video = torch.stack([self._previous_rgb, current_rgb_prepared], dim=0).permute(0, 3, 1, 2).unsqueeze(0)
        video = video.to(self.device, non_blocking=True)

        with torch.no_grad():
            tracks, visibility = predictor(video, queries=queries)

        current_points_xy = tracks[0, 1].detach().cpu().numpy().astype(np.float32)
        visibility_now = visibility[0, 1]
        if visibility_now.ndim > 1:
            visibility_now = visibility_now.squeeze(-1)
        visibility_now = visibility_now.detach().cpu().numpy().astype(bool)

        current_points_xy, visibility_now = self._filter_points_in_image(
            current_points_xy,
            visibility_now,
            width=current_depth_prepared.shape[1],
            height=current_depth_prepared.shape[0],
        )
        raw_visibility = visibility_now.copy()
        raw_visible_count = int(raw_visibility.sum())

        used_mask_fallback = False
        mask_visible_count = raw_visible_count
        if current_mask is not None:
            masked_visibility = self._filter_points_by_mask_array(
                current_points_xy,
                visibility_now,
                current_mask,
                output_shape=current_depth_prepared.shape,
            )
            mask_visible_count = int(masked_visibility.sum())
            if mask_visible_count >= self.min_track_points:
                visibility_now = masked_visibility
            else:
                visibility_now = raw_visibility
                used_mask_fallback = current_mask is not None

        correspondence_mask = visibility_now.copy()
        current_depth_values, current_depth_valid = self._sample_depth_bilinear(current_depth_prepared, current_points_xy)
        depth_compatible_mask = correspondence_mask & current_depth_valid
        depth_valid_count = int(depth_compatible_mask.sum())
        correspondence_mask = depth_compatible_mask

        prev_world = self._reference_world_points[correspondence_mask]
        curr_world = self._backproject_to_world(
            current_points_xy[correspondence_mask],
            current_depth_values[correspondence_mask],
            current_intrinsics,
            current_camera_to_world,
        )

        rotation = identity
        translation = zero
        success = False
        inlier_count = 0
        mean_residual = float("inf")
        median_residual = float("inf")
        track_count_after = int(correspondence_mask.sum())
        tracked_inlier_mask = np.zeros((len(current_points_xy),), dtype=bool)

        if len(prev_world) >= self.min_track_points and len(curr_world) >= self.min_track_points:
            ransac_result = self._estimate_rigid_transform_ransac(
                prev_world,
                curr_world,
                threshold=self.ransac_inlier_threshold,
                iterations=self.ransac_iterations,
                min_inliers=self.min_track_points,
            )
            if ransac_result is not None:
                rotation = ransac_result["rotation"]
                translation = ransac_result["translation"]
                inlier_mask = ransac_result["inlier_mask"]
                inlier_count = int(inlier_mask.sum())
                mean_residual = float(ransac_result["mean_residual"])
                median_residual = float(ransac_result["median_residual"])
                success = inlier_count >= self.min_track_points
                tracked_indices = np.nonzero(correspondence_mask)[0]
                tracked_inlier_mask[tracked_indices[inlier_mask]] = True
                if success:
                    track_count_after = inlier_count

        return CoTrackerMotionEstimate(
            success=success,
            ready=True,
            rotation=rotation,
            translation=translation,
            correspondence_count=int(correspondence_mask.sum()),
            inlier_count=inlier_count,
            track_count_before=track_count_before,
            track_count_after=track_count_after,
            raw_visible_count=raw_visible_count,
            mask_visible_count=mask_visible_count,
            depth_valid_count=depth_valid_count,
            used_mask_fallback=used_mask_fallback,
            mean_residual=mean_residual,
            median_residual=median_residual,
            previous_points_xy=debug_prev_points,
            current_points_xy=current_points_xy,
            tracked_inlier_mask=tracked_inlier_mask,
            previous_rgb=debug_prev_rgb,
            current_rgb=current_rgb_prepared,
        )

    def _advance_state(
        self,
        rgb: Tensor,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> None:
        self._previous_rgb = rgb
        self._previous_depth = depth
        self._previous_intrinsics = intrinsics
        self._previous_camera_to_world = camera_to_world

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        if self.checkpoint_path:
            try:
                from cotracker.predictor import CoTrackerPredictor
            except ImportError as exc:  # pragma: no cover - runtime dependency guard
                raise ImportError(
                    "CoTracker package is not installed. Install facebookresearch/co-tracker "
                    "or leave cotracker_checkpoint_path empty to use torch.hub."
                ) from exc
            predictor = CoTrackerPredictor(checkpoint=self.checkpoint_path, offline=True)
        else:
            try:
                predictor = torch.hub.load(self.hub_repo, self.hub_model)
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                raise RuntimeError(
                    "Failed to load Meta CoTracker offline predictor via torch.hub. "
                    "Install facebookresearch/co-tracker locally or provide cotracker_checkpoint_path."
                ) from exc

        predictor = predictor.to(self.device)
        predictor.eval()
        self._predictor = predictor
        return predictor



    @staticmethod
    def _prepare_tracking_rgb(image: Tensor) -> Tensor:
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected HxWxC image tensor for CoTracker, got shape {tuple(image.shape)}")
        if image.shape[-1] > 3:
            image = image[..., :3]
        image = image.detach().float().cpu()
        if image.max().item() <= 1.0 + 1e-6:
            image = image * 255.0
        return image.clamp(0.0, 255.0)

    @staticmethod
    def _prepare_depth_image(depth: Tensor) -> np.ndarray:
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth_np = depth.detach().float().cpu().numpy().astype(np.float32)
        return depth_np

    @staticmethod
    def _extract_intrinsics(camera: Cameras) -> np.ndarray:
        intrinsics = camera.get_intrinsics_matrices()[0].detach().cpu().numpy().astype(np.float32)
        return intrinsics

    @staticmethod
    def _extract_camera_to_world(camera: Cameras) -> np.ndarray:
        camera_to_world = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)
        if camera_to_world.shape == (4, 4):
            camera_to_world = camera_to_world[:3, :]
        return camera_to_world

    @staticmethod
    def _resize_mask(mask: Tensor, output_shape: tuple[int, int] | None) -> Tensor:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if output_shape is None or tuple(mask.shape[:2]) == tuple(output_shape):
            return mask
        mask_4d = mask[None, None].float()
        resized = F.interpolate(mask_4d, size=output_shape, mode="nearest")
        return resized[0, 0]

    @staticmethod
    def _subsample_points(points_xy: np.ndarray, max_points: int) -> np.ndarray:
        if len(points_xy) <= max_points:
            return points_xy.astype(np.float32)
        keep = np.linspace(0, len(points_xy) - 1, num=max_points)
        keep = np.unique(np.round(keep).astype(np.int64))
        return points_xy[keep].astype(np.float32)

    @staticmethod
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

    @staticmethod
    def _sample_mask_points(
        mask: Tensor,
        max_points: int,
        rgb: Tensor | None = None,
        output_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        mask = CoTrackerMotionEstimator._resize_mask(mask, output_shape)
        mask_np = (mask.detach().float().cpu().numpy() > 0.5)
        sample_mask_np = CoTrackerMotionEstimator._shrink_mask_for_sampling(mask_np)
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
                    return CoTrackerMotionEstimator._subsample_points(np.asarray(fast_points, dtype=np.float32), max_points)

        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        return CoTrackerMotionEstimator._subsample_points(coords, max_points)

    @staticmethod
    def _filter_points_in_image(
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

    @staticmethod
    def _filter_points_by_mask_array(
        points_xy: np.ndarray,
        visibility: np.ndarray,
        mask: Tensor,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        resized_mask = CoTrackerMotionEstimator._resize_mask(mask, output_shape)
        mask_np = (resized_mask.detach().float().cpu().numpy() > 0.5)
        if mask_np.size == 0:
            return np.zeros_like(visibility, dtype=bool)

        xs = np.clip(np.round(points_xy[:, 0]).astype(np.int64), 0, mask_np.shape[1] - 1)
        ys = np.clip(np.round(points_xy[:, 1]).astype(np.int64), 0, mask_np.shape[0] - 1)
        return visibility & mask_np[ys, xs]

    @staticmethod
    def _sample_depth_bilinear(depth: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            np.isfinite(d00)
            & np.isfinite(d01)
            & np.isfinite(d10)
            & np.isfinite(d11)
            & (d00 > 0.0)
            & (d01 > 0.0)
            & (d10 > 0.0)
            & (d11 > 0.0)
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

    @staticmethod
    def _backproject_to_world(
        points_xy: np.ndarray,
        depth_values: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
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

        # Nerfstudio camera_to_worlds use OpenGL-style camera coordinates:
        # +X right, +Y up, +Z back, and the camera looks along -Z.
        rotation = camera_to_world[:, :3]
        translation = camera_to_world[:, 3]
        return (camera_points @ rotation.T + translation[None, :]).astype(np.float32)

    @staticmethod
    def _estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
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

    def _estimate_rigid_transform_ransac(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        threshold: float,
        iterations: int,
        min_inliers: int,
    ) -> Optional[dict]:
        if source_points.shape != target_points.shape or source_points.shape[0] < max(min_inliers, 3):
            return None

        rng = np.random.default_rng(12345)
        best_rotation = None
        best_translation = None
        best_inlier_mask = None
        best_inlier_count = 0
        best_mean_residual = float("inf")

        all_indices = np.arange(source_points.shape[0])
        for _ in range(iterations):
            sample_indices = rng.choice(all_indices, size=3, replace=False)
            transform = self._estimate_rigid_transform(source_points[sample_indices], target_points[sample_indices])
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

        if best_inlier_mask is None or best_inlier_count < min_inliers:
            return None

        # The applied rigid transform is refit using only the final inlier set.
        refined_transform = self._estimate_rigid_transform(
            source_points[best_inlier_mask],
            target_points[best_inlier_mask],
        )
        if refined_transform is None:
            return None
        rotation, translation = refined_transform
        residuals = np.linalg.norm(source_points @ rotation.T + translation[None, :] - target_points, axis=1)
        inlier_mask = np.isfinite(residuals) & (residuals <= threshold)
        if int(inlier_mask.sum()) < min_inliers:
            return None

        return {
            "rotation": rotation.astype(np.float32),
            "translation": translation.astype(np.float32),
            "inlier_mask": inlier_mask,
            "mean_residual": float(residuals[inlier_mask].mean()),
            "median_residual": float(np.median(residuals[inlier_mask])),
        }
