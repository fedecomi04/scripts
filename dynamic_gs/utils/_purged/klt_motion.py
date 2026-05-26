"""Pyramidal Lucas-Kanade (KLT) optical-flow motion estimator.

Third backend for dynamic-gs's 2D point tracker, alongside CoTracker and
TAPIR. Same public surface as :class:`CoTrackerMotionEstimator`:

    * ``initialize(rgb, depth, camera, mask) -> int``
    * ``estimate_and_advance(rgb, depth, camera, mask, current_object_mask) -> CoTrackerMotionEstimate``
    * ``ready``, ``current_track_count`` properties
    * ``last_init_*`` diagnostic fields

Departures from CoTracker / TAPIR:

* Pure CPU. No model load, no GPU contention, no JIT warmup.
* No anchor at D0. Unlike TAPIR which back-projects D0 query points
  once and re-uses them as the world-frame anchor for every subsequent
  frame, KLT *resamples fresh keypoints every frame* and tracks them
  pairwise (frame N-1 → frame N). The resulting Kabsch fit yields a
  per-frame incremental rigid transform δT, and the cumulative
  reference-to-current transform is composed iteratively as
  ``T_total^N = δT^N · δT^(N-1) · ... · δT^1``. This drifts over time —
  composition error accumulates — but the regime KLT excels in is
  small inter-frame motion, which is exactly what high-fps capture
  delivers, so refreshing the point set frame-to-frame keeps the
  tracker on points that are still well-textured and visible rather
  than persisting through rotation / occlusion.
* Sample region is computed per frame from the rendered object mask
  passed in via the ``current_object_mask`` kwarg of
  ``estimate_and_advance``: erode the object mask so its area shrinks
  to ~``sample_inner_area_ratio`` (default 0.8) of the original, then
  intersect with the gripper "keep" mask passed via ``current_mask``.
  This biases keypoints toward the textured interior of the object
  while excluding gripper-occluded pixels.

Performance target on consumer GPU + CPU: ≈5-10 ms per tick (≈100-200
Hz cap), vs. TAPIR's ≈50 ms / CoTracker's ≈45 ms.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
from torch import Tensor

from .cotracker_motion import CoTrackerMotionEstimate, CoTrackerMotionEstimator

if TYPE_CHECKING:
    from nerfstudio.cameras.cameras import Cameras


class KltMotionEstimator:
    """Pyramidal Lucas-Kanade optical-flow motion estimator with per-frame resample.

    Per-frame loop in ``estimate_and_advance``:

    1. Convert current RGB to grayscale uint8.
    2. If a previous frame is cached, run ``cv2.calcOpticalFlowPyrLK``
       to track the previously sampled keypoints into the current
       frame.
    3. Filter status==1 ∧ in-image ∧ inside the current sample region
       ∧ depth-valid in both prev and curr frames.
    4. Back-project surviving correspondences (prev_xy via prev depth +
       prev c2w; curr_xy via curr depth + curr c2w) into world frame.
    5. Run RANSAC-Kabsch on the (prev_world, curr_world) pairs → δT.
    6. Compose δT into the cumulative D0→curr transform and return it.
    7. Resample fresh keypoints inside the current frame's sample
       region; cache them as the seed for the next call.
    """

    def __init__(
        self,
        device: torch.device | str,
        query_point_count: int,
        min_track_points: int,
        ransac_iterations: int,
        ransac_inlier_threshold: float,
        pyramid_levels: int = 3,
        window_size: int = 15,
        lk_iterations: int = 20,
        lk_eps: float = 0.03,
        fast_threshold: int = 28,
        sample_inner_area_ratio: float = 0.8,
    ) -> None:
        # Device is unused for KLT (pure CPU), but kept on the class for
        # API parity with the GPU-bound estimators.
        self.device = torch.device(device)
        self.query_point_count = max(int(query_point_count), 3)
        self.min_track_points = max(int(min_track_points), 3)
        self.ransac_iterations = max(int(ransac_iterations), 1)
        self.ransac_inlier_threshold = float(ransac_inlier_threshold)
        self.pyramid_levels = max(int(pyramid_levels), 0)
        self.window_size = max(int(window_size), 3)
        self.lk_criteria = (
            cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT,
            max(int(lk_iterations), 1),
            float(lk_eps),
        )
        self.fast_threshold = int(fast_threshold)
        self.sample_inner_area_ratio = float(sample_inner_area_ratio)
        self._fast_detector = cv2.FastFeatureDetector_create(
            threshold=self.fast_threshold, nonmaxSuppression=True,
        )

        # Per-frame state cached across calls. ``previous_points`` is in
        # OpenCV's expected (N, 1, 2) float32 layout for calcOpticalFlowPyrLK.
        self._previous_gray: Optional[np.ndarray] = None
        self._previous_points: Optional[np.ndarray] = None
        self._previous_depth: Optional[np.ndarray] = None
        self._previous_intrinsics: Optional[np.ndarray] = None
        self._previous_camera_to_world: Optional[np.ndarray] = None

        # Cumulative D0→current transform. Identity at init.
        self._cumulative_R: np.ndarray = np.eye(3, dtype=np.float32)
        self._cumulative_t: np.ndarray = np.zeros((3,), dtype=np.float32)

        # Diagnostic fields read by the pipeline log line at D0.
        self.last_init_fast_point_count = 0
        self.last_init_sampled_count = 0
        self.last_init_depth_valid_count = 0
        self.last_init_used_dense_fallback = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return (
            self._previous_gray is not None
            and self._previous_points is not None
            and len(self._previous_points) >= self.min_track_points
            and self._previous_depth is not None
            and self._previous_intrinsics is not None
            and self._previous_camera_to_world is not None
        )

    @property
    def current_track_count(self) -> int:
        if self._previous_points is None:
            return 0
        return int(len(self._previous_points))

    # ------------------------------------------------------------------
    # Initialisation (D0)
    # ------------------------------------------------------------------

    def initialize(self, rgb: Tensor, depth: Tensor, camera: "Cameras", mask: Tensor) -> int:
        """Seed KLT state with frame 0. ``mask`` is the rendered object mask."""
        # Reset cumulative pose every (re-)initialisation.
        self._cumulative_R = np.eye(3, dtype=np.float32)
        self._cumulative_t = np.zeros((3,), dtype=np.float32)

        rgb_t = CoTrackerMotionEstimator._prepare_tracking_rgb(rgb)
        depth_np = CoTrackerMotionEstimator._prepare_depth_image(depth)
        intrinsics = CoTrackerMotionEstimator._extract_intrinsics(camera)
        camera_to_world = CoTrackerMotionEstimator._extract_camera_to_world(camera)
        if depth_np.shape != rgb_t.shape[:2]:
            raise RuntimeError(
                "KLT initialization requires RGB and depth at the same resolution, "
                f"got rgb={tuple(rgb_t.shape[:2])} depth={tuple(depth_np.shape)}."
            )

        gray = self._rgb_tensor_to_gray_u8(rgb_t)
        obj_mask_np = self._mask_to_numpy(mask, gray.shape[:2])
        if obj_mask_np is None or not obj_mask_np.any():
            self._previous_points = None
            self.last_init_fast_point_count = 0
            self.last_init_sampled_count = 0
            self.last_init_depth_valid_count = 0
            self.last_init_used_dense_fallback = False
            return 0

        sample_region = self._compute_sample_region(obj_mask_np, gripper_mask_np=None)
        if sample_region is None:
            self._previous_points = None
            self.last_init_fast_point_count = 0
            self.last_init_sampled_count = 0
            self.last_init_depth_valid_count = 0
            self.last_init_used_dense_fallback = False
            return 0

        points, fast_count, used_dense = self._detect_keypoints(gray, sample_region)
        self.last_init_fast_point_count = int(fast_count)
        self.last_init_used_dense_fallback = bool(used_dense)
        self.last_init_sampled_count = int(len(points))
        if len(points) < self.min_track_points:
            self._previous_points = None
            self.last_init_depth_valid_count = 0
            return 0

        # Diagnostic only: count how many sampled points have a valid depth.
        depth_values, depth_valid = CoTrackerMotionEstimator._sample_depth_bilinear(
            depth_np, points.reshape(-1, 2),
        )
        self.last_init_depth_valid_count = int(depth_valid.sum())

        self._previous_gray = gray
        self._previous_points = points
        self._previous_depth = depth_np
        self._previous_intrinsics = intrinsics
        self._previous_camera_to_world = camera_to_world
        return int(len(points))

    # ------------------------------------------------------------------
    # Per-frame motion estimate
    # ------------------------------------------------------------------

    def estimate_and_advance(
        self,
        current_rgb: Tensor,
        current_depth: Tensor,
        current_camera: "Cameras",
        current_mask: Tensor | None = None,
        current_object_mask: Tensor | None = None,
    ) -> CoTrackerMotionEstimate:
        identity = np.eye(3, dtype=np.float32)
        zero = np.zeros((3,), dtype=np.float32)
        track_count_before = self.current_track_count
        timings: dict = {}

        # --- Sub-timing: input prep (CPU only — no CUDA syncs because
        # the pipeline already pulled rgb/depth to CPU before calling) ---
        t = time.time()
        current_rgb_prepared = CoTrackerMotionEstimator._prepare_tracking_rgb(current_rgb)
        current_depth_prepared = CoTrackerMotionEstimator._prepare_depth_image(current_depth)
        current_intrinsics = CoTrackerMotionEstimator._extract_intrinsics(current_camera)
        current_camera_to_world = CoTrackerMotionEstimator._extract_camera_to_world(current_camera)
        current_gray = self._rgb_tensor_to_gray_u8(current_rgb_prepared)
        timings["input_prep"] = time.time() - t

        if current_depth_prepared.shape != current_rgb_prepared.shape[:2]:
            raise RuntimeError(
                "KLT motion estimation requires RGB and depth at the same resolution, "
                f"got rgb={tuple(current_rgb_prepared.shape[:2])} depth={tuple(current_depth_prepared.shape)}."
            )

        if not self.ready:
            timings["klt_forward"] = 0.0
            timings["postprocess"] = 0.0
            timings["ransac_kabsch"] = 0.0
            # Even when not ready we still attempt a resample so the
            # NEXT call has something to track. This is the path taken
            # when D0 sampling failed (e.g. tiny object mask).
            t_re = time.time()
            self._resample_state(
                current_gray, current_depth_prepared,
                current_camera_to_world, current_intrinsics,
                current_mask, current_object_mask,
            )
            timings["resample"] = time.time() - t_re
            return CoTrackerMotionEstimate(
                success=False, ready=False,
                rotation=self._cumulative_R.copy(), translation=self._cumulative_t.copy(),
                correspondence_count=0, inlier_count=0,
                track_count_before=track_count_before, track_count_after=self.current_track_count,
                raw_visible_count=0, mask_visible_count=0, depth_valid_count=0,
                used_mask_fallback=False, mean_residual=float("inf"), median_residual=float("inf"),
                timings=timings,
            )

        debug_prev_points = self._previous_points.reshape(-1, 2).copy()
        debug_prev_rgb = torch.from_numpy(
            cv2.cvtColor(self._previous_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
        )

        # --- Sub-timing: KLT forward (single call to OpenCV PyrLK) ---
        t = time.time()
        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            self._previous_points,
            None,
            winSize=(self.window_size, self.window_size),
            maxLevel=self.pyramid_levels,
            criteria=self.lk_criteria,
        )
        timings["klt_forward"] = time.time() - t

        # --- Sub-timing: postprocess (filter, sample-region intersect, depth back-projection) ---
        t = time.time()
        prev_xy = self._previous_points.reshape(-1, 2).astype(np.float32)
        height, width = current_depth_prepared.shape

        if next_pts is None or status is None:
            timings["postprocess"] = time.time() - t
            timings["ransac_kabsch"] = 0.0
            t_re = time.time()
            self._resample_state(
                current_gray, current_depth_prepared,
                current_camera_to_world, current_intrinsics,
                current_mask, current_object_mask,
            )
            timings["resample"] = time.time() - t_re
            return CoTrackerMotionEstimate(
                success=False, ready=True,
                rotation=self._cumulative_R.copy(), translation=self._cumulative_t.copy(),
                correspondence_count=0, inlier_count=0,
                track_count_before=track_count_before, track_count_after=self.current_track_count,
                raw_visible_count=0, mask_visible_count=0, depth_valid_count=0,
                used_mask_fallback=False, mean_residual=float("inf"), median_residual=float("inf"),
                previous_points_xy=debug_prev_points, current_points_xy=None,
                tracked_inlier_mask=None,
                previous_rgb=debug_prev_rgb,
                current_rgb=current_rgb_prepared,
                timings=timings,
            )

        curr_xy = next_pts.reshape(-1, 2).astype(np.float32)
        status_flat = status.flatten().astype(bool)

        # In-image filter on the current points.
        in_image = (
            np.isfinite(curr_xy).all(axis=1)
            & (curr_xy[:, 0] >= 0.0) & (curr_xy[:, 0] <= max(width - 1, 0))
            & (curr_xy[:, 1] >= 0.0) & (curr_xy[:, 1] <= max(height - 1, 0))
        )
        valid = status_flat & in_image
        raw_visible_count = int(valid.sum())

        # Sample-region filter (intersect with current frame's eroded
        # object mask ∩ gripper-keep mask). Computed here for both
        # filtering and resampling — caching the result avoids two
        # erodes per frame.
        gripper_np = self._mask_to_numpy(current_mask, current_gray.shape[:2]) if current_mask is not None else None
        obj_np = self._mask_to_numpy(current_object_mask, current_gray.shape[:2]) if current_object_mask is not None else None
        sample_region = self._compute_sample_region(obj_np, gripper_np) if obj_np is not None else None

        used_mask_fallback = False
        mask_visible_count = raw_visible_count
        if sample_region is not None:
            xs = np.clip(np.round(curr_xy[:, 0]).astype(np.int64), 0, sample_region.shape[1] - 1)
            ys = np.clip(np.round(curr_xy[:, 1]).astype(np.int64), 0, sample_region.shape[0] - 1)
            in_region = sample_region[ys, xs]
            masked_valid = valid & in_region
            mask_visible_count = int(masked_valid.sum())
            if mask_visible_count >= self.min_track_points:
                valid = masked_valid
            else:
                used_mask_fallback = True
                # Fall through with the unmasked ``valid``.

        # Depth must be valid in both frames for the world-frame fit.
        prev_depth_values, prev_depth_valid = CoTrackerMotionEstimator._sample_depth_bilinear(
            self._previous_depth, prev_xy,
        )
        curr_depth_values, curr_depth_valid = CoTrackerMotionEstimator._sample_depth_bilinear(
            current_depth_prepared, curr_xy,
        )
        depth_compatible = valid & prev_depth_valid & curr_depth_valid
        depth_valid_count = int(depth_compatible.sum())
        correspondence_mask = depth_compatible

        prev_world = CoTrackerMotionEstimator._backproject_to_world(
            prev_xy[correspondence_mask], prev_depth_values[correspondence_mask],
            self._previous_intrinsics, self._previous_camera_to_world,
        )
        curr_world = CoTrackerMotionEstimator._backproject_to_world(
            curr_xy[correspondence_mask], curr_depth_values[correspondence_mask],
            current_intrinsics, current_camera_to_world,
        )
        timings["postprocess"] = time.time() - t

        # --- Sub-timing: RANSAC-Kabsch (numpy loop) ---
        t = time.time()
        delta_R = identity
        delta_t = zero
        success = False
        inlier_count = 0
        mean_residual = float("inf")
        median_residual = float("inf")
        tracked_inlier_mask = np.zeros((len(curr_xy),), dtype=bool)
        track_count_after = int(correspondence_mask.sum())

        if len(prev_world) >= self.min_track_points and len(curr_world) >= self.min_track_points:
            ransac_helper = CoTrackerMotionEstimator.__new__(CoTrackerMotionEstimator)
            ransac_helper.min_track_points = self.min_track_points
            ransac_helper.ransac_iterations = self.ransac_iterations
            ransac_helper.ransac_inlier_threshold = self.ransac_inlier_threshold
            ransac_result = ransac_helper._estimate_rigid_transform_ransac(
                prev_world, curr_world,
                threshold=self.ransac_inlier_threshold,
                iterations=self.ransac_iterations,
                min_inliers=self.min_track_points,
            )
            if ransac_result is not None:
                delta_R = ransac_result["rotation"]
                delta_t = ransac_result["translation"]
                inlier_mask = ransac_result["inlier_mask"]
                inlier_count = int(inlier_mask.sum())
                mean_residual = float(ransac_result["mean_residual"])
                median_residual = float(ransac_result["median_residual"])
                success = inlier_count >= self.min_track_points
                tracked_indices = np.nonzero(correspondence_mask)[0]
                tracked_inlier_mask[tracked_indices[inlier_mask]] = True
                if success:
                    track_count_after = inlier_count
        timings["ransac_kabsch"] = time.time() - t

        # --- Compose δT into the cumulative D0→current transform.
        # For p_0 in frame 0: p_curr = R_total · p_0 + t_total. Each
        # successful Kabsch fit yields R_δ, t_δ such that
        # p_curr ≈ R_δ · p_prev + t_δ, so the cumulative update is
        #   R_total ← R_δ · R_total
        #   t_total ← R_δ · t_total + t_δ
        if success:
            new_R = delta_R @ self._cumulative_R
            new_t = (delta_R @ self._cumulative_t.reshape(3, 1)).flatten() + delta_t
            self._cumulative_R = new_R.astype(np.float32)
            self._cumulative_t = new_t.astype(np.float32)
        rotation_out = self._cumulative_R.copy()
        translation_out = self._cumulative_t.copy()

        # --- Sub-timing: resample fresh keypoints for next call.
        # Done regardless of RANSAC success: even if the Kabsch fit
        # failed, we still want fresh points for the next iteration.
        t = time.time()
        self._resample_state(
            current_gray, current_depth_prepared,
            current_camera_to_world, current_intrinsics,
            current_mask, current_object_mask,
            cached_sample_region=sample_region,
        )
        timings["resample"] = time.time() - t

        return CoTrackerMotionEstimate(
            success=success, ready=True,
            rotation=rotation_out, translation=translation_out,
            correspondence_count=int(correspondence_mask.sum()), inlier_count=inlier_count,
            track_count_before=track_count_before, track_count_after=track_count_after,
            raw_visible_count=raw_visible_count, mask_visible_count=mask_visible_count,
            depth_valid_count=depth_valid_count, used_mask_fallback=used_mask_fallback,
            mean_residual=mean_residual, median_residual=median_residual,
            previous_points_xy=debug_prev_points, current_points_xy=curr_xy,
            tracked_inlier_mask=tracked_inlier_mask,
            previous_rgb=debug_prev_rgb, current_rgb=current_rgb_prepared,
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resample_state(
        self,
        gray: np.ndarray,
        depth_np: np.ndarray,
        camera_to_world: np.ndarray,
        intrinsics: np.ndarray,
        gripper_mask: Tensor | None,
        object_mask: Tensor | None,
        cached_sample_region: np.ndarray | None = None,
    ) -> int:
        """Cache the current frame as the seed for the next call.

        Resampling fresh FAST keypoints is the per-frame cost that
        compensates for the cumulative-transform drift: every call
        starts from a fresh, well-textured set of corners on the
        current frame, so the next pairwise LK track operates on the
        smallest possible inter-frame motion.
        """
        # State swap that always happens (so we don't end up with
        # stale gray/depth/c2w).
        self._previous_gray = gray
        self._previous_depth = depth_np
        self._previous_intrinsics = intrinsics
        self._previous_camera_to_world = camera_to_world

        if cached_sample_region is not None:
            sample_region = cached_sample_region
        else:
            obj_np = self._mask_to_numpy(object_mask, gray.shape[:2]) if object_mask is not None else None
            gripper_np = self._mask_to_numpy(gripper_mask, gray.shape[:2]) if gripper_mask is not None else None
            sample_region = self._compute_sample_region(obj_np, gripper_np) if obj_np is not None else None

        if sample_region is None:
            self._previous_points = None
            return 0

        points, _, _ = self._detect_keypoints(gray, sample_region)
        if len(points) < self.min_track_points:
            self._previous_points = None
            return 0
        self._previous_points = points
        return int(len(points))

    def _compute_sample_region(
        self,
        object_mask_np: np.ndarray | None,
        gripper_mask_np: np.ndarray | None,
    ) -> np.ndarray | None:
        """Erode object mask to ~``sample_inner_area_ratio`` of original area, then ∩ gripper-keep.

        Returns None when even the eroded-only fallback can't produce
        a region big enough to host ``min_track_points`` keypoints.
        """
        if object_mask_np is None or not np.any(object_mask_np):
            return None

        eroded = self._erode_to_inner_area(object_mask_np)
        if eroded is None or not np.any(eroded):
            return None

        if gripper_mask_np is None:
            return eroded

        intersected = eroded & gripper_mask_np
        if intersected.sum() < self.min_track_points:
            # Fallback: eroded only, without gripper subtraction. This
            # happens when the gripper visually covers the object —
            # rather than fail outright we accept gripper-bleed risk
            # and let the next stage's status/depth filters cull bad
            # points. Logged once per session at INFO via the print —
            # callers can see ``used_mask_fallback`` in the result.
            return eroded
        return intersected

    def _erode_to_inner_area(self, mask_np: np.ndarray) -> np.ndarray | None:
        """Erode ``mask_np`` so its area is ~``sample_inner_area_ratio`` of original.

        Uses an analytical kernel size based on the bounding box's
        longer side: ``margin = (1 - sqrt(area_ratio)) / 2 * side``.
        For default ratio 0.8, that's ~5.3% of the side length, which
        on a typical 100-200 px object is a 1-2 px margin — small
        enough to preserve the textured interior but large enough to
        push keypoints away from the silhouette where depth and color
        are unreliable.
        """
        ys, xs = np.nonzero(mask_np)
        if len(xs) == 0:
            return None
        side = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        margin = max(1, int(round(0.5 * (1.0 - np.sqrt(self.sample_inner_area_ratio)) * side)))
        kernel_size = 2 * margin + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1) > 0
        if not np.any(eroded):
            return mask_np  # erosion wiped the mask — return original as best-effort fallback
        return eroded

    def _detect_keypoints(
        self,
        gray: np.ndarray,
        sample_region: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """FAST inside ``sample_region``, falling back to goodFeaturesToTrack if FAST is sparse.

        Returns (points (N, 1, 2) float32, fast_count, used_dense_fallback).
        """
        gray_masked = gray.copy()
        gray_masked[~sample_region] = 0
        keypoints = self._fast_detector.detect(gray_masked, None)
        fast_points: list[list[float]] = []
        if keypoints:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
            for kp in keypoints:
                x = int(round(kp.pt[0]))
                y = int(round(kp.pt[1]))
                if x < 0 or x >= sample_region.shape[1] or y < 0 or y >= sample_region.shape[0]:
                    continue
                if not sample_region[y, x]:
                    continue
                fast_points.append([float(x), float(y)])
                if len(fast_points) >= self.query_point_count:
                    break
        fast_count = len(fast_points)

        if fast_count >= self.min_track_points:
            arr = np.asarray(fast_points, dtype=np.float32).reshape(-1, 1, 2)
            return arr, fast_count, False

        gft = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.query_point_count,
            qualityLevel=0.01, minDistance=5,
            mask=sample_region.astype(np.uint8) * 255,
        )
        if gft is not None and len(gft) >= self.min_track_points:
            return gft.astype(np.float32).reshape(-1, 1, 2), fast_count, True

        if fast_points:
            arr = np.asarray(fast_points, dtype=np.float32).reshape(-1, 1, 2)
            return arr, fast_count, False
        return np.zeros((0, 1, 2), dtype=np.float32), fast_count, False

    @staticmethod
    def _rgb_tensor_to_gray_u8(rgb_t: Tensor) -> np.ndarray:
        """Float HxWxC tensor in 0..255 → uint8 H×W grayscale."""
        rgb_np = rgb_t.detach().float().cpu().numpy()
        if rgb_np.ndim == 3 and rgb_np.shape[-1] >= 3:
            rgb_np = np.clip(rgb_np[..., :3], 0.0, 255.0).astype(np.uint8)
            return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
        if rgb_np.ndim == 3:
            rgb_np = rgb_np[..., 0]
        return rgb_np

    @staticmethod
    def _mask_to_numpy(mask, output_shape: tuple[int, int]) -> np.ndarray | None:
        """Resize ``mask`` (Tensor or ndarray) to ``output_shape`` and threshold."""
        if mask is None:
            return None
        if isinstance(mask, Tensor):
            resized = CoTrackerMotionEstimator._resize_mask(mask, output_shape)
            return resized.detach().float().cpu().numpy() > 0.5
        arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.shape[:2] != tuple(output_shape):
            arr = cv2.resize(
                arr.astype(np.float32),
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return arr > 0.5
