"""Online TAPIR (BootsTAPIR causal, PyTorch port) motion estimator.

Drop-in replacement for ``CoTrackerMotionEstimator`` that uses Google
DeepMind's online causal TAPIR. The 3D RANSAC anchor is still pinned at
D0 (back-projected through the depth at frame 0) so the recovered
``(R, t)`` is absolute world-frame; only the 2D tracker changes.

Why TAPIR over CoTracker:
  * Pure-PyTorch online API (no Jax/Haiku/CUDA-toolchain dance).
  * Per-frame stateful: a single feed-forward call refines ``causal_state``
    in place — no iterations, no support grid, no sliding window.
  * Native model resolution is 256x256; the encoder is small.

Vendored files live at ``third_party/tapnet/`` (torch model + transforms
helper only — the full repo is not cloned). Checkpoint at
``third_party/tapnet/checkpoints/causal_bootstapir_checkpoint.pt``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cotracker_motion import CoTrackerMotionEstimate, CoTrackerMotionEstimator

if TYPE_CHECKING:
    from nerfstudio.cameras.cameras import Cameras


_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
    "tapnet",
    "checkpoints",
    "causal_bootstapir_checkpoint.pt",
)
_TAPNET_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
    "tapnet",
)


def _ensure_tapnet_on_path() -> None:
    if _TAPNET_VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _TAPNET_VENDOR_ROOT)


class TapirMotionEstimator:
    """Online causal TAPIR motion estimator.

    Same public surface as :class:`CoTrackerMotionEstimator`:
      * ``initialize(rgb, depth, camera, mask) -> int``
      * ``estimate_and_advance(rgb, depth, camera, mask) -> CoTrackerMotionEstimate``
      * ``ready``, ``current_track_count`` properties.

    Differences in behavior:
      * No cadence skip — TAPIR's online forward is per-frame.
      * ``predictor_iters`` / ``predictor_resolution`` knobs are TAPIR-
        specific; resolution is fixed to 256x256 (model architecture
        constraint), iters is single-pass.
    """

    def __init__(
        self,
        device: torch.device | str,
        query_point_count: int,
        min_track_points: int,
        ransac_iterations: int,
        ransac_inlier_threshold: float,
        checkpoint_path: str = "",
        input_resolution: tuple[int, int] = (256, 256),
        visibility_threshold: float = 0.5,
        pyramid_level: int = 1,
        use_causal_conv: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.query_point_count = max(int(query_point_count), 3)
        self.min_track_points = max(int(min_track_points), 3)
        self.ransac_iterations = max(int(ransac_iterations), 1)
        self.ransac_inlier_threshold = float(ransac_inlier_threshold)
        self.checkpoint_path = checkpoint_path.strip() or _DEFAULT_CHECKPOINT
        self.input_resolution = tuple(int(v) for v in input_resolution)
        self.visibility_threshold = float(visibility_threshold)
        self.pyramid_level = int(pyramid_level)
        self.use_causal_conv = bool(use_causal_conv)

        self._model = None
        self._query_features = None
        self._causal_state = None
        self._original_size: Optional[tuple[int, int]] = None  # (H, W) of input frames

        self._previous_rgb: Optional[Tensor] = None
        self._previous_depth: Optional[np.ndarray] = None
        self._previous_intrinsics: Optional[np.ndarray] = None
        self._previous_camera_to_world: Optional[np.ndarray] = None
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
            and self._query_features is not None
            and self._causal_state is not None
            and len(self._current_points_xy) >= self.min_track_points
        )

    @property
    def current_track_count(self) -> int:
        if self._current_points_xy is None:
            return 0
        return int(len(self._current_points_xy))

    def _get_model(self):
        if self._model is not None:
            return self._model
        _ensure_tapnet_on_path()
        from tapnet.torch import tapir_model  # type: ignore

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"TAPIR checkpoint not found at {self.checkpoint_path}. Download "
                f"https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt"
            )

        model = tapir_model.TAPIR(
            pyramid_level=self.pyramid_level,
            use_casual_conv=self.use_causal_conv,
        )
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return model

    def _preprocess_frame_to_model_input(self, rgb_hw3_uint8_like: Tensor) -> Tensor:
        """Resize to ``input_resolution``, normalise to [-1, 1], shape [1, 1, h, w, 3]."""
        rgb = rgb_hw3_uint8_like
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        rgb = rgb.detach().float()
        if rgb.max().item() <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0)
        # CHW for interpolate
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        rgb = F.interpolate(rgb, size=self.input_resolution, mode="bilinear", align_corners=True)
        rgb = rgb.permute(0, 2, 3, 1)  # [1, h, w, 3]
        rgb = rgb / 255.0 * 2.0 - 1.0
        rgb = rgb.unsqueeze(1)  # [1, 1, h, w, 3]
        return rgb.to(self.device)

    def _xy_input_to_model(self, points_xy: np.ndarray) -> np.ndarray:
        """Scale (x, y) from input pixel space to model (256x256) space."""
        H_in, W_in = self._original_size  # type: ignore[misc]
        H_m, W_m = self.input_resolution
        sx = W_m / max(W_in - 1, 1)
        sy = H_m / max(H_in - 1, 1)
        out = points_xy.copy().astype(np.float32)
        out[:, 0] = out[:, 0] * sx
        out[:, 1] = out[:, 1] * sy
        return out

    def _xy_model_to_input(self, points_xy: np.ndarray) -> np.ndarray:
        H_in, W_in = self._original_size  # type: ignore[misc]
        H_m, W_m = self.input_resolution
        sx = max(W_in - 1, 1) / W_m
        sy = max(H_in - 1, 1) / H_m
        out = points_xy.copy().astype(np.float32)
        out[:, 0] = out[:, 0] * sx
        out[:, 1] = out[:, 1] * sy
        return out

    def initialize(self, rgb: Tensor, depth: Tensor, camera: Cameras, mask: Tensor) -> int:
        previous_rgb = CoTrackerMotionEstimator._prepare_tracking_rgb(rgb)
        previous_depth = CoTrackerMotionEstimator._prepare_depth_image(depth)
        intrinsics = CoTrackerMotionEstimator._extract_intrinsics(camera)
        camera_to_world = CoTrackerMotionEstimator._extract_camera_to_world(camera)
        if previous_depth.shape != previous_rgb.shape[:2]:
            raise RuntimeError(
                "TAPIR initialization requires RGB and depth at the same resolution, "
                f"got rgb={tuple(previous_rgb.shape[:2])} depth={tuple(previous_depth.shape)}."
            )
        self._previous_rgb = previous_rgb
        self._previous_depth = previous_depth
        self._previous_intrinsics = intrinsics
        self._previous_camera_to_world = camera_to_world
        self._original_size = (int(previous_rgb.shape[0]), int(previous_rgb.shape[1]))

        sampled_points = CoTrackerMotionEstimator._sample_mask_points(
            mask,
            max_points=self.query_point_count,
            rgb=previous_rgb,
            output_shape=previous_rgb.shape[:2],
        )
        self.last_init_fast_point_count = int(len(sampled_points))
        self.last_init_used_dense_fallback = False
        if len(sampled_points) < self.min_track_points:
            sampled_points = CoTrackerMotionEstimator._sample_mask_points(
                mask,
                max_points=self.query_point_count,
                rgb=None,
                output_shape=previous_rgb.shape[:2],
            )
            self.last_init_used_dense_fallback = True
        self.last_init_sampled_count = int(len(sampled_points))
        ref_depth, ref_valid = CoTrackerMotionEstimator._sample_depth_bilinear(previous_depth, sampled_points)
        self.last_init_depth_valid_count = int(ref_valid.sum())
        sampled_points = sampled_points[ref_valid]
        ref_depth = ref_depth[ref_valid]
        self._current_points_xy = sampled_points.astype(np.float32)
        self._reference_world_points = CoTrackerMotionEstimator._backproject_to_world(
            self._current_points_xy, ref_depth, intrinsics, camera_to_world,
        )

        # --- Initialise the TAPIR online state from frame 0 + query points. ---
        model = self._get_model()
        frame0 = self._preprocess_frame_to_model_input(previous_rgb)  # [1,1,h,w,3]
        query_xy_model = self._xy_input_to_model(self._current_points_xy)  # (N, 2) at 256
        N = query_xy_model.shape[0]
        # TAPIR expects (t, y, x) integer-ish coords (the demo uses int32).
        query_points_np = np.zeros((N, 3), dtype=np.float32)
        query_points_np[:, 0] = 0.0
        query_points_np[:, 1] = query_xy_model[:, 1]  # y
        query_points_np[:, 2] = query_xy_model[:, 0]  # x
        query_points = torch.from_numpy(query_points_np).to(self.device).unsqueeze(0)  # [1, N, 3]

        with torch.no_grad():
            feature_grids = model.get_feature_grids(frame0, is_training=False)
            self._query_features = model.get_query_features(
                frame0,
                is_training=False,
                query_points=query_points,
                feature_grids=feature_grids,
            )
            causal_state = model.construct_initial_causal_state(
                N, len(self._query_features.resolutions) - 1,
            )
            for i in range(len(causal_state)):
                for k, v in causal_state[i].items():
                    causal_state[i][k] = v.to(self.device)
            self._causal_state = causal_state
        return int(N)

    def estimate_and_advance(
        self,
        current_rgb: Tensor,
        current_depth: Tensor,
        current_camera: Cameras,
        current_mask: Tensor | None = None,
        current_object_mask: Tensor | None = None,  # noqa: ARG002 — KLT-only; TAPIR ignores
    ) -> CoTrackerMotionEstimate:
        identity = np.eye(3, dtype=np.float32)
        zero = np.zeros((3,), dtype=np.float32)
        track_count_before = self.current_track_count
        timings: dict = {}

        # Per-step CUDA-sync diagnostics. Each ``.cpu()`` / ``.numpy()`` /
        # ``.item()`` below stalls on the full GPU queue, so under
        # contention each line is a separate observable cost.
        t_all = time.time()
        t = time.time()
        current_rgb_prepared = CoTrackerMotionEstimator._prepare_tracking_rgb(current_rgb)
        timings["prep_rgb_cpu"] = time.time() - t
        t = time.time()
        current_depth_prepared = CoTrackerMotionEstimator._prepare_depth_image(current_depth)
        timings["prep_depth_cpu"] = time.time() - t
        t = time.time()
        current_intrinsics = CoTrackerMotionEstimator._extract_intrinsics(current_camera)
        timings["prep_intrinsics"] = time.time() - t
        t = time.time()
        current_camera_to_world = CoTrackerMotionEstimator._extract_camera_to_world(current_camera)
        timings["prep_c2w"] = time.time() - t
        timings["input_prep"] = time.time() - t_all

        if not self.ready:
            return CoTrackerMotionEstimate(
                success=False, ready=False, rotation=identity, translation=zero,
                correspondence_count=0, inlier_count=0,
                track_count_before=track_count_before, track_count_after=self.current_track_count,
                raw_visible_count=0, mask_visible_count=0, depth_valid_count=0,
                used_mask_fallback=False, mean_residual=float("inf"), median_residual=float("inf"),
                timings=timings,
            )

        debug_prev_points = self._current_points_xy.copy()
        debug_prev_rgb = self._previous_rgb.clone()

        # --- TAPIR online forward (per-frame, single pass) ---
        t_fwd = time.time()
        model = self._get_model()
        t = time.time()
        # `_preprocess_frame_to_model_input` does .max().item() + interpolate
        # + .to(device); the .item() is a CUDA sync.
        frame_t = self._preprocess_frame_to_model_input(current_rgb_prepared)
        timings["preprocess_frame"] = time.time() - t
        with torch.no_grad():
            t = time.time()
            feature_grids = model.get_feature_grids(frame_t, is_training=False)
            timings["feature_grids"] = time.time() - t
            t = time.time()
            trajectories = model.estimate_trajectories(
                frame_t.shape[-3:-1],
                is_training=False,
                feature_grids=feature_grids,
                query_features=self._query_features,
                query_points_in_video=None,
                query_chunk_size=64,
                causal_context=self._causal_state,
                get_causal_context=True,
            )
            timings["estimate_traj"] = time.time() - t
            self._causal_state = trajectories["causal_context"]
            # TAPIR returns tracks as [B, N, T, 2] and occlusion/expected_dist
            # as [B, N, T] (NOT [B, T, N, ...]).
            tracks = trajectories["tracks"][-1]            # [1, N, T=1, 2]
            occlusion = trajectories["occlusion"][-1]      # [1, N, T=1]
            expected_dist = trajectories["expected_dist"][-1]
            # TAPIR demo: visible iff (1 - sigmoid(occ)) * (1 - sigmoid(dist)) > 0.5
            visibles = (1.0 - torch.sigmoid(occlusion)) * (1.0 - torch.sigmoid(expected_dist)) > self.visibility_threshold

        # Final GPU→CPU sync: stalls on every kernel queued above.
        t = time.time()
        tracks_np_model = tracks[0, :, 0].detach().cpu().numpy().astype(np.float32)  # (N, 2) at 256
        visibility_now = visibles[0, :, 0].detach().cpu().numpy().astype(bool)        # (N,)
        timings["tracks_to_cpu"] = time.time() - t
        current_points_xy = self._xy_model_to_input(tracks_np_model)
        timings["predictor_forward"] = time.time() - t_fwd

        # --- Postprocess (image-bound + mask filter + depth back-projection) ---
        t = time.time()
        current_points_xy, visibility_now = CoTrackerMotionEstimator._filter_points_in_image(
            current_points_xy, visibility_now,
            width=current_depth_prepared.shape[1], height=current_depth_prepared.shape[0],
        )
        raw_visibility = visibility_now.copy()
        raw_visible_count = int(raw_visibility.sum())

        used_mask_fallback = False
        mask_visible_count = raw_visible_count
        if current_mask is not None:
            masked_visibility = CoTrackerMotionEstimator._filter_points_by_mask_array(
                current_points_xy, visibility_now, current_mask,
                output_shape=current_depth_prepared.shape,
            )
            mask_visible_count = int(masked_visibility.sum())
            if mask_visible_count >= self.min_track_points:
                visibility_now = masked_visibility
            else:
                visibility_now = raw_visibility
                used_mask_fallback = current_mask is not None

        correspondence_mask = visibility_now.copy()
        current_depth_values, current_depth_valid = CoTrackerMotionEstimator._sample_depth_bilinear(
            current_depth_prepared, current_points_xy,
        )
        depth_compatible_mask = correspondence_mask & current_depth_valid
        depth_valid_count = int(depth_compatible_mask.sum())
        correspondence_mask = depth_compatible_mask

        prev_world = self._reference_world_points[correspondence_mask]
        curr_world = CoTrackerMotionEstimator._backproject_to_world(
            current_points_xy[correspondence_mask], current_depth_values[correspondence_mask],
            current_intrinsics, current_camera_to_world,
        )
        timings["postprocess"] = time.time() - t

        rotation = identity
        translation = zero
        success = False
        inlier_count = 0
        mean_residual = float("inf")
        median_residual = float("inf")
        track_count_after = int(correspondence_mask.sum())
        tracked_inlier_mask = np.zeros((len(current_points_xy),), dtype=bool)

        t = time.time()
        if len(prev_world) >= self.min_track_points and len(curr_world) >= self.min_track_points:
            # Reuse the cotracker estimator's RANSAC implementation by binding
            # to a transient instance with the same hyperparameters.
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
        timings["ransac_kabsch"] = time.time() - t

        return CoTrackerMotionEstimate(
            success=success, ready=True, rotation=rotation, translation=translation,
            correspondence_count=int(correspondence_mask.sum()), inlier_count=inlier_count,
            track_count_before=track_count_before, track_count_after=track_count_after,
            raw_visible_count=raw_visible_count, mask_visible_count=mask_visible_count,
            depth_valid_count=depth_valid_count, used_mask_fallback=used_mask_fallback,
            mean_residual=mean_residual, median_residual=median_residual,
            previous_points_xy=debug_prev_points, current_points_xy=current_points_xy,
            tracked_inlier_mask=tracked_inlier_mask,
            previous_rgb=debug_prev_rgb, current_rgb=current_rgb_prepared,
            timings=timings,
        )
