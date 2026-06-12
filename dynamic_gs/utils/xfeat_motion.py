"""XFeat sparse + LighterGlue matcher with a multi-anchor keyframe pool.

Fourth backend for dynamic-gs's 2D point tracker, alongside CoTracker,
TAPIR, and KLT. Same public surface as :class:`KltMotionEstimator`:

    * ``initialize(rgb, depth, camera, mask) -> int``
    * ``estimate_and_advance(rgb, depth, camera, mask, current_object_mask) -> CoTrackerMotionEstimate``
    * ``ready``, ``current_track_count`` properties
    * ``last_init_*`` diagnostic fields

Anchor-pool tracking (vs the original pairwise prev-frame design):

* D0 is pinned as the first anchor with ``T_anchor = identity``. The pool grows
  over time: whenever the current estimated rotation is more than
  ``ROTATION_GATE_DEG`` away from every existing anchor, the current frame
  becomes a new anchor.
* Each tick selects the anchor whose stored rotation is closest to the
  predicted rotation (last frame's estimate; identity on the very first call).
  Matching is anchor↔current, so drift is bounded to one anchor-to-current hop
  no matter how far the object has rotated from D0.
* If the nearest anchor yields fewer than ``_MIN_INLIERS_DEFAULT`` Kabsch
  inliers, the 2nd-nearest is tried before declaring tracking lost.
* Composition follows the existing Kabsch convention (source→target): Kabsch
  is called with ``(anchor_world_3d, curr_world_3d)`` so the returned
  ``(R_local, t_local)`` is anchor→current. The canonical (D0→current) pose
  is then ``T_current = T_local · T_anchor`` →
  ``R_current = R_local @ R_anchor``,
  ``t_current = R_local @ t_anchor + t_local``.

Matching backend:

* XFeat (CVPR 2024, https://github.com/verlab/accelerated_features) detects
  *and* describes sparse keypoints in one CNN forward pass.
* By default, the 64-D descriptors are then matched by **LighterGlue** — the
  XFeat-author-trained slim LightGlue variant (6 transformer layers, FlashAttn,
  width pruning). LighterGlue uses positional encoding + cross-attention to
  reject ambiguous matches that MNN would happily keep, which is the entire
  motivation here: MNN over near-identical background descriptors produces
  thousands of "good" but anchor-rotation-confirming pairs that drown the
  object's motion. LighterGlue suppresses those.
* MNN over L2-normalised descriptors is kept as a fallback when LighterGlue
  cannot be constructed (kornia/lightglue missing or weights absent).

Bench targets: sparse extraction at 1280x720 is ~3-5 ms per frame on RTX 5070
Ti (native cu128); LighterGlue matching is ~3-8 ms per pair depending on
keypoint count. Total per-tick budget similar to the old dense+MNN path.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import tracker_common as _tc
from .tracker_common import MotionEstimate as CoTrackerMotionEstimate

if TYPE_CHECKING:
    from nerfstudio.cameras.cameras import Cameras


_XFEAT_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "xfeat")
)


# --- Multi-anchor keyframe pool ---------------------------------------------
# Replaces the pairwise prev-frame cache with a list of widely-separated
# rotational keyframes ("anchors"). Each tick matches against the anchor whose
# rotation is closest to the predicted rotation. Drift is bounded to ONE
# anchor-to-current hop, no matter how far the object has rotated from D0.

# Gate for adding a new anchor: if every existing anchor is more than this many
# degrees away from the current estimated rotation, the current frame becomes a
# new anchor. 22.5° → expect ~16 anchors across a full 360° sweep.
ROTATION_GATE_DEG: float = 22.5

# Apparent-scale gate for anchor keyframing: capture a fresh anchor when the
# camera<->object distance ratio to every existing anchor exceeds this (object
# appears 1/distance in pixels, so 1.3 == the object's on-screen size changed
# ~1.3x, beyond which XFeat/LighterGlue start shedding matches). Distance-
# invariant, unlike an absolute-cm threshold. Complements ROTATION_GATE_DEG:
# an anchor "covers" the current view only if within BOTH gates.
SCALE_GATE_RATIO: float = 1.3

# Fallback threshold for anchor selection: if Kabsch against the nearest
# anchor finds fewer inliers than this, retry against the 2nd-nearest before
# declaring tracking lost. Higher than min_track_points (12) so the fallback
# triggers earlier rather than waiting for full pose loss.
_MIN_INLIERS_DEFAULT: int = 20


@dataclass
class _Anchor:
    """One keyframe in the multi-anchor pose-tracking pool.

    Stores the minimum needed to redo matching + Kabsch against any future
    frame: XFeat descriptors, the corresponding kept-keypoint pixel coords
    on GPU (LighterGlue input), the pre-computed 3D points in WORLD frame
    (captured at this anchor's creation time), and the canonical D0→anchor
    pose. ``keypoints`` is the float32 ndarray copy used by the debug
    visualizer + numpy filters. ``keypoints_gpu`` is the matching path's
    tensor — kept on GPU to avoid an H2D copy per LighterGlue call.
    ``image_size`` is (W, H) — LighterGlue needs it for positional encoding.
    ``rgb`` is the HWC float tensor (0..255 range) used by the pipeline-side
    side-by-side debug visualizer; cheap to cache (~2.6 MB per anchor at
    1280×720) and skipping it would force the visualizer to bail out.
    """
    descriptors: Tensor        # (N, 64) on GPU, L2-normalised
    keypoints_gpu: Tensor      # (N, 2) float32 on GPU — LighterGlue input
    world_3d: np.ndarray       # (N, 3) float32 world-frame points at capture
    keypoints: np.ndarray      # (N, 2) float32 image pixels (debug + filtering)
    image_size: tuple[int, int]  # (W, H) — LighterGlue positional encoding scale
    rotation: np.ndarray       # (3, 3) float32 — D0→anchor rotation (object, world frame)
    translation: np.ndarray    # (3,) float32 — D0→anchor translation
    camera_rotation: np.ndarray  # (3, 3) float32 — R_cam_world of the anchor's
                                 # camera (camera_to_world[:3,:3]). Used to gate /
                                 # select anchors on the object's orientation AS
                                 # SEEN FROM THE CAMERA (relative orientation), so
                                 # a new anchor is captured when the viewpoint of
                                 # the object changes — covering camera-only and
                                 # object-only motion alike.
    camera_distance: float = 0.0  # cam<->object-centroid distance (m) at capture.
                                  # Gates + selects anchors on apparent SCALE so a
                                  # fresh keyframe is captured when the object moves
                                  # nearer/farther (matching degrades with scale).
    rgb: Optional[Tensor] = None      # (H, W, 3) float tensor on GPU, 0..255 range
    mask: Optional[Tensor] = None     # (H, W, 1) bool/float — region used to filter
                                      # this anchor's keypoints (for debug overlay)


def _rotation_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic SO(3) distance between two rotation matrices, in degrees.

    angle = arccos((trace(R_a^T · R_b) − 1) / 2), with arccos input clamped to
    [−1, 1] for numerical safety. Translation is never used.
    """
    cos_angle = (float(np.trace(rotation_a.T @ rotation_b)) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _relative_object_rotation(
    object_rotation_world: np.ndarray, camera_rotation_world: np.ndarray,
) -> np.ndarray:
    """Object orientation AS SEEN FROM THE CAMERA: ``R_cam_world^T @ R_obj_world``.

    Both inputs are world-frame rotations (object D0→current, and the camera's
    ``camera_to_world[:3,:3]``). The result lives in the camera frame, so two
    relative rotations are directly comparable via :func:`_rotation_distance_deg`
    regardless of how the camera moved. This is what XFeat/LighterGlue actually
    match on — appearance from a viewpoint — so anchoring on it captures a new
    keyframe whenever the *view* of the object changes (object moves, camera
    moves, or both)."""
    return camera_rotation_world.T @ object_rotation_world


def _scale_ratio(dist_a: float, dist_b: float) -> float:
    """Symmetric apparent-scale ratio between two camera<->object distances (>=1).
    On-screen object size ~ 1/distance, so this is how much the object's apparent
    size differs between the two viewpoints — what degrades XFeat matching."""
    if not (np.isfinite(dist_a) and np.isfinite(dist_b)) or dist_a <= 1e-6 or dist_b <= 1e-6:
        return 1.0  # unknown distance -> don't let scale veto (fall back to rotation)
    return max(dist_a / dist_b, dist_b / dist_a)


def _select_nearest_anchor_by_rotation(
    anchors: List[_Anchor],
    predicted_relative_rotation: np.ndarray,
    *,
    exclude: Optional[Set[int]] = None,
    current_distance: Optional[float] = None,
    rotation_gate_deg: float = ROTATION_GATE_DEG,
    scale_gate: float = SCALE_GATE_RATIO,
) -> int:
    """Index of the anchor whose stored viewpoint of the object is most similar to
    the current view — closest in BOTH relative (object-in-camera) rotation AND
    apparent scale, so the match runs against a similarly-posed AND similarly-sized
    keyframe.

    Scale mismatch is folded into the rotation distance as an equivalent-degrees
    penalty (a scale ratio of ``scale_gate`` costs ``rotation_gate_deg``), so the
    two trade off on one axis. When ``current_distance`` is None/unknown this
    reduces to nearest-rotation. ``exclude`` skips indices (2nd-nearest retry).
    Returns -1 if every anchor is excluded or the pool is empty.
    """
    excluded = exclude if exclude is not None else set()
    _scale_to_deg = rotation_gate_deg / max(scale_gate - 1.0, 1e-6)
    best_idx, best_cost = -1, float("inf")
    for i, anchor in enumerate(anchors):
        if i in excluded:
            continue
        anchor_rel = _relative_object_rotation(anchor.rotation, anchor.camera_rotation)
        cost = _rotation_distance_deg(predicted_relative_rotation, anchor_rel)
        if current_distance is not None:
            cost += max(0.0, _scale_ratio(current_distance, anchor.camera_distance) - 1.0) * _scale_to_deg
        if cost < best_cost:
            best_cost, best_idx = cost, i
    return best_idx


def _needs_new_anchor(
    anchors: List[_Anchor],
    relative_rotation: np.ndarray,
    current_distance: float,
    rotation_gate_deg: float,
    scale_gate: float,
) -> tuple[bool, float, float]:
    """Decide whether the current view needs a fresh anchor.

    An anchor "covers" the current view iff it is within BOTH the rotation gate
    AND the scale gate; a new anchor is captured when NO anchor covers both — so
    the pool tiles the (orientation x apparent-scale) viewpoint space, not just
    orientation. Returns (need_new, min_rotation_deg, min_scale_ratio) — the two
    minima (taken independently across the pool) are for the log line.
    """
    if not anchors:
        return True, float("inf"), float("inf")
    best_rot = float("inf")
    best_scale = float("inf")
    covered = False
    for a in anchors:
        rot = _rotation_distance_deg(
            relative_rotation, _relative_object_rotation(a.rotation, a.camera_rotation),
        )
        scl = _scale_ratio(current_distance, a.camera_distance)
        best_rot = min(best_rot, rot)
        best_scale = min(best_scale, scl)
        if rot <= rotation_gate_deg and scl <= scale_gate:
            covered = True
    return (not covered), best_rot, best_scale


def _ensure_repo_on_path() -> None:
    """XFeat ships as a repo, not a pip package. Modules import as
    ``from modules.xfeat import XFeat`` so the repo root must be on
    sys.path. Insert once; idempotent."""
    if _XFEAT_REPO not in sys.path:
        sys.path.insert(0, _XFEAT_REPO)


class XFeatMotionEstimator:
    """XFeat sparse matching + RANSAC-Kabsch pairwise rigid motion."""

    def __init__(
        self,
        device: torch.device | str,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
        min_cossim: float = -1.0,
        min_track_points: int = 12,
        ransac_iterations: int = 128,
        ransac_inlier_threshold: float = 0.008,
        weights_path: str = "",
        anchor_min_inliers: int = _MIN_INLIERS_DEFAULT,
        anchor_rotation_gate_deg: float = ROTATION_GATE_DEG,
        anchor_scale_gate: float = SCALE_GATE_RATIO,
        use_lighterglue: bool = True,
        lighterglue_min_conf: float = 0.1,
        lighterglue_depth_confidence: float = 0.95,
        object_search_radius_px: int = 80,
        use_semi_dense: bool = False,
        pose_filter_enabled: bool = True,
        pose_filter_accel_sigma: float = 0.05,
        pose_filter_alpha_sigma: float = 0.25,
        pose_filter_meas_trans_sigma_m: float = 0.003,
        pose_filter_meas_rot_sigma_deg: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.top_k = max(int(top_k), 8)
        self.detection_threshold = float(detection_threshold)
        self.min_cossim = float(min_cossim)
        self.min_track_points = max(int(min_track_points), 3)
        self.ransac_iterations = max(int(ransac_iterations), 1)
        self.ransac_inlier_threshold = float(ransac_inlier_threshold)
        # Anchor-pool knobs. Defaults match the module-level constants but the
        # pipeline can override per-run if a particular scene needs different
        # gating (e.g. tighter gate for fast rotators).
        self._anchor_min_inliers = max(int(anchor_min_inliers), self.min_track_points)
        self._anchor_rotation_gate_deg = float(anchor_rotation_gate_deg)
        # Apparent-scale anchor gate (>1). Object centroid in world at D0, used to
        # measure cam<->object distance each tick (1 matmul + norm); set in initialize.
        self._anchor_scale_gate = max(float(anchor_scale_gate), 1.0 + 1e-3)
        self._centroid_d0: Optional[np.ndarray] = None
        self._lighterglue_min_conf = float(lighterglue_min_conf)
        self._lighterglue_depth_confidence = float(lighterglue_depth_confidence)
        self._object_search_radius_px = max(int(object_search_radius_px), 0)
        self._use_semi_dense = bool(use_semi_dense)

        _ensure_repo_on_path()
        from modules.xfeat import XFeat  # type: ignore

        weights = weights_path.strip() or os.path.join(_XFEAT_REPO, "weights", "xfeat.pt")
        if not os.path.isfile(weights):
            raise RuntimeError(
                f"XFeat weights not found at {weights}. Either place "
                "xfeat.pt under third_party/xfeat/weights/ or set "
                "xfeat_weights_path in DynamicGSModelConfig."
            )
        # XFeat() reads CUDA_VISIBLE_DEVICES at construction. Make sure
        # it sees the GPU; the dynamic-gs runtime never unsets it.
        self._xfeat = XFeat(weights=weights, top_k=self.top_k,
                            detection_threshold=self.detection_threshold)

        # LighterGlue: XFeat-author-trained slim LightGlue (kornia backbone).
        # 6 transformer layers, FlashAttention if available, width-pruning at
        # 0.95. Constructed eagerly so the first tick doesn't pay the load
        # cost; if construction fails we fall back to MNN (logged once).
        self._lighterglue = None
        self._use_lighterglue = bool(use_lighterglue)
        if self._use_lighterglue:
            try:
                from modules.lighterglue import LighterGlue  # type: ignore
                lg_weights = os.path.join(_XFEAT_REPO, "weights", "xfeat-lighterglue.pt")
                self._lighterglue = LighterGlue(weights=lg_weights).to(self._xfeat.dev).eval()
                # Enable layer-depth early-exit. The LighterGlue
                # ``default_conf_xfeat`` ships with ``depth_confidence=-1``
                # (disabled) — the XFeat author left it off for accuracy
                # benchmarks. For live tracking we override at runtime so
                # easy matches skip the deeper transformer layers.
                try:
                    self._lighterglue.net.conf.depth_confidence = (
                        self._lighterglue_depth_confidence
                    )
                except Exception:  # noqa: BLE001
                    pass
                print(
                    f"[xfeat] LighterGlue matcher enabled (weights: {lg_weights}, "
                    f"depth_conf={self._lighterglue_depth_confidence})"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[xfeat] LighterGlue unavailable ({exc}); "
                    "falling back to MNN matching."
                )
                self._lighterglue = None
                self._use_lighterglue = False

        # cuDNN kernel JIT / autotune is paid on the first _extract call
        # at the actual input resolution (see _maybe_warmup). Deferring
        # avoids hardcoding a resolution and re-warming if the camera
        # ever changes.
        self._warmup_hw: Optional[tuple[int, int]] = None

        # Anchor pool. ``_anchors[0]`` is the D0 anchor (set in ``initialize``)
        # with T_anchor = identity. New anchors are appended whenever the
        # current rotation is more than ``_anchor_rotation_gate_deg`` from
        # every existing one.
        self._anchors: List[_Anchor] = []

        # Cumulative (D0 → current) rigid transform = last successful pose
        # estimate, also the predicted rotation for next tick's
        # nearest-anchor selection (constant-pose assumption).
        self._cumulative_R: np.ndarray = np.eye(3, dtype=np.float32)
        self._cumulative_t: np.ndarray = np.zeros((3,), dtype=np.float32)

        # Output-side pose Kalman filter (SE(3) constant-velocity ESKF).
        # Smooths ONLY the (rotation, translation) returned to the
        # pipeline; the raw ``_cumulative_*`` pose stays unfiltered so
        # anchor selection / anchor creation / next-tick prediction are
        # never contaminated by smoothing lag.
        self._pose_filter: Optional[_tc.PoseKalmanFilter] = None
        if pose_filter_enabled:
            self._pose_filter = _tc.PoseKalmanFilter(
                accel_sigma=float(pose_filter_accel_sigma),
                alpha_sigma=float(pose_filter_alpha_sigma),
                meas_trans_sigma=float(pose_filter_meas_trans_sigma_m),
                meas_rot_sigma=float(np.radians(pose_filter_meas_rot_sigma_deg)),
            )

        # Last-tick diagnostics (cleared by each call to estimate_and_advance).
        self.last_anchor_idx_used: int = -1
        self.last_used_fallback_anchor: bool = False
        self.last_inlier_count: int = 0
        self.last_pool_size: int = 0

        # Init diagnostics (mirror KLT's surface so pipeline logging
        # works without branching).
        self.last_init_fast_point_count = 0
        self.last_init_sampled_count = 0
        self.last_init_depth_valid_count = 0
        self.last_init_used_dense_fallback = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the D0 anchor has been seeded with at least
        ``min_track_points`` depth-valid keypoints."""
        return (
            len(self._anchors) > 0
            and self._anchors[0].descriptors.shape[0] >= self.min_track_points
        )

    @property
    def current_track_count(self) -> int:
        """Last frame's inlier count (i.e. how many anchor↔current keypoints
        survived RANSAC). Mirrors the previous "matched keypoints in cache"
        semantics — what the pipeline reads to gauge tracking health."""
        return int(self.last_inlier_count)

    # ------------------------------------------------------------------
    # D0 seed
    # ------------------------------------------------------------------

    def initialize(self, rgb: Tensor, depth: Tensor, camera: "Cameras", mask: Tensor) -> int:
        """Seed the D0 anchor (``T_anchor = identity``) into the pool.

        ``mask`` is the rendered object Gaussian mask captured by the pipeline
        at D0 (see _initialize_motion_estimator) — we use it to RESTRICT D0
        keypoints to the object region. Keeping the D0 descriptor pool
        object-focused is critical for the anchor-pool design: an unfiltered
        D0 would let MNN match thousands of background↔background pairs that
        confirm the identity transform and drown out the object's motion.
        Fallback: if filtering would leave fewer than min_track_points
        keypoints, the filter is skipped (mask probably wrong/empty).

        Only keypoints with valid depth survive into the anchor, since the
        Kabsch step needs 3D points; descriptors-only would be useless.
        """
        self._cumulative_R = np.eye(3, dtype=np.float32)
        self._cumulative_t = np.zeros((3,), dtype=np.float32)
        self._anchors = []
        if self._pose_filter is not None:
            self._pose_filter.reset()

        rgb_t = self._prepare_rgb_gpu(rgb)
        depth_np = _tc.prepare_depth_image(depth)
        intrinsics = _tc.extract_intrinsics(camera)
        camera_to_world = _tc.extract_camera_to_world(camera)
        if depth_np.shape != tuple(rgb_t.shape[:2]):
            raise RuntimeError(
                "XFeat initialization requires RGB and depth at the same resolution, "
                f"got rgb={tuple(rgb_t.shape[:2])} depth={tuple(depth_np.shape)}."
            )

        # Pre-mask the image before XFeat so the top_k keypoint budget is
        # spent on the object surface, not background. ``_pre_mask_image``
        # only zeros out-of-region pixels — it does NOT crop, so the
        # returned ``image_size`` is the full (W, H) of the live frame.
        # That's important: LighterGlue uses ``image_size`` only for
        # positional encoding scaling, and we want it to see the full
        # natural-image coord system on both sides so attention works as
        # trained.
        seed_mask_np = (
            self._mask_to_numpy(mask, rgb_t.shape[:2]) if mask is not None else None
        )
        # D0 seed: extract XFeat keypoints on the FULL image, then keep only
        # those that land inside the object mask. The descriptors stay clean
        # (computed on the full image, same as per-tick descriptors), so
        # LighterGlue matches them correctly without the boundary-descriptor
        # corruption that a masked-image extract would introduce.
        # `_restrict_depth_valid_to_image_mask` later applies depth-validity
        # as a second filter.
        keypoints, descriptors, keypoints_gpu, image_size = self._extract(rgb_t)
        if seed_mask_np is not None and seed_mask_np.any() and keypoints.shape[0] > 0:
            H, W = rgb_t.shape[:2]
            kp_xy_int = np.clip(keypoints.round().astype(np.int32), 0,
                                np.array([W - 1, H - 1], dtype=np.int32))
            mask_bool = seed_mask_np.astype(bool) if seed_mask_np.dtype != bool else seed_mask_np
            inside_obj_mask = mask_bool[kp_xy_int[:, 1], kp_xy_int[:, 0]]
            survivors = np.where(inside_obj_mask)[0]
            if survivors.size >= self.min_track_points:
                keypoints = keypoints[survivors]
                descriptors = descriptors[torch.from_numpy(survivors).to(descriptors.device).long()]
                keypoints_gpu = keypoints_gpu[torch.from_numpy(survivors).to(keypoints_gpu.device).long()]
        if keypoints.shape[0] < self.min_track_points:
            self.last_init_fast_point_count = int(keypoints.shape[0])
            self.last_init_sampled_count = int(keypoints.shape[0])
            self.last_init_depth_valid_count = 0
            self.last_init_used_dense_fallback = False
            return 0

        depth_values, depth_valid = _tc.sample_depth_bilinear(
            depth_np, keypoints,
        )

        # Belt-and-braces: also AND with the non-eroded seed mask in case
        # XFeat squeezed any keypoint into a residual boundary pixel.
        depth_valid = self._restrict_depth_valid_to_image_mask(
            keypoints, depth_valid, seed_mask_np,
        )

        d0_anchor = self._build_anchor(
            keypoints=keypoints, keypoints_gpu=keypoints_gpu,
            descriptors=descriptors, image_size=image_size,
            depth_values=depth_values, depth_valid=depth_valid,
            intrinsics=intrinsics, camera_to_world=camera_to_world,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros((3,), dtype=np.float32),
            rgb=rgb_t, mask=mask,
        )
        if d0_anchor is None or d0_anchor.descriptors.shape[0] < self.min_track_points:
            self.last_init_fast_point_count = int(keypoints.shape[0])
            self.last_init_sampled_count = int(keypoints.shape[0])
            self.last_init_depth_valid_count = int(depth_valid.sum())
            self.last_init_used_dense_fallback = False
            return 0

        self._anchors.append(d0_anchor)
        self.last_pool_size = 1
        # Fixed D0 object centroid in world — the point the scale gate transforms by
        # the cumulative pose each tick to get cam<->object distance. The D0 anchor's
        # own camera_distance already used this centroid, so the pool is consistent.
        self._centroid_d0 = d0_anchor.world_3d.mean(axis=0).astype(np.float32)

        self.last_init_fast_point_count = int(keypoints.shape[0])
        self.last_init_sampled_count = int(keypoints.shape[0])
        self.last_init_depth_valid_count = int(d0_anchor.descriptors.shape[0])
        self.last_init_used_dense_fallback = False
        return int(d0_anchor.descriptors.shape[0])

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
        self.last_anchor_idx_used = -1
        self.last_used_fallback_anchor = False
        self.last_inlier_count = 0

        # --- Sub-timing: input prep ---
        # XFeat lives on GPU, so we keep the RGB on GPU (no .cpu() round-trip,
        # no .max().item() sync). Depth/intrinsics/c2w still go to CPU since
        # bilinear depth sample + RANSAC-Kabsch is pure numpy.
        t = time.time()
        current_rgb_prepared = self._prepare_rgb_gpu(current_rgb)
        current_depth_prepared = _tc.prepare_depth_image(current_depth)
        current_intrinsics = _tc.extract_intrinsics(current_camera)
        current_camera_to_world = _tc.extract_camera_to_world(current_camera)
        timings["input_prep"] = time.time() - t

        if current_depth_prepared.shape != tuple(current_rgb_prepared.shape[:2]):
            raise RuntimeError(
                "XFeat motion estimation requires RGB and depth at the same resolution, "
                f"got rgb={tuple(current_rgb_prepared.shape[:2])} depth={tuple(current_depth_prepared.shape)}."
            )

        if not self.ready:
            # ``initialize`` must run first to seed the D0 anchor; with no
            # anchors there's nothing to match against. Return not-ready so
            # the caller knows to call initialize(D0).
            timings["klt_forward"] = 0.0
            timings["postprocess"] = 0.0
            timings["ransac_kabsch"] = 0.0
            timings["resample"] = 0.0
            return CoTrackerMotionEstimate(
                success=False, ready=False,
                rotation=self._cumulative_R.copy(), translation=self._cumulative_t.copy(),
                correspondence_count=0, inlier_count=0,
                track_count_before=track_count_before, track_count_after=self.current_track_count,
                raw_visible_count=0, mask_visible_count=0, depth_valid_count=0,
                used_mask_fallback=False, mean_residual=float("inf"), median_residual=float("inf"),
                timings=timings,
            )

        # Convert both pipeline masks to numpy once. They're reused for:
        #   - the per-frame match filter (gripper-keep ∩ dilated object mask),
        #   - the debug viz (obj_mask_for_debug, raw rendered object mask).
        gripper_keep_np = (
            self._mask_to_numpy(current_mask, current_rgb_prepared.shape[:2])
            if current_mask is not None else None
        )
        obj_mask_for_debug = (
            self._mask_to_numpy(current_object_mask, current_rgb_prepared.shape[:2])
            if current_object_mask is not None else None
        )

        # Per-frame post-match filter region:
        #   gripper_keep ∩ dilate(rendered_object_mask, search_radius).
        # Restricting current-frame matches to a halo around the object's
        # PREDICTED footprint stops STATIC background/scene features (which
        # survive gripper-keep + depth) from outvoting the object's real
        # motion in RANSAC. This is the grasped+lifted-object failure mode:
        # once the object leaves the table the crop fills with static
        # background, those features match consistently (high inlier count),
        # and the Kabsch majority becomes the background's zero motion — the
        # pose pins to the nearest anchor and the object "stops moving".
        # The mask lags the true object by one tick, hence the dilation halo.
        # We do NOT mask the anchor side (its mask was "where the object WAS"
        # at anchor time). When the object-masked set drops below
        # ``min_track_points`` the keep_region fallback below reverts to
        # gripper-keep-only, so an occluded/tiny mask can't kill tracking.
        keep_region = gripper_keep_np
        if obj_mask_for_debug is not None:
            obj_halo = obj_mask_for_debug
            if obj_halo.ndim == 3:
                obj_halo = obj_halo[..., 0]
            r = self._object_search_radius_px
            if r > 0:
                k = 2 * int(r) + 1
                obj_halo = cv2.dilate(
                    obj_halo.astype(np.uint8),
                    np.ones((k, k), np.uint8), iterations=1,
                ).astype(bool)
            keep_region = obj_halo if keep_region is None else (keep_region & obj_halo)
        current_rgb_for_extract = current_rgb_prepared

        # --- Sub-timing: GPU queue drain (diagnostic) ---
        # The extract below ends in a ``.cpu()`` pull, which is a sync point:
        # its wall time absorbs ALL GPU work enqueued earlier (CDN render, FF
        # inserts, viser render — possibly from other threads on the same
        # default stream). Syncing here first splits that wait out, so
        # ``xfeat_extract`` below measures the extract itself. Moving the sync
        # earlier only re-attributes time; it adds no work.
        t_sync = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["gpu_queue_wait"] = time.time() - t_sync

        # --- Sub-timing: XFeat extract on the current frame ---
        # Reports as ``xfeat_extract`` (sparse XFeat NN forward + .cpu() pull of
        # keypoints). The lumped legacy key ``klt_forward`` is also written so
        # the pipeline's generic predictor-forward slot stays populated for
        # backward compatibility, but ``xfeat_extract`` is the precise number.
        t = time.time()
        curr_keypoints, curr_descriptors, curr_keypoints_gpu, curr_image_size = self._extract(
            current_rgb_for_extract,
        )
        xfeat_extract_time = time.time() - t
        timings["xfeat_extract"] = xfeat_extract_time
        timings["klt_forward"] = xfeat_extract_time
        # Accumulators populated below as the anchor-attempt loop runs.
        timings["lighterglue_match"] = 0.0
        timings["ransac_kabsch"] = 0.0

        if curr_keypoints.shape[0] < self.min_track_points:
            timings["postprocess"] = 0.0
            timings["ransac_kabsch"] = 0.0
            timings["resample"] = 0.0
            return CoTrackerMotionEstimate(
                success=False, ready=True,
                rotation=self._cumulative_R.copy(), translation=self._cumulative_t.copy(),
                correspondence_count=0, inlier_count=0,
                track_count_before=track_count_before, track_count_after=self.current_track_count,
                raw_visible_count=int(curr_keypoints.shape[0]),
                mask_visible_count=0, depth_valid_count=0,
                used_mask_fallback=False, mean_residual=float("inf"), median_residual=float("inf"),
                timings=timings,
            )

        # Backproject ALL of the current keypoints (depth-invalid ones get
        # garbage 3D, masked out below). One pass — cheaper than doing it
        # twice if the 2nd-nearest anchor fallback fires.
        t = time.time()
        curr_depth_values, curr_depth_valid = _tc.sample_depth_bilinear(
            current_depth_prepared, curr_keypoints,
        )
        curr_world_all = _tc.backproject_to_world(
            curr_keypoints, curr_depth_values, current_intrinsics, current_camera_to_world,
        )

        # ``keep_region`` (= gripper_keep ∩ dilate(object_mask, R)) and
        # ``obj_mask_for_debug`` are already computed above the XFeat
        # forward and reused here for the per-frame match filter + the
        # debug viz red border.

        # --- Sub-timing: anchor selection + LighterGlue match + Kabsch ---
        # Try the rotation-nearest anchor first. If it gives fewer inliers
        # than the configured floor, retry against the 2nd-nearest. Keep the
        # better of the two results (more inliers wins).
        # Select on RELATIVE (object-in-camera) orientation: the current
        # object-world pose viewed from the current camera. This picks the
        # anchor whose stored viewpoint of the object is closest to the current
        # one (easiest LighterGlue match), correctly handling camera motion.
        current_cam_R = np.asarray(current_camera_to_world, dtype=np.float32)[:3, :3]
        predicted_R_rel = _relative_object_rotation(self._cumulative_R, current_cam_R)
        excluded: Set[int] = set()
        primary_result: Optional[dict] = None
        primary_anchor_idx: int = -1
        primary_used_mask_fallback = False
        primary_raw_visible_count = 0
        primary_mask_visible_count = 0
        primary_depth_valid_count = 0
        primary_correspondence_count = 0
        primary_curr_xy = np.empty((0, 2), dtype=np.float32)
        primary_anchor_xy = np.empty((0, 2), dtype=np.float32)
        primary_tracked_inlier_mask = np.zeros((0,), dtype=bool)
        primary_anchor_rgb: Optional[Tensor] = None
        primary_anchor_mask: Optional[Tensor] = None

        # Predicted cam<->object distance (object at last pose, camera now). Only
        # used by the scale-aware selection when DGS_XFEAT_SCALE_SELECT=1 — by
        # default selection stays nearest-rotation (the scale GATE already ensures
        # a similarly-scaled anchor exists; a steep scale penalty in selection was
        # measured to add tracking failures without improving the tail).
        _scale_select = os.environ.get("DGS_XFEAT_SCALE_SELECT") == "1"
        predicted_distance = (
            self._current_camera_object_distance(current_camera_to_world)
            if _scale_select else None
        )

        for attempt in range(2):
            anchor_idx = _select_nearest_anchor_by_rotation(
                self._anchors, predicted_R_rel, exclude=excluded,
                current_distance=predicted_distance,
                rotation_gate_deg=self._anchor_rotation_gate_deg,
                scale_gate=self._anchor_scale_gate,
            )
            if anchor_idx < 0:
                break  # no more anchors to try
            anchor = self._anchors[anchor_idx]

            attempt_result = self._match_and_solve_against_anchor(
                anchor=anchor,
                curr_keypoints=curr_keypoints,
                curr_keypoints_gpu=curr_keypoints_gpu,
                curr_descriptors=curr_descriptors,
                curr_image_size=curr_image_size,
                curr_world_all=curr_world_all,
                curr_depth_valid=curr_depth_valid,
                keep_region=keep_region,
                image_hw=current_rgb_prepared.shape[:2],
            )
            # Accumulate per-attempt sub-timings. With one anchor in the
            # pool this is one match + one RANSAC; with the 2nd-nearest
            # fallback it's the sum of both attempts.
            timings["lighterglue_match"] += float(attempt_result.get("match_time", 0.0))
            timings["ransac_kabsch"] += float(attempt_result.get("ransac_time", 0.0))

            inliers = int(attempt_result["ransac"]["inlier_mask"].sum()) if attempt_result["ransac"] else 0

            # Keep this attempt if it beats whatever we have so far.
            if primary_result is None or inliers > int(primary_result["inlier_mask"].sum()):
                primary_result = attempt_result["ransac"]
                primary_anchor_idx = anchor_idx
                primary_used_mask_fallback = attempt_result["used_mask_fallback"]
                primary_raw_visible_count = attempt_result["raw_visible_count"]
                primary_mask_visible_count = attempt_result["mask_visible_count"]
                primary_depth_valid_count = attempt_result["depth_valid_count"]
                primary_correspondence_count = attempt_result["correspondence_count"]
                primary_curr_xy = attempt_result["curr_xy"]
                primary_anchor_xy = attempt_result["anchor_xy"]
                primary_tracked_inlier_mask = attempt_result["tracked_inlier_mask"]
                primary_anchor_rgb = attempt_result["anchor_rgb"]
                primary_anchor_mask = attempt_result["anchor_mask"]
                self.last_used_fallback_anchor = (attempt > 0)

            # Stop if this attempt already cleared the inlier floor.
            if primary_result is not None and inliers >= self._anchor_min_inliers:
                break
            excluded.add(anchor_idx)

        # ``postprocess`` was the legacy CoTracker key — no postprocess stage
        # exists in this path (match + RANSAC happen back-to-back inside each
        # attempt). ``ransac_kabsch`` is already accumulated above.
        timings["postprocess"] = 0.0

        # Compose into the cumulative D0→current pose.
        success = False
        inlier_count = 0
        mean_residual = float("inf")
        median_residual = float("inf")
        if primary_result is not None:
            inlier_count = int(primary_result["inlier_mask"].sum())
            success = inlier_count >= self.min_track_points
            # Relative spike gate: a tick whose inlier count collapses below
            # 45% of the recent-success rolling median is a degenerate match
            # set (measured: spike frames carry 36% of jitter RMS with ~half
            # the inliers). Reject it (hold last pose) instead of applying a
            # bad pose. Relative -> adapts to scene difficulty; a FIXED high
            # threshold (min_track_points=45) killed tracking permanently.
            # DGS_SPIKE_GATE_FRAC=0 disables.
            if success:
                _hist = getattr(self, "_inlier_hist", None)
                if _hist is None:
                    _hist = self._inlier_hist = []
                _frac = float(os.environ.get("DGS_SPIKE_GATE_FRAC", "0"))  # off by default: cost longevity on the fixture
                if _frac > 0 and len(_hist) >= 8:
                    _med = float(np.median(_hist))
                    if inlier_count < _frac * _med:
                        success = False
                # Append on gate-rejected ticks too: a SUSTAINED inlier drop
                # (regime change: motion onset, new viewpoint) must lower the
                # median within ~8 ticks so tracking resumes — otherwise the
                # gate deadlocks at the easy-segment median and rejects
                # forever (measured: permanent death at the motion onset).
                # Transient spikes (1-3 ticks) barely move the median.
                _hist.append(inlier_count)
                if len(_hist) > 30:
                    del _hist[0]
            mean_residual = float(primary_result["mean_residual"])
            median_residual = float(primary_result["median_residual"])
            if success:
                anchor = self._anchors[primary_anchor_idx]
                R_local = primary_result["rotation"]
                t_local = primary_result["translation"]
                # T_current = T_local · T_anchor (matrix order). Existing
                # Kabsch returns source→target; here source=anchor, target=curr.
                new_R = R_local @ anchor.rotation
                new_t = (R_local @ anchor.translation.reshape(3, 1)).flatten() + t_local
                self._cumulative_R = new_R.astype(np.float32)
                self._cumulative_t = new_t.astype(np.float32)
        self.last_anchor_idx_used = primary_anchor_idx
        self.last_inlier_count = inlier_count

        # Output pose: feed the raw RANSAC pose through the Kalman filter
        # (success ticks only — failures produce no new measurement, so
        # the filter just holds its last estimate). Internals above keep
        # using the raw ``_cumulative_*``.
        if self._pose_filter is not None:
            if success:
                # Adaptive trust: spike frames have ~half the inliers
                # (measured: 38 vs 82 mean on the fixture); inflate the
                # measurement noise as inliers drop below the healthy norm.
                _inl = max(int(inlier_count), 1)
                _scale = min(4.0, max(1.0, (80.0 / _inl) ** 0.5))
                # Offline replays process ticks much slower than live (~0.4s
                # vs ~50ms): wall-clock dt makes process noise dominate and
                # the KF barely smooths. DGS_KF_SYNTHETIC_FPS feeds the filter
                # frame-cadence timestamps so offline == live filter behavior.
                _sfps = os.environ.get("DGS_KF_SYNTHETIC_FPS")
                if _sfps:
                    self._kf_tick = getattr(self, "_kf_tick", 0) + 1
                    _ts = self._kf_tick / float(_sfps)
                else:
                    _ts = time.time()
                rotation_out, translation_out = self._pose_filter.filter(
                    self._cumulative_R, self._cumulative_t, _ts,
                    meas_scale=_scale,
                )
            elif self._pose_filter.initialized:
                rotation_out, translation_out = self._pose_filter.current()
            else:
                rotation_out = self._cumulative_R.copy()
                translation_out = self._cumulative_t.copy()
        else:
            rotation_out = self._cumulative_R.copy()
            translation_out = self._cumulative_t.copy()

        # Anchor-creation gate: if the current RELATIVE (object-in-camera)
        # orientation is further than the gate from every existing anchor's
        # relative orientation, create a new one — i.e. capture fresh features
        # whenever the VIEW of the object has changed enough, whether the
        # object rotated, the camera moved, or both. Translation is never used.
        if success:
            # Re-anchor when no existing anchor covers the current view in BOTH
            # relative orientation AND apparent scale (cam<->object distance ratio).
            # The cumulative pose was just updated, so this is the current distance.
            current_distance = self._current_camera_object_distance(current_camera_to_world)
            need_new, min_dist_deg, min_scale_ratio = _needs_new_anchor(
                self._anchors, predicted_R_rel, current_distance,
                self._anchor_rotation_gate_deg, self._anchor_scale_gate,
            )
            if need_new:
                # Filter the new anchor's keypoints to (object ∩ gripper-keep)
                # at the time of creation. We CAN use the rendered object
                # mask here because the cumulative pose was just updated, so
                # the object Gaussians' position is the model's best
                # estimate. Unfiltered anchors flood future matches with
                # background pairs that drown out the object signal.
                anchor_keep_region = (
                    keep_region & obj_mask_for_debug
                    if keep_region is not None and obj_mask_for_debug is not None
                    else (obj_mask_for_debug if keep_region is None else keep_region)
                )
                # Build the anchor from the CURRENT frame's already-extracted
                # FULL-IMAGE keypoints, POST-filtered to the object — the SAME
                # process as the D0 seed. The previous masked-image re-extract
                # (a) corrupted descriptors at the erosion edge, (b) made the
                # anchor's descriptors inconsistent with the per-tick (full-
                # image) descriptors LighterGlue matches them against, and (c)
                # cost an extra ~7 ms XFeat forward. Post-filtering selects
                # keypoints by the mask without mutating the image the CNN sees,
                # so a few px of (rendered-mask) misalignment just trims/adds a
                # couple of edge points instead of corrupting descriptors, and
                # RANSAC drops any background point that slips in.
                anchor_kp = curr_keypoints
                anchor_desc = curr_descriptors
                anchor_kp_gpu = curr_keypoints_gpu
                anchor_image_size = curr_image_size
                if anchor_keep_region is not None and anchor_kp.shape[0] > 0:
                    _H, _W = anchor_keep_region.shape[:2]
                    _xy = np.clip(anchor_kp.round().astype(np.int64), 0,
                                  np.array([_W - 1, _H - 1], dtype=np.int64))
                    _surv = np.where(anchor_keep_region.astype(bool)[_xy[:, 1], _xy[:, 0]])[0]
                    anchor_kp = anchor_kp[_surv]
                    anchor_desc = anchor_desc[torch.from_numpy(_surv).to(anchor_desc.device).long()]
                    anchor_kp_gpu = anchor_kp_gpu[torch.from_numpy(_surv).to(anchor_kp_gpu.device).long()]
                if anchor_kp.shape[0] >= self.min_track_points:
                    anchor_depth_values, anchor_depth_valid = _tc.sample_depth_bilinear(
                        current_depth_prepared, anchor_kp,
                    )
                    # Belt-and-braces filter against the (same) region.
                    anchor_depth_valid = self._restrict_depth_valid_to_image_mask(
                        anchor_kp, anchor_depth_valid, anchor_keep_region,
                    )
                else:
                    anchor_depth_values = np.zeros((0,), dtype=np.float32)
                    anchor_depth_valid = np.zeros((0,), dtype=bool)
                # Save the (object ∩ gripper-keep) region as the anchor's
                # mask so the debug visualizer can draw its boundary.
                anchor_mask_t = (
                    torch.from_numpy(anchor_keep_region.astype(np.float32)).to(self.device)[..., None]
                    if anchor_keep_region is not None
                    else None
                )
                new_anchor = self._build_anchor(
                    keypoints=anchor_kp, keypoints_gpu=anchor_kp_gpu,
                    descriptors=anchor_desc, image_size=anchor_image_size,
                    depth_values=anchor_depth_values, depth_valid=anchor_depth_valid,
                    intrinsics=current_intrinsics, camera_to_world=current_camera_to_world,
                    rotation=self._cumulative_R.copy(),
                    translation=self._cumulative_t.copy(),
                    rgb=current_rgb_prepared, mask=anchor_mask_t,
                )
                if new_anchor is not None and new_anchor.descriptors.shape[0] >= self.min_track_points:
                    self._anchors.append(new_anchor)
                    self.last_pool_size = len(self._anchors)
                    print(
                        f"[xfeat-anchor] pool size: {len(self._anchors)} "
                        f"(added; nearest rot={min_dist_deg:.1f}deg scale={min_scale_ratio:.2f}x "
                        f"dist={current_distance:.2f}m, {new_anchor.descriptors.shape[0]} keypoints)"
                    )
        timings["resample"] = 0.0

        track_count_after = inlier_count if success else 0
        return CoTrackerMotionEstimate(
            success=success, ready=True,
            rotation=rotation_out, translation=translation_out,
            correspondence_count=primary_correspondence_count, inlier_count=inlier_count,
            track_count_before=track_count_before, track_count_after=track_count_after,
            raw_visible_count=primary_raw_visible_count,
            mask_visible_count=primary_mask_visible_count,
            depth_valid_count=primary_depth_valid_count,
            used_mask_fallback=primary_used_mask_fallback,
            mean_residual=mean_residual, median_residual=median_residual,
            previous_points_xy=primary_anchor_xy, current_points_xy=primary_curr_xy,
            tracked_inlier_mask=primary_tracked_inlier_mask,
            previous_rgb=primary_anchor_rgb, current_rgb=current_rgb_prepared,
            previous_mask=primary_anchor_mask,
            current_mask=obj_mask_for_debug,  # rendered object mask (for visualizer red border) — NOT used for filtering
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _extract(self, rgb_hwc: Tensor) -> tuple[np.ndarray, Tensor, Tensor, tuple[int, int]]:
        """Run XFeat detect+describe on an HxWx3 float tensor (0..255 range).

        Uses sparse ``detectAndCompute``: keypoints are sub-pixel-accurate
        corner-like features (NMS over the heatmap, then top-k by score),
        not grid points. Sparse pairs with LighterGlue (the matcher this
        estimator now uses): LighterGlue's attention rejects ambiguous
        background-to-background pairs that MNN would happily keep, so the
        pre-mask trick that motivated the dense path is no longer needed
        to keep the descriptor pool object-focused. Sub-pixel keypoints
        also give mm-scale 3D precision when back-projected through depth,
        so the RANSAC inlier threshold can stay tight (~8 mm).

        Returns:
            keypoints: (N, 2) float32 ndarray (x, y) in input pixel coords.
            descriptors: (N, 64) float32 Tensor on GPU, L2-normalised.
            keypoints_gpu: (N, 2) float32 Tensor on GPU (LighterGlue input).
            image_size: (W, H) tuple — needed by LighterGlue for positional
                encoding.
        """
        if rgb_hwc.ndim == 3:
            inp = rgb_hwc.permute(2, 0, 1).unsqueeze(0).to(self._xfeat.dev, non_blocking=True).float()
        else:
            inp = rgb_hwc.to(self._xfeat.dev, non_blocking=True).float()
        h, w = int(inp.shape[-2]), int(inp.shape[-1])
        self._maybe_warmup((h, w))
        if self._use_semi_dense:
            # detectAndComputeDense returns a SINGLE dict with batched tensors:
            #   keypoints   (B, top_k, 2)   coarse, integer-pixel
            #   descriptors (B, top_k, 64)  L2-normalised
            # (`scales` is also returned with multiscale=True but we don't use it.)
            out = self._xfeat.detectAndComputeDense(inp, top_k=self.top_k, multiscale=True)
            kp_gpu = out["keypoints"][0].detach().float()
            desc = out["descriptors"][0].detach()
        else:
            # detectAndCompute returns a list-per-batch of dicts; sparse path.
            out_list = self._xfeat.detectAndCompute(inp, top_k=self.top_k)
            out = out_list[0]
            kp_gpu = out["keypoints"].detach()
            desc = out["descriptors"].detach()
        kp = kp_gpu.cpu().numpy().astype(np.float32)
        return kp, desc, kp_gpu, (w, h)

    @staticmethod
    def _mnn_match(desc_a: Tensor, desc_b: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Mutual nearest neighbour over L2-normalised 64-D descriptors.

        Fallback path used when LighterGlue is unavailable.
        Returns (idx_a, idx_b) int64 arrays of matched index pairs.
        """
        if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        # cosine similarity = dot since descriptors are unit-norm
        sim = desc_a @ desc_b.T  # (Na, Nb)
        nn_ab = sim.argmax(dim=1)
        nn_ba = sim.argmax(dim=0)
        idx_a = torch.arange(desc_a.shape[0], device=sim.device)
        mutual = nn_ba[nn_ab] == idx_a
        idx_a = idx_a[mutual].cpu().numpy().astype(np.int64)
        idx_b = nn_ab[mutual].cpu().numpy().astype(np.int64)
        return idx_a, idx_b

    @torch.inference_mode()
    def _lighterglue_match(
        self,
        kp_a_gpu: Tensor, desc_a: Tensor, size_a: tuple[int, int],
        kp_b_gpu: Tensor, desc_b: Tensor, size_b: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match two sparse XFeat keypoint sets with LighterGlue.

        Both keypoint tensors must be on the LighterGlue device and in input
        pixel coordinates (LighterGlue does its own per-image normalisation
        from ``image_size``). Returns matched ``(idx_a, idx_b)`` int64
        numpy arrays. Returns empty arrays if either side has too few points
        or LighterGlue produced no matches.
        """
        n_a, n_b = int(kp_a_gpu.shape[0]), int(kp_b_gpu.shape[0])
        if n_a < self.min_track_points or n_b < self.min_track_points or self._lighterglue is None:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        dev = self._lighterglue.dev
        # LighterGlue's internal LightGlue expects (B, N, 2) keypoints and
        # (B, N, D) descriptors plus a (B, 2) image_size as [W, H].
        data = {
            "keypoints0": kp_a_gpu.to(dev, non_blocking=True)[None].float(),
            "keypoints1": kp_b_gpu.to(dev, non_blocking=True)[None].float(),
            "descriptors0": desc_a.to(dev, non_blocking=True)[None].float(),
            "descriptors1": desc_b.to(dev, non_blocking=True)[None].float(),
            "image_size0": torch.tensor(
                [size_a[0], size_a[1]], device=dev, dtype=torch.float32,
            )[None],
            "image_size1": torch.tensor(
                [size_b[0], size_b[1]], device=dev, dtype=torch.float32,
            )[None],
        }
        out = self._lighterglue(data, min_conf=self._lighterglue_min_conf)
        matches_list = out.get("matches", [])
        if not matches_list:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        matches = matches_list[0]
        if matches.numel() == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        m_np = matches.detach().cpu().numpy().astype(np.int64)
        return m_np[:, 0], m_np[:, 1]

    def _current_camera_object_distance(self, camera_to_world: np.ndarray) -> float:
        """Camera<->object-centroid distance (m) for the CURRENT tick: the fixed D0
        centroid pushed through the current cumulative pose, vs the camera position.
        One 3x3 matmul + a norm — the object's position is tracked implicitly by the
        pose, so we never iterate the point cloud. NaN before D0 is seeded."""
        if self._centroid_d0 is None:
            return float("nan")
        obj = (np.asarray(self._cumulative_R, dtype=np.float64) @ np.asarray(self._centroid_d0, dtype=np.float64)
               + np.asarray(self._cumulative_t, dtype=np.float64))
        cam = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
        return float(np.linalg.norm(cam - obj))

    def _build_anchor(
        self,
        *,
        keypoints: np.ndarray,
        keypoints_gpu: Tensor,
        descriptors: Tensor,
        image_size: tuple[int, int],
        depth_values: np.ndarray,
        depth_valid: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        rgb: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Optional[_Anchor]:
        """Filter to depth-valid keypoints, backproject to world, return the
        anchor (or None if no keypoints survive).

        Keypoints without valid depth are dropped here — Kabsch needs 3D
        points, so storing a descriptor without one would just inflate match
        cost. ``keypoints_gpu`` and ``descriptors`` are filtered with the
        same boolean mask so the GPU-side pair stays index-aligned with the
        numpy ``keypoints`` and ``world_3d``. We keep ``rotation``/
        ``translation`` exactly as the caller passed them (the D0→anchor
        canonical pose). ``rgb`` (if provided) is cloned and detached so the
        anchor owns a stable copy independent of whatever pipeline tensor it
        came from.
        """
        if int(depth_valid.sum()) == 0:
            return None
        kept_kp = keypoints[depth_valid].astype(np.float32)
        kept_depth = depth_values[depth_valid]
        depth_valid_t = torch.as_tensor(depth_valid, device=descriptors.device)
        kept_desc = descriptors[depth_valid_t]
        kept_kp_gpu = keypoints_gpu[depth_valid_t.to(keypoints_gpu.device)]
        world_3d = _tc.backproject_to_world(
            kept_kp, kept_depth, intrinsics, camera_to_world,
        )
        # cam<->object-centroid distance at this anchor's viewpoint (scale gate).
        # Use the fixed D0 centroid transformed by this anchor's pose so distances
        # are consistent across anchors; before D0 is set (the D0 anchor itself)
        # fall back to this anchor's own keypoint centroid (== the future D0 centroid).
        _centroid = self._centroid_d0 if self._centroid_d0 is not None else world_3d.mean(axis=0)
        _obj = (np.asarray(rotation, dtype=np.float64) @ np.asarray(_centroid, dtype=np.float64)
                + np.asarray(translation, dtype=np.float64))
        _cam = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
        camera_distance = float(np.linalg.norm(_cam - _obj))
        rgb_stored: Optional[Tensor] = None
        if rgb is not None:
            rgb_stored = rgb.detach().clone()
        mask_stored: Optional[Tensor] = None
        if mask is not None:
            mask_stored = mask.detach().clone() if isinstance(mask, Tensor) else mask.copy()
        return _Anchor(
            descriptors=kept_desc,
            keypoints_gpu=kept_kp_gpu,
            world_3d=world_3d.astype(np.float32),
            keypoints=kept_kp,
            image_size=image_size,
            rotation=rotation.astype(np.float32),
            translation=translation.astype(np.float32),
            camera_rotation=np.asarray(camera_to_world, dtype=np.float32)[:3, :3].copy(),
            camera_distance=camera_distance,
            rgb=rgb_stored,
            mask=mask_stored,
        )

    @staticmethod
    def _pre_mask_image(
        rgb: Tensor, mask: Optional[np.ndarray], erode_px: int = 5,
    ) -> Tensor:
        """Zero-out pixels outside ``mask`` (eroded by ``erode_px``) BEFORE
        feeding to XFeat.

        Used at anchor creation only — gives XFeat's top_k keypoint budget
        all to the object surface instead of wasting it on background.
        Erosion is critical: a sharp black/colour boundary at the mask's
        edge is exactly the kind of feature XFeat's detector loves to fire
        on, producing bogus boundary keypoints. Eroding by 5 px keeps the
        boundary at least that far from any kept pixel, so the detector
        only sees real surface edges.
        """
        if mask is None:
            return rgb
        if erode_px > 0:
            ksize = 2 * erode_px + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        mask_t = torch.from_numpy(mask.astype(np.float32))[..., None].to(rgb.device)
        return (rgb * mask_t).clamp(0, 255)

    def _restrict_depth_valid_to_image_mask(
        self,
        keypoints: np.ndarray,
        depth_valid: np.ndarray,
        image_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return a new ``depth_valid`` array that ALSO requires each keypoint
        to fall inside ``image_mask`` (HxW bool).

        Used at anchor creation time to keep the anchor's descriptor pool
        focused on the object — the previous pairwise design could afford an
        unfiltered prev cache because the per-frame match filter rejected
        background pairs, but for anchor-vs-current matching an unfiltered
        anchor floods MNN with background↔background matches that confirm
        the identity transform and drown out the object motion.

        Fallback: if the masked filter would leave fewer than
        ``min_track_points`` keypoints, returns the original ``depth_valid``
        unchanged. This handles the case where the object mask is small,
        wrongly placed, or unavailable — better to have a too-broad anchor
        than a starved one.
        """
        if image_mask is None:
            return depth_valid
        h, w = image_mask.shape[:2]
        xs = np.clip(np.round(keypoints[:, 0]).astype(np.int64), 0, w - 1)
        ys = np.clip(np.round(keypoints[:, 1]).astype(np.int64), 0, h - 1)
        in_mask = image_mask[ys, xs]
        masked_valid = depth_valid & in_mask
        if int(masked_valid.sum()) >= self.min_track_points:
            return masked_valid
        return depth_valid

    def _compose_keep_region(
        self,
        current_mask: Tensor | None,
        current_object_mask: Tensor | None,
        image_hw: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Combine the gripper-keep + object masks into a single boolean
        ``(H, W)`` array (or None if both are absent). Same recipe as the
        pre-anchor code: object ∩ gripper-keep when both are present."""
        gripper_np = self._mask_to_numpy(current_mask, image_hw) if current_mask is not None else None
        obj_np = self._mask_to_numpy(current_object_mask, image_hw) if current_object_mask is not None else None
        if obj_np is not None and gripper_np is not None:
            return obj_np & gripper_np
        if obj_np is not None:
            return obj_np
        return gripper_np

    def _match_and_solve_against_anchor(
        self,
        *,
        anchor: _Anchor,
        curr_keypoints: np.ndarray,
        curr_keypoints_gpu: Tensor,
        curr_descriptors: Tensor,
        curr_image_size: tuple[int, int],
        curr_world_all: np.ndarray,
        curr_depth_valid: np.ndarray,
        keep_region: Optional[np.ndarray],
        image_hw: tuple[int, int],
    ) -> dict:
        """LighterGlue (or MNN fallback) match + mask/depth filter +
        Kabsch-RANSAC against a single anchor. Returns a dict with the
        RANSAC output (or None) plus the bookkeeping the caller needs to
        fill out ``CoTrackerMotionEstimate``.

        We intentionally backproject the current frame ONCE (in the caller)
        and pass ``curr_world_all`` in — calling this with a 2nd anchor on
        fallback would otherwise repeat the depth sample / backproject.
        """
        height, width = image_hw

        match_t = time.time()
        if self._lighterglue is not None:
            anchor_idx, curr_idx = self._lighterglue_match(
                anchor.keypoints_gpu, anchor.descriptors, anchor.image_size,
                curr_keypoints_gpu, curr_descriptors, curr_image_size,
            )
        else:
            anchor_idx, curr_idx = self._mnn_match(anchor.descriptors, curr_descriptors)
        match_time = time.time() - match_t
        empty_result = {
            "ransac": None, "used_mask_fallback": False,
            "raw_visible_count": 0, "mask_visible_count": 0, "depth_valid_count": 0,
            "correspondence_count": 0,
            "curr_xy": np.empty((0, 2), dtype=np.float32),
            "anchor_xy": np.empty((0, 2), dtype=np.float32),
            "tracked_inlier_mask": np.zeros((0,), dtype=bool),
            "anchor_rgb": anchor.rgb,
            "anchor_mask": anchor.mask,
            "match_time": match_time,
            "ransac_time": 0.0,
        }
        if len(anchor_idx) < self.min_track_points:
            return empty_result

        anchor_xy = anchor.keypoints[anchor_idx].astype(np.float32)
        curr_xy = curr_keypoints[curr_idx].astype(np.float32)

        # In-image filter on current (XFeat clamps to the un-padded canvas
        # but this never hurts).
        in_image = (
            np.isfinite(curr_xy).all(axis=1)
            & (curr_xy[:, 0] >= 0.0) & (curr_xy[:, 0] <= max(width - 1, 0))
            & (curr_xy[:, 1] >= 0.0) & (curr_xy[:, 1] <= max(height - 1, 0))
        )
        valid = in_image
        raw_visible_count = int(valid.sum())

        # Restrict matches to ``keep_region`` on the CURRENT frame —
        # currently ``gripper_keep ∩ dilate(rendered_object_mask, R_px)``,
        # so matches must land within R px of the model's predicted object
        # location. We don't restrict the anchor side: that mask was for
        # "where the object WAS at anchor time" and the object has moved
        # since. Keeping only matches whose current pixel falls in the
        # predicted halo biases Kabsch toward the object's motion (not the
        # background's zero motion).
        used_mask_fallback = False
        mask_visible_count = raw_visible_count
        if keep_region is not None:
            xs = np.clip(np.round(curr_xy[:, 0]).astype(np.int64), 0, keep_region.shape[1] - 1)
            ys = np.clip(np.round(curr_xy[:, 1]).astype(np.int64), 0, keep_region.shape[0] - 1)
            in_region = keep_region[ys, xs]
            masked_valid = valid & in_region
            mask_visible_count = int(masked_valid.sum())
            if mask_visible_count >= self.min_track_points:
                valid = masked_valid
            else:
                used_mask_fallback = True

        # Anchor's 3D is pre-computed (only depth-valid points were stored at
        # anchor creation), so we just need curr depth validity here.
        curr_valid_for_match = curr_depth_valid[curr_idx]
        depth_compatible = valid & curr_valid_for_match
        depth_valid_count = int(depth_compatible.sum())

        if depth_valid_count < self.min_track_points:
            return {
                **empty_result,
                "used_mask_fallback": used_mask_fallback,
                "raw_visible_count": raw_visible_count,
                "mask_visible_count": mask_visible_count,
                "depth_valid_count": depth_valid_count,
                "curr_xy": curr_xy, "anchor_xy": anchor_xy,
            }

        anchor_world_pts = anchor.world_3d[anchor_idx[depth_compatible]]
        curr_world_pts = curr_world_all[curr_idx[depth_compatible]]

        ransac_t = time.time()
        ransac_result = _tc.estimate_rigid_transform_ransac(
            anchor_world_pts, curr_world_pts,
            threshold=self.ransac_inlier_threshold,
            iterations=self.ransac_iterations,
            min_inliers=self.min_track_points,
        )
        ransac_time = time.time() - ransac_t
        # Inlier mask indexed by MATCH (== position in anchor_xy/curr_xy),
        # so the pipeline-side debug visualizer can iterate matches and
        # colour-code green/red. RANSAC's own mask is over the depth_compatible
        # subset of matches — promote it back to the full match index space.
        n_matches = len(anchor_idx)
        tracked_inlier_mask = np.zeros((n_matches,), dtype=bool)
        if ransac_result is not None:
            compatible_match_indices = np.nonzero(depth_compatible)[0]
            tracked_inlier_mask[compatible_match_indices[ransac_result["inlier_mask"]]] = True
        return {
            "ransac": ransac_result,
            "used_mask_fallback": used_mask_fallback,
            "raw_visible_count": raw_visible_count,
            "mask_visible_count": mask_visible_count,
            "depth_valid_count": depth_valid_count,
            "correspondence_count": depth_valid_count,
            "curr_xy": curr_xy, "anchor_xy": anchor_xy,
            "tracked_inlier_mask": tracked_inlier_mask,
            "anchor_rgb": anchor.rgb,
            "anchor_mask": anchor.mask,
            "match_time": match_time,
            "ransac_time": ransac_time,
        }

    def _prepare_rgb_gpu(self, image: Tensor) -> Tensor:
        """GPU-native counterpart of ``_prepare_tracking_rgb``.

        Accepts (H, W, C) or (1, H, W, C) or (1, C, H, W) on any device,
        returns a contiguous HWC float tensor on the XFeat device, range
        normalised to 0..255 without a host sync.

        The critical difference from _tc.prepare_tracking_rgb
        is: NO ``.cpu()`` and NO ``.max().item()`` call. The legacy helper's
        15 ms cost was 99 % CUDA-sync wait, not transfer.
        """
        x = image
        if x.ndim == 4:
            if x.shape[0] == 1 and x.shape[-1] in (3, 4):
                x = x[0]
            elif x.shape[0] == 1 and x.shape[1] in (3, 4):
                x = x[0].permute(1, 2, 0).contiguous()
            else:
                raise ValueError(f"Unexpected 4D RGB shape {tuple(x.shape)}")
        if x.ndim != 3:
            raise ValueError(f"Expected HxWxC RGB tensor, got shape {tuple(x.shape)}")
        if x.shape[-1] > 3:
            x = x[..., :3]
        # Move to XFeat device + dtype without forcing a sync. If already
        # on the right device, .to is a no-op.
        x = x.detach().to(self._xfeat.dev, dtype=torch.float32, non_blocking=True)
        # Contract: live RGB comes through DynamicGSModel.get_live_rgb which
        # normalises by 255 on load, so the input range is always 0..1. Scale
        # to 0..255 unconditionally. No sync (no .max().item() call).
        x = x * 255.0
        return x.clamp(0.0, 255.0).contiguous()

    def _maybe_warmup(self, hw: tuple[int, int]) -> None:
        """Warm cuDNN kernels at the actual input resolution. Idempotent
        per (H, W); re-runs only if the resolution changes. Also warms
        LighterGlue at a representative keypoint count so the first match
        call doesn't pay LightGlue's transformer JIT cost."""
        target = (int(hw[0]), int(hw[1]))
        if self._warmup_hw == target:
            return
        H, W = target
        x = torch.randn(1, 3, H, W, device=self._xfeat.dev).float().mul(80).add(120).clamp(0, 255)
        with torch.inference_mode():
            if self._use_semi_dense:
                for _ in range(2):
                    out_d = self._xfeat.detectAndComputeDense(x, top_k=self.top_k, multiscale=True)
                # Massage into the same (list-of-dict-shaped) shape the LighterGlue
                # warmup branch below expects.
                out_list = [{"keypoints": out_d["keypoints"][0], "descriptors": out_d["descriptors"][0]}]
            else:
                for _ in range(2):
                    out_list = self._xfeat.detectAndCompute(x, top_k=self.top_k)
            if self._lighterglue is not None and out_list and out_list[0]["keypoints"].shape[0] >= 8:
                kp0 = out_list[0]["keypoints"]
                desc0 = out_list[0]["descriptors"]
                size = torch.tensor([W, H], device=self._xfeat.dev)
                data = {
                    "keypoints0": kp0[None], "keypoints1": kp0[None],
                    "descriptors0": desc0[None], "descriptors1": desc0[None],
                    "image_size0": size[None], "image_size1": size[None],
                }
                try:
                    self._lighterglue(data, min_conf=self._lighterglue_min_conf)
                except Exception as exc:  # noqa: BLE001
                    print(f"[xfeat] LighterGlue warmup failed: {exc}")
        torch.cuda.synchronize()
        self._warmup_hw = target

    @staticmethod
    def _mask_to_numpy(mask, output_shape: tuple[int, int]) -> np.ndarray | None:
        if mask is None:
            return None
        if isinstance(mask, Tensor):
            resized = _tc.resize_mask(mask, output_shape)
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
