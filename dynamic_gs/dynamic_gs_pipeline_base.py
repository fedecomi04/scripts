"""Phase 3 base class for the post-static dynamic pipeline.

Architecture (post-rewrite):

  * **static-gs**: trains the scene + runs Phase 0a/0b + writes
    ``post_fusion_state.pt`` at end-of-training. Owns the static phase
    entirely.
  * **dynamic-gs** (= ``RecordedDynamicGSPipeline``, subclass of this
    base): warm-loads the cache, then iterates the recorded dynamic
    dataset frame-by-frame, advancing the XFeat tracker and optionally
    firing the feedforward decoder (rgbd / anysplat). No optimization
    during dynamic phase; the trainer's ``get_train_loss_dict`` returns
    a zero-loss dummy.
  * **dynamic-gs-live** (= ``LiveDynamicGSPipeline``, subclass): same as
    recorded except the frame source is a ROS shared-memory subscriber
    and D0 object selection is by 3D-centroid-to-camera distance
    instead of 2D anchor.

What the rewrite deliberately drops vs the legacy monolith:

  * ``OptimPool`` / ``OptimFrame`` — no dynamic optimization, no need
    for a multi-frame round-robin queue. The feedforward dispatcher
    reads :attr:`_latest_tracker_frame` (single variable, replaced the
    pool).
  * ``DynamicKeyframeFilter`` on the recorded path — every dataset frame
    is fed to the tracker. Live keeps an on-the-fly accept/reject in the
    subclass.
  * ``depth_loss`` / ``rigid_static_loss`` / scene-opt gradient hooks —
    no dynamic-phase loss.
  * ``live: bool`` config field — the split into subclasses replaces it.
  * ``static_num_steps`` / ``static_convergence_*`` / Phase 0a/0b code —
    all moved to ``static-gs``. This pipeline REQUIRES a pre-existing
    ``post_fusion_state.pt`` and fails fast with a clear error if missing.

The 3 abstract hooks the subclass must implement
-------------------------------------------------

``_tracker_tick(step) -> None``
    Pull the next frame (from dataset or from the ROS subscriber),
    advance the XFeat tracker, write the resulting frame metadata onto
    :attr:`_latest_tracker_frame`, and force a viewer re-render.

``_pick_d0_object(camera, prefused_instance_ids) -> int``
    Choose which prefused instance to track from the D0 camera. Recorded:
    2D anchor (W/2, 0.75H) closest to a prefused centroid + CD0
    validation. Live: 3D-centroid closest to camera position.

``_on_tracker_frame(camera, batch, cdn, is_first) -> None``
    Post-tick callback. Subclass decides what to do with the captured
    tracker output (e.g. live runs the feedforward dispatcher per Mode B
    cadence; recorded writes anchor-pose video frames).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image

from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils.rich_utils import CONSOLE

from .change_detection import ChangeMaskConfig, compute_change_mask
from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig
from .persistence import load_post_fusion_state, save_post_fusion_state
from .utils import dilate_binary_mask


# ============================================================================
# Config
# ============================================================================


@dataclass
class DynamicGSPipelineBaseConfig(VanillaPipelineConfig):
    """Shared config for both recorded and live dynamic pipelines.

    The dropped legacy fields:

    * ``live: bool`` — the subclass IS the split.
    * ``static_num_steps``, ``static_convergence_*`` — static-gs owns it.
    * ``enable_dynamic_keyframe_filter``, ``dynamic_keyframe_*`` —
      recorded feeds every dataset frame; live filters in the subclass.
    * ``optim_pool_*``, ``disable_dynamic_optimization`` — no dynamic
      optimization, no pool.
    * ``enable_scene_optimization``, ``scene_opt_*`` — same reason.
    * ``depth_lambda``, ``rigid_*`` — no dynamic-phase loss.
    * ``feedforward_reuse_static_checkpoint`` — the warm cache is now
      load-required, not load-optional.
    * ``save_live_optim_debug*`` — optim is gone.
    """

    datamanager: DynamicGSDataManagerConfig = field(
        default_factory=DynamicGSDataManagerConfig
    )
    model: DynamicGSModelConfig = field(default_factory=DynamicGSModelConfig)

    # ---- Cache load (required) ----
    post_fusion_cache_subpath: str = "static_scene/static_state.pt"
    """Where to look for the warm-start cache written by ``static-gs`` or
    ``static-gs-preseg``, relative to ``datamanager.data``. The pipeline
    raises ``FileNotFoundError`` on construction if the file is missing —
    the dynamic pipeline does NOT do its own static training.

    Backward compat: the loader falls back to ``post_fusion_state.pt`` if
    the configured path is missing, so existing snapshots still warm-start."""

    save_final_snapshot: bool = True
    """If True, write a final ``post_dynamic_state.pt`` snapshot at exit so
    the post-feedforward scene can be visualized via
    ``scripts/dump_post_fusion_to_ply.py`` /
    ``scripts/dump_post_fusion_to_viser.py``. Same format as the warm-cache
    that ``static-gs`` writes (model.state_dict + num_points)."""
    final_snapshot_subpath: str = "dynamic_scene/post_dynamic_state.pt"
    """Where to write the final snapshot, relative to ``datamanager.data``."""

    # ---- Dynamic-phase pacing ----
    dynamic_steps_per_frame: int = 1
    """Trainer steps per tracker frame. For recorded mode this gates how
    often ``_tracker_tick`` advances to the next dataset frame. Live mode
    ignores it (the ROS supply is the rate limiter). With no dynamic-phase
    optimization, the trainer's per-step cost is just the tick + maybe FF;
    leaving this at 1 minimizes wall-clock overhead per frame."""

    save_debug_images: bool = False
    """Per-tick PNG dumps + motion logs. Off by default in the rewrite —
    they were ~600 ms / recorded frame in the legacy monolith and aren't
    needed for the new tracking-only + feedforward flow."""

    # ---- Viser-direct visualization ----
    enable_viser_direct: bool = True
    """Spin up a standalone viser server and push per-object rigid
    transforms + new feedforward splats each tracker tick. Browser does
    WebGL splatting so the training GPU never renders for the viewer.
    Use with ``--vis tensorboard`` so Nerfstudio's render_state_machine
    stays out of the way."""
    viser_direct_port: int = 8081

    # ---- Interactive object selection (default OFF) ----
    interactive_object_selection: bool = False
    """When True, the operator picks the tracked object via a viser GUI panel
    (the SAM3 input image with each object's mask overlaid + numbered, a
    button-group of object ids, and a Done button) instead of the heuristic.
    A persistent "Change object" button reopens the picker mid-run so the
    operator can switch objects after finishing one. Default False keeps
    headless/CI runs on the existing ``d0_force_instance_id`` / anchor-centroid
    heuristic. The picked id == SAM3 mask number + 1 == ``object_instance_ids``
    (true in both static-gs and static-gs-preseg as of the 2026-06-10 preseg
    id-order fix)."""
    object_selection_jpeg_quality: int = 85
    """JPEG quality for the overlay image pushed into the picker panel."""
    object_selection_timeout_s: float = 120.0
    """If ``interactive_object_selection`` is True but no Done click arrives
    within this many seconds (e.g. no viser client connected), fall back to
    ``d0_force_instance_id`` / the heuristic and log it. Checked on the trainer
    thread via a wall-clock comparison — never blocks."""

    d0_force_instance_id: Optional[int] = None
    """Force D0 to track this exact prefused ``object_instance_id``, bypassing
    the per-subclass anchor/centroid heuristic. Shared by both subclasses and
    used as the headless/timeout fallback when interactive selection is on."""

    # ---- Feedforward dispatcher (rgbd or anysplat) ----
    enable_feedforward_inpaint: Literal["off", "rgbd_decode", "anysplat_decode"] = "anysplat_decode"
    """``rgbd_decode`` back-projects sensor depth into per-pixel Gaussians
    (single GPU env, sub-ms). ``anysplat_decode`` calls the AnySplat
    subprocess worker (separate conda env, ~12 s/call) for multi-view
    feedforward Gaussian prediction + Umeyama alignment. Both write
    Gaussians with ``object_flags=1, object_instance_ids=999,
    inserted_flags=1`` so neither the rigid-transform pass nor any future
    scene-opt hooks touch them."""

    feedforward_oneshot_step: int = 0
    """Mode A: fire once at this dynamic-phase step. 0 disables."""
    feedforward_recurring_every_n_ticks: int = 6
    """Mode B cadence (>0 enables; 0 disables). Every Nth tracker tick,
    fire the dispatcher on the current frame."""
    feedforward_recurring_min_gap_s: float = 0.3
    """Wall-clock floor between consecutive FF firings, on top of the
    tick-count cadence. 0 disables (cadence-only)."""
    feedforward_anysplat_min_gap_s: float = 0.5
    """Same as above but for anysplat_decode. Overrides the rgbd floor
    when the active mode is anysplat."""

    feedforward_top_n_components: int = 3
    """Mode A: at most this many CDN components per call, area-sorted."""
    feedforward_dominant_area_ratio: float = 0.3
    """Mode A: drop top-N entries below this fraction of the largest
    component's area. Mode B sets this to 0.0."""

    feedforward_anchor_frame: Optional[int] = None
    """Recorded-mode video anchor frame (None = last dataset frame)."""
    feedforward_video_out: Optional[Path] = None
    feedforward_video_fps: int = 24

    feedforward_rgbd_opacity: float = 0.99
    feedforward_rgbd_min_valid_fraction: float = 0.95
    feedforward_rgbd_normal_smoothing_radius: int = 3
    feedforward_rgbd_leak_threshold_m: float = 0.01
    feedforward_rgbd_cliff_threshold_m: float = 0.05
    feedforward_rgbd_post_cliff_erode_px: int = 1
    """Pixels of erosion applied after the depth-cliff filter on the
    decoded component."""
    feedforward_rgbd_scale_multiplier: float = 1.0

    # ---- Feedforward dispatcher knobs (closed-loop additive mode) ----
    feedforward_skip_delete: bool = True
    """When True (closed-loop default), keep all prior Gaussians and just
    additively insert. The next-tick CDN will be near zero wherever the
    insert was correct, so the loop self-stabilizes without ever dropping
    scene content. Set False to restore the legacy delete-in-region path."""
    feedforward_cull_in_front: bool = True
    """Drop Gaussians sitting BETWEEN the camera and the real sensor
    surface in the component's footprint. Without this, leftover artifact
    Gaussians keep occluding the true geometry and re-triggering CDN
    every tick. Only touches instance_ids in {0, 999}; tracked objects
    are never culled."""
    feedforward_cull_in_front_depth_tol_m: float = 0.002
    """Per-Gaussian depth tolerance for the cull-in-front filter."""
    feedforward_cull_before_decode: bool = True
    """Run ONE in-front cull over the whole changed (CDN) region BEFORE the
    decoder, then recompute CDN on the freshly-culled scene. If the cull
    alone clears the change (cleaned CDN has no components), the decoder /
    AnySplat is skipped entirely — culling a stale occluder is often the
    whole fix, so this saves a decode/AnySplat call. The culled set is the
    same as the per-component cull (instance_ids in {0, 999}: both original
    point-cloud AND previously-inserted FF Gaussians; tracked objects never).
    When True, the redundant per-component in-front cull inside the decode
    loop is skipped (this pass already covered the CDN union)."""
    feedforward_object_mask_dilate_px: int = 2
    """Dilation applied to the rendered object mask before subtracting it
    from CDN in :meth:`_feedforward_clean_cdn`."""
    feedforward_object_mask_scale: float = 1.02
    """Enlarge the subtracted object footprint by this factor about its OWN
    centroid (1.02 = +2%) before subtracting from CDN. Unlike the fixed-px
    dilation this scales with object size, so it swallows the thin
    'misplacement ring' between the rendered tracked object and the live
    object (a few px of residual where tracking is slightly off) — without it
    the CDN flags that ring as change and the FF tries to insert a flat copy
    of the object onto the tracked 3D object. 1.0 disables."""

    feedforward_anysplat_conda_env: str = "anysplat_dynamic_gs"
    feedforward_anysplat_worker_timeout_s: float = 300.0
    feedforward_anysplat_min_opacity: float = 0.05
    feedforward_anysplat_scale_multiplier: float = 2.0
    """Multiplicative enlargement applied to the three per-axis scales of each
    AnySplat gaussian after world-frame reprojection (the reproject step already
    preserves image-space footprint). Tuned by eye on the recorded screwdriver
    scene: RAW (1.0) scales at full density render gritty / not smooth; 2.0
    closes the gaps. Density is the other half of the trade — the old
    thinned-then-2x combination (voxel dedup on) looked blurry and made
    change-detection re-flag the inserted regions; full density + 2.0 is the
    current setting."""

    feedforward_anysplat_min_visible_scene_points: int = 1000
    """If the per-call frustum cull keeps fewer than this many scene Gaussians,
    skip the entire FF call (no ICP, no AnySplat reproject, no insert). Triggers
    when the camera turns away from the captured scene — in that regime AnySplat
    would back-project whatever random surface the camera is now pointed at
    into a region with no scene context, polluting the model with floaters that
    have no spatial relation to the existing reconstruction. Set to 0 to disable."""
    feedforward_anysplat_voxel_dedup_m: float = 0.0
    """Voxel size (metres) for the NEAR dedup pass (points within
    ``feedforward_anysplat_dedup_near_radius_m`` of the current camera position).
    ``0.0`` (default) disables the near pass; the far pass has its own knob and
    BOTH must be 0 for truly RAW insertion. With both 0, AnySplat gaussians are
    inserted at full density. The old 2 mm default thinned the insert, which
    (with the 2x scale enlargement) degraded the rendered region enough that
    change-detection re-flagged it every call -> cull/reinsert churn. Set to
    0.002 to restore the TSDF-matched dedup if insert counts become a problem."""
    feedforward_anysplat_voxel_dedup_far_m: float = 0.0
    """Voxel size (metres) for the FAR dedup pass (points beyond
    ``feedforward_anysplat_dedup_near_radius_m``). ``0.0`` (default) disables
    the far pass — see ``feedforward_anysplat_voxel_dedup_m``; both must be 0
    for truly RAW insertion (the dedup block runs if EITHER is > 0). Set to
    0.010 to restore the coarse background compression."""
    feedforward_anysplat_dedup_near_radius_m: float = 0.5
    """Splits AnySplat output into NEAR (||xyz - cam|| <= radius) and FAR
    (> radius). NEAR uses the fine voxel, FAR uses the coarse voxel."""
    feedforward_anysplat_icp_refine: bool = True
    """Run point-to-plane ICP on the scene_c2w pose against the visible subset of
    the scene cloud before reprojecting AnySplat. Target is frustum-culled in
    :meth:`_anysplat_bg_run` so ICP only sees points the camera could actually
    observe — bounds cost AND avoids occluded-point bias."""
    feedforward_anysplat_icp_max_iters: int = 30
    feedforward_anysplat_icp_max_dist_m: float = 0.02


# ============================================================================
# Pipeline base
# ============================================================================


# Type alias documenting what ``_latest_tracker_frame`` carries. Replaces
# the legacy ``OptimFrame`` (which also carried per-frame optimization
# state — epochs_used, initial_loss, last_loss — that the rewrite doesn't
# need because there's no dynamic-phase optimization).
TrackerFrame = Any
"""Concrete shape: ``{
    "frame_idx": int,
    "camera": Cameras,
    "cdn": Tensor | None,         # change-detection mask, computed lazily
    "batch": dict,                # rgb, depth_image, mask (live or dataset batch)
    "stamp_sec": float | None,    # live-only; ROS frame stamp for dedup
}``. Subclasses build this from their respective frame sources.
"""


class DynamicGSPipelineBase(VanillaPipeline):
    """Shared foundation for ``RecordedDynamicGSPipeline`` and
    ``LiveDynamicGSPipeline``. See module docstring for architecture."""

    config: DynamicGSPipelineBaseConfig

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def __init__(
        self,
        config: DynamicGSPipelineBaseConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        # State that subclasses may read in their own __init__ overrides
        # BEFORE super().__init__ runs.
        self._timing: defaultdict[str, list] = defaultdict(list)
        # Fresh timing ledger for this teleop session (the static run already
        # rendered its own to timing_report_static.txt).
        try:
            from .utils import timing_ledger as _tl
            _tl.reset(config.datamanager.data)
        except Exception:
            pass
        self._motion_estimator = None
        self._latest_tracker_frame: Optional[TrackerFrame] = None
        self._global_frame_counter: int = 0
        self._dynamic_step_counter: int = 0
        self._tracker_tick_count: int = 0
        self._feedforward_call_counter: int = 0
        self._feedforward_oneshot_done: bool = False
        self._last_feedforward_wall_time: float = 0.0
        # Per-tick FF-fire decision, set by the subclass tick and reused by
        # _on_tracker_frame so the gate is evaluated exactly once (the CDN
        # render is gated on the same flag).
        self._ff_due_this_tick: bool = False
        self._d0_selected_instance_id: int = 0
        # cudnn benchmark mode OFF for the dynamic phase. ns-train sets
        # torch.backends.cudnn.benchmark=True globally (nerfstudio/scripts/
        # train.py:71), which makes cudnn run an exhaustive conv-algorithm
        # autotune for every NEW input shape — and the XFeat crop
        # (_crop_for_xfeat) presents a new H×W almost every tick. Measured on
        # new_env (192 frames, viser on + client, FF rgbd): DN.3c_xfeat_extract
        # avg 754.4 ms / max 5711.7 ms with benchmark=True vs avg 14.3 ms /
        # max 29.7 ms with it off (53×) — this was the visible per-tick object
        # freeze. Benchmark mode only pays off with FIXED shapes (static
        # training); the dynamic phase is shape-varying, so it's strictly
        # harmful here. gsplat (custom CUDA) and LighterGlue (matmul/cublas)
        # are unaffected by this flag.
        torch.backends.cudnn.benchmark = False
        # Interactive object-picker state machine (non-blocking; the viser
        # GUI callbacks run on viser's thread pool and only set these flags,
        # while the trainer-thread tick polls them). All inert unless
        # ``config.interactive_object_selection`` is True.
        self._selection_state: str = "IDLE"  # "IDLE" | "AWAITING_SELECTION"
        self._pending_selected_id: Optional[int] = None
        self._selection_event = threading.Event()
        self._selection_request_t: float = 0.0
        self._selection_gui_folder = None       # GuiFolderHandle (remove() clears)
        self._selection_dropdown = None         # GuiDropdownHandle (current pick)
        self._change_object_button = None       # persistent GuiButtonHandle
        self._reselect_requested: bool = False  # set by Change-object / stdin Enter
        self._initial_selection_done: bool = False  # first pick happened?
        self._sam3_objects = None               # cached (overlay_rgb, [ObjEntry])
        self._anysplat_persistent_worker = None
        # Per-tick cached tracked-object mask (see _render_object_mask_cached):
        # rendered once per tick, reused by CDN/FF/debug. None = needs render.
        self._obj_mask_cache = None
        self._feedforward_video_writer = None
        self._viser_direct_server = None
        self._viser_direct_handles_built: bool = False
        # Trainer back-ref populated by ``get_training_callbacks``; read by
        # ``_force_viewer_rerender`` so it must default to None so the
        # rerender hook is safe to call before training starts.
        self._trainer = None
        # Last successful XFeat motion estimate; read by
        # ``_push_viser_direct_transforms``. None until D1 succeeds.
        self._last_motion_estimate = None
        # Per-tick rolling diagnostics for the ``[tracker-rate]`` log
        # window (mean inliers / correspondences per window).
        self._last_inlier_count: int = 0
        self._last_correspondence_count: int = 0
        self._inlier_window: list[int] = []
        self._corr_window: list[int] = []
        # Recorded subclass populates this with kept dataset indices so
        # the FF anysplat path can resolve context frames. Empty in live.
        self._accepted_dynamic_frames: list[int] = []

        # Off-thread AnySplat dispatch: this lock is held while a bg
        # AnySplat call (worker.inference + reproject + cull + insert) is
        # in flight. Tracker ticks that find it locked skip the FF dispatch
        # so we never queue overlapping calls. _cleanup_anysplat_bg waits
        # for it to drain at shutdown.
        self._anysplat_slot_lock = threading.Lock()
        # Re-entrant model lock — held by every site that mutates model
        # state (rigid transform, FF cull, FF insert, capture_reference_*)
        # AND by the viser render thread around every get_outputs. Lives
        # here (not on the viser server) so it works even when viser is
        # disabled — otherwise the off-thread FF bg can race the main-
        # thread render and produce torn (means.shape != quats.shape)
        # snapshots. RLock so a single thread can re-enter (FF cull +
        # FF insert both grab it inside the bg run).
        self._model_lock = threading.RLock()

        atexit.register(self._cleanup_viser_direct)
        atexit.register(self._cleanup_anysplat_worker)
        atexit.register(self._cleanup_anysplat_bg)  # registered after worker -> runs BEFORE worker (LIFO)
        atexit.register(self._cleanup_anysplat_ipc_file)
        atexit.register(self._cleanup_feedforward_video_writer)
        atexit.register(self._save_final_snapshot_if_enabled)
        atexit.register(self._write_timing_report)
        self._final_snapshot_written: bool = False
        self._timing_report_written: bool = False

        # Build datamanager + model (cold model with SfM-seed Gaussians).
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

        # Share the model lock with the model so its
        # ``get_outputs_for_camera`` — which the Nerfstudio viewer's render
        # thread calls — joins the same exclusion zone as the FF bg insert
        # thread. Without this hook, the viser-direct lock-swap (line ~709)
        # protects only the viser-direct path, not the NS viewer path; the
        # NS viewer's render would still race mid-insert Parameter resizes
        # and CUDA-assert. Must be AFTER super().__init__() because that's
        # what builds ``self.model``.
        if hasattr(self.model, "attach_render_lock"):
            self.model.attach_render_lock(self._viser_lock_ctx)

        # Load the static-gs warm cache. This REPLACES the SfM seed
        # Gaussians with the post-fusion ones (~300k SfM + ~200k SAM3D
        # inserted), restores the four identity buffers, and skips the
        # entire Phase 0a/0b path.
        self._load_warm_cache_or_die()

        # Viser-direct setup (browser-side splatting).
        if config.enable_viser_direct:
            self._setup_viser_direct()

        # Eager-spawn the AnySplat persistent worker if AnySplat FF is on.
        # The worker loads the model ONCE inside its subprocess; every
        # subsequent _run_feedforward_anysplat() call routes through
        # worker.inference() (already wired in the dispatcher). Without
        # this hook, the fallback path (run_anysplat_subprocess) spins
        # up a fresh subprocess + reloads the model on every FF call,
        # which is the ~11s/call cost we measured.
        if str(getattr(config, "enable_feedforward_inpaint", "off")) == "anysplat_decode":
            self._start_anysplat_persistent_worker()

    def _load_warm_cache_or_die(self) -> None:
        cache_path = Path(self.config.datamanager.data) / self.config.post_fusion_cache_subpath
        if not cache_path.is_file():
            # Backward compat: accept the legacy filename written by older
            # static-gs runs. (The loader itself already has this fallback,
            # but the pre-flight existence check fires before the loader.)
            legacy = cache_path.with_name("post_fusion_state.pt")
            if legacy.is_file():
                CONSOLE.log(
                    f"[dynamic-gs] {cache_path.name} missing; using legacy "
                    f"{legacy.name} (rename it to silence)."
                )
                cache_path = legacy
            else:
                raise FileNotFoundError(
                    f"\n\n[dynamic-gs] Required warm-start cache not found:\n"
                    f"  {cache_path}\n"
                    f"  (also checked legacy name: {legacy})\n\n"
                    f"The dynamic-gs pipeline is now a stage-2 trainer: it does NOT\n"
                    f"do its own static training. Run a static method first:\n\n"
                    f"  ns-train static-gs --data {self.config.datamanager.data}\n"
                    f"  ns-train static-gs-preseg --data {self.config.datamanager.data}\n"
                )
        _t_ld = time.time()
        result = load_post_fusion_state(self.model, cache_path, self.device)
        try:
            from .utils import timing_ledger as _tl
            _tl.record(self.config.datamanager.data, "teleop_init", "static_state.pt",
                       "io", _t_ld, time.time())
        except Exception:
            pass
        if not result.success:
            raise RuntimeError(
                f"[dynamic-gs] post-fusion cache load failed: {result.error}\n"
                f"Delete {cache_path} and re-run static-gs to regenerate."
            )

        # Bypass Splatfacto's resolution + SH-degree schedules. The trainer
        # starts at step=0, but the warm-cache scene has already been trained
        # past the schedule. Without this offset, ``_get_downscale_factor()``
        # returns 4 for steps < 100 and 2 for steps < 200, so:
        #   * rendered RGB + depth come back at 1/4 or 1/2 grid size
        #   * camera.fx/fy/cx/cy stay at full resolution (intrinsics are the
        #     dataset's, not the model's)
        #   * decode_component_to_gaussians + _feedforward_cull_in_front
        #     back-project with full-res intrinsics on the downscaled grid →
        #     FF inserts land in completely wrong world locations.
        # Setting _step_offset = 10_000 forces all schedules to report
        # past-warmup; ``self.step`` becomes ``trainer_step + 10_000`` via
        # ``DynamicGSModel.step_cb``.
        try:
            self.model._step_offset = 10_000
            CONSOLE.log(
                f"[dynamic-gs] forced model._step_offset=10000 to bypass "
                f"Splatfacto's resolution+SH schedules (warm-cache scene is "
                f"already past warmup)."
            )
        except Exception as exc:
            CONSOLE.log(
                f"[dynamic-gs] WARNING: could not set _step_offset on model: "
                f"{exc}. Render+depth may be downscaled, inserts may land "
                f"in wrong locations until step >= ~200."
            )

        CONSOLE.log(
            f"[dynamic-gs] warm-cache loaded: {result.num_points} Gaussians "
            f"({int(self.model.object_instance_ids.gt(0).any(dim=-1).sum().item())} "
            f"with non-zero instance_id)"
        )

    # ---- Cleanup ----

    def _cleanup_viser_direct(self) -> None:
        srv = getattr(self, "_viser_direct_server", None)
        if srv is not None:
            # Force a final FF-handle flush so the post-run scene shows
            # every FF insert that arrived inside the last coalesce
            # window. Safe to call even if there's nothing pending.
            try:
                if getattr(self, "model", None) is not None:
                    srv.flush_pending_ff(self.model)
            except Exception:
                pass
            try:
                srv.close()
            except Exception:
                pass
            self._viser_direct_server = None

    def _cleanup_anysplat_worker(self) -> None:
        w = getattr(self, "_anysplat_persistent_worker", None)
        if w is not None:
            try:
                w.close()
            except Exception:
                pass
            self._anysplat_persistent_worker = None

    def _cleanup_anysplat_ipc_file(self) -> None:
        """Remove the /dev/shm IPC file written by the AnySplat worker.
        Best-effort; the file is small (a few MB) but living in tmpfs."""
        try:
            ipc_path = Path(f"/dev/shm/anysplat_ipc_{os.getpid()}.npz")
            if ipc_path.exists():
                ipc_path.unlink()
        except Exception:
            pass

    def _cleanup_anysplat_bg(self) -> None:
        """Block on the AnySplat bg slot at shutdown so the worker isn't
        killed mid-call (this hook is registered AFTER the worker cleanup
        and atexit runs in LIFO -> bg drain happens first). 60 s timeout
        in case a call wedged."""
        lock = getattr(self, "_anysplat_slot_lock", None)
        if lock is None:
            return
        acquired = lock.acquire(blocking=True, timeout=60.0)
        if acquired:
            lock.release()

    def _save_final_snapshot_if_enabled(self) -> None:
        """Dump the post-feedforward model state to disk at exit.

        Same format as ``static-gs``'s ``post_fusion_state.pt``, so the
        existing ``scripts/dump_post_fusion_to_ply.py`` and
        ``scripts/dump_post_fusion_to_viser.py`` work on it directly.
        Idempotent — only fires once even if atexit runs multiple times.
        """
        if self._final_snapshot_written:
            return
        if not getattr(self.config, "save_final_snapshot", True):
            return
        try:
            cache_path = (
                Path(self.config.datamanager.data) / self.config.final_snapshot_subpath
            )
            ok = save_post_fusion_state(self.model, cache_path)
            self._final_snapshot_written = True
            if ok:
                CONSOLE.log(
                    f"[dynamic-gs] final snapshot written → {cache_path} "
                    f"(N={int(self.model.num_points)}, "
                    f"object_flags={int(self.model.object_flags.sum().item())}, "
                    f"inserted_flags={int(self.model.inserted_flags.sum().item())})"
                )
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs] final snapshot save failed: {exc}")

    def _cleanup_feedforward_video_writer(self) -> None:
        w = getattr(self, "_feedforward_video_writer", None)
        if w is not None:
            try:
                w.release()
            except Exception:
                pass
            self._feedforward_video_writer = None

    def _write_timing_report(self) -> None:
        """Write ``<data_root>/timing_report.txt`` from ``self._timing``.

        Idempotent — atexit may fire twice on Ctrl+C; the
        ``_timing_report_written`` flag prevents a duplicate write.
        Covers FF.*, DN.*, S0.*, static_step, dynamic_step, and any
        other key collected in ``self._timing``.
        """
        if getattr(self, "_timing_report_written", False):
            return
        # If __init__ failed early, datamanager may not exist; abort.
        try:
            datamanager = object.__getattribute__(self, "datamanager")
        except AttributeError:
            return
        if datamanager is None or not hasattr(datamanager, "config"):
            return
        timing = getattr(self, "_timing", None)
        if not timing:
            return
        self._timing_report_written = True

        from datetime import datetime
        from pathlib import Path

        def _avg_ms(vals):
            if not vals:
                return 0.0
            return float(sum(vals)) / float(len(vals)) * 1000.0

        def _row(key: str, vals) -> str:
            if not vals:
                return f"  {key:<42s}        N/A"
            n = len(vals)
            avg = _avg_ms(vals)
            total = float(sum(vals))
            mn = min(vals) * 1000.0
            mx = max(vals) * 1000.0
            return (
                f"  {key:<42s} n={n:<6d} avg={avg:>8.1f}ms "
                f"min={mn:>7.1f}ms max={mx:>8.1f}ms total={total:>7.1f}s"
            )

        lines: list[str] = []
        lines.append("=" * 96)
        lines.append("DYNAMIC-GS-V2 TIMING REPORT")
        lines.append("=" * 96)
        lines.append(f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Data root:   {datamanager.config.data}")
        ff_mode = str(getattr(self.config, "enable_feedforward_inpaint", "off"))
        ff_n = int(getattr(self.config, "feedforward_recurring_every_n_ticks", 0) or 0)
        lines.append(f"FF mode:     {ff_mode} (every {ff_n} ticks)")
        lines.append(f"Viser:       {'on (port ' + str(getattr(self.config, 'viser_direct_port', '?')) + ')' if getattr(self.config, 'enable_viser_direct', False) else 'off'}")
        # Effective rates from wall-clock span of the dynamic tick loop (n-1
        # intervals over the first→last tick stamp). Tracker Hz = ticks/s; FF Hz
        # = AnySplat-dispatch calls/s. "N/A" if the span wasn't captured.
        _t0 = getattr(self, "_dyn_first_tick_wall", None)
        _t1 = getattr(self, "_dyn_last_tick_wall", None)
        _n_ticks = len(timing.get("DN.3_estimate_total", []))
        _n_ff = len(timing.get("FF.1_cdn_clean", []))
        if _t0 is not None and _t1 is not None and _t1 > _t0 and _n_ticks > 1:
            _span = _t1 - _t0
            _trk_hz = (_n_ticks - 1) / _span
            _ff_hz = _n_ff / _span
            lines.append(f"Tracker:     {_trk_hz:.1f} Hz  ({_n_ticks} ticks over {_span:.1f}s)")
            lines.append(f"FF:          {_ff_hz:.2f} Hz  ({_n_ff} calls over {_span:.1f}s)")
        else:
            lines.append("Tracker:     N/A (no tick-span captured)")
            lines.append("FF:          N/A")
        lines.append("")

        # By-phase bulleted load/inference report (teleop_init loads from the
        # ledger + the recurring per-tick algos folded in under dynamic_runtime,
        # all in the same n/avg/min/max row format).
        try:
            from .utils import timing_ledger as _tl
            _runtime_keys = [
                ("XFeat extract", "DN.3c_xfeat_extract"),
                ("LighterGlue match", "DN.3i_lighterglue_match"),
                ("RANSAC+Kabsch", "DN.3e_ransac_kabsch"),
                ("tracker estimate (total)", "DN.3_estimate_total"),
                ("CDN render", "DN.2_cdn_render"),
                ("apply rigid transform", "DN.3g_apply_transform"),
            ]
            _extra_dyn = [(label, list(timing[k])) for (label, k) in _runtime_keys
                          if k in timing and timing[k]]
            # Fold any feedforward-decode keys (FF.*) as algos too.
            for k in sorted(timing.keys()):
                if k.startswith("FF.") and timing[k]:
                    _extra_dyn.append((k, list(timing[k])))
            lines.append(_tl.render(datamanager.config.data,
                                    extra={"dynamic_runtime": _extra_dyn}))
            lines.append("")
            lines.append("DETAIL (per-substep, ms):")
        except Exception as _exc:
            lines.append(f"(timing-ledger render failed: {_exc})")
            lines.append("")

        # Group keys by prefix for readability.
        all_keys = sorted(timing.keys())
        groups: dict[str, list[str]] = {}
        for k in all_keys:
            prefix = k.split(".", 1)[0] if "." in k else k.split("_", 1)[0]
            groups.setdefault(prefix, []).append(k)

        for group_name in sorted(groups.keys()):
            group_keys = groups[group_name]
            lines.append(f"--- {group_name} ({len(group_keys)} key{'s' if len(group_keys)!=1 else ''}) ---")
            for k in group_keys:
                lines.append(_row(k, timing[k]))
            lines.append("")

        # AnySplat persistent worker sanity line: did the model load once?
        worker = getattr(self, "_anysplat_persistent_worker", None)
        if worker is not None:
            try:
                load_s = float(getattr(worker, "load_seconds", 0.0))
                lines.append(f"AnySplat persistent worker: load={load_s:.1f}s "
                             f"(then per-call inference only)")
            except Exception:
                pass
        elif ff_mode == "anysplat_decode":
            lines.append("AnySplat persistent worker: NOT SPAWNED (each FF call "
                         "reloaded the model via run_anysplat_subprocess — slow path)")
        lines.append("")

        out_path = Path(datamanager.config.data) / "timing_report.txt"
        try:
            out_path.write_text("\n".join(lines) + "\n")
            CONSOLE.log(f"[dynamic-gs] timing report written: {out_path}")
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs] timing report write failed: {exc}")

    # ====================================================================
    # Trainer entry — Nerfstudio calls this every step
    # ====================================================================

    def _recurring_ff_due(self, tick_count: int, is_first: bool) -> bool:
        """Whether recurring (Mode B) feedforward fires for the given tick count.

        PURE — no side effects — so the tick can decide UP FRONT whether to
        spend a CDN render. The per-tick CDN is consumed ONLY by feedforward, so
        on non-FF ticks the render is skipped entirely and the tracker thread
        stays undisturbed. Pass the tick count AS THE FF GATE WILL SEE IT (i.e.
        post-this-tick increment) — :meth:`_on_tracker_frame` runs after the
        subclass increments ``_tracker_tick_count``, so the tick passes
        ``_tracker_tick_count + 1`` and the hook passes ``_tracker_tick_count``.
        """
        if is_first:
            return False
        if str(self.config.enable_feedforward_inpaint) == "off":
            return False
        N = int(self.config.feedforward_recurring_every_n_ticks)
        if N <= 0 or (tick_count % N) != 0:
            return False
        gap = (
            self.config.feedforward_anysplat_min_gap_s
            if str(self.config.enable_feedforward_inpaint) == "anysplat_decode"
            else self.config.feedforward_recurring_min_gap_s
        )
        if gap > 0 and (time.time() - self._last_feedforward_wall_time) < gap:
            return False
        return True

    def _oneshot_ff_due(self, step: int) -> bool:
        """Whether the Mode A one-shot feedforward fires this step (also needs
        the per-tick CDN present)."""
        return (
            int(self.config.feedforward_oneshot_step) > 0
            and step >= int(self.config.feedforward_oneshot_step)
            and not self._feedforward_oneshot_done
        )

    def get_train_loss_dict(self, step: int):
        """Dynamic-only entry point.

        Each call from the trainer is a dynamic-phase step. Each step:

        1. Run the subclass tick (advance frame source + apply XFeat).
        2. Maybe fire feedforward (Mode A oneshot, or Mode B via subclass).
        3. Return a zero-loss dummy.
        """
        self._dynamic_step_counter += 1
        self._tracker_tick(step)
        if (
            self.config.feedforward_oneshot_step > 0
            and step >= self.config.feedforward_oneshot_step
            and not self._feedforward_oneshot_done
            and self._latest_tracker_frame is not None
        ):
            self._run_feedforward(self._latest_tracker_frame, mode_label="oneshot")
            self._feedforward_oneshot_done = True
        zero = torch.zeros((), device=self.device, requires_grad=True)
        return {}, {"main_loss": zero}, {}

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes):
        """Stash trainer back-ref so the pipeline can force viewer re-renders.

        The legacy monolith additionally set ``trainer.config.max_num_iterations``
        and ``trainer.config.logging.steps_per_log`` here. Both moves to
        subclasses (recorded → static_num_steps + N*frames; live → 1e9).
        Subclasses override and call super().
        """
        callbacks = super().get_training_callbacks(training_callback_attributes)
        self._trainer = training_callback_attributes.trainer
        return callbacks

    # ====================================================================
    # Abstract hooks — subclasses MUST implement
    # ====================================================================

    def _tracker_tick(self, step: int) -> None:
        """Advance the frame source by one tick, apply XFeat to update
        object Gaussian pose, write the result to
        :attr:`_latest_tracker_frame`, and push a viewer re-render."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _tracker_tick"
        )

    def _pick_d0_object(
        self,
        camera,
        prefused_instance_ids: torch.Tensor,
    ) -> int:
        """Select which prefused instance (1..K from Phase 0b) to track."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _pick_d0_object"
        )

    def _on_tracker_frame(
        self,
        camera,
        batch: dict,
        cdn: Optional[torch.Tensor],
        is_first: bool,
    ) -> None:
        """Post-tick callback fired by :meth:`_tracker_tick`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _on_tracker_frame"
        )

    # ====================================================================
    # Object (re)selection — shared by D0 first-pick + interactive switching
    # ====================================================================

    def _reset_d0_guard(self) -> None:
        """Reset the per-subclass 'D0 already happened' guard so the next tick
        treats the (re)selected object as a fresh D0. Base default resets the
        tick counter; live overrides to also flip ``_d0_completed``."""
        self._tracker_tick_count = 0

    @torch.no_grad()
    def _reseed_tracked_object(self, new_id: int, camera, batch) -> bool:
        """Switch the tracked object to ``new_id``. Returns True on success.

        Single entry point for BOTH the D0 first-pick and every interactive
        re-selection. Performs the full reset surface, then re-runs steps 2-5
        of the D0 bootstrap (object_flags + reference pose + object mask +
        XFeat anchor seed) for the new instance. ``object_instance_ids`` is the
        stable per-Gaussian identity; ``inserted_flags`` (FF, id 999) and
        ``sam3d_init_target_flags`` are intentionally untouched. Object A simply
        freezes at its last applied pose once B becomes the active object.

        On a failed XFeat seed the D0 guard is NOT reset, so the caller (live
        defer loop) can retry on a later frame instead of marking D0 done."""
        new_id = int(new_id)
        # --- reset surface (pipeline side) ---
        self._motion_estimator = None
        self._last_motion_estimate = None
        self._viser_direct_handles_built = False
        self._d0_selected_instance_id = new_id

        ids_buf = self.model.object_instance_ids
        ids = ids_buf.squeeze(-1) if ids_buf.ndim > 1 else ids_buf
        # --- step 2 (object_flags) + step 4 (reference pose) under the lock ---
        with self._viser_lock_ctx():
            self.model.object_flags.copy_((ids == new_id).float().unsqueeze(-1))
            if hasattr(self.model, "capture_reference_object_pose"):
                self.model.capture_reference_object_pose(instance_id=new_id)
        n_flagged = int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())
        # Tracked instance changed → drop the cached mask so step 3 (and the
        # rest of this tick) re-renders for the new object.
        self._invalidate_object_mask_cache()

        # --- step 3 (object mask) ---
        try:
            obj_mask = self._render_object_mask_cached(camera)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs] reseed render_object_mask failed: {exc}")
            obj_mask = None

        # --- step 5 (XFeat anchor seed on the NEW object) ---
        live_rgb = self._build_tracking_rgb(batch)
        depth = batch.get("depth_image")
        try:
            self._initialize_motion_estimator(live_rgb, depth, camera, obj_mask)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs] reseed _initialize_motion_estimator failed: {exc}")
            return False

        self._reset_d0_guard()
        CONSOLE.log(
            f"[dynamic-gs] reseeded tracked object -> instance_id={new_id} "
            f"({n_flagged} Gaussians flagged)"
        )
        return True

    # ---- Interactive picker (viser GUI, non-blocking state machine) ----

    def _present_object_ids(self) -> set:
        """Set of prefused object ids currently present in the buffer (>0)."""
        ids = self.model.object_instance_ids
        ids = ids.squeeze(-1) if ids.ndim > 1 else ids
        return {int(i) for i in torch.unique(ids[ids > 0]).tolist()}

    def _selection_fallback_id(self, camera) -> int:
        """Headless/timeout fallback: forced id if valid, else the heuristic."""
        forced = getattr(self.config, "d0_force_instance_id", None)
        if forced is not None and int(forced) in self._present_object_ids():
            return int(forced)
        try:
            return int(self._pick_d0_object(camera, self.model.object_instance_ids))
        except Exception:
            return 0

    def _open_picker_panel(self, camera, batch) -> None:
        """Build the viser picker GUI (SAM3 overlay + id button-group + Done).

        Non-blocking: returns immediately after constructing the panel; the
        trainer-thread tick polls :meth:`_poll_selection`. If no viser server
        is up, route straight to the fallback (caller reseeds + stays IDLE)."""
        import time as _time
        srv = getattr(self, "_viser_direct_server", None)
        if srv is None or getattr(srv, "server", None) is None:
            # No viser → immediate fallback (no AWAITING state).
            fb = self._selection_fallback_id(camera)
            CONSOLE.log(
                f"[picker] no viser client surface; falling back to "
                f"instance_id={fb}"
            )
            if fb > 0:
                self._reseed_tracked_object(fb, camera, batch)
            self._selection_state = "IDLE"
            return
        server = srv.server

        # Load + cache the SAM3 objects/overlay once per process.
        if self._sam3_objects is None:
            try:
                from .utils.object_picker import (
                    load_sam3_objects,
                    render_picker_overlay,
                )
                data_dir = Path(self.config.datamanager.data)
                loaded = load_sam3_objects(data_dir)
                if loaded is None:
                    raise RuntimeError("no SAM3 artifacts on disk")
                img, entries = loaded
                overlay = render_picker_overlay(img, entries)
                self._sam3_objects = (overlay, entries)
            except Exception as exc:
                fb = self._selection_fallback_id(camera)
                CONSOLE.log(
                    f"[picker] could not load SAM3 objects ({exc}); falling "
                    f"back to instance_id={fb}"
                )
                if fb > 0:
                    self._reseed_tracked_object(fb, camera, batch)
                self._selection_state = "IDLE"
                return

        overlay, entries = self._sam3_objects
        present = self._present_object_ids()
        # Offer only ids that exist in the buffer (intersection with SAM3 list).
        options = [str(e.instance_id) for e in entries
                   if e.instance_id in present] or [str(i) for i in sorted(present)]

        # Tear down any prior panel folder.
        if self._selection_gui_folder is not None:
            try:
                self._selection_gui_folder.remove()
            except Exception:
                pass
            self._selection_gui_folder = None
        self._selection_event.clear()
        self._pending_selected_id = None

        try:
            with server.gui.add_folder("Pick object to track") as folder:
                server.gui.add_markdown(
                    "Pick the object's number from the dropdown "
                    "(matches the labels on the image), then click **Done**."
                )
                server.gui.add_image(
                    overlay, label="SAM3 objects", format="jpeg",
                    jpeg_quality=int(self.config.object_selection_jpeg_quality),
                )
                # Dropdown (NOT button_group): a dropdown has a PERSISTENT,
                # single-select ``.value`` that updates on click — a
                # button_group is momentary and its ``.value`` doesn't track
                # the click, so Done would always read the default.
                dd = server.gui.add_dropdown("Object id", options)
                done = server.gui.add_button("Done", color="green")
            self._selection_gui_folder = folder
            self._selection_dropdown = dd
        except Exception as exc:
            fb = self._selection_fallback_id(camera)
            CONSOLE.log(f"[picker] GUI build failed ({exc}); falling back to {fb}")
            if fb > 0:
                self._reseed_tracked_object(fb, camera, batch)
            self._selection_state = "IDLE"
            return

        @done.on_click
        def _on_done(_evt) -> None:
            # viser thread pool — only set flags. Ignore unless awaiting.
            if self._selection_state != "AWAITING_SELECTION":
                return
            try:
                self._pending_selected_id = int(dd.value)
            except Exception:
                self._pending_selected_id = None
            CONSOLE.log(
                f"[picker] Done clicked -> selected id={self._pending_selected_id}"
            )
            self._selection_event.set()

        self._selection_state = "AWAITING_SELECTION"
        self._selection_request_t = _time.time()
        CONSOLE.log(
            f"[picker] awaiting object selection in viser "
            f"(options={options}); pick from the dropdown + click Done"
        )

    def _close_picker_panel(self) -> None:
        if self._selection_gui_folder is not None:
            try:
                self._selection_gui_folder.remove()
            except Exception:
                pass
            self._selection_gui_folder = None
        self._selection_event.clear()
        self._pending_selected_id = None

    def _ensure_change_object_button(self) -> None:
        """Add the persistent 'Change object' button once (reopens the picker)."""
        srv = getattr(self, "_viser_direct_server", None)
        if srv is None or getattr(srv, "server", None) is None:
            return
        if self._change_object_button is not None:
            return
        try:
            btn = srv.server.gui.add_button("Change object")

            @btn.on_click
            def _on_change(_evt) -> None:
                # viser thread pool — flag only; the tick reopens the panel.
                self._reselect_requested = True

            self._change_object_button = btn
        except Exception as exc:
            CONSOLE.log(f"[picker] could not add Change-object button: {exc}")

    def _tick_interactive_selection(self, camera, batch, is_first: bool) -> str:
        """Drive the interactive picker for one tick. Shared by both subclasses.

        Returns one of:
          * ``"seeded"`` — a selection was applied via reseed (the caller must
            SKIP its own D0/DN dispatch — the object is already seeded — but may
            still run CDN/publish so viser updates).
          * ``"none"``   — no picker active this tick; proceed normally
            (heuristic D0 or DN).

        Opens the picker for the FIRST selection (``not _initial_selection_done``)
        or whenever a ``Change object`` / stdin re-select was requested, then
        **blocks this tick** until the operator clicks Done (or the wall-clock
        timeout fires → fallback id). Blocking is safe: the viser render thread
        is independent (keeps the browser live) and the Done callback runs on
        viser's thread pool. We block deliberately so the trainer's step counter
        does NOT advance toward ``max_num_iterations`` while waiting — otherwise
        a recorded run blows through all its steps in seconds and the panel
        vanishes before the operator can click.

        Note: gating on ``_initial_selection_done`` rather than ``is_first`` is
        deliberate — ``_reseed_tracked_object`` resets the D0 guard
        (``_tracker_tick_count = 0``), which would make ``is_first`` True again
        on the very next tick and re-open the picker in a loop."""
        need_open = self._selection_state == "IDLE" and (
            not self._initial_selection_done or self._reselect_requested
        )
        if not need_open and self._selection_state != "AWAITING_SELECTION":
            return "none"

        if need_open:
            self._reselect_requested = False
            self._open_picker_panel(camera, batch)
            if self._selection_state != "AWAITING_SELECTION":
                # _open_picker_panel hit the no-viser / no-artifacts fallback
                # and already reseeded inline.
                self._initial_selection_done = True
                return "seeded"

        # Block (bounded) until the operator selects or the timeout fires.
        chosen = self._wait_for_selection(camera)
        if chosen is not None and int(chosen) > 0:
            self._reseed_tracked_object(int(chosen), camera, batch)
        self._selection_state = "IDLE"
        self._initial_selection_done = True
        return "seeded"

    def _wait_for_selection(self, camera) -> Optional[int]:
        """Block the trainer thread until the operator clicks Done (returns the
        chosen id) or ``object_selection_timeout_s`` elapses (returns the
        fallback id, or None if none). Polls the Event in 0.25 s slices so a
        ``stop`` / process kill is still responsive."""
        import time as _time
        timeout = float(getattr(self.config, "object_selection_timeout_s", 120.0))
        deadline = (self._selection_request_t + timeout) if timeout > 0 else None
        while True:
            if getattr(self, "_live_stop_requested", False):
                self._close_picker_panel()
                return None
            if self._selection_event.wait(timeout=0.25):
                chosen = self._pending_selected_id
                if chosen is not None and int(chosen) in self._present_object_ids():
                    self._close_picker_panel()
                    return int(chosen)
                # Bad/stale pick — re-arm and keep the panel open.
                self._selection_event.clear()
                self._pending_selected_id = None
                continue
            if deadline is not None and _time.time() > deadline:
                fb = self._selection_fallback_id(camera)
                CONSOLE.log(
                    f"[picker] selection timed out after {timeout:.0f}s; "
                    f"falling back to instance_id={fb}"
                )
                self._close_picker_panel()
                return fb if fb > 0 else None

    # ====================================================================
    # Viser-direct visualization (Path A — hybrid, browser-side splatting)
    # ====================================================================

    def _viser_lock_ctx(self):
        """Return the pipeline-owned model lock as a context manager.

        Acquired around every model-mutation site (rigid transform, FF
        insert, FF cull, capture_reference_object_pose) AND by the viser
        render thread inside every ``get_outputs`` call. Without this,
        the off-thread FF bg can reassign ``gauss_params["means"]`` /
        ``["quats"]`` mid-render and the rasterizer sees a torn (N != M)
        tensor pair — the error surfaces as
        ``render for CDN failed: torch.Size([..., 4])`` on the main
        tracker tick AND ``cannot register a hook on a tensor that
        doesn't require gradient`` on the bg insert path.

        Lives on the pipeline (not on the viser server) so it's always
        held regardless of whether viser-direct is enabled. RLock so a
        single thread can re-enter (FF cull + FF insert both grab it).
        """
        return self._model_lock

    def _setup_viser_direct(self) -> None:
        """Spin up the standalone viser server + attach the live model.

        Server-side rasterize + push-image pattern: the viser
        ``ViserDirectScene`` runs a background render thread that, for
        each connected client, polls the client's camera, calls
        ``model.get_outputs`` server-side, and pushes the resulting
        RGB frame via ``client.scene.set_background_image``. We attach
        the model here (right after warm-cache load) so the render loop
        has live state from the moment the first client connects.

        Legacy "build splat handles at D0" path is gone — the prior
        :meth:`_build_viser_direct_handles` call survives as a thin
        legacy stub that just records the D0 camera pose (so newly
        connecting clients land on the dataset frame instead of an
        arbitrary default).
        """
        from .utils.viser_direct import ViserDirectScene
        self._viser_direct_server = None
        self._viser_direct_handles_built = False
        try:
            self._viser_direct_server = ViserDirectScene(
                port=int(self.config.viser_direct_port),
            )
            # Replace viser's internal lock with the pipeline-owned one so
            # the render thread shares the same RLock as the FF bg thread
            # and the tracker tick (rather than racing on two separate
            # locks).
            self._viser_direct_server.model_lock = self._model_lock
            self._viser_direct_server.attach_model(self.model, device=self.device)
            CONSOLE.log(
                f"[viser-direct] server up on port {self.config.viser_direct_port} "
                f"+ model attached "
                f"— open http://localhost:{self.config.viser_direct_port}"
            )
            if getattr(self.config, "interactive_object_selection", False):
                self._ensure_change_object_button()
        except Exception as exc:
            CONSOLE.log(
                f"[viser-direct] failed to start: {exc} — "
                f"falling back to viewer pipeline"
            )
            self._viser_direct_server = None

    def _build_viser_direct_handles(self, camera) -> None:
        """Build the (static, tracked) splat handles ONCE at D0."""
        if self._viser_direct_server is None:
            return
        if self._viser_direct_handles_built:
            return
        try:
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :4] = (
                camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)
            )
            self._viser_direct_server.setup_handles(
                self.model,
                tracked_instance_id=getattr(self, "_d0_selected_instance_id", None),
                initial_c2w=c2w_4x4,
            )
            self._viser_direct_handles_built = True
        except Exception as exc:
            import traceback as _tb
            CONSOLE.log(
                f"[viser-direct] setup_handles failed ({type(exc).__name__}): "
                f"{exc!r}\n{_tb.format_exc()}"
            )
            self._viser_direct_server = None

    def _push_viser_direct_transforms(self) -> None:
        """Push the latest world-frame rigid (R, t) of the tracked object
        to the viser server. Called by the subclass each DN tick after
        :meth:`_apply_motion_estimator` succeeds."""
        if self._viser_direct_server is None:
            return
        est = getattr(self, "_last_motion_estimate", None)
        if est is None or not getattr(est, "success", False):
            return
        try:
            self._viser_direct_server.push_tracker_transform(
                est.rotation, est.translation
            )
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] push failed: {exc}")

    def _push_viser_camera_feed(self, camera, batch) -> None:
        """Push the current tracked frame's RGB (side-panel feed thumbnail) and
        its camera c2w (for the 'Follow tracked frame' toggle) to viser. Shared
        by recorded + live ticks; cheap reference swaps — the JPEG encode /
        camera snap happen render-side only when a client is connected."""
        srv = self._viser_direct_server
        if srv is None:
            return
        # --- live RGB for the feed thumbnail (uint8 HWC RGB) ---
        try:
            rgb = batch.get("image")
            if rgb is not None:
                import torch as _torch
                if isinstance(rgb, _torch.Tensor):
                    arr = rgb.detach()
                    if arr.dtype != _torch.uint8:
                        arr = (arr.clamp(0.0, 1.0) * 255.0).to(_torch.uint8)
                    rgb_np = arr.cpu().numpy()
                else:
                    rgb_np = np.asarray(rgb)
                    if rgb_np.dtype != np.uint8:
                        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                if rgb_np.ndim == 3 and rgb_np.shape[2] >= 3:
                    srv.update_camera_feed(np.ascontiguousarray(rgb_np[:, :, :3]))
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] camera-feed push failed: {exc}")
        # --- tracked frame camera c2w for the follow-pose toggle ---
        try:
            c2w = camera.camera_to_worlds
            c2w = c2w[0] if c2w.ndim == 3 else c2w
            srv.update_tracked_camera(c2w.detach().cpu().numpy())
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] tracked-camera push failed: {exc}")

    def _viser_direct_register_ff_insert(self, inserted_ids) -> None:
        """Trigger one server-side render after a FF insert. The actual
        FF gaussians are already in ``model`` (insert_inpaint_gaussians
        ran under model_lock just before this is called); we only need
        to wake the render thread so the browser sees them on the next
        push. The legacy ``add_ff_insert_chunk`` call is preserved as a
        no-op for backwards compat with the now-deleted Path A handle
        accounting."""
        if self._viser_direct_server is None:
            return
        # Skip the push if the server is tearing down — a late bg-thread call
        # here would submit onto a shutting-down executor and raise
        # "cannot schedule new futures after shutdown" (which previously aborted
        # the FF video finalize at "Training Finished").
        if getattr(self._viser_direct_server, "is_closing", False):
            return
        try:
            self._viser_direct_server.add_ff_insert_chunk(self.model, inserted_ids)
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] add_ff_insert_chunk failed: {exc}")
        try:
            self._viser_direct_server.request_render()
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] request_render failed: {exc}")

    def _refresh_viser_direct_after_feedforward(self) -> None:
        """Re-upload the static splat handle after an FF call."""
        if self._viser_direct_server is None:
            return
        try:
            self._viser_direct_server.refresh_static_handle(self.model)
        except Exception as exc:
            CONSOLE.log(f"[viser-direct] static refresh failed: {exc}")

    def _force_viser_direct_push(self) -> None:
        """Public alias for the per-tick push (kept for compat with
        subclass call sites that prefer the descriptive name)."""
        self._push_viser_direct_transforms()

    # ====================================================================
    # Viewer re-render (Nerfstudio fallback path)
    # ====================================================================

    def _force_viewer_rerender(self) -> None:
        """Trigger an immediate re-render on every connected viser client.
        Called right after the tracker mutates object Gaussian means so
        the visual update rate tracks the tracker rate.

        Best-effort: silently no-ops if the viewer isn't ready or the
        import fails. In the rewrite ``enable_viser_direct=True`` is the
        primary visualizer; this hook only matters when the Nerfstudio
        viewer is also up (``--vis viewer`` / ``--vis viewer+tensorboard``).
        """
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        viewer = getattr(trainer, "viewer_state", None)
        if viewer is None or not getattr(viewer, "ready", False):
            return
        statemachines = getattr(viewer, "render_statemachines", None)
        if not statemachines:
            return
        try:
            from nerfstudio.viewer.render_state_machine import RenderAction
        except Exception:
            return
        for client_id, sm in list(statemachines.items()):
            try:
                camera_state = viewer.get_camera_state(sm.client)
            except Exception:
                continue
            if camera_state is None:
                continue
            try:
                sm.state = "low_static"
                sm.next_action = RenderAction("step", camera_state)
                sm.render_trigger.set()
            except Exception:
                continue

    # ====================================================================
    # Render + change-mask shims (shared by FF dispatcher + tracker tick)
    # ====================================================================

    @torch.no_grad()
    def _render_object_mask_cached(self, camera):
        """Tracked-object mask, rendered AT MOST ONCE per tracker tick and
        shared by every consumer (CDN object-exclusion, FF clean, post-cull
        re-clean, _ff_debug dump). Enforces 'render once, use everywhere' so
        the mask is always self-consistent and the saved debug mask is exactly
        the one used. The cache is invalidated whenever the object mask can
        change: tick start (new camera), rigid transform (object moved), and
        reseed/D0 (instance set changed). Culls/FF-inserts don't touch the
        tracked instance's screen footprint, so the cache survives them — which
        is the whole point (one render reused across the cull→reclean)."""
        if self._obj_mask_cache is None:
            # Under the model lock: a concurrent FF bg insert re-allocates
            # gauss_params, so an unlocked read tears (instance_ids at the old
            # N vs means at the new N -> IndexError mid-rasterize).
            with self._viser_lock_ctx():
                self._obj_mask_cache = self.model.render_object_mask(camera)
        return self._obj_mask_cache

    def _invalidate_object_mask_cache(self) -> None:
        self._obj_mask_cache = None

    def _render_from_camera(self, camera):
        """Render from ``camera`` in training mode so we get the
        training-resolution output.

        Takes ``_model_lock`` so a concurrent FF bg insert (which
        re-allocates ``gauss_params`` Parameters) can't tear the
        means/quats pair under the rasterizer."""
        was_training = self.model.training
        self.model.train()
        try:
            with self._viser_lock_ctx():
                return self.model.get_outputs(camera.to(self.model.device))
        finally:
            if not was_training:
                self.model.eval()

    def _compute_change_mask(
        self,
        rendered_rgb,
        rendered_depth,
        live_rgb,
        gt_depth,
        gripper_mask,
        object_mask,
        downsample_factor: Optional[int] = None,
        keep_largest_only: bool = True,
        rendered_alpha=None,
    ):
        """Compute change mask between render and live, excluding gripper
        + object regions. Thin shim over ``change_detection.compute_change_mask``
        that pulls thresholds + cleanup knobs from ``self.model.config``.

        ``downsample_factor=None`` (the default for every per-tick caller)
        resolves the factor from ``change_mask_downsample_factor`` /
        ``change_mask_downsample_target_side`` on the model config — so the RGB
        MS-SSIM runs on a coarse grid (averaging away per-pixel noise) instead
        of full native res, which otherwise flags the whole scene as changed
        every tick. Pass an explicit int to override.

        ``rendered_alpha`` is the rasterizer's cumulative-alpha map from
        ``outputs['accumulation']``; below ``scene_coverage_threshold`` the
        rendered depth is the max-fallback at uncovered pixels and CDN must
        ignore those (otherwise the camera looking beyond the warm-cache
        scene generates huge spurious 'change' bands above the object)."""
        mc = self.model.config
        if downsample_factor is None:
            from .change_detection import resolve_downsample_factor
            # Env override for live A/B of CDN sensitivity (no relaunch): higher
            # target_side = finer MS-SSIM grid = MORE sensitive (smaller change
            # regions detected); lower = more conservative.
            _tgt = int(os.environ.get(
                "DGS_CDN_TARGET_SIDE",
                getattr(mc, "change_mask_downsample_target_side", 100)))
            downsample_factor = resolve_downsample_factor(
                rendered_rgb,
                int(getattr(mc, "change_mask_downsample_factor", 0)),
                _tgt,
            )
        cfg = ChangeMaskConfig(
            depth_threshold=mc.change_mask_depth_threshold,
            rgb_threshold=mc.change_mask_rgb_threshold,
            use_rgb=mc.change_mask_use_rgb,
            mode=mc.change_mask_mode,
            blur_kernel_size=mc.change_mask_blur_kernel_size,
            blur_sigma=mc.change_mask_blur_sigma,
            filter_radius=mc.change_mask_filter_radius,
            min_component_size=mc.change_mask_min_component_size,
            dilate_radius=mc.active_mask_dilate_radius,
            scene_coverage_threshold=float(getattr(mc, "change_mask_coverage_threshold", 0.5)),
            outlier_median_multiplier=float(getattr(mc, "change_mask_outlier_median_multiplier", 10.0)),
            outlier_min_threshold_m=float(getattr(mc, "change_mask_outlier_min_threshold_m", 0.01)),
            gripper_erode_px=int(getattr(mc, "change_mask_gripper_erode_px", 0)),
        )
        return compute_change_mask(
            rendered_rgb=rendered_rgb,
            rendered_depth=rendered_depth,
            live_rgb=live_rgb,
            gt_depth=gt_depth,
            gripper_mask=gripper_mask,
            object_mask=object_mask,
            config=cfg,
            downsample_factor=downsample_factor,
            keep_largest_only=keep_largest_only,
            rendered_alpha=rendered_alpha,
        )

    # ====================================================================
    # XFeat motion estimator — D0 seed + per-tick advance
    # ====================================================================

    @torch.no_grad()
    def _object_crop_bbox(self, camera, padding_px: int):
        """Compute a screen-space bbox covering the tracked object's
        projected Gaussian centres, padded and clamped to image bounds.

        Returns ``(x0, y0, x1, y1)`` (Python ints) or ``None`` if no
        tracked-object Gaussian centre is visible (camera looking away,
        or tracker has nothing to track yet) — caller falls back to the
        full image.
        """
        model = self.model
        obj_mask = model._tracked_object_mask()
        if not obj_mask.any():
            return None
        # Per-Gaussian buffers can drift out-of-sync with means after some
        # FF delete/insert sequences (separate bug). Guard so the tick
        # doesn't crash — return None and the caller falls back to the
        # full image.
        n_means = int(model.means.shape[0])
        if int(obj_mask.shape[0]) != n_means:
            return None
        means_obj = model.means[obj_mask]
        # Use the tracker's RAW cumulative pose for the crop, not the live
        # (Kalman-FILTERED) means: the filtered pose lags/diverges under
        # smoothing, dragging the crop window off the true object -> match
        # death (measured: KF defaults reproducibly lost tracking at frame
        # ~626 on the fixture while KF-off survived). Raw pose keeps the
        # tracking loop self-consistent; the filter stays display-only.
        est = getattr(self, "_motion_estimator", None)
        ref = getattr(model, "_reference_object_means", None)
        if (est is not None and ref is not None
                and getattr(est, "_cumulative_R", None) is not None
                and ref.shape[0] == int(obj_mask.sum().item())):
            R_raw = torch.as_tensor(est._cumulative_R, device=ref.device, dtype=ref.dtype)
            t_raw = torch.as_tensor(est._cumulative_t, device=ref.device, dtype=ref.dtype)
            means_obj = ref @ R_raw.T + t_raw[None, :]

        def _scalar(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0].item())
            return float(x)
        fx = _scalar(camera.fx); fy = _scalar(camera.fy)
        cx = _scalar(camera.cx); cy = _scalar(camera.cy)
        W = int(_scalar(camera.width)); H = int(_scalar(camera.height))

        c2w = camera.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        c2w = c2w.to(means_obj.device, dtype=means_obj.dtype)
        R = c2w[:3, :3]; t = c2w[:3, 3]
        means_cam = (means_obj - t[None, :]) @ R
        depths = -means_cam[:, 2]
        in_front = depths > 1e-6
        safe_d = torch.where(in_front, depths, torch.ones_like(depths))
        u = fx * (means_cam[:, 0] / safe_d) + cx
        v = fy * (-means_cam[:, 1] / safe_d) + cy
        visible = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not visible.any():
            return None
        u_v = u[visible]; v_v = v[visible]
        x0 = max(0, int(u_v.min().item()) - int(padding_px))
        y0 = max(0, int(v_v.min().item()) - int(padding_px))
        x1 = min(W, int(u_v.max().item()) + int(padding_px) + 1)
        y1 = min(H, int(v_v.max().item()) + int(padding_px) + 1)
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            return None
        return (x0, y0, x1, y1)

    @torch.no_grad()
    def _crop_for_xfeat(self, rgb, depth, camera, mask, bbox):
        """Crop ``rgb`` (H,W,3), ``depth`` (H,W) or (H,W,1), and ``mask``
        (H,W) or None to ``bbox = (x0, y0, x1, y1)``, and rebuild a single-
        camera ``Cameras`` with cx/cy shifted by (x0, y0) and width/height
        replaced. Depth backprojection inside the estimator stays metric
        because (fx, fy) are unchanged."""
        from nerfstudio.cameras.cameras import Cameras, CameraType
        x0, y0, x1, y1 = bbox
        if rgb.ndim == 3:
            rgb_c = rgb[y0:y1, x0:x1, :].contiguous()
        else:
            rgb_c = rgb[y0:y1, x0:x1].contiguous()
        if depth.ndim == 3:
            depth_c = depth[y0:y1, x0:x1, :].contiguous()
        else:
            depth_c = depth[y0:y1, x0:x1].contiguous()
        mask_c = None
        if mask is not None:
            if mask.ndim == 2:
                mask_c = mask[y0:y1, x0:x1].contiguous()
            else:
                mask_c = mask[y0:y1, x0:x1, ...].contiguous()

        def _scalar(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0].item())
            return float(x)
        fx = _scalar(camera.fx); fy = _scalar(camera.fy)
        cx = _scalar(camera.cx) - x0
        cy = _scalar(camera.cy) - y0
        c2w = camera.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        camera_c = Cameras(
            camera_to_worlds=c2w.unsqueeze(0).cpu(),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=int(x1 - x0), height=int(y1 - y0),
            camera_type=CameraType.PERSPECTIVE,
        ).to(rgb.device)
        return rgb_c, depth_c, camera_c, mask_c

    def _initialize_motion_estimator(self, rgb, depth, camera, mask) -> None:
        """Seed the XFeat tracker with the D0 reference frame.

        Samples XFeat keypoints inside the object mask of the D0 image,
        back-projects them via depth + intrinsics + c2w to get world-frame
        reference 3D positions. Subsequent :meth:`_apply_motion_estimator`
        calls extract on the new RGB, match against the anchor via
        LighterGlue, and run RANSAC-Kabsch to recover ``(R, t)``.

        ``enable_cotracker_rigid_motion`` is the legacy name kept for
        checkpoint compat — it gates XFeat.
        """
        if not self.model.config.enable_cotracker_rigid_motion:
            return
        from .utils.xfeat_motion import XFeatMotionEstimator
        _t_xf = time.time()
        self._motion_estimator = XFeatMotionEstimator(
            device=self.model.device,
            top_k=self.model.config.xfeat_top_k,
            detection_threshold=self.model.config.xfeat_detection_threshold,
            min_cossim=self.model.config.xfeat_min_cossim,
            min_track_points=self.model.config.xfeat_min_track_points,
            ransac_iterations=self.model.config.xfeat_ransac_iterations,
            ransac_inlier_threshold=self.model.config.xfeat_ransac_inlier_threshold,
            weights_path=self.model.config.xfeat_weights_path,
            use_lighterglue=self.model.config.xfeat_use_lighterglue,
            lighterglue_min_conf=self.model.config.xfeat_lighterglue_min_conf,
            lighterglue_depth_confidence=self.model.config.xfeat_lighterglue_depth_confidence,
            object_search_radius_px=self.model.config.xfeat_object_search_radius_px,
            use_semi_dense=self.model.config.xfeat_use_semi_dense,
            pose_filter_enabled=self.model.config.xfeat_pose_filter_enabled,
            pose_filter_accel_sigma=self.model.config.xfeat_pose_filter_accel_sigma,
            pose_filter_alpha_sigma=self.model.config.xfeat_pose_filter_alpha_sigma,
            static_hold_enabled=self.model.config.xfeat_static_hold,
            static_hold_window=self.model.config.xfeat_static_hold_window,
            static_hold_trans_m=self.model.config.xfeat_static_hold_trans_m,
            static_hold_rot_deg=self.model.config.xfeat_static_hold_rot_deg,
            pose_filter_meas_trans_sigma_m=self.model.config.xfeat_pose_filter_meas_trans_sigma_m,
            pose_filter_meas_rot_sigma_deg=self.model.config.xfeat_pose_filter_meas_rot_sigma_deg,
            pose_filter_fixed_fps=self.model.config.xfeat_pose_filter_fixed_fps,
        )
        try:
            from .utils import timing_ledger as _tl
            _tl.record(self.config.datamanager.data, "teleop_init",
                       "XFeat+LighterGlue", "load", _t_xf, time.time())
        except Exception:
            pass
        if getattr(self.model.config, "xfeat_crop_to_object_bbox", False):
            bbox = self._object_crop_bbox(
                camera, padding_px=int(self.model.config.xfeat_crop_padding_px),
            )
            if bbox is not None:
                CONSOLE.log(
                    f"[dynamic-gs] D0 crop bbox=({bbox[0]},{bbox[1]})-"
                    f"({bbox[2]},{bbox[3]}) [{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}]"
                )
                rgb, depth, camera, mask = self._crop_for_xfeat(
                    rgb, depth, camera, mask, bbox,
                )
        seeded = self._motion_estimator.initialize(
            rgb=rgb, depth=depth, camera=camera, mask=mask,
        )
        CONSOLE.log(
            f"[dynamic-gs] XFeat reference seed on D0 -> "
            f"fast={self._motion_estimator.last_init_fast_point_count}, "
            f"sampled={self._motion_estimator.last_init_sampled_count}, "
            f"depth_valid={self._motion_estimator.last_init_depth_valid_count}, "
            f"dense_fallback={self._motion_estimator.last_init_used_dense_fallback}, "
            f"tracks={seeded}, ready={self._motion_estimator.ready}"
        )
        if seeded < self._motion_estimator.min_track_points:
            CONSOLE.log(
                f"[dynamic-gs] XFeat seeded too few D0 points: "
                f"{seeded} < min_track_points={self._motion_estimator.min_track_points}"
            )

    def _apply_motion_estimator(self, camera, batch, current_mask=None) -> None:
        """Per-tick XFeat advance. Builds the gripper-blue-composited live RGB,
        runs the estimator's ``estimate_and_advance`` against the cached anchor,
        and on success calls ``model.apply_rigid_object_transform_from_reference``
        to move the flagged object Gaussians."""
        if self._motion_estimator is None:
            return
        t = time.time()
        current_live_rgb = self._build_tracking_rgb(batch)
        self._timing["DN.3a_get_live_rgb"].append(time.time() - t)
        t_mask = time.time()
        # Per-tick object mask (cached render of the tracked instance only) —
        # passed to the tracker so matches landing OUTSIDE the object's
        # predicted footprint (static background that survives gripper-keep)
        # are dropped before RANSAC. Without it the pose pins to the
        # background once the object is grasped+lifted ("stops moving").
        current_object_mask = (
            self._render_object_mask_cached(camera)
            if getattr(self.model.config, "xfeat_object_mask_filter", True)
            else None
        )
        if current_mask is None:
            current_mask = self.model._get_batch_mask(batch)
        self._timing["DN.3j_object_mask_render"].append(time.time() - t_mask)
        # Optional: crop rgb+depth+camera+mask to the tracked object's
        # projected bbox so XFeat's top_k keypoints all land on the object
        # instead of being spent on background. Critical for small objects.
        current_depth = batch["depth_image"]
        current_camera = camera
        if getattr(self.model.config, "xfeat_crop_to_object_bbox", False):
            bbox = self._object_crop_bbox(
                camera, padding_px=int(self.model.config.xfeat_crop_padding_px),
            )
            if bbox is not None:
                current_live_rgb, current_depth, current_camera, current_mask = (
                    self._crop_for_xfeat(
                        current_live_rgb, current_depth, camera, current_mask, bbox,
                    )
                )
                # Crop the object mask to the SAME bbox so it stays aligned
                # with the cropped frame the tracker extracts/matches on.
                if current_object_mask is not None:
                    x0, y0, x1, y1 = bbox
                    current_object_mask = (
                        current_object_mask[y0:y1, x0:x1, ...].contiguous()
                        if current_object_mask.ndim == 3
                        else current_object_mask[y0:y1, x0:x1].contiguous()
                    )
        t = time.time()
        motion_estimate = self._motion_estimator.estimate_and_advance(
            current_rgb=current_live_rgb,
            current_depth=current_depth,
            current_camera=current_camera,
            current_mask=current_mask,
            current_object_mask=current_object_mask,
        )
        self._timing["DN.3_estimate_total"].append(time.time() - t)
        # Wall-clock span of the dynamic tick loop, for the effective tracker /
        # FF Hz in the report header. Stamped once per tick (both pipelines call
        # this). Recorded mode runs as fast as compute allows, so the real rate
        # must come from wall-time, not from summing per-tick compute.
        _now_wall = time.time()
        if getattr(self, "_dyn_first_tick_wall", None) is None:
            self._dyn_first_tick_wall = _now_wall
        self._dyn_last_tick_wall = _now_wall
        sub = motion_estimate.timings or {}
        self._timing["DN.3b_estimator_input_prep"].append(float(sub.get("input_prep", 0.0)))
        self._timing["DN.3c_predictor_forward"].append(
            float(sub.get("predictor_forward",
                          sub.get("xfeat_extract",
                                  sub.get("klt_forward", 0.0))))
        )
        self._timing["DN.3c_xfeat_extract"].append(float(sub.get("xfeat_extract", 0.0)))
        self._timing["DN.3c0_gpu_queue_wait"].append(float(sub.get("gpu_queue_wait", 0.0)))
        self._timing["DN.3i_lighterglue_match"].append(
            float(sub.get("lighterglue_match", 0.0))
        )
        self._timing["DN.3d_postprocess"].append(float(sub.get("postprocess", 0.0)))
        self._timing["DN.3e_ransac_kabsch"].append(float(sub.get("ransac_kabsch", 0.0)))
        self._timing["DN.3h_resample"].append(float(sub.get("resample", 0.0)))
        t = time.time()
        try:
            frame_name = self.datamanager.get_current_dynamic_frame_name()
        except Exception:
            frame_name = f"step_{self._dynamic_step_counter:06d}"
        if self.config.save_debug_images:
            self._write_motion_log(frame_name, motion_estimate)
            self._save_motion_debug(frame_name, motion_estimate)
        self._timing["DN.3f_debug_io"].append(time.time() - t)
        if not motion_estimate.success:
            mean_res_mm = (
                motion_estimate.mean_residual * 1000.0
                if motion_estimate.mean_residual != float("inf") else float("inf")
            )
            med_res_mm = (
                motion_estimate.median_residual * 1000.0
                if motion_estimate.median_residual != float("inf") else float("inf")
            )
            CONSOLE.log(
                f"[dynamic-gs] XFeat rigid motion unavailable for {frame_name}: "
                f"raw={motion_estimate.raw_visible_count}, "
                f"mask={motion_estimate.mask_visible_count}, "
                f"depth={motion_estimate.depth_valid_count}, "
                f"correspondences={motion_estimate.correspondence_count}, "
                f"inliers={motion_estimate.inlier_count}, "
                f"mask_fallback={motion_estimate.used_mask_fallback}, "
                f"resid_mm(mean/med)={mean_res_mm:.1f}/{med_res_mm:.1f}"
            )
            return
        t = time.time()
        with self._viser_lock_ctx():
            moved_count = self.model.apply_rigid_object_transform_from_reference(
                motion_estimate.rotation, motion_estimate.translation,
            )
        # Object moved → its rendered mask is stale; next request re-renders.
        self._invalidate_object_mask_cache()
        self._last_motion_estimate = motion_estimate
        # Trajectory log for smoothness analysis (DGS_TRACK_TRAJ_LOG=<csv>).
        # Object centroid c(t)=R·c0+T from the FIXED D0 reference means, so it's
        # immune to FF inserts. Columns: wall_t, cx,cy,cz, rvx,rvy,rvz, inliers, corr.
        _tj = os.environ.get("DGS_TRACK_TRAJ_LOG")
        _ref = getattr(self.model, "_reference_object_means", None)
        if _tj and _ref is not None:
            try:
                import cv2 as _cv2
                _R = torch.as_tensor(motion_estimate.rotation, device=_ref.device, dtype=_ref.dtype).reshape(3, 3)
                _T = torch.as_tensor(motion_estimate.translation, device=_ref.device, dtype=_ref.dtype).reshape(3)
                _ct = (_R @ _ref.mean(0) + _T).detach().cpu().numpy()
                _rv = _cv2.Rodrigues(np.ascontiguousarray(motion_estimate.rotation, dtype=np.float64))[0].ravel()
                with open(_tj, "a") as _f:
                    _f.write(f"{time.time():.6f},{_ct[0]:.6f},{_ct[1]:.6f},{_ct[2]:.6f},"
                             f"{_rv[0]:.6f},{_rv[1]:.6f},{_rv[2]:.6f},"
                             f"{int(motion_estimate.inlier_count)},{int(motion_estimate.correspondence_count)},"
                             f"{frame_name}\n")
            except Exception:
                pass
        self._timing["DN.3g_apply_transform"].append(time.time() - t)
        # Tracker-tick-driven render: the per-tick model mutation just
        # completed, wake the viser render thread for exactly one push
        # so the browser sees this tick's motion without waiting on a
        # polling clock.
        srv = getattr(self, "_viser_direct_server", None)
        if srv is not None:
            srv.request_render()
        self._last_inlier_count = int(motion_estimate.inlier_count)
        self._last_correspondence_count = int(motion_estimate.correspondence_count)
        self._inlier_window.append(self._last_inlier_count)
        self._corr_window.append(self._last_correspondence_count)
        if moved_count == 0:
            CONSOLE.log(
                f"[dynamic-gs] XFeat estimated motion for {frame_name}, "
                "but no object Gaussians were moved. Check object_flags/reference pose consistency."
            )

    # ---- Motion-estimator helpers ----

    def _build_tracking_rgb(self, batch) -> "torch.Tensor":
        """Live RGB for the tracker, with the dataset mask composited onto
        the model background. Pixels where ``batch["mask"]`` is 0 (gripper)
        are replaced with the simulator-background color so the tracker
        cannot lock onto gripper texture."""
        rgb = self.model.get_live_rgb(batch, apply_training_downscale=False)
        mask = batch.get("mask")
        if mask is None:
            return rgb
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = mask.float().to(rgb.device)
        if mask.shape[:2] != rgb.shape[:2]:
            return rgb
        background = self.model._get_background_color().to(rgb.device).view(1, 1, -1)
        return rgb * mask + background * (1.0 - mask)

    def _get_debug_dir(self) -> Path:
        return Path(self.datamanager.config.data) / self.datamanager.config.dynamic_subdir / "debug"

    def _get_motion_debug_dir(self) -> Path:
        return self._get_debug_dir() / "tracker_debug"

    def _write_motion_log(self, frame_name: str, motion_estimate) -> None:
        debug_dir = self._get_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_path = debug_dir / f"{frame_name}_motion.txt"
        log_lines = [
            f"success: {motion_estimate.success}",
            f"ready: {motion_estimate.ready}",
            f"correspondence_count: {motion_estimate.correspondence_count}",
            f"inlier_count: {motion_estimate.inlier_count}",
            f"track_count_before: {motion_estimate.track_count_before}",
            f"track_count_after: {motion_estimate.track_count_after}",
            f"raw_visible_count: {motion_estimate.raw_visible_count}",
            f"mask_visible_count: {motion_estimate.mask_visible_count}",
            f"depth_valid_count: {motion_estimate.depth_valid_count}",
            f"used_mask_fallback: {motion_estimate.used_mask_fallback}",
            f"mean_residual: {motion_estimate.mean_residual}",
            f"median_residual: {motion_estimate.median_residual}",
            f"rotation: {motion_estimate.rotation.tolist()}",
            f"translation: {motion_estimate.translation.tolist()}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n")

    def _save_motion_debug(self, frame_name: str, est) -> None:
        """Side-by-side previous->current frame with tracked points + lines."""
        if est.previous_points_xy is None or est.current_points_xy is None:
            return
        if est.previous_rgb is None or est.current_rgb is None:
            return
        prev_img = est.previous_rgb.detach().float().cpu().numpy()
        curr_img = est.current_rgb.detach().float().cpu().numpy()
        if prev_img.max() > 1.5:
            prev_img = prev_img / 255.0
        if curr_img.max() > 1.5:
            curr_img = curr_img / 255.0
        prev_img = prev_img.clip(0, 1)
        curr_img = curr_img.clip(0, 1)
        # The XFeat crop bbox varies per tick, so prev/curr crops can differ in
        # height by a pixel — pad both to the taller one (bottom) so the
        # side-by-side concat (axis=1) doesn't raise. Point coords are unaffected
        # (they live within the original crop bounds).
        Hc = max(prev_img.shape[0], curr_img.shape[0])
        if prev_img.shape[0] != Hc:
            prev_img = np.pad(prev_img, ((0, Hc - prev_img.shape[0]), (0, 0), (0, 0)))
        if curr_img.shape[0] != Hc:
            curr_img = np.pad(curr_img, ((0, Hc - curr_img.shape[0]), (0, 0), (0, 0)))
        h, w = prev_img.shape[:2]
        canvas = np.concatenate([prev_img, curr_img], axis=1)
        canvas = (canvas * 255).astype(np.uint8).copy()
        # prev/curr crop widths differ per tick, so the canvas is
        # (prev_w + curr_w) wide, NOT 2*w — use the true canvas dims for all
        # draw-bounds checks below (a stale 2*w let edge markers overflow).
        cw = canvas.shape[1]
        prev_pts = est.previous_points_xy
        curr_pts = est.current_points_xy
        inlier_mask = est.tracked_inlier_mask
        n = min(len(prev_pts), len(curr_pts))
        for i in range(n):
            px, py = int(prev_pts[i, 0]), int(prev_pts[i, 1])
            cx, cy = int(curr_pts[i, 0]) + w, int(curr_pts[i, 1])
            is_inlier = bool(inlier_mask[i]) if inlier_mask is not None and i < len(inlier_mask) else False
            point_color = [0, 255, 0] if is_inlier else [255, 0, 0]
            line_color = [0, 180, 0] if is_inlier else [180, 0, 0]
            steps = max(abs(cx - px), abs(cy - py), 1)
            for t in range(steps + 1):
                lx = int(px + (cx - px) * t / steps)
                ly = int(py + (cy - py) * t / steps)
                if 0 <= ly < h and 0 <= lx < cw:
                    canvas[ly, lx] = line_color
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= py + dy < h and 0 <= px + dx < w:
                        canvas[py + dy, px + dx] = point_color
                    if 0 <= cy + dy < h and 0 <= cx + dx < cw:
                        canvas[cy + dy, cx + dx] = point_color
        dbg = self._get_motion_debug_dir()
        dbg.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(dbg / f"{frame_name}_tracker.png")

    # ====================================================================
    # Feedforward dispatcher — rgbd_decode + anysplat_decode
    # ====================================================================

    def _save_ff_debug_images(
        self,
        *,
        call_id: int,
        frame_idx: int,
        camera,
        cdn_raw,
        cdn_clean,
        prerendered_obj_mask,
        target_frame,
        pre_cull_render=None,
        post_cull_render=None,
    ) -> None:
        """Dump per-FF-call debug PNGs under <data>/dynamic_scene/_ff_debug/.

        Saved in a fixed numbered order so they sort + view in the raw→clean
        pipeline order (call_NNNN_frame_NNNNNN_<N>_<name>.png):
            1_gripper_mask          dataset mask (gripper=black, keep=white)
            2_object_mask           render_object_mask (tracked instance_id==d0 only)
            3_real                  live RGB
            4_rendered              scene render BEFORE the cull (== what RAW CDN saw)
            5_rerendered_after_cull scene render AFTER the in-front cull (== what CLEAN CDN saw)
            6_raw_mask              CDN before object-subtract + cull-reclean
            7_clean_mask            CDN after object-subtract + cull-reclean (== what FF decodes)

        Compare 4 vs 5 to see what the cull removed, and 6 vs 7 to see what that
        did to the change mask — the raw→clean delta that is neither object nor
        gripper is the cull-reclean re-render (4→5).
        """
        from pathlib import Path as _Path
        import cv2 as _cv2

        try:
            data_root = _Path(self.datamanager.config.data)
        except Exception:
            data_root = _Path("/tmp")
        out_dir = data_root / "dynamic_scene" / "_ff_debug"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"call_{call_id:04d}_frame_{frame_idx:06d}"

        def _to_u8(t):
            import numpy as _np
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().float().numpy()
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t[..., 0]
            if t.ndim == 3 and t.shape[-1] == 3:
                return (t.clip(0, 1) * 255).astype(_np.uint8)
            t = t.astype("float32")
            tmax = float(t.max()) if t.size else 1.0
            if tmax <= 1.5:
                return (t.clip(0, 1) * 255).astype("uint8")
            return _np.clip(t, 0, 255).astype("uint8")

        def _rgb_bgr_u8(x):
            """RGB tensor/array (float 0-1 or uint8) → BGR uint8 for cv2."""
            import numpy as _np
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            else:
                x = _np.asarray(x)
            if x.dtype == _np.uint8:
                u8 = x
            else:
                u8 = _np.clip(x.astype(_np.float32) * 255.0, 0, 255).astype(_np.uint8)
            if u8.ndim == 3 and u8.shape[-1] == 4:
                u8 = u8[..., :3]
            if u8.ndim == 3 and u8.shape[-1] == 3:
                return _cv2.cvtColor(u8, _cv2.COLOR_RGB2BGR)
            return None

        try:
            batch = target_frame.get("batch") if isinstance(target_frame, dict) else None

            def _save(name, img):
                try:
                    if img is not None:
                        _cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), img)
                except Exception:
                    pass

            # Fixed pipeline order (1..7) so the files sort the way they're read.
            # 1. gripper / dataset mask (0=gripper, 255=keep)
            _save("1_gripper_mask",
                  _to_u8(batch.get("mask")) if batch is not None else None)

            # 2. object mask — the EFFECTIVE subtracted footprint (scaled +2%
            #    about centroid + px-dilated), exactly as _feedforward_clean_cdn
            #    applies it, so the enlargement is visible here.
            try:
                om = prerendered_obj_mask if prerendered_obj_mask is not None \
                    else self._render_object_mask_cached(camera)
                if om is not None:
                    sc = float(getattr(self.config, "feedforward_object_mask_scale", 1.0))
                    if sc != 1.0:
                        om = self._scale_mask_about_centroid(om, sc)
                    dpx = int(self.config.feedforward_object_mask_dilate_px)
                    if dpx > 0:
                        om = dilate_binary_mask(om, dpx)
            except Exception:
                om = None
            _save("2_object_mask", _to_u8(om) if om is not None else None)

            # 3. real (live RGB)
            _save("3_real", _rgb_bgr_u8(batch.get("image")) if batch is not None else None)

            # 4. rendered scene BEFORE the cull (== what the RAW CDN compared to)
            _save("4_rendered", _rgb_bgr_u8(pre_cull_render))

            # 5. re-rendered scene AFTER the in-front cull (== what the CLEAN CDN compared to)
            _save("5_rerendered_after_cull", _rgb_bgr_u8(post_cull_render))

            # 6. raw change mask, 7. clean change mask (after object-subtract + cull-reclean)
            _save("6_raw_mask", _to_u8(cdn_raw))
            _save("7_clean_mask", _to_u8(cdn_clean))
        except Exception:
            pass

    def _scale_mask_about_centroid(self, mask, scale: float):
        """Enlarge a binary mask by ``scale`` (1.02 = +2%) about its OWN centroid.

        Used to slightly over-cover the tracked object's subtracted footprint so
        the thin rendered-vs-live misplacement ring isn't treated as change.
        """
        if scale is None or abs(float(scale) - 1.0) < 1e-6:
            return mask
        import numpy as _np
        import cv2 as _cv2
        was_tensor = isinstance(mask, torch.Tensor)
        dev = mask.device if was_tensor else None
        arr = mask.detach().cpu().numpy() if was_tensor else _np.asarray(mask)
        has_ch = (arr.ndim == 3 and arr.shape[-1] == 1)
        sq = arr[..., 0] if has_ch else arr
        ys, xs = _np.where(sq > 0.5)
        if xs.size == 0:
            return mask
        cx, cy = float(xs.mean()), float(ys.mean())
        H, W = sq.shape[:2]
        s = float(scale)
        M = _np.array([[s, 0.0, cx * (1.0 - s)],
                       [0.0, s, cy * (1.0 - s)]], dtype=_np.float32)
        out = _cv2.warpAffine((sq > 0.5).astype(_np.uint8), M, (W, H),
                              flags=_cv2.INTER_NEAREST)
        # Union with the original so the enlarged mask always CONTAINS it
        # (scaling-about-centroid can leave a few rounding gaps on thin parts).
        out = _np.maximum(out, (sq > 0.5).astype(_np.uint8)).astype(_np.float32)
        if has_ch:
            out = out[..., None]
        return torch.from_numpy(out).to(dev) if was_tensor else out

    def _feedforward_clean_cdn(self, camera, cdn, frame_name: Optional[str] = None, prerendered_obj_mask=None):
        """Subtract the moving object's rendered Gaussian footprint from CDN.
        Prevents the decoder from back-projecting the live object's surface
        as flat Gaussians on top of the tracked 3D object."""
        if prerendered_obj_mask is not None:
            obj_mask_now = prerendered_obj_mask
        else:
            try:
                obj_mask_now = self._render_object_mask_cached(camera)
            except Exception as exc:
                CONSOLE.log(f"[feedforward] render_object_mask failed: {exc}; using raw CDN")
                return cdn
        if obj_mask_now is None:
            return cdn
        if obj_mask_now.ndim == 2:
            obj_mask_now = obj_mask_now[..., None]
        if obj_mask_now.shape != cdn.shape:
            h, w = cdn.shape[:2]
            obj_mask_now = TF.interpolate(
                obj_mask_now.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)
        scale = float(getattr(self.config, "feedforward_object_mask_scale", 1.0))
        if scale != 1.0:
            obj_mask_now = self._scale_mask_about_centroid(obj_mask_now, scale)
        dilate_px = int(self.config.feedforward_object_mask_dilate_px)
        if dilate_px > 0:
            obj_mask_now = dilate_binary_mask(obj_mask_now, dilate_px)
        cleaned = (cdn * (1.0 - obj_mask_now)).detach()
        return cleaned

    def _run_feedforward(
        self,
        target_frame: TrackerFrame,
        mode_label: Literal["oneshot", "recurring"],
        *,
        prerendered_obj_mask=None,
        prerendered_depth=None,
    ) -> None:
        """Dispatcher for feedforward hole-fill at a target tracker frame."""
        # Defensive: the CDN is rendered only on FF-firing ticks; if it's
        # somehow absent (None), there is nothing to decode against — skip
        # rather than dereference None downstream.
        if target_frame is None or target_frame.get("cdn") is None:
            CONSOLE.log(f"[feedforward] {mode_label}: no CDN this tick — skip")
            return
        if str(getattr(self.config, "enable_feedforward_inpaint", "off")) == "anysplat_decode":
            return self._run_feedforward_anysplat(
                target_frame, mode_label,
                prerendered_obj_mask=prerendered_obj_mask,
            )

        from .utils.active_mask import select_top_n_components_filtered
        from .utils.rgbd_decode import decode_component_to_gaussians

        call_id = self._feedforward_call_counter
        self._feedforward_call_counter += 1

        t_call0 = time.time()
        camera = target_frame["camera"]
        frame_idx = int(target_frame["frame_idx"])
        cdn = target_frame["cdn"]

        batch = target_frame["batch"]
        if batch is None:
            CONSOLE.log(f"[feedforward] frame {frame_idx} has no batch — skip")
            return

        bg = self.model._get_background_color()
        live_rgb_full = self.model.composite_with_background(
            self.model.get_gt_img(batch["image"]), bg
        )
        gt_depth = self.model._get_gt_depth(batch)
        if gt_depth is None:
            CONSOLE.log(f"[feedforward] frame {frame_idx} has no depth — skip")
            return

        t0 = time.time()
        try:
            frame_name_for_cdn = self.datamanager.get_dynamic_frame_name(frame_idx)
        except Exception:
            frame_name_for_cdn = None
        cdn_clean = self._feedforward_clean_cdn(
            camera, cdn, frame_name=frame_name_for_cdn,
            prerendered_obj_mask=prerendered_obj_mask,
        )
        self._timing["FF.1_cdn_clean"].append(time.time() - t0)

        # Cull-before-decode: drop in-front occluders (original + prior FF
        # Gaussians) over the CDN region, recompute CDN. If culling alone
        # cleared the change, the component list below is empty and we skip
        # the decode loop entirely.
        t0 = time.time()
        cdn_clean, n_pre_culled = self._feedforward_cull_then_reclean_cdn(
            camera, batch, cdn_clean, gt_depth, frame_name=frame_name_for_cdn,
            prerendered_obj_mask=prerendered_obj_mask,
        )
        self._timing["FF.1b_cull_before_decode"].append(time.time() - t0)
        if n_pre_culled > 0:
            self._refresh_viser_direct_after_feedforward()

        t0 = time.time()
        if mode_label == "oneshot":
            components = select_top_n_components_filtered(
                cdn_clean,
                n=int(self.config.feedforward_top_n_components),
                area_ratio=float(self.config.feedforward_dominant_area_ratio),
                min_area=1,
            )
        else:
            components = select_top_n_components_filtered(
                cdn_clean, n=256, area_ratio=0.0, min_area=1,
            )
        self._timing["FF.2_component_select"].append(time.time() - t0)

        if not components:
            CONSOLE.log(
                f"[feedforward] {mode_label} call={call_id} step={self._dynamic_step_counter} "
                f"frame={frame_idx} no components above min_area "
                f"(pre-decode cull removed {n_pre_culled}; decode skipped)"
            )
            return

        total_inserted = 0
        total_deleted = 0
        per_component_diag: list[dict] = []
        for k, comp_mask in enumerate(components):
            t0 = time.time()
            decoded = decode_component_to_gaussians(
                camera,
                live_rgb_full,
                gt_depth,
                comp_mask,
                opacity=float(self.config.feedforward_rgbd_opacity),
                normal_smoothing_radius=int(self.config.feedforward_rgbd_normal_smoothing_radius),
                min_valid_fraction=float(self.config.feedforward_rgbd_min_valid_fraction),
                scale_multiplier=float(self.config.feedforward_rgbd_scale_multiplier),
                cliff_threshold_m=float(self.config.feedforward_rgbd_cliff_threshold_m),
                post_cliff_erode_px=int(self.config.feedforward_rgbd_post_cliff_erode_px),
                rendered_depth_m=prerendered_depth,
                leak_threshold_m=float(self.config.feedforward_rgbd_leak_threshold_m),
            )
            self._timing["FF.3_decode"].append(time.time() - t0)

            if decoded is None:
                CONSOLE.log(
                    f"[feedforward] {mode_label} call={call_id} comp={k} empty — skipped"
                )
                continue
            if decoded.get("skipped", False):
                per_component_diag.append({"component": k, **decoded["diagnostics"], "skipped": True})
                CONSOLE.log(
                    f"[feedforward] {mode_label} call={call_id} comp={k} skipped "
                    f"valid_fraction={decoded['diagnostics']['valid_fraction']:.3f}"
                )
                continue

            t0 = time.time()
            if self.device == torch.device("cuda") or self.device.type == "cuda":
                torch.cuda.synchronize()
            if not self.config.feedforward_skip_delete:
                try:
                    _ = self._render_from_camera(camera)
                except Exception as exc:
                    CONSOLE.log(f"[feedforward] pre-delete render failed: {exc}; skip comp")
                    continue
                n_deleted = self._feedforward_delete_in_region(camera, comp_mask)
            else:
                n_deleted = 0

            # Per-component in-front cull. Redundant when the pre-decode cull
            # already swept the CDN union (feedforward_cull_before_decode), so
            # only run it on the legacy path where that pass is disabled.
            n_culled = 0
            if self.config.feedforward_cull_in_front and not self.config.feedforward_cull_before_decode:
                with self._viser_lock_ctx():
                    n_culled = self._feedforward_cull_in_front_of_depth(
                        camera, comp_mask, gt_depth,
                        depth_tol_m=float(self.config.feedforward_cull_in_front_depth_tol_m),
                    )
            n_deleted += n_culled
            self._timing["FF.4_crop_and_delete"].append(time.time() - t0)

            t0 = time.time()
            with self._viser_lock_ctx():
                inserted_ids = self.model.insert_inpaint_gaussians(
                    xyz=decoded["xyz"],
                    features_dc=decoded["features_dc"],
                    features_rest=decoded["features_rest"],
                    opacities=decoded["opacities"],
                    scales=decoded["scales"],
                    quats=decoded["quats"],
                    instance_id=999,
                )
            self._timing["FF.5_insert"].append(time.time() - t0)

            self._viser_direct_register_ff_insert(inserted_ids)

            n_inserted = int(inserted_ids.numel())
            total_inserted += n_inserted
            total_deleted += n_deleted
            per_component_diag.append({
                "component": k,
                "inserted": n_inserted,
                "deleted": n_deleted,
                **decoded["diagnostics"],
            })

        total_per_call = time.time() - t_call0
        self._timing["FF.6_total_per_call"].append(total_per_call)

        obj_count = int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())
        ins_count = int((self.model.inserted_flags.squeeze(-1) > 0.5).sum().item())
        CONSOLE.log(
            f"[feedforward] {mode_label} call={call_id} step={self._dynamic_step_counter} "
            f"frame={frame_idx} components={len(per_component_diag)} "
            f"inserted={total_inserted} deleted={total_deleted} total_ms={total_per_call*1000:.1f} "
            f"object_flags_count={obj_count} inserted_flags_count={ins_count} "
            f"total_gauss={self.model.num_points}"
        )

    # ---- FF dispatcher helpers ----

    @torch.no_grad()
    def _feedforward_delete_in_region(self, camera, component_mask) -> int:
        """Delete Gaussians whose 2D footprint overlaps ``component_mask`` AND
        have ``object_instance_ids ∈ {0, 999}``. Tracked-object Gaussians are
        never touched. Requires a recent full-scene render."""
        from .utils.active_mask import build_active_mask, extract_projected_centers_and_radii

        model = self.model
        if model.num_points == 0:
            return 0
        try:
            centers_2d, radii = extract_projected_centers_and_radii(
                model.info, model.num_points
            )
        except Exception as exc:
            CONSOLE.log(f"[feedforward] projection failed: {exc}; skipping delete")
            return 0

        comp = component_mask
        if comp.ndim == 3 and comp.shape[-1] == 1:
            comp = comp[..., 0]
        comp_bool = (comp > 0.5).to(centers_2d.device)
        active_mask = build_active_mask(comp_bool, centers_2d, radii)
        in_region = active_mask.to(torch.bool)

        instance_ids = model.object_instance_ids.squeeze(-1)
        eligible = (instance_ids == 0) | (instance_ids == 999)
        to_delete = in_region & eligible
        indices = torch.nonzero(to_delete, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return 0
        return model.delete_gaussian_indices(indices)

    @torch.no_grad()
    def _feedforward_cull_in_front_of_depth(
        self, camera, component_mask, gt_depth_m, depth_tol_m: float = 0.005,
    ) -> int:
        """Delete Gaussians sitting in front of the real sensor surface.

        For every Gaussian whose 2D projection (centre) lands inside
        ``component_mask``, compare its camera-space depth to the sensor
        depth at that pixel. If shallower by more than ``depth_tol_m``,
        delete (it's an artifact occluding the true geometry). Restricted
        to ``object_instance_ids ∈ {0, 999}``.

        Direct projection — does NOT require a prior render.
        """
        model = self.model
        if model.num_points == 0:
            return 0

        def _scalar(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0].item())
            return float(x)
        fx = _scalar(camera.fx); fy = _scalar(camera.fy)
        cx = _scalar(camera.cx); cy = _scalar(camera.cy)
        W_cam = int(_scalar(camera.width)); H_cam = int(_scalar(camera.height))

        # Depth and component_mask are at the model's render resolution
        # (which may be downscaled vs the dataset's native camera resolution
        # during the resolution-schedule warm-up). Derive grid dims from the
        # depth tensor itself, then scale intrinsics so projection lands in
        # the depth grid's coordinate system.
        depth = gt_depth_m
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.to(model.means.device)
        H, W = int(depth.shape[0]), int(depth.shape[1])
        if (H, W) != (H_cam, W_cam):
            sx = W / float(W_cam)
            sy = H / float(H_cam)
            fx *= sx; cx *= sx
            fy *= sy; cy *= sy

        c2w = camera.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        c2w = c2w.to(model.means.device, dtype=model.means.dtype)
        R = c2w[:3, :3]; t = c2w[:3, 3]
        means_cam = (model.means - t[None, :]) @ R
        depths_g = -means_cam[:, 2]
        in_front_of_cam = depths_g > 1e-6

        safe_d = torch.where(in_front_of_cam, depths_g, torch.ones_like(depths_g))
        u = fx * (means_cam[:, 0] / safe_d) + cx
        v = fy * (-means_cam[:, 1] / safe_d) + cy
        u_idx = u.round().long().clamp(0, W - 1)
        v_idx = v.round().long().clamp(0, H - 1)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front_of_cam

        comp = component_mask
        if comp.ndim == 3 and comp.shape[-1] == 1:
            comp = comp[..., 0]
        comp = (comp > 0.5).to(model.means.device)
        # Resize comp to the depth grid if it landed at a different scale.
        if comp.shape[-2:] != (H, W):
            comp = TF.interpolate(
                comp.float().unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.bool)

        sensor_depth_at = depth[v_idx, u_idx]
        in_region = comp[v_idx, u_idx] & in_bounds
        has_valid_depth = sensor_depth_at > 0
        in_front = depths_g < (sensor_depth_at - float(depth_tol_m))

        instance_ids = model.object_instance_ids.squeeze(-1).to(in_front.device)
        eligible = (instance_ids == 0) | (instance_ids == 999)

        to_delete = in_region & has_valid_depth & in_front & eligible
        indices = torch.nonzero(to_delete, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return 0
        return model.delete_gaussian_indices(indices)

    @torch.no_grad()
    def _feedforward_cull_then_reclean_cdn(
        self, camera, batch, cdn_clean, gt_depth, *, frame_name=None, prerendered_obj_mask=None,
    ):
        """Cull-before-decode: delete every eligible Gaussian sitting in front
        of the true sensor surface within the changed (CDN) region, then
        recompute CDN on the freshly-culled scene.

        Eligible = ``object_instance_ids in {0, 999}`` — both the original
        point-cloud (0) AND previously-inserted feedforward Gaussians (999);
        tracked objects are never touched. This is the SAME test the
        per-component cull runs, applied once over the union CDN region so a
        stale FF occluder is removed regardless of which component it sits in.

        Returns ``(cdn_clean, n_culled)``. When ``n_culled > 0`` the returned
        CDN is freshly recomputed + re-cleaned (the cull may have cleared the
        change, in which case the caller skips the decoder). When nothing was
        culled the input ``cdn_clean`` is returned unchanged.
        """
        if not self.config.feedforward_cull_before_decode:
            return cdn_clean, 0
        n_culled = 0
        with self._viser_lock_ctx():
            n_culled = self._feedforward_cull_in_front_of_depth(
                camera, cdn_clean, gt_depth,
                depth_tol_m=float(self.config.feedforward_cull_in_front_depth_tol_m),
            )
        if n_culled <= 0:
            return cdn_clean, 0
        # The cull mutated the scene — re-render CDN so it reflects the
        # freshly-removed occluders, then re-subtract the object footprint.
        cdn_new = self._compute_tick_cdn(camera, batch)
        if cdn_new is None:
            return cdn_clean, n_culled
        # Reuse the same object mask as the initial clean + the debug save (the
        # cull never deletes tracked-object Gaussians, so it's still valid) —
        # don't re-render a fresh, unsaved one here.
        cdn_clean_new = self._feedforward_clean_cdn(
            camera, cdn_new, frame_name=frame_name,
            prerendered_obj_mask=prerendered_obj_mask,
        )
        return cdn_clean_new, n_culled

    # ---- AnySplat feedforward path ----

    def _start_anysplat_persistent_worker(self) -> None:
        """Acquire the long-lived AnySplat worker. Called once when the
        anysplat path is first hit (or at subclass-controlled init time).

        Adopt-first: the live capture session (bootstrap_live.sh with
        DGS_EAGER_ANYSPLAT=1) pre-spawns a detached FIFO-mode worker right
        after SAM3D finishes, so its ~17 s model load overlaps static
        training. If that worker is alive (or still loading — we wait), we
        adopt it for ~0 s startup; otherwise spawn fresh as before."""
        if self._anysplat_persistent_worker is not None:
            return
        from .utils.anysplat_decode import PersistentAnysplatWorker

        fifo_dir = Path(self.config.datamanager.data) / ".anysplat_worker"
        try:
            t0 = time.time()
            adopted = PersistentAnysplatWorker.adopt(fifo_dir, wait_ready_timeout_s=60.0)
        except Exception as exc:
            CONSOLE.log(f"[anysplat] adoption attempt failed: {exc}; spawning fresh")
            adopted = None
        if adopted is not None:
            self._anysplat_persistent_worker = adopted
            CONSOLE.log(
                f"[anysplat] ADOPTED pre-spawned worker in {time.time()-t0:.1f}s "
                f"(its load of {adopted.load_seconds:.1f}s already ran during capture/training)"
            )
            try:
                from .utils import timing_ledger as _tl
                _tl.record(self.config.datamanager.data, "teleop_init",
                           "AnySplat adopt (pre-warmed)", "io", t0, time.time())
            except Exception:
                pass
            return

        try:
            t0 = time.time()
            CONSOLE.log("[anysplat] spawning persistent worker (loading model in subprocess)...")
            self._anysplat_persistent_worker = PersistentAnysplatWorker(
                conda_env=str(self.config.feedforward_anysplat_conda_env),
                startup_timeout_s=120.0,
            )
            CONSOLE.log(
                f"[anysplat] persistent worker ready in {time.time()-t0:.1f}s "
                f"(worker reported load = {self._anysplat_persistent_worker.load_seconds:.1f}s)"
            )
        except Exception as exc:
            CONSOLE.log(f"[anysplat] persistent worker spawn FAILED: {exc}; "
                        f"will fall back to per-call subprocess spawn")
            self._anysplat_persistent_worker = None

    def _resolve_anysplat_context_image_paths(self, target_frame_idx: int) -> tuple[list[Path], list[int]]:
        """Return ([target_image_path], [target_frame_idx]).

        AnySplat is invoked with a single frame (the target). Multi-frame
        context was tested and gave no forward-time benefit while producing
        an inconsistent (often noisier) output cloud; the previous
        ``feedforward_anysplat_context_frames`` knob was removed.

        TODO(phase-3-stage-D): recorded-only because it reads
        ``dataset.image_filenames``. Live subclass overrides this with the
        in-memory RGB frame path."""
        try:
            ds = self.datamanager.dynamic_manager.train_dataset
        except AttributeError:
            return [], []
        target_path = Path(ds.image_filenames[target_frame_idx])
        return [target_path], [int(target_frame_idx)]

    def _scene_c2w_for_frame(self, frame_idx: int) -> np.ndarray:
        """Look up the post-camera-optimizer c2w (4x4) for a recorded dynamic frame.

        TODO(phase-3-stage-D): recorded-only. Live subclass should override."""
        ds = self.datamanager.dynamic_manager.train_dataset
        cam = ds.cameras[frame_idx : frame_idx + 1].to(self.device)
        c2w = cam.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        if c2w.shape == (3, 4):
            bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w.device, dtype=c2w.dtype)
            c2w = torch.cat([c2w, bottom], dim=0)
        return c2w.detach().cpu().numpy().astype(np.float32)

    def _run_feedforward_anysplat(self, target_frame: TrackerFrame, mode_label: str,
                                   prerendered_obj_mask=None) -> None:
        """Main-thread dispatch for the AnySplat FF path.

        Runs the fast prep (~10 ms: CDN clean, component selection, snapshot
        per-frame inputs) on the main thread, then hands the slow part
        (worker.inference + reproject + cull + insert, ~200 ms) to a daemon
        thread. The tracker loop is freed to keep ticking; the bg thread
        takes ``model_lock`` only during the actual cull + insert mutations
        (~5 ms total). At most one bg call is in flight; new dispatches that
        find the slot lock held are skipped (with a log line)."""
        from .utils.active_mask import select_top_n_components_filtered

        call_id = self._feedforward_call_counter
        self._feedforward_call_counter += 1

        t_call0 = time.time()
        camera = target_frame["camera"]
        frame_idx = int(target_frame["frame_idx"])
        cdn = target_frame["cdn"]

        # Render the tracked-object mask ONCE and reuse it for every cleaning
        # step (initial clean, post-cull re-clean) AND the _ff_debug dump — so
        # the objmask saved to disk is EXACTLY the mask that cleaned the CDN.
        # Previously each step re-rendered render_object_mask independently at a
        # different model state (the post-cull re-clean's render was never the
        # saved one), so the saved objmask could disagree with what was used.
        if prerendered_obj_mask is None:
            try:
                prerendered_obj_mask = self._render_object_mask_cached(camera)
            except Exception:
                prerendered_obj_mask = None

        t0 = time.time()
        try:
            frame_name_for_cdn = self.datamanager.get_dynamic_frame_name(frame_idx)
        except Exception:
            frame_name_for_cdn = None
        cdn_clean = self._feedforward_clean_cdn(
            camera, cdn, frame_name=frame_name_for_cdn,
            prerendered_obj_mask=prerendered_obj_mask,
        )
        self._timing["FF.1_cdn_clean"].append(time.time() - t0)

        # Need depth + batch up-front for the cull-before-decode pass (the
        # later AnySplat block reuses them).
        batch = target_frame["batch"]
        if batch is None:
            CONSOLE.log(f"[anysplat] frame {frame_idx} has no batch — skip")
            return
        gt_depth = self.model._get_gt_depth(batch)
        if gt_depth is None:
            CONSOLE.log(f"[anysplat] frame {frame_idx} has no depth — skip")
            return

        # PRE-cull render for the debug dump. DEBUG-ONLY: this is a full GPU
        # render on the MAIN tick thread that contends with the XFeat tracker
        # kernels, so it is skipped entirely unless save_debug_images is set
        # (otherwise every FF call stalls the tracker with 1-2 extra renders).
        save_dbg = bool(self.config.save_debug_images)
        pre_cull_render = None
        if save_dbg:
            try:
                pre_cull_render = self._render_from_camera(camera).get("rgb")
            except Exception:
                pre_cull_render = None

        # Cull-before-decode: drop in-front occluders (original + prior FF
        # Gaussians) over the CDN region, recompute CDN. If culling alone
        # clears the change, components below is empty and we skip the
        # (expensive) AnySplat forward + reproject entirely.
        t0 = time.time()
        cdn_clean, n_pre_culled = self._feedforward_cull_then_reclean_cdn(
            camera, batch, cdn_clean, gt_depth, frame_name=frame_name_for_cdn,
            prerendered_obj_mask=prerendered_obj_mask,
        )
        self._timing["FF.1b_cull_before_decode"].append(time.time() - t0)
        if n_pre_culled > 0:
            self._refresh_viser_direct_after_feedforward()
        # POST-cull render = the scene state the re-rendered (clean) CDN used
        # (debug-only; only re-render when the cull actually changed the scene).
        post_cull_render = pre_cull_render
        if save_dbg and n_pre_culled > 0:
            try:
                post_cull_render = self._render_from_camera(camera).get("rgb")
            except Exception:
                post_cull_render = pre_cull_render

        # --- Debug-image dump (debug-only): per-call ordered set for raw→clean.
        # gripper / object / real / rendered / rerendered-after-cull / raw / clean.
        if save_dbg:
            try:
                self._save_ff_debug_images(
                    call_id=call_id, frame_idx=frame_idx, camera=camera,
                    cdn_raw=cdn, cdn_clean=cdn_clean,
                    prerendered_obj_mask=prerendered_obj_mask, target_frame=target_frame,
                    pre_cull_render=pre_cull_render, post_cull_render=post_cull_render,
                )
            except Exception as exc:
                CONSOLE.log(f"[ff-debug] dump failed call={call_id}: {exc}")

        t0 = time.time()
        if mode_label == "oneshot":
            components = select_top_n_components_filtered(
                cdn_clean,
                n=int(self.config.feedforward_top_n_components),
                area_ratio=float(self.config.feedforward_dominant_area_ratio),
                min_area=1,
            )
        else:
            components = select_top_n_components_filtered(cdn_clean, n=256, area_ratio=0.0, min_area=1)
        self._timing["FF.2_component_select"].append(time.time() - t0)

        if not components:
            CONSOLE.log(
                f"[anysplat] {mode_label} call={call_id} step={self._dynamic_step_counter} "
                f"frame={frame_idx} no components "
                f"(pre-decode cull removed {n_pre_culled}; AnySplat skipped)"
            )
            return

        image_paths, _ = self._resolve_anysplat_context_image_paths(frame_idx)
        if len(image_paths) < 1:
            CONSOLE.log(f"[anysplat] call={call_id} no target image; skip")
            return
        image_paths = [image_paths[0]]

        sensor_depth_np = gt_depth.detach().cpu().numpy().astype(np.float32)
        if sensor_depth_np.ndim == 3:
            sensor_depth_np = sensor_depth_np[..., 0]

        def _scalar(x):
            return float(x.detach().cpu().reshape(-1)[0].item()) if isinstance(x, torch.Tensor) else float(x)
        scene_intr = {
            "w":    int(_scalar(camera.width)),
            "h":    int(_scalar(camera.height)),
            "fl_x": _scalar(camera.fx),
            "fl_y": _scalar(camera.fy),
            "cx":   _scalar(camera.cx),
            "cy":   _scalar(camera.cy),
        }
        scene_c2w_np = self._scene_c2w_for_frame(frame_idx)
        if scene_c2w_np.shape == (3, 4):
            scene_c2w_np = np.vstack([scene_c2w_np, [[0, 0, 0, 1]]]).astype(np.float64)
        else:
            scene_c2w_np = scene_c2w_np.astype(np.float64)

        # Adaptive AnySplat crop: AnySplat only sees a square, so pick the square
        # to ENCOMPASS the change mask (+50 px) instead of the image centre. One
        # window, or two when the change is wider than the image short side.
        cdn_np = cdn_clean.detach().cpu().numpy() if torch.is_tensor(cdn_clean) else np.asarray(cdn_clean)
        if cdn_np.ndim == 3:
            cdn_np = cdn_np[..., 0]
        cdn_np = cdn_np > 0.5
        crop_windows = self._anysplat_crop_windows(cdn_np, int(scene_intr["w"]), int(scene_intr["h"]))
        if not crop_windows:
            CONSOLE.log(f"[anysplat] call={call_id} empty change mask after clean; skip")
            return

        # Off-thread the slow part. If a previous AnySplat call is still
        # running we skip this dispatch instead of queueing (the FF
        # min-gap config usually prevents this but cold calls can exceed
        # the gap; queueing would just stack stale frames).
        if not self._anysplat_slot_lock.acquire(blocking=False):
            CONSOLE.log(
                f"[anysplat] {mode_label} call={call_id} step={self._dynamic_step_counter} "
                f"frame={frame_idx} skipped — previous FF call still in flight"
            )
            return

        bg_args = dict(
            t_call0=t_call0,
            call_id=call_id,
            mode_label=mode_label,
            frame_idx=frame_idx,
            camera=camera,
            components=components,
            source_image_path=str(image_paths[0]),
            crop_windows=crop_windows,
            cdn_np=cdn_np,
            gt_depth=gt_depth,
            sensor_depth_np=sensor_depth_np,
            scene_intr=scene_intr,
            scene_c2w_np=scene_c2w_np,
        )
        threading.Thread(
            target=self._anysplat_bg_run, args=(bg_args,),
            daemon=True, name=f"anysplat-bg-{call_id}",
        ).start()

    def _anysplat_crop_windows(self, change_mask_np, W: int, H: int, pad_px: int = 50):
        """Square scene crop window(s) that ENCOMPASS the change mask, for AnySplat.

        AnySplat only sees a square (its process_image resizes whatever square we
        give it to 448×448). So choose the square to cover the change region:
        ``size = max(bbox_w, bbox_h) + 2·pad_px`` at the change mask's natural
        scale — NOT forced to 448 (AnySplat up/down-samples internally; the
        reproject maps 448→scene via the window). One window normally; TWO
        horizontally-tiled windows ONLY when the change bbox is wider than the
        image short side (one square physically can't cover it). Capped at 2.

        Returns ``[(left, top, size), ...]`` in scene pixels.
        """
        m = change_mask_np
        if m.ndim == 3:
            m = m[..., 0]
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            return []
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        max_size = min(W, H)
        size = max(bw, bh) + 2 * int(pad_px)
        if size <= max_size:
            size = max(16, size)
            left = max(0, min(int(round(cx - size / 2.0)), W - size))
            top  = max(0, min(int(round(cy - size / 2.0)), H - size))
            return [(left, top, size)]
        # Padded bbox bigger than the largest square that fits → cap at short side.
        size = max_size
        top = max(0, min(int(round(cy - size / 2.0)), H - size))
        if bw <= size:
            left = max(0, min(int(round(cx - size / 2.0)), W - size))
            return [(left, top, size)]
        # Change wider than one square → two windows covering the left + right ends.
        left1 = max(0, min(x0 - pad_px, W - size))
        left2 = max(0, min(x1 + pad_px - size + 1, W - size))
        wins = [(left1, top, size)]
        if abs(left2 - left1) > 4:
            wins.append((left2, top, size))
        return wins

    def _anysplat_bg_run(self, args: dict) -> None:
        """Background worker for the AnySplat FF path. Calls the persistent
        subprocess (~150 ms), loads the IPC blob (~15 ms), reprojects each
        component (~45 ms), then takes ``model_lock`` briefly to cull + insert.
        Always releases ``_anysplat_slot_lock`` on exit so the next FF call
        can dispatch."""
        from .utils.anysplat_decode import (
            icp_refine_scene_c2w,
            reproject_anysplat_to_scene,
            run_anysplat_subprocess,
        )

        t_call0          = args["t_call0"]
        call_id          = args["call_id"]
        mode_label       = args["mode_label"]
        frame_idx        = args["frame_idx"]
        camera           = args["camera"]
        components        = args["components"]
        source_image_path = args["source_image_path"]
        crop_windows      = args["crop_windows"]
        cdn_np            = args["cdn_np"]
        gt_depth          = args["gt_depth"]
        sensor_depth_np   = args["sensor_depth_np"]
        scene_intr        = args["scene_intr"]
        scene_c2w_np      = args["scene_c2w_np"]

        try:
            import pickle
            import cv2 as _cv2

            # --- Frustum-cull scene cloud + ICP-refine scene_c2w (ONCE per FF call, on GPU) ---
            # The frustum cull and ICP are component-agnostic: align the whole visible
            # sensor cloud against the whole visible scene cloud. Unchanged regions agree
            # by definition and pull the pose toward the right answer; changed regions
            # are too few to bias it.
            scene_c2w_refined = scene_c2w_np
            if self.config.feedforward_anysplat_icp_refine:
                t_fc0 = time.time()
                with self._viser_lock_ctx():
                    means_all_t = self.model.gauss_params["means"].detach()  # (N, 3) on device
                # GPU frustum cull
                dev = means_all_t.device
                c2w_t = torch.as_tensor(scene_c2w_np, dtype=means_all_t.dtype, device=dev)
                R_sc_t = c2w_t[:3, :3]
                t_sc_t = c2w_t[:3, 3]
                p_cam_t = (means_all_t - t_sc_t) @ R_sc_t  # equiv to Rᵀ(p - t)
                z_cam_t = p_cam_t[:, 2]
                in_front_t = z_cam_t < -1e-3
                safe_z_t = torch.where(in_front_t, -z_cam_t, torch.ones_like(z_cam_t))
                fx_sc = float(scene_intr["fl_x"]); fy_sc = float(scene_intr["fl_y"])
                cx_sc = float(scene_intr["cx"]);   cy_sc = float(scene_intr["cy"])
                W_sc  = int(scene_intr["w"]);      H_sc  = int(scene_intr["h"])
                u_t = fx_sc * (p_cam_t[:, 0] / safe_z_t) + cx_sc
                v_t = fy_sc * (-p_cam_t[:, 1] / safe_z_t) + cy_sc
                in_image_t = (u_t >= 0) & (u_t < W_sc) & (v_t >= 0) & (v_t < H_sc)
                visible_t = in_front_t & in_image_t
                target_xyz_t = means_all_t[visible_t]
                self._timing["FF.3a.frustum_cull"].append(time.time() - t_fc0)

                # Camera-away-from-scene guard: if too few scene gaussians are
                # visible from the live camera, the AnySplat output would be
                # placed in a region with no scene context. Skip the call.
                n_visible = int(visible_t.sum().item())
                min_visible = int(self.config.feedforward_anysplat_min_visible_scene_points)
                if min_visible > 0 and n_visible < min_visible:
                    CONSOLE.log(
                        f"[anysplat] call={call_id} skipped: frustum kept "
                        f"{n_visible}/{int(visible_t.shape[0])} < {min_visible} "
                        f"(camera turned away from scene)"
                    )
                    return

                # ICP runs in Open3D Tensor API on CUDA when available.
                t_icp0 = time.time()
                scene_c2w_refined, icp_info = icp_refine_scene_c2w(
                    sensor_depth_m=sensor_depth_np, scene_c2w=scene_c2w_np,
                    scene_intr=scene_intr,
                    target_xyz_gpu=target_xyz_t,  # GPU tensor in
                    max_iters=int(self.config.feedforward_anysplat_icp_max_iters),
                    max_dist_m=float(self.config.feedforward_anysplat_icp_max_dist_m),
                )
                self._timing["FF.3a.icp_refine"].append(time.time() - t_icp0)
                CONSOLE.log(
                    f"[anysplat] call={call_id} frustum kept "
                    f"{int(visible_t.sum().item())}/{int(visible_t.shape[0])}, "
                    f"icp ran={icp_info.get('ran', False)} fitness={icp_info.get('fitness', float('nan')):.3f} "
                    f"rmse={icp_info.get('inlier_rmse', float('nan')):.4f}m "
                    f"(frustum {(time.time()-t_fc0)*1000-(time.time()-t_icp0)*1000:.1f} ms, "
                    f"icp {(time.time()-t_icp0)*1000:.1f} ms)"
                )

            total_inserted = 0
            total_culled = 0
            H_any, W_any = 448, 448
            # Decode per CROP WINDOW (1, or 2 when the change is wider than the
            # image short side). Each window is a square scene sub-region that
            # ENCOMPASSES the change mask (+pad); AnySplat resizes it to 448 and
            # reproject maps the 448 pixels back via the window. Every window is
            # filtered by the FULL change mask (cdn_np) and union-deduped below,
            # so overlapping windows do not double-insert.
            src_img = _cv2.imread(str(source_image_path))
            if src_img is None:
                CONSOLE.log(f"[anysplat] call={call_id} could not read {source_image_path}; skip")
                return
            per_component_decoded: list[dict] = []
            for wi, win in enumerate(crop_windows):
                left, top, size = int(win[0]), int(win[1]), int(win[2])
                crop_png = Path(f"/dev/shm/anysplat_crop_{os.getpid()}_{wi}.png")
                _cv2.imwrite(str(crop_png), src_img[top:top + size, left:left + size])
                out_npz = Path(f"/dev/shm/anysplat_ipc_{os.getpid()}_{wi}.npz")
                t0 = time.time()
                worker_timings: dict = {}
                try:
                    if self._anysplat_persistent_worker is not None:
                        worker_timings = self._anysplat_persistent_worker.inference(
                            [crop_png], out_npz,
                            timeout_s=float(self.config.feedforward_anysplat_worker_timeout_s),
                        )
                    else:
                        run_anysplat_subprocess(
                            [crop_png], out_npz,
                            conda_env=str(self.config.feedforward_anysplat_conda_env),
                            timeout_s=float(self.config.feedforward_anysplat_worker_timeout_s),
                        )
                except Exception as exc:
                    CONSOLE.log(f"[anysplat] call={call_id} win={wi} worker FAILED: {exc}")
                    continue
                self._timing["FF.3a_anysplat_inference"].append(time.time() - t0)
                for k_in, k_out in (
                    ("t_ipc_send_ms",    "FF.3a.ipc_send"),
                    ("t_images_load_ms", "FF.3a.images_load"),
                    ("t_forward_ms",     "FF.3a.forward"),
                    ("t_convert_ms",     "FF.3a.convert_to_numpy"),
                    ("t_npz_save_ms",    "FF.3a.npz_save"),
                    ("t_ipc_wait_ms",    "FF.3a.ipc_wait"),
                ):
                    v = worker_timings.get(k_in)
                    if v is not None:
                        self._timing[k_out].append(float(v) / 1000.0)
                t_load0 = time.time()
                with open(out_npz, "rb") as f:
                    data = pickle.load(f)
                self._timing["FF.3a.npz_load"].append(time.time() - t_load0)

                t0 = time.time()
                decoded = reproject_anysplat_to_scene(
                    means_canonical=data["means_canonical"], log_scales=data["log_scales"],
                    quats_wxyz=data["quats_wxyz"], opacity_logits=data["opacity_logits"],
                    features_dc=data["features_dc"], features_rest=data["features_rest"],
                    pred_c2w_0=data["pred_extrinsic_c2w"][0], pred_K_norm=data["pred_intrinsic_norm"][0],
                    pred_image_hw=(H_any, W_any),
                    sensor_depth_m=sensor_depth_np, scene_c2w=scene_c2w_refined,
                    scene_intr=scene_intr,
                    opacity_min=float(self.config.feedforward_anysplat_min_opacity),
                    component_mask=cdn_np,
                    scene_crop=(left, top, size),
                    voxel_dedup_m=None,  # dedup is done ONCE across all windows below
                    scale_multiplier=float(self.config.feedforward_anysplat_scale_multiplier),
                )
                self._timing["FF.3b_anysplat_reproject"].append(time.time() - t0)
                if int(decoded["xyz"].shape[0]) > 0:
                    per_component_decoded.append(decoded)

            # --- One union-wide voxel dedup on GPU, then one insert ---
            if per_component_decoded:
                t_dd0 = time.time()
                # Concatenate all components into single GPU tensors. From_numpy
                # + .to(self.device) handles the H2D copy in one fused call per key.
                def _cat(key: str) -> torch.Tensor:
                    return torch.cat(
                        [torch.from_numpy(d[key]).to(self.device, non_blocking=True)
                         for d in per_component_decoded],
                        dim=0,
                    )
                xyz_g           = _cat("xyz")            # (N, 3)
                features_dc_g   = _cat("features_dc")
                features_rest_g = _cat("features_rest")
                opacities_g     = _cat("opacities")
                scales_g        = _cat("scales")
                quats_g         = _cat("quats")

                vdm_near = float(self.config.feedforward_anysplat_voxel_dedup_m)
                vdm_far  = float(self.config.feedforward_anysplat_voxel_dedup_far_m)
                near_r   = float(self.config.feedforward_anysplat_dedup_near_radius_m)

                def _voxel_keep_idx(xyz_sub: torch.Tensor, vsize: float) -> torch.Tensor:
                    """Per-voxel first-index keeper on GPU. Returns indices into xyz_sub."""
                    if vsize <= 0.0 or xyz_sub.shape[0] == 0:
                        return torch.arange(xyz_sub.shape[0], device=xyz_sub.device)
                    cell = torch.floor(xyz_sub / vsize).to(torch.int64)
                    OFF = 1 << 20  # ±1 km at 1 mm voxels, ±5 km at 5 mm
                    key64 = ((cell[:, 0] + OFF) << 42) | ((cell[:, 1] + OFF) << 21) | (cell[:, 2] + OFF)
                    sort_keys, sort_perm = torch.sort(key64, stable=True)
                    first_mask = torch.ones_like(sort_keys, dtype=torch.bool)
                    if sort_keys.numel() > 1:
                        first_mask[1:] = sort_keys[1:] != sort_keys[:-1]
                    keep = sort_perm[first_mask]
                    keep, _ = torch.sort(keep)
                    return keep

                n_before = int(xyz_g.shape[0])
                if (vdm_near > 0.0 or vdm_far > 0.0) and n_before > 0:
                    # Split by distance to the (already-ICP-refined) camera centre.
                    # scene_c2w_refined[:3,3] is the camera world position.
                    cam_t = torch.as_tensor(scene_c2w_refined[:3, 3], device=xyz_g.device, dtype=xyz_g.dtype)
                    d2 = ((xyz_g - cam_t[None, :]) ** 2).sum(dim=-1)
                    is_near = d2 <= (near_r * near_r)

                    near_idx_all = torch.nonzero(is_near, as_tuple=False).squeeze(-1)
                    far_idx_all  = torch.nonzero(~is_near, as_tuple=False).squeeze(-1)

                    near_keep_local = _voxel_keep_idx(xyz_g[near_idx_all], vdm_near)
                    far_keep_local  = _voxel_keep_idx(xyz_g[far_idx_all],  vdm_far)
                    keep_idx = torch.cat([near_idx_all[near_keep_local],
                                          far_idx_all[far_keep_local]], dim=0)
                    keep_idx, _ = torch.sort(keep_idx)

                    n_near_in = int(near_idx_all.numel())
                    n_far_in  = int(far_idx_all.numel())
                    n_near_out = int(near_keep_local.numel())
                    n_far_out  = int(far_keep_local.numel())

                    xyz_g           = xyz_g[keep_idx]
                    features_dc_g   = features_dc_g[keep_idx]
                    features_rest_g = features_rest_g[keep_idx]
                    opacities_g     = opacities_g[keep_idx]
                    scales_g        = scales_g[keep_idx]
                    quats_g         = quats_g[keep_idx]
                    n_after = int(xyz_g.shape[0])
                else:
                    n_near_in = n_near_out = n_far_in = n_far_out = 0
                    n_after = n_before
                self._timing["FF.4b_voxel_dedup"].append(time.time() - t_dd0)

                t_ins0 = time.time()
                with self._viser_lock_ctx():
                    inserted_ids = self.model.insert_inpaint_gaussians(
                        xyz=xyz_g,
                        features_dc=features_dc_g,
                        features_rest=features_rest_g,
                        opacities=opacities_g,
                        scales=scales_g,
                        quats=quats_g,
                        instance_id=999,
                    )
                self._timing["FF.5_insert"].append(time.time() - t_ins0)
                self._viser_direct_register_ff_insert(inserted_ids)
                total_inserted += int(inserted_ids.numel())
                CONSOLE.log(
                    f"[anysplat] call={call_id} dedup {n_before}->{n_after} "
                    f"(near {n_near_in}->{n_near_out} @ {vdm_near*1000:.1f} mm, "
                    f"far {n_far_in}->{n_far_out} @ {vdm_far*1000:.1f} mm, "
                    f"r={near_r:.2f} m, {(time.time()-t_dd0)*1000:.1f} ms gpu)"
                )

            total_per_call = time.time() - t_call0
            self._timing["FF.6_total_per_call"].append(total_per_call)
            obj_count = int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())
            ins_count = int((self.model.inserted_flags.squeeze(-1) > 0.5).sum().item())
            def _last_ms(key: str) -> float:
                arr = self._timing.get(key, [])
                return arr[-1] * 1000.0 if arr else 0.0
            CONSOLE.log(
                f"[anysplat] {mode_label} call={call_id} step={self._dynamic_step_counter} "
                f"frame={frame_idx} components={len(components)} windows={len(crop_windows)} "
                f"inserted={total_inserted} culled={total_culled} total_ms={total_per_call*1000:.0f} "
                f"object_flags={obj_count} inserted_flags={ins_count} "
                f"total_gauss={self.model.num_points} "
                f"| cdn={_last_ms('FF.1_cdn_clean'):.0f} "
                f"precull={_last_ms('FF.1b_cull_before_decode'):.0f} "
                f"comp_sel={_last_ms('FF.2_component_select'):.0f} "
                f"anysplat_inf={_last_ms('FF.3a_anysplat_inference'):.0f} "
                f"reproj={_last_ms('FF.3b_anysplat_reproject'):.0f} "
                f"cull={_last_ms('FF.4_crop_and_delete'):.0f} "
                f"insert={_last_ms('FF.5_insert'):.0f}"
            )
            if "FF.3a.ipc_send" in self._timing:
                CONSOLE.log(
                    f"[anysplat]   ph(ms): "
                    f"snd={_last_ms('FF.3a.ipc_send'):.1f} "
                    f"img={_last_ms('FF.3a.images_load'):.0f} "
                    f"fwd={_last_ms('FF.3a.forward'):.0f} "
                    f"cvt={_last_ms('FF.3a.convert_to_numpy'):.0f} "
                    f"sav={_last_ms('FF.3a.npz_save'):.0f} "
                    f"wait={_last_ms('FF.3a.ipc_wait'):.0f} "
                    f"ld={_last_ms('FF.3a.npz_load'):.0f}"
                )
        finally:
            self._anysplat_slot_lock.release()
