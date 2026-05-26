from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image, ImageDraw
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig
from .utils import (
    CoTrackerMotionEstimator,
    DynamicKeyframeFilter,
    OptimFrame,
    OptimPool,
    build_change_mask,
    dilate_binary_mask,
    extract_projected_centers_and_radii,
    load_sam3d_gaussian_ply,
    load_sam3d_rotation_wxyz,
    register_and_fuse_sam3d_object,
    save_point_cloud,
)
from .utils.sam3_segmentation import load_sam3_masks, run_sam3_subprocess
from .utils.sam3d import (
    get_sam3d_output_paths,
    resolve_sam3d_pose_path,
    run_sam3d_multi_object_subprocess,
    sam3d_pose_has_rotation,
)


@dataclass
class DynamicGSPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: DynamicGSPipeline)

    datamanager: DynamicGSDataManagerConfig = field(default_factory=DynamicGSDataManagerConfig)
    model: DynamicGSModelConfig = field(default_factory=DynamicGSModelConfig)

    static_num_steps: int = 3000
    dynamic_steps_per_frame: int = 300
    save_debug_images: bool = False
    """If False, skip the per-frame debug PNG saves (D0.1f / D0.9 /
    DN.8) and the CoTracker debug PNG + motion log (DN.3f). These are
    pure disk I/O and not part of the tracking or change-detection
    critical path; disabling them removes ~210ms from D0, ~600ms from
    each recorded dynamic frame, and ~170ms from each live tracker
    tick."""

    enable_dynamic_keyframe_filter: bool = True
    """ORB-SLAM-style greedy keyframe filter on the dynamic dataset
    applied BEFORE any change-mask compute. Frame i is rejected iff
    some accepted keyframe j is within both τ_t in translation AND
    τ_r in rotation. Rejected frames cost zero — the trainer's step
    schedule indirects through ``_accepted_dynamic_frames``, so
    ``max_num_iterations`` shrinks to ``static_num_steps + K_accepted ·
    dynamic_steps_per_frame``. The same stateful filter is the API
    for future live ingestion (call ``accept(c2w)`` per arrival)."""
    dynamic_keyframe_translation_m: float = 0.01
    """τ_t in metric meters (poses are unscaled, ``auto_scale_poses=False``)."""
    dynamic_keyframe_rotation_deg: float = 20.0
    """τ_r in degrees, geodesic SO(3) distance."""

    tracker_tick_every_steps: int = 3
    """Tracker cadence for **recorded mode only**: a tracker tick
    (FP track + maybe push to optim pool) fires every N optim steps.
    With ~58 ms/step, N=3 fakes a ~5.7 Hz incoming-frame rate — close
    to the 5 Hz dataset capture rate. Decoupling is step-based
    (deterministic) rather than wall-clock-based, so optim doesn't idle
    when it's faster than the fake camera. The first dynamic step
    always fires a tick (D0 bootstrap). **Live mode ignores this and
    ticks every step** — the ROS supply is the real rate limiter, and
    the dedup-return makes over-polling free."""
    optim_pool_capacity: int = 15
    """Max number of accepted frames simultaneously queued for
    optimization. On overflow, the oldest entry is dropped (mirrors
    live behavior: if optim falls behind tracking, the queue forgets
    the oldest backlog rather than growing unboundedly)."""
    optim_pool_max_epochs: int = 10
    """Per-frame optimization budget. When ``epochs_used >= max``, the
    frame is evicted from the pool regardless of loss."""
    optim_pool_loss_relative_threshold: float = 0.3
    """Eviction threshold expressed as a fraction of the frame's
    *first-iteration* loss. Once ``last_loss < initial_loss * 0.3``,
    the frame is evicted. Relative semantics avoid scene-dependent
    absolute tuning."""
    optim_pool_min_change_pixels: int = 500
    """A captured CDN must have at least this many active pixels to be
    pushed to the optim pool. Below this, the change region is
    dominated by specular shimmer / sub-pixel JPEG noise and isn't
    worth a 50-step optimization budget. ~0.3% of a 400×400 frame."""

    enable_static_convergence_check: bool = True
    """Enable the MS-SSIM-based early static→dynamic transition. Once
    the rendered scene matches the GT for all static keyframes (change
    pixels per image below threshold), the static phase exits early
    even if ``static_num_steps`` is not yet reached."""
    static_convergence_first_check_step: int = 300
    """First step at which the convergence check is run. Sits 100
    steps after full-resolution training kicks in (full res reached
    at step ``resolution_schedule * num_downscales = 100 * 2 = 200``),
    giving the scene some full-res training time before the first
    metric is sampled. With the 1000-step static budget this leaves
    room for ~7 checks before the hard cap."""
    static_convergence_check_every: int = 400
    """Cadence of subsequent convergence checks while in the static
    phase."""
    static_convergence_rgb_threshold: float = 0.1
    """MS-SSIM dissimilarity threshold used **only by the static
    convergence check**. Independent of the dynamic-phase
    ``change_mask_rgb_threshold`` so the two concerns can be tuned
    separately. Lower = more sensitive (more pixels flagged as
    "different"), higher = stricter. At 0.1 a pixel counts as
    different if MSSIM dissimilarity exceeds 10 % — this matches the
    dynamic-phase default and keeps the static metric on the same
    sensitivity scale."""
    static_convergence_max_change_ratio: float = 0.01
    """Average per-image change-pixel ratio below which the static
    phase is considered converged. 0.01 = 1 %: as long as more than
    1 % of valid pixels (the average accepted-keyframe, mask-aware)
    are flagged different above ``static_convergence_rgb_threshold``,
    training continues. Below 1 %, transition to the dynamic phase."""

    live: bool = False
    """When True, run the interactive ROS-driven session before
    nerfstudio constructs the datamanager: prompt the user, capture
    SAM3 + SAM3D outputs, record static views to disk, build the
    SfM init PLY, then proceed with the standard pipeline against
    that just-recorded dataset. The dynamic phase reads frames live
    from rospy instead of advancing through a recorded dataset.
    Default False keeps recorded-mode behavior fully untouched."""

    disable_dynamic_optimization: bool = True
    """When True, the dynamic phase runs **tracking only**: tracker
    ticks fire and the rigid transform is applied to object
    Gaussians, but the change-mask compute, optim-pool push, pool
    round-robin pick, and per-step loss/backward are all skipped.
    Use this to iterate on tracking robustness in real time without
    the optim work taking GPU time. Object Gaussians still move
    (driven by FP / CoTracker); scene Gaussians stay frozen."""

    save_live_optim_debug: bool = True
    """When True, save the rendered image, live GT, and the effective
    loss mask (with overlay) for each live optim step into
    ``<data_root>/dynamic_scene/debug/live_optim/``. Throttled by
    ``save_live_optim_debug_every`` to avoid filling disk."""
    save_live_optim_debug_every: int = 25
    """How often (in optim steps) to dump the live-optim debug images
    when ``save_live_optim_debug=True``. 1 = every step (heavy)."""
# Fraction of the splat-rasterized object mask kept after distance-transform
# erosion before passing to the tracker. 0.85 = keep the inner 85% of the
# area, drop the boundary 15% — the low-opacity halo where keypoints would
# land on background pixels just outside the object's actual silhouette.
_TRACKER_MASK_INNER_FRACTION: float = 0.85


class DynamicGSPipeline(VanillaPipeline):
    config: DynamicGSPipelineConfig

    def __init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None):
        self.current_phase = None  # type: Optional[Literal["static", "dynamic"]]
        self.current_dynamic_frame_idx = None  # type: Optional[int]
        self.total_dynamic_frames = 0
        self.total_dynamic_steps = 0
        self._sam3d_inserted = False
        # CoTracker3 + RANSAC-Kabsch rigid-motion estimator. Stateful: holds
        # the D0 reference RGB + depth + 3D back-projected points; each
        # incoming frame's RGB-D advances the tracks pairwise and returns
        # an absolute (R, t) in world frame from the D0 anchor.
        self._motion_estimator = None
        self._global_frame_counter = 0
        self._timing = defaultdict(list)
        self._timing_report_written = False
        # One-shot torch.profiler diagnosis of the dynamic training step.
        # Toggled by DYNAMIC_GS_PROFILE=1. Captures `_torch_profile_active_dyn_steps`
        # active iterations after `_torch_profile_warmup_dyn_steps` warmup, exports
        # a Chrome trace + key_averages table to <data_root>/dynamic_step_profile.{json,txt},
        # then becomes a no-op.
        self._torch_profile_enabled = os.environ.get("DYNAMIC_GS_PROFILE", "0") == "1"
        self._torch_profile_warmup_dyn_steps = 5
        self._torch_profile_active_dyn_steps = 10
        self._torch_profiler = None
        self._torch_profile_started = False
        self._torch_profile_done = False
        self._cpd_info: dict = {}
        # Holds the SAM3 + SAM3D generation outputs (mask metadata + per-object PLY/pose
        # paths) produced pre-static. ``None`` means generation didn't run, ran with 0
        # objects, or was disabled — in which case the post-static fusion is a no-op.
        self._sam3d_generation_outputs: Optional[dict] = None
        # Live-mode runtime state — None when ``config.live=False``.
        self._live_subscriber = None
        self._live_stop_requested: bool = False
        self._live_last_processed_stamp: Optional[float] = None
        # Back-ref to the nerfstudio Trainer, populated in
        # get_training_callbacks. Lets _tracker_tick_live reach
        # trainer.viewer_state to force a re-render right after the
        # rigid transform is applied, bypassing update_scene's
        # step-count throttle (which otherwise rate-limits the viewer
        # to ~0.5–2 Hz in live tracking-only mode).
        self._trainer = None
        atexit.register(self._write_timing_report)

        # Live-mode pre-training session: drives the interactive ROS
        # capture (prompt → SAM3 → SAM3D → record static frames → build
        # init PLY) and points the dataparser at the resulting
        # LIVE_ROOT. Must run before ``super().__init__`` so the
        # datamanager constructor sees the populated static_scene/.
        if getattr(config, "live", False):
            from .utils.live_session import run_live_capture_session
            from .utils.live_shm_reader import LiveShmSubscriber
            live_root = run_live_capture_session()
            config.datamanager.data = live_root
            # The session already constructed the subscriber + spawned
            # the publisher subprocess. Just pull the singleton.
            self._live_subscriber = LiveShmSubscriber.get_singleton()
            self._start_stdin_stop_watcher()

        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self.total_dynamic_frames = self.datamanager.get_num_dynamic_frames()

        # Build the keyframe filter (kept around for the future live-data
        # path: per-arrival ``accept(c2w)`` calls share this state with
        # the recorded-mode bulk_filter we run below).
        self._dynamic_keyframe_filter: Optional[DynamicKeyframeFilter] = None
        if self.config.enable_dynamic_keyframe_filter:
            self._dynamic_keyframe_filter = DynamicKeyframeFilter(
                translation_thresh_m=self.config.dynamic_keyframe_translation_m,
                rotation_thresh_deg=self.config.dynamic_keyframe_rotation_deg,
            )

        # Recorded mode: pre-filter the entire dynamic dataset upfront so
        # ``_total_train_steps`` and ``_dynamic_frame_for_step`` see only
        # the kept frames. Live mode (future) replaces this bulk pass with
        # per-arrival calls to ``self._dynamic_keyframe_filter.accept(...)``
        # that append to ``_accepted_dynamic_frames`` and trigger
        # ``_prepare_dynamic_frame`` on True.
        self._accepted_dynamic_frames: list[int] = list(range(self.total_dynamic_frames))
        if self._dynamic_keyframe_filter is not None and self.total_dynamic_frames > 1:
            dyn_c2w = (
                self.datamanager.dynamic_manager.train_dataset.cameras.camera_to_worlds
            )
            self._accepted_dynamic_frames = self._dynamic_keyframe_filter.bulk_filter(dyn_c2w)
            CONSOLE.log(
                f"[dynamic-gs] dynamic keyframe filter: kept "
                f"{len(self._accepted_dynamic_frames)}/{self.total_dynamic_frames} "
                f"(τ_t={self.config.dynamic_keyframe_translation_m:.4f} m, "
                f"τ_r={self.config.dynamic_keyframe_rotation_deg:.1f}°)"
            )

        self.total_dynamic_steps = (
            len(self._accepted_dynamic_frames) * self.config.dynamic_steps_per_frame
        )
        # Fast-membership lookup: tracker tick checks frame_idx against this
        # set to decide whether to compute CDN + push to the optim pool.
        self._accepted_dynamic_frames_set: set[int] = set(self._accepted_dynamic_frames)

        # Decoupled tracking/optimization state. ``_next_frame_to_track``
        # iterates ALL dataset frames in order (FP track runs on every
        # frame for pose continuity); the pool only collects accepted
        # frames whose CDN passes the min-pixels gate.
        self._optim_pool: OptimPool = OptimPool(capacity=self.config.optim_pool_capacity)
        self._next_frame_to_track: int = 0
        self._dynamic_step_counter: int = 0

        # Step at which the static convergence check first reported
        # "scene matches GT for all keyframes". When set, ``_phase_for_step``
        # returns "dynamic" early.
        self._static_converged_step: Optional[int] = None

        # Pre-load ESAM at pipeline init so the ~300ms one-time model load
        # AND the first-call CUDA kernel JIT for our input shape are paid
        # here (well before D0) rather than inside the timed
        # `D0.1c_esam_render` window. We run two dummy forwards — one with
        # batch=2 for the common D0 path (`query_esam_mask_pair`) and one
        # with batch=1 for fallback paths — so all kernels are JIT-compiled
        # before training starts.
        try:
            esam_model = self.model._get_esam_model()
            try:
                from .utils.esam import ESAM_MAX_SIDE
            except Exception:
                ESAM_MAX_SIDE = 512
            warm_device = torch.device(self.model.device)
            with torch.no_grad():
                dummy_img = torch.zeros(
                    (1, 3, ESAM_MAX_SIDE, ESAM_MAX_SIDE),
                    device=warm_device,
                    dtype=torch.float32,
                )
                dummy_pts = torch.zeros(
                    (1, 1, 1, 2), device=warm_device, dtype=torch.float32
                )
                dummy_lbl = torch.ones(
                    (1, 1, 1), device=warm_device, dtype=torch.float32
                )
                esam_model(dummy_img, dummy_pts, dummy_lbl)
                # Batch=2 forward to JIT the kernels used by query_esam_mask_pair.
                dummy_img2 = dummy_img.expand(2, 3, ESAM_MAX_SIDE, ESAM_MAX_SIDE).contiguous()
                dummy_pts2 = dummy_pts.expand(2, 1, 1, 2).contiguous()
                dummy_lbl2 = dummy_lbl.expand(2, 1, 1).contiguous()
                esam_model(dummy_img2, dummy_pts2, dummy_lbl2)
            if warm_device.type == "cuda":
                torch.cuda.synchronize(warm_device)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs] ESAM warm-up skipped: {exc}")

        # Live warm-cache: if a post-fusion snapshot exists from a previous
        # session, load it and short-circuit static training + Phase 0b
        # fusion. Must run BEFORE ``_sync_phase(0)`` so the latter sees
        # ``_static_converged_step = 0`` and flips straight to "dynamic".
        self._warm_cache_loaded = False
        if self.config.live:
            cache_path = (
                Path(self.config.datamanager.data) / "static_scene" / "post_fusion_state.pt"
            )
            if cache_path.is_file():
                self._warm_cache_loaded = self._load_post_fusion_cache(cache_path)

        self._sync_phase(0)

        # Pre-static: run SAM3 segmentation + SAM3D 3D object generation now, so
        # the per-object PLYs and pose sidecars exist on disk before static
        # optimization begins. Fusion (insertion into the trained scene) is
        # deferred to the static→dynamic transition in ``_sync_phase`` so that
        # SAM3D's back-side Gaussians don't get opacity-eroded by static
        # photometric optimization.
        if (
            not self._warm_cache_loaded
            and self.model.config.use_sam3_graspable_prefusion
            and self.model.config.sam3_prompt_text
        ):
            self._sam3d_generation_outputs = self._run_sam3_and_sam3d_generation()

    def _reset_dynamic_segmentation_state(self) -> None:
        self._sam3d_inserted = False
        self._motion_estimator = None
        self._global_frame_counter = 0
        self._optim_pool = OptimPool(capacity=self.config.optim_pool_capacity)
        self._next_frame_to_track = 0
        self._dynamic_step_counter = 0

    def _run_sam3_and_sam3d_generation(self) -> Optional[dict]:
        """Pre-static: run SAM3 mask discovery + SAM3D 3D object generation.

        Saves per-object Gaussian PLYs and pose sidecars under
        ``initialization_artifacts/`` but does NOT mutate the Gaussian
        scene. Fusion (insertion + instance-id propagation) happens at
        the static→dynamic transition via
        ``_fuse_sam3d_objects_into_scene``.

        Returns a dict ``{"sam3_objects": [...], "sam3d_results": [...]}``,
        or ``None`` if SAM3 found 0 objects.
        """
        import gc
        import json
        import numpy as np

        t_total = time.time()
        model_cfg = self.model.config
        debug_dir = self.datamanager.get_initialization_debug_dir()
        artifact_dir = self.datamanager.get_initialization_artifact_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # First static image bytes + depth + intrinsics. We don't need a
        # camera tensor here — generation does no rendering. Fusion will
        # re-fetch its own camera at the static→dynamic transition.
        batch = self.datamanager.static_manager.cached_train[0]
        static_image = batch["image"]  # (H, W, 3) uint8 or float [0,1]

        static_ds = self.datamanager.static_manager.train_dataset
        depth_filenames = static_ds.metadata.get("depth_filenames")
        depth_scale = float(static_ds.metadata.get("depth_unit_scale_factor", 1.0))
        frame_idx_0 = int(batch.get("image_idx", 0))
        static_depth_m = None
        if depth_filenames is not None and frame_idx_0 < len(depth_filenames):
            try:
                depth_pil = Image.open(Path(depth_filenames[frame_idx_0]))
                depth_np = np.array(depth_pil).astype(np.float32) * depth_scale
                static_depth_m = torch.from_numpy(depth_np)
            except Exception as exc:
                CONSOLE.log(f"[phase-0] warning: failed to load static depth ({exc})")

        static_intrinsics = {
            "fx": float(static_ds.cameras.fx[frame_idx_0].item()),
            "fy": float(static_ds.cameras.fy[frame_idx_0].item()),
            "cx": float(static_ds.cameras.cx[frame_idx_0].item()),
            "cy": float(static_ds.cameras.cy[frame_idx_0].item()),
        }

        # Save static image (with gripper blacked out) for the SAM3 worker.
        static_image_path = debug_dir / "static0_rgb.png"
        img_cpu = static_image.cpu()
        if img_cpu.dtype == torch.uint8:
            static_np = img_cpu.numpy()
        else:
            static_np = (img_cpu.numpy() * 255).clip(0, 255).astype(np.uint8)
        gripper_mask_t = batch.get("mask")
        if gripper_mask_t is not None:
            m = gripper_mask_t.detach().cpu()
            if m.ndim == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)
            keep = (m > 0.5).numpy()
            if keep.shape != static_np.shape[:2]:
                resized = Image.fromarray(keep.astype(np.uint8) * 255).resize(
                    (static_np.shape[1], static_np.shape[0]), Image.NEAREST
                )
                keep = np.array(resized) > 127
            static_np = static_np.copy()
            static_np[~keep] = 0
        Image.fromarray(static_np).save(static_image_path)

        results_json = debug_dir / "static0_sam3_results.json"
        sam3_cached = model_cfg.sam3_reuse_cached and results_json.exists()

        # Move model + cached images off GPU so the SAM3/SAM3D subprocesses
        # have headroom on small GPUs.
        run_device = torch.device(self.model.means.device)
        if not sam3_cached:
            self.model.to("cpu")
            for mgr in [self.datamanager.static_manager, self.datamanager.dynamic_manager]:
                if hasattr(mgr, "cached_train"):
                    for batch_iter in mgr.cached_train:
                        for k in list(batch_iter.keys()):
                            if hasattr(batch_iter[k], "cpu") and hasattr(batch_iter[k], "device"):
                                if str(batch_iter[k].device).startswith("cuda"):
                                    batch_iter[k] = batch_iter[k].cpu()
                if hasattr(mgr, "cached_eval"):
                    for batch_iter in mgr.cached_eval:
                        for k in list(batch_iter.keys()):
                            if hasattr(batch_iter[k], "cpu") and hasattr(batch_iter[k], "device"):
                                if str(batch_iter[k].device).startswith("cuda"):
                                    batch_iter[k] = batch_iter[k].cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            # SAM3 segmentation
            t_sam3 = time.time()
            if sam3_cached:
                sam3_objects = load_sam3_masks(results_json)
                CONSOLE.log(f"[phase-0] reusing cached SAM3 results: {len(sam3_objects)} objects")
            else:
                sam3_objects = run_sam3_subprocess(
                    image_path=static_image_path,
                    text_prompt=model_cfg.sam3_prompt_text,
                    output_dir=debug_dir,
                    output_stem="static0",
                    sam3_conda_env=model_cfg.sam3_conda_env_name,
                    min_area_ratio=model_cfg.sam3_candidate_min_area_ratio,
                    max_area_ratio=model_cfg.sam3_candidate_max_area_ratio,
                    dedup_iou=model_cfg.sam3_candidate_dedup_iou,
                    max_objects=model_cfg.sam3_candidate_max_objects,
                    confidence_threshold=model_cfg.sam3_confidence_threshold,
                    min_score=model_cfg.sam3_min_score,
                )
            self._timing["S0.1_sam3_segmentation"].append(time.time() - t_sam3)
            # In live mode, the actual SAM3 subprocess ran in
            # ``live_session._run_sam3_worker`` before the pipeline was
            # constructed. The block above only re-loads the cached
            # results (~milliseconds), so the inline timer doesn't
            # reflect the real subprocess wall-clock. Re-inject the
            # measured duration from the live-session sidecar if it
            # exists.
            try:
                live_timings_path = artifact_dir / "live_sam3_timings.json"
                if live_timings_path.is_file():
                    import json as _json
                    live_timings = _json.loads(live_timings_path.read_text())
                    if "S0.1_sam3_segmentation" in live_timings:
                        # Replace the cached-check duration with the
                        # real subprocess one. Single-entry list since
                        # SAM3 only ever runs once per session.
                        self._timing["S0.1_sam3_segmentation"] = [
                            float(live_timings["S0.1_sam3_segmentation"]),
                        ]
            except Exception as _exc:
                CONSOLE.log(f"[phase-0] could not read live SAM3 timing sidecar: {_exc}")

            if not sam3_objects:
                CONSOLE.log("[phase-0] SAM3 found 0 objects; skipping Phase 0 prefusion")
                return None

            CONSOLE.log(f"[phase-0] SAM3 discovered {len(sam3_objects)} objects")
            self._save_sam3_debug_plots(
                rgb_path=static_image_path,
                sam3_objects=sam3_objects,
                out_dir=debug_dir,
                prefix="static0",
            )

            # SAM3D multi-object generation (full image + metric pointmap
            # from the static-scene depth, no crop). See sam3d.py and
            # scripts/old/test_sam3d_strategies.py for the rationale.
            t_sam3d = time.time()
            output_stems = [f"static0_obj_{i:02d}_sam3d" for i in range(len(sam3_objects))]
            sam3d_results: list = [None] * len(sam3_objects)

            to_run_indices: list[int] = []
            to_run_mask_paths: list[Path] = []
            to_run_stems: list[str] = []
            for obj_i, sam3_obj in enumerate(sam3_objects):
                stem = output_stems[obj_i]
                paths = get_sam3d_output_paths(artifact_dir, stem, image_dir=debug_dir)
                pose_path_resolved = resolve_sam3d_pose_path(paths["ply_path"], paths["pose_path"])
                if model_cfg.sam3_reuse_cached and paths["ply_path"].exists() and sam3d_pose_has_rotation(pose_path_resolved):
                    if pose_path_resolved is not None:
                        paths["pose_path"] = pose_path_resolved
                    sam3d_results[obj_i] = paths
                    CONSOLE.log(f"[phase-0] object {obj_i}: reusing cached SAM3D output")
                else:
                    to_run_indices.append(obj_i)
                    to_run_mask_paths.append(Path(sam3_obj["mask_path"]))
                    to_run_stems.append(stem)

            full_depth_path = None
            full_intrinsics_path = None
            if static_depth_m is not None and to_run_indices:
                H_img, W_img = int(static_np.shape[0]), int(static_np.shape[1])
                full_depth_path = artifact_dir / "static0_full_depth_meters.tiff"
                Image.fromarray(static_depth_m.cpu().numpy().astype(np.float32)).save(full_depth_path)
                full_intrinsics_path = artifact_dir / "static0_full_intrinsics.json"
                full_intrinsics_path.write_text(
                    json.dumps(
                        {
                            **static_intrinsics,
                            "width": W_img,
                            "height": H_img,
                        },
                        indent=2,
                    )
                    + "\n"
                )

            if to_run_indices:
                try:
                    multi_results = run_sam3d_multi_object_subprocess(
                        render_image_path=static_image_path,
                        object_mask_paths=to_run_mask_paths,
                        output_dir=artifact_dir,
                        output_stems=to_run_stems,
                        image_dir=debug_dir,
                        max_side=518,
                        depth_path=full_depth_path,
                        intrinsics_path=full_intrinsics_path,
                    )
                except Exception as exc:
                    CONSOLE.log(f"[phase-0] SAM3D multi-object subprocess failed: {exc}")
                    multi_results = [{} for _ in to_run_indices]

                for idx, result in zip(to_run_indices, multi_results):
                    if result:
                        sam3d_results[idx] = result
                        CONSOLE.log(f"[phase-0] object {idx}: SAM3D generation complete")
                    else:
                        sam3d_results[idx] = {}
                        CONSOLE.log(f"[phase-0] object {idx}: SAM3D failed (empty result)")

            sam3d_results = [r if r else {} for r in sam3d_results]
            self._timing["S0.2_sam3d_multi_generation"].append(time.time() - t_sam3d)
        finally:
            if not sam3_cached:
                self.model.to(run_device)
                for mgr in [self.datamanager.static_manager, self.datamanager.dynamic_manager]:
                    if hasattr(mgr, "cached_train"):
                        for batch_iter in mgr.cached_train:
                            for k in list(batch_iter.keys()):
                                if hasattr(batch_iter[k], "to") and hasattr(batch_iter[k], "device"):
                                    if not str(batch_iter[k].device).startswith("cuda"):
                                        batch_iter[k] = batch_iter[k].to(run_device)
                    if hasattr(mgr, "cached_eval"):
                        for batch_iter in mgr.cached_eval:
                            for k in list(batch_iter.keys()):
                                if hasattr(batch_iter[k], "to") and hasattr(batch_iter[k], "device"):
                                    if not str(batch_iter[k].device).startswith("cuda"):
                                        batch_iter[k] = batch_iter[k].to(run_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Phase 0a (pre-static, in __init__): SAM3 segmentation + SAM3D
        # multi-object generation. Reported separately from 0b in the timing
        # summary because the two halves run minutes apart on the timeline.
        self._timing["S0.4a_generation_total"].append(time.time() - t_total)
        CONSOLE.log(
            f"[phase-0] generation complete: {len(sam3_objects)} masks, "
            f"{sum(1 for r in sam3d_results if r)} SAM3D PLYs ready; "
            f"fusion deferred to static→dynamic transition"
        )
        return {
            "sam3_objects": sam3_objects,
            "sam3d_results": sam3d_results,
        }

    def _fuse_sam3d_objects_into_scene(self, gen_outputs: dict) -> dict:
        """Post-static: take the pre-generated SAM3D PLYs and fuse them
        into the trained scene. Runs at the static→dynamic transition.

        After ``static_num_steps`` of static optimization, the rendered
        depth used by ``_get_existing_object_subset`` is meaningful (vs
        being garbage at __init__ time when only SfM seeds exist), so
        instance-id propagation gets a denser set of seed Gaussians.
        SAM3D's back-side Gaussians also never see static photometric
        optimization, so they are not opacity-eroded before D0.
        """
        import json
        import numpy as np

        t_total = time.time()
        sam3_objects = gen_outputs.get("sam3_objects") or []
        sam3d_results = gen_outputs.get("sam3d_results") or []
        if not sam3_objects:
            return {}

        debug_dir = self.datamanager.get_initialization_debug_dir()
        artifact_dir = self.datamanager.get_initialization_artifact_dir()

        batch = self.datamanager.static_manager.cached_train[0]
        static_image = batch["image"]
        frame_idx_0 = int(batch.get("image_idx", 0))
        camera = self.datamanager.static_manager.train_dataset.cameras[
            frame_idx_0 : frame_idx_0 + 1
        ].to(self.device)

        # Apply the post-static camera-optimizer offset to the fusion camera.
        # `_should_apply_camera_optimizer` returns False here because phase
        # already flipped to "dynamic", so we call `apply_to_camera` directly
        # and overwrite `camera_to_worlds` once. Downstream `get_outputs`,
        # `_backproject_mask_to_world`, and the c2w read inside
        # `register_and_fuse_sam3d_object` all see the optimized pose.
        camera.metadata = dict(camera.metadata or {})
        camera.metadata["cam_idx"] = frame_idx_0
        if (
            self.model.camera_optimizer.config.mode != "off"
            and 0 <= frame_idx_0 < self.model.camera_optimizer.num_cameras
        ):
            optimized_c2w = self.model.camera_optimizer.apply_to_camera(camera).detach()
            if optimized_c2w.shape == camera.camera_to_worlds.shape:
                camera.camera_to_worlds = optimized_c2w
                CONSOLE.log(
                    f"[phase-0] using post-static optimized pose for cam_idx={frame_idx_0}"
                )

        static_ds = self.datamanager.static_manager.train_dataset
        depth_filenames = static_ds.metadata.get("depth_filenames")
        depth_scale = float(static_ds.metadata.get("depth_unit_scale_factor", 1.0))
        static_depth_m = None
        if depth_filenames is not None and frame_idx_0 < len(depth_filenames):
            try:
                depth_pil = Image.open(Path(depth_filenames[frame_idx_0]))
                depth_np = np.array(depth_pil).astype(np.float32) * depth_scale
                static_depth_m = torch.from_numpy(depth_np)
            except Exception as exc:
                CONSOLE.log(f"[phase-0] warning: failed to load static depth ({exc}); falling back to SfM targets")

        manifest: dict = {}
        n_objs = len(sam3_objects)
        print(
            f"\n==> [phase-0b] fusing {n_objs} SAM3D object(s) into the scene. "
            f"Each object runs CPD registration (~1.8-58s/obj per CLAUDE.md) "
            f"+ mesh build + FP tracker construct. Total can be several minutes; "
            f"per-object progress lines follow.\n",
            flush=True,
        )
        for obj_idx, (sam3_obj, sam3d_out) in enumerate(zip(sam3_objects, sam3d_results)):
            instance_id = obj_idx + 1
            print(f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: starting", flush=True)
            if not sam3d_out:
                CONSOLE.log(f"[phase-0] skipping object {obj_idx}: SAM3D failed or empty")
                continue

            t_fusion = time.time()

            # Re-render every iteration. ``insert_object_gaussians`` mutates the
            # Gaussian count, so ``model.info`` (populated by the previous render)
            # goes stale and ``_get_existing_object_subset`` →
            # ``extract_projected_centers_and_radii`` raises a length-mismatch.
            with torch.no_grad():
                outputs = self.model.get_outputs(camera)
            render_h, render_w = outputs["rgb"].shape[:2]

            ply_path = sam3d_out["ply_path"]
            pose_path = sam3d_out["pose_path"]
            try:
                source_points, source_colors = load_sam3d_gaussian_ply(ply_path)
                source_rotation_wxyz = load_sam3d_rotation_wxyz(pose_path)
            except Exception as exc:
                CONSOLE.log(f"[phase-0] skipping object {obj_idx}: {exc}")
                continue

            obj_mask_np = np.array(Image.open(sam3_obj["mask_path"]).convert("L"))
            obj_mask_tensor = torch.from_numpy((obj_mask_np > 127).astype(np.float32))
            obj_mask_tensor = obj_mask_tensor[..., None].to(self.device)
            if obj_mask_tensor.shape[0] != render_h or obj_mask_tensor.shape[1] != render_w:
                obj_mask_tensor = torch.nn.functional.interpolate(
                    obj_mask_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(render_h, render_w),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)

            existing_indices, existing_means, existing_colors = self.model._get_existing_object_subset(
                obj_mask_tensor,
                outputs["depth"],
            )
            existing_indices_cpu = existing_indices.detach().cpu()
            existing_means_np = existing_means.detach().cpu().numpy()
            existing_colors_np = existing_colors.detach().cpu().numpy()

            # Dense registration target via back-projection through the static
            # depth image (Gazebo GT). Falls back to SfM seeds when depth is
            # unavailable.
            target_points_np = existing_means_np
            target_colors_np = existing_colors_np
            if static_depth_m is not None:
                target_points_np, target_colors_np = self._backproject_mask_to_world(
                    obj_mask_tensor.squeeze(-1).cpu().numpy() > 0.5,
                    static_depth_m,
                    static_image,
                    camera,
                )
            if target_points_np.shape[0] < 3:
                CONSOLE.log(
                    f"[phase-0] skipping object {obj_idx}: only {target_points_np.shape[0]} target points for registration"
                )
                continue

            # `camera.camera_to_worlds` was already overwritten with the
            # post-static optimized pose above, so read it directly.
            c2w_rotation = camera.camera_to_worlds[0, :3, :3].detach().cpu().numpy().astype(np.float32)

            t_cpd = time.time()
            print(
                f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: running CPD registration "
                f"(source={len(source_points)} pts, target={len(target_points_np)} pts) "
                f"— this is the slow part",
                flush=True,
            )
            insertion_result = register_and_fuse_sam3d_object(
                source_points=source_points,
                source_colors=source_colors,
                target_points=target_points_np,
                target_colors=target_colors_np,
                source_rotation_wxyz=source_rotation_wxyz,
                camera_to_world_rotation=c2w_rotation,
                debug_dir=debug_dir,
                artifact_dir=artifact_dir,
                output_stem=f"static0_obj_{obj_idx:02d}_sam3d",
            )
            print(
                f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: CPD done in {time.time() - t_cpd:.1f}s "
                f"(kept {insertion_result.kept_point_count} pts)",
                flush=True,
            )

            # Cull tunables — see comments in the cull block below for what
            # each one does. Bumping CULL_STRENGTH or TAU_FLOOR_M removes more
            # SAM3D points on the camera-visible side; bumping CULL_DEPTH_TOL_M
            # makes E denser (more existing Gaussians in the slab) so tau is
            # estimated on a richer set.
            CULL_STRENGTH = 1.3
            TAU_FLOOR_M = 0.003
            CULL_DEPTH_TOL_M = 0.015
            FLAG_DEPTH_TOL_M = 0.02

            # Pre-insertion mask-slab extractions. Both gate candidates by
            # "projected center is in the object 2D mask AND projected depth is
            # within depth_tol_m of the rendered front-surface depth", but with
            # different tolerances tuned to their use:
            #   * cull: defines E for SAM3D point removal.
            #   * flag: slightly more generous, captures deeper interior
            #     original Gaussians for membership without leaking onto the table.
            # Both must run BEFORE insert_object_gaussians so self.info matches
            # self.num_points (extract_projected_centers_and_radii enforces it).
            e_indices_cull = self.model._get_object_mask_slab_indices(
                obj_mask_tensor, outputs["depth"], depth_tol_m=CULL_DEPTH_TOL_M
            )
            e_indices_flag = self.model._get_object_mask_slab_indices(
                obj_mask_tensor, outputs["depth"], depth_tol_m=FLAG_DEPTH_TOL_M
            )

            # Cull SAM3D points whose 1-NN in E_cull is within
            # max(_estimate_spacing(E) * CULL_STRENGTH, TAU_FLOOR_M). Larger
            # CULL_STRENGTH covers SAM3D points that fall in the gaps between
            # E points (SfM-sparse front surface vs. SAM3D-dense mesh). Floor
            # protects against locally dense E giving a tiny radius. Result:
            # front/sides/top of the object stay as the SfM-trained Gaussians;
            # SAM3D fills the camera-unobserved regions (back, bottom).
            cull_pts_np = insertion_result.kept_points.astype(np.float32)
            cull_colors_np = insertion_result.kept_colors.astype(np.float32)
            n_culled_sam3d = 0
            tau = 0.0
            if cull_pts_np.shape[0] > 0 and e_indices_cull.numel() >= 2:
                e_pts_np = (
                    self.model.means[e_indices_cull].detach().cpu().numpy().astype(np.float32)
                )
                tau = max(
                    self.model._estimate_spacing(e_pts_np) * CULL_STRENGTH,
                    TAU_FLOOR_M,
                )

                from sklearn.neighbors import NearestNeighbors as _CullNN

                e_nn = _CullNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(e_pts_np)
                sam3d_d, _ = e_nn.kneighbors(cull_pts_np)
                keep_mask = ~(np.isfinite(sam3d_d[:, 0]) & (sam3d_d[:, 0] <= tau))
                n_culled_sam3d = int((~keep_mask).sum())
                cull_pts_np = cull_pts_np[keep_mask]
                cull_colors_np = cull_colors_np[keep_mask]

            if cull_pts_np.shape[0] > 0:
                inserted_indices = self.model.insert_object_gaussians(
                    torch.from_numpy(cull_pts_np),
                    torch.from_numpy(cull_colors_np),
                    object_flag=False,
                    instance_id=instance_id,
                )
            else:
                inserted_indices = torch.zeros((0,), dtype=torch.long, device=self.model.means.device)

            # Flag the existing object Gaussians. Two-stage gate:
            #   (1) candidate pool restricted to the mask-slab — center inside
            #       the 2D object mask AND projected depth within depth_tol_m
            #       of the rendered front surface. Mask-bound → table Gaussians
            #       can never be flagged regardless of 3D proximity.
            #   (2) within that pool, accept Gaussians within proxy_radius of
            #       the inserted SAM3D cloud OR target_radius of the CPD
            #       registration target. Both radii hard-capped at MAX_RADIUS_M.
            MAX_RADIUS_M = 0.02
            n_flagged_existing = 0
            match_indices = torch.zeros((0,), dtype=torch.long, device=self.model.means.device)
            if e_indices_flag.numel() > 0 and insertion_result.kept_point_count > 0:
                device = self.model.means.device
                instance_ids_flat = self.model.object_instance_ids.squeeze(-1)
                slab_owners = instance_ids_flat[e_indices_flag]
                eligible_mask = (slab_owners == 0) | (slab_owners == instance_id)
                candidate_indices = e_indices_flag.to(device=device)[eligible_mask]

                if candidate_indices.numel() > 0:
                    from sklearn.neighbors import NearestNeighbors as _MatchNN

                    candidate_pts_np = (
                        self.model.means[candidate_indices].detach().cpu().numpy().astype(np.float32)
                    )

                    proxy_points_np = insertion_result.kept_points.astype(np.float32)
                    proxy_spacing = self.model._estimate_spacing(proxy_points_np)
                    proxy_radius = min(MAX_RADIUS_M, max(0.003, 1.5 * proxy_spacing))
                    proxy_nn = _MatchNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(proxy_points_np)
                    proxy_d, _ = proxy_nn.kneighbors(candidate_pts_np)
                    near_proxy_np = np.isfinite(proxy_d[:, 0]) & (proxy_d[:, 0] <= proxy_radius)

                    target_pts_np = existing_means_np.astype(np.float32)
                    near_target_np = np.zeros((len(candidate_pts_np),), dtype=bool)
                    if len(target_pts_np) > 0:
                        target_spacing = self.model._estimate_spacing(target_pts_np)
                        target_radius = min(MAX_RADIUS_M, max(0.002, 6.0 * target_spacing))
                        target_nn = _MatchNN(n_neighbors=1, algorithm="auto", metric="euclidean").fit(target_pts_np)
                        target_d, _ = target_nn.kneighbors(candidate_pts_np)
                        near_target_np = np.isfinite(target_d[:, 0]) & (target_d[:, 0] <= target_radius)

                    match_mask_np = near_proxy_np | near_target_np
                    if match_mask_np.any():
                        match_indices = candidate_indices[
                            torch.from_numpy(match_mask_np).to(device=device)
                        ]
                        self.model.object_instance_ids[match_indices] = instance_id
                        n_flagged_existing = int(match_indices.numel())

            obj_total = time.time() - t_fusion
            self._timing[f"S0.3_fusion_obj_{obj_idx}"].append(obj_total)
            print(
                f"==> [phase-0b] obj {obj_idx + 1}/{n_objs}: done in {obj_total:.1f}s",
                flush=True,
            )

            instance_count = int(
                (self.model.object_instance_ids.squeeze(-1) == instance_id).sum().item()
            )

            # CoTracker tracks 2D pixel points in the live RGB stream and
            # back-projects via depth, so Phase 0b doesn't need a mesh or any
            # per-instance pre-construction step. The tracker is initialized
            # at D0 (``_initialize_motion_estimator``) once the moved object is
            # known, against the live D0 RGB-D + the rendered object mask.
            manifest[instance_id] = {
                "object_index": obj_idx,
                "mask_path": str(sam3_obj["mask_path"]),
                "ply_path": str(ply_path),
                "score": sam3_obj.get("score", 0.0),
                "existing_gaussians": int(existing_indices.numel()),
                "sam3d_pre_cull_count": int(insertion_result.kept_point_count),
                "sam3d_culled": int(n_culled_sam3d),
                "sam3d_cull_rate": (
                    float(n_culled_sam3d) / float(insertion_result.kept_point_count)
                    if insertion_result.kept_point_count > 0
                    else 0.0
                ),
                "cull_tau_m": float(tau),
                "inserted_gaussians": int(inserted_indices.numel()),
                "flagged_existing_gaussians": int(n_flagged_existing),
                "instance_count": instance_count,
                "kept_points": insertion_result.kept_point_count,
                "chosen_scale": insertion_result.chosen_scale,
                "source_spacing": float(insertion_result.source_spacing),
            }
            CONSOLE.log(
                f"[phase-0] object {obj_idx} (instance_id={instance_id}): "
                f"existing={existing_indices.numel()}, "
                f"sam3d={insertion_result.kept_point_count}->{inserted_indices.numel()} "
                f"(culled {n_culled_sam3d}, tau={tau * 1000:.1f}mm), "
                f"flagged_existing={n_flagged_existing}, "
                f"instance_total={instance_count}, "
                f"scale={insertion_result.chosen_scale:.4f}"
            )

        manifest_path = artifact_dir / "phase0_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

        fusion_time = time.time() - t_total
        # Phase 0b (post-static, at static→dynamic transition): per-object
        # CPD register + insert + propagate. Reported separately from 0a in
        # the timing summary.
        self._timing["S0.4b_fusion_total"].append(fusion_time)
        num_prefused = int((self.model.object_instance_ids > 0).any(dim=-1).sum().item())
        CONSOLE.log(
            f"[phase-0] fusion complete: {len(manifest)} objects fused, "
            f"{num_prefused} Gaussians with instance IDs, "
            f"fusion time={fusion_time:.2f}s"
        )

        # Live mode: write a one-shot "post-fusion" snapshot so the next
        # ``--live`` run can skip static training + Phase 0b entirely.
        # The snapshot covers everything the dynamic loop needs: trained
        # Gaussian parameters at the post-insertion size, all persistent
        # buffers (object_flags / object_instance_ids / inserted_flags),
        # and the SO3xR3 camera-optimizer offsets keyed by frame index.
        if self.config.live:
            self._save_post_fusion_cache()
        return manifest

    def _save_post_fusion_cache(self) -> None:
        """Snapshot the post-fusion model so a future warm restart can
        jump straight to the dynamic phase. Only called from
        ``_fuse_sam3d_objects_into_scene`` in live mode.

        Writes ``<LIVE_ROOT>/static_scene/post_fusion_state.pt`` with the
        full ``model.state_dict()`` plus ``num_points``. The next session
        re-builds the model from the SfM seed PLY (small N), then
        ``_load_post_fusion_cache`` resizes the gauss_params to the saved
        N and copies the trained tensors in.
        """
        try:
            live_root = Path(self.config.datamanager.data)
            cache_path = live_root / "static_scene" / "post_fusion_state.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "num_points": int(self.model.num_points),
                },
                cache_path,
            )
            CONSOLE.log(
                f"[live-warm-cache] saved post-fusion state to "
                f"{cache_path.name} (N={int(self.model.num_points)} Gaussians)"
            )
        except Exception as exc:
            CONSOLE.log(f"[live-warm-cache] could not save snapshot: {exc}")
        # PROBLEM: the snapshot is not config-tagged. If sh_degree,
        # background color, or the camera-optimizer mode changes between
        # save and load, the loaded tensors won't match the rebuilt model
        # and load_state_dict will raise. Delete the .pt to recover.

    def _load_post_fusion_cache(self, cache_path: Path) -> bool:
        """Restore a post-fusion model snapshot from disk.

        The cold-start model was built by ``super().__init__`` from the
        SfM seed PLY (small N). The snapshot has N_post Gaussians (post
        Phase-0b insertions), so each ``gauss_params`` Parameter has to
        be re-allocated at N_post before ``load_state_dict`` can copy
        values in. The model's own ``load_state_dict`` override handles
        the persistent buffers (object_flags, object_instance_ids, ...).

        Side effects on success:
          * sets ``_static_converged_step = 0`` so the next ``_sync_phase``
            flips current_phase straight to "dynamic"
          * sets ``_sam3d_inserted = True`` so D0 takes Path A (prefused)
          * leaves ``_sam3d_generation_outputs = None`` so the
            static→dynamic transition does not re-run ``_fuse_sam3d_objects_into_scene``

        Returns True on success, False on any failure (caller falls back
        to the standard static+fusion path).
        """
        try:
            blob = torch.load(cache_path, map_location=self.device)
            state_dict = blob["model_state_dict"]
            target_n = int(blob["num_points"])
        except Exception as exc:
            CONSOLE.log(f"[live-warm-cache] could not read {cache_path.name}: {exc}")
            return False

        device = self.model.means.device
        try:
            # Reallocate each gauss_params Parameter at the saved size.
            # Mirrors the resize pattern in ``insert_object_gaussians``.
            for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
                sd_key = f"gauss_params.{name}"
                if sd_key not in state_dict:
                    CONSOLE.log(f"[live-warm-cache] missing {sd_key}; falling back to static+fusion")
                    return False
                old_param = self.model.gauss_params[name]
                new_tensor = state_dict[sd_key].to(device=device, dtype=old_param.dtype)
                self.model.gauss_params[name] = torch.nn.Parameter(
                    new_tensor.clone(),
                    requires_grad=old_param.requires_grad,
                )

            # The model's load_state_dict override sees ``object_flags.shape[0]
            # != target_n`` and rebuilds the persistent buffers at target_n
            # before copying. strict=False so any new/old keys (e.g. viewer
            # GUI handles) don't crash the load.
            self.model.load_state_dict(state_dict, strict=False)

            # The means-grad hook was bound to the old Parameter object; re-bind
            # to the freshly-allocated one so dynamic-phase gradient masking
            # fires correctly on object Gaussians.
            self.model.gauss_params["means"].register_hook(self.model._mask_means_grad)
        except Exception as exc:
            CONSOLE.log(f"[live-warm-cache] state_dict load failed: {exc}")
            return False

        # Tell the dynamic-loop bookkeeping that fusion has already happened.
        # D0 will go down Path A: pick the closest pre-fused instance to the
        # camera, set object_flags from object_instance_ids, init the CoTracker.
        self._sam3d_inserted = True
        self._sam3d_generation_outputs = None
        self._static_converged_step = 0

        # Advance the model's notional step so splatfacto's resolution +
        # SH-degree schedules immediately report full-res / final-SH.
        # step_cb in DynamicGSModel honors `_step_offset`. Without this,
        # the first dynamic render is 4× downscaled (200×200) and
        # build_change_mask raises a tensor-size mismatch against the
        # live RGB (800×800).
        self.model._step_offset = int(self.config.static_num_steps)

        CONSOLE.log(
            f"[live-warm-cache] loaded {cache_path.name} "
            f"(N={target_n} Gaussians); skipping static + Phase 0b."
        )
        print(
            f"\n==> [live] warm cache hit: jumping straight to dynamic phase "
            f"(N={target_n} Gaussians). Static training + Phase 0b skipped.\n",
            flush=True,
        )
        return True
        # PROBLEM: this assumes the saved camera-optimizer offsets still
        # correspond to the current transforms.json frame ordering. The
        # Tier 1 cache guarantees same LIVE_ROOT (so same transforms),
        # but if you manually edit transforms.json between runs, the
        # per-frame SO3xR3 offsets will load against the wrong indices.

    def _backproject_mask_to_world(
        self,
        mask_bool_np,
        depth_image: torch.Tensor,
        rgb_image: torch.Tensor,
        camera,
    ):
        """Back-project an image-plane mask through a depth image into world 3D points.

        Args:
            mask_bool_np: (H_mask, W_mask) boolean array on CPU.
            depth_image: (H, W) depth in meters (CPU tensor).
            rgb_image: (H, W, 3) uint8 or float RGB (CPU or GPU tensor).
            camera: ``Cameras`` with at least one element (we use index 0).

        Returns:
            ``(points_np, colors_np)`` where ``points_np`` is ``(N, 3)`` float32 in world
            coordinates and ``colors_np`` is ``(N, 3)`` float32 in [0, 1].  Points with
            missing/zero depth are filtered out.
        """
        import numpy as np

        H, W = int(depth_image.shape[0]), int(depth_image.shape[1])

        # Resize mask to depth resolution via nearest-neighbor
        if mask_bool_np.shape != (H, W):
            mask_resized = np.array(
                Image.fromarray(mask_bool_np.astype(np.uint8) * 255).resize((W, H), Image.NEAREST),
                dtype=np.uint8,
            ) > 127
        else:
            mask_resized = mask_bool_np

        depth_np = depth_image.detach().cpu().numpy().astype(np.float32)

        # Resize rgb to depth resolution and convert to float [0,1]
        if hasattr(rgb_image, "detach"):
            rgb_cpu = rgb_image.detach().cpu()
        else:
            rgb_cpu = rgb_image
        rgb_np = rgb_cpu.numpy() if hasattr(rgb_cpu, "numpy") else np.asarray(rgb_cpu)
        if rgb_np.dtype == np.uint8:
            rgb_np = rgb_np.astype(np.float32) / 255.0
        else:
            rgb_np = rgb_np.astype(np.float32)
        if rgb_np.shape[:2] != (H, W):
            rgb_np = np.array(
                Image.fromarray((rgb_np * 255).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

        ys, xs = np.where(mask_resized & (depth_np > 1e-4))
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        z = depth_np[ys, xs]

        # MAD-based depth outlier scrub on the back-projected target cloud.
        # Mask-boundary pixels frequently hit the background/table behind
        # the object (silhouette-edge depth bleed); without this scrub the
        # contaminated target points pull CPD's similarity fit toward a
        # smaller scale and a shifted centroid, inserting the SAM3D cloud
        # "super small inside the true object". CPD's outlier robustness
        # alone is insufficient because the bleed is structured along the
        # silhouette, not random.
        #
        # Threshold is intentionally permissive (``5.0 × 1.4826 ≈ 7.4 MAD``
        # vs. the original ``4.45 MAD``) so that objects with large legit
        # depth extent (e.g. tall/thick objects) are not truncated at the
        # back/bottom — only true silhouette-edge bleed (typically half a
        # meter or more behind the object) is removed.
        if z.size >= 10:
            med = float(np.median(z))
            mad = float(np.median(np.abs(z - med))) + 1e-6
            keep = np.abs(z - med) < 5.0 * 1.4826 * mad
            if keep.sum() >= 3:
                ys = ys[keep]
                xs = xs[keep]
                z = z[keep]

        # Camera intrinsics for frame 0
        fx = float(camera.fx[0].item())
        fy = float(camera.fy[0].item())
        cx = float(camera.cx[0].item())
        cy = float(camera.cy[0].item())
        c2w = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)  # (3, 4)

        # Intrinsics are defined at the original image resolution; if depth is at a different
        # resolution, scale them.
        src_H = int(camera.height[0].item()) if hasattr(camera.height[0], "item") else int(camera.height[0])
        src_W = int(camera.width[0].item()) if hasattr(camera.width[0], "item") else int(camera.width[0])
        if (H, W) != (src_H, src_W):
            sx = W / float(src_W)
            sy = H / float(src_H)
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        # Back-project: camera frame, Nerfstudio/OpenGL convention (x right, y up, z backwards)
        # so forward direction is -z.  This matches nerfstudio.cameras.cameras.Cameras.
        x_cam = (xs.astype(np.float32) - cx) / fx * z
        y_cam = -(ys.astype(np.float32) - cy) / fy * z
        z_cam = -z
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

        # Transform to world: p_w = R @ p_c + t
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = pts_cam @ R.T + t[None, :]
        colors = rgb_np[ys, xs]  # (N, 3)
        return pts_world.astype(np.float32), colors.astype(np.float32)

    def _save_sam3_debug_plots(
        self,
        rgb_path: Path,
        sam3_objects: list,
        out_dir: Path,
        prefix: str = "static0",
    ) -> None:
        """Save debug plots for SAM3 segmentation: overview (all objects) + per-object overlays.

        Produces:
          - {prefix}_sam3_overview.png: the input RGB with all masks tinted by distinct
            colors, plus bboxes + scores labeled on each object.
          - {prefix}_obj_{i:02d}_overlay.png: input RGB with a single object's mask
            tinted red, its bbox and score labeled.
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        rgb_pil = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_pil, dtype=np.uint8)
        H, W = rgb_np.shape[:2]

        palette = np.array(
            [
                (255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 128, 0),
                (200, 0, 200), (0, 200, 200), (255, 255, 0), (128, 64, 255),
            ],
            dtype=np.uint8,
        )

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        # Overview image: all objects
        overview = rgb_np.astype(np.float32)
        for i, obj in enumerate(sam3_objects):
            mask_path = obj.get("mask_path")
            if not mask_path:
                continue
            m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if m.shape != (H, W):
                m = np.array(
                    Image.fromarray(m).resize((W, H), Image.NEAREST),
                    dtype=np.uint8,
                )
            mask_bool = m > 127
            color = palette[i % len(palette)].astype(np.float32)
            alpha = 0.5
            overview[mask_bool] = overview[mask_bool] * (1 - alpha) + color * alpha

        overview_img = Image.fromarray(overview.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(overview_img)
        for i, obj in enumerate(sam3_objects):
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            color = tuple(int(c) for c in palette[i % len(palette)])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            label = f"#{i} s={obj.get('score', 0.0):.2f}"
            # Text background for readability
            text_xy = (x0 + 2, max(0, y0 - 22))
            try:
                tb = draw.textbbox(text_xy, label, font=font)
                draw.rectangle(tb, fill=(0, 0, 0))
            except Exception:
                pass
            draw.text(text_xy, label, fill=color, font=font)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overview_img.save(out_dir / f"{prefix}_sam3_overview.png")

        # Per-object overlays
        for i, obj in enumerate(sam3_objects):
            mask_path = obj.get("mask_path")
            if not mask_path:
                continue
            m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if m.shape != (H, W):
                m = np.array(
                    Image.fromarray(m).resize((W, H), Image.NEAREST),
                    dtype=np.uint8,
                )
            mask_bool = m > 127
            per_img = rgb_np.astype(np.float32).copy()
            per_img[mask_bool] = per_img[mask_bool] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
            per_pil = Image.fromarray(per_img.clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(per_pil)
            bbox = obj.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
                label = f"obj_{i:02d} score={obj.get('score', 0.0):.3f} area={obj.get('mask_area', 0)}"
                text_xy = (x0 + 2, max(0, y0 - 22))
                try:
                    tb = draw.textbbox(text_xy, label, font=font)
                    draw.rectangle(tb, fill=(0, 0, 0))
                except Exception:
                    pass
                draw.text(text_xy, label, fill=(255, 255, 0), font=font)
            per_pil.save(out_dir / f"{prefix}_obj_{i:02d}_overlay.png")

        CONSOLE.log(
            f"[phase-0] saved SAM3 debug plots: {prefix}_sam3_overview.png + "
            f"{len(sam3_objects)} per-object overlays in {out_dir}"
        )

    @staticmethod
    def _resize_mask_to(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize a (H,W,C) mask to (target_h, target_w, C) using nearest interpolation."""
        if mask.shape[0] == target_h and mask.shape[1] == target_w:
            return mask
        return TF.interpolate(
            mask.permute(2, 0, 1).unsqueeze(0), size=(target_h, target_w), mode="nearest",
        ).squeeze(0).permute(1, 2, 0)

    # ---- CoTracker helpers ----

    def _build_tracking_rgb(self, batch) -> "torch.Tensor":
        """Live RGB for CoTracker, with the dataset mask composited onto the model background.

        Pixels where ``batch["mask"]`` is 0 (gripper / out-of-camera) are replaced with the
        Gazebo-blue background color so CoTracker cannot lock onto gripper texture or generate
        spurious features at the mask boundary. Returned at full resolution (no training downscale).
        """
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

    @staticmethod
    def _has_nonempty_mask(mask) -> bool:
        return mask is not None and bool(torch.any(mask > 0.5))

    @staticmethod
    def _erode_mask_to_inner_fraction(mask: "torch.Tensor", fraction: float) -> "torch.Tensor":
        """Erode a binary mask to keep only the inner ``fraction`` of its area.

        Uses cv2's L2 distance transform: each foreground pixel gets its
        Euclidean distance to the nearest background pixel. Pixels closest
        to the boundary are dropped first. The threshold is picked as the
        ``(1 - fraction)`` percentile of those distances, so the kept set
        is exactly ``fraction`` of the original area (modulo discretization).

        ``mask`` is (H, W, 1) or (H, W) float/bool on any device. Returns
        the same shape/dtype/device. ``fraction >= 1`` returns the input
        unchanged. Empty mask returns the input unchanged.
        """
        import cv2 as _cv2
        if mask is None or fraction >= 1.0:
            return mask
        squeeze_back = mask.ndim == 3
        mask_2d = mask.squeeze(-1) if squeeze_back else mask
        mask_np = (mask_2d.detach().cpu().numpy() > 0.5).astype(np.uint8)
        total = int(mask_np.sum())
        if total == 0:
            return mask
        dist = _cv2.distanceTransform(mask_np, _cv2.DIST_L2, 3)
        nonzero = dist[mask_np > 0]
        thresh = float(np.percentile(nonzero, (1.0 - fraction) * 100.0))
        eroded = (dist > thresh).astype(np.float32)
        eroded_t = torch.from_numpy(eroded).to(mask.device, dtype=mask.dtype)
        return eroded_t.unsqueeze(-1) if squeeze_back else eroded_t

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
        """Side-by-side previous→current frame with tracked points + lines."""
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
        h, w = prev_img.shape[:2]
        canvas = np.concatenate([prev_img, curr_img], axis=1)
        canvas = (canvas * 255).astype(np.uint8).copy()
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
                if 0 <= ly < h and 0 <= lx < 2 * w:
                    canvas[ly, lx] = line_color
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= py + dy < h and 0 <= px + dx < w:
                        canvas[py + dy, px + dx] = point_color
                    if 0 <= cy + dy < h and 0 <= cx + dx < 2 * w:
                        canvas[cy + dy, cx + dx] = point_color
        dbg = self._get_motion_debug_dir()
        dbg.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(dbg / f"{frame_name}_tracker.png")

    _TRACKER_LABELS = {"tapir": "TAPIR", "tapnext": "TAPNext", "cotracker": "CoTracker", "klt": "KLT", "xfeat": "XFeat"}

    def _apply_motion_estimator(self, camera, batch, current_mask=None) -> None:
        if self._motion_estimator is None:
            return
        tracker_kind = getattr(self.model.config, "dynamic_tracker", "cotracker")
        tracker_label = self._TRACKER_LABELS.get(tracker_kind, "Tracker")
        if not self._motion_estimator.ready:
            # KLT and XFeat both become "ready" only after their first
            # call has cached a previous frame, so we must still call
            # estimate_and_advance on the not-ready first tick.
            if tracker_kind not in ("klt", "xfeat"):
                frame_name = self.datamanager.get_current_dynamic_frame_name()
                CONSOLE.log(
                    f"[dynamic-gs] {tracker_label} skipped for {frame_name}: tracker not ready "
                    f"({self._motion_estimator.current_track_count} tracks, "
                    f"min={self._motion_estimator.min_track_points})"
                )
                return
        # --- Sub-timing: DN.3a build live RGB ---
        # Neural trackers (TAPIR/TAPNext/CoTracker) want the FULL natural
        # frame — they were trained on natural images, and any pre-mask
        # destroys the texture they rely on. Painting the gripper region
        # with a flat background colour (the legacy ``_build_tracking_rgb``
        # path) made tracks collapse to constant-pixel hot-spots wherever
        # any residual texture survived. The mask is still applied as a
        # POST-FILTER on the tracked 2D points so matches that land on the
        # gripper get dropped before RANSAC. KLT/XFeat keep the legacy
        # composite because they're tuned against the gripper-blue
        # background and would otherwise lock onto gripper corners.
        t = time.time()
        if tracker_kind in ("klt", "xfeat"):
            current_live_rgb = self._build_tracking_rgb(batch)
        else:
            current_live_rgb = self.model.get_live_rgb(batch, apply_training_downscale=False)
        self._timing["DN.3a_get_live_rgb"].append(time.time() - t)
        # KLT samples FAST keypoints inside (eroded object ∩ gripper-keep).
        # XFeat extracts globally then filters MATCHES inside the same
        # region; both need the rendered object mask + gripper mask.
        # TAPIR / CoTracker ignore the kwarg.
        # --- Sub-timing: DN.3j object mask render + erode (XFeat/KLT only) ---
        t_mask = time.time()
        current_object_mask = None
        if tracker_kind in ("klt", "xfeat"):
            # Cache the object mask for ``xfeat_object_mask_cache_ticks``
            # consecutive ticks (XFeat only — KLT samples keypoints from
            # this mask and the staleness would shift sampled locations,
            # so KLT always re-renders). The downstream keep_region
            # dilates by ``xfeat_object_search_radius_px`` (default 80 px),
            # giving plenty of slack for a few ticks of object drift.
            self._object_mask_tick_count = getattr(self, "_object_mask_tick_count", 0) + 1
            cache_ticks = (
                int(getattr(self.model.config, "xfeat_object_mask_cache_ticks", 1))
                if tracker_kind == "xfeat" else 1
            )
            cached = getattr(self, "_cached_object_mask", None)
            stale = (cached is None) or (self._object_mask_tick_count % max(cache_ticks, 1) == 1)
            if stale:
                with torch.no_grad():
                    # Splat-rasterized mask, then eroded to keep only the inner
                    # 85% of the area. The outer 15% is the low-opacity halo
                    # where keypoints would land on background pixels just
                    # outside the object's actual silhouette.
                    current_object_mask = self.model.render_object_mask(camera).detach()
                    current_object_mask = self._erode_mask_to_inner_fraction(
                        current_object_mask, _TRACKER_MASK_INNER_FRACTION,
                    )
                self._cached_object_mask = current_object_mask
            else:
                current_object_mask = cached
            if current_mask is None:
                current_mask = self.model._get_batch_mask(batch)
        self._timing["DN.3j_object_mask_render"].append(time.time() - t_mask)
        # --- Sub-timing: DN.3b-e estimate_and_advance — inner sub-timings come from motion_estimate.timings ---
        t = time.time()
        motion_estimate = self._motion_estimator.estimate_and_advance(
            current_rgb=current_live_rgb,
            current_depth=batch["depth_image"],
            current_camera=camera,
            current_mask=current_mask,
            current_object_mask=current_object_mask,
        )
        self._timing["DN.3_estimate_total"].append(time.time() - t)
        sub = motion_estimate.timings or {}
        # The estimator emits one entry per sub-step it completed; missing
        # keys (e.g. "predictor_forward" on a not-ready early return) are
        # logged as 0.0 so the report column stays aligned.
        self._timing["DN.3b_estimator_input_prep"].append(float(sub.get("input_prep", 0.0)))
        # ``predictor_forward`` is the TAPIR/CoTracker key; ``klt_forward``
        # is the KLT key; ``xfeat_extract`` is the XFeat sparse-extract
        # leg. All flow into the same DN.3c slot so the generic "NN
        # forward" column stays populated across backends.
        self._timing["DN.3c_predictor_forward"].append(
            float(sub.get("predictor_forward",
                          sub.get("xfeat_extract",
                                  sub.get("klt_forward", 0.0))))
        )
        # XFeat-only: pure sparse-extract time (no matching, no RANSAC).
        # Other backends leave this at 0.
        self._timing["DN.3c_xfeat_extract"].append(float(sub.get("xfeat_extract", 0.0)))
        # XFeat-only: LighterGlue matcher forward pass (or MNN fallback
        # when LighterGlue isn't loaded).
        self._timing["DN.3i_lighterglue_match"].append(
            float(sub.get("lighterglue_match", 0.0))
        )
        self._timing["DN.3d_postprocess"].append(float(sub.get("postprocess", 0.0)))
        self._timing["DN.3e_ransac_kabsch"].append(float(sub.get("ransac_kabsch", 0.0)))
        # KLT-only: per-frame keypoint resample (FAST detect inside the
        # eroded sample region). TAPIR/CoTracker leave this at 0.
        self._timing["DN.3h_resample"].append(float(sub.get("resample", 0.0)))
        # Finer-grained breakdown of input_prep (TAPIR-only; cotracker
        # path leaves these zero). Each entry is a CUDA-sync candidate.
        self._timing["DN.3b1_prep_rgb_cpu"].append(float(sub.get("prep_rgb_cpu", 0.0)))
        self._timing["DN.3b2_prep_depth_cpu"].append(float(sub.get("prep_depth_cpu", 0.0)))
        self._timing["DN.3b3_prep_intrinsics"].append(float(sub.get("prep_intrinsics", 0.0)))
        self._timing["DN.3b4_prep_c2w"].append(float(sub.get("prep_c2w", 0.0)))
        # Finer-grained breakdown of predictor_forward (TAPIR-only).
        self._timing["DN.3c1_preprocess_frame"].append(float(sub.get("preprocess_frame", 0.0)))
        self._timing["DN.3c2_feature_grids"].append(float(sub.get("feature_grids", 0.0)))
        self._timing["DN.3c3_estimate_traj"].append(float(sub.get("estimate_traj", 0.0)))
        self._timing["DN.3c4_tracks_to_cpu"].append(float(sub.get("tracks_to_cpu", 0.0)))
        # --- Sub-timing: DN.3f debug I/O (motion log text + tracked-points overlay PNG; gated by save_debug_images) ---
        t = time.time()
        frame_name = self.datamanager.get_current_dynamic_frame_name()
        if self.config.save_debug_images:
            self._write_motion_log(frame_name, motion_estimate)
            self._save_motion_debug(frame_name, motion_estimate)
        self._timing["DN.3f_debug_io"].append(time.time() - t)
        if not motion_estimate.success:
            mean_res_mm = motion_estimate.mean_residual * 1000.0 if motion_estimate.mean_residual != float("inf") else float("inf")
            med_res_mm = motion_estimate.median_residual * 1000.0 if motion_estimate.median_residual != float("inf") else float("inf")
            CONSOLE.log(
                f"[dynamic-gs] {tracker_label} rigid motion unavailable for {frame_name}: "
                f"raw={motion_estimate.raw_visible_count}, "
                f"mask={motion_estimate.mask_visible_count}, "
                f"depth={motion_estimate.depth_valid_count}, "
                f"correspondences={motion_estimate.correspondence_count}, "
                f"inliers={motion_estimate.inlier_count}, "
                f"mask_fallback={motion_estimate.used_mask_fallback}, "
                f"resid_mm(mean/med)={mean_res_mm:.1f}/{med_res_mm:.1f}"
            )
            return
        # --- Sub-timing: DN.3g apply rigid transform (CUDA tensor op on flagged Gaussians: means rotation + translation, quaternion compose) ---
        t = time.time()
        moved_count = self.model.apply_rigid_object_transform_from_reference(
            motion_estimate.rotation, motion_estimate.translation,
        )
        self._timing["DN.3g_apply_transform"].append(time.time() - t)
        if moved_count == 0:
            CONSOLE.log(
                f"[dynamic-gs] {tracker_label} estimated motion for {frame_name}, "
                "but no object Gaussians were moved. Check object_flags/reference pose consistency."
            )
        # Per-frame success log is muted in tracking-only mode — at
        # 10+ Hz it floods the console and buries the [tracker-rate]
        # line below. Failure logs above (skipped / unavailable) still
        # fire because they're rare and diagnostic.
        if not self.config.disable_dynamic_optimization:
            CONSOLE.log(
                f"[dynamic-gs] {tracker_label} rigid motion -> {frame_name}, moved={moved_count}, "
                f"inliers={motion_estimate.inlier_count}/{motion_estimate.correspondence_count}, "
                f"median residual={motion_estimate.median_residual:.5f}, "
                f"mask_fallback={motion_estimate.used_mask_fallback}"
            )

    def _initialize_motion_estimator(self, rgb, depth, camera, mask) -> None:
        """Seed CoTracker with the D0 reference frame.

        Samples ``query_point_count`` 2D pixel points inside the object mask,
        back-projects via depth+intrinsics+c2w to get the world-frame
        reference 3D positions, and stores ``rgb`` as the previous frame.
        Subsequent ``estimate_and_advance`` calls track those points to the
        new frame and run RANSAC-Kabsch to recover (R, t).
        """
        if not self.model.config.enable_cotracker_rigid_motion:
            return
        tracker_kind = getattr(self.model.config, "dynamic_tracker", "cotracker")
        if tracker_kind == "tapnext":
            from .utils.tapnext_motion import TapnextMotionEstimator
            self._motion_estimator = TapnextMotionEstimator(
                device=self.model.device,
                query_point_count=self.model.config.tapnext_query_point_count,
                min_track_points=self.model.config.tapnext_min_track_points,
                ransac_iterations=self.model.config.tapnext_ransac_iterations,
                ransac_inlier_threshold=self.model.config.tapnext_ransac_inlier_threshold,
                checkpoint_path=self.model.config.tapnext_checkpoint_path,
                input_resolution=self.model.config.tapnext_input_resolution,
                visibility_threshold=self.model.config.tapnext_visibility_threshold,
            )
            tracker_label = "TAPNext"
        elif tracker_kind == "tapir":
            from .utils.tapir_motion import TapirMotionEstimator
            self._motion_estimator = TapirMotionEstimator(
                device=self.model.device,
                query_point_count=self.model.config.tapir_query_point_count,
                min_track_points=self.model.config.tapir_min_track_points,
                ransac_iterations=self.model.config.tapir_ransac_iterations,
                ransac_inlier_threshold=self.model.config.tapir_ransac_inlier_threshold,
                checkpoint_path=self.model.config.tapir_checkpoint_path,
                input_resolution=self.model.config.tapir_input_resolution,
                visibility_threshold=self.model.config.tapir_visibility_threshold,
            )
            tracker_label = "TAPIR"
        elif tracker_kind == "klt":
            from .utils.klt_motion import KltMotionEstimator
            self._motion_estimator = KltMotionEstimator(
                device=self.model.device,
                query_point_count=self.model.config.klt_query_point_count,
                min_track_points=self.model.config.klt_min_track_points,
                ransac_iterations=self.model.config.klt_ransac_iterations,
                ransac_inlier_threshold=self.model.config.klt_ransac_inlier_threshold,
                pyramid_levels=self.model.config.klt_pyramid_levels,
                window_size=self.model.config.klt_window_size,
                lk_iterations=self.model.config.klt_lk_iterations,
                lk_eps=self.model.config.klt_lk_eps,
                fast_threshold=self.model.config.klt_fast_threshold,
                sample_inner_area_ratio=self.model.config.klt_sample_inner_area_ratio,
            )
            tracker_label = "KLT"
        elif tracker_kind == "xfeat":
            # XFeat replaces the FAST/Shi-Tomasi-based feature
            # extraction that KLT uses: the CNN detects AND describes in
            # one forward pass, then MNN over the descriptors gives
            # frame-to-frame matches directly. There is no separate
            # corner-detector or descriptor step to load.
            from .utils.xfeat_motion import XFeatMotionEstimator
            self._motion_estimator = XFeatMotionEstimator(
                device=self.model.device,
                top_k=self.model.config.xfeat_top_k,
                detection_threshold=self.model.config.xfeat_detection_threshold,
                min_cossim=self.model.config.xfeat_min_cossim,
                min_track_points=self.model.config.xfeat_min_track_points,
                ransac_iterations=self.model.config.xfeat_ransac_iterations,
                ransac_inlier_threshold=self.model.config.xfeat_ransac_inlier_threshold,
                weights_path=self.model.config.xfeat_weights_path,
            )
            tracker_label = "XFeat"
        else:
            self._motion_estimator = CoTrackerMotionEstimator(
                device=self.model.device,
                query_point_count=self.model.config.cotracker_query_point_count,
                min_track_points=self.model.config.cotracker_min_track_points,
                ransac_iterations=self.model.config.cotracker_ransac_iterations,
                ransac_inlier_threshold=self.model.config.cotracker_ransac_inlier_threshold,
                checkpoint_path=self.model.config.cotracker_checkpoint_path,
                hub_repo=self.model.config.cotracker_hub_repo,
                hub_model=self.model.config.cotracker_hub_model,
                predictor_resolution=self.model.config.cotracker_predictor_resolution,
                predictor_iters=self.model.config.cotracker_predictor_iters,
            )
            tracker_label = "CoTracker"
        seeded = self._motion_estimator.initialize(
            rgb=rgb, depth=depth, camera=camera, mask=mask,
        )
        CONSOLE.log(
            f"[dynamic-gs] {tracker_label} reference seed on D0 -> "
            f"fast={self._motion_estimator.last_init_fast_point_count}, "
            f"sampled={self._motion_estimator.last_init_sampled_count}, "
            f"depth_valid={self._motion_estimator.last_init_depth_valid_count}, "
            f"dense_fallback={self._motion_estimator.last_init_used_dense_fallback}, "
            f"tracks={seeded}, ready={self._motion_estimator.ready}"
        )
        if seeded < self._motion_estimator.min_track_points:
            CONSOLE.log(
                f"[dynamic-gs] {tracker_label} seeded too few D0 points: "
                f"{seeded} < min_track_points={self._motion_estimator.min_track_points}"
            )

    # ---- Image helpers ----

    def _maybe_save_live_optim_debug(self, step: int, frame, model_outputs, effective_mask) -> None:
        """Dump two PNGs per live optim step, paired by name so they
        sort side-by-side in a file browser:

        - ``stepNNNNNNN_1_render_w_mask.png`` — rendered RGB with the
          effective loss mask (CDN ∩ not-current-object ∩ not-gripper)
          overlaid in red. This is "what the optimizer sees as the
          prediction, restricted to the pixels that actually
          contribute to the loss."
        - ``stepNNNNNNN_2_live.png`` — the live ROS image being
          compared against (the ground truth for the loss).

        Throttled by ``save_live_optim_debug_every``. Best-effort: any
        I/O failure is swallowed so a flaky disk can't kill training.
        """
        if not self.config.save_live_optim_debug:
            return
        every = max(1, int(self.config.save_live_optim_debug_every))
        if (step % every) != 0:
            return
        try:
            try:
                data_root = Path(self.datamanager.config.data)
            except AttributeError:
                data_root = Path(".")
            out_dir = data_root / "dynamic_scene" / "debug" / "live_optim"
            tag = f"step{step:07d}"

            render_rgb = model_outputs["rgb"]
            live_batch = frame.live_batch
            live_rgb = live_batch.get("image") if isinstance(live_batch, dict) else None
            if isinstance(live_rgb, torch.Tensor):
                live_rgb_t = live_rgb.detach().float()
                if live_rgb_t.max() > 1.5:
                    live_rgb_t = live_rgb_t / 255.0
            else:
                live_rgb_t = None

            self._save_overlay(render_rgb, effective_mask, out_dir / f"{tag}_1_render_w_mask.png")
            if live_rgb_t is not None:
                self._save_image(live_rgb_t, out_dir / f"{tag}_2_live.png")
        except Exception as exc:
            CONSOLE.log(f"[live-debug] save failed at step {step}: {exc}")

    @staticmethod
    def _save_image(image, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = image.detach().float().clamp(0.0, 1.0)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)
        image_uint8 = tensor.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    @staticmethod
    def _save_image_with_points(image, points_xy, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor = image.detach().float().clamp(0.0, 1.0)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)

        image_uint8 = tensor.mul(255).byte().cpu().numpy()
        pil_image = Image.fromarray(image_uint8)

        if points_xy is not None and points_xy.numel() > 0:
            draw = ImageDraw.Draw(pil_image)
            radius = max(2, int(round(0.006 * max(pil_image.size))))
            for point in points_xy.detach().cpu().tolist():
                x = int(round(point[0]))
                y = int(round(point[1]))
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(255, 255, 255))

        pil_image.save(path)

    @staticmethod
    def _resize_points(points_xy, source_shape, target_shape):
        if points_xy is None or points_xy.numel() == 0:
            return points_xy
        source_h, source_w = source_shape[:2]
        target_h, target_w = target_shape[:2]
        scaled = points_xy.detach().clone().float()
        scaled[:, 0] *= float(target_w) / float(max(source_w, 1))
        scaled[:, 1] *= float(target_h) / float(max(source_h, 1))
        return scaled

    @staticmethod
    def _save_depth_image(depth, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = depth.detach().float()
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        valid = torch.isfinite(tensor) & (tensor > 0.0)
        image = torch.zeros((*tensor.shape, 3), dtype=torch.float32, device=tensor.device)
        if bool(valid.any()):
            valid_values = tensor[valid]
            depth_min = float(valid_values.min().item())
            depth_max = float(valid_values.max().item())
            if depth_max > depth_min:
                normalized = (tensor - depth_min) / (depth_max - depth_min)
            else:
                normalized = torch.zeros_like(tensor)
            normalized = (1.0 - normalized).clamp(0.0, 1.0)
            image[valid] = normalized[valid][..., None].expand(-1, 3)
        image_uint8 = image.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    @staticmethod
    def _save_overlay(rgb, mask, path, color=(1.0, 0.0, 0.0), alpha=0.5):
        """Save rgb with a transparent colored mask overlay."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = rgb.detach().float().clamp(0.0, 1.0).clone()
        m = mask.detach().float()
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim == 2:
            m = (m > 0.5).float()
        # Resize mask to match image if needed
        if m.shape[:2] != img.shape[:2]:
            m = TF.interpolate(
                m.unsqueeze(0).unsqueeze(0), size=img.shape[:2], mode="nearest",
            ).squeeze(0).squeeze(0)
        overlay = torch.tensor(color, device=img.device).view(1, 1, 3)
        img[m > 0.5] = img[m > 0.5] * (1 - alpha) + overlay.expand_as(img)[m > 0.5] * alpha
        image_uint8 = img.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    def _get_debug_dir(self) -> Path:
        return Path(self.datamanager.config.data) / self.datamanager.config.dynamic_subdir / "debug"

    @torch.no_grad()
    def _render_from_camera(self, camera):
        """Render from camera in training mode (to get training-resolution output)."""
        was_training = self.model.training
        self.model.train()
        try:
            return self.model.get_outputs(camera.to(self.model.device))
        finally:
            if not was_training:
                self.model.eval()

    def _compute_change_mask(self, rendered_rgb, rendered_depth, live_rgb, gt_depth, gripper_mask, object_mask):
        """Compute change mask between render and live, excluding gripper + object regions."""
        target_h, target_w = rendered_rgb.shape[:2]
        valid_mask = None
        if object_mask is not None:
            obj = object_mask.float()
            if obj.ndim == 2:
                obj = obj[..., None]
            obj = self._resize_mask_to(obj.to(self.model.device), target_h, target_w)
            valid_mask = 1.0 - obj
        if gripper_mask is not None:
            grip = gripper_mask.float().to(self.model.device)
            if grip.ndim == 2:
                grip = grip[..., None]
            grip = self._resize_mask_to(grip, target_h, target_w)
            valid_mask = grip * valid_mask if valid_mask is not None else grip

        change_mask = build_change_mask(
            rendered_depth, gt_depth,
            pred_rgb=rendered_rgb, gt_rgb=live_rgb,
            valid_mask=valid_mask,
            depth_threshold=self.model.config.change_mask_depth_threshold,
            rgb_threshold=self.model.config.change_mask_rgb_threshold,
            use_rgb=self.model.config.change_mask_use_rgb,
            blur_kernel_size=self.model.config.change_mask_blur_kernel_size,
            blur_sigma=self.model.config.change_mask_blur_sigma,
            filter_radius=self.model.config.change_mask_filter_radius,
            min_component_size=self.model.config.change_mask_min_component_size,
        )
        if self.model.config.active_mask_dilate_radius > 0:
            change_mask = dilate_binary_mask(change_mask, self.model.config.active_mask_dilate_radius)
        # Re-clip to valid_mask so the dilation cannot bleed back into the
        # excluded object/gripper regions.
        if valid_mask is not None:
            change_mask = change_mask * valid_mask
        return change_mask

    # ---- Live-mode helpers ----

    def _start_stdin_stop_watcher(self) -> None:
        """Daemon thread that flips ``_live_stop_requested`` on 'stop'.

        Reads stdin line-by-line; non-'stop' lines are ignored. The
        flag is checked at the top of ``_dynamic_get_train_loss_dict``,
        which then short-circuits the dynamic loop and returns a
        zero-loss dict so the trainer keeps stepping (viewer stays
        responsive) without changing weights.
        """
        import threading
        import sys

        def _watch():
            try:
                for line in sys.stdin:
                    if line.strip().lower() == "stop":
                        self._live_stop_requested = True
                        CONSOLE.log("[live] stop requested; freezing model and keeping viewer alive")
                        return
            except Exception:
                return

        threading.Thread(target=_watch, name="live_stdin_watcher", daemon=True).start()
        # PROBLEM: stdin is shared with the user prompt earlier in
        # live_session — by the time this watcher starts, the prompts
        # have already returned and stdin is exclusively ours. If
        # someone re-introduces interactive prompts mid-training, the
        # watcher will eat their lines.

    def _wrap_live_tuple_as_batch(self, frame) -> tuple[object, dict]:
        """Build the (camera, batch) pair the rest of the pipeline expects.

        Mirrors ``DynamicGSDataManager._get_dynamic_batch`` so that
        every model method downstream — ``get_outputs``,
        ``get_live_rgb``, ``_get_batch_mask``, ``_get_gt_depth``,
        ``apply_rigid_object_transform_from_reference`` — sees a
        familiar shape and is none the wiser.
        """
        from .utils.live_shm_reader import cameras_from_live_frame

        device = self.model.device
        camera = cameras_from_live_frame(
            frame=frame,
            intrinsics=self._live_subscriber.intrinsics,
            device=device,
            cam_idx=0,
        )

        # NOTE: a previous attempt called ``.pin_memory()`` per tensor
        # to get async H2D. That regressed wrap_batch from 11 ms to
        # 32 ms because ``Tensor.pin_memory()`` is itself a synchronous
        # copy into a freshly-allocated pinned buffer (~20 ms total for
        # rgb + depth + mask). Doing it right requires a long-lived
        # pinned staging buffer that the shm reader writes directly
        # into — left as future work. For now stick with the plain
        # pageable upload.
        rgb_rgb = np.ascontiguousarray(frame.rgb_bgr[..., ::-1])
        image_t = torch.from_numpy(rgb_rgb).to(device, non_blocking=True)
        # depth is already float32 metres from the publisher; no rescale.
        depth_t = torch.from_numpy(frame.depth_m).to(device, non_blocking=True)
        mask_bool = (frame.mask_keep > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_bool).to(device, non_blocking=True).unsqueeze(-1)

        batch = {
            "image": image_t,
            "image_idx": 0,
            "mask": mask_t,
            "depth_image": depth_t,
        }
        return camera, batch
        # PROBLEM: `image_idx=0` re-uses the static frame 0 slot in the
        # camera optimizer. If the trainer's optimizer has already
        # updated slot 0 from static training, that pose adjustment
        # bleeds into every live frame's render. We only call this in
        # the dynamic phase where the camera optimizer is gated by
        # phase, but the slot collision is worth a glance during a
        # future cleanup.

    def _force_viewer_rerender(self) -> None:
        """Trigger an immediate re-render on every connected viser
        client. Used right after the tracker mutates object Gaussian
        means so the visual update rate tracks the tracker rate
        instead of being throttled by ``update_scene``'s step-count
        gate. Best-effort: silently no-ops if the viewer isn't ready
        or the import fails."""
        trainer = self._trainer
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
            # Bypass `sm.action()`'s state-machine filters and
            # directly push a high-res render. `action()` ignores
            # "step" while state is "low_move" (set by viser camera
            # on_update events from the browser), which would cap us
            # at the tiny low-state resolution forever. Setting
            # `state = "low_static"` lets the next transition
            # ("low_static" + "step" → "high") promote us cleanly,
            # so `_render_img` uses `max_res` instead of the
            # vis_rays_per_sec / target_fps fallback (~60 px).
            try:
                sm.state = "low_static"
                sm.next_action = RenderAction("step", camera_state)
                sm.render_trigger.set()
            except Exception:
                continue

    def _tracker_tick_live(self) -> None:
        """Live-mode replacement for ``_tracker_tick(frame_idx)``.

        Pulls the most recent ROS tuple, dedupes against the last one
        we processed, runs FP `track_one`, and — if the keyframe
        filter accepts AND CDN clears the min-pixel gate — pushes a
        capture-time CDN onto the optim pool.
        """
        # --- LIVE.tick_total measures whole-tick wall-clock for ticks
        # that actually do work (i.e. that pass the early-return gates
        # below). Compare to LIVE.peek_latest + LIVE.wrap_batch + ... to
        # find the unaccounted gap.
        tick_t0 = time.time()
        # --- LIVE.between_tick_gap: wall-clock time elapsed since the
        # PREVIOUS tick_total recording finished. If this is large
        # while tick_total is small, work happening OUTSIDE
        # _tracker_tick_live (trainer's outer loop, optim step,
        # callbacks, viewer render) is the bottleneck — not the tracker
        # itself.
        if hasattr(self, "_last_tick_end_t"):
            self._timing["LIVE.between_tick_gap"].append(tick_t0 - self._last_tick_end_t)
        # --- LIVE.frame_age: how stale was the most recent ROS frame
        # at the moment we picked it up? If this is consistently
        # similar to between_tick_gap, the supply (Gazebo camera plugin)
        # is publishing at a comparable rate to our consumption — i.e.
        # we're not starved by ROS. If frame_age << gap, we're
        # consuming slower than ROS supplies (we're the bottleneck).
        # If frame_age >> gap, ROS is publishing faster than we read —
        # ideal scenario.
        # --- LIVE.frame_dt_seq: stamp delta between this frame and the
        # previously processed frame. If this is ~5 Hz (200ms) it tells
        # us the camera plugin is rate-limited even if Gazebo physics
        # runs faster. If it's ~30 Hz (33ms) the camera is fine and the
        # consumer is the bottleneck.
        sub = self._live_subscriber
        if sub is None:
            return
        # --- LIVE.peek_latest: ROS subscriber atomic read of the most
        # recent (rgb, depth, pose, mask) tuple. Should be sub-ms.
        t0 = time.time()
        latest = sub.peek_latest()
        self._timing["LIVE.peek_latest"].append(time.time() - t0)
        if latest is None:
            return
        if (
            self._live_last_processed_stamp is not None
            and latest.stamp_sec == self._live_last_processed_stamp
        ):
            # Dedup return — count it (global + per-window) so the
            # [tracker-rate] line can show the dedup ratio.
            self._timing.setdefault("LIVE.dedup_returns", []).append(0.0)
            self._live_dedup_window_count = getattr(self, "_live_dedup_window_count", 0) + 1
            return
        # Frame is fresh. Record stamp-delta and wall-clock age before
        # we update the last-processed stamp.
        if self._live_last_processed_stamp is not None:
            self._timing["LIVE.frame_dt_seq"].append(
                latest.stamp_sec - self._live_last_processed_stamp
            )
        # Wall-clock age: how long ago did this frame land in the
        # subscriber buffer? Compares ROS stamp to current wall clock.
        # Note: assumes ROS stamps are wall clock (true with use_sim_time=False).
        # If sim time is on, this is sim-time age, still useful for
        # comparing against sim-time tick gap.
        self._timing["LIVE.frame_age"].append(max(0.0, time.time() - latest.stamp_sec))
        self._live_last_processed_stamp = latest.stamp_sec

        # --- LIVE.wrap_batch: 3x pageable H2D copies (rgb 5MB, depth
        # 3.6MB, mask 0.9MB at 720p) + Cameras object construction.
        t0 = time.time()
        camera, batch = self._wrap_live_tuple_as_batch(latest)
        self._timing["LIVE.wrap_batch"].append(time.time() - t0)
        frame_name = f"live_{latest.seq:06d}"

        # --- LIVE.gt_setup: model.composite_with_background + _get_gt_depth
        # + _get_batch_mask. Pure GPU ops on tensors already on device.
        t0 = time.time()
        bg = self.model._get_background_color()
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), bg)
        gt_depth = self.model._get_gt_depth(batch)
        gripper_mask = self.model._get_batch_mask(batch)
        self._timing["LIVE.gt_setup"].append(time.time() - t0)

        is_first = self._global_frame_counter == 0
        if is_first:
            live_rgb = self.model.get_live_rgb(batch, background=bg, apply_training_downscale=True)
            init_debug_dir = self.datamanager.get_initialization_debug_dir()
            init_artifact_dir = self.datamanager.get_initialization_artifact_dir()
            self._prepare_frame_0(
                camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
                frame_name, init_debug_dir, init_artifact_dir,
            )
            cdn = self.model.change_mask_image.detach().clone()
            self._optim_pool.push(OptimFrame(
                frame_idx=0, camera=camera, cdn=cdn, live_batch=batch,
            ))
            self._global_frame_counter += 1
            return

        # CoTracker advances on every live frame for object-pose continuity.
        t0 = time.time()
        if self._sam3d_inserted and self._motion_estimator is not None:
            self._apply_motion_estimator(camera, batch)
            # Object means just changed — force a viewer re-render
            # immediately. Otherwise update_scene's render_freq
            # step-count throttle caps visual update to ~0.5-2 Hz.
            self._force_viewer_rerender()
        self._timing["DN.3_tracker_motion"].append(time.time() - t0)

        # Rolling tick-rate report. Plain `print` (not CONSOLE.log) so
        # the line surfaces above nerfstudio's rich progress bar and
        # writer logs. Independent of what Viser paints: this is the
        # actual rate at which object Gaussians get moved.
        now = time.time()
        if not hasattr(self, "_live_tick_window_t0"):
            self._live_tick_window_t0 = now
            self._live_tick_window_count = 0
        if not hasattr(self, "_live_dedup_window_count"):
            self._live_dedup_window_count = 0
        self._live_tick_window_count += 1
        if (now - self._live_tick_window_t0) >= 2.0:
            elapsed = now - self._live_tick_window_t0
            fresh = self._live_tick_window_count
            dedup = self._live_dedup_window_count
            hz = fresh / elapsed
            # dedup ratio: fraction of tick CALLS that found no new ROS
            # frame. ~0% => loop-bound (tick-rate == trainer-loop-rate,
            # bottleneck is outside the tick). High => supply-bound
            # (publisher/synchronizer isn't delivering fresh frames).
            dedup_ratio = dedup / max(1, fresh + dedup)
            # outside-tick: mean wall-clock spent OUTSIDE _tracker_tick_live
            # over this window (trainer no-op step + callbacks + viewer
            # render). Large => the trainer loop / viewer is the gate.
            gaps = self._timing.get("LIVE.between_tick_gap", [])
            gap_ms = (sum(gaps[-fresh:]) / fresh * 1000.0) if (gaps and fresh) else 0.0
            tracker_ms = self._timing["DN.3_tracker_motion"][-1] * 1000
            tracker_label = {
                "tapir": "TAPIR",
                "tapnext": "TAPNext",
                "cotracker": "CoTracker",
                "klt": "KLT",
                "xfeat": "XFeat",
            }.get(getattr(self.model.config, "dynamic_tracker", "cotracker"), "CoTracker")
            # Per-window mean over last `fresh` ticks for the inner DN.3 sub-timers
            # so we can see WHERE the tick budget is going. Computed inline so we
            # don't have to kill the run to read timing_report.txt.
            def _last_mean_ms(key, n):
                vals = self._timing.get(key, [])
                if not vals or n <= 0:
                    return 0.0
                vals = vals[-n:]
                return sum(vals) / len(vals) * 1000.0
            n = max(1, fresh)
            a = _last_mean_ms("DN.3a_get_live_rgb", n)
            b = _last_mean_ms("DN.3b_estimator_input_prep", n)
            c = _last_mean_ms("DN.3c_predictor_forward", n)
            d = _last_mean_ms("DN.3d_postprocess", n)
            e = _last_mean_ms("DN.3e_ransac_kabsch", n)
            g = _last_mean_ms("DN.3g_apply_transform", n)
            tot = _last_mean_ms("DN.3_tracker_motion", n)
            est = _last_mean_ms("DN.3_estimate_total", n)
            tracker_kind_now = getattr(self.model.config, "dynamic_tracker", "cotracker")
            if tracker_kind_now == "xfeat":
                xf = _last_mean_ms("DN.3c_xfeat_extract", n)
                lg = _last_mean_ms("DN.3i_lighterglue_match", n)
                breakdown = (
                    f"rgb_prep={a:.1f} input_prep={b:.1f} "
                    f"xfeat={xf:.1f} match={lg:.1f} ransac={e:.1f} apply={g:.1f} "
                    f"estimate_total={est:.1f}  total={tot:.1f}ms"
                )
            else:
                breakdown = (
                    f"rgb_prep={a:.1f} input_prep={b:.1f} "
                    f"net+match={c:.1f} postproc={d:.1f} ransac={e:.1f} apply={g:.1f} "
                    f"estimate_total={est:.1f}  total={tot:.1f}ms"
                )
            print(
                f"\n==> [tracker-rate] {hz:.1f} Hz over {elapsed:.1f}s "
                f"({tracker_label} last={tracker_ms:.0f}ms | "
                f"dedup {dedup_ratio*100:.0f}% | outside-tick {gap_ms:.0f}ms)"
                f"\n    breakdown(mean/tick over {n}): {breakdown}\n",
                flush=True,
            )
            self._live_tick_window_t0 = now
            self._live_tick_window_count = 0
            self._live_dedup_window_count = 0

        # Tracking-only mode: skip the RDN render, object mask, CDN
        # compute, and pool push. The object Gaussians have already
        # been moved by `_apply_motion_estimator`; the viewer pulls
        # the latest pose on its own render.
        if self.config.disable_dynamic_optimization:
            self._timing["LIVE.tick_total"].append(time.time() - tick_t0); self._last_tick_end_t = time.time()
            self._global_frame_counter += 1
            return

        # Keyframe filter is purely pose-based (translation + rotation
        # thresholds vs. the set of accepted c2w's), so it's cheap and
        # doesn't need any rendering. Run it FIRST: rejected frames now
        # cost zero GPU on RDN + object mask + CDN (~258 ms saved per
        # rejected frame). The CDN pixel-count gate still runs after
        # the renders below since it requires the change mask itself.
        # --- LIVE.keyframe_filter: pose-only check (cheap, pure CPU)
        t0 = time.time()
        c2w_3x4 = camera.camera_to_worlds[0].detach().cpu()
        accepted = (
            self._dynamic_keyframe_filter is None
            or self._dynamic_keyframe_filter.accept(c2w_3x4)
        )
        self._timing["LIVE.keyframe_filter"].append(time.time() - t0)
        if not accepted:
            self._timing["LIVE.tick_total"].append(time.time() - tick_t0); self._last_tick_end_t = time.time()
            self._global_frame_counter += 1
            return

        t0 = time.time()
        rdn_outputs = self._render_from_camera(camera)
        self._timing["DN.5_render_rdn"].append(time.time() - t0)
        rdn_rgb = rdn_outputs["rgb"]
        rdn_depth = rdn_outputs["depth"]

        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["DN.6_render_object_mask"].append(time.time() - t0)

        t0 = time.time()
        cdn = self._compute_change_mask(
            rdn_rgb, rdn_depth, gt_rgb, gt_depth, gripper_mask, rendered_obj_mask,
        )
        self._timing["DN.7_change_mask_cdn"].append(time.time() - t0)

        cdn_px = int((cdn[..., 0] > 0.5).sum().item()) if cdn.ndim >= 3 else int((cdn > 0.5).sum().item())
        if cdn_px < self.config.optim_pool_min_change_pixels:
            self._timing["LIVE.tick_total"].append(time.time() - tick_t0); self._last_tick_end_t = time.time()
            self._global_frame_counter += 1
            return

        # Verify the gripper mask attached to this frame's batch is
        # the real URDF-rendered one (not the publisher's all-1
        # placeholder). A real mask has some zero pixels where the
        # robot is in view; an all-1 mask would let the gripper
        # region leak into CDN and pull the loss toward the robot.
        mask_t = batch.get("mask") if isinstance(batch, dict) else None
        if mask_t is not None and bool((mask_t > 0).all().item()):
            CONSOLE.log(
                f"[live] {frame_name}: dropping frame — gripper mask is all-1 "
                f"(publisher fell back to placeholder; image-mask mismatch)."
            )
            self._timing["LIVE.tick_total"].append(time.time() - tick_t0); self._last_tick_end_t = time.time()
            self._global_frame_counter += 1
            return

        self._optim_pool.push(OptimFrame(
            frame_idx=0, camera=camera, cdn=cdn.detach().clone(),
            live_batch=batch,
        ))
        CONSOLE.log(
            f"[live] {frame_name}: change px={cdn_px}, pool_size={len(self._optim_pool)}"
        )
        self._timing["LIVE.tick_total"].append(time.time() - tick_t0); self._last_tick_end_t = time.time()
        self._global_frame_counter += 1
        # PROBLEM: every accepted live frame holds its `camera` and
        # `cdn` tensors on GPU until evicted from the pool. With
        # capacity=15 and ~400×400×4B = ~640KB per CDN, that's tiny;
        # the camera ray helpers are heavier but still small. If the
        # operator drives the camera through hundreds of poses very
        # quickly, the FIFO drop-oldest keeps memory bounded.

    def _pick_closest_object_to_camera(self, camera) -> int:
        """Choose the prefused instance whose 3D centroid is nearest the camera.

        Live-mode replacement for the 2D anchor-distance pick at
        ``(W/2, 0.75 H)``. Used at D0 because the recorded teleop heuristic
        ("gripper-held object lives in the lower-centre of the image")
        does not generalise — in live mode the operator approaches the
        target from arbitrary viewpoints, so the closest object in
        world space is a more reliable signal for "what's about to be
        manipulated".
        """
        instance_ids = self.model.object_instance_ids.squeeze(-1)
        prefused_mask = instance_ids > 0
        if not bool(prefused_mask.any()):
            return 0
        unique_ids = torch.unique(instance_ids[prefused_mask])
        if unique_ids.numel() == 0:
            return 0
        cam_pos = camera.camera_to_worlds[0, :3, 3].to(self.model.means.device)
        best_id = 0
        best_dist = float("inf")
        for uid in unique_ids:
            uid_val = int(uid.item())
            mask = instance_ids == uid_val
            if not bool(mask.any()):
                continue
            centroid = self.model.means[mask].detach().mean(dim=0)
            dist = float(torch.linalg.norm(centroid - cam_pos).item())
            if dist < best_dist:
                best_dist = dist
                best_id = uid_val
        CONSOLE.log(f"[live] D0 closest-to-camera: instance_id={best_id} (dist={best_dist:.3f}m)")
        return best_id
        # PROBLEM: if two prefused objects sit at the same distance
        # (within float noise), the iteration order picks one by
        # `torch.unique` ordering rather than any user-meaningful
        # criterion. The operator can re-position the camera and try
        # again — this is a D0-only decision, not a per-frame one.

    # ---- Step/phase management ----

    def _total_train_steps(self) -> int:
        return self.config.static_num_steps + self.total_dynamic_steps

    def _phase_for_step(self, step: int) -> Literal["static", "dynamic"]:
        # Live mode: dynamic_scene/ holds a single stub frame just to
        # make the dataparser happy. The real gate is the MSSIM
        # convergence check; without it the trainer would sit in the
        # static phase until ``static_num_steps`` regardless of how
        # many live views the operator captured.
        if self.config.live:
            if (
                self._static_converged_step is not None
                and step >= self._static_converged_step
            ):
                return "dynamic"
            if step < self.config.static_num_steps:
                return "static"
            return "dynamic"

        if self.total_dynamic_frames == 0 or not self._accepted_dynamic_frames:
            return "static"
        # Early-convergence transition: once the MSSIM-based check sees the
        # scene match GT, we don't wait for ``static_num_steps``.
        if self._static_converged_step is not None and step >= self._static_converged_step:
            return "dynamic"
        if step < self.config.static_num_steps:
            return "static"
        return "dynamic"

    def _dynamic_frame_for_step(self, step: int) -> int:
        """Map a global step to the dataset frame index to optimize.

        Indirects through ``_accepted_dynamic_frames`` so rejected
        keyframes consume zero steps. The trainer's step counter sees
        ``static_num_steps + K_accepted · dynamic_steps_per_frame``
        steps total (set by ``_total_train_steps``).
        """
        dynamic_step = max(step - self.config.static_num_steps, 0)
        accepted_idx = dynamic_step // self.config.dynamic_steps_per_frame
        accepted_idx = min(accepted_idx, len(self._accepted_dynamic_frames) - 1)
        return self._accepted_dynamic_frames[accepted_idx]

    # ---- Core: per-frame processing ----

    def _prepare_dynamic_frame(self) -> None:
        frame_idx = self.current_dynamic_frame_idx
        self.datamanager.set_dynamic_frame_idx(frame_idx)
        camera, batch = self.datamanager.get_current_dynamic_train_batch()
        frame_name = self.datamanager.get_dynamic_frame_name(frame_idx)
        is_first = self._global_frame_counter == 0

        # All mask/change operations at training resolution for consistency
        bg = self.model._get_background_color()
        live_rgb = self.model.get_live_rgb(batch, background=bg, apply_training_downscale=True)
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), bg)
        gt_depth = self.model._get_gt_depth(batch)
        gripper_mask = self.model._get_batch_mask(batch)

        if is_first:
            init_debug_dir = self.datamanager.get_initialization_debug_dir()
            init_artifact_dir = self.datamanager.get_initialization_artifact_dir()
            self._prepare_frame_0(
                camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
                frame_name, init_debug_dir, init_artifact_dir,
            )
        else:
            debug_dir = self._get_debug_dir()
            self._prepare_frame_n(camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask, frame_name, debug_dir)

        self._global_frame_counter += 1

    def _prepare_frame_0(
        self, camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
        frame_name, debug_dir, artifact_dir,
    ):
        """Bootstrap: ESAM → SAM3D → rendered object mask → CoTracker D0 seed → CD0."""
        t_total = time.time()

        # Check if Phase 0 prefusion already inserted objects
        has_prefused = (self.model.object_instance_ids > 0).any().item()

        # --- TIMING: D0.1 Initial change detection (render RS, MSSIM change mask, ESAM on RS + D0, flag Gaussians) ---
        t0 = time.time()
        stats = self.model.prepare_dynamic_update(
            camera, batch, skip_object_flags_write=has_prefused,
        )
        self._timing["D0.1_initial_change_detection"].append(time.time() - t0)
        # Record substep breakdown for the timing report
        for k, v in stats.get("prepare_dynamic_update_substeps", {}).items():
            self._timing[k].append(v)

        # --- TIMING: D0.1f Post-D0.1 debug image saves (~10 PNGs from prepare_dynamic_update; gated by save_debug_images) ---
        t0 = time.time()
        render_mask_plain_path = debug_dir / f"{frame_name}_render_object_mask_binary.png"
        debug_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_debug_images:
            self._save_image(live_rgb, debug_dir / f"{frame_name}_live_input.png")
            self._save_depth_image(gt_depth, debug_dir / f"{frame_name}_live_depth.png")
            self._save_image(stats["rendered_rgb"], debug_dir / f"{frame_name}_render.png")
            self._save_depth_image(stats["rendered_depth"], debug_dir / f"{frame_name}_render_depth.png")
            self._save_image(stats["render_object_mask"], render_mask_plain_path)
            self._save_image_with_points(
                stats["render_object_mask"],
                stats.get("render_prompt_points"),
                debug_dir / f"{frame_name}_render_object_mask.png",
            )
            self._save_image(stats["live_object_mask"], debug_dir / f"{frame_name}_live_object_mask_binary.png")
            self._save_image_with_points(
                stats["live_object_mask"],
                stats.get("live_prompt_points"),
                debug_dir / f"{frame_name}_live_object_mask.png",
            )
            self._save_image_with_points(
                stats["render_propagation_mask"],
                stats.get("render_prompt_points"),
                debug_dir / f"{frame_name}_render_propagation_mask.png",
            )
            self._save_image_with_points(
                stats["live_propagation_mask"],
                stats.get("live_prompt_points"),
                debug_dir / f"{frame_name}_live_propagation_mask.png",
            )
            if gripper_mask is not None:
                self._save_image_with_points(gripper_mask.float(), None, debug_dir / f"{frame_name}_gripper_mask.png")
        self._timing["D0.1f_post_save"].append(time.time() - t0)

        # --- TIMING: D0.8 Change mask CD0 (MSSIM comparison non-inserted scene render vs D0, excluding gripper) ---
        # Build CD0 from ``non_inserted_rgb`` (scene rendered with all
        # ``inserted_flags == 1`` Gaussians removed) vs the live D0 image
        # with the gripper mask applied and no object-mask exclusion. Any
        # pixel whose live colour differs from this "object-free" render
        # is genuinely something the static scene didn't model — the
        # moved object's NEW position AND any prefused-but-stationary
        # objects' current positions. Computed once here and reused for
        # both Path A's moved-object validation and the final
        # ``_set_optim_mask`` / ``update_scene_opt_active_mask`` writes.
        t0 = time.time()
        cd0 = self._compute_change_mask(
            stats["non_inserted_rgb"], stats["rendered_depth"],
            gt_rgb, gt_depth,
            gripper_mask,
            object_mask=None,
        )
        self._timing["D0.8_change_mask_cd0"].append(time.time() - t0)

        # --- D0.2-D0.3: Path A (prefused) vs Path B (old SAM3D insertion) ---
        if has_prefused and self.config.live:
            # Live mode: the recorded "anchor at lower-image-centre"
            # heuristic doesn't apply — operator approaches the target
            # from arbitrary viewpoints. Pick the prefused instance
            # whose 3D centroid is closest to the camera position.
            t0 = time.time()
            best_id = self._pick_closest_object_to_camera(camera)
            if best_id > 0:
                self.model.object_flags.copy_(
                    (self.model.object_instance_ids == best_id).float()
                )
                self.model._persistent_object_membership_ready = True
                self._d0_selected_instance_id = int(best_id)
                CONSOLE.log(
                    f"[dynamic-gs] Path A (live): selected pre-fused object "
                    f"instance_id={best_id} (closest 3D centroid to camera)"
                )
            else:
                CONSOLE.log("[dynamic-gs] Path A (live): no pre-fused candidates; falling back to ESAM flags")
                self.model.object_flags.copy_(self.model.current_active_mask.float()[:, None])
                self._d0_selected_instance_id = 0
            self._timing["D0.2_sam3d_generation"].append(0.0)
            self._timing["D0.3_sam3d_insertion"].append(time.time() - t0)
            self._sam3d_inserted = True
        elif has_prefused:
            # Path A — robust moved-object pick, two stages:
            #   1. PRIMARY: closest prefused object centroid in 2D to the
            #      anchor point ``(W/2, 0.75 · H)`` (image is upper-left
            #      origin, so ``0.75 H`` is 1/4 of the way up from the
            #      bottom). The gripper-held object is consistently in the
            #      lower-centre of the image in this teleop setup, so
            #      anchoring there gives a 100 % robust pick that doesn't
            #      depend on change-detection signal quality.
            #   2. VALIDATE: confirm the picked candidate's projected
            #      centers lie inside CD0 (built earlier from
            #      ``non_inserted_rgb`` vs the live D0 image). Logs a
            #      warning if not — picks the anchor-closest candidate
            #      either way.
            t0 = time.time()
            centers_2d, radii = extract_projected_centers_and_radii(
                self.model.info, self.model.num_points
            )
            instance_ids = self.model.object_instance_ids.squeeze(-1)
            prefused_mask = instance_ids > 0
            unique_ids = torch.unique(instance_ids[prefused_mask])

            if cd0.ndim == 3:
                cdn_2d = cd0[..., 0]
            else:
                cdn_2d = cd0
            h, w = cdn_2d.shape[:2]

            # --- Anchor-distance pick.
            anchor_x = w * 0.5
            anchor_y = h * 0.75   # 1/4 from the bottom in upper-left origin
            best_id = 0
            best_distance = float("inf")
            best_centroid = (0.0, 0.0)
            best_overlap = 0
            best_total = 0
            best_ratio = 0.0
            score_log = []
            for uid in unique_ids:
                uid_val = uid.item()
                uid_mask = (instance_ids == uid_val) & torch.isfinite(radii) & (radii > 0)
                uid_centers = centers_2d[uid_mask]
                if uid_centers.shape[0] == 0:
                    continue
                cx = uid_centers[:, 0]
                cy = uid_centers[:, 1]
                in_bounds_f = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
                total_in_bounds = int(in_bounds_f.sum().item())
                if total_in_bounds == 0:
                    continue
                centroid_x = float(cx[in_bounds_f].mean().item())
                centroid_y = float(cy[in_bounds_f].mean().item())
                dist = float(((centroid_x - anchor_x) ** 2 + (centroid_y - anchor_y) ** 2) ** 0.5)
                # Validation overlap (purely informational at this stage).
                cx_long = torch.round(cx[in_bounds_f]).long().clamp(0, w - 1)
                cy_long = torch.round(cy[in_bounds_f]).long().clamp(0, h - 1)
                in_mask = cdn_2d[cy_long, cx_long] > 0.5
                overlap = int(in_mask.sum().item())
                ratio = overlap / float(total_in_bounds)
                score_log.append(
                    f"id={uid_val} centroid=({centroid_x:.1f},{centroid_y:.1f}) "
                    f"dist={dist:.1f}px overlap={overlap}/{total_in_bounds} ratio={ratio:.3f}"
                )
                if dist < best_distance:
                    best_distance = dist
                    best_id = uid_val
                    best_centroid = (centroid_x, centroid_y)
                    best_overlap = overlap
                    best_total = total_in_bounds
                    best_ratio = ratio

            # Diagnostic overlay PNG: change mask + each candidate's
            # projected centers + anchor point + winner outline.
            try:
                import matplotlib.pyplot as _plt
                from matplotlib import cm as _cm
                fig, ax = _plt.subplots(figsize=(8, 8))
                ax.imshow(cdn_2d.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                cmap_obj = _cm.get_cmap("tab10")
                for k, uid in enumerate(unique_ids):
                    uid_val = uid.item()
                    uid_mask = (instance_ids == uid_val) & torch.isfinite(radii) & (radii > 0)
                    uid_centers = centers_2d[uid_mask].detach().cpu().numpy()
                    if uid_centers.size == 0:
                        continue
                    col = cmap_obj(k % 10)
                    is_winner = uid_val == best_id
                    ax.scatter(
                        uid_centers[:, 0], uid_centers[:, 1],
                        s=4 if is_winner else 2,
                        c=[col],
                        alpha=0.9 if is_winner else 0.4,
                        edgecolors="red" if is_winner else "none",
                        linewidths=0.6 if is_winner else 0,
                        label=f"id={uid_val}{' (picked)' if is_winner else ''}",
                    )
                ax.scatter([anchor_x], [anchor_y], s=80, marker="x", c="cyan", linewidths=2,
                           label=f"anchor (W/2, 3H/4)")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"D0 Path A selection — picked id={best_id} (dist={best_distance:.1f}px)")
                ax.axis("off")
                fig.savefig(
                    debug_dir / f"{frame_name}_d0_selection_overlay.png",
                    bbox_inches="tight", dpi=120,
                )
                _plt.close(fig)
            except Exception as exc:
                CONSOLE.log(f"[dynamic-gs] Path A: overlay plot skipped ({exc})")

            VALIDATION_MIN_RATIO = 0.10
            if best_id > 0:
                self.model.object_flags.copy_(
                    (self.model.object_instance_ids == best_id).float()
                )
                self.model._persistent_object_membership_ready = True
                self._d0_selected_instance_id = int(best_id)
                if best_ratio < VALIDATION_MIN_RATIO:
                    CONSOLE.log(
                        f"[dynamic-gs] Path A WARNING: anchor-closest pick id={best_id} "
                        f"has only {best_ratio*100:.1f}% of its centers inside the change "
                        f"mask (< {VALIDATION_MIN_RATIO*100:.0f}%); pick may be wrong"
                    )
                CONSOLE.log(
                    f"[dynamic-gs] Path A: selected pre-fused object instance_id={best_id} "
                    f"(centroid=({best_centroid[0]:.1f},{best_centroid[1]:.1f}), "
                    f"dist={best_distance:.1f}px from anchor ({anchor_x:.1f},{anchor_y:.1f}); "
                    f"validation overlap={best_overlap}/{best_total}={best_ratio*100:.1f}%; "
                    f"all candidates: {'; '.join(score_log)})"
                )
            else:
                CONSOLE.log("[dynamic-gs] Path A: no pre-fused candidates; falling back to ESAM flags")
                self.model.object_flags.copy_(self.model.current_active_mask.float()[:, None])
                self._d0_selected_instance_id = 0

            self._timing["D0.2_sam3d_generation"].append(0.0)
            self._timing["D0.3_sam3d_insertion"].append(time.time() - t0)
            self._sam3d_inserted = True  # prevent old path from running

        elif not self._sam3d_inserted and self.model.config.use_sam3d_object_init:
            # Path B's SAM3D subprocess needs render.png + render_mask_plain_path
            # on disk; save them now if D0.1f skipped them.
            if not self.config.save_debug_images:
                self._save_image(stats["rendered_rgb"], debug_dir / f"{frame_name}_render.png")
                self._save_image(stats["render_object_mask"], render_mask_plain_path)
            sam3d_stats = self.model.initialize_object_from_sam3d(
                render_image_path=debug_dir / f"{frame_name}_render.png",
                object_mask_path=render_mask_plain_path,
                render_object_mask=stats["render_object_mask"],
                rendered_depth=stats["rendered_depth"],
                camera=camera, image_debug_dir=debug_dir, artifact_dir=artifact_dir, frame_name=frame_name,
            )
            if sam3d_stats:
                self._timing["D0.2_sam3d_generation"].append(sam3d_stats.get("sam3d_generation_time", 0.0))
                self._timing["D0.3_sam3d_insertion"].append(sam3d_stats.get("sam3d_insertion_time", 0.0))
                # Record D0.3 sub-step breakdown
                for sub_key in (
                    "D0.3a_load_ply",
                    "D0.3b_registration",
                    "D0.3b1_nn_distances",
                    "D0.3b2_voxel_downsample",
                    "D0.3b3_cpd_refinement",
                    "D0.3b4_correspondences",
                    "D0.3b5_dedup",
                    "D0.3b6_plot_and_save",
                    "D0.3c_save_aligned_ply",
                    "D0.3d_insert_gaussians",
                    "D0.3e_persistent_membership",
                    "D0.3f_save_fused_and_log",
                ):
                    if sub_key in sam3d_stats:
                        self._timing[sub_key].append(sam3d_stats[sub_key])
                # CPD metadata is a dict, not a float — store separately
                self._cpd_info = sam3d_stats.get("D0.3b3_cpd_meta", {})
                self._sam3d_inserted = True
                self.model.refresh_dynamic_state_after_insertion(
                    camera, stats["render_object_mask"], stats["optim_mask"],
                )
                CONSOLE.log(
                    f"[dynamic-gs] SAM3D object init -> existing={sam3d_stats['existing_object_gaussians']}, "
                    f"scale={sam3d_stats['chosen_scale']:.4f}, "
                    f"generated={sam3d_stats['sam3d_generated_points']}, kept={sam3d_stats['kept_points_after_dedup']}"
                )
            else:
                self._timing["D0.2_sam3d_generation"].append(0.0)
                self._timing["D0.3_sam3d_insertion"].append(0.0)
        else:
            self._timing["D0.2_sam3d_generation"].append(0.0)
            self._timing["D0.3_sam3d_insertion"].append(0.0)

        # --- TIMING: D0.4 Render object mask (rasterize only object_flags > 0.5 Gaussians, threshold, dilate) ---
        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["D0.4_render_object_mask"].append(time.time() - t0)

        # --- TIMING: D0.6 CoTracker init (sample 2D points inside the rendered object mask, back-project to world via depth, store as reference) ---
        t0 = time.time()
        # Match the per-tick path in ``_apply_motion_estimator``: KLT/XFeat
        # see the gripper-blue-composited image, neural trackers see the raw
        # natural image. The D0 reference must look exactly like the DN
        # input or the descriptor / appearance distribution diverges.
        _d0_tracker_kind = getattr(self.model.config, "dynamic_tracker", "cotracker")
        if _d0_tracker_kind in ("klt", "xfeat"):
            live_rgb_fullres = self._build_tracking_rgb(batch)
        else:
            live_rgb_fullres = self.model.get_live_rgb(batch, apply_training_downscale=False)
        self.model.capture_reference_object_pose()
        # Erode the splat-rasterized mask to its inner 85% so D0 keypoints
        # don't land on the low-opacity halo at the object's edge (those
        # halo pixels show background depth + texture, which would poison
        # the D0 anchor and every match against it). Then AND with the
        # gripper-keep mask so keypoints never fall on the gripper either
        # (gripper occludes the object — a keypoint at a gripper-covered
        # pixel would get gripper-surface depth and gripper-texture
        # descriptors).
        seed_mask = self._erode_mask_to_inner_fraction(
            rendered_obj_mask, _TRACKER_MASK_INNER_FRACTION,
        )
        gripper_keep_mask = self.model._get_batch_mask(batch)
        if seed_mask is not None and gripper_keep_mask is not None:
            gk = gripper_keep_mask
            if gk.ndim == 3 and gk.shape[-1] == 1 and seed_mask.ndim == 2:
                gk = gk[..., 0]
            elif gk.ndim == 2 and seed_mask.ndim == 3 and seed_mask.shape[-1] == 1:
                gk = gk[..., None]
            seed_mask = (seed_mask.bool() & gk.to(seed_mask.device).bool()).to(seed_mask.dtype)
        if not self._has_nonempty_mask(seed_mask):
            seed_mask = None
        instance_id_for_co = getattr(self, "_d0_selected_instance_id", 0)
        _tracker_kind = getattr(self.model.config, "dynamic_tracker", "cotracker")
        _tracker_label = self._TRACKER_LABELS.get(_tracker_kind, "Tracker")
        if instance_id_for_co > 0 and seed_mask is not None:
            self._initialize_motion_estimator(
                live_rgb_fullres, batch["depth_image"], camera, mask=seed_mask,
            )
        else:
            obj_pixels = int(rendered_obj_mask.sum().item()) if rendered_obj_mask is not None else -1
            obj_flag_count = int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())
            CONSOLE.log(
                f"[dynamic-gs] {_tracker_label} init skipped: "
                f"instance_id={instance_id_for_co}, "
                f"rendered_obj_mask_pixels={obj_pixels}, "
                f"object_flags_count={obj_flag_count}"
            )
        self._timing["D0.6_tracker_init"].append(time.time() - t0)

        if self.config.save_debug_images:
            self._save_image_with_points(rendered_obj_mask, None, debug_dir / f"{frame_name}_rendered_object_mask.png")

        # --- TIMING: D0.7 Render RS00 (re-render scene after SAM3D object insertion; debug overlay only) ---
        t0 = time.time()
        if self.config.save_debug_images:
            rs00_outputs = self._render_from_camera(camera)
            rs00_rgb = rs00_outputs["rgb"]
        else:
            rs00_outputs = None
            rs00_rgb = None
        self._timing["D0.7_render_rs00"].append(time.time() - t0)

        # --- TIMING: D0.9 Debug images (save overlay PNGs to disk) ---
        t0 = time.time()
        if self.config.save_debug_images:
            dbg = debug_dir
            self._save_overlay(gt_rgb, cd0, dbg / f"{frame_name}_live_w_cd0.png")
            self._save_overlay(rs00_rgb, cd0, dbg / f"{frame_name}_render_w_cd0.png")
            self._save_overlay(rs00_rgb, rendered_obj_mask, dbg / f"{frame_name}_render_w_objmask.png", color=(0, 0, 1))
            self._save_image(gt_rgb, dbg / f"{frame_name}_live.png")
            self._save_image(rs00_rgb, dbg / f"{frame_name}_rs00.png")
            self._save_image_with_points(cd0, None, dbg / f"{frame_name}_cd0.png")
        self._timing["D0.9_debug_images"].append(time.time() - t0)

        # Refresh the per-Gaussian scene-opt activation mask from CD0 + the
        # latest model.info (set by the RS00 render in D0.7). The means and
        # scene-opt grad hooks read this buffer to gate which Gaussians can
        # receive gradients in the dynamic phase.
        self.model.update_scene_opt_active_mask(cd0)
        self.model._set_optim_mask(cd0)
        self.model._dynamic_ready = True

        self._timing["D0.10_total_frame_0"].append(time.time() - t_total)

        change_px = int((cd0[..., 0] > 0.5).sum().item()) if cd0.ndim >= 3 else int((cd0 > 0.5).sum().item())
        CONSOLE.log(
            f"[dynamic-gs] frame 0 ({frame_name}): bootstrap complete, "
            f"change px={change_px}, object flags={int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())}"
        )
        CONSOLE.log(
            f"[timing] frame 0: total={self._timing['D0.10_total_frame_0'][-1]:.2f}s, "
            f"change_detect={self._timing['D0.1_initial_change_detection'][-1]:.2f}s, "
            f"sam3d_gen={self._timing['D0.2_sam3d_generation'][-1]:.2f}s, "
            f"sam3d_ins={self._timing['D0.3_sam3d_insertion'][-1]:.2f}s, "
            f"obj_mask={self._timing['D0.4_render_object_mask'][-1]:.2f}s, "
            f"tracker_init={self._timing['D0.6_tracker_init'][-1]:.2f}s, "
            f"render_rs00={self._timing['D0.7_render_rs00'][-1]:.2f}s, "
            f"change_mask={self._timing['D0.8_change_mask_cd0'][-1]:.2f}s, "
            f"debug_imgs={self._timing['D0.9_debug_images'][-1]:.2f}s"
        )

    def _prepare_frame_n(self, camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask, frame_name, debug_dir):
        """Frame N>=1: CoTracker pairwise advance → RANSAC-Kabsch (R, t) →
        absolute rigid transform → render → rendered obj mask → CDN.
        """
        t_total = time.time()

        # --- TIMING: DN.3 CoTracker advance (pairwise track of D0-sampled object points; RANSAC-Kabsch for absolute world (R, t)) ---
        if self._sam3d_inserted and self._motion_estimator is not None:
            t0 = time.time()
            self._apply_motion_estimator(camera, batch)
            self._timing["DN.3_tracker_motion"].append(time.time() - t0)

        # --- TIMING: DN.5 Render RDN (render full scene after rigid transform applied to object Gaussians) ---
        t0 = time.time()
        rdn_outputs = self._render_from_camera(camera)
        rdn_rgb = rdn_outputs["rgb"]
        rdn_depth = rdn_outputs["depth"]
        self._timing["DN.5_render_rdn"].append(time.time() - t0)

        # --- TIMING: DN.6 Render object mask (rasterize only object_flags > 0.5 Gaussians from simulation) ---
        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["DN.6_render_object_mask"].append(time.time() - t0)

        combined_obj_mask = rendered_obj_mask

        # --- TIMING: DN.7 Change mask CDN (MSSIM comparison RDN vs DN, excluding gripper + projected object mask) ---
        t0 = time.time()
        cdn = self._compute_change_mask(
            rdn_rgb, rdn_depth, gt_rgb, gt_depth, gripper_mask, combined_obj_mask,
        )
        self._timing["DN.7_change_mask_cdn"].append(time.time() - t0)

        # --- TIMING: DN.8 Debug images (save ~9 overlay PNGs to disk) ---
        t0 = time.time()
        if self.config.save_debug_images:
            dbg = self._get_debug_dir()
            self._save_overlay(gt_rgb, cdn, dbg / f"{frame_name}_live_w_cdn.png")
            self._save_overlay(rdn_rgb, cdn, dbg / f"{frame_name}_render_w_cdn.png")
            self._save_overlay(rdn_rgb, rendered_obj_mask, dbg / f"{frame_name}_render_w_objmask.png", color=(0, 0, 1))
            self._save_image(gt_rgb, dbg / f"{frame_name}_live.png")
            self._save_image(rdn_rgb, dbg / f"{frame_name}_rdn.png")
            self._save_image(cdn, dbg / f"{frame_name}_cdn.png")
        self._timing["DN.8_debug_images"].append(time.time() - t0)

        # Refresh the per-Gaussian scene-opt activation mask from CDN + the
        # latest model.info (set by the RDN render in DN.5). The means and
        # scene-opt grad hooks read this buffer to gate which Gaussians can
        # receive gradients in the dynamic phase.
        self.model.update_scene_opt_active_mask(cdn)
        self.model._set_optim_mask(cdn)
        self.model._dynamic_ready = True

        self._timing["DN.9_total_frame_n"].append(time.time() - t_total)

        change_px = int((cdn[..., 0] > 0.5).sum().item()) if cdn.ndim >= 3 else int((cdn > 0.5).sum().item())
        CONSOLE.log(
            f"[dynamic-gs] frame {self.current_dynamic_frame_idx} ({frame_name}): "
            f"change px={change_px}, "
            f"object flags={int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())}"
        )
        CONSOLE.log(
            f"[timing] frame {self.current_dynamic_frame_idx}: "
            f"total={self._timing['DN.9_total_frame_n'][-1]:.3f}s, "
            f"cotracker={self._timing.get('DN.3_tracker_motion', [0])[-1]:.3f}s, "
            f"render={self._timing['DN.5_render_rdn'][-1]:.3f}s, "
            f"obj_mask={self._timing['DN.6_render_object_mask'][-1]:.3f}s, "
            f"change={self._timing['DN.7_change_mask_cdn'][-1]:.3f}s, "
            f"debug={self._timing['DN.8_debug_images'][-1]:.3f}s"
        )

    # ---- Phase sync and training loop ----

    def _maybe_check_static_convergence(self, step: int) -> None:
        """Cadence gate around ``_compute_static_change_metric``.

        First runs at ``static_convergence_first_check_step`` (aligned
        with full-resolution training kicking in), then every
        ``static_convergence_check_every`` steps. Once the
        per-image change-pixel **ratio** drops below
        ``static_convergence_max_change_ratio`` (default 2 %), we set
        ``self._static_converged_step`` so the next ``_phase_for_step``
        call returns ``"dynamic"`` and the trainer flips phases this step.
        """
        if not self.config.enable_static_convergence_check:
            return
        if self._static_converged_step is not None:
            return
        if step < self.config.static_convergence_first_check_step:
            return
        offset = step - self.config.static_convergence_first_check_step
        if offset % self.config.static_convergence_check_every != 0:
            return

        t0 = time.time()
        avg_ratio, avg_px = self._compute_static_change_metric()
        elapsed = time.time() - t0
        self._timing["S.convergence_check"].append(elapsed)
        threshold_ratio = float(self.config.static_convergence_max_change_ratio)
        CONSOLE.log(
            f"[dynamic-gs] static convergence check @ step {step}: "
            f"avg change ratio = {avg_ratio*100:.2f}% ({avg_px:.0f} px/image; "
            f"threshold {threshold_ratio*100:.2f}%, "
            f"MSSIM_thresh={self.config.static_convergence_rgb_threshold}), "
            f"check took {elapsed:.2f}s"
        )
        if avg_ratio < threshold_ratio:
            self._static_converged_step = step
            CONSOLE.log(
                f"[dynamic-gs] static phase converged at step {step} "
                f"(early exit; configured static_num_steps was {self.config.static_num_steps})"
            )

    def _compute_static_change_metric(self) -> tuple[float, float]:
        """Returns ``(avg_change_ratio, avg_change_px)`` over all static keyframes.

        Renders each keyframe at full resolution (eval mode) and runs
        the MSSIM ``build_change_mask`` recipe with a higher MSSIM
        threshold (``static_convergence_rgb_threshold``) than the
        dynamic phase, so only substantially-different pixels count.

        The dataset's per-image mask (gripper / out-of-frame /
        background) is passed as ``valid_mask`` to the change-mask
        builder so masked-out pixels are excluded from the count, and
        the ratio is computed over the **valid pixel count**, not the
        full HxW — otherwise the metric is diluted by area the model
        was never trained to render.
        """
        ds = self.datamanager.static_manager.train_dataset
        cached = self.datamanager.static_manager.cached_train
        n = len(ds)
        if n == 0:
            return 0.0, 0.0

        device = self.model.device
        bg = self.model._get_background_color()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                total_px = 0
                total_ratio = 0.0
                for i in range(n):
                    camera = ds.cameras[i : i + 1].to(device)
                    if camera.metadata is None:
                        camera.metadata = {}
                    camera.metadata["cam_idx"] = i
                    outputs = self.model.get_outputs(camera)
                    pred_rgb = outputs["rgb"]
                    batch = {k: v for k, v in cached[i].items()}
                    if "image" in batch:
                        batch["image"] = batch["image"].to(device)
                    if "mask" in batch and batch["mask"] is not None:
                        batch["mask"] = batch["mask"].to(device)
                    gt_rgb = self.model.composite_with_background(
                        self.model.get_gt_img(batch["image"]), bg
                    )
                    valid_mask = self.model._get_batch_mask(batch)
                    cdn = build_change_mask(
                        pred_depth=None,
                        gt_depth=None,
                        pred_rgb=pred_rgb,
                        gt_rgb=gt_rgb,
                        valid_mask=valid_mask,
                        rgb_threshold=self.config.static_convergence_rgb_threshold,
                        blur_kernel_size=self.model.config.change_mask_blur_kernel_size,
                        blur_sigma=self.model.config.change_mask_blur_sigma,
                    )
                    if cdn.ndim >= 3:
                        cdn_2d = cdn[..., 0]
                    else:
                        cdn_2d = cdn
                    px_i = int((cdn_2d > 0.5).sum().item())
                    if valid_mask is not None:
                        vm_2d = valid_mask[..., 0] if valid_mask.ndim >= 3 else valid_mask
                        valid_count = int((vm_2d > 0.5).sum().item())
                    else:
                        h, w = cdn_2d.shape[:2]
                        valid_count = h * w
                    total_px = total_px + px_i
                    total_ratio += px_i / float(max(valid_count, 1))
        finally:
            if was_training:
                self.model.train()

        return total_ratio / float(n), total_px / float(n)

    def _sync_phase(self, step: int) -> None:
        """Phase-transition only.

        Frame stepping in the dynamic phase used to live here; with the
        decoupled tracking/optimization design it now lives in
        ``_tracker_tick`` (driven from ``_dynamic_get_train_loss_dict``).
        """
        # Run the convergence check BEFORE computing the phase, so a
        # convergence trigger flips the phase in this same step.
        if self.current_phase == "static":
            self._maybe_check_static_convergence(step)

        phase = self._phase_for_step(step)
        phase_changed = phase != self.current_phase

        if phase_changed:
            self.current_phase = phase
            self.datamanager.set_phase(phase)
            self.model.set_phase(phase, reset_means_optimizer=phase == "dynamic")
            if phase == "dynamic":
                self._reset_dynamic_segmentation_state()
                # Phase 0b: insert pre-generated SAM3D objects now that the
                # static scene is trained (back-side Gaussians never see
                # static photometric optimization).
                if self._sam3d_generation_outputs:
                    self._fuse_sam3d_objects_into_scene(self._sam3d_generation_outputs)
            CONSOLE.log(f"[dynamic-gs] phase -> {phase} at step {step}")

        if phase == "static":
            self.current_dynamic_frame_idx = None
            return

    # ---- Decoupled tracking + optimization ----

    def _tracker_tick(self, frame_idx: int) -> None:
        """One simulated camera frame: FP track always; pool push only if
        the frame is in the keyframe-accepted set AND its CDN clears the
        min-pixel gate.

        FP runs on every frame (even rejected ones) so the object pose
        stays continuous — rejected frames are near-duplicates with tiny
        pose deltas, so the per-tick FP cost is dominated by the small
        refinement.
        """
        self.datamanager.set_dynamic_frame_idx(frame_idx)
        self.current_dynamic_frame_idx = frame_idx
        camera, batch = self.datamanager.get_current_dynamic_train_batch()
        frame_name = self.datamanager.get_dynamic_frame_name(frame_idx)

        bg = self.model._get_background_color()
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), bg)
        gt_depth = self.model._get_gt_depth(batch)
        gripper_mask = self.model._get_batch_mask(batch)

        is_first = self._global_frame_counter == 0
        is_accepted = frame_idx in self._accepted_dynamic_frames_set

        if is_first:
            # D0 bootstrap (SAM3D fusion / FP setup / CD0). Reuses the
            # existing `_prepare_frame_0` end-to-end and pushes the
            # resulting CD0 + camera onto the pool.
            live_rgb = self.model.get_live_rgb(batch, background=bg, apply_training_downscale=True)
            init_debug_dir = self.datamanager.get_initialization_debug_dir()
            init_artifact_dir = self.datamanager.get_initialization_artifact_dir()
            self._prepare_frame_0(
                camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
                frame_name, init_debug_dir, init_artifact_dir,
            )
            cdn = self.model.change_mask_image.detach().clone()
            self._optim_pool.push(OptimFrame(
                frame_idx=frame_idx, camera=camera, cdn=cdn,
            ))
            self._global_frame_counter += 1
            return

        # FP track on every frame (object-pose continuity).
        # Capture the object centroid before / after FP so we can verify
        # in the log that the means actually shift on rejected frames.
        t0 = time.time()
        obj_mask_pre = self.model.object_flags.squeeze(-1) > 0.5
        centroid_before = (
            self.model.means[obj_mask_pre].detach().mean(dim=0)
            if obj_mask_pre.any()
            else None
        )
        if self._sam3d_inserted and self._motion_estimator is not None:
            self._apply_motion_estimator(camera, batch)
        self._timing["DN.3_tracker_motion"].append(time.time() - t0)
        centroid_after = (
            self.model.means[obj_mask_pre].detach().mean(dim=0)
            if obj_mask_pre.any()
            else None
        )
        if centroid_before is not None and centroid_after is not None:
            drift = float(torch.linalg.norm(centroid_after - centroid_before).item())
            CONSOLE.log(
                f"[fp-tick] dyn_step={self._dynamic_step_counter} "
                f"frame={frame_idx} ({frame_name}) "
                f"centroid_drift={drift*1000:.2f}mm "
                f"accepted={is_accepted}"
            )

        # Tracking-only mode: skip RDN render + CDN compute + pool push.
        if self.config.disable_dynamic_optimization:
            self._global_frame_counter += 1
            return

        # Render RDN + object mask + CDN on every tick, regardless of
        # keyframe-filter acceptance, so the rendered scene reflects the
        # latest object pose every time CoTracker moves it. The optim
        # pool push below stays gated by is_accepted AND the CDN pixel
        # threshold — so optimization cadence is unchanged, only the
        # render cadence improves.
        t0 = time.time()
        rdn_outputs = self._render_from_camera(camera)
        self._timing["DN.5_render_rdn"].append(time.time() - t0)
        rdn_rgb = rdn_outputs["rgb"]
        rdn_depth = rdn_outputs["depth"]

        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["DN.6_render_object_mask"].append(time.time() - t0)

        t0 = time.time()
        cdn = self._compute_change_mask(
            rdn_rgb, rdn_depth, gt_rgb, gt_depth, gripper_mask, rendered_obj_mask,
        )
        self._timing["DN.7_change_mask_cdn"].append(time.time() - t0)

        cdn_px = int((cdn[..., 0] > 0.5).sum().item()) if cdn.ndim >= 3 else int((cdn > 0.5).sum().item())

        if not is_accepted:
            CONSOLE.log(
                f"[dynamic-gs] frame {frame_idx} ({frame_name}): "
                f"CoTracker-tracked + rendered, keyframe-filter rejected"
            )
            self._global_frame_counter += 1
            return

        if cdn_px < self.config.optim_pool_min_change_pixels:
            CONSOLE.log(
                f"[dynamic-gs] frame {frame_idx} ({frame_name}): "
                f"change px={cdn_px} < {self.config.optim_pool_min_change_pixels}, skipped"
            )
            self._global_frame_counter += 1
            return

        if self.config.save_debug_images:
            dbg = self._get_debug_dir()
            self._save_overlay(gt_rgb, cdn, dbg / f"{frame_name}_live_w_cdn.png")
            self._save_overlay(rdn_rgb, cdn, dbg / f"{frame_name}_render_w_cdn.png")
            self._save_image(cdn, dbg / f"{frame_name}_cdn.png")

        self._optim_pool.push(OptimFrame(
            frame_idx=frame_idx, camera=camera, cdn=cdn.detach().clone(),
        ))
        CONSOLE.log(
            f"[dynamic-gs] frame {frame_idx} ({frame_name}): change px={cdn_px}, "
            f"pool_size={len(self._optim_pool)}"
        )
        self._global_frame_counter += 1

    def _zero_loss_dummy(self):
        """Trivial loss tuple used when the pool is empty.

        Returned shape matches ``VanillaPipeline.get_train_loss_dict``:
        ``(model_outputs, loss_dict, metrics_dict)``. CPU zero with
        ``requires_grad=True`` so the rare callers that DO go through
        the stock backward path (e.g. ``optim_pool`` empty while
        ``disable_dynamic_optimization=False``) still see a tensor whose
        ``.backward()`` is a valid no-op. In tracking-only mode
        (``disable_dynamic_optimization=True``) the ``NoSaveTrainer``
        short-circuit means ``backward`` is never called on this
        anyway.
        """
        zero = torch.zeros((), device="cpu", requires_grad=True)
        return {}, {"main_loss": zero}, {}

    def _dynamic_get_train_loss_dict(self, step: int):
        """One trainer step in the dynamic phase.

        1. Step-based tracker tick: every ``tracker_tick_every_steps``
           we advance ``_next_frame_to_track`` and run ``_tracker_tick``
           on it. The first dynamic step always ticks (to catch the D0
           bootstrap).
        2. Pool round-robin pick: pick the next pool entry, build the
           per-step effective mask
           ``cdn_at_capture · (1 − render_object_mask(camera_now))``,
           install it, point the datamanager at the entry's frame, and
           delegate to ``super().get_train_loss_dict`` for the loss.
        3. Update pool entry bookkeeping (epoch counter, initial/last
           loss). Evict if epoch budget exhausted OR loss dropped to
           ``optim_pool_loss_relative_threshold`` of initial.
        """
        # Live-mode `stop`: freeze weights, keep stepping no-ops so the
        # viewer remains responsive.
        if self.config.live and self._live_stop_requested:
            self.model.eval()
            return self._zero_loss_dummy()

        # 1. Tracker tick
        cadence = max(1, int(self.config.tracker_tick_every_steps))
        if self.config.live:
            # Live mode always ticks every step: the object pose should
            # update as fast as TAPIR + ROS allow, regardless of whether
            # optimization is on. The dedup-return in _tracker_tick_live
            # handles over-polling cheaply, and when optimization is on
            # the pool optim step still runs after the tick on the same
            # step. `tracker_tick_every_steps` only applies to recorded
            # mode (the elif branch below).
            self._tracker_tick_live()
        elif (
            self._dynamic_step_counter % cadence == 0
            and self._next_frame_to_track < self.total_dynamic_frames
        ):
            self._tracker_tick(self._next_frame_to_track)
            self._next_frame_to_track += 1
        self._dynamic_step_counter += 1

        # Tracking-only mode: skip pool pick + loss compute entirely.
        # Tracker ticks above already ran and applied the rigid
        # transform; weights stay frozen.
        if self.config.disable_dynamic_optimization:
            return self._zero_loss_dummy()

        # 2. Pool round-robin
        if len(self._optim_pool) == 0:
            return self._zero_loss_dummy()
        frame = self._optim_pool.pick_round_robin()

        # Per-step effective mask: capture-time CDN minus current object footprint.
        obj_mask_now = self.model.render_object_mask(frame.camera)
        if obj_mask_now.shape != frame.cdn.shape:
            h, w = frame.cdn.shape[:2]
            obj_mask_now = TF.interpolate(
                obj_mask_now.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)
        effective = (frame.cdn * (1.0 - obj_mask_now)).detach()
        self.model._set_optim_mask(effective)

        if frame.live_batch is None:
            # Recorded mode: pin the datamanager to the entry's frame
            # so super().get_train_loss_dict pulls disk-backed tensors.
            self.datamanager.set_dynamic_frame_idx(frame.frame_idx)
            self.current_dynamic_frame_idx = frame.frame_idx
            result = super().get_train_loss_dict(step)
        else:
            # Live mode: skip the datamanager entirely. Build the loss
            # directly against the OptimFrame's stored RGB/depth/mask so
            # the model is supervised by the actual ROS frame at capture
            # time, not whatever stub the dataparser is holding.
            model_outputs = self.model(frame.camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, frame.live_batch)
            loss_dict = self.model.get_loss_dict(model_outputs, frame.live_batch, metrics_dict)
            result = (model_outputs, loss_dict, metrics_dict)

            # Diagnostic dump: render, live GT, effective mask, overlay.
            # Throttled by `save_live_optim_debug_every`.
            self._maybe_save_live_optim_debug(step, frame, model_outputs, effective)

        # Refresh per-Gaussian gate using the just-set self.info from the
        # full-scene render. Backward (called by the trainer after this
        # function returns) reads scene_opt_active_mask via the registered
        # gradient hooks.
        self.model.update_scene_opt_active_mask(effective)

        # 3. Pool bookkeeping + eviction
        try:
            loss_value = float(sum(v.detach() for v in result[1].values()).item())
        except Exception:
            loss_value = 0.0
        if frame.initial_loss is None:
            frame.initial_loss = loss_value
        frame.last_loss = loss_value
        frame.epochs_used += 1

        evict = False
        if frame.epochs_used >= self.config.optim_pool_max_epochs:
            evict = True
        elif (
            frame.initial_loss is not None
            and frame.initial_loss > 0
            and frame.last_loss < frame.initial_loss * self.config.optim_pool_loss_relative_threshold
        ):
            evict = True
        if evict:
            self._optim_pool.evict(frame)
            CONSOLE.log(
                f"[dynamic-gs] evicted frame {frame.frame_idx} "
                f"(epochs={frame.epochs_used}, "
                f"loss={frame.last_loss:.4f}/{frame.initial_loss:.4f}, "
                f"pool_size={len(self._optim_pool)})"
            )

        return result

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes):
        callbacks = super().get_training_callbacks(training_callback_attributes)
        trainer = training_callback_attributes.trainer
        # Stash the trainer so _tracker_tick_live can reach
        # trainer.viewer_state.render_statemachines and force re-renders
        # the moment object means change.
        self._trainer = trainer
        if trainer is not None:
            if self.config.live:
                # Live mode has no a-priori frame budget — the operator
                # ends the session by typing 'stop' on stdin, after
                # which we keep returning zero-loss results so the
                # trainer keeps stepping (viewer stays alive).
                trainer.config.max_num_iterations = 10**9
            else:
                trainer.config.max_num_iterations = self._total_train_steps()
            # Tracking-only mode: each "step" is a no-op for the
            # writer (loss is identically 0). Bump the per-step log
            # cadence way up so the LocalWriter stops painting Train
            # Loss / GPU Memory lines over the [tracker-rate] print.
            if self.config.disable_dynamic_optimization:
                trainer.config.logging.steps_per_log = 10**9
        return callbacks

    def _maybe_start_torch_profiler(self) -> None:
        """Lazy-init torch.profiler.profile on the first dynamic step when enabled.

        Captures one window of dynamic training steps (warmup + active), then
        on_trace_ready fires once and exports a Chrome trace + a key_averages
        table sorted by CUDA time. Subsequent ``step()`` calls become no-ops.
        """
        if self._torch_profile_started or self._torch_profile_done or not self._torch_profile_enabled:
            return

        try:
            data_root = Path(self.datamanager.config.data)
        except AttributeError:
            data_root = Path(".")
        trace_path = data_root / "dynamic_step_profile.json"
        summary_path = data_root / "dynamic_step_profile.txt"

        def _on_trace_ready(prof):
            try:
                prof.export_chrome_trace(str(trace_path))
            except Exception as e:
                CONSOLE.log(f"[profile] export_chrome_trace failed: {e}")
            sort_key = "cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            try:
                table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
            except Exception as e:
                table = f"key_averages().table failed: {e}"
            try:
                summary_path.write_text(table)
            except Exception:
                pass
            CONSOLE.log("[profile] === DYNAMIC STEP TORCH PROFILE (top 40 by CUDA time) ===")
            CONSOLE.log(table)
            CONSOLE.log(f"[profile] Chrome trace: {trace_path}")
            CONSOLE.log(f"[profile] Op-table summary: {summary_path}")
            self._torch_profile_done = True

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        try:
            self._torch_profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=self._torch_profile_warmup_dyn_steps,
                    active=self._torch_profile_active_dyn_steps,
                    repeat=1,
                ),
                on_trace_ready=_on_trace_ready,
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self._torch_profiler.__enter__()
            self._torch_profile_started = True
            CONSOLE.log(
                f"[profile] torch.profiler started: "
                f"{self._torch_profile_warmup_dyn_steps} warmup + "
                f"{self._torch_profile_active_dyn_steps} active dynamic steps"
            )
        except Exception as e:
            CONSOLE.log(f"[profile] torch.profiler init failed: {e}")
            self._torch_profile_enabled = False
            self._torch_profiler = None

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        t0 = time.time()
        self._sync_phase(step)
        if self.current_phase == "dynamic":
            self._maybe_start_torch_profiler()
            result = self._dynamic_get_train_loss_dict(step)
        else:
            result = super().get_train_loss_dict(step)
        if self._torch_profiler is not None and self.current_phase == "dynamic":
            try:
                self._torch_profiler.step()
            except Exception:
                pass
        elapsed = time.time() - t0
        phase_key = "static_step" if self.current_phase == "static" else "dynamic_step"
        self._timing[phase_key].append(elapsed)

        # Print summary at end of static phase
        if self.current_phase == "dynamic" and step == self.config.static_num_steps:
            s = self._timing["static_step"]
            if s:
                CONSOLE.log(
                    f"[timing] === STATIC PHASE SUMMARY ===\n"
                    f"  steps: {len(s)}, total: {sum(s):.1f}s, avg: {sum(s)/len(s)*1000:.1f}ms/step"
                )
        # Write full report at the very last step
        total_steps = self._total_train_steps()
        if step == total_steps - 1:
            self._print_timing_summary()
            self._write_timing_report()

        return result

    def _print_timing_summary(self):
        """Print a concise timing summary to the console log."""
        CONSOLE.log("[timing] === FULL PIPELINE SUMMARY ===")
        for key in sorted(k for k in self._timing if k.startswith("S0.")):
            vals = self._timing[key]
            if vals:
                CONSOLE.log(f"  {key}: {sum(vals):.2f}s")
        s = self._timing["static_step"]
        if s:
            CONSOLE.log(f"  Static phase: {len(s)} steps, {sum(s):.1f}s total, {sum(s)/len(s)*1000:.1f}ms/step avg")
        for key in sorted(k for k in self._timing if k.startswith("D0.")):
            vals = self._timing[key]
            CONSOLE.log(f"  {key}: {sum(vals):.2f}s")
        for key in sorted(k for k in self._timing if k.startswith("DN.")):
            vals = self._timing[key]
            if vals:
                CONSOLE.log(f"  {key}: avg={sum(vals)/len(vals)*1000:.1f}ms, total={sum(vals):.2f}s ({len(vals)} frames)")
        d = self._timing["dynamic_step"]
        if d:
            CONSOLE.log(f"  Dynamic training: {len(d)} steps, {sum(d):.1f}s total, {sum(d)/len(d)*1000:.1f}ms/step avg")
        all_times = sum(sum(v) for v in self._timing.values())
        CONSOLE.log(f"  Grand total measured: {all_times:.1f}s")

    def _write_timing_report(self):
        """Write a detailed timing report file with per-phase breakdowns and percentages.

        Called both at the last training step and via atexit so that Ctrl+C
        interrupts still produce a report.  Missing timing keys (e.g. because
        the run was interrupted before that phase completed) show 0.0s; the
        last collected value for each key is always used as-is.
        """
        if self._timing_report_written:
            return
        # If the pipeline failed during __init__ before datamanager was set,
        # there is nothing to write.  Access via object.__getattribute__ to
        # avoid nn.Module's AttributeError on missing children.
        try:
            datamanager = object.__getattribute__(self, "datamanager")
        except AttributeError:
            return
        if datamanager is None or not hasattr(datamanager, "config"):
            return
        self._timing_report_written = True

        from datetime import datetime

        tracker_kind = getattr(self.model.config, "dynamic_tracker", "cotracker")
        tracker_label = {
            "tapir": "TAPIR",
            "tapnext": "TAPNext",
            "cotracker": "CoTracker",
            "klt": "KLT",
            "xfeat": "XFeat",
        }.get(tracker_kind, "CoTracker")
        # --- Descriptions for each timer key (chronological within phase) ---
        d0_keys = [
            ("D0.1_initial_change_detection", "Initial change detection (total)"),
            ("D0.1a_forward_render", "  -> Forward render (get_outputs in eval mode)"),
            ("D0.1b_change_mask", "  -> Change mask (MSSIM depth+RGB, morphological filtering)"),
            ("D0.1c_esam_render", "  -> ESAM on render (includes model load on first call)"),
            ("D0.1d_esam_live", "  -> ESAM on live image"),
            ("D0.1e_gaussian_flagging", "  -> Gaussian flagging (project centers, build active mask)"),
            ("D0.1f_post_save", "  -> Post-D0.1 debug image saves (~10 PNGs of inputs/intermediates)"),
            ("D0.2_sam3d_generation", "SAM3D object generation (subprocess)"),
            ("D0.3_sam3d_insertion", "SAM3D object insertion (total)"),
            ("D0.3a_load_ply", "  -> Load SAM3D PLY (plyfile read from disk)"),
            ("D0.3b_registration", "  -> Registration total (bbox scale + centroid + downsample + CPD + dedup)"),
            ("D0.3b1_nn_distances", "    -> Median NN distances for voxel size (Open3D, source + target)"),
            ("D0.3b2_voxel_downsample", "    -> Voxel downsample (source + target)"),
            ("D0.3b3_cpd_refinement", "    -> Probreg CPD similarity refinement (maxiter=50, tol=1e-2)"),
            ("D0.3b4_correspondences", "    -> Explicit correspondence build + point transforms"),
            ("D0.3b5_dedup", "    -> Dedup distance computation (Open3D point cloud distance)"),
            ("D0.3b6_plot_and_save", "    -> Correspondence plot (matplotlib 3D) + artifact PLY saves"),
            ("D0.3c_save_aligned_ply", "  -> Save aligned PLY"),
            ("D0.3d_insert_gaussians", "  -> Insert Gaussians (k_nearest for scale, param concat, optimizer rebuild)"),
            ("D0.3e_persistent_membership", "  -> Persistent membership (sklearn KNN over all candidate Gaussians)"),
            ("D0.3f_save_fused_and_log", "  -> Save fused PLY + write fusion log"),
            ("D0.4_render_object_mask", "Render object mask (rasterize object_flags Gaussians from simulation)"),
            ("D0.6_tracker_init", f"{tracker_label} D0 seed (sample 2D points inside the rendered object mask, back-project to world via depth)"),
            ("D0.7_render_rs00", "Render RS00 (re-render scene after SAM3D insertion)"),
            ("D0.8_change_mask_cd0", "Change mask CD0 (MSSIM RS00 vs D0, excluding gripper + object)"),
            ("D0.9_debug_images", "Debug images (save overlay PNGs to disk)"),
        ]
        dn_keys = [
            ("DN.3_tracker_motion", f"{tracker_label} 2D-track advance + RANSAC-Kabsch (R, t) on the moved-object Gaussians"),
            ("DN.3a_get_live_rgb", "  -> get_live_rgb (uint8->float, optional downscale, GPU upload, background composite)"),
            ("DN.3j_object_mask_render", "  -> render_object_mask + erode (XFeat/KLT only; cached every xfeat_object_mask_cache_ticks for XFeat)"),
            ("DN.3_estimate_total", "  -> estimate_and_advance total (sum of 3b+3c+3d+3e plus minor overhead)"),
            ("DN.3b_estimator_input_prep", "    -> input prep total (rgb/depth/intrinsics/c2w extraction, includes CUDA syncs)"),
            ("DN.3b1_prep_rgb_cpu", f"      -> rgb .cpu() + .item() ({tracker_label}-only; CUDA sync)"),
            ("DN.3b2_prep_depth_cpu", f"      -> depth .cpu().numpy() ({tracker_label}-only; CUDA sync)"),
            ("DN.3b3_prep_intrinsics", f"      -> intrinsics .cpu().numpy() ({tracker_label}-only; CUDA sync)"),
            ("DN.3b4_prep_c2w", f"      -> camera_to_world .cpu().numpy() ({tracker_label}-only; CUDA sync)"),
            ("DN.3c_predictor_forward", f"    -> predictor forward total ({tracker_label} NN inference + .cpu() pull; XFeat reports its sparse-extract leg here)"),
            ("DN.3c_xfeat_extract", "      -> XFeat extract (XFeat-only; sparse detectAndCompute forward + keypoint .cpu() pull)"),
            ("DN.3c1_preprocess_frame", f"      -> preprocess frame to model ({tracker_label}-only; .item() sync + interpolate)"),
            ("DN.3c2_feature_grids", f"      -> feature_grids ({tracker_label}-only; encoder forward)"),
            ("DN.3c3_estimate_traj", f"      -> estimate_trajectories ({tracker_label}-only; transformer forward)"),
            ("DN.3c4_tracks_to_cpu", f"      -> tracks/visibles .cpu().numpy() ({tracker_label}-only; final CUDA sync)"),
            ("DN.3i_lighterglue_match", "    -> LighterGlue match (XFeat-only; LightGlue transformer forward over 64-D descriptors; falls back to MNN time if LighterGlue unavailable)"),
            ("DN.3d_postprocess", "    -> postprocess (image-bound filter, mask filter, bilinear depth sample, world back-projection — CPU/numpy)"),
            ("DN.3e_ransac_kabsch", "    -> RANSAC-Kabsch (numpy loop, scales with ransac_iterations × correspondence count)"),
            ("DN.3h_resample", "    -> KLT keypoint resample (FAST inside eroded object ∩ gripper-keep; KLT-only)"),
            ("DN.3f_debug_io", "  -> debug I/O (motion log + tracked-points overlay PNG)"),
            ("DN.3g_apply_transform", "  -> apply rigid transform to flagged Gaussians (CUDA means+quat update)"),
            ("DN.5_render_rdn", "Render RDN (render scene after rigid transform)"),
            ("DN.6_render_object_mask", "Render object mask (rasterize object_flags Gaussians from simulation)"),
            ("DN.7_change_mask_cdn", "Change mask CDN (MSSIM RDN vs DN, excluding gripper + projected object mask)"),
            ("DN.8_debug_images", "Debug images (save overlay PNGs to disk)"),
        ]
        live_keys = [
            ("LIVE.tick_total", "Whole-tick wall-clock (live mode only). Includes early-return ticks that didn't reach the renders."),
            ("LIVE.between_tick_gap", "**Wall-clock between consecutive ticks** (= time spent OUTSIDE _tracker_tick_live: trainer outer loop, optim step, callbacks, viewer render). Large gap + small tick_total = bottleneck is here, not in the tracker."),
            ("LIVE.frame_dt_seq", "Stamp delta between consecutive PROCESSED ROS frames. Reflects effective camera publishing rate as seen by the consumer."),
            ("LIVE.frame_age", "Wall-clock age of the ROS frame at the moment we picked it up (now - frame.stamp)."),
            ("LIVE.peek_latest", "  -> ROS subscriber peek_latest (atomic read, should be sub-ms)"),
            ("LIVE.wrap_batch", "  -> _wrap_live_tuple_as_batch (3x pageable H2D copies: rgb~5MB + depth~3.6MB + mask~0.9MB + Cameras object)"),
            ("LIVE.gt_setup", "  -> gt_rgb composite + _get_gt_depth + _get_batch_mask (GPU ops on device-resident tensors)"),
            ("LIVE.keyframe_filter", "  -> dynamic keyframe filter accept (pure pose math, sub-ms)"),
        ]

        lines = []
        total_steps = self._total_train_steps()
        completed_steps = len(self._timing.get("static_step", [])) + len(self._timing.get("dynamic_step", []))
        interrupted = completed_steps < total_steps
        header = "=== PIPELINE TIMING REPORT (INTERRUPTED — PARTIAL) ===" if interrupted else "=== PIPELINE TIMING REPORT ==="
        lines.append(header)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if interrupted:
            lines.append(f"WARNING: training stopped early ({completed_steps}/{total_steps} steps). Timings below reflect only what was collected.")
        lines.append(
            f"Config: static_num_steps={self.config.static_num_steps}, "
            f"dynamic_steps_per_frame={self.config.dynamic_steps_per_frame}, "
            f"total_dynamic_frames={self.total_dynamic_frames}"
        )
        lines.append(
            f"Keyframe filter: static kept "
            f"{self.datamanager.static_accepted_frames}/{self.datamanager.static_total_frames}, "
            f"dynamic kept "
            f"{len(self._accepted_dynamic_frames)}/{self.total_dynamic_frames}"
        )
        lines.append("")

        # --- Phase 0: split into 0a (pre-static generation) + 0b (post-static fusion) ---
        # The two halves run minutes apart on the timeline (generation in
        # __init__, fusion at the static→dynamic boundary), so we report
        # them separately. Combined "Phase 0 total" is shown at the bottom
        # of the section for backward-compatible interpretation.
        s0_gen_vals = self._timing.get("S0.4a_generation_total", [])
        s0_fuse_vals = self._timing.get("S0.4b_fusion_total", [])
        s0_gen_total = sum(s0_gen_vals) if s0_gen_vals else 0.0
        s0_fuse_total = sum(s0_fuse_vals) if s0_fuse_vals else 0.0
        s0_phase_total = s0_gen_total + s0_fuse_total
        if s0_phase_total > 0:
            s0_keys: list[tuple[str, str]] = [
                ("S0.1_sam3_segmentation", "SAM3 text-prompted segmentation (subprocess in sam3_dynamic_gs env)"),
                ("S0.2_sam3d_multi_generation", "SAM3D multi-object generation (single subprocess, sequential per-mask)"),
                ("S0.4a_generation_total", "Phase 0a total (generation, runs in __init__ pre-static)"),
            ]
            # Per-object fusion keys (Phase 0b)
            for key in sorted(k for k in self._timing if k.startswith("S0.3_fusion_obj_")):
                obj_idx = key.split("_")[-1]
                s0_keys.append((key, f"Object {obj_idx} fusion (register + insert + propagate)"))
            s0_keys.append(("S0.4b_fusion_total", "Phase 0b total (fusion, runs at static→dynamic boundary)"))

            lines.append("--- PHASE 0: SAM3D OBJECT INITIALIZATION (0a generation + 0b fusion) ---")
            lines.append(
                f"Phase total: {s0_phase_total:.1f}s  "
                f"(0a generation: {s0_gen_total:.1f}s, 0b fusion: {s0_fuse_total:.1f}s)"
            )
            lines.append("")
            for key, desc in s0_keys:
                vals = self._timing.get(key, [])
                t = sum(vals) if vals else 0.0
                pct = (t / s0_phase_total * 100) if s0_phase_total > 0 else 0.0
                lines.append(f"  {key:<42s} {t:>8.2f}s  {pct:>6.1f}%    {desc}")
            lines.append("")

        # --- Phase 1: Static ---
        s = self._timing["static_step"]
        static_total = sum(s) if s else 0.0
        conv = self._timing.get("S.convergence_check", [])
        conv_total = sum(conv) if conv else 0.0
        # ``static_step`` is wall-clock around the whole get_train_loss_dict
        # call, so it already includes the convergence-check spikes that run
        # inside _sync_phase. Subtract for an honest "pure step" average.
        pure_total = max(static_total - conv_total, 0.0)
        pure_count = max(len(s) - len(conv), 1)
        lines.append("--- PHASE 1: STATIC TRAINING ---")
        lines.append(f"Phase total: {static_total:.1f}s  (training: {pure_total:.1f}s, convergence checks: {conv_total:.1f}s)")
        lines.append("")
        if s:
            pure_avg_ms = pure_total / pure_count * 1000
            lines.append(
                f"  S.1  Pure training step (avg over {pure_count} steps, excl. convergence checks)  "
                f"{pure_avg_ms:>10.1f}ms"
            )
        if conv:
            avg_s = conv_total / len(conv)
            lines.append(
                f"  S.2  Static convergence check (avg over {len(conv)} calls)  "
                f"{avg_s*1000:>10.1f}ms total {conv_total:.2f}s"
            )
        lines.append("")

        # --- Phase 2: Dynamic initialization (Frame 0) ---
        d0_total_vals = self._timing.get("D0.10_total_frame_0", [])
        d0_phase_total = sum(d0_total_vals) if d0_total_vals else 0.0
        lines.append("--- PHASE 2: DYNAMIC INITIALIZATION (Frame 0) ---")
        lines.append(f"Phase total: {d0_phase_total:.1f}s")
        lines.append("")
        for key, desc in d0_keys:
            vals = self._timing.get(key, [])
            t = sum(vals) if vals else 0.0
            pct = (t / d0_phase_total * 100) if d0_phase_total > 0 else 0.0
            lines.append(f"  {key:<42s} {t:>8.2f}s  {pct:>6.1f}%    {desc}")
            if key == "D0.3b3_cpd_refinement" and self._cpd_info:
                m = self._cpd_info
                if m.get("stop_reason") == "tol":
                    lines.append(
                        f"  {'':42s}                      "
                        f"CPD converged at iteration {m['iterations']}/{m['maxiter']} "
                        f"(tol={m['tol']:.0e} reached)"
                    )
                elif m.get("stop_reason") == "maxiter":
                    q_delta = m.get("last_q_delta")
                    q_str = f"{q_delta:.3e}" if q_delta is not None else "unavailable"
                    lines.append(
                        f"  {'':42s}                      "
                        f"CPD hit maxiter={m['maxiter']}, last |q_delta|={q_str}"
                    )
        lines.append("")

        # --- Phase 3: Dynamic loop ---
        dn_total_vals = self._timing.get("DN.9_total_frame_n", [])
        frame_prep_total = sum(dn_total_vals) if dn_total_vals else 0.0
        n_frames = len(dn_total_vals)
        d = self._timing["dynamic_step"]
        dyn_train_total = sum(d) if d else 0.0
        dyn_phase_total = frame_prep_total + dyn_train_total

        lines.append(
            f"--- PHASE 3: DYNAMIC LOOP "
            f"(kept {len(self._accepted_dynamic_frames)}/{self.total_dynamic_frames} keyframes) ---"
        )
        lines.append(f"Phase total: {dyn_phase_total:.1f}s")
        lines.append(f"  Frame prep total: {frame_prep_total:.1f}s")
        lines.append(f"  Training total: {dyn_train_total:.1f}s")
        lines.append("")

        lines.append(f"  [Per-frame prep averages over {n_frames} frames]")
        avg_frame_total = (frame_prep_total / n_frames) if n_frames > 0 else 0.0
        for key, desc in dn_keys:
            vals = self._timing.get(key, [])
            if vals:
                avg_ms = sum(vals) / len(vals) * 1000
                pct = (avg_ms / (avg_frame_total * 1000) * 100) if avg_frame_total > 0 else 0.0
                lines.append(f"  {key:<42s} {avg_ms:>8.1f}ms  {pct:>6.1f}%    {desc}")
            else:
                lines.append(f"  {key:<42s}      N/A     N/A    {desc}")
        lines.append("")

        lines.append(f"  [Per-epoch training average over {len(d)} steps]")
        if d:
            avg_dyn_ms = dyn_train_total / len(d) * 1000
            lines.append(f"  {'DT.1 dynamic_step':<42s} {avg_dyn_ms:>8.1f}ms  {100.0:>6.1f}%    Full training iteration (masked loss + backward + optimizer)")
        lines.append("")

        # --- Phase 3b: Live tick breakdown (live mode only) ---
        live_tick_vals = self._timing.get("LIVE.tick_total", [])
        if live_tick_vals:
            n_ticks = len(live_tick_vals)
            avg_tick_ms = sum(live_tick_vals) / n_ticks * 1000
            tick_hz = (1000.0 / avg_tick_ms) if avg_tick_ms > 0 else 0.0
            # Effective end-to-end rate from inter-tick gap.
            gap_vals = self._timing.get("LIVE.between_tick_gap", [])
            if gap_vals:
                avg_gap_ms = sum(gap_vals) / len(gap_vals) * 1000
                cycle_ms = avg_tick_ms + avg_gap_ms
                effective_hz = (1000.0 / cycle_ms) if cycle_ms > 0 else 0.0
            else:
                effective_hz = tick_hz
            # Dedup ratio: dedup-returns vs. processed ticks.
            dedup_count = len(self._timing.get("LIVE.dedup_returns", []))
            total_calls = n_ticks + dedup_count
            dedup_ratio = (dedup_count / total_calls) if total_calls > 0 else 0.0
            lines.append(
                f"--- LIVE TICK BREAKDOWN ({n_ticks} processed ticks + {dedup_count} dedup-returns; "
                f"in-tick avg {avg_tick_ms:.1f}ms; **effective rate {effective_hz:.1f} Hz** with "
                f"between-tick gap; dedup ratio {dedup_ratio*100:.0f}%) ---"
            )
            lines.append(
                "  [Sub-step averages. LIVE.tick_total includes early-return ticks (filter rejects, "
                "tracking-only short-circuit). LIVE.between_tick_gap is wall-clock spent OUTSIDE this "
                "function (trainer outer loop, optim, viewer). If gap >> tick_total, the bottleneck "
                "is outside the tracker.]"
            )
            for key, desc in live_keys:
                vals = self._timing.get(key, [])
                if vals:
                    avg_ms = sum(vals) / len(vals) * 1000
                    pct = (avg_ms / avg_tick_ms * 100) if avg_tick_ms > 0 else 0.0
                    lines.append(f"  {key:<42s} {avg_ms:>8.1f}ms  {pct:>6.1f}%    {desc}")
                else:
                    lines.append(f"  {key:<42s}      N/A     N/A    {desc}")
            # Cross-reference: how much of tick_total is the tracker
            # motion (which has its own DN.3 breakdown).
            motion_vals = self._timing.get("DN.3_tracker_motion", [])
            if motion_vals:
                avg_motion_ms = sum(motion_vals) / len(motion_vals) * 1000
                pct = (avg_motion_ms / avg_tick_ms * 100) if avg_tick_ms > 0 else 0.0
                lines.append(
                    f"  {'(reference) DN.3_tracker_motion':<42s} {avg_motion_ms:>8.1f}ms  {pct:>6.1f}%    "
                    f"tracker motion total — see DN.3* breakdown above for sub-syncs"
                )
            lines.append("")

        # --- Grand total ---
        grand_total = static_total + d0_phase_total + dyn_phase_total
        lines.append("--- GRAND TOTAL ---")
        if grand_total > 0:
            lines.append(f"  Static phase:           {static_total:>8.1f}s  {static_total/grand_total*100:>6.1f}%")
            lines.append(f"  Dynamic initialization: {d0_phase_total:>8.1f}s  {d0_phase_total/grand_total*100:>6.1f}%")
            lines.append(f"  Dynamic loop:           {dyn_phase_total:>8.1f}s  {dyn_phase_total/grand_total*100:>6.1f}%")
            lines.append(f"    Frame prep subtotal:  {frame_prep_total:>8.1f}s  {frame_prep_total/grand_total*100:>6.1f}%")
            lines.append(f"    Training subtotal:    {dyn_train_total:>8.1f}s  {dyn_train_total/grand_total*100:>6.1f}%")
            lines.append(f"  Pipeline total:         {grand_total:>8.1f}s")
        lines.append("")

        report_text = "\n".join(lines)

        # Write to data root (same level as CLAUDE.md equivalent for the data)
        try:
            data_root = Path(datamanager.config.data)
        except AttributeError:
            return
        report_path = data_root / "timing_report.txt"
        report_path.write_text(report_text)
        CONSOLE.log(f"[timing] Report written to {report_path}")

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self._sync_phase(step)
        return super().get_eval_loss_dict(step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self._sync_phase(step)
        return super().get_eval_image_metrics_and_images(step)

    @profiler.time_function
    def get_average_eval_image_metrics(self, step=None, output_path=None, get_std=False):
        if step is not None:
            self._sync_phase(step)
        return super().get_average_eval_image_metrics(step=step, output_path=output_path, get_std=get_std)
