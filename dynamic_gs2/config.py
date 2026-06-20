"""config.py — the single typed, frozen, env-overridable runtime config.

Per rewrite_spec/config.md + 00_DECISIONS.md. The ONLY place os.environ is read.
Every other module receives a frozen sub-config; none reaches back to os.environ.
Immutable => safe to share read-only across the 3 threads with no lock (Principle #1).

NOTE: the four nerfstudio MethodSpecifications (static-gs / -preseg / dynamic-gs /
-live) are intentionally NOT here yet — they reference pipeline.py (built last).
They get added once the orchestrator exists; the old dynamic_gs/ keeps owning the
entry-points meanwhile, so nothing breaks.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Tuple

# --- module constants (carried verbatim from dynamic_gs_config.py) -----------
STATIC_NUM_STEPS = 500
DEFAULT_DYNAMIC_RECORDED_STEPS = 5000
DEFAULT_DYNAMIC_LIVE_STEPS = 1_000_000


def _envs(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v != "0"


# --- typed config trees (frozen; defaults = the shipped values) --------------
@dataclass(frozen=True)
class TrackerConfig:
    top_k: int = 1024
    ransac_iterations: int = 32
    lighterglue_depth_confidence: float = -1.0
    crop_to_object_bbox: bool = True
    crop_padding_px: int = 150
    object_mask_filter: bool = True
    rotation_gate_deg: float = 22.5
    scale_gate_ratio: float = 1.3
    scale_select: bool = False
    static_hold_window: int = 15
    static_hold_trans_mm: float = 18.0
    static_hold_rot_deg: float = 6.0
    min_track_points: int = 12


@dataclass(frozen=True)
class PoseFilterConfig:
    # VERIFIED tuned params from the old commits (2f5b8d9 -> 7cb4f18), NOT CLAUDE.md's
    # stale 0.02/0.1. Default OFF to match the old ground-truth pipeline so the
    # deterministic A/B is apples-to-apples; the operator enables it at the live
    # (step-4) smoothness eval — the snap-gate handles reacquisition spikes, and
    # fixed_fps feeds event-time dt so it's rate-invariant (the live-detune fix).
    enabled: bool = False            # DGS_KF_ENABLED=1 to turn on
    fixed_fps: float = 9.0           # rate-invariant dt (event-time, never time.time())
    accel_sigma: float = 0.005       # final tuned (stronger smoothing than the 0.02 doc value)
    alpha_sigma: float = 0.025       # final tuned
    meas_trans_mm: float = 20.0
    meas_rot_deg: float = 10.0
    snap_trans_m: float = 0.05       # innovation gate: jumps > this are reacquisitions, snap
    snap_rot_deg: float = 10.0


@dataclass(frozen=True)
class ChangeMaskConfig:
    downsample_target_side: int = 150
    rgb_threshold: float = 0.2        # 1-SSIM cutoff on the per-pixel SSIM dissimilarity (see the score
                                      # heatmap debug dump). Pairs with the small morph_close_px=2 so a
                                      # low threshold no longer bridges scattered detections into a square.
    blur_sigma: float = 0.75           # Gaussian pre-blur on the downsampled SSIM input (0 = none)
    blur_kernel_size: int = 5
    ssim_window: int = 5              # SSIM Gaussian window size (single-scale; no pyramid). Runs on the
                                      # ds-downsampled grid, so a large window halos the detected region
                                      # (window 11 @ ds=10 -> ~50px full-res bleed); 5 keeps it tight.
    block_valid_min_frac: float = 0.0   # OFF: keep every pooled block that has any valid pixel (the
                                        # per-mask erosion already accounts for partial-validity blocks).
    min_component_area: int = 40        # drop change components smaller than this many downsampled-grid
                                        # cells (76 = ~87x87px full-res was too large; 10 too small/noisy).
    morph_close_px: int = 2           # fill holes INSIDE a detected region (downsampled-grid radius). Was
                                      # 10 = a 21x21 kernel that BRIDGED scattered faint detections into a
                                      # filled square; 2 fills genuine 1-2px holes only. (DGS_CDN_CLOSE_PX)
    morph_open_px: int = 1            # remove isolated speckle (downsampled-grid radius). Was 3 = a 7x7
                                      # erode that WIPED thin real change (<7 grid-cells wide, e.g. object
                                      # edges); only survived before because close=10 fattened regions
                                      # first. 1 removes single-px speckle only; min_component_area culls
                                      # the rest. (DGS_CDN_OPEN_PX)
    keep_largest_only: bool = False
    scene_coverage_threshold: float = 0.5
    live_depth_min_m: float = 0.05
    live_depth_max_m: float = 3.0
    gripper_erode_px: int = 6         # enlarge the gripper EXCLUSION in the CDN only (erode the keep-mask
                                      # by this many full-res px so the gripper silhouette never leaks into
                                      # the change region). FF-only: the tracker reads the raw keep mask.
                                      # 0 = off. (DGS_CDN_GRIPPER_ERODE_PX)


@dataclass(frozen=True)
class FeedforwardConfig:
    cadence_ticks: int = 10
    icp_refine: bool = True
    icp_stride: int = 4              # pixel stride for BOTH ICP clouds (source depth + projected scene
                                     # target). Measured 1920x1200: stride 4 -> src 139k/tgt 255k @ ~32ms;
                                     # stride 8 -> src 35k/tgt 107k but SLOWER (~47ms) — the target NN
                                     # search dominates and the target is geometry-bound (barely shrinks
                                     # with stride), so 4 is the sweet spot here. (DGS_FF_ICP_STRIDE)
    opacity_min: float = 0.05        # AnySplat opacity-logit keep floor (reproject)
    scale_multiplier: float = 1.0    # insert scales as predicted, no inflation
    max_scale_m: float = 0.02        # hard cap per insert axis (uniform-shrink gross outliers, e.g. 0.2/0.5m blobs)
    min_scale_m: float = 0.0         # tiny-splat cull floor (0 = off)
    voxel_merge_m: float = 0.001     # downsample inserts by moment-match MERGE per voxel (fuse cluster
                                     # -> 1 gaussian sized to its extent: thins WITHOUT holes). 0 = off.
    density_max_points: int = 80000  # CEILING on the corner-detection kNN input: a large revealed region
                                     # can yield 150k+ inserts and the O(N^2) GPU kNN then spikes to
                                     # seconds, starving the tracker (measured 2s freeze). Density shaping
                                     # is cosmetic flat-surface hole-fill, so subsample to this before the
                                     # kNN — bounds the worst case. 0 = no cap. (DGS_FF_DENSITY_MAX_PTS)
    grow_inplane_factor: float = 2.0  # after merge, grow the 2 in-plane (surface) axes by this, leaving
                                      # the normal axis (fills sub-splat surface holes, no blur). 1=off.
    # The in-plane grow is CORNERNESS-GATED (kNN-PCA): full grow on flat surfaces, tapering to no-grow
    # at corners/edges + their neighbours so the fill doesn't leak past edges. ONLY gates the grow.
    corner_knn_k: int = 80            # neighbours for crease+boundary detection (operator-tuned on the
                                      # screwdriver scene; 50-80 band measured best — wider = smoother,
                                      # full boundary recall, low flat false-positives at real density).
    corner_var_scale: float = 0.10    # CREASE: surface-variation that maps to corner_score=1
    corner_boundary_scale: float = 3.0   # BOUNDARY (silhouette/edge with nothing behind): neighbour-
                                         # centroid offset/spacing that maps to 1. Catches edges the
                                         # crease metric is blind to. Operator-tuned to 3.0: high enough
                                         # that gently-CURVED surfaces (table width) aren't mistaken for
                                         # silhouettes; lower = more edges caught + more false-positives.
    corner_halo_k: int = 50           # after detection, also flag a point as corner if any of its
                                      # corner_halo_k nearest neighbours is a detected corner (ONE
                                      # non-iterative hop). Decoupled from corner_knn_k so halo width
                                      # tunes independently of detection sharpness. 0 = no halo.
    corner_merge_threshold: float = 0.13  # corner_score >= this -> point is a CORNER: passed RAW (NOT
                                         # voxel-merged, NOT grown) so edges keep correct geometry.
                                         # Only flat points (< this) are downsampled + hole-filled.
    object_mask_scale: float = 1.1
    object_mask_dilate_px: int = 0
    cull_before_decode: bool = True   # in-front occlusion cull + CDN reclean before AnySplat decode
    cull_in_front_depth_tol_m: float = 0.01   # in-front occlusion cull margin: delete an eligible
                                     # gaussian only if it sits MORE than this in front of the live surface
                                     # (depths_g < sensor - tol). 0 had no margin, so depth noise at a
                                     # static object's silhouette (e.g. the droid) deleted real geometry;
                                     # 10mm only culls clear floaters well in front of the surface.
    cull_replaced_enabled: bool = True  # MASTER toggle for the replaced-surface cull (the SECOND cull):
                                        # delete the thin slab of old geometry the fresh insert overwrites.
                                        # True caps cumulative growth; False = insert-only (no replaced cull).
    cull_replaced_depth_tol_m: float = 0.0   # cull eligible (non-tracked) gaussians whose MEAN projects
                                              # into the (second/re-CDN) changed region AND lies in a THIN
                                              # slab JUST BEHIND the live surface: sensor <= depth_g <
                                              # sensor + this (0 = OFF, no old-surface removal). The insert
                                              # overwrites — NOT the whole column behind it (that culls deep
                                              # walls -> whole scene vanishes), NOT in front (in-front cull's
                                              # job). This value IS the slab thickness.
    crop_pad_px: int = 50
    insert_id: int = 999


@dataclass(frozen=True)
class GaussianBudgetConfig:
    live_gaussian_ceiling: int = 2_000_000
    dynamic_purge_every_n_ff: int = 0
    dynamic_purge_opacity_below: float = 0.05
    static_opacity_purge_threshold: float = 0.05
    static_scale_clamp_max_m: float = 0.05
    static_scale_reset_value_m: float = 0.01
    static_scale_clamp_every_n: int = 10


@dataclass(frozen=True)
class DepthConfig:
    filter_enabled: bool = True
    median_ksize: int = 5
    bilateral_sigma_color_m: float = 0.01
    bilateral_sigma_space: float = 0.0
    bilateral_d: int = 0
    depth_min_m: float = 0.05
    depth_max_m: float = 2.0
    scene_depth_max_m: float = 2.0   # MUST equal depth_max_m (validated)


@dataclass(frozen=True)
class FusionConfig:
    device: str = "auto"
    tsdf_voxel_m: float = 0.002
    far_voxel_m: float = 0.01
    near_radius_m: float = 1.0
    tsdf_trunc_m: float = 0.008
    defer_tsdf: bool = True


@dataclass(frozen=True)
class SimNoiseConfig:
    enabled: bool = True             # auto-derived from RuntimeConfig.source (sim->on, real->off)
    sigma0_m: float = 0.00147
    k_m: float = 0.000500
    hole_rate: float = 0.01
    z_min_m: float = 0.05
    z_max_m: float = 3.0


@dataclass(frozen=True)
class SegmentationConfig:
    backend: str = "fastsam"
    prompt_text: str = ""
    sam3d_no_trim: bool = False
    sam3d_registration_backend: str = "ndp"
    near_surface_max_object_slope_deg: float = 70.0
    near_surface_window_frac: float = 0.012


@dataclass(frozen=True)
class ViserConfig:
    enabled: bool = True
    port: int = 8081


@dataclass(frozen=True)
class StaticTrainConfig:
    num_steps: int = STATIC_NUM_STEPS
    early_stop_enabled: bool = True
    early_stop_loss: float = 0.02
    early_stop_patience: int = 8
    early_stop_min_steps: int = 100


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False            # DGS_DEBUG master (off-hot-path sink)
    ff_debug_images: bool = False
    fusion_debug: bool = False
    track_traj_log: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    source: str = "sim"              # "sim" | "real"  (#17: sim->noise on, real->off)
    resolution: Tuple[int, int] = (1920, 1200)
    background_color: Tuple[float, float, float] = (0.86, 0.92, 1.0)  # Inv #6 DO NOT CHANGE
    shm_name: str = "dgs_frame_v1"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose_filter: PoseFilterConfig = field(default_factory=PoseFilterConfig)
    change_mask: ChangeMaskConfig = field(default_factory=ChangeMaskConfig)
    feedforward: FeedforwardConfig = field(default_factory=FeedforwardConfig)
    budget: GaussianBudgetConfig = field(default_factory=GaussianBudgetConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    sim_noise: SimNoiseConfig = field(default_factory=SimNoiseConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    viser: ViserConfig = field(default_factory=ViserConfig)
    static_train: StaticTrainConfig = field(default_factory=StaticTrainConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


# --- env-override knobs that change at RUNTIME (live A/B, no relaunch) --------
# Only this whitelist is re-readable via reload_overrides(); everything else is
# frozen at boot. (#5: keep live tuning for the knobs the operator A/Bs.)
_RELOAD_WHITELIST = ("DGS_FF_ICP", "DGS_FF_MAX_SCALE_M", "DGS_FF_MIN_SCALE_M",
                     "DGS_HOLD_WINDOW", "DGS_HOLD_TRANS_MM", "DGS_HOLD_ROT_DEG",
                     "DGS_KF_ACCEL_SIGMA", "DGS_KF_ALPHA_SIGMA")


def load_runtime_config() -> RuntimeConfig:
    """Build the frozen RuntimeConfig: defaults overlaid with DGS_* env, parsed +
    validated ONCE. The only place os.environ is read."""
    source = _envs("DGS_SOURCE", "sim").lower()
    tracker = TrackerConfig(
        top_k=_envi("DGS_XFEAT_TOP_K", 1024),
        scale_select=_envb("DGS_XFEAT_SCALE_SELECT", False),
        static_hold_window=_envi("DGS_HOLD_WINDOW", 15),
        static_hold_trans_mm=_envf("DGS_HOLD_TRANS_MM", 18.0),
        static_hold_rot_deg=_envf("DGS_HOLD_ROT_DEG", 6.0),
    )
    pose_filter = PoseFilterConfig(
        enabled=_envb("DGS_KF_ENABLED", False),
        accel_sigma=_envf("DGS_KF_ACCEL_SIGMA", 0.005),
        alpha_sigma=_envf("DGS_KF_ALPHA_SIGMA", 0.025),
        meas_trans_mm=_envf("DGS_KF_MEAS_TRANS_MM", 20.0),
        meas_rot_deg=_envf("DGS_KF_MEAS_ROT_DEG", 10.0),
    )
    change_mask = ChangeMaskConfig(
        downsample_target_side=_envi("DGS_CDN_TARGET_SIDE", 150),
        rgb_threshold=_envf("DGS_CDN_RGB_THRESHOLD", ChangeMaskConfig.rgb_threshold),
        blur_sigma=_envf("DGS_CDN_BLUR_SIGMA", ChangeMaskConfig.blur_sigma),
        blur_kernel_size=_envi("DGS_CDN_BLUR_KERNEL_SIZE", ChangeMaskConfig.blur_kernel_size),
        ssim_window=_envi("DGS_CDN_SSIM_WINDOW", ChangeMaskConfig.ssim_window),
        gripper_erode_px=_envi("DGS_CDN_GRIPPER_ERODE_PX", ChangeMaskConfig.gripper_erode_px),
        morph_close_px=_envi("DGS_CDN_CLOSE_PX", ChangeMaskConfig.morph_close_px),
        morph_open_px=_envi("DGS_CDN_OPEN_PX", ChangeMaskConfig.morph_open_px),
    )
    feedforward = FeedforwardConfig(
        icp_refine=_envb("DGS_FF_ICP", True),
        icp_stride=_envi("DGS_FF_ICP_STRIDE", FeedforwardConfig.icp_stride),
        max_scale_m=_envf("DGS_FF_MAX_SCALE_M", FeedforwardConfig.max_scale_m),
        min_scale_m=_envf("DGS_FF_MIN_SCALE_M", FeedforwardConfig.min_scale_m),
        voxel_merge_m=_envf("DGS_FF_VOXEL_MERGE_M", FeedforwardConfig.voxel_merge_m),
        density_max_points=_envi("DGS_FF_DENSITY_MAX_PTS", FeedforwardConfig.density_max_points),
        grow_inplane_factor=_envf("DGS_FF_GROW_INPLANE", FeedforwardConfig.grow_inplane_factor),
        corner_knn_k=_envi("DGS_FF_CORNER_KNN_K", FeedforwardConfig.corner_knn_k),
        corner_halo_k=_envi("DGS_FF_CORNER_HALO_K", FeedforwardConfig.corner_halo_k),
        corner_var_scale=_envf("DGS_FF_CORNER_VAR_SCALE", FeedforwardConfig.corner_var_scale),
        corner_boundary_scale=_envf("DGS_FF_CORNER_BOUNDARY_SCALE", FeedforwardConfig.corner_boundary_scale),
        corner_merge_threshold=_envf("DGS_FF_CORNER_MERGE_THR", FeedforwardConfig.corner_merge_threshold),
        cull_replaced_enabled=_envb("DGS_FF_CULL_REPLACED", FeedforwardConfig.cull_replaced_enabled),
        cull_replaced_depth_tol_m=_envf("DGS_FF_CULL_REPLACED_TOL_M", FeedforwardConfig.cull_replaced_depth_tol_m),
    )
    depth_max = _envf("DGS_TSDF_DEPTH_MAX_M", 2.0)
    depth = DepthConfig(
        filter_enabled=_envb("DGS_DEPTH_FILTER", True),
        median_ksize=_envi("DGS_DEPTH_MEDIAN_KSIZE", 5),
        bilateral_sigma_color_m=_envf("DGS_DEPTH_BILATERAL_SIGMA_COLOR_M", 0.01),
        depth_max_m=depth_max,
        scene_depth_max_m=depth_max,                       # kept equal by construction
    )
    fusion = FusionConfig(
        device=_envs("DGS_FUSION_DEVICE", "auto").lower(),
        tsdf_voxel_m=_envf("DGS_TSDF_VOXEL_M", 0.002),
        defer_tsdf=_envb("DGS_LIVE_DEFER_TSDF", True),
    )
    # #17: sim-noise enabled is DERIVED from source; DGS_SIM_ZED_NOISE can still force off.
    noise_on = (source == "sim") and _envb("DGS_SIM_ZED_NOISE", True)
    sim_noise = SimNoiseConfig(
        enabled=noise_on,
        sigma0_m=_envf("DGS_SIM_ZED_SIGMA0_M", 0.00147),
        k_m=_envf("DGS_SIM_ZED_K_M", 0.000500),
        hole_rate=_envf("DGS_SIM_ZED_HOLE_RATE", 0.01),
    )
    segmentation = SegmentationConfig(
        backend=_envs("DGS_SEGMENTATION_BACKEND", "fastsam"),
        prompt_text=_envs("DGS_SAM3_PROMPT", ""),
        sam3d_no_trim=_envb("DGS_SAM3D_NO_TRIM", False),
    )
    static_train = StaticTrainConfig(
        early_stop_enabled=_envb("DGS_STATIC_EARLY_STOP", True),
        early_stop_loss=_envf("DGS_STATIC_EARLY_STOP_LOSS", 0.02),
        early_stop_patience=_envi("DGS_STATIC_EARLY_STOP_PATIENCE", 8),
        early_stop_min_steps=_envi("DGS_STATIC_EARLY_STOP_MIN_STEPS", 100),
    )
    debug = DebugConfig(
        enabled=_envb("DGS_DEBUG", False),
        ff_debug_images=_envb("DGS_FF_DEBUG", False),
        fusion_debug=_envb("DGS_FUSION_DEBUG", False),
        track_traj_log=_envb("DGS_TRACK_TRAJ_LOG", False),
    )
    cfg = RuntimeConfig(source=source, tracker=tracker, pose_filter=pose_filter,
                        change_mask=change_mask, feedforward=feedforward, depth=depth,
                        fusion=fusion, sim_noise=sim_noise, segmentation=segmentation,
                        static_train=static_train, debug=debug)
    _validate(cfg)
    return cfg


def _validate(cfg: RuntimeConfig) -> None:
    if cfg.source not in ("sim", "real"):
        raise ValueError("source must be 'sim' or 'real', got %r" % cfg.source)
    if cfg.depth.scene_depth_max_m != cfg.depth.depth_max_m:
        raise ValueError("depth.scene_depth_max_m must equal depth.depth_max_m (loss-mask invariant)")
    if cfg.budget.static_scale_reset_value_m >= cfg.budget.static_scale_clamp_max_m:
        raise ValueError("static_scale_reset must be BELOW clamp_max (avoid boundary re-trip)")
    if cfg.source == "real" and cfg.sim_noise.enabled:
        raise ValueError("sim_noise must be OFF on a real source (would double-noise real depth)")
    if cfg.background_color != (0.86, 0.92, 1.0):
        raise ValueError("background_color is Invariant #6 — must stay Gazebo sky")


def config_fingerprint(cfg: RuntimeConfig) -> str:
    """Stable hash of the config tree — stamped into the .pt warm-cache so a load
    fails loudly on config drift (Principle #8)."""
    blob = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def reload_overrides(cfg: RuntimeConfig) -> RuntimeConfig:
    """Return a NEW frozen config with ONLY the live-A/B whitelist re-read from env
    (#5). The caller atomically swaps it in; never mutates in place."""
    new = replace(cfg,
                  feedforward=replace(cfg.feedforward,
                                      icp_refine=_envb("DGS_FF_ICP", cfg.feedforward.icp_refine),
                                      max_scale_m=_envf("DGS_FF_MAX_SCALE_M", cfg.feedforward.max_scale_m),
                                      min_scale_m=_envf("DGS_FF_MIN_SCALE_M", cfg.feedforward.min_scale_m)),
                  tracker=replace(cfg.tracker,
                                  static_hold_window=_envi("DGS_HOLD_WINDOW", cfg.tracker.static_hold_window),
                                  static_hold_trans_mm=_envf("DGS_HOLD_TRANS_MM", cfg.tracker.static_hold_trans_mm),
                                  static_hold_rot_deg=_envf("DGS_HOLD_ROT_DEG", cfg.tracker.static_hold_rot_deg)),
                  pose_filter=replace(cfg.pose_filter,
                                      accel_sigma=_envf("DGS_KF_ACCEL_SIGMA", cfg.pose_filter.accel_sigma),
                                      alpha_sigma=_envf("DGS_KF_ALPHA_SIGMA", cfg.pose_filter.alpha_sigma)))
    _validate(new)
    return new
