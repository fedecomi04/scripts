# Module spec: `config.py` (layer: core)

## 1. Responsibility

One sentence: own every tunable knob of the dynamic-gs runtime as a single typed, frozen, env-overridable config object (resolution, prompts, FF/CDN cadences, scale clamps, depth caps, sim-noise, lock/thread params, ceilings) plus the four nerfstudio `MethodSpecification` entry points — so no other module reads `os.environ` or hardcodes a number.

This is a **pure-data + entry-point** module: no torch, no threads, no I/O beyond reading `os.environ` once at construction. It is the contract every other module imports its numbers from.

---

## 2. Public interface (the contract other modules call)

### 2a. The typed config trees

```python
@dataclass(frozen=True)
class TrackerConfig:
    """XFeat tracker knobs (consumed by tracker core, main thread)."""
    top_k: int = 1024                          # >=512; 300-era shake floor is 3x below
    ransac_iterations: int = 32
    lighterglue_depth_confidence: float = -1.0 # disabled; early-exit causes match-set shake
    crop_to_object_bbox: bool = True
    crop_padding_px: int = 150
    object_mask_filter: bool = True            # post-match footprint filter (Inv: stops bg-pin)
    rotation_gate_deg: float = 22.5            # relative cam<->object anchor gate
    scale_gate_ratio: float = 1.3
    scale_select: bool = False                 # DGS_XFEAT_SCALE_SELECT
    static_hold_window: int = 15               # DGS_HOLD_WINDOW
    static_hold_trans_mm: float = 18.0         # DGS_HOLD_TRANS_MM
    static_hold_rot_deg: float = 6.0           # DGS_HOLD_ROT_DEG
    min_track_points: int = 12

@dataclass(frozen=True)
class PoseFilterConfig:
    """SE(3) constant-velocity KF — DISABLED by default (lagged on jerky 1200p)."""
    enabled: bool = False                      # xfeat_pose_filter_enabled
    fixed_fps: float = 9.0                      # rate-invariant dt feed (NOT time.time())
    accel_sigma: float = 0.02                   # DGS_KF_ACCEL_SIGMA
    alpha_sigma: float = 0.1                    # DGS_KF_ALPHA_SIGMA
    meas_trans_mm: float = 20.0                 # DGS_KF_MEAS_TRANS_MM
    meas_rot_deg: float = 10.0                  # DGS_KF_MEAS_ROT_DEG
    snap_trans_m: float = 0.05                  # DGS_KF_SNAP_TRANS_M
    snap_rot_deg: float = 10.0                  # DGS_KF_SNAP_ROT_DEG

@dataclass(frozen=True)
class ChangeMaskConfig:
    """CDN / change-detection knobs (consumed by change-detection, FF-bg + main)."""
    downsample_target_side: int = 150           # DGS_CDN_TARGET_SIDE; single scalar, keeps aspect
    rgb_threshold: float = 0.07
    msssim_pyramid_weights: tuple[float, float, float] = (0.15, 0.30, 0.55)  # coarse-heavy
    block_valid_min_frac: float = 0.5           # fractional block validity (1.0 = old strict)
    min_component_area: int = 76                # at pooled grid res
    keep_largest_only: bool = False             # keep ALL change components
    scene_coverage_threshold: float = 0.5
    live_depth_min_m: float = 0.05              # gate holes vs void on live sensor
    live_depth_max_m: float = 3.0

@dataclass(frozen=True)
class FeedforwardConfig:
    """AnySplat feedforward decode + insert hygiene (consumed by FF dispatcher, bg thread)."""
    cadence_ticks: int = 10                      # FF fires every N ticks
    icp_refine: bool = True                      # DGS_FF_ICP; ON (seam reduction)
    icp_voxel_m: float = 0.0                      # DGS_FF_ICP_VOXEL_M (0 = no downsample)
    scale_multiplier: float = 2.0
    max_scale_m: float = 0.05                    # DGS_FF_MAX_SCALE_M; uniform-shrink oversized
    min_scale_m: float = 0.0                     # DGS_FF_MIN_SCALE_M; cull-tiny off
    voxel_dedup_m: float = 0.0                   # OFF (either >0 enables)
    voxel_dedup_far_m: float = 0.0
    object_mask_scale: float = 1.02              # enlarge subtracted footprint about centroid
    object_mask_dilate_px: int = 0
    cull_in_front_depth_tol_m: float = 0.0
    crop_pad_px: int = 50
    insert_id: int = 999                         # inserted_flags marker (Inv #8)

@dataclass(frozen=True)
class GaussianBudgetConfig:
    """Bounded FF growth + static hygiene (Architecture principle #3)."""
    live_gaussian_ceiling: int = 2_000_000       # hard cap; shed when exceeded
    dynamic_purge_every_n_ff: int = 0            # 0 = off; periodic FF-insert purge (TODO-wired)
    dynamic_purge_opacity_below: float = 0.05
    static_opacity_purge_threshold: float = 0.05 # one-shot end-of-static (0 disables)
    static_scale_clamp_max_m: float = 0.05       # trigger
    static_scale_reset_value_m: float = 0.01     # reset target (below trigger)
    static_scale_clamp_every_n: int = 10

@dataclass(frozen=True)
class DepthConfig:
    """Depth filtering + caps shared seed/tracker/FF (consumed by depth + fusion)."""
    filter_enabled: bool = True                  # DGS_DEPTH_FILTER
    median_ksize: int = 5                        # DGS_DEPTH_MEDIAN_KSIZE
    bilateral_sigma_color_m: float = 0.01        # DGS_DEPTH_BILATERAL_SIGMA_COLOR_M
    bilateral_sigma_space: float = 0.0           # DGS_DEPTH_BILATERAL_SIGMA_SPACE
    bilateral_d: int = 0                         # DGS_DEPTH_BILATERAL_D
    depth_min_m: float = 0.05                    # hardcoded floor
    depth_max_m: float = 2.0                     # DGS_TSDF_DEPTH_MAX_M; seed + train mask cap
    scene_depth_max_m: float = 2.0              # MUST equal depth_max_m (loss-mask)

@dataclass(frozen=True)
class FusionConfig:
    """TSDF seed build (consumed by fusion, capture-time / subprocess)."""
    device: str = "auto"                         # DGS_FUSION_DEVICE: auto|cpu (auto->GPU if avail)
    tsdf_voxel_m: float = 0.002                  # DGS_TSDF_VOXEL_M
    far_voxel_m: float = 0.01
    near_radius_m: float = 1.0
    tsdf_trunc_m: float = 0.008                  # GPU trunc_voxel_multiplier = trunc/voxel
    defer_tsdf: bool = True                       # DGS_LIVE_DEFER_TSDF (live: subprocess seed)

@dataclass(frozen=True)
class SimNoiseConfig:
    """Sim-to-real ZED-X depth noise injection (consumed by live publisher)."""
    enabled: bool = True                         # DGS_SIM_ZED_NOISE; ON by default
    sigma0_m: float = 0.00147                    # DGS_SIM_ZED_SIGMA0_M (p90 upper-bound fit)
    k_m: float = 0.000500                        # DGS_SIM_ZED_K_M (sigma_z = sigma0 + k*z^2)
    hole_rate: float = 0.01                      # DGS_SIM_ZED_HOLE_RATE
    z_min_m: float = 0.05                        # DGS_SIM_ZED_Z_MIN
    z_max_m: float = 3.0                         # DGS_SIM_ZED_Z_MAX

@dataclass(frozen=True)
class SegmentationConfig:
    """SAM3/FastSAM + SAM3D + NDP (consumed by phase0 / static pipeline)."""
    backend: str = "fastsam"                     # DGS_SEGMENTATION_BACKEND: fastsam|sam3
    prompt_text: str = ""                        # DGS_SAM3_PROMPT
    sam3d_no_trim: bool = False                  # DGS_SAM3D_NO_TRIM (fp16 gaussian trim)
    sam3d_registration_backend: str = "ndp"      # ndp|cpd|teaser
    near_surface_max_object_slope_deg: float = 70.0
    near_surface_window_frac: float = 0.012

@dataclass(frozen=True)
class ViserConfig:
    """viser-direct live viz (Invariant #9 — NEVER NS viewer)."""
    enabled: bool = True                         # enable_viser_direct
    port: int = 8081                             # viser_direct_port

@dataclass(frozen=True)
class StaticTrainConfig:
    """Static-phase training schedule (Invariants #1, #2)."""
    num_steps: int = 500                         # STATIC_NUM_STEPS
    early_stop_loss: float = 0.02                # DGS_STATIC_EARLY_STOP_LOSS
    early_stop_patience: int = ...               # DGS_STATIC_EARLY_STOP_PATIENCE
    early_stop_min_steps: int = ...              # DGS_STATIC_EARLY_STOP_MIN_STEPS
    early_stop_enabled: bool = True              # DGS_STATIC_EARLY_STOP

@dataclass(frozen=True)
class DebugConfig:
    """Opt-in tracing — MUST stay off the hot path (Principle #4)."""
    ff_debug_images: bool = False                # DGS_FF_DEBUG (140ms/tick I/O when on)
    fusion_debug: bool = False                   # DGS_FUSION_DEBUG
    track_traj_log: bool = False                 # DGS_TRACK_TRAJ_LOG
    torch_profile: bool = False                  # DGS_TORCH_PROFILE

@dataclass(frozen=True)
class RuntimeConfig:
    """The god-config: one frozen tree the pipeline reads everything from."""
    resolution: tuple[int, int] = (1920, 1200)   # (W, H) live default; 800x800 / 960x600 valid
    background_color: tuple[float, float, float] = (0.86, 0.92, 1.0)  # Inv #6 (DO NOT CHANGE)
    tracker: TrackerConfig = ...
    pose_filter: PoseFilterConfig = ...
    change_mask: ChangeMaskConfig = ...
    feedforward: FeedforwardConfig = ...
    budget: GaussianBudgetConfig = ...
    depth: DepthConfig = ...
    fusion: FusionConfig = ...
    sim_noise: SimNoiseConfig = ...
    segmentation: SegmentationConfig = ...
    viser: ViserConfig = ...
    static_train: StaticTrainConfig = ...
    debug: DebugConfig = ...
```

### 2b. Functions

```python
def load_runtime_config() -> RuntimeConfig:
    """Build the frozen RuntimeConfig: dataclass defaults overlaid with os.environ
    DGS_* overrides, parsed+validated ONCE. The only place os.environ is read.
    Validates cross-field invariants (depth.scene_depth_max_m == depth.depth_max_m;
    static_scale_reset < clamp_max; resets below trigger) and raises ValueError on violation."""

def config_fingerprint(cfg: RuntimeConfig) -> str:
    """Stable hash of the config tree, stamped into post_fusion_state.pt so a warm-cache
    load fails loudly on config drift (Architecture principle #8)."""
```

### 2c. nerfstudio entry points (unchanged contract — pyproject `method_configs`)

```python
StaticGS:       MethodSpecification   # "static-gs"        — train static + Phase 0a/0b, write .pt
StaticGSPreseg: MethodSpecification   # "static-gs-preseg" — per-Gaussian SAM IDs, fuse-before-train
DynamicGS:      MethodSpecification   # "dynamic-gs"       — recorded tracker+FF runtime
DynamicGSLive:  MethodSpecification   # "dynamic-gs-live"  — live SHM tracker+FF runtime
```

Module-level constants kept (referenced by entry points / CLAUDE.md): `STATIC_NUM_STEPS`,
`DEFAULT_DYNAMIC_RECORDED_STEPS`, `DEFAULT_DYNAMIC_LIVE_STEPS`, `_ZERO_LR_OPTIMIZERS`,
and a new shared `_STATIC_OPTIMIZERS` (dedup of the two hand-copied static optimizer blocks).

---

## 3. Depends on (NEW modules only)

**None at import time.** This is the leaf of the dependency graph — `config.py` imports only
stdlib (`os`, `dataclasses`, `hashlib`) + nerfstudio plumbing (`TrainerConfig`,
`MethodSpecification`, `AdamOptimizerConfig`, `CameraOptimizerConfig`, `ViewerConfig`) for the
four MethodSpecs. The MethodSpecs reference the NEW `pipeline.py` (the god-file orchestrator),
the model config, and the datamanager config by `_target` — but those are construction-time
references, not config logic dependencies. Every other module **depends on `config.py`**, never
the reverse (Principle #9 single-source-of-truth, narrow seam).

---

## 4. Consumes / produces

**Consumes:** `os.environ` (the `DGS_*` table above) — read exactly once in
`load_runtime_config()`. nerfstudio CLI args still flow through the `MethodSpecification`/
`TrainerConfig` surface for `--data` etc.

**Produces:**
- `RuntimeConfig` — the frozen typed tree handed to `pipeline.py` at construction; every other
  module receives the relevant sub-config (e.g. tracker gets `cfg.tracker`) — never reaches
  back into `os.environ`.
- A `config_fingerprint(cfg)` string written into the `.pt` warm-cache by `persistence`.
- Four `MethodSpecification` objects discovered by nerfstudio's entry-point loader.

---

## 5. Source moved in (current symbol -> what it becomes)

| Current | Becomes |
|---|---|
| `dynamic_gs_config.py` `STATIC_NUM_STEPS`, `DEFAULT_DYNAMIC_RECORDED_STEPS`, `DEFAULT_DYNAMIC_LIVE_STEPS` | module constants, unchanged |
| `dynamic_gs_config.py` `_ZERO_LR_OPTIMIZERS` | unchanged (Invariant #4 enforcer) |
| `dynamic_gs_config.py` hand-duplicated static optimizer dicts (`StaticGS` + `StaticGSPreseg`) | one shared `_STATIC_OPTIMIZERS` (smell #7) |
| `dynamic_gs_config.py` four `MethodSpecification`s | unchanged surface; `_target` repointed to the new `pipeline.py` classes |
| `DynamicGSModelConfig.xfeat_*` fields (model) | `TrackerConfig` |
| `DynamicGSModelConfig.xfeat_pose_filter_*` + `DGS_KF_*`/`DGS_HOLD_*` env reads scattered in `xfeat_motion.py`/`tracker_common.py` | `PoseFilterConfig` + `TrackerConfig` (the env reads centralize here) |
| `ChangeMaskConfig` (change_mask.py) defaults + `change_mask_*` (model) + `DGS_CDN_TARGET_SIDE`/`DGS_SPIKE_GATE_FRAC` | `ChangeMaskConfig` (this module) |
| `DynamicGSPipelineBaseConfig.feedforward_*` fields + `DGS_FF_ICP*`/`DGS_FF_*_SCALE_M` env reads | `FeedforwardConfig` |
| `static_phase_opacity_purge_threshold`, `scale_clamp_*`, `scale_reset_*` (static model/pipeline) | `GaussianBudgetConfig` |
| `online_fusion.py` `TSDF_VOXEL_M`/`FAR_VOXEL_M`/`NEAR_RADIUS_M`/`DEPTH_MIN_M`/`DEPTH_MAX_M`/`TSDF_TRUNC_M` module consts + `DGS_TSDF_*`/`DGS_FUSION_DEVICE`/`DGS_LIVE_DEFER_TSDF` | `FusionConfig` + `DepthConfig` |
| `depth_filter.py` `DGS_DEPTH_*` env reads | `DepthConfig` |
| `zed_depth_noise.py` `DGS_SIM_ZED_*` env reads + `enabled()` | `SimNoiseConfig` |
| `segmentation_backend`, `sam3_prompt_text`, `sam3d_registration_backend`, near-surface knobs | `SegmentationConfig` |
| `enable_viser_direct`, `viser_direct_port` (pipeline base) | `ViserConfig` |
| `STATIC_EARLY_STOP_LOSS` + `DGS_STATIC_EARLY_STOP*` (static_gs_pipeline) | `StaticTrainConfig` |
| `DGS_FF_DEBUG`/`save_debug_images`, `DGS_FUSION_DEBUG`, `DGS_TRACK_TRAJ_LOG`, `DGS_TORCH_PROFILE` reads | `DebugConfig` |

---

## 6. Dropped (NOT carried — with reason + audit ref)

| Dropped | Reason | Audit ref |
|---|---|---|
| `feedforward_video_out` (field) | Zero reads; no mp4 writer ever implemented | 00_DEAD_CODE.md "feedforward_video_out … Config field, zero reads; CLAUDE.md confirms no video writer" |
| `feedforward_video_fps` (field) | Zero reads; dies with the video machinery | 00_DEAD_CODE.md "feedforward_video_fps … zero reads" |
| `feedforward_anchor_frame` (field) | Zero reads | 00_DEAD_CODE.md "feedforward_anchor_frame … Config field, zero reads" |
| `_oneshot_ff_due` predicate / oneshot-FF path | Inlined; the cadence model is recurring-only. Spec carries only `cadence_ticks` | 00_DEAD_CODE.md "_oneshot_ff_due … def only; get_train_loss_dict inlines"; RUNTIME doc keeps only `is_first` gate + cadence |
| `change_mask_*` fields on **StaticGSModelConfig** (6 fields, no `change_mask_mode`) | Partial dead copy; zero `config.change_mask_*` reads on the static model | 00_DEAD_CODE.md UNCERTAIN table rows `change_mask_*` (StaticGSModelConfig) |
| `mode='depth'` / `'depth_outlier'` CDN knobs (`_depth_diff_score`/`_depth_outlier_score` config branch) | Live always `mode='rgb'`; never-configured branch. Config exposes only the rgb path | 00_DEAD_CODE.md UNCERTAIN `_depth_diff_score`/`_depth_outlier_score` |
| `live_render_kick_every_n_ticks` (inert N=1 experiment field) | Inert; the train_lock-disable experiment was reverted | CLAUDE.md "train_lock disable was a mistake"; field is inert N=1 |
| `reuse_sam3d_generated_ply=True` forced-truthy hack | Config value forced to dodge an over-eager validator; the new model config should not require it | datamanager_config.md smell #6 |
| `interactive_object_selection` / `object_selection_timeout_s` / `d0_force_instance_id` picker knobs | Live D0 picker is roadmap/recorded-debug, not the live tracker+FF hot path; keep out of the runtime god-config until the multi-object roadmap lands | RUNTIME doc thread model (D0 = single selection); CLAUDE roadmap items #2-4 still future |
| `static_keyframe_translation_m` / `static_keyframe_rotation_deg` on the **DataManager** config | Confusing 2nd dedup mechanism colliding with capture-time `keyframe_filter.py`; capture-side filter is the live one | datamanager_config.md smell #5 |
| Per-stage `filter_depth_torch(median=, bilateral=)` split as a *config* knob | All prod callers run the full filter; no production behavior | 00_DEAD_CODE.md LIVE row "median=/bilateral= kwargs … unused" |

Note: dropping a *field* here does not delete an Invariant-protected buffer or LR — those
(`_ZERO_LR_OPTIMIZERS`, means LR=0, camera_opt off, background color, the 4 identity buffers)
are explicitly **kept**.

---

## 7. Invariants preserved (which CLAUDE.md invariants + how)

- **#1 (static means LR = 0):** `_STATIC_OPTIMIZERS["means"]` LR stays `0.0`. The config never
  exposes a static means-LR knob, so it cannot be turned on by env override.
- **#2 (static camera_opt = "off"):** `StaticGS`/`StaticGSPreseg` MethodSpecs hardcode
  `CameraOptimizerConfig(mode="off")`; no `SegmentationConfig`/`StaticTrainConfig` field can flip it.
- **#4 (dynamic ALL LRs = 0):** `_ZERO_LR_OPTIMIZERS` carried verbatim; both dynamic MethodSpecs
  use it. No dynamic-phase LR knob is exposed.
- **#5 (`outputs/` suppressed):** all four MethodSpecs keep `vis="tensorboard"`; the config never
  offers `vis="viewer"`.
- **#6 (background color):** `RuntimeConfig.background_color = (0.86, 0.92, 1.0)` is the single
  source; documented DO-NOT-CHANGE. Both models read it from here.
- **#8 (identity buffers):** config carries `feedforward.insert_id = 999` (the `inserted_flags`
  marker) but defines NO knob that writes/clears `sam3d_init_target_flags` or `object_flags` —
  those stay phase-owned in the model.
- **#9 (viser-direct only):** `ViserConfig.enabled=True`, `port=8081`; `vis="tensorboard"` on all
  specs. No `--vis viewer` knob exists in this module.
- **Architecture principle #3 (bounded growth):** `GaussianBudgetConfig.live_gaussian_ceiling`
  makes the FF cap a first-class typed knob instead of "hope."
- **Architecture principle #8 (versioned contract):** `config_fingerprint()` is the stamp that
  makes the `.pt` warm-cache fail loudly on drift.

---

## 8. Threading

- **Constructed once, on the main/trainer thread**, at pipeline `__init__`, before any
  tracker/FF/viser thread spins. `load_runtime_config()` reads `os.environ` exactly once here.
- **`RuntimeConfig` is `frozen=True` and deeply immutable** (only ints/floats/str/tuples — no
  mutable containers). It is therefore **safe to share read-only across all three threads** (main
  tracker, FF-bg, viser-render) with **no lock**. This is the cleanest possible concurrency story:
  immutable shared data needs no synchronization (Principle #1 — "concurrency by design").
- **May not block on anything** — no I/O, no torch, no subprocess. `os.environ` read is the only
  side effect and it is synchronous + instant.
- **Lock discipline:** none required. The config object is never mutated after construction, so it
  is never a torn-read hazard. (Contrast: the gaussian_set SSOT needs `_model_lock`; the config
  does not.) If a runtime re-tune is ever needed (live A/B), it MUST be a *new frozen* `RuntimeConfig`
  swapped atomically by the owning thread — never an in-place field write — but no current knob
  requires runtime mutation.

---

## 9. Open questions for the human

1. **Live A/B env knobs at runtime.** Several `DGS_*` (e.g. `DGS_FF_ICP`, `DGS_HOLD_*`,
   `DGS_FF_MAX_SCALE_M`) are documented as "live A/B, no relaunch." A frozen config read once at
   startup means changing them mid-run requires a relaunch. Is the no-relaunch A/B workflow still
   needed, or has it been superseded? If still needed, do we want a small `reload_overrides()`
   path (re-read env -> new frozen config atomically swapped), or accept relaunch-only?

2. **Resolution as a runtime knob vs dataset-derived.** I modeled `resolution=(1920,1200)` as a
   config default, but the real W/H come from the dataset's `transforms.json` / camera intrinsics
   at load. Should `resolution` be a *validation/assertion target* (raise if dataset != configured)
   rather than a source value? Several derived knobs (CDN `downsample_target_side` scaling, crop
   pads) only make sense relative to actual resolution.

3. **`static-gs-preseg` survival.** It's a parallel third method tied to the roadmap (#1
   per-Gaussian IDs). The live-purge north star is the tracker+FF live path. Keep its
   `MethodSpecification` in this module, or split it to a separate `config_preseg.py` so the core
   config stays lean? (I kept it; flag if you want it out.)

4. **Recorded-vs-live split.** `DynamicGS` (recorded) and `DynamicGSLive` differ only in pipeline
   `_target` + iteration cap. Should the recorded path stay a first-class MethodSpec in the core
   config, or is it now purely a debug/replay tool that could live behind a flag? The audit shows a
   large recorded-only dead branch in the datamanager; if recorded is debug-only, several knobs
   (e.g. `defer_tsdf`, the recorded frame-index plumbing) could be demoted.

5. **`StaticTrainConfig.early_stop_*` exact defaults.** I left `early_stop_patience` /
   `early_stop_min_steps` as `...` — the current values live in `static_gs_pipeline.py` and I did
   not want to fabricate numbers. Please confirm the intended defaults (the env names are
   `DGS_STATIC_EARLY_STOP_PATIENCE` / `DGS_STATIC_EARLY_STOP_MIN_STEPS`).

6. **Sim-noise on the real-camera path.** `SimNoiseConfig.enabled=True` by default corrupts sim
   depth to ZED-realistic. On a REAL ZED capture this must be OFF. Should the config auto-derive
   `enabled` from a `source=sim|real` field rather than rely on the operator setting
   `DGS_SIM_ZED_NOISE=0`? (Avoids double-noising real depth.)
