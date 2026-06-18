# `static_fit.py` — module spec (layer: static)

## (1) RESPONSIBILITY

Run the Splatfacto static fit over the TSDF seed under the hard static invariants (NoRefine/no-densify, means LR=0, camera-opt off) with the depth/scene loss-mask cutoff and mid-training scale-reset, then hand back a trained `GaussianSet` (+ the one-shot opacity purge) for Phase-0b — it is the *training-loop policy* for the static phase, not the model and not the surgery.

---

## (2) PUBLIC INTERFACE

```python
@dataclass
class StaticFitConfig:
    """The static-phase knobs. Carried/produced by config.py; passed in by pipeline.py.
    NO Splatfacto LR/optimizer fields here (those live in config.py's optimizer block,
    invariant #1/#2); NO Phase-0a/0b segmentation/registration fields (those move to the
    static_fusion module's config). This is ONLY the fit-loop + loss-mask + hygiene policy."""
    num_steps: int = 500                       # STATIC_NUM_STEPS budget (full run; early-stop may cut it)
    scene_depth_max_m: float = 2.0             # loss-mask depth cap; KEEP == seed DEPTH_MAX_M. 0 disables.
    scene_depth_min_m: float = 0.05            # loss-mask near cut (was hardcoded 0.05)
    scale_clamp_max_m: float = 0.05            # mid-train shrink TRIGGER (largest world axis). 0 disables.
    scale_reset_value_m: float = 0.01          # mid-train shrink RESET target (< trigger, > 0)
    scale_clamp_every_n: int = 10              # shrink cadence in steps
    opacity_purge_threshold: float = 0.05      # one-shot end-of-fit purge: drop sigmoid(opacity)<thr. 0 disables.
    early_stop_enabled: bool = True            # photometric-loss-EMA early exit
    early_stop_loss: float = 0.02              # EMA threshold
    early_stop_patience: int = 8               # consecutive sub-threshold steps to stop
    early_stop_min_steps: int = 100            # warmup before early-stop can fire


def build_static_loss_mask(batch_mask: torch.Tensor | None,
                           depth_m: torch.Tensor | None,
                           cfg: StaticFitConfig,
                           device: torch.device) -> torch.Tensor | None:
    """AND a depth-keep mask (scene_depth_min_m < depth < scene_depth_max_m) into the gripper
    keep-mask. Pure function (no model state). Returns the combined (H,W,1) mask, or the
    gripper mask unchanged when the cap is disabled / no depth in batch. The static model's
    get_loss_dict calls this; FF/dynamic never do."""


def shrink_oversized_log_scales(log_scales: torch.Tensor, cfg: StaticFitConfig) -> torch.Tensor:
    """Pure tensor op: uniformly divide all 3 axes of any Gaussian whose largest log-axis
    exceeds log(scale_clamp_max_m) so its largest axis becomes scale_reset_value_m (shape
    preserved). Returns the new (N,3) log-scales. (Delegates to gaussian_set.uniform_shrink_log_scales;
    re-exported here so the static callback site reads cohesively.)"""


def low_opacity_purge_indices(opacity_logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Indices where sigmoid(opacity) < threshold — the one-shot end-of-fit purge set.
    (Thin wrapper over gaussian_set.low_opacity_indices.)"""


class StaticFit:
    """Drives the Splatfacto static fit to convergence/budget over a GaussianSet, owning ONLY
    the training-loop hooks (loss-mask, scale-reset callback, early-stop, opacity purge).
    Does NOT segment, register, fuse, or save — those are the static_fusion + warm-cache modules."""

    def __init__(self, cfg: StaticFitConfig, gaussians: "GaussianSet",
                 timing: "TimingLedger") -> None: ...

    def install_loss_mask(self, model) -> None:
        """Wire build_static_loss_mask into the static model's get_loss_dict (or register it as
        the model's mask provider). The depth-cut belongs to the fit policy, not the model."""

    def training_callbacks(self, attrs) -> list:
        """Return the nerfstudio TrainingCallbacks this module owns:
          - AFTER_TRAIN_ITERATION every scale_clamp_every_n: scale-reset via shrink_oversized_log_scales
          - BEFORE_TRAIN_ITERATION: stamp train-start wall (timing)
        (Phase-0b + cache-save callbacks are NOT here — pipeline.py composes them from static_fusion.)"""

    def on_train_iteration(self, step: int, loss_dict: dict) -> bool:
        """Per-step early-stop bookkeeping (loss-EMA). Returns True when the fit should stop
        early (caller sets trainer.stop_training). Pure of side effects on the model."""

    def finalize_opacity_purge(self) -> int:
        """One-shot end-of-fit: cull scene Gaussians with sigmoid(opacity)<threshold via
        gaussians.cull(...). Runs BEFORE Phase-0b inserts any object. Returns count dropped.
        object_flags are all 0 here, so nothing tracked is at risk."""
```

(Whether `StaticFit` is a standalone helper or a `nerfstudio.Trainer` subclass is OQ1. The interface above keeps the loop policy as composable hooks so `pipeline.py` — the one god-file orchestrator — owns the trainer object and just installs these.)

---

## (3) DEPENDS ON (NEW modules only)

- **`gaussian_set.py`** — the SSOT. `StaticFit` reads/writes Gaussian state ONLY through it: `gaussians.cull(low_opacity_purge_indices(...))` for the purge; the scale-reset callback writes back via the locked surgery API (or the model's in-place `scales` param under the same lock — OQ3). Never hand-resizes params/buffers.
- **`config.py`** — owns `StaticFitConfig` and the Splatfacto optimizer block (means LR=0, camera-opt off). `StaticFit` receives the config; it does NOT define the LRs.
- **`scene_model.py`** (the splat-model module) — provides `gauss_params`, `get_loss_dict` hookpoint, `NoRefineStrategy` install, the sim background (#6), the four identity buffers (via `GaussianSet`). `StaticFit` installs the loss-mask provider into it; the model owns render + LR/phase policy.
- **`timing.py`** (timing ledger) — records `static_training`/"Splatfacto"/"train" wall + the AnySplat-overlap fold-in.

It does NOT depend on `static_fusion` (Phase-0a/0b), the warm-cache module, `frame.py`, `shm_channel.py`, the tracker, or any FF/decoder module. `pipeline.py` sequences `StaticFit` → `static_fusion` → warm-cache-save.

---

## (4) CONSUMES / PRODUCES

**Consumes:**
- `StaticFitConfig` (from `config.py`).
- A constructed static `GaussianSet` seeded from `depth_camera_init_points.ply` (allocated by the static model's `populate_modules`; the seed PLY is produced upstream by the fusion seed pipeline — NOT this module).
- Per-batch `batch["mask"]` (gripper keep) + `batch["depth_image"]` (CPU-resident, injected by the datamanager) for the loss mask.
- Per-step `loss_dict["main_loss"]` for early-stop.

**Produces:**
- A trained, opacity-purged `GaussianSet` (means on the seed, colors/scales/quats/opacities converged) — handed to Phase-0b by `pipeline.py`.
- The combined static loss-mask tensor (consumed by the model's `get_loss_dict`).
- A `stop` signal (early-stop) the caller applies to the trainer.
- Timing rows: `static_training`/"Splatfacto"/"train".

Does NOT produce: `post_fusion_state.pt` (warm-cache module), any SAM3D object, any manifest, any viser state.

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `static_gs_model.py:get_loss_dict` (depth-cut mask AND-in) | `build_static_loss_mask` free fn + `StaticFit.install_loss_mask` (logic out of the model, into the fit policy) |
| `static_gs_model.py:_shrink_oversized_scales_cb` | `shrink_oversized_log_scales` (→ `gaussian_set.uniform_shrink_log_scales`) + the scale-reset `TrainingCallback` in `StaticFit.training_callbacks` |
| `static_gs_model.py:get_training_callbacks` (the scale-reset append) | `StaticFit.training_callbacks` (the model keeps only NoRefine/buffer setup) |
| `static_gs_pipeline.py:StaticGSTrainer.train_iteration` (loss-EMA + early-stop) | `StaticFit.on_train_iteration` (returns stop-flag) — no separate Trainer subclass needed if `pipeline.py` drives it (OQ1) |
| `static_gs_pipeline.py` module-level `STATIC_EARLY_STOP_*` consts | `StaticFitConfig.early_stop_*` fields (env-override moves to `config.py`'s loader, ARCHITECTURE_PRINCIPLES §8) |
| `static_gs_pipeline.py:_finalize_static_training` (opacity-purge half only) | `StaticFit.finalize_opacity_purge` via `low_opacity_purge_indices` + `gaussians.cull` |
| `static_gs_pipeline.py:_finalize_static_training` (Phase-0b + save half) | NOT here — moves to `static_fusion` + warm-cache, composed by `pipeline.py` |
| `static_gs_pipeline.py:_stamp_train_start` + the `static_training` ledger row | `StaticFit` BEFORE_TRAIN callback + `timing.py` record |
| `StaticGSModelConfig` fit-relevant fields (`scene_depth_max_m`, `scale_*`, `num_downscales=0`, `sh_degree_interval`, `resolution_schedule`, `output_depth_during_training`) | `StaticFitConfig` (the loss-mask/scale/schedule subset) — the Splatfacto-schedule fields stay model config in `config.py`, the *fit* fields move here |

---

## (6) DROPPED (NOT carried, with reason + audit ref)

- **The 7 `change_mask_*` fields on `StaticGSModelConfig`** (`change_mask_depth_threshold/rgb_threshold/use_rgb/blur_kernel_size/blur_sigma/filter_radius/min_component_size`) — declared "for a future static-convergence early-exit", read **zero** times on a static instance; all grep hits resolve to `DynamicGSModelConfig`, and `change_mask_mode` isn't even declared so it isn't a faithful mirror. The convergence check that would use them was never built; the photometric-EMA early-stop covers the same job. *Audit: static_model_pipeline.md §2 "Dead config fields" + §4 "(Medium) Dead config block".*
- **The `StaticGSTrainer` subclass as a structural layer** — its only real content is the early-stop, which becomes `on_train_iteration`. The docstring's stated reason ("a landing zone for trainer tweaks") is speculative scaffolding; the rewrite's god-file `pipeline.py` owns the trainer directly (OQ1). *Audit: static_model_pipeline.md §1 ("paper-thin subclass"); ARCHITECTURE_PRINCIPLES §9 (narrow seams, no speculative layers).*
- **Phase-0a-in-`__init__` + eager-AnySplat spawn + the whole `_write_timing_report` body + the `static_state.pt` save + Phase-0b call** — none of these are the *static fit*. They move to `static_fusion`, the warm-cache module, the AnySplat lifecycle owner, and `timing.py`, sequenced by `pipeline.py`. Carrying them here would re-make the god-pipeline. *Audit: static_model_pipeline.md §1 (`StaticGSPipeline.__init__`, `_write_timing_report`, `_finalize_static_training`); ARCHITECTURE_PRINCIPLES §9.*
- **`local import math` / `import time as _t` inside the hot callback + per-step loop** — moved to module top. *Audit: static_model_pipeline.md §4 "(Low) Local imports inside hot/looped code"; ARCHITECTURE_PRINCIPLES §4 (no per-step allocation surprises).*
- **Broad swallowed exceptions** around the timing-ledger reset and ledger render (`except Exception: pass`) — replaced by narrow handling / loud failure on config errors. *Audit: static_model_pipeline.md §4 "(Low) Broad swallowed exceptions"; ARCHITECTURE_PRINCIPLES §7 (no bare except).*
- **Stale module docstring claims** ("reuses `DynamicGSModel` as-is", "Phase 0b CPD/TEASER++") — not carried; the rewrite docstring states the real flow (own static model + GaussianSet; NDP default). *Audit: static_model_pipeline.md §4 "(High) Stale/contradictory module docstring".*
- **FF-video / oneshot-FF machinery** — never existed on the static path; explicitly out of scope. *Audit: gaussian_set.md §6; CLAUDE.md (no mp4 writer).*

---

## (7) INVARIANTS PRESERVED (CLAUDE.md) + how

- **#1 (static means LR = 0):** `StaticFit` never sets LRs and never moves means — the lr=0.0 means group lives in `config.py`'s optimizer block; the scale-reset callback touches `scales` only, the purge only deletes. The seed positions stay locked. *(This module must assert it does not register a means LR > 0 — it imports the LRs read-only.)*
- **#2 (camera-opt = "off" in static):** untouched — `camera_optimizer.mode="off"` is set in `config.py`; `StaticFit` adds no camera-opt path.
- **#3 (ICP-refined transforms.json):** out of scope — the seed/poses are produced upstream; `StaticFit` consumes them unmodified.
- **#5 (`outputs/` suppressed):** `StaticFit` writes nothing to `outputs/`; its only output is the in-memory `GaussianSet` + timing rows (under `<data_dir>/`). The nerfstudio write-suppression monkeypatches stay in the framework-init module.
- **#6 (Gazebo-sky background):** the sim background is set in the static model's `populate_modules` (kept there); `StaticFit` does not change it, so the loss is computed against the correct bg.
- **#8 (identity-buffer ownership):** `StaticFit` only ever calls `gaussians.cull(...)` (purge) — which slices all 4 buffers in lockstep inside `GaussianSet`. It writes NO identity buffer directly; `object_flags` stay 0 through the whole static fit (D0 is the dynamic phase's job). *Audit confirms object_flags=0 is the correct post-fusion state.*
- **NoRefine / no-densify:** the `NoRefineStrategy` install + `step_post_backward` no-op stay on the static model; `StaticFit` relies on them (the scale-reset exists precisely because densification's own prune is off) but does not re-implement them.

---

## (8) THREADING

- **Single-threaded, main (trainer) thread only.** The entire static fit runs before any tracker/FF/viser thread exists — there is no live SHM source, no FF-bg, no viser-direct during the static phase. So `StaticFit` has no concurrency hazard of its own.
- **Lock discipline:** even though there's no contention yet, all Gaussian-count/value mutation (the opacity purge `cull`, the scale-reset write) goes through `GaussianSet`'s locked surgery API — NOT hand-resized — so the static path uses the *same* one chokepoint as the dynamic path (ARCHITECTURE_PRINCIPLES §1/§2; this is what kills the static↔dynamic surgery drift the audit found). The scale-reset in-place `scales.sub_` must take the `_model_lock` if it touches the live param (OQ3).
- **May block on:** the GPU (render+backward per step) and the `_model_lock` for the brief purge/scale-reset critical sections — both bounded.
- **May NOT block on:** SHM, any subprocess (SAM3/SAM3D/AnySplat are sequenced by `pipeline.py` *around* the fit, not inside it), disk I/O on the hot loop, the FF slot lock, the publisher pose/joint lock. Debug/timing I/O is off the per-step path (write at finalize, not per iteration). *Audit: ARCHITECTURE_PRINCIPLES §4 (no I/O on the hot path).*
- **Early-stop signal:** `on_train_iteration` returns a stop-flag the caller sets on the trainer; the AFTER_TRAIN composition (purge → Phase-0b → save) still fires because it's a callback chain `pipeline.py` owns, not gated by the early break.

---

## (9) OPEN QUESTIONS

1. **Trainer ownership:** keep a thin `nerfstudio.Trainer` subclass that calls `StaticFit` hooks, or have the god-`pipeline.py` own the vanilla trainer and install `StaticFit`'s callbacks + early-stop check directly? The spec assumes the latter (no `StaticGSTrainer` layer). Confirm — it changes whether `on_train_iteration` is an override or a callback.
2. **Loss-mask wiring:** should `StaticFit` install the depth-cut by *patching the model's `get_loss_dict`* (current behavior, model-side), or by passing a `mask_provider` callable into the model at construction? The latter keeps the mask policy fully in this module (cleaner seam) but needs the model to accept the hook. Which seam do you want?
3. **Scale-reset write path:** the mid-train shrink mutates `gauss_params["scales"]` in place every N steps. Route it through a new `GaussianSet.write_log_scales(mask, ...)` locked method (consistent SSOT), or allow a direct in-place `sub_` on the param under the shared lock (cheaper, but a second mutation path)? Spec leans SSOT method.
4. **Early-stop default at high res:** the `early_stop_loss=0.02` + `num_downscales=0` pairing is tuned to the 1200p scene (the 0.09→0.02 history). Should these defaults be resolution-aware (auto-relax for square/800p), or stay a single tuned value with an env override? The audit/CLAUDE history shows 0.09 *undertrained* at 1200p — don't silently regress that.
5. **Config split boundary:** which of the Splatfacto schedule fields (`num_downscales`, `sh_degree_interval`, `resolution_schedule`, `output_depth_during_training`) are "fit policy" (→ `StaticFitConfig`) vs "model config" (→ `config.py` model block)? They affect the fit but are read by the model's render. Proposed: they stay model config; only the loss-mask + scale + early-stop + purge knobs move here. Confirm the line.
6. **Depth in batch contract:** `build_static_loss_mask` assumes the datamanager injects `batch["depth_image"]` (CPU). If the rewrite's `frame.py`/datamanager changes that key/location/device, this mask silently degrades to gripper-only (no error). Should a missing-depth-when-cap-enabled raise loudly (ARCHITECTURE_PRINCIPLES §8 versioned contract) instead of degrading?
