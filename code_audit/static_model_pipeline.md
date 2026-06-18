# Code Audit — `static_gs_model.py` + `static_gs_pipeline.py`

Repo: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`
Audited files:
- `dynamic_gs/static_gs_model.py`
- `dynamic_gs/static_gs_pipeline.py`

Both methods `static-gs` and `static-gs-preseg` (entry points in `pyproject.toml`
→ `dynamic_gs_config.py:StaticGS` / `StaticGSPreseg`) instantiate
`StaticGSModelConfig` (whose `_target` is `StaticGSModel`), `StaticGSPipelineConfig`
(`static-gs`) / `StaticGSPresegPipelineConfig` (`static-gs-preseg`), and
`StaticGSTrainer`. **NOTE the `static_gs_pipeline.py` module docstring is wrong**
where it says it "reuses `DynamicGSModel` as-is" and "Phase 0b runs CPD/TEASER++":
the method actually uses `StaticGSModel` and the default Phase-0b backend is NDP
(`sam3d_registration_backend="ndp"`). See DESIGN SMELLS.

---

## 1) FUNCTION / CLASS MAP

### `static_gs_model.py`

- **`StaticGSModelConfig` (dataclass, SplatfactoModelConfig)** — `static_gs_model.py:64-190` — config surface for the static model (render bg, depth cap, scale-reset hysteresis, schedule overrides, Phase 0a SAM3/FastSAM, Phase 0a/0b SAM3D/registration, plus an unused change-mask block). Refs: 8 external — `static_gs_preseg_pipeline.py:57`, `dynamic_gs_config.py:14,46,104`, `static_gs_pipeline.py:59,77`, plus "keep in sync" comments in `dynamic_gs_model.py:166,540`. **Live entry-point config.**

- **`StaticGSModel(SplatfactoModel)`** — `static_gs_model.py:193` — stripped Splatfacto subclass: 4 identity buffers, insert/delete machinery, Phase-0b subset queries, sim bg, NoRefineStrategy. Refs: 0 direct external by name (it is reached only via `StaticGSModelConfig._target` at `static_gs_model.py:66`). **Live — `_target` instantiation, not dead.**

  - **`__init__(self, config, metadata=None, **kwargs)`** — `:196` — sets `_optimizers_wrapper=None`, calls super. Override. Called by nerfstudio model construction. (constructor)

  - **`get_loss_dict(self, outputs, batch, metrics_dict=None)`** — `:200` — ANDs a depth-keep `(0.05, scene_depth_max_m]` mask into `batch["mask"]` before delegating to Splatfacto. Override; called by the training loop (`pipeline.get_train_loss_dict`). Refs: nerfstudio engine (not by name in repo).

  - **`get_training_callbacks(self, training_callback_attributes)`** — `:228` — appends the mid-training scale-reset callback to the base callbacks. Override; nerfstudio engine calls it. Refs (by name): 8 (mostly other models' own overrides + nerfstudio). **Live — callback registration.**

  - **`_shrink_oversized_scales_cb(self, step)`** — `:244` — mid-training: uniformly shrink any Gaussian whose largest log-axis > `log(scale_clamp_max_m)` down to `scale_reset_value_m`, shape preserved. Refs: 0 external; **registered as a `TrainingCallback` func at `:235`** (callback entry point — NOT dead).

  - **`populate_modules(self)`** — `:272` — sets sim bg, registers the 4 persistent identity buffers, installs `NoRefineStrategy`. Override; nerfstudio model build calls it.

  - **`_get_background_color(self)`** — `:314` — returns the sim bg tensor when enabled, else super. Override. Refs (by name): 7 (parent + splatfacto + dynamic model). Called by render path.

  - **`step_cb(self, optimizers, step)`** — `:319` — caches the optimizers wrapper onto `self._optimizers_wrapper`, then super. Override; nerfstudio `BEFORE_TRAIN_ITERATION` callback. Refs: 3.

  - **`step_post_backward(self, step)`** — `:323` — no-op override (NoRefineStrategy => no densify/prune; parent would raise). Override; nerfstudio `AFTER_TRAIN_ITERATION`. Refs: 3.

  - **`load_state_dict(self, state_dict, **kwargs)`** — `:333` — resizes the 4 identity buffers to the saved `gauss_params.means` count, fills any missing buffer key with zeros, then super. Override; called by `persistence.load_post_fusion_state` (`post_fusion_cache.py:138`) and torch.

  - **`_resize_dynamic_buffers(self, num_points)`** — `:381` — resize the 4 identity buffers in lockstep with a gauss_params insert/delete, preserving leading entries. Refs: 3 (this file + `dynamic_gs_model.py`). Called by `insert_object_gaussians` (`:495`).

  - **`_refresh_gaussian_optimizers(self, reset_means_optimizer)`** — `:411` — re-point every optimizer's `params[0]` at the new gauss_params Parameter + clear Adam state. Refs: 5. Called by `insert_object_gaussians` (`:501`) + `delete_gaussian_indices` (`:535`). **Note: `reset_means_optimizer` arg is unused** (see DESIGN SMELLS).

  - **`_build_new_gaussian_tensors(self, new_xyz, new_rgb)`** — `:428` — default per-Gaussian attribute tensors for a batch of new points (kNN-spaced scales, opacity 0.1). Refs: 2 (this file + dynamic model). Called by `insert_object_gaussians` (`:487`).

  - **`insert_object_gaussians(self, new_xyz, new_rgb, object_flag=True, instance_id=0)`** — `:465` — concat new Gaussians, write identity flags for the inserted range, refresh optimizers; returns inserted index range. Refs: 5 — Phase 0b (`fusion/phase0.py`). **Live — Phase 0b insertion.**

  - **`delete_gaussian_indices(self, indices)`** — `:507` — prune Gaussians + slice the 4 buffers in lockstep + refresh optimizers; returns deleted count. Refs: 3 — `static_gs_pipeline.py:237` (opacity purge) + dynamic model. **Live.**

  - **`_estimate_spacing(points, max_samples=50000)` (staticmethod)** — `:542` — median per-point mean kNN distance (k=3). Refs: 8 — Phase 0b (`phase0.py:1027,1083,1092`) + dynamic model. **Live.**

  - **`_get_object_mask_slab_indices(self, render_object_mask, rendered_depth, depth_tol_m=0.01)`** — `:559` — Gaussian indices whose projected center is in the 2D mask AND within depth_tol of the rendered front surface. Refs: 4 — Phase 0b (`phase0.py:1011,1014`) + dynamic model. **Live.**

  - **`_get_existing_object_subset(self, render_object_mask, rendered_depth)`** — `:603` — Phase 0b CPD/registration target: frontmost-per-pixel, depth-thinned, ~50% downsampled subset → `(indices, means, colors)`. Refs: 6 — Phase 0b + dynamic model. **Live.** Reads `self.info` (set by Splatfacto rasterization in the prior `get_outputs`) — implicit coupling, see SMELLS.

### `static_gs_pipeline.py`

- **`StaticGSPipelineConfig(VanillaPipelineConfig)`** — `:67-95` — pipeline config (`static_num_steps`, cache subpath, opacity purge threshold). Refs: 4 — `dynamic_gs_config.py`, `static_gs_preseg_pipeline.py` (subclass). **Live entry-point config.**

- **`StaticGSPipeline(VanillaPipeline)`** — `:98` — end-to-end static-only pipeline. Refs: 7 — `dynamic_gs_config.py`, `static_gs_preseg_pipeline.py`. **Live.**

  - **`__init__(...)`** — `:103` — timing ledger reset, super, atexit timing report, **runs Phase 0a in the constructor**, optional eager AnySplat spawn. Constructor; nerfstudio instantiates via `_target`.

  - **`get_training_callbacks(self, training_callback_attributes)`** — `:176` — adds a `_stamp_train_start` (BEFORE_TRAIN_ITERATION) and `_finalize_static_training` (AFTER_TRAIN) callback. Override; nerfstudio engine calls.

  - **`_finalize_static_training(self, step)`** — `:203` — AFTER_TRAIN: record train wall, one-shot opacity purge, Phase 0b fusion, save `static_state.pt`. Refs: 1 — registered as callback at `:198`. **Callback entry point.**

  - **`_write_timing_report(self)`** — `:284` — atexit: write `timing_report_static.txt`. Refs: 4 — registered at `:136` (`atexit.register`). **atexit entry point.**

- **`StaticGSTrainer(NoSaveTrainer)`** — `:400` — adds photometric-loss EMA early-stop. Refs: 3 — `dynamic_gs_config.py:23,...` (`_target` of both static method TrainerConfigs). **Live entry-point.**

  - **`train_iteration(self, step)`** — `:408` — calls `Trainer.train_iteration`, logs loss EMA once/sec, sets `self.stop_training` when EMA < threshold for PATIENCE steps. Override; nerfstudio train loop calls.

- **Module-level early-stop constants** `STATIC_EARLY_STOP_ENABLED/_LOSS/_PATIENCE/_MIN_STEPS` — `:387-397` — env-overridable; read in `StaticGSTrainer.train_iteration`. **Live.**

---

## 2) DEAD-CODE CANDIDATES

After grep, **no top-level function/method/class is fully dead.** The two zero-external-ref symbols are both protected entry points:

- `_shrink_oversized_scales_cb` — `static_gs_model.py:244` — 0 external refs, **but registered as a `TrainingCallback` func** at `static_gs_model.py:235`. NOT dead (callback dispatch).
- `_finalize_static_training` — `static_gs_pipeline.py:203` — 1 ref, the callback registration at `:198`. NOT dead.

`StaticGSModel` (`:193`) and `StaticGSPipeline`/`StaticGSTrainer` show 0 "by-name" usage outside their own files but are reached via `_target` factory / `pyproject.toml` method_configs — entry points, NOT dead.

### Dead *config fields* (declared on `StaticGSModelConfig`, never read on a static instance)

The whole **change-mask block** on `StaticGSModelConfig` is unread. The grep hits for these names all resolve to `DynamicGSModelConfig` reads in `dynamic_gs_model.py:1974-1981` (`self.config.change_mask_*`), and there is **zero `config.change_mask_*` read on the static model/pipeline**. The fields' own docstring admits "not consumed yet". Note `change_mask_mode` (read at `dynamic_gs_model.py:1981`) is **not even declared** on `StaticGSModelConfig` — so this is not a kept-in-sync mirror, it is a partial dead copy.

- `change_mask_depth_threshold` — `static_gs_model.py:184` — 0 static reads.
- `change_mask_rgb_threshold` — `:185` — 0 static reads.
- `change_mask_use_rgb` — `:186` — 0 static reads.
- `change_mask_blur_kernel_size` — `:187` — 0 static reads.
- `change_mask_blur_sigma` — `:188` — 0 static reads.
- `change_mask_filter_radius` — `:189` — 0 static reads.
- `change_mask_min_component_size` — `:190` — 0 static reads.

(Confidence: medium — they are knobs reserved by docstring for a future early-exit, so removal is a judgment call, not a guaranteed safe delete.)

---

## 3) DATA-LIFECYCLE

### The four identity buffers (INVARIANT-PROTECTED — see CLAUDE.md Invariant #8)

- Allocated `persistent=True` in `populate_modules` (`:287-306`), all zeros. `sam3d_init_target_flags` is never written at runtime by design (writer has no caller — invariant-protected; do not flag).
- **Write owners (correct per invariant):** `object_flags`/`object_instance_ids`/`inserted_flags` written only inside `insert_object_gaussians` (`:496-500`); `object_flags` left 0 here (D0 selection is the dynamic pipeline's job). Opacity-purge `delete_gaussian_indices` slices all 4 (`:530-533`).
- **Lockstep correctness:** `insert_object_gaussians` resizes gauss_params (6 tensors, `:489-493`) then `_resize_dynamic_buffers` (`:495`); `delete_gaussian_indices` slices gauss_params (6, `:526-528`) then the 4 buffers (`:530-533`) with the SAME `keep` mask. Lockstep holds in both paths. **No desync found within this file.**
- **Save/load shape contract:** `save_post_fusion_state` writes the full `model.state_dict()` (`post_fusion_cache.py:57`), which includes all 4 buffers + 6 gauss_params. `StaticGSModel.load_state_dict` (`:333`) reshapes the 4 buffers to `gauss_params.means.shape[0]` and zero-fills any missing buffer key (`:372-373`). `load_post_fusion_state` reallocates the 6 gauss_params then calls `load_state_dict(strict=False)`. Contract consistent. **(Low) potential silent-zero:** if a snapshot is missing a buffer key, `load_state_dict` fills it with zeros silently (`:372-373`) — fine for the documented all-zero buffers, but it would also mask a genuinely-truncated/corrupt save of a populated `object_instance_ids`/`inserted_flags`.

### Optimizer state

- `_refresh_gaussian_optimizers` (`:411`) clears Adam `m/v` state for every gauss_params optimizer after insert/delete (state shapes no longer match). Correct. **Smell:** clears ALL optimizers' state unconditionally and ignores its `reset_means_optimizer` parameter (see SMELLS) — no leak, just a misleading API.
- `_optimizers_wrapper` cached in `step_cb` (`:321`); used in `_refresh_gaussian_optimizers` (`:423-426`). If a refresh happens before the first `step_cb` (e.g. Phase 0b before any training step), `_optimizers_wrapper is None` and the wrapper-side param list is skipped — benign here because Phase 0b runs AFTER training, but it is an order dependency.

### `self.info` (GPU rasterization output)

- `_get_object_mask_slab_indices` / `_get_existing_object_subset` read `self.info["depths"]` (`:575,618`), which Splatfacto sets only inside `get_outputs` (`splatfacto.py:555`). These methods will use a STALE or absent `self.info` unless the caller (Phase 0b) renders the right camera immediately before. Implicit temporal coupling, not a leak. `_get_existing_object_subset` raises `RuntimeError` if `info["depths"]` is None (`:620`); the slab variant returns empty instead (`:577`) — inconsistent failure modes for the same precondition.

### Process / file handles

- **Eager AnySplat worker** (`static_gs_pipeline.py:160-170`): spawns a DETACHED FIFO worker process at `<data>/.anysplat_worker` and **never joins/kills it here** — intentional (the subsequent `dynamic-gs-live` adopts it). If no dynamic run follows, this is an orphaned process + FIFO dir (lifecycle owned elsewhere; flagged as a low cross-process concern).
- **atexit timing report** (`:136`): `_write_timing_report` is idempotent via `_timing_report_written` (`:290,301`) — guards the double-fire on Ctrl+C. No leak.
- The Phase 0a SAM3D subprocess moves the model CPU↔GPU around its run (per docstring `:139-141`); not visible in this file, no handle owned here.

### GPU tensors

- Insert/delete build new `torch.nn.Parameter`s under `@torch.no_grad()`; old tensors dropped to GC. No explicit `del`/`empty_cache`, but no retained references → no leak detectable in-file.

---

## 4) DESIGN SMELLS

- **(High) Stale / contradictory module docstring** — `static_gs_pipeline.py:5-9` says the pipeline "reuses `DynamicGSModel` as-is"; it actually uses `StaticGSModel` (`dynamic_gs_config.py:46` → `StaticGSModelConfig` → `_target=StaticGSModel`). Lines `:15-16` say Phase 0b is "CPD / TEASER++ registration", but the default backend is NDP (`static_gs_model.py:150` `sam3d_registration_backend="ndp"`). `static_gs_model.py:477` docstring on `insert_object_gaussians` likewise still says "after CPD/TEASER++". Misleading; should say "NDP (default) / CPD / TEASER++". Recommendation: correct the model name and registration backend in all three docstrings.

- **(Medium) Dead config block on `StaticGSModelConfig`** — `static_gs_model.py:181-190`, the 7 `change_mask_*` fields, are never read on a static instance (all reads are on `DynamicGSModelConfig`). The companion `change_mask_mode` isn't even declared, so it isn't a faithful mirror. Recommendation: either delete the block or add a one-line "WIRED? no" note distinguishing it from the genuinely synced SAM3D block.

- **(Medium) Unused parameter `reset_means_optimizer`** — `_refresh_gaussian_optimizers(self, reset_means_optimizer)` (`static_gs_model.py:411`) never reads the arg; both call sites pass `True` (`:501,535`) and the body clears ALL optimizer state regardless. The docstring (`:413-415`) describes per-means behavior that the code does not implement. Recommendation: drop the parameter or implement the documented conditional clear.

- **(Medium) Inconsistent missing-`info` handling** — same precondition (`self.info["depths"]` present), two failure modes: `_get_existing_object_subset` raises (`:620`), `_get_object_mask_slab_indices` returns empty (`:577`). Recommendation: pick one.

- **(Low) Implicit `self.info` temporal coupling** — `_get_existing_object_subset` / `_get_object_mask_slab_indices` depend on a render having just populated `self.info`; nothing in the signature signals this. Recommendation: accept `info` as an explicit argument, or assert freshness.

- **(Low) Local imports inside hot/looped code** — `import math` inside `_shrink_oversized_scales_cb` (`:253`, runs every `scale_clamp_every_n` steps), `import time as _t` inside `train_iteration` (`static_gs_pipeline.py:419`, every step). Negligible cost but stylistically noisy; move to module top.

- **(Low) Broad swallowed exceptions** — `static_gs_pipeline.py`: timing-ledger reset `except Exception: pass` (`:123-124`), Phase 0a `except Exception` logs + continues (`:148-152`), eager-AnySplat spawn (`:169-170`), timing-record blocks (`:217-218`, `:343-344`), and the whole `_write_timing_report` body. Phase 0a swallow is partly intentional (degrade to no-prefusion) but the bare `pass` on the ledger reset hides config errors. Recommendation: narrow or at least log.

- **(Low) `StaticGSTrainer` instance attrs set lazily, not declared** — `_loss_t0/_loss_last/_loss_ema` first assigned inside `train_iteration` (`static_gs_pipeline.py:421-424`) guarded by `getattr(..., None)`; only `_early_stop_hits` is a class attr (`:406`). Works, but the implicit lazy-init is easy to miss. Recommendation: declare all four as class attrs for clarity.

- **(Low) Duplicated subset/insert/spacing logic** — `_estimate_spacing`, `_get_object_mask_slab_indices`, `_get_existing_object_subset`, `_build_new_gaussian_tensors`, `_resize_dynamic_buffers`, `insert_object_gaussians`, `delete_gaussian_indices` are near-duplicated between `StaticGSModel` and `DynamicGSModel` (grep shows twin definitions in both files). The CLAUDE.md note "StaticGSModel does NOT subclass DynamicGSModel … so the dynamic-phase code paths can't be reached" is the deliberate reason, but the shared Phase-0b machinery is a copy-paste maintenance hazard (the buffers + insert path must stay byte-identical for warm-restart). Recommendation: extract the shared Phase-0b/identity-buffer machinery into a mixin both subclass, keeping the dynamic-only render/tracker paths off the static model.
