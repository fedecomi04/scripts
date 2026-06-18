# Adversarial Code Audit — `dynamic_gs/dynamic_gs_model.py`

Module: `dynamic_gs/dynamic_gs_model.py` (2412 lines, LIVE-PATH, scheduled for purge).
Grep base: `grep -rn <sym> dynamic_gs scripts --include=*.py`.

Key structural fact established by grep (drives most of this report):
**`StaticGSModel` (`dynamic_gs/static_gs_model.py`) is a SEPARATE subclass of `SplatfactoModel`, NOT a subclass of `DynamicGSModel`.** Phase-0 / Phase-0b (`fusion/phase0.py`, `static_gs_pipeline.py`) operate on the *static* model and call `_get_existing_object_subset`, `_get_object_mask_slab_indices`, `insert_object_gaussians`, `_build_new_gaussian_tensors`, `_estimate_spacing` on **StaticGSModel's own copies** (those defs exist at `static_gs_model.py:603` etc.). Therefore the SAM3D-init suite *inside `dynamic_gs_model.py`* is reached ONLY from `DynamicGSModel.initialize_object_from_sam3d`, which itself has no caller. The "external refs" the grep shows for those names are StaticGSModel/phase0 doc-comments + StaticGSModel's own defs — none of them call into `DynamicGSModel`.

---

## 1) FUNCTION / CLASS MAP

### `DynamicGSModelConfig` (dataclass) — `dynamic_gs_model.py:56`
Config for `DynamicGSModel`; ~120 fields (Splatfacto + CDN + XFeat + SAM3D-teaser + segmentation). `_target` → `DynamicGSModel`. Consumed by the entry-point method configs (`pyproject.toml` `dynamic-gs`, `dynamic-gs-live`).

### `class DynamicGSModel(SplatfactoModel)` — `dynamic_gs_model.py:549`

- `__init__(self, config, metadata=None, **kwargs)` — :552 — Init instance attrs (phase, lr caches, reference buffers, `_d0_tracked_instance_id`) then `super().__init__`. Caller: nerfstudio model construction via `_target`. Entry point.
- `load_state_dict(self, state_dict, **kwargs)` — :568 — Resizes the 4 identity buffers to match checkpoint point count, injects zeros for missing buffers. Caller: nerfstudio/`persistence.load_post_fusion_state` (warm-cache `.pt`). Override, framework-driven.
- `populate_modules(self)` — :619 — Sets sim background, registers the 4 identity buffers + `current_active_mask` + `change_mask_image`, installs `NoRefineStrategy`, hooks means grad, creates `viewer_object_selector`. Caller: nerfstudio framework. Entry point.
- `_refresh_viewer_object_options(self)` — :678 — Rebuild the "Visualize Object" dropdown from `object_instance_ids`. Caller: `get_outputs_for_camera` (:721) only. NO EXTERNAL REFS.
- `_viewer_keep_mask(self, selection)` — :693 — Boolean keep-mask for the viewer object isolator. Caller: `get_outputs_for_camera` (:726) only. NO EXTERNAL REFS.
- `get_outputs_for_camera(self, camera, obb_box=None)` — :708 — Locked render entry for the NS viewer; applies the per-object isolator. Callers: `dynamic_gs_pipeline_base.py:512` (comment), `scripts/view_anysplat_nerfstudio.py:185`. NOTE invariant #9: NS viewer is OFF in live (`vis="tensorboard"`), so the only real live use of the lock wrapper is the standalone viewer script; in live this method is essentially not driven.
- `attach_render_lock(self, lock_ctx_factory)` — :741 — Store the shared `_model_lock` factory used by `get_outputs_for_camera`. Caller: `dynamic_gs_pipeline_base.py:520`. LIVE.
- `_get_background_color(self)` — :749 — Return sim background. Callers: 4 (base pipeline + internal). LIVE.
- `step_cb(self, optimizers, step)` — :754 — Per-step: apply `_step_offset` warm-cache shift, cache base LRs/scheduler states, re-apply phase trainability + optimizers. Caller: nerfstudio strategy callback (framework). Entry point. LIVE.
- `get_gaussian_param_groups(self)` — :779 — Return the 6 gauss param groups. Caller: nerfstudio (framework override). NO EXTERNAL REFS in repo (framework call).
- `set_phase(self, phase, reset_means_optimizer=False)` — :789 — Switch static/dynamic; on "static" calls `reset_dynamic_state`. Callers: recorded/live pipelines + datamanager. LIVE.
- `_apply_phase_trainability(self)` — :796 — Toggle requires_grad per phase (means trainable only in dynamic). Internal. LIVE.
- `_apply_phase_optimizers(self, reset_means_optimizer)` — :802 — Zero LRs of inactive groups, clear optimizer state. Internal. LIVE.
- `_mask_means_grad(self, grad)` — :825 — Means-grad hook: in dynamic returns zeros (invariant #4). Caller: `register_hook` (3 refs are the hook installs). INVARIANT-PROTECTED. LIVE.
- `_set_optim_mask(self, mask)` — :833 — Store CDN mask into `change_mask_image`. Caller: `prepare_dynamic_update`/`refresh_dynamic_state_after_insertion` only. NO EXTERNAL REFS (both callers are themselves dead — see §2).
- `_get_optim_mask(self, target_shape=None)` — :842 — Fetch/resize `change_mask_image`. Caller: `get_loss_dict` (dynamic branch, :2372). NO EXTERNAL REFS.
- `get_live_rgb(self, batch, background=None, apply_training_downscale=True)` — :854 — Composite live RGB with bg. Caller: `dynamic_gs_pipeline_base.py:2214` (always `apply_training_downscale=False`). LIVE.
- `_normalize_quaternions(quats)` staticmethod — :865 — Unit-normalize quats. Caller: internal (rigid transforms). NO EXTERNAL REFS.
- `_quaternion_multiply(lhs, rhs)` staticmethod — :869 — Hamilton product. Caller: internal. NO EXTERNAL REFS.
- `_rotation_matrix_to_quaternion(rotation)` staticmethod — :883 — R→quat. Caller: internal rigid transforms. NO EXTERNAL REFS.
- `apply_rigid_object_transform(self, rotation, translation)` — :922 — Rigid transform of `object_flags>0.5` Gaussians (absolute, from live means). NO EXTERNAL REFS — superseded by `_from_reference` (the doc/comment refs are all to the `_from_reference` variant). See §2.
- `_tracked_object_mask(self)` — :941 — Strict `instance_ids == d0_id` mask. Callers: internal (`capture_reference_object_pose`, `apply_rigid_object_transform_from_reference`, `render_object_mask`). LIVE.
- `capture_reference_object_pose(self, instance_id=None)` — :965 — Snapshot tracked object means/quats at D0. Callers: `dynamic_gs_pipeline_base.py:1167`. LIVE.
- `apply_rigid_object_transform_from_reference(self, rotation, translation)` — :986 — Apply tracker pose to the D0-reference means/quats. Callers: `dynamic_gs_pipeline_base.py:2164`. **The core live tracker write-path.** LIVE.
- `_has_persistent_object_membership(self)` — :1010 — True if sam3d+object flags set. Callers: internal (`reset_dynamic_state`, `prepare_dynamic_update`). LIVE (via reset path).
- `reset_dynamic_state(self)` — :1018 — Zero dynamic buffers + reference state. Caller: `set_phase("static")` (:794). Reachable via datamanager init. LIVE-adjacent.
- `_resize_dynamic_buffers(self, num_points)` — :1030 — Grow/copy the 4 identity buffers on point-count change. Callers: `insert_inpaint_gaussians`, `insert_object_gaussians`. LIVE (insert path).
- `_refresh_gaussian_optimizers(self, reset_means_optimizer)` — :1063 — Rebind optimizers + re-install means hook after a resize. Callers: `delete_gaussian_indices`, `insert_inpaint_gaussians`, `insert_object_gaussians`. LIVE.
- `_build_new_gaussian_tensors(self, new_xyz, new_rgb)` — :1082 — Build SH/scale/opacity tensors for new Gaussians. Caller: `insert_object_gaussians` only. (External refs are StaticGSModel's own copy.) Dead-in-dynamic-model — see §2.
- `delete_gaussian_indices(self, indices)` — :1115 — Drop Gaussians; slices all 6 params + 4 buffers + active mask; refresh optimizers. Callers: `static_gs_pipeline.py:237`, `dynamic_gs_pipeline_base.py:2769,2852` (dynamic-phase purge). LIVE.
- `insert_inpaint_gaussians(self, xyz, features_dc, features_rest, opacities, scales, quats, instance_id=999)` — :1149 — Concat FF-decoded Gaussians, mark `object_flags=1/instance_id/inserted_flags=1`, preserve requires_grad. Callers: `dynamic_gs_pipeline_base.py:2700,3498` (FF bg thread). **HOT LIVE PATH.**
- `insert_object_gaussians(self, new_xyz, new_rgb, object_flag=True, instance_id=0)` — :1230 — Insert SAM3D object Gaussians (uses `_build_new_gaussian_tensors`). Caller: `initialize_object_from_sam3d` (:1745) only. Dead-in-dynamic-model (phase0 uses StaticGSModel's copy) — see §2.
- `_should_apply_camera_optimizer(self, camera)` — :1259 — Gate camera-opt (training+static+mode!="off"). Caller: `_get_optimized_camera_to_world`. NO EXTERNAL REFS; the True branch is unreachable in dynamic phase (invariant #2 keeps camera-opt off; phase is "dynamic").
- `_get_optimized_camera_to_world(self, camera)` — :1275 — Return (optionally optimized) c2w. Callers: internal (`get_outputs`, `render_object_mask`-adjacent, SAM3D). NO EXTERNAL REFS.
- `_clone_camera(camera)` staticmethod — :1280 — `camera.to(device)`. Caller: `_get_scaled_camera`. NO EXTERNAL REFS.
- `_get_scaled_camera(self, camera)` — :1284 — Clone + rescale by downscale factor. Callers: internal (`get_outputs`, `render_object_mask`, `_get_render_projection_params`). NO EXTERNAL REFS.
- `_get_render_projection_params(self, camera)` — :1297 — Return (viewmat, K, w, h) numpy. **ZERO refs anywhere (internal or external).** DEAD — see §2.
- `_estimate_spacing(points, max_samples=50000)` staticmethod — :1305 — Median kNN spacing. Callers: internal (`_build_persistent_object_membership`, `_propagate_instance_membership`). External refs are StaticGSModel's own copy.
- `_build_persistent_object_membership(self, ...)` — :1317 — Flag `object_flags` near proxy/target clouds (sklearn KNN). Caller: `initialize_object_from_sam3d` (:1763) only. Dead-in-dynamic-model — see §2.
- `_propagate_instance_membership(self, ...)` — :1390 — Assign `object_instance_ids` for unassigned Gaussians near a proxy cloud (multi-object). **ZERO refs anywhere.** DEAD — see §2.
- `_get_object_mask_slab_indices(self, render_object_mask, rendered_depth, depth_tol_m=0.01)` — :1456 — Gaussian indices under mask within depth tol (SAM3D cull set). NO refs into the DynamicGSModel copy (phase0 calls StaticGSModel's copy at `static_gs_model.py:559`). Dead-in-dynamic-model — see §2.
- `_get_existing_object_subset(self, render_object_mask, rendered_depth)` — :1500 — Frontmost-per-pixel thinned object subset for SAM3D registration target. Caller: `initialize_object_from_sam3d` (:1618) only. Dead-in-dynamic-model (phase0 uses StaticGSModel's copy at :603) — see §2.
- `initialize_object_from_sam3d(self, render_image_path, object_mask_path, render_object_mask, rendered_depth, camera, image_debug_dir, artifact_dir, frame_name)` — :1603 — Full SAM3D generate→register→insert→membership pipeline (~235 lines). **Only writer of `sam3d_init_target_flags`; documented in CLAUDE.md Invariant #8 as having NO caller ON PURPOSE.** INVARIANT-PROTECTED (not flagged dead). Its whole transitive helper subtree (`_get_existing_object_subset`, `_build_persistent_object_membership`, `insert_object_gaussians`, `_build_new_gaussian_tensors`, `_estimate_spacing`) is reachable only from here.
- `refresh_dynamic_state_after_insertion(self, camera, render_object_mask, optim_mask)` — :1841 — Recompute active mask after insertion. **ZERO refs anywhere.** DEAD — see §2.
- `_get_gt_depth(self, batch)` — :1868 — Downscale GT depth to render res. Callers: internal + base pipeline (5 refs). LIVE.
- `_get_batch_mask(self, batch)` — :1885 — Downscale batch mask. Callers: internal + base (2 refs). LIVE.
- `_masked_rgb_l1(pred, gt, mask)` staticmethod — :1902 — Masked L1. Caller: `get_loss_dict` (:2377). NO EXTERNAL REFS.
- `_get_esam_model(self)` — :1909 — Lazy-build ESAM. Caller: `prepare_dynamic_update` (:2017) only. NO EXTERNAL REFS (caller is dead) — see §2.
- `prepare_dynamic_update(self, camera, batch, external_object_mask=None, skip_object_flags_write=False)` — :1914 — Legacy ESAM-based change-mask + object-segmentation god-method (~210 lines). **ZERO real callers** (only its own return dict + doc-comments in `static_gs_model.py`/`esam.py`). DEAD — the XFeat tracker + CDN render in `dynamic_gs_pipeline_base.py` replaced it. See §2.
- `render_object_mask(self, camera)` — :2131 — Rasterize tracked-instance Gaussians → binary mask. Callers: `dynamic_gs_pipeline_base.py:1680` (via `_render_object_mask_cached`). **HOT LIVE PATH** (every few ticks).
- `get_outputs(self, camera)` — :2215 — Full forward render; sets `self.info`, returns rgb/depth/flagged/non_flagged/non_inserted/accumulation/background. Callers: framework + base pipeline (`get_outputs(camera)` at :1699,1722, plus internal). **HOT LIVE PATH** (CDN render).
- `get_metrics_dict(self, outputs, batch)` — :2359 — Add object/active counts. Caller: nerfstudio framework. **Unreachable in live** — live `get_train_loss_dict` returns a zero-loss dummy and never invokes loss/metrics (`dynamic_gs_pipeline_base.py:975-999`).
- `get_loss_dict(self, outputs, batch, metrics_dict=None)` — :2368 — Static→super; dynamic→masked rgb+depth+rigid loss. Caller: nerfstudio framework. **Dynamic branch unreachable in live** (same reason — trainer loss is bypassed). The static branch runs only under `static-gs` (StaticGSModel overrides loss anyway).
- `step_post_backward(self, step)` — :2404 — Assert + no-op (LRs zeroed). Caller: framework strategy callback. LIVE (no-op).

---

## 2) DEAD-CODE CANDIDATES

Entry points (framework method-configs, `register_hook`/strategy callbacks, monkeypatch targets, the 4 invariant buffers, and `initialize_object_from_sam3d` per Invariant #8) are EXCLUDED. Remaining genuine zero-ref suspects:

| Symbol | Line | Ref evidence | Confidence | Note |
|---|---|---|---|---|
| `prepare_dynamic_update` | 1914 | grep returns only its own return-dict key + doc-comments in `static_gs_model.py:27`, `esam.py:251`. No call site. | **high** | Largest dead body (~210 lines). Legacy ESAM segmentation path superseded by the XFeat tracker + CDN render in `dynamic_gs_pipeline_base.py`. Pulls in the entire ESAM machinery (`_get_esam_model`, `query_esam_mask_pair`, `build_esam_ti`, `combine_object_masks`, `build_active_mask` imports) as dead weight. |
| `refresh_dynamic_state_after_insertion` | 1841 | 0 refs anywhere (`grep` empty). | **high** | Was the post-SAM3D-insert active-mask refresh; no caller. |
| `_propagate_instance_membership` | 1390 | 0 refs anywhere. | **high** | Multi-object instance propagation; never wired (multi-object is roadmap, not implemented). |
| `_get_render_projection_params` | 1297 | 0 refs anywhere (internal or external). | **high** | Unused projection-params helper. |
| `_get_esam_model` | 1909 | only caller is `prepare_dynamic_update` (dead). | **high** | Dead-by-transitivity. The `_esam_model` attr is otherwise unused. |
| `apply_rigid_object_transform` | 922 | All refs are to `apply_rigid_object_transform_from_reference`; the bare variant has 0 real callers. | **high** | Superseded by `_from_reference`. |
| `get_live_rgb` True-branch (downscale) | 860-861 | only caller passes `apply_training_downscale=False` (`dynamic_gs_pipeline_base.py:2214`). | medium | The method is live; only its downscale branch is dead. |
| `_get_optim_mask` / `_set_optim_mask` | 842 / 833 | callers are `get_loss_dict`(dynamic, unreachable-in-live) + the dead `prepare_dynamic_update`/`refresh_dynamic_state_after_insertion`. | medium | Live-dead via the loss-bypass; keep if recorded-pipeline loss is ever used. |
| `_masked_rgb_l1` | 1902 | only caller `get_loss_dict` dynamic branch (unreachable in live). | medium | Same loss-bypass reasoning. |

**Dead-in-dynamic-model (the SAM3D-init subtree).** These have refs that LOOK external but actually resolve to **StaticGSModel's own identically-named copies** (phase0 runs on the static model): `_get_existing_object_subset` (:1500), `_get_object_mask_slab_indices` (:1456), `insert_object_gaussians` (:1230), `_build_new_gaussian_tensors` (:1082), `_estimate_spacing` (:1305), `_build_persistent_object_membership` (:1317). Inside `DynamicGSModel` their only caller is `initialize_object_from_sam3d`, which is invariant-protected-no-caller. So this whole subtree is unreachable on the dynamic model. **NOT flagged as repo-dead** (the StaticGSModel twins are live) — flagged here as duplicated/unreachable-on-this-class (see §4). Confidence the *DynamicGSModel copies* are unreachable: **high**.

**Explicitly NOT dead (invariant-protected / framework):** the 4 identity buffers, `sam3d_init_target_flags` + `initialize_object_from_sam3d` (Invariant #8), `_mask_means_grad` (Invariant #4 hook), `step_cb`/`step_post_backward`/`populate_modules`/`load_state_dict`/`get_gaussian_param_groups`/`get_metrics_dict`/`get_loss_dict` (nerfstudio framework overrides — reachable under `static-gs` / recorded even if loss-bypassed in live).

---

## 3) DATA-LIFECYCLE

**The 4 identity buffers + `current_active_mask` + `change_mask_image`** (registered at :626-662):
- `object_flags`, `sam3d_init_target_flags`, `object_instance_ids`, `inserted_flags` are `persistent=True` (saved to `.pt`); `current_active_mask` and `change_mask_image` are `persistent=False`.
- **Resize correctness:** three independent resize paths — `load_state_dict` (:577), `_resize_dynamic_buffers` (:1030), `delete_gaussian_indices` (:1139). All five buffers are kept in lockstep in `_resize_dynamic_buffers` and `delete_gaussian_indices`. **`load_state_dict` resizes only `object_flags/current_active_mask/sam3d_init_target_flags/object_instance_ids/inserted_flags`** — consistent. OK.
- **Desync risk (medium):** `delete_gaussian_indices` (:1143) guards `current_active_mask` with `if self.current_active_mask.shape[0] == num_points` but unconditionally slices the 4 persistent buffers with `[keep]`. If a prior code path left a persistent buffer at a stale length (e.g. a partial resize), the `[keep]` slice would raise or silently mis-map. The buffers are assumed already == `num_points`; there is no assertion. Low likelihood given the lockstep resizers, but no defensive check.
- **`object_flags` overwrite hazard (by design, Invariant #8):** `_build_persistent_object_membership` does `self.object_flags.copy_(persistent_flags)` (:1340,1380) — *overwrites* (erases prior identity). `prepare_dynamic_update` (dead) and the active-mask paths also write `object_flags.copy_(active...)` (:2085). Only relevant if those paths revive.

**Warm-cache `.pt` (`persistence/`):**
- `load_state_dict` (:568) is the load hook; injects zero buffers for any missing key (back-compat with older `.pt`). Shape derived from `gauss_params.means`. **No format/shape mismatch found** — the 6 gauss params + 4 buffers are all handled.
- `_step_offset` (read in `step_cb` :764, `step_post_backward` :2409) is set by the pipeline's warm-cache loader; default 0. The `step_post_backward` assert `step == self.step - _step_offset` (:2409) will fire (AssertionError, kills the trainer thread) if the pipeline sets `_step_offset` but the strategy passes an already-shifted step — fragile coupling across modules.

**`self.info` (rasterization output dict) — THREAD-SAFETY (HIGH, live):**
- Written once per render at `get_outputs` :2297 (`self.info = info`).
- Read by `_get_object_mask_slab_indices` (:1467,1472), `_get_existing_object_subset` (:1505,1510), `refresh_dynamic_state_after_insertion` (:1847), `prepare_dynamic_update` (:2061) via `extract_projected_centers_and_radii(self.info, ...)`.
- It is a **single shared mutable attribute**. In live, `get_outputs` is called from the tracker tick, the FF bg thread (`dynamic_gs_pipeline_base.py:1699,1722`), and (if NS viewer were on) the render thread — all under `_model_lock`. The reads of `self.info` in the still-live `render_object_mask`/CDN path happen via `get_outputs` returns, not via the dead `self.info` consumers — but **any future read of `self.info` outside the lock races a concurrent `get_outputs` overwrite**, producing centers/radii from the wrong render (silent wrong masks, or a shape-mismatch CUDA assert if a resize interleaved). The dead consumers (`_get_existing_object_subset` etc.) would reintroduce this if revived. Flag: `self.info` has no per-call ownership; correctness depends entirely on the external `_model_lock` discipline.

**Per-tick GPU/heap allocations (live hot path):**
- `render_object_mask` (:2131) and `get_outputs.render_subset_rgb` (:2303) each run extra `rasterization` passes per call. `get_outputs` runs **three** subset renders every call (`flagged_rgb`, `non_flagged_rgb`, `non_inserted_rgb`, :2329-2334) — i.e. 4 rasterizations per CDN render. `non_inserted_rgb`/`flagged_rgb`/`non_flagged_rgb` are only consumed by the dead/legacy CDN-Path-A code; in the live XFeat path these are computed-and-discarded per tick (wasted GPU + allocation). See §4.
- `insert_inpaint_gaussians`/`insert_object_gaussians`/`delete_gaussian_indices` rebuild full `torch.nn.Parameter` tensors via `torch.cat`/slice on every call (:1218, :1242, :1137) and `optimizer.state.clear()` — O(N) realloc each FF insert; on the documented 459k→1.29M growth this is heavy churn, but unavoidable given the resize model.

**Process/file handles:** `initialize_object_from_sam3d` (dead path) does `self.to("cpu")` + `gc.collect()` + `torch.cuda.empty_cache()` around the SAM3D subprocess (:1658-1682) inside try/finally — handle/device restore is correct there. No leaked file handles in the live methods.

**No SHM in this module** — SHM lifecycle lives in `live_shm_reader.py`/`live_ros_publisher.py`, not here.

---

## 4) DESIGN SMELLS

- **God method (high): `prepare_dynamic_update` (:1914, ~210 lines)** — change-mask + ESAM segmentation + flagging + reference capture + return-dict-of-18-keys, all in one. It is also entirely DEAD (§2). Highest-value purge target.
- **God method (high): `initialize_object_from_sam3d` (:1603, ~235 lines)** — generate/register/insert/membership/log/timing in one. Invariant-protected (no caller by design), but its size makes the dead-vs-live boundary hard to read.
- **Duplicated logic across two model classes (high):** `_get_existing_object_subset`, `_get_object_mask_slab_indices`, `insert_object_gaussians`, `_build_new_gaussian_tensors`, `_estimate_spacing` exist as **near-identical copies** in both `dynamic_gs_model.py` and `static_gs_model.py`. The DynamicGSModel copies are unreachable (phase0 uses the static twins). This is the single biggest cleanup lever and a maintenance trap (a fix to one won't reach the other).
- **Wasted per-tick compute (high, live):** `get_outputs` always renders `flagged_rgb` / `non_flagged_rgb` / `non_inserted_rgb` (3 extra `rasterization` calls, :2329-2334) regardless of whether the caller needs them. The live XFeat+CDN path consumes only `rgb`/`depth`/`accumulation`/`background`; the three subset renders are computed and discarded on most ticks. No flag to skip them.
- **Leaky abstraction / fragile cross-module coupling (medium):** `self.info` is a public mutable attribute used as an implicit channel between `get_outputs` and several consumers; correctness depends on the pipeline's `_model_lock` being held externally (see §3). `_step_offset`, `_render_lock_ctx`, `_optimizers_wrapper` are all attributes poked in from the pipeline via `getattr`/`setattr` with silent defaults — no contract enforced.
- **Swallowed exceptions (medium):** `_refresh_viewer_object_options` (:690) `except Exception: pass` (viser-not-bound — acceptable but blanket). `get_outputs_for_camera` (:716) defaults the lock to `nullcontext` when `_render_lock_ctx` is unset — silently disables the race guard for any caller that forgot to `attach_render_lock` (tests/scripts), which is intended but masks misconfiguration in live if `attach_render_lock` ever isn't called.
- **Dead config fields on `DynamicGSModelConfig` (medium):**
  - `enable_dynamic_mean_optimization` (:201) — only read in `_mask_means_grad` (:828), but that branch requires `not enable_cotracker_rigid_motion` AND the flag; with the tracker master-switch default True, the flag's effect is unreachable in the live config. Effectively inert.
  - `change_mask_*` family (`change_mask_depth_threshold`, `change_mask_use_rgb`, `change_mask_blur_*`, `change_mask_filter_radius`, `change_mask_min_component_size`, `change_mask_outlier_*`, `change_mask_mode`, `active_mask_dilate_radius`) — read ONLY by `prepare_dynamic_update` (dead) inside `build_change_mask`. The live CDN path in `dynamic_gs_pipeline_base.py` reads its OWN `ChangeMaskConfig` defaults (per CLAUDE.md notes), NOT these model-config fields. So these are config-dead in live. (`change_mask_downsample_target_side` IS live via `DGS_CDN_TARGET_SIDE`/the base pipeline — verify per-field before deleting.)
  - `rigid_static_lambda`, `rigid_inlier_threshold`, `depth_lambda` (:153,154,63) — read only in `get_loss_dict` dynamic branch, which is loss-bypassed in live (§2). Inert in live.
  - The huge `sam3d_teaser_*` block (:176-200) is consumed only by `initialize_object_from_sam3d` (no-caller-by-design) → never read at runtime on the dynamic model; the static model has its own teaser fields. Threaded as a 13-key dict into `register_and_fuse_sam3d_object` (:1718) — a long param-bag passed through a dead path.
- **Params threaded through many layers (medium):** `initialize_object_from_sam3d` builds a 13-entry `teaser_params` dict (:1718-1732) forwarded to `register_and_fuse_sam3d_object`, and returns an 30+-key timing/stats dict (:1814) — heavy plumbing for a dead-on-this-class method.
- **Misleading name (low):** `enable_cotracker_rigid_motion` (:209) gates **XFeat**, not CoTracker (documented, but actively misleading at call sites like `_mask_means_grad`).
- **Misleading docstring (low):** `prepare_dynamic_update` docstring (:1922) says "The CoTracker tracks the moved object's pose" — CoTracker was purged 2026-05-26; the whole method is dead anyway.
- **`get_outputs` early-return shape (low):** `if not isinstance(camera, Cameras): return {}` (:2216-2217) returns an empty dict that downstream `outputs["rgb"]` access would KeyError on — silent contract break rather than a raise.
