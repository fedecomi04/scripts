# Code Audit — `dynamic_gs/dynamic_gs_pipeline_base.py`

Module: `DynamicGSPipelineBase` + `DynamicGSPipelineBaseConfig`. Shared dynamic-phase foundation for `RecordedDynamicGSPipeline` and `LiveDynamicGSPipeline`. 3554 lines, LIVE-PATH (highest priority).

Grep scope for all "caller count" figures:
`grep -rn "<sym>" scripts/dynamic_gs scripts/scripts --include=*.py` (defs excluded). "self-only" = referenced only inside this file.

Invariant-protected symbols touched here (NEVER flag as dead/buggy): the 4 identity buffers (`object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`), `_ZERO_LR_OPTIMIZERS`/means-LR=0 (enforced in config, not here), the warm-cache `.pt` load/save, and `current_phase = "dynamic"` (trainer fast-path flag).

---

## 1) FUNCTION / CLASS MAP

### Config
- `DynamicGSPipelineBaseConfig(VanillaPipelineConfig)` — :86 — dataclass holding all shared dynamic-phase knobs. Subclassed by both pipeline configs; instantiated via nerfstudio `method_configs` (`dynamic-gs`, `dynamic-gs-live`). Entry point.

### Module-level
- `TrackerFrame = Any` — :348 — type alias documenting the `_latest_tracker_frame` dict shape. Used as annotation; self-only.

### Class `DynamicGSPipelineBase(VanillaPipeline)` — :359

**Lifecycle**
- `__init__(config, device, test_mode, world_size, local_rank, grad_scaler)` — :369 — builds all per-tick counters/locks, registers 8 atexit cleanups, calls `super().__init__`, attaches model render-lock, loads warm cache, sets up viser-direct + AnySplat worker. Called by nerfstudio trainer via the subclass. Entry point.
- `_load_warm_cache_or_die()` — :542 — resolves `post_fusion_cache_subpath` (with legacy `post_fusion_state.pt` fallback), `load_post_fusion_state`, sets `model._step_offset=10000`. 1 ref (`__init__`).
- `_cleanup_viser_direct()` — :614 — flush pending FF + close viser server. atexit (registered :476). self-only.
- `_cleanup_anysplat_worker()` — :631 — close persistent AnySplat worker. atexit (:477). self-only.
- `_cleanup_anysplat_ipc_file()` — :640 — unlink `/dev/shm/anysplat_ipc_<pid>.npz`. atexit (:479). self-only.
- `_cleanup_anysplat_bg()` — :650 — block (60s) on `_anysplat_slot_lock` so the worker isn't killed mid-call. atexit (:478). self-only.
- `_save_final_snapshot_if_enabled()` — :662 — `save_post_fusion_state` → `post_dynamic_state.pt`; idempotent via `_final_snapshot_written`. atexit (:481). self-only.
- `_cleanup_feedforward_video_writer()` — :690 — release the (never-created) video writer. atexit (:480). self-only.
- `_capture_static_sequence_total()` — :704 — on first tracked frame, read+consume `.static_sequence_t0` sidecar, build the from-scratch timing block. 1 real caller each in recorded (:341) + live (:425).
- `_render_static_sequence_section(total_s, rows)` @staticmethod — :779 — format the from-scratch section string. 1 ref (`_capture_static_sequence_total`).
- `_write_timing_report()` — :795 — render `timing_report.txt` from `self._timing` + ledger; idempotent. atexit (:482). self-only.

**Trainer entry**
- `_recurring_ff_due(tick_count, is_first) -> bool` — :939 — PURE gate for Mode-B FF firing (cadence + wall-clock min-gap). Called by both subclasses (recorded :230, live :314).
- `_oneshot_ff_due(step) -> bool` — :966 — **NO REFS FOUND** (only def). The inline logic at :987 duplicates it.
- `get_train_loss_dict(step)` — :975 — per-step dynamic entry: tick → maybe oneshot FF → return zero loss. Called by trainer (override of VanillaPipeline). Entry point.
- `_maybe_torch_profile_step()` — :1005 — lazily build+step `torch.profiler` gated by `DGS_TORCH_PROFILE=1`. 1 ref (`get_train_loss_dict` :985).
- `_finalize_torch_profile()` — :1053 — stop profiler, dump chrome trace + tables. atexit (:483) + 1 internal (:1048). self-only.
- `get_training_callbacks(attrs)` — :1084 — stash `self._trainer`. Override; subclasses call super. Entry point (nerfstudio callback).

**Abstract hooks**
- `_tracker_tick(step)` — :1100 — raises NotImplementedError; implemented by both subclasses.
- `_pick_d0_object(camera, prefused_instance_ids) -> int` — :1108 — raises; implemented by both subclasses; also called by `_selection_fallback_id` (:1210).
- `_on_tracker_frame(camera, batch, cdn, is_first)` — :1118 — raises; implemented by both subclasses.

**Object (re)selection**
- `_reset_d0_guard()` — :1134 — resets `_tracker_tick_count`; live overrides. 1 internal ref (:1189).
- `_reseed_tracked_object(new_id, camera, batch) -> bool` @no_grad — :1140 — full reseed (object_flags + reference pose + mask + XFeat anchor). Called by both subclasses' pickers + 3 internal fallback sites. 5 real call sites.
- `_present_object_ids() -> set` — :1198 — set of buffer instance ids >0. 3 internal refs.
- `_selection_fallback_id(camera) -> int` — :1204 — forced id or heuristic. 3 internal refs.
- `_open_picker_panel(camera, batch)` — :1214 — build viser picker GUI (non-blocking). 1 ref (`_tick_interactive_selection`).
- `_close_picker_panel()` — :1323 — remove picker folder. 3 internal refs.
- `_ensure_change_object_button()` — :1333 — add persistent "Change object" button. 1 ref (`_setup_viser_direct` :1488).
- `_tick_interactive_selection(camera, batch, is_first) -> str` — :1352 — drive picker for one tick; blocks until Done/timeout. Called by both subclasses (recorded :207, live :288).
- `_wait_for_selection(camera) -> Optional[int]` — :1399 — poll the selection Event in 0.25s slices. 1 ref (`_tick_interactive_selection`).

**Viser-direct**
- `_viser_lock_ctx()` — :1433 — return `self._model_lock` (RLock). The central model-mutation/render exclusion. Many internal refs + passed to `model.attach_render_lock` (:520) and `viser_direct_server.model_lock` (:1480).
- `_setup_viser_direct()` — :1452 — spin up `ViserDirectScene`, swap in shared lock, attach model. 1 ref (`__init__` :530).
- `_build_viser_direct_handles(camera)` — :1496 — legacy stub recording the D0 camera pose. Called by both subclasses (recorded :249, live :333).
- `_push_viser_direct_transforms()` — :1521 — push tracked-object (R,t) to viser. Called by `_force_viser_direct_push` (:1608) + subclasses (3 refs).
- `_push_viser_camera_feed(camera, batch)` — :1537 — push live RGB thumbnail + tracked c2w. 2 refs (both subclasses).
- `_viser_direct_register_ff_insert(inserted_ids)` — :1571 — wake render thread after FF insert. 2 internal refs (rgbd + anysplat insert paths).
- `_refresh_viser_direct_after_feedforward()` — :1596 — re-upload static handle. 2 internal refs.
- `_force_viser_direct_push()` — :1605 — public alias for `_push_viser_direct_transforms`. **NO REFS FOUND** outside def.

**Viewer re-render (NS fallback)**
- `_force_viewer_rerender()` — :1614 — poke NS render state machines (only matters with `--vis viewer`). Called by both subclasses (recorded :252, live :336).

**Render + change-mask shims**
- `_render_object_mask_cached(camera)` @no_grad — :1655 — render tracked-object mask at most once/tick under model lock; cached. 6 refs (self + subclasses).
- `_invalidate_object_mask_cache()` — :1685 — null the cache. 4 refs (self + subclasses).
- `_render_from_camera(camera)` — :1688 — train-mode `get_outputs` under model lock. 6 internal refs.
- `_render_from_camera_at_scale(camera, scale)` — :1704 — rescaled render. **NO REFS FOUND** outside def.
- `_compute_tick_cdn(camera, batch)` — :1727 — render + compute change mask (`keep_largest_only=False`). 2 internal refs.
- `_compute_change_mask(...)` — :1781 — thin shim over `change_detection.compute_change_mask` pulling thresholds from `model.config`. 3 internal refs (incl. `DGS_CDN_TARGET_SIDE` env).

**XFeat motion estimator**
- `_object_crop_bbox(camera, padding_px)` @no_grad — :1855 — screen-space bbox of tracked object Gaussians (uses RAW cumulative pose, not KF). 2 internal refs.
- `_crop_for_xfeat(rgb, depth, camera, mask, bbox)` @no_grad — :1935 — crop + rebuild single-camera `Cameras` with shifted cx/cy. 2 internal refs.
- `_initialize_motion_estimator(rgb, depth, camera, mask)` — :1977 — build `XFeatMotionEstimator`, seed D0 reference. 3 refs (self + `_reseed_tracked_object` + subclasses).
- `_apply_motion_estimator(camera, batch, current_mask)` — :2053 — per-tick XFeat advance → `apply_rigid_object_transform_from_reference` under lock. 5 refs (subclasses).
- `_build_tracking_rgb(batch)` — :2209 — gripper-composited live RGB. 3 refs.
- `_get_debug_dir()` — :2226 — debug dir path. 2 refs.
- `_get_motion_debug_dir()` — :2229 — tracker-debug subdir. 1 internal ref.
- `_write_motion_log(frame_name, est)` — :2232 — write per-frame motion txt (debug only). 1 internal ref (`_apply_motion_estimator` :2139, gated `save_debug_images`).
- `_save_motion_debug(frame_name, est)` — :2254 — side-by-side tracked-points PNG (debug only). 1 internal ref (:2140).

**Feedforward dispatcher**
- `_save_ff_debug_images(...)` — :2314 — dump ordered 1..7 PNGs to `_ff_debug/`. 1 internal ref (anysplat path, gated `save_debug_images`).
- `_scale_mask_about_centroid(mask, scale)` — :2437 — enlarge mask about centroid (warpAffine). 2 internal refs.
- `_feedforward_clean_cdn(camera, cdn, frame_name, prerendered_obj_mask)` — :2469 — subtract object footprint from CDN. 4 internal refs.
- `_dispatch_feedforward_async(target_frame, mode_label) -> bool` — :2501 — acquire single-in-flight slot, spawn bg thread. 3 refs (self + subclasses).
- `_feedforward_threaded(target_frame, mode_label)` — :2519 — bg-thread wrapper: render CDN if absent, run FF, release slot. 1 internal ref.
- `_run_feedforward(target_frame, mode_label, *, prerendered_obj_mask, prerendered_depth)` — :2546 — dispatcher; routes to anysplat or runs rgbd_decode loop. 2 internal refs.
- `_feedforward_delete_in_region(camera, component_mask) -> int` @no_grad — :2738 — legacy delete (only on `not feedforward_skip_delete` path). 1 internal ref (:2681).
- `_feedforward_cull_in_front_of_depth(camera, component_mask, gt_depth_m, depth_tol_m) -> int` @no_grad — :2771 — delete Gaussians in front of sensor surface (instance_ids in {0,999}). 2 internal refs.
- `_feedforward_cull_then_reclean_cdn(camera, batch, cdn_clean, gt_depth, *, frame_name, prerendered_obj_mask)` @no_grad — :2854 — cull-before-decode + recompute CDN. 2 internal refs.
- `_start_anysplat_persistent_worker()` — :2899 — adopt-or-spawn the persistent AnySplat worker. 1 ref (`__init__` :540).
- `_resolve_anysplat_context_image_paths(target_frame_idx)` — :2949 — recorded-only target image path; live overrides. 1 internal ref (:3102).
- `_scene_c2w_for_frame(frame_idx) -> np.ndarray` — :2967 — recorded-only c2w lookup; live overrides (raises if pre-D0). 1 internal ref (:3125).
- `_run_feedforward_anysplat(target_frame, mode_label, prerendered_obj_mask)` — :2981 — main prep (CDN clean, components, intrinsics, crop windows) then inline `_anysplat_bg_run`. 1 internal ref (:2562).
- `_anysplat_crop_windows(change_mask_np, W, H, pad_px)` — :3165 — 1-or-2 square crop windows encompassing change. 1 internal ref.
- `_anysplat_bg_run(args)` — :3209 — frustum cull + ICP + per-window AnySplat inference + reproject + union dedup + insert. 1 internal ref (inline at :3163).

---

## 2) DEAD-CODE CANDIDATES (genuine zero-ref, entry points excluded)

| Symbol | file:line | Grep evidence | Confidence |
|---|---|---|---|
| `_force_viser_direct_push` | base:1605 | 0 refs outside def across `dynamic_gs` + `scripts`. Docstring says "kept for compat with subclass call sites that prefer the descriptive name" — but no subclass uses it; they call `_push_viser_direct_transforms` directly. | high |
| `_render_from_camera_at_scale` | base:1704 | 0 refs outside def. Docstring claims "The CDN uses this" but `_compute_tick_cdn` calls `_render_from_camera` (full-res), and the reduced-res CDN approach was explicitly reverted (comment at :1730-1737). | high |
| `_oneshot_ff_due` | base:966 | 0 refs outside def. `get_train_loss_dict` (:987) inlines the identical predicate instead of calling it. Pure dead duplicate. | high |
| `feedforward_anchor_frame` (config) | base:207 | Declared on Config; 0 reads anywhere (`grep feedforward_anchor_frame` → only the declaration). | high |
| `feedforward_video_out` (config) | base:209 | 0 reads. CLAUDE.md confirms "no writer is implemented — no mp4 is produced". | high |
| `feedforward_video_fps` (config) | base:210 | 0 reads. Same dead video feature. | high |

Dead-adjacent (NOT listed as candidates, but noted): `_feedforward_video_writer` attribute (:440) + `_cleanup_feedforward_video_writer` (:690) are the cleanup half of the never-implemented video feature — the writer is only ever set to `None`, so `.release()` never fires. Harmless atexit no-op; remove together with the 3 video config fields if the feature is dropped. NOT flagged as a dead *callable* because the cleanup is atexit-registered (an entry point) and safe.

NOT dead (verified, despite low/zero raw grep counts):
- `_maybe_torch_profile_step` — called at :985 (grep initially miscounted due to a `#` comment on the call line).
- `_oneshot_ff_due`'s sibling `_recurring_ff_due` IS used (subclasses).
- All `_cleanup_*`, `_write_timing_report`, `_save_final_snapshot_if_enabled`, `_finalize_torch_profile` — atexit-registered (entry points).
- All abstract hooks — implemented in subclasses.

---

## 3) DATA-LIFECYCLE

### Warm-cache `.pt` (load) — invariant-protected
- **Load:** `_load_warm_cache_or_die` (:542) resolves `static_scene/static_state.pt` (config `post_fusion_cache_subpath`) with legacy `post_fusion_state.pt` fallback (:548). `load_post_fusion_state(self.model, cache_path, self.device)` REPLACES the SfM-seed gauss_params + restores the 4 identity buffers. Sets `model._step_offset = 10_000` (:593) to bypass Splatfacto resolution/SH schedules.
- **Failure path:** `result.success` False → RuntimeError (:573). Missing file → FileNotFoundError (:556). Good — fails fast, no silent corruption.
- **`_step_offset` fragility:** set in a `try/except` that only logs a warning on failure (:599). If it silently fails, the comment (:584-588) says FF inserts back-project with full-res intrinsics on a downscaled grid → inserts land in wrong world locations. This is a load-bearing side-effect hidden behind a warning, not a hard error. **Flag (low):** a missing `_step_offset` should arguably hard-fail rather than warn.

### Final-snapshot `.pt` (save)
- **Save:** `_save_final_snapshot_if_enabled` (:662) via atexit → `save_post_fusion_state(self.model, dynamic_scene/post_dynamic_state.pt)`. Idempotent via `_final_snapshot_written`. Same format as static-gs. No leak.

### Identity buffers (the 4) — invariant-protected
- Read/written through `self.model` only. `_reseed_tracked_object` (:1165) writes `object_flags` under `_model_lock`; intentionally leaves `inserted_flags` / `sam3d_init_target_flags` / `object_instance_ids` untouched (correct per Invariant #8). FF inserts write `inserted_flags`+`object_instance_ids=999` via `model.insert_inpaint_gaussians`.
- **Desync risk (medium):** `_object_crop_bbox` (:1873-1875) explicitly guards `obj_mask.shape[0] != model.means.shape[0]` and returns None, with the comment "Per-Gaussian buffers can drift out-of-sync with means after some FF delete/insert sequences (separate bug)." This is an acknowledged latent buffer-desync bug between the identity buffers and `means` after delete/insert. The guard is defensive but the root desync is not fixed here.

### GPU tensors / per-tick allocations (LIVE-PATH hot)
- `_render_object_mask_cached` (:1655) caches one mask render per tick; invalidated on tick start, rigid transform (:2168), reseed (:1171). Good — single render reused.
- `_apply_motion_estimator` allocates per-tick crops + composited RGB; bounded by `xfeat_crop_max_side`.
- `_anysplat_bg_run` per-FF allocations: frustum-cull copies `means` (:3282) under lock, builds GPU tensors, `_cat` does H2D copies (:3430). Bounded by single-in-flight slot. Union dedup builds several full-length GPU index tensors — scales with cumulative insert count (CLAUDE.md: 459k→1.29M growth). No periodic dynamic-phase purge implemented → **unbounded GPU growth** over a long run (known TODO, not a new finding).

### SHM / file / process handles
- This module does NOT own the ROS SHM (that's `live_shm_reader`/the live subclass). It owns:
  - `/dev/shm/anysplat_crop_<pid>_<wi>.png` (:3367) and `/dev/shm/anysplat_ipc_<pid>_<wi>.npz` (:3369) — written per FF window. **LEAK (medium):** `_cleanup_anysplat_ipc_file` (:640) only unlinks `anysplat_ipc_<pid>.npz` (no `_<wi>` suffix and no crop PNGs). The actual per-window files written at :3367/:3369 use the `_<wi>` suffix and are **never cleaned up** — they accumulate in tmpfs across every FF call until reboot. The cleanup path targets a filename that the current code never writes.
  - AnySplat persistent worker subprocess: spawned (:2936) or adopted (:2915); closed via `_cleanup_anysplat_worker` (atexit). `_cleanup_anysplat_bg` (atexit, LIFO-ordered before worker) drains the slot lock first so the worker isn't killed mid-call. Good.
  - Viser server: closed via `_cleanup_viser_direct`. Good.
- **Double-cleanup safety:** all atexit handlers wrap in try/except and null their handle; `_write_timing_report`/`_save_final_snapshot` use written-flags. Idempotent.

### No double-loads found. Warm cache loads exactly once in `__init__`.

---

## 4) DESIGN SMELLS

**Thread-safety / races (LIVE-PATH critical)**
- **(high) `_run_feedforward_anysplat` / `_anysplat_bg_run` run on the FF bg thread, but read `self.config` and `self.model.config` freely and mutate `self._timing` and `self._feedforward_call_counter` without a lock.** `_feedforward_call_counter += 1` (:2995, :2571) is a non-atomic read-modify-write; `self._timing[key].append(...)` is hit from BOTH the tracker thread (DN.* keys, e.g. :2062) and the FF bg thread (FF.* keys). `defaultdict.__getitem__` + `list.append` on distinct keys is *mostly* safe under CPython's GIL, but `defaultdict` key-creation for a brand-new key concurrently is not guaranteed. Given the single-in-flight slot, only one FF thread runs at a time, so counter races are limited to FF-vs-FF (serialized) — but tracker-vs-FF timing-dict writes are concurrent. Low blast radius (timing only) but technically unsynchronized shared mutable state.
- **(medium) `_obj_mask_cache` is read/written from multiple threads without the model lock around the cache slot itself.** `_render_object_mask_cached` (:1666) checks `if self._obj_mask_cache is None` then renders under the lock, but the *cache assignment/read* of `_obj_mask_cache` is not itself guarded. The tracker thread invalidates it (`_invalidate_object_mask_cache`) while the FF bg thread may be reading it via `_feedforward_clean_cdn` → `_render_object_mask_cached`. Worst case a redundant re-render or a stale mask, not a crash. The deeper render IS lock-protected.
- **(medium) `scene_c2w_np` is captured on the main thread (:3125) but the model `means` used for the frustum cull is read on the bg thread under lock at a LATER time (:3282).** Between capture and use, the tracker may have applied a rigid transform (moving object Gaussians). The frustum cull/ICP uses whatever `means` exist at lock-acquire time against a pose snapshot from dispatch time — minor temporal skew, acknowledged-style in the ICP comments.

**God function**
- **(high) `_anysplat_bg_run` (:3209-3553, ~345 lines).** Does: depth re-filter, frustum cull, camera-away guard, ICP, per-window crop/inference/reproject loop, union voxel dedup (with a nested `_voxel_keep_idx` GPU helper + manual 64-bit Morton key packing), insert, and ~30 lines of logging. Multiple concerns; the dedup block alone (:3425-3515) is independently testable and should be extracted. Hard to reason about thread-safety because the lock is acquired/released in 3 separate inner scopes.
- **(medium) `_write_timing_report` (:795-933, ~140 lines)** and `_run_feedforward` (:2546-2734) are large but linear/readable.

**Duplicated logic**
- **(medium) `_oneshot_ff_due` (:966) is duplicated inline in `get_train_loss_dict` (:987-997).** The method exists but the caller re-implements the same `feedforward_oneshot_step > 0 and step >= ... and not done` predicate. Either call the method or delete it.
- **(medium) The `_scalar` local helper is redefined 4×** (:1892, :1959, :2789, :3115) — identical body each time. Should be a module/private helper.
- **(medium) The CDN-clean → cull-before-decode → component-select → insert pipeline is duplicated** between `_run_feedforward` (rgbd, :2592-2734) and `_run_feedforward_anysplat` (:3014-3163). The prefixes (clean, cull-reclean, component-select, debug) are near-identical; only the decode/insert tail differs.

**Dead config fields** (see §2): `feedforward_anchor_frame`, `feedforward_video_out`, `feedforward_video_fps` — declared, never read.

**Params threaded through many layers**
- **(low) `prerendered_obj_mask`** is threaded through `_run_feedforward` → `_feedforward_clean_cdn` → `_feedforward_cull_then_reclean_cdn` → `_run_feedforward_anysplat` → `_save_ff_debug_images` (5 layers) to enforce "render the object mask once." This is the intended "render-once" optimization but it makes every signature heavier; `_render_object_mask_cached` already provides a per-tick cache that arguably makes most of this redundant.
- **(low) `_anysplat_bg_run` takes a single `args: dict`** (:3147-3161) that is unpacked into 14 locals — an args-bag anti-pattern left over from when this was dispatched to a separate thread (the comment at :3143-3146 confirms it now runs inline, so the dict marshalling is vestigial).

**Leaky abstraction**
- **(medium) `_resolve_anysplat_context_image_paths` (:2949) and `_scene_c2w_for_frame` (:2967) are base methods that are recorded-only** (read `dataset.image_filenames` / `dataset.cameras`) with `TODO(phase-3-stage-D)` comments saying the live subclass must override — and it does (live:438, :458). A base method that is wrong for one of its two subclasses and relies on being overridden is a leaky default; the recorded-specific lookups arguably belong in the recorded subclass with an abstract stub in the base.
- **(low) `enable_cotracker_rigid_motion`** is read at :1989 to gate XFeat — a legacy name kept for checkpoint compat (documented), confusing on read.

**Confusing / misleading naming**
- `_force_viser_direct_push` (dead alias for `_push_viser_direct_transforms`).
- `_build_viser_direct_handles` (:1496) — name implies it builds splat handles, but it's now a "thin legacy stub that just records the D0 camera pose" (docstring). Misleading.
- `_dispatch_feedforward_async` spawns a thread, but `_run_feedforward_anysplat` then runs `_anysplat_bg_run` *inline* (already on that thread) — "async" vs "bg_run" naming makes the actual threading topology hard to trace.

**Swallowed exceptions (pervasive)**
- **(medium)** Bare `except Exception: pass` / log-and-continue appears ~30+ times (e.g. :391, :571, :648, :718, :732, :776, :1271, :1328, :2022, :2418, :2434, :3011, ...). Most are defensive cleanup/best-effort (acceptable). But several swallow real failures silently: `_capture_static_sequence_total` (:732 returns on any parse error), `_save_ff_debug_images` (:2434 swallows the whole dump), and the `_step_offset` warning path (:599) which downgrades a correctness-affecting failure to a log line. The FF CDN render failure (:2537) logs but then `_run_feedforward` proceeds to a `cdn is None` guard — OK. Volume makes genuine errors easy to miss in this LIVE-PATH module.

**Branches unreachable in live mode** (not strictly dead, but never exercised live)
- `_run_feedforward` rgbd path (:2567-2734) — live config default is `enable_feedforward_inpaint="anysplat_decode"`, so `_run_feedforward` returns early into `_run_feedforward_anysplat` (:2561). The entire rgbd decode loop, `_feedforward_delete_in_region`, and the `not feedforward_skip_delete` branch (:2675-2681) are unreachable on the default live path.
- `run_anysplat_subprocess` fallback (:3379) — only when `_anysplat_persistent_worker is None`; live eager-spawns the worker, so the slow fallback is effectively never hit live.
