# LIVE dynamic-gs — Runtime State Mutation Map

Scope: the **LIVE** dynamic phase (`ns-train dynamic-gs-live`). Traces every site that mutates
`gauss_params.*`, the four identity buffers (`object_flags`, `object_instance_ids`,
`sam3d_init_target_flags`, `inserted_flags`), or the scene Gaussian COUNT, plus the locks/threads
involved. Verified by reading the source, not inferred. File paths are absolute; sites are quoted by
`file:line` against the current tree (`feedforward_dev` branch).

Files traced:
- `dynamic_gs/dynamic_gs_pipeline_base.py` (base)
- `dynamic_gs/dynamic_gs_pipeline_live.py` (live subclass)
- `dynamic_gs/dynamic_gs_model.py` (model)
- `dynamic_gs/utils/anysplat_decode.py`, `dynamic_gs/utils/rgbd_decode.py` (decoders — produce tensors only, no model writes)
- `dynamic_gs/utils/viser_direct.py` (reader thread)

---

## 0. Threads in play (LIVE mode)

| Thread | Spawned where | What it does | Locks it takes |
|---|---|---|---|
| **Trainer/Tracker thread** | Nerfstudio `Trainer.train` loop → `DynamicGSTrainer.train_iteration` (`dynamic_gs_trainer.py:60`) → `pipeline.get_train_loss_dict` (`:97`) | Calls `get_train_loss_dict` (`dynamic_gs_pipeline_base.py:975`) → `_tracker_tick` (`dynamic_gs_pipeline_live.py:243`). Runs D0 bootstrap, XFeat motion, applies rigid transform, dispatches FF. This is the SINGLE thread that drives the per-tick loop. | `_model_lock` (RLock) around rigid transform / reseed / object-mask render |
| **FF bg thread** (`ff-recurring` / `ff-oneshot`) | `_dispatch_feedforward_async` (`base:2513`) spawns one `threading.Thread` per FF dispatch, daemon | Renders CDN, cull-before-decode (delete), runs AnySplat subprocess + reproject, voxel-dedup, **inserts** new Gaussians. Single-in-flight via `_anysplat_slot_lock`. | `_anysplat_slot_lock` (held whole call, acquired in `_dispatch_feedforward_async`, released in `_feedforward_threaded` finally `:2542`); `_model_lock` (RLock) re-entrantly around each cull/insert/render |
| **Viser render thread** | `ViserDirectScene.attach_model` (`viser_direct.py:437`) starts `_render_thread` | Event-driven `_render_once` (`:658`): per connected client calls `model.get_outputs(camera)` — a **READER** of `gauss_params`. | `self.model_lock` — which the pipeline **swaps** to be the SAME `_model_lock` RLock (`base:1480`), so it shares the exclusion zone |
| **Viser GUI threadpool** | `viser.ViserServer` internal | `on_click` (picker Done, Change-object) + `on_update` (camera move → `request_render`) callbacks. Only set flags / `Event`s. | none (only sets `self._reselect_requested`, `self._pending_selected_id`, `_selection_event`) |
| **stdin stop watcher** | `_start_stdin_stop_watcher` (`live:237`), daemon | Sets `self._live_stop_requested` / `self._reselect_requested`. | none |
| **NS viewer render thread** | NOT USED in live (Invariant #9: `vis="tensorboard"`, viser-direct only). `get_outputs_for_camera` lock plumbing (`model:708`) exists but is inert unless `--vis viewer`. | (`_render_lock_ctx` = `_model_lock` if attached) |

**Key design fact:** `_model_lock` is a re-entrant `threading.RLock` created in `base.__init__` (`:474`),
shared with the model via `attach_render_lock(self._viser_lock_ctx)` (`:519-520`) and with the viser
server via `self._viser_direct_server.model_lock = self._model_lock` (`:1480`). All three threads that
touch `gauss_params` therefore contend on ONE lock. `_viser_lock_ctx()` (`:1433-1450`) just returns it.

---

## 1. Annotated call chain (per LIVE tracker tick)

```
Trainer.train (NS) [trainer thread]
└─ DynamicGSTrainer.train_iteration (dynamic_gs_trainer.py:60)
   └─ pipeline.get_train_loss_dict(step)            base:975
      ├─ _tracker_tick(step)                          live:243
      │  ├─ _shm_sub.peek_latest()                    live:250   (lock-free SHM read)
      │  ├─ cameras_from_live_frame / _batch_from_live_frame  live:271-272
      │  ├─ _invalidate_object_mask_cache()           live:275 → base:1685 (sets _obj_mask_cache=None)
      │  ├─ [interactive] _tick_interactive_selection live:288 → base:1352
      │  │     └─ _reseed_tracked_object()            base:1141   ★WRITE object_flags + reference pose (under lock)
      │  ├─ if is_first: _bootstrap_d0(camera,batch)  live:480
      │  │     └─ _reseed_tracked_object(picked,...)  live:508 → base:1141   ★WRITE object_flags (under lock)
      │  ├─ else: _apply_motion_estimator(camera,batch) live:303 → base:2053
      │  │     ├─ _render_object_mask_cached(camera)  base:2070 → base:1655  (READ under lock)
      │  │     ├─ estimate_and_advance(...)            base:2102  (XFeat, no model write)
      │  │     └─ with _viser_lock_ctx():              base:2163
      │  │           model.apply_rigid_object_transform_from_reference(R,t)  base:2164 → model:987
      │  │                                             ★WRITE gauss_params["means"/"quats"][mask]  model:1006-1007
      │  ├─ self._ff_due_this_tick = _recurring_ff_due(...) live:314
      │  ├─ self._latest_tracker_frame = {...}         live:323   (publishes frame for bg thread)
      │  ├─ _build_viser_direct_handles / _push_* / _push_camera_feed  live:333-335 (reader pushes)
      │  ├─ _force_viewer_rerender()                   live:336   (NS viewer, inert in live)
      │  └─ _on_tracker_frame(camera,batch,cdn,is_first) live:338 → live:414
      │        └─ if _ff_due_this_tick:
      │              _dispatch_feedforward_async(_latest_tracker_frame,"recurring")  live:432 → base:2501
      │                 ├─ _anysplat_slot_lock.acquire(blocking=False)  base:2509
      │                 └─ Thread(_feedforward_threaded).start()        base:2513   →→→ FF BG THREAD
      └─ return {}, {"main_loss": zero}, {}            base:998-999
```

### FF bg thread chain (anysplat path — the live default `enable_feedforward_inpaint="anysplat_decode"`)

```
_feedforward_threaded(target_frame, mode)            base:2519   [ff bg thread; holds _anysplat_slot_lock]
├─ with _viser_lock_ctx(): target_frame["cdn"]=_compute_tick_cdn(cam,btch)  base:2535-2536  (READ render under lock)
└─ _run_feedforward(...)                              base:2539 → base:2546
   └─ _run_feedforward_anysplat(...)                  base:2561 (returns into method at ~base:2960)
      ├─ _render_object_mask_cached(camera)           base:3010 → base:1655   (READ under lock)
      ├─ _feedforward_clean_cdn(...)                   base:3019
      ├─ _feedforward_cull_then_reclean_cdn(...)       base:3053 → base:2854
      │     └─ with _viser_lock_ctx():                 base:2876
      │           _feedforward_cull_in_front_of_depth  base:2877 → base:2771
      │              └─ model.delete_gaussian_indices  base:2852 → model:1116  ★COUNT↓ + buffers re-sliced
      │     └─ _compute_tick_cdn(...) (re-render)       base:2885 (READ; NOT under lock — see HAZARD H2)
      ├─ select_top_n_components_filtered(...)          base:3091
      ├─ _resolve_anysplat_context_image_paths(...)     base:3102 → live:438  (dumps /dev/shm PNG)
      ├─ _anysplat_bg_run(args)                          base:3163 → base:3209
      │  ├─ depth_filter.filter_depth_torch(...)        base:3253  (live filters depth here, off tracker)
      │  ├─ with _viser_lock_ctx(): means_all_t=gauss_params["means"].detach()  base:3281-3282  (READ snapshot under lock)
      │  ├─ icp_refine_scene_c2w(...)                    base:3322  (anysplat_decode.py, no model write)
      │  ├─ worker.inference(...) / run_anysplat_subprocess  base:3374/3379  (subprocess; no model write)
      │  ├─ reproject_anysplat_to_scene(...)             base:3405  (anysplat_decode.py:659; returns tensors only)
      │  ├─ GPU voxel-dedup of decoded tensors           base:3447-3494
      │  └─ with _viser_lock_ctx():                       base:3497
      │        model.insert_inpaint_gaussians(...,instance_id=999)  base:3498 → model:1150
      │            ★COUNT↑ + ★WRITE object_flags/object_instance_ids/inserted_flags on new tail  model:1224-1226
      └─ (finally in _feedforward_threaded) _anysplat_slot_lock.release()  base:2542
```

The **rgbd_decode** path (`enable_feedforward_inpaint="rgbd_decode"`, NOT the live default) is
`_run_feedforward` `base:2566-2734`: same structure, deletes via `_feedforward_delete_in_region`
(`base:2739` → `model.delete_gaussian_indices`) / `_feedforward_cull_in_front_of_depth`, and inserts
via `model.insert_inpaint_gaussians` at `base:2700` (under `_viser_lock_ctx`, `base:2699`).

**Decoders write nothing to the model.** `anysplat_decode.reproject_anysplat_to_scene`
(`anysplat_decode.py:659`) and `rgbd_decode.decode_component_to_gaussians` (`rgbd_decode.py:244`) only
build/return tensors. All model mutation is in `dynamic_gs_model.py`, called from the pipeline under the lock.

---

## 2. Shared-state access table

Legend — Thread: **T**=trainer/tracker, **FF**=feedforward bg, **V**=viser render. Lock = `_model_lock`
(the shared RLock) unless noted. "Racy?" = can it tear/race a concurrent reader/writer given the locking.

| State | Site (file:line) | Thread | Lock held? | Racy? |
|---|---|---|---|---|
| `gauss_params["means"/"quats"][mask]` write (rigid transform) | `model:1006-1007` via `base:2163-2164` | T | YES (`_viser_lock_ctx` at `base:2163`) | No — in-place index write on existing tensor; V reads under same lock |
| `gauss_params["means"/"quats"][mask]` write (legacy `apply_rigid_object_transform`) | `model:937-938` | T (only called pre-D0 / preprocess) | **depends on caller** | This method is `@torch.no_grad` but takes NO lock itself; live calls the `_from_reference` variant which IS wrapped. See H4. |
| `gauss_params[*]` REALLOC (insert) | `model:1219` (`insert_inpaint_gaussians`) | FF | YES (`base:3497` anysplat / `base:2699` rgbd) | No — reassigns Parameter; readers gated by lock |
| `gauss_params[*]` REALLOC (delete) | `model:1137` (`delete_gaussian_indices`) | FF | YES (`base:2876` precull / `base:2690` per-comp) | No — but see H2 (re-render after cull is unlocked) |
| scene COUNT ↑ | `model:1228` return / concat `model:1218` | FF | YES | No |
| scene COUNT ↓ | `model:1131-1137` | FF | YES | No |
| `object_flags` write (D0/reseed) | `base:1165` (`copy_`) | T | YES (`base:1164`) | No |
| `object_flags` write (insert tail) | `model:1224` | FF | YES | No |
| `object_flags` re-slice (delete) | `model:1139` | FF | YES | No |
| `object_flags` resize/zero-fill | `model:1057`, `_resize_dynamic_buffers` | FF | YES (within insert/delete) | No |
| `object_instance_ids` write (insert tail) | `model:1225` | FF | YES | No |
| `object_instance_ids` re-slice (delete) | `model:1141` | FF | YES | No |
| `object_instance_ids` READ (D0 pick / mask / cull eligibility) | `live:485`, `base:1161`, `base:2763`, `base:2845`, `model:958` | T, FF | mixed | mostly under lock or pre-D0; cull eligibility read `base:2763/2845` runs under lock |
| `inserted_flags` write (insert tail) | `model:1226` | FF | YES | No |
| `inserted_flags` re-slice (delete) | `model:1142` | FF | YES | No |
| `sam3d_init_target_flags` | `model:1140` (re-slice on delete), `model:1054/1059` (resize) | FF | YES | No — never value-written at runtime (Invariant #8 holds; only sliced/resized to keep length == num_points) |
| `current_active_mask` re-slice | `model:1144` | FF | YES | No |
| `_reference_object_means/_quats` write | `model:982-983` (`capture_reference_object_pose`) | T | YES (called inside `base:1164` lock in reseed) | No |
| `_obj_mask_cache` write (None / render) | `base:1686` (invalidate), `base:1680` (set) | T (invalidate + set), FF (set) | set is under lock (`base:1677`); **invalidate at `live:275` / `base:1171/2168` is NOT locked** | **YES — see H1** |
| `_latest_tracker_frame` publish | `live:323` | T | NO | Low — single-writer (T), single-reader snapshot passed by value to FF at dispatch; but `batch`/`camera` tensors are shared (read-only) |
| `_anysplat_slot_lock` acquire/release | `base:2509` / `base:2542` | T acquires, FF releases | n/a | **Cross-thread acquire/release — see H3** |
| `model.get_outputs(camera)` READ | `viser_direct.py:690` | V | YES (`with self.model_lock`) | No |
| `model.info` READ (projected centers) | `base:2750`, `base:3282` etc | FF | partially | `base:3282` snapshot is locked; `extract_projected_centers_and_radii(model.info,...)` in `_feedforward_delete_in_region` (`base:2749`) reads `model.info` populated by the last render — stale-but-consistent |

---

## 3. Hazards

### H1 — `_obj_mask_cache` invalidated unlocked while FF bg thread may be reading/setting it  (race, medium)
- **Sites:** invalidate (no lock) at `dynamic_gs_pipeline_live.py:275`, `dynamic_gs_pipeline_base.py:1171`, `dynamic_gs_pipeline_base.py:2168`; populate (under lock) at `base:1680`; FF reads/sets it at `base:3010`/`base:1655`.
- **Thread interaction:** T sets `_obj_mask_cache=None` at tick start (`live:275`) and after a rigid transform (`base:2168`) with NO lock. The FF bg thread reads `_render_object_mask_cached` (`base:3010`) which checks `if self._obj_mask_cache is None:` and, on miss, takes the lock and renders. A T-thread `None`-store racing the FF-thread `is None` check is a benign data race on a Python attribute (worst case: FF re-renders a fresh mask, or uses a mask one tick stale). It cannot tear a tensor (the render itself is locked). **Not crash-class, but the "render once per tick, reuse everywhere" invariant the cache documents (`base:1655-1665`) is silently violated** whenever FF runs concurrently with a new tick — FF may cache a mask built against a different camera than the tick that dispatched it.
- **Recommendation:** treat `_obj_mask_cache` as owned by the FF dispatch snapshot, OR guard read+set+invalidate with a tiny dedicated lock. Low risk; document as accepted if left.

### H2 — CDN re-render after cull runs WITHOUT `_model_lock`  (race, medium→high)
- **Site:** `dynamic_gs_pipeline_base.py:2885` — `cdn_new = self._compute_tick_cdn(camera, batch)` inside `_feedforward_cull_then_reclean_cdn`, AFTER the `with _viser_lock_ctx():` block (`base:2876-2880`) has closed.
- **Thread interaction:** `_compute_tick_cdn` (`base:1727`) calls `_render_from_camera` (`base:1688`) which DOES take `_viser_lock_ctx()` internally (`base:1698`), so the render itself is locked. BUT it also calls `_render_object_mask_cached` (`base:1756`) and `_feedforward_clean_cdn`. The render is safe; the concern is only that the cull (delete, `base:2877`) and the re-render are two separate lock acquisitions, so another FF cannot interleave (single-in-flight via `_anysplat_slot_lock`) — only the **viser render thread** could mutate nothing (it's read-only). **Net: actually safe today** because (a) FF is single-in-flight, (b) the only other gauss_params writer is the trainer's rigid transform which is locked, and (c) `_render_from_camera` re-locks. Flag retained as **fragile**: the unlocked gap between the cull and re-render relies on `_anysplat_slot_lock` serialization + the rigid-transform lock; if a future change moves a gauss_params write off the lock, this tears.
- **Recommendation:** no action required for correctness today; add a comment that the cull→reclean sequence is safe ONLY because FF is single-in-flight and every gauss_params writer holds `_model_lock`.

### H3 — `_anysplat_slot_lock` acquired on trainer thread, released on FF bg thread  (lock, low — by design but exception-fragile)
- **Sites:** acquire `base:2509` (T thread, in `_dispatch_feedforward_async`); release `base:2542` (FF thread, `_feedforward_threaded` finally).
- **Detail:** `threading.Lock` allows release from a different thread (unlike some RLock semantics), so this is legal. The `finally` at `base:2540-2544` swallows a double-release `RuntimeError`. The risk: if `_dispatch_feedforward_async` acquires the lock (`base:2509`) then `threading.Thread(...).start()` (`base:2513`) **raises** before the thread body runs (e.g. OS thread-creation failure), the lock is never released and ALL future FF dispatches no-op forever (`base:2510` "previous FF still in flight"). Tracking continues but feedforward dies silently.
- **Recommendation:** wrap the `Thread(...).start()` in try/except that releases `_anysplat_slot_lock` on failure. Also `_cleanup_anysplat_bg` (`base:650`) blocks up to 60 s at exit on this lock — a wedged FF call delays shutdown.

### H4 — `apply_rigid_object_transform` (legacy variant) takes no lock  (lock, low)
- **Site:** `dynamic_gs_model.py:922-939` writes `gauss_params["means"/"quats"]` with `@torch.no_grad` but NO `_model_lock`. The LIVE per-tick path uses `apply_rigid_object_transform_from_reference` (`model:987`), which is wrapped by the caller at `base:2163`. The legacy `apply_rigid_object_transform` has no live caller on the tick path (grep shows it is unused in the live tick), but it is a public method.
- **Recommendation:** if confirmed dead on the live/recorded paths, it is a **purge candidate** (the `_from_reference` variant supersedes it). If kept, it must not be called without holding `_model_lock` or it races the viser render thread.

### H5 — per-tick & per-FF tensor allocations / Parameter realloc churn  (allocation, medium)
- **Sites:** every FF insert (`model:1218-1221`) does `torch.cat([old.detach(), new])` for ALL SIX gauss_params, allocating the full (N+M)×… tensors, then `_refresh_gaussian_optimizers` (`model:1063`) clears optimizer state and **re-registers the means grad hook** (`model:1078`). Every FF delete (`model:1136-1137`) reallocates all six via boolean index `[keep]`. With `feedforward_recurring_every_n_ticks=10` and accumulating inserts (CLAUDE.md notes 459k→1.29M on real-1200p), each FF call copies the entire (growing) param set twice (cull + insert) on the FF thread under the lock — directly contends with the tracker render. This is the documented scene-bloat / GPU-contention cost.
- **Recommendation:** out of scope for a pure race audit, but the per-call full-tensor `cat` + `[keep]` realloc is the dominant per-FF allocation; a periodic dynamic-phase purge (already TODO in CLAUDE.md `[[static-phase-opacity-purge-todo]]`) bounds it. Never drop `object_instance_ids==d0_id`.

### H6 — `_latest_tracker_frame` dict shared T→FF without lock  (race, low)
- **Site:** published at `live:323` (T), read by FF at dispatch (`base:2531-2536` mutates `target_frame["cdn"]`). T overwrites `self._latest_tracker_frame` every tick; FF was handed the dict reference at dispatch time (`base:432` passes `self._latest_tracker_frame`). Because the dispatch passes the CURRENT reference and FF is single-in-flight, the in-flight FF holds its own reference; T rebinding `self._latest_tracker_frame` to a NEW dict next tick does not mutate the in-flight one. **The shared objects are `camera`/`batch` tensors, which FF only reads.** Safe today. The one mutation FF makes (`target_frame["cdn"]=...` at `base:2536`) is on the dict FF owns. Low risk.
- **Recommendation:** none; document that `_latest_tracker_frame` is a per-tick immutable snapshot consumed by at most one FF thread.

### H7 — viser GUI/`request_render` from callback threads vs teardown  (lifecycle, low — already guarded)
- **Sites:** `request_render` (`viser_direct.py:611`) and FF insert push `_viser_direct_register_ff_insert` (`base:1571`) both check `is_closing` / `_stop_event` (`base:1585`, `viser_direct.py:620`) before submitting, to avoid "cannot schedule new futures after shutdown". `add_ff_insert_chunk`/`flush_pending_ff`/`refresh_static_handle`/`push_tracker_transform` are all legacy NO-OP stubs (`viser_direct.py:506-528`). Guarded correctly.
- **Recommendation:** none. Note the legacy stubs are dead-but-harmless call sites — purge candidates if simplifying the viser interface.

### H8 — interactive picker BLOCKS the trainer thread (and the FF dispatch with it)  (lifecycle, low; default OFF)
- **Site:** `_tick_interactive_selection`→`_wait_for_selection` (`base:1399`) blocks the trainer thread up to `object_selection_timeout_s` (120 s). During the block no ticks advance, but an in-flight FF bg thread keeps running and can `insert_inpaint_gaussians` under the lock — fine (the picker doesn't hold `_model_lock` while waiting; it only held it during the `_reseed_tracked_object` `copy_` at `base:1164`). Default `interactive_object_selection=False`, so inert in normal runs.
- **Recommendation:** none for correctness.

### H9 — unreachable / dead branches on the live path  (deadbranch, info)
- `_force_viewer_rerender` (`base:1614`) — NS viewer only; inert in live (Invariant #9, `vis="tensorboard"`, no `viewer_state.ready`). Pure no-op every tick.
- `get_outputs_for_camera` opacity-hide branch (`model:726-739`) — only reached via NS viewer object-selector; live uses viser-direct `get_outputs` (`viser_direct.py:690`), never `get_outputs_for_camera`. The `_render_lock_ctx` plumbing there is inert in live.
- `_mask_means_grad` (`model:825`) returns `zeros_like(grad)` whenever `enable_cotracker_rigid_motion` (always True in live) — so means never move via gradient (Invariant #4). The hook fires only if a backward runs; the trainer fast path (`dynamic_gs_trainer.py:60`) skips backward, so the hook is effectively never invoked in live. Re-registered on every insert/delete (`model:1078`) regardless.
- `apply_rigid_object_transform` legacy variant (`model:922`) — see H4; appears to have no live caller.
- `push_tracker_transform`/`add_ff_insert_chunk`/`flush_pending_ff`/`refresh_static_handle`/`maybe_flush_ff_handle` (`viser_direct.py:506-528`) — all NO-OP stubs.

---

## 4. Summary for the purge

- **The single load-bearing concurrency primitive is `_model_lock` (RLock)**, shared across trainer, FF bg, and viser render threads via `attach_render_lock` (`base:519`) and `viser_direct_server.model_lock = _model_lock` (`base:1480`). Do NOT remove either hookup — it is the only thing preventing the viser render thread from reading a torn `gauss_params` mid-insert/resize.
- **`_anysplat_slot_lock` (plain Lock)** is the single-in-flight FF guard; cross-thread acquire(T)/release(FF) is intentional but exception-fragile (H3).
- **All four identity buffers** are only value-written under `_model_lock`: `object_flags` at `base:1165` (T) + `model:1224` (FF), `object_instance_ids`/`inserted_flags` at `model:1225-1226` (FF). `sam3d_init_target_flags` is never value-written at runtime (Invariant #8 confirmed — only sliced/resized to track `num_points`). All re-sliced together on delete (`model:1139-1144`) and resized together on insert (`_resize_dynamic_buffers`, `model:1030`).
- **Decoders mutate nothing** — `anysplat_decode`/`rgbd_decode` return tensors only; safe to refactor without touching model state.
- **Highest-value fixes:** H3 (lock leak on thread-spawn failure → FF silently dies), then H1 (object-mask cache cross-thread invalidation breaks the "render once" guarantee). H2/H4 are latent (safe today, fragile under future edits). H9 lists concrete dead code purge candidates.
