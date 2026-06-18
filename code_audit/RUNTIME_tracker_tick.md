# RUNTIME trace — LIVE dynamic-gs TRACKER path

Scope: `LiveDynamicGSPipeline._tracker_tick` → XFeat estimate → Kabsch/RANSAC →
`apply_rigid_object_transform_from_reference`. Verified by reading the source
(not inferred). File paths are absolute; line numbers are from the state of the
tree at audit time and will drift — anchor on symbol names.

Repo root: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`

---

## Threading model (ground truth)

There are **three threads** that touch model state in the live runtime:

| Thread | Started where | What it does on the tracker path |
|---|---|---|
| **TRAINER thread** (the nerfstudio `Trainer.train()` loop — runs on the process **main thread**) | `NoSaveTrainer.train_iteration` (`dynamic_gs/dynamic_gs_trainer.py:60`) calls `pipeline.get_train_loss_dict` | Runs the ENTIRE `_tracker_tick`: peek SHM → XFeat estimate → `apply_rigid_object_transform_from_reference` → viser pushes → `_on_tracker_frame`. The whole tracker write path is on this single thread. |
| **FF bg thread** (`ff-recurring` / `ff-oneshot`, daemon) | `_dispatch_feedforward_async` → `threading.Thread(target=_feedforward_threaded …)` (`dynamic_gs_pipeline_base.py:2513`) | Renders CDN, culls + **inserts** Gaussians (`insert_inpaint_gaussians` reallocates every `gauss_params.*` Parameter and resizes the 4 identity buffers). This is the ONLY writer that changes `N` (Gaussian count) during the dynamic phase. |
| **VISER render thread** (`daemon`, in `ViserDirectScene`) | `attach_model` → `threading.Thread(target=self._render_loop …)` (`dynamic_gs/utils/viser_direct.py:437`) | READ-ONLY: `model.get_outputs(camera)` under `model_lock` (`viser_direct.py:689`). The legacy `push_tracker_transform` / `add_ff_insert_chunk` are now **stubs** (`viser_direct.py:506,512`) — render thread never mutates model. |

Plus a daemon **stdin-stop watcher** (`_start_stdin_stop_watcher`,
`dynamic_gs_pipeline_live.py:237`) which only flips `_live_stop_requested` /
`_reselect_requested` (plain bool writes, no model touch), and viser's own GUI
thread-pool callbacks (`_on_done`, `_on_change`) which only set flags.

**Key lock:** `self._model_lock = threading.RLock()`
(`dynamic_gs_pipeline_base.py:474`). The SAME object is shared three ways:
- the model via `attach_render_lock(self._viser_lock_ctx)` →
  `model._render_lock_ctx` (`dynamic_gs_pipeline_base.py:519-520`, model side
  `dynamic_gs_model.py:741-747`);
- the viser server via `self._viser_direct_server.model_lock = self._model_lock`
  (`dynamic_gs_pipeline_base.py:1480`) — replaces viser's own RLock so all three
  threads contend on ONE lock.

Single-in-flight FF guard: `self._anysplat_slot_lock = threading.Lock()`
(`dynamic_gs_pipeline_base.py:465`), `acquire(blocking=False)` in
`_dispatch_feedforward_async` (`:2509`), released in `_feedforward_threaded`'s
`finally` (`:2542`).

`_ff_due_this_tick` decide-once flag: set EXACTLY ONCE per tick in
`_tracker_tick` (`dynamic_gs_pipeline_live.py:314`) and only *read* in
`_on_tracker_frame` (`:428`). Both run on the trainer thread, same tick, so no
cross-thread race on it.

---

## Annotated call chain (TRAINER thread unless noted)

```
NoSaveTrainer.train_iteration(step)                       dynamic_gs_trainer.py:60
  └─ pipeline.get_train_loss_dict(step)                   dynamic_gs_pipeline_base.py:975
       ├─ self._dynamic_step_counter += 1                 :984
       ├─ _tracker_tick(step)   [LIVE override]           dynamic_gs_pipeline_live.py:243
       │    ├─ guard: _live_stop_requested / _shm_sub None :245-248
       │    ├─ latest = self._shm_sub.peek_latest()       :250    (lock-free SHM read)
       │    ├─ stamp dedup + sim-clock-reset re-arm        :253-266 (WRITE _last_processed_stamp_sec)
       │    ├─ camera = cameras_from_live_frame(...)       :271
       │    ├─ batch  = _batch_from_live_frame(...)        :272    (RAW depth; H2D copies)
       │    ├─ _invalidate_object_mask_cache()             :275    (WRITE _obj_mask_cache=None)
       │    ├─ frame_idx=_next_live_frame_counter; ++      :277-278
       │    ├─ is_first = not _d0_completed                :279
       │    ├─ [interactive picker — usually OFF]          :287-288
       │    ├─ if is_first: _bootstrap_d0(camera,batch)    :295  ── D0 path (see below)
       │    │      else:    _apply_motion_estimator(...)   :303  ── DN path (see below)
       │    ├─ self._ff_due_this_tick = _recurring_ff_due(tick+1, is_first)  :314 (WRITE, decide-once)
       │    ├─ cdn = None                                  :317
       │    ├─ self._latest_live_rgb_bgr = latest.rgb_bgr  :321   (WRITE, read by FF thread)
       │    ├─ self._latest_tracker_frame = {...}          :323   (WRITE dict, read by FF thread)
       │    ├─ _global_frame_counter++; _tracker_tick_count++ :330-331
       │    ├─ _build_viser_direct_handles(camera)         :333   (legacy stub-ish; sets initial cam)
       │    ├─ _push_viser_direct_transforms()             :334   (no-op: push_tracker_transform stub)
       │    ├─ _push_viser_camera_feed(camera, batch)      :335   (WRITE srv feed buffers)
       │    ├─ _force_viewer_rerender()                    :336   (NS-viewer only; no-op w/ viser-direct)
       │    └─ _on_tracker_frame(camera, batch, None, is_first)  dynamic_gs_pipeline_live.py:414
       │         ├─ if is_first: _capture_static_sequence_total()  :425
       │         ├─ if not self._ff_due_this_tick: return  :428   (READ decide-once flag)
       │         └─ _dispatch_feedforward_async(_latest_tracker_frame,"recurring")  :432
       │              └─ spawns FF bg thread (see threading table) — NON-BLOCKING
       └─ return {}, {"main_loss": zero}, {}               :999
```

### DN path: `_apply_motion_estimator` (dynamic_gs_pipeline_base.py:2053)

```
_apply_motion_estimator(camera, batch)                    :2053  [TRAINER thread]
  ├─ if _motion_estimator is None: return                 :2058
  ├─ current_live_rgb = _build_tracking_rgb(batch)        :2061
  ├─ current_object_mask = _render_object_mask_cached(camera)   :2070
  │     └─ if _obj_mask_cache is None:                     :1666
  │          ├─ lock.acquire()  [_model_lock]             :1677   ◀── LOCK HELD
  │          ├─ self._obj_mask_cache = model.render_object_mask(camera)  :1680 (READ means/quats/...)
  │          └─ lock.release()  (finally)                 :1682   ◀── LOCK RELEASED
  ├─ [optional bbox crop of rgb/depth/cam/mask]           :2082-2100
  ├─ motion_estimate = _motion_estimator.estimate_and_advance(...)  :2102
  │     └─ XFeat extract + LighterGlue + RANSAC/Kabsch    utils/xfeat_motion.py:578
  │        — operates ONLY on its own anchors + passed tensors; does NOT touch self.model
  ├─ if not success: log + return                         :2142-2161
  ├─ with self._viser_lock_ctx():                         :2163   ◀── LOCK HELD
  │     moved = model.apply_rigid_object_transform_from_reference(R, t)  :2164
  │        └─ WRITES gauss_params["means"][mask], ["quats"][mask]  dynamic_gs_model.py:1006-1007
  │   (lock released at end of with)                                ◀── LOCK RELEASED
  ├─ _invalidate_object_mask_cache()                      :2168   (WRITE _obj_mask_cache=None)
  ├─ self._last_motion_estimate = motion_estimate         :2169   (WRITE, read by viser push)
  ├─ [DGS_TRACK_TRAJ_LOG csv append — opt-in]             :2173-2188
  └─ srv.request_render() (wake viser thread)             :2196
```

`apply_rigid_object_transform_from_reference` (dynamic_gs_model.py:987):
- READ `_tracked_object_mask()` = `object_instance_ids == _d0_tracked_instance_id`
  (`:988`, `:957-963`);
- count check vs `_reference_object_means/_quats` (`:994`);
- index-assign `gauss_params["means"][object_mask]` and `["quats"][object_mask]`
  (`:1006-1007`) — IN-PLACE on the existing Parameter tensors, does NOT change `N`.

### D0 path: `_bootstrap_d0` → `_reseed_tracked_object` (first tick / picker)

```
_bootstrap_d0(camera, batch)                              dynamic_gs_pipeline_live.py:480 (@torch.no_grad)
  ├─ picked = _pick_d0_object(camera, model.object_instance_ids)  :486 (READ ids + means; projection)
  ├─ if picked==0: defer (no D0) return                   :487-499
  └─ _reseed_tracked_object(picked, camera, batch)        :508 → base :1141
        ├─ _motion_estimator=None; _last_motion_estimate=None  :1156-1157
        ├─ _d0_selected_instance_id = new_id              :1159
        ├─ with self._viser_lock_ctx():                   :1164   ◀── LOCK HELD
        │     model.object_flags.copy_((ids==new_id)...)  :1165   (WRITE object_flags)
        │     model.capture_reference_object_pose(new_id) :1167   (WRITE _d0_tracked_instance_id,
        │                                                          _reference_object_means/_quats; READ means/quats)
        ├─ _invalidate_object_mask_cache()                :1171
        ├─ obj_mask = _render_object_mask_cached(camera)  :1175   (re-locks internally)
        ├─ _initialize_motion_estimator(rgb,depth,cam,mask) :1184 (builds XFeatMotionEstimator; no model write)
        └─ _reset_d0_guard()  [LIVE]                       :1189 → live :510  (WRITE _tracker_tick_count=0, _d0_completed=True)
```

---

## Shared-state access table

Legend — Thread: **T**=trainer (tracker), **F**=FF bg, **V**=viser render.
Lock = `_model_lock` (the shared RLock) unless noted.

| Shared state | Site (file:line) | Thread | Lock held? | Racy? |
|---|---|---|---|---|
| `gauss_params["means"][mask]=` (write) | dynamic_gs_model.py:1006 (`apply_rigid_..._from_reference`) | T | **YES** (held by caller `:2163`) | No — in-place, count unchanged, lock vs V and vs F insert |
| `gauss_params["quats"][mask]=` (write) | dynamic_gs_model.py:1007 | T | YES (`:2163`) | No |
| `gauss_params["means"] = Parameter(cat(...))` (REALLOC, N grows) | dynamic_gs_model.py:1219 (`insert_inpaint_gaussians`) | F | **YES** (held by FF bg via `_viser_lock_ctx` around insert, base `:2700/3498`) | No, **iff** every reader holds the lock (see hazards) |
| `object_flags` write (copy_) | dynamic_gs_pipeline_base.py:1165 (`_reseed_tracked_object`) | T | YES (`:1164`) | No |
| `object_flags[old:]=1` (resize+write) | dynamic_gs_model.py:1224 (insert) | F | YES | No |
| `object_instance_ids` READ (D0 pick / reseed) | dynamic_gs_pipeline_live.py:486; base:1161,1200,1210 | T | mixed (pick: NO; reseed body: YES) | Low — D0 pick runs before any FF can fire (is_first), buffer stable |
| `object_instance_ids[old:]=id` (resize+write) | dynamic_gs_model.py:1225 (insert) | F | YES | No |
| `inserted_flags[old:]=1` (resize+write) | dynamic_gs_model.py:1226 (insert) | F | YES | No |
| `sam3d_init_target_flags` | only resized in `_resize_dynamic_buffers` (model.py:589, 1059); value never written at runtime | F (resize only) | YES | No — placeholder, all-zeros expected (Invariant #8) |
| `_d0_tracked_instance_id`, `_reference_object_means/_quats` (write) | dynamic_gs_model.py:975,982,983 (`capture_reference_object_pose`) | T | YES (`:1164`) | No |
| `_reference_object_means` READ | dynamic_gs_model.py:1001; base:2174 (traj log) | T | partial | No — only T reads/writes it |
| `model.get_outputs(camera)` READ all gauss_params | viser_direct.py:690 (`_render_once`) | V | **YES** (`with self.model_lock` :689) | No |
| `model.render_object_mask(camera)` READ gauss_params + `_tracked_object_mask` | dynamic_gs_pipeline_base.py:1680 | T | YES (`:1677`) | No |
| `_obj_mask_cache` (read/write) | base:1666,1680,1686 | T (also F via `_render_object_mask_cached` in `_feedforward_clean_cdn`) | render path locks `_model_lock`; the cache pointer R/W itself unlocked | **LOW race** — see Hazard H4 |
| `_latest_tracker_frame` (dict write) | dynamic_gs_pipeline_live.py:323 | T | none | See Hazard H1 (publish→consume handoff) |
| `_latest_tracker_frame["cdn"]=` (write) | dynamic_gs_pipeline_base.py:2536 | F | none on the dict slot | Single-writer (F) after handoff; OK |
| `_latest_live_rgb_bgr` (write/read) | live:321 (T write) / live:446 (F read in `_resolve_anysplat_context_image_paths`) | T,F | none | Low — pointer swap; FF reads a slightly newer frame at worst |
| `_last_motion_estimate` (write) | base:2169 (T) / read base:1527 (`_push_viser_direct_transforms`, T) | T | none | No — same thread |
| `_ff_due_this_tick` (write live:314 / read live:428) | T | none | No — same thread, same tick |
| `_anysplat_slot_lock` | acquire base:2509 (T) / release base:2542 (F) | T→F | n/a | Cross-thread acquire/release by design (see Hazard H3) |
| `_last_processed_stamp_sec` | live:263,266 (write) | T | none | No — single thread |
| `_live_stop_requested` | stdin thread write live:230 / T read live:245 | stdin,T | none | Benign bool flag |

---

## Hazards

### H1 — `_latest_tracker_frame` published BEFORE the FF dispatch reads it; tracker keeps mutating the model the snapshot points into
- **Severity: medium**  | kind: race
- **Where:** `_tracker_tick` writes `self._latest_tracker_frame = {...}`
  (`dynamic_gs_pipeline_live.py:323`); `_on_tracker_frame` passes that SAME dict
  into `_dispatch_feedforward_async` (`:432`) which runs `_feedforward_threaded`
  on the FF bg thread. The dict captures `camera`/`batch` (immutable per tick —
  fine), but the FF then renders the CDN and culls/inserts against
  `self.model`, whose `gauss_params` the TRAINER thread continues to mutate via
  `apply_rigid_object_transform_from_reference` on every subsequent tick.
- **Why it is mostly contained:** both the FF model writes/reads
  (`_compute_tick_cdn` at `:2536` under `_viser_lock_ctx`, the cull and
  `insert_inpaint_gaussians`) and the tracker write (`:2163`) take the SAME
  `_model_lock`, so there is no torn-tensor corruption. The remaining issue is
  *semantic staleness*, not memory corruption: the FF decodes against a model
  pose a few ticks newer than the frame it snapshotted. This is the documented
  "_4_rendered captured a few ticks after _6_raw_mask" caveat in CLAUDE.md and
  is accepted.
- **Recommendation:** none required for the purge — do NOT "fix" by holding the
  lock across the whole FF (that would serialize the tracker behind the ~270 ms
  AnySplat call, the exact thing the off-thread design avoids). Leave as-is;
  just don't assume `_latest_tracker_frame` is a deep snapshot of model state.

### H2 — Reader of `gauss_params` that does NOT take `_model_lock` would tear against an FF insert
- **Severity: high (latent)** | kind: race
- **Where:** the FF insert reallocates every Parameter and grows `N`
  (`insert_inpaint_gaussians`, `dynamic_gs_model.py:1219-1228`) on the FF bg
  thread. Correctness of the WHOLE design depends on every concurrent reader of
  `means`/`quats`/buffers holding `_model_lock`. Verified holders: viser render
  (`viser_direct.py:689`), `_render_object_mask_cached` (`:1677`),
  `_render_from_camera` (`:1698`), the rigid-transform write (`:2163`),
  `_compute_tick_cdn` via `_feedforward_threaded` (`:2535`).
- **The gap:** `_pick_d0_object` (`dynamic_gs_pipeline_live.py:381-398`) reads
  `self.model.means` / `object_instance_ids` WITHOUT the lock. This is safe
  TODAY only because it runs while `is_first` is True and the FF gate
  `_recurring_ff_due` returns False when `is_first` (`base:950-951`), so no FF
  thread can be inserting during D0. **If the purge ever moves D0 pick to a
  point where FF can be in flight (e.g. mid-run reseed via picker while a prior
  FF is still draining), this read tears.** Note `_reseed_tracked_object`'s
  `object_flags.copy_` IS locked (`:1164`) but the `_pick_d0_object` call that
  precedes a live reseed is not.
- **Recommendation:** wrap the `means`/`ids` reads in `_pick_d0_object` in
  `with self._viser_lock_ctx():` (cheap — it's one projection). Cite this
  invariant explicitly so the purge doesn't remove the `is_first` FF gate that
  currently makes it safe.

### H3 — `_anysplat_slot_lock` acquired on the trainer thread, released on the FF bg thread; not released if the bg thread never starts
- **Severity: medium** | kind: lock / leak
- **Where:** `_dispatch_feedforward_async` does
  `self._anysplat_slot_lock.acquire(blocking=False)` (`base:2509`) then
  `threading.Thread(...).start()` (`:2513`). The lock is released only inside
  `_feedforward_threaded`'s `finally` (`:2542`), which runs on the bg thread.
- **The gap:** if `Thread(...).start()` raises (e.g. OS thread-limit /
  `RuntimeError: can't start new thread`) AFTER the `acquire` succeeded, the lock
  is held forever → every future FF dispatch logs "previous FF still in flight"
  and FF silently stops for the rest of the run. Acquire on one thread / release
  on another is legal for `threading.Lock` (not owner-checked), so the mechanism
  itself is fine; the unguarded `start()` is the hole.
- **Recommendation:** wrap `start()` in try/except and release the slot lock on
  failure before returning False. Low-probability but a single-point silent
  death of the whole FF subsystem.

### H4 — `_obj_mask_cache` pointer read/written without a lock; cross-thread use by the FF thread
- **Severity: low** | kind: race
- **Where:** `_render_object_mask_cached` reads `self._obj_mask_cache`
  (`base:1666`) and writes it (`:1680`); `_invalidate_object_mask_cache` sets it
  `None` (`:1686`). The render itself is locked, but the cache-pointer R/W is
  not. The trainer thread invalidates it at tick start (`live:275`) and after
  the rigid transform (`:2168`); the FF bg thread also calls
  `_render_object_mask_cached` (via `_feedforward_clean_cdn` /
  `_feedforward_cull_then_reclean_cdn`).
- **Why low:** Python attribute assignment is atomic (GIL); worst case the FF
  thread renders the mask once extra or reuses a one-tick-stale mask. Mask is a
  coarse +2%-scaled, dilated region filter so a one-tick lag is within tolerance
  (matches the documented `xfeat_object_mask_cache_ticks` rationale). No
  corruption (each actual render is under `_model_lock`).
- **Recommendation:** none for the purge. If touched, document that the cache is
  intentionally lock-free-pointer + lock-protected-render.

### H5 — Per-tick GPU allocations on the tracker critical path
- **Severity: low** | kind: allocation
- **Where:** `_batch_from_live_frame` (`live:530-543`) does H2D copies of RGB
  (`/255`), depth, mask EVERY tick; `_render_object_mask_cached` rasterizes
  ~45k object Gaussians every tick the cache is cold (`render_object_mask`,
  model:2179); the optional bbox-crop allocates cropped tensors (`base:2087`).
- **Why noted, not flagged high:** these are inherent to the per-frame tracker
  and already measured/tuned (CLAUDE.md timing notes). No leak — tensors are
  freed each tick. The `DGS_TRACK_TRAJ_LOG` path (`base:2173-2188`) opens+writes
  a file per successful tick when the env var is set — opt-in only.
- **Recommendation:** none. Do not micro-optimize during the purge.

### H6 — `_force_viewer_rerender` iterates NS-viewer state machines — dead branch under the documented config
- **Severity: low** | kind: deadbranch
- **Where:** `_tracker_tick` calls `_force_viewer_rerender()` (`live:336`),
  which early-returns when `self._trainer.viewer_state` is None
  (`base:1624-1629`). Per Invariant #9 the live runtime uses `vis="tensorboard"`
  / viser-direct and never `--vis viewer`, so `viewer_state` is None and this is
  a no-op every tick.
- **Recommendation:** safe to KEEP (cheap guard, supports `--vis viewer+...` for
  static-ckpt debugging). If the purge wants to drop NS-viewer support entirely,
  this method + `get_training_callbacks`' `_trainer` stash could go — but it does
  not break any invariant either way. Flagging only so it isn't mistaken for
  live-path logic.

### H7 — `request_render()` from the trainer thread onto a possibly-closing viser server
- **Severity: low** | kind: lifecycle
- **Where:** `_apply_motion_estimator` calls `srv.request_render()` (`base:2196`)
  and `_push_viser_camera_feed` writes feed buffers (`base:1560,1567`) each tick.
  At shutdown the FF-insert push path guards with `is_closing`
  (`base:1585`), but the per-tick `request_render`/`update_camera_feed`/
  `update_tracked_camera` calls here are wrapped in try/except (feed) or bare
  (`:2196`). `request_render` just sets an Event (`viser_direct.py:611`), so a
  late call is harmless; `set_background_image` on a closing client is caught in
  `_render_once`.
- **Recommendation:** none. The teardown noise is the known cosmetic
  "cannot schedule new futures after shutdown" already documented.

---

## Summary of invariant compliance (do-not-break list for the purge)

- **Invariant #4 (dynamic LRs = 0):** enforced by `_ZERO_LR_OPTIMIZERS`/the
  trainer fast path (`dynamic_gs_trainer.py:69`). The tracker mutates means/quats
  via direct index-assignment under `@torch.no_grad`, NOT via the optimizer — do
  not reintroduce a backward/step on this path.
- **Invariant #8 (identity buffers):** `object_flags` written by D0 reseed
  (T, locked) and by FF insert (F, locked); `object_instance_ids`/`inserted_flags`
  written by FF insert (F, locked); `sam3d_init_target_flags` only ever resized,
  never value-written — all-zeros is correct. The tracker mask is
  `object_instance_ids == _d0_tracked_instance_id` only (model:957-963), so FF
  inserts (id=999) are never moved by the rigid transform — keep this.
- **Invariant #9 (viser-direct, model lock):** the shared `_model_lock` is the
  single correctness mechanism for the 3-thread design. Every gauss_params
  read/write on a non-trainer thread, and every count-changing write, MUST hold
  it. The render thread reads under it (`viser_direct.py:689`); the FF inserts
  under it; the rigid transform under it. Do not split this into per-subsystem
  locks (the comment at base:466-474 explains why it lives on the pipeline).
