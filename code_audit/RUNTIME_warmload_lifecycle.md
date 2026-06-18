# Runtime trace: LIVE warm-load + worker/thread lifecycle

Scope: the STATE-LOAD path into `dynamic-gs-live` — `LiveDynamicGSPipeline.__init__`
→ `DynamicGSPipelineBase.__init__` → warm-cache load of `static_state.pt` into the
live model, plus all worker/thread/SHM spawns, their shutdown, and what leaks.
Read against CLAUDE.md invariants #4 (`_ZERO_LR_OPTIMIZERS` / dynamic LRs=0),
#8 (the 4 identity buffers), #9 (viser-direct only, never NS viewer; the shared
`_model_lock`). No source was edited.

All file:line references are against the code as read 2026-06-17.

---

## (a) Annotated call chain

### Construction order (single thread — the trainer/main thread)

```
LiveDynamicGSPipeline.__init__                         dynamic_gs_pipeline_live.py:89
  set live-only state (_shm_sub=None, _d0_completed=False, ...)        :101-107
  super().__init__(...)  ──────────────────────────────────────────►  :109
    DynamicGSPipelineBase.__init__                     dynamic_gs_pipeline_base.py:369
      timing_ledger.reset(data)                                        :388-391
      init shared-state attrs (_latest_tracker_frame, counters, ...)   :393-458
      torch.backends.cudnn.benchmark = False  (dynamic-phase global)   :421
      self._anysplat_slot_lock = threading.Lock()                      :465  (single-in-flight FF gate)
      self._model_lock = threading.RLock()                             :474  (THE shared model lock, invariant #9)
      atexit.register x8  (viser, anysplat worker, anysplat bg,        :476-483
                           ipc file, video writer, final snapshot,
                           timing report, torch profile) — LIFO
      self.current_phase = "dynamic"  (trainer fast-path flag)         :492
      self._filter_depth_at_ff = False  (live overrides to True later) :499
      super().__init__()  → VanillaPipeline                            :502
        builds DynamicGSDataManager + DynamicGSModel (COLD, SfM seed)
          DynamicGSModel.populate_modules                  dynamic_gs_model.py:619
            register_buffer object_flags / current_active_mask /       :626-662
              sam3d_init_target_flags / object_instance_ids /
              inserted_flags / change_mask_image  (the 4 identity
              buffers + 2 non-persistent, invariant #8)
            self.strategy = NoRefineStrategy()                         :663
            gauss_params["means"].register_hook(_mask_means_grad)      :665  (hook #1, on the COLD means)
            _apply_phase_trainability()                                :666  (phase still "static" here)
            build ViewerDropdown "Visualize Object"                    :671
      model.attach_render_lock(self._viser_lock_ctx)                   :519-520  (model render lock == pipeline _model_lock)
      self._load_warm_cache_or_die()  ───────────────────────────────►:526
        resolve cache_path (static_state.pt, legacy fallback)          :542-564
        load_post_fusion_state(model, cache_path, device)   post_fusion_cache.py:92
          torch.load(cache_path, map_location=device)                  :110  (reads N_post state_dict to GPU)
          reallocate gauss_params[means/features/opac/scales/quats]    :121-132  (Parameter realloc at N_post)
          model.load_state_dict(state_dict, strict=False)   dynamic_gs_model.py:568
            if object_flags.shape[0] != N_post:  rebuild ALL 4 buffers :577-606  (+ current_active_mask) at N_post
            inject missing buffer keys as zeros                        :608-615
            super().load_state_dict(...)                               :617  (copies gauss_params + buffers)
          gauss_params["means"].register_hook(_mask_means_grad)        :143  (hook #2, RE-BIND on the NEW means Param)
        model._step_offset = 10_000  (bypass SH/res schedules)         :593
      if config.enable_viser_direct: self._setup_viser_direct()        :529-530
        ViserDirectScene(port=8081)  → starts viser server + render thread  :1473
        server.model_lock = self._model_lock  (SWAP in shared RLock)   :1480  (invariant #9)
        server.attach_model(self.model, device)                        :1481
      if FF mode == anysplat_decode: _start_anysplat_persistent_worker():539-540
        PersistentAnysplatWorker.adopt(fifo_dir, 60s)  OR spawn fresh  :2915 / :2936
          → subprocess.Popen in anysplat_dynamic_gs env (or adopt FIFO worker)
  # back in LiveDynamicGSPipeline.__init__, AFTER super():
  self._filter_depth_at_ff = True   (tracker raw / FF-filtered)        live:117
  datamanager.set_phase("dynamic")                                     live:123-124
  model.set_phase("dynamic")  → _apply_phase_trainability:             live:128-129
     means.requires_grad=True, others False                  dynamic_gs_model.py:796-800
     _apply_phase_optimizers(): active={"means"}, ALL group lr=0       :802-817 (invariant #4)
  LiveShmSubscriber(...)  ──────────────────────────────────────────► live:139
     _spawn_publisher → subprocess.Popen(live_ros_publisher) in        live_shm_reader.py:278/162
        dynamic_gs_ros env; block on {"event":"ready"}                 :291
     SharedMemory(name, create=False) + resource_tracker.unregister    :314-317  (attach, not own)
     atexit.register(self._atexit_close)                               :331
  wait_for_first_frame(30s)                                            live:153
  _start_stdin_stop_watcher()  → daemon thread "dgs-live-stop-watcher" live:157,224-237
  atexit.register(_cleanup_live_subscriber)                            live:161
  atexit.register(_cleanup_live_ff_dump)                               live:162
  signal.signal(SIGINT/SIGTERM, _on_signal)  (if on main thread)       live:185-191
```

### Per-tick runtime (trainer thread)

`DynamicGSTrainer.train_iteration` (dynamic_gs_trainer.py:60) sees
`current_phase == "dynamic"` → fast path, skips zero_grad/backward/optimizer/
scheduler/AFTER_TRAIN callbacks (correct, all no-ops under invariant #4) →
`pipeline.get_train_loss_dict(step)` (base:975) → `_tracker_tick(step)`
(live:243): peek SHM frame, build camera+batch, D0 bootstrap OR
`_apply_motion_estimator`, decide `_ff_due_this_tick`, publish
`_latest_tracker_frame`, push viser, `_on_tracker_frame` →
`_dispatch_feedforward_async` (base:2501) → spawns daemon thread
`_feedforward_threaded` (base:2519) holding `_anysplat_slot_lock`.

### Threads at steady state

| Thread | Spawned at | Holds | Lifetime |
|---|---|---|---|
| main/trainer | ns-train | `_model_lock` at mutation sites | process |
| viser render thread | `_setup_viser_direct` (base:1473) | `_model_lock` (swapped in, base:1480) around `get_outputs` | until `srv.close()` |
| `dgs-live-stop-watcher` (daemon) | live:237 | none (sets flags) | daemon, dies w/ process |
| `ff-{recurring,oneshot}` (daemon) | base:2513 | `_anysplat_slot_lock` whole call; `_model_lock` for cull/insert | per FF call |
| AnySplat worker | subprocess (anysplat env) | own GPU | until `close()` |
| ROS publisher | subprocess (ros env) | own SHM region (owner) | until `close()` |

Note: `SamWorkerClient` (sam_worker.py) is **NOT** spawned in the live dynamic
runtime — it lives only in the static-gs capture / Phase-0a path. In live, the
fused object is already baked into `static_state.pt`. ESAM is the only
segmentation model touched at live runtime, and only if the interactive picker
fires (`_get_esam_model` → `build_esam_ti`, dynamic_gs_model.py:1909-1912).

---

## (b) Shared-state access table

| State | Site (file:line) | Thread(s) | Lock held? | Racy? |
|---|---|---|---|---|
| `model.gauss_params[*]` (Parameter realloc) | post_fusion_cache.py:129; dynamic_gs_model.py:617 | main (init only) | NO | No — init is single-threaded, before any worker/render thread can touch the model |
| `model.gauss_params["means"]` grad hook | populate_modules :665 (cold) + post_fusion_cache.py:143 (rebind) | main (init) | NO | No (init); rebind is mandatory — old hook bound to freed tensor |
| 4 identity buffers (`object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`) | rebuilt load_state_dict :577-606; read everywhere | main + ff + viser | partial | See hazard H3 |
| `model._render_lock_ctx` | set base:520; read dynamic_gs_model.py:716 | main set / viser+main read | n/a | No — set once at init before threads start |
| `_model_lock` (RLock) | base:474; viser swap :1480 | main, ff, viser | self | Correct — shared RLock is the cross-thread guard |
| viser `server.model_lock` swap | base:1480 | main (init) | NO | **H1**: swap is non-atomic w.r.t. an already-running render thread |
| `insert_inpaint_gaussians` (resize gauss_params + buffers) | base:3497-3506; dynamic_gs_model.py:1150 | ff bg | `_model_lock` (base:3497) | No — under lock; viewer joins via attach_render_lock |
| `model.get_outputs_for_camera` | dynamic_gs_model.py:708-739 | viser render + main CDN | `_render_lock_ctx`==`_model_lock` | No — both paths share the RLock |
| `_latest_tracker_frame` (dict) | written live:323; read base FF dispatch + `_scene_c2w_for_frame` | main writes / ff reads | NO | **H2**: ff bg reads the dict the main thread re-points each tick |
| `_anysplat_slot_lock` | base:2509 acquire / 2542 release / 658 drain | main acquire, ff release | self | Correct (acquire main, release in ff `finally`) |
| `_shm_sub` (+`_slot_views`) | live:139/250; cleanup :197-205; reader close :587 | main + atexit + signal handler | NO | **H4**: `_cleanup_live_subscriber` callable from atexit AND signal handler concurrently |
| `_last_feedforward_wall_time` | base:2512 write; :962 read | main | NO | benign (float store/load) |
| `_reselect_requested` / `_pending_selected_id` / `_selection_state` | stdin watcher live:234; viser cb base:1308-1316; tick base:1376 | watcher + viser + main | `_selection_event` for handoff | mostly OK; flags are plain bools — see H5 |
| `_motion_estimator` (XFeat model) | base:1156 (drop) / 1993 (new) | main | NO (reseed on main) | No race, but see leak L4 |
| `model._esam_model` | dynamic_gs_model.py:1910-1912 | main | NO | No race; lazy GPU load, never freed (L3) |

---

## (c) Hazards

### H1 — viser render thread can race the `model_lock` swap during init  ·  MEDIUM · race
`_setup_viser_direct` constructs `ViserDirectScene(port)` which starts the viser
server + its render thread (base:1473), THEN swaps `server.model_lock =
self._model_lock` (base:1480), THEN `attach_model` (base:1481). If a browser
client is already connected (e.g. a fast resume on a port that a previous tab
points at), the render thread can enter `get_outputs` between server-construction
and the lock swap, acquiring viser's *original* internal lock while a later FF
insert uses `_model_lock` — two different locks, no mutual exclusion → the exact
torn-snapshot CUDA assert invariant #9 exists to prevent. In practice the model
isn't attached until :1481 so `get_outputs` has nothing to render, but the
ordering is fragile.
Fix: pass the shared lock into the `ViserDirectScene` constructor (so it is the
render thread's lock from thread-start), or don't start the render thread until
after `attach_model`. Verify the constructor (`utils/viser_direct.py`) does not
spin the render loop before `model_lock`/`attach_model` are set.

### H2 — FF bg thread reads `_latest_tracker_frame` the main thread keeps re-pointing  ·  LOW · race
`_dispatch_feedforward_async` passes `self._latest_tracker_frame` (a dict) to the
bg thread (base:2514). The main tick reassigns `self._latest_tracker_frame = {...}`
every tick (live:323) — that reassignment is safe (the bg holds its own
reference to the old dict). BUT the bg thread also *mutates* the passed dict:
`target_frame["cdn"] = self._compute_tick_cdn(...)` (base:2536). Since the main
thread has already moved on to a NEW dict object, this writes the stale dict only
— no cross-thread tear. `_scene_c2w_for_frame` (live:458) reads
`self._latest_tracker_frame["camera"]` (the *current* one), not the captured
`target_frame`, so a fast tracker can make the FF reproject against a camera pose
one-or-more ticks newer than the frame it is decoding. This is a correctness
smell (pose/frame skew in FF inserts), not a crash.
Fix: have `_scene_c2w_for_frame` read the pose from the captured `target_frame`
passed to the bg thread, not from `self._latest_tracker_frame`.

### H3 — identity buffers read without `_model_lock` while FF bg resizes them  ·  MEDIUM · race
`insert_inpaint_gaussians` (dynamic_gs_model.py:1150) and `delete_gaussian_indices`
(:1116) reassign `self._buffers["object_flags"/...]` to new-length tensors on the
ff bg thread under `_model_lock`. But several reads of these buffers happen on the
main/viser threads WITHOUT the lock: e.g. base:3519-3520 (`object_flags.sum()`,
`inserted_flags.sum()` logging — same thread, fine), live:485
(`self.model.object_instance_ids` in `_bootstrap_d0`), live:357-398
(`_pick_d0_object` reads `object_instance_ids`/`means`), and the timing-report /
final-snapshot atexit hooks (base:684-685, :519-520) read `object_flags.sum()` /
`num_points`. D0 reads happen before any FF can fire (FF is gated `is_first` →
returns False, base:950), so those are safe. The atexit reads can race a final
in-flight FF insert — but `_cleanup_anysplat_bg` (base:650) is registered AFTER
the worker/snapshot hooks so LIFO drains the bg slot FIRST. Net: the dangerous
window is narrow but real if an FF insert is mid-resize when an unlocked reader
indexes a buffer by an out-of-date length.
Fix: take `_model_lock` around buffer reads that can overlap FF (the snapshot/
report hooks, and any future per-tick buffer indexing), or snapshot lengths under
the lock.

### H4 — `_cleanup_live_subscriber` is not idempotent against concurrent callers  ·  MEDIUM · lifecycle
`_cleanup_live_subscriber` (live:197) is registered with `atexit` (live:161) AND
called from the SIGINT/SIGTERM handler `_on_signal` (live:173). On Ctrl+C the
signal handler runs on the main thread, calls `sub.close()`, sets
`self._shm_sub = None`; then normal interpreter shutdown fires the same atexit
hook. The guard `if sub is None: return` makes the second call a no-op, so this
is *mostly* safe — but `LiveShmSubscriber.close()` itself (live_shm_reader.py:563)
is guarded by `self._closed` so double-close is fine too. The real gap:
`close()` does `self._proc.wait(timeout=5.0)` then on timeout `terminate()` but
**never `kill()` or a second `wait()`** (live_shm_reader.py:579-585). A publisher
that ignores SIGTERM (it holds `/camera_info` subscriptions + the SHM region)
becomes an orphan — exactly the "stale `dynamic_gs_live_pub` node" failure the
CLAUDE.md "Live publisher restart cleanup" memory documents.
Fix: after `terminate()`, add `try: self._proc.wait(timeout=5); except: self._proc.kill()`.

### H5 — interactive-picker flags are plain bools shared across 3 threads  ·  LOW · race
`_reselect_requested` is set by the stdin watcher (live:234) and the viser
"Change object" button callback (base:1346), and read+cleared by the main tick
(`_tick_interactive_selection`, base:1376/1383). `_pending_selected_id` is set in
the viser Done callback (base:1308) and read in `_wait_for_selection` (base:1412),
with `_selection_event` providing the actual happens-before handoff for the
selection value. The `_reselect_requested` flag has no such barrier — a set from
the watcher can be missed/clobbered if it lands exactly as the tick clears it. Low
impact (operator just clicks again), and only when `interactive_object_selection`
is True (default False).
Fix: route `_reselect_requested` through the same `threading.Event` discipline,
or guard with a small `threading.Lock`.

### L1 — ROS publisher orphaned on hard kill  ·  MEDIUM · leak
See H4. On SIGKILL of ns-train (or any path that skips atexit + signal handlers),
the publisher subprocess + its depth-republisher child survive holding the SHM
region (`/dgs_live_shm`) and ROS subscriptions. The reader only *attaches* to the
SHM and explicitly `resource_tracker.unregister`s it (live_shm_reader.py:317), so
the reader will never unlink it — by design (publisher owns it). If the publisher
dies without unlinking, `/dev/shm/dgs_live_shm` leaks until reboot or manual
removal, and the next run's `SharedMemory(create=False)` attaches to a stale
region (magic check at :321 is the only guard).
Fix: the publisher should `unlink` its SHM in its own atexit/signal path
(verify live_ros_publisher.py); document a `rm /dev/shm/dgs_live_shm` recovery
step. The existing CLAUDE.md scoped-`pkill` + `rosnode cleanup` is the current
mitigation.

### L2 — AnySplat worker + its /dev/shm IPC file leak on hard kill  ·  LOW · leak
`_cleanup_anysplat_worker` (base:631) and `_cleanup_anysplat_ipc_file` (base:640)
are atexit-only. On SIGKILL the worker subprocess (or adopted detached worker)
stays resident on the GPU and `/dev/shm/anysplat_ipc_<pid>.npz` leaks. The adopt
path (base:2915) targets-kills a verified pid (`_pid_is_anysplat_worker`), so a
later run can reclaim/replace it, but a crashed run leaves the GPU memory pinned
until the next adopt or manual kill.
Fix: none required if the adopt-reclaim is reliable; document the leak.

### L3 — ESAM model never freed  ·  LOW · leak
`DynamicGSModel._get_esam_model` (dynamic_gs_model.py:1909) lazily builds
`build_esam_ti(...)` into `self._esam_model` on first interactive-picker use and
never releases it. It co-resides with splatfacto + AnySplat for the rest of the
session. Only matters when `interactive_object_selection=True`.
Fix: add an unload after picker selection completes if VRAM is tight.

### L4 — old XFeat estimator GPU weights not explicitly freed on object switch  ·  LOW · leak
`_reseed_tracked_object` sets `self._motion_estimator = None` (base:1156) then
builds a fresh `XFeatMotionEstimator` (base:1993) each switch. The old estimator's
XFeat + LighterGlue weights are reclaimed only by Python GC + later
`empty_cache`, not deterministically. Repeated mid-run object switches transiently
double the tracker's GPU footprint.
Fix: explicitly `del`/move-to-CPU + `torch.cuda.empty_cache()` before constructing
the replacement.

### B1 — `sam3d_init_target_flags` rebuilt/zeroed but never written at runtime  ·  N/A · deadbranch (expected per invariant #8)
`load_state_dict` rebuilds it (dynamic_gs_model.py:589) and `insert_*` /
`delete_*` carry it (`:1059`, `:1141`), but the only value-writer
(`initialize_object_from_sam3d`, model:1636) has no live caller. All-zeros is the
documented expected state (invariant #8). NOT a bug — flagged so the purge does
not "clean it up" by deleting the buffer; the buffer is load-bearing for
`state_dict` shape compatibility and the dump scripts.

### B2 — per-tick allocations on the tracker critical path  ·  LOW · allocation
`_batch_from_live_frame` (live:523) allocates 3 fresh GPU tensors per tick
(rgb/depth/mask copies). `_pick_d0_object` (live:381-387) projects ALL means each
deferred D0 attempt. These are expected (live frames are new each tick) and D0
runs only until the first lock, so not a steady-state cost. No fix needed; noted
so the purge does not mistake them for waste.

---

## Summary for the purge

- **Do NOT delete** the 4 identity buffers or `sam3d_init_target_flags`'s rebuild/
  carry logic (invariant #8; load-bearing for `state_dict` shape + dump scripts),
  the `_step_offset=10_000` set (base:593; FF inserts land wrong without it), the
  means-grad hook re-bind (post_fusion_cache.py:143; the cold hook is bound to a
  freed tensor after realloc), or the `_model_lock` swap into viser (base:1480;
  invariant #9 cross-thread guard).
- `SamWorkerClient` (sam_worker.py) is **not on the live runtime path** — it is
  static-gs/Phase-0 only. Safe to ignore for the live trace; do not assume the
  live pipeline spawns or closes it.
- The real lifecycle weaknesses are shutdown-on-hard-kill (H4/L1/L2: publisher +
  AnySplat worker + 2 `/dev/shm` files can orphan) and two narrow unlocked-read
  races (H1 viser lock-swap window, H3 identity-buffer reads vs FF resize). None
  block the purge; H4's missing `kill()` after `terminate()` is the highest-value
  one-line fix.
