# RUNTIME target architecture — purged live dynamic-gs path

Goal of tomorrow's purge: collapse the live runtime to a minimal, race-free, clearly-threaded path with ONE shared lock and ONE source of truth for shared state, without breaking any CLAUDE.md invariant. This doc states the intended end-state and the keep/merge/delete plan to reach it.

## Invariants this architecture must respect (NON-NEGOTIABLE)

- **#4** — dynamic phase = pure tracker + FF runtime; ALL gauss-param LRs = 0 (`_ZERO_LR_OPTIMIZERS`). No per-step gradient descent. Keep `set_phase("dynamic")` → `_apply_phase_optimizers` lr=0. Find the best way to do this, i really dont want any compute to be spent on computing a gradient which then is not being used, or even loading and unloading tensors for example.
- **#8** — identity buffers owned by phases: `object_instance_ids` (Phase 0b), `inserted_flags` (Phase 0b + FF Mode B / anysplat insert id=999), `sam3d_init_target_flags` (never value-written at runtime — KEEP all-zeros buffer), `object_flags` (D0 selection only). Never delete a buffer because it "looks unused."
- **#9** — live viz is viser-direct ONLY (port 8081), NEVER the NS viewer (`vis="tensorboard"`, no `--vis viewer`). The viser render thread is read-only and does NOT call `get_outputs_for_camera` for viz — it calls `get_outputs` directly under the shared lock.
- **#6** — background = Gazebo sky `(0.86,0.92,1.0)`.

## Thread model (3 threads + 2 subprocesses)

```
TRAINER / TRACKER thread (main, nerfstudio Trainer)
  NoSaveTrainer.train_iteration → get_train_loss_dict → _tracker_tick
  Per tick: peek SHM (lock-free seqlock) → build camera+batch (RAW depth) →
            D0 bootstrap OR _apply_motion_estimator (XFeat, no model touch) →
            UNDER _model_lock: apply_rigid_object_transform_from_reference
              (in-place means/quats[object_mask] write) →
            publish _latest_tracker_frame (single-writer, per-tick immutable dict) →
            request viser render → dispatch FF (non-blocking)
  SOLE writer of: rigid transform, object_flags (D0), _latest_tracker_frame,
                  _obj_mask_cache invalidation.

FF-BG thread (single-in-flight, daemon, gated by _anysplat_slot_lock)
  _feedforward_threaded → CDN render (UNDER lock) → _anysplat_bg_run
  → AnySplat subprocess IPC → reproject (tensors only) → voxel dedup →
    UNDER _model_lock: delete_gaussian_indices (cull) + insert_inpaint_gaussians (id=999)
  SOLE count-changing writer. SOLE writer of inserted_flags/object_instance_ids tail.

VISER-RENDER thread (daemon, viser_direct._render_loop)
  waits on _render_requested Event → per client UNDER _model_lock: model.get_outputs
  READ-ONLY of gauss_params. Never mutates model.

ROS PUBLISHER subprocess (dynamic_gs_ros py3.8) — owns the SHM segment.
ANYSPLAT WORKER subprocess (anysplat_dynamic_gs) — stateless decode-on-IPC.
```

## Single source of truth for shared state

- **Gaussian state** = `model.gauss_params` (6 Parameters) + 4 identity buffers, living on the `DynamicGSModel`. This is THE source of truth. Decoders (`anysplat_decode`/`rgbd_decode`) produce tensors ONLY; all mutation funnels through `dynamic_gs_model.py` methods (`apply_rigid_object_transform_from_reference`, `delete_gaussian_indices`, `insert_inpaint_gaussians`) called by the pipeline under the lock.
- **Latest sensor frame** = the publisher's 4-slot SHM ring (publisher-owned). Reader does a lock-free seqlock copy into a per-tick `LiveFrame`.
- **Per-tick handoff to FF** = `_latest_tracker_frame` dict, rebound each tick (immutable snapshot of camera/batch/cdn/stamp). FF is handed the dispatch-time dict and is single-in-flight, so it owns its snapshot. RULE: FF must read its camera/RGB from the dispatch snapshot, NOT re-read `_latest_tracker_frame` (fixes H-live-camera + H-live-rgb staleness).

## The one lock

- **`self._model_lock`** — a single re-entrant `RLock` created in `DynamicGSPipelineBase.__init__` (base:474). It is:
  - shared into the model via `attach_render_lock(self._viser_lock_ctx)` (base:519-520),
  - shared into the viser server via `server.model_lock = self._model_lock` (base:1480).
- **Discipline (the rule for the purged code):** EVERY read OR write of `gauss_params` / identity buffers / `model.info` / `self.training` happens inside `with self._viser_lock_ctx():`. No exceptions. Today three sites violate this and are the top race fixes:
  1. `_object_crop_bbox` reads `means`/`ids` unlocked (HIGH) → wrap.
  2. `_pick_d0_object` reads `means`/`ids` unlocked (HIGH) → wrap.
  3. `model.train()/eval()` toggled outside the lock (HIGH) → move inside, or pass explicit render-mode.
- **What the lock must NOT do:** it must NOT be held across the whole ~270 ms AnySplat call — that would block the tracker. FF acquires it only for the CDN render, the cull, and the insert. The publisher's pose/joint history needs its OWN separate lock (do not overload `_model_lock` or `_state_lock`).
- **Setup ordering fix:** pass `_model_lock` into the `ViserDirectScene` ctor (or defer starting the render thread until after the swap) so the render thread never runs with the wrong lock (H viser-swap).

## SHM + worker lifecycle (spawn / own / free)

- **SHM segment** — OWNED by the publisher: `SharedMemory(create=True)` at live_ros_publisher.py:603 after unlinking any stale name (:595-602). Reader attaches `create=False` + `resource_tracker.unregister` (live_shm_reader.py:314-317) so it never tries to unlink. `shutdown()` deliberately does NOT clear views nor unlink (avoids racing late callbacks / a still-reading reader). Reclamation = next publisher launch unlinks+recreates.
  - PURGE ADDITIONS: best-effort unlink in the publisher's SIGTERM/SIGINT/atexit path for NORMAL exits (keep next-run unlink as the real cleanup); reader `peek_latest` early-returns None when `_closed`.
- **ROS publisher subprocess** — spawned by `LiveShmSubscriber._spawn_publisher` with LD_LIBRARY_PATH/CPATH/LIBRARY_PATH/CUDA_HOME stripped (load-bearing, KEEP). Teardown MUST `terminate()` → `wait(timeout)` → `kill()` → `wait()` to stop orphaning the node (current code stops at `terminate()` — the documented stale-node bug). Pair with scoped `pkill` + `rosnode cleanup`.
- **AnySplat worker subprocess** — adopted-or-spawned at init; closed in atexit. Add best-effort `/dev/shm` IPC-file glob-unlink (`anysplat_*_<pid>*`, `dgs_live_ff_frame_<pid>.png`).
- **Slot lock** — `_anysplat_slot_lock` enforces single-in-flight FF; wrap `Thread.start()` in try/except that releases on failure (else FF dies silently for the whole run).

## Keep / merge / delete

### KEEP (load-bearing — do not touch)
- `_model_lock` RLock + `attach_render_lock` wiring + the `server.model_lock` swap (base:1480).
- The is_closing guard (base:1585), `request_render` after writes (base:2196,3508).
- `update_camera_feed` / `update_tracked_camera` real viser hooks.
- The env-strip in `_spawn_publisher` (live_shm_reader.py:225-226) + its comment.
- `sam3d_init_target_flags` buffer (invariant #8) and all 4 identity buffers.
- The seqlock SHM read/write (correct, lock-free on x86) and the publisher-owns-segment lifecycle.
- The `is_first` FF gate (base:950-951) — load-bearing for D0 safety.
- `_step_offset=10_000`, `set_phase("dynamic")`, lr=0 enforcement (invariants #4).
- Decoders as tensor-only producers; all mutation in `dynamic_gs_model.py`.

### MERGE / FIX (correctness, no invariant impact)
- Wrap `_object_crop_bbox`, `_pick_d0_object`, and `model.train()/eval()` toggles under `_model_lock`.
- Thread the dispatch camera + dispatch RGB through FF `bg_args`; make live `_scene_c2w_for_frame` and `_resolve_anysplat_context_image_paths` read the snapshot, not the live `_latest_tracker_frame`.
- Add the publisher pose/joint history lock (separate from `_model_lock`).
- Double-checked locking (or per-dispatch snapshot) for `_obj_mask_cache`.
- `terminate()→kill()` second-wait in the subscriber close; try/except on `Thread.start()` for the slot lock.
- Snapshot `srv = self._viser_direct_server` at the top of the two read-then-use methods.

### DELETE (dead / inert under the mandated config)
- Path-A viser no-op stubs: `push_tracker_transform`, `add_ff_insert_chunk`, `maybe_flush_ff_handle`, `flush_pending_ff`, `refresh_static_handle` (viser_direct.py:506-528) AND their call sites (`_push_viser_direct_transforms` per-tick call; the `*_ff_handle` wrappers).
- `_force_viewer_rerender` (base:1614) — no-op under invariant #9; remove with the `get_training_callbacks` `_trainer` stash IF dropping NS-viewer-for-static-ckpt support (otherwise keep guarded).
- The `model.info` rgbd-cull path (`_feedforward_delete_in_region` → `extract_projected_centers_and_radii`) — the anysplat default path proves direct-projection cull works without `model.info`; deleting it removes a whole cross-thread race class.
- Legacy `apply_rigid_object_transform` (dynamic_gs_model.py:922-939) if confirmed to have no live caller.

## Confidence
- HIGH on the thread model, the single-lock discipline, and the SHM/worker ownership (directly traced across all 5 reports, corroborated by source reads in 4 of them).
- HIGH that H-CROP / H-PICK / train-eval are real unlocked cross-thread accesses; MEDIUM on exact crash frequency (interleave window is narrow; masked today by FF cadence + is_first gate).
- MEDIUM on the delete list being fully dead — confirm `apply_rigid_object_transform` and the rgbd `model.info` path have no live caller before removing (grep the call graph).
