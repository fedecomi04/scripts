# DYNAMIC-GS PURGE PLAN — Human-Review Checklist

Synthesized from a per-module adversarial code-audit + a repo-wide dead-code grep pass (60 candidate symbols verified). Tick each box, fill the `DECISION:` field, then execute. **Live path is purged FIRST.**

Confidence legend: **DEAD** = verified zero live refs; **UNCERTAIN** = reachable only through a code path that is itself bypassed/unconfigured (verify before deleting); **LIVE-but-functionally-dead** = write-only / multi-site, needs coordinated removal not a single delete.

---

## EXECUTIVE SUMMARY — TOP 10 (ranked by impact)

1. **`/dev/shm` AnySplat per-window file leak (LIVE, correctness-adjacent).** `dynamic_gs_pipeline_base.py:640` `_cleanup_anysplat_ipc_file` unlinks `anysplat_ipc_<pid>.npz`, but the code at `:3367/:3369` actually writes `anysplat_crop_<pid>_<wi>.png` + `anysplat_ipc_<pid>_<wi>.npz` (per-window `_wi` suffix). Cleanup is a **no-op** → crop PNGs + per-window npz accumulate in tmpfs every FF call until reboot. **FIX FIRST.**
2. **NameError latent bug aborts the canonical live SAM3D path.** `dynamic_gs/fusion/phase0.py:661` reads `static_np.shape` unconditionally, but `static_np` is assigned only on the non-anchor fallback branch (`:492/:494`). On the live `anchor_ref` path with an uncached SAM3D run — exactly what `anchor_ref` exists for — Phase 0a raises `NameError`. **This is a live crash, not a smell.**
3. **FF bg-thread frame-skew race (LIVE).** `dynamic_gs_pipeline_live.py:446/463` re-read `self._latest_live_rgb_bgr` / `self._latest_tracker_frame` at FF-execution time on the bg thread, but the tracker rebinds both every tick. The AnySplat context RGB + insert back-projection c2w can be N ticks newer than the CDN the inserts were computed against → misplaced inserts (same bug-family as the documented FF/CDN frame-consistency issues). Snapshot into the dispatched `target_frame`.
4. **`self.info` shared-mutable race across 3 threads (LIVE).** `dynamic_gs_model.py:2297` `get_outputs` writes `self.info` on tracker / FF-bg / viser threads; mask helpers read it. Correctness depends entirely on the external `_model_lock`; any unlocked read races an overwrite (wrong centers/radii, or a shape-mismatch CUDA assert if a resize interleaves).
5. **Two large dead god-methods carrying entire dead subsystems.** `prepare_dynamic_update` (~210 LOC, `dynamic_gs_model.py:1915`) is dead and drags in the whole ESAM chain (`_get_esam_model`, `combine_object_masks`, `_depth_diff_score`, `_set/_get_optim_mask`, `_masked_rgb_l1`). Deleting it unlocks the largest single cleanup in the repo.
6. **`get_outputs` wastes 3 extra rasterizations per CDN render (LIVE perf).** `dynamic_gs_model.py:2329` computes `flagged_rgb/non_flagged_rgb/non_inserted_rgb` (4 rasterizations total) every render; only the dead/legacy Path-A consumed them. Gate behind a flag → skip ~3 GPU rasterizations/tick.
7. **Unbounded anchor-pool VRAM growth (LIVE).** `xfeat_motion.py:1206` clones a full-res RGB GPU tensor (~27.6 MB @1200p) + descriptors + mask per anchor, with **no cap/eviction**; resets only on reseed. Monotonic VRAM growth competing with the splat scene on the same 16 GB card. RGB/mask kept *solely* for the debug visualizer.
8. **Unlocked cross-thread pose/joint sample lists (LIVE publisher).** `live_ros_publisher.py:762/781` are sorted-inserted by rospy callback threads and read by the worker (`_interpolate_c2w :784`) with no lock; a multi-step `bisect+index` races a concurrent `list.insert` that may reallocate.
9. **`peek_latest` races unsynchronized `close()` (LIVE reader).** `live_shm_reader.py:373` reads slot views + mmap lock-free (correct seqlock), but `close()` (`:587-589`, atexit thread) drops `_slot_views` and unmaps with no lock → tracker-thread `peek_latest` mid-read can `IndexError`/segfault at shutdown.
10. **`.pt` warm-cache `strict=False` with no post-load verification.** `persistence/post_fusion_cache.py:138` verifies only the 6 `gauss_params`; the 4 invariant-protected identity buffers + config-sensitive params (sh_degree/bg/camera-opt) ride on `strict=False` with no N/shape assertion → silent buffer desync against the resized `gauss_params`.

---

## SECTION 1 — LIVE PATH: PURGE FIRST

### `dynamic_gs/dynamic_gs_pipeline_base.py`
- [ ] **HIGH — base.py:640 vs :3367/:3369** — `/dev/shm` cleanup targets a filename the code never writes (per-window crop PNGs + `_wi` npz leak forever) — **REFACTOR** (make cleanup unlink the actual per-window paths, or unlink in the per-call finally) — DECISION: ____
- [ ] HIGH — base.py:3209 — `_anysplat_bg_run` is a ~345-line god-function with 3 inner `_model_lock` scopes (hardest code to verify for races) — **REFACTOR** (extract union voxel-dedup `:3425-3515` + `_voxel_keep_idx`; split frustum-cull/ICP from per-window loop) — DECISION: ____
- [ ] HIGH — base.py:2995/:2571 — FF bg thread does non-atomic `_feedforward_call_counter += 1` and concurrent `self._timing[key].append` with tracker thread — **REFACTOR or KEEP+DOCUMENT** (guard with a lock, or explicitly document GIL-only safety) — DECISION: ____
- [ ] MED — base.py:592/:599 — `_step_offset=10000` (bypasses Splatfacto res/SH schedules; missing offset → FF inserts back-project with wrong intrinsics → wrong world locations) set in a try/except that only **warns** — **REFACTOR** (hard-fail) — DECISION: ____
- [ ] MED — base.py:2949 — `_resolve_anysplat_context_image_paths` / `_scene_c2w_for_frame` are recorded-only (read `dataset.image_filenames/cameras`, TODO(phase-3-stage-D)) but live on the base — **REFACTOR** (make base an abstract stub; move recorded lookups to recorded subclass) — DECISION: ____
- [ ] MED — base.py:1873 — `_object_crop_bbox` guards buffer-vs-means desync and returns None with "separate bug" comment — **KEEP+DOCUMENT** (underlying desync not fixed here; tracked in Section 3) — DECISION: ____
- [ ] DEAD — base.py:1605 — `_force_viser_direct_push` (public alias, 0 refs; subclasses call `_push_viser_direct_transforms`) — **DELETE** — DECISION: ____
- [ ] DEAD — base.py:1704 — `_render_from_camera_at_scale` (0 refs; reduced-res CDN was reverted, CDN uses full-res `_render_from_camera`) — **DELETE** — DECISION: ____
- [ ] DEAD — base.py:966 — `_oneshot_ff_due` (0 refs; `get_train_loss_dict :987` inlines the identical predicate) — **DELETE** (or call the method instead of inlining) — DECISION: ____
- [ ] DEAD-config — base.py:207 — `feedforward_anchor_frame` (config field, 0 reads) — **DELETE** — DECISION: ____
- [ ] DEAD-config — base.py:209 — `feedforward_video_out` (0 reads; no video writer implemented) — **DELETE** — DECISION: ____
- [ ] DEAD-config — base.py:210 — `feedforward_video_fps` (0 reads) — **DELETE** — DECISION: ____
- [ ] LOW dead-infra — base.py:440/:690 — `_feedforward_video_writer` only ever set None; `_cleanup_feedforward_video_writer` never releases anything — **DELETE** with the 3 video config fields — DECISION: ____
- [ ] LOW naming — base.py:1496 — `_build_viser_direct_handles` is now a D0-pose-only stub; `_run_feedforward_anysplat` runs `_anysplat_bg_run` inline despite "async"/"bg_run" naming — **KEEP+DOCUMENT** (rename/comment the real threading topology) — DECISION: ____
- [ ] MED dup — base.py:1892/:1959/:2789/:3115 — `_scalar` local redefined 4×; FF clean→cull→reclean→select prelude duplicated rgbd vs anysplat — **MERGE** into one private helper — DECISION: ____
- [ ] LOW lifecycle — base.py:3496 — no dynamic-phase FF-insert purge (insert_inpaint_gaussians instance_id=999 grows unbounded, 459k→1.29M) — **KEEP+DOCUMENT** (known TODO `[[static-phase-opacity-purge-todo]]`) — DECISION: ____

### `dynamic_gs/dynamic_gs_pipeline_live.py`
- [ ] **HIGH — live.py:446/:463 vs :321/:323** — FF bg thread re-reads tracker-mutated `_latest_live_rgb_bgr` / `_latest_tracker_frame` instead of the dispatched snapshot → RGB + back-proj c2w lag the CDN by N ticks — **REFACTOR** (snapshot RGB.copy() + c2w into dispatched `target_frame`) — DECISION: ____
- [ ] MED — live.py:321/:455 — bare cross-thread ndarray `_latest_live_rgb_bgr` shared (no copy); `_anysplat_slot_lock` guards only the dump filename, not the source array — **REFACTOR** (`.copy()` at tick time) — DECISION: ____
- [ ] MED — live.py:243 — `_tracker_tick` is a ~95-line god-method (stop-check, SHM peek, sim-clock dedup, camera/batch, picker, D0, motion, FF gating, publish, 4 viser pushes, hook) — **REFACTOR** (extract picker/bootstrap/motion + publish block) — DECISION: ____
- [ ] MED lifecycle — live.py:197/:161/:173/:203 — SHM subscriber freed only via atexit/signal (unreliable on SIGTERM), no explicit close on clean loop-exit; `close()` exceptions swallowed → orphan-publisher failure mode silent — **REFACTOR** (explicit close on stop; log in cleanup except) — DECISION: ____
- [ ] LOW — live.py:317/:326/:338 — `cdn=None` threaded through state + hook signature though always None (recomputed on FF bg thread) — **REFACTOR** (drop vestigial cdn param / document) — DECISION: ____
- [ ] LOW — live.py:458 — `_scene_c2w_for_frame` ignores its `frame_idx` arg (returns current tracker c2w) — **KEEP+DOCUMENT** (compounds the high-sev skew; intentional per docstring) — DECISION: ____
- [ ] LOW — live.py:203/:213/:235 — broad `except: pass` in all cleanup paths hides orphan-publisher — **REFACTOR** (debug-log) — DECISION: ____

### `dynamic_gs/dynamic_gs_pipeline_recorded.py`
- [ ] LIVE-funcdead — recorded.py:120 (+ live.py:132, base.py:458) — `self._accepted_dynamic_frames` write-only, 0 reads at all 3 sites — **DELETE** (coordinated, all 3 sites; drop the redundant `get_num_dynamic_frames` feeding it) — DECISION: ____
- [ ] MED — recorded.py:237 — `_latest_tracker_frame` rebind races FF bg thread holding the snapshot's GPU tensors (shared with datamanager) — **REFACTOR / VERIFY** (confirm datamanager returns fresh per-call tensors; else copy) — DECISION: ____
- [ ] MED — recorded.py:234 — `cdn` hard-wired None yet typed `Optional[Tensor]`, stored, passed to `_on_tracker_frame` (docstring still describes CDN) — **REFACTOR** (drop dead param / document) — DECISION: ____
- [ ] LOW — recorded.py:261/:307-312 — `_pick_d0_object` hand-rolls OpenGL 3D→2D projection; forked from live variant — **MERGE** (shared projection helper in base) — DECISION: ____
- [ ] LOW — recorded.py:196 — in-place depth filter mutates datamanager-returned batch — **VERIFY** datamanager hands back a fresh batch — DECISION: ____

### `dynamic_gs/dynamic_gs_model.py`
- [ ] **HIGH dead — model.py:1915** — `prepare_dynamic_update` (~210 LOC, 0 callers; legacy ESAM path superseded by XFeat+CDN) — **DELETE** (unlocks the ESAM + optim-mask chain below) — DECISION: ____
- [ ] HIGH dead — model.py:1842 — `refresh_dynamic_state_after_insertion` (0 callers) — **DELETE** — DECISION: ____
- [ ] HIGH dead — model.py:1391 — `_propagate_instance_membership` (0 refs; multi-object never wired) — **DELETE** — DECISION: ____
- [ ] HIGH dead — model.py:1297 — `_get_render_projection_params` (0 refs) — **DELETE** — DECISION: ____
- [ ] HIGH dead — model.py:923 — `apply_rigid_object_transform` (bare variant, 0 real callers; `_from_reference` is the live write-path) — **DELETE** (do NOT touch `_from_reference`) — DECISION: ____
- [ ] DEAD-transitive — model.py:1909 — `_get_esam_model` (only caller = dead `prepare_dynamic_update`) — **DELETE** + the `_esam_model` attr — DECISION: ____
- [ ] UNCERTAIN — model.py:842 — `_get_optim_mask` (reached only via dynamic `get_loss_dict` branch, which loss-bypassing `get_train_loss_dict` never invokes; also dead callers) — **DELETE after verifying** the dynamic loss branch is truly unreachable — DECISION: ____
- [ ] UNCERTAIN — model.py:833 — `_set_optim_mask` (callers are dead `refresh_/prepare_dynamic_update`; writes `change_mask_image` read only by loss-bypassed path) — **DELETE after verifying** — DECISION: ____
- [ ] UNCERTAIN — model.py:1903 — `_masked_rgb_l1` (only caller = unreachable dynamic `get_loss_dict`) — **DELETE after verifying** — DECISION: ____
- [ ] **HIGH perf — model.py:2329** — `get_outputs` always renders `flagged_rgb/non_flagged_rgb/non_inserted_rgb` (3 wasted rasterizations/tick) — **REFACTOR** (gate behind a flag so live XFeat path skips them) — DECISION: ____
- [ ] **HIGH race — model.py:2297** — `self.info` single shared mutable attr written by `get_outputs` on 3 threads — **REFACTOR** (return info by value instead of storing on self) — DECISION: ____
- [ ] MED — model.py:764/:2409 — `_step_offset`/`_render_lock_ctx`/`_optimizers_wrapper` poked via getattr with silent defaults; `step_post_backward` assert can kill trainer thread — **REFACTOR** (explicit contracts) — DECISION: ____
- [ ] MED dead-config — model.py:74 — `change_mask_*` (read only by dead `prepare_dynamic_update`), `rigid_static_lambda/rigid_inlier_threshold/depth_lambda` (loss-bypassed), `sam3d_teaser_*` (no-caller init) — **DELETE after verifying** each (NOTE: `change_mask_downsample_target_side` IS live — keep) — DECISION: ____
- [ ] MED dead-config — model.py:201 — `enable_dynamic_mean_optimization` unreachable (needs `not enable_cotracker_rigid_motion`, which defaults True) — **DELETE / KEEP+DOCUMENT** — DECISION: ____
- [ ] LOW — model.py:209/:828 — `enable_cotracker_rigid_motion` gates XFeat not CoTracker; docstring references purged CoTracker — **KEEP+DOCUMENT** (config-compat name; refresh docstring) — DECISION: ____
- [ ] LOW — model.py:2216 — `get_outputs` returns `{}` for non-Cameras arg (downstream `['rgb']` KeyErrors) — **REFACTOR** (raise TypeError) — DECISION: ____
- [ ] HIGH dup — model.py:1500+ — SAM3D-init helper suite duplicated with StaticGSModel; DynamicGSModel copies reachable only from invariant-protected `initialize_object_from_sam3d` — **MERGE** (shared mixin) — see Section 3 — DECISION: ____

### `dynamic_gs/utils/xfeat_motion.py`
- [ ] **HIGH lifecycle — xfeat_motion.py:1206** — unbounded anchor pool, each anchor pins ~27.6 MB RGB GPU tensor (kept only for debug viz) — **REFACTOR** (cap/evict pool; or store RGB on CPU / gate behind debug flag) — DECISION: ____
- [ ] HIGH — xfeat_motion.py:578 — `estimate_and_advance` ~455-line god-method (duplicates `initialize` anchor-build recipe + inlined keep-region) — **REFACTOR** (extract anchor-create `:944-1013`, pose/KF/static-hold tail `:821-928`, keep-region `:656-668`) — DECISION: ____
- [ ] MED perf — xfeat_motion.py:679 — unconditional per-tick `torch.cuda.synchronize()` shipped as diagnostic (serializes vs FF/viser GPU work) — **REFACTOR** (gate behind `DGS_DIAG_SYNC`) — DECISION: ____
- [ ] MED dead-config — xfeat_motion.py:285 — `xfeat_min_cossim`/`min_cossim` stored, never read (plumbed end-to-end from `dynamic_gs_model.py:247`) — **DELETE** (field + ctor param + pipeline pass-through) — DECISION: ____
- [ ] DEAD — xfeat_motion.py:1281 — `_compose_keep_region` (0 callers; recipe inlined) — **DELETE** — DECISION: ____
- [ ] DEAD — xfeat_motion.py:1224 — `_pre_mask_image` (0 callers; full-image-extract+post-filter is the live path) — **DELETE** — DECISION: ____
- [ ] LIVE-funcdead — xfeat_motion.py:430/:431/:433 — `last_anchor_idx_used` / `last_used_fallback_anchor` / `last_pool_size` write-only diagnostics, 0 reads — **DELETE** (coordinated, multi-site writes) — DECISION: ____
- [ ] KEEP — xfeat_motion.py:456 — `current_track_count` (read 3×/tick internally) — **KEEP** (false alarm) — DECISION: ____
- [ ] LOW — xfeat_motion.py:338 — silent `except: pass` on LighterGlue depth_confidence override — **REFACTOR** (log) — DECISION: ____
- [ ] LOW — xfeat_motion.py:3 — stale CoTracker/KLT/TAPIR docstring + always-0.0 timing slots (`klt_forward`/`postprocess`/`resample`) — **KEEP+DOCUMENT / cleanup names** — DECISION: ____

### `dynamic_gs/utils/tracker_common.py` — DEAD ISLAND (delete as one unit)
- [ ] DEAD — tracker_common.py:165 — `sample_mask_points` (KLT residue, 0 refs) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:153 — `_shrink_mask_for_sampling` (only caller dead) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:145 — `_subsample_points` (only caller dead) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:215 — `filter_points_in_image` (0 refs) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:229 — `filter_points_by_mask_array` (0 refs) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:84 — `prepare_tracking_rgb` (0 callers; "preferred path" docstring is a lie) — **DELETE** — DECISION: ____
- [ ] DEAD — tracker_common.py:100 — `prepare_tracking_rgb_gpu` (0 callers; xfeat has own impl) — **DELETE** — DECISION: ____
- [ ] MED dormant — tracker_common.py:334 — `PoseKalmanFilter` (~185 LOC) OFF by default (`xfeat_pose_filter_enabled=False`), disabled 2026-06-13 for lagging — **KEEP+DOCUMENT or DELETE** (decision: is the smoother-motion re-enable still planned?) — DECISION: ____
- [ ] LOW — tracker_common.py:571 — RANSAC re-reads `DGS_RANSAC_SEED` env + builds fresh `default_rng` every tick — **REFACTOR** (resolve seed once) — DECISION: ____

### `dynamic_gs/utils/anysplat_decode.py`
- [ ] DEAD-pair — anysplat_decode.py:832 — `filter_gaussians_by_component_mask` (0 refs; filtering inlined in `reproject_anysplat_to_scene`) — **DELETE** — DECISION: ____
- [ ] DEAD-pair — anysplat_decode.py:502 — `_world_to_image_opengl` (only caller = dead above) — **DELETE** — DECISION: ____
- [ ] HIGH — anysplat_decode.py:624-829 — `reproject_anysplat_to_scene` ~205-line, ~20-kwarg god-function hand-re-slicing 6 parallel arrays across 5 filter blocks (silent attr-desync risk) — **REFACTOR** (bundle/struct sliced in one place) — DECISION: ____
- [ ] MED — anysplat_decode.py:680/:739/:756 — asymmetric early-return (full 6-key dict on success, 1-key on empty) → caller KeyError risk — **REFACTOR** (consistent full empty dict) — DECISION: ____
- [ ] LOW — anysplat_decode.py:178 — `PersistentAnysplatWorker` no internal lock; safe only by `_anysplat_slot_lock` discipline — **KEEP+DOCUMENT** (single-consumer requirement) — DECISION: ____
- [ ] LOW naming — anysplat_decode.py:178 — `*.npz` outputs are pickle blobs (read with `pickle.load`) — **REFACTOR / DOCUMENT** (rename `.pkl`) — DECISION: ____
- [ ] LOW lifecycle — anysplat_decode.py:278/:312 — detached FIFO worker orphanable (~3.5 GB VRAM) across datasets; cmd/res.fifo inodes never unlinked — **KEEP+DOCUMENT** — DECISION: ____

### `dynamic_gs/utils/viser_direct.py` (Invariant #9 — canonical live viz)
- [ ] DEAD — viser_direct.py:518 — `maybe_flush_ff_handle` (0 refs; sibling stubs ARE called) — **DELETE** — DECISION: ____
- [ ] MED — viser_direct.py:506 — five legacy no-op stubs (`push_tracker_transform`/`add_ff_insert_chunk`/`maybe_flush_ff_handle`/`flush_pending_ff`/`refresh_static_handle`) still threaded; pipeline computes args for nothing (base `:1531/:1588/:622/:1601`) — **DELETE** (stubs + call sites together) — DECISION: ____
- [ ] MED dead-config — viser_direct.py:203/:283 — `render_hz` never read (loop is event-driven) but printed in banner as if it caps cadence — **DELETE** (or stop printing) — DECISION: ____
- [ ] MED dead-state — viser_direct.py:472/:456/:581 — `set_initial_camera` stores `look_at`/`fov_y_rad` that `_apply_initial_camera` never reads; docstring lies — **DELETE** (or actually apply) — DECISION: ____
- [ ] LOW dead-config — viser_direct.py:191 — 4 explicitly-ignored legacy `__init__` kwargs (`opacity_floor`/`static_refresh_min_gap_s`/`push_min_gap_s`/`ff_coalesce_gap_s`) — **DELETE** with call site — DECISION: ____
- [ ] LOW race — viser_direct.py:486/:569/:578 — `_initial_camera_applied` mutated lock-free from server thread + lock-inconsistently from pipeline thread (cosmetic snap race) — **REFACTOR** (mutate under `_client_state_lock`) — DECISION: ____
- [ ] LOW — viser_direct.py:208 — shared `model_lock` correct only because base swaps it before `attach_model` starts render thread (fragile ordering) — **REFACTOR** (inject lock via constructor) — DECISION: ____

### `dynamic_gs/utils/live_ros_publisher.py` (LIVE publisher subprocess)
- [ ] **HIGH race — live_ros_publisher.py:762/:781 vs :784** — pose/joint sample lists sorted-inserted by rospy callbacks, read by worker `_interpolate_c2w` + RobotMaskGenerator by-reference, NO lock — **REFACTOR** (lock or snapshot the slice) — DECISION: ____
- [ ] MED — live_ros_publisher.py:1020 vs :1021-1025 — SHM slot writes `seq` BEFORE payload (inverse seqlock; safe only by 4-slot >100 ms reuse window) — **REFACTOR** (write payload then seq) — DECISION: ____
- [ ] MED — live_ros_publisher.py:1146/:1145 — `meta['frames']` aliases lock-guarded `_record_frames_written`; reads outside `_record_lock` see concurrent mutation — **REFACTOR** (copy on read) — DECISION: ____
- [ ] DEAD — live_ros_publisher.py:687 — `_spawn_depth_republisher` (never called; raw 32FC1 subscribed directly, proc stays None) — **DELETE** (+ the no-op shutdown teardown branch) — DECISION: ____
- [ ] DEAD — live_ros_publisher.py:265 — `_total_shm_bytes` (def only; `__init__` recomputes inline) — **DELETE** (or call it) — DECISION: ____
- [ ] DEAD — live_ros_publisher.py:1046 — `wait_first_frame` (op handler exists but reader never dispatches `wait_first_frame`) — **DELETE** (+ unawaited `_first_frame_event`) — DECISION: ____
- [ ] DEAD — live_ros_publisher.py:349 — `_KeyframeFilter.num_kept` (0 refs; counting via `len(_record_frames_written)`) — **DELETE** — DECISION: ____
- [ ] HIGH — live_ros_publisher.py:972 — `_process_synced_pair` does ~10 jobs in the hot path (~70 LOC) — **REFACTOR** (split decode/publish/record) — DECISION: ____
- [ ] MED — live_ros_publisher.py:458/:479 — `__init__` ~225 LOC with hardcoded `_DATASETS_ROOT` glob — **REFACTOR** (extract intrinsics resolution; parameterize root) — DECISION: ____
- [ ] MED — live_ros_publisher.py:858 — `_on_synced` double-nested `except: pass` (genuine fault silently stalls stream) — **REFACTOR** (throttled warn on non-Full/Empty) — DECISION: ____
- [ ] LOW dup — live_ros_publisher.py:811/:992 — lazy `_mask_gen` construction duplicated — **MERGE** (`_get_mask_gen()`) — DECISION: ____
- [ ] LOW — live_ros_publisher.py:101/:117 — module-level `[publisher-debug]` spew + eager pyparsing probe on every launch — **DELETE / gate** — DECISION: ____
- [ ] LOW — live_ros_publisher.py:1295/:1306 — SHM intentionally never unlinked; replay stream closed only in shutdown — **KEEP+DOCUMENT** (leak-by-design; relies on next-launch stale-unlink `:595`) — DECISION: ____

### `dynamic_gs/utils/live_shm_reader.py` (LIVE reader)
- [ ] **HIGH race — live_shm_reader.py:373 vs :587-589** — lock-free `peek_latest` races unsynchronized `close()` dropping `_slot_views` + unmapping mmap (IndexError/segfault at shutdown) — **REFACTOR** (guard close / a teardown flag checked in peek) — DECISION: ____
- [ ] DEAD — live_shm_reader.py:334 — `get_singleton` (0 external refs) — **DELETE** — DECISION: ____
- [ ] DEAD — live_shm_reader.py:265 — `_singleton` (read only by dead `get_singleton`) — **DELETE** — DECISION: ____
- [ ] DEAD — live_shm_reader.py:520 — `save_anchor_for_sam3` (live_session uses module-level `_save_anchor_for_sam3`) — **DELETE** (+ publisher-side op handler) — DECISION: ____
- [ ] DEAD — live_shm_reader.py:529 — `save_anchor_intrinsics_and_depth` (live_session uses module-level variant) — **DELETE** (+ publisher op) — DECISION: ____
- [ ] MED — live_shm_reader.py:551/:558 — `pause/unpause_gazebo_physics` swallow all exceptions → False (dead pipe == unsupported op) — **REFACTOR** (let RuntimeError propagate / log) — DECISION: ____
- [ ] MED perf — live_shm_reader.py:458-466 — `_read_response` per-byte `dbg.write+flush` (O(n) syscalls, re-opens log per response) on the production read path — **REFACTOR** (buffer + gate behind debug flag) — DECISION: ____
- [ ] MED — live_shm_reader.py:57/:78 — `_HDR_FMT`/`_compute_header_field_offsets` duplicated verbatim from publisher with "keep in sync" comment — **REFACTOR** (share format / version-assert against ready msg) — DECISION: ____
- [ ] MED — live_shm_reader.py:236 — parent-side `log_fd` from `_spawn_publisher` never closed (1 fd/session) — **REFACTOR** (stash + close) — DECISION: ____
- [ ] LOW perf — live_shm_reader.py:398-401 — ~18 MB heap churn/tick (4 fresh np copies, no double-buffer) — **REFACTOR if Hz binds** (copies required; add scratch reuse) — DECISION: ____
- [ ] LOW — live_shm_reader.py:446 vs :569 — `_read_response` 600s timeout held under `_proc_lock` that `close()` also needs (stalls graceful shutdown) — **REFACTOR** — DECISION: ____
- [ ] LOW — live_shm_reader.py:5 — stale docstring names env `radiance_ros_4060` (code uses `dynamic_gs_ros`) — **KEEP+DOCUMENT** (fix header) — DECISION: ____

### `dynamic_gs/utils/depth_filter.py` (LIVE — both threads)
- [ ] DEAD-params — depth_filter.py:87 — `filter_depth_torch median=/bilateral=` split kwargs (no prod caller; CLAUDE.md confirms unused) — **DELETE params** (or wire the documented tracker-median/FF-bilateral policy) — DECISION: ____
- [ ] MED — depth_filter.py:117 — cv2 `filter_depth` and torch `filter_depth_torch` are hand-maintained parallel impls with no parity test — **REFACTOR** (add parity assertion / share kernel) — DECISION: ____
- [ ] MED perf — depth_filter.py:142 — ~230 MB transient `F.unfold` GPU spike/call @1200p on the shared GPU — **KEEP+DOCUMENT** (no leak; flag for VRAM accounting) — DECISION: ____
- [ ] LOW — depth_filter.py:29 — kernel/sigma env knobs frozen at import while `enabled()` reads env per-call; docstring claims all are hot — **KEEP+DOCUMENT** (correct the doc) — DECISION: ____

### `dynamic_gs/fusion/phase0.py` (LIVE static→dynamic handoff)
- [ ] **HIGH bug — phase0.py:661** — `static_np` read unconditionally but assigned only on non-anchor branch (`:492/:494`) → `NameError` on live `anchor_ref` path with uncached SAM3D — **REFACTOR** (derive H/W from `static_image.shape` or anchor intrinsics) — DECISION: ____
- [ ] MED — phase0.py:762 — `run_phase0b_fusion` ~405 LOC (ref-frame resolve, render, load, registration, 2 NN-culls, flag, manifest, ledger) — **REFACTOR** (extract `_resolve_reference_frame`/`_cull_inserted_points`/`_flag_existing_gaussians`) — DECISION: ____
- [ ] MED dup — phase0.py:479 vs :824 — reference-frame + `static_dir` resolution duplicated across phase0a/0b — **MERGE** (one helper) — DECISION: ____
- [ ] MED config — phase0.py:1006-1009/:1044/:1065/:1084/:1093 — cull/flag registration knobs hardcoded as locals (can't be A/B'd) — **REFACTOR** (promote to config/env) — DECISION: ____
- [ ] MED — phase0.py:1056/:1103 — Phase-0b per-object insert loop assumes all 4 identity buffers stay length-synced after each `insert_object_gaussians` — **VERIFY** lockstep resize — DECISION: ____
- [ ] LOW — phase0.py:689 — hard SAM3D-subprocess failure silently degrades to 0 objects fused — **REFACTOR** (surface louder) — DECISION: ____
- [ ] LOW labels — phase0.py:867/:588/:1157 — "CPD/TEASER++" print + `S0.1_fastsam` key on SAM3 + hardcoded "NDP register+fuse" ledger label regardless of backend — **REFACTOR** (reflect configured backend) — DECISION: ____

### `dynamic_gs/utils/live_session.py` (LIVE capture flow)
- [ ] DEAD — live_session.py:175 — `_wipe_live_root` (0 refs; wipe done via `LiveShmSubscriber(wipe_live_root=True) :567`) — **DELETE** — DECISION: ____
- [ ] HIGH — live_session.py:495-1203 — `run_live_capture_session` ~700-line linear procedure — **REFACTOR** (extract `_run_segmentation_loop`/`_run_sam3d`/`_build_seed`) — DECISION: ____
- [ ] HIGH naming — live_session.py:159 — pervasive `SAM3` naming while default is FastSAM (`_prompt_user`/`sam3_objects`/`t_sam3`/`DEFAULT_SAM3_PROMPT`/`S0.1_fastsam` key/docstring) — **REFACTOR** (rename `segmenter_*`; refresh docstring) — DECISION: ____
- [ ] MED — live_session.py:549/:567 — subscriber+publisher+SHM never closed on abort/warm paths (leaks unless caller adopts) — **REFACTOR** (explicit close on abort) — DECISION: ____
- [ ] MED — live_session.py:1046-1048 — SAM3D failure → `[{}]`, capture "succeeds" with nothing to track — **REFACTOR** (hard error when n_ok==0) — DECISION: ____
- [ ] MED — live_session.py:189-197/:319-324/:285/:326 — gripper-blackout + uint16-mm depth write duplicated across `_save_anchor_for_sam3`/`_write_anchor_ref` — **MERGE** (helpers) — DECISION: ____
- [ ] LOW — live_session.py:85-92 — `_has_complete_recording_cache` checks existence not content (stale partial cache trips Phase-0b) — **REFACTOR** (count cross-check) — DECISION: ____
- [ ] LOW — live_session.py:165 — `SAM3_MIN_SCORE=0.0` inert but threaded into every infer — **DELETE / document** — DECISION: ____
- [ ] LOW — live_session.py:605 — 6+ `DGS_*`/`AUTONOMOUS_*` env flags read ad-hoc inside the god function — **KEEP+DOCUMENT** (note for config consolidation) — DECISION: ____

### change-detection (`utils/active_mask.py` + `change_detection/change_mask.py`) — LIVE CDN
- [ ] DEAD — active_mask.py:680 — `build_active_mask_center_only` (exported, 0 call sites) — **DELETE** (+ import/`__all__`) — DECISION: ____
- [ ] UNCERTAIN — active_mask.py:373 — `_depth_diff_score` (reachable only via `mode=='depth'`; never configured, default 'rgb') — **DELETE after verifying** `change_mask_mode` never set — DECISION: ____
- [ ] UNCERTAIN — active_mask.py:419 — `_depth_outlier_score` (`mode=='depth_outlier'`, never configured) — **DELETE after verifying** — DECISION: ____
- [ ] UNCERTAIN-transitive — active_mask.py:216 — `combine_object_masks` (sole real call inside dead `prepare_dynamic_update`; exported) — **DELETE after `prepare_dynamic_update` purged** — DECISION: ____
- [ ] MED — active_mask.py:548 — `build_change_mask` `del`s 3 params (`use_rgb`/`filter_radius`/`min_component_size`) plumbed through 3 layers from model config — **DELETE** (params + 3 ChangeMaskConfig fields + 3 model-config fields + plumbing) — DECISION: ____
- [ ] MED — active_mask.py:521 — after `prepare_dynamic_update` purge, make `build_change_mask` private to `compute_change_mask` + inline rgb-only path (drop depth dispatch) — **REFACTOR** — DECISION: ____
- [ ] MED — active_mask.py:521 — neither CDN entry wraps conv2d/SSIM chain in `torch.no_grad()` (safe only because live caller is no_grad) — **REFACTOR** (self-defensive no_grad) — DECISION: ____
- [ ] LOW — change_mask.py:56 — `live_depth_max_m=3.0` desyncs from fusion `DEPTH_MAX_M=2.0` (2-3 m band CDN-flagged but never fillable) — **KEEP+DOCUMENT / wire** — DECISION: ____
- [ ] LOW — change_mask.py:55 — `block_valid_min_frac`/`live_depth_min_m`/`live_depth_max_m` never wired from model config despite docstring claim — **KEEP+DOCUMENT / wire** — DECISION: ____
- [ ] LOW — active_mask.py:545 — stale `a760` min-area in docstring (constant lowered to 76) — **KEEP+DOCUMENT** (fix doc) — DECISION: ____
- [ ] LOW — change_mask.py:41/:52 — orphan triple-quoted strings don't attach to fields; `scene_coverage_threshold` undocumented — **REFACTOR** (move docstrings under fields) — DECISION: ____

---

## SECTION 2 — REST OF CODEBASE (lighter)

### `dynamic_gs/static_gs_model.py` + `static_gs_pipeline.py`
- [ ] **HIGH docs — static_gs_pipeline.py:5-16 / static_gs_model.py:477** — docstrings claim it reuses DynamicGSModel + CPD/TEASER++ Phase 0b; actually StaticGSModel + NDP default — **REFACTOR** (fix all 3 docstrings) — DECISION: ____
- [ ] MED dead-config — static_gs_model.py:181-190 — 7 `change_mask_*` fields, 0 static-instance reads, partial copy (missing `change_mask_mode`) — **DELETE / mark NOT WIRED** — DECISION: ____
- [ ] MED — static_gs_model.py:411 — `_refresh_gaussian_optimizers` `reset_means_optimizer` param never read (body clears ALL state); docstring describes unimplemented conditional — **REFACTOR** (drop param or implement) — DECISION: ____
- [ ] LOW — static_gs_model.py:381-714 — Phase-0b/identity-buffer machinery near-duplicated with DynamicGSModel (must stay byte-identical for warm-restart) — **MERGE** (shared mixin) — see Section 3 — DECISION: ____
- [ ] LOW — static_gs_pipeline.py:123/421-424 — bare `except: pass` ledger-reset; lazily-init loss-EMA attrs (only 1 declared) — **REFACTOR** — DECISION: ____
- [ ] LOW — static_gs_model.py:253 — `import math`/`import time` inside hot per-step callbacks — **REFACTOR** (module top) — DECISION: ____

### `dynamic_gs/utils/sam3d.py` + `sam3d_fusion.py`
- [ ] DEAD — sam3d_fusion.py:453 — `_largest_extent` (0 refs; live helper is `_bbox_diagonal`) — **DELETE** — DECISION: ____
- [ ] HIGH dup — sam3d.py:458/:744 + sam_worker.py — single/multi SAM3D runners near-duplicate (OOM ladder + magic param dicts copy-pasted 3×) — **MERGE** (`_build_inference` + `_run_inference_with_oom_ladder`) — DECISION: ____
- [ ] HIGH — sam3d_fusion.py:1110 — `register_and_fuse_sam3d_object` ~250 LOC (validation, rigid init, 3-way dispatch, plot, save, 4x4, timing) — **REFACTOR** (`_rigid_init`/`_refine`/`_finalize_result`) — DECISION: ____
- [ ] MED — sam3d_fusion.py:1120 — default `registration_backend='cpd'` never effective (callers pass 'ndp') — **REFACTOR** (default 'ndp' or require) — DECISION: ____
- [ ] MED — sam3d_fusion.py:1296 — module docstring + vestigial result fields (`dedup_threshold`/`kept_points`/`visible_source_point_count`) describe disabled CPD+dedup path — **REFACTOR** (update doc) — DECISION: ____
- [ ] MED — sam3d_fusion.py:168 — `reconstruct_mesh_from_points`/`reconstruct_mesh_from_gaussian_ply` exported, no runtime caller (FoundationPose-only, FP unwired) — **DELETE / mark FP-only** — DECISION: ____
- [ ] MED lifecycle — sam3d.py:559 — single-object path reloads ~11 GB model on every OOM retry (multi-object loads once) — **REFACTOR** (load once before loop) — DECISION: ____
- [ ] MED — sam3d.py:449 vs sam3d_fusion.py:46 — p3d↔ns camera convention encoded twice, no link/assert — **REFACTOR** (share + assert) — DECISION: ____
- [ ] LOW — sam3d.py:219 — `_write_runtime_config` overwrites shared yaml in vendored third_party tree per call (race-prone) — **KEEP+DOCUMENT** (single-GPU serial) — DECISION: ____
- [ ] LOW — sam3d.py:28 — hardcoded `_CONDA_ROOT` (has sys.executable fallback) — **KEEP+DOCUMENT** — DECISION: ____

### segmenters (`fastsam_segmentation.py` / `sam3_segmentation.py` / `esam.py`)
- [ ] DEAD — fastsam_segmentation.py:49 — `_compute_iou` (fastsam copy; sam_worker uses sam3 copy, fastsam dedup uses inline IoU) — **DELETE** — DECISION: ____
- [ ] UNCERTAIN-trio — esam.py:156/:133/:121 — `query_esam_mask`/`_run_esam_query`/`_select_esam_mask` (exported+imported but live path uses `query_esam_mask_pair`, which reimplements `_pick` inline) — **DELETE after verifying** no single-image caller — DECISION: ____
- [ ] **MED contract — fastsam_segmentation.py:384 vs sam3_segmentation.py:168** — shared `objects[].score` JSON key means model-confidence (SAM3) vs survivor-softmax-prob (FastSAM); any numeric threshold behaves differently per backend — **REFACTOR** (namespace or normalize) — DECISION: ____
- [ ] MED — sam3_segmentation.py:96 vs :334 — `min_score` default differs function (0.2) vs CLI (0.44) — **REFACTOR** (unify) — DECISION: ____
- [ ] MED — fastsam_segmentation.py:374 — `min_score=0.2` floor fights the default `auto_threshold` path (can drop a valid lone object) — **REFACTOR** (floor opt-in / 0 when auto) — DECISION: ____
- [ ] MED — fastsam_segmentation.py:554/:594 — FastSAM knobs enumerated in 3 places; subprocess silently drops unlisted kwargs — **REFACTOR** (single source) — DECISION: ____
- [ ] LOW dup — fastsam_segmentation.py:49/:55 vs sam3_segmentation.py:50/:59; `_resolve_env_python` (5 modules) — **MERGE** (shared mask-utils + env helper) — DECISION: ____
- [ ] LOW — fastsam_segmentation.py:478 — `infer_raw` RPC wired, no client caller — **KEEP+DOCUMENT** (backend parity) — DECISION: ____

### preseg (`preseg_seed.py` / `static_gs_preseg_pipeline.py`)
- [ ] **MED bug — static_gs_preseg_pipeline.py:296** — `_load_sidecar_into_buffer` silently no-ops (leaves `object_instance_ids` all-zero) on `np.load` failure → trains+saves cache with no ids, no error — **REFACTOR** (hard-fail) — DECISION: ____
- [ ] MED — static_gs_preseg_pipeline.py:303 — labeling assumes dataparser preserves open3d PLY row order; only count-guarded (same-N permutation mislabels every Gaussian) — **REFACTOR / VERIFY+assert** — DECISION: ____
- [ ] MED — static_gs_preseg_pipeline.py:150 — ICP-refine failure silently degrades to raw URDF poses (violates Invariant #3) — **REFACTOR** (hard-fail / flag) — DECISION: ____
- [ ] MED lifecycle — preseg_seed.py:429 — SAM2 video predictor leaked on early-error paths (no try/finally before `del :517`) — **REFACTOR** — DECISION: ____
- [ ] MED — preseg_seed.py:393 — `build_labeled_seed` ~155-line god-function (SAM2/AMG/SAM3/overlay/merge/video/propagate/vote/2 sidecars); mutable-default `amg_cfg=AmgConfig()` — **REFACTOR** — DECISION: ____
- [ ] LOW — preseg_seed.py:251/:520 — `_video_frames` JPEGs never cleaned; `seg_ids.npz` written but never re-read (QA-only) — **KEEP+DOCUMENT / clean** — DECISION: ____
- [ ] LOW — preseg_seed.py:432/:257 — computed-then-discarded `_seg0`, `rgbs`, `_static_dir`/`_num_instances`/`_labeled_instance_count` — **DELETE** unused returns/attrs — DECISION: ____
- [ ] LOW — static_gs_preseg_pipeline.py:346 — `_phase0b_done` repurposed + `P0a.*` timing keys in a method with no phases — **REFACTOR** (rename) — DECISION: ____

### fusion-init / pose-refine / NDP (`rgbd_fusion_init.py` / `icp_pose_refine.py` / `ndp_register.py`)
- [ ] DEAD — rgbd_fusion_init.py:114 — `_make_o3d_with_normals` (0 refs; inlined `estimate_normals` 3×) — **DELETE** — DECISION: ____
- [ ] MED — icp_pose_refine.py:134/:152 — backup created BEFORE rewrite; if rewrite raises, dataset wedged (re-run hard-aborts on backup-collision guard, needs manual `force=True`) — **REFACTOR** (create backup atomically with success) — DECISION: ____
- [ ] MED — rgbd_fusion_init.py:75/:459 — CPU fallback seed uses hardcoded `DEPTH_TRUNC_M=3.0`/`TSDF_VOXEL_M=0.0025`, diverges from online_fusion's 2.0 m / 2-3 mm → same PLY path gets different geometry per path — **REFACTOR** (align to `DGS_TSDF_*` or document) — DECISION: ____
- [ ] MED — rgbd_fusion_init.py:281 — `build_tsdf_seed` ~210-line god-function — **REFACTOR** — DECISION: ____
- [ ] LOW — rgbd_fusion_init.py:104 vs rgbd_decode.py:91 — `_backproject_world` defined twice, same name diff signatures — **REFACTOR** (rename one) — DECISION: ____
- [ ] LOW — rgbd_fusion_init.py:233 — `_load_gripper_keep_mask` returns the inverted (drop) mask despite "keep" name — **REFACTOR** (rename) — DECISION: ____
- [ ] LOW — ndp_register.py:46 — `_setup_seed` sets global `torch.backends.cudnn.deterministic=True`, never restored — **KEEP+DOCUMENT / save-restore** — DECISION: ____

### fusion-runtime (`fusion_runner.py` / `online_fusion.py`) — runs during capture, before tracker threads
- [ ] DEAD — online_fusion.py:367 — `_GpuOnlineFusion._sync` (0 refs; local o3c import dead with it) — **DELETE** — DECISION: ____
- [ ] **MED lifecycle — fusion_runner.py:344 vs online_fusion.py:626** — `transforms.json` mtime bump (M1) present in `stop_and_finalize` but MISSING in `fuse_recorded_dataset` (the DEFAULT `DGS_LIVE_DEFER_TSDF=1` path) → redundant re-fusion may trigger — **VERIFY** against `rgbd_fusion_init._output_is_fresh`, then **MERGE** the finalize-and-write logic — DECISION: ____
- [ ] MED dup — fusion_runner.py:300 vs online_fusion.py:611; fusion_runner.py:84 vs online_fusion.py:594 — finalize/downsample/write + per-frame image-load loop duplicated and divergent — **MERGE** — DECISION: ____
- [ ] MED — online_fusion.py:500/:530 — `OnlineFusion` wrapper bypassed by `profile_fusion.py`/`bench_gpu_fusion.py` (reach `_impl` attrs via wrapper → AttributeError post-2026-06-01 split); `idx` read-only property they try to `+=` — **REFACTOR / retire stale bench scripts** — DECISION: ____
- [ ] LOW — fusion_runner.py:152/:275 — watcher swallows all exceptions; bare-literal 30s/2s drain timeouts can silently drop final keyframes — **REFACTOR** — DECISION: ____
- [ ] LOW — online_fusion.py:84/:482 — `WITH_COLOR` code-edit-only (no env unlike siblings); `finalize` nulls `_slam` with no post-finalize guard — **KEEP+DOCUMENT / guard** — DECISION: ____

### misc (`timing_ledger.py` / `object_picker.py` / `keyframe_filter.py`)
- [ ] DEAD — keyframe_filter.py:33 — `DynamicKeyframeFilter` class (0 instantiations; recorded dropped it, live uses inlined `_KeyframeFilter`; `bulk_filter`/`num_kept`/`reset`/`accept` die with it) — **DELETE** (+ utils/__init__ export) — DECISION: ____
- [ ] UNCERTAIN — timing_ledger.py:86 — `timed` contextmanager (0 callers; all sites use raw `record()`; advertised as ergonomic API) — **KEEP+DOCUMENT or DELETE** — DECISION: ____
- [ ] LOW — timing_ledger.py:100 — `render` does parse+order+aggregate+fold+format in 100 LOC — **REFACTOR if touched** — DECISION: ____
- [ ] LOW — object_picker.py:118/:170 — broad except downgrades corrupt artifacts to fallback id silently; mask-resize recomputed twice — **REFACTOR** — DECISION: ____

---

## SECTION 3 — DATA-LIFECYCLE & ARCHITECTURE (cross-cutting)

### `.pt` warm-cache (`persistence/post_fusion_cache.py`)
- [ ] **MED — post_fusion_cache.py:138** — `load_post_fusion_state` uses `model.load_state_dict(strict=False)` and verifies only the 6 `gauss_params`. The 4 identity buffers + config-sensitive params (sh_degree/bg/camera-opt) silently skip-on-missing → buffer desync against resized N. **ADD** post-load assertions: `num_points==target_n`, `object_flags.shape[0]==target_n`, all 4 buffers present + right length/dtype. Snapshot is not config-tagged (`:18-21`). — DECISION: ____
- [ ] LOW — post_fusion_cache.py:72 vs base.py:548 — `static_state`→`post_fusion_state` legacy fallback duplicated — **MERGE** — DECISION: ____
- [ ] LOW — post_fusion_cache.py:67 — `save_post_fusion_state` failure → `False`, callers (static_gs_pipeline.py:268, preseg:354) don't hard-fail on `ok==False` → surfaces only at next run's `FileNotFoundError` — **REFACTOR** (hard-fail on save failure) — DECISION: ____
- [ ] LOW — post_fusion_cache.py:128 — transient ~2× VRAM spike during gauss_params reallocation (means-grad hook correctly re-bound `:142-143`) — **KEEP+DOCUMENT** — DECISION: ____

### The 4 identity buffers (`object_flags` / `object_instance_ids` / `sam3d_init_target_flags` / `inserted_flags`) — Invariant #8
- [ ] **MED — model.py:1500+ vs static_gs_model.py:381-714** — the insert/delete/subset/resize machinery that keeps the buffers length-synced with `gauss_params` is COPY-PASTED between StaticGSModel and DynamicGSModel and MUST stay byte-identical for warm-restart. Largest structural risk in the repo. **MERGE** into a shared mixin (keep dynamic-only paths off the static model). — DECISION: ____
- [ ] MED — dynamic_gs_model.py:1139 — `delete_gaussian_indices` slices `current_active_mask` only under a length-guard, but slices the 4 persistent buffers with `[keep]` unconditionally (no assertion) — **REFACTOR** (assert all buffers == num_points before slicing) — DECISION: ____
- [ ] MED — base.py:1873 — acknowledged buffer-vs-means desync (`_object_crop_bbox` defensive guard) after some FF delete/insert sequences — **INVESTIGATE root cause** (the guard masks, doesn't fix) — DECISION: ____
- [ ] LOW — static_gs_model.py:372-373 — `load_state_dict` zero-fills missing identity buffers silently (correct for all-zero defaults, but masks a truncated populated save) — **KEEP+DOCUMENT / assert** — DECISION: ____

### SHM lifecycle (publisher + reader)
- [ ] Already in Section 1: publisher unlocked pose/joint lists (`live_ros_publisher.py:762`), seqlock write-order (`:1020`), reader peek-vs-close race (`live_shm_reader.py:373`), leak-by-design never-unlink (`:1295`). **Decide a consistent teardown discipline** (a shared shutdown flag checked by `peek_latest`, explicit close on clean stop). — DECISION: ____

### `self.info` shared mutable (the cross-thread render-state hazard)
- [ ] **HIGH — model.py:2297** — single `self.info` attr written by `get_outputs` on tracker/FF-bg/viser threads, read by 4 mask helpers; safe only under `_model_lock`. The dead `self.info` consumers (in `prepare_dynamic_update` etc.) would reintroduce the race if revived — another reason to delete them. **REFACTOR** to return `info` by value. — DECISION: ____

### Biggest worthwhile refactors (consolidated)
- [ ] Delete `prepare_dynamic_update` chain → frees ESAM (`esam.py:121-156`, `model.py:1909`), optim-mask (`model.py:833/842/1903`), `combine_object_masks`, `_depth_diff/_outlier_score`, and 7+ dead config fields in one sweep. — DECISION: ____
- [ ] Shared identity-buffer mixin (Static/Dynamic) — DECISION: ____
- [ ] Single SAM3D `_run_inference_with_oom_ladder` (sam3d single/multi + sam_worker) — DECISION: ____
- [ ] One shared fusion finalize-and-write helper (fixes the divergent mtime-guard bug) — DECISION: ____
- [ ] Shared mask-utils (`_compute_iou`/`_touches_n_borders`) + `_resolve_env_python` (5 modules) — DECISION: ____

---

## APPENDIX — DO NOT TOUCH (CLAUDE.md invariant-protected; load-bearing weirdness)

These look deletable/odd but are deliberate. **Do not purge without explicit re-review.**

1. **`means` LR = 0.0** in `dynamic_gs_config.py` (Invariant #1) — NOT "effectively 0"; Adam moves means via `.grad` regardless of densification.
2. **Static `camera_optimizer.mode = "off"`** in `dynamic_gs_config.py` (Invariant #2) — NOT `SO3xR3`.
3. **`rewrite_transforms_with_icp.py` + `transforms_urdf_backup.json` + `pose_source` flag** (Invariant #3) — ICP-refined poses; idempotent backup.
4. **`_ZERO_LR_OPTIMIZERS`** in `dynamic_gs_config.py` (Invariant #4) — ALL dynamic-phase gauss-param LRs = 0; the `_mask_means_grad` hook (`model.py:828`) enforces it.
5. **3 monkeypatches in `dynamic_gs/__init__.py:39/66/91`** (Invariant #5) — suppress `ExperimentConfig.save_config` / `dataparser_transforms.json` / tensorboard. The bare `except: pass` is a smell (log it) but the patches themselves are load-bearing. NoSaveTrainer overrides + `outputs/` pre-create are part of this.
6. **Background `(0.86, 0.92, 1.0)`** in StaticGSModel/DynamicGSModel `populate_modules` + viewer default (Invariant #6).
7. **`SamWorkerClient`** persistent worker (Invariant #7) — canonical SAM3/SAM3D/FastSAM live path; the per-call subprocess fallbacks are intentional.
8. **`initialize_object_from_sam3d`** (`dynamic_gs_model.py`) — invariant-protected NO-CALLER (Invariant #8). The DynamicGSModel SAM3D-init copies it reaches are NOT dead by this rule. `sam3d_init_target_flags` all-zeros is the expected state. `object_instance_ids` written by Phase-0b only; `inserted_flags` by Phase-0b + rgbd_decode; `object_flags` by D0 selection only — `=0` in `post_fusion_state.pt` is correct.
9. **viser-direct on `:8081`, never NS viewer `:7007`** (Invariant #9) — `vis="tensorboard"` in every method config; `enable_viser_direct=True` default. The 5 no-op stubs are still safe to delete (Section 1) but the WebGL/handle topology must stay.
10. **`apply_rigid_object_transform_from_reference`** (`model.py`) — the live tracker write-path. Only the BARE `apply_rigid_object_transform` (`:923`) is dead.
11. **`_step_offset=10000`** (base.py:592) — bypasses Splatfacto res/SH schedules so FF inserts back-project with correct intrinsics. Make the failure hard-fail (Section 1) but DO NOT remove the offset itself.
12. **`STATIC_NUM_STEPS=500`, `STATIC_EARLY_STOP_LOSS=0.02`, `OFFICIAL_FILTER_MIN_AREA=76`, `change_mask_downsample_target_side=150`, depth-cap `DEPTH_MAX_M=2.0`, scale-reset cadence (every 10)** — tuned values with dated measurements; do not "tidy" to round numbers.
13. **`current_track_count` (xfeat:456), `LabeledSeed` (preseg:49), `filter_depth_torch` function** — flagged by grep but LIVE (false alarms).
