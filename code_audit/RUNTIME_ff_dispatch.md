# RUNTIME trace — LIVE dynamic-gs feedforward dispatch path

Scope: the `dynamic-gs-live` runtime FF path, from the trainer step down through the
AnySplat background insert, mapping shared state / locks / threads / hazards. Verified by
reading the code (no speculation). Files:

- `dynamic_gs/dynamic_gs_pipeline_base.py` (base, FF dispatcher + cull + CDN)
- `dynamic_gs/dynamic_gs_pipeline_live.py` (live tick + overrides)
- `dynamic_gs/dynamic_gs_model.py` (gauss_params mutations + render)
- `dynamic_gs/utils/anysplat_decode.py` (reproject / ICP / worker IPC)
- `dynamic_gs/utils/active_mask.py`, `dynamic_gs/change_detection/change_mask.py` (CDN)

Default config for live: `enable_feedforward_inpaint="anysplat_decode"` (base config),
so the **AnySplat** path is the live one. The `rgbd_decode` path (`_run_feedforward`,
`decode_component_to_gaussians`, `_feedforward_delete_in_region`) is only reached when the
mode is `rgbd_decode`; it is traced for completeness because it has a distinct hazard
(`model.info`), but it is NOT on the live default path.

## Threading model (3 threads + N subprocess)

| Thread | What runs | Mutates gauss_params? | Renders? |
|---|---|---|---|
| **Main / trainer** | `get_train_loss_dict` → `_tracker_tick` (live) → XFeat estimate → `apply_rigid_object_transform_from_reference`; dispatches FF | YES (rigid transform: in-place means/quats writes) | YES (object-mask render under lock) |
| **FF bg thread** (`ff-recurring`, daemon) | `_feedforward_threaded` → CDN render + clean + cull (delete) + AnySplat reproject + `insert_inpaint_gaussians` | YES (delete + insert: REASSIGN gauss_params Parameters) | YES (CDN full-scene render, object-mask render) |
| **viser render thread** (`ViserDirectScene`) | per client, `model.get_outputs` server-side then `set_background_image` | no | YES (reads gauss_params) |
| AnySplat worker subprocess | model forward; communicates via stdin/stdout JSON + `/dev/shm` npz | no (separate process) | — |

Single-in-flight guarantee: only ONE FF bg thread can exist at a time
(`_anysplat_slot_lock`, non-blocking acquire in `_dispatch_feedforward_async`
base.py:2509). So FF-vs-FF concurrency is impossible; the real concurrency is
**FF-bg vs main-tracker vs viser-render**, mediated by the re-entrant `_model_lock`.

## Annotated call chain (live, AnySplat, recurring Mode B)

```
[MAIN THREAD]
Trainer.train step
└─ get_train_loss_dict(step)                              base.py:975
   ├─ _tracker_tick(step)  [LIVE override]                live.py:243
   │  ├─ _shm_sub.peek_latest()                           live.py:250   (reads ROS SHM)
   │  ├─ cameras_from_live_frame / _batch_from_live_frame live.py:271-272
   │  ├─ _invalidate_object_mask_cache()                  live.py:275  -> _obj_mask_cache=None
   │  ├─ _apply_motion_estimator(camera, batch)           live.py:303 / base.py:2053
   │  │  ├─ _render_object_mask_cached(camera)            base.py:2070,1656
   │  │  │  └─ LOCK acquire(_model_lock) ... render_object_mask ... release   base.py:1676-1682
   │  │  ├─ _object_crop_bbox(camera)        ** reads model.means UNLOCKED **  base.py:1856,1876
   │  │  ├─ motion_estimator.estimate_and_advance(...)     base.py:2102
   │  │  └─ with _model_lock: apply_rigid_object_transform_from_reference()    base.py:2163-2166
   │  │     └─ in-place writes gauss_params["means"/"quats"][object_mask]      model.py:1006-1007
   │  │     then _invalidate_object_mask_cache()           base.py:2168
   │  ├─ _ff_due_this_tick = _recurring_ff_due(tick+1)     live.py:314   (PURE gate; cdn left None)
   │  ├─ _latest_tracker_frame = {... cdn=None ...}        live.py:323
   │  ├─ _build_viser_direct_handles / _push_* / _force_viewer_rerender  live.py:333-336
   │  └─ _on_tracker_frame(camera, batch, None, is_first)  live.py:338,414
   │     └─ if _ff_due_this_tick: _dispatch_feedforward_async(_latest_tracker_frame,"recurring")  live.py:432
   │        ├─ _anysplat_slot_lock.acquire(blocking=False) base.py:2509   (skip if held)
   │        ├─ _last_feedforward_wall_time = now           base.py:2512
   │        └─ Thread(_feedforward_threaded, daemon).start()  base.py:2513
   └─ return {}, {"main_loss": zero}, {}                   base.py:998

[FF BG THREAD] _feedforward_threaded(target_frame,"recurring")   base.py:2519
├─ if target_frame["cdn"] is None:                              base.py:2531
│   with _model_lock: target_frame["cdn"]=_compute_tick_cdn(cam,btch)  base.py:2535-2536
│     └─ _compute_tick_cdn -> _render_from_camera (re-acquires _model_lock, re-entrant)  base.py:1739,1698
│        + _render_object_mask_cached (lock) + _compute_change_mask (pure tensor)        base.py:1756,1761
├─ _run_feedforward(target_frame,"recurring")                   base.py:2539
│  └─ (mode anysplat) _run_feedforward_anysplat(...)            base.py:2561,2981
│     ├─ prerendered_obj_mask = _render_object_mask_cached      base.py:3010 (lock)
│     ├─ cdn_clean = _feedforward_clean_cdn(...)                base.py:3019 (pure tensor on cdn + objmask)
│     ├─ gt_depth = model._get_gt_depth(batch)                  base.py:3031
│     ├─ cdn_clean,n = _feedforward_cull_then_reclean_cdn(...)  base.py:3053,2855
│     │   ├─ with _model_lock: _feedforward_cull_in_front_of_depth(...)  base.py:2876-2880
│     │   │     └─ delete_gaussian_indices(...)  REASSIGNS gauss_params   model.py:1116,1137
│     │   └─ if n>0: _compute_tick_cdn (re-render) + _feedforward_clean_cdn  base.py:2885-2891
│     ├─ select_top_n_components_filtered(cdn_clean,n=256)      base.py:3091 (pure)
│     ├─ _resolve_anysplat_context_image_paths [LIVE] dumps /dev/shm png  live.py:438
│     ├─ scene_intr / scene_c2w_np = _scene_c2w_for_frame [LIVE reads _latest_tracker_frame.camera]  live.py:458
│     ├─ _anysplat_crop_windows(cdn_np)                         base.py:3138,3165 (pure numpy)
│     └─ _anysplat_bg_run(bg_args)  [INLINE, same bg thread]    base.py:3163,3209
│        ├─ depth_filter.filter_depth_torch (live: _filter_depth_at_ff=True)  base.py:3248-3254
│        ├─ if icp_enabled:
│        │   ├─ with _model_lock: means_all_t = gauss_params["means"].detach()  base.py:3281-3282
│        │   ├─ frustum cull (pure tensor on snapshot)          base.py:3284-3299
│        │   └─ icp_refine_scene_c2w(target_xyz_gpu=...)        base.py:3322 / anysplat_decode.py:525
│        ├─ for win in crop_windows:
│        │   ├─ cv2.imwrite /dev/shm crop png                   base.py:3368
│        │   ├─ worker.inference([crop_png], out_npz)           base.py:3374 (subprocess IPC, blocking)
│        │   ├─ pickle.load(out_npz)                            base.py:3400-3401
│        │   └─ reproject_anysplat_to_scene(...)                base.py:3405 / anysplat_decode.py:624
│        ├─ concat + GPU voxel dedup                            base.py:3430-3494
│        ├─ with _model_lock: insert_inpaint_gaussians(...)     base.py:3497-3506
│        │     └─ REASSIGNS all 6 gauss_params Parameters + resizes 4 ID buffers  model.py:1216-1227
│        └─ _viser_direct_register_ff_insert(inserted_ids)      base.py:3508,1571 (request_render)
└─ finally: _anysplat_slot_lock.release()                       base.py:2542
```

Invariants respected by this path (verified, do NOT break in the purge):
- **Inv #4** (dynamic = all gauss LR 0): no optimizer step touches gauss_params; only the
  rigid transform + FF insert/delete mutate. `get_train_loss_dict` returns a zero dummy.
- **Inv #8** (identity buffers): `insert_inpaint_gaussians` sets `object_flags=1`,
  `object_instance_ids=999`, `inserted_flags=1` for new rows (model.py:1224-1226);
  `sam3d_init_target_flags` left 0 (never written — matches doc). Cull eligibility is
  `instance_ids ∈ {0,999}` only — tracked-object id is never deleted.
- **Inv #9** (viser-direct not NS viewer): FF only calls `request_render()` to wake the
  viser render thread; no `get_outputs_for_camera` for visualization.

## Shared-state access table

Legend for "racy?": **N** = protected by `_model_lock` on all writers+readers, or the
read is of a detached snapshot; **Y** = a writer or reader is outside the lock and can
observe a torn/resized tensor; **benign** = technically unsynchronized but safe under the
CPython GIL or by disjoint keys.

| Shared state | Site (file:line) | Thread | Lock held? | Racy? |
|---|---|---|---|---|
| `gauss_params["means"/"quats"]` in-place write (rigid xform) | base.py:2163 / model.py:1006-1007 | main | `_model_lock` | N |
| `gauss_params[*]` REASSIGN (delete) | base.py:2876 → model.py:1135-1137 | FF bg | `_model_lock` | N |
| `gauss_params[*]` REASSIGN (insert) | base.py:3497 → model.py:1216-1227 | FF bg | `_model_lock` | N |
| `gauss_params["means"].detach()` snapshot for frustum cull | base.py:3281-3282 | FF bg | `_model_lock` (read only); used after release | N (detached copy) |
| `model.means` read in `_object_crop_bbox` | base.py:1873-1876,1905 | main | **NONE** | **Y** (see H1) |
| `model._tracked_object_mask()` / `object_instance_ids` read | base.py:1866 (crop_bbox) | main | **NONE** | **Y** (H1) |
| `render_object_mask` reads means/quats/scales/opacities/features | model.py:2166-2184 | main + FF bg | caller `_render_object_mask_cached` holds `_model_lock` | N |
| `model.get_outputs` (CDN render) reads gauss_params | base.py:1698-1699 | FF bg | `_render_from_camera` holds `_model_lock` | N |
| `model.get_outputs_for_camera` (viser) reads gauss_params | model.py:708-739 | viser render | `_render_lock_ctx` = `_model_lock` (attached base.py:520) | N |
| `model.info` (means2d/radii) — rgbd cull only | active_mask.py:626-627 via base.py:2749 | FF bg (rgbd mode) | `_model_lock` NOT held around the projection read | **Y** (H2, rgbd path only) |
| `_obj_mask_cache` | base.py:1666,1680,1686 / live.py:275 | main writes+reads; FF bg reads+writes | partial (`_render_object_mask_cached` under lock; invalidate not) | **Y** (H3) |
| `_latest_tracker_frame` (dict ref) | live.py:323 (write main); base.py:2531/2574/2998 (read FF bg) | main writes; FF bg reads | NONE | benign* (H4) |
| `_latest_tracker_frame["cdn"]` mutated | base.py:2536 (FF bg writes the dict it was handed) | FF bg | NONE | benign* (H4) |
| `_latest_live_rgb_bgr` | live.py:321 (write main) / live.py:446 (read FF bg) | main+FF bg | NONE | **Y** (H5) |
| `_last_feedforward_wall_time` | base.py:2512 (write main) / base.py:962 (read main) | main only | n/a | N |
| `_feedforward_call_counter` | base.py:2994-2995 | FF bg only | n/a | N |
| `self._timing[...]` defaultdict append | base.py many | main (DN.*) + FF bg (FF.*) | NONE | benign (GIL + disjoint keys, H6) |
| `_anysplat_slot_lock` | base.py:2509 acq / 2542 rel / 658 drain | main acq, FF bg rel | — | N (see H7 ownership) |
| `/dev/shm/dgs_live_ff_frame_<pid>.png` | live.py:454-455 (write FF bg via dispatch) | FF bg | serialized by slot lock | N |
| `/dev/shm/anysplat_crop_<pid>_<wi>.png`, `anysplat_ipc_<pid>_<wi>.npz` | base.py:3367-3369 | FF bg | serialized by slot lock | N (but H8 cleanup) |
| `_d0_tracked_instance_id` / `_reference_object_means/_quats` | model.py:975-983 | main (D0/reseed) | `_model_lock` at reseed (base.py:1164) | N |

\* "benign" for `_latest_tracker_frame`: the main thread REPLACES the dict reference each
tick (live.py:323) while the FF bg thread holds its OWN reference passed at dispatch time
(`target_frame` arg). So the bg thread never sees a half-written dict — it sees a complete
prior dict. But see H4 for the staleness consequence.

## Hazards

### H1 — `_object_crop_bbox` reads `model.means` on the main thread WITHOUT `_model_lock` (RACE)
- **Where:** `base.py:1873-1876` (and the projection at 1905). Called from
  `_apply_motion_estimator` (base.py:2083) on the **main tracker thread**, every tick when
  `xfeat_crop_to_object_bbox` is True (it is True by default per CLAUDE.md).
- **Hazard:** the FF bg thread can be inside `insert_inpaint_gaussians` /
  `delete_gaussian_indices`, which **reassign** `gauss_params["means"]` to a new Parameter
  and resize the 4 ID buffers. `_object_crop_bbox` does `n_means = model.means.shape[0]`
  then `obj_mask = model._tracked_object_mask()` (reads `object_instance_ids`) then
  `model.means[obj_mask]`. These are three separate unsynchronized reads. If an insert/
  delete lands between them, `obj_mask` (sized to the old/new `object_instance_ids`) can
  mismatch the current `means` row count → `IndexError`/CUDA index assert at line 1876,
  or a silently wrong crop. The guard at 1874 (`obj_mask.shape[0] != n_means → None`)
  reduces but does not close the window: `object_instance_ids` and `means` are reassigned
  in separate Python statements inside the insert (model.py:1219 vs 1223-1225), so a read
  interleaved there sees a transient `means.shape != object_instance_ids.shape`.
- **Why it usually survives:** the insert/delete critical sections are short (~ms) and FF
  fires every 10 ticks, so the collision probability is low — but at the documented
  459k→1.29M growth the insert resize is not instantaneous.
- **Fix:** wrap the `model.means` / `_tracked_object_mask` reads in `_object_crop_bbox`
  with `with self._viser_lock_ctx():` (snapshot `means_obj`, `obj_mask`, and the count under
  one lock acquisition), matching the discipline already used in `_render_object_mask_cached`
  and `_apply_motion_estimator`'s transform. Cheap (microseconds) and re-entrant.

### H2 — `_feedforward_delete_in_region` reads stale/cross-thread `model.info` (RACE, rgbd path only)
- **Where:** `base.py:2749` → `extract_projected_centers_and_radii(model.info, model.num_points)`
  (active_mask.py:618). `model.info` is the rasterizer metadata from the **most recent**
  `get_outputs` call, which can be the viser render thread's or the CDN render's, not the
  pre-delete render at base.py:2677. The code re-renders right before (base.py:2677) to
  refresh `model.info`, but that render is NOT atomic with the read — a viser render on
  another thread can overwrite `model.info` in between, and `model.info["means2d"]` then has
  a row count for a DIFFERENT gauss_params size → `extract_projected_centers_and_radii`
  raises `ValueError` (count mismatch, active_mask.py:639-642), which is caught and the
  delete is skipped (acceptable) — but if the count *coincidentally* matches a stale info,
  it deletes the WRONG Gaussians.
- **Scope:** only on `enable_feedforward_inpaint="rgbd_decode"`. The live default is
  `anysplat_decode`, whose cull (`_feedforward_cull_in_front_of_depth`) uses direct
  projection of `model.means` (base.py:2818) and does NOT touch `model.info`. So **not on
  the live hot path** — but it is a latent bug if anyone flips the mode.
- **Fix:** either (a) make `_render_from_camera` + `model.info` read atomic under
  `_model_lock` in `_feedforward_delete_in_region`, or (b) deprecate the `model.info` path
  and use the direct-projection cull everywhere (the anysplat path already proves it works
  without `model.info`). Given the purge, (b) removes a whole class of `model.info` aliasing.

### H3 — `_obj_mask_cache` is shared by main + FF bg threads with partial locking (RACE)
- **Where:** written/read in `_render_object_mask_cached` (base.py:1666-1683) under
  `_model_lock`; but **invalidated** without the lock at `live.py:275`
  (`_invalidate_object_mask_cache`, main thread, tick start) and base.py:2168 (main, after
  rigid xform). The FF bg thread reads it via `_render_object_mask_cached` at base.py:3010,
  2477, 1756.
- **Hazard:** the check-then-fill in `_render_object_mask_cached` is:
  `if self._obj_mask_cache is None: <lock> render; set cache`. The `is None` test is OUTSIDE
  the lock. Main thread sets `_obj_mask_cache=None` (live.py:275) while the FF bg thread is
  between its `is None` check and its locked render → both can render; or the FF bg thread
  caches a mask rendered against the OLD object pose, then the main thread moves the object
  (rigid xform) and the FF reuses the stale mask for CDN cleaning. Functionally the worst
  case is a slightly-misaligned object-exclusion mask for one FF call (it feeds the
  documented "misplacement ring" churn), not a crash.
- **Fix:** move the `is None` test inside the locked region (double-checked locking), and/or
  give the cache a frame-id stamp so a bg consumer that reads it knows whether it matches the
  frame it is decoding. Low severity (cosmetic/churn, not corruption).

### H4 — FF bg thread decodes against a frame snapshot that the main thread has since advanced (STALENESS, by design but lossy)
- **Where:** `_latest_tracker_frame` is replaced every tick (live.py:323) but the FF bg
  thread holds the reference captured at dispatch (`target_frame` arg). `_scene_c2w_for_frame`
  (live.py:458) reads `self._latest_tracker_frame["camera"]` — the CURRENT one, NOT the one
  the FF was dispatched with. So the AnySplat reproject pose (`scene_c2w_np`) can be from a
  LATER camera than the CDN / depth the FF is decoding, because the tracker kept ticking
  while the ~200 ms FF ran.
- **Hazard:** the inserts are placed using a camera pose newer than the depth/CDN they were
  selected from → spatial mismatch (a few mm-cm of camera motion over the FF latency),
  feeding the documented CDN churn loop. Not a crash; a correctness/quality drift.
- **Fix:** snapshot the dispatch camera into `bg_args` (it is already passed as
  `camera`=base.py:2998/3225) and make the LIVE `_scene_c2w_for_frame` use the passed-in
  camera instead of re-reading `self._latest_tracker_frame["camera"]`. The base already
  threads `camera` into `bg_args` — the live override just ignores it. Medium severity
  (this is plausibly part of the churn the docs chase).

### H5 — `_latest_live_rgb_bgr` read on FF bg thread, written on main thread (RACE)
- **Where:** main writes at live.py:321 every tick; FF bg reads at live.py:446 inside
  `_resolve_anysplat_context_image_paths` (called from `_run_feedforward_anysplat`).
- **Hazard:** the FF bg can `cv2.imwrite` an RGB frame that is NEWER than the
  CDN/depth/camera it was dispatched with (same root cause as H4 — the reference is a bare
  attribute, replaced each tick). At worst the AnySplat source image is one tick ahead of the
  depth used to back-project it → small reproject error. No tearing (reference swap is
  atomic under GIL), so no crash.
- **Fix:** same as H4 — snapshot `latest.rgb_bgr` into the dispatched frame dict /
  `bg_args` at dispatch time rather than reading the live attribute on the bg thread.

### H6 — `self._timing` defaultdict written by two threads (BENIGN)
- **Where:** main thread appends DN.* keys (base.py:2062…), FF bg appends FF.* keys
  (base.py:2601…). Disjoint key sets. `defaultdict.__getitem__` may create a key on either
  thread.
- **Assessment:** CPython GIL makes `list.append` and dict-slot insert individually atomic;
  keys are disjoint so no lost updates. No fix needed, but if the GIL-free build is ever a
  target, guard with a lock. Low.

### H7 — `_anysplat_slot_lock` acquired on main, released on FF bg (correct, but fragile)
- **Where:** acquired non-blocking in `_dispatch_feedforward_async` (base.py:2509, main),
  released in `_feedforward_threaded`'s `finally` (base.py:2542, FF bg). `_anysplat_bg_run`
  deliberately does NOT release (base.py:3549-3553 — comment confirms ownership is the
  wrapper's).
- **Exception safety:** `_feedforward_threaded` wraps `_run_feedforward` in try/finally and
  releases unconditionally — GOOD: even if `_run_feedforward_anysplat` raises (e.g. worker
  death, reproject error), the slot is released. The `release()` is itself wrapped in
  `try/except RuntimeError` (base.py:2543) to swallow a double-release. **One gap:** if
  `Thread(...).start()` at base.py:2513 raises (OS thread-exhaustion), the slot acquired at
  2509 is never released and ALL future FF is permanently skipped ("previous FF still in
  flight" forever). Very unlikely but unbounded.
- **Fix:** wrap the `Thread.start()` in try/except that releases the slot on failure.
  Low severity.

### H8 — `/dev/shm` crop/ipc files: per-window leak + only window-0 cleaned at exit (LEAK)
- **Where:** FF bg writes `anysplat_crop_<pid>_<wi>.png` (base.py:3367) and
  `anysplat_ipc_<pid>_<wi>.npz` (base.py:3369) for each window `wi`. The atexit cleanup
  `_cleanup_anysplat_ipc_file` (base.py:640-648) only unlinks
  `anysplat_ipc_<pid>.npz` — the NON-indexed name, which this path never writes. So every
  `anysplat_crop_*` and `anysplat_ipc_*_<wi>.npz` accumulates in tmpfs for the process
  lifetime and is never cleaned at exit. Also `dgs_live_ff_frame_<pid>.png` (live.py:454)
  is never cleaned.
- **Hazard:** unbounded /dev/shm growth across a long live session is bounded by reuse
  (fixed filenames per `(pid, wi)`, overwritten each call), so it is NOT unbounded growth —
  it is a fixed handful of stale files left behind at exit. Low severity (tmpfs litter).
- **Fix:** update `_cleanup_anysplat_ipc_file` to glob `anysplat_*_<pid>*` and
  `dgs_live_ff_frame_<pid>.png` and unlink all. Trivial.

### H9 — per-FF-call CPU↔GPU sync + host allocations (PERF, not correctness)
- **Where:** `_anysplat_bg_run`: `torch.cuda.synchronize()` is NOT called here, but the
  reproject path does many `.detach().cpu().numpy()` round-trips (base.py:3111 sensor depth,
  3116 scalars, 3134 cdn). `reproject_anysplat_to_scene` is pure numpy on CPU (anysplat_decode.py:624+)
  and allocates several full-N arrays per call per window. ICP does a GPU→Open3D dlpack
  handoff per call (anysplat_decode.py:585-588). All on the FF bg thread, so they don't block
  the tracker tick directly, but they SHARE the GPU (no MPS) and the `_model_lock` windows
  (frustum-cull read 3281, insert 3497) serialize against the tracker's `_render_object_mask_cached`.
- **Assessment:** consistent with the documented "FF contends with tracker on shared GPU".
  Not a bug; flagged so the purge doesn't "optimize" by moving these back onto the main
  thread. The DN.3k_model_lock_wait timing key (base.py:1678) exists precisely to measure
  the tracker stall caused by these locked FF windows.

### H10 — `_compute_tick_cdn` swallows render failure → FF silently no-ops (DEAD-ish BRANCH)
- **Where:** `_compute_tick_cdn` returns `None` on render exception (base.py:1740-1742);
  `_feedforward_threaded` sets `target_frame["cdn"]=None`; `_run_feedforward` then logs
  "no CDN this tick — skip" (base.py:2558-2560) and returns. So a transient render failure
  silently drops an FF firing. Benign (next cadence fires again) but means FF coverage can
  silently degrade if renders fail repeatedly. Low; documented here so it isn't mistaken for
  "FF never runs".

## Summary of lock discipline (the load-bearing part)

`_model_lock` is a single re-entrant RLock shared by: every gauss_params mutation (rigid
xform main-thread; delete+insert FF-bg-thread), every `model.get_outputs*` render (CDN on
FF bg, viser render thread via `attach_render_lock` base.py:520, object-mask render), and
reseed/D0 capture. The discipline is **correct and complete for the writers/renderers**:
every Parameter REASSIGN (the dangerous resize) and every rasterization that reads
gauss_params is inside the lock. RLock re-entry is used intentionally
(`_feedforward_threaded` holds it around `_compute_tick_cdn`, which re-enters via
`_render_from_camera`; the cull holds it around `delete_gaussian_indices`).

**The two genuine gaps are reads that bypass the lock on the main thread:** H1
(`_object_crop_bbox`) is the only one on the live hot path and the only one that can crash;
H3 (`_obj_mask_cache` check-then-act) is a quality/churn issue. H2 (`model.info`) is a real
race but only on the non-default rgbd mode. Everything else is staleness (H4/H5 — the FF bg
decodes a frame the tracker has moved past) or tmpfs litter (H8).
