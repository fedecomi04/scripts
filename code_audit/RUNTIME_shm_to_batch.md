# RUNTIME trace: ROS publisher → SHM → LiveShmSubscriber → LiveDynamicGSPipeline batch

Scope: the **ingest path only** of the live dynamic-gs pipeline — how a synced
(rgb, depth, mask, pose) tuple travels from the ROS publisher subprocess, through
POSIX shared memory, into the per-tick `batch` dict the tracker/FF consume. All
file:line refs are from the state of the tree on the trace date. Source untouched.

Two OS processes are involved:

| Process | Conda env | Python | Role |
|---|---|---|---|
| **Publisher** | `dynamic_gs_ros` | 3.8 | `live_ros_publisher.py` — owns ROS subscribers, the SHM segment, the disk recorder. |
| **Reader / trainer** | `dynamic_gs` | 3.12 | `ns-train dynamic-gs-live` — `LiveShmSubscriber` attaches the SHM, `LiveDynamicGSPipeline` runs the tracker. |

---

## (a) Annotated call chain

### Process boundary + SHM creation (publisher side)

```
LiveShmSubscriber.__init__                       live_shm_reader.py:267
  └─ _spawn_publisher(...)                        live_shm_reader.py:162-249
       → subprocess.Popen(bash -c "source ROS && exec python live_ros_publisher.py ...")
         stdin=PIPE, stdout=PIPE, stderr=<logfile>     :238-245
         env: LD_LIBRARY_PATH/CPATH/CUDA_HOME STRIPPED  :225-226 (load-bearing; see hazards)
  └─ _read_response(ready_timeout=30s)            live_shm_reader.py:291  (blocks on publisher stdout)

[publisher process]
_main()                                           live_ros_publisher.py:1403
  ├─ if --wipe-live-root: _wipe_live_root()        :1415  (rmtree everything except 2 timing sidecars)
  ├─ LivePublisher.__init__                        :458
  │   ├─ _wait_for_camera_info_primed(20s)         :404  (subscribe RGB to wake Gazebo lazy publisher)
  │   │     fallback chain to transforms.json / ~/.cache  :503-539
  │   ├─ slot_bytes, offsets = _slot_layout(H,W)   :591  / :234
  │   ├─ UNLINK any stale SHM with same name       :593-602  (prev-crash cleanup)
  │   ├─ self.shm = SharedMemory(create=True, size=total)  :603  (PUBLISHER OWNS THE SEGMENT)
  │   ├─ _write_header(... ready=1, shutdown=0)    :607
  │   ├─ build self._slot_views (np.frombuffer on shm.buf, zero-copy)  :616-638
  │   ├─ subscribe joint_states / gazebo_pose / rgb(/compressed) / depth(raw)  :644-677
  │   ├─ _sync = ApproximateTimeSynchronizer([rgb, depth], slop=0.02)  :672  → registerCallback(_on_synced)
  │   └─ start_worker()                            :685  → Thread(_worker_loop, daemon)  :876
  └─ _send_response({"event":"ready", ... slot_bytes, header_bytes})  :1431
  └─ for line in sys.stdin: <control command loop>  :1469  (capture_anchor/start_recording/...)
```

### Frame production (publisher, two threads)

```
[rospy receive thread]  _on_synced(image_msg, depth_msg)   :847
  └─ self._frame_queue.put_nowait((image, depth, t))  :859  (maxsize=4, drop-oldest on full :862-869)
     — MINIMAL: enqueues raw ROS msgs only, returns immediately.

[publisher worker thread "dgs-publisher-worker"]  _worker_loop  :881
  └─ _process_synced_pair(image_msg, depth_msg)   :972
       ├─ c2w = _interpolate_c2w(stamp)            :975 / :783  (slerp gazebo poses; lazy RobotMaskGenerator)
       ├─ rgb_bgr = _decode_compressed_rgb(jpeg)   :982 / :822
       ├─ depth_m = _decode_raw_depth(32FC1→f32 m) :983 / :828
       ├─ if ZED_NOISE.enabled(): depth_m = apply_zed_depth_noise(depth_m)  :984-985  ← noise injected HERE (raw)
       ├─ mask_keep = _mask_gen._render_robot_exclusion_mask(...)  :999
       ├─ with _state_lock: _frame_seq += 1; self._latest = _StoredFrame(...)  :1006-1013
       ├─ SHM WRITE (seqlock):                     :1015-1027
       │     slot_idx = seq % NUM_SLOTS            :1015
       │     slot["seq"][0] = seq                  :1020  ← tag FIRST
       │     slot["pose"/"stamp"/"rgb"/"depth"/"mask"][:] = ...  :1021-1025  ← payload
       │     struct.pack_into latest_seq = seq     :1027  ← publish LAST
       ├─ replay tap (if --record-replay)          :1032-1036
       └─ if should_write: _write_frame_to_disk()  :1041  ← DEPTH FILTERED here (recorder only), see below
```

**Depth-filter split (important for the purge):**
- Tracker batch depth is **RAW** (no median/bilateral) — see `_batch_from_live_frame` and `_filter_depth_at_ff=True`.
- FF-insert depth is filtered **on the FF bg thread** at `_anysplat_bg_run` `dynamic_gs_pipeline_base.py:3248-3254`.
- The **on-disk static-recording** depth is filtered independently in `_write_frame_to_disk` `live_ros_publisher.py:1129-1135` (writes raw to `depth_raw/`, filtered to `depth/`).
- The SHM depth payload itself is **never** median/bilateral-filtered — only ZED-noise-corrupted (`:985`). CLAUDE.md's "depth filtered once at the batch source for live" claim is **stale**: in live the batch carries raw depth and the FF filters its own copy. (See hazard H8.)

### Frame consumption (reader / trainer, main thread)

```
[trainer MAIN thread]  Trainer.train() loop → pipeline.get_train_loss_dict(step)  base:975
  └─ _tracker_tick(step)                          dynamic_gs_pipeline_live.py:243
       ├─ latest = self._shm_sub.peek_latest()    :250  → live_shm_reader.py:373
       │     SEQLOCK READ:                          :386-414
       │       s_pre = latest_seq                    :387
       │       if s_pre==0: return None              :388
       │       slot = slot_views[s_pre % num_slots]  :390
       │       if slot.seq != s_pre: retry           :393-396
       │       rgb/depth/mask/pose = np.array(slot[..], copy=True)  :398-402  ← THE per-tick copy
       │       if latest_seq==s_pre AND slot.seq==s_pre: return LiveFrame(...)  :404-410
       ├─ sim-clock-reset detect + stamp dedup     :253-266
       ├─ camera = cameras_from_live_frame(latest) :271  → live_shm_reader.py:605  (c2w[:3,:4]→Cameras, .to(device))
       ├─ batch = _batch_from_live_frame(latest)   :272  → live:523
       │     rgb_t  = torch(rgb[...,::-1]).float()/255  :531  (BGR→RGB, ascontiguousarray copy → GPU)
       │     depth_t= torch(depth_m).float().unsqueeze(-1)  :537  (RAW depth → GPU)
       │     mask_t = torch(mask_keep).bool().unsqueeze(-1) :538
       ├─ D0 bootstrap (is_first) OR _apply_motion_estimator  :294-303
       ├─ _ff_due_this_tick = _recurring_ff_due(...)  :314  (cdn deferred to FF thread; cdn=None)
       ├─ self._latest_tracker_frame = {camera, cdn, batch, stamp}  :323-329  ← single-slot handoff to FF thread
       ├─ viser-direct pushes + force rerender     :333-336
       └─ _on_tracker_frame → _dispatch_feedforward_async(...)  :338 / :432
            └─ if _anysplat_slot_lock.acquire(blocking=False): Thread(_feedforward_threaded)  base:2509-2516
```

### SHM lifecycle summary

| Event | Publisher | Reader |
|---|---|---|
| **create** | `SharedMemory(create=True)` `:603` after unlinking stale `:595-597` | — |
| **attach** | — | `SharedMemory(create=False)` `live_shm_reader.py:314`; then `resource_tracker.unregister` `:317` so reader atexit won't try to unlink |
| **read** | writes slots in-place | lock-free seqlock `peek_latest` `:386` |
| **free** | process exit releases memoryview refs; `shutdown()` deliberately does **not** clear `_slot_views` (`:1286-1293`) nor `unlink()` (`:1296-1299`) | `close()` drops `_slot_views=[]` then `_shm.close()` `:587-591` |
| **unlink** | only at **next startup** (`:595-597`), never at shutdown | never (unregistered from resource_tracker) |

So: **the publisher owns the segment for its whole life and the OS name persists after it exits**; the next publisher launch unlinks the stale name and recreates. On a crash the segment leaks under `/dev/shm/dgs_live_shm` until the next run reclaims it (intentional — see hazard H1).

---

## (b) Shared-state access table

Legend for thread: **PRX**=publisher rospy receive thread, **PWK**=publisher worker thread, **MAIN**=reader trainer main thread, **FF**=reader FF bg thread, **VIS**=reader viser-direct render thread, **STDIN**=reader stdin stop-watcher, **CTRL**=publisher stdin control loop (its own thread = main of publisher).

| State | Site (file:line) | Thread(s) | Lock held? | Racy? |
|---|---|---|---|---|
| `_frame_queue` (Queue, maxsize 4) | put `:859` / get `:888` | PRX (put), PWK (get) | Queue-internal | No (Queue is thread-safe; drop-oldest at `:862` is a benign 2-op race, see H6) |
| `_latest` (`_StoredFrame`) | write `:1010` / read `:1055`,`:1493`,`:1508`,`:1516` | PWK (write), CTRL (read) | `_state_lock` on all sites | No |
| `_frame_seq` | `:1007` / `:969`,`:1051` | PWK, CTRL | `_state_lock` | No |
| SHM slot payload | write `:1020-1025` / read `:398-402` | PWK (write), MAIN (read) | **none** (cross-process seqlock) | Mitigated by seqlock — see H2 |
| SHM `latest_seq` hdr | write `:1027` / read `:387`,`:404` | PWK / MAIN | none | Mitigated by seqlock |
| SHM `slot["seq"]` | write `:1020` / read `:393`,`:405` | PWK / MAIN | none | Mitigated by seqlock |
| `_joint_state_times/_positions`, `_gazebo_pose_*` | append `:760-781` / read `:784-820` | PRX (pose/joint cbs), PWK (interp) | **none** | **Yes — H3** (list insert vs read across threads) |
| `_mask_gen` (lazy) | `:811`,`:992` | PWK only (and `_interpolate_c2w` also PWK) | none | No (single thread) |
| `_record_active/_record_dir/_record_meta/_record_frames_written/_inflight_writes` | `:988`,:1071,`:1108-1156` | PWK, CTRL | `_record_lock` (+`_inflight_cv`) | No |
| `_slot_views` (publisher) | build `:616` / read `:1016` / NOT cleared on shutdown `:1286` | PWK | none | No (built once pre-worker) |
| `_replay_*` | `:1032`,`:944-961`,`:1302-1311` | PWK (enqueue), replay-writer thread, CTRL/shutdown | none on counters | low (H7) |
| `_shm` / `_slot_views` (reader) | build `:328` / read `:391` / clear in close `:587` | MAIN, FF(indirect via peek? no — only MAIN calls peek) | none | H4 (close vs in-flight peek) |
| `_last_processed_stamp_sec` | live:266 | MAIN only | none | No |
| `_latest_tracker_frame` | write MAIN `:323` / read FF `_feedforward_threaded` `:2531`, `_scene_c2w_for_frame` live:465 | MAIN (write), FF (read) | **none** | **Yes — H5** (dict swap is atomic, but FF reads multiple keys of a possibly-newer dict) |
| `model.gauss_params` / `means`/`quats`/identity buffers | render MAIN `:1699`,`:1739`; insert/cull FF `:2700`,`:2691`; render VIS viser_direct:690 | MAIN, FF, VIS | **`_model_lock` (RLock)** on all three | No — invariant-protected (see below) |
| `_anysplat_slot_lock` | acquire MAIN `:2509` / release FF `:2542` | MAIN (acquire), FF (release) | self | No (single-in-flight guard; acquire-in-one-thread/release-in-another is intentional) |
| `_obj_mask_cache` | set `:1680` / invalidate MAIN live:275 / read MAIN+FF `:1666` | MAIN, FF | partial (`_model_lock` only around the render, not the None-check) | low (H9) |
| `_live_stop_requested`, `_reselect_requested` | STDIN write / MAIN read | STDIN, MAIN | none | No (bool flag, single writer) |
| `_proc.stdin/stdout` (control pipe) | `_send_command` `:433` | MAIN | `_proc_lock` | No |

### Model-lock invariant (load-bearing, do NOT break)

Per CLAUDE.md invariant #9, all three readers/writers of `gauss_params` share **one** RLock:
`pipeline._model_lock` (`base:474`) is the same object handed to (a) the model via
`attach_render_lock` (`base:519-520` → `dynamic_gs_model.py:741` `get_outputs_for_camera` `:716`),
and (b) the viser-direct server (`base:1480`, replacing its internal lock). The FF bg
thread re-allocates `gauss_params` Parameters on insert/cull, so every render
(`_compute_tick_cdn`, `_render_from_camera`, viser `_render_once`) and every mutation
must be inside `_viser_lock_ctx()`. This is correct as written. **Any purge that removes
a `with self._viser_lock_ctx()` around a `get_outputs`/insert/delete reintroduces the
torn-tensor race documented at `base:1438-1444`.**

---

## (c) Hazards

### H1 — SHM segment leaks on publisher crash (LOW, by design but worth a guard)
`live_ros_publisher.py:1296-1299`, `:603`. `shutdown()` deliberately never calls
`self.shm.unlink()`; cleanup relies on (a) process exit releasing fd refs and (b) the
**next** publisher launch unlinking the stale name (`:595-602`). If the publisher is
SIGKILLed (OOM, `kill -9`) the `/dev/shm/dgs_live_shm` file persists (its size = HEADER
+ 4·slot_bytes; at 1920×1200 that is ~44 MB). Benign because the next run reclaims it,
but a sequence of crashed runs with *different* `--shm-name` would accumulate.
**Recommendation:** keep the next-run unlink (it is the real cleanup), but add a
best-effort `unlink()` inside the `_on_signal` handler path / `atexit` for the common
SIGTERM/SIGINT case so `/dev/shm` stays clean on normal Ctrl-C. Do NOT unlink in
`shutdown()` while the reader may still hold the fd (the comment's reasoning is sound).

### H2 — SHM seqlock has no payload-vs-seq write barrier (MEDIUM)
`live_ros_publisher.py:1020-1027`. Order is `slot.seq = seq` (`:1020`) → payload
(`:1021-1025`) → `latest_seq = seq` (`:1027`). The reader (`peek_latest` `:386-410`)
checks `slot.seq == s_pre` before the copy and `latest_seq==s_pre AND slot.seq==s_pre`
after. The protection works **only because the slot is reused every NUM_SLOTS=4 frames**:
a torn read is caught when, on the *next* write to the same slot, `slot.seq` is bumped to
seq+4 first. But within a single write there is **no memory barrier** between the payload
stores and the `latest_seq` store — on the publisher these are plain numpy slice
assignments + a `struct.pack_into`, executed under the CPython GIL but with **no
cross-process fence**. On x86 store ordering is strong enough that this is safe in
practice; the design comment at `:14-16` is correct for x86 but is **architecture
dependent**. The real residual race: the reader copies the slot (`:398-402`) and only
*then* re-checks (`:404-405`) — if the publisher wrote seq+4 into the same slot between
the copy of `rgb` and the copy of `depth`, the post-check catches it and the loop retries
(up to 5×, then returns None `:414`). Correct, but the 5-retry cap means under sustained
publisher saturation `peek_latest` can return None and the tick silently skips a frame.
**Recommendation:** leave as-is for x86 (it is correct and lock-free); if ever ported off
x86, add an explicit ordering guarantee. Optionally raise the retry cap or log when it is
hit, so frame-skip-under-load is observable rather than silent.

### H3 — pose/joint history lists are mutated across threads without a lock (MEDIUM, real race)
`live_ros_publisher.py`: `_on_joint_state` `:762-763` and `_on_gazebo_pose` `:780-781`
run on the **rospy receive thread(s)** and do `list.insert(at, ...)` into
`_joint_state_times_sec` / `_gazebo_pose_times_sec` / matching value lists. Meanwhile
`_interpolate_c2w` (`:784-820`, on the **worker thread**) and the lazy `RobotMaskGenerator`
constructor (`:812-816`, passed the **same list objects** by reference) read/iterate them.
There is **no lock** around any of this. A `bisect_left` + `insert` on the receive thread
concurrent with an index read on the worker thread can read a stale/half-updated pair
(times list grown but matrices list not yet, or vice-versa → `IndexError` or a
time/matrix mismatch → wrong interpolated pose). rospy callbacks are typically serialized
onto one thread by default, which is why this hasn't blown up, but it is not guaranteed
and `RobotMaskGenerator` captures the live list references.
**Recommendation:** guard the four history lists with a dedicated lock (or use a single
lock for both pairs), held in `_on_joint_state`, `_on_gazebo_pose`, and the read window in
`_interpolate_c2w`. Keep it separate from `_state_lock` to avoid serializing the SHM write.

### H4 — reader `close()` can race an in-flight `peek_latest` (LOW)
`live_shm_reader.py:587-591` sets `self._slot_views = []` then `self._shm.close()`.
`peek_latest` (`:391`) indexes `self._slot_views[slot_idx]`. Both `close()` (via atexit /
`_cleanup_live_subscriber`) and `peek_latest` are normally MAIN-thread, but `close()` also
runs from the SIGINT/SIGTERM handler (`live:170-184`) which can fire **while** the trainer
main thread is mid-`peek_latest`. `_slot_views=[]` → `slot_views[idx]` IndexError, or
`_shm.close()` → reading a freed memoryview → segfault risk.
**Recommendation:** guard `peek_latest` and `close()` with a lightweight lock or a
`self._closed` check at the top of `peek_latest` (the flag already exists `:286`); after
`close()` returns, `peek_latest` should early-return None.

### H5 — `_latest_tracker_frame` handoff to FF thread is a shared mutable dict (LOW/MEDIUM)
MAIN writes a fresh dict each tick (`live:323-329`); FF reads it
(`_feedforward_threaded` `base:2531`, and `_scene_c2w_for_frame` live:465 reads
`self._latest_tracker_frame["camera"]`). The reference swap is atomic, but:
(a) `_feedforward_threaded` mutates the *captured* dict in place (`target_frame["cdn"]=...`
`base:2536`) — that captured ref is safe; but
(b) `_scene_c2w_for_frame` reads `self._latest_tracker_frame` **live** (not the captured
`target_frame`), so the FF computes inserts against a frame whose pose may be a *newer*
tick than the one it was dispatched for. Under the "ICP places inserts in a different
frame than the CDN judges" churn (CLAUDE.md note), this is one of the frame-consistency
seams. Not a crash; a correctness drift.
**Recommendation:** have the FF path read the pose from its captured `target_frame`
(thread it through `args`) rather than `self._latest_tracker_frame`. Pure correctness,
no lock needed.

### H6 — `_on_synced` drop-oldest is a 3-statement non-atomic sequence (LOW)
`live_ros_publisher.py:858-869`: on a full queue it does `get_nowait()` then
`put_nowait()`, each in its own try/except. If two receive callbacks interleave (multiple
rospy threads) the get/put pair isn't atomic, so it can briefly exceed maxsize or drop two.
Benign for tracking (freshest-frame semantics) but worth noting.
**Recommendation:** none required; if rospy is ever configured multi-threaded, switch to a
single `deque(maxlen=4)` guarded by one lock.

### H7 — replay counters `_replay_count/_replay_dropped` updated unlocked from two threads (LOW)
`:959` (replay-writer thread) and `:1036` (worker thread). Only affects the metadata
count written at finalize (`:937-938`); off by default (`--record-replay`). Cosmetic.
**Recommendation:** ignore unless replay-recording is promoted to a default path.

### H8 — stale doc claim: "live depth filtered at the batch source" (DOC, not code)
CLAUDE.md ("Depth filter" note) says the filter is applied "ONCE at the batch source …
live `_batch_from_live_frame`". The code does the opposite for live: `_batch_from_live_frame`
(`live:537`) keeps **raw** depth, and filtering happens on the FF bg thread
(`base:3248-3254`), gated by `self._filter_depth_at_ff=True` (`live:117`). The static
on-disk recording is filtered separately in the publisher (`:1129-1135`). This is a
documentation inconsistency that could mislead a purge into "removing the duplicate filter
at the batch source" — there is no filter there to remove.
**Recommendation:** fix the CLAUDE.md note to state the live tracker uses raw depth and
only the FF site filters (per the in-code comments at `live:532-536` and `base:3239-3245`).

### H9 — `_obj_mask_cache` None-check is outside the lock (LOW)
`base:1666` checks `if self._obj_mask_cache is None:` *before* acquiring the lock; only the
render inside is locked (`:1676-1682`). MAIN invalidates it (`live:275`) while FF may read
it. Worst case two threads both render the mask (wasted work) or FF reads a mask from a
slightly newer camera. No tearing (each assignment is a whole-tensor ref swap). Acceptable.
**Recommendation:** none; documented behavior. If tightened, double-check under the lock.

### H10 — env-strip on publisher spawn is load-bearing (NOT a bug — flag for purge safety)
`live_shm_reader.py:225-226` pops `LD_LIBRARY_PATH/CPATH/LIBRARY_PATH/CUDA_HOME` before
spawning the ROS-env publisher. Removing this (looks like dead cleanup) makes rospy/pyrender
load the `dynamic_gs` conda libstdc++/CUDA `.so`s and crash at import. **Do not purge.**

---

## Notes for the purge

- **Single owner of SHM = publisher.** The reader is a pure attach-er and has
  deliberately unregistered itself from the multiprocessing resource_tracker
  (`live_shm_reader.py:314-319`) — that `_rt.unregister` line looks orphaned but prevents
  the "leaked shared_memory" warning + a double-unlink. Keep it.
- **`shutdown()` NOT clearing `_slot_views` (`:1286-1293`) and NOT unlinking (`:1296-1299`)
  are intentional** — both have explicit comments and prevent a "looks-like-a-crash" race
  with late rospy callbacks / a still-reading reader.
- **Per-tick allocations (steady-state):** `peek_latest` copies 4 arrays out of SHM
  (`:398-402`), `_batch_from_live_frame` makes 1 contiguous RGB copy + 2 host→GPU transfers
  (`:531-538`), `cameras_from_live_frame` builds one Cameras + `.to(device)` (`:612-622`).
  All bounded per frame; none accumulate. No per-tick leak found.
- The only truly **unlocked cross-thread mutable state** that can corrupt (not just drift)
  is **H3** (pose/joint history lists). That is the one to fix before trusting the live
  capture under any multi-threaded rospy config.
