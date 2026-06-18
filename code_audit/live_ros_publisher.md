# Code Audit: `dynamic_gs/utils/live_ros_publisher.py`

LIVE-PATH module (Python 3.8, runs in the `dynamic_gs_ros` conda env as a **standalone script**, not a package import). Owns the POSIX shared-memory frame stream and the disk recorder. It is driven over a stdin/stdout JSON line protocol by the reader-side `LiveShmSubscriber` (`dynamic_gs/utils/live_shm_reader.py`). Method invocation is therefore **by `op` string in `_main`**, not by direct Python reference — grep for the method name returns "no refs" but the op string in the reader is the real caller. Cross-references below cite the matching `op`.

---

## 1. FUNCTION / CLASS MAP

### Module-level functions

- `_load_recorder_module()` — :139 — importlib-loads `save_data_img_depth_mask_pose.py` (the disk recorder) as `_dgs_live_recorder` so the publisher can reuse its ROS topic names, mask render, intrinsics. **1 ref** — called at module import (:149).
- `_load_zed_depth_noise()` — :188 — importlib-loads `zed_depth_noise.py`. **1 ref** — :210.
- `_load_depth_filter()` — :199 — importlib-loads `depth_filter.py` (CPU cv2 path). **1 ref** — :211.
- `_slot_layout(height, width)` — :234 — computes per-slot byte size + field offsets for the SHM layout. **3 refs** — :266 (`_total_shm_bytes`), :591 (`__init__`).
- `_total_shm_bytes(height, width)` — :265 — returns total SHM bytes. **NO REFS FOUND** outside its own def (grep `_total_shm_bytes` → only :265). `__init__` re-derives the same value inline at :592 (`total = HEADER_BYTES + NUM_SLOTS * slot_bytes`). Dead.
- `_write_header(...)` — :270 — packs the full header struct into the SHM buffer once at init. **1 ref** — :607.
- `_compute_header_field_offsets()` — :293 — walks `_HDR_FMT` prefixes to get byte offsets of fields mutated post-init (`latest_seq`, `shutdown`). **1 ref** — :331 (`HDR_OFFSETS = ...`).
- `_send_response(payload)` — :1377 — writes one JSON line to the saved IPC fd (`_IPC_OUT`). **Many refs** — :1427, :1431, and ~15 sites in `_main`'s command loop.
- `_wipe_live_root(live_root)` — :1383 — clears the dataset dir (preserving `.static_sequence_t0` + `timing_static_sequence.txt`) and recreates `static_scene/`+`dynamic_scene/`. **1 ref** — :1416 (`_main`, gated on `--wipe-live-root`, which the reader passes — `live_shm_reader.py:196`).
- `_main()` — :1403 — argparse + construct `LivePublisher` + stdin JSON command loop. **1 ref** — :1541 (`__main__`). Entry point.

### `_KeyframeFilter` (class) — :339
Inlined ORB-SLAM greedy keyframe filter (numpy only — the package version isn't importable in this env). **1 ref** — :578 (`_record_keyframe_filter`).
- `__init__(translation_thresh_m, rotation_thresh_deg)` — :342
- `num_kept` (property) — :348 — **NO REFS FOUND** outside def. Dead accessor.
- `reset()` — :352 — **1 ref** — :1075 (`start_recording`).
- `accept(c2w_3x4) -> bool` — :356 — **refs** — :990, :1076.

### `_StoredFrame` (dataclass) — :385
Per-frame latched snapshot (seq, stamp, rgb, depth, mask, c2w). **Refs** — :549, :1010, and as type hint on `capture_anchor`/`start_recording`/`_write_frame_to_disk`.

### `LivePublisher` (class) — :400
Single-process publisher. Constructed once in `_main` (:1419).

- `_wait_for_camera_info_primed(timeout_s=20.0)` (staticmethod) — :403 — resolves `camera_info`, holding a primer subscriber to wake Gazebo's lazy publisher. **1 ref** — :492.
- `__init__(live_root, shm_name, keyframe_translation_m, keyframe_rotation_deg, record_replay_dir=None)` — :458 — resolves intrinsics (ROS → cached fallback chain), allocates SHM, pre-builds zero-copy slot views, registers ROS subscribers + the `ApproximateTimeSynchronizer`, starts the worker thread.
- `_spawn_depth_republisher()` — :687 — auto-launches `rosrun image_transport republish` for `/compressedDepth`. **NO REFS FOUND** — defined but never called anywhere (the depth subscriber subscribes to the **raw** `DEPTH_TOPIC`, :668-671; `_decode_raw_depth` explicitly rejects the compressedDepth transport). Dead method (but `_depth_republisher_proc` stays `None`, so the `shutdown()` teardown at :1322 is a harmless no-op).
- `_on_joint_state(msg)` — :747 — throttled (20 ms) sorted-insert of joint positions for FK mask render. **1 ref** — registered at :645.
- `_on_gazebo_pose(msg)` — :765 — throttled sorted-insert of camera poses. **1 ref** — registered at :648.
- `_interpolate_c2w(stamp_sec)` — :783 — slerp/lerp the camera pose at an image stamp, lazily build `RobotMaskGenerator`, apply optical offset. **1 ref** — :975.
- `_decode_compressed_rgb(msg)` (staticmethod) — :822 — JPEG→BGR. **1 ref** — :982.
- `_decode_raw_depth(msg)` (staticmethod) — :828 — 32FC1/16UC1 → float32 metres. **1 ref** — :983.
- `_on_synced(image_msg, depth_msg)` — :847 — rospy sync callback; enqueues (drop-oldest) onto `_frame_queue` and returns. **1 ref** — registered at :677.
- `start_worker()` — :871 — spawns the worker thread. **1 ref** — :685.
- `_worker_loop()` — :881 — drains `_frame_queue`, calls `_process_synced_pair`. **1 ref** — :877 (thread target).
- `_setup_replay_recording(record_replay_dir)` — :898 — opens `stream.bin`, starts the replay writer thread. **1 ref** — :683 (gated on `--record-replay`).
- `_write_replay_meta(finalized)` — :928 — writes `replay_meta.json`. **2 refs** — :924, :1308.
- `_replay_writer_loop()` — :944 — drains `_replay_queue`, appends fixed-size records to `stream.bin`. **1 ref** — :922 (thread target).
- `record_control_event(op, **kw)` — :963 — logs a reader control op mapped to current seq (replay only). **Refs** — :1488, :1498, :1501 (`_main`).
- `_process_synced_pair(image_msg, depth_msg)` — :972 — the heavy worker body: interp pose, decode rgb+depth, apply ZED noise, render mask, latch `_latest`, write SHM slot, bump `latest_seq`, optionally enqueue replay + write to disk. **1 ref** — :892.
- `wait_first_frame(timeout_s)` — :1046 — blocks on `_first_frame_event`. **NO REFS FOUND** — no `op == "wait_first_frame"` exists in `_main` and the reader never sends it (grep `wait_first` in `live_shm_reader.py` → none). Dead public method.
- `capture_anchor(timeout_s=30.0)` — :1049 — wait for a fresh frame, return it. Driven by `op "capture_anchor"` (:1483; reader `live_shm_reader.py:491`).
- `start_recording(anchor)` — :1060 — create static_scene dirs, reset keyframe filter, set `_record_active`, write anchor frame. `op "start_recording"` (:1490; reader :505).
- `stop_recording()` — :1081 — flip `_record_active` off, quiesce in-flight writes (M3). `op "stop_recording"` (:1500; reader :510).
- `_write_frame_to_disk(frame, stamp)` — :1107 — rgb/depth/mask imwrite + atomic transforms.json swap; depth median+bilateral filtered, raw preserved under `depth_raw/`. **Refs** — :1042, :1078.
- `num_recorded_frames()` — :1158 — count written frames. `op "num_recorded"` (:1504; reader :515).
- `build_static_init_pointcloud()` — :1162 — back-project recorded depth+mask into a world PLY. `op "build_init_pcd"` (:1522; reader :541).
- `save_anchor_for_sam3(anchor, debug_dir)` — :1232 — gripper-blacked anchor RGB → `static0_rgb.png`. `op "save_anchor_for_sam3"` (:1506; reader :521).
- `save_anchor_intrinsics_and_depth(anchor, artifact_dir)` — :1249 — anchor depth tiff + intrinsics json. `op "save_anchor_depth_intrinsics"` (:1514; reader :532).
- `pause_gazebo()` / `unpause_gazebo()` — :1265 / :1274 — gazebo physics service calls. `op "pause_gazebo"`/`"unpause_gazebo"` (:1525/:1527; reader :549/:556).
- `shutdown()` — :1283 — finalize replay, signal worker stop, kill republisher, set header.shutdown, unregister subscribers, `rospy.signal_shutdown`. **Refs** — :1447 (atexit), :1458 (signal handler).

---

## 2. DEAD-CODE CANDIDATES

(Entry points / op-string-driven methods excluded; those are the live IPC surface.)

- **`_spawn_depth_republisher()` — :687** — confidence **high**. grep across `dynamic_gs/` + `scripts/` for `_spawn_depth_republisher` → only its own definition (:687). Never called, including from within the file. Superseded by the move to subscribe the raw 32FC1 depth topic directly (:668-671, and `_decode_raw_depth`'s docstring at :828 explicitly says it does NOT use the compressedDepth transport). The whole method body — plus the 5 s `get_published_topics` poll — is unreachable. `_depth_republisher_proc` remains `None`, so the teardown branch at :1322 never fires.
- **`_total_shm_bytes(height, width)` — :265** — confidence **high**. grep → only :265. `__init__` (:592) computes the identical value inline instead of calling it.
- **`wait_first_frame(timeout_s)` — :1046** — confidence **high**. No `op == "wait_first_frame"` branch in `_main`; reader never sends it (`grep wait_first live_shm_reader.py` → none). The `_first_frame_event` it waits on is still *set* (:1038) but never *awaited*. Dead public method.
- **`_KeyframeFilter.num_kept` (property) — :348** — confidence **high**. grep `num_kept` → only the definition. The recorder counts frames via `len(self._record_frames_written)` instead.

Note: the four identity buffers / monkeypatches / means-LR / `_ZERO_LR_OPTIMIZERS` invariants are not present in this module (it has no torch). Nothing here is invariant-protected.

---

## 3. DATA-LIFECYCLE

### POSIX shared memory (the live frame stream)
- **Create** — :594-603. Pre-creation unlink of any stale region by name (crash recovery), then `SharedMemory(name=shm_name, create=True, size=total)`. Good.
- **Zero-copy slot views** — :616-638 builds 4 slots × 6 numpy views (`np.frombuffer(self.shm.buf, ...)`). These views hold a writable reference into `shm.buf` for the process lifetime — intentional (per-frame writes at :1020-1025 are in-place, no per-frame allocation of the SHM side).
- **Write path** — :1015-1027: `slot_idx = seq % NUM_SLOTS`, write the 6 fields, then `struct.pack_into latest_seq`. The reader samples `latest_seq` then re-checks `slot.seq` — the documented torn-read guard. **RACE / correctness concern (medium):** the publisher writes `slot["seq"]` *first* (:1020) then the payload (:1021-1025), then bumps `latest_seq` (:1027). The reader's contract (header docstring, :13-16) is: read `latest_seq` → read that slot → re-check `slot.seq == latest_seq`. Because `seq` is written *before* the payload, a reader that samples `latest_seq` after :1027 and reads the slot will re-check `seq` and find it equal **even though the rgb/depth/mask of that same slot may still be mid-`[:]=` copy** if `seq` overran by `NUM_SLOTS` — but with `NUM_SLOTS=4` @ ~25 Hz the reuse window is >100 ms, so in practice safe. The ordering is nonetheless *write-seq-before-payload*, which is the inverse of the usual seqlock discipline (write payload, then publish seq). Worth a comment/flag for the purge.
- **Free / unlink** — deliberately NOT freed in `shutdown()` (:1295-1299): the comment explains slot views are left intact (callbacks may fire during teardown) and `unlink()` is skipped so the reader can finish. Relies on **process exit** to release the mmap + POSIX name. This is a documented leak-by-design; if the process is `os._exit(...)`'d (signal handler, :1460) the name is never unlinked, but the next launch unlinks-stale at :595. Acceptable, but the SHM name can linger in `/dev/shm` after an abnormal exit until the next run.

### Replay recording (`--record-replay`, off by default)
- **Open** — :915 `open(stream.bin, "wb")` + :921 writer thread. **Lifecycle:** `_replay_stream` is flushed+closed only in `shutdown()` (:1306-1307). If `shutdown()` is bypassed (hard kill before atexit/signal), the file handle leaks and the tail of the queue is lost — acceptable for a diagnostic-only path. Drop-counting via `_replay_dropped` (:1036) is honest.
- The replay record size (:914) is `16*8 + 8 + 8 + H*W*(3+4+1)` — note it packs `struct.pack("<qd", seq, stamp)` = i64+f64 (:954), matching the `replay_meta.json` layout string (:935). Format self-consistent.

### Disk recorder state (the 4 record-* buffers + transforms.json)
- `_record_active` / `_record_dir` / `_record_meta` / `_record_frames_written` — guarded by `_record_lock`. `start_recording` (:1071-1077) sets them under lock; `stop_recording` (:1089) clears `_record_active` under lock and quiesces `_inflight_writes` via `_inflight_cv` (M3 fix, :1094-1101). Good.
- **`meta["frames"] = self._record_frames_written` aliasing — :1146.** `_record_meta["frames"]` is bound to the *same list object* as `_record_frames_written`. Subsequent `.append(frame_entry)` (:1145) mutates both. This works because the json dump happens under the same lock immediately after, but it's a fragile alias: any future code that reads `meta["frames"]` outside the lock sees a live, concurrently-mutated list. Flag as a desync hazard.
- **transforms.json** written atomically (tmp + `os.replace`) at :1147-1150 and again in `build_static_init_pointcloud` (:1224-1227). Consistent.
- `build_static_init_pointcloud` (:1162) **re-reads depth/mask/rgb from disk twice** (counting pass :1173-1186, then sampling pass :1193-1215) — loads every recorded frame's depth+mask once to count, then loads depth+mask+rgb again to sample. Not a leak (each is local), but 2× disk I/O over the whole static set. Noted under smells.

### ZED noise / depth filter modules
- `_ZED_NOISE`, `_DEPTH_FILTER` loaded once at import (:210-211); `_zed_noise_rng` is one RNG reused (:584). No per-frame module reload. Good.

### Lazy `RobotMaskGenerator`
- Built lazily and cached on first use in **two** places: `_interpolate_c2w` (:811-816) and `_process_synced_pair` (:992-997). Both guard on `self._mask_gen is None` and pass the *same* lists. Benign double-init guard, but it means the generator can be constructed from whichever path runs first; both run on the worker thread so no race. The generator holds references to `self._joint_state_times_sec`/`_positions` (the live, growing lists) — it reads them by reference each render, so it stays current. Worth confirming `RobotMaskGenerator` doesn't snapshot.

---

## 4. DESIGN SMELLS

- **God function: `_process_synced_pair` (:972-1042)** — single method does pose interp, RGB decode, depth decode, ZED-noise injection, keyframe-accept decision, lazy mask-gen build, mask render, `_latest` latch, 6-field SHM write + `latest_seq` bump, replay enqueue, first-frame event, and disk write dispatch. ~70 lines, the core of the live hot path. High blast radius for the purge.
- **God function: `LivePublisher.__init__` (:458-685)** — ~225 lines: intrinsics resolution with a 4-tier fallback chain (nested `_write_cache` closure, glob over the datasets root), SHM alloc, slot-view construction, subscriber wiring, worker spawn. The hardcoded `_DATASETS_ROOT` absolute path (:479) is a leaky abstraction baked into the publisher.
- **Swallowed exceptions (broad `except Exception: pass`)** — pervasive and load-bearing in teardown but risky elsewhere: `_write_cache` (:489), stale-shm unlink (:600), primer unregister (:455), `_on_synced` enqueue (:860-869, double-nested), replay enqueue (:1035), and the entire `shutdown()` (every step wrapped). The `_on_synced` drop-oldest logic silently swallows *all* exceptions from `put_nowait`/`get_nowait`, so a genuine bug there is invisible. The worker loop's `logwarn_throttle` (:894) at least logs, but throttled to 2 s — a per-frame failure shows ~1 line / 2 s and the SHM stream silently stalls.
- **Duplicated SHM-bytes math** — `_total_shm_bytes` (:265) vs inline `__init__` (:592). Dead helper + magic duplication.
- **Duplicated lazy `_mask_gen` construction** — :811-816 and :992-997 (identical block).
- **Duplicated atomic transforms.json write** — :1147-1150 and :1224-1227 (same tmp+`os.replace` idiom, could be one helper).
- **`meta["frames"]` list-aliasing (:1146)** — see lifecycle §3; misleading because it looks like an assignment-of-a-copy but is an alias to the lock-guarded list.
- **Module-level debug spew (:101-121)** — a block of `[publisher-debug]` prints (sys.path dump, eager `pyparsing` import probe) with a hardcoded site-packages path (:117). This is leftover diagnostic scaffolding that runs on every launch, on the live path. Not dead (it executes), but it is noise that should be gated or removed.
- **Numpy-alias monkeypatch (:94-99)** — restores `np.float` etc. removed in numpy 1.24, for urdfpy 0.0.22. Load-bearing (the docstring explains the mask-gen crash without it) but a global mutation of `numpy` — fragile and easy to mistake for cruft. Keep, but it's a smell rooted in a vendored-dependency pin.
- **`SYNC_SLOP_SEC` (:180)** overrides the recorder's value by re-binding a module constant after the bulk `_REC.*` re-export — easy to miss that this one constant is intentionally NOT taken from `_REC`.
- **Dead config-ish constants/flags**: `--keyframe-translation-m`/`--keyframe-rotation-deg` defaults (:1407-1408) are real and threaded through. No dead CLI flags found. `wait_first_frame` is the one declared-but-unreachable public capability.
- **`record_control_event` is a no-op when `_replay_dir is None` (:966-967)** — fine, but it's called unconditionally from `_main` (:1488 etc.), so it acquires `_state_lock` on every control op even when replay is off (minor; not hot path).

### Thread-safety summary (this module's threads)
Threads here: (1) rospy receive thread(s) → `_on_synced` (enqueue only), `_on_joint_state`, `_on_gazebo_pose`; (2) `dgs-publisher-worker` → `_process_synced_pair`; (3) `replay_writer` (optional); (4) the main thread in `_main`'s stdin loop (calls `capture_anchor`/`start_recording`/`stop_recording`/`build_*`/`save_*`).
- `_joint_state_times_sec`/`_positions` and `_gazebo_pose_*` lists are **written by the rospy callbacks** (sorted-insert, :762/:781) and **read by the worker** (`_interpolate_c2w`, :784) and by `RobotMaskGenerator` (by reference) — **with NO lock**. Concurrent `list.insert` during a `bisect_left`/read is the clearest race in the file: a sorted-insert that reallocates the list while the worker is mid-`bisect`/index can yield a stale or out-of-range read. In CPython the GIL makes individual ops atomic, but a multi-step read (`bisect_left` then `mats[idx]`) is not atomic against an insert. Flag **medium/high** for the live path. (The model lock / feedforward / viser threads referenced in the prompt live on the *reader* side, not here — this process has no torch and no viser.)
- `_record_*` state correctly under `_record_lock`; `_latest`/`_frame_seq` under `_state_lock`. SHM slot writes are lockless-by-design (seqlock-ish, see §3).
