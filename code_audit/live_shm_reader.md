# Code Audit — `dynamic_gs/utils/live_shm_reader.py`

LIVE-PATH module (highest priority). POSIX-shm reader + control-pipe client that spawns the ROS publisher subprocess (`live_ros_publisher.py`, run in the `dynamic_gs_ros` env) and exposes a lock-free `peek_latest()` frame read plus JSON control commands. One instance per training session, owned by `LiveDynamicGSPipeline`, `live_session.py`, and `capture_only.py`.

Grep base for ref counts:
`grep -rn "<sym>" scripts/dynamic_gs scripts/scripts --include=*.py` (excluding `live_shm_reader.py` itself).

---

## 1) FUNCTION / CLASS MAP

### Module-level functions

- **`_decode_header(buf: memoryview)`** — `live_shm_reader.py:61` — Unpacks the full SHM header struct (`_HDR_FMT`) into a dict (magic, intrinsics, offsets, ready/shutdown flags). **Callers: 1**, internal only (`__init__` at `:320`). No external refs (grep "_decode_header" → NO REFS FOUND outside file).
- **`_compute_header_field_offsets()`** — `live_shm_reader.py:78` — Computes per-field byte offsets of the header by cumulative `struct.calcsize` of growing prefixes; used to seek `latest_seq` directly without re-decoding the whole header. **Callers: 1**, module-init `HDR_OFFSETS = _compute_header_field_offsets()` at `:106`. (A same-named function exists in `live_ros_publisher.py:293` — that is the publisher's independent copy, not a ref to this one.)
- **`_spawn_publisher(live_root, shm_name, keyframe_translation_m, keyframe_rotation_deg, wipe_live_root) -> Popen`** — `live_shm_reader.py:162` — Builds the bash-wrapped command that sources ROS Noetic, pins `PYTHONNOUSERSITE`, strips `LD_LIBRARY_PATH`/`CPATH`/etc., and launches the publisher with pipes. **Callers: 1**, `LiveShmSubscriber.__init__` at `:278`. (The grep hit in `anysplat_decode.py:63` is a comment referencing the *pattern*, not a call.)
- **`cameras_from_live_frame(frame, intrinsics, device, cam_idx=0) -> Cameras`** — `live_shm_reader.py:605` — Builds a single-frame Nerfstudio `Cameras` from a `LiveFrame`'s c2w + intrinsics. **Callers: 1**, `dynamic_gs_pipeline_live.py:271` (imported at `:269`).

### Dataclasses

- **`CameraIntrinsicsLite(width, height, fx, fy, cx, cy)`** — `live_shm_reader.py:114` — Minimal intrinsics holder so consumers don't import the recorder. **Direct refs: 0** (the only repo hits are a docstring mention in `fusion_runner.py:27` and an *independent* `_CameraIntrinsicsLite` in `test_concurrent_fusion.py:48`). However it is **NOT dead** — it is instantiated internally (`__init__:296`) and flows out via the `intrinsics` property, which callers consume (`dynamic_gs_pipeline_live.py:145,271`, etc.). The class name is just never written at a call site because construction is encapsulated.
- **`LiveFrame(seq, stamp_sec, rgb_bgr, depth_m, mask_keep, c2w_4x4)`** — `live_shm_reader.py:125` — One synced (rgb/depth/mask/pose/stamp) tuple, the unit returned by `peek_latest`/`capture_anchor`. **Callers: many** — imported in `capture_only.py:69`, `live_session.py:45`, used as a type/return throughout `dynamic_gs_pipeline_live.py`, `live_session.py`. (Note: `live_ros_publisher.py:389` and `diag_validate_fix.py` mentions are independent definitions/comments, not refs to this class.)

### `LiveShmSubscriber` (class) — `live_shm_reader.py:257`
Imported/instantiated in `dynamic_gs_pipeline_live.py:137-139`, `capture_only.py:69,168`, `live_session.py:46,549,567`.

- **`__init__(self, live_root, shm_name, keyframe_translation_m=0.02, keyframe_rotation_deg=20.0, wipe_live_root=True, ready_timeout_s=30.0)`** — `:267` — Spawns publisher, blocks on `{"event":"ready"}`, attaches SHM, unregisters from resource_tracker, builds slot views, registers atexit. **Callers: 3** call sites above.
- **`get_singleton(cls) -> LiveShmSubscriber`** (classmethod) — `:334` — Returns existing instance or constructs a default one. **Callers: 0** — grep "get_singleton" → NO REFS FOUND anywhere in repo. (`_singleton` is still *set* in `__init__:330`, but never read by anything; see Dead Code.)
- **`intrinsics(self) -> CameraIntrinsicsLite`** (property) — `:339` — Returns cached intrinsics. **Callers: many** — `dynamic_gs_pipeline_live.py:145,271`, `live_session.py:904`.
- **`_build_slot_views(self)`** — `:345` — Builds NUM_SLOTS dicts of `np.frombuffer` views over the mmap'd SHM (pose/seq/stamp/rgb/depth/mask). **Callers: 1**, `__init__:328`. No external refs.
- **`peek_latest(self) -> Optional[LiveFrame]`** — `:373` — Lock-free seqlock read (read latest_seq, copy slot, re-verify seq); up to 5 retries. **Callers: 3** — `dynamic_gs_pipeline_live.py:250`, `capture_only.py:292`, internal `wait_for_first_frame`/`capture_anchor`.
- **`wait_for_first_frame(self, timeout_s=30.0) -> LiveFrame`** — `:416` — Spins on `peek_latest` until a frame or deadline. **Callers: 3** — `dynamic_gs_pipeline_live.py:153`, `capture_only.py:188`, `live_session.py:550,570`.
- **`_send_command(self, op, **kwargs) -> dict`** — `:433` — Under `_proc_lock`: checks `poll()`, writes JSON line to publisher stdin, reads one response. **Callers: many internal** (`capture_anchor`, `start_recording`, …). No external refs.
- **`_read_response(self, timeout_s=600.0) -> dict`** — `:446` — Reads one JSON line from publisher stdout byte-by-byte, tee'ing to `/tmp/.../publisher.stdout.log`. **Callers: 2 internal** (`__init__:291`, `_send_command:444`).
- **`capture_anchor(self) -> LiveFrame`** — `:485` — Sends `capture_anchor`, then spins `peek_latest` until `seq >= target_seq`. **Callers: 3** — `live_session.py:586,763`, `capture_only.py:215`.
- **`start_recording(self, anchor) -> None`** — `:504` — Sends `start_recording` with anchor seq. **Callers: 2** — `live_session.py:587`, `capture_only.py:216`.
- **`stop_recording(self) -> None`** — `:509` — Sends `stop_recording`. **Callers: 3** — `live_session.py:751,871`, `capture_only.py:222`.
- **`num_recorded_frames(self) -> int`** — `:514` — Sends `num_recorded`. **Callers: 1** — `live_session.py:872`.
- **`save_anchor_for_sam3(self, anchor, debug_dir) -> Path`** — `:520` — Sends `save_anchor_for_sam3`. **Callers: 0** — no `.save_anchor_for_sam3` call site exists; `live_session.py` uses its own module-level `_save_anchor_for_sam3` (`:182`) on the already-peeked `LiveFrame`. See Dead Code.
- **`save_anchor_intrinsics_and_depth(self, anchor, artifact_dir) -> tuple[Path, Path]`** — `:529` — Sends `save_anchor_depth_intrinsics`. **Callers: 0** — `live_session.py` uses module-level `_save_anchor_intrinsics_and_depth` (`:207`). See Dead Code.
- **`build_static_init_pointcloud(self) -> Path`** — `:540` — Sends `build_init_pcd`. **Callers: 2** — `live_session.py:1148,1165` (both inside a fallback branch; see Data-Lifecycle note).
- **`pause_gazebo_physics(self) -> bool`** — `:547` — Sends `pause_gazebo`, swallows exceptions → False. **Callers: 1** — `live_session.py:115` (the module-level wrapper `pause_gazebo_physics(sub)` calls `sub.pause_gazebo_physics()`).
- **`unpause_gazebo_physics(self) -> bool`** — `:554` — Sends `unpause_gazebo`, swallows exceptions → False. **Callers: 1** — `live_session.py:129,145`.
- **`close(self) -> None`** — `:563` — Best-effort: send `shutdown`, wait 5s/terminate, drop slot views, close SHM. **Callers: 1** — `_atexit_close:595`; also referenced in `dynamic_gs_pipeline_live.py:25` docstring as the atexit target.
- **`_atexit_close(self) -> None`** — `:593` — atexit wrapper around `close()`. **Callers: 1** — `atexit.register` at `__init__:331`.

---

## 2) DEAD-CODE CANDIDATES (zero external refs after grep)

| Symbol | file:line | grep evidence | Confidence |
|---|---|---|---|
| `LiveShmSubscriber.get_singleton` (classmethod) | `:334` | `grep -rn "get_singleton" scripts --include=*.py` → only the def + the `_singleton` writes in this file; **NO REFS FOUND** elsewhere. The `_singleton` class attr (`:265`) is *set* (`:330`) but never *read*. | **high** |
| `LiveShmSubscriber._singleton` (class attr) | `:265` | Written at `:330`, read only by `get_singleton` which itself is unused. Pure orphan support state for the dead classmethod. | **high** |
| `LiveShmSubscriber.save_anchor_for_sam3` | `:520` | `grep -rn "\.save_anchor_for_sam3"` → 0 caller-style hits; `live_session.py` uses module-level `_save_anchor_for_sam3` (`:182`) instead. Publisher-side `save_anchor_for_sam3` (`live_ros_publisher.py:1232`) and the JSON op (`:1506`) exist but are never reached because nothing sends the command. | **high** |
| `LiveShmSubscriber.save_anchor_intrinsics_and_depth` | `:529` | `grep -rn "\.save_anchor_intrinsics_and_depth"` → 0 caller-style hits; `live_session.py` uses module-level `_save_anchor_intrinsics_and_depth` (`:207`). | **high** |

Notes / explicitly NOT flagged dead:
- `CameraIntrinsicsLite` — 0 name-refs but **alive** (constructed in `__init__`, returned via `.intrinsics`, widely consumed). Not dead.
- `build_static_init_pointcloud` (`:540`) — kept; it has 2 callers in `live_session.py` (`:1148,1165`), both inside a legacy/fallback branch. Live + the default seed path use the GPU-subprocess/`online_fusion` route instead, so this is rarely-reached but not zero-ref. **medium** that it is effectively dead at runtime, but it has real call sites so it is NOT a zero-ref candidate.
- No invariant-protected buffers, monkeypatch targets, `__main__` blocks, or method_configs entry points live in this file.

---

## 3) DATA-LIFECYCLE — persistent state, SHM, handles, GPU

### SHM region (attach / read / free) — the core lifecycle
- **Attach** `__init__:314` `shared_memory.SharedMemory(name=shm_name, create=False)`. Reader is *not* the owner (publisher creates it).
- **resource_tracker unregister** `:316-319` — explicitly unregisters the name so the reader's atexit does NOT try to unlink a SHM it didn't create (Python bug #38119). Correct and load-bearing.
- **Views** `_build_slot_views:345` builds NUM_SLOTS×6 `np.frombuffer` views over `self._shm.buf`. These views hold references into the mmap.
- **Free** `close:586-591` clears `self._slot_views = []` *before* `self._shm.close()`. **Order is correct** — closing the mmap while numpy views still alias it would risk a segfault / BufferError; dropping views first is the right sequence. `_shm.close()` only detaches the reader's mapping; the publisher unlinks the name.
- **LEAK RISK (low/medium):** if `__init__` raises **after** `self._shm = SharedMemory(...)` (`:314`) but before `atexit.register` (`:331`) — e.g. the magic mismatch `raise` at `:322` — the SHM handle is opened but neither `close()` nor the atexit hook is wired, so the reader's mmap fd leaks until process exit. Narrow window, but it exists. Severity low (process-lifetime fd).
- **Stale-view hazard (medium):** the slot views are built once from `self._intrinsics` H/W. If the publisher ever re-creates the SHM with a different resolution mid-session, the views silently alias wrong byte ranges. Not reachable in current flow (intrinsics fixed at ready), but the views are never rebuilt or validated against a per-read header.

### Process handle (publisher subprocess)
- **Spawn** `__init__:278` → `_spawn_publisher` → `subprocess.Popen` with `stdin/stdout=PIPE`, `stderr=log_fd`.
- **`log_fd` (`_spawn_publisher:236`) is never closed.** `open(log_path, "wb")` is passed as `stderr=` and the fd is owned by the child, but the *parent's* file object is never `.close()`d and is not stashed for cleanup. Leaks one fd per `LiveShmSubscriber` (one per session → benign, but a true unclosed-handle finding). **medium** (file:`live_shm_reader.py:236`).
- **stdout/stdin pipes** are never explicitly closed in `close()` — only `_proc.wait()`/`terminate()`. On graceful shutdown the child exits and the OS reaps the pipes; on the `terminate()` path the pipe fds linger until GC. Minor.
- **Teardown ordering bug (medium):** `__init__` on the ready-failure path (`:292-294`) calls `self._proc.terminate()` and re-raises, but does **not** close `log_fd` (it's local to `_spawn_publisher`, already leaked above) nor the SHM (not yet attached at that point — fine). The bigger issue: at `:293` `_proc_lock` and `_closed` are already set (`:285-286`) but `close()`/atexit is NOT yet registered, so a terminate without `wait()` can leave a zombie if `terminate` is slow. Low severity.

### LiveFrame / per-tick heap allocation (HOT PATH — live)
- `peek_latest:398-401` does **4 fresh `np.array(..., copy=True)` allocations per successful read** (rgb HxWx3 uint8, depth HxW f32, mask HxW u8, pose 4x4 f64). At 1920×1200 that is ~6.9 MB rgb + 9.2 MB depth + 2.3 MB mask ≈ **~18 MB/frame churned every tracker tick**. No buffer reuse / no preallocated scratch. This is the dominant per-tick heap allocation in the reader. **medium** (`:398-401`) — by design (callers need owned copies that survive the next publisher write), but a reusable double-buffer would cut allocator pressure on the live thread.
- No GPU tensors are created in `peek_latest`; `cameras_from_live_frame:612` does one small `torch.from_numpy(...).to(device)` (3×4 c2w) per camera build — negligible, but it IS a per-call host→device copy on the live path (`dynamic_gs_pipeline_live.py:271`).

### Persistent `.pt` / identity buffers
- This module touches **none** of the `post_fusion_state.pt` warm-cache, none of the 4 identity buffers (`object_flags` / `object_instance_ids` / `sam3d_init_target_flags` / `inserted_flags`). Those live in `persistence/` + the models — invariant-protected and out of scope here. No desync risk originates in this file.

### Header format coupling (save/load shape mismatch risk)
- `_HDR_FMT` (`:57`) is a hand-copied mirror of the publisher's `_HDR_FMT` (`live_ros_publisher.py`), with the comment "keep these in sync." Likewise `_compute_header_field_offsets` is duplicated verbatim. **A silent struct-format divergence between the two files would mis-read every field** (intrinsics, offsets) with no checksum beyond the `magic == b"DGS\0"` guard at `:321`. The magic catches a totally wrong region but NOT a field-layout drift (e.g. an added field). **medium** structural risk (`:57` ↔ publisher).

---

## 4) DESIGN SMELLS

### Thread-safety / races (live: tracker tick ‖ FF bg thread ‖ viser render share the model lock)
- **`peek_latest` is intentionally lock-free** (seqlock) and that is correct for the single-producer/single-consumer SHM. BUT it is also re-entered from multiple reader-side threads: the tracker tick (`dynamic_gs_pipeline_live.py:250`) and `capture_only` polling all call `peek_latest`. The 6 slot views (`self._slot_views`) and the private copy arrays are per-call locals, so concurrent `peek_latest` calls do not corrupt each other — **but `close():587` sets `self._slot_views = []` with no lock**, so a `peek_latest` racing a `close()` from the atexit thread can hit `self._slot_views[slot_idx]` → IndexError, or read `slot["seq"]` on a view backed by an mmap that `_shm.close()` (`:589`) just unmapped → potential segfault/BufferError. **high** (`:373` vs `:586-591`): no synchronization between the lock-free read path and teardown. In live mode the atexit/close happens at shutdown so the window is narrow, but the tracker thread can still be mid-`peek_latest` when the main thread exits.
- `_send_command`/`_read_response` are serialized by `_proc_lock` (`:434`) — good; the control pipe is correctly single-threaded. But **`close():569` takes `_proc_lock` to send "shutdown", while `_read_response` inside a concurrent `_send_command` may be blocked reading stdout under the *same* lock** — `close` cannot acquire it until the in-flight command's `_read_response` returns or times out (up to 600 s default, `:446`). A long-running command (`build_init_pcd` uses 300 s) would stall shutdown. **medium** (`:446` 600 s default timeout under a lock that shutdown also wants).

### Swallowed exceptions
- `__init__:316-319` swallows the resource_tracker unregister failure with bare `except Exception: pass`. Acceptable (best-effort), but masks a real attribute-rename if Python internals (`self._shm._name`) change.
- `pause_gazebo_physics:551` / `unpause_gazebo_physics:558` swallow *all* exceptions and return False — a dead publisher pipe looks identical to "gazebo doesn't support pause." **medium**: the caller (`live_session.py:115`) can't distinguish "pause unsupported" from "publisher crashed."
- `close()` has **three** nested bare `except Exception: pass` (`:577,584,591`) and `_atexit_close:596` a fourth. Standard for teardown, but collectively they would hide a SHM-close BufferError that indicates the view-vs-close race above.

### Leaky abstraction / duplication
- `LiveShmSubscriber` is documented (`:15-19`) as mirroring the old `LiveRosSubscriber` API surface, but ships **4 dead control methods** (`get_singleton`, `save_anchor_for_sam3`, `save_anchor_intrinsics_and_depth` — see §2) that exist only because the publisher-side handlers exist. The reader/publisher JSON-op contract has stale ops (`save_anchor_for_sam3`, `save_anchor_depth_intrinsics`) that nothing sends. **API-surface bloat** on a module being purged.
- `_HDR_FMT` + `_compute_header_field_offsets` are **verbatim-duplicated** from `live_ros_publisher.py` with only a "keep in sync" comment as the contract (see §3). Classic copy-paste-coupling smell.
- `_read_response` tees **every byte** to `/tmp/.../publisher.stdout.log` with a `dbg.flush()` per byte (`:465-466`). On a large response (e.g. `build_init_pcd`) that is one syscall write+flush per byte — a debug-instrumentation hot loop left in the production read path. **medium** (`:458-466`): per-byte flush is O(n) syscalls; fine for short JSON but pathological for any large line, and it opens/appends the log file on *every* response.

### Misleading naming / config
- The module docstring (`:5`) says the publisher runs in env `radiance_ros_4060`, but the code constant `ROS_PUBLISHER_CONDA_ENV = "dynamic_gs_ros"` (`:154`) — the docstring is stale (the inline comment at `:148-153` corrects it, but the top-of-file docstring still lies). **low** documentation drift.
- `keyframe_translation_m` / `keyframe_rotation_deg` are accepted by `__init__` and forwarded to the publisher, but are **publisher-side dedup knobs** — the reader never reads them. Pass-through params threaded through `__init__ → _spawn_publisher → CLI`. Not a bug, but the reader is just a courier for them.

### God function / param-threading
- No god functions here (largest is `__init__` at ~63 lines, cohesive). `peek_latest` is the only complexity-dense method and is well-scoped.

---

## TOP CONCERNS (live-purge priority)
1. **Lock-free `peek_latest` vs unsynchronized `close()` teardown** (`:373` vs `:587-589`) — segfault/IndexError window when the tracker thread reads while atexit drops slot views + closes the mmap. Highest-severity race on the live path.
2. **Per-tick ~18 MB heap churn** in `peek_latest` (`:398-401`) — 4 fresh copies/frame at 1200p, no buffer reuse.
3. **Unclosed `log_fd`** in `_spawn_publisher` (`:236`) — parent-side fd leaked per session.
4. **4 genuinely-dead control methods/attrs** (`get_singleton`, `_singleton`, `save_anchor_for_sam3`, `save_anchor_intrinsics_and_depth`) — safe to delete on purge.
5. **Per-byte flush in `_read_response`** (`:465-466`) + **600 s `_read_response` timeout held under `_proc_lock`** that `close()` also needs (`:446` / `:569`).
