# `shm_channel.py` — module spec (layer: contract)

## 1. Responsibility

One source of truth for the live RGB-D-mask-pose POSIX shared-memory ring: the
header/slot byte layout, a producer-owned create/write-slot path and a
consumer-only attach/read-latest path, sharing one seqlock so a single producer
and one-or-more readers exchange the latest frame lock-free without torn reads.

## 2. Public interface (the contract other modules call)

Pure stdlib + numpy. NO torch, NO nerfstudio, NO rospy, NO control pipe.
Header layout constants and the seqlock discipline live here once; both
processes import them so the publisher's writer and the reader's `peek` can
never silently diverge.

```python
# --- layout / contract constants (single source; publisher + reader import these) ---
SHM_MAGIC: bytes              # b"DGS\0" — region sanity guard
SHM_VERSION: int              # header schema version; bumped on any field-layout change
DEFAULT_SHM_NAME: str         # "/dgs_live_shm"
NUM_SLOTS: int                # 4 (ring depth)
HEADER_BYTES: int             # 4096 (fixed header pad)

@dataclass(frozen=True)
class ShmLayout:
    """Fully-resolved byte layout for a given H/W + intrinsics. Computed once
    by the producer, re-derived by the consumer from the header it reads."""
    height: int; width: int
    fx: float; fy: float; cx: float; cy: float
    num_slots: int; slot_bytes: int
    rgb_off: int; depth_off: int; mask_off: int
    pose_off: int; seq_off: int; stamp_off: int
    @property
    def total_bytes(self) -> int: ...          # HEADER_BYTES + num_slots*slot_bytes

@dataclass
class ShmFrame:
    """One synced tuple copied OUT of shm (owned arrays — safe past next write).
    The wire/contract unit; the higher Frame contract is built from this."""
    seq: int; stamp_sec: float
    rgb_bgr: np.ndarray        # (H,W,3) uint8, BGR
    depth_m: np.ndarray        # (H,W)   float32, metres
    mask_keep: np.ndarray      # (H,W)   uint8, keep-mask
    c2w_4x4: np.ndarray        # (4,4)   float64, OpenGL c2w

def compute_layout(height:int, width:int, fx,fy,cx,cy, num_slots:int=NUM_SLOTS) -> ShmLayout
    # Derive slot_bytes + per-field offsets (largest-first 8B-aligned, slot rounded to 64).

# --- producer side (owns the segment for its whole life) ---
class ShmProducer:
    def __init__(self, name:str, layout:ShmLayout): ...
        # Unlink any stale region with `name` (prior-crash cleanup), create=True,
        # write header (ready=1, shutdown=0, latest_seq=0), build NUM_SLOTS writable views.
    def write(self, seq:int, stamp_sec:float, c2w_4x4, rgb_bgr, depth_m, mask_keep) -> None
        # Seqlock write into slot[seq % N]: tag slot.seq=seq FIRST, payload, then
        # struct.pack_into header.latest_seq=seq LAST. Caller owns the monotonic seq.
    def mark_shutdown(self) -> None
        # Set header.shutdown=1. Does NOT clear slot views, does NOT unlink (see Inv/H notes).
    def close(self, unlink:bool=True) -> None
        # Drop views then shm.close(); unlink ONLY on a normal exit (producer owns the name).

# --- consumer side (attach only; NEVER unlinks) ---
class ShmConsumer:
    def __init__(self, name:str): ...
        # Attach create=False; resource_tracker.unregister(name) so reader atexit never unlinks;
        # verify magic==SHM_MAGIC and version==SHM_VERSION (raise on mismatch);
        # re-derive ShmLayout from the header; build read-only slot views.
    @property
    def layout(self) -> ShmLayout: ...
    def read_latest(self) -> Optional[ShmFrame]
        # Lock-free seqlock read: latest_seq -> slot -> copy-out -> re-verify; up to RETRY tries,
        # else None. Early-returns None if closed. Each call returns freshly-owned arrays.
    def close(self) -> None
        # Idempotent + close-safe vs an in-flight read_latest (see THREADING). Detach only.
```

`read_latest()` is the hot-path call (one per tracker tick). `write()` is the
hot-path call on the producer worker thread. Everything else is setup/teardown.

## 3. Depends on (NEW modules only)

- **None.** This is the lowest contract layer — stdlib (`struct`, `mmap`/
  `multiprocessing.shared_memory`, `multiprocessing.resource_tracker`, `time`,
  `dataclasses`) + `numpy` only. It must stay importable from the minimal
  `dynamic_gs_ros` py3.8 env (the producer side runs there), so it pulls in
  nothing heavy. The `Frame` contract module and the source-adapter / publisher
  depend on THIS, not the reverse.

## 4. Consumes / produces

- **Consumes (producer side):** a monotonically-increasing `seq`, a capture
  `stamp_sec` (event-time, NOT `now()`), and four already-decoded arrays
  (BGR uint8 rgb, float32-metres depth, uint8 keep-mask, float64 OpenGL c2w) —
  all produced upstream by the publisher's frame-decode/sync step.
- **Produces (consumer side):** `ShmFrame` (owned-copy arrays) + a resolved
  `ShmLayout` carrying intrinsics. The source-adapter wraps `ShmFrame` into the
  pipeline's `Frame` contract and builds the nerfstudio `Cameras` from
  `layout` intrinsics + `c2w_4x4`. **This module does not touch torch/Cameras.**
- **Wire contract:** `[ HEADER_BYTES header | NUM_SLOTS × slot_bytes ]`; header =
  magic, version, H, W, num_slots, fx/fy/cx/cy, latest_seq(u64), slot_bytes(u64),
  6 offsets(u32), ready(u32), shutdown(u32). Slot = pose(16×f64), seq(u64),
  stamp(f64), rgb, depth, mask. Single x86 store-ordering seqlock; no cross-process
  fence (documented, x86-only).

## 5. Source moved in

| Current `file:symbol` | Becomes |
|---|---|
| `live_ros_publisher.py` `_HDR_FMT` / `_HDR_SIZE` | the single header struct, owned here (was duplicated in BOTH files) |
| `live_ros_publisher.py` `_slot_layout` + `_total_shm_bytes` | `compute_layout()` + `ShmLayout.total_bytes` |
| `live_ros_publisher.py` `_write_header` | `ShmProducer.__init__` header write |
| `live_ros_publisher.py` `_compute_header_field_offsets` / `HDR_OFFSETS` | computed once here; both sides import (was duplicated verbatim) |
| `live_ros_publisher.py` SHM alloc block (`unlink-stale → create=True → _slot_views`) `:593-638` | `ShmProducer.__init__` |
| `live_ros_publisher.py` SHM write `:1015-1027` (`slot.seq` tag → payload → `pack_into latest_seq`) | `ShmProducer.write` |
| `live_ros_publisher.py` `shutdown()` header.shutdown pack `:1335` | `ShmProducer.mark_shutdown` |
| `live_shm_reader.py` `_HDR_FMT` / `_decode_header` / `_compute_header_field_offsets` / `HDR_OFFSETS` `:57-106` | DELETED as duplicates; reader imports the shared layout |
| `live_shm_reader.py` `CameraIntrinsicsLite` `:114` | folded into `ShmLayout` fields (intrinsics carried by layout) |
| `live_shm_reader.py` `LiveFrame` `:125` | `ShmFrame` (renamed; same six fields, depth_m float32) |
| `live_shm_reader.py` shm attach + `resource_tracker.unregister` + magic check `:314-322` | `ShmConsumer.__init__` (adds version check) |
| `live_shm_reader.py` `_build_slot_views` `:345` | `ShmConsumer` slot-view build (read-only) |
| `live_shm_reader.py` `peek_latest` `:373` | `ShmConsumer.read_latest` (+ `_closed` early-return guard) |
| `live_shm_reader.py` `close()` view-drop + `shm.close` `:563-591` | `ShmConsumer.close` (close-safe vs in-flight read) |

## 6. Dropped (NOT carried)

| Dropped | Reason | Audit ref |
|---|---|---|
| Control-pipe client: `_send_command`, `_read_response`, `capture_anchor`, `start_recording`, `stop_recording`, `num_recorded_frames`, `build_static_init_pointcloud`, `pause/unpause_gazebo_physics` | Not SHM primitives — JSON-over-stdin command/response is a separate **source-adapter / capture-session** concern. This contract module is data-plane only. | `live_shm_reader.md §1` (control methods), `DUP_worker_subprocess.md` (publisher spawn is a distinct concern) |
| `_spawn_publisher` + the ROS bash-wrap / env-strip / `log_fd` | Process spawning + the load-bearing `LD_LIBRARY_PATH/CPATH/CUDA_HOME` strip belong to the **publisher source adapter**, not the byte-layout contract. (The env-strip is KEPT, just relocated.) `log_fd` leak fixed there. | `live_shm_reader.md §3` (unclosed log_fd), `RUNTIME_target_arch §SHM lifecycle`, `RUNTIME_shm_to_batch H10` |
| `cameras_from_live_frame` (`:605`) | torch + nerfstudio `Cameras` — would poison the import-light contract usable in the py3.8 ROS env. Camera build moves to the source adapter / `Frame` module. | `live_shm_reader.md §1`, `RUNTIME_shm_to_batch (b)` |
| `get_singleton` classmethod + `_singleton` class attr (`:333,265`) | Zero external refs; orphan support for a dead classmethod. | `live_shm_reader.md §2` (high-confidence dead) |
| `save_anchor_for_sam3`, `save_anchor_intrinsics_and_depth` (`:520,529`) | Zero caller-style refs; `live_session.py` uses its own module-level helpers. | `live_shm_reader.md §2` (high-confidence dead) |
| Per-byte stdout tee + per-byte `flush` in `_read_response` (`:458-466`) | O(n)-syscall debug hot loop on the production read path; goes away with the whole control pipe. | `live_shm_reader.md §4` (per-byte flush smell) |
| Replay tap (`--record-replay`, `_replay_*`) in the SHM write path | Off-by-default diagnostic, not part of the live contract; stays in the publisher if kept at all. | `RUNTIME_shm_to_batch H7` (replay counters, cosmetic) |
| Stale top-of-file docstring naming env `radiance_ros_4060` | Drift; the contract module has no ROS env reference at all. | `live_shm_reader.md §4` (misleading naming) |

## 7. Invariants preserved

This module touches **none** of the gauss-param / identity-buffer invariants
(#1–#9 on the model) — it is a sensor-ingest contract. The invariants it must
honor are the SHM-lifecycle disciplines the audits flag as load-bearing:

- **Producer owns the segment for its whole life.** Only `ShmProducer` does
  `create=True` / `unlink`; `ShmConsumer` attaches `create=False` and calls
  `resource_tracker.unregister(name)` so the reader's atexit never unlinks a
  name it did not create (Python bug #38119). `mark_shutdown()` must NOT clear
  slot views nor unlink — late producer-side callbacks may still index a slot,
  and a reader may still be mid-read. (`RUNTIME_target_arch §SHM lifecycle`,
  `RUNTIME_shm_to_batch (a)` "free/unlink" row.)
- **terminate → kill orphan fix** is honored at the boundary this module
  defines: `ShmProducer.close(unlink=True)` is the producer's *normal-exit*
  reclamation, and `ShmConsumer.close()` is detach-only. The actual
  `terminate()→wait→kill()→wait` of the publisher subprocess is the source
  adapter's job, but this module makes the "next launch unlinks stale" path
  correct by always unlinking-stale-before-create in `ShmProducer.__init__`, so
  a SIGKILLed prior producer that leaked `/dev/shm/<name>` is reclaimed.
  (`RUNTIME_warmload_lifecycle H4/L1`, MEMORY "Live publisher restart cleanup".)
- **Event-time, not `now()`** (ARCH principle #5): `stamp_sec` is a passed-in
  capture timestamp carried verbatim through the slot; this module never calls
  `time.time()` for frame metadata (only for the read-retry yield).
- **Versioned boundary contract** (ARCH principle #8): `ShmConsumer.__init__`
  validates `magic` AND `version`, failing loudly on a layout-schema drift
  instead of silently mis-reading fields — the gap the audit named (magic
  catches a wrong region but not an added/removed field).
  (`live_shm_reader.md §3` header-coupling risk.)

## 8. Threading

- **Producer thread:** `ShmProducer.write` is called from the publisher's single
  worker thread (`PWK`). It is the sole writer; it MUST NOT block (no I/O, no
  alloc beyond the in-place slot stores). The seqlock ordering (slot.seq first,
  `latest_seq` last) is the only synchronization; no lock. `mark_shutdown`/`close`
  run on the producer's shutdown path. (`RUNTIME_shm_to_batch (b)` SHM rows.)
- **Consumer thread:** `ShmConsumer.read_latest` is called from the reader's
  tracker MAIN thread once per tick (and possibly the capture-poll path). It is
  lock-free (seqlock), copies out owned arrays, may return `None` under producer
  saturation (retry cap) or when `_closed` — callers must tolerate `None`.
  It may block only on `time.sleep(0)` yields between retries (bounded by RETRY).
- **Close-vs-read race (FIX vs current code):** in the current reader `close()`
  sets `_slot_views = []` then `_shm.close()` with no synchronization, so a
  concurrent `peek_latest` can `IndexError` / read a freed mmap (segfault).
  `ShmConsumer` must close-safely: set `_closed` first, have `read_latest`
  early-return `None` when `_closed` is observed, and serialize view-drop /
  `shm.close()` against an in-flight read (a lightweight lock around the
  view-access + close, or a quiescence check) so teardown from a signal handler
  on a different stack can never free memory a read is touching.
  (`live_shm_reader.md §4 / TOP CONCERN #1`, `RUNTIME_shm_to_batch H4`.)
- **Cross-process:** producer and consumer are separate OS processes; the only
  shared state is the mmap, mediated entirely by the seqlock. No in-process lock
  spans the boundary. This module never takes `_model_lock` or any pipeline lock
  (it predates the model entirely).

## 9. Open questions

1. **Owned-copy vs reusable double-buffer in `read_latest`.** Current code does
   4 fresh `np.array(copy=True)` per read (~18 MB/frame at 1200p). Keep the
   simple owned-copy contract, or add a caller-supplied scratch double-buffer to
   cut allocator pressure on the tracker thread? (audit names this MEDIUM but
   "by design".) Recommend: ship owned-copy; revisit only if the ledger shows it.
2. **Retry cap / observability.** `read_latest` returns `None` after N retries
   under producer saturation, silently skipping a frame. Expose a hit-counter /
   log, and is N=5 still right? (ARCH #4 latency-tail visibility.)
3. **Single-consumer assumption.** The seqlock is correct for 1 producer / N
   readers, but is there ever >1 reader process in the new design (e.g. viser on
   a separate process)? If strictly single-consumer, document it; if not,
   confirm N-reader safety (it holds — readers only read).
4. **Does `ShmFrame` stay separate from the higher `Frame` contract, or does the
   source adapter map one to the other?** Spec assumes the adapter maps
   `ShmFrame + layout → Frame + Cameras`. Confirm the `Frame` module owns the
   torch/Cameras build so this module stays import-light for the py3.8 producer.
5. **Best-effort SIGTERM/SIGINT unlink for clean `/dev/shm`** (ARCH H1
   recommendation): should `ShmProducer` register an atexit/signal best-effort
   `unlink()` for the normal-Ctrl-C case (keeping next-run unlink as the real
   reclamation), or leave all unlink to the source adapter's signal path?
