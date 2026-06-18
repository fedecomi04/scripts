# `frame.py` — the Frame data contract (layer: contract)

## (1) RESPONSIBILITY

Define `Frame` — the single immutable per-tick sensor tuple (rgb, depth, pose, intrinsics, mask, timestamp) with exact formats — plus the versioned SHM ring-buffer layout and its lock-free (de)serialize, so every data source (live publisher, recorded replay) produces *one* boundary type the rest of the pipeline consumes.

This is a **pure contract module**: it owns the *shape, units, conventions, and byte layout* of a frame. It does NOT own ROS, the publisher subprocess, the tracker, or any model state. Decoders/cameras are downstream of it.

---

## (2) PUBLIC INTERFACE

```python
# ---- Format constants (the contract's source of truth) ----
LAYOUT_VERSION: int                 # bump on ANY field/dtype/order change to header or slot
SHM_MAGIC: bytes                    # b"DGS\0" — guards a wrong-region attach
NUM_SLOTS: int = 4                  # ring depth; producer writes seq % NUM_SLOTS
HEADER_BYTES: int = 4096            # fixed header reservation (slot 0 starts after this)
DEPTH_SCALE_MM: float = 1000.0      # depth_ops re-exports; mm<->m boundary constant

@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics + image size. OpenGL/Nerfstudio camera frame downstream."""
    width: int; height: int
    fx: float; fy: float; cx: float; cy: float

@dataclass(frozen=True)
class Frame:
    """One synced sensor tuple. THE boundary every source must produce.
    rgb_bgr  : uint8  (H,W,3), BGR (publisher/cv2 order; consumer flips to RGB).
    depth_m  : float32(H,W),   metres, 0.0 == invalid (no return).
    mask_keep: uint8  (H,W),   1 == keep, 0 == robot/gripper-excluded.
    c2w_4x4  : float64(4,4),   OpenGL c2w (x right, +y up, +z back; look -z).
    stamp_sec: float,          CAPTURE event-time (sim or sensor clock) — NOT now().
    seq      : int,            monotonic producer sequence id (>=1; 0 == none yet).
    """
    seq: int
    stamp_sec: float
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    mask_keep: np.ndarray
    c2w_4x4: np.ndarray

# ---- SHM layout (single owner of the byte map; producer + consumer import this) ----
@dataclass(frozen=True)
class ShmLayout:
    """Resolved byte map for a fixed (H,W). Built once at ready-handshake."""
    version: int; height: int; width: int; num_slots: int
    intrinsics: Intrinsics
    header_bytes: int; slot_bytes: int
    rgb_off: int; depth_off: int; mask_off: int
    pose_off: int; seq_off: int; stamp_off: int

def compute_layout(intr: Intrinsics) -> ShmLayout:
    """Deterministic offset/size computation for (H,W); identical on both sides."""

def total_shm_bytes(layout: ShmLayout) -> int:
    """header_bytes + num_slots * slot_bytes (the SharedMemory size to allocate)."""

# ---- Header (de)serialize ----
def pack_header(layout: ShmLayout, *, latest_seq: int, ready: int, shutdown: int) -> bytes:
    """Pack the fixed header. LAYOUT_VERSION + magic written here."""

def read_header(buf: memoryview) -> ShmLayout | dict:
    """Parse the header; raise on magic mismatch OR version != LAYOUT_VERSION (loud)."""

def latest_seq_offset() -> int:
    """Byte offset of latest_seq in the header (consumer reads it lock-free, no full parse)."""

# ---- Slot views (zero-copy producer write / consumer read aliases) ----
def build_slot_views(buf: memoryview, layout: ShmLayout) -> list[dict[str, np.ndarray]]:
    """NUM_SLOTS dicts of np.frombuffer views over `buf` (pose/seq/stamp/rgb/depth/mask).
    Producer writes through them; consumer copies out of them. Same code both sides."""

def write_frame(buf: memoryview, slot_views, layout: ShmLayout, frame: Frame) -> None:
    """Seqlock WRITE: slot.seq=seq (tag FIRST) -> payload -> header.latest_seq=seq (LAST).
    Producer-only. No fence beyond CPython store order (x86-correct; see OQ4)."""

def read_latest(buf: memoryview, slot_views, layout: ShmLayout) -> Frame | None:
    """Seqlock READ: read latest_seq, copy slot, re-verify (<=5 retries). None if
    no frame / publisher saturating. Lock-free, single-consumer. Copies are owned."""

# ---- Convenience constructor for non-SHM sources (recorded replay) ----
def frame_from_arrays(seq, stamp_sec, rgb_bgr, depth_m, mask_keep, c2w_4x4) -> Frame:
    """Validate dtypes/shapes/contiguity and build a Frame (the recorded adapter's exit)."""
```

Notes on the interface:
- `read_latest`/`write_frame` are the **only** seqlock implementors; the duplicated `_compute_header_field_offsets` (one copy in reader, one in publisher today) collapses to `latest_seq_offset()` + `read_header` here.
- `Frame` is `frozen=True` so it cannot be partially mutated after a source produces it (Principle #2 "make illegal states unrepresentable" applied to the per-tick contract).

---

## (3) DEPENDS ON (other NEW modules only)

- **None at runtime.** `frame.py` is the lowest contract layer; it imports only `numpy`, `struct`, `dataclasses` (stdlib + numpy).
- `depth_ops.py` (the consolidation target from `DUP_depth_handling.md`) DEPENDS ON `frame.py` for `DEPTH_SCALE_MM` (or re-exports it) — not the other way around. To avoid a cycle, the canonical `DEPTH_SCALE_MM` / depth-band constants live in ONE of the two; per `DUP_depth_handling.md` §B the cleanest is: `frame.py` owns the SHM/format constants, `depth_ops.py` owns the numeric helpers and imports `DEPTH_SCALE_MM` from `frame.py`.
- `cameras_from_frame(...)` (the old `cameras_from_live_frame`) does NOT belong here — it imports nerfstudio `Cameras`. It moves to the source-adapter / pipeline layer and merely *consumes* a `Frame`. Keeping nerfstudio out of the contract module keeps this importable from the publisher's minimal py3.8 env if ever needed (today the publisher has its own copy of the layout code; this contract is the single definition both sides import — see OQ2).

---

## (4) CONSUMES / PRODUCES

CONSUMES (inputs to its functions):
- A writable `memoryview` over a `multiprocessing.shared_memory.SharedMemory.buf` (owned/created by the source adapter / publisher, NOT by this module).
- Raw decoded arrays from a source: BGR uint8 rgb, float32-metres depth (already mm→m converted and ZED-noised by the source — this module does NOT decode or noise), uint8 keep-mask, float64 OpenGL c2w, capture stamp.

PRODUCES:
- `Frame` (immutable) — the single object the tracker, FF dispatcher, viser bridge, and recorded/live adapters all speak. Down-contract: `depth_ops`/decoders read `frame.depth_m`; the camera builder reads `frame.c2w_4x4` + `Intrinsics`; the tracker reads `frame.rgb_bgr`/`mask_keep`/`stamp_sec`.
- `ShmLayout` + packed header bytes — the versioned wire contract between the source subprocess and the reader.

Invariants of the produced data (the format guarantees other modules may rely on):
- depth in **metres, 0==invalid** (NOT mm; NOT NaN-as-invalid). The mm↔m conversion is the source's job before `frame_from_arrays`/`write_frame`.
- c2w is **OpenGL** (`DUP_depth_handling.md` guardrail) — any OpenCV consumer must convert; this module never silently mixes conventions.
- `seq` strictly increasing, `0` reserved for "nothing published".

---

## (5) SOURCE MOVED IN

| Current `file:symbol` | Becomes |
|---|---|
| `live_shm_reader.py: LiveFrame` (dataclass) | `Frame` (frozen, renamed; `depth_m`/`mask_keep`/`c2w_4x4`/`stamp_sec`/`seq` kept) |
| `live_shm_reader.py: CameraIntrinsicsLite` | `Intrinsics` (frozen, renamed; same 6 fields) |
| `live_shm_reader.py: _HDR_FMT, _HDR_SIZE` + `live_ros_publisher.py: _HDR_FMT` (the duplicate) | one `_HEADER_STRUCT` + `LAYOUT_VERSION` in `frame.py`; both sides import it |
| `live_shm_reader.py: _decode_header` / `live_ros_publisher.py: _write_header` | `read_header` / `pack_header` |
| `live_shm_reader.py: _compute_header_field_offsets` + `HDR_OFFSETS` AND the verbatim publisher copy | `latest_seq_offset()` (only field still read post-init by the seqlock) + offsets inside `read_header` |
| `live_ros_publisher.py: _slot_layout` / `_total_shm_bytes` | `compute_layout` / `total_shm_bytes` |
| `live_shm_reader.py: LiveShmSubscriber._build_slot_views` AND `live_ros_publisher.py` slot-view build (`:616-638`) | one `build_slot_views` (shared by producer + consumer) |
| `live_shm_reader.py: peek_latest` (seqlock read body, `:386-414`) | `read_latest` (pure function; no `self`, no proc handle) |
| `live_ros_publisher.py: _process_synced_pair` SHM-write block (`:1015-1027`) | `write_frame` (pure function; the surrounding decode/noise/mask stays in the source) |

What does NOT move in (stays in the source adapter / pipeline, just *uses* `Frame`):
- `LiveShmSubscriber` process spawn/control-pipe, `peek_latest` retry-policy *caller*, atexit/close, `cameras_from_live_frame`, `wait_for_first_frame`, all `_send_command`/control ops. Those are the **source adapter**, not the contract.

---

## (6) DROPPED (current code NOT carried)

| Dropped | Reason | Audit ref |
|---|---|---|
| `LiveShmSubscriber.get_singleton` + `_singleton` class attr | Zero external refs; `_singleton` set but never read. | `live_shm_reader.md` §2 (high confidence) |
| `save_anchor_for_sam3`, `save_anchor_intrinsics_and_depth` (reader methods) | Zero caller-style refs; `live_session.py` uses its own module-level helpers; the publisher JSON-ops are never sent. | `live_shm_reader.md` §2 |
| The verbatim-duplicated `_HDR_FMT` + `_compute_header_field_offsets` (publisher copy) | Copy-paste coupling with only a "keep in sync" comment; silent layout drift = mis-read every field. Collapse to one definition. | `live_shm_reader.md` §3 (header coupling), `RUNTIME_shm_to_batch.md` notes |
| Per-byte tee + `dbg.flush()` in `_read_response` | Debug instrumentation hot loop on the control read path (not part of the Frame contract at all — it's a control-pipe concern that lives in the source adapter, and even there should drop the per-byte flush). | `live_shm_reader.md` §4 (per-byte flush) |
| `depth_mm` legacy field / a second mm→m conversion in the contract | `Frame` carries depth **in metres only**; conversion is the source's responsibility (single conversion site). Do not re-add a mm variant. | `LiveFrame` docstring (`:128-134`); `DUP_depth_handling.md` §B |
| FF-video machinery / oneshot FF path | Not in the ingest contract at all; `feedforward_video_out` has no writer (CLAUDE.md picker note). Frame contract has nothing to do with FF output. | CLAUDE.md "interactive object picker" note |
| Re-reading `latest_seq`/header via a full `_decode_header` on the hot read | Only `latest_seq` is needed per read; full header parse is init-time. `read_latest` uses `latest_seq_offset()` directly. | `live_shm_reader.md` §1 (`_decode_header` 1 caller, init only) |

Explicitly NOT dropped (would be a mistake): the 4-slot ring, the seqlock tag-first/publish-last order, `np.array(..., copy=True)` owned copies on read (callers need copies that survive the next producer write), the `magic` guard. These are load-bearing (`RUNTIME_shm_to_batch.md` H2, notes for the purge).

---

## (7) INVARIANTS PRESERVED

This is a contract module — it touches **no** gauss_params or identity buffers, so the gaussian/identity invariants (#1–#4, #8) are unaffected by construction. The ones it must honor:

- **CLAUDE.md depth/format guardrails (`DUP_depth_handling.md`):** depth is uint16-mm on disk / **float32-metres, 0==invalid** in `Frame`; c2w is **OpenGL** in the contract. `frame.py` is the single place these are declared, so other modules stop re-hardcoding `1e-3`/`3.0`/sign conventions. The module never converts conventions silently.
- **Principle #5 (event-time, not now()):** `Frame.stamp_sec` is the **capture** timestamp; the contract forbids any `time.time()` substitution. Downstream dt-math (KF if re-enabled) must read this field. (`ARCHITECTURE_PRINCIPLES.md` #5; CLAUDE.md KF wall-clock-dt note.)
- **Principle #2 (illegal states unrepresentable):** `Frame` and `Intrinsics` are `frozen` — a source cannot half-mutate a produced frame; arrays are validated for dtype/shape/contiguity at construction (`frame_from_arrays`).
- **Principle #8 (versioned contracts at boundaries):** `LAYOUT_VERSION` + `SHM_MAGIC` are stamped in the header and **checked on attach** (`read_header` raises loudly on mismatch). This is the boundary-versioning the audit demands so a publisher/reader layout drift fails clearly instead of mis-reading every field. (`ARCHITECTURE_PRINCIPLES.md` #8.)
- **Invariant #6 (background sky):** unaffected — `Frame` carries only sensor data, no background color.

---

## (8) THREADING

`frame.py` is **stateless pure functions + frozen dataclasses** — it owns no thread, no lock, no handle. Thread-safety is entirely a property of the seqlock protocol it implements and the discipline of its callers:

- **`write_frame`** — called **only** by the producer (publisher worker thread, single writer). Seqlock order (`slot.seq` first, payload, `latest_seq` last) is the synchronization; **no lock**, correct lock-free on x86 single-producer (`RUNTIME_shm_to_batch.md` H2).
- **`read_latest`** — called by the **consumer** thread(s) only (tracker MAIN tick in live; the recorded adapter). It does NOT block, NOT sleep beyond `time.sleep(0)` yields, NOT allocate beyond the 4 owned copies it must return. Up to 5 retries then returns `None` (frame-skip-under-load is intentional and should be observable, not silent — see OQ3). May be called from more than one consumer thread safely (all locals + owned copies), BUT the **source adapter** that owns the SHM handle must guarantee the buffer is not freed under an in-flight `read_latest` (the `close()` vs `peek_latest` race, `live_shm_reader.md` TOP CONCERN #1 / `RUNTIME_shm_to_batch.md` H4) — that guard is the adapter's job, NOT this module's, since this module never closes the segment.
- **Lock discipline:** none internal. The contract explicitly does **not** participate in `_model_lock` (that guards gaussian state, a different SSOT) nor the publisher pose/joint-history lock. Keeping `frame.py` lock-free and owner-less is what lets the tracker hot path stay allocation-/lock-light (Principle #4: no logging/IO/extra-lock on the hot read path).
- **What it may NOT block on:** ROS, subprocess pipes, GPU, disk, any worker. A `read_latest` that needs none of those is the whole point.

---

## (9) OPEN QUESTIONS

1. **Frozen `Frame` vs in-place batch building.** Today the consumer flips BGR→RGB, casts depth to a `(H,W,1)` tensor, and `.to(device)` — those allocations live in the source adapter / `depth_ops`. Confirm `Frame` should stay raw-numpy (no torch, no device) so the contract is import-light and env-portable, with all torch/device work downstream. (Recommended: yes.)
2. **Can the publisher (py3.8, `dynamic_gs_ros` env) import `frame.py` directly?** The publisher currently keeps its *own* copy of the layout code precisely so it doesn't import the heavy `dynamic_gs` package. If `frame.py` is kept numpy+struct-only (no nerfstudio/torch), both sides can import the *one* definition — eliminating the duplicate-drift risk. Needs a check that `frame.py`'s import graph stays minimal (it should, per §3). If not feasible, fall back to a generated/checked-in shared constant + a startup version assert.
3. **Surface frame-skip when `read_latest` exhausts retries?** Current code silently returns `None` after 5 tries (`RUNTIME_shm_to_batch.md` H2 rec). Should the contract expose a cheap counter/flag so the pipeline's always-on ledger (Principle #4) can report skip-under-load, or keep `None`-only and count at the caller?
4. **Memory-ordering on non-x86.** The seqlock has no explicit fence; correct on x86 store ordering only (`RUNTIME_shm_to_batch.md` H2). Document "x86-only" in the module, or add an explicit ordering primitive now? (Target hardware is x86; recommend a docstring note + a single `assert` site, defer the fence.)
5. **`LAYOUT_VERSION` bump policy.** Confirm that ANY change to header struct, slot order, dtypes, or `NUM_SLOTS` bumps `LAYOUT_VERSION`, and that `read_header` rejecting a mismatch with a "delete stale SHM / restart publisher" message is the desired loud-fail (mirrors the `.pt` config-fingerprint pattern, Principle #8).
6. **Reusable read buffers (Principle #6 / `live_shm_reader.md` #2).** `read_latest` allocates ~18 MB/frame at 1200p (4 owned copies). A double-buffer would cut allocator pressure but complicates the "copy survives next write" guarantee. Is per-tick churn acceptable for v1 (it's bounded, no leak), deferring buffer reuse to a later optimization?
