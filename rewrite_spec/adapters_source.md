# adapters_source.py — SPEC (blueprint, not code)

> Layer: **adapter** (hardware/middleware-specific, the ONE file a new user writes).
> Module owns the **producer** end of the Frame contract: it fills the SHM ring.
> The rest of the pipeline (`pipeline.py`, tracker, FF, viser) only ever *reads* SHM.

---

## (1) RESPONSIBILITY

Define the `FrameSource` interface and ship the producers that fill the publisher-owned SHM ring with `Frame`s — a `ReplaySource` that drives recorded-on-disk datasets so they "feel live", and thin `Ros1Source`/`Ros2Source` stubs that turn (rgb, depth, joint-state, camera-pose) topics into FK-posed, robot-masked `Frame`s — so the rest of the pipeline has exactly ONE ingest path (peek the SHM ring) regardless of where frames come from.

---

## (2) PUBLIC INTERFACE (the contract other modules call)

### The producer-side Frame (numpy, pre-SHM)

```python
@dataclass(frozen=True)
class Frame:
    """One synced sensor tuple in the producer's own (numpy/CPU) domain.

    This is the ONLY thing a source emits. Geometry is metric; depth is
    float32 metres (NOT uint16 mm — the disk/uint16 boundary is internal
    to ReplaySource). c2w is OpenGL 4x4 (camera->world), the scene's native
    convention. Immutable so it can be handed across the SHM boundary safely.
    """
    seq: int                 # monotone, source-assigned, starts at 1
    stamp_sec: float         # CAPTURE event-time (sim or sensor clock), NEVER now()
    rgb_bgr: np.ndarray      # (H,W,3) uint8, BGR (cv2 convention)
    depth_m: np.ndarray      # (H,W)   float32, metres, 0 == invalid/no-return
    mask_keep: np.ndarray    # (H,W)   uint8 {0,1}, 1 == keep (robot/gripper excluded)
    c2w_4x4: np.ndarray      # (4,4)   float64, OpenGL camera->world
```

### Intrinsics (carried out-of-band, set once at attach)

```python
@dataclass(frozen=True)
class Intrinsics:
    width: int; height: int
    fx: float; fy: float; cx: float; cy: float
```

### The interface every source implements

```python
class FrameSource(Protocol):
    """A producer of Frames that owns and fills the SHM ring.

    Contract: __init__/attach() creates the SHM segment (producer owns it),
    publishes the header (intrinsics + layout) and then drives frames into
    slots via a seqlock write. The pipeline never constructs this directly
    except through open_source()."""

    def intrinsics(self) -> Intrinsics: ...
        # Camera intrinsics, valid after attach(); used to build the header + Cameras.

    def attach(self, shm_name: str) -> None: ...
        # Create/own the SHM segment under shm_name, write the header, begin producing.
        # Blocks until the first Frame is published (so the reader can peek immediately).

    def next_frame(self) -> Optional[Frame]: ...
        # Produce ONE Frame, write it into the SHM ring (seqlock), and return it
        # (or None when the source is exhausted — ReplaySource at end-of-dataset).
        # Live sources block briefly for the next synced tuple; ReplaySource paces by stamp.

    def close(self) -> None: ...
        # Release the SHM segment + any subprocess/middleware handles. try/finally-safe.
```

### Factory + SHM reader shared by all sources

```python
def open_source(kind: Literal["replay","ros1","ros2"],
                data_dir: Path | None = None,
                shm_name: str = "/dgs_live_shm",
                replay_mode: Literal["paced","fast"] = "paced",
                **opts) -> FrameSource: ...
    # Construct + attach the requested source. data_dir required for "replay".
    # replay_mode (ReplaySource only): "paced" = emit each frame at its capture
    #   stamp relative to wall-clock start (REAL-TIME proxy: the consumer peek-latest
    #   then drops/trails exactly like live → honest real-time test). DEFAULT.
    #   "fast" = emit as fast as the consumer pulls, frame-exact, no drops
    #   (batch reprocessing / deterministic debugging; NOT a real-time proxy).
    # This is the ONE call pipeline.py makes to obtain its ingest producer.

class ShmRing:
    """Reader-side attach to the producer's SHM ring (lock-free seqlock peek).

    NOT a FrameSource — the consumer side. pipeline.py holds one of these and
    calls peek_latest() each tick. Producer-owns / consumer-attaches discipline."""

    def __init__(self, shm_name: str = "/dgs_live_shm"): ...
        # create=False attach + resource_tracker.unregister (reader never unlinks).

    def intrinsics(self) -> Intrinsics: ...
        # Read from the header that the producer wrote.

    def peek_latest(self) -> Optional[Frame]: ...
        # Lock-free seqlock copy of the freshest slot, or None (no frame yet / mid-write).

    def close(self) -> None: ...
        # Drop slot views + shm.close(). Sets _closed; peek_latest returns None after.
```

### Camera builder (numeric, used by the consumer)

```python
def camera_from_frame(frame: Frame, intr: Intrinsics,
                      device: torch.device) -> Cameras: ...
    # Build a single nerfstudio Cameras (OpenGL c2w[:3,:4]) on device. Stamps cam_idx=0.
```

> **Deliberately NOT in this module's public surface:** anchor/record/SAM3 control-pipe
> ops (`capture_anchor`, `start_recording`, `save_anchor_for_sam3`, `build_init_pcd`,
> `pause_gazebo`, …). Those belong to the *static-capture* tool, not the dynamic ingest
> adapter. See DROPPED.

---

## (3) DEPENDS ON (other NEW modules only)

- **`frame_contract`** (or wherever `Frame`/`Intrinsics`/the SHM header struct live) — IF the contract types are hoisted into a shared dataclass module so `pipeline.py` and `adapters_source.py` agree byte-for-byte on the SHM layout. If the rewrite keeps them co-located here, then this module has **no NEW-module dependency** and `pipeline.py` imports them FROM here. (OPEN QUESTION #1.)
- Nothing else. This module is a **leaf adapter**: it must NOT import the model, the gaussian_set, the tracker, the FF dispatcher, or the viser bridge. It produces `Frame`s; it knows nothing about gaussians.

External (non-NEW) deps it legitimately uses: `numpy`, `cv2`, `torch` (only for `camera_from_frame`), `multiprocessing.shared_memory`, `struct`, and — *only inside the ros1/ros2 stub bodies* — `rospy`/`rclpy` + the robot-mask FK helper (which is itself middleware-specific and may stay vendored alongside the stub).

---

## (4) CONSUMES / PRODUCES (data contracts)

**ReplaySource CONSUMES (in):**
- A recorded dataset dir `<data>/dynamic_scene/` (or any frame folder): `rgb/*.png` (BGR), `depth/*.tiff` (uint16 mm), `masks/*.png` (uint8 keep), `transforms.json` (per-frame OpenGL c2w + intrinsics + frame stamps). Read once into an ordered frame list; depth uint16→`depth_m = mm * 1e-3`.

**Ros1Source/Ros2Source CONSUME (in):**
- Topics: compressed RGB, raw 32FC1/16UC1 depth, `joint_states`, `gazebo_pose` (camera link pose). FK + slerp → per-stamp c2w; robot-exclusion mask render → `mask_keep`. (Mirrors today's publisher `_process_synced_pair`.)

**ALL sources PRODUCE (out):**
- `Frame` objects (numpy, the dataclass above), AND the side-effect of writing each into the **SHM ring** via the seqlock discipline:
  1. `slot["seq"] = seq` (tag first)
  2. payload (`pose/stamp/rgb/depth/mask`) copied in-place
  3. `header.latest_seq = seq` (publish last)
- The SHM **header** (written once at `attach`): magic `b"DGS\0"`, version, H, W, num_slots, fx/fy/cx/cy, slot byte offsets, `latest_seq`, `ready`, `shutdown`.

**Consumer reads (out → pipeline):** `ShmRing.peek_latest()` → `Frame` (lock-free copy); `camera_from_frame(...)` → `Cameras`. The tracker then builds its batch (BGR→RGB, host→GPU) — **batch construction is the pipeline's job, not this module's** (this module stays torch-light).

---

## (5) SOURCE MOVED IN (current file:symbol → what it becomes)

| Current | Becomes |
|---|---|
| `live_shm_reader.py :: LiveFrame` (dataclass, `depth_m`) | `Frame` (this module) — same fields, made `frozen`. |
| `live_shm_reader.py :: CameraIntrinsicsLite` | `Intrinsics` (this module). |
| `live_shm_reader.py :: _decode_header / _compute_header_field_offsets / HDR_OFFSETS / _HDR_FMT` | The SHM header codec — kept verbatim (the layout is a versioned contract), shared by producer + `ShmRing`. |
| `live_shm_reader.py :: LiveShmSubscriber._build_slot_views / peek_latest / _shm attach + resource_tracker.unregister` | `ShmRing` (reader side). `peek_latest` keeps the 5-retry seqlock loop; add the `_closed` early-return guard (H4). |
| `live_shm_reader.py :: cameras_from_live_frame` | `camera_from_frame` (this module). |
| `live_ros_publisher.py :: LivePublisher.__init__` (SHM alloc, slot-view build, `_write_header`, stale-name unlink) | The **producer** half shared by all sources: a `_ShmProducer` helper that `attach()` uses to create+own the segment and write slots. The intrinsics 4-tier fallback chain is dropped (see DROPPED). |
| `live_ros_publisher.py :: _slot_layout / _write_header / _StoredFrame` | `_ShmProducer` internals + the producer-side `Frame` latch. |
| `live_ros_publisher.py :: _process_synced_pair` (pose interp, rgb/depth decode, ZED-noise, mask render, SHM write) | Split: the **decode+pose+mask** body → `Ros1Source.next_frame()` (and a `Ros2Source` twin); the **SHM write** → `_ShmProducer.publish(frame)`. ZED-noise stays an opt-in hook inside the ros source (sim-only). |
| `live_ros_publisher.py :: _on_synced / _frame_queue / _worker_loop / start_worker` | The ros source's internal receive→queue(maxsize=4, drop-oldest)→produce loop. Kept as the bounded-queue load-shed pattern (ARCH principle #3). |
| `live_ros_publisher.py :: _on_joint_state / _on_gazebo_pose / _interpolate_c2w / RobotMaskGenerator` | Ros source internals — **now guarded by a dedicated history lock** (fixes H3, the one real corruption race). |
| `dynamic_gs_pipeline_recorded.py :: _tracker_tick` disk-frame advance + `dynamic_gs_datamanager.py :: get_current_dynamic_train_batch / set_dynamic_frame_idx` | The **frame-advance + pacing** logic moves into `ReplaySource.next_frame()` (read ordered disk list, pace by `stamp_sec`). The recorded pipeline's separate dynamic datamanager path is DELETED — replay now feeds the SAME SHM ingest the live path uses. |
| `_spawn_publisher` env-strip (LD_LIBRARY_PATH/CPATH/…) + PYTHONNOUSERSITE + bash-source-ROS wrapper | Kept verbatim **inside `Ros1Source`** (load-bearing, H10). ReplaySource needs NO subprocess (it runs in-process in the `dynamic_gs` env). |

---

## (6) DROPPED (current code NOT carried — with reason + audit ref)

- **The whole stdin/stdout JSON control-pipe** (`_send_command`, `_read_response`, `capture_anchor`, `start_recording`/`stop_recording`/`num_recorded`, `save_anchor_for_sam3`, `save_anchor_depth_intrinsics`, `build_init_pcd`, `pause/unpause_gazebo`). Reason: these are **static-capture orchestration**, not dynamic ingest. The dynamic pipeline only needs `peek_latest`. They belong to a separate capture tool. (Pipeline-recorded audit confirms recorded mode never uses the control pipe; RUNTIME target arch scopes this module to ingest.)
- **`live_ros_publisher.py :: _spawn_depth_republisher` / `_depth_republisher_proc`** — dead; never called (publisher.md §2, high confidence; raw 32FC1 subscribed directly).
- **`_total_shm_bytes`** — dead duplicate of inline math (publisher.md §2).
- **`wait_first_frame` (publisher op) + `_first_frame_event`** — no op-string caller (publisher.md §2). Replaced by `attach()` blocking on first publish.
- **`_KeyframeFilter` + record/keyframe-dedup + `_write_frame_to_disk` + replay-recording (`--record-replay`, `stream.bin`, `_replay_*`)** — disk recording + ORB-SLAM dedup are CAPTURE concerns, not ingest. Dropped from this module. (Replay *playback* stays; replay *recording* goes.)
- **`get_current_dynamic_eval_batch` / eval-split plumbing / `DynamicGSDataManager` dynamic-phase branches** — the dynamic phase is a tracker runtime (Invariant #4), not training; there is no eval. Replay reads frames straight off disk; no FullImageDatamanager wrapper for the dynamic stream. The static-phase datamanager stays (separate module), untouched.
- **FF-video machinery** (`feedforward_video_out`) — declared but no writer implemented (CLAUDE.md); not an ingest concern; dropped.
- **Oneshot-FF / `_oneshot_ff_due` plumbing** — belongs to the FF dispatcher module, never this one.
- **Intrinsics 4-tier fallback chain + hardcoded `_DATASETS_ROOT`** (publisher `__init__` god-function, publisher.md §4) — the ros stub takes intrinsics from `camera_info` (live) and ReplaySource from `transforms.json`; no glob-over-datasets fallback.
- **Module-level `[publisher-debug]` print block + numpy-alias monkeypatch** (publisher.md §4) — the print spew is dropped; the numpy-alias patch (if still needed by `urdfpy`) is scoped INSIDE the ros stub, not module-global.
- **`_accepted_dynamic_frames`** — write-only dead attribute across all 3 pipeline files (pipeline_recorded.md §2).
- **`cdn` threaded-through-as-None** and the recorded `_pick_d0_object`/`_bootstrap_d0` — these are tracker/FF/D0 logic, NOT ingest; they live in `pipeline.py`. This module ends at "a Frame is in SHM".

---

## (7) INVARIANTS PRESERVED (which CLAUDE.md invariants + how)

- **#5 (outputs/ suppressed)** — this module writes NOTHING under `outputs/`; ReplaySource reads `<data>/dynamic_scene/`, ros sources write only SHM. No new artifact tree.
- **#6 (background = Gazebo sky)** — not set here, but this module must NOT recolor/composite RGB; it passes sensor BGR through untouched so the pipeline's fixed background stays the only one.
- **#8 (identity buffers)** — this module **never touches gauss_params or the 4 identity buffers**. It produces sensor `Frame`s only; all gaussian state is downstream. By construction it cannot desync identity from geometry.
- **#9 (viser-direct only)** — no viewer/render here; ingest is render-free.
- **ARCH principle #5 (event-time, not now())** — `Frame.stamp_sec` is the CAPTURE timestamp (ros header stamp / replay transforms stamp). ReplaySource paces playback off these stamps; the ros source uses them for slerp. **No `time.time()` delta is ever fed to dt/velocity math.** (Fixes the documented KF wall-clock detune class at the source.)
- **ARCH principle #3 (bounded work / load-shed)** — ros source keeps the `queue(maxsize=4, drop-oldest)`; SHM is a fixed 4-slot ring. No unbounded growth on the ingest side.
- **ARCH principle #6 (guaranteed release)** — `close()` has try/finally release of SHM + subprocess; `ShmRing.close()` sets `_closed` and `peek_latest` early-returns (H4 fix).
- **SHM single-owner discipline (RUNTIME target arch)** — producer creates with `create=True` after unlinking stale; reader attaches `create=False` + `resource_tracker.unregister`; reader never unlinks; reclamation is next-launch unlink. Seqlock write order (seq→payload→latest_seq) preserved verbatim; documented x86-only (H2). **Add** best-effort unlink in the producer's SIGTERM/atexit for clean Ctrl-C (H1).

---

## (8) THREADING (which threads, what may/may-not block, lock discipline)

- **Consumer side (`ShmRing.peek_latest`)** — called on the pipeline **MAIN/tracker thread** only. Lock-free (seqlock). MUST NOT block (5-retry then return None → tick skips a frame, by design). MUST NOT allocate beyond the 4 fixed per-frame copies. Add a `_closed` flag check so a teardown-thread `close()` can't segfault an in-flight peek (H4).
- **ReplaySource** — runs **in-process** in the `dynamic_gs` env, on a **dedicated producer thread** (so it paces independently of the consumer, exactly like a real camera). Always writes through SHM (ONE ingest path; live + replay identical downstream — decided, OQ#2). Two modes:
  - **`paced` (DEFAULT, real-time proxy):** the producer thread sleeps until `start_wall + (frame.stamp_sec − stamp0)`, then writes the frame to SHM. It writes **every** frame on schedule; the *consumer* (main-thread `peek_latest`) naturally drops/trails when the tracker is slower than the cadence — reproducing live frame-drop + latency faithfully. This is the mode that would have exposed the zed_final 1.7 Hz slow-motion instead of hiding it.
  - **`fast` (deterministic):** lock-step — the producer advances only after the consumer has acked the current `seq` (so no frame is ever skipped), as-fast-as-the-consumer-pulls. Frame-exact + reproducible, for batch reprocessing / debugging. NOT a real-time proxy (by construction it can't drop). The one place replay and live diverge — and only when you opt in.
  Both honor a stop flag. No middleware, no subprocess, one owner of its frame list. The SHM write is single-writer.
- **Ros1Source/Ros2Source** — owns a **subprocess** (ROS env, py3.8) exactly like today's publisher, OR runs the middleware in-process if the env allows; internally: rospy receive thread(s) enqueue → ONE worker thread decodes+masks+publishes to SHM (single SHM writer). **Two locks, kept separate:**
  - the SHM-write is single-writer (no lock needed beyond the seqlock store order),
  - the **pose/joint history lists get a dedicated lock** held in `_on_joint_state`/`_on_gazebo_pose` (writers) and the read window of `_interpolate_c2w` (reader) — fixes H3, the one cross-thread *corruption* race. This lock is **separate from any model lock** (this module has no model lock) and must not serialize the SHM write.
- **Lock discipline summary:** this module introduces NO shared lock with the rest of the pipeline. Its only internal lock is the ros history lock. It never holds a lock across a decode/render/SHM-write of another frame. The `_model_lock` lives entirely downstream — this module is on the producer side of the contract and never sees it.

---

## (9) OPEN QUESTIONS for the human

1. **Where do `Frame`/`Intrinsics`/the SHM header codec live** — co-located in `adapters_source.py` (then `pipeline.py` imports them from here), or hoisted into a tiny shared `frame_contract` module? The latter is cleaner if any other module needs the types without importing the adapter; the former keeps the "one file a user writes" promise tighter. (Affects DEPENDS-ON.)
2. **RESOLVED (2026-06-18):** ReplaySource **always goes through SHM** (ONE ingest path, live + replay identical downstream). **`paced` is the default** (real-time proxy; consumer drops/trails like live). Frame-exact determinism is the opt-in **`fast`** lock-step mode (producer waits for consumer ack → no drops). So you get faithful real-time testing by default AND deterministic replay when you ask for it — no conflict. See §8.
3. **Static-capture orchestration home** — the dropped control-pipe ops (anchor/record/SAM3/build-seed/pause-gazebo) must land somewhere. Confirm they move to a separate `capture_tool` (not this module) and that `bootstrap_live.sh`/`capture_only.sh` retarget to it.
4. **ros1 vs ros2** — are both stubs in scope now, or ros1-only (matching the current Noetic publisher) with ros2 as a documented `NotImplementedError` stub? The audit only covers ros1 (Noetic).
5. **Who builds the GPU batch** — confirmed this module stops at numpy `Frame` + `camera_from_frame`, and the BGR→RGB + host→GPU + depth-filter batch build is `pipeline.py`'s job (keeps the adapter torch-light and env-portable). Confirm depth-filtering is NOT done here (CLAUDE.md H8 notes live tracker uses RAW depth; FF filters its own copy).
6. **RESOLVED (2026-06-18):** `paced` uses per-frame `stamp_sec` from `transforms.json` if present; otherwise it synthesizes them from a fixed `replay_fps` (the recording rate, e.g. 15) — `stamp = frame_idx / replay_fps`. Either way the cadence is wall-clock-paced off those stamps (principle #5). `replay_fps` is an `open_source` opt.
