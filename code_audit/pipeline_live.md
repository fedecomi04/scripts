# Code Audit — `dynamic_gs/dynamic_gs_pipeline_live.py`

Module: `LiveDynamicGSPipeline`, the live-ROS-fed dynamic pipeline (SHM frame source → XFeat tracker tick → off-thread feedforward → viser-direct push). Subclass of `DynamicGSPipelineBase`. **LIVE-PATH module.**

Audited 2026-06-17 against repo root `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`. All grep counts exclude self-references inside `dynamic_gs_pipeline_live.py` unless noted.

---

## 1) FUNCTION / CLASS MAP

### `LiveDynamicGSPipelineConfig(DynamicGSPipelineBaseConfig)` — `dynamic_gs_pipeline_live.py:57`
Dataclass config; adds SHM publisher knobs (`live_shm_name`, `live_keyframe_translation_m`, `live_keyframe_rotation_deg`, `live_wipe_root`). `_target` → `LiveDynamicGSPipeline`.
Refs: **2** — imported in `dynamic_gs_config.py:11` and instantiated in the `dynamic-gs-live` method config (`dynamic_gs_config.py:216`). Registered as ns-train entry point `dynamic-gs-live` (`pyproject.toml:23` → `DynamicGSLive`). **Entry point.**

### `LiveDynamicGSPipeline(DynamicGSPipelineBase)` — `dynamic_gs_pipeline_live.py:84`
The live pipeline class.
Refs: **4** (config `_target` factory, docstrings/notes). Instantiated by nerfstudio via the `_target` factory. **Entry point.**

### `LiveDynamicGSPipeline.__init__(self, config, device, test_mode, world_size, local_rank, grad_scaler)` — `dynamic_gs_pipeline_live.py:89`
Initializes live-specific state before `super().__init__`, flips datamanager+model to `"dynamic"` phase, spawns `LiveShmSubscriber` (publisher subprocess + SHM), waits for first frame, registers atexit cleanups + SIGINT/SIGTERM handlers, starts stdin watcher.
Caller: nerfstudio trainer setup via `_target`. **Entry point.**

### `LiveDynamicGSPipeline._cleanup_live_subscriber(self)` — `dynamic_gs_pipeline_live.py:197`
Closes the SHM subscriber + publisher subprocess, nulls `_shm_sub`.
Refs (all internal): registered `atexit` (`:161`), called in signal handler (`:173`). **NOT dead** — atexit + signal-driven.

### `LiveDynamicGSPipeline._cleanup_live_ff_dump(self)` — `dynamic_gs_pipeline_live.py:207`
Unlinks the per-PID FF input PNG from `/dev/shm`.
Refs (all internal): registered `atexit` (`:162`), called in signal handler (`:177`). **NOT dead.**

### `LiveDynamicGSPipeline._start_stdin_stop_watcher(self)` — `dynamic_gs_pipeline_live.py:216`
Spawns a daemon thread reading stdin: `stop`/`quit`/`exit` → set `_live_stop_requested`; bare-Enter → `_reselect_requested` (object switch).
Refs (all internal): called once in `__init__` (`:157`). **NOT dead.**

### `LiveDynamicGSPipeline._tracker_tick(self, step)` — `dynamic_gs_pipeline_live.py:243`
The per-tick core. Peeks SHM, dedups on `stamp_sec` (with sim-clock-reset re-arm), builds `Cameras`+batch, runs interactive picker / D0 bootstrap / motion estimate, decides FF cadence, publishes `_latest_tracker_frame`, pushes viser, fires `_on_tracker_frame`.
Overrides the abstract base hook (`dynamic_gs_pipeline_base.py:1100`, raises `NotImplementedError`). Driven by base loop at `dynamic_gs_pipeline_base.py:986`. Refs **23** repo-wide (mostly base + recorded). **Entry point (hook).**

### `LiveDynamicGSPipeline._pick_d0_object(self, camera, prefused_instance_ids) -> int` — `dynamic_gs_pipeline_live.py:346`
Live D0: project all means once, gate instances on `>=_D0_MIN_VISIBLE_GAUSSIANS` visible centres, rank by camera↔centroid distance. Returns 0 → defer.
Overrides abstract base hook (`base.py:1108`). Called by `_bootstrap_d0` (`:486`). Refs **10** repo-wide. **Entry point (hook).**

### `LiveDynamicGSPipeline._on_tracker_frame(self, camera, batch, cdn, is_first)` — `dynamic_gs_pipeline_live.py:414`
Post-tick callback: on first frame stamp the from-scratch timing section; otherwise, if FF is due this tick, dispatch async FF.
Overrides abstract base hook (`base.py:1118`). Called from base `_tracker_tick` flow / live `_tracker_tick:338`. Refs **10**. **Entry point (hook).**

### `LiveDynamicGSPipeline._resolve_anysplat_context_image_paths(self, target_frame_idx) -> (list[Path], list[int])` — `dynamic_gs_pipeline_live.py:438`
Live override: dump `_latest_live_rgb_bgr` to a fixed `/dev/shm` PNG and return its path (AnySplat worker reads from disk across conda envs).
Overrides base default (`base.py:2949`). Refs **2** (base def + this). Called by the AnySplat FF path in the base. **NOT dead** (override).

### `LiveDynamicGSPipeline._scene_c2w_for_frame(self, frame_idx) -> np.ndarray` — `dynamic_gs_pipeline_live.py:458`
Live override: return the c2w of `_latest_tracker_frame` (live frame_idx is a monotonic counter, not a dataset index). Pads (3,4)→(4,4).
Overrides base default (`base.py:2967`). Refs **2**. **NOT dead** (override).

### `LiveDynamicGSPipeline._bootstrap_d0(self, camera, batch)` — `dynamic_gs_pipeline_live.py:480`
First-tick bootstrap: pick D0 object; if 0, increment defer counter + occasionally log; else `_reseed_tracked_object`. `@torch.no_grad()`.
Overrides recorded/base `_bootstrap_d0` (recorded.py:356). Called in live `_tracker_tick:295`. Refs **2**. **NOT dead.**

### `LiveDynamicGSPipeline._reset_d0_guard(self)` — `dynamic_gs_pipeline_live.py:510`
Live override: reset `_tracker_tick_count` AND set `_d0_completed=True`. Called by the shared `_reseed_tracked_object` in the base.
Overrides base default (`base.py:1134`). Refs **2**. **NOT dead** (override; relied on by `_reseed_tracked_object`).

### `LiveDynamicGSPipeline._batch_from_live_frame(self, frame, device) -> dict` — `dynamic_gs_pipeline_live.py:523`
Convert a `LiveFrame` (BGR uint8 + float32 depth + uint8 keep-mask) into the Splatfacto batch dict. Tracker gets RAW depth (filter deferred to FF).
Refs (all internal): called once in `_tracker_tick:272`. **NOT dead.**

### Nested `_watch()` — `dynamic_gs_pipeline_live.py:224` (inside `_start_stdin_stop_watcher`)
Thread target reading stdin. Internal closure.

### Nested `_on_signal(signum, _frame)` — `dynamic_gs_pipeline_live.py:170` (inside `__init__`)
Signal handler closure for SIGINT/SIGTERM; runs cleanups, re-raises.

### Nested `_scalar(x)` — `dynamic_gs_pipeline_live.py:373` (inside `_pick_d0_object`)
Coerce a possibly-tensor camera intrinsic to a python float. Internal helper.

### Module attr `LiveDynamicGSPipeline._D0_MIN_VISIBLE_GAUSSIANS = 200` — `dynamic_gs_pipeline_live.py:340`
Class const; read in `_pick_d0_object:404` and logging `:496`. Used.

---

## 2) DEAD-CODE CANDIDATES

**None found.** Every method with zero *external* refs is one of:
- An **abstract-base hook override** driven by the base trainer loop: `_tracker_tick`, `_pick_d0_object`, `_on_tracker_frame` (base raises `NotImplementedError` at `base.py:1100/1108/1118`; loop calls `self._tracker_tick(step)` at `base.py:986`).
- A **base-default override**: `_reset_d0_guard` (base.py:1134), `_resolve_anysplat_context_image_paths` (base.py:2949), `_scene_c2w_for_frame` (base.py:2967), `_bootstrap_d0` (recorded.py:356).
- **Internally wired**: `_batch_from_live_frame` (called `:272`), `_start_stdin_stop_watcher` (called `:157`), `_cleanup_live_subscriber`/`_cleanup_live_ff_dump` (atexit `:161/:162` + signal handler `:173/:177`).

No symbol in this module is a genuine zero-ref suspect.

---

## 3) DATA-LIFECYCLE

### SHM subscriber + publisher subprocess (`_shm_sub`)
- **Create/attach**: `LiveShmSubscriber(...)` in `__init__:139`; blocks on `wait_for_first_frame(timeout_s=30.0)` (`:153`). Owns the publisher subprocess + POSIX SHM.
- **Read**: `peek_latest()` each tick (`:250`); intrinsics via `_shm_sub.intrinsics` (`:145,:271`).
- **Free**: `_cleanup_live_subscriber()` → `sub.close()`. Triggered three ways: `atexit` (`:161`), explicit SIGINT/SIGTERM handler (`:173`), or — note — **never on a clean `stop`/`max_num_iterations` return path inside this module** (relies on atexit firing at interpreter shutdown). The class docstring + signal-handler comment (`:164-169`) acknowledge atexit is unreliable on SIGTERM, which is why the signal handlers exist. **Low-risk gap**: if the trainer thread exits the loop on `_live_stop_requested` but the process keeps running (e.g. nerfstudio teardown), the publisher stays alive until atexit. Idempotent: `close()` nulls `_shm_sub`, second call early-returns (`:199`). Swallows all exceptions (`:203`) — a failed close is silent (orphan publisher possible; CLAUDE.md memory `project_live_publisher_restart_cleanup.md` documents the orphan-publisher failure mode this guards against).

### `/dev/shm` FF input PNG (`dgs_live_ff_frame_<pid>.png`)
- **Write**: `_resolve_anysplat_context_image_paths:455` (`cv2.imwrite`), once per FF call, fixed filename per PID. Overwritten each call (not accumulated).
- **Free**: `_cleanup_live_ff_dump()` unlinks it (atexit `:162` + signal `:177`). Per-PID name → no cross-run collision. **Clean.**

### `_latest_live_rgb_bgr` (np.ndarray, CPU)
- **Write**: `_tracker_tick:321` (`self._latest_live_rgb_bgr = latest.rgb_bgr`) — a **reference** into the `LiveFrame` returned by `peek_latest()`. Initialized `None` in `__init__:103`.
- **Read**: `_resolve_anysplat_context_image_paths:446` on the **FF bg thread**.
- **Lifecycle concern**: this is a bare ndarray reference, not a copy. It is read on the FF thread at FF-execution time, but written by the tracker thread every tick. See §4 race note. Not freed explicitly (GC'd when overwritten); no leak.

### `_latest_tracker_frame` (TrackerFrame dict)
- **Write**: `_tracker_tick:323` **rebinds** the attribute to a *new* dict each tick (camera, cdn=None, batch, frame_idx, stamp_sec).
- **Read**: passed by reference to `_dispatch_feedforward_async` (`_on_tracker_frame:432`) — that captured reference is safe (rebinding doesn't mutate the prior dict). BUT `_scene_c2w_for_frame:463-465` reads `self._latest_tracker_frame` **freshly** on the FF thread, not the dispatched snapshot. See §4 race note.
- Holds GPU tensors (`camera`, `batch["image"/"depth_image"/"mask"]`). One frame's worth retained at a time; replaced each tick. The in-flight FF thread pins one extra (its captured `target_frame`). Bounded → no leak.

### Per-tick GPU/heap allocations (`_batch_from_live_frame`, `_tracker_tick`)
- `_batch_from_live_frame:531/537/538`: three new GPU tensors per tick (image float32 H×W×3, depth float32 H×W×1, mask bool H×W×1). At 1920×1200 the image alone is ~27 MB/tick. These are freed when `_latest_tracker_frame` is rebound next tick (minus the one pinned by an in-flight FF). Expected, bounded.
- `cameras_from_live_frame` (`:271`) allocates a `Cameras` per tick — small.
- `_pick_d0_object:381` allocates a full `means_cam` (N×3) projection **only during D0 bootstrap** (not steady state) — fine.

### The 4 identity buffers (`object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`)
- This module **reads** `object_instance_ids` (`_bootstrap_d0:485`) and **writes** `object_flags` indirectly via `_reseed_tracked_object` (`:508`, base). It does **not** touch `sam3d_init_target_flags` or `inserted_flags` directly. **Invariant-protected** (CLAUDE.md Invariant #8): `object_flags` written by D0 selection here is correct/expected; `sam3d_init_target_flags` all-zeros is expected. No desync introduced by this module.

### `.pt` warm-cache (`post_fusion_state.pt`)
- Not loaded/saved in this module — handled by the base/persistence on warm-restart. `live_wipe_root=True` would let the publisher wipe it (`:75-81` docstring warns), but that is the publisher's action, not this module's.

### Phase flips
- `datamanager.set_phase("dynamic")` (`:124`) + `model.set_phase("dynamic")` (`:129`) in `__init__`. The model flip is load-bearing: it keeps `means.requires_grad=True` so FF-insert `register_hook` works (comment `:125-129`). One-shot, no leak.

---

## 4) DESIGN SMELLS

### HIGH — FF bg thread reads tracker-thread-mutated state freshly (frame skew / mild race)
`_dispatch_feedforward_async` snapshots `_latest_tracker_frame` by reference, but two pieces of FF input are re-read from `self.*` at FF *execution* time on the bg thread:
- `_resolve_anysplat_context_image_paths:446` reads `self._latest_live_rgb_bgr` (rebound every tick by `_tracker_tick:321`).
- `_scene_c2w_for_frame:463-465` reads `self._latest_tracker_frame` (rebound every tick by `_tracker_tick:323`).

Because the tracker keeps ticking while FF runs (FF is off-thread, single-in-flight), by the time FF samples these the live RGB and the c2w can be several ticks newer than the `cdn`/`batch` the FF dispatched on. The RGB dump and the back-projection pose therefore may not correspond to the CDN frame the inserts were computed against. CLAUDE.md already documents an FF/CDN frame-consistency class of bug ("FF ICP places inserts in a DIFFERENT frame than the CDN judges them in"). This is a *different* skew (RGB/pose vs CDN) but same family. **Recommendation:** read RGB + c2w from the dispatched `target_frame` snapshot, not from `self.*`. The `target_frame` dict already carries `camera` and `stamp_sec`; add the BGR frame to the snapshot and have `_scene_c2w_for_frame` accept/use the passed frame. (No source edits made — flagged only.)
Note: it's a *value*-skew, not a tearing race — Python attribute rebinds are atomic, so no corruption — but the FF consumes a mismatched frame.

### MEDIUM — bare ndarray reference shared across threads (`_latest_live_rgb_bgr`)
`_tracker_tick:321` stores a reference into the `LiveFrame.rgb_bgr` buffer; the FF thread `cv2.imwrite`s it later (`:455`). If `peek_latest()` ever returns a view into a recycled/over-written SHM staging buffer rather than a fresh copy, the FF thread could read a half-written frame. Depends on `LiveShmSubscriber.peek_latest` copy semantics (out of scope of this file). The comment at `:445` claims serialization by `_anysplat_slot_lock` protects the *dump filename*, but that lock does **not** protect the *source ndarray* from tracker-thread overwrite. **Recommendation:** snapshot `latest.rgb_bgr.copy()` into the dispatched `target_frame` at tick time.

### MEDIUM — `_tracker_tick` is a god-method (~95 lines, many responsibilities)
`_tracker_tick:243-338` does: stop-check, SHM peek, sim-clock-reset dedup, camera/batch build, object-mask cache invalidation, interactive picker, D0 bootstrap, motion estimate, FF-cadence decision, state publish, 4 viser pushes, and the post-tick hook. The cadence-gating block (`:305-317`) carries a 13-line comment explaining a subtle ordering invariant (`_tracker_tick_count + 1`). It is the hottest live path and the hardest to reason about. Mostly inherent to the design, but the picker/bootstrap/motion branch (`:286-303`) and the publish block (`:323-338`) could be extracted for clarity. Low-priority given imminent purge.

### LOW — `cdn` is unconditionally `None` in this module, yet threaded through state
`_tracker_tick:317` sets `cdn = None` (CDN render moved to the FF bg thread), then stores it in `_latest_tracker_frame["cdn"]` (`:326`) and passes it to `_on_tracker_frame(..., cdn, ...)` (`:338`), whose signature still takes `cdn` (`:418`) and ignores it. The `cdn` parameter on `_on_tracker_frame` is now vestigial in the live path (and the base hook signature). The dispatched FF recomputes the CDN on its own thread (`_feedforward_threaded` in base). Not a bug — but the `cdn` plumbing through tick → frame dict → hook is dead weight in live mode. The `_feedforward_threaded` "cdn is None → render it" path (base.py:3144) is exactly what consumes the `None`. Misleading-naming-adjacent: a reader expects `cdn` to carry a tensor.

### LOW — broad `except Exception: pass` swallows in cleanup + watcher
`_cleanup_live_subscriber:203`, `_cleanup_live_ff_dump:213`, `_start_stdin_stop_watcher:235`, and the signal handler's two try/except (`:174,:178`) all swallow silently. Defensible for teardown idempotency, but a failing `sub.close()` (orphan publisher) is the documented top failure mode (`project_live_publisher_restart_cleanup.md`) and goes unlogged. **Recommendation:** at least log at debug level in `_cleanup_live_subscriber`'s except.

### LOW — `_resolve_anysplat_context_image_paths` imports `cv2` lazily on the FF thread every call
`:450` `import cv2` inside the method (per FF call). Cheap after first import (module cache), but the try/except-around-import (`:449-453`) is per-call overhead on the FF path. Minor.

### LOW — `_scene_c2w_for_frame` ignores its `frame_idx` argument
`:458` takes `frame_idx` but never uses it (live frame_idx is a meaningless counter; it returns the current tracker c2w). Documented in the docstring, so intentional — but the parameter is dead. Compounds the §4-HIGH skew (it can't even key on the dispatched frame because it discards the index).

### INFO — config fields all read
`live_shm_name`/`live_keyframe_translation_m`/`live_keyframe_rotation_deg`/`live_wipe_root` are all consumed in `__init__:141-144`. No dead config fields declared on `LiveDynamicGSPipelineConfig`.

---

## Invariant-protected items touched (NOT flagged as bugs)
- `model.set_phase("dynamic")` keeping `means.requires_grad=True` (`:125-129`) — required for FF `register_hook`; not the static means-LR=0 invariant (that's the static phase). Dynamic-phase LRs are zeroed via `_ZERO_LR_OPTIMIZERS` in config, not here.
- `object_flags` written by D0 selection (via `_reseed_tracked_object`) — Invariant #8, expected.
- `object_instance_ids` read-only here — Invariant #8.
