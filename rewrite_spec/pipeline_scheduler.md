# `pipeline_scheduler.py` — the orchestrator god-file (layer: orchestrator)

> This is the SHORT god-file that the rewrite keeps deliberately central: it builds the
> run (open source → static.build → dynamic loop) and owns the **scheduling policy** + the
> **3-thread model** (tracker on main / FF on a single-in-flight bg thread / viser daemon).
> It collapses the current `Recorded`/`Live` subclasses into ONE `DynamicLoop` core fed by a
> `FrameSource` adapter. It is the ONLY place threads are spawned, joined, and coordinated.

---

## (1) RESPONSIBILITY

Drive the whole pipeline from one place — wire the source adapter, the static build, and the
dynamic runtime together, then run the dynamic loop's scheduling policy: **tracker every tick
on the main thread; decide-FF-once-per-tick via a pure `ff_due` predicate; dispatch FF on a
single-in-flight background thread (skip if a prior FF is still running, i.e. skip-if-behind);
keep the viser render thread fed** — all coordinated through the one shared `_model_lock` and
the one `GaussianSet` SSOT, with bounded FF growth and guaranteed thread/resource teardown.

---

## (2) PUBLIC INTERFACE (the contract other modules / the entry script call)

```python
# ---------- top-level run entry (what bootstrap_live.sh / ns-train wiring calls) ----------

@dataclass
class RunSpec:
    """Everything needed to build + run a dynamic session. Built by config.py from CLI/env
    (NO os.environ reads in this module — config folds DGS_* in already)."""
    mode: Literal["live", "replay"]      # selects the FrameSource adapter kind
    data_dir: Path                        # dataset root (warm-cache, replay frames, debug dirs)
    runtime_cfg: "RuntimeConfig"          # scheduling + FF cadence + tracker/viser knobs (config.py)
    device: torch.device

def run_dynamic(spec: RunSpec) -> None:
    """Build (source + warm-loaded model/GaussianSet + tracker + FF dispatcher + viser bridge),
    then run the dynamic loop to completion (source exhausted / stop requested), then tear down.
    The ONE call the launcher makes for the dynamic phase. Blocks on the main thread."""


class DynamicLoop:
    """The collapsed Recorded+Live core. One class, fed by a FrameSource; no subclasses.
    Owns thread spawn/join, the per-tick schedule, and the single _model_lock RLock."""

    def __init__(self,
                 source: "FrameSource",            # adapters_source.open_source(...) — producer of Frames
                 gaussians: "GaussianSet",         # the SSOT (warm-loaded), shares _model_lock
                 model: "SceneModel",              # render + phase/LR policy (set_phase("dynamic"))
                 tracker: "XFeatTracker",          # main-thread rigid-object tracker
                 ff: "FeedforwardDispatcher",      # owns AnySplat worker + single-in-flight slot
                 viser: "ViserBridge",             # read-only render thread + camera-feed push
                 cfg: "RuntimeConfig") -> None:
        """Wire collaborators. Create the shared _model_lock here and hand the SAME instance to
        gaussians, model.attach_render_lock, viser, and ff. Register teardown (atexit + SIGINT/
        SIGTERM). Spawn NO worker thread yet (viser/FF threads start on first need)."""

    # ---- the trainer-driven entry (nerfstudio NoSaveTrainer calls this each step) ----
    def step(self, step_idx: int) -> "ZeroLoss":
        """One dynamic step = one tracker tick + maybe-dispatch-FF + viser kick. Returns the
        zero-loss dummy (Invariant #4: no gradient descent). This IS get_train_loss_dict's body."""

    # ---- the scheduling policy (pure; the heart of this module) ----
    def _ff_due(self, tick_count: int, *, is_first: bool, stamp_sec: float) -> bool:
        """PURE predicate, evaluated EXACTLY ONCE per tick and cached in _ff_due_this_tick.
        False when: is_first (D0 not safe yet) / FF mode off / cadence not hit (tick%N) /
        within min-gap of last dispatch. Uses cadence (tick count) + event-time gap, never now()."""

    def _tracker_tick(self, frame: "Frame") -> "MotionEstimate | None":
        """MAIN thread. Build batch (BGR->RGB, host->GPU, RAW depth) + Cameras from the Frame;
        render the object-footprint mask UNDER _model_lock; run tracker.seed (D0) or tracker.track;
        on success write the rigid pose via gaussians.write_object_pose (UNDER lock). Returns the
        estimate (None if no frame / deferred D0). SOLE writer of the rigid transform + object_flags."""

    def _dispatch_ff_if_due(self, snapshot: "TickSnapshot") -> bool:
        """MAIN thread. If _ff_due_this_tick and the FF slot is free, hand the FROZEN per-tick
        snapshot (camera, batch, rgb_bgr copy, stamp) to ff.dispatch(...) which spawns the bg
        thread. Non-blocking; returns False (skip-if-behind) when the slot is held. Sole place FF starts."""

    # ---- lifecycle ----
    def run(self) -> None:
        """Pull frames from the source (peek SHM ring) and call step() until exhausted/stopped.
        Used by replay/standalone; under nerfstudio the Trainer drives step() instead (OQ2)."""
    def request_stop(self) -> None:
        """Idempotent: set the stop flag (stdin 'stop'/SIGTERM/source-exhausted all route here)."""
    def teardown(self) -> None:
        """Drain the in-flight FF slot, stop+join viser, close the source, write timing report,
        save final snapshot. try/finally-safe, idempotent (guarded by a _torn_down flag)."""


@dataclass(frozen=True)
class TickSnapshot:
    """The immutable per-tick handoff to FF (fixes the H4/H5 staleness class). Rebound each
    tick on the main thread; the FF bg thread owns the instance it was dispatched with and
    reads camera/rgb/stamp ONLY from here — never re-reads self.* live state."""
    camera: "Cameras"            # OpenGL c2w + K at dispatch time
    batch: dict                  # GPU rgb/depth(RAW)/keep-mask tensors for this tick
    rgb_bgr: np.ndarray          # COPY of the source RGB (not a live SHM view)
    stamp_sec: float             # capture event-time
    frame_seq: int               # source sequence id (for debug naming + frame correspondence)
```

Notes:
- The whole `DynamicGSPipelineBase` + `RecordedDynamicGSPipeline` + `LiveDynamicGSPipeline`
  surface collapses to `DynamicLoop` + `run_dynamic` + the `TickSnapshot` contract. The
  recorded/live difference becomes the *FrameSource* (`ReplaySource` vs `Ros1Source`), not a
  pipeline subclass.
- `step()` is the body the nerfstudio trainer's `get_train_loss_dict` delegates to (the thin
  trainer override stays in the framework-glue module, not here).

---

## (3) DEPENDS ON (other NEW modules only)

- **`adapters_source.py`** — `FrameSource` / `ShmRing` / `open_source(...)` (the ONE ingest path; peek SHM each tick) and `camera_from_frame`.
- **`frame.py`** — `Frame` / `Intrinsics` contract (what the source produces; the batch builder consumes).
- **`gaussian_set.py`** — `GaussianSet` (the SSOT; `write_object_pose`, `set_object_flags`, `snapshot`, `cull`/`insert` invoked indirectly by FF). The scheduler creates the shared `_model_lock` and passes it into the ctor.
- **`scene_model.py`** — the splat model (render under lock, `set_phase("dynamic")`, `attach_render_lock`).
- **`dynamic_track.py`** — `XFeatTracker` (`seed`/`track`, `TrackerInputs`/`MotionEstimate`).
- **`feedforward_dispatcher.py`** — the FF module that owns the AnySplat worker, the single-in-flight slot, CDN render, cull+insert (the scheduler only calls `ff.dispatch(snapshot)` and queries `ff.slot_busy`). *(Sibling spec not yet written; this module depends on its `dispatch`/`slot_busy`/`drain`/`close` surface — OQ3.)*
- **`viser_bridge.py`** — `ViserBridge` (read-only render thread, `request_render`, camera-feed push, `keep_alive_until_shutdown`). *(Sibling spec not yet written — OQ3.)*
- **`config.py`** — `RuntimeConfig` (cadence `feedforward_recurring_every_n_ticks`, `*_min_gap_s`, `dynamic_steps_per_frame`, FF-mode, growth cap, keep-viser-alive, tracker/viser sub-configs). All `DGS_*` env resolution lives there, not here.
- **a timing ledger module** (`timing.py` or equivalent) — p90/p99 always-on, debug opt-in OFF the hot path.
- **the warm-cache loader** (`static_persist.py` / `persistence`) — `load_post_fusion_state` into the `GaussianSet` at build time (called from `run_dynamic`, not `DynamicLoop.__init__`).

It does NOT import the decoders (`anysplat_decode`/`rgbd_decode`) directly — those are inside the FF dispatcher. It does NOT import ROS/cv2-heavy ingest internals — those are inside the source adapter.

---

## (4) CONSUMES / PRODUCES

**CONSUMES (in):**
- `Frame` objects via `source.peek_latest()` / the SHM ring (rgb_bgr uint8, depth_m float32 metres 0==invalid, mask_keep uint8, c2w OpenGL, stamp_sec capture-time, seq).
- `Intrinsics` (once, at attach) to build `Cameras`.
- A warm-cache `.pt` (loaded into `GaussianSet` before the loop starts).
- `RuntimeConfig` (scheduling cadence, gaps, growth cap, mode, keep-alive).
- `MotionEstimate` from the tracker; the FF dispatcher's `slot_busy` flag.

**PRODUCES (out):**
- Per-tick: a rigid pose write into `GaussianSet` (`write_object_pose`) on success; an FF dispatch on due ticks; a viser render kick.
- The `TickSnapshot` handed to the FF dispatcher (immutable per-tick boundary).
- A zero-loss dummy back to the trainer each step (Invariant #4).
- `timing_report.txt` + a final `post_dynamic_state.pt` snapshot at teardown.
- Stop/teardown side-effects (drain FF, join viser, close source).

**Data-format guarantees relied on:** depth in metres (0==invalid), c2w OpenGL, `stamp_sec` is capture event-time (used for the FF min-gap, never `time.time()` deltas for any dt math) — all from `frame.py`.

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `pipeline_base.py:get_train_loss_dict` (tick → maybe-FF → zero-loss) | `DynamicLoop.step` (the per-step body) |
| `pipeline_base.py:_recurring_ff_due` | `DynamicLoop._ff_due` (kept PURE; cadence + event-time min-gap; `is_first` guard kept — load-bearing per RUNTIME_ff_dispatch Inv#4/H2) |
| `pipeline_live.py:_tracker_tick` + `pipeline_recorded.py:_tracker_tick` (the two god-methods) | ONE `DynamicLoop._tracker_tick(frame)`; the recorded/live divergence (disk-advance vs SHM-peek) moves entirely into the `FrameSource` |
| `pipeline_live.py:_batch_from_live_frame` + `pipeline_recorded.py` depth-filter-at-source block | the batch builder inside `_tracker_tick` (BGR→RGB, host→GPU, RAW depth to tracker; FF filters its own copy) — single site for both modes |
| `pipeline_base.py:_dispatch_feedforward_async` + `_feedforward_threaded` slot-acquire/release wrapper | `DynamicLoop._dispatch_ff_if_due` (the *decision* + handoff) calling `ff.dispatch(...)`; the slot lifecycle + bg-thread body moves into `feedforward_dispatcher.py` |
| `pipeline_base.py:_ff_due_this_tick` decide-once flag + the `_tracker_tick_count + 1` ordering invariant | `DynamicLoop._ff_due_this_tick` (decided once in `step`, read by `_dispatch_ff_if_due`; same-thread, no race) |
| `pipeline_base.py:_latest_tracker_frame` dict + `_latest_live_rgb_bgr` | `TickSnapshot` (frozen, carries the RGB **copy** + camera + stamp — fixes H4/H5; FF reads ONLY the snapshot) |
| `pipeline_base.py:_apply_motion_estimator` (the orchestration half: render mask under lock → tracker → write transform under lock) | `_tracker_tick` body; the XFeat call goes to `tracker.track`; the rigid write goes to `gaussians.write_object_pose` |
| `pipeline_base.py:_render_object_mask_cached` / `_invalidate_object_mask_cache` | kept as a `_tracker_tick`-local cached render (per-tick, under lock); double-checked so the FF read can't tear (H3) — OR moved to `gaussian_set`/`scene_model` (OQ5) |
| `pipeline_base.py:_object_crop_bbox` / `_crop_for_xfeat` | the tracker-input crop builder; the `means`/`ids` read is now via `gaussians.snapshot()` (closes H1 — no unlocked model read) |
| `pipeline_live.py:_bootstrap_d0`/`_pick_d0_object` + `pipeline_recorded.py` twins + `_reseed_tracked_object` | `DynamicLoop` D0/reselect logic: pick reads `gaussians.snapshot()` (UNDER lock semantics), reseed calls `gaussians.set_object_flags` + `tracker.seed` |
| `pipeline_base.py:_capture_static_sequence_total`/`_render_static_sequence_section` + `_write_timing_report` | the timing-report assembly in `teardown` (delegates rendering to the timing module) |
| `pipeline_base.py` 8 `atexit` `_cleanup_*` + live `_cleanup_live_subscriber`/`_cleanup_live_ff_dump` + signal handlers + `_save_final_snapshot_if_enabled` | ONE `DynamicLoop.teardown` (try/finally, idempotent) wiring source.close / ff.close / viser.stop / snapshot save / timing flush |
| `pipeline_recorded.py:block_until_viser_shutdown` + `keep_viser_alive_*` | `teardown`'s optional keep-alive (delegates to `viser.keep_alive_until_shutdown`); main-thread, pre-finalization (kept rationale) |
| `pipeline_live.py:_start_stdin_stop_watcher` (stop/reselect) | a small daemon in `DynamicLoop` that only flips `request_stop()` / a reselect flag (no model touch) |
| `pipeline_recorded.py:dynamic_steps_per_frame` pacing | the source's pacing (`ReplaySource`) + the loop's per-frame advance; the knob stays in `RuntimeConfig` |
| `pipeline_base.py:_setup_viser_direct` lock-swap (`server.model_lock = self._model_lock`) | `DynamicLoop.__init__` passes the shared lock into `ViserBridge` at construction (fixes the H-viser-swap ordering) |

---

## (6) DROPPED (current code NOT carried — with reason + audit ref)

| Dropped | Reason | Audit ref |
|---|---|---|
| `RecordedDynamicGSPipeline` / `LiveDynamicGSPipeline` as separate classes | The recorded/live split collapses to ONE `DynamicLoop` + a `FrameSource` adapter — the rewrite goal. Disk-advance vs SHM-peek is the source's job. | pipeline_recorded.md / pipeline_live.md (the two `_tracker_tick` twins); ARCH #9 |
| `_oneshot_ff_due` + the inlined oneshot block in `get_train_loss_dict` (`feedforward_oneshot_step`) | Zero-ref method; the oneshot path is a dead duplicate of the recurring gate. Only the recurring Mode-B FF survives. | pipeline_base.md §2 (`_oneshot_ff_due` 0 refs, high); RUNTIME_target_architecture (single recurring FF path) |
| `feedforward_video_out` / `feedforward_video_fps` / `_feedforward_video_writer` / `_cleanup_feedforward_video_writer` | No mp4 writer was ever implemented; declared-never-read. | pipeline_base.md §2 (high); CLAUDE.md picker note |
| `feedforward_anchor_frame` config | Declared, 0 reads. | pipeline_base.md §2 (high) |
| `_accepted_dynamic_frames` (all 3 sites) | Write-only dead attribute, never read anywhere. | pipeline_recorded.md §2 (high) |
| `_force_viser_direct_push` (dead alias) + `_render_from_camera_at_scale` (reduced-res CDN reverted) | 0 refs; the reduced-res render was explicitly reverted. | pipeline_base.md §2 (both high) |
| `_force_viewer_rerender` + the `get_training_callbacks` `_trainer` stash | No-op under Invariant #9 (viser-direct only; no NS viewer). Drop unless static-ckpt NS-viewer debugging is kept (OQ). | RUNTIME_target_architecture DELETE; RUNTIME_tracker_tick H6 |
| Path-A viser stubs called per tick (`push_tracker_transform`, `add_ff_insert_chunk`, `_push_viser_direct_transforms` per-tick) | No-op stubs under the mandated config; the camera-feed/transform push collapses to the `ViserBridge` surface. | RUNTIME_target_architecture DELETE; viser_direct.py:506-528 |
| `cdn` threaded through `_tracker_tick`→frame-dict→`_on_tracker_frame` as always-`None` | The CDN is rendered on the FF bg thread now; the `cdn` param is dead weight in the tick path. The FF dispatcher renders its own CDN from the `TickSnapshot`. | pipeline_live.md §4 (LOW, `cdn` always None); pipeline_recorded.md §4 |
| The `model.info` rgbd-cull path (`_feedforward_delete_in_region`) + whole rgbd-decode branch of `_run_feedforward` | Unreachable on the default `anysplat_decode` live path; the AnySplat direct-projection cull proves `model.info` is unnecessary — removing it kills a cross-thread race class (H2). Lives in FF dispatcher anyway, not here. | pipeline_base.md §4 (rgbd unreachable); RUNTIME_ff_dispatch H2; RUNTIME_target_architecture DELETE |
| `DGS_*` env reads scattered in the tick/gate | All env resolution centralized in `config.py` → `RuntimeConfig`; this module reads typed config only. | ARCH #4/§ config; dynamic_track.md (same pattern) |
| `_scalar` redefined 4×, the `args`-bag marshalling, per-tick `cuda.synchronize`/`gpu_queue_wait` diag | God-function residue + hot-path sync; diag moves opt-in OFF the hot path. | pipeline_base.md §4 (`_scalar` ×4, args-bag); ARCH #4 (no work on hot path) |
| The dynamic `DynamicGSDataManager` / `get_current_dynamic_*_batch` / eval-split plumbing | Dynamic phase is a tracker runtime (Inv #4), not training — no eval, no datamanager wrapper for the dynamic stream; replay feeds the same SHM ingest. | adapters_source.md DROPPED; RUNTIME_target_architecture |

Explicitly NOT dropped (load-bearing): the **single-in-flight FF slot** + skip-if-behind; the **`is_first` FF gate** (D0 safety, H2); the **decide-FF-once-per-tick** flag + the `+1` ordering; the **one shared `_model_lock`** discipline; `set_phase("dynamic")` keeping `means.requires_grad=True`; `_step_offset=10_000`; the keep-viser-alive main-thread block at end-of-run.

---

## (7) INVARIANTS PRESERVED (CLAUDE.md) + how

- **#4 (dynamic phase = pure runtime, ALL gauss LRs = 0):** `step()` returns a zero-loss dummy; the loop performs NO backward/optimizer step. The only scene mutations are the tracker's rigid pose write (`gaussians.write_object_pose`, in-place, `@no_grad`, under lock) and FF cull+insert (in the FF dispatcher, under lock). The scheduler never touches LRs or runs a training step. (RUNTIME_ff_dispatch Inv#4; RUNTIME_tracker_tick.)
- **#8 (identity buffers, per-phase ownership):** the loop writes ONLY `object_flags` (D0 selection, via `gaussians.set_object_flags`); `object_instance_ids`/`inserted_flags`/the id=999 tail are written by the FF insert (in the dispatcher); `sam3d_init_target_flags` is never written. The rigid transform moves only `object_instance_ids == d0_id`, so FF inserts (id=999) are never tracked. All writes funnel through `GaussianSet` (the lockstep chokepoint), so identity can't desync from geometry. (RUNTIME_tracker_tick Inv#8; gaussian_set.md.)
- **#9 (viser-direct ONLY, single model lock):** the loop creates the ONE `_model_lock` RLock and shares the SAME instance into `GaussianSet`, `model.attach_render_lock`, and `ViserBridge` at construction (fixes the H-viser-swap ordering — never starts the render thread with the wrong lock). The viser bridge is read-only (`snapshot()` / locked render); the loop never enables the NS viewer (`vis="tensorboard"`); no `get_outputs_for_camera` for viz. (RUNTIME_target_architecture §The one lock; RUNTIME_tracker_tick Inv#9.)
- **#6 (background = Gazebo sky):** unchanged — the loop never recolors; background is the model's fixed render bg.
- **#1/#2 (static means LR=0, camera-opt off):** not this module's concern (static phase is `static_fit.py`); the loop only runs the dynamic phase. By never running a static step it cannot violate them.
- **ARCH #1/#2 (concurrency by snapshot+one-chokepoint):** all cross-thread reads go through `gaussians.snapshot()`; the H1 (`_object_crop_bbox`) and H2-pick unlocked reads are closed by reading the snapshot, not `model.means` directly.
- **ARCH #3 (bounded work + load-shed):** the **single-in-flight slot = skip-if-behind** (drop an FF firing rather than queue); the **FF growth cap** is honored by having the loop's purge policy call `gaussians.cull(low_opacity_indices(...), protect_mask=tracked)` periodically (mechanism in `GaussianSet`, policy here) — never dropping `object_flags==1`.
- **ARCH #4 (latency tail, no hot-path work):** the per-tick `cuda.synchronize`/file-I/O/debug-dump are dropped from the hot path; the timing ledger emits p90/p99; debug dumps are opt-in and off-thread.
- **ARCH #5 (event-time, not now()):** the FF min-gap is the one wall-clock use kept (it gates *dispatch cadence*, not any dt/velocity math) — and any dt-dependent math (KF, if re-enabled) reads `TickSnapshot.stamp_sec`. No `time.time()` delta feeds pose/velocity. (CLAUDE.md KF wall-clock note.)
- **ARCH #6/#7 (guaranteed release + fault isolation):** `teardown` is try/finally + idempotent; every thread/handle (source SHM, FF slot+worker, viser server) has a release path including on exception; the FF slot is released even if `Thread.start()` raises (closes H7/H3-slot); a dead FF thread degrades (skip FF) rather than hangs.
- **ARCH #8 (versioned contracts):** the warm-cache load is delegated to the persistence module which validates the config fingerprint; the loop fails loudly on a missing `_step_offset` rather than warn-and-continue (pipeline_base.md §3 flag).

---

## (8) THREADING (which threads, what they may/may-not block on, lock discipline)

**Three threads + 2 subprocesses (the rewrite's thread model lives HERE):**

- **MAIN / TRACKER thread** (the nerfstudio `Trainer.train` loop calls `step()`):
  - Runs the ENTIRE `_tracker_tick`: peek the SHM ring (lock-free seqlock; MUST NOT block — 5-retry → None → skip a frame), build batch + Cameras, render the object mask **UNDER `_model_lock`**, run `tracker.track` (no model touch), write the rigid pose via `gaussians.write_object_pose` **UNDER `_model_lock`** (in-place, no count change), build the frozen `TickSnapshot`, decide `_ff_due` once, dispatch FF (non-blocking), kick viser.
  - SOLE writer of: the rigid transform, `object_flags` (D0), the per-tick snapshot, the FF-due flag.
  - May block ONLY on: `_model_lock` (briefly — the object-mask render + the pose write). MUST NOT block on: the FF slot, the AnySplat subprocess, SHM-write, disk, or the publisher pose/joint lock.

- **FF-BG thread** (single-in-flight, daemon, owned by the FF dispatcher; the scheduler only *starts* it via `ff.dispatch`):
  - Gated by the FF slot (`acquire(blocking=False)`); if held, the scheduler skips (skip-if-behind). The scheduler's `_dispatch_ff_if_due` is the SOLE place the bg thread is spawned. The slot must be released on `Thread.start()` failure (closes H7).
  - Reads its inputs ONLY from the `TickSnapshot` it was handed (never re-reads live `self.*` — fixes H4/H5). All gauss mutation (cull+insert) goes through `GaussianSet` under `_model_lock`; the dispatcher must NOT hold the lock across the ~270 ms AnySplat IPC.

- **VISER-RENDER thread** (daemon, in `ViserBridge`):
  - Read-only: `snapshot()` or `model.get_outputs` UNDER the SAME `_model_lock`. Never mutates. Started by `__init__` with the shared lock already swapped in (no wrong-lock window).
  - Plus a tiny **stdin/stop daemon** that only flips `request_stop()` / a reselect bool (no model touch, no lock).

- **Subprocesses:** the ROS publisher (owns SHM, inside the source adapter) and the AnySplat worker (inside the FF dispatcher) — the scheduler does not own these directly; it owns their *lifecycle hooks* via `source.close()` / `ff.close()`.

**Lock discipline (the load-bearing rule):** there is exactly ONE `_model_lock` (re-entrant RLock), created here and shared into `GaussianSet` + `model.attach_render_lock` + `ViserBridge` + the FF dispatcher. EVERY read or write of `gauss_params` / identity buffers / `model.info` / a render happens under it OR reads an immutable `snapshot()`. The publisher pose/joint history lock is SEPARATE and lives in the source adapter — the scheduler never overloads `_model_lock` with it. The FF slot lock is distinct from `_model_lock`.

---

## (9) OPEN QUESTIONS for the human

1. **Does the loop drive frames, or does the nerfstudio Trainer?** Today `Trainer.train` calls `get_train_loss_dict` per step (the loop is *pulled*). The `DynamicLoop.run()` self-driven loop is the cleaner model for replay/standalone. Confirm we keep the trainer-pulled `step()` as the live entry (with `run()` only for headless replay), or fully replace the trainer with our own loop (drops a lot of nerfstudio glue but loses `--vis`/callbacks).
2. **Where does the framework glue (`NoSaveTrainer`, the `method_configs` entry points) live** — a thin `framework_glue.py` that forwards `get_train_loss_dict → DynamicLoop.step` and `train_complete → teardown`, or folded into this module? Keeping it separate keeps `pipeline_scheduler.py` framework-agnostic.
3. **FF dispatcher + viser bridge sibling specs are not yet written.** This spec assumes their surface: `ff.dispatch(snapshot) -> bool`, `ff.slot_busy`, `ff.drain()`, `ff.close()`; `viser.request_render()`, `viser.push_camera_feed(...)`, `viser.keep_alive_until_shutdown()`, `viser.stop()`. Confirm those signatures so the seam is exact.
4. **FF growth-cap policy home.** The mechanism (`cull(low_opacity_indices(...), protect_mask=tracked)`) is in `GaussianSet`; should the *when/how-much* periodic purge policy live in the scheduler (every N FF dispatches) or inside the FF dispatcher (right after each insert)? Spec assumes the scheduler owns the cadence; confirm.
5. **Object-mask render ownership.** The tracker needs the object footprint rendered under `_model_lock`. Does the scheduler render it (it holds the lock and builds `TrackerInputs`) — the spec's assumption — or does `scene_model`/`GaussianSet` expose a `render_object_mask` the scheduler just calls under lock? Either keeps the tracker lock-free; pick one to avoid a second render-mask site.
6. **Interactive object picker / mid-run reselect** — keep in v1? It adds a blocking-until-Done path in the tick (the trainer step-counter race the current code guards against). If kept, it must read `gaussians.snapshot()` (not unlocked `model.means`, closing H2-pick). Confirm in-scope or defer.
7. **Replay determinism vs skip-if-behind.** If replay goes through the same SHM ingest (adapters OQ2) AND the FF uses skip-if-behind, a recorded run is no longer frame-exact reproducible (FF fires depend on wall-clock gap). Confirm replay either (a) accepts non-determinism, or (b) the loop uses a *tick-count-only* FF gate (no wall-clock min-gap) in replay mode for reproducibility.
