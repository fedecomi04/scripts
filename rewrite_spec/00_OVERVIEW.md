# 00 — BLUEPRINT OVERVIEW

Assembled from the 13 per-module rewrite specs in this directory. This file is the
coherence contract for the dynamic-gs rewrite: a short god-file `pipeline.py`, a single
`Frame` contract over producer-owned SHM, one locked `GaussianSet` SSOT read via
immutable snapshots, three threads (tracker on main / FF single-in-flight bg / viser
daemon), bounded FF growth, and Recorded+Live collapsed into one `DynamicLoop` + source
adapters.

---

## 1. MODULE MAP

```
scripts/
├── pipeline.py                  ORCHESTRATOR. wire source→static.build→dynamic loop; owns the ONE _model_lock + 3-thread scheduling. (spec: pipeline_scheduler.md)
│
├── config.py                    CORE. single frozen RuntimeConfig (every DGS_* knob) + 4 nerfstudio MethodSpecifications + config_fingerprint. (config.md)
│
├── contract/  (leaf, import-light, py3.8-safe — numpy+struct only)
│   ├── frame.py                 Frame/Intrinsics frozen dataclasses + versioned SHM header/slot codec + seqlock (de)serialize. (frame.md)
│   └── shm_channel.py           ShmProducer (create/write) + ShmConsumer (attach/read-latest) over the frame.py layout; lock-free x86 seqlock. (shm_channel.md)
│
├── adapters_source.py           ADAPTER. FrameSource Protocol + open_source() factory: ReplaySource (disk→SHM) + ros1/ros2 producers + ShmRing reader + camera_from_frame. (adapters_source.md)
│
├── gaussian_set.py              CORE/SSOT. 6 gauss_params + 4 identity buffers behind ONE locked surgery API (cull/insert/write-pose/reload) + immutable snapshot() read path. (gaussian_set.md)
├── scene_model.py               CORE. the splat model (gauss_params host, optimizers, means-grad hook, phase-apply, render). *** NO SPEC YET — user-supplied / TBD. ***
│
├── static_segment.py            STATIC. Phase-0a: ref frame → FastSAM/SAM3 masks → Fast-SAM3D per-object clouds via persistent SAM worker. (static_segment.md)
├── static_fuse.py               STATIC. TSDF seed PLY (GPU/CPU online-fusion, subprocess-isolated) + Phase-0b register/cull → insert-ready GaussTensors. (static_fuse.md)
├── static_fit.py                STATIC. Splatfacto static fit policy: depth/scene loss-mask, scale-reset, EMA early-stop, opacity purge. (static_fit.md)
├── static_persist.py            STATIC. warm-cache .pt save/load (config-fingerprint + schema versioned) + seed/init path constants. (static_persist.md)
│
├── dynamic_track.py             DYNAMIC. XFeat→LighterGlue→RANSAC/Kabsch multi-anchor tracker → MotionEstimate (main thread, read-only). (dynamic_track.md)
├── dynamic_feedforward.py       DYNAMIC. FF bg worker: CDN→AnySplat decode→reproject→in-front cull→bounded insert; load-shed ceiling. (dynamic_feedforward.md)
├── dynamic_viz.py               DYNAMIC. ViserBridge: viser-direct server-side rasterize + push-image (Invariant #9), camera-feed/follow. (dynamic_viz.md)
│
└── (referenced-but-unspecced helpers — see Coherence Check §6.B)
    ├── timing.py                timing ledger (record/render p90/p99). dep of static_fit, pipeline.
    ├── sam_worker.py            SamWorkerClient (spawn-once SAM3/FastSAM/SAM3D, _request mutex + close fix). dep of static_segment.
    ├── anysplat_client.py       AnySplat worker IPC wrapper (spawn/adopt/inference). dep of dynamic_feedforward.
    ├── camera_conventions.py    gl↔cv c2w, deproject_to_world, project_world_to_pixel (DUP A/B/C target). dep of static_fuse.
    ├── rotations.py             quat_wxyz→rotmat (DUP D target). dep of static_fuse.
    └── depth_ops.py             mm↔m + pytorch3d pointmap back-projection. dep of static_segment.
```

---

## 2. DEPENDENCY DAG

Read top→bottom = depends-on. Leaf contracts at the bottom, `pipeline.py` at the top.
**No cycle exists** (verified in §6.A).

```
                              pipeline.py  (orchestrator)
        ┌──────────┬──────────┬───────┬──────────┬──────────┬──────────┬───────────┬──────────┐
        v          v          v       v          v          v          v           v          v
 adapters_source  static_*  dynamic_track  dynamic_feedforward  dynamic_viz  static_persist  timing  config
        │          (segment, fuse, fit)        │                    │            │              │       │
        │            │   │   │                  │                    │            │              │       │
        v            v   v   v                  v                    v            v              │       │
   frame.py     scene_model.py            scene_model.py        scene_model.py  gaussian_set.py  │       │
   shm_channel  gaussian_set.py           gaussian_set.py       gaussian_set.py     │            │       │
   (frame)      config / frame /          config / frame /      config / frame /    v            │       │
                timing / sam_worker /      anysplat_client       (read snapshot)  scene_model.py  │       │
                camera_conv / rotations /                                                          │       │
                depth_ops / ndp                                                                    │       │
        │            │                          │                    │                            │       │
        └────────────┴──────────────┬───────────┴────────────────────┴────────────────────────────┘       │
                                     v                                                                      │
                              gaussian_set.py ───────────────────────────────────────────────────► config.py
                                     │                                                                      │
                                     v                                                                      │
                              scene_model.py ──────────────────────────────────────────────────────► config.py
                                                                                                            │
   LEAF CONTRACTS (zero deps):  frame.py     shm_channel.py     config.py    rotations.py   camera_conventions.py
```

Key edges (from each spec's `depends_on`):
- **frame.py** → ∅ (leaf). **shm_channel.py** → ∅ (its spec declares no deps; in practice it imports frame.py's layout — see §6.A note).
- **config.py** → ∅ (leaf; reads os.environ only).
- **adapters_source.py** → frame_contract (CONDITIONAL) — owns Frame/Intrinsics itself OR imports them.
- **gaussian_set.py** → scene_model.py + the shared `_model_lock` instance.
- **scene_model.py** → config.py (assumed; no spec).
- **static_segment** → frame, config, sam_worker, depth_ops.
- **static_fuse** → config, frame, gaussian_set, camera_conventions, rotations, ndp.
- **static_fit** → gaussian_set, config, scene_model, timing.
- **static_persist** → gaussian_set, config.
- **dynamic_track** → frame, gaussian_set, config.
- **dynamic_feedforward** → gaussian_set, config, frame, scene_model, anysplat_client, `_model_lock`.
- **dynamic_viz** → config, gaussian_set, frame.
- **pipeline.py** → adapters_source, frame, gaussian_set, scene_model, dynamic_track, dynamic_feedforward, dynamic_viz, config, timing, static_persist (+ static_* for the static phase).

---

## 3. pipeline.py SKELETON

Short orchestrator. Cites the real spec interfaces. ~150 lines target.

```python
# pipeline.py
from config import load_runtime_config, config_fingerprint
from adapters_source import open_source, camera_from_frame   # FrameSource + ShmRing
from gaussian_set import GaussianSet, GaussianSnapshot
from scene_model import build_scene_model                    # (unspecced) splat model
from dynamic_track import XFeatTracker, TrackerInputs, MotionEstimate
from dynamic_feedforward import FeedforwardWorker, FeedforwardConfig, FeedforwardDispatch
from dynamic_viz import ViserBridge
import static_segment, static_fuse, static_fit, static_persist, timing
import threading


# ---------------------------------------------------------------- STATIC PHASE
def build_static(data_dir, cfg, device) -> GaussianSet:
    lock  = threading.RLock()                                  # the ONE _model_lock
    model = build_scene_model(cfg, seed_ply=static_persist.seed_ply_path(data_dir))
    gset  = GaussianSet(model, lock)

    fit = static_fit.StaticFit(cfg.static_train, gset, timing)
    fit.install_loss_mask(model)
    # ... run the Splatfacto trainer with fit.training_callbacks + fit.on_train_iteration early-stop ...
    fit.finalize_opacity_purge()                               # gset.cull(low_opacity)

    # Phase 0a (segmentation) — main thread, BEFORE any worker thread
    ref = static_fuse.load_reference_frame(data_dir, datamanager, model, device)
    with static_segment.StaticSegmenter(cfg.segmentation) as seg:
        seg_result = seg.segment(seg_ref, out_dir=...)         # masks + SAM3D clouds

    # Phase 0b (register/cull) — returns insert-ready tensors, never mutates gset itself
    fused = static_fuse.register_objects(scene=gset.snapshot(), ref=ref,
                                         generation_artifacts=seg_result.clouds, ...)
    for obj in fused:                                          # surgery via the SSOT chokepoint
        idx = gset.insert(obj.tensors, object_flag=0.0, instance_id=obj.instance_id)

    static_persist.save_warm_cache(gset, config_fingerprint(cfg), data_dir=data_dir)
    return gset


# --------------------------------------------------------------- DYNAMIC LOOP
class DynamicLoop:
    def __init__(self, source, gaussians, model, tracker, ff, viser, cfg):
        self.source, self.g, self.model = source, gaussians, model
        self.tracker, self.ff, self.viser = tracker, ff, viser
        self.cfg, self.lock = cfg, gaussians._lock          # SAME RLock instance everywhere
        self._tick = 0
        self._stop = threading.Event()

    def _ff_due(self, tick, *, is_first, stamp_sec) -> bool:   # PURE — decide once/tick
        return (not is_first) and (tick % self.cfg.feedforward.cadence_ticks == 0) \
               and (not self.ff.in_flight()) and self.ff.due(tick, stamp_sec)

    def _tracker_tick(self, frame) -> MotionEstimate | None:
        cam   = camera_from_frame(frame, self.source.intrinsics(), self.model.device)
        snap  = self.g.snapshot()                              # immutable read for crop bbox
        with self.lock:
            obj_mask = self.model.render_object_mask(cam, snap, d0_id=self.d0_id)  # under lock
        inp = TrackerInputs(rgb=..., depth=frame.depth_m, camera=cam,
                            keep_mask=frame.mask_keep, object_mask=obj_mask,
                            stamp_sec=frame.stamp_sec)
        est = self.tracker.track(inp)                          # NO model touch, no lock
        if est.success:
            with self.lock:
                self.g.write_object_pose(est_means, est_quats, object_mask=(snap.buffers.object_instance_ids == self.d0_id))
        return est

    def step(self, step_idx) -> ZeroLoss:                      # Invariant #4 — zero-loss dummy
        frame = self.source.peek_latest()                     # lock-free seqlock, None→skip
        if frame is None:
            return ZERO_LOSS
        self._tick += 1
        est = self._tracker_tick(frame)
        is_first = (self._tick == 1)                           # D0 gate
        if self._ff_due(self._tick, is_first=is_first, stamp_sec=frame.stamp_sec):
            disp = FeedforwardDispatch(seq=frame.seq, camera=cam_copy,
                                       rgb_bgr=frame.rgb_bgr.copy(),    # frozen snapshot, fixes staleness
                                       depth_m=frame.depth_m, object_mask=obj_mask,
                                       gripper_keep=frame.mask_keep,
                                       scene_intr=self.source.intrinsics(), d0_instance_id=self.d0_id)
            self.ff.dispatch(disp)                             # non-blocking, skip-if-behind
        self.viser.request_render()                            # coalesced read-only kick
        self.viser.update_camera_feed(frame.rgb_bgr)
        return ZERO_LOSS

    def teardown(self):                                       # idempotent try/finally
        self._stop.set()
        self.ff.close(); self.viser.close(); self.source.close()
        static_persist.save_warm_cache(self.g, ..., data_dir=...)   # post_dynamic snapshot
        timing.write_report(...)


# --------------------------------------------------------------------- LAUNCH
def run_dynamic(spec):  # RunSpec(mode, data_dir, runtime_cfg, device)
    cfg = spec.runtime_cfg
    lock = threading.RLock()
    model = build_scene_model(cfg)
    gset  = GaussianSet(model, lock)
    static_persist.load_warm_cache(gset, config_fingerprint(cfg), data_dir=spec.data_dir)  # loud-on-drift

    source = open_source(spec.mode, spec.data_dir, shm_name=..., )      # replay|ros1|ros2
    tracker = XFeatTracker(spec.device, cfg.tracker, cfg.pose_filter)
    viser   = ViserBridge(cfg.viser)
    viser.attach(render_fn=lambda cam: locked_render(gset, model, lock, cam),  # closure carries lock
                 snapshot_fn=gset.snapshot)
    ff = FeedforwardWorker(gset, lock, cfg.feedforward, anysplat, renderer=model.render,
                           on_insert=viser.request_render)

    loop = DynamicLoop(source, gset, model, tracker, ff, viser, cfg)
    # nerfstudio NoSaveTrainer pulls loop.step(step_idx); teardown in try/finally
```

---

## 4. THREAD MODEL

One re-entrant `_model_lock` (RLock), created in `pipeline.run_dynamic`/`build_static` and
passed by reference into `GaussianSet`, `model.attach_render_lock`, `ViserBridge` (via the
render closure), and `FeedforwardWorker`. The publisher pose/joint lock and the FF slot
lock are **separate** locks, never the model lock.

| Thread | Owner | Runs | Lock discipline | Snapshots |
|---|---|---|---|---|
| **MAIN / tracker** | pipeline `DynamicLoop.step` | peek SHM (lock-free seqlock, 5-retry→None→skip), build batch+Cameras, render object-mask **under lock**, `tracker.track` (no model touch), `write_object_pose` **under lock**, freeze `TickSnapshot`, decide `_ff_due` once, `ff.dispatch` non-blocking, `viser.request_render`. | acquires `_model_lock` only for the ms-scale mask-render + pose-write windows; MUST NOT block on FF slot / AnySplat / SHM-write / disk / publisher lock. | `gaussians.snapshot()` for the crop bbox + tracked-object mask. |
| **FF background** (single-in-flight daemon) | `FeedforwardWorker._run` | reads ONLY its frozen `FeedforwardDispatch`; CDN render (under lock) → AnySplat decode/reproject/ICP (lock-free, ~400ms) → in-front cull (under lock) → `enforce_ceiling` → `gaussians.insert` (under lock) → maybe purge → `on_insert`. | `_model_lock` for exactly 3 short windows (snapshot-target, cull, insert); NEVER across the AnySplat decode. Slot is a non-blocking try-acquire; released on `Thread.start()` failure. | pulls a `GaussianSnapshot` internally for the ICP target + live count. |
| **VISER render** (daemon) | `ViserBridge` render loop | event-driven on `request_render` Event; per client: `render_fn(camera)` → JPEG → `set_background_image`; apply follow-pose/feed thumbnail. | holds NO scene lock of its own; the only critical section is inside `render_fn` (pipeline's locked render). Read-only. | `snapshot_fn()` for version-skip; render reads live scene under `render_fn`'s lock. |
| **(aux) stdin/stop** (tiny daemon) | pipeline | flips `request_stop()` / reselect bool. | no lock. | none. |
| **(subprocess) ROS publisher** | adapters_source ros source | rospy receive → 1 worker decodes+masks+`ShmProducer.write` (seqlock store). | a NEW dedicated pose/joint history lock (writers + `_interpolate_c2w` read window); separate from `_model_lock`. | none (cross-process via mmap seqlock). |
| **(subprocess) AnySplat / SAM worker** | anysplat_client / sam_worker | model inference. | none shared. | none. |

**Invariant:** every `gauss_params` / identity-buffer / `model.info` / render access is either
under `_model_lock` OR via an immutable `snapshot()`. No thread reads live mutable scene
state another thread mutates without one of those two.

---

## 5. BUILD ORDER

Write leaves first so every consumer compiles against a real interface.

1. **Leaf contracts (no deps):** `config.py`, `frame.py`, then `shm_channel.py` (on frame's layout). Plus pure helpers `rotations.py`, `camera_conventions.py`, `depth_ops.py`, `timing.py`.
2. **Core SSOT:** `scene_model.py` *(USER-SUPPLIED STUB — no spec; must expose gauss_params, optimizers/_optimizers_wrapper, `_mask_means_grad`, phase-apply callbacks, `render(camera)`, `render_object_mask`, device/dtype, `attach_render_lock`)*, then `gaussian_set.py` (binds model + lock).
3. **Adapters + worker clients:** `adapters_source.py`, `sam_worker.py` *(client; SAM worker subprocess body unchanged)*, `anysplat_client.py` *(client; AnySplat subprocess body unchanged)*.
4. **Static phase:** `static_segment.py`, `static_fuse.py`, `static_fit.py`, `static_persist.py`.
5. **Dynamic phase:** `dynamic_track.py`, `dynamic_feedforward.py`, `dynamic_viz.py`.
6. **Orchestrator:** `pipeline.py` + thin framework glue (NoSaveTrainer, the 4 MethodSpecification `_target` repoints).

**User-supplied / external (not generated by this rewrite, only wrapped):** `scene_model.py`
internals (TBD — has no spec), the SAM/SAM3D/FastSAM worker subprocess (`sam_worker` body),
the AnySplat worker subprocess, the vendored `ndp/`, the ROS publisher subprocess body
(env-strip + rospy), nerfstudio itself.

---

## 6. ADVERSARIAL COHERENCE CHECK

### A. Circular dependencies — NONE, with one boundary to nail down.
- The DAG is acyclic: `pipeline → {dynamic_*, static_*, adapters} → {gaussian_set, frame, config} → {scene_model → config}`. `frame`/`config` are true leaves.
- **POTENTIAL CYCLE / OWNERSHIP COLLISION (must resolve before coding):** three modules each claim to *define* `Frame`/`Intrinsics`/SHM-codec — `frame.md`, `shm_channel.md` (as `ShmFrame`/`ShmLayout`), and `adapters_source.md` (which says it owns them "CONDITIONAL — only if not hoisted"). If `shm_channel` imports `frame` AND `frame` declares the SHM layout that `shm_channel` re-declares, you get duplicate-drift (the exact bug both specs are trying to kill). **DECISION REQUIRED (D1):** `frame.py` is the single source of `Frame`, `Intrinsics`, `LAYOUT_VERSION`, `SHM_MAGIC`, header struct, `compute_layout`, `pack/read_header`, `build_slot_views`, `write_frame`, `read_latest`; `shm_channel.py` becomes the thin `ShmProducer`/`ShmConsumer` lifecycle wrapper that *imports* all layout/codec from `frame.py` (NOT a second `ShmFrame`/`ShmLayout`); `adapters_source.py` imports both, never redefines. Note `shm_channel.md` lists `depends_on: []` — that is WRONG once it imports frame; mark it `depends_on: [frame]`.

### B. Missing / referenced-but-unspecced modules.
- **`scene_model.py`** is a `depends_on` of `gaussian_set`, `static_fit`, `dynamic_feedforward` and is implied by `pipeline`. **It has NO spec.** It is load-bearing: GaussianSet "wraps and calls back into it" for the optimizer-refresh tail, `_mask_means_grad`, phase optimizers, `render`, `render_object_mask`, device/dtype, `attach_render_lock`. **DECISION REQUIRED (D2):** write a `scene_model.md` spec (or explicitly declare it user-supplied) pinning that interface — every dynamic/static module is contracted against it.
- **`timing.py`, `sam_worker.py`, `anysplat_client.py`, `camera_conventions.py`, `rotations.py`, `depth_ops.py`** are all cited as deps with no spec. They are mostly "kept verbatim / extracted dedup," lower-risk, but `camera_conventions`/`rotations` are the DUP-A/B/C/D consolidation targets and `anysplat_client` absorbs the spawn/adopt lifecycle the FF spec explicitly delegates — confirm these exist as real extraction targets.

### C. Producer↔consumer contract mismatches.
- **Module-name drift (orchestrator vs module list):** `pipeline_scheduler.md` `depends_on` says `feedforward_dispatcher.py` and `viser_bridge.py`; the actual specs are `dynamic_feedforward.md` (class `FeedforwardWorker`) and `dynamic_viz.md` (class `ViserBridge`). **Names must reconcile** before imports are written: file `dynamic_feedforward.py`/class `FeedforwardWorker`, file `dynamic_viz.py`/class `ViserBridge`. The orchestrator's assumed methods (`ff.dispatch(snapshot)->bool`, `ff.slot_busy`, `ff.drain()`, `ff.close()`) vs the FF spec's surface (`dispatch`, `in_flight()`, `close()` — no `drain`/`slot_busy`) **DO NOT MATCH**: FF exposes `in_flight()` not `slot_busy`, and has no `drain()` — `close()` does join. **DECISION REQUIRED (D3):** pick one surface; recommend FF spec's (`in_flight()` + `close()` joins).
- **`FeedforwardDispatch` field names:** orchestrator builds `FeedforwardDispatch(seq=, camera=, rgb_bgr=, depth_m=, object_mask=, gripper_keep=, scene_intr=, d0_instance_id=)`; FF spec declares `{seq, camera, rgb_bgr, depth_m, object_mask, gripper_keep, scene_intr, d0_instance_id}`. ✓ MATCH. The orchestrator's `TickSnapshot(camera, batch, rgb_bgr, stamp_sec, frame_seq)` is a *different* type than `FeedforwardDispatch` — confirm the orchestrator converts TickSnapshot→FeedforwardDispatch (it must; FF reads only the latter). Minor, but spell it out.
- **GaussTensors as the universal insert contract:** `gaussian_set.GaussTensors` is consumed by `static_fuse.register_objects` (returns `FusedObject.tensors: GaussTensors`) and produced by `dynamic_feedforward.reproject_to_scene`. ✓ Single contract, good — but its `validate(sh_rest_dim)` couples to `sh_degree`/features_rest width sourced from `config`. Confirm static and dynamic build tensors with the SAME `sh_rest_dim`, or an FF insert with the wrong width silently corrupts (flagged in FF open-Q).
- **`snapshot()` shape:** tracker, FF, viz, static_fuse, static_persist all consume `GaussianSnapshot`. `static_persist.save_warm_cache` reads `gset.snapshot()` while an FF insert could be in flight — snapshot is atomic (locks briefly to bundle detached refs), so this is safe ✓. But `static_fuse.register_objects(scene=gset.snapshot())` runs in the static phase where no FF thread exists — also safe.
- **`render_object_mask` site:** orchestrator open-Q asks who renders the object mask; the tracker spec *requires* the pipeline render it under lock and pass it in `TrackerInputs.object_mask` (tracker takes NO lock). The skeleton above does this. But `FeedforwardDispatch.object_mask` must be the SAME mask the tracker used (same tick) — confirm one render, reused for both tracker input and FF dispatch (skeleton reuses `obj_mask`). ✓ if wired as drawn.

### D. DROPPED logic that may actually be needed.
- **`self.info` (model.info) cross-thread reads** are dropped by `gaussian_set`, `dynamic_feedforward`, and `pipeline` (the rgbd-cull path). FF cull now uses "direct projection." **Verify** the AnySplat in-front cull truly never needs projected radii from `model.info` — both specs assert it, but `cull_points_in_front` is the replacement; make sure its `render_hw`/projection path is self-contained. (Low risk — the spec says AnySplat default already proves it.)
- **Entire `rgbd_decode.py` path dropped** as "unreachable on default live config." Correct given AnySplat is the single FF backend — but this removes Mode-A/B RGB-D inpaint entirely. Confirm no flow (recorded debug?) still selects it. The specs are consistent (config drops `enable_feedforward_inpaint` rgbd branch).
- **`DynamicGSDataManager` + dynamic eval split dropped** across adapters/pipeline. Correct under Invariant #4 (dynamic = runtime, no eval). But the STATIC phase still needs a datamanager for `load_reference_frame(static_dir, datamanager, model, device)` (static_fuse) — confirm the static datamanager survives (it's the nerfstudio `FullImageDatamanager` for `static_scene/`, untouched). ✓ only the *dynamic*-stream datamanager is dropped.
- **`current_active_mask` (5th buffer) dropped** — all its writers are dead. Confirmed across gaussian_set + dynamic_model audits. Safe.
- **CPD/TEASER registration dropped to "optional-import"** in static_fuse — NDP is default. Keep selectable per the config (`sam3d_registration_backend`), just behind optional import so NDP-only is lean. No invariant impact.
- **Oneshot-FF path dropped** everywhere (config, FF, pipeline). The is_first/D0 gate stays in pipeline (`_ff_due(is_first=...)`). Verify D0 bootstrap still fires the initial scene-fill it needs — D0 is the tracker SEED (`tracker.seed`), not an FF call; the first FF happens on the first `cadence_ticks` boundary with `is_first=False`. Acceptable; just confirm no design relied on a one-shot full-scene FF at D0.

### E. Threads reading state another mutates without snapshot/lock — NONE found, given:
- Tracker reads object-mask = a tensor handed in (rendered under lock by main). ✓
- FF reads ONLY its frozen `FeedforwardDispatch` (fixes the H4/H5 "bg thread reads live `self.*`" race) + an internal `snapshot()`. ✓
- Viser reads via `snapshot_fn()` + locked `render_fn`. ✓
- `static_persist.save_warm_cache` ↔ in-flight FF insert: atomic `snapshot()`. ✓
- **WATCH:** the skeleton renders `obj_mask` under lock in `_tracker_tick`, then the SAME `obj_mask` is referenced when building `FeedforwardDispatch` later in `step()`. Since both run on the main thread within one `step()`, no race — but the variable must be captured in `step()` scope, not re-rendered. Wire it so the mask is rendered once.
- **Publisher pose/joint history race (H3)** is fixed by a NEW dedicated lock in the ros source (adapters_source), separate from `_model_lock`. ✓ Cross-process SHM is seqlock-only, no in-process lock spans it. ✓

### F. Consolidated HUMAN-DECISION LIST (from every spec's open questions + the checks above).

**Blocking (resolve before writing module interfaces):**
1. **D1 — Frame/SHM ownership:** `frame.py` is the sole owner of Frame/Intrinsics/layout/codec; `shm_channel.py` imports it (fix its `depends_on: []`); `adapters_source.py` imports, never redefines. (frame.md/shm_channel.md/adapters_source.md open-Qs all ask this.)
2. **D2 — `scene_model.py` has no spec.** Write one (or declare user-supplied) pinning: gauss_params host, optimizers/`_optimizers_wrapper`, `_mask_means_grad`, phase-apply callbacks, `render(camera)→rgb,depth,alpha`, `render_object_mask`, device/dtype, `attach_render_lock`, config.sh_degree. Every dynamic/static module contracts against it.
3. **D3 — FF + Viser surface names:** reconcile orchestrator's `feedforward_dispatcher.py`/`slot_busy`/`drain()` vs FF spec's `dynamic_feedforward.py`/`FeedforwardWorker`/`in_flight()`/`close()`. Recommend FF spec wins.
4. **GaussianSet identity:** BE the splat model or WRAP nerfstudio SplatfactoModel? (gaussian_set open-Q; spec assumes wrap-and-callback — confirms D2 boundary.)
5. **Config reload vs relaunch:** documented no-relaunch A/B env knobs (DGS_FF_ICP, DGS_HOLD_*, DGS_FF_MAX_SCALE_M) vs frozen read-once config → add `reload_overrides()` atomic-swap or accept relaunch-only? (config open-Q.)

**Architectural (resolve before dynamic/static phase):**
6. **ReplaySource through-SHM vs in-process:** one identical ingest path (preferred) costs a copy/frame and—combined with FF skip-if-behind—makes replay non-frame-exact. Accept non-determinism, or replay must NOT drop frames + FF gates on tick-count-only (no wall-clock min-gap)? (adapters_source + pipeline open-Qs.)
7. **Static-capture orchestration home:** the dropped control-pipe ops (anchor/record/SAM3/build-seed/pause-gazebo) move to a separate `capture_tool`; retarget `bootstrap_live.sh`/`capture_only.sh`. (adapters_source/shm_channel open-Qs.)
8. **Phase-0a/0b boundary:** SAM3/FastSAM/SAM3D generation lives in `static_segment` producing `Sam3dArtifact`; `static_fuse` consumes. Confirm `static_segment` stops at SegmentResult and `static_fuse.register_objects` returns `FusedObject` for the caller to insert (no interleaved per-object insert that re-renders for the next object's existing-subset query). (static_segment/static_fuse open-Qs — note the "post-insert scene required?" caveat.)
9. **FF growth-cap policy home:** scheduler owns purge cadence (every N dispatches → `gaussians.cull(low_opacity_indices, protect_mask=tracked)`) vs FF purges after each insert. Spec assumes scheduler; FF spec also has `enforce_ceiling`/`purge_ff_inserts`. Pick ONE owner. (pipeline + FF open-Qs.)
10. **`max_live_gaussians` value:** 1.5M placeholder (zed_final blew to 3M) — derive from a VRAM probe vs fixed count? (FF open-Q.)
11. **KF retention:** OFF by default, measured worse on jerky 1200p. Drop entirely (remove PoseFilterConfig from tracker ctor) or keep dormant behind a flag in a `_pose_filter.py` sibling? (dynamic_track open-Q.)
12. **ICP-refine default:** kept ON by eyeball, not a rigorous A/B; depth-scaled bg divergence is real. Ship `icp_refine=True` or default-off pending A/B? (FF + CLAUDE.md open-Q.)

**Trainer/framework:**
13. **Trainer ownership:** nerfstudio `NoSaveTrainer` pulls `step()` (live) vs `DynamicLoop.run()` self-drives (headless replay)? Where does framework glue (MethodSpecifications, `get_train_loss_dict→step`) live — a thin `framework_glue.py` or folded into pipeline? (pipeline + static_fit open-Qs.)
14. **Static trainer layer:** drop `StaticGSTrainer` subclass; pipeline owns the vanilla trainer + installs `StaticFit` callbacks + early-stop directly? (static_fit open-Q — spec assumes yes.)
15. **Loss-mask wiring:** patch model `get_loss_dict` vs pass a `mask_provider` callable into the model ctor? (static_fit open-Q.)

**Persistence/contract hygiene:**
16. **Warm-cache fingerprint scope:** what the config_fingerprint must cover (sh_degree/features_rest width, background, camera-opt mode, SH-rest dim) and whether a narrower tensor-shape-only fingerprint should coexist so benign tuning doesn't invalidate a valid cache; legacy `post_fusion_state.pt` has no fingerprint → treat as drift / accept-with-warning? (static_persist open-Qs.)
17. **SimNoise enabled-source:** `SimNoiseConfig.enabled` defaults True (corrupts sim depth); REAL ZED must be OFF — auto-derive from a `source=sim|real` field instead of `DGS_SIM_ZED_NOISE=0`? (config open-Q.)
18. **read_latest churn:** ~18MB owned-copy per tick at 1200p — accept for v1 or add caller-supplied scratch double-buffer (complicates copy-survives-next-write)? (frame/shm_channel open-Qs.)
19. **Multi-object surface:** keep `SegmentResult.masks/clouds` list now (roadmap #3) + picker downstream, or single-object v1? (static_segment + pipeline open-Qs.)

**Lower-risk / defer:**
20. ros1-only vs ros1+ros2 stub; camera-feed thumbnail gate-on-clients vs drop; render_size/jpeg_quality into ViserConfig; CPD/TEASER optional-import; ConcurrentSeedRunner carry vs subprocess-seed-only; `object_mask_scale=1.02` re-tune; tracker_common survivors inline vs `_tracker_geom.py` sibling.

---

## EXECUTIVE SUMMARY

The 13 module specs compose into a clean, acyclic blueprint: leaf contracts (`frame`,
`config`, `shm_channel`) at the bottom, the `GaussianSet` SSOT + `scene_model` in the core,
static and dynamic phases as thin policy layers, and a short `pipeline.py` god-file that
owns the one `_model_lock`, the 3-thread schedule (tracker-on-main, single-in-flight FF bg,
read-only viser daemon), and collapses Recorded/Live into one `DynamicLoop` fed by a
`FrameSource` adapter — with every cross-thread scene read going through an immutable
`snapshot()` or the lock, which is exactly what kills the H-CROP/self.info/staleness race
class. The blueprint is coherent and contains no dependency cycle, but three blocking
mismatches must be decided before any code is written: **(D1)** collapse the triplicate
`Frame`/`Intrinsics`/SHM-layout ownership into `frame.py`-only (and fix `shm_channel`'s
false `depends_on: []`); **(D2)** `scene_model.py` is a load-bearing dependency of five
modules yet has **no spec** — its interface (`render`, `render_object_mask`,
`_mask_means_grad`, optimizer-refresh callbacks, `attach_render_lock`) must be pinned
before GaussianSet/FF/tracker can compile against it; and **(D3)** the orchestrator's
assumed FF/viser surface (`feedforward_dispatcher`/`slot_busy`/`drain()`) does not match the
actual FF/viz specs (`dynamic_feedforward`/`FeedforwardWorker`/`in_flight()`/`close()`).
All other findings are consolidated into a 20-item human-decision list (§6.F), led by the
FF-growth-cap owner, replay determinism vs skip-if-behind, and the KF/ICP default calls.
