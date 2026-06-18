# 00 — DECISIONS LOG (the settled calls the code is generated against)

Running record of resolved design decisions for the dynamic-gs rewrite. Updated 2026-06-18.
The eventual code-gen reads THIS + the module specs.

## SETTLED
- **D1 — contract ownership:** `frame.py` is the sole owner of `Frame`, `Intrinsics`, the SHM layout + codec + `LAYOUT_VERSION`. `shm_channel.py` imports it (producer/consumer lifecycle only); adapters import, never redefine.
- **D2 / #4 — WRAP (not BE):** `GaussianSet` = pure, lockable state (6 tensors + 4 identity buffers + locked surgery + `snapshot()`). `scene_model.py::SceneModel` = thin adapter that renders/trains those tensors via a *wrapped* nerfstudio `SplatfactoModel` (same tensor objects, no copy; `rebind()` after surgery). Drops the vestigial NS-viewer integration (Inv #9). See `scene_model.md`.
- **D3 — FF/viz surface:** `dynamic_feedforward.py::FeedforwardWorker` with `dispatch()` + `in_flight()` + `close()`-joins (no `slot_busy`/`drain()`); `dynamic_viz.py::ViserBridge`. Orchestrator adopts these names.
- **#6 — replay:** `ReplaySource` always goes through SHM (one ingest path). **`paced` is the default** (wall-clock-paced off capture stamps → faithful live proxy; consumer drops/trails like live). **`fast`** = lock-step, frame-exact, deterministic (debugging only). Stamps from `transforms.json` or synthesized from `replay_fps`. See `adapters_source.md`.
- **#17 — sim vs real noise:** `config.source: Literal["sim","real"]`. `sim` auto-enables the ZED depth-noise model; `real` auto-disables it. Replaces the `DGS_SIM_ZED_NOISE` env footgun. (bake into `config.md`.)
- **#19 — multi-object:** contracts carry the multi-object surface — `SegmentResult.masks[]/clouds[]`, `object_instance_ids` 1..K, the viser picker, multi-id `gaussian_set`. v1 may exercise a single object, but the interfaces support N (roadmap #3 needs no contract change).
- **#11 — Kalman filter: KEPT, first-class.** Lives in the tracker, fed **event-time dt** (`Frame.stamp_sec` capture timestamps), NOT wall-clock `now()`-deltas — which is the exact fix for "great on recorded, bad live" (the live timing detune). Tune params later (a reload knob). NOT dropped, NOT a dormant sibling.
- **#5 — config reload (kept for tuning knobs):** a small `reload_overrides()` atomic-swaps a whitelist of live-tunable knobs (KF params, `ICP on/off`, FF scale caps) mid-run with no relaunch; everything else is frozen at boot. (Relaunch = restart the process + reload `.pt` + re-spawn workers; reload = next tick picks up the new value. Kept because live A/B tuning is part of the workflow.)

## OPEN (recommendation noted; decide when ready — none block writing the leaf/core modules)
- **#9 FF growth-cap owner** — REC: scheduler owns the purge *cadence*; `FeedforwardWorker.enforce_ceiling` does *enforcement* before insert. Never drop `object_flags==1`.
- **#10 `max_live_gaussians`** — REC: VRAM-probe-derived with a floor; start ~1.5M (zed_final blew to 3M).
- **#12 ICP-refine default** — REC: ship ON; A/B (`ICP on` vs `off`) early on a live run — it's the #1 thing to settle (memory: ICP-off measured *slower*).
- **#13/#14/#15 framework glue** — REC: thin `framework_glue.py`; `DynamicLoop` self-drives, a nerfstudio facade wraps it for `ns-train`; drop `StaticGSTrainer` subclass; static loss-mask via a `mask_provider` callable (no monkeypatch).
- **#7 static-capture home** — REC: separate `capture_tool` owns the anchor/record/SAM3/build-seed/pause-gazebo control-pipe ops; retarget `bootstrap_live.sh`.
- **#8 phase-0 boundary** — REC: `static_segment`→`SegmentResult`; `static_fuse.register_objects`→`FusedObject`; caller inserts (no interleaved per-object re-render).
- **Defer:** #16 warm-cache fingerprint scope, #18 the ~18 MB/tick `read_latest` copy (accept for v1), #20 grab-bag (ros1-only first, thumbnail gating, CPD/TEASER optional-import).
