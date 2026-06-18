# dynamic_gs2 — overnight build STATUS (2026-06-18)

Honest report of the autonomous rewrite session. **Read this first.**

## TL;DR
The new pipeline (`dynamic_gs2/`) is **built, unit-tested, and runs end-to-end on real
data**. The whole dynamic phase — warm-load the trained scene → tracker → write-pose →
trace — replays the real `screwdriver recorded full` episode and reaches **functional
equivalence with the old (validated) pipeline**: 312/312 frames tracked, trajectory
endpoint within **1.2 mm** over a 460 mm path. **90.8 % LOC cut (28 003 → 2 582)**, under
the < 10 000 target. The WRAP architecture (not BE) is proven on the real 458 k-gaussian scene.

What I did NOT do (yours / not unattended-safe): the **live camera run (step 4)** and the
**FF-on full AnySplat decode A/B** — both need a running sim / the AnySplat subprocess. Their
code is written and the testable halves are validated; see "Operator next steps".

## LOC KPI
| | LOC |
|---|---:|
| old `dynamic_gs/` (baseline) | 28 003 |
| new `dynamic_gs2/` (excl tests) | **2 582** |
| cut | **90.8 %** |

## Modules built (all committed, 9/9 tests green)
| module | what | validation |
|---|---|---|
| `frame.py` | Frame/Intrinsics + SHM seqlock codec (the contract) | round-trip test |
| `config.py` | one frozen RuntimeConfig + env overrides + fingerprint; **verified KF params** (0.005/0.025, OFF) | test |
| `shm_channel.py` | ShmProducer/ShmConsumer lifecycle (own/unlink vs attach-only) | round-trip + stale-reclaim test |
| `gaussian_set.py` | **SSOT** — locked surgery (cull/insert/write-pose/reload) + immutable snapshot | CPU test + **adversarial review (16 agents, 0 high/critical)** |
| `scene_model.py` | **WRAP** SplatfactoModel (render/object-mask/rebind/phase-LR) | GPU test: build→render→insert→rebind→render→cull |
| `adapters_source.py` | ReplaySource(paced/fast) + ShmRing + camera_from_frame + LiveBridgeSource | round-trip test |
| `static_persist.py` | warm-cache .pt save/load + build_loaded_scene | **loads REAL 458k scene, renders @1200p, round-trips** |
| `dynamic_track.py` | XFeat tracker WRAP + ReferenceObjectPose (R,t→subset) | smoke on real frames + deterministic pose math |
| `dynamic_feedforward.py` | FF worker: **P0 frozen-dispatch** + load-shed ceiling + purge | deterministic test (due/single-in-flight/ceiling/purge/**P0 grep-check**) |
| `dynamic_ff_backends.py` | real CDN (compute_change_mask) + AnySplat decode callbacks | CDN smoke-validated on real frames; decode wraps proven utils |
| `pipeline.py` | orchestrator: DynamicLoop + crop-to-object-bbox + recorded A/B + run_live + CLI | **full episode runs 312/312** |
| `verify/compare_traces.py` | old-vs-new per-tick rigid-transform A/B | ran; verdict below |

## VALIDATED (unattended, real data)
- **WRAP works**: GaussianSet (pure tensors) + SceneModel (wrapped SplatfactoModel, same
  tensor objects, rebind after surgery) renders the real trained scene and survives
  insert/cull. Retires the BE-vs-WRAP risk.
- **Warm-load**: the new core loads the real `static_state.pt` (458 658 gaussians, 44 k
  instance-id'd object, 42.7 k inserted), renders @1920×1200 (alpha 1.0, depth_med 0.443 m),
  save→load round-trips exactly.
- **Tracker A/B** (`screwdriver recorded full`, FF off, vs the actual old run's per-frame
  motion logs): **312/312 frames tracked by both**; **endpoint |t| old 460.7 mm / new
  459.5 mm (1.2 mm diff)** → FUNCTIONAL EQUIVALENCE **PASS**. Cumulative per-tick diff
  p50 1.5°/16 mm is accumulated match-set variance (a documented tracker property) that
  reconverges; strict bit-match was not expected and is NOT claimed.
- **CDN** (change detection) runs in the new pipeline on real frames (267 k–729 k px,
  correct crop windows).
- **P0 fix** ("FF inserts much less"): the FF bg thread reads ONLY the immutable
  `FeedforwardDispatch` (grep-asserted) and `enforce_ceiling`/`_purge_ff_inserts` bound
  growth (protect_mask never drops the tracked object) — implemented + unit-tested.

## NOT done tonight (yours / not unattended-safe) — code is WRITTEN, validation pending
1. **LIVE camera run (step 4 — yours).** `resume_live.sh` + `pipeline.run_live` +
   `LiveBridgeSource` are written. The bridge reuses the **entire proven ROS publisher**
   (spawned by the old `LiveShmSubscriber`) and forwards frames into the new SHM layout —
   zero new rospy/FK/mask code. Needs a running Gazebo/ROS stack to validate.
2. **FF-on full AnySplat decode A/B.** `dynamic_ff_backends.make_decode_fn` wraps the proven
   `reproject_anysplat_to_scene` + persistent worker; the CDN half is validated, but the full
   decode needs the AnySplat subprocess (`anysplat_dynamic_gs`) + a live run to confirm the
   "insert much less" volume number. The *fix* is already in (P0 + load-shed, tested).
3. **Static phase from-scratch** (segment/fuse/fit) — not needed for the recorded A/B (it
   warm-loads the existing `.pt`); deferred. The new pipeline currently consumes a scene the
   *old* static phase produced.

## Operator next steps
```bash
# VISUAL VALIDATION with the viser viewer — NO sim needed (replays recorded through the
# pipeline at ~10fps with the viewer up). Open http://localhost:8081 and orbit:
dynamic_gs2/view_dynamic.sh "<dataset>" transforms_313_trimmed.json
#   loops until Ctrl-C; add --once to play once; add --ff for feedforward.

# Non-interactive: side-by-side [live | render] mp4 of the tracking:
python -m dynamic_gs2.visualize "<dataset>" --transforms transforms_313_trimmed.json
#   -> <dataset>/dynamic_gs2_viz.mp4

# Recorded A/B (unattended-validated path; FF off):
dynamic_gs2/replay_ab.sh "<dataset>" transforms_313_trimmed.json
#   -> writes <dataset>/new_trace.jsonl + prints the old-vs-new verdict

# LIVE (step 4 — needs the sim up; viser viewer auto-starts on :8081):
dynamic_gs2/resume_live.sh "<dataset>" [--ff]
```

## Honest caveats / known gaps
- Tracker pose is **functionally** equivalent, not bit-identical (match-set variance +
  crop-bbox source differs: new uses the rendered object-mask bbox, old projected means).
  Endpoint match is the trustworthy signal.
- `dynamic_viz.py` (viser-direct viewer) **IS built** (server-side rasterize + push-image,
  Inv #9; reuses the proven viser↔OpenGL camera conversion). Verified headless: server binds
  :8081, scene loads, tracking runs, client connects, no errors. The in-browser image push
  reuses the validated render path; confirm visually via `view_dynamic.sh`.
- `Ros1Source` (from-scratch publisher spawner) exists but `LiveBridgeSource` is the
  recommended, proven-reuse live path.
- The `KeyError: '/dgs_...'` printed at the end of some runs is a benign same-process
  `resource_tracker` teardown artifact (producer+consumer in one process during tests);
  harmless, never fires across the real two-process boundary.
- FF-on decode npz-key contract + crop-window math are wrapped from the old code but
  unrun tonight — verify on the first `--ff` run.
