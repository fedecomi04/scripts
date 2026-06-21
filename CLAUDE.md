# CLAUDE.md

Guidance for Claude Code in this repo. This file is **always-loaded current truth only** —
no dated session notes, no frozen-baseline reference. Those live in:

- [`HISTORY.md`](HISTORY.md) — all dated `### … (YYYY-MM-DD)` session notes + superseded reasoning.
- [`BASELINE.md`](BASELINE.md) — the frozen OLD `dynamic_gs/` package reference (arch map, splatfacto trace).
- [`dynamic_gs2/STATUS_LIVE.md`](dynamic_gs2/STATUS_LIVE.md) — **the SSOT for the live pipeline** (most recent reality).

If you find yourself adding a dated note or a war-story to *this* file, it belongs in HISTORY.md instead.

---

## 1. Behavioral guidelines

**Think before coding.** State assumptions explicitly; if uncertain, ask. If multiple interpretations
exist, present them — don't pick silently. If a simpler approach exists, say so. If something is
unclear, stop and name it.

**Simplicity first.** Minimum code that solves the problem. No speculative features, abstractions for
single-use code, unrequested configurability, or error handling for impossible scenarios. If 200 lines
could be 50, rewrite it.

**Surgical changes.** Touch only what you must. Don't refactor what isn't broken, don't "improve"
adjacent code, match existing style. Remove orphans *your* change created; mention pre-existing dead
code, don't delete it. Every changed line should trace to the request.

**Goal-driven execution.** Turn tasks into verifiable goals ("fix the bug" → "write a test that
reproduces it, then make it pass"). For multi-step work, state a brief plan with a verify-check per step.

---

## 2. CURRENT STATE — read this first

**`scripts/dynamic_gs2/` is the active, validated pipeline** (clean ~3k-LOC rewrite, ~90% LOC cut from
the old 28k). The old **`scripts/dynamic_gs/` package is the frozen ground-truth baseline** — its
`ns-train` methods still run and the rewrite is verified *against* it, but new work goes through
`dynamic_gs2/`.

Where to look:
- Live pipeline reality → [`dynamic_gs2/STATUS_LIVE.md`](dynamic_gs2/STATUS_LIVE.md)
- Build report → [`dynamic_gs2/STATUS.md`](dynamic_gs2/STATUS.md) · LOC KPI → [`dynamic_gs2/LOC.md`](dynamic_gs2/LOC.md)
- Module map the rewrite was generated against → [`rewrite_spec/00_OVERVIEW.md`](rewrite_spec/00_OVERVIEW.md)
- Settled design calls → [`rewrite_spec/00_DECISIONS.md`](rewrite_spec/00_DECISIONS.md)
- Launch commands → [`commands.md`](commands.md) — `dynamic_gs2/` has FOUR mode scripts (`full_live.sh` / `full_recorded.sh` / `warm_live.sh` / `warm_recorded.sh`); the core is source-agnostic (`--source {live_bridge,replay}`).

**Architecture (current, `dynamic_gs2`) — WRAP not BE:**
- [`gaussian_set.py`](dynamic_gs2/gaussian_set.py) is the SSOT: 6 gauss params + 4 identity buffers behind
  one locked surgery API + immutable `snapshot()`.
- [`scene_model.py`](dynamic_gs2/scene_model.py) renders/trains those same tensor objects via a *wrapped*
  nerfstudio `SplatfactoModel` (`rebind()` after surgery, no copy).
- One SHM ingest path ([`frame.py`](dynamic_gs2/frame.py) + [`shm_channel.py`](dynamic_gs2/shm_channel.py)
  + [`adapters_source.py`](dynamic_gs2/adapters_source.py)) for recorded (`ReplaySource`) and live
  (`LiveBridgeSource`, which reuses the proven ROS publisher into the new SHM layout).
- Live mask render → [`dynamic_gs2/ros_mask.py`](dynamic_gs2/ros_mask.py) (`RobotMaskGenerator`).
- FF dispatch → [`dynamic_feedforward.py`](dynamic_gs2/dynamic_feedforward.py) (bg thread reads only the
  immutable `FeedforwardDispatch`).

> The full module list + current launch flows live in STATUS_LIVE.md — keep that authoritative, not this summary.

---

## 3. Keeping this file accurate (MANDATORY)

This file is loaded as authoritative instructions every session — a stale claim actively misleads.

1. **If you change code this file references, update this file in the SAME change.** Config defaults/values,
   the lines enforcing an Invariant, default-flag states, any symbol/file/module name mentioned here. The
   diff is not done until the doc matches the code.
2. **Reference code by symbol name, NOT line number.** Write `` [`config.py`](path) (`_ZERO_LR_OPTIMIZERS`) ``,
   never `:138`. Line numbers drift on any unrelated edit; symbol names only break on an actual rename (and
   then rule 1 applies). Exception: the pinned vendored-nerfstudio trace in BASELINE.md.
3. **Never rewrite a dated note's numbers** (that fabricates). Dated notes record measurements *at that date*.
   If a conclusion was later reverted, prepend a `> **SUPERSEDED (date):**` banner in HISTORY.md — don't edit
   the body. **Banners stack in HISTORY.md, never here:** this file states only current reality, so a fact
   that flips gets *replaced* here and the old version + banner moves to HISTORY.md.

---

## 4. Design Invariants (NON-NEGOTIABLE — DO NOT VIOLATE)

Hard rules the pipeline depends on. If a change appears to require breaking one, **stop and flag it
explicitly** — do not silently violate. Each has a stated reason so you can judge edge cases.

> Symbol refs below may still point at the frozen `dynamic_gs/` baseline. The invariants themselves are
> pipeline-agnostic — `dynamic_gs2` preserves all of them (it wraps the same `SplatfactoModel`). **TODO:**
> confirm/replace each `Verified:` pointer with its `dynamic_gs2` equivalent (the `gaussian_set.py` SSOT
> owns the surgery the old `StaticGSModel`/`DynamicGSModel` did). Don't guess the gs2 symbol — verify it.

1. **Static phase: `means` LR = 0.** Positions stay locked on the TSDF-fused seed
   (`depth_camera_init_points.ply`); only `features_dc/_rest`, `opacities`, `scales`, `quats` train.
   **Why:** the seed is geometrically correct (ICP-fused depth); means drifting under photometric loss
   smears the output. Adam moves means via `.grad` regardless of densification, so this must be an
   explicit `lr=0.0`, not "effectively 0". *History: HISTORY.md (was 1.6e-4 until 2026-06-02).*

2. **Static phase: `camera_optimizer.mode = "off"`.** Poses are NOT optimized during static training.
   **Why:** `transforms.json` holds ICP-refined poses (invariant #3), residual is sub-mm — nothing to fix.
   Leaving it on drifts cameras by visible amounts and smears the scene. *History: HISTORY.md (was
   `"SO3xR3"` until 2026-06-02).*

3. **`<data>/static_scene/transforms.json` holds ICP-refined poses, not raw URDF FK.** Raw capture
   preserved at `transforms_urdf_backup.json`. **Why:** the seed PLY lives in the ICP-refined frame;
   raw-FK training cameras leave a systematic 1–4 mm misalignment camera-opt can't undo (it's off, #2).
   Tool: [`scripts/rewrite_transforms_with_icp.py`](rewrite_transforms_with_icp.py) — idempotent, refuses
   to overwrite an existing backup. Measured drift: median 0.96 mm / 0.053°, max 3.94 mm / 0.41° over 68 frames.

4. **Dynamic phase: ALL gauss-param LRs = 0.** The dynamic phase is a runtime, not a training loop — only
   the tracker's rigid transform + feedforward inserts mutate the scene. Per-step gradient descent would
   fight the tracker. **Verified:** `_ZERO_LR_OPTIMIZERS` in the config (used by `dynamic-gs` + `dynamic-gs-live`).

5. **`outputs/` is suppressed across all runs.** Nerfstudio's `outputs/<exp>/…` tree is unused; all
   artifacts live under `<data_dir>/`. **Why:** the dataset dir is self-contained and portable.
   **Verified:** three monkeypatches in [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py)
   (`ExperimentConfig.save_config`, the `dataparser_transforms.json` write, the tensorboard branch).
   Note: `--vis viewer` still `mkdir`s `outputs/<run>` — pre-create the parent if ever enabled.

6. **Background color = Gazebo sky `(0.86, 0.92, 1.0)`.** Used by the model render + viser default.
   **Why:** the sim renders against this sky; any other background injects a constant photometric bias
   the renderer would compensate for via opacity tweaks at silhouettes.

7. **Persistent SAM3 + Fast-SAM3D worker is the canonical SAM3/SAM3D path during live capture.**
   `SamWorkerClient` (spawn-once, load-on-demand, JSON-over-pipe). **Why:** measured 9.4 s/call (SAM3
   cold-start) + 22 s/call (SAM3D reuse) savings. The per-call subprocess paths remain fallbacks; the
   live flow auto-spawns the worker at `fusion_runner.start()`.

8. **Per-object identity buffers are owned by specific phases:**
   - `object_instance_ids` — written by Phase 0b fusion only (1..K).
   - `inserted_flags` — written by Phase 0b (SAM3D inserts) + `rgbd_decode.insert_inpaint_gaussians` (FF Mode B).
   - `sam3d_init_target_flags` — initialized to zeros, **never written at runtime**. The only value-writer
     (`DynamicGSModel.initialize_object_from_sam3d`) has no caller. All-zeros is the expected state.
   - `object_flags` — written by D0 selection on the first dynamic frame, **never by the static pipeline**.
     `object_flags=0` in `post_fusion_state.pt` is correct/expected.

9. **Live visualization uses viser-direct, NEVER Nerfstudio's viewer.** Connect to
   **`http://localhost:8081`**, never `:7007`. Do NOT pass `--vis viewer`.
   - **Mechanism:** viser-direct is **server-side rasterize + push-image** (one dedicated render thread
     calls `model.get_outputs(camera)`, ~25 ms at 512×512, and pushes the result via an atomic
     `client.scene.set_background_image(...)`; a single `model_lock` guards every tracker write + FF insert
     + render). *(The old client-side `GaussianSplatHandle` "Path A" was removed; its API is no-op stubs.)*
   - **Why avoid the NS viewer:** its render-state-machine render path wraps `get_outputs` in a
     render-interrupt (`IOChangeException`) that, under the shared FF lock, can **deadlock** (observed
     2026-06-02). viser-direct has no state machine and no render-interrupt.
   - **Enforced:** every method config sets `vis="tensorboard"` (NS viewer off; tensorboard writes
     suppressed by `_suppress_nerfstudio_output_writes`). `enable_viser_direct=True` is the default, so
     viser-direct spins up on port 8081 (`viser_direct_port`).
   - *History (Path-A removal, the deadlock, the flash-on-write bug): HISTORY.md.*

---

## 5. Data format + conventions (stable)

```
<data_dir>/
├── static_scene/   rgb/(BGR PNG) depth/(uint16 mm TIFF) masks/(uint8) transforms.json
│                   depth_camera_init_points.ply (TSDF seed)  post_fusion_state.pt (warm cache)
├── dynamic_scene/  rgb/ depth/ masks/ transforms.json  initialization_{debug,artifacts}/
└── timing_report*.txt
```

- Dataparser: `orientation_method="none"`, `center_method="none"`, `auto_scale_poses=False` — metric, not recentered.
- Camera poses are OpenGL c2w; converted to OpenCV internally via `diag(1, -1, -1, 1)`.
- Depth is uint16 mm on disk (`depth_unit_scale_factor=1e-3`); publisher emits float32 m at the SHM boundary.

## 6. Conda environments

| Env | Python | Role |
|---|---|---|
| `dynamic_gs` | 3.12 | Main: ns-train methods, tracker, Open3D 0.19 (GPU TSDF), nerfstudio, gsplat. |
| `sam3_dynamic_gs` | 3.12 | SAM3 + Fast-SAM3D (+ FastSAM/CLIP) worker subprocess. |
| `dynamic_gs_ros` | 3.8 | Minimal ROS Noetic env for the live publisher subprocess. |
| `anysplat_dynamic_gs` | 3.12 | AnySplat feedforward decoder (persistent worker). |

## 7. Open roadmap

1. Per-Gaussian SAM IDs — every Gaussian gets a real instance ID at the source, not just SAM3D inserts.
2. Auto-pick by gripper TCP (closest-point to gripper, not centroid-to-camera).
3. Multi-object Fast-SAM3D (`prompt_text` → `list[str]`, distinct instance IDs).
4. Multi-object switching tracker (track whichever instance is moving).
- Gaussian hygiene purge: drop sub-0.05-opacity + super-small-scale Gaussians, one-shot at static end
  AND periodically in the dynamic phase to cap FF accumulation; never drop `object_flags==1`.

## 8. Timing

Per-substep numbers live in `<data_dir>/timing_report.txt` after each run. **Don't quote a timing number
from memory** — historical estimates have been wildly off. Verify against a recent report.
