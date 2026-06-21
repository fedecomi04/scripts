# BASELINE.md — frozen OLD `dynamic_gs/` package reference

The **old `dynamic_gs/` package is the frozen ground-truth baseline**, NOT the active pipeline. It still
runs and the `dynamic_gs2/` rewrite is verified *against* it, and much of `dynamic_gs2/` WRAPs these same
vendored utils — so this architecture map stays useful as reference. For the CURRENT pipeline see
[`dynamic_gs2/STATUS_LIVE.md`](dynamic_gs2/STATUS_LIVE.md) + [`commands.md`](commands.md). Dated war-stories
are in [`HISTORY.md`](HISTORY.md); the live-truth + invariants are in [`CLAUDE.md`](CLAUDE.md).

## Project Overview

**dynamic-gs** is a static + dynamic Gaussian Splatting system for robotic teleoperation, integrated with [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The static phase fits a Splatfacto scene; the dynamic phase tracks objects via XFeat and (optionally) feedforward-decodes newly revealed surfaces. Designed for live RGB-D streams from a single arm-mounted camera.

The codebase was rewritten 2026-05-30 → 2026-06-01. The historical monolith (`dynamic_gs_pipeline.py`, 5329 LOC) was deleted; capabilities are now split across three thin pipelines, the dynamic logic into a shared base. *(That refactor was itself later superseded by the full `dynamic_gs2/` rewrite — see CURRENT STATE at the top.)*

## Installation

```bash
pip install -e .   # from scripts/
```

ns-train auto-discovers our methods via the `nerfstudio.method_configs` entry-point in `pyproject.toml`. Method names registered: `static-gs`, `static-gs-preseg`, `dynamic-gs`, `dynamic-gs-live`.

## Conda Environments

| Env | Python | torch | sm_120 native | Role |
|---|---|---|---|---|
| `dynamic_gs` | 3.12 | 2.11+cu128 | ✅ | Main env: hosts all four ns-train methods, XFeat tracker, Open3D 0.19 (TSDF on GPU), nerfstudio, gsplat. |
| `sam3_dynamic_gs` | 3.12 | 2.11+cu128 | ✅ | SAM3 + Fast-SAM3D subprocess env. Invoked via `conda run -n sam3_dynamic_gs python ...` from `utils/sam3d.py` and `utils/sam3_segmentation.py`. |
| `dynamic_gs_ros` | 3.8 | none | n/a | Minimal ROS Noetic env for the live publisher subprocess. ROS bindings come from `/opt/ros/noetic/lib/python3/dist-packages` via `source /opt/ros/noetic/setup.bash`. The publisher spawn wrapper sets `PYTHONNOUSERSITE=1` — without it, user-local pyrender shadows the env's. |
| `anysplat_dynamic_gs` | 3.12 | — | ✅ | AnySplat feedforward decoder env (persistent worker; see `utils/anysplat_decode.py`). |

## Running (OLD pipeline)

> For the CURRENT pipeline use the `dynamic_gs2/` four mode scripts (`full_live.sh` / `full_recorded.sh` /
> `warm_live.sh` / `warm_recorded.sh`, see [`commands.md`](commands.md)). The scripts below drive the
> frozen `dynamic_gs/` baseline.

Three top-level scripts cover the common flows. Defaults are chosen so most invocations are zero-argument.

```bash
# Capture-only: record a fresh static + dynamic dataset, no training
scripts/capture_only.sh
# default: data → datasets/<YYYY-MM-DD_HHMMSS>/, 2cm/20° dedup static, 30 fps dynamic

# Full pipeline: capture + train static + go live
scripts/bootstrap_live.sh <data_dir> [sam3_prompt]

# Resume on a pre-trained dataset (skip capture + static training)
scripts/resume_live.sh <data_dir>
# requires <data_dir>/static_scene/post_fusion_state.pt
```

Direct method invocations also work:
```bash
ns-train static-gs        --data <data_dir> --pipeline.model.sam3_prompt_text "..."
ns-train static-gs-preseg --data <data_dir> --pipeline.text-prompts "..."   # per-Gaussian SAM IDs, no SAM3D/CPD
ns-train dynamic-gs       --data <data_dir>   # recorded dataset
ns-train dynamic-gs-live  --data <data_dir>   # live SHM stream
```

`outputs/` is intentionally empty across runs — see [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py) for the three monkeypatches that suppress nerfstudio's `config.yml` / `dataparser_transforms.json` / tensorboard writes. All artifacts live under the dataset dir.

## High-Level Architecture

### Pipelines (4 modules)

```
DynamicGSPipelineBase            (dynamic_gs_pipeline_base.py)
├─ RecordedDynamicGSPipeline    (dynamic_gs_pipeline_recorded.py)
└─ LiveDynamicGSPipeline        (dynamic_gs_pipeline_live.py)

StaticGSPipeline                 (static_gs_pipeline.py)
```

* **`StaticGSPipeline`** — fits Splatfacto on the static dataset, then runs Phase 0a (SAM3 + Fast-SAM3D, in `fusion/phase0.py`) and Phase 0b (NDP non-rigid registration by default — CPD/TEASER++ still selectable — + insertion + post-fusion cull). Writes the warm-cache snapshot `<data>/static_scene/post_fusion_state.pt`.
* **`DynamicGSPipelineBase`** — shared dynamic-phase logic: XFeat tracker tick, feedforward dispatcher (`rgbd_decode` or `anysplat_decode`), viser-direct push, persistent per-object identity buffers.
* **`RecordedDynamicGSPipeline`** — feeds the base from a recorded `dynamic_scene/`.
* **`LiveDynamicGSPipeline`** — feeds the base from a `LiveShmSubscriber` polling the ROS publisher's shared memory.

### Models (2 modules)

* **`StaticGSModel`** (`static_gs_model.py`) — straight subclass of `SplatfactoModel` + the four persistent buffers (`object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`). Uses `NoRefineStrategy` so densification is OFF during static training. Means LR is zeroed so seed positions stay fixed.
* **`DynamicGSModel`** (`dynamic_gs_model.py`) — superset used by recorded + live dynamic pipelines. Adds `render_object_mask`, the rigid-transform helpers, ESAM lazy load, a `means`-gradient zeroing hook (enforces invariant #4), and the dynamic-phase config knobs. *(The scene-optimization machinery — `enable_scene_optimization`, `scene_opt_*`, `scene_opt_active_mask` — was purged in `06d2c47`; the dynamic phase does no per-step gradient descent.)*

### Data + persistence

* **Datamanager**: [`DynamicGSDataManager`](dynamic_gs/dynamic_gs_datamanager.py) wraps two `FullImageDatamanager`s (`static_scene/`, `dynamic_scene/`). Live mode pulls frames from SHM via [`LiveShmSubscriber`](dynamic_gs/utils/live_shm_reader.py).
* **Persistence**: [`dynamic_gs/persistence/`](dynamic_gs/persistence/) — `save_post_fusion_state` writes `gauss_params.*` + all persistent buffers; `load_post_fusion_state` warm-restarts the dynamic pipelines from a static-gs snapshot.
* **Fusion phase 0**: [`dynamic_gs/fusion/phase0.py`](dynamic_gs/fusion/phase0.py) — `run_phase0a_sam3_and_sam3d` + `run_phase0b_fusion`.

### Static-phase seed pipeline (2026-06-01)

The PLY at `<data>/static_scene/depth_camera_init_points.ply` is what Splatfacto inits Gaussians from. Today:

1. **During capture**, [`utils/fusion_runner.py`](dynamic_gs/utils/fusion_runner.py) runs an `OnlineFusion` worker thread that watches `transforms.json` and integrates each new keyframe (`add_frame`). GPU TSDF + ICP at 2 mm voxel (`TSDF_VOXEL_M=0.002`), only depth in the 0.05–3.0 m band (`DEPTH_MIN_M`/`DEPTH_MAX_M`, hardcoded), ~16 ms/frame at 800×800.
2. **On capture stop**, `stop_and_finalize()` drains the queue, calls `finalize()`, and writes the PLY (~0.6 s).
3. **Optional adaptive downsample**: [`scripts/adaptive_downsample.py`](scripts/adaptive_downsample.py) keeps the <1 m near-zone at full density, voxel-downsamples the rest to 5 mm. Not yet auto-wired into the bootstrap flow; run manually if seed size matters.

### Utility modules ([`dynamic_gs/utils/`](dynamic_gs/utils/))

| Module | Role |
|---|---|
| `online_fusion.py` | `OnlineFusion` class: TSDF + ICP per frame, GPU. Drives the init seed. |
| `fusion_runner.py` | `ConcurrentFusionRunner` — watcher + worker for streaming fusion. |
| `rgbd_fusion_init.py` | Legacy offline post-pass refinement; only used as a fallback. |
| `xfeat_motion.py` | The XFeat-only dynamic tracker (5-tracker dispatch purged 2026-05-26). |
| `tracker_common.py` | Kabsch + RANSAC helpers + `MotionEstimate` dataclass shared by the tracker. |
| `live_ros_publisher.py` | The ROS publisher run inside `dynamic_gs_ros` (subprocess). Owns the SHM. |
| `live_shm_reader.py` | Reader-side wrapper: spawns the publisher, polls SHM, gives `peek_latest()`. |
| `live_session.py` | The bootstrap-time interactive capture flow (SAM3 retry loop, SAM3D, fusion). |
| `keyframe_filter.py` | ORB-SLAM-style greedy 2 cm/20° pose dedup; shared between recorded + live. |
| `sam3_segmentation.py`, `sam3d.py`, `sam3d_fusion.py` | SAM3 + Fast-SAM3D subprocess wrappers + Phase-0b registration & fusion (NDP default; CPD/TEASER++ fallbacks). |
| `ndp_register.py`, `ndp/` | Vendored NDP non-rigid deformation (`deform_source_to_target`) — the default Phase-0b backend. Pure-torch, no pytorch3d, in-process GPU. |
| `esam.py` | ESAM interactive object-mask query (D0 bootstrap). |
| ~~`optim_pool.py`~~ | **REMOVED** — dynamic-phase per-step optimization was dropped (the dynamic phase is a pure tracker+FF runtime; see invariant #4). File no longer exists; `OptimPool`/`optim_pool_*` are dead references. |
| `active_mask.py` | `build_change_mask`, `select_top_n_components_filtered`, projection helpers. |
| `viser_direct.py` | Standalone viser server pushed by the tracker; bypasses ns-viewer state machine. |
| `rgbd_decode.py` | Feedforward Mode A/B: direct RGB-D back-projection into frozen Gaussians. |
| `anysplat_decode.py` | Feedforward via the AnySplat persistent subprocess. |
| `depth_loss.py`, `rigid_regularization.py`, `no_refine_strategy.py` | Smaller pieces used by the model. |

The legacy trackers (cotracker / tapir / tapnext / klt) and the old live-subscriber were **deleted** (commit `5de7fab`, 2026-05-31) — recover from git history if ever needed.

### Three-phase training (overview)

**Phase 0 (Static)** — `static-gs`. Splatfacto fit on the SfM/TSDF seed for `static_num_steps` (default 500, `STATIC_NUM_STEPS` in `dynamic_gs_config.py`). Densification OFF, means LR = 0, camera-pose optimizer = `off` (NOT `SO3xR3` — see invariant #2). At end: Phase 0a SAM3 + Fast-SAM3D, then Phase 0b NDP non-rigid registration (default; CPD/TEASER++ selectable) + insertion + post-fusion cull (proximity de-dup + in-front occlusion). Writes `post_fusion_state.pt`.

**Phase 1 (Dynamic)** — `dynamic-gs` / `dynamic-gs-live`. Warm-load from `.pt`. Per tracker tick: XFeat motion estimation → `apply_rigid_object_transform_from_reference` → viser-direct push. Optionally feedforward-decode CDN regions (rgbd or anysplat) into the scene.

The legacy "Phase 0 split" (object insertion AFTER static training) is preserved by `static-gs`'s `_finalize_static_training` AFTER_TRAIN callback. The dynamic pipelines load the post-fusion snapshot directly and skip retraining.

### Per-object identity buffers

| Buffer | Type | Set by | Purpose |
|---|---|---|---|
| `object_instance_ids` | long (N,1) | Phase 0b fusion | Multi-object identity, 1..K |
| `object_flags` | float (N,1) | D0 selection | Active dynamic object (0/1) |
| `sam3d_init_target_flags` | float (N,1) | nobody (writer uncalled) | Placeholder — intended to mark SAM3D-inserted Gaussians; never written at runtime (see Invariant #8) |
| `inserted_flags` | float (N,1) | rgbd_decode | Feedforward Mode B inserts |

`object_instance_ids` only carries IDs for Fast-SAM3D-inserted Gaussians today. Future #1 (top of roadmap) will give every TSDF-seeded Gaussian a real ID via per-frame SAM2 propagation.

### Optimizer groups + LRs

Standard Splatfacto 7 groups: `means`, `features_dc`, `features_rest`, `opacities`, `scales`, `quats`, `camera_opt`.

* Static phase (`static-gs`): all groups active, but `means` LR = **0.0** (explicit, in [`dynamic_gs_config.py`](dynamic_gs/dynamic_gs_config.py)) to lock seed positions — Adam moves means via `.grad` regardless of densification, so the old 1.6e-4 did NOT stay put (see Invariant #1).
* Dynamic phase (`dynamic-gs` + `dynamic-gs-live`): all LRs zeroed (`_ZERO_LR_OPTIMIZERS` in `dynamic_gs_config.py`). The trainer's optimizer step is a no-op; mutations come from `apply_rigid_object_transform_from_reference` and feedforward inserts.

## Data Format

```
<data_dir>/
├── static_scene/
│   ├── rgb/                          (BGR PNG)
│   ├── depth/                        (uint16 mm TIFF)
│   ├── masks/                        (uint8 robot-exclusion mask)
│   ├── transforms.json               (Nerfstudio-formatted)
│   ├── depth_camera_init_points.ply  (TSDF-fused seed)
│   └── post_fusion_state.pt          (warm-cache after static-gs)
├── dynamic_scene/
│   ├── rgb/  depth/  masks/  transforms.json
│   ├── initialization_debug/         (SAM3 anchor + debug images)
│   └── initialization_artifacts/     (per-object SAM3D PLY + pose JSON)
└── timing_report*.txt
```

Dataparser settings: `orientation_method="none"`, `center_method="none"`, `auto_scale_poses=False` — poses are kept in metric units, not recentered.

## ROS Data Collection

The live publisher subprocess is auto-spawned by `LiveShmSubscriber`. It runs the URDF FK + frame sync + atomic frame writes; the reader-side process never imports rospy.

Required: `dynaarm_with_gripper_for_gazebo_only_no_wrist_collision.urdf` must load the `libactive_camera_arm_link_pose_publisher.so` Gazebo plugin (publishes the camera pose to `/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose`). See the historic 2026-05-04 version in `~/.config/Code/User/History/-45f4ea38/KHwu.urdf` for the canonical content.

**Camera-pose plugin (`StampedLinkPosePublisher`) — what it actually publishes + the reset-survival fix (2026-06-14):** the plugin source is `active_camera_arm_control/active_camera_arm_gazebo/src/StampedLinkPosePublisher.cpp` (in the teleop catkin_ws, NOT this repo). Corrections to earlier notes here: it publishes the pose of **`dynaarm_WRIST_2_base`** (relative to `dynaarm_base`), **not** `camera_pose_link` (that link exists but is unused by this plugin); `updateRate=250.0` is set in the URDF `<plugin>` block, not the `.cpp`. **Failure mode + fix:** the plugin throttles by `world->SimTime()` and cached `last_publish_time_`. Until 2026-06-14 it had **no `Reset()` override**, so a `reset_world`/`reset_simulation` (or any `/clock` reset) rewound SimTime to ~0 while `last_publish_time_` kept its large value → the throttle delta went negative → it skipped publishing **every tick forever** (plugin stays loaded, world keeps stepping, but `gazebo_pose` goes permanently silent → the live publisher hangs waiting for the pose topic; the preflight `dgs_check_sim_alive` catches this). Fixed (teleop repo `federico/dynamic-gaussian-splat`, commit `d905560`) by adding a `Reset()` override + a backwards-SimTime guard in `OnUpdate`, so it re-arms on reset and keeps publishing across world resets. **The rebuilt `.so` only loads on a fresh model spawn / Gazebo restart** — a `reset_world` on an already-running pre-fix sim won't pick it up. If `gazebo_pose` is ever silent while the sim is otherwise healthy (joint_states flowing, RTF≈1, physics not paused), this is the first thing to check.

`urdf/dynamic_gaussian_splat/` and `worlds/dynamic_gaussian_splat/` symlinks are required under the catkin workspace — the publisher expects them at those paths.

## Third-Party Dependencies (`third_party/`)

* **`sam-3d-objects/`** — SAM3D model for single-view 3D object reconstruction. Multi-object path via `utils/sam3d.run_sam3d_multi_object`.
* **SAM3** ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) — text-prompted segmentation. Invoked via `conda run -n sam3_dynamic_gs python`.
* **ESAM** — interactive segmentation, D0 bootstrap.
* **AnySplat** — feedforward decoder, persistent subprocess in `anysplat_dynamic_gs`.
* **NDP (Neural Deformation Pyramid)** — vendored in `utils/ndp/` (`nets.py` + `rigid_body.py`); the default non-rigid Phase-0b backend via `utils/ndp_register.py`. Upstream: github.com/rabbityl/DeformationPyramid (no-learned path, no checkpoint).
* **PROBREG / Open3D** — `utils/sam3d_fusion.py` CPD fallback; TEASER++ the rigid alternative (both still selectable, no longer the default).
* **XFeat / LighterGlue** — vendored under `dynamic_gs/utils/xfeat_motion.py`'s dependencies.
* **FoundationPose** — `third_party/FoundationPose/` kept on disk but no longer wired into the runtime (XFeat purge 2026-05-26).

## Background + Camera Conventions

* Camera poses are OpenGL c2w in `transforms.json`. `OnlineFusion` and FoundationPose helpers convert to OpenCV internally via `diag(1, -1, -1, 1)`.
* Depth is uint16 millimetres on disk (`depth_unit_scale_factor = 1e-3` in the dataparser). The publisher converts to float32 metres at the SHM boundary.
* Simulator background: Gazebo sky color `(0.86, 0.92, 1.0)` is set as the model's render background — not the dataparser `mask_color`. Defined in `StaticGSModel.populate_modules()` and `DynamicGSModel.populate_modules()` and as the viewer default in `nerfstudio/viewer/control_panel.py`.

## Open Roadmap

(Detailed in the memory entry `project_multi_object_roadmap`.)

1. **Per-Gaussian SAM IDs** — port from `experiments/icp_fusion_mvp/`. Every Gaussian gets a real instance ID at the source, not just Fast-SAM3D inserts.
2. **Auto-pick by gripper TCP** — D0 picker uses closest-point to gripper, not 3D centroid to camera.
3. **Multi-object Fast-SAM3D** — `sam3_prompt_text` becomes a `list[str]`; multi-mask insertion with distinct instance IDs.
4. **Multi-object switching tracker** — track whichever instance is currently moving; swap on detected motion change.

Also pending: **Gaussian hygiene purge** — drop sub-0.05-opacity AND super-small-scale Gaussians, one-shot at end of static phase (~26 % reduction, no visible change) AND periodically during the dynamic phase to cap FF-insert accumulation (459k→1.29M on real-1200p); never drop `object_flags==1`. Complements the oversized-insert clamp TODO above (one combined `min_scale < s < max_scale` + opacity filter on every insert batch). See [[static-phase-opacity-purge-todo]]. Also: Phase 0b CPD vs TEASER++ comparison.

## Timing Reference

Per-substep numbers live in `<data_dir>/timing_report.txt` after each run. Don't trust any number quoted here without verifying against a recent report — historical estimates have been wildly off (see memory entry `feedback_no_timing_guesses`).

Most recent measurements (validate_run_1, 800×800, 71 frames, 2026-06-01):
* Online fusion (GPU): mean 16 ms/frame, p90 21 ms — see `scripts/bench_gpu_fusion.py`.
* Static training (Splatfacto, 1000 steps, no densify): under 20 s.
* XFeat tick: 17–30 Hz steady, ~21 ms/tick at `xfeat_top_k=300`.

---

## Vendored nerfstudio / Splatfacto reference trace

_Pinned reference into the vendored `../nerfstudio/` (line numbers point into that pinned dependency — the one place line-refs are allowed, per CLAUDE.md rule 2). Describes what stock Splatfacto does per step; the baseline the dynamic-gs custom phase/optim logic is compared against._

### Splatfacto per-iteration sequence (with code references)

Reference trace of what Nerfstudio + Splatfacto actually do per training step. Useful as the baseline to compare the dynamic-gs custom phase/optim logic against.

- Outer loop sets `self.step = step` — [trainer.py:247](../nerfstudio/nerfstudio/engine/trainer.py#L247)
- Fire `BEFORE_TRAIN_ITERATION` callbacks — [trainer.py:260-263](../nerfstudio/nerfstudio/engine/trainer.py#L260-L263)
- `step_cb` stashes step, optimizers, schedulers onto the model — [splatfacto.py:407-410](../nerfstudio/nerfstudio/models/splatfacto.py#L407-L410)
- Call `train_iteration(step)` — [trainer.py:266](../nerfstudio/nerfstudio/engine/trainer.py#L266)
- Zero gradients on this step's active param groups — [trainer.py:497](../nerfstudio/nerfstudio/engine/trainer.py#L497)
- `pipeline.get_train_loss_dict(step)` called — [trainer.py:502](../nerfstudio/nerfstudio/engine/trainer.py#L502)
- Pipeline calls `model.get_outputs(camera)` — [splatfacto.py:485](../nerfstudio/nerfstudio/models/splatfacto.py#L485)
  - Apply learned camera-pose correction — [splatfacto.py:501](../nerfstudio/nerfstudio/models/splatfacto.py#L501)
  - Build view matrix + intrinsics K — [splatfacto.py:534-535](../nerfstudio/nerfstudio/models/splatfacto.py#L534-L535)
  - Call `gsplat.rasterization(...)` → `render, alpha, self.info` — [splatfacto.py:555-575](../nerfstudio/nerfstudio/models/splatfacto.py#L555-L575)
  - `strategy.step_pre_backward(...)` (registers `means2d` to retain its gradient) — [splatfacto.py:577-579](../nerfstudio/nerfstudio/models/splatfacto.py#L577-L579)
  - Composite rendered RGB with background — [splatfacto.py:583](../nerfstudio/nerfstudio/models/splatfacto.py#L583)
- Pipeline calls `model.get_loss_dict(outputs, batch)` — [splatfacto.py:652](../nerfstudio/nerfstudio/models/splatfacto.py#L652)
  - Composite GT image with background — [splatfacto.py:660](../nerfstudio/nerfstudio/models/splatfacto.py#L660)
  - Compute L1 = `mean(|gt − pred|)` — [splatfacto.py:673](../nerfstudio/nerfstudio/models/splatfacto.py#L673)
  - Compute `1 − SSIM(gt, pred)` — [splatfacto.py:674](../nerfstudio/nerfstudio/models/splatfacto.py#L674)
  - Combine: `(1 − ssim_lambda)·L1 + ssim_lambda·(1 − SSIM)` — [splatfacto.py:689](../nerfstudio/nerfstudio/models/splatfacto.py#L689)
- Sum loss_dict into scalar `loss` — [trainer.py:503](../nerfstudio/nerfstudio/engine/trainer.py#L503)
- `grad_scaler.scale(loss).backward()` fills `.grad` on every param tensor and on `info["means2d"]` — [trainer.py:504](../nerfstudio/nerfstudio/engine/trainer.py#L504)
- `optimizer_scaler_step_some` → Adam step on each active param group — [trainer.py:510](../nerfstudio/nerfstudio/engine/trainer.py#L510)
- `scheduler_step_all` → schedulers decay LRs for next step — [trainer.py:527](../nerfstudio/nerfstudio/engine/trainer.py#L527)
- `train_iteration` returns to outer loop
- Fire `AFTER_TRAIN_ITERATION` callbacks — [trainer.py:269-272](../nerfstudio/nerfstudio/engine/trainer.py#L269-L272)
- `step_post_backward` dispatcher runs — [splatfacto.py:365](../nerfstudio/nerfstudio/models/splatfacto.py#L365)
- Delegates to `strategy.step_post_backward(...)` — [splatfacto.py:367-374](../nerfstudio/nerfstudio/models/splatfacto.py#L367-L374)
  - If outside refinement window or wrong step → return early
  - Else read `info["means2d"].grad`, decide clone/split/prune, mutate `gauss_params` + each Adam's `m`,`v` state in lockstep
  - Every `reset_alpha_every × refine_every` steps: reset all opacities low
- Loop to next step

### Splatfacto `get_outputs(camera)` — the render function

Pure forward render: given one camera, returns the rendered image (plus depth/alpha). Does **not** compute loss, does **not** call backward, does **not** modify Gaussians.

1. **Apply pose correction** — `camera_optimizer.apply_to_camera(camera)` adds the learned 6D offset to the dataset c2w (training only, if camera-opt is on) — [splatfacto.py:501](../nerfstudio/nerfstudio/models/splatfacto.py#L501).
2. **Pick which Gaussians to render** — all of them, unless a `crop_box` is set (viewer feature) — [splatfacto.py:506-528](../nerfstudio/nerfstudio/models/splatfacto.py#L506-L528).
3. **Build camera matrices** — `viewmat` from corrected c2w, intrinsics `K` — [splatfacto.py:534-535](../nerfstudio/nerfstudio/models/splatfacto.py#L534-L535).
4. **Pick render mode** — `"RGB+ED"` if depth is needed this step, else `"RGB"` — [splatfacto.py:544-547](../nerfstudio/nerfstudio/models/splatfacto.py#L544-L547).
5. **Active SH degree** — `min(step // sh_degree_interval, max_sh_degree)` — coarse-to-fine color schedule — [splatfacto.py:549-553](../nerfstudio/nerfstudio/models/splatfacto.py#L549-L553).
6. **Call `gsplat.rasterization(...)`** — the differentiable splatting kernel; inputs the 7 param tensors + camera matrices; returns `render`, `alpha`, `self.info` — [splatfacto.py:555-575](../nerfstudio/nerfstudio/models/splatfacto.py#L555-L575).
7. **`strategy.step_pre_backward(...)`** — registers `means2d` so its gradient is retained through backward (needed by densification) — [splatfacto.py:577-579](../nerfstudio/nerfstudio/models/splatfacto.py#L577-L579).
8. **Composite with background** — `rgb = render + (1 − alpha) · background`, clamp [0,1] — [splatfacto.py:582-584](../nerfstudio/nerfstudio/models/splatfacto.py#L582-L584).
9. **Apply bilateral grid** — only if enabled and training; per-image color correction — [splatfacto.py:587-589](../nerfstudio/nerfstudio/models/splatfacto.py#L587-L589).
10. **Extract depth** — mask out empty regions (alpha = 0) — [splatfacto.py:591-595](../nerfstudio/nerfstudio/models/splatfacto.py#L591-L595).
11. **Return** `{"rgb", "depth", "accumulation" (= alpha), "background"}` — [splatfacto.py:600-604](../nerfstudio/nerfstudio/models/splatfacto.py#L600-L604).

**`alpha`** is per-pixel accumulated opacity in `[0, 1]`: `alpha = 1 − Π(1 − αᵢ)` over all Gaussians touching that pixel. Used to composite the background and to mask depth.

### Splatfacto `get_loss_dict(outputs, batch, metrics_dict)` — the loss function

Takes the rendered output and GT batch, returns a dict of scalar losses that the trainer sums and backprops.

1. **Composite GT with background** — same background as the render, so they're compared on equal footing — [splatfacto.py:660](../nerfstudio/nerfstudio/models/splatfacto.py#L660).
2. **Apply mask if present** — if `batch["mask"]` exists, multiply both GT and pred so masked pixels contribute zero — [splatfacto.py:665-671](../nerfstudio/nerfstudio/models/splatfacto.py#L665-L671).
3. **L1 loss** — `mean(|gt − pred|)` — [splatfacto.py:673](../nerfstudio/nerfstudio/models/splatfacto.py#L673).
4. **SSIM loss** — `1 − SSIM(gt, pred)`; windowed structural similarity — [splatfacto.py:674](../nerfstudio/nerfstudio/models/splatfacto.py#L674).
5. **Combine** — `main_loss = (1 − ssim_lambda)·L1 + ssim_lambda·(1 − SSIM)`, default `ssim_lambda = 0.2` — [splatfacto.py:689](../nerfstudio/nerfstudio/models/splatfacto.py#L689).
6. **Scale regularization** (optional, only if `use_scale_regularization=True`, every 10 steps) — penalizes Gaussians with large max/min scale ratio (PhysGaussian) — [splatfacto.py:675-686](../nerfstudio/nerfstudio/models/splatfacto.py#L675-L686).
7. **MCMC regularizers** (only if `strategy="mcmc"`) — L1 on opacity and exp(scale) — [splatfacto.py:693-702](../nerfstudio/nerfstudio/models/splatfacto.py#L693-L702).
8. **Camera optimizer loss** (training only) — regularization on learned pose offsets — [splatfacto.py:704-706](../nerfstudio/nerfstudio/models/splatfacto.py#L704-L706).

Returns: `{"main_loss", "scale_reg", possibly "mcmc_opacity_reg", "mcmc_scale_reg", "camera_opt_*"}`.

### Splatfacto optional features (opt-in via config)

| Feature | Flag | Default | What it does |
|---|---|---|---|
| **Bilateral grid** | `use_bilateral_grid` | `False` | Per-image learnable color correction (exposure/WB drift) |
| **Camera-pose optimization** | `camera_optimizer.mode` | `"off"` | Learnable 6D pose offset per training image (`"SO3xR3"` or `"SE3"`) |
| **Antialiased rasterization** | `rasterize_mode` | `"classic"` | `"antialiased"` adjusts opacity to keep splats consistent across resolutions; reduces aliasing |
| **Scale regularization** | `use_scale_regularization` | `False` | Penalizes spiky/elongated Gaussians (max/min scale ratio > `max_gauss_ratio`); from PhysGaussian |
| **Absolute-gradient densification** | `use_absgrad` | `True` | Uses absolute screen-space grad instead of signed; densifies more aggressively |
| **MCMC strategy** | `strategy` | `"default"` | `"mcmc"` swaps clone/split/prune for Langevin-dynamics sampling; adds opacity + scale L1 regs |
| **Random init** | `random_init` | `False` | Init Gaussians in a random cube instead of SfM points |
| **Output depth during training** | `output_depth_during_training` | `False` | Render depth every train step (slower, enables depth losses) |
| **Color-corrected metrics** | `color_corrected_metrics` | `False` | Histogram match before PSNR/SSIM — fair comparison under color drift |
| **Background color** | `background_color` | `"random"` | `"random"` / `"black"` / `"white"`; randomization prevents memorizing a fixed bg |
| **Max Gaussian cap** | `max_gs_num` | `1_000_000` | Hard cap; densification stops past this |
| **SH degree schedule** | `sh_degree` + `sh_degree_interval` | `3`, every `1000` steps | Activates one extra SH band per interval — coarse-to-fine on color |
| **Resolution schedule** | `num_downscales` + `resolution_schedule` | `2`, every `3000` steps | Start at 1/4 res, double up to full — coarse-to-fine on image res |

In dynamic-gs, `camera_optimizer.mode="SO3xR3"` is overridden on in `DynamicGSModelConfig`.
