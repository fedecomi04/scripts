# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dynamic-gs** is a two-phase static + dynamic Gaussian Splatting system integrated with [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). It reconstructs and tracks dynamic objects (e.g., robot arms, manipulated objects) in scenes where most of the environment is static. It is designed for robotic teleoperation scenarios.

## Installation

```bash
# Install in development mode (from the scripts/ directory)
pip install -e .
```

After installation, `dynamic-gs` will be registered as a Nerfstudio method via the entry-point in `pyproject.toml`.

## Running

```bash
# Train from scratch
ns-train dynamic-gs --data /path/to/data_root

# Resume from checkpoint
ns-train dynamic-gs --data /path/to/data_root --load-dir /path/to/checkpoint
```

## Testing Utilities

Individual components can be tested with the scripts in `scripts/`:

```bash
# Test SAM3D 3D segmentation
python scripts/test_sam3d_single_object.py

# Test ESAM interactive mask queries
python scripts/test_esam_from_change_mask.py

# Test SAM3D fusion alignment
python scripts/test_probreg_sam3d_refine.py

# Visualize SAM3D outputs
python scripts/view_sam3d_output.py

# Build a presentation image from a completed dataset root
python scripts/generate_pipeline_presentation.py /path/to/dataset_root
```

`generate_pipeline_presentation.py` expects the dataset root path that contains both `static_scene/` and `dynamic_scene/`. It reads `static_scene/transforms.json`, `dynamic_scene/transforms.json`, initialization debug images, SAM3D artifacts, and dynamic tracker debug outputs to assemble a single presentation PNG.

## Architecture

### Three-Phase Training

Phase 0 is split into two non-contiguous halves so the static optimization
sees only the SfM seed scene, then the SAM3D Gaussians are inserted just
in time for the dynamic phase (this protects SAM3D's back-side Gaussians
from opacity erosion during static photometric optimization).

**Phase 0a — Pre-Static SAM3 + SAM3D Generation (optional, in `__init__`):**
When `use_sam3_graspable_prefusion=True` and `sam3_prompt_text` is non-empty,
SAM3 discovers graspable objects via text prompt on the first static image,
then SAM3D generates 3D Gaussians for each discovered mask in one subprocess
(one model load, sequential per-mask, full image + metric pointmap from
the static depth — see `dynamic_gs/utils/sam3d.py:run_sam3d_multi_object`).
Per-object PLYs and pose JSONs are written to disk; the Gaussian scene is
NOT mutated yet. Runs in `_run_sam3_and_sam3d_generation()` from `__init__`
after `_sync_phase(0)`. Result is stashed on `self._sam3d_generation_outputs`.

**Phase 1 — Static (steps 0 → static_num_steps − 1):** Optimizes the
standard Splatfacto Gaussian scene on the SfM seed. `means` LR is zeroed,
so static training updates appearance/orientation/opacity/scale but not
Gaussian positions. Gaussian refinement (densification/pruning) is
disabled via `NoRefineStrategy`. **No SAM3D objects are present in the
scene during this phase.**

**Phase 0b — Post-Static Fusion (at static→dynamic transition):** When
`_sync_phase` detects the boundary it calls `_fuse_sam3d_objects_into_scene(
self._sam3d_generation_outputs)` — for each pre-generated PLY: back-project
the SAM3 mask through the static depth → registration target, run
`register_and_fuse_sam3d_object` (rotation init from pose quaternion +
bbox-scale + centroid + voxel + probreg CPD similarity), insert Gaussians
via `insert_object_gaussians`, propagate `object_instance_ids` (each
object gets a unique 1..K id), reconstruct the Poisson mesh from the
Gaussian centers, **and eagerly construct a FoundationPoseTracker for that
instance** (paying its ~4 s nvdiffrast + model-weight + first-call CUDA JIT
cost here, while the boundary is otherwise idle). The trackers are stored
on the pipeline keyed by `instance_id` in `_fp_trackers_by_instance`.
`object_flags` stays 0 until D0 selection. The trained scene's render depth
gives `_get_existing_object_subset` a denser target than the SfM seeds
alone, so propagation is more robust than it would have been pre-static.

The fusion camera uses the **post-static optimized pose**: after 4000 static
steps the Nerfstudio `CameraOptimizer` has trained an SO3xR3 offset that
makes `transforms.json[frame_idx_0]` reproject the recorded image. We
build a single-frame `Cameras` slice, set `metadata["cam_idx"]`, call
`self.model.camera_optimizer.apply_to_camera(camera).detach()` directly
(bypassing the model's `_should_apply_camera_optimizer` phase guard which
returns False here because `phase` already flipped to `"dynamic"` when
this runs), and overwrite `camera.camera_to_worlds` with the optimized
c2w. Downstream `get_outputs`, `_backproject_mask_to_world`, and the
c2w extraction inside `register_and_fuse_sam3d_object` then all see the
optimized pose, which slightly improves reprojection alignment between
the SAM3D object cloud and the static-depth registration target.

**Phase 2 — Dynamic (from step `static_num_steps`, `dynamic_steps_per_frame`
steps/frame):** Per-frame optimization of the dynamic sequence. The phase
transition resets the `means` optimizer state and scheduler. `means`
gradients are masked so object Gaussians are moved by rigid transforms,
while non-object Gaussians can still be optimized by scene optimization.

- **Frame 0 bootstrap (Path A — prefused):** `prepare_dynamic_update(skip_object_flags_write=True)` → select moved object from pre-fused candidates via ESAM overlap → set `object_flags = (object_instance_ids == selected_k).float()` + `_persistent_object_membership_ready = True` → rendered object mask → **pick the pre-built FoundationPose tracker for the selected instance** from `_fp_trackers_by_instance`, drop the rest, seed `est.pose_last` from the known scene pose (no construction cost here — Phase 0b paid it) → CD0. No SAM3D generation/insertion at D0; no `register()` call (skipped because the initial pose is already known from Phase 0b fusion). No SAM2 — the projected object Gaussians give the authoritative object mask.
- **Frame 0 bootstrap (Path B — no prefusion):** `prepare_dynamic_update` (render RS + change mask + ESAM on render/live + Gaussian flagging) → SAM3D 3D generation → CPD-based fusion/insertion → render object mask → FoundationPose `register()` fallback on D0 live image/depth/mask (no manifest entry) → capture reference object pose → compute CD0
- **Frame N tracking:** FoundationPose `track_one(rgb, depth, K, ...)` from the cached `pose_last` (FP refines its previous pose against the new RGB-D — i.e., **consecutive-frame tracking** internally; the convert-to-D0-reference is only the output convention) → convert mesh-to-camera 4×4 to world-frame absolute SE(3) from the D0 reference pose → apply via `apply_rigid_object_transform_from_reference` → render RDN → render object mask → compute CDN. **The CDN's object exclusion comes from the projected object-Gaussian mask alone** (no SAM2 propagation; FoundationPose handles per-frame pose, so the Gaussian projection is the authoritative "where the object is" signal).

### Phase Transition Details

`_sync_phase(step)` is called at the start of every `get_train_loss_dict`. It:
1. Detects phase changes (static → dynamic) and calls `model.set_phase(phase)`.
2. **At the static→dynamic boundary, runs Phase 0b fusion** — calls `_fuse_sam3d_objects_into_scene(self._sam3d_generation_outputs)` if generation produced objects pre-static. This insertion happens AFTER `model.set_phase("dynamic")` so the scene-opt + means-grad hooks just registered are then re-bound by `insert_object_gaussians` → `_refresh_gaussian_optimizers` onto the resized parameter tensors. The hooks read `object_flags` at backward time, so writing `object_flags` later in D0 Path A still engages them correctly.
3. Detects frame transitions within the dynamic phase and calls `_prepare_dynamic_frame()`.
4. `model.set_phase("dynamic")` sets `requires_grad` on all Gaussian parameter groups, registers scene optimization gradient hooks on features/opacities/scales/quats, and resets the `means` optimizer + scheduler.
5. The `max_num_iterations` in the Nerfstudio trainer is updated to `static_num_steps + total_dynamic_frames * dynamic_steps_per_frame` at callback time.

### Gradient Masking Strategy

During the dynamic phase, gradients are gated **twice**: once per pixel by the masked loss, and once per Gaussian by the `scene_opt_active_mask` buffer. Both gates must let a Gaussian through for it to receive a non-zero gradient.

- **Per-pixel: masked loss.** `_masked_rgb_l1` and `masked_l1_depth_loss` multiply `(pred - gt)` by the change mask CDN before reducing. Pixels outside CDN contribute 0 to the loss, so any Gaussian whose 2D footprint lies entirely outside CDN gets a zero gradient signal from the loss alone.
- **Per-Gaussian: `scene_opt_active_mask`.** A non-persistent bool buffer (N,) on the model that holds, for each Gaussian, "this Gaussian's 2D footprint overlaps the change region AND it is not flagged as belonging to the moved object". The buffer is refreshed once per dynamic frame by `model.update_scene_opt_active_mask(cdn)` (called by the pipeline in both `_prepare_frame_0` after CD0 and `_prepare_frame_n` after CDN). Both `_mask_means_grad` and the scene-opt parameter hooks (registered on `features_dc`, `features_rest`, `opacities`, `scales`, `quats`) read this buffer and zero gradients for any Gaussian outside it. This is what stops a Gaussian whose footprint crosses CDN pixels from being pulled by the masked loss when its actual 3D position is far behind the change region.
- **Object Gaussians** are always excluded from `scene_opt_active_mask` (the `& ~object_flags` clause), so they receive zero gradients on every parameter group. They move exclusively via `apply_rigid_object_transform_from_reference` driven by FoundationPose.

### Core Class Hierarchy

```
DynamicGSModel          (dynamic_gs_model.py)
  └─ extends SplatfactoModel
     - object_flags: (N,1) float persistent buffer — 1.0 for active dynamic object Gaussians, 0.0 for scene
     - object_instance_ids: (N,1) long persistent buffer — 0=scene, 1..K=Phase 0 pre-fused object ID
     - sam3d_init_target_flags: (N,1) persistent buffer — flags added by SAM3D
     - scene_opt_active_mask: (N,) non-persistent bool buffer — per-Gaussian "footprint overlaps CDN AND not object". Refreshed each frame by `update_scene_opt_active_mask(cdn)`. Read by `_mask_means_grad` and the scene-opt parameter hooks to gate gradients per Gaussian.
     - change_mask_image: (H,W,1) non-persistent buffer — current frame's CDN mask
     - _mask_means_grad: backward hook filtering means gradients via `scene_opt_active_mask`
     - apply_rigid_object_transform_from_reference: applies the current rigid transform to the stored reference object pose
     - capture_reference_object_pose: stores object means/quats once at D0
     - initialize_object_from_sam3d: bbox scale + CPD similarity registration + Gaussian insertion
     - prepare_dynamic_update: ESAM segmentation + change mask + flag Gaussians (skip_object_flags_write param)
     - render_object_mask: rasterize only object Gaussians using gsplat rasterization
     - refresh_dynamic_state_after_insertion: re-flags Gaussians after SAM3D insertion
     - _propagate_instance_membership: Phase 0 per-object identity propagation (writes object_instance_ids)
     - _build_persistent_object_membership: D0 binary object membership (writes object_flags — NOT for Phase 0)

DynamicGSPipeline       (dynamic_gs_pipeline.py)
  └─ extends VanillaPipeline
     - _timing: defaultdict(list) — per-step timing accumulator
     - _fp_tracker: FoundationPoseTracker instance for the moved object (set at D0 by picking from `_fp_trackers_by_instance`)
     - _fp_trackers_by_instance: dict[int, FoundationPoseTracker] — one tracker per Phase 0b instance, built eagerly during fusion. D0 picks one by `instance_id` and discards the rest so their nvdiffrast contexts + model weights are eligible for GC.
     - _d0_selected_instance_id: int — the Phase 0b instance id chosen at D0 Path A (used to look up the pre-built tracker)
     - _sam3d_generation_outputs: dict|None — {sam3_objects, sam3d_results} stash
                                  written by Phase 0a, consumed by Phase 0b
     - _run_sam3_and_sam3d_generation(): Phase 0a — SAM3 segmentation + SAM3D
                                          multi-object generation (full image
                                          + metric pointmap), called from __init__
     - _fuse_sam3d_objects_into_scene(): Phase 0b — register + insert + propagate
                                         instance ids, called from _sync_phase at
                                         the static→dynamic boundary
     - _prepare_frame_0(): D0 bootstrap with Path A (prefused) / Path B (old SAM3D) branching
     - _prepare_frame_n(): absolute-reference dynamic update sequence
     - _compute_change_mask(): MSSIM change detection excluding gripper+object
     - _print_timing_summary(): logs full pipeline timing at last step
     - _write_timing_report(): writes <data_root>/timing_report.txt at exit

DynamicGSDataManager    (dynamic_gs_datamanager.py)
  └─ Wraps two FullImageDatamanager instances
     - static_manager: FullImageDatamanager for static_scene/ (loads 3D points)
     - dynamic_manager: DynamicFrameFullImageDatamanager for dynamic_scene/ (loads depth)
     - DynamicFrameDataset: extends InputDataset with uint16 depth loading (scale 1e-3)
     - set_phase(): switches active_manager, train_dataset, eval_dataset
     - set_dynamic_frame_idx(): pins manager to a specific frame
     - next_train/next_eval: return pinned frame during dynamic phase
```

### Utility Modules (`dynamic_gs/utils/`)

| Module | Role |
|--------|------|
| `active_mask.py` | Change detection from RGB/depth deltas; MSSIM-based `build_change_mask`; morphological filtering; `keep_largest_component_with_min_area` combines the prior `remove_small_components` + `keep_largest_component` into a single scipy.label CPU round-trip |
| `foundationpose_tracker.py` | `FoundationPoseTracker`: wraps NVIDIA FoundationPose (`third_party/FoundationPose/`); takes a triangle mesh + initial mesh-to-world; seeds `est.pose_last` from the known scene pose at D0 (skips `register()`); `track_one` per frame; converts mesh-to-camera output to world-frame absolute (R, t) for `apply_rigid_object_transform_from_reference` |
| `sam3_segmentation.py` | SAM3 text-prompted segmentation: worker (`run_sam3_segmentation`) runs in `sam3_dynamic_gs` conda env; subprocess launcher (`run_sam3_subprocess`) uses `conda run`; mask filtering (area, border, IoU dedup) |
| `sam3d.py` | SAM3D subprocess invocation: single-object (`run_sam3d_single_object_subprocess`, uses cropping), multi-object (`run_sam3d_multi_object_subprocess`, no cropping, one model load); output path management; pose sidecar save/resolve/validation |
| `sam3d_fusion.py` | SAM3D rotation-aware object initialization: apply SAM3D quaternion in camera/world frame, then current bbox scale + centroid alignment + voxel downsampling + probreg CPD similarity; Gaussian insertion and deduplication via `register_and_fuse_sam3d_object` |
| `sam2.py` | **No longer wired into the runtime.** Originally provided a SAM2 video predictor + `query_sam2_propagated_mask` for pairwise live-mask propagation D(N-1)→DN; replaced by FoundationPose pose tracking + the projected object-Gaussian mask. File is kept on disk for reference but is not imported by `dynamic_gs_pipeline.py` / `dynamic_gs_model.py`, and `utils/__init__.py` does not re-export it. |
| `esam.py` | ESAM interactive object mask query (`query_esam_mask`, `query_esam_mask_pair`); `build_esam_ti` model builder; input downsampled to `ESAM_MAX_SIDE=512` and a single forward pass (no inner convergence loop) |
| `depth_loss.py` | `masked_l1_depth_loss`: per-pixel L1 depth loss masked to valid regions |
| `rigid_regularization.py` | `rigid_or_static_loss`: Kabsch-based rigid body consistency loss; promotes coherent group motion |
| `no_refine_strategy.py` | `NoRefineStrategy`: disables gsplat's default Gaussian refinement (densification + pruning) |

### Key Configuration (`dynamic_gs_config.py`)

```python
STATIC_NUM_STEPS = 4000
DYNAMIC_STEPS_PER_FRAME = 50
```

Pipeline-level config in `DynamicGSPipelineConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `static_num_steps` | 4000 | Number of static training steps before the dynamic phase |
| `dynamic_steps_per_frame` | 50 | Optimization steps per dynamic frame |
| `save_debug_images` | True | Gates per-frame debug PNG saves (D0.1f / D0.9 / DN.8). Set False to remove ~210ms from D0 and ~600ms from each dynamic frame; SAM3D Path B inputs (`render.png`, `render_object_mask_binary.png`) are still saved when needed |

Key model parameters in `DynamicGSModelConfig` (inherits from `SplatfactoModelConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_lambda` | 0.4 | Depth supervision loss weight |
| `change_mask_depth_threshold` | 0.02 | Meters — pixel flagged as changed if depth delta > threshold |
| `change_mask_rgb_threshold` | 0.10 | Per-pixel MS-SSIM dissimilarity threshold for RGB change detection. Higher = less sensitive (drops marginal-difference pixels), lower = more sensitive. Plumbed through `_compute_change_mask` and `prepare_dynamic_update` to `build_change_mask`; default fallback when no value is passed is `OFFICIAL_RGB_MSSSIM_THRESHOLD` in `active_mask.py`. |
| `change_mask_use_rgb` | False | Use RGB (not just depth) for change mask |
| `change_mask_blur_kernel_size` | 5 | Gaussian blur kernel for change mask smoothing |
| `change_mask_filter_radius` | 1 | Morphological filter radius on change mask |
| `change_mask_min_component_size` | 64 | Min connected component size (pixels) to keep |
| `active_mask_dilate_radius` | 0 | Extra dilation on final change mask |
| `object_mask_dilate_px` | 1 | Dilation on the rendered object mask |
| `rigid_static_lambda` | 0.1 | Weight for rigid body regularization loss |
| `use_sam3d_object_init` | True | Generate SAM3D object cloud at frame 0 |
| `reuse_sam3d_generated_ply` | True | Skip SAM3D subprocess if PLY already exists (set False for fresh generation) |
| `use_simulator_background` | True | Render/composite against the Gazebo background color |
| `simulator_background_rgb` | (0.86, 0.92, 1.0) | Gazebo background color used by the model background override |
| `camera_optimizer.mode` | `SO3xR3` | Enable Nerfstudio camera-pose optimization during training |
| `enable_dynamic_mean_optimization` | False | Alternative path. When True with `enable_scene_optimization=False` and `enable_fp_rigid_motion=False`, allows gradient descent on object means using `current_active_mask` as the per-Gaussian gate. Off in the default runtime. |
| `enable_fp_rigid_motion` | True | Apply FoundationPose 6D rigid transform to object Gaussians |
| `fp_init_refine_iter` | 6 | FoundationPose refinement iterations on D0 (after seeding `pose_last` from the known scene pose) |
| `fp_track_refine_iter` | 2 | FoundationPose refinement iterations per frame N≥1 |
| `fp_mesh_unit_scale` | 1.0 | Scale factor applied to the mesh on load; default 1.0 since the post-CPD `mesh_to_world_4x4` already carries the scale |
| `enable_scene_optimization` | True | Continue optimizing scene Gaussians during dynamic phase |
| `scene_opt_refine_every` | 100 | Steps between scene Gaussian densification/pruning in dynamic phase |
| `scene_opt_densify_grad_thresh` | 0.0002 | Grad norm threshold for densification |
| `scene_opt_cull_alpha_thresh` | 0.1 | Alpha threshold for pruning |
| `use_sam3_graspable_prefusion` | True | Enable Phase 0 pre-static SAM3 object discovery and fusion |
| `sam3_prompt_text` | "" | Text prompt for SAM3 segmentation (empty = Phase 0 skipped) |
| `sam3_conda_env_name` | "sam3_dynamic_gs" | Conda env for SAM3 subprocess (Python 3.12+, PyTorch 2.7+, CUDA 12.6+) |
| `sam3_candidate_min_area_ratio` | 0.002 | Min mask area as fraction of image area (0.2%) |
| `sam3_candidate_max_area_ratio` | 0.25 | Max mask area as fraction of image area (25%) |
| `sam3_candidate_dedup_iou` | 0.6 | IoU threshold for deduplicating overlapping SAM3 masks |
| `sam3_candidate_max_objects` | 8 | Max objects to keep from SAM3 |
| `sam3_reuse_cached` | True | Reuse cached SAM3/SAM3D outputs if they exist |

### Per-Object Identity Buffers

| Buffer | Type | Purpose | When set |
|--------|------|---------|----------|
| `object_instance_ids` | `torch.long` (N,1) | Persistent multi-object identity from Phase 0 | Pre-static fusion: `instance_id=k` (1..K) |
| `object_flags` | `float` (N,1) | Single active dynamic object indicator | D0 selection: `(object_instance_ids == selected_k).float()` |
| `sam3d_init_target_flags` | `float` (N,1) | SfM Gaussians targeted by old D0 SAM3D insertion | D0 Path B only |

- `object_instance_ids` is set during Phase 0 via `_propagate_instance_membership()`. It survives `load_state_dict`, `_resize_dynamic_buffers`, and the prune path.
- `object_flags` is set at D0 either by `prepare_dynamic_update()` (Path B) or by the pipeline's moved-object selection (Path A).
- `_propagate_instance_membership()` writes to `object_instance_ids` for unassigned Gaussians only; `_build_persistent_object_membership()` writes to `object_flags` and is NOT used for Phase 0.
- `prepare_dynamic_update()` accepts `skip_object_flags_write=True` to prevent overwriting `object_flags` when pre-fused objects exist. The pipeline sets `_persistent_object_membership_ready = True` after D0 object selection in Path A.

### Optimizer Groups

Seven Adam optimizer groups are defined in `dynamic_gs_config.py`:

| Group | LR | Active in static | Active in dynamic |
|-------|-----|------------------|-------------------|
| `means` | 1.6e-4 | No | Yes (object only via hook) |
| `features_dc` | 0.0025 | Yes | Yes (scene only via hook) |
| `features_rest` | 0.0025/20 | Yes | Yes (scene only via hook) |
| `opacities` | 0.05 | Yes | Yes (scene only via hook) |
| `scales` | 0.005 | Yes | Yes (scene only via hook) |
| `quats` | 0.001 | Yes | Yes (scene only via hook) |
| `camera_opt` | 1e-3 | Yes | Yes |

LR is zeroed (not disabled) for inactive groups. When transitioning to the dynamic phase, the `means` optimizer state and scheduler are reset so it restarts from initial LR.

### FoundationPose 6D Tracker Detail

`FoundationPoseTracker` (in `dynamic_gs/utils/foundationpose_tracker.py`) wraps NVIDIA's [FoundationPose](https://github.com/NVlabs/FoundationPose) (cloned at `third_party/FoundationPose/`). One tracker instance per moved object.

**Inputs at construction:**
- `mesh_path`: triangle mesh PLY produced by SAM3D's mesh decoder (Phase 0a writes `<stem>_sam3d_mesh.ply` next to the existing `<stem>_sam3d_raw_output.ply`).
- `mesh_to_world`: 4×4 post-CPD similarity transform persisted in `phase0_manifest.json` under key `mesh_to_world_4x4` (Phase 0b extension).

**`initialize_from_known_pose(rgb, depth, K, camera_to_world, refine_iterations)`**: Skips FP's `register()` entirely. Computes `mesh_to_camera = inv(camera_to_world) @ mesh_to_world`, then `est.pose_last = mesh_to_camera @ inv(get_tf_to_centered_mesh())` (FP keeps `pose_last` in centered-mesh frame). Runs `track_one` for `refine_iterations` to settle the pose against the actual D0 observation. Logs `[FP] initialized from known scene pose`.

**`track_one(rgb, depth, K, camera_to_world, iterations)`** per frame ≥1: calls FP's `est.track_one(...)` (no register), receives mesh-to-camera 4×4. Converts back to world-frame absolute SE(3) from the D0 reference: `delta_world = (camera_to_world @ pose_in_camera) @ inv(mesh_to_world_init)`. Returns `(R, t)` ready for `apply_rigid_object_transform_from_reference`.

**`fallback_register(rgb, depth, K, ob_mask, camera_to_world, refine_iterations)`**: Exception path used only when the manifest lacks `mesh_to_world_4x4` (legacy datasets). Runs FP's standard `register()` with a 2D mask, then captures the resulting mesh-to-world as the new D0 reference.

**Camera convention conversion:** Nerfstudio cameras are y-up / z-back (OpenGL); FP / OpenCV are y-down / z-forward. Pipeline applies `cv_c2w = ns_c2w @ diag(1, -1, -1, 1)` before passing to the tracker. Depth is meters float32; the datamanager applies `depth_unit_scale_factor=1e-3` so `batch["depth_image"]` is already in meters.

**Frame-0 register skip:** the FP `register()` call is the slowest path (multi-iteration global pose search). By seeding `pose_last` directly from the known scene pose, we skip ~5 seconds of D0 cost AND avoid needing a 2D segmentation mask for first-frame registration.

### Dynamic Tracking Roadmap (open work)

The dynamic phase has two conceptually independent jobs and they should
be treated separately when optimizing:

1. **Object tracking — per-frame, must be real-time.** Tracking now uses
   FoundationPose (`DN.3_fp_track`) with `track_refine_iter=2` per frame.
   FP holds `pose_last` between calls and only refines the previous estimate
   against the new RGB-D, so per-frame cost is dominated by the refiner +
   scorer forward. Possible knobs: `fp_track_refine_iter` (default 2 — try
   1 if drift stays bounded), or skip `track_one` on frames where the
   live RGB-D hasn't moved enough.
2. **Optimization of unseen parts via change detection — currently a
   hard-coded "if change detected, optimize for X epochs" rule.** Not
   yet real-time-bound. The user's plan is to redesign this later;
   today the change mask CDN simply gates the next 50 training-step
   loss masks (`dynamic_steps_per_frame=50`), regardless of whether
   the change is small enough to be ignored or large enough to need
   more steps. Do not optimize this until the user's redesign is
   specified.

These two paths share `DN.7_change_mask_cdn` (≈240 ms/frame) — the MSSIM
compute is needed both to seed the optimization and (potentially) to
gate it. The change-detection optimizations already shipped in D0
(`build_change_mask` cleanup collapsed to one scipy.label, ESAM
batched/downsampled) flow through to DN automatically; the remaining
cost here is the multiscale SSIM convolutions themselves.

### SAM3D Integration

The runtime config (`pipeline_runtime_small.yaml`, written at runtime by
`_write_runtime_config()` in `dynamic_gs/utils/sam3d.py`) now loads
**both** `slat_decoder_gs` (Gaussian splat) and `slat_decoder_mesh`
(triangle mesh) with `decode_formats=["gaussian", "mesh"]`. Per object,
`run_sam3d_multi_object` writes:
- `<stem>_sam3d_raw_output.ply` — Gaussian splat (used by Phase 0b CPD
  registration)
- `<stem>_sam3d_mesh.ply` — triangle mesh (used by FoundationPose at D0)
- `<stem>_sam3d_pose.json` — SAM3D rotation/translation/scale prior

The mesh PLY is required for the FP tracker. The Gaussian splat PLY
remains the registration source for fusion; both are produced from the
same SLAT in a single inference pass.

Phase 0b's `_fuse_sam3d_objects_into_scene` now extends each
`phase0_manifest.json` entry with:
- `mesh_path`: absolute path to the mesh PLY
- `mesh_to_world_4x4`: post-CPD `Sam3DInsertionResult.similarity_transform`
  (4×4 list of lists)

These two fields are what the FP tracker reads at D0 to skip `register()`.

`initialize_object_from_sam3d` in the model:

1. Saves the rendered RGB and live object mask to disk (required as SAM3D inputs).
2. Calls `run_sam3d_single_object_subprocess` — launches a separate Python process running SAM3D (`sam-3d-objects`) which generates a Gaussian `.ply` file.
3. `register_and_fuse_sam3d_object` loads the PLY and the matching SAM3D pose sidecar, applies only the SAM3D rotation as the source orientation prior, then uses the current pipeline's bbox scale + centroid translation + voxel downsampling + probreg CPD similarity, and appends only non-overlapping Gaussians to the existing set.
4. `refresh_dynamic_state_after_insertion` re-runs Gaussian flagging to label the newly inserted points as object members.

Key parameters: `reuse_sam3d_generated_ply=True` (default) skips the subprocess only when both the cached PLY and a valid matching pose sidecar with `rotation` exist. Set to `False` when you need fresh generation (e.g., for accurate timing measurements or after changing the object). If `{frame_name}_d0_true_sam3d_raw_output.ply` exists in the debug directory, the pipeline prefers that file as the SAM3D source for insertion only when its matching pose sidecar is also valid.

SAM3D checkpoints live in `third_party/sam-3d-objects/checkpoints/hf/`. The pipeline uses `pipeline_runtime_small.yaml` which requires only: `ss_generator`, `slat_generator`, `ss_decoder`, `slat_decoder_gs`. The mesh and GS4 decoders are null and their checkpoints are not needed.

### Change Mask Computation

`_compute_change_mask` (pipeline) and `build_change_mask` (utils):

1. Computes an RGB MS-SSIM change score between the render and live image.
2. Thresholds that score using the current official threshold/cleanup recipe in `active_mask.py`.
3. Re-applies the valid mask from the dataset batch, and excludes gripper/object regions through the `valid_mask` passed in by the pipeline/model.
4. Applies the configured cleanup recipe and optional dilation.

The resulting mask (CDN) is stored in `model.change_mask_image` and used as the optimization mask for the next 50 training steps of that frame.

### Debug Outputs

Initialization images are saved to `<data_root>/dynamic_scene/initialization_debug/`.
Initialization non-image SAM3D artifacts are saved to `<data_root>/dynamic_scene/initialization_artifacts/`.
Post-initialization dynamic debug remains under `<data_root>/dynamic_scene/debug/`.

| File | Content |
|------|---------|
| `{frame}_render.png` | Rendered RGB at static phase end (RS) |
| `{frame}_live_input.png` | First dynamic RGB image used during initialization |
| `{frame}_live.png` | Live camera RGB |
| `{frame}_render_depth.png` / `{frame}_live_depth.png` | Depth visualizations saved for initialization review |
| `{frame}_rs00.png` | Re-rendered RGB after SAM3D insertion |
| `{frame}_rdn.png` | Re-rendered RGB after rigid transform (frame N) |
| `{frame}_change_mask.png` | Raw change mask from ESAM step |
| `{frame}_render_object_mask.png` | Rendered Gaussian object mask with sampled prompt points |
| `{frame}_render_object_mask_binary.png` | Clean rendered object mask used by SAM3D |
| `{frame}_live_object_mask.png` | ESAM live object mask with sampled prompt points |
| `{frame}_live_object_mask_binary.png` | Clean live object mask |
| `{frame}_optim_mask.png` | Combined optimization mask |
| `{frame}_cd0.png` / `{frame}_cdn.png` | Final change mask used for optimization |
| `{frame}_render_w_cd0.png` | Render with CDN overlay (red) |
| `{frame}_live_w_cdn.png` | Live with CDN overlay (red) |
| `{frame}_render_w_objmask.png` | Render with projected object-Gaussian mask overlay (blue) |
| `{frame}_fp_motion.txt` | FoundationPose rigid transform stats: R, t (world-frame absolute from D0 reference) |
| `{frame}_sam3d_preview.png` / `{frame}_sam3d_crop_*.png` | Initialization images given to SAM3D |
| `{frame}_sam3d_*.ply` | SAM3D-generated object point clouds (`initialization_artifacts/`) |

### Timing Profile

Timing is instrumented in `get_train_loss_dict`, `_run_sam3_and_sam3d_generation`,
`_fuse_sam3d_objects_into_scene`, and `_prepare_frame_{0,n}` using
`self._timing` (a `defaultdict(list)`). The exact numbers depend on whether
SAM3D generation is cached, the static-step budget, whether debug images
are saved, and GPU model. **Always check the generated
`<data_root>/timing_report.txt` from the actual run.**

#### Reference numbers (4-object scene, 4000 static steps + ~119 dynamic frames × 50 steps, 8 GiB GPU)

The values below are from a representative run **after the change-detection
optimizations** (ESAM pre-warm + downsample to 512 + batched render/live
forward + bbox-restricted EDT + combined scipy.label cleanup) and
**after the optimized-camera-pose fusion fix**. The numbers will shift
once the dynamic-phase tracking / scene-opt work lands, so always check
the generated `<data_root>/timing_report.txt` from the actual run.

| Phase | Total | Notes |
|---|---|---|
| Phase 0a generation | reported as `S0.4a_generation_total` | Cached on re-runs (`sam3_reuse_cached=True`) |
| Phase 0b fusion | reported as `S0.4b_fusion_total` | Per-object CPD, runs at static→dynamic boundary |
| Phase 1 — static training | ≈12 ms/step × 4000 ≈ 49 s | |
| Phase 2 — D0 bootstrap | ≈1.2 s | Down from ≈3.6 s pre-optimization |
| Phase 3 — dynamic loop | per-frame prep + training, see below | |

Within Phase 0 (uncached, 4-object scene):

| Substep | Time | Notes |
|---|---|---|
| `S0.2_sam3d_multi_generation` | ≈380 s | Dominant. Cached on re-runs |
| `S0.3_fusion_obj_*` | 1.8 s — 58 s/object | Probreg CPD; scales with point count |
| `S0.3_mesh_recon_obj_*` | ≈ a few s/object | Open3D Poisson reconstruction from gaussian centers |
| `S0.3_fp_construct_obj_*` | ≈4 s/object | Eager FP construction (paid here so D0.6 stays fast) |
| `S0.1_sam3_segmentation` | ≈0 s when cached | |

D0 bootstrap (Path A, post-optimization):

| Substep | Time |
|---|---|
| `D0.1_initial_change_detection` | ≈0.29 s (was ≈0.70 s) |
|   `D0.1a_forward_render` | ≈0.03 s |
|   `D0.1b_change_mask` | ≈0.09 s (single scipy.label) |
|   `D0.1c_esam_render` | ≈0.16 s (batched render+live in one forward) |
|   `D0.1d_esam_live` | 0.00 s (batched into D0.1c) |
|   `D0.1e_gaussian_flagging` | ≈0.01 s |
|   `D0.1f_post_save` | ≈0.21 s with debug saves on, ≈0 s off |
| `D0.6_fp_init` | ≈milliseconds — picks the pre-built tracker from `_fp_trackers_by_instance` (constructed in Phase 0b) and seeds `pose_last`. Set `fp_init_refine_iter>0` to add per-iteration `track_one` cost on D0. |

Per-frame averages in the dynamic loop (DN, with default `save_debug_images=True`):

| Substep | Time/frame | Notes |
|---|---|---|
| `DN.8_debug_images` | ≈600 ms | Disk I/O — set `save_debug_images=False` to drop to ≈0 |
| `DN.3_fp_track` | ≈64 ms | FoundationPose `track_one` (refiner forward, no RANSAC) |
| `DN.7_change_mask_cdn` | ≈240 ms | MSSIM + cleanup; uses `_compute_change_mask` (pipeline) which still calls the full `build_change_mask` recipe |
| `DN.5_render_rdn` | ≈15 ms | gsplat rasterize |
| `DN.6_render_object_mask` | ≈3 ms | object-only rasterize |

(`DN.1_sam2_live_propagation` no longer exists — SAM2 was removed; the
projected object-Gaussian mask carries the object exclusion alone.)

Per-frame dynamic training step (50 steps/frame):

| Substep | Time/step |
|---|---|
| `DT.1 dynamic_step` | ≈58 ms |

**Hotspot summary (open work — see "Dynamic Tracking Roadmap" below):**

1. **`DN.3_fp_track`** — FoundationPose `track_one` (refiner forward at `fp_track_refine_iter=2`). FP is stateful (`pose_last` is cached and refined per frame), so there's no per-frame D0 encoder cost or RANSAC step. Per-frame cost is dominated by the refiner + scorer forward.
2. **`DN.7_change_mask_cdn`** ≈240 ms — MSSIM compute. Cleanup recipe is already collapsed to one scipy.label, but the 3-level multiscale SSIM convolutions still dominate.
3. **`DN.8_debug_images`** ≈600 ms — pure disk I/O. Already gated by `save_debug_images=False` for production runs.
4. **`S0.2_sam3d_multi_generation`** ≈380 s for 4 objects — dominated by SAM3D inference, cached on re-runs.

#### Known timing instrumentation notes

These are NOT bugs — they explain how the timing report is structured.

1. **Phase 0 is reported as separate `S0.4a_generation_total` (Phase 0a,
   runs in `__init__`) and `S0.4b_fusion_total` (Phase 0b, runs at the
   static→dynamic boundary).** The report shows both totals plus a
   combined "Phase total" header. The two halves run minutes apart on
   the wall clock, so lumping them together was misleading.

2. **`DN.2`, `DN.4` are not present.** The CoTracker-era reseed/post-filter
   stages have no analog in FP — FP is stateful and refines `pose_last` in place.

3. **ESAM is pre-warmed in `DynamicGSPipeline.__init__`** with two
   dummy forwards (batch=1 and batch=2 at 512×512) so both the ~300 ms
   one-time model load and the CUDA kernel JIT for both batch shapes
   are paid before training starts, not inside `D0.1c_esam_render`.
   `query_esam_mask` and `query_esam_mask_pair` also: (a) downsample
   the input to a max side of `ESAM_MAX_SIDE = 512` (encoder cost is
   quadratic in input area), (b) run a single ESAM forward (no inner
   convergence loop), and (c) restrict `compute_prompt_interior`'s
   `distance_transform_edt` to the prompt mask's bounding box.

4. **Render and live ESAM are batched.** `prepare_dynamic_update` calls
   `query_esam_mask_pair` once: it computes the prompt-interior EDT +
   sample points once (the prompt is shared) and forwards a stacked
   `(2,3,H,W)` image batch. The combined cost lands under
   `D0.1c_esam_render`; `D0.1d_esam_live` is 0 in this path. (The
   previous "mixed / SAM2 fallback" branch was removed when SAM2 was
   deactivated — the bootstrap path now has no fallback.)

5. **`D0.1b_change_mask`** uses `keep_largest_component_with_min_area`,
   a single scipy.label CPU round-trip that replaces the prior
   `remove_small_components` (one scipy.label) + outer
   `keep_largest_component` (another scipy.label) sequence. The outer
   `keep_largest_component` in `prepare_dynamic_update` only runs when
   `active_mask_dilate_radius > 0` (default 0), since dilation can fuse
   previously-separate components and require relabeling.

6. **`save_debug_images=False`** removes the `D0.1f_post_save`,
   `D0.9_debug_images`, and `DN.8_debug_images` saves entirely. The
   SAM3D Path B branch still saves the two SAM3D-required files
   (`render.png`, `render_object_mask_binary.png`) when needed.

## Pipeline Step-by-Step Reference (Corrected + Timed)

### Symbol Glossary

| Symbol | Meaning |
|--------|---------|
| RS | Rendered static scene (fully trained, no object inserted) |
| DN | N-th image from the dynamic dataset (live camera) |
| MN | Gripper/background mask from transforms.json for DN |
| RS00 | Render of static scene + inserted SAM3D object, from D0's camera pose |
| RDN | Render from DN's pose, after rigid transform applied to flagged Gaussians |
| F0_render | ESAM object mask queried on RS (rendered image, frame 0 only — Path B and overlap selection only) |
| F0_live | ESAM object mask queried on D0 (live image, frame 0 only — Path B and overlap selection only) |
| CDN | Change mask between RDN and DN, excluding the projected object-Gaussian region + gripper |

### Phase 1 — Static Training

From an initial SfM pointcloud, optimize all Gaussian parameters except `means` (LR=0). No densification/pruning (`NoRefineStrategy`). The current runtime config uses 4000 static steps before the dynamic bootstrap starts.

### Phase 2 — Dynamic Frame 0 Bootstrap

#### D0.1 Initial change detection (≈0.29 s total, was ≈0.70 s)

1. **D0.1a** Forward render (≈0.03 s): call `get_outputs(camera)` in eval mode → RS image + depth
2. **D0.1b** Change mask (≈0.09 s): `build_change_mask(...)` — RGB MS-SSIM + single-scipy.label cleanup → C0
3. **D0.1c** ESAM on render+live in one batched forward (≈0.16 s): `query_esam_mask_pair(esam, RS_rgb, D0_rgb, C0)` → (F0_render, F0_live). The change mask C0 is the shared prompt for both, so EDT + interior point sampling runs once. ESAM is pre-warmed at pipeline init, and the input is downsampled to `ESAM_MAX_SIDE=512` before the forward; the output mask is upsampled with nearest-neighbor.
4. **D0.1d** ESAM on D0 live: 0 s — folded into D0.1c since the bootstrap is always batched.
5. **D0.1e** Gaussian flagging (≈0.01 s): project ~300K+ Gaussian centers to 2D (`extract_projected_centers_and_radii`), check which fall inside union(F0_render, F0_live) (`build_active_mask`) → set `object_flags = 1` for object Gaussians (suppressed via `skip_object_flags_write=True` in Path A; the pipeline writes the per-instance flags itself)

#### D0.2 SAM3D object generation

- Subprocess: launch `sam-3d-objects` with RS image + F0_live mask → generates a Gaussian `.ply` file
- One-time cost; skip on re-runs with `reuse_sam3d_generated_ply=True`

#### D0.3 SAM3D insertion: bbox scale + CPD similarity + dedup

- Load generated PLY, register against existing object cloud: bbox scale init → centroid alignment → voxel downsample → probreg CPD similarity refinement → dedup
- Deduplicate overlapping Gaussians → append to `means`, `features_dc`, etc.
- Mark newly inserted Gaussians as `object_flags = 1`

#### D0.4 Render object mask

- `render_object_mask(camera)`: rasterize only `object_flags > 0.5` Gaussians, threshold + dilate
- **Simulation-based**, NOT a new ESAM query on RS00. There is no F00.

(D0.5 is no longer present — the SAM2 live tracker has been removed. The
projected object-Gaussian mask alone now drives the change-mask exclusion.)

#### D0.6 FoundationPose D0 seed (no construction cost)

- The expensive FoundationPose construction (nvdiffrast `RasterizeCudaContext`, `ScorePredictor` + `PoseRefinePredictor` weight load, first-call CUDA JIT, mesh upload — together ≈4 s per instance) **already ran in Phase 0b** for every prefused candidate. The trackers live on the pipeline in `_fp_trackers_by_instance: dict[int, FoundationPoseTracker]`.
- D0.6 just: pop the tracker for the Path-A-selected `instance_id` into `self._fp_tracker`, delete the rest of the dict (so unused nvdiffrast contexts + model weights are GC-eligible), then call `initialize_from_known_pose(D0_rgb, D0_depth, K, c2w_cv, refine_iterations=fp_init_refine_iter)`. With the default `fp_init_refine_iter=0`, this is `est.pose_last = (mesh_to_camera @ inv(centered_mesh))` and a return — milliseconds. With `fp_init_refine_iter>0`, it additionally runs `track_one` to settle the seed against the actual D0 RGB-D.
- Capture the flagged object means/quats once as the reference object pose for the rigid transform application path.
- **Fallback**: if `_fp_trackers_by_instance` is empty for the selected id (legacy datasets without the manifest, or Phase 0b construction failed), `_initialize_fp_tracker` constructs the tracker on the spot using `phase0_manifest.json` — the slow path is preserved as a safety net.
- **No 2D mask required for first-frame init** — the known scene pose replaces what `register()` would have searched for.

#### D0.7 Render RS00

- Re-render scene after SAM3D insertion from D0's camera pose → RS00
- Reference render for CD0 computation

#### D0.8 Change mask CD0

- Object mask = `render_object_mask` — projected object-Gaussian mask (no SAM2 union).
- `_compute_change_mask(RS00_rgb, RS00_depth, D0_rgb, D0_depth, gripper_mask, render_object_mask)` returns `(cd0, valid_mask)`.
- Excludes gripper + projected object from change detection: `valid_mask = gripper_mask × (1 − object_mask)`. After the MS-SSIM cleanup recipe, `cd0` is re-clipped to `valid_mask` so the closing operation cannot bleed back into the excluded regions.
- CD0 stored in `model.change_mask_image` for the next training steps.

#### D0.9 Debug images

- Save ~9 overlay PNGs to `<data_root>/dynamic_scene/debug/`

### Phase 3 — Dynamic Loop, Frame N >= 1

(DN.1 is no longer present — the SAM2 live mask propagation step was removed
when SAM2 was deactivated. **FoundationPose tracks the object pose
statefully (consecutive frame to frame) and the projected object-Gaussian
mask after the rigid transform is the authoritative "where the object is"
mask** for change-mask exclusion. SAM2 was redundant.)

#### DN.3 FoundationPose track_one

- `tracker.track_one(rgb=DN_rgb, depth=DN_depth, K, camera_to_world=cv_c2w, iterations=fp_track_refine_iter)` — FP refines the cached `pose_last` against the new RGB-D and updates it in place.
- Convert `pose_in_camera` (mesh-to-camera) → world-frame absolute (R, t) from the D0 reference pose: `delta_world = (cv_c2w @ pose_in_camera) @ inv(mesh_to_world_init)`.
- `apply_rigid_object_transform_from_reference(R, t)` — the same model method used by CoTracker before; only the source of (R, t) changed.
- Writes `{frame_name}_fp_motion.txt` with R, t.

(DN.2, DN.4 are not present — FP is stateful and does not need a per-frame
reseed or post-filter.)

#### DN.5 Render RDN

- Re-render from DN's camera pose after rigid transform has been applied → RDN

#### DN.6 Render object mask

- `render_object_mask(camera)`: rasterize only `object_flags > 0.5` Gaussians → simulation-based mask
- **NOT SAM2 propagation.** There is no SAM2; the projected object-Gaussian mask is the authoritative "where the object is" signal.

#### DN.7 Change mask CDN

- Object mask = `render_object_mask` — projected object-Gaussian mask alone (no SAM2 union).
- `_compute_change_mask(RDN_rgb, RDN_depth, DN_rgb, DN_depth, gripper_mask, render_object_mask)` returns `(cdn, valid_mask)`.
- Excludes gripper + projected object from change detection: `valid_mask = gripper_mask × (1 − object_mask)`. After the MS-SSIM cleanup recipe, `cdn` is re-clipped to `valid_mask` so the closing operation cannot bleed back into the excluded regions.
- CDN stored in `model.change_mask_image` for the next 50 training steps.

#### DN.8 Debug images

- Save overlay PNGs to `dynamic_scene/debug/`. FP debug artifacts (rendered mesh overlays, score visualizations) live under `dynamic_scene/debug/fp_debug/`.

### Dynamic Training Step

- Forward render → masked RGB + depth loss (masked to CDN region)
- Rigid regularization loss on object Gaussians (`rigid_or_static_loss`)
- Backward → optimizer step
- `means` grad hook: only Gaussians in `scene_opt_active_mask` (footprint ∩ CDN AND non-object) receive `means` gradients (object moved via rigid transforms only)
- Scene opt hooks: zero gradient for object Gaussians on features/opacities/scales/quats

### Data Format

The data root must contain two subdirectories:
```
<data_root>/
  static_scene/    # Nerfstudio-formatted static frames (RGB + camera poses + transforms.json)
  dynamic_scene/   # Per-frame dynamic data: RGB, uint16 depth PNGs (scale 1e-3 m/unit), optional masks
```

- Static scene uses standard Nerfstudio `transforms.json` with `load_3D_points=True` (SfM points used to initialize Gaussians).
- Dynamic scene must include `depth_filenames` metadata; depth images are uint16 PNG scaled by `depth_unit_scale_factor=1e-3` (so value 1000 = 1 metre).
- Dataparser settings: `orientation_method="none"`, `center_method="none"`, `auto_scale_poses=False` — poses are used as-is without recentering.

### ROS Data Collection

For live robot teleoperation data:
```bash
source /path/to/devel/setup.bash
conda activate radiance_ros
python scripts/save_data_img_depth_mask_pose.py    # Collects RGB, depth, gripper mask, camera poses
python scripts/joint_state_merger.py               # Merges robot + gripper joint states
```

Mask generation note:
- The saved per-frame dataset masks live under `dynamic_scene/masks/` and are written by `RobotModelMaskSaver.save_mask(...)`.
- To keep the background, `save_mask(...)` must use `keep_mask = robot_exclusion_mask` and must not combine it with `background_keep_mask` / `background_black_mask`.
- To remove the background, restore the combination with `cv2.bitwise_and(robot_exclusion_mask, background_keep_mask)` or `cv2.bitwise_and(robot_black_mask, background_black_mask)`.
- There are multiple mask saver variants in this repo:
  - `scripts/save_data_img_depth_mask_pose.py`
  - `scripts/old/save_dynaarm_camera1_rgb_tf_current.py`
  - `scripts/old/ros1_robot_mask_saver_stl.py`
  - `scripts/old/ros1_robot_mask_saver_stl_tfdata.py`
- If the background still looks masked after code changes, the usual reason is that the masks were generated earlier. In that case, regenerate the dataset masks; changing the script alone does not modify existing `.png` masks already on disk.

Simulator background note:
- For dynamic-gs / Splatfacto, the correct place to set the Gazebo background is the model render background, not the dataparser `mask_color`.
- `mask_color` rewrites masked input pixels in the dataset loader and can make the training images look wrong.
- The correct implementation is in `DynamicGSModel.populate_modules()`: call `self.set_background(torch.tensor((0.86, 0.92, 1.0), ...))` and override `_get_background_color()` to return that fixed color.
- The Nerfstudio viewer also overrides Splatfacto background from the viewer control panel on every render (`viewer/render_state_machine.py`), so the viewer default background must also be set to the Gazebo color. In this repo that default now lives in:
  - `nerfstudio/nerfstudio/viewer/control_panel.py`
  - `nerfstudio/nerfstudio/viewer_legacy/server/control_panel.py`
- Keep the `NoRefineStrategy`, `strategy_state`, means-grad hook, and `_apply_phase_trainability()` inside `populate_modules()`. Accidentally moving them below `_get_background_color()` makes them unreachable and can severely slow or break the dynamic transition.

### Output Structure

Important runtime outputs currently live mainly under the dataset root, not only under Nerfstudio `outputs/`:

```
<data_root>/
  ├─ dynamic_scene/
  │  ├─ initialization_debug/
  │  │  ├─ *.png
  │  │  └─ *_sam3d_preview.png
  │  ├─ initialization_artifacts/
  │  │  ├─ *_sam3d_*.ply
  │  │  ├─ *_sam3d_*.txt
  │  │  └─ *_sam3d_*.json
  │  └─ debug/
  │     ├─ *_fp_motion.txt
  │     └─ fp_debug/
  └─ timing_report.txt
```

The method currently uses `NoSaveTrainer`, so checkpoint saving is intentionally disabled during these runs unless that trainer choice is changed.

### Third-Party Dependencies (`third_party/`)

- **`sam-3d-objects/`**: SAM3D model for single-view 3D object reconstruction from RGB + mask. Uses `pipeline_runtime_small.yaml`. Required checkpoints: `ss_generator`, `slat_generator`, `ss_decoder`, `slat_decoder_gs`. The mesh decoder, GS4 decoder, and `ss_encoder` are not used. Multi-object path (`run_sam3d_multi_object`) loads the model once and processes masks sequentially on the full image (no cropping).
- **SAM3** ([github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)): Text-prompted segmentation for Phase 0 object discovery. Requires separate conda env `sam3_dynamic_gs` (Python 3.12+, PyTorch 2.7+, CUDA 12.6+) because the training env is incompatible. Invoked via `conda run -n sam3_dynamic_gs python`.
- **FoundationPose** ([github.com/NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)): Cloned at `third_party/FoundationPose/`. Weights at `third_party/FoundationPose/weights/{2023-10-28-18-33-37,2024-01-11-20-02-45}/model_best.pth`. Runtime deps in `radiance_ros` env: `torch + cu118` matching nvcc 11.8 (installed into the env via `conda-forge cuda-nvcc`), `nvdiffrast` (built from source against torch 2.1/cu118), `pytorch3d` (already there), `kornia`, `warp-lang`, `transformations`, `ruamel.yaml`, `pyrender`. The native `mycpp` extension is built per-Python-version via `cmake .. -DPYTHON_EXECUTABLE=...`; both `mycpp.cpython-38-*.so` (radiance_ros) and `mycpp.cpython-311-*.so` (foundationpose env) coexist in `mycpp/build/`.
- **SAM2**: **Deactivated** in the current pipeline. The previous code path (live-mask propagation D(N-1)→DN to gate the change mask) was made redundant by FoundationPose's stateful pose tracking — the projected object-Gaussian mask after the rigid transform now drives the change-mask exclusion alone. The wrapper file `dynamic_gs/utils/sam2.py` is kept on disk but is no longer imported or referenced by `dynamic_gs_pipeline.py`, `dynamic_gs_model.py`, or `dynamic_gs/utils/__init__.py`.
- **ESAM**: Interactive segmentation model for frame 0 object mask extraction.
- **PROBREG / Open3D**: Used in `sam3d_fusion.py` for point cloud registration and CPD-based similarity refinement.

### Timing Instrumentation

The pipeline includes reusable timing instrumentation in `dynamic_gs_pipeline.py`. All timing data is accumulated in `self._timing` (a `defaultdict(list)`) and a report is written at the end of training.

**Timer key naming convention:** `{phase}.{number}_{description}` — e.g., `D0.2_sam3d_generation`, `DN.3_fp_track`.

**How to add a new timer:** Wrap the code section with a descriptive comment and `time.time()` calls:
```python
# --- TIMING: DN.X Description of what is being timed (details for future editors) ---
t0 = time.time()
...code...
self._timing["DN.X_short_name"].append(time.time() - t0)
```
The comment before each timer block describes what is being timed so that future code modifications can correctly move or update the timing boundaries. Always keep the comment and the timer key in sync.

**Timer keys by phase:**

| Phase | Keys | Location |
|-------|------|----------|
| Phase 0a | `S0.1`, `S0.2`, `S0.4a_generation_total` | `_run_sam3_and_sam3d_generation` |
| Phase 0b | `S0.3_fusion_obj_*`, `S0.4b_fusion_total` | `_fuse_sam3d_objects_into_scene` |
| Static | `static_step` | `get_train_loss_dict` |
| Frame 0 | `D0.1` through `D0.10` (incl. `D0.1a-f`, `D0.3a-f`) | `_prepare_frame_0` + `prepare_dynamic_update` + `initialize_object_from_sam3d` |
| Frame N | `DN.3`, `DN.5` through `DN.9` | `_prepare_frame_n` |
| Dynamic training | `dynamic_step` | `get_train_loss_dict` |

SAM3D generation vs insertion timing is split inside `initialize_object_from_sam3d()` in `dynamic_gs_model.py`, which returns `sam3d_generation_time` and `sam3d_insertion_time` in its stats dict. The pipeline reads these to populate `D0.2` and `D0.3`.

**Output:** `<data_root>/timing_report.txt` — written at the last training step. Contains chronological per-phase breakdowns with absolute times and percentages. Console summary is also printed via `_print_timing_summary()`.
