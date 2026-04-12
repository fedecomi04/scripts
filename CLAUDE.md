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
```

## Architecture

### Two-Phase Training

**Phase 1 — Static (steps 0–1000):** Optimizes the standard Splatfacto Gaussian scene (appearance + geometry). Only non-positional parameters are updated (features_dc, scales, quats, opacities). The `means` optimizer has its LR set to 0. Gaussian refinement (densification/pruning) is disabled via `NoRefineStrategy`.

**Phase 2 — Dynamic (steps 1000+, 50 steps/frame):** Per-frame optimization of dynamic objects. The phase transition resets the `means` optimizer state and its LR scheduler. `means` gradients are masked via a registered backward hook (`_mask_means_grad`) so only eligible Gaussians receive updates.

- **Frame 0 bootstrap:** `prepare_dynamic_update` (ESAM + render RS + change mask + Gaussian flagging) → SAM3D 3D generation → insert object Gaussian cloud → seed SAM2 tracker → CoTracker initialization on RS00 → compute change mask CD0
- **Frame N tracking:** SAM2 propagation of live object mask → CoTracker refresh (fill point gaps) → CoTracker advance (2D track + RANSAC rigid transform) → apply rigid transform to object Gaussians → render RDN → render object mask → filter CoTracker points by mask → compute change mask CDN

### Phase Transition Details

`_sync_phase(step)` is called at the start of every `get_train_loss_dict`. It:
1. Detects phase changes (static → dynamic) and calls `model.set_phase(phase)`.
2. Detects frame transitions within the dynamic phase and calls `_prepare_dynamic_frame()`.
3. `model.set_phase("dynamic")` sets `requires_grad` on all Gaussian parameter groups, registers scene optimization gradient hooks on features/opacities/scales/quats, and resets the `means` optimizer + scheduler.
4. The `max_num_iterations` in the Nerfstudio trainer is updated to `static_num_steps + total_dynamic_frames * dynamic_steps_per_frame` at callback time.

### Gradient Masking Strategy

During the dynamic phase, two separate masking mechanisms run in parallel:

- **`means` grad hook (`_mask_means_grad`):** If `enable_scene_optimization=True`, only non-object Gaussians (where `object_flags < 0.5`) receive `means` gradients. Object Gaussians are moved exclusively via CoTracker rigid transforms, not gradient descent.
- **Scene opt hooks:** Registered on features_dc, features_rest, opacities, scales, quats to zero gradients for object Gaussians, so their appearance is not changed by scene optimization.
- **`enable_dynamic_mean_optimization` (default False):** Legacy flag. When True (and `enable_scene_optimization=False`, `enable_cotracker_rigid_motion=False`), allows gradient descent on object means using the change mask. Not the recommended approach.

### Core Class Hierarchy

```
DynamicGSModel          (dynamic_gs_model.py)
  └─ extends SplatfactoModel
     - object_flags: (N,1) persistent buffer — 1.0 for object Gaussians, 0.0 for scene
     - sam3d_init_target_flags: (N,1) persistent buffer — flags added by SAM3D
     - current_active_mask: (N,) non-persistent buffer — change mask projected to Gaussians
     - change_mask_image: (H,W,1) non-persistent buffer — current frame's CDN mask
     - _mask_means_grad: backward hook filtering means gradients
     - apply_rigid_object_transform: rotates+translates means+quats for object Gaussians
     - insert_sam3d_object: bbox scale + CPD similarity registration + Gaussian insertion
     - prepare_dynamic_update: ESAM interactive segmentation + change mask + flag Gaussians
     - render_object_mask: rasterize only object Gaussians using gsplat rasterization
     - refresh_dynamic_state_after_insertion: re-flags Gaussians after SAM3D insertion

DynamicGSPipeline       (dynamic_gs_pipeline.py)
  └─ extends VanillaPipeline
     - _timing: defaultdict(list) — per-step timing accumulator
     - _cotracker_motion: CoTrackerMotionEstimator instance (frame 0+)
     - _live_tracker_rgb/mask: previous live frame for SAM2 propagation
     - _prepare_frame_0(): full bootstrap sequence
     - _prepare_frame_n(): incremental update sequence
     - _compute_change_mask(): MSSIM change detection excluding gripper+object
     - _print_timing_summary(): logs full pipeline timing at last step

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
| `active_mask.py` | Change detection from RGB/depth deltas; MSSIM-based `build_change_mask`; morphological filtering |
| `cotracker_motion.py` | `CoTrackerMotionEstimator`: pairwise CoTracker3 tracking + RANSAC rigid transform; `refresh_tracking_points` fills coverage gaps from SAM2 mask; `filter_points_by_mask` removes drift |
| `sam3d.py` | SAM3D subprocess invocation (`run_sam3d_single_object_subprocess`); output path management; PLY loading |
| `sam3d_fusion.py` | Point cloud registration via bbox scale + centroid alignment + voxel downsampling + probreg CPD similarity; Gaussian insertion and deduplication via `register_and_fuse_sam3d_object` |
| `sam2.py` | SAM2 video predictor (`build_sam2_tiny_video_predictor`); `query_sam2_propagated_mask` for pairwise frame-to-frame mask propagation |
| `esam.py` | ESAM interactive object mask query (`query_esam_mask`); `build_esam_ti` model builder |
| `depth_loss.py` | `masked_l1_depth_loss`: per-pixel L1 depth loss masked to valid regions |
| `rigid_regularization.py` | `rigid_or_static_loss`: Kabsch-based rigid body consistency loss; promotes coherent group motion |
| `no_refine_strategy.py` | `NoRefineStrategy`: disables gsplat's default Gaussian refinement (densification + pruning) |

### Key Configuration (`dynamic_gs_config.py`)

```python
STATIC_NUM_STEPS = 1000
DYNAMIC_STEPS_PER_FRAME = 50
```

Key model parameters in `DynamicGSModelConfig` (inherits from `SplatfactoModelConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_lambda` | 0.4 | Depth supervision loss weight |
| `change_mask_depth_threshold` | 0.02 | Meters — pixel flagged as changed if depth delta > threshold |
| `change_mask_rgb_threshold` | 0.15 | MSSIM threshold for RGB change detection |
| `change_mask_use_rgb` | False | Use RGB (not just depth) for change mask |
| `change_mask_blur_kernel_size` | 5 | Gaussian blur kernel for change mask smoothing |
| `change_mask_filter_radius` | 1 | Morphological filter radius on change mask |
| `change_mask_min_component_size` | 64 | Min connected component size (pixels) to keep |
| `active_mask_dilate_radius` | 0 | Extra dilation on final change mask |
| `object_mask_dilate_px` | 1 | Dilation on the rendered object mask |
| `rigid_static_lambda` | 0.1 | Weight for rigid body regularization loss |
| `use_sam3d_object_init` | True | Generate SAM3D object cloud at frame 0 |
| `reuse_sam3d_generated_ply` | True | Skip SAM3D subprocess if PLY already exists (set False for fresh generation) |
| `enable_dynamic_mean_optimization` | False | Gradient descent on object means (legacy; not recommended) |
| `enable_cotracker_rigid_motion` | True | Apply CoTracker rigid transform to object Gaussians |
| `enable_scene_optimization` | True | Continue optimizing scene Gaussians during dynamic phase |
| `scene_opt_refine_every` | 100 | Steps between scene Gaussian densification/pruning in dynamic phase |
| `scene_opt_densify_grad_thresh` | 0.0002 | Grad norm threshold for densification |
| `scene_opt_cull_alpha_thresh` | 0.1 | Alpha threshold for pruning |
| `cotracker_query_point_count` | 256 | Points sampled from object mask as CoTracker seeds |
| `cotracker_min_track_points` | 12 | Min tracked points required to compute rigid transform |
| `cotracker_ransac_iterations` | 128 | RANSAC iterations for outlier rejection |
| `cotracker_ransac_inlier_threshold` | 0.02 | Inlier threshold (meters in 3D) |
| `cotracker_point_refresh_min_distance` | 8.0 | Min pixel distance for new refresh points |
| `cotracker_checkpoint_path` | "" | Path to local CoTracker3 checkpoint (empty = torch.hub) |
| `cotracker_hub_model` | "cotracker3_offline" | Model name for torch.hub loading |

### Optimizer Groups

Six Adam optimizer groups are defined in `dynamic_gs_config.py`:

| Group | LR | Active in static | Active in dynamic |
|-------|-----|------------------|-------------------|
| `means` | 1.6e-4 | No | Yes (object only via hook) |
| `features_dc` | 0.0025 | Yes | Yes (scene only via hook) |
| `features_rest` | 0.0025/20 | Yes | Yes (scene only via hook) |
| `opacities` | 0.05 | Yes | Yes (scene only via hook) |
| `scales` | 0.005 | Yes | Yes (scene only via hook) |
| `quats` | 0.001 | Yes | Yes (scene only via hook) |

LR is zeroed (not disabled) for inactive groups. When transitioning to the dynamic phase, the `means` optimizer state and scheduler are reset so it restarts from initial LR.

### CoTracker Motion Estimation Detail

`CoTrackerMotionEstimator` (in `cotracker_motion.py`) operates pairwise on consecutive frames:

1. **`initialize(rgb, depth, camera, mask)`**: Samples `query_point_count` (256) random 2D points inside the object mask. Stores as `_current_points_xy`. Loads CoTracker3 (offline variant) from `torch.hub` or local checkpoint.

2. **`estimate_and_advance(current_rgb, current_depth, current_camera)`**:
   - Builds a 2-frame video tensor `[prev_frame, curr_frame]`
   - Calls CoTracker3 predictor with query points from previous frame
   - Gets tracked positions + visibility on current frame
   - Back-projects visible points to 3D using depth + camera intrinsics + extrinsics
   - RANSAC: random 3-point minimal samples → SVD rigid body fit → count inliers within threshold
   - Returns `CoTrackerMotionEstimate` with R, t, inlier stats, residuals

3. **`refresh_tracking_points(mask)`**: After SAM2 propagation, fills gaps in coverage by sampling new 2D points from under-covered SAM2 mask regions. Uses `point_refresh_min_distance` to avoid crowding.

4. **`filter_points_by_mask(mask)`**: Removes tracked points that fall outside the current SAM2 object mask (handles drift after gripper releases object).

5. **`CoTrackerMotionEstimate.success`**: False if `inlier_count < min_track_points` or RANSAC failed. Pipeline skips `apply_rigid_object_transform` on failure and logs a warning.

**Known limitation:** CoTracker3 is used in offline mode on 2-frame pairs. Confidence scores (`vis`, `conf`) are uniformly ~1.0 on such short sequences because the model has no temporal context to detect drift. Confidence-based filtering was investigated and removed.

### SAM3D Integration

`initialize_object_from_sam3d` in the model:

1. Saves the rendered RGB and live object mask to disk (required as SAM3D inputs).
2. Calls `run_sam3d_single_object_subprocess` — launches a separate Python process running SAM3D (`sam-3d-objects`) which generates a Gaussian `.ply` file.
3. `register_and_fuse_sam3d_object` loads the PLY, then aligns it with the existing object cloud using bbox scale + centroid alignment + voxel downsampling + probreg CPD similarity, and appends only non-overlapping Gaussians to the existing set.
4. `refresh_dynamic_state_after_insertion` re-runs Gaussian flagging to label the newly inserted points as object members.

Key parameters: `reuse_sam3d_generated_ply=True` (default) skips the subprocess if a PLY already exists from a previous run. Set to `False` when you need fresh generation (e.g., for accurate timing measurements or after changing the object).

SAM3D checkpoints live in `third_party/sam-3d-objects/checkpoints/hf/`. The pipeline uses `pipeline_runtime_small.yaml` which requires only: `ss_generator`, `slat_generator`, `ss_decoder`, `slat_decoder_gs`. The mesh and GS4 decoders are null and their checkpoints are not needed.

### Change Mask Computation

`_compute_change_mask` (pipeline) and `build_change_mask` (utils):

1. Compares rendered depth vs GT depth: pixels with `|depth_render - depth_gt| > change_mask_depth_threshold` are flagged.
2. Optionally compares rendered RGB vs live RGB using MSSIM if `change_mask_use_rgb=True`.
3. Excludes gripper mask region (from batch data) and current object mask from change detection.
4. Applies Gaussian blur + morphological filtering + connected component size filtering.
5. Optionally dilates the final mask by `active_mask_dilate_radius`.

The resulting mask (CDN) is stored in `model.change_mask_image` and used as the optimization mask for the next 50 training steps of that frame.

### Debug Outputs

Per-frame debug images saved to `<data_root>/dynamic_scene/render_masks_esam/` and `<data_root>/dynamic_scene/debug/`:

| File | Content |
|------|---------|
| `{frame}_render.png` | Rendered RGB at static phase end (RS) |
| `{frame}_live.png` | Live camera RGB |
| `{frame}_rs00.png` | Re-rendered RGB after SAM3D insertion |
| `{frame}_rdn.png` | Re-rendered RGB after rigid transform (frame N) |
| `{frame}_change_mask.png` | Raw change mask from ESAM step |
| `{frame}_render_object_mask.png` | Rendered Gaussian object mask |
| `{frame}_live_object_mask.png` | ESAM live object mask |
| `{frame}_cd0.png` / `{frame}_cdn.png` | Final change mask used for optimization |
| `{frame}_render_w_cd0.png` | Render with CDN overlay (red) |
| `{frame}_live_w_cdn.png` | Live with CDN overlay (red) |
| `{frame}_render_w_objmask.png` | Render with object mask overlay (blue) |
| `{frame}_render_w_combined.png` | Render with union mask overlay (cyan) |
| `{frame}_live_w_gripper.png` | Live with gripper mask overlay (green) |
| `{frame}_cotracker.png` | Side-by-side prev/curr with tracked points (green dots) + yellow connecting lines |
| `{frame}_cotracker_motion.txt` | CoTracker rigid transform stats: R, t, inliers, residuals |
| `{frame}_sam3d_*.ply` | SAM3D-generated object point clouds |

### Timing Profile (measured, 61 dynamic frames)

| Phase | Time | Notes |
|-------|------|-------|
| Static training | 23.7s | 1000 steps @ 23.7ms/step |
| Frame 0 bootstrap | 80.8s | SAM3D gen=72.8s, SAM3D ins=5.5s, change detect=2.2s, rest <0.1s |
| Frame N prep (avg) | 601ms/frame | CoTracker=300ms, debug images=120ms, SAM2=107ms, refresh=44ms |
| Frame N prep (total, 60 frames) | 36.1s | |
| Dynamic training | 163.5s | 3050 steps @ 53.6ms/step |
| **Grand total** | **304.2s** | (pipeline total, not wall time) |

Bottlenecks: SAM3D subprocess (one-time, 72.8s), CoTracker advance (50% of per-frame prep), debug image saving (20% of per-frame prep). Disabling debug saves would cut ~7s from frame prep.

D0.1 breakdown — ESAM is NOT the bottleneck despite the name "Initial change detection (ESAM)":
- Change mask MSSIM + morphological filtering: 0.88s (41%)
- Gaussian center projection + flagging (~300K centers): 0.79s (37%)
- ESAM on render (incl. first-call model load): 0.24s (11%)
- ESAM on live image: 0.11s (5%)
- Forward render (eval mode): 0.15s (7%)

Timing is instrumented in `get_train_loss_dict` and `_prepare_frame_{0,n}` using `self._timing` (a `defaultdict(list)`). Full report written to `<data_root>/timing_report.txt` at the last training step.

## Pipeline Step-by-Step Reference (Corrected + Timed)

### Symbol Glossary

| Symbol | Meaning |
|--------|---------|
| RS | Rendered static scene (fully trained, no object inserted) |
| DN | N-th image from the dynamic dataset (live camera) |
| MN | Gripper/background mask from transforms.json for DN |
| RS00 | Render of static scene + inserted SAM3D object, from D0's camera pose |
| RDN | Render from DN's pose, after rigid transform applied to flagged Gaussians |
| F0_render | ESAM object mask queried on RS (rendered image, frame 0 only) |
| F0_live | ESAM object mask queried on D0 (live image, frame 0 only) |
| FDN_live | SAM2-propagated object mask on DN (live tracker: D0→D1→D2→...) |
| CDN | Change mask between RDN and DN, excluding object + gripper regions |

### Phase 1 — Static Training (~23.7s total)

From an initial SfM pointcloud, optimize all Gaussian parameters except `means` (LR=0 via zero'd optimizer). No densification/pruning (`NoRefineStrategy`). RGB loss only. 1000 steps @ ~23.7ms/step.

### Phase 2 — Dynamic Frame 0 Bootstrap (~80.8s total)

#### D0.1 Initial change detection (2.16s total)

1. **D0.1a** Forward render (0.15s): call `get_outputs(camera)` in eval mode → RS image + depth
2. **D0.1b** Change mask (0.88s): `build_change_mask(RS_depth, D0_depth)` — MSSIM depth comparison, Gaussian blur, morphological open/close, connected-component filtering → C0
3. Sample points from C0 interior (90% inward erosion from mask boundary) — these points lie inside the object on both RS and D0 (object barely moved)
4. **D0.1c** ESAM on RS (0.24s, incl. one-time model load): `query_esam_mask(esam, RS_rgb, C0)` → F0_render
5. **D0.1d** ESAM on D0 live (0.11s): `query_esam_mask(esam, D0_rgb, C0)` → F0_live
6. **D0.1e** Gaussian flagging (0.79s): project ~300K+ Gaussian centers to 2D (`extract_projected_centers_and_radii`), check which fall inside union(F0_render, F0_live) (`build_active_mask`) → set `object_flags = 1` for object Gaussians

#### D0.2 SAM3D object generation (72.77s, 90% of frame 0)

- Subprocess: launch `sam-3d-objects` with RS image + F0_live mask → generates a Gaussian `.ply` file
- One-time cost; skip on re-runs with `reuse_sam3d_generated_ply=True`

#### D0.3 SAM3D insertion: bbox scale + CPD similarity + dedup (5.53s, 6.8%)

- Load generated PLY, register against existing object cloud: bbox scale init → centroid alignment → voxel downsample → probreg CPD similarity refinement → dedup
- Deduplicate overlapping Gaussians → append to `means`, `features_dc`, etc.
- Mark newly inserted Gaussians as `object_flags = 1`

#### D0.4 Render object mask (~0ms)

- `render_object_mask(camera)`: rasterize only `object_flags > 0.5` Gaussians, threshold + dilate
- **Simulation-based**, NOT a new ESAM query on RS00. There is no F00.

#### D0.5 Seed live SAM2 tracker (~0ms)

- Store `(D0_live_rgb, F0_live)` as the starting state for the SAM2 propagation chain
- D0 → D1 → D2 → ... live mask propagation in subsequent frames

#### D0.6 CoTracker initialization (~10ms)

- Create `CoTrackerMotionEstimator`, call `initialize(D0_live_rgb, D0_depth, camera, F0_live)`
- Loads CoTracker3 (offline) model, samples 256 2D seed points from F0_live on D0's **live image**
- **Seeded on D0 live image + mask, NOT on RS00**
- **No rigid transform at frame 0.** The first `estimate_and_advance()` + rigid transform happen at frame 1.

#### D0.7 Render RS00 (~20ms)

- Re-render scene after SAM3D insertion from D0's camera pose → RS00
- Reference render for CD0 computation

#### D0.8 Change mask CD0 (~10ms)

- Combined object mask = `max(render_object_mask, F0_live_resized)` — union of simulation mask + ESAM live mask
- `_compute_change_mask(RS00_rgb, RS00_depth, D0_rgb, D0_depth, gripper_mask, combined_obj_mask)` → CD0
- Excludes gripper + object union from change detection (valid_mask = M0 × (1 − obj_union))
- CD0 stored in `model.change_mask_image` for the next 50 training steps

#### D0.9 Debug images (~60ms)

- Save ~9 overlay PNGs to `<data_root>/dynamic_scene/debug/`

### Phase 3 — Dynamic Loop, Frame N ≥ 1 (60 frames: ~36.1s prep + ~163.5s training)

Per-frame prep average: ~601ms. Per-frame training: 50 steps × 53.6ms = ~2.7s.

#### DN.1 SAM2 live mask propagation (107ms, 17.8% of frame prep)

- `query_sam2_propagated_mask(predictor, prev_live_rgb, curr_live_rgb, prev_live_mask)` → FDN_live
- Propagates D(N-1) object mask → DN using SAM2 tiny video predictor (pairwise, not full video)
- Updates `_live_tracker_rgb` and `_live_tracker_mask` for next frame
- **Live tracker only.** No render tracker exists. FRN does not exist.

#### DN.2 CoTracker refresh: fill point gaps (44.5ms, 7.4%)

- `refresh_tracking_points(FDN_live)`: sample new 2D seed points in under-covered SAM2 mask regions
- Avoids crowding: new points must be ≥ `point_refresh_min_distance=8px` from existing tracked points

#### DN.3 CoTracker advance + rigid transform (299.7ms, 49.8%)

- Build 2-frame video `[prev_live_frame, curr_live_frame DN]`, query CoTracker3 with seed points from D(N-1)
- Back-project visible tracked points to 3D using DN depth + camera intrinsics/extrinsics
- RANSAC (128 iterations, 3-point minimal samples) → SVD rigid body fit → SE(3) = (R, t)
- If inliers ≥ 12: `apply_rigid_object_transform(R, t)` — moves all `object_flags > 0.5` Gaussians (means + quats)
- If RANSAC fails: skip transform, log warning
- Writes `{frame_name}_cotracker_motion.txt` with R, t, inlier count, residuals

#### DN.4 CoTracker filter by mask (1.1ms, 0.2%)

- `filter_points_by_mask(FDN_live)`: remove tracked points that drifted outside the current SAM2 mask
- Handles drift when gripper releases or occludes the object

#### DN.5 Render RDN (14.2ms, 2.4%)

- Re-render from DN's camera pose after rigid transform has been applied → RDN

#### DN.6 Render object mask (4.2ms, 0.7%)

- `render_object_mask(camera)`: rasterize only `object_flags > 0.5` Gaussians → simulation-based mask
- **NOT SAM2 propagation.** There is no render tracker. FRN does not exist.

#### DN.7 Change mask CDN (11.4ms, 1.9%)

- Combined mask = `max(render_object_mask, FDN_live_resized)` — union of simulation + SAM2 live
- `_compute_change_mask(RDN_rgb, RDN_depth, DN_rgb, DN_depth, gripper_mask, combined_obj_mask)` → CDN
- CDN stored in `model.change_mask_image` for the next 50 training steps

#### DN.8 Debug images (120ms, 19.9%)

- Save ~9 overlay PNGs: RDN, DN live, CDN overlay, object mask overlay, CoTracker visualization, etc.

### Dynamic Training Step (~53.6ms/step, 50 steps/frame)

- Forward render → masked RGB + depth loss (masked to CDN region)
- Rigid regularization loss on object Gaussians (`rigid_or_static_loss`)
- Backward → optimizer step
- `means` grad hook: only non-object Gaussians receive `means` gradients (object moved via rigid transforms only)
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

### Output Structure

```
outputs/<run_date>/dynamic-gs/<timestamp>/
  ├─ checkpoints/
  ├─ config.yaml
  └─ dynamic_scene/render_masks_esam/
      ├─ *_sam3d_*.ply        # Object point clouds
      ├─ *_cotracker_motion.txt
      └─ debug images (*_render.png, *_live.png, ...)
```

### Third-Party Dependencies (`third_party/`)

- **`sam-3d-objects/`**: SAM3D model for single-view 3D object reconstruction from RGB + mask. Uses `pipeline_runtime_small.yaml`. Required checkpoints: `ss_generator`, `slat_generator`, `ss_decoder`, `slat_decoder_gs`. The mesh decoder, GS4 decoder, and `ss_encoder` are not used.
- **CoTracker3**: Loaded via `torch.hub` from `facebookresearch/co-tracker` or from a local checkpoint path set in `cotracker_checkpoint_path`.
- **SAM2**: Tiny video predictor loaded via `build_sam2_tiny_video_predictor` for mask propagation.
- **ESAM**: Interactive segmentation model for frame 0 object mask extraction.
- **PROBREG / Open3D**: Used in `sam3d_fusion.py` for point cloud registration and CPD-based similarity refinement.

### Timing Instrumentation

The pipeline includes reusable timing instrumentation in `dynamic_gs_pipeline.py`. All timing data is accumulated in `self._timing` (a `defaultdict(list)`) and a report is written at the end of training.

**Timer key naming convention:** `{phase}.{number}_{description}` — e.g., `D0.2_sam3d_generation`, `DN.3_cotracker_advance`.

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
| Static | `static_step` | `get_train_loss_dict` |
| Frame 0 | `D0.1` through `D0.10` | `_prepare_frame_0` |
| Frame N | `DN.1` through `DN.9` | `_prepare_frame_n` |
| Dynamic training | `dynamic_step` | `get_train_loss_dict` |

SAM3D generation vs insertion timing is split inside `initialize_object_from_sam3d()` in `dynamic_gs_model.py`, which returns `sam3d_generation_time` and `sam3d_insertion_time` in its stats dict. The pipeline reads these to populate `D0.2` and `D0.3`.

**Output:** `<data_root>/timing_report.txt` — written at the last training step. Contains chronological per-phase breakdowns with absolute times and percentages. Console summary is also printed via `_print_timing_summary()`.
