# Dynamic-GS Duplication Consolidation Plan

Synthesizes the four cross-module duplication scans (coord conventions, depth
handling, mask handling, worker/subprocess, Gaussian tensor-surgery) into a
single ranked, ordered purge plan.

Source reports:
- `code_audit/DUP_coord_conventions.md`
- `code_audit/DUP_depth_handling.md`
- `code_audit/DUP_mask_handling.md`
- `code_audit/DUP_worker_subprocess.md`
- `code_audit/DUP_tensor_surgery.md`

Total estimated LOC removable: **~660** across ~110 call sites, in **6 new
shared modules** + extensions to 2 existing ones (`active_mask.py`,
`depth_filter.py`).

---

## CLAUDE.md invariants this plan must respect

These gate several of the helpers below — the consolidation must NOT erase
behavior the invariants depend on:

- **Inv #8 (per-object identity buffers).** The 4 buffers (`object_flags`,
  `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`) are
  written by specific phases. Tensor-surgery helpers must subset/concat them in
  lockstep with the 6 gauss_params, and must NOT introduce a writer for
  `sam3d_init_target_flags` (it stays all-zeros) or write `object_flags` from
  the static path.
- **Inv #1/#4 (means LR / dynamic LRs).** The optimizer-refresh helper must keep
  `reset_means_optimizer` controllable per-caller — it must not silently re-enable
  a means optimizer where LR is meant to be 0.
- **Depth-band single-source.** `DEPTH_MIN_M=0.05`, `DEPTH_MAX_M=2.0`
  (env-overridable `DGS_TSDF_DEPTH_MAX_M`) live in `online_fusion.py` and are
  the canonical band. `zed_depth_noise.py` deliberately gates WIDER (3.0) — that
  asymmetry is documented and must remain a separate explicit value, not be
  collapsed onto the fusion cap.
- **Disk format.** uint16-mm depth, OpenGL c2w on disk. mm<->m and GL<->CV
  helpers must preserve these.
- **Convention split is a FLAG, not a merge.** OpenGL / OpenCV / pytorch3d
  back-projection variants coexist on purpose; the shared back-project helper
  takes an explicit `convention=` arg and must default to nothing surprising.
- **Import-weight of the ROS py3.8 env.** `live_ros_publisher.py` runs in the
  minimal `dynamic_gs_ros` env (numpy + os only). Any module it imports
  (atomic-write, mm<->m) MUST stay stdlib/numpy-only — no torch/nerfstudio.

---

## 1. Ranked proposed shared helpers

Ranked by (correctness-risk × LOC saved). Highest first.

### R1 — `dynamic_gs/utils/gaussian_surgery.py` (NEW)  — ~148 LOC, CORRECTNESS-CRITICAL
The densest and riskiest cluster: the 6-param + 4-buffer lockstep surgery is
copy-pasted between `StaticGSModel` and `DynamicGSModel` and **has already
drifted** (see Correctness-Risk §2.1).

Functions and the sites that collapse onto each:

- **`subset_gaussians_in_place(model, keep)`** — replaces both
  `delete_gaussian_indices` bodies.
  - `dynamic_gs/static_gs_model.py:508-536`
  - `dynamic_gs/dynamic_gs_model.py:1116-1147`
  - LOC saved ~25. (Helper slices 6 params + every name in
    `model._identity_buffer_names()`; the dynamic-only `current_active_mask`
    becomes just another buffer in that list — drift removed by construction.)

- **`concat_gaussians_in_place(model, new_tensors) -> old_num_points`** —
  replaces the insert/concat skeleton (cat loop w/ `requires_grad` preservation
  + `_resize_dynamic_buffers` + `_refresh_gaussian_optimizers`); each caller
  keeps its own identity-flag write over `[old:]`.
  - `dynamic_gs/static_gs_model.py:465-505` (`insert_object_gaussians`)
  - `dynamic_gs/dynamic_gs_model.py:1200-1228`
  - `dynamic_gs/dynamic_gs_model.py:1230-1257` (`insert_inpaint_gaussians`)
  - LOC saved ~30.

- **`build_default_gaussian_tensors(model, new_xyz, new_rgb) -> dict`** —
  byte-identical `_build_new_gaussian_tensors`.
  - `dynamic_gs/static_gs_model.py:428-463`
  - `dynamic_gs/dynamic_gs_model.py:1082-1113`
  - LOC saved ~33.

- **`refresh_optimizer_params(model, *, reset_means_optimizer)`** +
  **`resize_identity_buffers(model, num_points)`** — the `_refresh_gaussian_optimizers`
  and `_resize_dynamic_buffers` near-duplicates (static uses a closure, dynamic
  inlines 5 blocks). Buffer set driven by `model._identity_buffer_names()`.
  - `dynamic_gs/static_gs_model.py:411-426` + `:381-409`
  - `dynamic_gs/dynamic_gs_model.py:1063-1080` + `:1030-1061`
  - LOC saved ~30. (Each method calls the core then its own model-specific tail:
    static = none; dynamic = `current_active_mask` + means-grad hook/phase tail.)

- **`uniform_shrink_log_scales(log_scales, max_scale_m, *, min_scale_m=0.0) -> (log_scales, keep)`**
  — backend-dispatched (torch/numpy); codifies the documented
  uniform-shrink-not-clamp policy once.
  - `dynamic_gs/static_gs_model.py:244-266` (mid-train scale-reset callback)
  - `dynamic_gs/utils/anysplat_decode.py:791-803` (insert cap)
  - (related metres->log seed / log-add: `anysplat_decode.py:495`,
    `rgbd_decode.py:448`)
  - LOC saved ~12.

- **`activated_opacity(logits)` + `low_opacity_indices(logits, thr)`** —
  torch/numpy-dispatched sigmoid+threshold.
  - `dynamic_gs/static_gs_pipeline.py:234-235`
  - `dynamic_gs/utils/anysplat_decode.py:667-668`
  - LOC saved ~6.

- **(in-file)** `reproject_anysplat_to_scene` 5× lockstep subset of the same 6
  arrays — fold into a local `_subset(keep)` closure or a small `GaussArrays`
  dataclass.
  - `dynamic_gs/utils/anysplat_decode.py:669-670, 676-677, 729-730, 751-752, 799-803`
  - LOC saved ~12.

**Prereq:** add `_identity_buffer_names()` to each model (static returns 4
buffers; dynamic returns 4 + `current_active_mask`). This makes the
static/dynamic difference *data*, not *forked code*.

### R2 — `dynamic_gs/utils/depth_ops.py` (NEW)  — ~138 LOC, mixed correctness/maintenance
Single home for depth math + the depth-band single-source.

- **`backproject_pixels_to_world(u,v,z,fx,fy,cx,cy,c2w,*,convention,backend='auto')`**
  — pinhole deproject, convention-flagged (this is the SAME helper as
  coord-conventions R3 §below; build it once, in `camera_conventions.py`, and
  re-export from `depth_ops.py`, or place it in `depth_ops.py` and import into
  `camera_conventions.py` — pick ONE owner to avoid a new circular dup).
  - `dynamic_gs/utils/tracker_common.py:289`
  - `dynamic_gs/utils/rgbd_decode.py:91`
  - `dynamic_gs/fusion/phase0.py:57` / `:177`
  - `dynamic_gs/utils/anysplat_decode.py:560`/`:571`
  - `dynamic_gs/utils/online_fusion.py:251` (OpenCV variant)
  - `dynamic_gs/utils/rgbd_fusion_init.py:104` (OpenCV variant)
  - scripts: `reproject_static_frames.py:80`, `diag_pose_drift.py:24`,
    `compare_depth_filters.py:69`, `reproject_anchor.py:32`,
    `diag_target_repro.py:68`, `render_reproj_o3d.py:29`,
    `preview_double_noise.py:48`
  - LOC saved ~45 (depth report) + ~40 (coord report) — overlapping; net ~45.

- **`project_world_to_image(pts,fx,fy,cx,cy,W,H,c2w,*,convention='opengl') -> (u,v,z,in_img)`**
  — the inverse + finite/in-front/in-bounds gate.
  - `dynamic_gs/fusion/phase0.py:226`
  - `dynamic_gs/utils/anysplat_decode.py:505`
  - `dynamic_gs/dynamic_gs_pipeline_live.py:381`
  - `dynamic_gs/dynamic_gs_pipeline_recorded.py:307`
  - scripts: `d0_instance_overlay.py:86`, `preseg_seed.py:350`
  - LOC saved ~22 (depth) / ~20 (coord) overlapping; net ~22.

- **`depth_mm_to_m(u16)` / `depth_m_to_mm_u16(m)` + `DEPTH_SCALE_MM=1000.0`** —
  STDLIB/numpy only (imported by `live_ros_publisher` in the ROS env).
  - `online_fusion.py:255`, `rgbd_fusion_init.py:226`,
    `live_ros_publisher.py:844,1128,1134`, `live_session.py:285,326`,
    `scripts/capture_only.py:111`
  - LOC saved ~20.

- **`depth_band_valid(depth,*,z_min=DEPTH_MIN_M,z_max=DEPTH_MAX_M)`** + re-export
  single-source `DEPTH_MIN_M`/`DEPTH_MAX_M` (imported by `online_fusion`,
  `static_gs_model`, `zed_depth_noise`). **Keep `zed_depth_noise`'s wider 3.0
  cap as its own explicit override (Inv).**
  - `online_fusion.py:252`, `static_gs_model.py:216`, `zed_depth_noise.py:57`,
    `anysplat_decode.py:565`, scripts `render_reproj_o3d.py:21`,
    `preview_double_noise.py:29`
  - LOC saved ~12.

- **`rescale_intrinsics(fx,fy,cx,cy,src_wh,dst_wh)`** — verbatim block.
  - `phase0.py:141` / `:218`, `dynamic_gs_model.py:1874`
  - LOC saved ~14.

### R3 — `dynamic_gs/utils/rotations.py` (NEW)  — ~90 LOC, CORRECTNESS-CRITICAL
The four-case Shepperd R->wxyz branch is the highest single-function correctness
risk (one mistranscribed sign breaks insert orientation).

- **`quat_wxyz_to_rotmat` / `rotmat_to_quat_wxyz`** (numpy + torch),
  **`quat_multiply`, `quat_normalize`, `rvec_to_rotmat`/`rotmat_to_rvec`**.
  - `anysplat_decode.py:425,438`, `sam3d_fusion.py:114`, `rgbd_decode.py:119`,
    `dynamic_gs_model.py:866,869,883`, `tracker_common.py:322`,
    `xfeat_motion.py:925`, scripts `debug_rgbd_decode.py:113`
  - LOC saved ~90.

### R4 — `dynamic_gs/utils/camera_conventions.py` (NEW)  — ~16 LOC, correctness
Small but high-correctness named constants + the GL<->CV flip. Owns the
convention enum used by R2's deproject/project helpers.

- **`gl_c2w_to_cv(c2w)` / `cv_c2w_to_gl` (alias)** — 4x4 right-mul by
  `diag([1,-1,-1,1])`.
  - `online_fusion.py:528`, `rgbd_fusion_init.py:101`, `preseg_seed.py:339`,
    scripts `bench_gpu_fusion.py:66`, `sweep_tsdf_voxel.py:61`,
    `step_reproject_static.py:47`, `icp_refine_dynamic_transforms.py:38`,
    `test_foundationpose_static_scene.py:66`, `view_anysplat_nerfstudio.py:215`
  - LOC saved ~12.
- **`CV_TO_GL_ROT` / `GL_TO_CV_ROT`** named 3x3 `diag([1,-1,-1])` constants.
  - `anysplat_decode.py:777`, scripts `anysplat_patch_to_scene.py:61`
  - LOC saved ~2.

### R5 — `dynamic_gs/utils/seg_subprocess.py` (NEW) + `conda_env.py` (NEW) + `io_atomic.py` (NEW)  — ~70 LOC, correctness
Subprocess/worker orchestration. `io_atomic`/`conda_env` MUST be stdlib-only
(ROS env imports `io_atomic`).

- **`atomic_write_transforms(path, meta)`** (`io_atomic.py`)
  - `fusion_runner.py:335-337`, `online_fusion.py:627-630`,
    `live_session.py:295-297,444-446`,
    `live_ros_publisher.py:1148-1150,1225-1227`,
    scripts `capture_only.py:125-127`, `test_concurrent_fusion.py:76-78`
  - LOC saved ~14. **Highest correctness item here** — `fusion_runner` polls
    `transforms.json` on a watcher thread; a torn (non-atomic) write yields
    bad reads.
- **`conda_subprocess_env(...)`** + **`resolve_env_python(conda_env)`**
  (`conda_env.py`, single `_CONDA_ROOT` source)
  - `sam3d.py:54-63,32-41`, `sam3_segmentation.py:289-293,38-42`,
    `fastsam_segmentation.py:564-569,41-43`, `sam_worker.py:753-759,742-743`,
    `anysplat_decode.py:374-376,358-361`
  - LOC saved ~21.
- **`run_segmentation_subprocess(...)`** (`seg_subprocess.py`)
  - `sam3_segmentation.py:258-315`, `fastsam_segmentation.py:529-585`
  - LOC saved ~35.

### R6 — extend `dynamic_gs/utils/active_mask.py` (+ optional `gaussian_projection.py`)  — ~140 LOC, mixed
- **`select_gaussians_in_mask_near_depth(...)` + `frontmost_object_subset(...)`**
  (new `gaussian_projection.py`, next to `extract_projected_centers_and_radii`) —
  the copied projected-Gaussian-into-mask subset helpers. **Drives Phase-0b
  SAM3D registration/cull; the two models are meant to behave identically.**
  - `static_gs_model.py:556-601` ↔ `dynamic_gs_model.py:1456-1498`
  - `static_gs_model.py:603-701` ↔ `dynamic_gs_model.py:1502-1601`
  - LOC saved ~90.
- **`erode_mask_np` / `dilate_mask_np`** (thin numpy wrappers in `active_mask.py`,
  ONE `2*px+1` cv2 kernel) — torch callers keep existing `*_binary_mask`.
  - `xfeat_motion.py:1241-1244,662-667`, `tracker_common.py:159-161`,
    `rgbd_decode.py:368-369`
  - LOC saved ~18.
- **`mask_bbox(mask)` + `square_crop_about_bbox(bbox,H,W,pad,min_side)`**
  - `sam3d.py:331-347`, `dynamic_gs_pipeline_base.py:3178-3199,1856-1935`,
    `fastsam_segmentation.py:213,262`
  - LOC saved ~24.
- **export `to_hw1` + add `to_hw`** (squeeze) — folds away once R6/R1 helpers
  use them internally. ~8 LOC.
- **`scale_mask_about_centroid` — DO NOT extract yet** (single impl at
  `dynamic_gs_pipeline_base.py:2437-2467`); promote only if a 2nd caller appears.

### Explicitly NOT to merge (over-merge guards, verified non-duplications)
- AnySplat resize+center-crop intrinsics inversion (single complex site).
- `sam3d` crop-origin cx/cy shift; viser fx/fy-from-FOV; viser `_FLIP_YZ`
  (a distinct nerfstudio<->viser basis change — NOT the GL<->CV flip).
- `SamWorkerClient` load/unload/infer (already consolidated over one `_request`).
- `live_shm_reader` ROS spawn (deliberately STRIPS LD_LIBRARY_PATH — opposite logic).
- post_fusion_cache load path; viewer/diag read-only display masking; rigid
  transform in-place means/quats writes (mutate values, don't change count).

---

## 2. CORRECTNESS-RISK section — where drift between copies is a real bug

Ordered by severity.

### 2.1 Gaussian tensor-surgery — params + identity buffers MUST stay in sync (Inv #8)  [TOP RISK]
`StaticGSModel` and `DynamicGSModel` carry copy-pasted delete/insert/refresh/resize.
**Already drifted** — confirmed by reading the code:
- `delete_gaussian_indices`: dynamic (`dynamic_gs_model.py:1143-1144`) additionally
  subsets `current_active_mask`; static (`static_gs_model.py:530-533`) does not.
  If `current_active_mask` ever applied on the static path, the static copy would
  desync it from the params.
- insert/concat: static `insert_object_gaussians:489-493` rebuilds Parameters
  WITHOUT explicit `requires_grad=` preservation; the dynamic copies preserve it
  under `no_grad`. A future change relying on grad state would behave differently
  per model.
**Why it's a real bug class:** if any of the 6 params and 4 buffers fall out of
lockstep (different length, wrong order), the identity buffers point at the wrong
Gaussians — silently corrupting tracking/insert ownership, the exact failure
Inv #8 exists to prevent. R1 removes the fork: ONE helper, buffer set from
`_identity_buffer_names()`, so static/dynamic differ only in data.
**Mitigation:** before/after, the helper must (a) assert all params + buffers have
equal length post-op, (b) preserve `reset_means_optimizer` per caller (Inv #1/#4),
(c) never introduce a `sam3d_init_target_flags` writer (Inv #8).

### 2.2 wxyz<->rotmat Shepperd branch (R3)
The four-case largest-component R->wxyz is implemented in ≥6 runtime files. A
single sign error in one copy mis-orients inserted/tracked Gaussians and is hard
to spot (looks "almost right"). One audited implementation removes the risk.

### 2.3 Back-projection convention mixing (R2/R4)
6 hand-written pinhole deprojections across **two incompatible conventions**
(OpenGL y/z-negated vs OpenCV homogeneous). Using the wrong one flips y/z sign and
mis-places inserts — this is the documented 1200p ghost / SAM3D-offset failure
class (CLAUDE.md). The shared helper takes an explicit `convention=` flag;
**must not** auto-detect or default in a way that silently picks the wrong frame.

### 2.4 Depth-band drift 2.0 vs 3.0 (R2)
`zed_depth_noise.py` (3.0) and `static_gs_model.scene_depth_max_m` (hand-kept
== `DEPTH_MAX_M`) drift from canonical `online_fusion` 2.0 — CLAUDE.md itself
flags the mismatch. Single-source the constant; keep `zed_depth_noise`'s wider
gate as an explicit documented override, not an accidental copy.

### 2.5 Atomic transforms.json write (R5)
`fusion_runner` polls `transforms.json` on a watcher thread. Any site that writes
non-atomically (or a future copy that drops the tmp+`os.replace`) causes torn
reads mid-fusion. Verbatim across 7 prod sites — consolidate to one helper.

### 2.6 Divergent depth bilateral filter (R6/depth)
`rgbd_decode.py:62`'s bilateral copy LACKS the weight-correction the canonical
`depth_filter.py` documents fixing — so FF-mode depth is filtered differently
from the tracker/seed path. Route `rgbd_decode` through
`depth_filter.filter_depth_torch(median=False, bilateral=True)` and delete the
local copy.

### 2.7 Projected-Gaussian-into-mask subset (R6)
Static/dynamic copies drive Phase-0b SAM3D registration & cull and are *meant*
to be identical; the static docstring even claims reuse while the code is copied.
Drift here changes which Gaussians are culled/targeted → registration offset.
Note: these helpers only READ/select indices, they don't reorder arrays, so
Inv #8 is not directly at risk — but the registration result is.

---

## 3. Suggested consolidation order

Principle: do the pure-extraction, low-blast-radius, stdlib-only modules first
(they unblock the rest and can't regress training math), then the
correctness-critical model surgery last under the protection of the helpers
built earlier. Validate each step with the existing end-to-end run before moving on.

1. **R5 `io_atomic.py` + `conda_env.py` + `seg_subprocess.py`** (~70 LOC).
   Pure orchestration, no math, no torch. `io_atomic`/`conda_env` stdlib-only
   (ROS-env safe). Lowest risk, immediate correctness win (2.5).
   *Verify:* a capture run writes a valid `transforms.json`; SAM3 + FastSAM
   subprocess paths still return their JSON contract.

2. **R3 `rotations.py`** (~90 LOC). Self-contained pure functions; add a unit
   test asserting round-trip `R -> wxyz -> R` and against `cv2.Rodrigues` /
   `scipy` for a battery of rotations BEFORE swapping call sites. Removes 2.2.

3. **R4 `camera_conventions.py`** (~16 LOC) — named flip constants +
   `gl_c2w_to_cv`. Trivial, and it establishes the `convention` enum that R2
   depends on. Removes 2.3's flip half.

4. **R2 `depth_ops.py`** (~138 LOC). Build `backproject_pixels_to_world` /
   `project_world_to_image` with the explicit `convention` flag (single owner —
   decide depth_ops vs camera_conventions, re-export from the other). Then
   mm<->m, depth-band single-source, intrinsics-rescale. Removes 2.3 (deproject
   half) + 2.4. *Verify:* reproject-static diagnostic clouds match pre-change
   byte-for-byte per convention; seed PLY unchanged.

5. **R6 `active_mask.py` extensions + `gaussian_projection.py`** (~140 LOC).
   Morphology wrappers + bbox/crop + the projected-into-mask subset helpers +
   route `rgbd_decode` bilateral through `depth_filter`. Removes 2.6, 2.7.
   *Verify:* Phase-0b registration/cull on a known object produces the same
   kept-point count; tracker gating masks unchanged.

6. **R1 `gaussian_surgery.py`** (~148 LOC) — LAST and most carefully. Add
   `_identity_buffer_names()` to both models; extract subset/concat/build/refresh/
   resize; collapse both `delete_gaussian_indices`, all three inserts,
   `uniform_shrink_log_scales`, opacity threshold, and the anysplat 5× subset.
   Removes 2.1. *Verify (hard gate):* after a full `static-gs` run, assert (a)
   all 6 params + all buffers in `_identity_buffer_names()` are equal-length at
   every delete/insert, (b) `post_fusion_state.pt` round-trips through
   load/save unchanged, (c) means LR stays 0 (Inv #1) and dynamic LRs stay 0
   (Inv #4), (d) `sam3d_init_target_flags` remains all-zeros (Inv #8). Run the
   static opacity purge + mid-train scale reset and confirm identical drop counts
   to a pre-change baseline.

Each step is independently shippable and LOC-positive; if step 6 proves risky it
can be deferred without blocking R2–R5.
