# Duplication audit: camera / coordinate-convention conversions

Scope: `dynamic_gs/` (package) + `scripts/`. Class hunted: OpenGL c2w ↔ OpenCV
flips (`diag(1,-1,-1,1)`), world↔camera back-projection, intrinsics
scaling/cropping, pose inversions, quaternion↔matrix. Every site below was
read and confirmed to be the SAME logic (not a coincidental name match).

Convention anchors (from CLAUDE.md "Background + Camera Conventions"): poses in
`transforms.json` are **OpenGL c2w**; `OnlineFusion`/FoundationPose convert to
OpenCV via `diag(1,-1,-1,1)`; depth is uint16 mm on disk (`1e-3` scale). Any
shared helper MUST preserve these and must NOT touch the identity buffers.

---

## Pattern A — OpenGL c2w → OpenCV c2w (`@ diag([1,-1,-1,1])`)

Identical 4×4 right-multiply by `diag(1,-1,-1,1)`. Confirmed runtime copies:

- `dynamic_gs/utils/online_fusion.py:528` — `OnlineFusion._cv_c2w` (static method)
- `dynamic_gs/utils/rgbd_fusion_init.py:101` — `cv_c2w_from_opengl`
- `dynamic_gs/utils/preseg_seed.py:339` — inline `flip = np.diag([1,-1,-1,1])`

Script copies (ad-hoc, lower priority but the same constant):
- `scripts/bench_gpu_fusion.py:66-67` — `opengl_to_cv`
- `scripts/sweep_tsdf_voxel.py:61-62` — `opengl_to_cv` (byte-identical to bench)
- `scripts/step_reproject_static.py:47` — `_GL2CV = np.diag([1,-1,-1,1])`
- `scripts/icp_refine_dynamic_transforms.py:38` — `_FLIP = np.diag([1,-1,-1,1])`
- `scripts/test_foundationpose_static_scene.py:66` — inline `flip`
- `scripts/view_anysplat_nerfstudio.py:215` — the INVERSE direction (`c2w_cv @ diag` to go CV→GL); same matrix, self-inverse

Risk: **correctness.** A wrong sign here silently mis-fuses the seed / mis-places
inserts (this exact family produced the historical ICP/registration offsets). If
one copy is "fixed" and others aren't, frames diverge in a way masked by GS
training (per the GPU-TSDF-doubling note) — i.e. very hard to spot.

Proposed helper: `gl_c2w_to_cv(c2w)` (and self-documenting `cv_c2w_to_gl = gl_c2w_to_cv`)
in a new `dynamic_gs/utils/camera_conventions.py`. `online_fusion`,
`rgbd_fusion_init`, `preseg_seed` import it; the script copies import it too.

---

## Pattern B — OpenGL-frame pixel+depth → world back-projection

Camera-space ray as `((u-cx)/fx, -(v-cy)/fy, -1)` (or the `*z` variant), then
`@ R.T + t` with an OpenGL c2w. Same convention, same algebra, re-implemented:

- `dynamic_gs/utils/tracker_common.py:289-319` — `backproject_to_world` (per-point, `*z` form, OpenGL)
- `dynamic_gs/utils/rgbd_decode.py:91-115` — `_backproject_world` (torch, ray-dir form, OpenGL)
- `dynamic_gs/fusion/phase0.py:57+` — `backproject_mask_to_world` (the canonical, feature-rich one: mask resize + near-surface filter + MAD scrub; the *core* deprojection inside it is this same form)
- `dynamic_gs/utils/anysplat_decode.py:571-579` — inline in `icp_refine_scene_c2w` (`d*(u-cx)/fx, -d*(v-cy)/fy, -d`, OpenGL)
- `dynamic_gs/utils/preseg_seed.py:350-351` — project side, but the deproject sibling lives here too
- `scripts/reproject_static_frames.py:80-81`, `scripts/diag_pose_drift.py:24-25`, `scripts/compare_depth_filters.py:69-70`, `scripts/reproject_anchor.py:32-33`, `scripts/diag_target_repro.py:68-69`, `scripts/render_reproj_o3d.py:29-30`, `scripts/preview_double_noise.py:48-49`, `scripts/diag_fix_isolate.py:68`, `scripts/diag_confirm_icp.py:39` — ad-hoc diag copies (OpenGL `-(v-cy)/fy` form)

NOTE a DIFFERENT (OpenCV) deprojection also exists and must NOT be merged with B:
`dynamic_gs/utils/online_fusion.py:256-257` and `rgbd_fusion_init.py:107-108`
use `(u-cx)*z/fx, (v-cy)*z/fy, +z` (OpenCV frame, paired with a CV c2w from
Pattern A) and `live_ros_publisher.py:1207-1208` uses `+x, -y` for its SHM cloud.
`sam3d.py:449-450` uses pytorch3d's `-x,-y,+z`. These are three distinct frames;
a shared helper must take an explicit `convention=` arg, not assume one.

Risk: **correctness.** The y/z sign is the exact thing that flips a cloud upside
down or front-to-back; scattered copies make it easy to paste the wrong frame's
form (the sam3d/pytorch3d vs OpenGL vs OpenCV confusion is already live here).

Proposed helper: `deproject_to_world(u, v, z, fx, fy, cx, cy, c2w, *, convention)`
with `convention ∈ {"opengl", "opencv", "pytorch3d"}`, numpy+torch dispatch, in
`dynamic_gs/utils/camera_conventions.py`. Keep `backproject_mask_to_world` as the
mask-aware wrapper that calls it. Migrate `tracker_common`, `rgbd_decode`,
`anysplat_decode` ICP, scripts.

---

## Pattern C — OpenGL-frame world → pixel projection

The forward of B: `cam = (p - t) @ R; depth = -cam_z; u = fx*cam_x/depth+cx;
v = fy*(-cam_y)/depth+cy`. Re-implemented nearly verbatim:

- `dynamic_gs/dynamic_gs_pipeline_live.py:381-387` — bulk means projection in `_select_*` D0
- `dynamic_gs/dynamic_gs_pipeline_recorded.py:307-312` — per-instance centroid projection in D0 (same algebra, scalar)
- `dynamic_gs/utils/preseg_seed.py:350-351` — `u = fx*cam[...,0]/z + cx`, `v = fy*cam[...,1]/z + cy`
- `scripts/d0_instance_overlay.py:86-87` — overlay projection (same form)

Risk: **maintenance** (drifts from the matching deproject) bordering on
**correctness** — live and recorded D0 picker must agree pixel-for-pixel or the
two pipelines select different objects from the same scene.

Proposed helper: `project_world_to_pixel(points, fx, fy, cx, cy, c2w, *, convention="opengl")`
→ `(u, v, depth, in_front)`, in `dynamic_gs/utils/camera_conventions.py`. The two
pipeline D0 selectors collapse onto it.

---

## Pattern D — wxyz quaternion ↔ rotation matrix

Two sub-patterns, each duplicated.

**D1 — wxyz → rotmat** (the `1-2(y²+z²)…` matrix), three copies:
- `dynamic_gs/utils/anysplat_decode.py:425-435` — `quat_wxyz_to_rotmat` (numpy, batched)
- `dynamic_gs/utils/sam3d_fusion.py:114-130` — `_quaternion_wxyz_to_rotation_matrix` (numpy, single, with norm-guard)
- `scripts/debug_rgbd_decode.py:113-114+` — inline (torch)

**D2 — rotmat → wxyz** (four-cases-largest-component / Shepperd), three copies:
- `dynamic_gs/utils/anysplat_decode.py:438-480` — `rotmat_to_quat_wxyz` (numpy, batched)
- `dynamic_gs/utils/rgbd_decode.py:119-163+` — `_rotmat_to_wxyz` (torch, batched)
- `dynamic_gs/dynamic_gs_model.py:883-919` — `DynamicGSModel._rotation_matrix_to_quaternion` (torch, single 3×3)

Also adjacent (same family, single copies — fold in for free): `dynamic_gs_model.py:866`
`_normalize_quaternions`, `:869` `_quaternion_multiply`; `tracker_common.py:322-330`
`_so3_exp/_so3_log` (`cv2.Rodrigues` wrappers) and `xfeat_motion.py:925-927`
(another raw `cv2.Rodrigues` for the same rvec↔R job).

Risk: **correctness.** All four-case branches must order components identically
(wxyz) and pick the same numerically-stable branch; a transcription slip in one
copy yields a subtly-wrong rotation only on certain orientations (trace ≤ 0) —
the worst kind to debug. The norm-guard exists in only some copies.

Proposed helper: a small `dynamic_gs/utils/rotations.py` with
`quat_wxyz_to_rotmat`, `rotmat_to_quat_wxyz` (numpy), `quat_wxyz_to_rotmat_t`,
`rotmat_to_quat_wxyz_t` (torch), `quat_multiply`, `quat_normalize`, plus
`rvec_to_rotmat`/`rotmat_to_rvec` wrapping `cv2.Rodrigues`. `anysplat_decode`,
`sam3d_fusion`, `rgbd_decode`, `dynamic_gs_model`, `tracker_common`,
`xfeat_motion` all import from it.

---

## Pattern E — OpenCV → OpenGL ROTATION-only flip (`diag([1,-1,-1])`)

3×3 flip applied to a rotation (positions handled by Pattern A/the formula):
- `dynamic_gs/utils/anysplat_decode.py:777` — `R_scene @ diag([1,-1,-1]) @ R_pred.T` (AnySplat CV → scene GL)
- `scripts/anysplat_patch_to_scene.py:61` — `OPENCV_TO_OPENGL = np.diag([1,-1,-1])`

Risk: **maintenance.** Only 2 sites and one is a script, but it is the same
constant as Pattern A's rotation block; documenting it as one named constant
(`CV_TO_GL_ROT = diag([1,-1,-1])`) prevents a future third copy guessing the sign.

Proposed helper: expose `CV_TO_GL_ROT` (and `GL_TO_CV_ROT`, identical) from
`camera_conventions.py`.

---

## NOT duplication (verified, called out so the purge doesn't over-merge)

- `dynamic_gs/utils/anysplat_decode.py:694-704` resize+center-crop intrinsics
  inversion — complex, SINGLE site (the one that fixed the 1200p ghost bug). The
  other "intrinsics edit" sites are unrelated: `sam3d.py:386-387` shifts cx/cy by
  a crop ORIGIN (different op), `viser_direct.py:149-152` derives fx/fy from a
  viser FOV (different op). Do not merge.
- `viser_direct.py` `_FLIP_YZ` (3×3) is the nerfstudio↔viser basis change, a
  THIRD distinct convention (see CLAUDE.md invariant #9 / the viser notes). Keep
  separate from Patterns A/E; it could live beside them in the new module but
  must stay a distinct named matrix.
