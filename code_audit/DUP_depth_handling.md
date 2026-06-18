# Duplication audit — depth handling

Scope: every place where the dynamic-gs repo re-implements the same depth logic —
uint16-mm <-> float-m conversion, the 0.05..3.0 m depth-band gate, pinhole
back-projection of depth pixels to 3D world points, depth resize/filter, and
NaN/inf/zero scrubbing. Each pattern below was confirmed by reading the cited
sites, not by name match.

Convention guardrails the proposed helpers MUST keep (from CLAUDE.md):
- Depth on disk is **uint16 millimetres**, `0 = invalid`; dataparser
  `depth_unit_scale_factor = 1e-3`. Writers clip to `[0, 65535]` before cast.
- Back-projection is **Nerfstudio/OpenGL camera frame** (x right, +y **up**,
  +z **back**, looking along -z): `x=(u-cx)z/fx`, `y=-(v-cy)z/fy`, `z=-z`.
  The **fusion/Open3D path uses OpenCV** (`y` down, `z` forward, no negation),
  so a single helper must take a `convention` flag — they are NOT
  interchangeable and silently mixing them flips the sign of y/z.
- The depth band is `(DEPTH_MIN_M, DEPTH_MAX_M]` with
  `DEPTH_MIN_M = 0.05`, `DEPTH_MAX_M = float(os.environ.get("DGS_TSDF_DEPTH_MAX_M","2.0"))`
  — the canonical pair already lives in `online_fusion.py`. Several copies still
  hardcode the OLD `3.0` upper bound (see Pattern C).
- These are read-only-by-design buffers, not touched here; the helpers only move
  numeric depth logic, so the identity buffers / background color invariants are
  unaffected.

---

## Pattern A — pinhole back-projection of (u, v, depth) → world points

The exact same back-projection (build camera-frame ray from pixel + intrinsics,
rotate by c2w, add translation) is hand-written in **6 places**, in two
conventions. They differ only in array library (numpy vs torch), homogeneous vs
explicit R/t, and the OpenGL-vs-OpenCV sign of y/z.

OpenGL/Nerfstudio convention (`y=-(v-cy)z/fy`, `z=-z`):
- `dynamic_gs/utils/tracker_common.py:289` — `backproject_to_world(points_xy, depth_values, intrinsics, c2w)` (the most general, intrinsics matrix in).
- `dynamic_gs/utils/rgbd_decode.py:91` — `_backproject_world(u, v, depth_z, fx, fy, cx, cy, c2w)` (torch, ray-direction form).
- `dynamic_gs/fusion/phase0.py:177-185` — inside `backproject_mask_to_world` (numpy, explicit x/y/z stack).
- `dynamic_gs/utils/anysplat_decode.py:560-579` — inside `icp_refine_scene_c2w` (torch+meshgrid, builds source cloud).

OpenCV convention (no y/z negation, homogeneous c2w_cv multiply):
- `dynamic_gs/utils/online_fusion.py:251-259` — `_GpuOnlineFusion._src_cloud` (numpy, `cam=[x,y,z,1] ; world = c2w_cv @ cam.T`).
- `dynamic_gs/utils/rgbd_fusion_init.py:104-111` — `_backproject_world(depth_m, valid, c2w_cv, fx, fy, cx, cy)` (numpy, homogeneous).

Risk: **correctness**. The y/z sign IS the whole convention; a copy-paste that
crosses the OpenGL/OpenCV line (or a future intrinsics-resize fix applied to
only some copies) silently mis-places every inserted/registered point. This is
exactly the class of bug that produced the "ghost copy / sideways insert" at
1200p (CLAUDE.md, anysplat reproject note) and the SAM3D-insert offset.

Proposed helper: `backproject_pixels_to_world(u, v, z, fx, fy, cx, cy, c2w, *, convention="opengl", backend="auto")` in **`dynamic_gs/utils/depth_ops.py`** (new module). `backend="auto"` dispatches numpy vs torch on the input type; `convention` ∈ {"opengl","opencv"}. The matrix-intrinsics entry point (`tracker_common`) and the homogeneous entry point can be thin wrappers.

Est LOC saved: ~45.

---

## Pattern B — uint16-mm depth file <-> float32-metres conversion

Read-side `mm/1000 -> m` and write-side `clip(m*1000, 0, 65535) -> uint16`.

Read side (runtime):
- `dynamic_gs/utils/online_fusion.py:255` — `zz = depth_u16[..] / DEPTH_SCALE` (DEPTH_SCALE=1000.0); also `:598` loads tiff as uint16.
- `dynamic_gs/utils/rgbd_fusion_init.py:78` `DEPTH_SCALE=1000.0`, `:226` `_load_depth_u16_mm`, `:325`.
- `dynamic_gs/utils/live_ros_publisher.py:844` — `arr_mm.astype(np.float32) * 1e-3` (16UC1 branch); `:1206` `depth_mm[..] / 1000.0`.
- `dynamic_gs/dynamic_gs_datamanager.py:81` / `dynamic_gs/utils/preseg_seed.py:103` — `depth_unit_scale_factor=1e-3` (the dataparser-driven version of the same constant).

Write side (mm uint16 cast, identical `clip(.,0,65535)` line):
- `dynamic_gs/utils/live_ros_publisher.py:1128` and `:1134` (raw + filtered).
- `dynamic_gs/utils/live_session.py:285` and `:326` (static keyframe + anchor_ref).
- `scripts/capture_only.py:111`.
- `scripts/fuse_bilateral_experiment.py:86`.

Risk: **maintenance** (and latent correctness on the scale constant). The
`1e-3` / `1000.0` / `65535` magic numbers are spread across IO sites; if the
disk depth unit ever changes (or a sensor publishes a different scale) every
copy must be found. Today they agree, so the risk is drift, not a present bug.

Proposed helpers: `depth_mm_to_m(arr_u16) -> float32` and
`depth_m_to_mm_u16(arr_m) -> uint16` (the canonical `clip(m*1000,0,65535)`),
plus a `DEPTH_SCALE_MM = 1000.0` constant, in **`dynamic_gs/utils/depth_ops.py`**.
The 4 write-side `np.clip(... * 1000.0, 0.0, 65535.0).astype(np.uint16)` lines
collapse to one call each; the read-side `/1000` likewise.

Est LOC saved: ~20.

---

## Pattern C — the (0.05, MAX] m depth-band validity gate

The canonical pair is `DEPTH_MIN_M=0.05` / `DEPTH_MAX_M` (env, default 2.0) in
`online_fusion.py:66,76`. The same "valid = depth in (0.05, MAX)" predicate is
re-derived elsewhere, and **several copies hardcode the stale 3.0 upper bound**
or a different near floor:

- `dynamic_gs/utils/online_fusion.py:252` — `(depth_u16 > DEPTH_MIN_M*DEPTH_SCALE) & (< DEPTH_MAX_M*DEPTH_SCALE)` (the source of truth).
- `dynamic_gs/static_gs_model.py:216` — `(d > 0.05) & (d < dmax)` re-hardcodes the `0.05` floor (dmax = `scene_depth_max_m`, kept equal to DEPTH_MAX_M by hand per CLAUDE.md).
- `dynamic_gs/utils/zed_depth_noise.py:57-58,89` — `_Z_MIN=0.05`, `_Z_MAX=3.0` range gate; comment at `:39,56` admits it must match DEPTH_MIN/MAX but **defaults to 3.0, not 2.0** (the documented misalignment).
- `dynamic_gs/utils/anysplat_decode.py:565` — `valid_t = depth_t > 0.01` (near floor only, different constant).
- `scripts/render_reproj_o3d.py:17,21` and `scripts/preview_double_noise.py:22,29` — `Z_MAX, Z_MIN = 2.0, 0.05`.

Risk: **correctness**. CLAUDE.md explicitly flags that `zed_depth_noise` gates to
3.0 while fusion caps at 2.0, and that `static_gs_model.scene_depth_max_m` must
be kept EQUAL to `DEPTH_MAX_M` by hand. These are exactly the "keep this equal"
hazards that drift. A copy that still says 3.0 means depth in the 2–3 m band is
treated as valid in one place and discarded in another.

Proposed helper: `depth_band_valid(depth, *, z_min=DEPTH_MIN_M, z_max=DEPTH_MAX_M)`
returning the boolean keep-mask (works in metres for float arrays and accepts a
`scale` for the uint16-mm variant) in **`dynamic_gs/utils/depth_ops.py`**, and
make `zed_depth_noise` / `static_gs_model` import `DEPTH_MIN_M` / `DEPTH_MAX_M`
from there instead of redeclaring. Keeps the env override in one place.

Est LOC saved: ~12 (the real win is removing the silent 2.0-vs-3.0 divergence).

---

## Pattern D — bilateral / median depth filter ignoring invalid (==0) pixels

Two independent implementations of "bilateral-filter depth in metres, hold out
the `==0` invalid pixels, restore them after".

- `dynamic_gs/utils/depth_filter.py:41` `filter_depth` (cv2, weight-corrected bilateral + median) and `:87` `filter_depth_torch` (GPU equivalent) — the canonical, A/B'd implementation, ON by default.
- `dynamic_gs/utils/rgbd_decode.py:62-88` — `_bilateral_filter_depth` (cv2, `np.where(valid,0)` → `cv2.bilateralFilter` → restore). A second, simpler hand-rolled copy of the same idea (no weight-correction, no median).

Risk: **maintenance / cosmetic** drift. `rgbd_decode._bilateral_filter_depth`
is NOT weight-corrected, so zeros in the window pull valid pixels toward 0 — the
exact bug `depth_filter` documents fixing at `:67-73`. They will diverge in
quality. Per CLAUDE.md the FF path is supposed to consume already-filtered batch
depth (no re-filter), so this second filter may also be partially dead.

Proposed helper: route `rgbd_decode` through `depth_filter.filter_depth_torch(..., median=False, bilateral=True)` (already supports the per-stage split) and delete `_bilateral_filter_depth`. No new module needed.

Est LOC saved: ~25.

---

## Pattern E — depth resize / downscale (bilinear for depth, nearest for masks)

Depth is bilinearly down-sized in several spots; the related
mask/seg/component resize is `INTER_NEAREST` in several others. The depth ones
re-derive the same `interpolate(..., mode="bilinear", align_corners=False)` or
`cv2.resize` call.

- `dynamic_gs/dynamic_gs_model.py:1874-1882` — `_get_gt_depth` bilinear downscale by `_get_downscale_factor`.
- `dynamic_gs/utils/anysplat_decode.py` (reproject) — full-res sensor depth sampled at scaled scene pixels (the intrinsics-resize math at `phase0.py:141-147` and `cull_points_in_front:218-222` is the SAME `fx*=W/srcW; cx*=...` block, duplicated).
- Mask/component nearest-resize (same call, different interp): `preseg_seed.py:300`, `object_picker.py:171,179`, `anysplat_decode.py:747`.

Risk: **correctness** for the **intrinsics-rescale** sub-block specifically:
`fx *= W/src_W; fy *= H/src_H; cx *= ...; cy *= ...` is copy-pasted verbatim in
`phase0.py:141-147` and `phase0.py:218-222` (cull). If one is fixed (e.g. for a
non-square aspect) and the other isn't, projection and back-projection disagree
— precisely the 1200p square-only-assumption failure class.

Proposed helper: `rescale_intrinsics(fx, fy, cx, cy, src_wh, dst_wh) -> (fx,fy,cx,cy)` in **`dynamic_gs/utils/depth_ops.py`** for the duplicated rescale block. The depth-resize call itself is thin enough that consolidating it is lower value; flag the intrinsics-rescale as the real duplicate.

Est LOC saved: ~14.

---

## Pattern F — NaN/inf/zero "is this depth/point usable" scrub before use

`np.isfinite(...) & (... > 0)` over depth or back-projected points, repeated at
nearly every consumer. This is the loosest pattern — many are genuinely local
one-liners — but a few are the SAME multi-term predicate.

Confirmed same-shape copies (finite AND in-front AND in-image projection gate):
- `dynamic_gs/fusion/phase0.py:236` and `:244` — `np.isfinite(z) & (z>0) & (ui>=0)&(ui<W)&(vi>=0)&(vi<H)` twice in `cull_points_in_front` (target vs inserted), and again the projection lambda `_project` at `:226-232` is duplicated against `anysplat_decode._world_to_image_opengl:505-520` and the recorded/live `_project`-style blocks (`dynamic_gs_pipeline_live.py:381-387`, `dynamic_gs_pipeline_recorded.py:307-312`).
- depth-finite-and-positive: `tracker_common.py:284`, `depth_loss.py:14`, `rgbd_decode.py:280,407`.

Risk: **cosmetic / maintenance**. Individually trivial; the value is in the
*world→image projection + in-bounds gate* (the `_project` + `in_img` pair), which
recurs 4–5 times and is genuinely the same math as the back-projection inverse.

Proposed helper: `project_world_to_image(pts_world, fx, fy, cx, cy, W, H, c2w, *, convention="opengl") -> (u, v, z, in_img_mask)` in **`dynamic_gs/utils/depth_ops.py`** (the forward inverse of Pattern A's helper). Folds in the `isfinite & z>0 & in-bounds` gate. The bare `isfinite & >0` depth one-liners are NOT worth a helper — leave them.

Est LOC saved: ~22.

---

## Suggested consolidation target

A new **`dynamic_gs/utils/depth_ops.py`** holding: `DEPTH_SCALE_MM`,
`DEPTH_MIN_M`/`DEPTH_MAX_M` (moved from / re-exported by `online_fusion.py` to
avoid a circular import — or keep the source in `online_fusion` and have
`depth_ops` import them), `depth_mm_to_m`, `depth_m_to_mm_u16`,
`depth_band_valid`, `rescale_intrinsics`, `backproject_pixels_to_world`,
`project_world_to_image`. Patterns A, C, E-intrinsics and F-projection are the
correctness-bearing ones; B and D are maintenance/cosmetic.
