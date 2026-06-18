# Duplication audit — mask handling

Scope: gripper/object mask rendering, erode/dilate, mask-about-centroid scaling,
applying a mask to zero depth/rgb, connected-component selection, mask↔bbox.

Method: grepped `dynamic_gs/` + `scripts/` for the relevant cv2/scipy/numpy/torch
mask idioms, then Read each hit to confirm it is genuinely the same logic. Only
TRUE, verified duplication is reported below. The repo already has a canonical
torch morphology module — `dynamic_gs/utils/active_mask.py` (`dilate_binary_mask`,
`erode_binary_mask`, `open_binary_mask`, `close_binary_mask`, `remove_small_components`,
`_to_hw1`) — yet several sites re-implement the same primitives inline with cv2/scipy.
That is the core of the problem: a second, divergent set of morphology kernels.

Conventions to respect (from CLAUDE.md) when adding helpers: depth is uint16 mm on
disk / float32 m in memory; binary masks are conventionally `[H,W,1]` float in the
torch path; the four per-object identity buffers must stay index-aligned (a mask
helper must NOT reorder/subset Gaussian arrays — these helpers only operate on 2D
image masks, so that invariant is not at risk here); background sky `(0.86,0.92,1.0)`
is only relevant to Pattern 5 (compositing).

---

## Pattern 1 — cv2/scipy erode-dilate inline, duplicating `active_mask.py`

`active_mask.py` already provides `dilate_binary_mask` / `erode_binary_mask` (torch
`max_pool2d`, the `k = 2*r+1` kernel). These sites re-derive the SAME
`k = 2*px + 1; np.ones((k,k)); cv2.erode/dilate` (or scipy) by hand, on numpy
arrays, instead of calling the helper.

Sites (all verified same logic — odd kernel `2*px+1`, single iteration):
- `dynamic_gs/utils/xfeat_motion.py:1241-1244` — `_pre_mask_image`: `ksize = 2*erode_px+1; np.ones; cv2.erode` (eroded anchor mask).
- `dynamic_gs/utils/xfeat_motion.py:662-667` — object-halo `cv2.dilate` with `k = 2*r+1; np.ones`.
- `dynamic_gs/utils/tracker_common.py:159-161` — `_shrink_mask_for_sampling`: `kernel_size = 2*margin_px+1; np.ones; cv2.erode`.
- `dynamic_gs/utils/rgbd_decode.py:368-369` — `structure = np.ones((2*erode_px+1, ...)); _scipy_binary_erosion`.
- `dynamic_gs/utils/active_mask.py:697-701` — `build_active_mask_center_only`: inline `k = 2*dilate_px+1` + `F.max_pool2d` dilation, duplicating its own module-level `dilate_binary_mask`.

Risk: **correctness**. Two divergent dilate/erode implementations. The numpy/cv2
copies operate on bool/uint8 HxW arrays; the canonical helper on `[H,W,1]` float
torch tensors with `>0.5` thresholding. A future change to kernel shape, border
handling (cv2 BORDER_CONSTANT vs torch zero-pad), or radius convention would have
to be made in 5 places or they silently diverge — and several of these masks
(anchor pre-mask, object halo) directly gate which keypoints the tracker sees.

Proposed helper: extend `active_mask.py` with numpy-facing thin wrappers
`erode_mask_np(mask, px)` / `dilate_mask_np(mask, px)` (uint8/bool in, uint8/bool
out, the single `2*px+1` cv2 kernel) so the cv2/scipy callers share ONE kernel
definition; the torch callers keep using `erode_binary_mask`/`dilate_binary_mask`.
Then route all five sites through one of the two. (Or, where the array is already
torch, just call the existing torch helper.)

est_loc_saved: 18

---

## Pattern 2 — mask → bbox → padded square crop (clamped to image)

The "compute the tight bbox of a binary mask, expand to a centred square of
`side = max(w,h) + 2*pad` (with a min side), clamp the square to image bounds" recipe
is implemented three times, nearly line-for-line.

Sites (verified same algorithm):
- `dynamic_gs/utils/sam3d.py:331-347` — `prepare_cropped_sam3d_inputs`: `np.nonzero(mask); xs/ys.min/max; center; side = max(...) + 2*padding; side = max(side, 32, min_crop_side); clamp crop_x0/y0/x1/y1`.
- `dynamic_gs/dynamic_gs_pipeline_base.py:3178-3199` — `_anysplat_crop_windows`: `np.where(m>0); xs/ys.min/max; size = max(bw,bh) + 2*pad_px; size = max(16, size); left/top = clamp(round(c - size/2), 0, W-size)`.
- `dynamic_gs/dynamic_gs_pipeline_base.py:1856-~1935` — `_object_crop_bbox` + `_crop_for_xfeat`: same bbox-min/max + pad + clamp, but the bbox comes from *projected Gaussian centres* (rectangular, not forced square) rather than a binary mask — partial overlap (the pad-and-clamp tail is identical; the bbox source differs).

Risk: **maintenance** (verging on correctness). The square-crop math is fiddly
(the `crop_x0 = max(0, crop_x1 - side)` re-anchor after the first clamp in sam3d.py
exists specifically to keep a full `side×side` square against the image edge; the
pipeline copy uses `min(round(c - size/2), W-size)` to achieve the same). These two
formulations are equivalent only by careful reading — easy to "fix" one and break
the edge-touching case in the other. The crop drives what SAM3D / AnySplat actually
see, so a regression here silently degrades 3D reconstruction quality.

Proposed helper: `mask_bbox(mask) -> (x0,y0,x1,y1) | None` and
`square_crop_about_bbox(bbox, img_h, img_w, pad_px, min_side) -> (x0,y0,size)` in
`dynamic_gs/utils/active_mask.py` (or a new `mask_geom.py`). sam3d.py and
`_anysplat_crop_windows` call both; `_object_crop_bbox` can at least share
`mask_bbox` + the clamp tail.

est_loc_saved: 24

---

## Pattern 3 — projected-Gaussian-into-2D-mask subset helpers duplicated across the two models

`StaticGSModel` and `DynamicGSModel` carry byte-for-byte equivalent methods that
project Gaussian centres to pixels, squeeze the object mask to HxW, sample it, and
keep Gaussians inside-mask-and-near-the-rendered-depth. `static_gs_model.py`'s own
module docstring (line 26) says it "reuses" these — but the code is copied, not
shared.

Sites (verified near-identical bodies):
- Slab / cull-index helper: `static_gs_model.py:556-601` ↔ `dynamic_gs_model.py:1456-~1498` (`_select_sam3d_cull_indices` / `_get_object_slab_indices`-style). Same `mask = render_object_mask[...,0] if ndim==3 else ...`, same `in_bounds`/`isfinite`/`radii>0`/`cx,cy` bounds test, same `(proj_depth - sampled).abs() <= depth_tol_m`.
- CPD-target subset: `static_gs_model.py:603-701` ↔ `dynamic_gs_model.py:1502-1601` (`_get_existing_object_subset`). Same preamble, then the frontmost-per-pixel / depth-thin / downsample tail.

Risk: **correctness**. These select the registration/cull set for Phase 0b SAM3D
fusion. Two copies on two models that are explicitly meant to behave identically; a
fix to the depth-tolerance handling or the bounds test applied to one model and not
the other would make `static-gs` (which actually runs Phase 0b) and the dynamic
warm-load path diverge in how they define the object surface.

Proposed helper: a free function `select_gaussians_in_mask_near_depth(info,
num_points, render_object_mask, rendered_depth, depth_tol_m, ...)` (and a
`frontmost_object_subset(...)` for the stricter variant) in a shared module
(e.g. `dynamic_gs/utils/gaussian_projection.py`, alongside the already-shared
`extract_projected_centers_and_radii`). Both models call it.

est_loc_saved: 90

---

## Pattern 4 — "squeeze mask to HxW" / "ensure mask is [H,W,1]" sprinkled inline

The inverse pair of one-liners — `m = mask[...,0] if mask.ndim==3 else mask`
(squeeze) and `mask = mask[...,None] if mask.ndim==2 else mask` (expand) — is
written inline at many sites. `active_mask.py` already has `_to_hw1` for the expand
direction (private, underscored, not exported), but callers don't use it and there
is no squeeze counterpart.

Sites (verified — exact same idiom):
- Squeeze `[...,0]`: `static_gs_model.py:571, 572, 614, 615`; `dynamic_gs_model.py:1468, 1469, 1506, 1507`.
- Expand `[...,None]` / `ndim==2` guard: `dynamic_gs_model.py:1999-2000, 2031-2034`.
- `active_mask.py:29-32` — the canonical `_to_hw1` (expand) that the others should reuse.

Risk: **cosmetic** (low correctness risk — the idiom is trivial), but it is genuine
copy-paste and clutters the projection helpers; consolidating it falls out for free
once Pattern 3 is extracted (the shared helper does the squeeze once).

Proposed helper: export `to_hw1(mask)` and add `to_hw(mask)` (squeeze) in
`active_mask.py`; use inside the Pattern-3 helper so the per-call-site guards vanish.

est_loc_saved: 8

---

## Pattern 5 — mask-about-centroid scaling (single copy, NOT yet duplicated)

`_scale_mask_about_centroid` (warpAffine scale about the mask centroid, then union
with the original) exists ONCE: `dynamic_gs_pipeline_base.py:2437-~2467`, called at
:2414 and :2494. No second copy was found.

Risk: none today — reported only because the audit class names it and to flag that
it is currently a *method* on the pipeline base (uses `self` only for nothing
substantive — it imports cv2/numpy locally). If a future site needs it (e.g. the FF
object-footprint subtract in another module), promote it to a free function
`scale_mask_about_centroid(mask, scale)` in `active_mask.py` rather than copying.
No consolidation action needed now.

est_loc_saved: 0

---

## Notes on what was checked and is NOT duplication

- `combine_object_masks` (`active_mask.py:216`) — single definition, used via import. Not duplicated.
- The bbox dicts in SAM/FastSAM (`sam3_segmentation.py`, `fastsam_segmentation.py`,
  `sam_worker.py`, `compare_sam3_fastsam.py`) share a `{mask_path, score, bbox, mask_area, object_index}`
  output *contract*, but each computes bbox from its own backend's boxes/components
  — `fastsam_segmentation.py:213` and `:262` DO re-derive bbox from `np.nonzero` like
  Pattern 2's `mask_bbox`, so those two lines also benefit from the Pattern-2
  `mask_bbox` helper (folded into that estimate).
- K-intrinsics rescale-with-resolution (`dynamic_gs_model.py:2156`,
  `dynamic_gs_pipeline_base.py:2810`, `fusion/phase0.py:144,219`) is a related but
  distinct idiom (camera intrinsics, not masks) — out of scope for this class.
- `render_object_mask` exists once (`dynamic_gs_model.py:2132`); the `static_gs_model`
  references are parameter names, not a second implementation.
