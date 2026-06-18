# Code Audit — Change Detection module

Files audited:
- `dynamic_gs/utils/active_mask.py` (716 LOC) — morphology, MS-SSIM / depth scoring, `build_change_mask`, active-mask projection.
- `dynamic_gs/change_detection/change_mask.py` (290 LOC) — `ChangeMaskConfig`, `compute_change_mask` (the high-level shim entry).

LIVE path (per-tick CDN): `dynamic_gs_pipeline_base.py::_compute_change_mask` (shim) → `change_mask.compute_change_mask` → `active_mask.build_change_mask` (`mode="rgb"`) → `_rgb_msssim_score` + cleanup. Grep evidence cited inline; ref counts are "outside the symbol's own `def`/`class`".

---

## 1) FUNCTION / CLASS MAP

### `dynamic_gs/utils/active_mask.py`

- `_to_hw1(mask)` — :29 — Coerce a `[H,W]` or `[H,W,1]` mask to float `[H,W,1]`. — 17 refs (file-internal helper used by nearly every fn here). LIVE (transitively).
- `dilate_binary_mask(mask, radius)` — :35 — Max-pool binary dilation. — 12 refs incl. `change_mask.py:21,282`, `dynamic_gs_model.py`, `dynamic_gs_pipeline_base.py`. LIVE.
- `erode_binary_mask(mask, radius)` — :47 — Inverted max-pool erosion. — 4 refs; external caller `change_mask.py:166` (gripper-erode), internal in `open_binary_mask`. LIVE.
- `open_binary_mask(mask, radius)` — :59 — erode∘dilate (speckle removal). — 2 refs, both INTERNAL (`combine_object_masks:221`, `_apply_cleanup_recipe:503`). No external caller.
- `close_binary_mask(mask, radius)` — :65 — dilate∘erode (hole fill). — 2 refs, both INTERNAL (`combine_object_masks:220`, `_apply_cleanup_recipe:501`). No external caller.
- `remove_small_components(mask, min_area)` — :71 — scipy-label drop of small CCs. — 2 refs; only real call is INTERNAL `combine_object_masks:222` (other is a docstring). No external caller.
- `keep_largest_component(mask)` — :93 — Keep the single largest CC. — 10 refs incl. `dynamic_gs_model.py:1992`, `static_gs_model.py`, `phase0.py`. LIVE-adjacent (model side; the 1992 call lives in the dead `prepare_dynamic_update`, but the symbol has live callers elsewhere).
- `keep_all_components_above_min_area(mask, min_area)` — :112 — Drop CCs < min_area, keep ALL survivors (multi-blob). — 1 ref: INTERNAL only (`_apply_cleanup_recipe:507`). No external caller; reachable in live via `keep_largest_only=False` path.
- `keep_largest_component_with_min_area(mask, min_area)` — :135 — Largest CC if ≥ min_area else empty. — 4 refs: 2 are `utils/__init__.py` export rows, 1 is its own docstring, 1 is INTERNAL (`_apply_cleanup_recipe:505`). No call site outside this file other than the re-export.
- `select_top_n_components_filtered(mask, n, area_ratio, min_area)` — :167 — Up-to-n largest CCs as a list of per-CC masks (FF decode regions). — 11 refs incl. `dynamic_gs_pipeline_base.py`. LIVE (FF region selection).
- `combine_object_masks(render_mask, live_mask, valid_mask=None)` — :216 — Union render+live object masks → cleaned optim mask. — 4 refs: `dynamic_gs_model.py:31`(import),`:2052`(call inside DEAD `prepare_dynamic_update`), `utils/__init__.py` import+export. The only call site is dead-by-transitivity (see §2).
- `_gaussian_blur_image(image, k, sigma, valid_mask=None)` — :230 — Mask-weighted separable Gaussian blur (holds out invalid pixels). — 7 refs, all INTERNAL to this file (`_rgb_msssim_score`, `_depth_diff_score`, `_depth_outlier_score`). LIVE (via rgb score).
- `_ssim_map(gray_pred, gray_gt, k=11, sigma=1.5)` — :276 — Windowed per-pixel SSIM map. — 1 ref: INTERNAL (`_rgb_msssim_score:353`). LIVE.
- `_rgb_msssim_score(pred,gt,valid_mask,...,pyramid_weights)` — :297 — 3-level coarse-weighted MS-SSIM dissimilarity. — 4 refs: 1 internal call (`build_change_mask:556`) + docstring/comment mentions. LIVE (the active per-tick scorer).
- `_depth_diff_score(pred_depth,gt_depth,...)` — :373 — Per-pixel |Δdepth| metres. — 1 ref: INTERNAL `build_change_mask:569` (`mode="depth"` branch). UNREACHABLE in live (mode is always "rgb"; see §2).
- `_depth_outlier_score(pred,gt,...,median_multiplier,min_threshold_m)` — :419 — Robust bidirectional median-gated depth-outlier score (GaME-style). — 2 refs: INTERNAL `build_change_mask:586` (`mode="depth_outlier"`) + docstring. UNREACHABLE in live.
- `_threshold_mask(score, valid_mask, threshold)` — :490 — Threshold a score → binary `[H,W,1]`, re-AND valid_mask. — 2 refs: INTERNAL (`build_change_mask:598`). LIVE.
- `_apply_cleanup_recipe(mask, valid_mask, close_radius, open_radius, min_area, keep_largest_only)` — :498 — Close/open/CC-filter the thresholded mask; empty→empty (no raw fallback). — 1 ref: INTERNAL (`build_change_mask:599`). LIVE.
- `build_change_mask(pred_depth,gt_depth,pred_rgb,gt_rgb,valid_mask,...,mode,...)` — :521 — Dispatch rgb/depth/depth_outlier scorer → threshold → cleanup. — 12 refs; LIVE caller is `change_mask.py:252` (via `compute_change_mask`). The `dynamic_gs_model.py:1968` caller is DEAD (inside `prepare_dynamic_update`).
- `extract_projected_centers_and_radii(info, num_points)` — :618 — Read `means2d`/`radii` from gsplat info, shape-validate. — 13 refs incl. `static_gs_model.py:570,613`, `dynamic_gs_pipeline_base.py:2749`. LIVE.
- `build_active_mask(mask, centers_2d, radii)` — :649 — Mark Gaussian active if projected footprint (integral-image rect) overlaps mask. — 11 refs incl. `dynamic_gs_pipeline_base.py:2760`, `dynamic_gs_model.py:1848` (live), `:2063,:2077` (dead path). LIVE.
- `build_active_mask.rect_sum(...)` — :668 — Local integral-image rectangle sum closure. — internal to `build_active_mask`. LIVE.
- `build_active_mask_center_only(mask, centers_2d, dilate_px=0)` — :680 — Mark Gaussian active iff its projected 2D CENTER lands in mask. — 2 refs: BOTH are `utils/__init__.py` (import row + `__all__` entry). NO CALL SITE anywhere. DEAD (see §2).

### `dynamic_gs/change_detection/change_mask.py`

- `class ChangeMaskConfig` — :25 — Dataclass of CDN thresholds + cleanup knobs (mirrors `change_mask_*` model-config fields). — 7 refs; instantiated at `dynamic_gs_pipeline_base.py:1823`. LIVE.
- `_resize_mask_to(mask, target_h, target_w)` — :82 — Nearest-resize a `(H,W,C)` mask. — 4 refs, all INTERNAL to `compute_change_mask`. LIVE.
- `resolve_downsample_factor(rgb_or_shape, configured_factor, target_side)` — :93 — Auto-scale MS-SSIM downsample so it runs on ~target_side² px. — 4 refs: export rows + `dynamic_gs_pipeline_base.py:1811,1818`. LIVE.
- `compute_change_mask(*, rendered_rgb, rendered_depth, live_rgb, gt_depth, gripper_mask, object_mask, config, downsample_factor=1, keep_largest_only=True, rendered_alpha=None)` — :114 — Top-level CDN: build valid_mask (object/gripper/coverage-gate), masked-avg-pool downsample, `build_change_mask`, upsample, dilate. — 12 refs; LIVE caller `dynamic_gs_pipeline_base.py:1838`.
- `compute_change_mask._avg_pool / _masked_avg_rgb / _masked_depth` — :204/:214/:220 — Local pooling closures (downsample path). — internal. LIVE.

---

## 2) DEAD-CODE CANDIDATES

All verified by repo-wide grep over `dynamic_gs/` + `scripts/` (and full-repo for the cross-module ones). Entry points / invariant buffers excluded per instructions.

| Symbol | file:line | Evidence (ref count) | Confidence |
|---|---|---|---|
| `build_active_mask_center_only` | active_mask.py:680 | 2 refs, BOTH in `utils/__init__.py` (import + `__all__`). `grep -rn build_active_mask_center_only` over `dynamic_gs`+`scripts`+`.` (excl third_party) → ZERO call sites. | **high** |
| `_depth_diff_score` | active_mask.py:373 | 1 ref (`build_change_mask:569`, `mode=="depth"`). `change_mask_mode` default `"rgb"` (dynamic_gs_model.py:77), never reassigned, no env override (grep for `DGS_CDN_MODE`/`change_mask_mode=` → none). The only live `build_change_mask` caller passes `mode="rgb"`. UNREACHABLE. | **medium** (reachable only if someone flips `change_mask_mode`) |
| `_depth_outlier_score` | active_mask.py:419 | 1 call ref (`build_change_mask:586`, `mode=="depth_outlier"`) + docstring. Same reasoning as `_depth_diff_score` — `mode` is always `"rgb"`. UNREACHABLE in live. | **medium** |
| `combine_object_masks` | active_mask.py:216 | Sole call site is `dynamic_gs_model.py:2052`, inside `prepare_dynamic_update` — which has ZERO real callers (full-repo grep: only its own return-dict key + doc-comments in `esam.py:251`, `static_gs_model.py:27`). Dead-by-transitivity. | **medium** (live-dead; symbol revives if `prepare_dynamic_update` ever wired back) |
| `dynamic_gs_model.py:1968 build_change_mask` call | dynamic_gs_model.py:1968 | This is the model's OWN CDN call, inside the dead `prepare_dynamic_update`. The LIVE CDN goes through `compute_change_mask`. So the model-side `build_change_mask` / `keep_largest_component`(:1992) / `combine_object_masks` usages are all dead-by-transitivity. (Reported here because it makes `build_change_mask`'s `mode`-dispatch surface reachable from a dead path only.) | **medium** |

Borderline (kept OUT of the high-confidence dead list — single internal caller but on the live path):
- `open_binary_mask`, `close_binary_mask`, `remove_small_components`, `keep_all_components_above_min_area`, `keep_largest_component_with_min_area`, `_ssim_map`, `_threshold_mask`, `_apply_cleanup_recipe`, `_gaussian_blur_image`, `_resize_mask_to` — each has exactly one (or few) callers but those callers are LIVE (`build_change_mask` / `compute_change_mask` / `combine_object_masks`). NOT dead. They are private helpers with a single live consumer; consolidation candidates, not removal candidates.
- `keep_largest_component_with_min_area` — only `_apply_cleanup_recipe:505` calls it (live, when `keep_largest_only=True`). The default per-tick FF path passes `keep_largest_only=False`, so in the live FF dispatcher this branch is the OFF one — but `keep_largest_only=True` is still hit by the static-convergence CDN check. NOT dead.

`build_change_mask` dead PARAMETERS (not symbols): `use_rgb`, `filter_radius`, `min_component_size` are accepted then immediately discarded — `del use_rgb, filter_radius, min_component_size` at active_mask.py:548. They are still threaded all the way from model config → `ChangeMaskConfig` → `compute_change_mask` → here, then dropped. See §4.

---

## 3) DATA-LIFECYCLE

These two modules are **pure / stateless** — no `.pt` warm-cache I/O, no SHM create/attach/free, no process handles, no persistent identity-buffer writes. (`change_mask.py:1` docstring asserts this; verified: no `torch.save/load`, no `multiprocessing.shared_memory`, no file handles in either file.) So most lifecycle hazards live in the callers, not here. What this module DOES touch:

- **GPU tensor allocations, per CDN call (not per tick — CDN is FF-gated, ~every 10 ticks):** `compute_change_mask` allocates, per invocation: `valid_mask` (HxWx1), resized object/gripper/coverage masks, `valid_chw`, two pooled RGB tensors, two pooled depth tensors, `valid_frac`/`valid_block`, then inside `build_change_mask` the MS-SSIM pyramid builds 3 SSIM maps + interpolations + two blurred RGB images. At 1920×1200 the full-res `_rgb_msssim_score` level-0 SSIM runs **before** any downsample benefit (the downsample is applied to the inputs in `compute_change_mask`, but `_rgb_msssim_score` then re-builds a 3-level pyramid on top). All are transient (freed at scope exit / next GC). No leak, but the level-0 full-grid `F.conv2d` SSIM on the already-downsampled grid is the per-call hot path. No `torch.no_grad()` wrapper inside either module — relies on the caller's `@torch.no_grad()` (`_compute_change_mask` is under `_render_*` paths; the live shim is fine, but `build_change_mask` called directly with grad enabled would retain the autograd graph across all the conv2d ops = a hidden per-call graph allocation). Flagged below.
- **CPU round-trips (scipy.ndimage.label):** `remove_small_components`, `keep_largest_component`, `keep_all_components_above_min_area`, `keep_largest_component_with_min_area`, `select_top_n_components_filtered` each do `binary.detach().cpu().numpy()` → scipy label → `torch.from_numpy(...).to(device)`. This is a GPU→CPU→GPU sync per call. `_apply_cleanup_recipe` (live) calls exactly one of these per CDN. `build_change_mask` cleanup → 1 scipy round-trip; the dead `dynamic_gs_model.py:1992` adds a second when dilation is on. No leak; it's a latency / sync cost (documented in `keep_largest_component_with_min_area`'s docstring as "the dominant cost on 800×800 masks").
- **No buffer desync risk introduced here** — the 4 identity buffers (`object_flags` etc.) are NOT read or written in either file. `combine_object_masks`/`build_active_mask` consume *masks* derived from them upstream; they return tensors, never mutate model state. (The mutation lives in the dead `prepare_dynamic_update` and in the live model methods, out of scope.)
- **Config→ChangeMaskConfig field plumbing (potential silent desync, not a leak):** `_compute_change_mask` (pipeline_base.py:1823) constructs `ChangeMaskConfig` from `model.config` but **omits `block_valid_min_frac`, `live_depth_min_m`, `live_depth_max_m`**. Those three always fall back to the dataclass defaults (0.5 / 0.05 / 3.0). CLAUDE.md states this is intentional ("`ChangeMaskConfig` defaults, NOT wired to model config"), so it is a *known* desync surface: tuning the model config for these does nothing, and the live coverage-gate band (`live_depth_max_m=3.0`) is hardcoded to differ from the fusion `DEPTH_MAX_M=2.0` — flagged in §4 as a data-lifecycle / config-consistency item below.

### Data-lifecycle issues (structured)

- **MEDIUM — `build_change_mask` builds the full autograd graph if a caller forgets `no_grad`** — active_mask.py:521 — Neither `build_change_mask` nor `compute_change_mask` wraps its conv2d / SSIM / avg_pool chain in `torch.no_grad()`. The live shim path is safe (caller is no-grad), but the contract is implicit; a non-no_grad caller (e.g. a future eval probe) silently retains a multi-conv autograd graph per call = per-call GPU graph allocation that won't free until backward/zero_grad. Make the public entry self-defensive.
- **LOW — coverage-gate `live_depth_max_m=3.0` desyncs from fusion `DEPTH_MAX_M=2.0`** — change_mask.py:56 / pipeline_base.py:1823 — `ChangeMaskConfig.live_depth_max_m` defaults to 3.0 and is NOT wired from model config, while `online_fusion.DEPTH_MAX_M` is now 2.0. The 2–3 m band is kept as a "fillable hole" by the coverage gate but discarded by fusion/static-mask — pixels CDN flags there can never be filled consistently. Already noted in CLAUDE.md; harmless but a standing inconsistency.
- **LOW — per-CDN GPU↔CPU sync via scipy.label** — active_mask.py:82,101,123,153,187 — Every CC-filter helper round-trips to CPU. One per live CDN today; if a future change runs CDN per-tick this becomes a per-tick sync stall. Not a leak.

---

## 4) DESIGN SMELLS

- **MEDIUM — Dead parameters threaded through 3 layers** — active_mask.py:548 — `build_change_mask` accepts `use_rgb`, `filter_radius`, `min_component_size` then `del`s them on the first line. They are still passed from `dynamic_gs_model.py` config → `ChangeMaskConfig.use_rgb/filter_radius/min_component_size` → `compute_change_mask` (change_mask.py:260,263,264) → `build_change_mask`. Three config fields (`change_mask_use_rgb`, `change_mask_filter_radius`, `change_mask_min_component_size`) exist and are read ONLY to be discarded. Recommendation: drop the params + the three config fields + the plumbing.
- **MEDIUM — Two parallel CDN entry points with diverging signatures** — active_mask.py:521 vs change_mask.py:114 — `build_change_mask` (low-level) and `compute_change_mask` (high-level shim) both exist; the model also calls `build_change_mask` directly (dynamic_gs_model.py:1968, dead) with a *different* arg set (no coverage gate, no downsample, no block-validity). The live path only uses `compute_change_mask`. The direct-`build_change_mask` surface (and its depth/depth_outlier modes) is reachable only from dead code. Recommendation: after the `prepare_dynamic_update` purge, `build_change_mask` becomes a private helper of `compute_change_mask` and the `mode` dispatch can be inlined to rgb-only.
- **MEDIUM — `_rgb_msssim_score` re-pyramids on an already-downsampled grid** — active_mask.py:297 + change_mask.py:202 — `compute_change_mask` masked-avg-pools inputs by `downsample_factor` (≈10 at 1200p), then `_rgb_msssim_score` builds ANOTHER 3-level pyramid (full/½/¼ of the downsampled grid). The "full-res" SSIM band is therefore already ¹⁄₁₀ res; the docstring's "full-res reads sharpness mismatch" reasoning predates the auto-downsample and the band names are now misleading. Confusing naming + double-downsample.
- **LOW — Misleading docstring in `build_change_mask`** — active_mask.py:545 — Says "the cleanup recipe (c10_o3_a760)" but `OFFICIAL_FILTER_MIN_AREA` was lowered to 76 (active_mask.py:20). The `a760` literal in the docstring is stale (CLAUDE.md rule 1 — doc must match code).
- **LOW — Three config fields documented as "NOT wired"** — change_mask.py:55,72 — `block_valid_min_frac`, `live_depth_min_m`, `live_depth_max_m` are `ChangeMaskConfig` defaults the shim never sets. Intentional per CLAUDE.md, but it means three live-behaviour knobs are editable only by editing the dataclass default, not the model config — a leaky abstraction (config dataclass claims to "mirror `change_mask_*`" at change_mask.py:28 but three fields don't).
- **LOW — `ChangeMaskConfig` docstring placement bug** — change_mask.py:41-54 — Two bare triple-quoted strings (the `scene_coverage_threshold` explainer and the `mode` explainer) sit AFTER `outlier_min_threshold_m` and BEFORE `live_depth_min_m`, so they are NOT attached as the docstring of any field (they're orphan string expressions). The `scene_coverage_threshold` field at :36 has no attached doc; the long explainer at :41 floats free. Cosmetic but confusing — the field-to-doc mapping is wrong.
- **LOW — Naming collision: module `active_mask.py` vs the variable `active_mask`** — active_mask.py / pipeline_base.py:2760 — The file is named for `build_active_mask`, but it actually hosts the entire change-mask scorer (`build_change_mask`, MS-SSIM, depth scores). Half the module is "change detection", half is "active mask projection" — two unrelated concerns in one file. The real change-detection home (`change_detection/change_mask.py`) imports back into `utils/active_mask.py`, an inverted dependency.
- **LOW — `_apply_cleanup_recipe` `keep_largest_only` default mismatch** — active_mask.py:498 — Defaults to `True`, but the live FF path always overrides to `False`; `build_change_mask` defaults it to `True` too (active_mask.py:534). The "default" is the non-live behaviour, which invites mistakes when called fresh.

---

### Note on invariant-protected items
None of the 4 identity buffers, the 3 `__init__.py` monkeypatches, means-LR=0, `_ZERO_LR_OPTIMIZERS`, or the writer-with-no-caller (`initialize_object_from_sam3d`) are referenced in either audited file. Nothing here was flagged against those. `combine_object_masks`/`build_change_mask` dead-by-transitivity findings hinge on `prepare_dynamic_update` being dead — that is a model-side method, not an invariant-protected symbol (confirmed zero real callers full-repo).
