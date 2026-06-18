# Code Audit — `dynamic_gs/utils/rgbd_decode.py`

Module purpose: closed-form per-pixel RGB-D back-projection over a 2D change-component
mask, producing a frozen per-pixel Gaussian batch for the feedforward (FF) hole-fill
path. Pure GPU compute, no model state, no checkpoint, no subprocess. It is the
**non-default** FF backend (`enable_feedforward_inpaint` defaults to `"anysplat_decode"`,
see `dynamic_gs_pipeline_base.py:176`); the `"rgbd_decode"` branch is selected only when
that config field is set to `"rgbd_decode"`.

Important context for the audit: **this module does NOT touch any persistent state.**
`decode_component_to_gaussians` returns a dict of CPU/GPU tensors; the *caller*
(`_run_feedforward` in `dynamic_gs_pipeline_base.py`) is what mutates the model,
identity buffers, viser, and timing. So the thread-safety / lifecycle concerns live at
the call boundary, not inside this file. Findings reflect that.

---

## 1) FUNCTION / CLASS MAP

(No classes; module is all module-level functions.)

- `_camera_intrinsics(target_camera) -> (fx,fy,cx,cy,width,height)` — `rgbd_decode.py:41`
  — Extracts scalar intrinsics + width/height from a single-frame nerfstudio `Cameras`,
  coercing tensor fields to python floats.
  Callers: 1 internal — `rgbd_decode.py:293` (`decode_component_to_gaussians`). The
  `scripts/debug_rgbd_decode.py:59` and `live_ros_publisher.py:473/478` hits are a
  docstring mention and an unrelated `dgs_camera_intrinsics.json` cache path
  (substring collision, NOT a call). **Effectively internal-only.**

- `_camera_c2w(target_camera) -> Tensor(3,4)` — `rgbd_decode.py:58`
  — Returns the camera-to-world matrix, squeezing a leading batch dim if present.
  Callers: 1 internal — `rgbd_decode.py:294`. The `debug_rgbd_decode.py:59` hit is a
  docstring. **Internal-only.**

- `_bilateral_smooth_depth(depth_m, valid_mask, radius, sigma_space=2.0, sigma_depth=0.02) -> Tensor` — `rgbd_decode.py:67`
  — CPU `cv2.bilateralFilter` of depth in metres, holding out invalid (==0) pixels;
  used to produce a smooth depth grid for normal estimation.
  Callers: 1 internal — `rgbd_decode.py:406`. **Internal-only.**

- `_backproject_world(u, v, depth_z, fx, fy, cx, cy, c2w) -> Tensor(N,3)` — `rgbd_decode.py:91`
  — OpenGL/Nerfstudio-convention per-pixel back-projection to world XYZ.
  Callers: 1 internal — `rgbd_decode.py:434`. NOTE: a *different* `_backproject_world`
  exists in `rgbd_fusion_init.py:104` (different signature, OpenCV `c2w_cv`); the two
  are unrelated module-local helpers that share a name. **Internal-only here.**

- `_rotmat_to_wxyz(R) -> Tensor(N,4)` — `rgbd_decode.py:119`
  — Numerically-stable rotation-matrix → wxyz quaternion (largest-component branch).
  Callers: 1 internal — `rgbd_decode.py:456`. **Internal-only.**

- `_normals_from_xyz_grid(xyz_grid) -> Tensor(H,W,3)` — `rgbd_decode.py:180`
  — Per-pixel surface normals via central differences on a per-pixel XYZ grid.
  Callers: 1 internal — `rgbd_decode.py:422`. **Internal-only.**

- `_rotation_from_normal(normals) -> Tensor(N,3,3)` — `rgbd_decode.py:204`
  — Builds per-pixel orthonormal frames with z-column = surface normal (Gram-Schmidt).
  Callers: 1 internal — `rgbd_decode.py:455`. **Internal-only.**

- `decode_component_to_gaussians(target_camera, live_rgb, gt_depth_m, component_mask, *, opacity=0.99, normal_smoothing_radius=3, min_valid_fraction=0.95, thin_axis_ratio=0.25, scale_multiplier=5.0, cliff_threshold_m=0.05, post_cliff_erode_px=1, rendered_depth_m=None, leak_threshold_m=0.02) -> Optional[dict]` — `rgbd_decode.py:225`
  — Public entry point. Validates shapes, applies valid-fraction / leak-cut / cliff-cut
  gates, computes normals, back-projects each cliff-split sub-component into per-pixel
  Gaussians, returns a dict (`xyz/features_dc/features_rest/opacities/scales/quats/diagnostics`)
  or a `{"skipped":True,...}` / `None`.
  Callers: 2 real — `dynamic_gs_pipeline_base.py:2643` (live + recorded FF dispatcher,
  via `from .utils.rgbd_decode import ...` at `:2568`) and `scripts/debug_rgbd_decode.py:210`
  (offline debug tool). Re-exported from `dynamic_gs/utils/__init__.py:12,52`.
  **Live-path entry point (when `enable_feedforward_inpaint=="rgbd_decode"`).**

---

## 2) DEAD-CODE CANDIDATES

No genuinely dead symbols. Every private helper has exactly one internal caller, and
`decode_component_to_gaussians` is the live FF entry point (config-gated) plus the
public re-export and an offline debug driver. All grep counts above are non-zero.

Two near-dead notes (NOT recommended for removal without owner sign-off):

- `thin_axis_ratio` parameter (`rgbd_decode.py:234`) — has **no corresponding config
  field**. `_run_feedforward` (`dynamic_gs_pipeline_base.py:2643-2655`) never passes it,
  so it is permanently the default `0.25` in the live path; only `debug_rgbd_decode.py`
  could override it. It is read internally (`rgbd_decode.py:446`), so it is live, but it
  is an un-plumbed knob — call it a latent-but-not-dead parameter.

- `sigma_space` / `sigma_depth` params of `_bilateral_smooth_depth` (`rgbd_decode.py:71-72`)
  — never overridden by the one caller (`rgbd_decode.py:406`), always defaults. Live but
  unconfigurable.

(Module-level fallback flags `_HAS_CV2` / `_HAS_SCIPY` and the `cv2` / `scipy` import
guards are used — `rgbd_decode.py:79,348` — not dead.)

---

## 3) DATA-LIFECYCLE

This module is **stateless**: no `.pt` warm-cache, no SHM, no identity buffers
(`object_flags` / `object_instance_ids` / `sam3d_init_target_flags` / `inserted_flags`),
no model handle, no file handles, no process handles. It allocates transient GPU/CPU
tensors per call and returns them; ownership transfers to the caller. Lifecycle issues
are therefore about (a) per-tick allocation cost on a background thread, and (b) the
contract handed to the caller.

- **Per-call CPU↔GPU round-trips** — `rgbd_decode.py:82-88` (`_bilateral_smooth_depth`),
  `:349-350,387` (cliff cut). `_bilateral_smooth_depth` copies the *entire* HxW depth +
  validity to numpy and back every call; the cliff cut copies depth + valid to numpy,
  does the scipy `label`/`binary_erosion`, then copies each sub-mask back to GPU. At
  1920×1200 these are multi-MB host transfers **per component, per FF call** (the cliff
  block runs once but `_bilateral_smooth_depth` is run once too — both on the full frame).
  This is the dominant heap/transfer churn and runs on the FF background thread. Not a
  leak (all transient), but allocation-heavy on the hot live path. (medium)

- **Per-component `.item()` host syncs** — `rgbd_decode.py:276,281,322,324`. Each `int(...sum().item())`
  forces a CUDA sync. `total_count`/`valid_count` syncs are unavoidable for the gate, but
  `leak_dropped = int(valid_mask.sum().item() - new_valid.sum().item())` (`:322`) is a
  pure-diagnostic value computed with two extra syncs on every leak-enabled call. (low)

- **Returned tensors carry the caller's dtype, not necessarily float32** — `rgbd_decode.py:257`
  sets `dtype = live_rgb.dtype if floating else float32`. All outputs (`xyz`, `scales`,
  `quats`, `opacities`, …) inherit this. If `live_rgb` is ever float16, the FF batch is
  float16 and `insert_inpaint_gaussians` (`dynamic_gs_model.py:1150`) concatenates it
  into the model's gauss_params. Today `composite_with_background` upstream
  (`dynamic_gs_pipeline_base.py:2584`) yields float32 so it's fine, but the dtype is an
  implicit cross-module contract with no assertion. (low)

- **Output-dict shape contract** — `rgbd_decode.py:438` hardcodes
  `sh_degree_max_coeffs = 15` ("matches DynamicGSModelConfig.sh_degree=3"). `features_rest`
  is `(N,15,3)`. If `sh_degree` is ever changed on the model config, this silently
  produces a mismatched-shape `features_rest` that `insert_inpaint_gaussians` would
  concatenate against the model's `(N, K, 3)` → shape error or silent corruption. This is
  a **save/load-adjacent format coupling** (the inserted Gaussians persist into
  `post_fusion_state.pt` via the model later). Magic constant, no runtime check against
  the actual model. (medium)

- **No GPU frees / no caching** — every call rebuilds `vs,us` meshgrid (`:410`),
  `dir_grid`, `rays_d_un_grid`, `xyz_smooth_grid`, `normals_grid` over the FULL HxW frame
  (`:417-422`) even though only masked pixels are used. These are freed by GC after the
  call; no leak, but full-frame normal grids are computed even for a tiny component.
  (low → efficiency)

- **Skip/None contract is three-valued** — returns `None` (empty / shape-reject early at
  `:278`), `{"skipped":True,...}` (gated), or a full dict. The caller handles all three
  (`dynamic_gs_pipeline_base.py:2659,2664`). Consistent — no desync — but the tri-state
  return is a mild lifecycle smell (see §4).

No double-loads, no missing frees of persistent resources (there are none), no buffer
desync (this module owns no buffers).

---

## 4) DESIGN SMELLS

- **God function** — `decode_component_to_gaussians` (`rgbd_decode.py:225-505`) is ~280
  lines doing: shape validation, three independent pixel-rejection gates (valid-fraction,
  leak, cliff+resegment), full-frame normal estimation, per-sub-component back-projection,
  Gaussian-param synthesis, and diagnostics assembly. The cliff-cut block (`:344-401`) and
  leak-cut block (`:303-335`) are each self-contained passes that could be helpers; their
  inlining makes the early-return matrix hard to follow (5 distinct `return {"skipped":...}`
  sites with subtly different diagnostics keys). (medium)

- **Tri-state return** — mixing `None` and `{"skipped":True}` for "nothing produced"
  forces every caller to check both (`dynamic_gs_pipeline_base.py:2659` and `:2664`). One
  sentinel would be cleaner. Minor; the contract is documented in the docstring. (low)

- **`thin_axis_ratio` declared but never config-plumbed** — `rgbd_decode.py:234`. It is a
  real knob (used at `:446`) but no `feedforward_rgbd_thin_axis_ratio` config field exists,
  so the live path can never tune it. Either wire it or drop it from the signature.
  Inconsistent with the other six `feedforward_rgbd_*` fields which ARE plumbed
  (`dynamic_gs_pipeline_base.py:212-220`). (low)

- **Default-value drift between signature and caller** — the function defaults
  `scale_multiplier=5.0` and `leak_threshold_m=0.02` (`rgbd_decode.py:235,239`), but the
  live caller passes `feedforward_rgbd_scale_multiplier=1.0`
  (`dynamic_gs_pipeline_base.py:220`) and `feedforward_rgbd_leak_threshold_m=0.01`
  (`:215`). So the signature defaults are misleading documentation — they are never the
  live values and only the offline debug script could see them. (low)

- **Magic constant `sh_degree_max_coeffs = 15`** — `rgbd_decode.py:438`, with a comment
  asserting it "matches DynamicGSModelConfig.sh_degree=3" but no programmatic link. See
  §3; this couples output shape to a config the function can't see. (medium)

- **Swallowed import exceptions** — `rgbd_decode.py:28,37` catch bare `Exception` for
  `cv2` / `scipy` and silently flip a `_HAS_*` flag. When scipy is absent the cliff cut is
  silently skipped (`:348` falls to the `else: sub_component_masks=[valid_mask]` at `:401`)
  — a meaningful behavior change (no cliff/normal cleanup, no resegment) with **no log
  warning**. On a machine missing scipy the FF would degrade quietly. Both are
  `# pragma: no cover` so the degraded path is untested. (low — both are hard deps
  elsewhere, so unlikely, but silent.)

- **Normal-orientation sign hint is a global heuristic** — `_normals_from_xyz_grid`
  (`rgbd_decode.py:199`) flips ALL normals based on the *mean* z of the whole grid, with a
  self-aware comment that it "won't matter at insert time." For a frame containing both
  front- and back-facing surfaces this picks one global sign; benign for the symmetric
  Gaussian (the splat is two-sided) but the comment admits the logic is a guess. Leaky
  reasoning baked into geometry. (low)

- **Misleading docstring vs code on cliff "thickness"** — comments at `:362-364` say the
  cliff is dilated to 2 px on both neighbours; combined with `post_cliff_erode_px` default
  1 and the 8-pixel minimum (`:384`), small revealed slivers can be entirely consumed and
  silently dropped (`:389` returns skipped). Behavior is intentional but the interaction
  of three thresholds (cliff/erode/min-8) is undocumented as a combined gate. (low)

- **Thread-safety (call-boundary, not this file):** `decode_component_to_gaussians` itself
  is pure/reentrant — it reads only its arguments and allocates locals, so concurrent FF
  bg-thread + tracker-tick + viser-render is safe *within* this function. The actual race
  surface is entirely in the caller: `insert_inpaint_gaussians` + cull + viser register
  are wrapped in `self._viser_lock_ctx()` (`dynamic_gs_pipeline_base.py:2690,2699`), and
  the decode runs lock-free (correct — it touches no shared state). No finding against this
  module; flagged here so the synthesis doesn't expect a lock inside `rgbd_decode.py`.

- **Branch reachability in live mode:** the whole module is reachable in live mode ONLY
  when `enable_feedforward_inpaint == "rgbd_decode"`; the shipped default is
  `"anysplat_decode"` (`dynamic_gs_pipeline_base.py:176`), which short-circuits at
  `:2561` before importing this module. So in the default live configuration this entire
  file is **unreachable** — relevant given the module is slated for purge. Not dead
  (config-selectable + offline debug tool), but dormant on the default path. (informational)
