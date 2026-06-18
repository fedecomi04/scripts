# Duplication audit — Gaussian tensor-surgery

Scope: subsetting / concatenating the 6 `gauss_params` + 4 identity buffers in lockstep,
`delete_gaussian_indices` / insert, Adam-optimizer-state refresh, log-scale↔metres
conversions, opacity sigmoid thresholding. These MUST stay in sync — drift here is a
correctness bug (silently mismatched per-Gaussian arrays, stale Adam `m`/`v` state, or a
buffer that desyncs from `gauss_params`).

Conventions respected by all proposed helpers (per CLAUDE.md):
- The 6 param names are always processed in the fixed order
  `["means", "features_dc", "features_rest", "scales", "quats", "opacities"]`.
- The 4 identity buffers (`object_flags`, `sam3d_init_target_flags`, `object_instance_ids`
  (long), `inserted_flags`) are owned by specific phases (Invariant #8) — a shared helper
  must subset/resize them in lockstep, never reorder or drop one. `DynamicGSModel` also
  carries a 5th non-persistent `current_active_mask` (bool, 1-D) that participates in its
  delete/resize but is absent from `StaticGSModel`.
- Scales are stored as natural-log metres; opacity stored as logit; render thresholds use
  `sigmoid`. `background_rgb = (0.86, 0.92, 1.0)`.

---

## Pattern A — `delete_gaussian_indices` (whole method duplicated across the two models)

`StaticGSModel.delete_gaussian_indices` and `DynamicGSModel.delete_gaussian_indices` are
the same method copy-pasted, differing ONLY in that the dynamic one also subsets the extra
`current_active_mask` buffer. Identical: index dedup/bounds-clip, `keep` bool-mask build,
the 6-param `for name in [...]: detach()[keep] → nn.Parameter` loop, the 4-buffer
`self._buffers[...] = ...[keep]` block, the `_refresh_gaussian_optimizers(reset_means_optimizer=True)`
call, and the `n_deleted` return.

Sites:
- `dynamic_gs/static_gs_model.py` (`StaticGSModel.delete_gaussian_indices`) — lines 508–536
- `dynamic_gs/dynamic_gs_model.py` (`DynamicGSModel.delete_gaussian_indices`) — lines 1116–1147

Risk: **correctness.** If one model's param/buffer list is edited (e.g. a new identity
buffer is added, or the param order changes) and the other is not, the surviving Gaussians
get mismatched per-Gaussian arrays — a tracked object's `object_instance_ids` could point
at the wrong splats. Already drifted once: the dynamic copy has `current_active_mask`, the
static copy does not.

Proposed helper: a free function
`subset_gaussians_in_place(model, keep: Tensor)` in a new
`dynamic_gs/utils/gaussian_surgery.py`. It slices the 6 `gauss_params` (re-wrapping as
`nn.Parameter` preserving `requires_grad`) and every registered identity buffer the model
declares via a small `model._identity_buffer_names()` list (so the static/dynamic
difference is data, not code), then calls `model._refresh_gaussian_optimizers(...)`.
`delete_gaussian_indices` in both models reduces to: validate indices → build `keep` →
call the helper.

Est. LOC saved: ~25.

---

## Pattern B — insert / concatenate (`insert_object_gaussians`, `insert_inpaint_gaussians`)

Three insert methods share the same skeleton: stash `old_num_points`, run the 6-param
`for name in [...]: torch.cat([old.detach(), new[name]]) → nn.Parameter(...,
requires_grad=old.requires_grad)` loop, `self._resize_dynamic_buffers(self.num_points)`,
write the identity flags for the `[old_num_points:]` range (`object_flags`,
`object_instance_ids`, `inserted_flags`), `_refresh_gaussian_optimizers(reset_means_optimizer=True)`,
and `return torch.arange(old_num_points, self.num_points, ...)`.

Sites:
- `dynamic_gs/static_gs_model.py` (`StaticGSModel.insert_object_gaussians`) — lines 465–505
  (this copy does the cat WITHOUT preserving `requires_grad` — see risk)
- `dynamic_gs/dynamic_gs_model.py` (`DynamicGSModel.insert_object_gaussians`) — lines 1230–1257
- `dynamic_gs/dynamic_gs_model.py` (`DynamicGSModel.insert_inpaint_gaussians`) — lines 1200–1228
  (the cat + flag-write tail; the head differs — it takes pre-built tensors)

Risk: **correctness.** The static copy's concat uses `torch.nn.Parameter(concatenated)`
with no `requires_grad=` (lines 489–493), whereas both dynamic copies explicitly pass
`requires_grad=old_param.requires_grad` because they run inside `@torch.no_grad()` where
the `Parameter` default flips to False (documented at dynamic_gs_model.py:1209–1215).
This is exactly the kind of subtle divergence a shared helper prevents — the static path is
only safe today because it isn't called under `no_grad`.

Proposed helper: `concat_gaussians_in_place(model, new_tensors: dict) -> int` (returns
`old_num_points`) in `dynamic_gs/utils/gaussian_surgery.py`, doing the cat loop with
`requires_grad` preservation + `_resize_dynamic_buffers` + `_refresh_gaussian_optimizers`.
Each `insert_*` method builds `new_tensors`, calls the helper, then writes its own
identity-flag pattern over `[old_num_points:]`. The flag-write itself is small and
genuinely per-method (different default `instance_id` / `object_flag` semantics) so leave
it at the call sites.

Est. LOC saved: ~30.

---

## Pattern C — `_build_new_gaussian_tensors` (identical method on both models)

Byte-identical method: kNN-spacing → log-scale seed, `RGB2SH`/logit features_dc, zero
features_rest, identity quats, `logit(0.1)` opacity. Both copies bundle the same log-scale
seeding (`torch.log(avg_dist.repeat(1,3))`) and opacity-logit init.

Sites:
- `dynamic_gs/static_gs_model.py` (`StaticGSModel._build_new_gaussian_tensors`) — lines 428–463
- `dynamic_gs/dynamic_gs_model.py` (`DynamicGSModel._build_new_gaussian_tensors`) — lines 1082–1113

Risk: **maintenance** (leaning correctness). A change to the default scale/opacity seeding
in one model silently produces differently-initialized object inserts depending on which
phase ran the insert.

Proposed helper: `build_default_gaussian_tensors(model, new_xyz, new_rgb) -> dict` in
`dynamic_gs/utils/gaussian_surgery.py` (reads `model.config.sh_degree`,
`model.features_rest.shape`). Both methods delete their body and delegate.

Est. LOC saved: ~33.

---

## Pattern D — `_refresh_gaussian_optimizers` + `_resize_dynamic_buffers` (near-identical)

`_refresh_gaussian_optimizers`: the optimizer-state-clear loop (`param_groups[0]["params"]
= [gauss_params[name]]; optimizer.state.clear()`) + the `_optimizers_wrapper.parameters`
re-point loop are identical across both models; the dynamic copy appends three extra calls
(`register_hook(self._mask_means_grad)`, `_apply_phase_trainability()`,
`_apply_phase_optimizers(...)`).

`_resize_dynamic_buffers`: same lockstep "allocate-zeros-then-copy-leading-`keep`-entries"
for the 4 identity buffers; the dynamic copy adds `current_active_mask`. The static copy
factors a local `_resize(old, long=...)` closure; the dynamic copy inlines 5 explicit
`new_* = torch.zeros(...)` + `new_*[:keep] = ...` blocks — same logic, two spellings.

Sites:
- `dynamic_gs/static_gs_model.py` (`_refresh_gaussian_optimizers`) — lines 411–426
- `dynamic_gs/dynamic_gs_model.py` (`_refresh_gaussian_optimizers`) — lines 1063–1080
- `dynamic_gs/static_gs_model.py` (`_resize_dynamic_buffers`) — lines 381–409
- `dynamic_gs/dynamic_gs_model.py` (`_resize_dynamic_buffers`) — lines 1030–1061

Risk: **correctness.** These two are the actual "keep Adam state + the 4 buffers in sync
with the params" enforcers. The static `_resize` already diverged from the dynamic inlined
form; a buffer added to one resize-list but not the other desyncs after any insert/delete.

Proposed helper: a free `refresh_optimizer_params(model, *, reset_means_optimizer)` core
(the two identical loops) that each model's method calls before its model-specific tail; and
`resize_identity_buffers(model, num_points)` driven by `model._identity_buffer_names()` +
per-name dtype (long for `object_instance_ids`, bool for `current_active_mask`). Both in
`dynamic_gs/utils/gaussian_surgery.py`.

Est. LOC saved: ~30.

---

## Pattern E — opacity-logit → `sigmoid` thresholding (purge / filter idiom)

The "threshold Gaussians by activated opacity" idiom appears in three spellings (torch
`sigmoid`, numpy logistic), each followed by an index/keep-mask build feeding a delete or a
parallel-array subset.

Sites:
- `dynamic_gs/static_gs_pipeline.py` (`_finalize_static_training` opacity purge) — lines 234–235:
  `torch.sigmoid(gauss_params["opacities"]...) < purge_thr → nonzero → delete_gaussian_indices`
- `dynamic_gs/utils/anysplat_decode.py` (`reproject_anysplat_to_scene` opacity filter) — lines 667–668:
  `opac = 1.0/(1.0+exp(-opacity_logits)); keep = opac >= opacity_min`

Risk: **cosmetic→maintenance.** Both implement "activated opacity = sigmoid(logit), compare
to a threshold." Not a correctness hazard on its own (the threshold semantics are local) but
two implementations of the same activation (one torch, one hand-rolled numpy logistic) is an
easy place for an off-by-one in the comparison direction.

Proposed helper: `activated_opacity(logits)` (backend-dispatched torch/numpy) +
`low_opacity_indices(opacity_logits, thr)` in `dynamic_gs/utils/gaussian_surgery.py`. Thin,
but it names the convention once.

Est. LOC saved: ~6.

---

## Pattern F — log-scale (natural-log metres) arithmetic: shrink-oversized / scale-from-metres

Three independent implementations of "operate on log-scale to cap/seed a metric scale":
the static mid-training UNIFORM-SHRINK callback, the AnySplat insert-batch max/min-scale
cap (numpy, same uniform-divide-in-log algorithm), and the rgbd/anysplat metres→log seed.

Sites:
- `dynamic_gs/static_gs_model.py` (`_shrink_oversized_scales_cb`) — lines 244–266:
  `log_max = scales.max(1); shift = where(log_max>log_trigger, log_max-log_reset, 0); scales.sub_(shift)`
- `dynamic_gs/utils/anysplat_decode.py` (max_scale_m / min_scale_m cap) — lines 791–803:
  numpy mirror — `log_max_axis = log_scales.max(1); shift = clip(log_max-log_cap,0); log_scales -= shift`,
  then `keep_s = exp(log_scales).max(1) >= min_scale_m`
- `dynamic_gs/utils/anysplat_decode.py` (`_apply_similarity_to_gaussians`) — line 495:
  `log_scales_world = log_scales + log(similarity_s)` (and lines 772–774 `+ log(s_per_gauss)`,
  `+ log(scale_multiplier)`)
- `dynamic_gs/utils/rgbd_decode.py` — line 448: `scales = torch.log(scales_lin)`
  (metres→log seed; pairs with static_gs_model.py:453 `torch.log(avg_dist.repeat(1,3))` in Pattern C)

Risk: **maintenance.** CLAUDE.md documents the uniform-shrink (preserve aspect, divide all
axes by the same factor) as the deliberate choice over a per-axis clamp; it is implemented
twice (torch in the model callback, numpy in the AnySplat filter). If the policy changes
(e.g. clamp vs shrink) both must move together — they were introduced for the same reason in
the same session.

Proposed helper: `uniform_shrink_log_scales(log_scales, max_scale_m, *, min_scale_m=0.0)`
returning `(log_scales, keep_mask)`, backend-dispatched, in
`dynamic_gs/utils/gaussian_surgery.py`. Multiply-in-log helpers
(`log_scales + log(factor)`) are trivial and probably not worth a wrapper; flag only the
shrink/cap.

Est. LOC saved: ~12.

---

## Pattern G — parallel-array lockstep subset inside `reproject_anysplat_to_scene`

Within ONE function, the same 6-array subset
(`means_canonical / log_scales / quats_wxyz / opacity_logits / features_dc / features_rest`)
is hand-applied FIVE times by re-listing all six arrays on a `keep` mask. This is the same
"keep the per-Gaussian arrays in lockstep" hazard as Pattern A, but for AnySplat's plain
numpy arrays rather than the model's params+buffers.

Sites (all in `dynamic_gs/utils/anysplat_decode.py`, `reproject_anysplat_to_scene`):
- lines 669–670 (opacity `keep`)
- lines 676–677 (`keep_bg`)
- lines 729–730 (`keep_d` no-sensor-depth)
- lines 751–752 (`keep_c`)
- lines 799–803 (`keep_s` scale cap)

Risk: **correctness.** Five separate re-listings of the six arrays; forgetting to subset one
array on a new `keep` (or adding a 7th array and missing a site) silently misaligns the
inserted Gaussians' attributes — exactly the failure class that produced "ghost"/misplaced
inserts before. Highest-density duplication found.

Proposed helper: a tiny local `def _subset(keep): ... ` closure (or a
`GaussArrays` dataclass with a `.subset(mask)` method) that owns the six arrays and applies
one mask to all of them — replacing each two-line re-listing with `_subset(keep)`. Lives at
the top of `reproject_anysplat_to_scene` (or a small `numpy`-side helper in
`gaussian_surgery.py` if reused elsewhere).

Est. LOC saved: ~12.

---

## Not flagged (verified NOT this duplication class)

- `dynamic_gs/persistence/post_fusion_cache.py` lines 121–143: reallocates the 6
  `gauss_params` from a saved state_dict and re-binds `_mask_means_grad`. This is the
  load/warm-restart path (relies on the model's `load_state_dict` override to resize the 4
  buffers) — it does NOT hand-roll the buffer subset, so it's a consumer of the convention,
  not a duplicate of it. Left as-is.
- `scripts/view_splats_viser.py`, `dump_post_fusion_to_viser.py`, `dump_viser_pt_to_ply.py`,
  `diag_*.py`: read-only viewer/diagnostic subsetting of a saved `.pt` (numpy masking for
  display). Not on the train/insert/delete path; coincidental `inserted_flags[keep]` naming
  only. Left as-is.
- `dynamic_gs_model.py` lines 937–938 / 1006–1007 (`gauss_params["means"][object_mask] = ...`):
  the rigid-transform in-place WRITE — mutates values, does not change count or subset arrays.
  Different operation. Left as-is.
