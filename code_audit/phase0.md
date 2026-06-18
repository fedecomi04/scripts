# Audit: `dynamic_gs/fusion/phase0.py`

Phase 0a (SAM3/FastSAM mask discovery + Fast-SAM3D 3D-object generation) and
Phase 0b (NDP/CPD/TEASER registration + insertion + instance-id propagation).
Pure-function lift from the deleted monolith. 6 top-level symbols, all module-level
functions (no classes/methods).

Caller grep run over `dynamic_gs/` + `scripts/` (`--include=*.py`).

---

## 1) FUNCTION / CLASS MAP

### `backproject_mask_to_world(mask_bool_np, depth_image, rgb_image, camera, max_object_slope_deg=70.0, near_surface_window_frac=0.012) -> (points_np, colors_np)` — phase0.py:57
Back-projects an image-plane mask through a depth image into world 3D points; applies a MAD depth-outlier scrub + a geometry-derived local near-surface filter to scrub silhouette-edge table/see-through bleed. Returns `(N,3)` world points + `(N,3)` colors.
**Callers: 3.** `phase0.py:920` (`run_phase0b_fusion` registration target); `scripts/view_object_reconstruction.py:78`; re-exported `fusion/__init__.py:31,39`. Also referenced in 3 diag scripts as comment cross-refs.

### `cull_points_in_front(points_world, target_points_world, camera, render_hw, band_m=0.0, radius_px=2) -> keep_mask` — phase0.py:190
Builds a front-surface depth buffer by projecting the trusted real/GT target cloud, then returns a boolean keep-mask that drops inserted points lying between the camera and that surface (occlusion cull). Inverse projection of `backproject_mask_to_world`.
**Callers: 1.** `phase0.py:1047` (`run_phase0b_fusion`, in-front cull stage). NOT re-exported in `fusion/__init__.py`.

### `load_anchor_reference(static_dir, device) -> Optional[dict]` — phase0.py:252
Loads the canonical `<static>/anchor_ref/` (rgb/depth/intrinsics/c2w) written by `live_session._write_anchor_ref` and rebuilds a `Cameras(1)` so Phase-0b back-projects the mask through the exact frame SAM3/SAM3D ran on. Returns `None` (→ caller falls back to `cached_train[-1]`) when the folder is absent or load fails.
**Callers: 1.** `phase0.py:827` (`run_phase0b_fusion`). NOT re-exported in `fusion/__init__.py`.

### `save_sam3_debug_plots(rgb_path, sam3_objects, out_dir, prefix="static0") -> None` — phase0.py:297
Writes a SAM3 segmentation-review overview PNG (all masks colored + bbox/labels) plus one per-object overlay PNG.
**Callers: 1.** `phase0.py:628` (`run_phase0a_sam3_and_sam3d`); re-exported `fusion/__init__.py:34,42`.

### `run_phase0a_sam3_and_sam3d(*, model, datamanager, timing=None) -> Optional[dict]` — phase0.py:399
Pre-static: segments the last static / anchor frame (SAM3 or FastSAM), then runs Fast-SAM3D per mask, writing per-object PLYs + pose sidecars under `initialization_artifacts/`. Does NOT mutate the Gaussian scene. Returns `{"sam3_objects","sam3d_results"}` or `None` (0 objects).
**Callers: 1.** `static_gs_pipeline.py:143`; imported `static_gs_pipeline.py:57`; re-exported `fusion/__init__.py:32,40`.

### `run_phase0b_fusion(*, model, datamanager, generation_outputs, device, timing=None) -> dict` — phase0.py:762
Post-static: loads each pre-generated SAM3D PLY, registers it (NDP default; CPD/TEASER selectable) against the back-projected real-depth target, culls (proximity de-dup + in-front occlusion), inserts via `model.insert_object_gaussians`, and propagates `object_instance_ids` onto matched existing Gaussians. Writes `phase0_manifest.json`; returns the manifest dict.
**Callers: 1.** `static_gs_pipeline.py:248`; imported `static_gs_pipeline.py:57`; re-exported `fusion/__init__.py:33,41`.

---

## 2) DEAD-CODE CANDIDATES

**None.** All 6 top-level functions have at least one live caller (verified by grep above). `cull_points_in_front` and `load_anchor_reference` are internal-only (single in-module caller, not re-exported) but genuinely live — not dead.

No nested helper is dead: `_project` (phase0.py:226) is called twice within `cull_points_in_front` (234, 242).

---

## 3) DATA-LIFECYCLE

Phase 0 touches the 4 identity buffers, the model's GPU tensors, image/model caches, and several on-disk sidecars. The post-fusion `.pt` warm-cache (`persistence/`) is NOT written here — the caller (`static_gs_pipeline`) saves it.

### Identity buffers (invariant-protected — see CLAUDE.md Invariant #8)
- **`object_instance_ids`** — WRITTEN here, as documented (Phase-0b is the only writer). Inserted Gaussians get `instance_id` via `insert_object_gaussians(..., instance_id=instance_id)` (phase0.py:1056-1061); existing matched Gaussians get `model.object_instance_ids[match_indices] = instance_id` (phase0.py:1103). Correct per invariant.
- **`object_flags`** — inserts pass `object_flag=False` (phase0.py:1059); never set to 1 here. Matches Invariant #8 ("`object_flags=0` in `post_fusion_state.pt` is correct").
- **`sam3d_init_target_flags`** — NOT touched here. Invariant-protected (writer `initialize_object_from_sam3d` is the only value-writer and has no caller). Correct — do NOT flag.
- **`inserted_flags`** — NOT touched here (FF Mode B owns it). Correct.
  - **MEDIUM — buffer desync risk on insert.** `run_phase0b_fusion` calls `model.insert_object_gaussians` (phase0.py:1056) which appends rows to all 6 gauss_params + the 4 identity buffers. Phase 0b inserts per-object **inside a loop** (n_objs iterations) and re-renders each iteration (phase0.py:885-887, comment explains the stale-`model.info` reason). If `insert_object_gaussians` does not resize all 4 buffers in lockstep, the buffers desync. Not a bug in *this* file, but this file is the trigger site — the per-iteration insert + the later `model.object_instance_ids[match_indices]` write (phase0.py:1103) both assume buffer length == gauss_params length at that moment. Verify against `dynamic_gs_model.py:1230 insert_object_gaussians`.

### Model GPU tensor / cache lifecycle (phase0.py:528-743, `run_phase0a`)
Around the SAM3D subprocess the model + image caches are evicted CPU-ward then restored:
- `model.to("cpu")` (530), restored `model.to(run_device)` (719) in `finally`. **OK** — symmetric, guarded by `if not sam3_cached`.
- Image-cache CPU eviction (531-543) is unconditional on `not sam3_cached`; restore (720-740) is **conditional** on `cache_images=="gpu"` (727). This asymmetry is INTENTIONAL and documented (the unconditional CUDA restore was an 11.3 GB OOM on the 16 GB card). For `cache_images="cpu"` managers the batches were already on CPU, so the eviction loop at 531-543 was a no-op for them and skipping restore is correct. **OK, but subtle** — the eviction loop and restore loop use different guards; a future change to the default `cache_images` could break the symmetry silently.
- `gc.collect()` + `torch.cuda.empty_cache()` on both eviction (544-546) and restore (741-743). **OK.**

### On-disk sidecars (read/write)
- `static0_rgb.png` — written (508) only on the non-anchor path; the anchor path references `anchor_ref/rgb.png` instead of overwriting (the documented fix). **OK.**
- `static0_full_depth_meters.tiff` + `static0_full_intrinsics.json` — written 662-675 only when SAM3D actually runs (`to_run_indices` non-empty). Not cleaned up (left on disk). Low severity (small, reused as cache).
- `phase0_manifest.json` — written (1151). **OK.**
- Timing sidecars read: `live_sam3_timings.json` (596), `_sam3_timing.json`/`_fastsam_timing.json` (609), `_sam3d_timing.json` (705). All wrapped in try/except. The SAM3D sidecar read is gated on `st_mtime >= t_sam3d` (706) to avoid reading a stale prior-run file — **good defensive check**, but the FastSAM/SAM3 sidecars (609) have NO such mtime guard, so a stale split sidecar from a prior run could be recorded against this run's `t_sam3` window. LOW.

### Depth load (both phases)
- `Image.open(depth_filenames[idx])` → `np.float32 * depth_scale` (461-463, 858-860). uint16-mm → meters via `depth_unit_scale_factor`. Consistent with `load_anchor_reference` (278: `*1e-3`). **OK.** Both depth loads are wrapped in try/except and degrade to SfM targets (`static_depth_m=None`).

### No leak of file/process handles
SAM3 / SAM3D / FastSAM all run as subprocesses inside their own wrappers; phase0 only passes paths. No open file handles held (all `Image.open`/`read_text` are transient). No SHM touched here.

---

## 4) DESIGN SMELLS

### HIGH — `NameError` when anchor_ref present AND SAM3D must run (uncached)
`static_np` is assigned ONLY in the non-anchor `else` branch (phase0.py:492/494, inside `if _anchor_rgb is None/absent`). But phase0.py:661 unconditionally reads `static_np.shape` to compute `H_img,W_img` for the SAM3D full-depth TIFF — reached whenever `static_depth_m is not None and to_run_indices` (660). In the **canonical live anchor path** (`_anchor_rgb` exists, line 485 branch) `static_np` is never defined, so if SAM3D actually needs to run (not cached) line 661 raises `NameError: name 'static_np' is not defined`, caught nowhere until the `except Exception` at 689 only wraps the *subprocess* call — but 661 is *before* that try, so it propagates and aborts Phase 0a. This is exactly the live flow the anchor_ref machinery was built for. **Recommendation:** derive `H_img,W_img` from `static_image.shape` (or the anchor camera intrinsics) instead of `static_np`, which only exists on the fallback path.
- file:line — phase0.py:661 (read) vs phase0.py:492-508 (sole assignment, other branch)

### MEDIUM — god function: `run_phase0b_fusion` (~405 LOC, phase0.py:762-1167)
One function does: ref-frame resolution (anchor vs cached + camera-opt apply), per-object render, PLY/pose load, mask resize, registration dispatch with a 14-field TEASER param dict (952-966), TEASER stage logging (976-1000), proximity cull (1011-1038), in-front cull (1040-1053), insert, NN-based existing-Gaussian flagging (1065-1104), manifest assembly (1117-1139), and ledger recording. The proximity-cull and the existing-flag matching each build their own `sklearn NearestNeighbors` with near-duplicate logic (1031-1038 vs 1085-1096). **Recommendation:** extract `_resolve_reference_frame`, `_cull_inserted_points`, `_flag_existing_gaussians` helpers.

### MEDIUM — duplicated reference-frame resolution across the two phases
`run_phase0a` (451-472) and `run_phase0b` (824-863) independently re-derive the static frame, depth, and intrinsics from `cached_train[-1]` / `depth_filenames`, with `run_phase0b` additionally trying `load_anchor_reference` first. The `static_dir = Path(depth_filenames[0]).resolve().parent.parent` derivation appears twice (479-481 vs 824-826) with subtly different variable names (`_anchor_static_dir` vs `static_dir`). The two phases can pick *different* reference frames: 0a uses `static0_rgb`/anchor for segmentation but 0b independently re-resolves — they only stay consistent because both prefer anchor_ref. Fragile coupling. file:line — phase0.py:479-481, 824-826.

### MEDIUM — magic-number cull tunables hardcoded as locals
`CULL_STRENGTH=1.3`, `TAU_FLOOR_M=0.003`, `CULL_DEPTH_TOL_M=0.015`, `FLAG_DEPTH_TOL_M=0.02` (1006-1009), `MAX_RADIUS_M=0.02` (1065), `IN_FRONT_BAND_M=0.0` (1044), proxy/target radius constants `0.003/1.5`, `0.002/6.0` (1084,1093). These are load-bearing registration-quality knobs buried as function locals (the comment at 1002-1005 even says "see legacy pipeline comments"), not on any `*Config`, so they can't be A/B'd without editing source — inconsistent with the rest of the codebase's env-overridable knobs (`DGS_FF_*`, etc.). recommendation: promote to config or env. file:line — phase0.py:1006-1009.

### LOW — swallowed exceptions hide real failures
Multiple bare `except Exception: pass` / log-and-continue blocks: timing-ledger writes (620, 715, 1159), live-timing sidecar (602), font load (318/351). The timing ones are genuinely cosmetic. But the SAM3D-subprocess `except` at 689-691 sets `multi_results=[{}]*N` and continues — a hard SAM3D failure silently degrades to "0 objects fused" with only a CONSOLE.log, no raised error. For the primary product path this should at least be surfaced louder. file:line — phase0.py:689-691.

### LOW — misleading docstrings / stale naming vs runtime reality
- The module docstring + `run_phase0b_fusion` print (867-869) and timing key `S0.1_fastsam_segmentation` (588,599) say "CPD/TEASER++" / "fastsam" regardless of the actual configured backend. The default backend is **NDP** (per CLAUDE.md), and `S0.1_fastsam_segmentation` is recorded even for the SAM3 backend (588). Misleading to anyone reading the timing report or stdout. file:line — phase0.py:867-869, 588.
- The ledger record at 1156-1158 hardcodes the label `"NDP register+fuse"` even when `backend` is `cpd`/`teaser` — inverse mislabel of the above. file:line — phase0.py:1157.

### LOW — `frame_idx_0` reused as both depth-list index and camera-opt index
In `run_phase0b` fallback path, `frame_idx_0` (read from `batch["image_idx"]`, 837) indexes both `depth_filenames` (856) and `camera_optimizer` cameras (847). If the static manager ever reorders/filters cached_train so `image_idx` ≠ dataset row, the depth and the camera-opt offset would be fetched for different frames. Currently safe (no reordering) but a latent coupling. file:line — phase0.py:837,847,856.
