# Code Audit — preseg (`utils/preseg_seed.py` + `static_gs_preseg_pipeline.py`)

Module scope: per-Gaussian instance-id labeling of the TSDF seed PLY (SAM2-AMG + SAM3 + SAM2-video propagate + occlusion-voted 3D transfer) and the `static-gs-preseg` nerfstudio pipeline that wires it into a fuse-before-train flow.

Entry point: `static-gs-preseg` is registered in `pyproject.toml:25` → `dynamic_gs.dynamic_gs_config:StaticGSPreseg` (`dynamic_gs_config.py:91`), so the pipeline class + its config are live nerfstudio method targets.

---

## 1) FUNCTION / CLASS MAP

### `dynamic_gs/utils/preseg_seed.py`

- **`AmgConfig` (dataclass)** — `preseg_seed.py:39` — AMG knobs (points_per_side, area gates, max_masks, gripper-blackout, SAM2 HF model). Constructed once from pipeline config. Refs: imported + constructed in `static_gs_preseg_pipeline.py:60,239`; referenced as default arg in `build_labeled_seed` and in `_sam2_amg_frame0`. ~6 refs.
- **`LabeledSeed` (dataclass)** — `preseg_seed.py:49` — return descriptor (paths + counts) of `build_labeled_seed`. Refs: ONLY the return type/constructor inside `build_labeled_seed` (`:405,:543`) + docstring mentions. No external consumer reads its fields by type name — the caller uses `result.instance_ids_path` etc. via duck typing. See §2.
- **`_log(msg)`** — `preseg_seed.py:65` — prefixed print. Module-local, called ~25× within the file. (repo-wide `_log` count 110 is collisions with other modules' own `_log`.)
- **`_abspath(dataset_dir, rel)`** — `preseg_seed.py:69` — join dataset dir + relative path stripping `./`. Called by `_load_gripper_mask`, `_load_rgb`, `_load_depth_m`. 3 refs.
- **`_load_transforms(dataset_dir)`** — `preseg_seed.py:73` — load+sort `transforms.json`, return frames/intrinsics/base/depth_scale; accepts root or static_scene dir. Called once in `build_labeled_seed:419`. 1 ref.
- **`_load_gripper_mask(base, fr)`** — `preseg_seed.py:108` — read mask PNG → bool (mask==0 = robot exclusion). Called by `_load_rgb:116`. 1 ref.
- **`_load_rgb(base, fr, black_out_gripper)`** — `preseg_seed.py:113` — read RGB, optionally zero gripper pixels. Called by `_write_video_frames` + `build_labeled_seed:423`. 4 refs.
- **`_load_depth_m(base, fr, depth_unit_scale)`** — `preseg_seed.py:120` — read uint16 depth → float metres. Called by `_transfer_labels_to_ply:358`. 2 refs.
- **`_load_sam2_predictor(sam2_hf_model)`** — `preseg_seed.py:128` — load the SAM2 video predictor once (shared by AMG + propagation). Called in `build_labeled_seed:429`. 2 refs.
- **`_sam2_amg_frame0(frame0_rgb, cfg, sam2_model)`** — `preseg_seed.py:143` — AMG on frame 0, area/gripper-filter, top-K masks → `(masks, seg0)`. Called in `build_labeled_seed:432`. 1 ref.
- **`_assign_and_merge(amg_masks, sam3_masks, coverage_threshold)`** — `preseg_seed.py:185` — coverage-assign each AMG mask to its best SAM3 instance, union per SAM3 id (id-preserving, SAM3-raw fallback for uncovered instances) → `(merged, target)`. Called in `build_labeled_seed:504`. 1 ref (+ doc mention in `object_picker.py:12`).
- **`_write_video_frames(frames, base, black_out_gripper, video_frames_dir)`** — `preseg_seed.py:245` — write per-frame JPEGs for SAM2-video; returns the RGB list. Called in `build_labeled_seed:513`. 1 ref.
- **`_propagate(seed_masks, num_frames, H, W, video_frames_dir, predictor)`** — `preseg_seed.py:260` — SAM2-video propagate seeded objects (obj_id = sam3_index+1) → `seg_ids (F,H,W)`. Called in `build_labeled_seed:514`. 1 ref (+ doc mentions in `object_picker.py:12`, `static_gs_preseg_pipeline.py:71` comment, and unrelated `dynamic_gs_model.py:1391 _propagate_instance_membership`).
- **`_transfer_labels_to_ply(ply_path, frames, intr, base, seg_ids, depth_unit_scale, min_obj_votes, occ_tol_m)`** — `preseg_seed.py:318` — visibility-aware per-point vote transfer → `(ids, K)`. Called in `build_labeled_seed:524`. 1 ref.
- **`build_labeled_seed(*, dataset_dir, ply_path, out_dir, sam_worker, text_prompts, sam3_confidence_threshold, coverage_threshold, amg_cfg, min_obj_votes, occ_tol_m)`** — `preseg_seed.py:393` — public entry point orchestrating steps 1–6, writes `<ply>.instance_ids.npy` + `seg_ids.npz`. Called in `static_gs_preseg_pipeline.py:250`. 1 functional call (10 grep hits, mostly comments/docstrings).

### `dynamic_gs/static_gs_preseg_pipeline.py`

- **`StaticGSPresegPipelineConfig` (dataclass, extends `StaticGSPipelineConfig`)** — `static_gs_preseg_pipeline.py:64` — config + the seg/behavior knobs. Imported in `dynamic_gs_config.py:16`, constructed at `:101`. Entry-point config. ~6 refs.
- **`StaticGSPresegPipeline` (class, extends `StaticGSPipeline`)** — `static_gs_preseg_pipeline.py:94` — the pipeline. `_target` of the config; nerfstudio instantiates it. Entry-point class. ~8 refs.
- **`.__init__(config, device, test_mode, world_size, local_rank, grad_scaler)`** — `:99` — ICP-refine poses, label seed (or reuse sidecar), build datamanager+model via `VanillaPipeline.__init__` (bypassing `StaticGSPipeline.__init__`), load sidecar. Called by nerfstudio.
- **`._run_segmentation_and_label_seed(config)`** — `:190` — spawn SamWorkerClient, bg-load SAM3, call `build_labeled_seed`, unload+close worker. Called in `__init__:162`. 1 ref.
- **`._load_sidecar_into_buffer()`** — `:287` — load sidecar `.npy`, validate shape vs `model.num_points`, copy into `model.object_instance_ids[:,0]`. Called in `__init__:184`. 1 ref.
- **`.get_training_callbacks(training_callback_attributes)`** — `:327` — VanillaPipeline callbacks + AFTER_TRAIN `_save_post_fusion_state` (deliberately skips `StaticGSPipeline`'s Phase 0b callback). Nerfstudio trainer override.
- **`._save_post_fusion_state(step)`** — `:345` — AFTER_TRAIN: write `static_state.pt` via `save_post_fusion_state`, idempotency-guarded by `_phase0b_done`. Registered as callback (`:340`). 2 refs.
- Inherited (NOT redefined here, used as entry points): `_write_timing_report` (`static_gs_pipeline.py:284`, atexit-registered at `:181`).

---

## 2) DEAD-CODE CANDIDATES

- **`LabeledSeed` dataclass — `preseg_seed.py:49` — LOW confidence.** Grep shows it is referenced only inside `preseg_seed.py` (lines 15 docstring, 405 return annotation, 408 docstring, 543 constructor). The single consumer (`static_gs_preseg_pipeline.py:265-267`) reads `result.instance_ids_path` / `.num_labeled_points` / `.num_instances` by attribute, never by the type name, so the dataclass *type* has zero external references. NOT genuinely dead — it is the live return value of the public `build_labeled_seed`; it is part of the declared public API (module docstring line 15). Listed for completeness only; do not remove. **`seg_ids_path` field is written (`:546`) but never read by any consumer** — kept "for QA" per its inline comment.
- No other zero-ref symbols. Every private helper has ≥1 live call inside the module, and both public pipeline symbols are wired through the nerfstudio entry point. `object_instance_ids` writes are invariant-protected (Invariant #8 / "Per-object identity buffers": `object_instance_ids` is the buffer this whole module exists to populate; not dead).

---

## 3) DATA-LIFECYCLE

Persistent state touched and how it flows:

- **Seed PLY** (`static_scene/depth_camera_init_points.ply`, read-only) — read in `_transfer_labels_to_ply:328` (open3d) AND independently re-read by the nerfstudio dataparser when `VanillaPipeline.__init__` builds the model. **Desync risk is real and explicitly guarded:** `_load_sidecar_into_buffer:303` asserts `ids.shape[0] == model.num_points` and raises on mismatch (comment "the dataparser may have permuted points"). The labeling votes are computed against the open3d point ORDER; if the dataparser reorders/filters points the ids would be misassigned — the shape check catches *count* drift but NOT a same-count permutation. MEDIUM: a same-N reorder would silently mislabel every point. The module relies on the dataparser preserving PLY row order (undocumented contract).
- **Sidecar `<ply>.instance_ids.npy`** (N, int64) — written in `build_labeled_seed:536`, read back in `_load_sidecar_into_buffer:295`. Round-trips correctly (int64 both ways). Reuse cache: `__init__:157` skips re-segmentation if it exists AND `reuse_sidecar_if_present`. NOTE: reuse keys only on file existence — a stale sidecar from a DIFFERENT seed PLY/prompt is reused blindly (only the count check protects it). Stale-id hazard is documented in CLAUDE.md ("Datasets labeled before 2026-06-11 carry stale ids").
- **`seg_ids.npz`** (F,H,W uint8) — written `:520`, passed in-memory to `_transfer_labels_to_ply`; the on-disk copy is QA-only and never re-read by the pipeline. Loaded-but-the-disk-copy-never-freed/used.
- **`object_instance_ids` buffer** (model, invariant-protected) — zero-initialized by the model at `num_points`, populated at `_load_sidecar_into_buffer:313` (`[:,0] = ids`), persisted at `_save_post_fusion_state:354` via `save_post_fusion_state`. The other 3 identity buffers (`object_flags`, `sam3d_init_target_flags`, `inserted_flags`) stay zero here by design (Invariant #8) — saved as zeros into `static_state.pt`. Correct/expected.
- **`static_state.pt` warm cache** — written `:354` (`post_fusion_cache_subpath = "static_scene/static_state.pt"`, from `static_gs_pipeline.py:83`). **Doc/path note:** the pipeline docstring (`:30`) and module header repeatedly call it `static_scene/static_state.pt` — that is correct; CLAUDE.md's data-format section still calls it `post_fusion_state.pt` (doc drift, not a code bug here).
- **SAM model GPU memory:**
  - SAM3 — spawned `SamWorkerClient` (`:204`), loaded in bg thread (`:216`), unloaded + closed in the `finally` (`:275-281`). Freed on both success and exception paths. OK.
  - SAM2 video predictor — loaded `build_labeled_seed:429`, `del sam2_predictor` + `empty_cache()` at `:517-518`. The AMG generator is `del amg`'d at `_sam2_amg_frame0:176`. If `build_labeled_seed` raises BETWEEN load and the `del` (e.g. AMG returns 0 masks → `RuntimeError:433`, or SAM3 0 instances → `:464`), the SAM2 predictor is NOT freed — there is no try/finally around it. MEDIUM: GPU leak on the error path (process usually exits anyway, so impact is bounded).
- **`_video_frames/` JPEG scratch dir** — created + filled in `_write_video_frames:251`, NEVER deleted. Per-frame JPEGs accumulate under `preseg_artifacts/_video_frames/` and are overwritten (same `{i:05d}.jpg` names) on re-runs but never cleaned. LOW: disk growth bounded by frame count, but the dir is left behind.
- **Worker process handle** — `SamWorkerClient()` spawns a subprocess; `.close()` in `finally:279`. On the reuse-sidecar path the worker is never spawned (correct).

No double-loads of the same tensor. No save/load shape mismatch (the one shape check is the desync guard, working as intended).

---

## 4) DESIGN SMELLS

- **Swallowed exceptions (multiple):**
  - `_run_segmentation_and_label_seed:274-281` — `unload_sam3()` and `close()` wrapped in bare `except Exception: pass`. Acceptable for teardown, but completely silent (no log).
  - `__init__:150` — ICP refine failure caught and logged, then training continues on possibly-unrefined poses (violates Invariant #3 silently if it fails; only a CONSOLE.log warns). MEDIUM: a real ICP failure degrades to raw URDF poses with only a log line.
  - `_load_sidecar_into_buffer:296` — `np.load` failure caught, logged, and the method RETURNS leaving `object_instance_ids` all-zero. The run then trains + saves a cache with NO instance ids and no hard failure. MEDIUM: the whole point of the method (per-Gaussian ids) can silently no-op.
  - `build_labeled_seed:500` — overlay-PNG write failure swallowed (cosmetic, fine).
- **Dead/unused local state:**
  - `_seg0` (`build_labeled_seed:432`) — `_sam2_amg_frame0` returns `(masks, seg0)` but `seg0` is discarded (`_seg0`). The `seg0` label image is fully computed in `_sam2_amg_frame0:172-174` and never used by anyone → wasted work + dead return value. LOW.
  - `_write_video_frames` returns `rgbs` (`:257`) but the caller (`:513`) ignores it → dead return.
  - `_static_dir` (`:129`) assigned, never read after.
  - `_num_instances` / `_labeled_instance_count` (`:116-117`) assigned in 3 places, never read anywhere (no getter, not in timing report). Bookkeeping-only attributes that go nowhere.
  - `field` imported from dataclasses (`:39`) — used (`:68` `_target`), OK. `defaultdict` imported at module top (`:39`) AND re-imported locally inside `_assign_and_merge:190` — duplicate import, the top-level one in `static_gs_preseg_pipeline.py` is used for `_timing`. Minor.
- **`build_labeled_seed` is a god function** (`:393-549`, ~155 lines) — orchestrates SAM2 load, AMG, SAM3 loop, overlay rendering, merge, video write, propagate, vote transfer, and two sidecar writes in one body. The SAM3 overlay-PNG rendering block (`:477-501`) is pure visualization inlined into the core path; it belongs in a helper. Mutable default arg `amg_cfg: AmgConfig = AmgConfig()` (`:402`) — shared instance default (safe only because AmgConfig is never mutated, but a known footgun).
- **Duplicated logic:** the OpenGL→OpenCV `flip = diag(1,-1,-1,1)` + back-projection math in `_transfer_labels_to_ply:339-352` re-implements the same convention used across the repo (e.g. `online_fusion`, anysplat reproject). Not factored. LOW (intentional verbatim port from the MVP per docstring).
- **Misleading naming:** `_phase0b_done` (`:113,346`) is repurposed as a generic "post-fusion state saved" idempotency guard even though this method has NO Phase 0b — the `__init__` comment (`:108-110`) flags the reuse, but the name lies about its meaning. Similarly the timing keys are prefixed `P0a.*` (`:205,234,262`) for a method that has no Phase 0a. Confusing for anyone reading timing reports.
- **Leaky abstraction / undocumented ordering contract:** the whole correctness of `_load_sidecar_into_buffer` depends on the open3d PLY read order (used to compute votes) matching the nerfstudio dataparser's Gaussian row order. This cross-module invariant is enforced only by a count assert and a comment ("check Risk #2 in the plan") referencing an external doc not in the repo.
- **Config knobs all defaulted, none overridden:** every preseg-specific field on `StaticGSPresegPipelineConfig` (`:72-91`) is left at its dataclass default in `dynamic_gs_config.py:101` (only base StaticGSModel fields are set). Fine, but means the CLI is the only way to change them — and they ARE all read (verified), so none are dead config fields.
