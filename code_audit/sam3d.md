# Adversarial Code Audit — `sam3d.py` + `sam3d_fusion.py`

Scope: `dynamic_gs/utils/sam3d.py` (SAM3D subprocess/worker runners) and
`dynamic_gs/utils/sam3d_fusion.py` (Phase-0b geometry registration + fusion).

Grep basis (all ref counts below): `grep -rn <sym> dynamic_gs scripts --include=*.py`,
excluding the symbol's own definition file. "External" = refs outside the
defining module. Re-exports through `utils/__init__.py` + `fusion/__init__.py`
are counted as references (they are the public surface that real callers import).

---

## 1) FUNCTION/CLASS MAP

### `dynamic_gs/utils/sam3d.py`

- `_resolve_sam3d_python() -> str` — sam3d.py:32 — Picks the interpreter that has `sam_3d_objects` installed (the `sam3_dynamic_gs` env python if present, else `sys.executable`). — 2 internal callers (subprocess builders L954, L1028); no external refs.
- `_sam3d_subprocess_env() -> Dict[str,str]` — sam3d.py:44 — Builds a hardened env (pins `CONDA_PREFIX`/`CUDA_HOME`/`LD_LIBRARY_PATH`/`PYTHONNOUSERSITE`) so the cross-env subprocess doesn't KeyError on Fast-SAM3D's import-time `os.environ["CONDA_PREFIX"]`. — 2 internal callers (L969, L1050); no external refs.
- `get_sam3d_output_paths(output_dir, output_stem, image_dir=None) -> Dict[str,Path]` — sam3d.py:66 — Canonical path map (ply/pose/preview/run_info/glb/mesh_ply) for a SAM3D output stem. — 16 refs: `utils/__init__`, `fusion/__init__`, `live_session.py:1015`, `sam_worker.py:434/470`, `dynamic_gs_model.py:1627`, `phase0.py:646`.
- `resolve_sam3d_pose_path(raw_ply_path, fallback_pose_path=None) -> Path|None` — sam3d.py:83 — Resolves the `_pose.json` sidecar for a `_raw_output.ply`, with a fallback candidate. — 14 refs: `dynamic_gs_model.py:1629/1640/1683`, `phase0.py:647`, `live_session.py:1016`, both `__init__`s.
- `load_sam3d_pose(path) -> Dict[str,np.ndarray]` — sam3d.py:101 — Loads translation/rotation/scale arrays from a pose JSON. — 7 refs: `sam3d_fusion.py:107`, both `__init__`s.
- `sam3d_pose_has_rotation(path) -> bool` — sam3d.py:112 — True iff the pose sidecar has a finite 4-element rotation quaternion. — 15 refs: `dynamic_gs_model.py:1630/1646/1684`, `live_session.py:82/1017`, `phase0.py:648`, both `__init__`s.
- `_load_binary_mask(mask_path, target_size) -> np.ndarray` — sam3d.py:123 — Loads + thresholds + (NEAREST) resizes a mask PNG to a uint8 binary mask. — Internal callers L327/L493/L701 + 1 external: `sam_worker.py:430/450` (private-import via `_sd._load_binary_mask`).
- `_install_kaolin_stub() -> None` — sam3d.py:130 — Inserts dummy `kaolin.*` modules into `sys.modules` so Fast-SAM3D imports without the real kaolin. — 1 caller: `_import_official_api` L166 (internal only).
- `_import_official_api()` — sam3d.py:164 — Stubs kaolin, extends `sys.path`, sets `LIDRA_SKIP_INIT`, imports `Inference`. — 6 refs: internal L527/L727 + `sam_worker.py:358/363`, `sam3d_trim_probe.py:45`, `measure_vram.py:165`.
- `_write_runtime_config() -> Path` — sam3d.py:177 — Writes the runtime `pipeline_runtime_small.yaml` (gaussian-only decode, fp16, MoGe HF override, workspace_dir fix). — 6 refs: internal L526/L726 + `sam_worker.py:359`, `sam3d_trim_probe.py:44`, `measure_vram.py:164`.
- `apply_sam3d_gaussian_trim(inference) -> Dict[str,list]` — sam3d.py:223 — fp16's the diffusion generators + DINOv2 embedders and CPU-offloads never-invoked modules to drop SAM3D resident VRAM (~11.7→7.3 GB). `DGS_SAM3D_NO_TRIM=1` disables. — refs in both runners (L573/L772) + `sam_worker.py`, `sam3d_trim_probe.py`, `measure_vram.py` (per grep set).
- `_save_preview(mask, image_rgb, preview_path) -> None` — sam3d.py:283 — Saves a red-overlay preview PNG of the segmented region. — Internal callers L541/L792; no external refs.
- `prepare_cropped_sam3d_inputs(render_image_path, object_mask_path, output_dir, output_stem, image_dir=None, padding=100, min_crop_side=800, depth_path=None, depth_scale=1.0, camera_intrinsics=None) -> Dict[str,Path]` — sam3d.py:289 — Crops RGB+mask (+optional depth+crop-shifted intrinsics sidecar) tightly around the mask for lighter/contextful inference. — refs: `dynamic_gs_model.py:1663`, `live_session.py:972`, both `__init__`s.
- `_resize_image_and_mask(image_rgb, mask, max_side) -> (np.ndarray,np.ndarray)` — sam3d.py:403 — Aspect-preserving downscale of image (BILINEAR) + mask (NEAREST) to `max_side`. — Internal callers L539/L791/L808 + external `sam_worker.py:431/492`, `sam3d_trim_probe.py:113`, `measure_vram.py:180`.
- `_build_pytorch3d_pointmap(depth_m, intrinsics) -> np.ndarray` — sam3d.py:421 — Back-projects metric depth into a `(H,W,3)` pytorch3d-convention pointmap (x,y flipped), NaN holes for invalid depth. — Internal callers L517/L723 + external `sam_worker.py:433/463`, `sam3d_trim_probe.py:117`, `measure_vram.py:190`.
- `run_sam3d_single_object(render_image_path, object_mask_path, output_dir, output_stem, image_dir=None, max_side=518, depth_path=None, intrinsics_path=None) -> Dict[str,Path]` — sam3d.py:458 — In-process single-object SAM3D run with OOM resize ladder; saves ply/pose/glb/run_info. — 1 caller: `_main` L1116 (CLI worker entry). NO direct external callers (always reached via the `_subprocess` wrapper → CLI).
- `run_sam3d_multi_object(render_image_path, object_mask_paths, output_dir, output_stems, image_dir=None, max_side=518, depth_path=None, intrinsics_path=None) -> list[Dict[str,Path]]` — sam3d.py:644 — In-process multi-object SAM3D run (single model load, sequential per-mask inference, OOM ladder, mesh export, timing sidecar). — 1 caller: `_main` L1102 (CLI worker entry). Only comment-refs elsewhere (`sam_worker.py:426/479`).
- `run_sam3d_multi_object_subprocess(... ) -> list[Dict[str,Path]]` — sam3d.py:922 — Launches `run_sam3d_multi_object` in a cross-env subprocess and rebuilds the result path maps. — 9 refs: `live_session.py:402`, `phase0.py:679`, both `__init__`s.
- `run_sam3d_single_object_subprocess(... ) -> Dict[str,Path]` — sam3d.py:999 — Launches `run_sam3d_single_object` in a fresh subprocess; validates the rotation sidecar. — 6 refs: `dynamic_gs_model.py:1670`, both `__init__`s.
- `_parse_args() -> argparse.Namespace` — sam3d.py:1072 — CLI arg parser for the worker subprocess. — 1 caller: `_main` L1096.
- `_main() -> int` — sam3d.py:1095 — CLI dispatch (multi vs single), deep `__cause__` traceback unwinder. — 1 caller: `__main__` L1148 (entry point).

### `dynamic_gs/utils/sam3d_fusion.py`

- `Sam3DInsertionResult` (dataclass) — sam3d_fusion.py:49 — Full fusion result bundle (aligned/kept points+colors, scale, transforms, timing). — 6 refs: `dynamic_gs_model.py:27/1707`, both `__init__`s.
- `_require_open3d()` — sam3d_fusion.py:82 — Guard returning the `open3d` module or raising. — Internal-only (used by `_to_pcd`, mesh/save helpers, etc.); 0 external refs.
- `_require_plyfile()` — sam3d_fusion.py:88 — Guard returning `PlyData` or raising. — Internal-only (L95, L251); 0 external refs.
- `load_sam3d_gaussian_ply(ply_path) -> (xyz,rgb)` — sam3d_fusion.py:94 — Reads a SAM3D gaussian PLY → xyz + SH-decoded RGB. — 15 refs: `run_teaser_registration_only.py`, `smoke_test_pipeline_teaser_v13.py`, both `__init__`s, etc.
- `load_sam3d_rotation_wxyz(pose_path) -> np.ndarray` — sam3d_fusion.py:106 — Loads + validates the wxyz rotation quaternion from a pose sidecar. — 10 refs: `phase0.py:38`, both `__init__`s, etc.
- `_quaternion_wxyz_to_rotation_matrix(quaternion) -> np.ndarray` — sam3d_fusion.py:114 — wxyz quaternion → 3×3 rotation. — 1 internal caller (`_apply_sam3d_rotation_init` L138); 0 external refs.
- `_apply_sam3d_rotation_init(source_points, rotation_wxyz, camera_to_world_rotation) -> np.ndarray` — sam3d_fusion.py:133 — Rotates the canonical SAM3D cloud by its pose, p3d→ns camera flip, then into world. — 1 internal caller (`register_and_fuse_sam3d_object` L1139); 0 external refs.
- `_to_pcd(points, colors=None)` — sam3d_fusion.py:145 — numpy → Open3D `PointCloud`. — Internal-only (many call sites); 0 external refs.
- `_ensure_rgb_colors(colors, point_count) -> np.ndarray` — sam3d_fusion.py:154 — Normalizes/decodes colors into [0,1] RGB, fills gray on mismatch. — 2 internal callers (L1130/1131); 0 external refs.
- `reconstruct_mesh_from_points(points, mesh_ply_path, voxel_size=0.005, poisson_depth=8, density_quantile_trim=0.05) -> bool` — sam3d_fusion.py:168 — Poisson mesh reconstruction from a point cloud (normals + density trim). — 4 refs: both `__init__`s + 1 internal caller (`reconstruct_mesh_from_gaussian_ply` L259).
- `reconstruct_mesh_from_gaussian_ply(gaussian_ply_path, mesh_ply_path, ...) -> bool` — sam3d_fusion.py:236 — Poisson mesh from SAM3D gaussian-center PLY (FoundationPose fallback when the SAM3D mesh decoder OOMs). — 4 refs: both `__init__`s. **No runtime caller besides the re-exports** — see Dead/Smells.
- `save_point_cloud(path, points, colors=None) -> None` — sam3d_fusion.py:268 — Writes a PLY (empty-cloud safe). — 11 refs: `register_and_fuse_sam3d_object` (internal L1320-1322) + `run_teaser_registration_only.py`, both `__init__`s.
- `_sample_rows_for_plot(points, max_points) -> np.ndarray` — sam3d_fusion.py:285 — Uniform row subsample for plotting. — 1 internal caller (`_save_correspondence_plot` L318/319); 0 external refs.
- `_set_equal_axes(ax, points) -> None` — sam3d_fusion.py:293 — Equal-aspect 3D axis limits for the correspondence plot. — Internal-only; 0 external refs.
- `_save_correspondence_plot(debug_dir, output_stem, source_points, target_points, correspondences, threshold) -> Path` — sam3d_fusion.py:304 — Saves a 2-panel matplotlib correspondence diagnostic PNG. — 1 internal caller (L1309); 0 external refs.
- `_centroid(points) -> np.ndarray` — sam3d_fusion.py:424 — Robust (5–95 pct trimmed) centroid. — 2 internal callers (L1147/1148). The 2 `xfeat_motion.py:1199` hits are an unrelated **local variable** named `_centroid`, NOT this function.
- `_bbox_diagonal(points) -> float` — sam3d_fusion.py:440 — Robust bbox diagonal (percentile extents). — 2 internal callers (L1142/1143); 0 external refs.
- `_largest_extent(points) -> float` — sam3d_fusion.py:453 — Max single-axis extent. — **0 refs anywhere (internal or external).**
- `_median_nn_distance(points) -> float` — sam3d_fusion.py:460 — Median nearest-neighbor spacing. — Internal (L1154/1155/1263) + external `run_teaser_registration_only.py`, `smoke_test_pipeline_teaser_v13.py`.
- `_voxel_downsample(points, colors, voxel_size) -> (pts,colors)` — sam3d_fusion.py:470 — Voxel down-sample keeping colors. — 2 internal callers (L1161/1162); 0 external refs.
- `_transform_points(points, transform) -> np.ndarray` — sam3d_fusion.py:479 — Homogeneous 4×4 transform of points. — Internal (many) + external `run_teaser_registration_only.py`, `smoke_test_pipeline_teaser_v13.py`.
- `_extract_isotropic_scale(transform) -> float` — sam3d_fusion.py:486 — Mean column-norm scale of a 3×3 linear block. — 2 internal callers (L1264/1277); 0 external refs.
- `_compose_similarity_transform(scale, rotation, translation) -> np.ndarray` — sam3d_fusion.py:495 — Builds a 4×4 similarity transform. — Internal callers (`_teaser_solve_xyz` L638, teaser/cpd refiners); 0 external refs.
- `_build_explicit_correspondences(source_points, target_points, max_distance)` — sam3d_fusion.py:502 — KD-tree 1-NN correspondences within a radius (for the diagnostic plot). — 1 internal caller (L1266); 0 external refs.
- `_l2_normalize_rows(arr, eps=1e-12) -> np.ndarray` — sam3d_fusion.py:530 — Row L2 normalize. — Internal (FPFH helpers); 0 external refs.
- `_multiscale_fpfh_descriptors(...) -> np.ndarray` — sam3d_fusion.py:535 — Concatenated multi-radius FPFH (+color) descriptors. — 2 internal callers (L860/865, teaser path); 0 external refs.
- `_euclidean_nn_correspondences(src_pts, tgt_pts, max_dist) -> (idx_s,idx_t)` — sam3d_fusion.py:569 — Mutual-NN 3D correspondences for the TEASER reproject pass. — 1 internal caller (L980); 0 external refs.
- `_estimate_normals_np(points, voxel_size, normal_max_nn=30) -> np.ndarray` — sam3d_fusion.py:586 — Open3D normals → numpy. — 2 internal callers (L886/887); 0 external refs.
- `_normal_consistency_filter(...) -> (idx_s,idx_t)` — sam3d_fusion.py:599 — Drops correspondences whose normals disagree > max_deg. — 1 internal caller (L888); 0 external refs.
- `_teaser_solve_xyz(src_xyz, dst_xyz, noise_bound) -> (T,scale)` — sam3d_fusion.py:619 — TEASER++ on paired (3,N) points. — 1 internal caller (L995); 0 external refs.
- `_run_icp_polish(...) -> (final_T, meta)` — sam3d_fusion.py:646 — Scale-aware point-to-plane ICP polish with a count+RMSE acceptance guard. — Internal (L1236) + external `run_teaser_registration_only.py`, `smoke_test_pipeline_teaser_v13.py`.
- `_color_aware_fpfh_descriptors(...) -> np.ndarray` — sam3d_fusion.py:747 — Single-radius FPFH + RGB descriptor block. — 2 internal callers (L871/876); 0 external refs.
- `_mutual_nearest_neighbor(feat_a, feat_b, ratio_thresh=None) -> (idx_a,idx_b)` — sam3d_fusion.py:782 — Mutual-NN feature matching with optional Lowe ratio. — 1 internal caller (L882); 0 external refs.
- `_run_teaser_similarity_refinement(...) -> (T,count,meta)` — sam3d_fusion.py:811 — FPFH→mutual-NN→TEASER similarity (teaser backend stage 1). — Internal (L1193) + external `run_teaser_registration_only.py`, `smoke_test_pipeline_teaser_v13.py`.
- `_run_teaser_reproject_refinement(...) -> (T,count,meta)` — sam3d_fusion.py:949 — 2nd TEASER pass via Euclidean-NN pairs. — Internal (L1222) + external benches.
- `_run_probreg_similarity_refinement(...) -> (T,count,meta)` — sam3d_fusion.py:1015 — probreg CPD similarity refinement (cpd backend). — Internal (L1246) + external `run_teaser_registration_only.py`.
- `register_and_fuse_sam3d_object(...) -> Sam3DInsertionResult` — sam3d_fusion.py:1110 — The Phase-0b driver: rotation init → bbox-scale+centroid → voxel down → backend refine (ndp/cpd/teaser) → correspondences/plot → result. — 19 refs: `dynamic_gs_model.py:1707`, `phase0.py:951`, both `__init__`s, benches.

---

## 2) DEAD-CODE CANDIDATES

Genuine zero-reference suspects (after grep; excluding the protected entry-point
classes from the prompt). Pure-private helpers with a single internal caller are
NOT listed (they're live). Listed below = symbols whose ONLY references are their
own definition / unused.

- **`_largest_extent` — sam3d_fusion.py:453 — confidence HIGH.** `grep -rn "\b_largest_extent\b"` across `dynamic_gs` + `scripts` returns ONLY the definition line. Zero callers anywhere (not internal, not external, not re-exported). Truly dead. (`_bbox_diagonal` is the live extent helper; `_largest_extent` appears to be a leftover from an earlier scale-estimation approach.)

Considered-but-NOT-dead (documented so the re-verifier doesn't re-flag):

- `run_sam3d_single_object` / `run_sam3d_multi_object` (sam3d.py:458/644) — reached only via the `__main__` CLI worker (`_main` L1102/1116), which the `_subprocess` wrappers invoke as a child process. **Entry-point reachable, NOT dead.** (`run_sam3d_multi_object` also has an in-process twin in `sam_worker.py` that the live worker prefers, but the subprocess path is still used by `phase0.py:679` / `live_session.py:402` for recorded datasets.)
- `reconstruct_mesh_from_points` / `reconstruct_mesh_from_gaussian_ply` (sam3d_fusion.py:168/236) — only the `__init__` re-exports show in grep; no live caller. These are the FoundationPose mesh-fallback path, and FoundationPose "is kept on disk but no longer wired into the runtime" (CLAUDE.md). They are PUBLIC API (exported in both `__init__.py` `__all__`) but effectively orphaned at runtime. Flagged as a smell (4.6), not as hard-dead, because they're an exported surface and one calls the other internally.
- All the TEASER/CPD refinement helpers (`_run_teaser_*`, `_run_probreg_*`, `_run_icp_polish`, `_color_aware_fpfh_descriptors`, `_multiscale_fpfh_descriptors`, `_mutual_nearest_neighbor`, `_euclidean_nn_correspondences`, `_normal_consistency_filter`, `_estimate_normals_np`, `_teaser_solve_xyz`, `_compose_similarity_transform`) — live via `register_and_fuse_sam3d_object`'s `teaser`/`cpd` branches + the standalone bench scripts. Default backend is `"ndp"`, so they don't run in the default pipeline, but CLAUDE.md explicitly keeps cpd/teaser selectable. NOT dead.

---

## 3) DATA-LIFECYCLE

### GPU model state (SAM3D `Inference`)
- **Per-resize-attempt model reload (single-object), sam3d.py:559.** `run_sam3d_single_object` constructs a fresh `Inference(...)` *inside* the OOM resize loop and `del`s it in the `finally` (L590-594) every iteration. So on each OOM it pays the full ~11 GB checkpoint reload again at the next smaller size. The multi-object path (L744) loads ONCE before the loop — correct. Lifecycle is leak-free (finally + `gc.collect` + `empty_cache`) but the single-object path is load-thrashy on OOM. (medium)
- **Multi-object model freed once, in `finally` (L905-910).** `output` is `del`'d per-mask (L903) but only when a result is appended; on the OOM/`continue`/empty-mask branches (`all_results.append({})`, L787/847/852/887) the previous `output` (if any) is NOT explicitly del'd before the next iteration — it's overwritten by `output = None` at L794, so it's GC-eligible, but the large gaussian tensors linger until the next `gc.collect` inside the resize loop. Minor; not a true leak.
- **`apply_sam3d_gaussian_trim` (sam3d.py:223)** mutates the loaded model in place (`.half()`, `.to("cpu")`). The fp16 cast is irreversible on that handle — fine because the handle is discarded after inference. Buffers/embedders moved to CPU stay there for the model's life (intended). No free issue.

### File/process handles
- **Subprocess (`subprocess.run`, capture_output=True), sam3d.py:966 / L1047.** Synchronous, fully buffered; child stdout/stderr captured into memory. For a multi-object 1200p run the captured stderr can be large but is bounded. Handles are closed by `subprocess.run`. No leak.
- **Runtime config write (sam3d.py:219).** `_write_runtime_config` overwrites `pipeline_runtime_small.yaml` IN the repo tree (`third_party/Fast-SAM3D/checkpoints/hf/`) on every call. Two concurrent SAM3D runs (e.g. live worker + a bench) would race on this single shared file. (low — single-GPU serial use in practice)
- **Timing sidecar `_sam3d_timing.json` (sam3d.py:915)** written best-effort (swallowed except). Read by the cross-env caller. No handle leak.

### Persistent identity buffers / `.pt` warm-cache
- **None touched directly in these two files.** Neither `sam3d.py` nor `sam3d_fusion.py` reads/writes `post_fusion_state.pt`, the SHM, or the 4 identity buffers (`object_flags` / `object_instance_ids` / `sam3d_init_target_flags` / `inserted_flags`). Those are handled by `persistence/` + `phase0.py` (the caller of `register_and_fuse_sam3d_object`). `object_instance_ids` is set by Phase-0b downstream of `register_and_fuse_sam3d_object`'s output, not here. **Invariant-protected; not in scope of these files.**

### Pose / PLY format round-trips (cross-process contract)
- **Pose sidecar shape contract.** `run_sam3d_*` writes `{translation, rotation, scale}` lists (L603-611 / L880-889); `load_sam3d_pose` reads them as `reshape(-1)` float32 (L101-109); `sam3d_pose_has_rotation` requires `rotation.size == 4` (L120); `load_sam3d_rotation_wxyz` requires `size==4` and finite (L109). Consistent. The producer enforces `len(rotation)==4` before writing (L608/885), so a malformed sidecar can't be produced silently. Good.
- **Pointmap convention coupling.** `_build_pytorch3d_pointmap` (sam3d.py:449) hard-codes the x/y axis flip to match SAM3D's `camera_to_pytorch3d_camera`. This is an undocumented-at-the-boundary contract: if SAM3D's camera convention changes upstream, the pointmap silently mis-projects (no assert/validation). The same convention is duplicated in `SAM3D_P3D_TO_NS_CAMERA = diag([-1,1,-1])` in sam3d_fusion.py:46 — two independent encodings of "p3d↔ns camera" that must stay in sync but are not linked. (medium — desync risk)
- **`run_sam3d_multi_object_subprocess` result reconstruction (sam3d.py:986-996).** Rebuilds output paths from stems and validates `ply exists && rotation`; a per-mask failure yields `{}` in the list — callers must position-match by index. Order is preserved (built from `output_stems` in order). The empty-dict-as-failure sentinel is implicit and easy to mis-handle downstream. (low)

---

## 4) DESIGN SMELLS

- **4.1 Near-duplicate runners — `run_sam3d_single_object` (L458-641) vs `run_sam3d_multi_object` (L644-919). HIGH.** The OOM resize ladder, pointmap resize-to-match, `Inference` ctor + `hfer_2d=0` + `ss_params`/`slat_params`/`mesh_params`/`enable_mesh=False` + `apply_sam3d_gaussian_trim`, pose extraction, and run_info writing are copy-pasted with small differences (single reloads the model per attempt, multi loads once; multi exports mesh, single exports glb). The magic param dicts (`ss_params={"ss_faster_stride":3,...}`) appear verbatim at L563-571 and L762-770. A third copy of the same param block + resize ladder lives in `sam_worker.py`. Any tuning change must be made in 3 places. Recommendation: extract a single `_build_inference(config_obj)` + `_run_inference_with_oom_ladder(inference, image, mask, pointmap, sizes)` helper.

- **4.2 `register_and_fuse_sam3d_object` is a god function — sam3d_fusion.py:1110-1363 (~250 lines). HIGH.** It mixes: input validation, rotation init, bbox-scale + centroid recentering, NN-distance/voxel sizing, a 3-way backend dispatch (ndp/cpd/teaser each with their own sub-stages), correspondence building, plotting, artifact PLY saving, the full `canonical_to_world_4x4` composition, AND a 12-field timing dict. `refinement_meta` / `refinement_meta_key` are conditionally bound per branch and only safe because every branch sets them before use (a 4th backend that forgot would `UnboundLocalError`). Recommendation: split into `_rigid_init`, `_refine(backend, ...)`, `_finalize_result`.

- **4.3 Dead config field threaded but unused in default path — `Sam3DInsertionResult.dedup_threshold` / `kept_points` (sam3d_fusion.py:1298-1300). MEDIUM.** Dedup is hard-disabled (`dedup_threshold = 0.0`, `kept_points = aligned_points`); the docstring at the top of the file (steps 4-6) still describes "append only non-overlapping points / CPD similarity" as "the active fusion path", but the default backend is `ndp` and dedup is off. The module docstring is stale relative to the code. (Also: `visible_source_point_count` / `registration_source_point_count` both just report `len(source_down_points)` — the "visible filtering" they were named for is disabled, per the L1337 comment.)

- **4.4 Misleading timing keys — `register_and_fuse_sam3d_object` timing dict (sam3d_fusion.py:1348-1361). MEDIUM.** The key `"D0.3b3_refinement"` holds a variable named `t_cpd_refinement` (L1189/1212/1255) regardless of backend, and `refinement_meta_key` is `"D0.3b3_cpd_meta"` / `"D0.3b3_teaser_meta"` / `"D0.3b3_ndp_meta"` depending on branch — so a reader scanning for "cpd" gets a backend-agnostic value. Naming carries a stale "cpd-is-default" assumption.

- **4.5 `registration_backend="cpd"` default is dead — sam3d_fusion.py:1120. MEDIUM.** The function signature defaults to `"cpd"`, but every runtime caller passes `model.config.sam3d_registration_backend` which defaults to `"ndp"` (`static_gs_model.py:150`, `dynamic_gs_model.py:162`, `phase0.py:934/951`, `dynamic_gs_model.py:1717`). The `"cpd"` default is never the effective value — a maintainer reading the signature would wrongly conclude CPD is the default backend. Recommendation: default to `"ndp"` to match reality, or drop the default and make it required.

- **4.6 Orphaned public API — `reconstruct_mesh_from_points` / `reconstruct_mesh_from_gaussian_ply` (sam3d_fusion.py:168/236). MEDIUM.** Exported in both `__init__.py` `__all__` but have no runtime caller (only re-exports in grep). They exist for the FoundationPose mesh path, which CLAUDE.md says is "no longer wired into the runtime." Leaving them exported implies they're live API. Recommendation: note as FoundationPose-only / candidate for removal once FP is confirmed gone.

- **4.7 Swallowed exceptions hide failure cause. MEDIUM.**
  - `sam3d_pose_has_rotation` (sam3d.py:117) `except Exception: return False` — a corrupt JSON looks identical to a missing rotation; the caller can't distinguish "no SAM3D run" from "SAM3D wrote a broken sidecar."
  - `apply_sam3d_gaussian_trim` (sam3d.py:259-276) per-module `try/except: pass` — a failed `.half()`/`.to("cpu")` is silently dropped; the returned `{moved_to_cpu, fp16}` dict would just omit it, but VRAM expectations could be silently violated.
  - `_run_probreg_similarity_refinement` (sam3d_fusion.py:1085) `except Exception: return init_transform, 0, _empty_meta` — a CPD solver failure silently returns the unrefined init with `stop_reason="skipped"`, indistinguishable from "cpd not installed."
  - `_save_correspondence_plot` import + `_sam3d_timing.json` write are best-effort; acceptable (diagnostics).

- **4.8 Deeply-threaded optional params — `depth_path` + `intrinsics_path` (+ `depth_scale`, `camera_intrinsics`).** Threaded through `prepare_cropped_sam3d_inputs` → `run_sam3d_single_object_subprocess` → CLI args → `_main` → `run_sam3d_single_object` → `_build_pytorch3d_pointmap` (and the parallel multi-object chain). Five layers, each re-validating `exists()` and re-doing the depth-resize-to-image branch (sam3d.py:370-376, 513-516, 719-722). The depth-resize block is triplicated. (low-medium)

- **4.9 Author-machine + env coupling. LOW.** `_CONDA_ROOT = Path("/home/mrc-cuhk/miniconda3")` (sam3d.py:28) is a hard-coded absolute path; `_resolve_sam3d_python` falls back to `sys.executable` if it's absent, so it degrades gracefully, but it's machine-specific. `_write_runtime_config` mutates a file inside `third_party/` (vendored dep) on every run — config generation writing into a vendored tree is a leaky abstraction.

- **4.10 `min_crop_side` / `padding` semantics comment-only. LOW.** `prepare_cropped_sam3d_inputs` has rich docstring guidance but no validation that `min_crop_side <= image short side`; if the crop side exceeds the image it's clamped by the `crop_x1=min(...)` / `crop_x0=max(0, crop_x1-side)` logic (L344-347) which can silently shrink the crop below `min_crop_side` near image edges. Behavior is correct but the intent (a floor) isn't actually guaranteed. (low)
