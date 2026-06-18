# Code Audit — `dynamic_gs_datamanager.py` + `dynamic_gs_config.py`

Repo root: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`
Modules audited:
- `dynamic_gs/dynamic_gs_datamanager.py` (354 lines)
- `dynamic_gs/dynamic_gs_config.py` (231 lines)

Grep command used for caller counts (excludes the symbol's own definition file where noted):
`grep -rn "<name>" .../dynamic_gs .../scripts --include=*.py`

---

## 1) FUNCTION / CLASS MAP

### `dynamic_gs_datamanager.py`

| Symbol | file:line | Purpose | Callers |
|---|---|---|---|
| `class DynamicFrameDataset(InputDataset)` | datamanager.py:20 | InputDataset subclass that loads per-frame depth (uint16 mm TIFF → metres) and excludes `depth_image` from device transfer. | **NO EXTERNAL REFS** — only instantiated indirectly via `DynamicFrameFullImageDatamanager` (set as `cfg._target` at datamanager.py:155). Reachable, not dead. |
| `DynamicFrameDataset.__init__(self, dataparser_outputs, scale_factor=1.0, cache_compressed_images=False)` | datamanager.py:25 | Validates `depth_filenames` present, stores depth scale factor. | Called by FullImageDatamanager base via `create_train_dataset`/`create_eval_dataset` (nerfstudio interface). |
| `DynamicFrameDataset.get_metadata(self, data)` | datamanager.py:37 | Loads depth image for `data["image_idx"]`, returns `{"depth_image": ...}`. | nerfstudio `InputDataset.__getitem__` interface (no direct repo ref). |
| `class DynamicFrameFullImageDatamanager(FullImageDatamanager[DynamicFrameDataset])` | datamanager.py:52 | Typed FullImageDatamanager whose dataset type is `DynamicFrameDataset`. Empty body. | Set as `cfg._target` at datamanager.py:155 (`_build_manager`, dynamic path). NO other refs. |
| `class DynamicGSDataManagerConfig(DataManagerConfig)` | datamanager.py:56 | Config for the wrapper datamanager. `_target=DynamicGSDataManager`. | 16 refs — config.py:45/103/175/217 (all 4 MethodSpecs), static_gs_pipeline.py:74, dynamic_gs_pipeline_base.py:105, static_gs_preseg_pipeline.py:55, __init__.py lazy-export. **Entry-point reachable.** |
| `class DynamicGSDataManager(DataManager)` | datamanager.py:97 | Wraps two `FullImageDatamanager`s (static + dynamic); pins one dynamic frame at a time; phase switch. | 22 refs — `_target` of the config; exported lazily from `__init__.py`; referenced in phase0.py:417 docstring, static_gs_model.py:208 comment. Instantiated by nerfstudio via `cfg.setup()`. |
| `DynamicGSDataManager.__init__(...)` | datamanager.py:102 | Builds both managers, runs the keyframe filter, sets phase to static. | nerfstudio `setup()` path. |
| `._build_manager(self, data_path, use_depth_dataset)` | datamanager.py:148 | Deep-copies inner config, points it at a subdir, swaps dataset type for the depth path, calls `.setup()`. | **NO REFS FOUND** outside file (called internally twice, datamanager.py:129/137). Internal-only — not dead. |
| `._filter_static_keyframes(self, *, translation_thresh_m, rotation_thresh_deg)` | datamanager.py:163 | ORB-SLAM greedy keyframe dedup on the static train set; mutates dataparser_outputs in place. | 1 ref (doc only): keyframe_filter.py:20. Called internally at datamanager.py:132. Internal-only — not dead. |
| `.set_phase(self, phase)` | datamanager.py:239 | Switches `active_manager` between static/dynamic; rebinds `train_dataset`/`eval_dataset`/samplers/dataparser_outputs. | recorded:113-114, live:123-124. **Live-path.** |
| `.set_dynamic_frame_idx(self, frame_idx)` | datamanager.py:249 | Range-checked set of `current_dynamic_frame_idx`. | recorded:190 only. **Recorded-only.** |
| `.get_num_dynamic_frames(self)` | datamanager.py:255 | `len(dynamic_manager.train_dataset)`. | recorded:117,176. Recorded-only. |
| `.get_dynamic_frame_name(self, frame_idx)` | datamanager.py:258 | Stem of the frame's image filename. | base:2594,3016. |
| `.get_current_dynamic_frame_name(self)` | datamanager.py:261 | Name of the currently-pinned dynamic frame. | base:2135. |
| `.get_initialization_debug_dir(self)` | datamanager.py:264 | `<data>/dynamic_scene/initialization_debug`. | phase0.py:434,810 (+418 doc). |
| `.get_initialization_artifact_dir(self)` | datamanager.py:267 | `<data>/dynamic_scene/initialization_artifacts`. | phase0.py:435,811 (+419 doc). |
| `.get_dynamic_debug_dir(self)` | datamanager.py:270 | Alias returning `get_initialization_debug_dir()`. | **NO REFS FOUND** — dead (see §2). |
| `._get_dynamic_batch(self, frame_idx, split)` | datamanager.py:273 | Copies cached frame, moves image/mask to device, builds a 1-camera slice with `cam_idx` metadata. | **NO REFS FOUND** outside file (called internally at 294/297). Internal. |
| `.get_current_dynamic_train_batch(self)` | datamanager.py:293 | `_get_dynamic_batch(current, "train")`. | recorded:191 only. **Recorded-only.** |
| `.get_current_dynamic_eval_batch(self)` | datamanager.py:296 | `_get_dynamic_batch(current, "eval")`. | **NO EXTERNAL REFS** — called only internally at 302/323/328. |
| `.fixed_indices_eval_dataloader` (property) | datamanager.py:299 | Returns `[eval batch]` in dynamic phase, else delegates. | **NO REFS FOUND** — nerfstudio eval interface (see §2). |
| `.setup_train(self)` / `.setup_eval(self)` | datamanager.py:305/308 | No-ops (return None). | nerfstudio `DataManager` interface (called by trainer). |
| `.forward(self)` | datamanager.py:311 | `raise NotImplementedError`. | Abstract-method satisfier (nn.Module/DataManager). Never called. |
| `.next_train(self, step)` / `.next_eval(self, step)` / `.next_eval_image(self, step)` | datamanager.py:314/320/326 | Phase-aware batch fetch; in dynamic phase returns current frame, else delegates to active_manager. | nerfstudio trainer interface. No direct repo ref. |
| `.get_train_rays_per_batch(self)` / `.get_eval_rays_per_batch(self)` | datamanager.py:331/338 | W·H of the active camera. | nerfstudio interface. |
| `.get_datapath(self)` | datamanager.py:346 | Delegates to active_manager. | nerfstudio interface. |
| `.get_param_groups(self)` | datamanager.py:349 | Delegates to active_manager (returns `{}` for FullImage). | nerfstudio interface. |
| `.get_training_callbacks(self, attrs)` | datamanager.py:352 | Delegates to active_manager. | nerfstudio interface (pipeline base also defines one at base:1084). |

### `dynamic_gs_config.py`

| Symbol | file:line | Purpose | Callers |
|---|---|---|---|
| `STATIC_NUM_STEPS = 500` | config.py:31 | Static-phase iteration count. | Used config.py:41/44/99/101. Documented in CLAUDE.md invariant #1. |
| `StaticGS` (MethodSpecification) | config.py:33 | `static-gs` method. | **Entry point** pyproject.toml:24; __init__.py:141 lazy export. |
| `StaticGSPreseg` (MethodSpecification) | config.py:91 | `static-gs-preseg` method. | **Entry point** pyproject.toml:25. |
| `_ZERO_LR_OPTIMIZERS` (dict) | config.py:138 | All-zero LR optimizer block for dynamic phases (invariant #4). | config.py:183/225. CLAUDE.md invariant #4. |
| `DEFAULT_DYNAMIC_RECORDED_STEPS = 5000` | config.py:160 | Iteration cap for recorded dynamic. | config.py:172. |
| `DynamicGS` (MethodSpecification) | config.py:164 | `dynamic-gs` recorded method. | **Entry point** pyproject.toml:22. |
| `DEFAULT_DYNAMIC_LIVE_STEPS = 10**9` | config.py:202 | Effectively-infinite cap for live. | config.py:214. |
| `DynamicGSLive` (MethodSpecification) | config.py:206 | `dynamic-gs-live` live method. | **Entry point** pyproject.toml:23. **LIVE-PATH.** |

---

## 2) DEAD-CODE CANDIDATES

Entry points (MethodSpecs, `_ZERO_LR_OPTIMIZERS`, all nerfstudio `DataManager` interface overrides, invariant buffers) are excluded.

| Symbol | file:line | Evidence | Confidence |
|---|---|---|---|
| `get_dynamic_debug_dir` | datamanager.py:270 | `grep get_dynamic_debug_dir … = 0 refs` outside definition. Pure alias of `get_initialization_debug_dir`. Genuinely dead. | **high** |
| `static_total_frames` (attr) | datamanager.py:130 | `grep static_total_frames` → only the assignment line. Never read anywhere (internal or external). Computed pre-filter count, dead. | **high** |
| `static_accepted_frames` (attr) | datamanager.py:136 | `grep static_accepted_frames` → only the assignment line. Never read. (The kept-count is also logged inside `_filter_static_keyframes`, so this attribute is redundant.) | **high** |
| `get_current_dynamic_eval_batch` | datamanager.py:296 | No external refs; only reached via `fixed_indices_eval_dataloader` (datamanager.py:302) and `next_eval`/`next_eval_image` (323/328). Those callers are themselves never exercised (see below) — so this is dead *in practice* but on the nerfstudio eval interface. | **medium** |
| `fixed_indices_eval_dataloader` (property) | datamanager.py:299 | 0 repo refs. It is the nerfstudio eval-loop interface, but eval is fully disabled in all 4 MethodSpecs (`steps_per_eval_*` = 1e9 / 0). So it is never invoked. Keep (interface contract) but note unreachable. | **medium** |
| `next_eval` / `next_eval_image` | datamanager.py:320/326 | nerfstudio eval interface; eval is disabled in every MethodSpec (see config.py:37-39 etc.). Unreachable at runtime but part of the abstract contract — NOT recommended to delete. | **low** (interface) |
| `forward` | datamanager.py:311 | `raise NotImplementedError`. Abstract-method placeholder; never called. Keep (satisfies base). | **low** (interface) |

> NOT flagged dead (verified reachable): `_build_manager`, `_filter_static_keyframes`, `_get_dynamic_batch` (internal callers), `DynamicFrameDataset`/`DynamicFrameFullImageDatamanager` (instantiated via `_target`).

---

## 3) DATA-LIFECYCLE

This module is the *data source* layer. It does NOT touch the `.pt` warm-cache (that is `persistence/`), SHM (that is `live_shm_reader.py`), or the 4 identity buffers (those live on the model). What it owns: two wrapped `FullImageDatamanager`s and their image/depth/mask caches.

1. **Two FullImageDatamanager caches, both CPU-resident.** `_build_manager` (datamanager.py:148) is called twice → `static_manager` (datamanager.py:129) and `dynamic_manager` (datamanager.py:137). Inner config sets `cache_images="cpu"` (config.py:92) deliberately (the comment cites the 16 GB OOM at 1200p). **Both** caches are fully materialized at `__init__` and held for the whole run — `dynamic_manager`'s `cached_train` is loaded even in **live mode where it is never read** (live pipeline comment at live:120-121 says it "doesn't actually pull from the datamanager at runtime"). So in live mode the entire dynamic image+depth+mask cache is loaded and pinned in CPU RAM but never consumed — a wasted-load. Not a leak (freed at process exit), but a sizeable idle allocation on the live path. **(medium)**

2. **Static cache built BEFORE the keyframe filter trims it.** `__init__` order: `static_manager` setup (datamanager.py:129) → `_filter_static_keyframes` (132). But `FullImageDatamanager.setup()` already materialized `cached_train` for ALL frames before the filter prunes `image_filenames`/`cameras` (datamanager.py:224-228). The filter shrinks `cameras`/`image_filenames` and re-samples `train_unseen_cameras` (datamanager.py:232) but does **not** prune `cached_train`/`cached_eval`. So the dropped frames' cached tensors stay resident, and there is now a **length mismatch** between `len(cameras)` (trimmed) and `len(cached_train)` (untrimmed). For static training this is masked because `sample_train_cameras()` only yields kept indices — but any code that indexes `cached_train` by the new contiguous range, or zips cameras↔cache, would desync. **(medium — latent desync / stale-cache retention)**

3. **`train_unsampled_epoch_count` deleted, possibly absent.** datamanager.py:230-231 `delattr` is guarded by `hasattr`, fine. But `train_unseen_cameras = sample_train_cameras()` (232) is called unconditionally — relies on the base attr existing. Safe for FullImageDatamanager; brittle to base refactors. **(low)**

4. **Per-frame batch copy + per-tick H2D allocation.** `_get_dynamic_batch` (datamanager.py:281) does `cached[frame_idx].copy()` (shallow dict copy) then `.to(self.device)` on `image` and `mask` (282-284) and `cameras[i:i+1].to(device)` (287) — a fresh GPU allocation **every recorded tick**. Note `depth_image` is NOT moved here (it's in `exclude_batch_keys_from_device`, datamanager.py:23) and is left as a CPU view into the shared `cached` tensor. `.copy()` is shallow, so `data["depth_image"]` aliases the cache entry — if any downstream consumer mutates `batch["depth_image"]` in place it corrupts the cache. recorded.py:196 does `batch["depth_image"] = filter_depth_torch(...)` which **reassigns** the key (no in-place write) so the cache is safe today, but it's a sharp edge. **(low — aliasing footgun, recorded-only)**

5. **No model/process/file handles opened here.** No `open()`, no torch.save/load, no mmap, no subprocess. Lifecycle risk is confined to the two image caches above.

6. **Format/shape assumptions on depth.** `get_metadata` (datamanager.py:37) reads H/W from `cameras.height/width[image_idx]` and applies `depth_unit_scale_factor * dataparser_scale`. With `auto_scale_poses=False` (config.py:84) `dataparser_scale==1.0`, so depth is metres. If a dataset is ever loaded with scaling on, depth would be silently rescaled — coupling that is implicit, not asserted. **(low)**

---

## 4) DESIGN SMELLS

1. **Live-mode loads an unused dynamic dataset cache.** (Cross-ref lifecycle #1.) `DynamicGSDataManager.__init__` unconditionally builds `dynamic_manager` with full CPU caching (datamanager.py:137), but the live pipeline never calls `get_current_dynamic_train_batch` (live:120-121 comment). The whole recorded-batch surface (`_get_dynamic_batch`, `get_current_dynamic_train_batch`, `get_current_dynamic_eval_batch`, `set_dynamic_frame_idx`, `next_train` dynamic branch) is **unreachable in live mode** — frames come from SHM. This is the largest smell given the live-path purge priority: a whole branch of the class is recorded-only dead weight on the live path. **(medium)**

2. **Keyframe-filter cache desync (lifecycle #2) — leaky abstraction.** The filter mutates `dataparser_outputs.cameras`/`image_filenames` but leaves the already-built `cached_train` untouched, relying on undocumented FullImageDatamanager internals (`train_unseen_cameras`, `sample_train_cameras`, `train_unsampled_epoch_count`) to stay consistent. Reaches deep into base-class private state. **(medium)**

3. **Dead alias `get_dynamic_debug_dir`** (datamanager.py:270) duplicates `get_initialization_debug_dir`. Zero callers. Pure noise. **(low)**

4. **Two write-only attributes** `static_total_frames` / `static_accepted_frames` (datamanager.py:130/136) — set, never read. The "accepted" count is also separately logged inside the filter. Dead state. **(low)**

5. **Misleading naming: "static keyframe filter" config knobs live on the *DataManager* config but are described in terms identical to the capture-time `keyframe_filter.py` (2cm/20°).** Two independent dedup mechanisms with the same thresholds (`static_keyframe_translation_m=0.02`, `static_keyframe_rotation_deg=20.0`, config-side default) vs the capture-side `keyframe_filter.py`. Easy to confuse which one ran. The docstring says "OR semantics" (datamanager.py:175) but the code at datamanager.py:209 rejects when `(dt<=t) & (dr<=r)` — i.e. a frame is *kept* unless it is close in BOTH → the keep rule is "far in T OR far in R", which matches the doc, but the inline `near = … & …` reads as AND and is easy to misread. **(low — confusing naming/comment vs code)**

6. **`reuse_sam3d_generated_ply=True` set on `static-gs-preseg`** (config.py:111) even though preseg has no SAM3D step — kept "truthy so the validator doesn't complain" (config.py:108-110). A config field forced to a meaningless value to dodge a validator is a smell pointing at an over-eager validator on `StaticGSModelConfig`. **(low)**

7. **Duplicated optimizer block.** `static-gs` (config.py:63-69) and `static-gs-preseg` (config.py:115-121) hand-duplicate the identical 7-group optimizer dict (means=0, camera_opt=1e-3, etc.). Two copies that must stay in sync by hand; the file even has a shared `_ZERO_LR_OPTIMIZERS` for the dynamic pair but no shared `_STATIC_OPTIMIZERS`. **(low)**

8. **Eval interface fully wired but globally disabled.** `next_eval`/`next_eval_image`/`fixed_indices_eval_dataloader`/`get_eval_rays_per_batch`/`get_current_dynamic_eval_batch` all exist and branch on phase, but every MethodSpec disables eval (`steps_per_eval_*` 1e9/0, config.py:37-39/95-97/168-171/210-213). Substantial unreachable surface kept only to satisfy the base contract. **(low)**

### Thread-safety note (live-path priority)
The datamanager itself is **not** touched by the feedforward bg thread, tracker thread, or viser render thread at runtime in live mode (frames bypass it via SHM). `set_phase` (datamanager.py:239) is called **once** at D0 from the tick thread before the threaded FF/render machinery spins (live:123-124) — no concurrent mutation of `active_manager`/`train_dataset` observed. So no datamanager-level race exists on the live path. The recorded path's `_get_dynamic_batch` per-tick `.to(device)` (datamanager.py:282-287) runs on the single tick thread; the shallow-copy depth-aliasing (lifecycle #4) is the only sharp edge, and it is currently safe because recorded.py:196 reassigns rather than mutates in place.
