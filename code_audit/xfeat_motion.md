# Audit: `dynamic_gs/utils/xfeat_motion.py`

XFeat sparse + LighterGlue matcher with a multi-anchor keyframe pool. **LIVE-PATH module** — the only surviving dynamic tracker (5-tracker dispatch purged 2026-05-26). Constructed once per dynamic run at `_initialize_motion_estimator` (`dynamic_gs_pipeline_base.py:1993`), driven on the **tracker thread** via `_apply_motion_estimator`.

Grep base for caller counts:
`grep -rn "<sym>" /home/.../scripts/dynamic_gs /home/.../scripts/scripts --include=*.py` (own-file matches subtracted; diag-script same-name redefinitions treated as NOT-a-ref).

---

## 1) FUNCTION / CLASS MAP

### Module-level

- `ROTATION_GATE_DEG: float = 22.5` — xfeat_motion.py:84 — new-anchor rotation gate. Default for the ctor `anchor_rotation_gate_deg`. **0 external refs** (ctor uses the value via the kwarg default; the symbol itself is only referenced inside this file). Not dead — feeds defaults.
- `SCALE_GATE_RATIO: float = 1.3` — xfeat_motion.py:92 — apparent-scale anchor gate. Same: **0 external refs**, used as kwarg default + internal.
- `_MIN_INLIERS_DEFAULT: int = 20` — xfeat_motion.py:98 — fallback inlier floor for 2nd-anchor retry. **0 external refs**; used as the default for `anchor_min_inliers`.
- `class _Anchor` (dataclass) — xfeat_motion.py:101 — one keyframe in the pool (descriptors, keypoints_gpu, world_3d, rotation/translation, camera_rotation, camera_distance, rgb, mask). **0 external refs** — internal type only.
- `_rotation_distance_deg(R_a, R_b) -> float` — xfeat_motion.py:140 — geodesic SO(3) distance in degrees. **0 external refs**; called internally by `_select_nearest_anchor_by_rotation`, `_needs_new_anchor`.
- `_relative_object_rotation(object_R_world, camera_R_world) -> np.ndarray` — xfeat_motion.py:151 — object-in-camera relative rotation `R_cam^T @ R_obj`. **0 external refs**; called internally (selection, gating, `estimate_and_advance`).
- `_scale_ratio(dist_a, dist_b) -> float` — xfeat_motion.py:166 — symmetric apparent-scale ratio (>=1). **0 external refs**; internal.
- `_select_nearest_anchor_by_rotation(anchors, predicted_relative_rotation, *, exclude, current_distance, rotation_gate_deg, scale_gate) -> int` — xfeat_motion.py:175 — picks the anchor whose stored view is most similar (rotation + folded scale penalty). **0 external refs**; called in the per-tick attempt loop.
- `_needs_new_anchor(anchors, relative_rotation, current_distance, rotation_gate_deg, scale_gate) -> (bool,float,float)` — xfeat_motion.py:210 — decides whether the current view needs a fresh anchor (must miss BOTH gates). **0 external refs**; internal.
- `_ensure_repo_on_path() -> None` — xfeat_motion.py:242 — inserts `third_party/xfeat` onto `sys.path`. **0 external refs to THIS symbol** (the 2 grep hits in `scripts/test_foundationpose_static_scene.py:40,100` are a separate, unrelated definition). Called once in `__init__`.

### `XFeatMotionEstimator` (xfeat_motion.py:250)

- `__init__(device, top_k, detection_threshold, min_cossim, min_track_points, ransac_iterations, ransac_inlier_threshold, weights_path, anchor_min_inliers, anchor_rotation_gate_deg, anchor_scale_gate, use_lighterglue, lighterglue_min_conf, lighterglue_depth_confidence, object_search_radius_px, use_semi_dense, pose_filter_*, static_hold_*)` — xfeat_motion.py:253 — builds XFeat + LighterGlue, the KF, the static-hold deque, the anchor pool. **1 caller**: `dynamic_gs_pipeline_base.py:1993`.
- `ready` (property) — xfeat_motion.py:447 — True once D0 anchor seeded with >= min_track_points. **2 callers**: `dynamic_gs_pipeline_base.py:2045,2238` (log lines); also read internally in `estimate_and_advance`.
- `current_track_count` (property) — xfeat_motion.py:456 — returns `last_inlier_count`. **0 external refs**; used internally (return-struct fields). See DEAD-CODE candidate note.
- `initialize(rgb, depth, camera, mask) -> int` — xfeat_motion.py:466 — seeds the D0 anchor (identity pose), resets KF + static-hold history, sets `_centroid_d0`. **1 caller**: `dynamic_gs_pipeline_base.py:2036`.
- `estimate_and_advance(current_rgb, current_depth, current_camera, current_mask=None, current_object_mask=None) -> MotionEstimate` — xfeat_motion.py:578 — the per-tick god method: prep, extract, anchor-attempt loop (match+RANSAC), compose cumulative pose, KF, static-hold, anchor creation. **1 caller**: `dynamic_gs_pipeline_base.py:2102`.
- `_extract(rgb_hwc) -> (kp_np, desc, kp_gpu, (w,h))` — xfeat_motion.py:1039 — XFeat detect+describe forward. **0 external refs to THIS method** (grep hits in `dynamic_gs_pipeline_base.py` are the timing-key STRING `"DN.3c_xfeat_extract"`; `scripts/diag_*.py` have own local `_extract`s). Called internally by `initialize`, `estimate_and_advance`.
- `_mnn_match(desc_a, desc_b) -> (idx_a, idx_b)` (staticmethod) — xfeat_motion.py:1083 — MNN fallback when LighterGlue absent. **0 external refs**; called in `_match_and_solve_against_anchor`.
- `_lighterglue_match(kp_a_gpu, desc_a, size_a, kp_b_gpu, desc_b, size_b) -> (idx_a, idx_b)` — xfeat_motion.py:1102 — LighterGlue match. **0 external refs**; called in `_match_and_solve_against_anchor`.
- `_current_camera_object_distance(camera_to_world) -> float` — xfeat_motion.py:1144 — cam↔object-centroid distance for the scale gate. **0 external refs**; called twice in `estimate_and_advance`.
- `_build_anchor(*, keypoints, keypoints_gpu, descriptors, image_size, depth_values, depth_valid, intrinsics, camera_to_world, rotation, translation, rgb=None, mask=None) -> Optional[_Anchor]` — xfeat_motion.py:1156 — filters to depth-valid, backprojects, computes camera_distance, returns anchor. **0 external refs**; called by `initialize` + `estimate_and_advance`.
- `_pre_mask_image(rgb, mask, erode_px=5) -> Tensor` (staticmethod) — xfeat_motion.py:1224 — zeros pixels outside eroded mask BEFORE XFeat. **0 external refs; only a docstring mention at :499.** See DEAD-CODE.
- `_restrict_depth_valid_to_image_mask(keypoints, depth_valid, image_mask) -> np.ndarray` — xfeat_motion.py:1248 — AND depth-validity with an image mask (min_track_points fallback). **0 external refs**; called by `initialize` + the anchor-creation block.
- `_compose_keep_region(current_mask, current_object_mask, image_hw) -> Optional[np.ndarray]` — xfeat_motion.py:1281 — combines gripper-keep + object masks. **0 external refs; ZERO internal callers either.** See DEAD-CODE (the per-tick path inlines this recipe at :656-668 instead).
- `_match_and_solve_against_anchor(*, anchor, curr_keypoints, curr_keypoints_gpu, curr_descriptors, curr_image_size, curr_world_all, curr_depth_valid, keep_region, image_hw) -> dict` — xfeat_motion.py:1298 — match + mask/depth filter + Kabsch-RANSAC against one anchor. **0 external refs**; called in the attempt loop.
- `_prepare_rgb_gpu(image) -> Tensor` — xfeat_motion.py:1431 — GPU-native HWC float 0..255 prep, no host sync. **0 external refs**; called by `initialize` + `estimate_and_advance`.
- `_maybe_warmup(hw) -> None` — xfeat_motion.py:1463 — warms cuDNN + LighterGlue per resolution (idempotent). **0 external refs**; called in `_extract`.
- `_mask_to_numpy(mask, output_shape) -> np.ndarray|None` (staticmethod) — xfeat_motion.py:1499 — Tensor/ndarray → bool HxW. **0 external refs**; called 5× internally.

**Diagnostic / init fields** (read by pipeline logging, `dynamic_gs_pipeline_base.py:2041-2045`): `last_init_fast_point_count`, `last_init_sampled_count`, `last_init_depth_valid_count`, `last_init_used_dense_fallback` — all **have a consumer**. `min_track_points` read at `:2047,2050`.

**Fields with NO external consumer** (set but only used internally or never read outside): `last_anchor_idx_used`, `last_used_fallback_anchor`, `last_pool_size` — **0 external refs** (grep returned nothing). `last_inlier_count` is surfaced only through `current_track_count`, which itself has 0 external refs.

---

## 2) DEAD-CODE CANDIDATES

Genuine zero-ref suspects (after subtracting own-file matches and same-name redefinitions in diag scripts). None of these are entry points / callbacks / monkeypatch targets / invariant-protected buffers.

| Symbol | file:line | Ref evidence | Confidence |
|---|---|---|---|
| `_compose_keep_region` | xfeat_motion.py:1281 | 0 external refs AND 0 internal callers — the only grep hit is its own `def`. The per-tick keep-region recipe is **inlined** at :656-668 (`keep_region = gripper_keep_np`; dilate object halo; `&`). This method is an orphaned earlier extraction. | high |
| `_pre_mask_image` | xfeat_motion.py:1224 (staticmethod) | 0 external refs; only a *docstring* reference at :499. No code call. Docstrings note the D0/anchor paths now extract on the FULL image then post-filter (no pre-masking) — this is the old pre-mask path left behind. | high |
| `current_track_count` (property) | xfeat_motion.py:456 | 0 external refs (`grep current_track_count` outside file = empty). Used internally as `self.current_track_count` at :588,623,707 (track_count_before/after). Borderline: it has internal readers, so it is not strictly dead, but its documented purpose ("what the pipeline reads to gauge tracking health") is unfulfilled — no pipeline reads it. | low |
| `last_anchor_idx_used` | xfeat_motion.py:430,866 | set on every tick; **0 external refs**. Pure write-only diagnostic. | medium |
| `last_used_fallback_anchor` | xfeat_motion.py:431,809 | set on every tick; **0 external refs**. Write-only. | medium |
| `last_pool_size` | xfeat_motion.py:433,562,1008 | set; **0 external refs**. Write-only. | medium |

NOT flagged (verified live-reachable): `_mnn_match` (LighterGlue-unavailable fallback in `_match_and_solve_against_anchor:1329` — reachable when `use_lighterglue=False` / kornia missing), `_maybe_warmup`, `_extract`, `_build_anchor`, `_match_and_solve_against_anchor`, all selection/gate helpers.

`_use_semi_dense` branch (xfeat_motion.py:1066-1073, 1071, 1474-1479): wired to config `xfeat_use_semi_dense` (`dynamic_gs_model.py:271`, default `False`) → constructor `use_semi_dense` → `_extract`/`_maybe_warmup`. **Unreachable in live mode at the default** (`detectAndComputeDense` path), but it IS a config-gated branch, not dead. Note: dead config below.

---

## 3) DATA-LIFECYCLE

This module owns **no** `.pt` warm-cache, no SHM, and none of the 4 identity buffers (those live on the model; this estimator is a transient pipeline member, rebuilt each dynamic run — `dynamic_gs_pipeline_base.py:393,1156,1993`). It does own significant **GPU tensors** in the anchor pool, plus the XFeat + LighterGlue models.

### Anchor pool — UNBOUNDED GPU growth (the headline lifecycle issue)

- **Create**: `_anchors` starts `[]` (:362), reset to `[]` in `initialize` (:483), D0 appended (:561), new anchors appended in `estimate_and_advance` (:1007).
- **Per anchor**: `descriptors` (N×64 GPU), `keypoints_gpu` (N×2 GPU), and crucially **`rgb` = a full-frame HWC float tensor cloned onto GPU** (`_build_anchor:1206 rgb.detach().clone()`; stored at :1220). The docstring (:113-115) cites ~2.6 MB/anchor at 1280×720, but at the live 1920×1200 path it is ~27.6 MB/anchor (1920·1200·3·4 bytes) of float32 RGB held on GPU.
- **No cap, no eviction, no pruning**: there is no upper bound on `len(self._anchors)` anywhere (grep confirms appends at :561,1007 and resets only in `initialize`). A long sweep (CLAUDE.md notes "~16 anchors across a full 360°", but camera-only + scale gates add more) accumulates anchors for the entire run. Each one pins a full-res RGB tensor + descriptors on the GPU.
- **Free**: only on the next `initialize` (D0 reseed, e.g. "Change object" / re-pick) which drops the whole list to `[]` — Python ref-drop frees the GPU tensors. Between reseeds, monotonic growth. **Leak class: bounded-by-run-length, not bounded-by-a-cap.** On the same card hosting the splat scene (459k→1.29M gauss per CLAUDE notes) this competes for VRAM. Severity raised because this is the live path being scaled to 1200p.
- **`mask`** (:1208-1209): also cloned/copied per anchor (HxWx1 GPU tensor when built from `anchor_keep_region`, :992-996). Same unbounded accumulation. Stored purely for the debug visualizer's red border.

### XFeat + LighterGlue models

- Loaded eagerly in `__init__` (`XFeat(...)` :315; `LighterGlue(...).to(...).eval()` :328). Held for the estimator's lifetime. **Never explicitly freed/unloaded** — relies on the estimator object being GC'd at process exit / reseed-replacement. For a transient that is rebuilt on reseed (`_initialize_motion_estimator` at :1184 re-assigns `self._motion_estimator`), the OLD estimator (and its XFeat/LighterGlue + entire anchor pool) is dropped only when the new assignment replaces the reference — fine, but no `del`/`empty_cache`, so VRAM is reclaimed lazily.

### Per-tick heap / GPU allocations (live hot path)

- `_extract` ends in `kp_gpu.cpu().numpy()` (:1080) — a host sync every tick (intentional; the `gpu_queue_wait` split at :678-681 is a SECOND `torch.cuda.synchronize()` added per tick for diagnostics — pure overhead on the live path, every tick, unconditionally when CUDA is available).
- `cv2.dilate` on the object halo every tick (:664-667) allocates a new uint8 frame-sized array.
- `_lighterglue_match` builds a fresh `data` dict with `.to(dev)[None].float()` copies of keypoints + descriptors every call (:1122-1133), plus two `torch.tensor([...])` image_size allocations per call (×1-2 attempts/tick).
- `_mask_to_numpy` × up to 5/tick, each potentially `cv2.resize` + `.cpu().numpy()`.
None are leaks (all transient), but they are per-tick churn on a module flagged for Hz.

### State that can desync

- **`_centroid_d0`** (:566) is the FIXED D0 object centroid. `_build_anchor`'s camera_distance and `_current_camera_object_distance` both push it through the cumulative pose. If `initialize` is called again (reseed) it is correctly recomputed. No desync within a run.
- **`_inlier_hist`** (:837-839,851-853) and **`_kf_tick`** (:890) are LAZILY created via `getattr(self, ..., default)` rather than initialized in `__init__`. On a reseed, `initialize` (:481-486) does **NOT** clear `_inlier_hist` or `_kf_tick` — only `_pose_filter.reset()`, `_static_hold_hist.clear()`, `_anchors`, `_cumulative_*`. So after a mid-run object switch, the spike-gate history and the KF synthetic-tick counter carry over from the previous object. `_kf_tick` carry-over only shifts synthetic time forward (harmless to the constant-velocity KF after `reset()`); `_inlier_hist` carry-over (only active when `DGS_SPIKE_GATE_FRAC>0`, off by default) could mis-gate the first ~8 ticks of the new object. Low severity given the env-gate is off by default, but it is a real reset-incompleteness.
- **Diagnostic fields** (`last_*`) are cleared at the top of `estimate_and_advance` (:590-592) but NOT all of them (`last_pool_size` persists). Harmless since unconsumed.

### Threading / lifecycle

- The estimator is touched **only** on the tracker thread (`_apply_motion_estimator` → `estimate_and_advance`; `_initialize_motion_estimator` → `initialize`). Grep of `_motion_estimator` across all three pipeline files shows no access from `_feedforward_threaded` / `_anysplat_bg_run` / the viser render thread. So **no cross-thread mutation of the anchor pool or `_cumulative_*`** — the FF/render/tracker race the CLAUDE invariants warn about is on the MODEL (`_model_lock`), not on this estimator. Good. (One caveat: the per-tick `torch.cuda.synchronize()` at :680 and the implicit `.cpu()` sync at :1080 serialize against whatever the FF/viser threads enqueued on the default CUDA stream — that's a GPU-contention cost, not a data race.)

---

## 4) DESIGN SMELLS

- **God method `estimate_and_advance` (xfeat_motion.py:578-1033, ~455 lines).** Does input prep, resolution check, mask compose (inlined), GPU sync diagnostic, extract, backproject-all, anchor-selection-and-attempt loop, pose compose, spike gate, Kalman filter, static-hold median, anchor-creation (with its own ~70-line filter/build block at :944-1013), and result packing. High severity for a module being purged/refactored — almost everything interesting lives in one function. The anchor-creation block (:944-1013) duplicates the D0-seed filtering logic in `initialize` (:517-543) — same "extract full image → post-filter by mask → depth-valid → `_build_anchor`" recipe written twice.

- **Duplicated keep-region recipe.** `_compose_keep_region` (:1281) encodes "object ∩ gripper-keep", but the live tick inlines a richer variant (gripper ∩ dilated object halo) at :656-668, and the anchor-creation path inlines yet another (`keep_region & obj_mask_for_debug`) at :951-955. Three places, one concept, one of them dead.

- **Dead config field: `xfeat_min_cossim` / `min_cossim`.** Declared `dynamic_gs_model.py:247` (`xfeat_min_cossim: float = -1.0`), threaded through the ctor (`dynamic_gs_pipeline_base.py` passes `min_cossim=...`), stored at xfeat_motion.py:285 (`self.min_cossim = float(min_cossim)`) — **and never read again** (grep: only :258 param, :285 assignment). A config knob + constructor param + instance attr that does nothing.

- **`current_track_count` leaky/misleading.** Documented as "what the pipeline reads to gauge tracking health" (:459) but **no pipeline reads it** (0 external refs). The pipeline reads `last_init_*` and the returned `MotionEstimate` instead. Misleading docstring.

- **Write-only diagnostics.** `last_anchor_idx_used`, `last_used_fallback_anchor`, `last_pool_size` are maintained every tick but never consumed anywhere in the repo. Maintenance cost (and the `last_used_fallback_anchor` logic at :809) for no readers.

- **Legacy-name leakage / misleading naming.** The return type is aliased `MotionEstimate as CoTrackerMotionEstimate` (:64) and timing keys carry dead-tracker names: `timings["klt_forward"]` is written as a duplicate of `xfeat_extract` (:694) "for backward compatibility", `timings["postprocess"]=0.0` and `timings["resample"]=0.0` are CoTracker-era slots always zero (:819,1014, comments admit no such stage exists). The module docstring still calls XFeat the "Fourth backend ... alongside CoTracker, TAPIR, and KLT" (:3-4) — all three were deleted 2026-05-26. Stale.

- **Swallowed exceptions.** `__init__` LighterGlue construction (:344-350) catches `Exception` and silently falls back to MNN (logged, acceptable). The `depth_confidence` override (:334-339) catches `Exception` and `pass`es with **no log** — if the LighterGlue internals change shape, the depth-confidence knob silently no-ops. `_maybe_warmup` LighterGlue warmup (:1492-1495) catches and logs but continues. Low severity, but the silent `pass` at :338 hides config failures.

- **Per-tick unconditional `torch.cuda.synchronize()` (:679-681).** Labeled a "diagnostic" sub-timing split (`gpu_queue_wait`) but runs on every live tick with no env-gate — it forces a full device sync that the comment itself says "absorbs ALL GPU work enqueued earlier." On the live path (shared GPU with FF/viser) this is a real serialization point shipped as instrumentation.

- **Params threaded through many layers.** The ctor takes 30 params; nearly all are individually re-read from `self.model.config.xfeat_*` at the call site (`dynamic_gs_pipeline_base.py:1993-2017`). Several (`min_cossim`, `use_semi_dense`) are vestigial. Env-var overrides (`DGS_KF_*`, `DGS_HOLD_*`, `DGS_XFEAT_SCALE_SELECT`, `DGS_KF_SYNTHETIC_FPS`, `DGS_SPIKE_GATE_FRAC`) are read inline inside `__init__` and `estimate_and_advance`, so the effective config is split across config fields + ctor args + scattered `os.environ.get` reads — hard to see the true live behavior in one place.

- **Lazy-init attributes (`_inlier_hist`, `_kf_tick`) via `getattr` instead of `__init__`** — see lifecycle desync note; also obscures the object's state surface.
