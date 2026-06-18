# Audit: `dynamic_gs/utils/tracker_common.py`

Shared tracker primitives extracted from the legacy `cotracker_motion` module. The only live consumer is the XFeat tracker (`dynamic_gs/utils/xfeat_motion.py`), imported as `from . import tracker_common as _tc` (xfeat_motion.py:63) plus the `MotionEstimate` alias re-export in `dynamic_gs/utils/__init__.py:21`.

Grep basis (run against `dynamic_gs/` and `scripts/`, `--include=*.py`, excluding the definition file):
```
grep -rn "<symbol>" dynamic_gs scripts --include=*.py | grep -v tracker_common.py
```

---

## 1) FUNCTION / CLASS MAP

### `MotionEstimate` (dataclass) — tracker_common.py:43
Result dataclass returned by every tracker backend's `estimate_and_advance`; field names are read verbatim by the pipeline (`inlier_count`, `rotation`, `timings`, etc.).
- Callers: **referenced only via the alias** `CoTrackerMotionEstimate`. Imported as that alias in `dynamic_gs/utils/__init__.py:21` and `xfeat_motion.py:64`; instantiated in xfeat_motion.py:619/703/1017; annotated at xfeat_motion.py:585. The bare name `MotionEstimate` has **NO refs** outside its def except the two `import ... as` lines. LIVE (the active tracker constructs it every tick).

### `CoTrackerMotionEstimate` (alias = `MotionEstimate`) — tracker_common.py:81
Back-compat alias for the legacy name.
- Callers: 9 refs — `dynamic_gs/utils/__init__.py:21,54`; `xfeat_motion.py:64,585,619,703,1017` (+ docstrings at 7, 1314). LIVE.

### `prepare_tracking_rgb(image: Tensor) -> Tensor` — tracker_common.py:84
CPU-resident `[0,255]` float HWC RGB; docstring says it's for init paths without a GPU device handy.
- Callers: **NO REFS FOUND.** The only hit (xfeat_motion.py:1438) is a *docstring* ("The critical difference from `_tc.prepare_tracking_rgb`"), not a call. Not in `__init__.py` exports. Dead candidate.

### `prepare_tracking_rgb_gpu(image, device) -> Tensor` — tracker_common.py:100
GPU-resident HWC float; docstring claims it's the per-tick preferred path to avoid a `.cpu()` sync.
- Callers: **NO REFS FOUND** (zero, including docstrings). The active tracker uses its own `_prepare_tracking_rgb_gpu` method (xfeat_motion.py:1432), NOT this one. Dead candidate.

### `prepare_depth_image(depth: Tensor) -> np.ndarray` — tracker_common.py:118
Squeeze trailing channel, `.cpu().numpy()` float32.
- Callers: 2 — xfeat_motion.py:489, 600. LIVE (per-tick).

### `extract_intrinsics(camera) -> np.ndarray` — tracker_common.py:124
First-camera 3×3 K to numpy float32.
- Callers: 2 — xfeat_motion.py:490, 601. LIVE (per-tick).

### `extract_camera_to_world(camera) -> np.ndarray` — tracker_common.py:128
First-camera c2w to numpy float32, sliced to 3×4.
- Callers: 2 — xfeat_motion.py:491, 602. LIVE (per-tick).

### `resize_mask(mask, output_shape) -> Tensor` — tracker_common.py:135
Nearest-neighbour mask resize via `F.interpolate`.
- Callers: 1 real — xfeat_motion.py:1504 (`_tc.resize_mask`). (The 6 raw grep hits include `change_mask.py`'s unrelated local `_resize_mask_to`.) Also called internally by `sample_mask_points` and `filter_points_by_mask_array` (both dead — see below). LIVE via xfeat_motion.

### `_subsample_points(points_xy, max_points) -> np.ndarray` — tracker_common.py:145
Even-stride subsample of 2D points to a cap.
- Callers: used only inside `sample_mask_points` (tracker_common.py:209,212), which is itself dead. **NO external REFS.** Dead candidate (transitively).

### `_shrink_mask_for_sampling(mask_np) -> np.ndarray` — tracker_common.py:153
Erode a mask inward by ~2.5% of its bbox side before sampling.
- Callers: used only inside `sample_mask_points` (tracker_common.py:178), itself dead. **NO external REFS.** Dead candidate (transitively).

### `sample_mask_points(mask, max_points, rgb=None, output_shape=None) -> np.ndarray` — tracker_common.py:165
Sample 2D pixel coords inside a mask, FAST-keypoint-aware when `rgb` given.
- Callers: **NO REFS FOUND.** Per CLAUDE.md ("Per-tick object mask removal — XFeat", 2026-05-26) the FAST/mask-sampling path was structurally needed by the deleted KLT tracker; XFeat never calls it. Dead candidate.

### `filter_points_in_image(points_xy, visibility, width, height)` — tracker_common.py:215
AND visibility with in-bounds + finite check.
- Callers: **NO REFS FOUND.** Dead candidate.

### `filter_points_by_mask_array(points_xy, visibility, mask, output_shape)` — tracker_common.py:229
AND visibility with a resized mask lookup at rounded pixel coords.
- Callers: **NO REFS FOUND.** Dead candidate.

### `sample_depth_bilinear(depth, points_xy)` — tracker_common.py:244
Bilinear depth sample with per-corner validity (finite + >0).
- Callers: 3 — xfeat_motion.py:535, 718, 980. LIVE (per-tick).

### `backproject_to_world(points_xy, depth_values, intrinsics, camera_to_world)` — tracker_common.py:289
Pixel+depth → world in OpenGL camera convention.
- Callers: 2 — xfeat_motion.py:721, 1192. LIVE (per-tick).

### `_so3_exp(w) -> R` — tracker_common.py:322
Rodrigues exp (rotvec→matrix).
- Callers: used only inside `PoseKalmanFilter` (tracker_common.py:441,513). **NO external REFS.** Live only if the KF is enabled (default OFF — see §3).

### `_so3_log(R) -> w` — tracker_common.py:328
Rodrigues log (matrix→rotvec).
- Callers: used only inside `PoseKalmanFilter` (tracker_common.py:466). **NO external REFS.** KF-internal.

### `class PoseKalmanFilter` — tracker_common.py:334
12-state constant-velocity error-state SE(3) Kalman filter smoothing per-tick RANSAC pose; methods `reset`, `initialized`, `current`, `filter`.
- Callers: 3 — annotated/constructed at xfeat_motion.py:392,394; documented at dynamic_gs_model.py:315. **Constructed only when `pose_filter_enabled` is True (xfeat_motion.py:393), and `xfeat_pose_filter_enabled: bool = False` by default (dynamic_gs_model.py:313).** So on the default live path the filter is never instantiated and `_pose_filter is None` (xfeat_motion.py:484,873). Live-reachable ONLY via opt-in env/config; see §4.

### `estimate_rigid_transform(source_points, target_points)` — tracker_common.py:521
Closed-form Kabsch SVD rigid alignment.
- Callers: called only inside `estimate_rigid_transform_ransac` (tracker_common.py:581,617). **NO external REFS** (the `estimate_rigid_transform` hit in `rigid_regularization.py` is a *different* function, `estimate_rigid_transform_kabsch`). Public-looking but internal-only — see §4.

### `estimate_rigid_transform_ransac(source, target, threshold, iterations, min_inliers)` — tracker_common.py:550
3-point RANSAC over Kabsch with inlier refit and residual-stats diagnostics.
- Callers: 1 — xfeat_motion.py:1400. LIVE (per-tick, the core pose solver).

---

## 2) DEAD-CODE CANDIDATES

Confirmed by grep (excluding the definition file). None are entry points, callbacks, monkeypatch targets, `__main__`, or invariant-protected buffers.

| Symbol | file:line | External refs | Confidence |
|---|---|---|---|
| `prepare_tracking_rgb_gpu` | tracker_common.py:100 | 0 (zero, even docstrings) | high |
| `prepare_tracking_rgb` | tracker_common.py:84 | 0 calls (1 docstring-only mention at xfeat_motion.py:1438) | high |
| `sample_mask_points` | tracker_common.py:165 | 0 | high |
| `_shrink_mask_for_sampling` | tracker_common.py:153 | 0 external (only `sample_mask_points`, dead) | high |
| `_subsample_points` | tracker_common.py:145 | 0 external (only `sample_mask_points`, dead) | high |
| `filter_points_in_image` | tracker_common.py:215 | 0 | high |
| `filter_points_by_mask_array` | tracker_common.py:229 | 0 | high |

Notes:
- The whole "mask-point sampling" cluster (`sample_mask_points` + `_shrink_mask_for_sampling` + `_subsample_points` + the FAST detector path) is a self-contained dead island: the only caller of the privates is the dead public sampler, which itself has no callers. This is the residue of the KLT-tracker removal documented in CLAUDE.md (2026-05-26). Removable as one unit.
- `filter_points_in_image` / `filter_points_by_mask_array` are the legacy point-filtering helpers; XFeat does its filtering inline. Dead.
- `prepare_tracking_rgb` / `prepare_tracking_rgb_gpu`: both docstrings claim they are the "init path" / "per-tick preferred path", but xfeat_motion implements its OWN `_prepare_tracking_rgb_gpu` and never calls either of these. The docstrings are actively misleading (a maintainer would assume they're load-bearing). Removable.

**NOT dead (internal-only, keep):**
- `_so3_exp`, `_so3_log` — used by `PoseKalmanFilter`.
- `estimate_rigid_transform` — used by `estimate_rigid_transform_ransac`.
- `_subsample_points`/`_shrink_mask_for_sampling` would survive only if `sample_mask_points` survives; flagged above because the cluster dies together.

---

## 3) DATA-LIFECYCLE

This module holds **no** persistent state of its own — no `.pt` warm-cache I/O, no SHM, no file/process handles, none of the 4 identity buffers (`object_flags`/`object_instance_ids`/`sam3d_init_target_flags`/`inserted_flags` are owned by the model + persistence layer, not here — invariant-protected, untouched). The only stateful object is `PoseKalmanFilter`; everything else is pure numpy/torch transforms.

**`PoseKalmanFilter` internal state lifecycle**
- Allocated: `__init__` → `reset()` (tracker_common.py:387,389) sets `_t_nom`, `_R_nom`, `_x (12,)`, `_P (12,12)`, `_last_time`. All host (numpy float64), small, no GPU. No leak risk.
- Reset: xfeat_motion.py:485 on tracker re-seed; internally on the snap-gate branch (tracker_common.py:476-483) re-inits `_t_nom/_R_nom/_x/_P`. Consistent shapes on every reset path.
- `_last_time` uses **wall-clock dt** (`timestamp` is `time.time()` per CLAUDE.md "rate-sensitive" note). dt is clamped to a 20 Hz fallback only outside `(0, 0.5)` (tracker_common.py:436-437); a 0.4 s hiccup is NOT clamped → an over-large `dt⁴` Q inflation → transient over-trust of the model. Cosmetic given the KF is OFF by default, but a real desync-with-real-time hazard if re-enabled without the rate-invariant `fixed_fps` path (which lives in xfeat_motion, not here).

**GPU / heap per-tick allocations (live-path, called every tracker tick)**
- `prepare_depth_image` (489,600): forces a `.cpu().numpy()` host sync + allocates a new HWC float32 array every tick. Per-tick D→H copy of the full depth map (1920×1200 → ~9 MB). This is on the live tracker hot path; the sync stalls the tick on the GPU stream. Documented elsewhere as acceptable, flagged here as a per-tick allocation + sync.
- `extract_intrinsics`/`extract_camera_to_world` (490-491,601-602): `.cpu().numpy()` per tick — tiny (3×3 / 3×4), negligible.
- `sample_depth_bilinear` / `backproject_to_world` / `estimate_rigid_transform_ransac`: pure-numpy, allocate temporaries proportional to match count (small). `estimate_rigid_transform_ransac` allocates a fresh `np.random.default_rng` **per call** (tracker_common.py:571) seeded from an env read every call — a re-`os.environ.get` + RNG construct per tick. Cheap but avoidable churn.
- No tensors are cached/retained across ticks here; nothing loaded-but-never-freed; no handles opened. No leak.

**Format/shape contracts**
- `extract_camera_to_world` returns 3×4 (slices 4×4 → 3×4 at :131). `backproject_to_world` consumes `camera_to_world[:, :3]`/`[:, 3]` — consistent with 3×4. OK.
- `MotionEstimate` shape contract: `rotation`/`translation` are np arrays; the KF (`filter`) reshapes to `(3,3)`/`(3,)` defensively (tracker_common.py:417-418). The optional debug tensor fields (`previous_rgb` etc.) default to `None` — no leak, but if a caller stashes live GPU tensors there and the dataclass is retained, those tensors stay resident; not a problem in this module (it never retains a `MotionEstimate`).

No double-loads, no missing frees, no save/load mismatches in this module.

---

## 4) DESIGN SMELLS

1. **Misleading "preferred path" docstrings on dead functions** — tracker_common.py:84-87 (`prepare_tracking_rgb`: "init paths…") and :100-106 (`prepare_tracking_rgb_gpu`: "Per-tick paths should prefer…"). Both are dead (§2) and the live tracker has its own implementation. The docstrings assert a contract that no longer exists → a maintainer purging tomorrow could keep them believing they're load-bearing. Recommend: delete with the functions.

2. **`estimate_rigid_transform` is public but internal-only** — tracker_common.py:521. Only caller is `_ransac` in the same file. Reads like a reusable public Kabsch (the module docstring even advertises it at line 19), but nothing outside uses it. Either underscore-prefix it or accept it as a deliberate public utility; right now it's a leaky abstraction (looks shared, isn't).

3. **`PoseKalmanFilter` is a ~185-line opt-in component that is OFF by default** — tracker_common.py:334-518, gated by `xfeat_pose_filter_enabled=False` (dynamic_gs_model.py:313). Per CLAUDE.md it was disabled 2026-06-13 because it lagged on real-1200p. So the most complex class in the file (ESKF math, snap gate, robust-R inflation, env knobs `DGS_KF_SNAP_*`) is unreachable on the default live path. Not dead (config-reachable), but it is dormant weight on a "being-purged-tomorrow" module — flag for the purge decision: keep only if the smoother-motion re-enable is still planned.

4. **`filter()` mixes wall-clock dt with a `meas_scale` AND a self-derived robust-R** — tracker_common.py:491-503. Three independent noise-inflation mechanisms (`meas_scale²`, the `_rob` quadratic past-3-sigma term, and the snap gate) stack multiplicatively into `R_noise`. The interaction is subtle and only "tuned on a fixture" (comment at :496). Hard to reason about; if the KF is ever re-enabled this is where mis-tuning will hide.

5. **`MotionEstimate` carries 6 optional debug fields (`previous_rgb`/`current_rgb`/masks/point arrays)** — tracker_common.py:67-74. These are debug-viz only; mixing them into the per-tick result dataclass that the live pipeline reads by name couples the hot-path contract to debug plumbing. The `previous_mask: Optional[object]` / `current_mask: Optional[object]` typing as bare `object` is a typing smell (no contract on what they hold). Minor.

6. **Per-call `os.environ.get` reads in the hot path** — `DGS_RANSAC_SEED` re-read every RANSAC call (tracker_common.py:571); `DGS_KF_SNAP_*` read in the KF ctor (acceptable, ctor-only). The RANSAC env read + RNG re-construction every tick is needless churn; the seed could be resolved once.

7. **Two parallel "rigid transform" implementations in the repo** — `estimate_rigid_transform` here (numpy Kabsch) vs `estimate_rigid_transform_kabsch` in `rigid_regularization.py:7` (torch Kabsch). Different names, different backends, same math. Not a bug, but duplicated logic worth noting for the purge.

8. **No swallowed exceptions** — the two `try/except np.linalg.LinAlgError` blocks (tracker_common.py:534-537) correctly return `None` and propagate failure to the caller; not swallowed. (Noted as a positive.)

### Thread-safety (live: FF bg thread / tracker tick / viser render share a model lock)
This module is **stateless except `PoseKalmanFilter`**, and the filter instance is owned privately by the tracker (`self._pose_filter`), called only on the tracker tick thread. None of these functions touch the shared model `gauss_params` or the identity buffers, so they introduce **no cross-thread races** of their own — the FF/render/tracker contention lives in the pipeline, not here. Caveat: the module-level `os.environ` reads are process-global but read-only, so no race. No module-level mutable state.

### Branches unreachable in live mode
- Entire `PoseKalmanFilter` (default `enabled=False`).
- All of the dead-island samplers (§2) are unreachable in any mode.
- `sample_mask_points`'s `rgb is not None` FAST-keypoint branch (tracker_common.py:183-209) — doubly unreachable (function is dead).
