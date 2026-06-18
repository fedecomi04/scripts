# `dynamic_track.py` — the XFeat rigid-object tracker (layer: dynamic)

## (1) RESPONSIBILITY

Per-tick rigid object pose estimation: given a `Frame` and an immutable `GaussianSnapshot` (for the object footprint mask), run XFeat extract → LighterGlue match → RANSAC/Kabsch against a multi-anchor keyframe pool, optionally smooth via static-hold (KF off by default), and return a `MotionEstimate` (R, t, diagnostics) — all on the MAIN thread, holding no long lock and never mutating the scene.

---

## (2) PUBLIC INTERFACE

```python
@dataclass
class MotionEstimate:
    """Result of one tracker tick. The pipeline reads these fields by name to
    decide whether to apply the rigid transform and to log/feed timing."""
    success: bool                 # inlier_count >= min_track_points AND not spike-gated
    ready: bool                   # D0 anchor seeded; False => caller must call seed()
    rotation: np.ndarray          # (3,3) float32 — D0->current OBJECT rotation, world frame (post-hold/KF)
    translation: np.ndarray       # (3,)  float32 — D0->current OBJECT translation, world frame
    inlier_count: int             # RANSAC inliers on the winning anchor
    correspondence_count: int     # depth-compatible matches fed to RANSAC
    mean_residual: float          # winning-anchor inlier residual stats (metres)
    median_residual: float
    timings: dict[str, float]     # per-stage seconds: input_prep, xfeat_extract, lighterglue_match, ransac_kabsch

@dataclass
class TrackerInputs:
    """The per-tick inputs the pipeline assembles from a Frame + snapshot.
    rgb/depth/camera are torch tensors / nerfstudio Cameras already built by the
    pipeline's frame->batch step (the tracker does NOT build cameras or convert
    units). object_mask is the rendered footprint of the tracked object."""
    rgb: Tensor                   # (H,W,3) float [0,1] on GPU (model.get_live_rgb output)
    depth: Tensor                 # (H,W) or (H,W,1) float32 metres, filtered, 0==invalid
    camera: "Cameras"             # nerfstudio Cameras (1 cam) — OpenGL c2w + intrinsics
    keep_mask: Optional[Tensor]   # (H,W) gripper/robot-exclusion keep mask (1==keep)
    object_mask: Optional[Tensor] # (H,W) rendered tracked-object footprint (post-match halo filter)
    stamp_sec: float              # Frame capture event-time (KF dt source if KF re-enabled)

class XFeatTracker:
    """XFeat sparse + LighterGlue + RANSAC-Kabsch multi-anchor rigid tracker.
    One instance per dynamic run; rebuilt on D0 re-pick. Touched ONLY on the main thread."""

    def __init__(self, device, cfg: "TrackerConfig", pose_filter_cfg: "PoseFilterConfig") -> None:
        """Build XFeat + LighterGlue (eager), the optional KF, the static-hold deque,
        and the empty anchor pool. All knobs come from cfg/pose_filter_cfg — NO os.environ
        reads here (config module already folded the DGS_* env vars in)."""

    @property
    def ready(self) -> bool:
        """True once the D0 anchor holds >= min_track_points depth-valid keypoints."""

    def seed(self, inp: TrackerInputs) -> int:
        """Seed the D0 anchor (T=identity) from this frame's object-masked keypoints;
        reset pose, KF, static-hold, pool, D0 centroid. Returns kept-keypoint count (0 == failed)."""

    def track(self, inp: TrackerInputs) -> MotionEstimate:
        """One tick: extract -> select+match nearest anchor(s) -> RANSAC/Kabsch ->
        compose cumulative pose -> (KF) -> static-hold -> maybe-create-anchor. Pure
        read of inp + own pool; never touches the GaussianSet. Returns a MotionEstimate."""
```

Notes:
- `MotionEstimate.rotation/translation` are the **object's** D0→current world-frame pose; the pipeline turns them into the means/quats subset write via `GaussianSet.write_object_pose` (the tracker never writes Gaussians).
- The interface is deliberately two methods (`seed`, `track`) + one `ready` property — the old `initialize`/`estimate_and_advance`/`current_track_count` surface collapses to this.
- The KF / static-hold knobs are passed as config objects, not 12 ctor scalars; env-var resolution happens once in `config.py` (see config.md `TrackerConfig`/`PoseFilterConfig`).

---

## (3) DEPENDS ON (other NEW modules only)

- **`frame.py`** — only for `Frame.stamp_sec` semantics / event-time contract (the tracker consumes a `stamp_sec` float that originates from `Frame`). No direct `Frame` object is required inside `track()`; the pipeline pre-builds `TrackerInputs`.
- **`gaussian_set.py`** — for the `GaussianSnapshot` type the pipeline reads `object_mask` from. The tracker itself takes the already-rendered `object_mask` tensor in `TrackerInputs`; it does NOT call `snapshot()` or render (the pipeline renders the object mask under `_model_lock` and hands it in). So the dependency is on the *snapshot-derived mask*, not on `GaussianSet` surgery.
- **`config.py`** — `TrackerConfig`, `PoseFilterConfig` (consumed by the ctor).
- **vendored XFeat / LighterGlue** (`third_party/xfeat`) — not a "new module" but a load-bearing third-party dependency; the repo-on-path shim stays internal to this module.

It does NOT depend on the pipeline god-file, the FF dispatcher, the viser bridge, or `scene_model`. Camera construction and frame→batch live in the pipeline/source-adapter, upstream of `track()`.

---

## (4) CONSUMES / PRODUCES

**CONSUMES (per tick):**
- `TrackerInputs`: GPU rgb [0,1], float32-metres depth (already depth-filtered upstream — the tracker does NOT re-filter), nerfstudio `Cameras` (OpenGL c2w + K), gripper keep-mask, rendered object-footprint mask, capture `stamp_sec`.
- Internal owned state: the anchor pool (descriptors/keypoints/world-3D/pose/camera-rotation/distance per anchor), cumulative `(R,t)`, KF state (if enabled), static-hold deque, D0 centroid.

**PRODUCES:**
- `MotionEstimate` (R, t, success/ready, inlier/correspondence counts, residuals, per-stage timings). Pure value object — no tensors retained, no model handles. The pipeline applies it via `GaussianSet.write_object_pose`.

**Data-format guarantees relied on (from `frame.py`):** depth in **metres, 0==invalid** (the bilinear depth sample treats 0 as invalid); c2w is **OpenGL** (backprojection uses the OpenGL camera convention `+x right, +y up, -z forward`). `stamp_sec` is capture event-time, never `now()`.

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `xfeat_motion.py:XFeatMotionEstimator` | `dynamic_track.py:XFeatTracker` (renamed, slimmed) |
| `xfeat_motion.py:initialize` | `XFeatTracker.seed` (same D0 logic: full-image extract → object-mask post-filter → depth-valid → build D0 anchor → reset pose/KF/hold/centroid) |
| `xfeat_motion.py:estimate_and_advance` | `XFeatTracker.track` — but the ~455-line god method is split into private steps: `_extract`, `_select_and_match` (anchor loop), `_compose_pose`, `_smooth_output` (KF + static-hold), `_maybe_create_anchor` |
| `xfeat_motion.py:_Anchor` | unchanged internal dataclass (kept; see DROPPED for the `rgb`/`mask` debug fields) |
| `xfeat_motion.py:_rotation_distance_deg / _relative_object_rotation / _scale_ratio / _select_nearest_anchor_by_rotation / _needs_new_anchor` | unchanged module-level helpers (anchor selection + gating) |
| `xfeat_motion.py:_extract / _lighterglue_match / _mnn_match / _build_anchor / _match_and_solve_against_anchor / _current_camera_object_distance / _restrict_depth_valid_to_image_mask / _prepare_rgb_gpu / _maybe_warmup / _mask_to_numpy / _ensure_repo_on_path` | kept as private methods/helpers (the live-reachable internals) |
| `xfeat_motion.py` ctor scalar knobs + scattered `os.environ.get(DGS_KF_* / DGS_HOLD_* / DGS_XFEAT_SCALE_SELECT / DGS_KF_SYNTHETIC_FPS)` | replaced by `TrackerConfig` + `PoseFilterConfig` (env resolution centralized in `config.py`) |
| `tracker_common.py:MotionEstimate` | `dynamic_track.py:MotionEstimate` (trimmed: drop the 6 debug-viz fields — see DROPPED) |
| `tracker_common.py:prepare_depth_image / extract_intrinsics / extract_camera_to_world / resize_mask / sample_depth_bilinear / backproject_to_world` | kept (the live per-tick numpy/camera helpers) — either inlined into `dynamic_track.py` or a thin `_tracker_geom.py` sibling (these are the only survivors of `tracker_common`) |
| `tracker_common.py:estimate_rigid_transform + estimate_rigid_transform_ransac` | kept (the Kabsch core); `estimate_rigid_transform` becomes underscore-private (internal-only) |
| `tracker_common.py:_so3_exp / _so3_log / PoseKalmanFilter` | kept ONLY if the KF survives the purge (see OQ) — KF is OFF by default |

---

## (6) DROPPED (current code NOT carried, with reason + audit ref)

| Dropped | Reason | Audit ref |
|---|---|---|
| `tracker_common.py:prepare_tracking_rgb`, `prepare_tracking_rgb_gpu` | 0 callers (tracker uses its own `_prepare_rgb_gpu`); docstrings falsely claim "preferred path". | `tracker_common.md` §2 (high) |
| `tracker_common.py:sample_mask_points`, `_shrink_mask_for_sampling`, `_subsample_points`, `filter_points_in_image`, `filter_points_by_mask_array` | Dead island — KLT-tracker residue; XFeat filters inline. Whole cluster has 0 external refs. | `tracker_common.md` §2 (high) |
| `xfeat_motion.py:_compose_keep_region` | 0 callers AND 0 internal callers; the keep-region recipe is inlined per-tick. Orphaned earlier extraction. | `xfeat_motion.md` §2 (high) |
| `xfeat_motion.py:_pre_mask_image` | 0 callers; only a docstring mention. The current design extracts on the full image then post-filters (pre-mask corrupts boundary descriptors). | `xfeat_motion.md` §2 (high); MEMORY `feedback_xfeat_d0_seed_no_premask` |
| `xfeat_motion.py:current_track_count` property + `last_anchor_idx_used`/`last_used_fallback_anchor`/`last_pool_size` write-only diagnostics | No pipeline reader; documented purpose unfulfilled. Keep only `inlier_count` via the `MotionEstimate`. | `xfeat_motion.md` §2/§4 (write-only diagnostics) |
| `xfeat_motion.py:_use_semi_dense` branch + `xfeat_use_semi_dense` config | Config-gated but unreachable at default `False`; speculative. Drop the branch and the knob. | `xfeat_motion.md` §2 (`_use_semi_dense` note) |
| `min_cossim` / `xfeat_min_cossim` ctor param + attr | Threaded through but never read after assignment — fully dead config. | `xfeat_motion.md` §4 (dead config field) |
| Legacy timing keys `klt_forward`, `postprocess`, `resample` (always 0 or duplicate of `xfeat_extract`) + the `CoTrackerMotionEstimate` alias + stale "fourth backend alongside CoTracker/TAPIR/KLT" docstring | CoTracker-era leakage; the three trackers were purged 2026-05-26. Keep only real stage timings. | `xfeat_motion.md` §4 (legacy-name leakage); CLAUDE.md tracker-purge note |
| Per-tick unconditional `torch.cuda.synchronize()` + `gpu_queue_wait` timing | Ships a full device sync as "instrumentation" on the shared-GPU hot path — a real serialization point against FF/viser. Move to an opt-in diag flag, OFF on the hot path. | `xfeat_motion.md` §4 (per-tick sync); ARCHITECTURE_PRINCIPLES #4 (no work on hot path) |
| `MotionEstimate` debug-viz fields: `previous_points_xy`, `current_points_xy`, `tracked_inlier_mask`, `previous_rgb`, `current_rgb`, `previous_mask`, `current_mask` | Debug-overlay only; couple the hot-path result contract to viz plumbing, and `previous_rgb`/`previous_mask` pin a full-res GPU tensor per result. No live consumer in the purged path. | `tracker_common.md` §4 (#5); `xfeat_motion.md` anchor-pool lifecycle |
| `_Anchor.rgb` (full-frame GPU tensor cloned per anchor) + `_Anchor.mask` | Stored ONLY for the side-by-side debug visualizer's red border; ~27.6 MB/anchor at 1200p, unbounded pool growth, on the GPU hosting the splat scene. Drop with the debug-viz. | `xfeat_motion.md` §3 (UNBOUNDED GPU growth — the headline lifecycle issue) |
| `DGS_SPIKE_GATE_FRAC` inlier-spike gate + lazy `_inlier_hist`/`_kf_tick` `getattr` attrs | Off by default; reset-incompleteness on re-seed. If kept, init in `__init__` and reset in `seed`; otherwise drop (config-gated, unreachable at default). | `xfeat_motion.md` §3 (lazy-init desync), §4 |
| `tracker_common.py:_so3_exp/_so3_log/PoseKalmanFilter` (conditional) | OFF by default since 2026-06-13 (lagged on jerky 1200p). Carry ONLY if smoother-motion re-enable is still planned — else this 185-line ESKF + its `DGS_KF_*` knobs are dormant weight. | `tracker_common.md` §4 (#3); CLAUDE.md KF-off note; MEMORY `project_tracker_jitter_findings` |

Explicitly NOT dropped (load-bearing): the multi-anchor pool + relative (object-in-camera) rotation gate + scale gate; the 2nd-nearest-anchor fallback; the per-tick `keep_region = gripper_keep ∩ dilate(object_mask, R)` post-match filter (stops background-pin once the object is grasped — `object_mask_filter` invariant); the static-hold median; full-image-extract-then-post-filter for D0 + new anchors; the fixed RANSAC seed (deterministic per match set).

---

## (7) INVARIANTS PRESERVED (CLAUDE.md) + how

- **#4 (dynamic phase = pure runtime, ALL gauss LRs = 0):** the tracker performs NO gradient descent and NEVER touches `gauss_params`/optimizers. It returns a `MotionEstimate`; the pipeline applies it via `GaussianSet.write_object_pose` (in-place index-assign under `_model_lock`, `@torch.no_grad`). The tracker must not reintroduce any backward/optimizer step. (`RUNTIME_tracker_tick.md` summary; ARCHITECTURE_PRINCIPLES.)
- **#8 (identity buffers):** the tracker reads only the *rendered object-footprint mask* (derived from `object_instance_ids == d0_id`), never writes any of the 4 identity buffers. `object_flags` (D0) and the FF tail writes stay with the pipeline/`GaussianSet`. The tracker moves only the tracked instance (the mask is `object_instance_ids == d0_id`), so FF inserts (id=999) are never tracked — invariant kept by construction since the tracker takes the mask as input. (`RUNTIME_tracker_tick.md` invariant summary.)
- **#9 (viser-direct, model lock):** the tracker holds NO lock and does NOT render — it consumes a pre-rendered `object_mask` that the pipeline produced under `_model_lock`. It runs on the main thread and never reads `gauss_params` directly, so it introduces no cross-thread race with the FF insert or viser render. (`RUNTIME_tracker_tick.md` H2/H5 — the tracker estimator is single-threaded, races live on the model not here.)
- **#5 event-time (ARCHITECTURE_PRINCIPLES):** any dt-dependent math (KF, if re-enabled) reads `TrackerInputs.stamp_sec` (capture time) or the rate-invariant `fixed_fps` feed — never `time.time()` deltas. The wall-clock-dt detuning bug must not return. (CLAUDE.md KF rate-sensitivity note; ARCHITECTURE_PRINCIPLES #5.)
- **#4 hot-path discipline (ARCHITECTURE_PRINCIPLES):** drop the per-tick `cuda.synchronize()` + per-tick file I/O; keep extract→match→RANSAC allocation-bounded (no anchor-pool RGB clone). (ARCHITECTURE_PRINCIPLES #4.)

---

## (8) THREADING

- **Runs on the MAIN (trainer/tracker) thread ONLY.** `seed()` and `track()` are called from the pipeline's `_tracker_tick`. The audit confirms `_motion_estimator` is never accessed from the FF-bg or viser-render threads, so the anchor pool, cumulative pose, KF, and static-hold deque are single-threaded — **no internal lock needed**. (`xfeat_motion.md` §3 threading; `RUNTIME_tracker_tick.md` threading table.)
- **Holds NO long lock.** The tracker does not acquire `_model_lock`. It must NOT block on: the model lock, the FF slot lock, SHM, subprocess IPC, disk, or the publisher pose/joint lock. The object-footprint mask is rendered by the pipeline *under* `_model_lock` and handed in as a tensor — the tracker only reads that tensor. This keeps the FF insert (which takes `_model_lock`) and the tracker decoupled.
- **GPU contention, not data race:** XFeat extract ends in a `.cpu()` pull (an implicit sync against whatever the FF/viser threads enqueued on the default CUDA stream). This is a *latency* cost on the shared GPU, not a correctness race. The explicit extra `cuda.synchronize()` is dropped (item in §6); the implicit `.cpu()` sync stays (it is the keypoint read-back).
- **Lifecycle:** rebuilt on D0 re-pick; the old instance (XFeat + LighterGlue models + anchor pool) is dropped when the pipeline reassigns the reference. With the per-anchor RGB clone dropped (§6), pool growth is bounded to descriptors+keypoints+world-3D per anchor — no full-res GPU image accumulation. No `.pt` cache, no SHM, no subprocess owned here.

---

## (9) OPEN QUESTIONS

1. **Keep the Kalman filter at all?** It is OFF by default and was measured worse than raw+static-hold on jerky 1200p (CLAUDE.md, MEMORY). Carrying it keeps `PoseKalmanFilter` + `_so3_exp/_so3_log` + the `DGS_KF_*`/`PoseFilterConfig` surface. Drop entirely (and remove `PoseFilterConfig` from the ctor), or keep dormant for a future smoother-motion scene? (Recommend: keep behind the config flag but move the KF into a small `_pose_filter.py` sibling so `dynamic_track.py` stays lean — confirm.)
2. **Object-mask: rendered-by-pipeline-and-handed-in vs tracker-renders-it.** The spec assumes the pipeline renders the object footprint under `_model_lock` and passes it in `TrackerInputs.object_mask` (keeps the tracker lock-free + render-free). Confirm the pipeline owns that render (it must, to hold the lock) — the alternative (tracker calls back to render) would force the tracker to take `_model_lock`, which we explicitly do not want.
3. **`tracker_common` survivors: inline or sibling module?** Only 8 functions survive (depth/camera/backproject/Kabsch). Inline them into `dynamic_track.py`, or keep a thin `_tracker_geom.py`? Inlining removes a module but grows the god-file; a sibling keeps the seam. (Lean toward inline since they have exactly one consumer now.)
4. **Spike gate + scale-select selection:** both are env-gated OFF by default (`DGS_SPIKE_GATE_FRAC`, `DGS_XFEAT_SCALE_SELECT`). Drop both for the v1 purge (simpler, matches default behavior), or keep as `TrackerConfig` booleans? The spec drops the spike gate and keeps `scale_select` as a config bool (already in `TrackerConfig`). Confirm dropping the spike gate is acceptable.
5. **Debug-viz removal blast radius.** Dropping the `_Anchor.rgb`/`mask` clones and the 6 `MotionEstimate` debug fields removes the side-by-side tracker visualizer. Is that visualizer still used by any live diagnostic, or fully superseded by viser-direct + the `_ff_debug` dump? If still wanted, it needs a separate opt-in path that clones RGB only when a debug flag is set (NOT unconditionally per anchor).
6. **`stamp_sec` plumbing when KF is off.** If the KF is dropped (OQ1), `TrackerInputs.stamp_sec` has no consumer in the tracker. Keep the field for forward-compat (and event-time discipline) or drop it until the KF returns? (Recommend keep — it is one float and documents the event-time contract.)
