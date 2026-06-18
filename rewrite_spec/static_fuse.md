# `static_fuse.py` — module spec (layer: static)

## (1) RESPONSIBILITY

One sentence: build the static-phase TSDF seed PLY from a posed RGB-D dataset (GPU/CPU online_fusion + ICP, contained in a subprocess for OOM isolation) and run Phase-0b — register each pre-generated SAM3D object cloud onto the back-projected real-depth target (NDP non-rigid by default; CPD/TEASER selectable), cull it (proximity de-dup + in-front occlusion), and hand the caller the `GaussTensors` + instance-id to insert via the one `GaussianSet` chokepoint.

This module owns **geometry and registration math only**. It NEVER mutates the `GaussianSet` itself (it returns insert-ready tensors + the target/match masks); the caller (`pipeline.py` static path) performs the locked `insert` / `write_instance_ids`. SAM3/FastSAM mask discovery and the SAM3D subprocess generation (Phase-0a) are NOT in this module — they are a segmentation/generation concern; this module *consumes* their on-disk artifacts.

---

## (2) PUBLIC INTERFACE (the contract other modules call)

```python
# ===================== TSDF SEED FUSION =====================

def build_seed_ply(static_dir: Path, intr: Intrinsics, cfg: FusionConfig,
                   depth_cfg: DepthConfig) -> Path:
    """Fuse every posed frame in <static_dir>/transforms.json into
    <static_dir>/depth_camera_init_points.ply (TSDF + per-frame ICP, GPU auto / CPU
    fallback), adaptive near/far downsample, atomic transforms.json `ply_file_path`
    patch + PLY mtime bump. Reads depth/rgb/mask off disk, zeros robot pixels.
    Returns the written PLY path. This is the in-process entry."""

def build_seed_ply_subprocess(static_dir: Path, cfg: FusionConfig) -> Path:
    """Run `build_seed_ply` in an isolated child process (python -m ...) so an
    Open3D GPU OOM (which poisons the CUDA cache + aborts the process at teardown)
    cannot take down the caller. Parent retries CPU on non-zero exit. The DEFAULT
    live/capture seed path (FusionConfig.defer_tsdf=True). Returns the PLY path."""


class SeedFuser:
    """Streaming TSDF+ICP fuser. add_frame() per arriving keyframe, finalize() once.
    Wraps the GPU/CPU impl auto-select; the ONLY object holding Open3D fusion state.
    Used by the concurrent-capture path (DGS_LIVE_DEFER_TSDF=0) and the offline driver."""

    def __init__(self, intr: Intrinsics, cfg: FusionConfig, depth_cfg: DepthConfig) -> None: ...
    def add_frame(self, depth_u16: np.ndarray, c2w_opengl: np.ndarray,
                  rgb_u8: np.ndarray | None = None) -> np.ndarray:
        """Integrate ONE frame (gripper pixels pre-zeroed by caller). Converts
        OpenGL c2w -> OpenCV internally. Returns the ICP-refined OpenCV c2w
        (or the FK pose unchanged when ICP fitness < threshold)."""
    def finalize(self) -> "o3d.geometry.PointCloud":
        """Extract the fused cloud, adaptive-downsample near/far, release GPU state.
        Single-shot; no add_frame after."""


class ConcurrentSeedRunner:
    """Capture-time watcher+worker: polls transforms.json, integrates each new
    keyframe on a worker thread (single-producer/single-consumer queue), finalizes
    on stop. Only used when defer_tsdf is False; otherwise build_seed_ply_subprocess
    runs at capture end."""

    def __init__(self, static_dir: Path, intr: Intrinsics, cfg: FusionConfig,
                 depth_cfg: DepthConfig) -> None: ...
    def start(self) -> None: ...                          # idempotent; spawns watcher + worker
    def stop_and_finalize(self) -> Path: ...              # drain, finalize, write PLY, return path
    def per_frame_add_stats(self) -> dict | None: ...     # {mean,p90,max,n,fail} | None (worker never ran)


# ===================== PHASE-0B REGISTRATION + CULL =====================

@dataclass(frozen=True)
class FusedObject:
    """One registered+culled object ready for GaussianSet.insert. Tensors are in
    scene frame / log-scale / logit-opacity (the GaussTensors contract)."""
    instance_id: int
    tensors: "GaussTensors"             # insert-ready (means/features/scales/quats/opacities)
    match_mask: torch.Tensor            # bool, len==scene num_points: existing Gaussians to tag with instance_id
    stats: dict                         # n_source/n_kept/scale/backend/chamfer/timing (manifest row)


def register_objects(*, scene: "GaussianSnapshot", ref: "ReferenceFrame",
                     generation_artifacts: list["Sam3dArtifact"],
                     seg_cfg: SegmentationConfig, depth_cfg: DepthConfig,
                     sh_degree: int, device, timing: dict | None = None
                     ) -> list[FusedObject]:
    """Phase-0b core. For each SAM3D artifact: back-project its mask through `ref`
    into the real-depth target cloud, register the SAM3D source onto it (NDP default;
    cpd/teaser selectable via seg_cfg), cull (proximity de-dup vs the trusted visible
    surface + in-front occlusion), build GaussTensors, and find the existing-Gaussian
    match_mask for instance-id propagation. Pure geometry: returns FusedObjects;
    does NOT touch GaussianSet. The caller inserts under the lock."""


@dataclass(frozen=True)
class ReferenceFrame:
    """The exact frame the masks/SAM3D were computed on, with a Cameras(1) rebuilt for
    back-projection. anchor_ref/ when present (live), else the last static keyframe."""
    image: torch.Tensor                 # (H,W,3) RGB
    depth_m: np.ndarray                 # (H,W) metres
    camera: "Cameras"                   # nerfstudio Cameras(1), OpenGL c2w
    H: int; W: int

def load_reference_frame(static_dir: Path, datamanager, model, device) -> ReferenceFrame:
    """Resolve the Phase-0b reference frame: prefer <static>/anchor_ref/ (the
    canonical live anchor); fall back to cached_train[-1] + camera-opt offset for
    recorded datasets. Single resolver for both phases (kills the dup in phase0)."""


# ===================== GEOMETRY PRIMITIVES (reused by register_objects + diag) =====================

def backproject_mask_to_world(mask_bool, depth_m, rgb, camera, *,
                              max_object_slope_deg=70.0, near_surface_window_frac=0.012
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an image-plane mask through depth into world points+colors,
    with MAD outlier scrub + geometry-derived local near-surface filter (silhouette
    table/see-through bleed). OpenGL convention via camera_conventions."""

def cull_points_in_front(points_world, target_points_world, camera, render_hw, *,
                         band_m=0.0, radius_px=2) -> np.ndarray:
    """Boolean keep-mask: drop inserted points lying between the camera and the
    trusted front surface (occlusion cull). Inverse projection of backproject_mask."""
```

Notes:
- `Intrinsics`, `GaussTensors`, `GaussianSnapshot` come from the NEW contract modules (`frame.py`, `gaussian_set.py`) — this module imports, never redefines, them.
- `Sam3dArtifact` is a small frozen tuple `{ply_path, pose_path, mask_path, instance_id}` produced by the segmentation/generation module (Phase-0a, NOT here); the registration step only reads those paths. (Open question 1 on whether this lives here or in the generation module.)
- The three-way backend dispatch (`ndp`/`cpd`/`teaser`) is INTERNAL to `register_objects`; the public surface does not expose the per-backend helpers.

---

## (3) DEPENDS ON (NEW modules only)

- **`config.py`** — `FusionConfig`, `DepthConfig`, `SegmentationConfig` (every knob; no `os.environ` read in this module).
- **`frame.py`** — `Intrinsics` (camera params for the fuser + Cameras build) and `DEPTH_SCALE_MM` (the single mm↔m constant).
- **`gaussian_set.py`** — `GaussTensors` (the insert contract this module *produces*), `build_default_gauss_tensors` (the kNN-spacing/SH/opacity seed builder shared with the FF decoders), `GaussianSnapshot` (read-view of the trained scene it registers against). This module returns `GaussTensors`; it does NOT call any mutating `GaussianSet` method.
- **`camera_conventions.py`** (the DUP_coord_conventions consolidation target) — `gl_c2w_to_cv`, `deproject_to_world(..., convention="opengl")`, `project_world_to_pixel` — so the OpenGL↔OpenCV flip and back-projection are defined once (this module is one of the worst current duplicators, Patterns A + B + C).
- **`rotations.py`** (DUP Pattern D target) — `quat_wxyz_to_rotmat` (the SAM3D rotation-init currently in `sam3d_fusion:_quaternion_wxyz_to_rotation_matrix`).
- No dependency on `pipeline.py`, the tracker, FF dispatcher, or viser. Phase-0a segmentation/SAM3D generation is a *sibling* module this module's caller sequences before it; the artifacts arrive as data.

Third-party (not "modules" but hard deps): `open3d` (TSDF/ICP), the vendored `ndp/` package (`deform_source_to_target`), `sklearn.NearestNeighbors` (cull/match), optional `probreg`/TEASER (only on the non-default backends).

---

## (4) CONSUMES / PRODUCES

**Consumes:**
- A `<static_dir>` with `transforms.json` (OpenGL c2w per frame, `fl_x/fl_y/cx/cy/w/h`, per-frame `depth_file_path`/`mask_path`/`file_path`) + the uint16-mm depth, BGR rgb, uint8 keep-mask files on disk.
- `Intrinsics`, `FusionConfig`, `DepthConfig` for the seed build.
- For Phase-0b: a trained-scene `GaussianSnapshot` (the registration target: existing object Gaussians + their colors), a `ReferenceFrame`, and a list of `Sam3dArtifact` (per-object SAM3D gaussian-PLY + `_pose.json` rotation sidecar + mask).
- `SegmentationConfig.sam3d_registration_backend` selects ndp/cpd/teaser; near-surface knobs feed `backproject_mask_to_world`.

**Produces:**
- `<static_dir>/depth_camera_init_points.ply` (the Splatfacto seed; near/far-downsampled) + an atomic `transforms.json` `ply_file_path` patch + PLY mtime bump.
- A `list[FusedObject]`: per-object insert-ready `GaussTensors` (scene frame, log-scale, logit-opacity), the `instance_id`, the `match_mask` over existing Gaussians for id propagation, and a `stats` dict.
- `initialization_artifacts/phase0_manifest.json` (the per-object fusion manifest) — written by this module's caller from the returned `stats`, OR optionally here (Open question 3).
- per-frame add timing (`per_frame_add_stats`) for the capture timing ledger.

Data-format guarantees: depth metres/0==invalid internally; all poses OpenGL on the public surface, OpenCV only inside the fuser; `object_flags=0` and `inserted_flags` untouched on the returned tensors (those buffer writes are `GaussianSet`'s job at insert time — this module supplies only `instance_id` and `object_flag=False`).

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `online_fusion.py: OnlineFusion` (+`_GpuOnlineFusion`/`_CpuOnlineFusion` impls, `_cv_c2w`, `add_frame`, `finalize`) | `SeedFuser` (+ the two private impls, auto-select kept). `_cv_c2w` delegates to `camera_conventions.gl_c2w_to_cv`. |
| `online_fusion.py: fuse_recorded_dataset` + `__main__` subprocess block | `build_seed_ply` (in-process) + `build_seed_ply_subprocess` (the isolation wrapper). The frame-read loop is shared, not duplicated. |
| `online_fusion.py: adaptive_downsample` / `_adaptive_downsample_{gpu,cpu}` | private `SeedFuser._downsample` (GPU→CPU dispatch kept); near/far constants come from `FusionConfig`. |
| `online_fusion.py` module consts (`TSDF_VOXEL_M`, `DEPTH_MAX_M`, `TSDF_TRUNC_M`, `FAR_VOXEL_M`, `NEAR_RADIUS_M`, `WITH_COLOR`, ICP_*) | `FusionConfig` / `DepthConfig` fields (the `DGS_TSDF_*`/`DGS_FUSION_DEVICE` reads move to `config.py`). ICP tunables become `FusionConfig` fields. |
| `fusion_runner.py: ConcurrentFusionRunner` + `_FusionWorker` + `_FrameWatcher` | `ConcurrentSeedRunner` (watcher+worker collapsed into one class; `start`/`stop_and_finalize`/`per_frame_add_stats` kept). The `cv2.imread`+gripper-zero loop is the SAME helper `build_seed_ply` uses (fixes SMELL-3 dup). |
| `fusion_runner.py: stop_and_finalize` (the god method) | split into `_drain`, `finalize()`, `_write_seed_ply` (the atomic patch + **mtime bump kept on BOTH paths**, fixing LIFECYCLE-1 divergence). |
| `phase0.py: run_phase0b_fusion` registration/cull/manifest core | `register_objects` — the Phase-0a segmentation/SAM3D-subprocess half stays in the generation module; only register+cull+match moves here. The per-object loop + insert call is split out: this module returns `FusedObject`, the caller inserts. |
| `phase0.py: backproject_mask_to_world` | kept verbatim (signature preserved; deproject core calls `camera_conventions`). |
| `phase0.py: cull_points_in_front` | kept verbatim. |
| `phase0.py: load_anchor_reference` + the cached_train[-1] fallback block | `load_reference_frame` (single resolver, fixes the phase0 dup + the `static_np` NameError by deriving H,W from the resolved image/camera, not the fallback-only `static_np`). |
| `sam3d_fusion.py: register_and_fuse_sam3d_object` | the registration body becomes `register_objects`' inner `_register_one` + `_refine(backend)`; the god-function is split (rigid-init / refine / finalize). |
| `sam3d_fusion.py: load_sam3d_gaussian_ply`, `load_sam3d_rotation_wxyz`, `_apply_sam3d_rotation_init`, `_bbox_diagonal`, `_centroid`, `_median_nn_distance`, `_voxel_downsample`, `_transform_points`, `_run_icp_polish`, the NDP/CPD/TEASER refiners | private helpers of this module (NDP path is the live one; cpd/teaser kept selectable). `_apply_sam3d_rotation_init` uses `rotations.quat_wxyz_to_rotmat`. |
| `ndp_register.py: deform_source_to_target` + `ndp/` package | unchanged dependency; called by the `ndp` branch. |
| `phase0.py` magic cull locals (`CULL_STRENGTH`, `TAU_FLOOR_M`, `CULL_DEPTH_TOL_M`, `FLAG_DEPTH_TOL_M`, `IN_FRONT_BAND_M`, `MAX_RADIUS_M`) | `FusionConfig`/`SegmentationConfig` fields (env-overridable, fixing phase0 MEDIUM "magic-number cull tunables"). |

---

## (6) DROPPED (NOT carried, with reason + audit ref)

| Dropped | Reason | Audit ref |
|---|---|---|
| `_GpuOnlineFusion._sync` + its `import open3d.core as o3c` companion | Defined, never called (zero refs). | fusion_runtime.md DEAD-1 (high) |
| `OnlineFusion.idx` read-only property | No functional reader; `profile_fusion`/`bench_gpu_fusion` mutate `_impl.idx` and are already broken against the wrapper. | fusion_runtime.md DEAD-2 / SMELL-4 |
| `bench_gpu_fusion.py`/`profile_fusion.py`/`sweep_tsdf_voxel.py` reach-through into `_src_cloud`/`_integrate`/`_pend`/`idx` | Stale dead-on-arrival benches that `AttributeError` against the current wrapper; the new `SeedFuser` is not built to keep them working. | fusion_runtime.md SMELL-4 |
| `sam3d_fusion.py: _largest_extent` | Zero callers anywhere (`_bbox_diagonal` is the live extent helper). | sam3d.md §2 (high) |
| `sam3d_fusion.py: reconstruct_mesh_from_points` / `reconstruct_mesh_from_gaussian_ply` | FoundationPose mesh-fallback path; FoundationPose "kept on disk but no longer wired into the runtime". Orphaned public API. | sam3d.md §2 / smell 4.6; CLAUDE.md FoundationPose note |
| `Sam3DInsertionResult.dedup_threshold` / `kept_points` / `visible_source_point_count` fields | Dedup hard-disabled (`dedup_threshold=0`, `kept_points=aligned`) under the NDP default; the "append-only / CPD similarity" doc is stale. The new `FusedObject` carries only what the inserter needs. | sam3d.md smell 4.3 |
| `register_and_fuse_sam3d_object(registration_backend="cpd")` default | The `"cpd"` default is never the effective value (callers always pass `"ndp"`). New code takes backend from `SegmentationConfig`, no misleading default. | sam3d.md smell 4.5 |
| `register_and_fuse_sam3d_object`'s `canonical_to_world_4x4` / `similarity_transform` outputs | NDP is non-linear → `similarity_transform` is identity, `canonical_to_world` carries only the rigid-init approximation; no live consumer needs the matrices (FoundationPose, the only consumer, is unwired). Keep the warped cloud only. | sam3d.md smell 4.3; CLAUDE.md NDP note |
| The stale `"D0.3b3_cpd_meta"` / `t_cpd_refinement` timing keys | Backend-agnostic value mislabeled "cpd"; rename to backend-keyed. | sam3d.md smell 4.4; phase0.md LOW (mislabel) |
| Phase-0a (`run_phase0a_sam3_and_sam3d`, `save_sam3_debug_plots`, the model-CPU-evict dance around the SAM3D subprocess) | Segmentation + SAM3D-generation concern, not seed/registration; lives in the generation module. This module consumes the resulting `Sam3dArtifact`s. | phase0.md §1 (separate function); module boundary |
| The per-object **insert call** (`model.insert_object_gaussians`) + the `object_instance_ids[...]=id` write inside the loop | Surgery is `GaussianSet`'s sole chokepoint; this module returns `FusedObject` and the caller inserts/tags under the lock. Removes the per-iteration re-render + the buffer-desync trigger. | gaussian_set.md §5; phase0.md MEDIUM (buffer desync), RUNTIME_target_architecture (one surgery chokepoint) |
| `WITH_COLOR` as an editable-only module constant | Becomes a config field (consistent env pattern); geometry-only fusion stays available via config. | fusion_runtime.md SMELL-8 |
| The watcher's full re-parse of transforms.json every 0.25 s + swallowed `except` | Worker/watcher merge; surface a persistent parse error instead of console-only spin. (Behavior kept correct; the swallowed-failure smell fixed.) | fusion_runtime.md SMELL-5 / LIFECYCLE-note |
| FF-video / oneshot-FF machinery | Not a fusion concern at all; no writer exists. | CLAUDE.md picker note; gaussian_set.md §6 |

---

## (7) INVARIANTS PRESERVED (CLAUDE.md) + how

- **#3 (transforms.json = ICP-refined poses):** `build_seed_ply` consumes the poses already in transforms.json and fuses the seed in that exact frame (unrecentered: dataparser `orientation/center=none`, `auto_scale=False`). The seed PLY lands in the same frame as the cameras — no recentering, no silent re-frame. The ICP inside `SeedFuser` refines per-frame *fusion* registration only; it does not rewrite transforms.json poses.
- **#8 (per-object identity buffer ownership):** this module is the geometry source for Phase-0b but it does NOT write any identity buffer. It returns `instance_id` + `object_flag=False` + a `match_mask`; the actual writes (`object_instance_ids` on insert + matched existing rows, `inserted_flags` untouched at Phase-0b, `object_flags=0`) happen in `GaussianSet.insert`/`write_instance_ids`. `sam3d_init_target_flags` is never touched (correct — all-zeros). So Phase-0b stays the only `object_instance_ids` writer, via the chokepoint.
- **#1 / #2 (static means LR=0 / camera-opt off):** untouched — this module does no training and sets no LRs. `load_reference_frame` applies the camera-opt offset only when `mode != "off"` (a no-op under static-gs where it is off), so it reads the same poses the trained scene used.
- **#6 (background sky):** N/A to geometry; the returned `GaussTensors` carry per-Gaussian color (SH-decoded from the SAM3D PLY), not a scene background.
- **GPU-TSDF-doubling fix:** the GPU integrate MUST pass `trunc_voxel_multiplier = FusionConfig.tsdf_trunc_m / tsdf_voxel_m` so GPU truncation equals the CPU's 8 mm at any voxel (the 2026-06-16 fix). Carried as a non-negotiable in `SeedFuser`'s GPU path.
- **Decoders/registration produce tensors only:** enforced by `register_objects` returning `FusedObject(tensors=GaussTensors, ...)` — there is no path from this module into a `GaussianSet` mutation.
- **Coordinate conventions:** all OpenGL↔OpenCV / back-projection goes through `camera_conventions` (single sign-correct source), so a stray flip can't silently mis-fuse the seed or mis-place an insert (DUP_coord_conventions Pattern A/B risk).

---

## (8) THREADING

- **`build_seed_ply` / `build_seed_ply_subprocess` / `register_objects`** run on the **capture/static-orchestration thread** at the static→dynamic boundary — BEFORE the dynamic tracker/FF/viser threads exist. By construction there is no contention with `_model_lock`, FF-bg, or the viser render thread (fusion_runtime.md thread-safety note). They may block freely on disk, GPU, and the fusion subprocess — this is not a hot path.
- **`build_seed_ply_subprocess`** spawns a child process (Open3D OOM isolation). The child owns its own Open3D CUDA cache; a GPU OOM aborts only the child. The parent must `wait(timeout)` then fall back to CPU (`FusionConfig.device` honored) — timeout + non-zero-exit handling is mandatory (ARCHITECTURE_PRINCIPLES #7 fault isolation; no silent hang).
- **`ConcurrentSeedRunner`** (non-default `defer_tsdf=False`) runs a single-producer/single-consumer pattern over a bounded `queue.Queue`: a watcher thread enqueues newly-written keyframes, one worker thread integrates them. The only shared mutable state is the queue + the worker's own `timings_ms`/`fail_count`. **`per_frame_add_stats` MUST be called only after the worker is joined** (post `stop_and_finalize`); reading mid-capture races the appending list (fusion_runtime.md RACE-note). Document this precondition on the method.
- **`register_objects` reads a `GaussianSnapshot`**, never the live `GaussianSet` — so even if (hypothetically) it ran concurrent with a mutator it sees a consistent detached view. It returns tensors; the caller does the locked `insert`. This module **never acquires `_model_lock`**.
- **Resource discipline (#6):** the `SeedFuser` `finalize()` drops `_slam`/`_model_pcd`/`_pend` + `gc.collect()` + `o3c.cuda.release_cache()` (kept). Every `cv2.imread`/subprocess/temp-file in the read loop is transient; the subprocess and any temp TIFF (`static0_full_depth_meters.tiff`) get a `try/finally` cleanup. Watcher/worker threads + the queue sentinel are joined with bounded timeouts (the 2 s/30 s magic literals become named `FusionConfig`-derived constants).

---

## (9) OPEN QUESTIONS

1. **Phase-0a/0b split ownership of `Sam3dArtifact`.** This spec assumes Phase-0a (SAM3/FastSAM + SAM3D subprocess) lives in a separate generation module that produces `Sam3dArtifact`s, and `static_fuse.register_objects` only registers them. Confirm the boundary — or should `static_fuse` own the SAM3D-subprocess orchestration too (keeping today's single `phase0.py` surface)? The clean split is recommended; the cost is one more module + a small artifact contract.
2. **`register_objects` returns vs inserts.** The spec has it return `FusedObject` and the caller insert under the lock (keeps surgery in `GaussianSet`). The current code inserts per-object inside the loop and re-renders each iteration to refresh `model.info` for the next object's existing-subset query. If the existing-subset query needs the *post-previous-insert* scene, returning-all-then-inserting changes behavior. Confirm whether per-object inserts must interleave with the registration loop (if so, `register_objects` needs a callback or must take a live `GaussianSet`, weakening the "tensors-only" guarantee).
3. **Manifest writer.** Should `register_objects` write `phase0_manifest.json`, or return `stats` and let the caller write it? Leaning caller-writes (keeps this module I/O-light on the registration path), but the manifest is per-object and assembled here.
4. **`match_mask` / instance-id propagation generality.** Today the existing-Gaussian match is a KNN within a fixed radius (a roadmap-#1 placeholder until per-Gaussian SAM IDs). Keep it as-is (return `match_mask`), drop it pending the per-Gaussian-ID work, or make it config-gated? The KNN radius constants move to config either way.
5. **CPD/TEASER retention.** CLAUDE.md keeps cpd/teaser selectable, but they are dead on the default path and pull in `probreg`/TEASER deps. Keep both fully wired in `static_fuse`, or move them behind an optional-import boundary (NDP-only by default, cpd/teaser importable on demand)? Recommend the latter to keep the default import graph lean.
6. **Concurrent runner survival.** `ConcurrentSeedRunner` only runs when `defer_tsdf=False`, which is non-default and (per the capture-only fix) prone to 1200p GPU OOM. Is the concurrent path worth carrying at all, or should the rewrite ship subprocess-seed-only and delete the watcher/worker entirely? Keeping it preserves an option; deleting it removes ~a class + the SMELL-1/2/3 dup surface.
