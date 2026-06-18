# `static_segment.py` — object segmentation + SAM3D object-cloud generation (layer: static)

## (1) RESPONSIBILITY

Given one reference frame (the head-on static/anchor view), discover the target object's 2D mask(s) — FastSAM+CLIP by default, SAM3 optional — then run Fast-SAM3D per mask to produce per-object 3D object clouds (gaussian-center PLY + pose sidecar), all through the persistent SAM worker subprocess; it produces masks + SAM3D object clouds and writes NOTHING into the live gaussian set.

This is **Phase 0a only**. Registration of the SAM3D cloud onto the real depth, the proximity/in-front culls, insertion into the scene, and `object_instance_ids` propagation are a *separate* concern (the fusion/register module) — see DROPPED.

---

## (2) PUBLIC INTERFACE

This is the contract the god-file `pipeline.py` calls (once, at the static→dynamic handoff, on the main thread, before any dynamic tick).

```python
@dataclass(frozen=True)
class SegmentReference:
    """The exact frame SAM/SAM3D run on. Caller builds this from the chosen
    static/anchor keyframe so the mask, depth, intrinsics and pose all belong
    to ONE frame (the anchor-ref consistency fix, see INVARIANTS)."""
    rgb_bgr: np.ndarray        # uint8 (H,W,3) BGR, gripper-blacked (mask_keep==0 -> 0)
    depth_m: np.ndarray        # float32 (H,W) metres, 0.0 == invalid
    mask_keep: np.ndarray      # uint8 (H,W) 1==keep (robot/gripper excluded)
    intrinsics: Intrinsics     # frame.Intrinsics (fx,fy,cx,cy,width,height)
    c2w_4x4: np.ndarray        # float64 (4,4) OpenGL c2w of THIS frame

@dataclass(frozen=True)
class ObjectMask:
    """One segmented object instance. Backend-neutral contract."""
    index: int                 # 0-based; instance_id := index+1 downstream (INVARIANT #8)
    mask_path: Path            # uint8 PNG, 1==object
    bbox_xyxy: tuple[int,int,int,int]
    area_px: int
    score: float               # backend-confidence; semantics differ per backend (see CONSUMES/PRODUCES)

@dataclass(frozen=True)
class ObjectCloud:
    """One SAM3D 3D object reconstruction (canonical frame, pre-registration)."""
    index: int                 # matches the ObjectMask.index it came from
    ply_path: Path             # SAM3D gaussian-center PLY (xyz + SH color)
    pose_path: Path            # _pose.json sidecar {translation,rotation(wxyz),scale}
    has_rotation: bool         # pose sidecar carries a finite 4-elt quaternion
    mesh_ply_path: Optional[Path]   # None if mesh export skipped/failed

@dataclass(frozen=True)
class SegmentResult:
    """The whole Phase-0a output handed back to the orchestrator."""
    masks: list[ObjectMask]        # [] when 0 objects found
    clouds: list[ObjectCloud]      # index-aligned to masks; entry may be a no-cloud sentinel
    backend: Literal["fastsam","sam3"]
    timing: dict[str,float]        # S0.* per-step wall (mean only; off hot path)

class StaticSegmenter:
    """Owns the persistent SAM worker for the duration of one Phase-0a run.
    Spawn-once, load-on-demand, explicit close. NOT thread-safe: single caller
    on the main thread."""

    def __init__(self, cfg: SegmentConfig) -> None:
        """cfg = config.SegmentConfig (backend, prompt, thresholds, sam3d knobs).
        Does NOT spawn the worker yet (lazy on first segment call)."""

    def segment(self, ref: SegmentReference, *, out_dir: Path) -> SegmentResult:
        """Full Phase-0a: spawn/adopt worker -> load segmenter -> mask discovery
        -> load SAM3D -> per-mask reconstruct -> write artifacts under out_dir.
        Returns SegmentResult; masks==[] (and clouds==[]) when 0 objects. Blocks."""

    def close(self) -> None:
        """Terminate the worker subprocess (terminate->wait->kill->wait) and
        release its pipe FDs. Idempotent. MUST be called (use try/finally)."""

    def __enter__(self) -> "StaticSegmenter": ...
    def __exit__(self, *exc) -> None: ...   # calls close() — exception-safe teardown
```

Notes on minimality:
- ONE public class (`StaticSegmenter`) + 4 frozen dataclasses + the call returns a single `SegmentResult`. No separate "run_phase0a" free function, no separate FastSAM/SAM3 entry points exposed (backend chosen by `cfg.backend` internally).
- No `infer_raw` / preseg raw-mask path exposed (that belongs to the preseg method, out of scope — see DROPPED).
- No subprocess CLI / `_main` here: the worker subprocess + its CLI live in the worker module; this module is the *client side* only.

---

## (3) DEPENDS ON (other NEW modules only)

- `frame.py` — `Frame`, `Intrinsics` (the `SegmentReference` reuses `Intrinsics`; the orchestrator builds `SegmentReference` from a `Frame` + the chosen keyframe's pose). Depth/mm-scale constants.
- `config.py` — `SegmentConfig` (backend `Literal["fastsam","sam3"]`, prompt text, FastSAM conf/iou/imgsz + auto-threshold knobs, SAM3 confidence_threshold, area/border/dedup filters, max_objects, SAM3D crop padding / max_side / trim toggle, `reuse_cached`).
- `sam_worker.py` (NEW client) — the persistent `SamWorkerClient` (spawn, `load_fastsam`/`load_sam3`/`load_sam3d`, `fastsam_infer`/`sam3_infer`/`sam3d_infer`, `unload_*`, `close`) with the **single internal `_request` lock added** (audit §4 thread-safety). `StaticSegmenter` owns one client instance.
- `depth_ops.py` (NEW, if it exists in the rewrite) — only for the mm<->m boundary + the pytorch3d pointmap back-projection helper, IF that helper is hoisted out of SAM3D-input prep. Otherwise the pointmap build stays internal to the worker call.

Explicitly does NOT depend on: `gaussian_set.py`, the model, the tracker, viser, SHM, persistence. Segmentation never touches the gaussian SSOT.

---

## (4) CONSUMES / PRODUCES

CONSUMES (in):
- `SegmentReference` — one frame's rgb (gripper-blacked) + depth(m) + mask + intrinsics + c2w. The caller is responsible for choosing the anchor/last-keyframe and blacking the gripper; this module does NOT re-derive the reference frame from a datamanager.
- `SegmentConfig` — backend + prompt + thresholds.
- `out_dir: Path` — where to write artifacts (the orchestrator passes `<static>/initialization_*`).

PRODUCES (out, in `SegmentResult` + on disk):
- Per-object mask PNGs (uint8, 1==object) + a `<stem>_results.json` summary `{objects:[{mask_path,score,bbox,mask_area,object_index}], segmentation_backend}` (backend tag retained — used for cache invalidation).
- Per-object SAM3D PLY (gaussian centers + SH color) + `_pose.json` `{translation, rotation(wxyz, len==4 enforced by producer), scale}`. `has_rotation` validated.
- Optional per-object mesh PLY (best-effort; None on skip/fail).
- `timing` dict (S0.1 segmentation, S0.2 sam3d_generation, S0.4a total — mean only).

Contract honesty notes (carried from audit, must be documented at the boundary):
- `score` semantics differ by backend (SAM3 = model confidence; FastSAM = survivor-softmax prob). Downstream MUST NOT numerically compare scores across backends. The field stays but is documented as backend-relative.
- `clouds` is index-aligned to `masks`; a per-mask SAM3D failure yields a no-cloud sentinel (e.g. `ObjectCloud` with `ply_path=None`) at that index, NOT a shifted list — preserves the index⇔instance_id invariant.

---

## (5) SOURCE MOVED IN (current file:symbol -> what it becomes)

- `fusion/phase0.py:run_phase0a_sam3_and_sam3d` -> `StaticSegmenter.segment` (the Phase-0a orchestration body: backend branch, cache check, SAM3D-needs-run gate, artifact writes). The model/datamanager coupling is REMOVED — input becomes the explicit `SegmentReference` (kills the duplicated reference-frame derivation, phase0 audit §MEDIUM, and the `NameError static_np` HIGH bug, phase0 audit §HIGH).
- `fusion/phase0.py:save_sam3_debug_plots` -> internal `_save_review_plots` helper of `StaticSegmenter` (debug-only; gated, off the result path).
- `utils/fastsam_segmentation.py:FastSamTextSegmenter` (+ `_run_fastsam`/`_clip_scores`/`_split_into_components`/`select_kept_indices`/`infer`) -> stays *inside the worker subprocess module*; `StaticSegmenter` reaches it via `SamWorkerClient.load_fastsam`/`fastsam_infer`. Not re-implemented here.
- `utils/sam3_segmentation.py:run_sam3_segmentation` + `load_sam3_masks` -> SAM3 path inside the worker; client-side `sam3_infer` + result-JSON parse. `load_sam3_masks` collapses into the `<stem>_results.json` reader in `_parse_segment_results`.
- `utils/sam3d.py:run_sam3d_multi_object` (per-mask loop, OOM resize ladder, pose extraction, `apply_sam3d_gaussian_trim`) -> stays inside the worker (`SamWorkerClient.sam3d_infer`); `StaticSegmenter` calls it once. `get_sam3d_output_paths`/`resolve_sam3d_pose_path`/`load_sam3d_pose`/`sam3d_pose_has_rotation` -> the `ObjectCloud` constructor's validation helpers (one canonical path map + one rotation-presence check).
- `utils/sam3d.py:prepare_cropped_sam3d_inputs` + `_build_pytorch3d_pointmap` -> the SAM3D-input prep step inside `segment` (or hoisted into the worker call args). The depth-resize-to-image block is deduplicated to ONE site (was triplicated, sam3d audit §4.8).
- `utils/sam_worker.py:SamWorkerClient` -> reused as-is (the NEW client), with the `_request` mutex + `terminate->kill->wait` close fix folded in; `StaticSegmenter` is its sole owner during Phase-0a.

---

## (6) DROPPED (current code NOT carried, with reason + audit ref)

- **All of Phase 0b** (`run_phase0b_fusion`, `backproject_mask_to_world`, `cull_points_in_front`, `load_anchor_reference`, `register_and_fuse_sam3d_object` + the entire `sam3d_fusion.py` registration stack) — belongs to the fusion/register module, NOT segmentation. This module's job ends at "masks + SAM3D clouds". (phase0.md §1; sam3d.md §register_and_fuse god-function.)
- **CPD / TEASER++ registration backends** and their ~20 FPFH/normal/ICP-polish helpers (`_run_teaser_*`, `_run_probreg_*`, `_multiscale_fpfh_descriptors`, …) — registration, not segmentation; out of scope here. (sam3d.md §2 "NOT dead, but register-only".)
- **The raw-mask path** (`fastsam_infer_raw` client + `FastSamTextSegmenter.infer_raw` + `sam3_infer_raw` where FastSAM): the FastSAM raw RPC has **no in-repo caller** (sam_worker.md §2, segmenters.md §2 high-confidence). The SAM3 raw path is preseg-only (`preseg_seed.py`) — the preseg method is a separate static method, not this object-track flow. Drop from this module's surface.
- **ESAM** (`esam.py`, `query_esam_mask`/`_run_esam_query`/`_select_esam_mask`) — single-image ESAM path is dead (segmenters.md §2 medium); the live `_pair` variant is a *dynamic-phase* D0 mask query, not static segmentation. Not in this module.
- **`reconstruct_mesh_from_points` / `reconstruct_mesh_from_gaussian_ply`** (FoundationPose mesh fallback) — FoundationPose is no longer wired into the runtime (CLAUDE.md); orphaned public API (sam3d.md §4.6). The worker's best-effort `enable_mesh` export already covers the optional mesh.
- **Duplicated helpers**: `_compute_iou`/`_touches_n_borders` (fastsam copies are dead, segmenters.md §2), `_resolve_env_python` (4x copy), `_largest_extent` (0 refs, sam3d.md §2). Collapse to one copy each (in the worker or a small shared util).
- **`SamWorkerClient.fastsam_infer_raw` + `__enter__/__exit__`-as-CM** dead surface (sam_worker.md §2) — drop the raw client method; keep CM only because `StaticSegmenter` itself offers `with` (the client's own CM is unused).
- **FF-video machinery / oneshot FF path** — not present in segmentation; nothing to carry (mentioned in the prompt's drop list for the broader purge, N/A here).
- **Model CPU-eviction dance** (`model.to("cpu")` around the SAM3D subprocess, image-cache CPU/GPU shuffle, phase0.py:528-743) — was needed because Phase-0a ran *inside* the training model's process holding GPU. In the rewrite, segmentation runs at the static→dynamic boundary with the model NOT yet on GPU (or under the orchestrator's explicit VRAM sequencing), so the asymmetric eviction/restore (phase0.md §3 "subtle, guard-asymmetric") is dropped. If VRAM sequencing is still needed, it becomes ONE explicit orchestrator call, not buried here.
- **Swallowed-exception SAM3D-failure-as-"0 objects"** (phase0.py:689-691) — replaced by a surfaced, logged-loud degradation (a no-cloud sentinel + a warning), per ARCHITECTURE_PRINCIPLES §7 fault isolation.

---

## (7) INVARIANTS PRESERVED (CLAUDE.md + audit)

- **Invariant #8 (per-object identity buffers).** This module NEVER writes `object_instance_ids`, `object_flags`, `sam3d_init_target_flags`, or `inserted_flags` — it only emits masks/clouds with a 0-based `index`. The `index⇔instance_id=index+1` mapping is preserved by the index-aligned, no-shift-on-failure contract (a per-mask failure leaves a sentinel at its index). Phase-0b downstream is the sole writer of `object_instance_ids`. (phase0.md §3 confirms 0a does not touch buffers.)
- **Invariant #7 (persistent SAM worker is canonical).** Segmentation goes through the spawn-once `SamWorkerClient` (load-on-demand SAM3/FastSAM/SAM3D, JSON-over-pipe). Legacy per-call subprocess paths are NOT the primary path here. (CLAUDE.md #7.)
- **Gripper-blacked reference (the anchor_ref consistency fix).** The caller supplies ONE `SegmentReference` whose rgb/depth/intrinsics/c2w all belong to the same frame; this module does not independently re-resolve "the last frame", eliminating the mask-on-wrong-frame class (CLAUDE.md "SAM3D insert offset root cause"; phase0.md §MEDIUM duplicated reference-frame resolution).
- **Does NOT touch the gaussian SSOT or any lock.** No `gauss_params`/buffer access (ARCHITECTURE_PRINCIPLES §1/§9) — runs entirely before the dynamic threads exist.
- **Background color (#6), means-LR (#1), camera-opt (#2), zero-LR dynamic (#4)** — not in this module's surface; nothing here can violate them (segmentation produces clouds, not optimizer config).

---

## (8) THREADING

- **Runs on the MAIN thread only**, once, at the static→dynamic handoff, BEFORE the tracker/FF/viser threads are spawned. There is no concurrency with the live path.
- **Blocks freely**: `segment()` is a long synchronous call (worker spawn ~seconds, SAM3D per-mask seconds). This is acceptable because it is a one-time setup step, not on any per-tick hot path (ARCHITECTURE_PRINCIPLES §4 — no hot-path work here, by construction).
- **May block on**: the SAM worker subprocess (with a per-call timeout — REQUIRED, per audit §4: a corrupted response must fail fast, not hang 600s); disk I/O for artifacts.
- **Lock discipline**: owns ONE `SamWorkerClient`; that client's `_request` MUST be mutex-guarded internally (sam_worker.md §4 high latent) even though this module is single-threaded, so the client is safe if ever reused. `StaticSegmenter` itself holds no shared-state lock (touches no shared state).
- **Resource lifecycle (ARCHITECTURE_PRINCIPLES §6)**: the worker subprocess + its pipe FDs are owned here and released in `close()` via `terminate()->wait(timeout)->kill()->wait()` AND explicit pipe close, wrapped so `__exit__`/`finally` always runs (fixes the FD/zombie leak, sam_worker.md §3). One owner, guaranteed release on exception.

---

## (9) OPEN QUESTIONS

1. **Module boundary with the registration module.** Confirm Phase-0b (`register_and_fuse_sam3d_object`, culls, `backproject_mask_to_world`, instance-id propagation) lives in a *separate* `static_register.py`/fusion module and that `static_segment.py` stops at `SegmentResult`. The task statement ("produces masks + the SAM3D object cloud") implies yes — please confirm so I don't pull the register stack in.
2. **Who builds `SegmentReference`?** I assumed the orchestrator (god-file) chooses the anchor/last keyframe + blacks the gripper and constructs `SegmentReference`. Alternative: pass a `Frame` + a "which keyframe" selector and let this module black the gripper. The first keeps segmentation frame-agnostic (preferred); confirm.
3. **VRAM sequencing ownership.** The old code did model-CPU-eviction inside Phase-0a because it ran in the training process. In the rewrite, who guarantees SAM3D's ~7.3 GB (trimmed) co-resides safely (model not on GPU yet, or explicit unload)? If it's the orchestrator's job, this module just loads/unloads its own worker models; confirm that's the intended split.
4. **Multi-object scope now or later.** Roadmap #3 makes `prompt` a `list[str]`. The `SegmentResult` already returns a list of objects (multi-mask), but the picker today tracks one. Keep the multi-object surface (list) and let the picker select downstream? (I designed for the list — confirm we want it now.)
5. **`reuse_cached` semantics + backend-tag invalidation.** Keep the on-disk `<stem>_results.json` cache with `segmentation_backend` tag invalidation (phase0.py:516-526) as the only caching mechanism, or drop caching entirely for the rewrite's clean-slate runs? Caching saves a re-segment on resume; it also adds a stale-frame foot-gun. Recommend keep with the tag check; confirm.
6. **SAM3 vs FastSAM default.** CLAUDE.md says FastSAM is the default (`segmentation_backend="fastsam"`). Confirm SAM3 stays as a selectable fallback in `SegmentConfig` rather than being dropped — it's heavier (3.8 GB) and FastSAM passed the IoU gate (0.79).
