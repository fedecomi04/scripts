# `dynamic_feedforward.py` — the feedforward decode+insert worker (layer: dynamic)

## (1) RESPONSIBILITY

On a single-in-flight background thread: take a per-tick dispatch snapshot (camera, RGB, depth, CDN region), run change-detection → AnySplat decode → scene-frame reproject → in-front cull → bounded insert, holding the `_model_lock` only for the millisecond snapshot-read / cull / insert and never across the ~400 ms decode, while enforcing a hard live-gaussian-count ceiling (load-shed) so the zed_final 3M-gaussian blowup cannot happen.

---

## (2) PUBLIC INTERFACE (the contract `pipeline.py` calls)

```python
@dataclass(frozen=True)
class FeedforwardConfig:
    """FF + CDN knobs. Carried by config.py, passed in once. Anysplat-only path."""
    # cadence / gating
    cadence_ticks: int = 10                    # recurring FF every N ticks (the only FF trigger)
    min_gap_s: float = 0.0                     # wall-floor between dispatches (0 = cadence-only)
    # CDN (change detection)
    change_mask: "ChangeMaskConfig"            # the CDN thresholds/cleanup dataclass (see CONSUMES)
    cdn_downsample_target_side: int = 150      # MS-SSIM grid target side (ds = sqrt(H*W)/side)
    cdn_min_component_area: int = 76           # pooled-grid min CC area (OFFICIAL_FILTER_MIN_AREA)
    cdn_top_n_components: int = 256            # max revealed regions fed to the decoder
    # decode / reproject
    opacity_min: float = 0.05                  # AnySplat opacity-logit keep floor
    scale_multiplier: float = 2.0              # log-scale inflate before clamp
    max_scale_m: float = 0.05                  # per-insert uniform-shrink trigger (world m). 0 disables.
    min_scale_m: float = 0.0                   # tiny-speck cull floor (world m). 0 disables.
    voxel_dedup_m: float = 0.0                 # post-decode voxel dedup (0 = off, current default)
    crop_pad_px: int = 50                      # adaptive-crop margin around the change bbox
    # cull
    icp_refine: bool = True                    # ICP live-cloud->scene before reproject (DGS_FF_ICP)
    cull_in_front_depth_tol_m: float = 0.0     # in-front occlusion cull band (m)
    object_mask_scale: float = 1.02            # object-footprint enlarge before CDN-clean subtract
    object_mask_dilate_px: int = 0
    # load-shed (THE blowup guard)
    max_live_gaussians: int = 1_500_000        # hard ceiling; never insert past it (protect object_flags/id)
    purge_opacity_threshold: float = 0.05      # periodic dynamic purge of FF inserts below this
    purge_every_n_calls: int = 0               # 0 = purge only when ceiling pressed; >0 = also periodic


@dataclass(frozen=True)
class FeedforwardDispatch:
    """The immutable per-tick snapshot the pipeline hands to dispatch(). The bg thread
    owns this for its whole run and reads camera/rgb/depth ONLY from here — never from
    the live _latest_tracker_frame (fixes H4/H5 staleness)."""
    seq: int                       # Frame.seq this was built from (event-time provenance)
    camera: "Cameras"              # dispatch-time scene camera (OpenGL c2w + intr) — frozen
    rgb_bgr: np.ndarray            # (H,W,3) uint8 BGR, dispatch-time live frame (gripper-blacked)
    depth_m: torch.Tensor          # (H,W) float32 metres, FILTERED, 0==invalid (already cleaned at batch source)
    object_mask: torch.Tensor      # (H,W,1) rendered tracked-object footprint at dispatch
    gripper_keep: torch.Tensor     # (H,W,1) robot-exclusion keep mask
    scene_intr: dict               # {fl_x,fl_y,cx,cy,w,h}
    d0_instance_id: int            # the tracked id to PROTECT from cull/purge (never dropped)


class FeedforwardWorker:
    """Owns the AnySplat persistent worker handle + the single-in-flight slot. The
    pipeline constructs ONE of these and calls due()/dispatch() each tick from main."""

    def __init__(self, gaussian_set: "GaussianSet", lock: "threading.RLock",
                 cfg: "FeedforwardConfig", anysplat: "AnysplatClient",
                 renderer: "SceneRenderer", *, on_insert=None) -> None:
        """Bind the SSOT GaussianSet + shared _model_lock + the AnySplat IPC client +
        a render callback (renderer.render(camera)->rgb,depth,alpha under lock). on_insert
        is the viser request_render hook (called with inserted index range after insert)."""

    def due(self, tick: int, now_s: float) -> bool:
        """PURE gate: cadence_ticks AND min_gap_s AND not currently in-flight. No side effects,
        no model touch. Pipeline calls this once per tick and stores the result (decide-FF-once)."""

    def dispatch(self, d: FeedforwardDispatch) -> bool:
        """Non-blocking. Acquire the single in-flight slot (try, return False if held), then
        start the daemon bg thread on _run(d). Releases the slot on Thread.start() failure
        (no permanent FF death). Returns True iff a thread was started."""

    def in_flight(self) -> bool: ...           # True while a decode thread is running
    def close(self) -> None: ...               # join-with-timeout + AnysplatClient.close + /dev/shm glob-unlink

    # ---- bg-thread body (NOT called by other modules; documented for the thread contract) ----
    def _run(self, d: FeedforwardDispatch) -> None:
        """[FF-BG THREAD] try/finally(release slot):
        1. compute_cdn(d)                      -> region mask (locked render, lock-free score)
        2. if empty: return (skipped, next cadence fires)
        3. cull_in_front(d, cdn)               -> delete occluders UNDER lock (ms), re-clean CDN
        4. select regions + adaptive crops     -> lock-free numpy
        5. icp + decode + reproject(d, regions)-> GaussTensors (lock-free, ~400 ms)
        6. shed = enforce_ceiling(tensors)     -> trim/skip if would breach max_live_gaussians
        7. gaussian_set.insert(tensors, object_flag=1, instance_id=999)  UNDER lock (ms)
        8. maybe purge_ff_inserts()            -> bounded growth
        9. on_insert(range)                    -> viser request_render"""
```

```python
# ---- CDN: pure functions (no model state; the locked render is the caller's) ----
def compute_change_mask(*, rendered_rgb, rendered_depth, rendered_alpha,
                        live_rgb, live_depth, gripper_mask, object_mask,
                        cfg: ChangeMaskConfig, downsample_factor: int,
                        keep_largest_only: bool = False) -> torch.Tensor:
    """The CDN: coverage-gate valid_mask (alpha + live-depth fillable-hole rescue),
    masked-avg-pool downsample, MS-SSIM dissimilarity, threshold, close/open/min-area
    cleanup, upsample. Returns (H,W,1) bool change mask. keep_largest_only=False (live)."""

def resolve_downsample_factor(rgb_or_shape, configured: int, target_side: int) -> int:
    """Auto MS-SSIM downsample so the score runs on ~target_side**2 px (aspect-preserving)."""

def select_change_regions(cdn: torch.Tensor, *, n: int, min_area: int) -> list[torch.Tensor]:
    """Up-to-n largest CCs above min_area as per-CC bool masks (the decode regions)."""

# ---- decode/reproject: pure (no model state, no torch device side effects on shared state) ----
def reproject_to_scene(decoded: dict, *, sensor_depth_m, scene_c2w, scene_intr,
                       component_mask, scene_crop, cfg: FeedforwardConfig) -> "GaussTensors":
    """Canonical AnySplat decode -> scene frame via per-pixel sensor-depth back-projection
    through SCENE intrinsics (NOT pred intrinsics), un-crop the center-crop, opacity/bg
    filter, scale-multiplier + uniform-shrink/min-cull hygiene, optional voxel dedup.
    Returns a GaussTensors (the GaussianSet.insert contract); empty -> 0-row GaussTensors."""

def icp_refine_scene_c2w(*, sensor_depth_m, scene_c2w, scene_intr,
                         target_xyz_gpu, max_iters=30, max_dist_m=0.02) -> tuple[np.ndarray, dict]:
    """GPU point-to-plane ICP of the live cloud onto a frustum-culled scene-means snapshot;
    returns refined c2w + info. target_xyz_gpu is a detached snapshot taken under lock by _run."""

def adaptive_crop_windows(cdn: np.ndarray, *, pad_px: int) -> list[tuple[int,int,int]]:
    """Square (left,top,size) crop window(s) ENCOMPASSING the change bbox (>=1, <=2 tiled
    when wider than the image short side); fed to AnySplat so edge change isn't lost."""
```

---

## (3) DEPENDS ON (other NEW modules only)

- **`gaussian_set.py`** — the SSOT. This module calls `gaussian_set.snapshot()` (lock-free read of scene means for ICP target + num_points for the ceiling), `gaussian_set.cull(indices, protect_mask=...)` (in-front occlusion cull + the periodic FF-insert purge), `gaussian_set.insert(GaussTensors, object_flag=1, instance_id=999)` (the FF Mode-B tail), and consumes `GaussTensors` + `low_opacity_indices` + `uniform_shrink_log_scales` (purge/shrink helpers). It NEVER resizes a tensor itself.
- **`config.py`** — `FeedforwardConfig` + `ChangeMaskConfig` are produced there (env-overridable once at construction); this module reads no `os.environ`.
- **`frame.py`** — `Intrinsics` / the camera-convention constants and `seq`/`stamp_sec` provenance carried on the dispatch snapshot.
- **`scene_model.py` (renderer seam)** — a narrow `render(camera) -> (rgb, depth, alpha)` callback (used under `_model_lock` for the CDN render + the object-mask render). FF does not import the model; it holds the callback handle.
- **`anysplat_client.py`** (the rewrite's wrapper around the AnySplat subprocess/IPC) — `inference(image_paths, out) -> (npz, timings)`; single-consumer by construction (this worker is the only caller, serialized by the slot). Spawn/adopt/lifecycle live there, not here.
- **the shared `_model_lock`** — a handle created by `pipeline.py`, passed into the ctor. Same instance the tracker, viser, and GaussianSet hold.

NOT a dependency: the tracker, the viser bridge, the SHM/source adapter, persistence. The pipeline mediates all of those; FF only sees the dispatch snapshot + the GaussianSet.

---

## (4) CONSUMES / PRODUCES

**Consumes (inputs at the boundary):**
- `FeedforwardDispatch` (immutable per-tick: dispatch camera, dispatch RGB-BGR, filtered depth-m, object+gripper masks, scene intrinsics, `d0_instance_id`). Depth is already filtered at the batch source — FF does NOT re-filter (would double-apply).
- `ChangeMaskConfig` / `FeedforwardConfig` (all CDN + decode + load-shed knobs).
- a `GaussianSnapshot` (pulled internally) for the ICP target + the live count.
- the AnySplat worker `.npz`/pickle decode output (read once per crop window).

**Produces (outputs):**
- a `GaussTensors` batch inserted via `gaussian_set.insert(...)` with `object_flag=1, instance_id=999, inserted_flags=1` on the new tail (the write itself happens in GaussianSet under the lock — FF supplies the tensors + the flag/id args).
- cull index sets (in-front occluders + periodic low-opacity FF-insert purge) passed to `gaussian_set.cull(..., protect_mask=object_instance_ids==d0_instance_id)`.
- a viser `request_render` signal (via `on_insert`) after a successful insert.
- per-stage timing rows (`FF.*`) appended to the pipeline's timing ledger (p90/p99-capable; off the tracker hot path).

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `dynamic_gs_pipeline_base.py:_dispatch_feedforward_async` + `_feedforward_threaded` + `_run_feedforward` + `_run_feedforward_anysplat` + `_anysplat_bg_run` | `FeedforwardWorker.dispatch` + `_run` (one linear bg-thread body; the 5-method maze collapses) |
| `_recurring_ff_due` + `_ff_due_this_tick` plumbing | `FeedforwardWorker.due(tick, now_s)` (pure gate; pipeline stores the bool once) |
| `_anysplat_slot_lock` acquire/release + `Thread.start()` | the worker's internal slot with try/except-release on start failure (closes H7) |
| `change_mask.py:compute_change_mask` + `ChangeMaskConfig` | `compute_change_mask` + `ChangeMaskConfig` (rgb-only; depth/depth_outlier modes dropped — see §6) |
| `change_mask.py:resolve_downsample_factor` | same name, kept |
| `active_mask.py:build_change_mask`, `_rgb_msssim_score`, `_ssim_map`, `_gaussian_blur_image`, `_threshold_mask`, `_apply_cleanup_recipe`, morphology (`dilate/erode/open/close/remove_small`) | private helpers inside this module's CDN section (one rgb scorer; the mode-dispatch inlined) |
| `active_mask.py:select_top_n_components_filtered`, `keep_all_components_above_min_area` | `select_change_regions` |
| `active_mask.py:OFFICIAL_FILTER_MIN_AREA` | `FeedforwardConfig.cdn_min_component_area` (76) |
| `anysplat_decode.py:reproject_anysplat_to_scene` (the ~205-line god fn) | `reproject_to_scene` returning a `GaussTensors` (parallel-array re-slicing replaced by one masked-subset over a struct; empty -> 0-row GaussTensors, NOT the asymmetric `{"xyz":...}` dict) |
| `anysplat_decode.py:icp_refine_scene_c2w` | `icp_refine_scene_c2w` (kept; target snapshot taken under lock by `_run`) |
| `_anysplat_crop_windows` | `adaptive_crop_windows` |
| `_feedforward_cull_in_front_of_depth` + `_feedforward_cull_then_reclean_cdn` | `_run` step 3 + a `cull_in_front` helper → `gaussian_set.cull` |
| `_feedforward_clean_cdn` + `_scale_mask_about_centroid` (object-footprint subtract/enlarge) | CDN-clean helper using `object_mask_scale`/`object_mask_dilate_px` |
| `anysplat_decode.py:apply_sam3d_*`/`uniform-shrink` scale block (`:791-803`) | `gaussian_set.uniform_shrink_log_scales` (shared helper; called inside `reproject_to_scene`) |
| `_cleanup_anysplat_ipc_file` (atexit, broken glob) | `FeedforwardWorker.close` glob-unlinks `anysplat_*_<pid>*` + `dgs_live_ff_frame_<pid>.png` (fixes H8) |
| the (NEW) `max_live_gaussians` ceiling + periodic insert purge | `enforce_ceiling` + `purge_ff_inserts` (the load-shed the current code lacks — Principle #3) |

---

## (6) DROPPED (NOT carried, with reason + audit ref)

- **The entire `rgbd_decode.py` path** (`decode_component_to_gaussians` + all 7 private helpers). Reason: `enable_feedforward_inpaint` default is `anysplat_decode`; in the shipped live config this file is *unreachable* (short-circuits before import). *Audit: rgbd_decode.md §1/§4 "Branch reachability in live mode: this entire file is unreachable on the default path"; RUNTIME_ff_dispatch.md (rgbd path traced only for completeness).* The single FF backend is AnySplat. If RGB-D-direct is ever wanted again, recover from git.
- **The `rgbd` cull-via-`model.info`** (`_feedforward_delete_in_region` → `extract_projected_centers_and_radii` → `build_active_mask`). Reason: only used by the rgbd path; the AnySplat cull uses direct projection of scene means and proves `model.info` is unnecessary; removing it deletes the H2 cross-thread `model.info` aliasing race entirely. *Audit: RUNTIME_target_architecture.md "DELETE: the model.info rgbd-cull path"; RUNTIME_ff_dispatch.md H2.*
- **`filter_gaussians_by_component_mask` + `_world_to_image_opengl`** (anysplat_decode.py). Reason: zero callers anywhere; a superseded back-project-the-means filtering approach — the live path filters via pred-pixel un-crop + scene-mask index inside `reproject`. *Audit: anysplat_decode.md §2 (both high-confidence dead, self-contained pair).*
- **CDN depth modes** (`_depth_diff_score`, `_depth_outlier_score`, the `mode=="depth"/"depth_outlier"` branches of `build_change_mask`, and config `change_mask_mode`/`use_rgb`/`filter_radius`/`min_component_size`). Reason: `mode` is always `"rgb"` (no env override exists); the depth branches are unreachable, and `use_rgb`/`filter_radius`/`min_component_size` are `del`-ed on the first line of `build_change_mask`. *Audit: change_detection.md §2 (depth scorers unreachable), §4 (dead params threaded through 3 layers).*
- **`build_active_mask_center_only`**, **`combine_object_masks`**, **`build_change_mask` raw-mask fallback**, and the model-side `prepare_dynamic_update` CDN call. Reason: `combine_object_masks`/the model-side `build_change_mask` are dead-by-transitivity through dead `prepare_dynamic_update`; `build_active_mask_center_only` has no call site. *Audit: change_detection.md §2.*
- **`umeyama_similarity` + `apply_similarity_to_gaussians`** (anysplat_decode.py). Reason: only offline diagnostic scripts call them; not on any live/runtime path. The live reproject does its own inline per-pixel transform. *Audit: anysplat_decode.md §1 (callers are `scripts/dump_*`, `scripts/merge_*` only), §2 note.*
- **The one-shot ("oneshot") FF firing** (`_oneshot_ff_due` predicate, the D0-time one-shot dispatch). Reason: the target design fires FF on exactly ONE schedule — recurring cadence (`due()`); the `is_first`/D0 gate stays in the pipeline, not as a second FF trigger. *Audit: RUNTIME_target_architecture.md (FF cadence every 10 ticks, single dispatch); RUNTIME_ff_dispatch.md (predicates "harmless; CDN is unconditional now").*
- **FF-video machinery** (`feedforward_video_out` config + any mp4 writer). Reason: declared but no writer is implemented; produces nothing. *Audit: CLAUDE.md "`feedforward_video_out` is currently declared but no writer is implemented — no mp4 is produced".*
- **`_save_ff_debug_images` on the hot path** (the `_ff_debug` numbered-montage dump + `DGS_FF_DEBUG` I/O). Reason: 140 ms/tick I/O on the dispatch path violates "no I/O on the hot path". Kept ONLY as an opt-in, off-thread, post-insert hook writing from the bg thread after the lock is released (never gating dispatch). *Audit: ARCHITECTURE_PRINCIPLES.md #4 (DGS_FF_DEBUG 140 ms tail); project_next_session_tasks memory.*
- **The asymmetric `{"xyz": empty}` early-return** of `reproject_anysplat_to_scene`. Reason: latent KeyError footgun; replaced by a uniform 0-row `GaussTensors`. *Audit: anysplat_decode.md §3 "asymmetric early-return shape", §4.*
- **`run_anysplat_subprocess` slow one-shot fallback** as an in-module branch. Reason: worker spawn/adopt/one-shot lifecycle moves wholesale into `anysplat_client.py`; FF only calls `.inference()`. *Audit: anysplat_decode.md §1 (it's a worker-availability fallback, a lifecycle concern).*

---

## (7) INVARIANTS PRESERVED (which CLAUDE.md invariants + how)

- **#4 (dynamic phase = ALL gauss-param LRs = 0; pure tracker+FF runtime).** FF performs NO optimizer step — it only `insert`s and `cull`s via the GaussianSet surgery chokepoint. No `.grad`-driven mutation. The pipeline's `get_train_loss_dict` still returns a zero dummy.
- **#8 (per-object identity buffers owned by phases).** FF inserts set ONLY `object_flags=1`, `object_instance_ids=999`, `inserted_flags=1` on the new tail (via `gaussian_set.insert` args) — it never writes `sam3d_init_target_flags` (stays all-zeros) and never touches Phase-0b ids. Cull eligibility is `instance_id ∈ {0,999}` enforced by passing `protect_mask=(object_instance_ids==d0_instance_id)` to every `cull` — the tracked object is structurally undeletable, including by the load-shed purge. *(GaussianSet asserts the 4-buffer length invariant on every mutation; FF cannot desync them because it routes through the one API.)*
- **#9 (viser-direct only, NEVER the NS viewer).** FF's only render is the CDN render via the `render(camera)` callback under `_model_lock` (read-only of gauss_params); it signals the viser thread purely via `on_insert→request_render`. It never calls `get_outputs_for_camera`, never passes `--vis viewer`.
- **#6 (background = Gazebo sky `(0.86,0.92,1.0)`).** The reproject background-color filter and the CDN coverage-gate both use this exact sky tuple as the "uncovered/background" reference (carried in config, not hardcoded per call-site).
- **Load-shed (ARCHITECTURE_PRINCIPLES #3 — the blowup guard).** `enforce_ceiling` refuses inserts that would breach `max_live_gaussians` (trim the insert batch, or skip + schedule a purge), and `purge_ff_inserts` drops below-opacity FF inserts (`inserted_flags==1`, `instance_id==999`) under pressure — never `object_flags==1`. This is the named fix for the zed_final 3M growth.

---

## (8) THREADING

- **Runs on:** the single FF background daemon thread (`_run`), dispatched from the MAIN/tracker thread (`due`/`dispatch`). The CDN/decode/reproject/ICP helpers are pure and run on the bg thread; they are also safe to call from a test thread (no shared mutable state).
- **Single-in-flight:** `dispatch` acquires the worker's slot non-blocking; if held, it returns False and the tick proceeds without FF (fail-safe: no-FF-this-tick > stall). `Thread.start()` is wrapped so a start failure releases the slot (no permanent FF death — closes H7).
- **Lock discipline (the load-bearing rule):** the bg thread acquires `_model_lock` for exactly three short windows — (a) the snapshot read of scene means for the ICP target + num_points, (b) the in-front cull (`gaussian_set.cull`), (c) the insert (`gaussian_set.insert`) — each milliseconds. The ~400 ms AnySplat decode + reproject + ICP-solve run **lock-free** on detached tensors / numpy. The lock is NEVER held across the AnySplat IPC call (would stall the tracker). Re-entrant: the CDN render re-enters `_model_lock` via the render callback.
- **May block on:** the AnySplat worker IPC (`anysplat_client.inference`, off-lock, with a timeout enforced in the client) and acquiring `_model_lock` for the three short windows.
- **May NOT block on:** the tracker tick, the viser render, or hold the lock during decode. Must not read `_latest_tracker_frame` / live RGB / live camera on the bg thread — everything comes from the frozen `FeedforwardDispatch` (fixes H4/H5 staleness).
- **Reads of shared state:** scene means/num_points ONLY via `gaussian_set.snapshot()` (atomic, detached) or under the lock for the cull/insert. No unlocked `model.means` read (the H1/H3 class of races is eliminated because crop/object-mask reads belong to the tracker/pipeline, not FF, and FF's own reads are snapshot-or-locked).
- **Resources:** the AnySplat worker handle + slot + `/dev/shm` IPC files are released in `close()` (`try/finally`, glob-unlink the indexed crop/ipc PNGs/NPZs — fixes H8). No per-frame persistent allocation beyond the transient decode tensors (GC'd at `_run` exit).
- **Event-time:** the dispatch carries `seq`/`stamp_sec`; FF does no `time.time()`-delta math (cadence is a tick count; `min_gap_s` is a coarse wall floor on dispatch, not a fusion/filter dt).

---

## (9) OPEN QUESTIONS for the human

1. **Load-shed policy when at the ceiling:** trim the *new* insert batch to fit (insert as many as room allows), OR skip the insert entirely + trigger a purge next call, OR purge-then-insert in the same `_run`? The purge frees `inserted_flags==1` low-opacity rows — confirm purging mid-`_run` (under the same insert lock window) is acceptable vs deferring to a dedicated purge call.
2. **`max_live_gaussians` value:** 1.5M is my placeholder (zed_final blew to 3M). What is the real ceiling for the 16 GB card with the tracker + viser + AnySplat worker co-resident? Should it derive from a VRAM probe rather than a fixed count?
3. **ICP sign is unconfirmed** (CLAUDE.md: kept ON by eyeball, not a rigorous A/B; the depth-scaled background divergence is a real mechanism). Keep `icp_refine=True` default in the rewrite, or ship it off pending the A/B? The toggle stays in config either way.
4. **CDN coverage-gate `live_depth_max_m` vs fusion `DEPTH_MAX_M`:** currently 3.0 vs 2.0 — pixels in the 2–3 m band get flagged as fillable holes but fusion discards them. Should the rewrite wire these to ONE depth-cap so CDN can't flag what FF can't consistently fill? *(change_detection.md §3 LOW finding.)*
5. **Object-footprint subtract scale (`object_mask_scale=1.02`, the misplacement-ring swallow):** is the +2% still right after the tracker/CDN fixes, or should it be re-tuned / dropped now that the churn loop was tamed by the coverage-gate + keep-all-blobs fixes?
6. **Decode dtype contract:** `reproject_to_scene` output dtype must match the model's gauss_params (float32). Should `GaussTensors.validate()` assert this, or should FF coerce explicitly before `insert`? (rgbd_decode.md flagged the implicit float16 leak; AnySplat is numpy→float32 today, but worth pinning.)
7. **`features_rest` width / `sh_degree` coupling:** the decode hardcodes 15 SH-rest coeffs (sh_degree=3). Should `reproject_to_scene` read `sh_degree` from config and build the width dynamically (so changing sh_degree doesn't silently corrupt inserts), per the rgbd_decode.md §3 medium finding?
