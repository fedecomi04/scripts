# Audit: fusion-init / pose-refine / non-rigid-register

Files audited:
- `dynamic_gs/utils/rgbd_fusion_init.py` — ICP+TSDF static-seed builder (legacy/fallback path).
- `dynamic_gs/utils/icp_pose_refine.py` — idempotent wrapper enforcing Design Invariant #3.
- `dynamic_gs/utils/ndp_register.py` — NDP non-rigid registration (default Phase-0b backend).

Grep basis (callers):
`grep -rn "<sym>" scripts/dynamic_gs scripts/scripts --include=*.py`, def/class lines excluded.

---

## 1) FUNCTION / CLASS MAP

### rgbd_fusion_init.py

- `cv_c2w_from_opengl(c2w_opengl) -> np.ndarray` — :99 — OpenGL c2w → OpenCV c2w via `diag(1,-1,-1,1)`. **1 caller**, internal only (`:332`).
- `_backproject_world(depth_m, valid, c2w_cv, fx, fy, cx, cy)` — :104 — back-projects valid depth pixels to world points. **1 caller**, internal (`:343`). NOTE: a *different* `_backproject_world` (8-arg, `def` at `rgbd_decode.py:91`) is the one referenced at `rgbd_decode.py:434` — not this one. So this symbol is internal-only.
- `_make_o3d_with_normals(points) -> o3d.geometry.PointCloud` — :114 — builds an Open3D cloud and estimates normals. **0 callers — DEAD.**
- `_image_gradient(rgb_u8) -> np.ndarray` — :126 — per-pixel Sobel magnitude as a uint8 RGB image (adaptive-density signal). **1 caller**, internal (`:421`).
- `_nn_lookup(pos, src_pc) -> np.ndarray` — :141 — nearest-neighbour color/value lookup via cKDTree. **1 caller**, internal (`:436`).
- `_grid_keep(pos, idx, vx) -> np.ndarray` — :147 — one representative index per voxel. **2 callers**, internal (`:164`, `:443`).
- `_adaptive_keep(pos, detail) -> np.ndarray` — :156 — 3-tier voxel decimation by gradient quantile. **1 caller**, internal (`:437`).
- `_Frame` (dataclass) — :172 — per-frame paths + OpenGL c2w. **Used** at `:205`, `:213`, `:180` in this file. (The `_FrameWatcher` hits at `fusion_runner.py:108/196/212` are a *different* class — substring match only.)
- `_load_static_dataset(static_dir) -> tuple[list[_Frame], dict]` — :180 — parses transforms.json + resolves frame paths + intrinsics. **1 caller**, internal (`:308`).
- `_load_rgb_rgb_order(path) -> np.ndarray` — :217 — read BGR PNG, return RGB. **1 caller**, internal (`:418`).
- `_load_depth_u16_mm(path) -> np.ndarray` — :226 — read uint16 mm depth TIFF. **1 caller**, internal (`:325`).
- `_load_gripper_keep_mask(path, h, w) -> np.ndarray` — :233 — load gripper mask, return inverted bool (True=arm-to-drop). **1 caller**, internal (`:326`).
- `_output_is_fresh(out_ply, static_dir) -> bool` — :254 — idempotency mtime check. **1 caller**, internal (`:298`). (Other two grep hits at `fusion_runner.py:339` / `live_session.py:1106` are comments referencing it.)
- `build_tsdf_seed(data_root, *, force=False, verbose=True) -> Path` — :281 — **public entry**; runs full ICP+TSDF fusion and writes the seed PLY. **3 real call sites:** `live_session.py:1141`, `live_session.py:1168`, `capture_only.py:253` (CPU fallback paths), plus the CLI at `:507`. (Remaining grep hits are comments.)
- `_main_cli()` — :499 — argparse entry. **1 caller** (`__main__` block `:511`). Entry point.

### icp_pose_refine.py

- `_log(msg)` — :54 — prefixed print. **Used** at `:114,120,154,156,169,171,199,207,216` internally.
- `refine_poses_and_refuse(dataset_dir, *, force=False) -> dict` — :58 — **public entry**; enforces Invariant #3 (backup URDF transforms, run ICP rewrite in CPU mode, stale-PLY warn). **1 real caller:** `static_gs_preseg_pipeline.py:137` (import at `:59`). Other hits are docstring/`__all__`.

### ndp_register.py

- `_setup_seed(seed=0)` — :40 — seeds torch/np/random + cudnn deterministic. **1 caller**, internal (`:88`).
- `_truncated_chamfer(a, b, trunc=1e9) -> torch.Tensor` — :49 — symmetric truncated Chamfer via dense cdist. **1 caller**, internal (`:126`).
- `deform_source_to_target(source_xyz, target_xyz, *, device=None, config=None, seed=0) -> tuple[np.ndarray, dict]` — :63 — **public entry**; NDP non-rigid warp of source onto target. **2 real callers:** `sam3d_fusion.py:1185` (import `:1181`) and `scripts/view_object_reconstruction.py:133` (import `:132`).
- `_NDP_CONFIG` (module dict) — :24 — default NDP hyperparams. **Read** at `:81` and imported by `scripts/view_object_reconstruction.py:132,134`.

---

## 2) DEAD-CODE CANDIDATES

- **`_make_o3d_with_normals` — rgbd_fusion_init.py:114 — confidence HIGH.**
  Grep across `dynamic_gs/` + `scripts/`: **0 references** outside its own `def`. Not an entry point, not a callback, not invariant-protected. The two normal-estimation call sites in this file (`:350`, `:376`) inline `pc.estimate_normals(...)` rather than calling this helper. Genuine dead function.

No other zero-ref symbols. Every other private helper has ≥1 internal caller; the three public entry points (`build_tsdf_seed`, `refine_poses_and_refuse`, `deform_source_to_target`) all have live external callers.

---

## 3) DATA-LIFECYCLE

State touched by these three modules is **file-based** (transforms.json, depth/rgb/mask PNG/TIFF, the seed PLY) and **transient GPU tensors** (NDP). None of these modules touch the `.pt` warm-cache, SHM, or the 4 identity buffers directly — those are handled in `persistence/` and the pipelines (invariant-protected; not in scope here).

**rgbd_fusion_init.build_tsdf_seed**
- **Read:** `transforms.json` (`_load_static_dataset`), then per-frame depth TIFF / rgb PNG / mask PNG. All depth preloaded into Python lists `depth_u16`, `valid`, `c2w_cv` (`:321-332`) — held resident for the whole run; RGB is loaded lazily per-frame in the TSDF loop (`:418`).
- **Write:** seed PLY `depth_camera_init_points.ply` **in place, overwriting** the publisher's naive seed (`:459`), plus a timing sidecar `init_seed_timing.txt` (`:485`).
- **Idempotency / format contract:** PLY path is the same one transforms.json's `ply_file_path` points at, so nerfstudio `load_3D_points=True` consumes it unchanged. PLY carries RGB colors (required for `features_dc` init) — preserved (`:451`).
- **Lifecycle observations (low severity):**
  - Open3D `ScalableTSDFVolume` objects `rgbvol`/`gradvol` (`:413-414`) are never explicitly released; they free on GC at function exit. The `gradvol` doubles peak memory during integration when `ADAPTIVE_DENSITY=True`. Acceptable for a one-shot CLI/fallback, but on 1200p this is the same class of VoxelBlockGrid pressure that OOMs the GPU path (CLAUDE.md). This module is CPU Open3D (`ScalableTSDFVolume`, not the tensor VoxelBlockGrid) so the OOM risk is lower, but worth noting it holds two full TSDF volumes + all per-frame depth simultaneously.
  - The full per-frame `depth_u16` list stays alive through both the ICP pass and the TSDF pass (`:330`) — necessary since both passes reuse it, but it is the dominant resident allocation for a large episode.

**icp_pose_refine.refine_poses_and_refuse**
- **Backup lifecycle:** copies `transforms.json` → `transforms_urdf_backup.json` via `shutil.copy2` (preserves mtime) (`:152`). Idempotency guard reads `pose_source` tag (`:113`); backup-collision guard refuses to clobber unless `force=True` (`:134`). Atomic rewrite is delegated (`.tmp`+replace inside `rewrite_transforms_with_icp`).
- **Env-var lifecycle (correct):** sets `DGS_FUSION_DEVICE=cpu` only if unset, restores prior value in a `finally` (`:166-196`) — no leak into the calling process. Good.
- **Return-contract / format match (verified):** `refine_poses_and_refuse` reads `result["frames"]`, `result["dt_mm"]`, `result["dR_deg"]`, `result["elapsed_seconds"]` (`:221-236`); `rewrite_transforms_with_icp` returns exactly `{frames, dt_mm, dR_deg, elapsed_seconds}` on the real (non-dry-run) path (`rewrite_transforms_with_icp.py:193-198`). `.get(..., default)` is used for all, so the dry-run early-return shape `{frames, dt_mm_median}` (rewrite `:186`) would not KeyError — but `refine_poses_and_refuse` always calls with `dry_run=False` (`:180`), so the full shape is the only one reachable here. No mismatch.
- **Partial-write safety:** documented at `:182-189` — delegated rewrite is atomic, backup is source of truth. Sound.
- **MEDIUM — double-backup ambiguity (see Smells):** both this wrapper AND the delegated `rewrite_transforms_with_icp` perform a backup step. The wrapper creates the backup first (`:152`), then the delegate "skips if backup_path already exists" (`:176-178` comment). If the wrapper's `copy2` succeeds but the delegate later raises, `transforms.json` is unchanged (atomic) and the backup now exists but is *unmarked* → a re-run hits the backup-collision guard (`:134`) and **hard-fails** requiring manual `force=True`. So a transient ICP failure leaves the dataset in a state where the next automated run aborts. Recoverable but operator-facing.

**ndp_register.deform_source_to_target**
- **GPU tensor lifecycle:** `src`/`tgt` moved to CUDA (`:91-92`); subsample tensors and per-level Adam optimizers created inside the hierarchy loop (`:119-121`). No explicit free — released on function return / GC. `torch.cuda.synchronize()` before building meta (`:155-156`).
- **No persistent state, no file I/O, no caches.** Pure compute; returns a numpy array + meta dict. No leak across calls. (Repeated calls per object in Phase-0b each allocate a fresh `Deformation_Pyramid` — `sam3d_fusion.py:1185` calls once per object; fine.)
- **Determinism side effect (low):** `_setup_seed` mutates global RNG state (torch/np/random) and sets `torch.backends.cudnn.deterministic = True` **globally and never restores it** (`:46`). This leaks a cudnn mode change into the rest of the process for the remainder of the run. Benign for correctness, but it is a hidden global mutation from a "register two clouds" call.

---

## 4) DESIGN SMELLS

- **MEDIUM — `build_tsdf_seed` is a god function (rgbd_fusion_init.py:281-492, ~210 lines).** Single function does: idempotency check, dataset load, depth preload+mask, frame-to-model ICP loop (with its own nested `_frame_cloud` closure `:342`), dual-TSDF integration (nested `_new_vol`/`_integrate` closures `:393-411`), extraction, adaptive/uniform decimation, SOR, PLY write, timing-sidecar render. Hard to test any stage in isolation; the closures capture loop-local state. Recommendation: not urgent given it is a fallback path, but the ICP loop and the TSDF-integrate+extract stages are the natural extract points.

- **LOW — module-constant config block vs. config-dataclass elsewhere (rgbd_fusion_init.py:65-92).** ~25 tunables live as ALL-CAPS module constants (`ICP_VOXEL_M`, `TSDF_VOXEL_M`, `ADAPT_Q`, …). The newer `online_fusion.py` exposes equivalents as env-overridable (`DGS_TSDF_VOXEL_M`, `DGS_TSDF_DEPTH_MAX_M`). Here `DEPTH_TRUNC_M=3.0` (`:77`) and `TSDF_VOXEL_M=0.0025` (`:75`) are hardcoded and **diverge from the live/online path** (online_fusion default `DEPTH_MAX_M=2.0`, `TSDF_VOXEL_M=0.002` per CLAUDE.md). Since `build_tsdf_seed` is the CPU fallback the live path falls back to (`live_session.py:1141`, `capture_only.py:253`), the fallback seed has a **different depth cap (3.0 vs 2.0 m) and voxel (2.5 vs 2.0/3.0 mm)** than the primary GPU seed — a silent behavioral inconsistency between primary and fallback. Worth flagging given the 2026-06-15 2 m-depth-cap invariant work assumed a single cap value.

- **LOW — duplicated back-projection logic.** `rgbd_fusion_init._backproject_world` (:104) and `rgbd_decode._backproject_world` (:91) are two same-named functions implementing the same camera math with different signatures. Same-name collision is a readability/maintenance trap (a grep for the symbol returns both); they should not share a name.

- **LOW — duplicated normal-estimation + cloud-build.** `_make_o3d_with_normals` (:114, dead) was presumably meant to dedupe the two inline `estimate_normals` blocks at `:350` and `:376` plus `_frame_cloud`'s build; instead it is unused and the logic is inlined three times with the same `KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30)`.

- **LOW — swallowed exception (rgbd_fusion_init.py:488-490).** The timing-sidecar write is wrapped in `except Exception` that only prints a warning when `verbose`. A non-verbose run silently swallows any sidecar write error. Cosmetic (sidecar is non-essential) but it is a blanket catch.

- **LOW — misleading mask-convention naming (`_load_gripper_keep_mask`, :233).** Named "keep" but returns the **inverted** "drop" mask (`m == 0` → True = arm to zero). The docstring explains it, but the function name says the opposite of what it returns. Easy to misuse.

- **LOW — `w_reg` semantics were historically inert (ndp_register.py:127-136).** The inline comment documents that the vendored NDP port had dropped the non-rigidity BCE term, leaving `w_reg` a no-op that only toggled `nonrigidity_est`. It is now restored, but this is exactly the kind of config field that *looked* live while being dead — worth keeping the note. Currently correct (`w_reg=1.0` does weight the BCE loss).

- **LOW — params threaded as a 7-positional-arg closure call.** `_backproject_world(depth_m, valid, c2w_cv, fx, fy, cx, cy)` and `_frame_cloud`/`_integrate` pass intrinsics as loose positional floats through several layers rather than the `intr` dict that `_load_static_dataset` already returns. Minor; raises the chance of an fx/fy or cx/cy swap.

No dead config fields found *on a Config dataclass* in these three modules — none of them define a `*Config` (tunables are module constants here and a plain dict in NDP).
