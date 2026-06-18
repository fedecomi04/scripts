# Fusion runtime audit — `fusion_runner.py` + `online_fusion.py`

Scope: `dynamic_gs/utils/fusion_runner.py` and `dynamic_gs/utils/online_fusion.py`.
Audited 2026-06-17. Repo root: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`.

Grep command used for caller counts:
`grep -rn "<name>" .../dynamic_gs .../scripts --include=*.py` (excluding the defining file).

Live-path note: on the **default** live flow `DGS_LIVE_DEFER_TSDF=1` (live_session.py:605) the
`ConcurrentFusionRunner` is **constructed but never `start()`ed**; the seed is built by a GPU
**subprocess** running `online_fusion.fuse_recorded_dataset` (`python -m dynamic_gs.utils.online_fusion <dir>`).
So in live mode the watcher/worker threads do NOT run, and `OnlineFusion`/`fuse_recorded_dataset`
run in a *separate process* (the `__main__` block, online_fusion.py:634). The concurrent threaded
path only runs when `DGS_LIVE_DEFER_TSDF=0` or via `capture_only.py`/`test_concurrent_fusion.py`.

---

## 1) FUNCTION / CLASS MAP

### `fusion_runner.py`

- **`_FusionWorker(threading.Thread)`** — fusion_runner.py:61 — Worker thread; pulls `(idx, depth_path, rgb_path, mask_path, c2w)` tuples off a queue, reads images, zeros gripper pixels, calls `OnlineFusion.add_frame`, records per-frame ms + fail count. Caller: only `ConcurrentFusionRunner` (fusion_runner.py:216). No external refs.
  - **`__init__(self, fuser, q)`** — fusion_runner.py:68 — stores fuser/queue, inits `timings_ms`/`fail_count`. Caller: 1 (fusion_runner.py:216).
  - **`run(self)`** — fusion_runner.py:75 — thread body; sentinel `None` exits. Caller: `Thread.start()` machinery only.
- **`_FrameWatcher(threading.Thread)`** — fusion_runner.py:108 — Polls `static_scene/transforms.json` every `poll_period_s`, enqueues newly-appended frames; one final sweep on stop. Caller: only `ConcurrentFusionRunner` (fusion_runner.py:212). No external refs.
  - **`__init__(self, static_dir, q, stop_evt, poll_period_s=0.25)`** — fusion_runner.py:116 — Caller: 1 (fusion_runner.py:212).
  - **`_enqueue_from_meta(self, frames)`** — fusion_runner.py:130 — resolves paths from frame dicts, enqueues `[_last_count, len(frames))`. Callers: 2 internal (fusion_runner.py:151, :164). NO external refs.
  - **`run(self)`** — fusion_runner.py:143 — poll loop + final sweep. Caller: `Thread.start()`.
- **`ConcurrentFusionRunner`** — fusion_runner.py:174 — Public API: owns watcher + worker + `OnlineFusion`; `start()` / `stop_and_finalize()`. Callers: `live_session.py:597`, `scripts/test_concurrent_fusion.py:138`.
  - **`__init__(self, static_dir, intrinsics, poll_period_s=0.25)`** — fusion_runner.py:185 — duck-typed `intrinsics` (fx/fy/cx/cy/width/height). Callers: 2 (above).
  - **`start(self)`** — fusion_runner.py:202 — constructs `OnlineFusion`, queue, stop-event, starts both threads; idempotent via `_started`. Callers: `live_session.py:607`, `test_concurrent_fusion.py:139`.
  - **`per_frame_add_stats(self)`** — fusion_runner.py:221 — returns `{mean,p90,max,n,fail}` or `None` if worker never ran. Caller: `live_session.py:727`.
  - **`_last_camera_world_xyz(self)`** — fusion_runner.py:237 — reads last frame's translation from transforms.json. Caller: 1 internal (fusion_runner.py:311). NO external refs.
  - **`stop_and_finalize(self)`** — fusion_runner.py:254 — stops watcher, drains queue, `finalize()`, adaptive downsample, writes PLY, patches transforms.json, bumps PLY mtime. Callers: `live_session.py:719`, `test_concurrent_fusion.py:154`.

### `online_fusion.py`

- **Module constants** — `TSDF_VOXEL_M` (:63), `TSDF_TRUNC_M` (:64), `DEPTH_SCALE` (:65), `DEPTH_MIN_M` (:66), `DEPTH_MAX_M` (:76), `ICP_SRC_STRIDE` (:78), `ICP_VOXEL_M` (:79), `MODEL_REFRESH_EVERY` (:80), `NORMAL_RADIUS_M` (:81), `ICP_STAGES` (:82), `ICP_FITNESS_MIN` (:83), `WITH_COLOR` (:84), `NEAR_RADIUS_M` (:92), `FAR_VOXEL_M` (:93). Re-exported via `fusion_runner` (`FAR_VOXEL_M`, `NEAR_RADIUS_M`, `WITH_COLOR`) and read by `scripts/bench_gpu_fusion.py`, `scripts/sweep_tsdf_voxel.py`, `scripts/fuse_bilateral_experiment.py`.
- **`_adaptive_downsample_gpu(pc, last_cam_world_xyz, near_radius_m, far_voxel_m)`** — online_fusion.py:96 — GPU tensor near/far split + far voxel-downsample. Caller: 1 internal (online_fusion.py:218). NO external refs.
- **`_adaptive_downsample_cpu(pc, last_cam_world_xyz, near_radius_m, far_voxel_m)`** — online_fusion.py:167 — CPU fallback. Caller: 1 internal (online_fusion.py:221). NO external refs.
- **`adaptive_downsample(pc, last_cam_world_xyz, near_radius_m=…, far_voxel_m=…)`** — online_fusion.py:188 — dispatcher GPU→CPU. Callers: `fusion_runner.py:314`, `online_fusion.py:617` (`fuse_recorded_dataset`), `scripts/fuse_bilateral_experiment.py:94`.
- **`_CpuOnlineFusion`** — online_fusion.py:229 — legacy `ScalableTSDFVolume` + `registration_icp`. Caller: only `OnlineFusion.__init__` (online_fusion.py:519, :522). No external refs.
  - **`__init__`** (:233), **`_src_cloud`** (:251), **`_integrate`** (:266), **`add_frame`** (:280), **`finalize`** (:317) — all called only via the `OnlineFusion._impl` delegation. (`bench_gpu_fusion.py`/`profile_fusion.py` reach for these on the wrapper — see SMELL-4.)
- **`_GpuOnlineFusion`** — online_fusion.py:327 — Open3D tensor SLAM VoxelBlockGrid + `multi_scale_icp`. Caller: only `OnlineFusion.__init__` (online_fusion.py:513). No external refs.
  - **`__init__`** (:332), **`_sync`** (:367), **`_src_cloud`** (:371), **`_integrate`** (:387), **`add_frame`** (:422), **`finalize`** (:465). `_sync` — see DEAD-1.
- **`OnlineFusion`** — online_fusion.py:500 — public dispatcher; auto-selects GPU/CPU. Callers: `fusion_runner.py:205`, `live_session.py` (via subprocess `__main__`), `scripts/{profile_fusion,bench_gpu_fusion,fuse_bilateral_experiment}.py`.
  - **`__init__(fx,fy,cx,cy,W,H)`** — online_fusion.py:508.
  - **`_cv_c2w(c2w_opengl)`** (static) — online_fusion.py:525 — OpenGL→OpenCV. Callers: 2 internal (online_fusion.py:553) + `scripts/profile_fusion.py:65`.
  - **`idx` (property)** — online_fusion.py:530 — delegates to `_impl.idx`. NO functional external ref (bench/profile mutate `_impl` attrs, not this property — see SMELL-4).
  - **`add_frame(depth_u16, c2w_opengl, rgb_u8=None)`** — online_fusion.py:534 — converts pose, delegates. Caller: `_FusionWorker.run` (fusion_runner.py:99) + `fuse_recorded_dataset` (:609) + bench scripts.
  - **`finalize()`** — online_fusion.py:556 — delegates. Callers: fusion_runner.py:301, online_fusion.py:611, bench scripts.
- **`fuse_recorded_dataset(static_dir)`** — online_fusion.py:566 — one-shot fusion over all frames in transforms.json; writes PLY + patches transforms.json. Callers: `scripts/renoise_static_depth.py:70`, and the `__main__` subprocess (online_fusion.py:640) invoked by `live_session.py` (the **default live seed build**).
- **`__main__`** — online_fusion.py:634 — subprocess entry `python -m dynamic_gs.utils.online_fusion <static_dir>`. Invoked by `live_session.py` deferred-TSDF path.

---

## 2) DEAD-CODE CANDIDATES

- **DEAD-1 (high): `_GpuOnlineFusion._sync(self)`** — online_fusion.py:367. Defined but **never called** anywhere (grep `_sync` returns only the definition; the only `cuda.synchronize` call site is inside `_sync` itself, which nothing invokes). Companion `import open3d.core as o3c` line is also dead. Ref count outside def: 0.
- **DEAD-2 (low, NOT removable): `OnlineFusion.idx` property** — online_fusion.py:530. Zero functional external readers: `profile_fusion.py`/`bench_gpu_fusion.py` mutate `fuser.idx` expecting the *impl*'s plain attribute, but the wrapper's `idx` is a **read-only property** with no setter, so `fuser.idx += 1` would raise `AttributeError` against the current wrapper (those scripts predate the dispatcher split — SMELL-4). The property is harmless and cheap; flagged only as "no live consumer," not recommended for deletion without confirming the bench scripts.

No other zero-ref symbols. `_FusionWorker`, `_FrameWatcher`, `_CpuOnlineFusion`, `_GpuOnlineFusion`,
`_adaptive_downsample_{gpu,cpu}`, `_last_camera_world_xyz`, `_enqueue_from_meta` all have at least one
in-module caller and are reachable from the public API. Entry points excluded per instructions
(`__main__` subprocess, `ConcurrentFusionRunner` public API).

---

## 3) DATA-LIFECYCLE

### Image / disk handles
- `_FusionWorker.run` (fusion_runner.py:84–98) reads depth/mask/rgb fresh per frame via `cv2.imread`; numpy arrays are local, freed by GC after `add_frame`. No leak. Mask is applied **in-place** (`depth[m==0]=0`) on the just-loaded copy — fine.
- `_FrameWatcher.run` (fusion_runner.py:148) re-reads + re-parses the **entire** transforms.json every 0.25 s. O(frames) JSON parse per tick; for long captures this is wasteful but not a leak (LIFECYCLE-note, not a bug). The final sweep (fusion_runner.py:161–166) re-enqueues from `_last_count`, so frames written during stop are not lost.

### GPU tensors (live path — `_GpuOnlineFusion`)
- VoxelBlockGrid (`_slam`) grows from 8k blocks (online_fusion.py:348). Per `add_frame` it allocates `depth_t`/`rgb_t`/frustum tensors (online_fusion.py:373,394,397,401,404) — these are per-frame and dropped on scope exit; `_pend` accumulates source clouds until `MODEL_REFRESH_EVERY` (online_fusion.py:455,462). Bounded.
- `finalize` (online_fusion.py:465–493) explicitly drops `_slam`, `_model_pcd`, `_pend`, runs `gc.collect()` + `o3c.cuda.release_cache()`. Good — addresses the documented M4 VRAM concern.
- `_adaptive_downsample_gpu` (online_fusion.py:96–164) explicitly `del`s `diff/sq/dist2/tpc/masks` mid-function (M4). The fallback `return pc` at online_fusion.py:141 deletes `dist2/near_mask/far_mask` first. Good.

### Persistent `.pt` warm-cache / identity buffers / SHM
- **None touched here.** Neither module imports `dynamic_gs/persistence/`, the 4 identity buffers (`object_flags` / `object_instance_ids` / `sam3d_init_target_flags` / `inserted_flags`), nor `LiveShmSubscriber` SHM. The SHM lifecycle and `post_fusion_state.pt` live in `live_shm_reader.py` / `persistence/` — out of scope for these two files. (Worth stating explicitly: the "fusion runtime" module produces only the `.ply` seed + a `transforms.json` patch; it does not read/write the warm-cache.)

### transforms.json / PLY save format
- `stop_and_finalize` (fusion_runner.py:331–344) and `fuse_recorded_dataset` (online_fusion.py:626–630) both do atomic `tmp`+`os.replace` rewrites of transforms.json with `ply_file_path` set. **LIFECYCLE-1 (medium):** these two writers are **not consistent** — `stop_and_finalize` also `os.utime(ply_path)` to bump mtime (M1, fusion_runner.py:344) so the redundant `bootstrap_live.sh` re-fusion sees the seed as fresh; `fuse_recorded_dataset` does NOT. In the default live flow the seed IS built by `fuse_recorded_dataset` (subprocess), so the mtime guard the M1 comment relies on is **absent on the actual default path** — if `rgbd_fusion_init._output_is_fresh` is consulted after a deferred build, transforms.json (just rewritten) can have mtime ≥ ply mtime and trigger a redundant re-fusion. Verify whether `_output_is_fresh` is on the deferred path.
- **LIFECYCLE-2 (low):** `stop_and_finalize` patches transforms.json only `if tp.exists()` (fusion_runner.py:332) but never warns when it does not, so a missing transforms.json silently produces a PLY the dataparser won't auto-pick.

### Pose / coordinate convention
- `OnlineFusion.add_frame` converts OpenGL→OpenCV once (online_fusion.py:553) before delegating; both impls receive `c2w_cv`. `_FusionWorker` passes the raw OpenGL c2w from transforms.json (fusion_runner.py:99) → correct. `fuse_recorded_dataset` likewise passes raw `transform_matrix` (online_fusion.py:609) → correct. No double-conversion.

---

## 4) DESIGN SMELLS

- **SMELL-1 (medium) — `stop_and_finalize` is a god method** (fusion_runner.py:254–345, ~90 lines): thread teardown + queue drain + timing print + finalize + adaptive downsample + PLY write + transforms.json patch + mtime bump, all inline. Five distinct responsibilities; hard to unit-test in isolation.
- **SMELL-2 (medium) — duplicated finalize/downsample/write logic** between `ConcurrentFusionRunner.stop_and_finalize` (fusion_runner.py:300–344) and `fuse_recorded_dataset` (online_fusion.py:611–630): both call `finalize()` → `adaptive_downsample` → `write_point_cloud` → atomic transforms.json patch, with subtly **divergent** behavior (the mtime bump exists only in one — see LIFECYCLE-1). This is exactly the kind of drift that bites: the default live path uses the one *without* the mtime guard.
- **SMELL-3 (medium) — duplicated frame-reading loop** between `_FusionWorker.run` (fusion_runner.py:84–99) and `fuse_recorded_dataset` (online_fusion.py:594–609): same `cv2.imread` depth/mask/rgb + gripper-zero + BGR→RGB sequence, with near-identical "depth is pre-filtered, don't double-apply" comments. Two copies to keep in sync.
- **SMELL-4 (medium) — `OnlineFusion` is a leaky abstraction; bench scripts reach through it.** `scripts/profile_fusion.py` (lines 62,65,70,86,88) and `scripts/bench_gpu_fusion.py` (94,107,109) access `fuser._src_cloud`, `fuser._integrate`, `fuser.model`, `fuser._pend`, and **set** `fuser.idx`. After the 2026-06-01 dispatcher split those attributes live on `_impl`, and `idx` became a read-only property — so these scripts are **broken against the current wrapper** (`AttributeError` on first `_src_cloud`/`idx +=`). Not on the live path, but they are stale dead-on-arrival benches masquerading as working tools.
- **SMELL-5 (low) — swallowed exceptions in the watcher.** `_FrameWatcher.run` catches every exception and only `print`s `[watcher] transient` (fusion_runner.py:152–157) / `final pass failed` (:165). A persistent malformed transforms.json (not a transient partial read) would silently stall enqueue forever with only console noise — no failure surfaced to `stop_and_finalize`. Likewise `_last_camera_world_xyz` swallows all exceptions and returns `None` (fusion_runner.py:251), which downstream turns into "adaptive downsample SKIPPED" — a silently un-downsampled (≈9× larger) seed rather than an error.
- **SMELL-6 (low) — busy-wait drain with magic sleeps.** `stop_and_finalize` (fusion_runner.py:281–288) polls `qsize`/`join(2.0)` in a loop; the 2.0 s and 30.0 s timeouts are bare literals. The 30 s watcher-join timeout (fusion_runner.py:275) can proceed-after-warn and silently drop the final keyframes (the warning at :277 is the only signal).
- **SMELL-7 (low) — `_GpuOnlineFusion.finalize` mutates type-incorrect state.** Sets `self._slam = None` / `self._model_pcd = None` (online_fusion.py:482–483) with `# type: ignore`; any subsequent `add_frame` would `AttributeError`/`NoneType`. The class has no guard preventing post-finalize `add_frame`; relies on the docstring contract only. Minor for the single-shot usage but a footgun.
- **SMELL-8 (low) — `WITH_COLOR` is a module-level constant masquerading as config.** Read at import time (online_fusion.py:84) and branched on in 4 places across both files; the docstring advertises it as a tuning knob but there is no env override (unlike `TSDF_VOXEL_M`/`DEPTH_MAX_M`), so toggling it requires a code edit. Inconsistent with the other tunables' `os.environ` pattern.

### Thread-safety / race assessment (live-path priority)
- The concurrent threaded path (`_FusionWorker` + `_FrameWatcher`) is **single-producer/single-consumer over a `queue.Queue`** — safe; no shared mutable state besides the queue and the worker's own `timings_ms`/`fail_count` (written only by the worker thread, read by `per_frame_add_stats`/`stop_and_finalize` **after** the worker is joined — RACE-free because reads happen post-join). `_last_count` is touched only by the watcher thread.
- **RACE-note (low):** `per_frame_add_stats` (fusion_runner.py:226) reads `_worker.timings_ms` while the worker may still be running (it's called from `live_session.py:727` — confirm it's post-finalize). If called mid-capture, `np.asarray(list-being-appended)` could race; in the default deferred path the worker never starts so `_worker.timings_ms` is empty → returns `None` safely. Only a hazard if `DGS_LIVE_DEFER_TSDF=0` and stats are read before drain.
- **No shared model lock here.** This module does NOT touch the `DynamicGSModel` `_model_lock` / viser render lock — feedforward/tracker/render concurrency hazards are in the pipeline modules, not fusion. The fusion runner runs during the *static capture* phase, before the dynamic tracker/FF threads exist, so cross-thread contention with FF/render is not possible by construction.
- **Open3D global CUDA state:** `finalize`'s `o3c.cuda.release_cache()` (online_fusion.py:490) is process-global; harmless because in the default flow it runs in the isolated subprocess (online_fusion.py:634 `__main__`), which is the whole reason for the subprocess (a GPU OOM poisons Open3D's CUDA cache + aborts the process at teardown — contained to the subprocess, parent falls back to CPU).
