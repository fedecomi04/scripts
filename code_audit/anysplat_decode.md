# Code Audit — `dynamic_gs/utils/anysplat_decode.py`

LIVE-PATH module. AnySplat feedforward decode: subprocess dispatcher + canonical→world alignment. 868 lines, 12 top-level symbols.

Caller grep run:
`grep -rn "<sym>" .../dynamic_gs .../scripts --include=*.py` (definition file excluded).

---

## 1) FUNCTION / CLASS MAP

### `class PersistentAnysplatWorker` — anysplat_decode.py:41
Long-lived AnySplat subprocess; loads the model once then issues many inferences over a pipe (spawn mode) or FIFO (adopt mode).
**Callers:** 3 refs in `dynamic_gs/dynamic_gs_pipeline_base.py` (`_start_anysplat_persistent_worker`, lines 2910/2915/2936). Constructed at 2936; `.adopt()` at 2915.

- `__init__(self, conda_env="anysplat_dynamic_gs", startup_timeout_s=60.0)` — :49
  Spawns the worker process directly via the env python (skips `conda run`), prepends env lib to `LD_LIBRARY_PATH`, then blocks reading stdout until a `{"status":"ready"}` JSON sentinel. **Caller:** `dynamic_gs_pipeline_base.py:2936`.
- `load_seconds(self)` (property) — :107
  Returns cached model-load seconds. **Callers:** `dynamic_gs_pipeline_base.py:918, 2923, 2942` (via `getattr(worker,"load_seconds",...)` / `.load_seconds`).
- `_alive(self)` — :111
  True if the spawned proc is running, or (adopt mode) the adopted pid's cmdline still matches the worker. **Callers:** internal only — `inference` (193, 210), `close` (235, 248, 250, 256). No external refs.
- `adopt(cls, fifo_dir, wait_ready_timeout_s=60.0)` (classmethod) — :118
  Adopts a worker pre-spawned by `spawn_detached_anysplat_worker` in an earlier process; connects to its `cmd.fifo`/`res.fifo`. Returns `None` (caller should spawn fresh) on any failure. **Caller:** `dynamic_gs_pipeline_base.py:2915`.
- `inference(self, image_paths, output_npz, timeout_s=60.0)` — :178
  Sends one JSON request line, blocks for one JSON response line, returns the output `.npz` path + per-phase timing dict. Raises on worker death/error/timeout. **Caller:** `dynamic_gs_pipeline_base.py:3374` (the FF bg thread).
- `close(self)` — :234
  Sends `{"cmd":"quit"}`, waits, and targeted-kills the verified pid if it lingers (adopt mode) or `.kill()`s the proc (spawn mode); closes pipes. **Caller:** `dynamic_gs_pipeline_base.py:635` (`_cleanup_anysplat_worker`).

### `_pid_is_anysplat_worker(pid)` — anysplat_decode.py:268
True iff `pid` is alive AND its `/proc/<pid>/cmdline` contains `anysplat_worker.py` (guards against recycled pids).
**Callers:** internal only — `_alive` (115), `adopt` (144, 156), `spawn_detached_anysplat_worker` (303). NO external refs.

### `spawn_detached_anysplat_worker(fifo_dir, conda_env="anysplat_dynamic_gs")` — anysplat_decode.py:278
Fire-and-forget spawn of the FIFO-mode worker (`start_new_session=True`) so its model load overlaps capture/static-training; replaces any stale worker recorded in the dir, writes `spawn.json`, returns the pid.
**Callers (3):** `static_gs_pipeline.py:164`, `live_session.py:1066, 1154`.

### `run_anysplat_subprocess(image_paths, output_npz, *, conda_env=..., timeout_s=300.0)` — anysplat_decode.py:342
One-shot (non-persistent) worker spawn; blocks with `subprocess.run`, returns the output `.npz`. The slow fallback when the persistent worker is unavailable.
**Caller:** `dynamic_gs_pipeline_base.py:3379` (FF bg thread `else` branch; imported at 3218).

### `umeyama_similarity(src, dst)` — anysplat_decode.py:394
Closed-form 7-DoF similarity (s,R,t) from corresponding (K,3) point sets (Umeyama 1991).
**Callers (2, both offline scripts):** `scripts/dump_anysplat_outputs_to_ply.py:145`, `scripts/merge_anysplat_with_scene_ply.py:132`. **NOT called from any live/runtime path** (see Smells).

### `quat_wxyz_to_rotmat(q)` — anysplat_decode.py:425
Batched wxyz quaternion → (...,3,3) rotmat.
**Callers:** in-module `apply_similarity_to_gaussians` (496) and `reproject_anysplat_to_scene` (778); external `scripts/anysplat_npz_to_viser_pt.py:38`, `scripts/anysplat_patch_to_scene.py:233`. (`viser_direct.py` has its own private `_quat_wxyz_to_rotmat_np` — NOT this one.)

### `rotmat_to_quat_wxyz(R)` — anysplat_decode.py:438
Batched (...,3,3) rotmat → wxyz quat (Shepperd's method).
**Callers:** in-module `apply_similarity_to_gaussians` (498), `reproject_anysplat_to_scene` (780); external `scripts/anysplat_patch_to_scene.py:235, 268`. (`viser_direct.py`/`dump_viser_pt_to_ply.py` use their own private copies.)

### `apply_similarity_to_gaussians(*, means_canonical, log_scales, quats_wxyz, similarity_s, similarity_R, similarity_t)` — anysplat_decode.py:483
Applies an (s,R,t) similarity to a gaussian set (means, log-scales, quats).
**Callers (2, both offline scripts):** `scripts/merge_anysplat_with_scene_ply.py:135`, `scripts/dump_anysplat_outputs_to_ply.py:148`. **NOT on the live path** (the live reproject in `reproject_anysplat_to_scene` does its own inline per-pixel transform, NOT this function).

### `_world_to_image_opengl(xyz_world, c2w, fx, fy, cx, cy, width, height)` — anysplat_decode.py:502
Projects world xyz → pixel (u,v) + validity using the Nerfstudio OpenGL convention.
**Callers:** internal only — `filter_gaussians_by_component_mask` (856). NO external refs. (Transitively dead — see §2.)

### `icp_refine_scene_c2w(*, sensor_depth_m, scene_c2w, scene_intr, target_xyz_gpu, max_iters=30, max_dist_m=0.02, stride=4, min_pts=1000, target_voxel_m=0.0)` — anysplat_decode.py:525
GPU point-to-plane ICP of the live sensor cloud against a caller-supplied frustum-culled target tensor; returns (refined_c2w, info).
**Caller:** `dynamic_gs_pipeline_base.py:3322` (FF bg thread; imported at 3216). LIVE PATH.

### `reproject_anysplat_to_scene(*, ~20 kwargs)` — anysplat_decode.py:624
The core canonical-AnySplat → scene reprojection. Opacity/background/mask filtering, pred-pixel→scene-pixel un-crop, per-pixel sensor-depth back-projection through scene intrinsics, scale hygiene, optional voxel dedup. Returns the insert dict for `model.insert_inpaint_gaussians`.
**Caller:** `dynamic_gs_pipeline_base.py:3405` (FF bg thread; imported at 3217). LIVE PATH, hottest function.

### `filter_gaussians_by_component_mask(*, means_world, target_camera, component_mask)` — anysplat_decode.py:832
Returns a bool (N,) mask of gaussians whose 2D projection lands inside the CDN component, using `_world_to_image_opengl`.
**Callers:** NO REFS FOUND anywhere (`dynamic_gs`, `scripts`, experiments). DEAD — see §2.

---

## 2) DEAD-CODE CANDIDATES

| symbol | file:line | grep evidence | confidence |
|---|---|---|---|
| `filter_gaussians_by_component_mask` | anysplat_decode.py:832 | 0 refs in `dynamic_gs/`, `scripts/`, or anywhere under repo root (`grep -rn ... --include=*.py` → empty). Not an entry point, not a config target, not invariant-protected. | **high** |
| `_world_to_image_opengl` | anysplat_decode.py:502 | 0 external refs; only caller is `filter_gaussians_by_component_mask` (line 856), which is itself dead. Transitively dead. | **high** |

Both are a self-contained dead pair: the live FF path does component-mask filtering *inside* `reproject_anysplat_to_scene` (the `component_mask` kwarg, lines 742–756) via a different mechanism (pred-pixel un-crop + scene-resolution mask index), not via projecting `means_world` back. `filter_gaussians_by_component_mask`/`_world_to_image_opengl` appear to be a superseded earlier filtering approach.

Not flagged (verified live/offline-script callers exist): `umeyama_similarity`, `apply_similarity_to_gaussians`, `quat_wxyz_to_rotmat`, `rotmat_to_quat_wxyz`, `_pid_is_anysplat_worker`, `PersistentAnysplatWorker.*` (`_alive` is internal-but-live), `icp_refine_scene_c2w`, `reproject_anysplat_to_scene`, `spawn_detached_anysplat_worker`, `run_anysplat_subprocess`.

Note on `umeyama_similarity` / `apply_similarity_to_gaussians`: only referenced by offline diagnostic scripts (`scripts/dump_anysplat_outputs_to_ply.py`, `scripts/merge_anysplat_with_scene_ply.py`). They are NOT dead, but they are NOT on the live path — if the offline scripts are also being purged tomorrow, these become dead. Confidence that they are dead *today*: low (scripts still reference them).

---

## 3) DATA-LIFECYCLE

### Subprocess / process handles
- **Spawn (pipe mode), `__init__`:** `subprocess.Popen` with `stdin/stdout/stderr=PIPE` (:77). On startup timeout the proc is `.kill()`ed (:91) and on poll-death the stderr tail is read (:94). **`close()` (:234)** sends quit, waits 5 s, falls back to `.kill()`. Pipes closed in `finally` (:261). Lifecycle looks complete. The owning pipeline calls `close()` via `_cleanup_anysplat_worker` (pipeline_base:635).
- **Spawn (detached/FIFO mode), `spawn_detached_anysplat_worker`:** `start_new_session=True` (:335) → the worker SURVIVES the spawning process by design (overlaps capture/training). `worker.log` fd is opened (:328) and `log_f.close()` at :337 — comment says "child holds its own fd" (OK). **Stale-worker replacement:** before spawning, an old recorded pid is `os.kill(...,9)`ed only after `_pid_is_anysplat_worker` verifies the cmdline (:303) — safe against pid recycling. **Leak risk (low/by-design):** if the adopting process never runs (e.g. static training spawns it, then the dynamic pipeline is never launched), the detached worker is orphaned holding ~3.5 GB VRAM until the next `spawn_detached_anysplat_worker` on the same `fifo_dir` kills it. There is no independent reaper; the only killer is the next spawn for the same dataset dir. Cross-dataset orphans would persist.
- **FIFOs:** `os.mkfifo(cmd.fifo / res.fifo)` created in `spawn_detached_anysplat_worker` (:312–315) but **never unlinked anywhere** in this module. They are recreated only `if not p.exists()` so they don't multiply, but they leak one FIFO pair per dataset dir on disk (small, in the dataset dir). `adopt` opens `send_f`/`recv_f` on them (:165) and `close()` closes those fds (:261), but the FIFO inodes themselves are never removed.

### IPC files (.npz / .png in /dev/shm) — produced/consumed by this module
- `inference` / `run_anysplat_subprocess` write the worker output to a caller-supplied `output_npz`. The live caller (`_anysplat_bg_run`, pipeline_base:3369) uses `/dev/shm/anysplat_ipc_{pid}_{wi}.npz` and a crop input `/dev/shm/anysplat_crop_{pid}_{wi}.png` (:3367).
- **LEAK (medium):** the atexit cleanup `_cleanup_anysplat_ipc_file` (pipeline_base:644) unlinks only `/dev/shm/anysplat_ipc_{pid}.npz` — **no `_{wi}` suffix**, so it does NOT match the files actually written (`..._{pid}_0.npz`, `..._{pid}_1.npz`) nor the crop PNGs (`anysplat_crop_{pid}_{wi}.png`). Those tmpfs files are never cleaned. Bounded per process (same path reused each FF call, ≤2 windows), so it's a fixed-size residue in tmpfs, not unbounded growth — but it is a real cleanup mismatch and lives in tmpfs (RAM). This is a caller-side defect, but it concerns this module's output contract. Flagging for the synthesis.
- `out_npz` is read back via `pickle.load` (pipeline_base:3393) — note the file written by the worker as `.npz` is actually a pickle blob; the `.npz` extension is misleading (naming smell, worker-side).

### GPU tensors (this module)
- `icp_refine_scene_c2w`: builds a source cloud on GPU from `sensor_depth_m` (`torch.from_numpy(...).to(dev)`, :557), strided + masked; hands both source and `target_xyz_gpu` to Open3D CUDA via DLPack (`from_dlpack(to_dlpack(...))`, :585, :588). These are transient locals freed at function return (Python refcount). The DLPack zero-copy share means Open3D aliases the torch buffers; both are released when the locals drop. No explicit `torch.cuda.empty_cache`; per-call allocations rely on the caching allocator. Acceptable but per-FF-call churn (called once per FF call on the bg thread).
- `target_xyz_gpu` is supplied by the caller (`target_xyz_t = means_all_t[visible_t]`, pipeline_base:3293) — a fancy-index COPY of the scene means, read under `_viser_lock_ctx()` (pipeline_base:3281). Snapshotted before the lock is released, so concurrent inserts can't mutate it mid-ICP. OK.
- `reproject_anysplat_to_scene` is pure-numpy on CPU (no torch device tensors); returns numpy arrays. No GPU lifecycle here.

### Identity buffers (the 4 invariant-protected buffers)
This module **does not touch** `object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, or `inserted_flags` directly. It returns the insert dict; the actual `inserted_flags` write happens downstream in `model.insert_inpaint_gaussians` (per CLAUDE.md Invariant #8). Invariant-protected — not analyzed further here. No desync introduced by this module.

### `.pt` warm-cache / persistence
Not touched by this module (no `dynamic_gs/persistence/` import, no `post_fusion_state.pt` read/write here).

### Shape/format notes
- Output dict shapes are documented and consistent with `insert_inpaint_gaussians`: `opacities` (N,1), `scales` log-scales (N,3), `quats` wxyz (N,4) (:822–828). The early-return on empty returns ONLY `{"xyz": empty (0,3)}` (lines 680, 739, 756) — a **partial dict missing all other keys**. The caller must guard on `xyz.shape[0]==0` before indexing other keys; if it ever reads `decoded["scales"]` unconditionally it would `KeyError`. (Caller does check — but this asymmetric return shape is a latent footgun; see §4.)

---

## 4) DESIGN SMELLS

- **God function — `reproject_anysplat_to_scene` (:624–829, ~205 lines, ~20 kwargs).** Does opacity filter → background filter → pred-projection → un-crop (two branches) → sensor-depth lookup → optional drop-no-depth → optional component-mask filter → back-projection → per-gauss scale → rotation basis change → scale hygiene (shrink+cull) → optional voxel dedup. Each stage re-slices the same 6 parallel arrays (`means_canonical, log_scales, quats_wxyz, opacity_logits, features_dc, features_rest`) by hand. The parallel-array re-slicing is repeated **5 times** (lines 669–670, 676–677, 729–730, 751–753, 798–803, 815–820) — extremely error-prone: any future filter that forgets one of the 6 arrays (or the auxiliary `u_scene/v_scene/z_cam/d_per_gauss` carried in some blocks but not others) silently desyncs the gaussian attributes. High-risk duplication on the hottest live function.

- **Asymmetric early-return shape (medium).** `reproject_anysplat_to_scene` returns the full 6-key dict on success but a 1-key `{"xyz": empty}` on every empty path (:680, :739, :756). Callers must special-case the empty dict. Misleading contract for a function documented as returning a fixed key set.

- **`stride` param of `icp_refine_scene_c2w` is effectively a dead config knob from the caller (low).** The live caller (pipeline_base:3322) passes `sensor_depth_m, scene_c2w, scene_intr, target_xyz_gpu, max_iters, max_dist_m, target_voxel_m` but **not** `stride` or `min_pts` — both keep their defaults (4 / 1000). Not dead in the function (used at :558, :545/566), but never tunable from the live path. Minor.

- **Duplicated quaternion math across the repo (low, naming/dup).** `quat_wxyz_to_rotmat`/`rotmat_to_quat_wxyz` here, `_quat_wxyz_to_rotmat_np`/`_rotmat_to_quat_wxyz_np` in `viser_direct.py:61/76`, and `_rotmat_to_quat_wxyz` in `scripts/dump_viser_pt_to_ply.py:37` are three independent implementations of the same transforms. Not this module's bug, but a maintenance hazard noted for the synthesis.

- **Swallowed exceptions (low, mostly defensible).** `close()` (:252–259) swallows all exceptions during teardown (acceptable for cleanup). `adopt()` returns `None` on bare `except Exception` (:138, :154) — masks the *reason* a worker couldn't be adopted (stale json vs IO error vs json parse), making "why did it spawn fresh / slow path" hard to diagnose on a live run. The `_pid_is_anysplat_worker` `except OSError` (:273) is correct.

- **Misleading `.npz` extension (low, worker-side contract).** `inference`/`run_anysplat_subprocess` accept `output_npz: Path` and the file is named `*.npz`, but the live caller reads it with `pickle.load` (pipeline_base:3393), i.e. it's a pickle, not a numpy `.npz`. The name lies about the format.

- **Two parallel projection conventions for "is this gaussian in the component".** `reproject_anysplat_to_scene` filters via pred-pixel un-crop + scene-resolution mask index (:742–756), while the DEAD `filter_gaussians_by_component_mask` projects `means_world` through the camera. Having both (one live, one dead) invites a future maintainer to "fix" the wrong one.

### Thread-safety (LIVE — extra adversarial)
- **`_alive()` TOCTOU in `inference` (low).** `inference` checks `_alive()` at :193 then writes to `self._send_f` at :199; in adopt mode the worker can die between the check and the write (broken-FIFO write → `BrokenPipeError`/`SIGPIPE`). It's caught one level up in `_anysplat_bg_run` (pipeline_base:3382 `except Exception`), so it degrades to a skipped FF call rather than a crash. Acceptable but worth noting since the worker is a detached process that can vanish independently.
- **`PersistentAnysplatWorker` instance is single-consumer by construction.** Only the FF bg thread calls `inference` (pipeline_base:3374), and dispatch is gated by `_anysplat_slot_lock` (acquired non-blocking at pipeline_base:2509, held for the whole bg run). So even though the pipe handles (`_send_f`/`_recv_f`) are not internally locked, there is never a second concurrent `inference`. `close()` runs at teardown (atexit/cleanup), which could in principle race a final in-flight `inference` on the bg thread, but the FF thread is daemon/joined-by-process-exit; low risk. The class itself has NO internal lock — its thread-safety is entirely a property of the single-slot dispatch discipline in the caller. If a future caller ever issues `inference` from two threads (or calls `close()` while `inference` is mid-readline), the pipe read/write would interleave and corrupt the JSON stream. Document this single-consumer requirement on the class (currently undocumented).
- **`icp_refine_scene_c2w` / `reproject_anysplat_to_scene` are pure (no shared mutable state).** They read only their args; the GPU snapshot of scene means is taken under `_viser_lock_ctx()` in the caller before ICP. No race introduced by this module. The model-lock for the cull+insert is held by the caller, not here. Good separation.
