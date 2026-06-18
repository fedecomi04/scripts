# Code Audit — `dynamic_gs/utils/live_session.py`

Module: the bootstrap-time interactive live capture flow (SAM3/FastSAM retry loop → SAM3D → TSDF seed). Runs **entirely on the main thread before nerfstudio's pipeline `__init__`**, spawned by `bootstrap_live.sh` stage 1 via an inline `from dynamic_gs.utils.live_session import run_live_capture_session`.

Grep base for "callers" = `grep -rn "\bNAME\b" dynamic_gs scripts --include=*.py` minus the definition in this file. `scripts/diag_validate_fix.py` exercises two helpers by **textual extraction** (`src.index("def …")` + `exec`), not by import — counted as a non-runtime test caller.

---

## 1) FUNCTION / CLASS MAP

- `_has_complete_recording_cache(live_root: Path) -> bool` — live_session.py:57 — Warm-path probe: returns True iff a prior session left transforms.json + seed PLY + SAM3 results JSON + ≥1 valid SAM3D PLY/pose pair, so capture+SAM3+SAM3D can be skipped. **1 caller** — internal (line 536).
- `pause_gazebo_physics(sub) -> bool` — live_session.py:107 — Pauses Gazebo via the publisher subprocess control pipe; sets module flag `_GAZEBO_PHYSICS_PAUSED`/`_PAUSE_SUB`. **1 caller** — internal (line 912). (The phase0 grep hit is `LiveShmSubscriber.pause_gazebo_physics`, a different method in live_shm_reader.py:547.)
- `unpause_gazebo_physics(sub) -> bool` — live_session.py:125 — Idempotent unpause; clears the module flag. **2 internal callers** (lines 145 inside `_atexit_unpause` calls `_PAUSE_SUB.unpause_gazebo_physics()` — actually the *subscriber* method; the module function itself called at line 1185). External hit live_shm_reader.py:554 is the subscriber method, not this fn.
- `_atexit_unpause() -> None` — live_session.py:137 — atexit safety net: unpauses Gazebo if the process dies while paused. **Entry point** — registered via `atexit.register(_atexit_unpause)` at line 150. Not dead.
- `_wipe_live_root() -> None` — live_session.py:175 — Wipes LIVE_ROOT and recreates static/dynamic skeletons. **NO REFS FOUND** (zero internal calls, zero external). The actual wipe is delegated to `LiveShmSubscriber(wipe_live_root=True)` (line 567). The same-named function in live_ros_publisher.py:1383 takes a `live_root` arg and is unrelated. → DEAD.
- `_save_anchor_for_sam3(anchor, debug_dir) -> Path` — live_session.py:182 — Writes the gripper-blacked anchor RGB (`static0_rgb.png`) that SAM3/FastSAM segments. **2 internal callers** (lines 318 doc-ref in comment? actually a call inside `_write_anchor_ref` mirrors the logic; grep line 318 is the comment "mirrors `_save_anchor_for_sam3`" — the real call is line 764).
- `_save_anchor_intrinsics_and_depth(anchor, intrinsics, artifact_dir) -> tuple[Path,Path]` — live_session.py:207 — Writes `static0_full_depth_meters.tiff` (float32 m) + `static0_full_intrinsics.json` that SAM3D needs for the metric pointmap. **1 caller** — internal (line 765).
- `_append_anchor_as_static_keyframe(anchor, intrinsics, static_dir) -> Optional[str]` — live_session.py:238 — Frame-consistency fix: appends the segmented anchor as the FINAL static keyframe so the dataparser sorts it to `cached_train[-1]` (the frame the mask/SAM3D were built on). **1 runtime caller** (internal line 893) + test extraction in scripts/diag_validate_fix.py:93,98 + doc-ref phase0.py:449.
- `_write_anchor_ref(anchor, sam3_objects, intrinsics, static_dir) -> Path` — live_session.py:301 — Writes the canonical `<static>/anchor_ref/` folder (rgb/mask_NN/depth/intrinsics/c2w/overlay/manifest) — the EXACT geometry SAM3D used; Phase-0 reads from here. **1 caller** (internal line 904) + doc-refs phase0.py:254,819.
- `_prompt_user(prompt_text: str) -> str` — live_session.py:353 — Blocking `input()` with a headless EOFError holdoff (`AUTONOMOUS_PROMPT_HOLDOFF_S`). **1 caller** (internal line 749) + test-extraction boundary marker in diag_validate_fix.py:94.
- `_spawn_sam3d_in_thread(anchor_rgb_path, sam3_objects, artifact_dir, debug_dir, depth_path, intrinsics_path) -> tuple[Thread,dict]` — live_session.py:382 — Subprocess-fallback path: runs `run_sam3d_multi_object_subprocess` on a daemon thread, mutating a `result_slot` dict. **1 caller** — internal (line 1029, only in the `sam_worker is None` branch).
- `_register_seed_ply_path(static_dir) -> None` — live_session.py:427 — Atomically writes `ply_file_path` into transforms.json (Invariant #1: Splatfacto inits from the TSDF seed). **1 caller** — internal (line 1142, CPU-TSDF fallback branch only).
- `_seed_dynamic_scene_stub(static_dir, dynamic_dir) -> None` — live_session.py:449 — Symlinks the first static frame into `dynamic_scene/` to satisfy the dataparser's non-empty check (live never reads it). **1 caller** — internal (line 1187).
- `run_live_capture_session(sam3_prompt_text: Optional[str]=None) -> Path` — live_session.py:495 — The public entry: drives the entire pre-training session and returns LIVE_ROOT. **Entry point** — `bootstrap_live.sh:163-164` imports + calls it. ~700-line god function (see smells).

Module-level state: `INIT_CLOUD_NAME` (54), `_GAZEBO_PHYSICS_PAUSED`/`_PAUSE_SUB` (103-104), `DEFAULT_SAM3_PROMPT`/`SAM3_CONDA_ENV`/`SEGMENTATION_BACKEND` + `SAM3_CANDIDATE_*`/`SAM3_CONFIDENCE_THRESHOLD`/`SAM3_MIN_SCORE` (154-165).

---

## 2) DEAD-CODE CANDIDATES

- **`_wipe_live_root()` — live_session.py:175 — HIGH confidence.** Zero internal callers, zero external refs (`grep -rn "\b_wipe_live_root\b"` returns only the def here + an unrelated same-named function in live_ros_publisher.py:1383). The module's own docstring step 1 ("Wipe LIVE_ROOT") is performed by `LiveShmSubscriber(wipe_live_root=True)` at line 567, not by this function. Safe to delete. (Not invariant-protected; not an entry point, callback, or monkeypatch target.)

No other zero-ref symbols. `_atexit_unpause` is an `atexit.register` entry point. All other helpers have exactly one internal caller. `run_live_capture_session` is the bootstrap entry point.

---

## 3) DATA-LIFECYCLE

### SHM / ROS subscriber (`LiveShmSubscriber` = `sub`)
- **Created twice on mutually exclusive branches:** warm path `sub = LiveShmSubscriber(wipe_live_root=False)` (549) then returns LIVE_ROOT; cold path `sub = LiveShmSubscriber(wipe_live_root=True)` (567). Only one runs per call → no double-spawn.
- **No explicit `sub` teardown in the cold path.** `start_recording` (587) and `stop_recording` (751/871) bracket recording, but the subscriber + publisher subprocess + SHM segment are **never closed** by `run_live_capture_session`. This is intentional handoff — the live pipeline (`dynamic_gs_pipeline_live.py`) adopts the running publisher/SHM after `run_live_capture_session` returns LIVE_ROOT. **Risk:** on the **SAM3-abort** path (line 750-758) and on the **warm path** (549), the function `raise`s/returns with the publisher still running; if the caller does not adopt it (e.g. abort), the publisher subprocess + SHM leak until process exit. Worth a flag — see lifecycle issues.
- `_PAUSE_SUB` module global holds a reference to `sub` for the atexit hook (118). Never cleared on normal exit; benign (process-lifetime).

### GPU / SAM worker (`sam_worker: SamWorkerClient`)
- **Created** at 650. Loads FastSAM/SAM3 on a bg thread `_bg_load_sam3` (653-687); optionally **preloads SAM3D** (671) during capture (`_preload_sam3d`).
- **Closed up to THREE times**, each guarded by `is not None` / try-except:
  1. SAM3-abort: 754 (`sam_worker.close()`), then not nulled (but function raises immediately).
  2. Deferred-TSDF early close: 1093-1099 (`sam_worker.close()` then `sam_worker = None`) — frees VRAM before the GPU TSDF subprocess.
  3. `finally`: 1175-1179 (`sam_worker.close()`) — no-op safety net once nulled at #2.
  - In the **non-deferred** path #2 is skipped, so the `finally` close (1175) is the real one. No double-free crash (close is try-wrapped + None-guarded), but the close-then-null discipline is asymmetric across branches (abort path closes without nulling → if any later code ran it could double-close; here it raises, so safe).
- **Model load/unload pairing:** FastSAM/SAM3 loaded (657-659), unloaded at 932-937 before SAM3D. SAM3D loaded (preload 671 OR at-Enter 952), unloaded at 1025 after infer. **Desync risk:** if `unload_fastsam`/`unload_sam3` (932) raises (caught, line 938) but the model stays resident, the subsequent `load_sam3d` (952) can OOM — handled only by the outer SAM3D try/except (1046) → `sam3d_results = [{}…]` (all-failed), capture continues with no 3D objects. Silent functional degradation, not a crash.
- **CUDA cache:** `gc.collect()` + `torch.cuda.empty_cache()` called at 772-774 (before SAM3) and 923-925 (before SAM3D) — frees the parent's reservation. Correct.

### SAM3D subprocess thread (fallback only, `sam_worker is None`)
- `_spawn_sam3d_in_thread` (1029) spawns a **daemon** thread; the loop 1038-1042 joins it. On exception inside, `result_slot["error"]` is re-raised (1043-1044). Daemon → if the process dies mid-run the thread is abandoned but the SAM3D *subprocess* it launched is a child process and may outlive (not reaped here). Minor leak on crash, fallback path only.

### TSDF / fusion (`ConcurrentFusionRunner`)
- Constructed at 597 (allocates zero GPU). `.start()` only if NOT deferred (607); deferred is the default → worker never runs.
- `_finalize_safe` (712) is idempotent via `_finalize_done["value"]`; called on abort (757), happy-path (1161), and `finally` (1182). In the deferred default path, `stop_and_finalize()` is a no-op (worker never started) and the seed is built by the **GPU subprocess** (1120) or CPU fallback (1141). So in the default flow `_finalize_safe` does nothing useful but still runs in `finally` — harmless.
- **GPU OOM isolation:** the deferred GPU TSDF runs as a **subprocess** (1120-1123) specifically because an Open3D OOM poisons the CUDA cache + aborts at teardown; subprocess contains the abort. Good. CPU `build_tsdf_seed` (1141) + naive `build_static_init_pointcloud` (1148) are nested fallbacks.

### File / persisted state (the 4 identity buffers are NOT touched here)
- This module writes only the dataset on disk: `static0_rgb.png`, depth/intrinsics sidecars, anchor keyframe (rgb/depth/mask + transforms.json append), `anchor_ref/`, the seed PLY, `live_sam3_timings.json`, and `timing_report_capture.txt`. **None** of `object_flags / object_instance_ids / sam3d_init_target_flags / inserted_flags` are read/written here (invariant-protected; they live in the model + persistence layer, populated later by Phase-0b and the dynamic pipeline). No `.pt` warm-cache is written by this module — it only *detects* `static_state.pt`/`post_fusion_state.pt` existence for the warm-path tier message (538-543).

### Format / convention mismatch hazards (data-lifecycle)
- **Depth convention split (documented, latent):** `_save_anchor_intrinsics_and_depth` writes `static0_full_depth_meters.tiff` as **float32 metres** (214) while `_append_anchor_as_static_keyframe`/`_write_anchor_ref` write `depth.tiff`/`{stem}.tiff` as **uint16 mm** (285, 326). SAM3D reads the float-m file with `depth_scale=1.0` (979). Any future reader that assumes the dataset uint16-mm convention on the float-m sidecar is off by 1000× — the module's own PROBLEM comment (232-235) flags this.
- **Transforms.json frame-count desync:** `_append_anchor_as_static_keyframe` derives the new stem index from `len(frames)` (280), but the recorder's numbering is owned by the publisher. If the recorder ever skips/duplicates an index the appended anchor could collide; relies on the publisher's monotone 5-digit pad. The TSDF seed is built AFTER the append (so it includes the anchor), and `transforms.json` is rebuilt per recording (no accumulation) — currently consistent.

---

## 4) DESIGN SMELLS

- **God function — `run_live_capture_session` (495-1203, ~700 lines).** Single linear procedure holding: warm-path short-circuit, subscriber spawn, fusion-runner arm, SAM worker spawn + bg load thread, SAM3 retry loop, SAM3D (worker path + subprocess-thread fallback, each with crop + full-frame sub-fallback), eager-AnySplat spawn (TWICE, 1063 and 1151), deferred/non-deferred TSDF (3-level nested fallback), and timing-report render. Deeply nested try/except/finally. Highest-risk surface in the live path. Recommend extracting at least: `_run_segmentation_loop`, `_run_sam3d`, `_build_seed`.

- **Duplicated gripper-blackout logic.** `_save_anchor_for_sam3` (189-197) and `_write_anchor_ref` (319-324) inline the identical "BGR→RGB, resize keep-mask, zero outside keep" block. The `_write_anchor_ref` docstring even says "mirrors `_save_anchor_for_sam3`". Should call the helper.

- **Duplicated uint16-mm depth write.** Lines 285 and 326 repeat `np.clip(anchor.depth_m*1000, 0, 65535).astype(np.uint16)` (and again the publisher does it). Trivial helper candidate.

- **Eager-AnySplat spawn duplicated** across the non-deferred (1063-1072) and deferred-post-TSDF (1151-1158) branches — same `spawn_detached_anysplat_worker(LIVE_ROOT/".anysplat_worker")` + try/except. The `not _defer_tsdf` guard at 1063 means in the default deferred flow the first block is skipped and only 1151 fires; correct but the two copies can drift.

- **Misleading naming.** `_prompt_user` / log strings / `S0.1_fastsam_segmentation` / `DEFAULT_SAM3_PROMPT` / `SAM3_CONDA_ENV` / `sam3_objects` / `sam3_duration` / `t_sam3` all say "SAM3" while the **default backend is FastSAM** (`SEGMENTATION_BACKEND` default "fastsam", 159). Almost every "SAM3" identifier actually carries FastSAM data in the default path. High confusion cost for the next reader. (Also docstring step 6 still says "SAM3 subprocess … ~1-2s".)

- **Swallowed exceptions, broad `except Exception` everywhere.** Failed-segmentation save (859), anchor-keyframe append (895), anchor_ref write (906), segmenter unload (938), SAM3D unload (1027), every TSDF fallback (1136/1144/1149), SAM worker closes (755/938/1098/1178), timing-ledger reset/record/render (620/1195). Many are deliberate best-effort, but the SAM3D-failed path (1046-1048) silently yields zero 3D objects and continues to seed-build — capture "succeeds" with nothing to track, surfacing only downstream. The `_tl.reset/record/render` blanket-swallows mean a broken ledger is invisible.

- **`_has_complete_recording_cache` integrity gap (self-documented, 85-92).** Checks file *existence* only, not SAM3-mask-count vs SAM3D-PLY-count. A stale partial cache can pass the warm-path check and trip Phase-0b. Acceptable but a real foot-gun (must `rm -rf` LIVE_ROOT manually).

- **Stale docstring vs code.** Module docstring (1-23) describes the OLD flow ("SAM3 subprocess blocking ~1-2s", "wait for second Enter", "stub frame") — the current flow is single-Enter, FastSAM-default, deferred-TSDF. Misleads readers of the live path.

- **Env-flag sprawl threaded as implicit config.** `DGS_LIVE_DEFER_TSDF`, `DGS_SAM3D_LOAD_DURING_CAPTURE`, `DGS_EAGER_ANYSPLAT`, `DGS_TSDF_VOXEL_M`, `AUTONOMOUS_PROMPT_HOLDOFF_S`, `DGS_SEGMENTATION_BACKEND`, `DGS_SAM3_PROMPT` all read ad-hoc inside the god function with inline defaults — no single config surface; behavior changes silently with the environment.

- **Dead config / module constant:** `SAM3_MIN_SCORE = 0.0` (165) is "disabled" but still threaded into every infer call (796/808/823) as `min_score=0.0` — inert plumbing kept for symmetry; not harmful but a thread-through with no effect in the default path.

### Thread-safety / race notes (live path)
- `run_live_capture_session` is **single-threaded on the main thread**; the concurrency hazards of the dynamic phase (FF bg thread vs tracker vs viser render sharing the model lock) do **not** apply here — this module runs entirely *before* the pipeline + tracker exist. The only background threads are `_bg_load_sam3` (684, daemon) and the SAM3D fallback thread (417, daemon).
- **`_bg_load_sam3` ↔ main race:** the bg thread mutates `_sam3_load_err`/`_sam3d_preload` dicts; the main thread reads them only AFTER `_sam3_load_thread.join()` (782-785) → no torn read. The preload (`load_sam3d` at 671) runs on the bg thread and shares the GPU with nothing concurrent (the operator is sweeping; no trainer yet) → the doc's measured 12.65 GB peak. Safe by sequencing, not by lock.
- **No shared mutable model state with locks here** — all model handoff is via files on disk (LIVE_ROOT) consumed by the next process (`ns-train static-gs` / `dynamic-gs-live`). Cross-process, file-mediated, no in-process lock needed.
