# Code Audit — `dynamic_gs/utils/sam_worker.py`

Persistent SAM3 / Fast-SAM3D / FastSAM worker subprocess (runs in the `sam3_dynamic_gs` conda env) + the in-`dynamic_gs`-env `SamWorkerClient` that spawns and drives it over a JSON-over-stdin/stdout line protocol. This is a LIVE-PATH module (segmentation during capture).

Grep basis for "callers":
`grep -rn "<sym>" /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/{dynamic_gs,scripts} --include=*.py` (excluding self-references in `sam_worker.py`).

---

## 1) FUNCTION / CLASS MAP

### Module-level

- `_load_sibling_module(name) -> module` — sam_worker.py:72 — imports a sibling `.py` (e.g. `sam3_segmentation`, `sam3d`, `fastsam_segmentation`) directly from disk, bypassing `utils/__init__.py` (which would drag in nerfstudio, absent in the sam env); caches in `sys.modules` under `_sam_worker_sib_<name>`. Callers: 4, all internal (`load_sam3d`, `sam3_infer`, `sam3d_infer`, `load_fastsam`). NO EXTERNAL REFS (internal helper).

- `_worker_main() -> int` — sam_worker.py:655 — subprocess entrypoint: reroutes fd-1 for protocol, sends `{"status":"ready"}`, then the request loop dispatching `cmd` to `_SamWorker` handlers. Caller: `_main` (sam_worker.py:1012). NO EXTERNAL REFS (entry point, reached via `--worker`).

- `_parse_args() -> Namespace` — sam_worker.py:1003 — argparse with the single `--worker` flag. Caller: `_main`. NO EXTERNAL REFS.

- `_main() -> int` — sam_worker.py:1009 — top-level CLI; runs `_worker_main` when `--worker`, else errors. Caller: `__main__` block. NO EXTERNAL REFS (entry point).

### `class _SamWorker` — sam_worker.py:97

Lives in the worker subprocess; holds loaded models. Class itself: instantiated only in `_worker_main` (sam_worker.py:672). NO EXTERNAL REFS (subprocess-internal).

- `__init__(self)` — sam_worker.py:104 — null out the 5 model slots, import torch once.
- `_gpu_mem(self) -> dict` — sam_worker.py:115 — torch allocator resident/peak MiB. Internal, ~7 call sites in load/infer responses.
- `_reset_peak(self)` — sam_worker.py:126 — `reset_peak_memory_stats`. Internal, called in load/infer handlers.
- `load_sam3(self,*,confidence_threshold=0.1) -> dict` — sam_worker.py:132 — build SAM3 image model + processor. Reached via the `load_sam3` cmd from client `SamWorkerClient.load_sam3`.
- `unload_sam3(self) -> dict` — sam_worker.py:144 — drop refs + `empty_cache`. Via `unload_sam3` cmd.
- `sam3_infer(self,*,image_path,text_prompt,output_dir,output_stem,...) -> dict` — sam_worker.py:154 — run SAM3, apply area/border/dedup/score filters, write per-object PNG masks + `_sam3_results.json`. Via `sam3_infer` cmd.
- `sam3_infer_raw(self,*,image_path,text_prompt,output_dir,output_stem,min_score=0.0) -> dict` — sam_worker.py:267 — run SAM3, write ALL masks above min_score to `_raw_masks.npz` (no area/border/dedup). Via `sam3_infer_raw` cmd.
- `load_sam3d(self) -> dict` — sam_worker.py:354 — build SAM3D `Inference`, apply hfer/ss/slat/mesh patches + gaussian-only trim; releases partial alloc + re-raises on load failure. Via `load_sam3d` cmd.
- `unload_sam3d(self) -> dict` — sam_worker.py:402 — drop refs + `empty_cache`. Via `unload_sam3d` cmd.
- `sam3d_infer(self,*,render_image_path,object_mask_paths,output_stems,output_dir,...) -> dict` — sam_worker.py:412 — per-mask SAM3D reconstruction with an OOM size-retry ladder; writes PLY/pose/preview/mesh. Via `sam3d_infer` cmd.
- `load_fastsam(self,*,fastsam_weights,clip_model,clip_pretrained) -> dict` — sam_worker.py:582 — build `FastSamTextSegmenter`. Via `load_fastsam` cmd.
- `unload_fastsam(self) -> dict` — sam_worker.py:596 — drop ref + `empty_cache`. Via `unload_fastsam` cmd.
- `fastsam_infer(self,*,image_path,...,imgsz=1024) -> dict` — sam_worker.py:605 — delegate to `FastSamTextSegmenter.infer`. Via `fastsam_infer` cmd.
- `fastsam_infer_raw(self,*,image_path,...,imgsz=1024) -> dict` — sam_worker.py:632 — delegate to `FastSamTextSegmenter.infer_raw`. Via `fastsam_infer_raw` cmd — **NO CLIENT CALLER for this cmd** (see §2).

### `class SamWorkerClient` — sam_worker.py:724

Client in the `dynamic_gs` env. Callers (constructor): `static_gs_preseg_pipeline.py:204`, `live_session.py:650`. Also referenced (imports/type hints/docstrings) in `static_gs_preseg_pipeline.py:61`, `live_session.py:51,639`.

- `__init__(self,conda_env=_DEFAULT_ENV,startup_timeout_s=30.0)` — sam_worker.py:739 — spawn env-python `--worker` with LD_LIBRARY_PATH/PYTHONPATH set, block until `{"status":"ready"}`. 2 callers above.
- `spawn_seconds` (property) — sam_worker.py:790 — startup wall. Callers: `static_gs_preseg_pipeline.py:208`, `live_session.py:651`.
- `_request(self,cmd,*,timeout_s=600.0,**kwargs) -> dict` — sam_worker.py:794 — write one JSON request, poll stdout for one dict-with-`status` response, raise on `error`/dead proc/timeout. Internal; all public methods route through it.
- `load_sam3(self,confidence_threshold=0.1,timeout_s=60.0) -> float` — sam_worker.py:825 — `static_gs_preseg_pipeline.py:216`, `live_session.py:659`.
- `unload_sam3(self,timeout_s=15.0)` — sam_worker.py:830 — `static_gs_preseg_pipeline.py:275`, `live_session.py:936`.
- `sam3_infer(self,*,image_path,...) -> list` — sam_worker.py:833 — `live_session.py:799`.
- `sam3_infer_raw(self,*,image_path,...) -> dict` — sam_worker.py:854 — `preseg_seed.py:451`.
- `load_sam3d(self,timeout_s=60.0) -> float` — sam_worker.py:877 — `live_session.py:670,952`.
- `unload_sam3d(self,timeout_s=15.0)` — sam_worker.py:881 — `live_session.py:1025`.
- `sam3d_infer(self,*,render_image_path,...) -> list` — sam_worker.py:884 — `live_session.py:985,993`.
- `load_fastsam(self,fastsam_weights=...,...) -> float` — sam_worker.py:910 — `live_session.py:657`.
- `unload_fastsam(self,timeout_s=15.0)` — sam_worker.py:924 — `live_session.py:933`.
- `fastsam_infer(self,*,image_path,...) -> list` — sam_worker.py:927 — `live_session.py:787`.
- `fastsam_infer_raw(self,*,image_path,...) -> dict` — sam_worker.py:952 — **NO REFS FOUND** (no `.fastsam_infer_raw` call site anywhere; see §2).
- `close(self)` — sam_worker.py:978 — send `shutdown`, wait 5s, kill on failure. `static_gs_preseg_pipeline.py:232,279`, `live_session.py:754,1095,1177`.
- `__enter__`/`__exit__` — sam_worker.py:991,994 — context-manager support. **NO REFS FOUND** — no `with SamWorkerClient(...)` anywhere; all call sites use explicit `close()`.

---

## 2) DEAD-CODE CANDIDATES

- **`SamWorkerClient.fastsam_infer_raw` (client wrapper)** — sam_worker.py:952 — ref count 0. `grep -rn "\.fastsam_infer_raw" {dynamic_gs,scripts} --include=*.py` (excluding self) = 0 hits. The matching worker handler `_SamWorker.fastsam_infer_raw` (sam_worker.py:632) + its `handler_map` entry (sam_worker.py:701) are therefore also unreachable via any client. Confidence: **high** (the raw-mask path is only exercised for SAM3 via `sam3_infer_raw` from `preseg_seed.py`; FastSAM preseg has no analogous caller). Not an entry point, not invariant-protected.

- **`SamWorkerClient.__enter__` / `__exit__`** — sam_worker.py:991,994 — ref count 0 as a context manager (`with SamWorkerClient` = 0 hits; every site does `client = SamWorkerClient(); ... client.close()`). Confidence: **medium** — trivially correct API surface kept for symmetry; low harm. Reported for completeness, not urgent.

Not flagged as dead (verified live): every `load_*`/`unload_*`/`*_infer` (except the two above), `close`, `spawn_seconds`, `_request`, all `_SamWorker` handlers except `fastsam_infer_raw`. Entry points (`_main`, `_worker_main`, `_parse_args`, `__main__`) excluded by rule. The `sam3d_init_target_flags` / identity-buffer machinery does not appear in this module (no overlap to mis-flag).

---

## 3) DATA-LIFECYCLE

This module owns **GPU model state inside a subprocess** and a **process+pipe handle in the parent**. It does NOT touch the `.pt` warm-cache, SHM, or the 4 identity buffers directly — those live in `persistence/` and the pipelines. State traced here:

### Subprocess + pipes (parent side)
- **Create:** `subprocess.Popen([env_python,"-u",file,"--worker"], stdin=PIPE, stdout=PIPE, stderr=None)` — sam_worker.py:762. Startup blocks reading stdout until `ready` (sam_worker.py:770-788); on timeout it `kill()`s (sam_worker.py:772). On exit-during-startup it raises but does NOT `kill()` (proc already dead — fine).
- **Use:** `_request` writes stdin + readlines stdout (sam_worker.py:798-821).
- **Free:** `close()` sends `shutdown`, `wait(5s)`, else `kill()` (sam_worker.py:978-989). All 5 client sites call `close()`. **LEAK RISK (medium):** `close()` only kills the process; it does NOT close `self._proc.stdin`/`stdout` pipe FDs. After `wait()`/`kill()` Popen's destructor will eventually close them, but on a `TimeoutError`/`RuntimeError` raised mid-session (worker died) callers may drop the client without `close()` and the FDs + zombie linger until GC. CLAUDE.md "Killing ns-train safely" already documents zombie-process pain — same family.
- **DOUBLE-FREE-safe:** `close()` early-returns if `poll() is not None` (sam_worker.py:979). Re-entrant close is a no-op. Good.

### GPU model state (worker side) — load/unload symmetry
- SAM3: `load_sam3` builds (sam_worker.py:139-140); `unload_sam3` nulls + `empty_cache` (sam_worker.py:147-151). Symmetric.
- SAM3D: `load_sam3d` builds + trims (sam_worker.py:362-397); on build failure it `empty_cache` + re-raises so a partial OOM alloc is freed (sam_worker.py:365-374) — explicitly documented. `unload_sam3d` symmetric (sam_worker.py:405-409).
- FastSAM: `load_fastsam`/`unload_fastsam` symmetric (sam_worker.py:591-602).
- **Idempotent loads:** each `load_*` early-returns `already_loaded` if the slot is set (sam_worker.py:133,355,586) — no double-alloc. Good.

### Lifecycle desync / orchestration hazards
- **Worker holds models until explicit unload OR process death.** If the parent forgets an `unload_*` and reuses the worker for a different model, VRAM stacks (SAM3 3.8 + SAM3D 12.0 = 17.7 GB > 15.8 — documented in the header, sam_worker.py:29). The class provides no guard that two heavy models aren't co-resident; correctness depends entirely on the caller (`live_session.py`) sequencing `unload` before the next `load`. **FRAGILE (medium):** `live_session.py:933` unloads fastsam, `:952` loads sam3d, `:1025` unloads sam3d — but there is no `unload_sam3d`/`unload_fastsam` on the early-abort/exception paths beyond `close()` (which kills the whole proc, so VRAM is reclaimed at OS level — acceptable).
- **`close()` does not unload models first** — it relies on process exit to free GPU. Correct (OS reclaims), but means a `close()` mid-load (worker stuck importing) just `kill()`s after 5s; the in-flight CUDA context dies with the process. Fine.

### File outputs (worker writes, parent reads paths back)
- `sam3_infer` writes PNG masks + `<stem>_sam3_results.json` (sam_worker.py:236,246). `sam3_infer_raw` writes `<stem>_raw_masks.npz` (sam_worker.py:343). `sam3d_infer` writes PLY/pose/preview/run_info/mesh (sam_worker.py:523-559). These are passed by **path string** in the JSON response; the parent re-reads from disk — no in-memory tensor crosses the pipe. No shape/format mismatch risk in the IPC itself.
- **NPZ shape contract (`sam3_infer_raw`):** writes `masks (K,H,W) bool, scores (K,) f32, boxes (K,4) f32`; on zero-keep it writes empty `(0,H,W)` etc. (sam_worker.py:336-339). The `H,W` fallback when `masks_bool.ndim != 3` uses `image.height/width` — consumer `preseg_seed.py` must tolerate the empty case. No mismatch observed but the empty-mask branch is the riskiest (untested-looking) path.

### GPU allocator counters (cross-call mutable state)
- `_reset_peak()` is called at the start of `load_sam3`, `load_fastsam`, `fastsam_infer*` but **NOT** at the start of `sam3_infer`/`sam3_infer_raw`/`sam3d_infer`/`load_sam3d` — so the `gpu_peak_mb` reported by those reflects the peak since the last reset (possibly a much earlier load), not since this call. Reporting inconsistency, not a leak. (`load_sam3d` returns `_gpu_mem()` without resetting peak → peak may be a stale higher value.)

---

## 4) DESIGN SMELLS

- **Duplicated SAM3 post-processing logic (high):** `sam3_infer` (sam_worker.py:154-265) re-implements the area/border/dedup/score filter + mask-PNG + results-JSON write that `sam3_segmentation.run_sam3_segmentation` already contains — the docstring at sam_worker.py:167-169 admits this is a deliberate copy to avoid importing argparse/subprocess. The tensor→numpy normalization block (sam_worker.py:187-205) is then duplicated AGAIN verbatim inside `sam3_infer_raw` (sam_worker.py:298-316). Three copies of the same `masks/scores/boxes` cpu/reshape/ndim-fix dance; a private `_normalize_sam3_outputs(output)` helper would collapse it. Drift risk: a filter-semantics change must be made in ≥2 places.

- **`sam3d_infer` is a god method (high):** sam_worker.py:412-578, ~166 lines. It does mask loading, pointmap build, the OOM size-retry ladder, gaussian save, best-effort mesh export (nested try with its own `trimesh` import), pose extraction + validation, run-info text, and per-result dict assembly. The docstring (sam_worker.py:423-426) notes it deliberately forks `sam3d.run_sam3d_multi_object`'s per-mask loop rather than refactoring it — so this is a second copy of that loop's body, with the same drift risk as the SAM3 duplication.

- **Thread-safety: `_request` is NOT lock-guarded (high, latent).** `SamWorkerClient` shares one stdin/stdout pipe pair, and `live_session.py` calls `load_*` from a daemon background thread (`_bg_load_sam3`, live_session.py:653-687) while the main thread later issues `*_infer`. There is no mutex around the `stdin.write`+`stdout.readline` cycle in `_request` (sam_worker.py:798-821). Today this is SAFE ONLY because the main thread does `_sam3_load_thread.join()` (live_session.py:782-783) before any `infer`, serializing access by construction — but nothing in `SamWorkerClient` enforces it. If a future caller (or the CLAUDE.md-described feedforward/tracker/viser three-thread world) ever issues two overlapping requests, the interleaved writes/reads will desync the line protocol (response of cmd A consumed as response of cmd B) with no detection beyond the loose `isinstance(resp,dict) and "status" in resp` filter. Recommend an internal `threading.Lock` around `_request`.

- **Swallowed exceptions / silent skips (medium):**
  - `_request` silently `continue`s on any non-dict / no-`status` / JSON-parse-fail line (sam_worker.py:809-818) — necessary to skip loguru noise, but it ALSO swallows a genuinely malformed real response; the only backstop is `timeout_s` (default 600s for `sam3d_infer`), so a corrupted response = a 10-minute hang rather than a fast error.
  - `load_sam3d` trim failure is caught + printed but ignored (sam_worker.py:395-396) — acceptable (trim is best-effort) but means a silently-untrimmed 12 GB SAM3D can OOM downstream with no signal in the response dict.
  - `sam3d_infer` mesh export failure is caught + printed (sam_worker.py:539-540) — best-effort, fine.

- **Misleading naming (low):** the module/class is named "SAM3 + Fast-SAM3D worker" but now hosts three model families (SAM3, SAM3D, FastSAM); `_DEFAULT_ENV` env var is `DYNAMIC_GS_SAM_WORKER_ENV` while the conda env constant in callers is `SAM3_CONDA_ENV` — two names for the same concept across the boundary.

- **Param fan-out (medium):** the filter knobs (`min_area_ratio`, `max_area_ratio`, `dedup_iou`, `max_objects`, `min_score`, `fastsam_conf`, `fastsam_iou`, `imgsz`) are threaded literally through 4 layers: caller `live_session.py` → `SamWorkerClient.fastsam_infer` → JSON → `_SamWorker.fastsam_infer` → `FastSamTextSegmenter.infer`, each restating the same 8-11 kwargs with duplicated defaults. The defaults are declared independently on the client method AND the worker method AND `FastSamTextSegmenter` — three places that can silently disagree (e.g. `min_score=0.2` default appears on both client sam_worker.py:843 and worker sam_worker.py:163).

- **Per-call hidden imports (low):** `sam3_infer`/`sam3_infer_raw`/`sam3d_infer`/`load_sam3d`/`load_fastsam` each do `import numpy`/`PIL`/`omegaconf`/`_load_sibling_module(...)` inside the method body (e.g. sam_worker.py:170-172, 285, 427-434). `_load_sibling_module` caches, so repeat cost is a dict lookup; `import numpy` inside the hot loop is cheap after first import but is stylistically a per-call allocation that a module-level import (where env allows) would avoid. Not a perf problem at segmentation cadence (not per-tick).

- **No `--worker` arg validation beyond the flag (low):** the worker trusts every JSON request's kwargs verbatim via `h(**args)` (sam_worker.py:709-710); an unexpected key raises `TypeError`, caught and returned as `error` (sam_worker.py:712-714) — acceptable, but the keyword-only signatures mean a single typo'd kwarg from a future caller fails the whole call rather than being ignored.
