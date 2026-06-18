# Code Audit — persistence / package-init / trainer / keyframe-filter

Scope: `dynamic_gs/persistence/post_fusion_cache.py`, `dynamic_gs/__init__.py`,
`dynamic_gs/dynamic_gs_trainer.py`, `dynamic_gs/utils/keyframe_filter.py`.

Repo root: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`. Grep base:
`grep -rn <name> dynamic_gs scripts --include=*.py`.

---

## 1) FUNCTION / CLASS MAP

### `dynamic_gs/persistence/post_fusion_cache.py`

- **`PostFusionLoadResult` (@dataclass: success/num_points/error)** — post_fusion_cache.py:35-39 — result struct for a warm-cache load attempt. — **Refs: used.** Re-exported in `persistence/__init__.py:2,8`; returned/constructed within this file (116,126,147,153); the load result is consumed at `dynamic_gs_pipeline_base.py:566` (`result.success`).

- **`save_post_fusion_state(model, cache_path) -> bool`** — post_fusion_cache.py:45 — `torch.save({model_state_dict, num_points})` to disk; returns True on success. — **Refs: used.** `static_gs_pipeline.py:268`, `static_gs_preseg_pipeline.py:354`, `dynamic_gs_pipeline_base.py:678` (final-snapshot dump); re-exported `persistence/__init__.py:4,10`.

- **`_resolve_cache_path(cache_path) -> Path`** — post_fusion_cache.py:72 — back-compat: if `cache_path` (`static_state.pt`) is absent but legacy `post_fusion_state.pt` sits beside it, return the legacy path. — **Refs: used.** Internal only: called at post_fusion_cache.py:108 (inside `load_post_fusion_state`).

- **`load_post_fusion_state(model, cache_path, device) -> PostFusionLoadResult`** — post_fusion_cache.py:92 — reallocate 6 `gauss_params` Parameters at saved N, `load_state_dict(strict=False)`, re-bind the means-grad hook. — **Refs: used.** `dynamic_gs_pipeline_base.py:566`; re-exported `persistence/__init__.py:3,9`.

- **`_LEGACY_CACHE_NAME = "post_fusion_state.pt"`** — post_fusion_cache.py:42 — module constant. — **Refs: used.** post_fusion_cache.py:82.

### `dynamic_gs/__init__.py`

- **`_suppress_nerfstudio_output_writes()`** — __init__.py:24 — monkeypatches 3 nerfstudio write-sites (`ExperimentConfig.save_config`, `Trainer.train`'s dataparser dump, `writer.setup_event_writer` tb branch) to suppress all `outputs/` disk writes. **INVARIANT-PROTECTED** (Design Invariant #5). — **Refs: used.** Self-invoked at import time, __init__.py:125. (Referenced by name in `dynamic_gs_config.py:72,185,227` comments.)

- **`_train_no_dataparser_dump(self)`** — __init__.py:47 (nested in the suppress fn) — wrapper installed over `Trainer.train` that stubs `save_dataparser_transform` for the call duration. — **Refs: used** (installed at __init__.py:65). Closure, not module-level.

- **`_setup_event_writer_no_tb(...)`** — __init__.py:80 (nested) — wrapper over `writer.setup_event_writer` forcing tensorboard=False. — **Refs: used** (installed at __init__.py:90). Closure.

- **`_ensure_ninja_on_path()`** — __init__.py:95 — prepend the pip `ninja` package BIN_DIR to PATH if the `ninja` executable isn't resolvable (gsplat JIT needs it). — **Refs: used.** Self-invoked at import time, __init__.py:126.

- **`__getattr__(name)`** — __init__.py:140 — lazy attribute shim re-exporting config/model/datamanager classes (`DynamicGS`, `DynamicGSLive`, `StaticGS`, `DynamicGSModel`, ...). — **Refs: used** (Python module protocol; backs `from dynamic_gs import DynamicGSModel`-style imports). Entry-point-ish — leave.

### `dynamic_gs/dynamic_gs_trainer.py`

- **`NoSaveTrainer(Trainer)`** — dynamic_gs_trainer.py:10 — Trainer with no-op checkpoint, viser-keep-alive `_train_complete_viewer`, and a dynamic-phase fast-path `train_iteration`. — **Refs: used.** `_target` in `dynamic_gs_config.py:166,208`; subclassed by `StaticGSTrainer` in `static_gs_pipeline.py:400`.

- **`NoSaveTrainer.save_checkpoint(step)`** — dynamic_gs_trainer.py:31 — no-op override (suppress multi-GB `outputs/` snapshots). — **Refs: used** (nerfstudio Trainer callback; entry-point — leave).

- **`NoSaveTrainer._train_complete_viewer()`** — dynamic_gs_trainer.py:34 — after the train loop, delegate to the pipeline's `block_until_viser_shutdown` hook to keep the viser-direct scene interactive; falls back to stock. — **Refs: used** (nerfstudio Trainer override, fires post-loop). Hook provider: `dynamic_gs_pipeline_recorded.py:135` `block_until_viser_shutdown`.

- **`NoSaveTrainer.train_iteration(step)`** — dynamic_gs_trainer.py:60 — dynamic-phase short-circuit: when `pipeline.current_phase == "dynamic"`, fire `get_train_loss_dict` (tracker tick side-effect) and skip backward/optimizer/scheduler; else call `super()`. — **Refs: used** (nerfstudio Trainer override).

### `dynamic_gs/utils/keyframe_filter.py`

- **`DynamicKeyframeFilter(translation_thresh_m, rotation_thresh_deg)`** — keyframe_filter.py:33 — greedy ORB-SLAM-style stateful keyframe dedup (Euclidean trans + SO(3) geodesic rot gate). — **Refs: NO INSTANTIATION FOUND.** Imported+re-exported (`utils/__init__.py:22,56`), named only in *docstrings* (`dynamic_gs_pipeline_base.py:25,94`) that explicitly say it was DROPPED. `grep -rn "DynamicKeyframeFilter(" dynamic_gs scripts` → 0 hits. See §2.

- **`DynamicKeyframeFilter.num_kept` (property)** — keyframe_filter.py:43 — count of kept keyframes. — **Refs: NO REFS FOUND** on this class (`.num_kept` only matches the unrelated inlined `_KeyframeFilter` in `live_ros_publisher.py:349`).

- **`DynamicKeyframeFilter.reset()`** — keyframe_filter.py:46 — clear kept state. — **Refs: NO REFS FOUND** on this class (the `.reset()` hits at `xfeat_motion.py:485` / `live_ros_publisher.py:1075` are other objects).

- **`DynamicKeyframeFilter.accept(c2w_3x4) -> bool`** — keyframe_filter.py:50 — per-frame accept/reject; first frame always accepted. — **Refs: internal only** (called by `bulk_filter` at :101). No external caller (`live_ros_publisher.py:990` calls its OWN inlined `_KeyframeFilter.accept`).

- **`DynamicKeyframeFilter.bulk_filter(c2w_Nx3x4) -> List[int]`** — keyframe_filter.py:86 — loop `accept` over a pose stack, return kept indices. — **Refs: NO REFS FOUND.** `grep -rn "\.bulk_filter(" dynamic_gs scripts` → 0 hits.

---

## 2) DEAD-CODE CANDIDATES

> Entry points / invariant-protected symbols excluded per instructions
> (`_suppress_nerfstudio_output_writes`, `_ensure_ninja_on_path`, `__getattr__`,
> all `NoSaveTrainer` overrides = nerfstudio Trainer callbacks, the identity buffers).

- **`DynamicKeyframeFilter` (entire class + `accept`/`bulk_filter`/`num_kept`/`reset`)** — keyframe_filter.py:33 — **HIGH confidence dead.**
  Evidence: `grep -rn "DynamicKeyframeFilter(" dynamic_gs scripts --include=*.py` → **0** instantiations. `grep -rn "\.bulk_filter(" …` → **0**. The only references are the import/re-export in `utils/__init__.py:22,56` and two docstrings in `dynamic_gs_pipeline_base.py:25,94` that *state it was dropped* ("`DynamicKeyframeFilter` on the recorded path … every dataset frame is fed to the tracker"). The live path uses a separately-defined, inlined `_KeyframeFilter` in `live_ros_publisher.py:335` (deliberately separate — that subprocess avoids importing `dynamic_gs.utils`). So this file is an orphaned module kept alive only by its own `__init__` re-export. `accept` is reachable only from `bulk_filter` (itself dead); `num_kept`/`reset` have no callers at all.
  Caveat (why not "delete now"): it's a public-ish util export and a clean, self-contained implementation; the matching live logic lives elsewhere. Flagging as dead, not recommending deletion blindly — confirm no out-of-tree/notebook import before removal.

No other dead symbols in these four files. `_resolve_cache_path` (internal, 1 caller), `_LEGACY_CACHE_NAME` (1 caller), and the nested closures in `__init__.py` are all live.

---

## 3) DATA-LIFECYCLE — persistent state

State unit: the `.pt` warm-cache (`static_scene/static_state.pt`, legacy `post_fusion_state.pt`). Holds `{"model_state_dict", "num_points"}`. The `model_state_dict` carries the 6 `gauss_params` AND the 4 identity buffers (`object_flags`/`object_instance_ids`/`sam3d_init_target_flags`/`inserted_flags`) — all invariant-protected.

**SAVE path** (`save_post_fusion_state`, post_fusion_cache.py:45)
- Writers: `static_gs_pipeline.py:268` (after Phase 0b), `static_gs_preseg_pipeline.py:354` (AFTER_TRAIN cb), `dynamic_gs_pipeline_base.py:678` (final post-FF snapshot at exit).
- Writes `model.state_dict()` + `int(model.num_points)`. No config tag stored (documented caveat, post_fusion_cache.py:18-21).
- Exception-swallowing: `except Exception` → logs + returns False (post_fusion_cache.py:67-69). Acceptable for a best-effort snapshot, but a real save failure is reduced to a one-line log; callers at static_gs_pipeline.py:268 / preseg:354 don't appear to hard-fail on `ok==False` — a silent "no warm cache produced" is possible (LOW; verify the caller's handling).

**LOAD path** (`load_post_fusion_state`, post_fusion_cache.py:92)
- Reader: `dynamic_gs_pipeline_base.py:566` (only caller), pre-checked by `_load_warm_cache_or_die` (base.py:542) which raises FileNotFoundError if neither name exists.
- Reallocates each of the 6 `gauss_params` to a fresh `nn.Parameter` at saved N (post_fusion_cache.py:121-132), then `model.load_state_dict(state_dict, strict=False)` (:138). The model's own `load_state_dict` override resizes the 4 identity buffers to match N before copying (documented :134-137).
- Re-binds `_mask_means_grad` hook to the new means Parameter (:142-143) — correct: without this the dynamic-phase means-zeroing hook (invariant #4) would be bound to the discarded pre-resize tensor and silently never fire. Good catch in the code.

**Lifecycle observations / risks:**
1. **`strict=False` masks shape/key drift (MEDIUM).** post_fusion_cache.py:138. The docstring (:18-21) admits the snapshot is NOT config-tagged: if `sh_degree` / background / camera-opt mode differ between save and load, `features_rest`/etc. won't match. With `strict=False`, a *missing* key is silently skipped rather than erroring — a model param could keep its cold-start (SfM-seed) value instead of the saved one, with no warning. Only the 6 `gauss_params` are explicitly re-checked (:121-126 returns failure on a missing `gauss_params.*`); everything else (including the 4 identity buffers) rides on `strict=False` and is NOT verified post-load. An identity buffer that fails to copy (wrong dtype/shape) would desync silently. No post-load assertion that `model.num_points == target_n` or that `object_flags.shape[0] == target_n`.
2. **Two independent legacy-fallback implementations (LOW, duplication/desync).** `_resolve_cache_path` (post_fusion_cache.py:72) AND `_load_warm_cache_or_die` (base.py:548) both implement the `static_state.pt → post_fusion_state.pt` fallback. They agree today, but the doc on `_resolve_cache_path` (:73-78) notes the loader's fallback exists "but the caller's pre-flight check fires first" — two copies of the same rule that must be kept in lockstep.
3. **No explicit free of the old gauss_params Parameters.** post_fusion_cache.py:129 rebinds `model.gauss_params[name]` to a new Parameter; the old tensors are dropped to GC. The new tensor is built `.to(means_device)` then `.clone()` (:128-130) — a transient extra copy on GPU during load (≈2× the largest param). For ~1.3M-Gaussian scenes this is a momentary VRAM spike, not a leak (LOW).
4. **`num_points` is the only integrity signal.** A truncated/corrupt `.pt` is caught by the `torch.load` try/except (:113) → returns `success=False` → base.py raises RuntimeError. No checksum, but failure is loud. Fine.

No SHM, image-cache, or process-handle lifecycle is touched by these four files. (The `_KeyframeFilter` SHM/recorder lives in `live_ros_publisher.py`, out of scope.)

**Trainer state (`dynamic_gs_trainer.py`):**
- `save_checkpoint` is a hard no-op (:31) — by design, suppresses `outputs/` writes (invariant #5). The dynamic final-snapshot is written via `_save_final_snapshot_if_enabled` (base.py:662), NOT the trainer checkpoint path. Consistent.
- `_train_complete_viewer` (:34) swallows all exceptions from `block_until_viser_shutdown` AND from the stock fallback (:51-58). A failing keep-alive hook is silently downgraded; see §4.

---

## 4) DESIGN SMELLS

- **Swallowed exceptions, monkeypatch installers (LOW, acceptable).** __init__.py:39,66,91 each wrap a patch install in bare `except Exception: pass`. Defensible (a missing nerfstudio symbol shouldn't break import), but if upstream renames `setup_event_writer`/`save_config`, the suppression silently stops working and `outputs/` writes resume with no warning — a silent regression of invariant #5. Consider logging the patch-miss.

- **Swallowed exceptions in `_train_complete_viewer` (LOW).** dynamic_gs_trainer.py:51-53,55-58 — both the keep-alive hook and the stock fallback are wrapped in bare `except Exception: pass`. A genuine bug in the viser-shutdown block becomes an invisible "viewer just closed". `record(...)` of the load timing (base.py:567-572) is similarly `except Exception: pass`.

- **Stale / misleading docstrings reference dropped symbols (LOW, naming/doc).** `dynamic_gs_pipeline_base.py:25` and `:94` describe `DynamicKeyframeFilter` as a "dropped legacy field" while the class is still imported+exported in `utils/__init__.py:22,56`. The doc says it's gone; the symbol isn't. Misleads a reader grepping for it. (The class itself is the §2 dead candidate.)

- **`post_fusion` naming kept after the rename to `static_state.pt` (LOW, intentional but confusing).** The function names (`save_post_fusion_state`/`load_post_fusion_state`/`PostFusionLoadResult`) and `_LEGACY_CACHE_NAME` all say "post_fusion", but the default file is now `static_state.pt` and the saver is also used for the *dynamic* final snapshot (base.py:678) which is post-feedforward, not post-fusion. The module docstring (:11-12) explicitly keeps the prefix "to preserve the public API used by external dump/merge scripts" — so this is deliberate, but a reader sees three different names (`post_fusion`, `static_state`, `static-cache` log prefix) for one artifact.

- **`StaticGSTrainer` "Same as NoSaveTrainer today" (LOW).** static_gs_pipeline.py:400-401 — a subclass that the comment says is identical-plus-early-stop. The `train_iteration` dynamic fast-path in the base is dead for static (gated on `current_phase=="dynamic"`, static is never that phase) — harmless inheritance, but the base `train_iteration` comment (dynamic_gs_trainer.py:66-68) notes the gate "used to also require disable_dynamic_optimization … leaving the fast path dead until now," confirming this code path has churned and is worth a re-read for correctness on the static side.

- **No god functions / heavy param-threading in these files.** `load_post_fusion_state` is the longest at ~60 lines and is cohesive (reallocate → load → rebind hook).

- **No dead config fields declared on a *Config in these files.** The `enable_dynamic_keyframe_filter` / `dynamic_keyframe_*` names appear ONLY in the base.py:94 docstring as "dropped" — confirmed NOT declared on any `*Config` (`grep -rn "enable_dynamic_keyframe_filter\s*[:=]" dynamic_gs` → 0). So they're not dead fields, just dead doc references.
