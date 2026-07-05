# DETACH_PLAN — cut every `dynamic_gs2` → `dynamic_gs` dependency

Plan of record for making `scripts/dynamic_gs2/` import-independent of the frozen
`scripts/dynamic_gs/` baseline. Grounded in the 2026-07-03 audit (every claim below was
verified against source; line numbers are as of that audit — re-check before editing).

**Verbatim baseline (before any change): 22 load-bearing old-package import sites**
(excluding `verify/ab_isolated_phase0b.py` + `verify/ab_unit_phase0b.py`, which import the
old package *by design* as the A/B reference and stay that way), plus one grep-invisible
subprocess string (`static_pipeline.py:89` runs `python -m dynamic_gs.utils.online_fusion`).

## Decisions (settled)

1. **COPY, never move.** Every one of the 17 shared files is also on an old-baseline
   ns-train method path (verified call sites, e.g. `dynamic_gs_pipeline_base.py:1991`
   imports `xfeat_motion`; `static_gs_pipeline.py:57` pulls `fusion.phase0`). The frozen
   baseline stays byte-untouched. **Divergence policy: bug fixes land only in the gs2
   copies; `dynamic_gs/` stays frozen.**
2. **Destination = flat `dynamic_gs2/`** (+ one `ndp/` subpackage). Flat placement makes
   the self-spawning workers' sibling-loads (`sam_worker.py` `_THIS_FILE.parent/<name>.py`)
   and the publisher's `ros_mask.py`/`frame.py` path-loads resolve with near-zero edits.
   Cost: six `parents[2]→parents[1]` depth edits (enumerated per phase below).
3. **Trim only where the pin-down proved separability** (measured, zero cross-references
   into the dropped remainder): `live_shm_reader` → 117-line extract, `viser_direct` →
   104-line inline, `phase0` → 193-line extract, `active_mask` → 29-line inline,
   `anysplat_decode` → 576-line subset. Everything battle-tested/self-spawning is copied
   whole (`online_fusion`, `depth_filter`, xfeat set, SAM trio, publisher, `sam3d_fusion`+ndp).
4. **Invariant #5 (outputs/ suppression) = DOCUMENT-UNREACHABLE for gs2.** Verified:
   `dynamic_gs2` never constructs `ExperimentConfig`/`Trainer` and never calls
   `writer.setup_event_writer` — the three `dynamic_gs/__init__.py` monkeypatches are
   incidental import side effects, not load-bearing for gs2. **Exception:
   `_ensure_ninja_on_path` IS load-bearing** (empirically proven: gsplat's JIT backend
   fails under the exact `full_live.sh` env without ninja on PATH) → port it into
   `dynamic_gs2/__init__.py` in Phase 1. If gs2 ever adopts nerfstudio Trainer/ns-train,
   the three patches must be ported then — CLAUDE.md must state this condition.
5. **`Ros1Source` is deleted** (Phase 1). Verified dead: execs the nonexistent
   `dynamic_gs2.ros_publisher`, zero importers, zero `--source ros1` users in any script,
   test, or commands.md.
6. **LOC honesty:** `dynamic_gs2` *.py is 11,630 LOC today; this plan adds ~10.8k
   (measured per-phase below) → ~22.4k end state. Still ~20% of the old 28k package, and
   the copies are leaf utilities, not orchestration.

## Standing verify discipline (applies to every phase)

- `PY=/home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python` (absolute env python — avoids
  the conda-run libffi/LD_LIBRARY_PATH gotchas).
- After every whole-file copy: `diff` against the original **must show only the enumerated
  hunks** for that phase (empty diff where no fixups are listed).
- After every phase: full unit tests + a recorded run, comparing `timing_report.txt` and
  the named log lines against the Phase-0 reference **verbatim** (no rounding).
- Live smoke runs: ~20 s with a scheduled stop; publisher cleanup via `pgrep -af` +
  specific PIDs + rosnode cleanup — never a blanket python kill.
- Self-spawning workers run the copied file **in a different conda env** — an import check
  proves nothing; each such phase has an actual spawn test.

---

## Phase 0 — Freeze the A/B reference (no code changes)

1. Settle the uncommitted working-tree changes first: `git status` shows
   `dynamic_gs/utils/live_ros_publisher.py` and `dynamic_gs2/ros_mask.py` modified —
   commit them so every later copy forks a reviewed state.
2. Run `bash /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/dynamic_gs2/full_recorded.sh`
   and `bash /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/dynamic_gs2/warm_recorded.sh`.
3. Stash `<data_dir>/timing_report.txt` + the tracker/FF telemetry lines + seed point count
   + `kept_point_count` lines as the verbatim comparison reference for Phases 1–6.

## Phase 1 — Zero-risk inlines + dead-code removal (+342 LOC, kills 5 import sites) — DONE (2026-07-03)

> Landed. `_ensure_ninja_on_path` ported into `dynamic_gs2/__init__.py`; `NoRefineStrategy`
> inlined into `scene_model.py`; `extract_projected_centers_and_radii` inlined into
> `static_phase0b.py`; `backproject_mask_to_world`+`cull_points_in_front` copied verbatim into
> new `dynamic_gs2/backproject.py` (byte-exact diff vs phase0.py:57-249); viser helpers inlined
> into `dynamic_viz.py`; `Ros1Source` deleted (class + factory branch + `import signal` +
> `_STRIP_ENV_VARS`/`_ros_publisher_env` + docstrings/help strings). Verified: all changed files
> py_compile + full import graph OK; ninja on PATH after package import; `open_source('ros1')`
> rejected; 8 CPU tests + `test_scene_model` + `test_dynamic_track` pass; old-package import count
> dropped 22→17 (excl. ab_). `test_dynamic_feedforward` fails identically on the pristine baseline
> (pre-existing, unrelated). Docs still TODO: this is a plan-file note only; CLAUDE.md/STATUS_LIVE
> updates are deferred to Phase 6 per the plan.

**Original plan below (kept for reference):**


**Changes**
- `dynamic_gs2/__init__.py` (currently empty): port `_ensure_ninja_on_path` verbatim
  (~25 lines from `dynamic_gs/__init__.py`), called at import time. Do NOT port the three
  outputs/-suppression monkeypatches (Decision 4).
- `scene_model.py`: inline the 16-line `NoRefineStrategy` class (whole of
  `dynamic_gs/utils/no_refine_strategy.py`); drop the try-import + `_HAVE_NOREFINE` plumbing.
- `static_phase0b.py`: inline `extract_projected_centers_and_radii` verbatim
  (`active_mask.py:618-646`, 29 lines, zero helper deps); delete the import at `:29`.
  Verify byte-identical: `diff <(sed -n '618,646p' dynamic_gs/utils/active_mask.py) <(inlined block)`.
- NEW `dynamic_gs2/backproject.py`: paste `dynamic_gs/fusion/phase0.py:57-249` verbatim
  (`backproject_mask_to_world` + `cull_points_in_front`, 193 lines — verified they
  reference nothing else in phase0) + ~6 header import lines. Repoint
  `static_phase0b.py:33` and `_diag_insert_pose.py:47` to `from .backproject import …`.
- `dynamic_viz.py`: paste `viser_direct.py:61-164` (104 lines: both quat helpers,
  `_FLIP_YZ`, `_build_camera_from_viser`) above `ViserBridge`, outside the `_HAVE_VISER`
  try; add the function-local `import torch`; delete the import at `:26`.
- **Delete `Ros1Source`**: `adapters_source.py` class body `:221-289`, the `ros1` branch in
  `open_source()` `:381`, `_STRIP_ENV_VARS`+`_ros_publisher_env` `:210-218`, the now-unused
  `import signal` `:19`; drop `"ros1"` from `static_pipeline.py:340` tuple and both
  `--source` help strings (`pipeline.py:605`, `static_pipeline.py:555`); fix the module
  docstrings that name Ros1Source as the live half (`adapters_source.py:5,11`,
  `pipeline.py:5`, `static_pipeline.py:13,325`); update `STATUS.md:120` +
  `PIPELINE_NOTES.md:522,528,559`. Leave the frozen `rewrite_spec/*.md` as-is.

**Verify**
```
cd /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
$PY -m pytest dynamic_gs2/tests -q
$PY -c "from dynamic_gs2.backproject import backproject_mask_to_world, cull_points_in_front; import dynamic_gs2.dynamic_viz, dynamic_gs2.scene_model, dynamic_gs2.static_phase0b; print('ok')"
bash dynamic_gs2/full_recorded.sh   # compare vs Phase-0 reference
```
Expected remaining import sites (17): `adapters_source.py:312`, `static_pipeline.py:414`,
`static_seed.py:35,65`, `static_seed_stream.py:62`, `pipeline.py:209`,
`dynamic_ff_backends.py:184,259`, `dynamic_track.py:50`, `model_loader.py:80,143,213`,
`static_phase0b.py:30`, 4× `verify/_probe_*` online_fusion (+ the `:89` subprocess string).

## Phase 2 — Small self-contained copies (+933 LOC, kills 9 sites + the subprocess string) — DONE (2026-07-05)

> Landed. `depth_filter.py` + `online_fusion.py` copied byte-identical into dynamic_gs2/ (diff empty);
> new `publisher_spawn.py` extracts LIVE_ROOT + launcher constants + `_spawn_publisher` verbatim
> (spawn body byte-identical) with the INTERIM `_PUBLISHER_SCRIPT` -> old-package publisher (retargeted
> in Phase 6). Repointed: static_seed.py (OnlineFusion + adaptive_downsample), static_seed_stream.py,
> pipeline.py (depth_filter, behind try/except-pass — explicit-import verified), static_pipeline.py:89
> subprocess `-m dynamic_gs2.online_fusion` + the LIVE_ROOT sentinel import, adapters_source.py
> (_spawn_publisher+LIVE_ROOT), 4 verify/_probe_* online_fusion imports. Verified: compile OK; explicit
> imports resolve (incl. interim _PUBLISHER_SCRIPT.exists()); 11/11 tests pass; real load-bearing
> old-package sites 17->8. SEMANTIC A/B: `-m dynamic_gs2.online_fusion` vs old on zed_final/static_scene
> both produced 322,343 vertices — md5 differs but the OLD module is ALSO non-deterministic run-to-run
> (Open3D CPU TSDF/ICP fp-ordering; proven by old-vs-old md5 mismatch), so identical vertex count = the
> copy is a true reproduction. NOTE: during this phase found 3 Phase-1 edits had been reverted by an
> intervening git stash/pop (scene_model NoRefineStrategy inline, static_phase0b extract inline +
> backproject repoint) — re-applied and re-verified.

**Original plan below (kept for reference):**


**Changes**
- `cp dynamic_gs/utils/depth_filter.py dynamic_gs2/depth_filter.py` (155, verbatim, zero
  fixups inside). Repoint `pipeline.py:209` → `from .depth_filter import filter_depth_torch`.
  **CAUTION:** that import sits inside `try/except Exception: pass` — a typo silently
  disables depth filtering; the explicit import check below is mandatory.
- NEW `dynamic_gs2/publisher_spawn.py` (~120): the three measured blocks from
  `live_shm_reader.py` — `LIVE_ROOT` (42-50), launcher constants (148-159),
  `_spawn_publisher` (162-257) — plus `os/subprocess/pathlib` imports. **INTERIM** fixup:
  `_PUBLISHER_SCRIPT = Path(__file__).resolve().parents[1] / "dynamic_gs" / "utils" / "live_ros_publisher.py"`
  (retargeted in Phase 6; the mv-detach proof is INVALID until then). Keep verbatim: the
  `bash -c "source … && exec …"` wrapper (adapters_source `close()` depends on exec + same
  process group), the env strip, `DGS_PUB_MAX_HZ` injection, `/tmp/dgs_live_publisher/` logs.
- Repoint `adapters_source.py:312` and `static_pipeline.py:414` → `from .publisher_spawn import …`.
- `cp dynamic_gs/utils/online_fusion.py dynamic_gs2/online_fusion.py` (658, verbatim — no
  `__file__` math). Repoint `static_seed.py:35,65`, `static_seed_stream.py:62`, and the four
  probes (`verify/_probe_cpu_stage_breakdown.py:9`, `_probe_cpu_tsdf_sweep.py:18`,
  `_probe_noicp_fix.py:7`, `_probe_stride_quality.py:15`).
- **`static_pipeline.py:89`: change the subprocess arg `"-m", "dynamic_gs.utils.online_fusion"`
  → `"-m", "dynamic_gs2.online_fusion"`** (grep-invisible — this is the one everything missed).

**Verify**
```
$PY -c "from dynamic_gs2.depth_filter import filter_depth_torch, filter_depth, enabled; from dynamic_gs2.publisher_spawn import _spawn_publisher, LIVE_ROOT, _PUBLISHER_SCRIPT; assert _PUBLISHER_SCRIPT.exists(); import dynamic_gs2.online_fusion; print('ok')"
diff dynamic_gs/utils/depth_filter.py dynamic_gs2/depth_filter.py        # must be empty
diff dynamic_gs/utils/online_fusion.py dynamic_gs2/online_fusion.py      # must be empty
$PY -m pytest dynamic_gs2/tests -q
# Semantic A/B: same static_scene dir, DGS_FUSION_DEVICE=cpu — ply point counts must match EXACTLY
DGS_FUSION_DEVICE=cpu $PY -m dynamic_gs2.online_fusion <static_scene_dir>   # vs the dynamic_gs.utils.online_fusion run
bash dynamic_gs2/full_recorded.sh    # seed via the copied module; check seed POINT COUNT + colors, not just exit code
bash dynamic_gs2/warm_live.sh        # ~20s scheduled stop; ready-handshake in /tmp/dgs_live_publisher/publisher.stderr.log
```
Expected remaining (8): `dynamic_ff_backends.py:184,259`, `dynamic_track.py:50`,
`model_loader.py:80,143,213`, `static_phase0b.py:30` — plus nothing else.

## Phase 3 — Mechanical medium copies: anysplat subset + xfeat set (+2,787 LOC, kills 3)

**Changes**
- NEW `dynamic_gs2/anysplat_decode.py` (576): from the old file take header/constants
  (1-38), `PersistentAnysplatWorker` minus the unused `adopt()`, `_pid_is_anysplat_worker`,
  both quat helpers, `icp_refine_scene_c2w`, `reproject_anysplat_to_scene` (all verified
  zero references into the dropped remainder). **Depth fixup:** `_REPO_ROOT =
  Path(__file__).resolve().parents[2]` → `parents[1]` (must resolve to `scripts/` so
  `anysplat_worker.py` + `third_party/AnySplat` are found). Repoint
  `dynamic_ff_backends.py:184` and `:259`.
- `cp` `xfeat_motion.py` (1525) + `tracker_common.py` (686) into `dynamic_gs2/` whole
  (95% live; only `_pre_mask_image`/`_compose_keep_region` are dead — leave them, verbatim
  copy wins). **Fixups:** `_XFEAT_REPO` at `xfeat_motion.py:70-72` drops one `".."`;
  `from .tracker_common import …` / `from . import tracker_common as _tc` work unchanged
  flat. Repoint `dynamic_track.py:50`.

**Verify**
```
$PY -c "from dynamic_gs2.anysplat_decode import PersistentAnysplatWorker, reproject_anysplat_to_scene, icp_refine_scene_c2w, _ANYSPLAT_REPO, _WORKER_SCRIPT; from pathlib import Path; assert Path(_ANYSPLAT_REPO).is_dir() and Path(_WORKER_SCRIPT).is_file(); from dynamic_gs2.xfeat_motion import XFeatMotionEstimator; print('ok')"
diff dynamic_gs/utils/tracker_common.py dynamic_gs2/tracker_common.py    # must be empty
diff dynamic_gs/utils/xfeat_motion.py dynamic_gs2/xfeat_motion.py        # ONLY the _XFEAT_REPO hunk
$PY -m pytest dynamic_gs2/tests/test_dynamic_track.py dynamic_gs2/tests/test_dynamic_feedforward.py -q
bash dynamic_gs2/warm_recorded.sh    # tracker + ≥1 FF decode (worker spawn proves the parents fix); compare tracker/FF.3a rows + inlier counts
```
Expected remaining (4): `model_loader.py:80,143,213`, `static_phase0b.py:30`.

## Phase 4 — Registration math: sam3d_fusion + ndp (+1,968 LOC, kills 1)

**Changes**
- `cp` `sam3d_fusion.py` → `dynamic_gs2/sam3d_fusion.py` whole (1363; all three
  registration backends stay reachable). **Fixup at `:24`:** replace
  `from .sam3d import load_sam3d_pose` with the 9-line `load_sam3d_pose` inlined verbatim
  (`sam3d.py:101-109`) — *temporary; Phase 5 swaps it back to `from .sam3d import
  load_sam3d_pose` once the copied sam3d.py exists, so no end-state duplicate.*
- `cp` `ndp_register.py` (164) + `ndp/{__init__,nets,rigid_body}.py` (7+304+121) →
  `dynamic_gs2/` (the `ndp` chain is LIVE: `config.py` defaults
  `sam3d_registration_backend="ndp"`). Zero path math in all 5 files; relative imports
  survive the layout unchanged.
- Repoint `static_phase0b.py:30-32`. Leave `verify/ab_isolated_phase0b.py:92-93` on the
  OLD package (its `run_old()` is the A/B reference — that's the point).

**Verify**
```
$PY -c "from dynamic_gs2.sam3d_fusion import load_sam3d_gaussian_ply, load_sam3d_rotation_wxyz, register_and_fuse_sam3d_object; from dynamic_gs2.ndp_register import deform_source_to_target; print('ok')"
$PY dynamic_gs2/verify/ab_isolated_phase0b.py    # old-vs-copied A/B on kept_points/kept_colors
bash dynamic_gs2/full_recorded.sh                # NDP path end-to-end; compare kept_point_count lines vs Phase-3 run
```
Expected remaining (3): `model_loader.py:80,143,213`.

## Phase 5 — SAM worker trio (+2,850 LOC, kills 3; cross-env self-spawner)

**Changes**
- `cp sam_worker.py` (1018) → `dynamic_gs2/`. **Edits:** delete the dead PYTHONPATH
  injection + wrong comment at `:756-759` (verified: the worker NEVER package-imports
  `dynamic_gs.*` — only `_load_sibling_module` path-loads); the self-spawn
  `[env_python, '-u', str(_THIS_FILE), '--worker']` auto-corrects by colocation.
- `cp sam3d.py` (1148) → `dynamic_gs2/`. **Depth edits:** `:18` `SAM3D_REPO_ROOT`
  `parents[2]→parents[1]` (→ `scripts/third_party/Fast-SAM3D`), `:968` + `:1049` cwd
  `parents[2]→parents[1]`. Self-spawn argv untouched.
- `cp fastsam_segmentation.py` (684) → `dynamic_gs2/`. **Depth edit:** `:608` cwd
  `parents[2]→parents[1]`. Keep the `~/.cache/dynamic_gs/fastsam` weights path.
- `sam3_segmentation.py` is NOT copied (gs2's `load_sam3`/`sam3_infer` handlers have zero
  callers — verified). Flag in the copied `sam_worker.py` header: wiring SAM3 back into
  gs2 requires copying it (+360 LOC).
- Swap the Phase-4 interim: `dynamic_gs2/sam3d_fusion.py` inline → `from .sam3d import load_sam3d_pose`.
- Repoint `model_loader.py:80,143,213` to the three local siblings.
- Known shared write: BOTH packages' `sam3d.py` copies write the same
  `third_party/Fast-SAM3D/checkpoints/hf/pipeline_runtime_small.yaml` — same content today;
  note it in the file header as a divergence hazard.

**Verify**
```
$PY -c "from dynamic_gs2.sam_worker import SamWorkerClient; from dynamic_gs2.sam3d import run_sam3d_multi_object_subprocess, SAM3D_REPO_ROOT; from dynamic_gs2.fastsam_segmentation import run_fastsam_subprocess; assert SAM3D_REPO_ROOT.is_dir(); print('ok')"
# REAL spawn proof (worker runs the COPIED file in sam3_dynamic_gs and sibling-loads the COPIED fastsam):
$PY -c "from dynamic_gs2.sam_worker import SamWorkerClient; c=SamWorkerClient(conda_env='sam3_dynamic_gs'); c.load_fastsam(); c.unload_fastsam(); c.close(); print('worker spawn+load ok')"
diff dynamic_gs/utils/sam3d.py dynamic_gs2/sam3d.py                      # ONLY the 3 parents hunks
diff dynamic_gs/utils/fastsam_segmentation.py dynamic_gs2/fastsam_segmentation.py   # ONLY the :608 hunk
bash dynamic_gs2/full_recorded.sh    # FastSAM segment + Fast-SAM3D insert via the copied worker; compare segment/sam3d timing rows + inserted_flags count
```
Expected remaining import sites: **none** (only the interim `_PUBLISHER_SCRIPT` path string).

## Phase 6 — Publisher copy, retarget, gates + the detach proof (+1,937 LOC) — DONE (2026-07-05)

> Landed. `live_ros_publisher.py` (1937) + `zed_depth_noise.py` (97, byte-identical) copied into
> dynamic_gs2/; publisher diff = ONLY the :136 `_RECORDER_SCRIPT` + :683 frame.py `parents[2]/'dynamic_gs2'`
> → `parent` hunks (+ two stale docstring fixes). Its ros_mask.py/frame.py/depth_filter.py/zed_depth_noise.py
> loaders now resolve as plain siblings — the OLD→NEW circular path-load is dissolved. `publisher_spawn.py`
> `_PUBLISHER_SCRIPT` retargeted to the local sibling. NEW permanent gate `tests/test_no_old_package_imports.py`
> (strips comments/docstrings; allowlists the two ab_* A/B files). Verified: publisher compiles under BOTH
> py3.12 AND py3.8 (dynamic_gs_ros); 12/12 tests pass; grep gates empty (imports + code -m strings + path
> strings); **DETACH PROOF: `mv dynamic_gs /tmp` → all 23 main-env runtime modules import + 9/9 CPU tests pass
> with dynamic_gs ABSENT, then restored.** (ros_mask.py needs pyrender = py3.8-env-only, pre-existing, never
> imported into the main env — not a detach issue.)
>
> **dynamic_gs2 is now fully import-independent of dynamic_gs.** Remaining note: LIVE runtime not smoke-tested
> since detach (needs the Gazebo/ROS rig — recorded path fully validated). The old baseline was never modified.

**Original plan below (kept for reference):**


**Changes**
- `cp dynamic_gs/utils/live_ros_publisher.py dynamic_gs2/live_ros_publisher.py` (1937).
  **Fixups (py3.8-safe, script is run by path, never imported):**
  - `:136` `_RECORDER_SCRIPT` `parents[2]/'dynamic_gs2'/'ros_mask.py'` → `parent/'ros_mask.py'`
  - `:683` frame.py loader → `parent/'frame.py'`
  - `:190` zed_depth_noise loader: `parent/'zed_depth_noise.py'` now points at a
    nonexistent gs2 sibling — either copy `zed_depth_noise.py` too or repoint to
    `parents[1]/'dynamic_gs'/'utils'/…`. **Decide at implementation: copy it (small file,
    keeps detach complete).**
  - `:201` depth_filter loader: `parent/'depth_filter.py'` — resolves to the **Phase-2 gs2
    copy** by colocation. Zero edit, verify it loads.
- `dynamic_gs2/publisher_spawn.py`: retarget `_PUBLISHER_SCRIPT` →
  `Path(__file__).resolve().parent / 'live_ros_publisher.py'`; delete the interim comment.
- NEW permanent gate `dynamic_gs2/tests/test_no_old_package_imports.py`: fails on any
  `dynamic_gs.` (dotted, non-gs2) string in `dynamic_gs2/**/*.py` outside the two `ab_*`
  allowlisted files — catches imports AND subprocess `-m` strings forever.
- **Docs in the SAME change:**
  - CLAUDE.md §2: drop "reuses the proven ROS publisher" old-package wording; state
    dynamic_gs2 is import-independent; resolve the §4 TODO `Verified:` pointers now
    answerable with gs2 symbols; Invariant #5 gains the gs2 clause (unreachable-by-design +
    the port-if-Trainer-adopted condition; ninja port lives in `dynamic_gs2/__init__.py`).
  - STATUS_LIVE.md: module list + spawn-chain description (+ `STATUS_LIVE.md:107` fastsam path).
  - BASELINE.md: record the surviving REVERSE dependency — the frozen
    `dynamic_gs/utils/live_ros_publisher.py` still path-loads `dynamic_gs2/ros_mask.py`
    (`:136`) and `frame.py` (`:683`): deleting/renaming those gs2 files breaks the OLD live
    path. Plus the two-copy divergence policy (Decision 1).
  - HISTORY.md: dated detach note (what was copied/inlined, mv-proof result).

**Verify**
```
grep -rnE 'from dynamic_gs\.|import dynamic_gs\.' --include='*.py' dynamic_gs2/ | grep -vE 'verify/ab_(isolated|unit)_phase0b\.py'    # MUST be empty
grep -rnE 'dynamic_gs\.utils|dynamic_gs\.fusion' --include='*.py' dynamic_gs2/ | grep -vE 'verify/ab_|#'                              # dotted strings incl. subprocess -m — MUST be empty
grep -rn 'dynamic_gs/' --include='*.py' dynamic_gs2/ | grep -vE 'dynamic_gs2|verify/ab_'                                              # path strings (missed retargets)
grep -rnE 'ExperimentConfig|from nerfstudio.engine' --include='*.py' dynamic_gs2/ | grep -v verify    # MUST be empty — Invariant #5 unreachability evidence
$PY -m pytest dynamic_gs2/tests -q
# Invariant #5 runtime proof (no outputs/ writes without the old monkeypatches):
touch /tmp/dgs2_detach_marker && bash dynamic_gs2/full_recorded.sh && find outputs -newer /tmp/dgs2_detach_marker 2>/dev/null | wc -l   # expect 0
# ~20s live smoke; pgrep -af live_ros_publisher must show dynamic_gs2/live_ros_publisher.py
bash dynamic_gs2/full_live.sh   # scheduled ~20s stop; targeted cleanup only
# THE DETACH PROOF:
mv /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/dynamic_gs /tmp/dynamic_gs_detach_proof
$PY -m pytest dynamic_gs2/tests -q && bash dynamic_gs2/full_recorded.sh && bash dynamic_gs2/full_live.sh   # (~20s stop)
mv /tmp/dynamic_gs_detach_proof /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/dynamic_gs        # ALWAYS restore, even on failure
git status --short dynamic_gs/    # only the Phase-0 commit; nothing else touched across all phases
```

## Known residual risks

- `pipeline.py` depth-filter import is inside `try/except: pass` — silent-disable trap on
  any future rename.
- The six `parents[N]` depth edits fail only at runtime spawn/weight-load, never at import
  — that's what the per-phase spawn tests are for.
- The mv-detach proof is invalid before Phase 6 (interim `_PUBLISHER_SCRIPT` points into
  `dynamic_gs/` from Phase 2 to Phase 6).
- Duplicate-code drift between the two packages is by design (frozen baseline) — the
  BASELINE.md divergence-policy paragraph is the guard.
- `verify/ab_*` scripts become unrunnable if `dynamic_gs/` is ever physically deleted;
  they are the documented allowlist.
