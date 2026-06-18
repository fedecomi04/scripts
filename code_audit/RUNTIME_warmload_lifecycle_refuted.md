# Adversarial re-verification — RUNTIME_warmload_lifecycle hazards

Stance: SKEPTIC. For each hazard I tried to DISPROVE it against the actual code
(live config: `vis="tensorboard"`, viser-direct on, anysplat FF default,
`interactive_object_selection=False`, KF off). All file:line refs read 2026-06-18.
No source edited.

Live-config facts established up front (dynamic_gs_config.py + base config):
- `enable_viser_direct=True` (base:143) — H1 path reachable.
- `enable_feedforward_inpaint="anysplat_decode"` (base:176) — FF resize path reachable (H2/H3/L2).
- `interactive_object_selection=False` (base:152; live method config dynamic_gs_config.py:216-224 does NOT override it) — H5 and L3 gated OFF by default.
- `vis="tensorboard"` (dynamic_gs_config.py:227) — NS viewer OFF (invariant #9); only viser-direct renders.

---

## H1 — viser render thread races the model_lock swap during init · MEDIUM → **REFUTED (none)**

**Claim:** `ViserDirectScene(port)` (base:1473) starts the server + its render
thread before `server.model_lock = self._model_lock` (base:1480) and `attach_model`
(base:1481); a fast-resume client could enter `get_outputs` between construction
and the swap, acquiring viser's *original* internal lock → two-lock split.

**Refutation:** the render thread is NOT started in the constructor. `__init__`
(viser_direct.py:184-243) only does `viser.ViserServer(port)` (:201) and sets
`self.model_lock = threading.RLock()` (:208) + `self._model = None` (:209). The
render thread is created **only inside `attach_model`** (viser_direct.py:435-442,
`threading.Thread(target=self._render_loop ...).start()`). The pipeline calls
`model_lock = self._model_lock` (base:1480) BEFORE `attach_model` (base:1481), so
by the time `_render_loop` exists the lock is already the shared RLock and the
model is attached. Even if a browser client connects in the construction gap, the
only code that reaches `get_outputs` is `_render_once` (viser_direct.py:689-690),
which runs solely on that not-yet-started render thread; and `_render_once` early-
returns on `model is None` (:661-663) anyway. There is no window where a render
acquires the pre-swap lock while the model is renderable.

**Verdict:** refuted (unreachable as described). revised_severity: none.

---

## H2 — FF bg thread reads `_latest_tracker_frame` the main thread re-points · LOW → **CONFIRMED (low)**

**Claim:** the bg FF reprojects against a camera pose newer than the frame it is
decoding, because `_scene_c2w_for_frame` reads `self._latest_tracker_frame` (the
live one) not the captured `target_frame`.

**Refutation attempt failed — the structure is real:** `_on_tracker_frame` passes
`self._latest_tracker_frame` (the dict the main tick reassigns every tick,
live:323) to `_dispatch_feedforward_async` (live:432). The bg thread's
`_scene_c2w_for_frame` reads `self._latest_tracker_frame["camera"]` (live:465),
the *current* dict, not the `target_frame` argument it was dispatched with. FF
fires every ~10 ticks and runs under `_anysplat_slot_lock` (single-in-flight,
base:2509), so by FF-run time the main thread has re-pointed the dict 1+ times →
the FF back-projects with a camera pose newer than the CDN/frame it decodes. The
live:459-462 docstring even *claims* "the FF dispatcher snapshots state at dispatch
time" — but no snapshot of the pose actually happens; it re-reads the live dict.

This is a correctness smell (pose/frame skew in inserts), NOT a crash or torn
read — the camera object itself is a consistent tensor. LOW is justified.

**Verdict:** confirmed. revised_severity: low.

---

## H3 — identity buffers read without `_model_lock` while FF bg resizes them · MEDIUM → **CONFIRMED (low)**, and the audit's own atexit reasoning is WRONG

**Claim:** `insert_inpaint_gaussians`/`delete_gaussian_indices` reassign the buffer
tensors on the ff bg thread; unlocked reads on other threads can race. D0 reads
are safe (pre-FF); the atexit snapshot/report reads could race a final in-flight
FF, BUT (audit claims) `_cleanup_anysplat_bg` is registered AFTER the snapshot so
LIFO drains the bg slot first.

**Refutation attempt — partly refuted, partly the audit UNDERSTATED it:**
- Confirmed FF mutations ARE locked: `insert_inpaint_gaussians` is called under
  `with self._viser_lock_ctx()` (base:3497) and `_viser_lock_ctx()` returns
  `self._model_lock` (base:1433-1448). The methods themselves don't lock
  (dynamic_gs_model.py:1116, :1150) but every FF caller wraps them. Viser render
  + main CDN read `get_outputs` under the same RLock (viser_direct.py:689). So
  the steady-state per-tick / per-render races are GUARDED.
- D0 reads (`object_instance_ids` in `_pick_d0_object`, live:357) are pre-FF
  (`is_first` gate returns FF=False, base:950-952) — safe, as the audit said.
- **The atexit claim is BACKWARDS.** Registration order (base:476-481):
  viser(476), worker(477), bg(478), ipc(479), video(480), **snapshot(481)**.
  atexit is LIFO → executes snapshot(481) **FIRST**, then video, ipc, then
  bg-drain(478) **LATER**. So `_save_final_snapshot_if_enabled` (which reads
  `model.state_dict()`, `model.num_points`, `object_flags.sum()`,
  `inserted_flags.sum()` at base:678-685 with NO `_model_lock`) runs BEFORE the
  FF slot is drained. If an FF insert is mid-resize at process exit, the snapshot
  can read a torn (mismatched-N) state_dict. The audit's "LIFO drains bg first"
  is incorrect; the hazard is *more* real than the audit reasoned.

Net: real, but shutdown-only and narrow (requires an FF daemon thread still mid-
`insert_inpaint_gaussians` exactly as atexit fires). Real-but-rare → LOW.

**Verdict:** confirmed (with corrected mechanism). revised_severity: low.

---

## H4 — `_cleanup_live_subscriber` not idempotent vs concurrent callers · MEDIUM → **CONFIRMED (low)** (idempotency is fine; the real gap is the missing kill())

**Claim:** registered atexit (live:161) + called from `_on_signal` (live:173).
Idempotency is mostly safe via `if sub is None: return` (live:199) + `close()`'s
`self._closed` guard (live_shm_reader.py:564). The real gap: `close()` does
`wait(timeout=5)` then on timeout `terminate()` but never `kill()`/second `wait()`.

**Refutation attempt — idempotency refuted-as-safe; the kill gap is real but rare:**
- Idempotency IS guarded: `_cleanup_live_subscriber` nulls `self._shm_sub`
  (live:205) and re-checks `sub is None` (live:199); `close()` is `self._closed`-
  guarded (live_shm_reader.py:564-566). Concurrent atexit+signal double-call is a
  no-op the second time. So the "not idempotent" framing is REFUTED.
- The missing-`kill()` IS real: `close()` sends the "shutdown" JSON, `wait(5s)`,
  and on timeout calls `terminate()` (SIGTERM) with NO follow-up wait or kill
  (live_shm_reader.py:579-585). **However**, the publisher installs its own
  SIGTERM/SIGINT/SIGHUP handler (`_on_signal → pub.shutdown(); os._exit`,
  live_ros_publisher.py:1456-1464), so a delivered SIGTERM normally does drop it.
  The orphan only survives if the publisher is wedged in an uninterruptible state
  AND ignores both the shutdown JSON and SIGTERM — which the CLAUDE.md "Live
  publisher restart cleanup" memory confirms DOES happen occasionally in practice
  (stale `dynamic_gs_live_pub` node). So a one-line `kill()` after `terminate()`
  is a genuine robustness fix, but the residual orphan is rare, not the default.

**Verdict:** confirmed (real gap, idempotency claim refuted). revised_severity: low.

---

## H5 — interactive-picker flags are plain bools shared across 3 threads · LOW → **REFUTED (unreachable in default live config)**

**Claim:** `_reselect_requested` is set by the stdin watcher (live:234) + viser
"Change object" callback (base:1346) and read/cleared by the main tick with no
barrier — a set can be clobbered.

**Refutation:** the entire path is gated by `interactive_object_selection`, which
is **False** by default (base:152) and is NOT enabled by the live method config
(dynamic_gs_config.py:216-224). The stdin watcher only sets `_reselect_requested`
inside `if s == "" and self.config.interactive_object_selection` (live:232-234),
and `_tick_interactive_selection` is only invoked under the same flag (live:287).
With the flag off the flag is never set or read. The race is real ONLY when an
operator opts into the picker, and even then the impact is "operator clicks again"
(the audit's own LOW). Unreachable in the default live config.

**Verdict:** refuted (unreachable at default config). revised_severity: none
(low only if `interactive_object_selection=True` is explicitly set).

---

## L1 — ROS publisher orphaned on hard kill · MEDIUM → **CONFIRMED (low)** (stale-region claim REFUTED)

**Claim:** on SIGKILL the publisher + depth-republisher child survive holding
`/dgs_live_shm` + ROS subs; the reader only attaches + unregisters
(live_shm_reader.py:317) so never unlinks; `/dev/shm/dgs_live_shm` leaks until
reboot, and the next run's `SharedMemory(create=False)` attaches to a STALE region.

**Refutation attempt — orphan real, stale-region claim REFUTED:**
- The publisher deliberately does NOT unlink its SHM on shutdown
  (live_ros_publisher.py:1295-1299, by design — process exit releases it, and
  unlinking mid-teardown would yank it from a still-reading reader).
- **The "next run attaches to a stale region" claim is false.** Every publisher
  start unlinks any pre-existing region with the same name BEFORE creating a fresh
  one (live_ros_publisher.py:593-603: `SharedMemory(name).unlink()` then
  `SharedMemory(create=True)`). So a leaked `/dev/shm/dgs_live_shm` is auto-
  reclaimed on the next launch — it does not poison the next run.
- The genuine residual is the orphaned *process* on a true double-SIGKILL (the
  publisher otherwise handles SIGTERM/SIGINT/SIGHUP itself, line 1456-1464).
  This matches the CLAUDE.md "Live publisher restart cleanup" memory; the existing
  scoped-`pkill` + `rosnode cleanup` is the mitigation. Real but rare.

**Verdict:** confirmed (process orphan on hard kill); the stale-region / leaks-
until-reboot framing is refuted (next start auto-unlinks). revised_severity: low.

---

## L2 — AnySplat worker + /dev/shm IPC file leak on hard kill · LOW → **CONFIRMED (low)**

**Claim:** `_cleanup_anysplat_worker` (base:631) + `_cleanup_anysplat_ipc_file`
(base:640) are atexit-only; on SIGKILL the worker stays GPU-resident and
`/dev/shm/anysplat_ipc_<pid>.npz` leaks. Adopt-reclaim (base:2915) can replace a
verified pid later.

**Refutation attempt failed — confirmed as stated, already LOW.** Both cleanups
are atexit-registered (base:477, :479) so SIGKILL skips them. The IPC file is
per-pid (`anysplat_ipc_{os.getpid()}.npz`, base:644) so it does not collide across
runs; the adopt path targets a verified worker pid for reclaim. Real leak on hard
kill, self-healing on the next adopt. Severity LOW is already correct.

**Verdict:** confirmed. revised_severity: low.

---

## L3 — ESAM model never freed · LOW → **REFUTED-as-unreachable in default config**

**Claim:** `_get_esam_model` lazily builds `build_esam_ti` into `self._esam_model`
on first interactive-picker use and never releases it.

**Refutation:** ESAM is only touched on the interactive-picker path
(`interactive_object_selection`, default False — base:152, not set by the live
config). With the picker off, `_get_esam_model` is never called, so nothing is
allocated and nothing leaks. The audit itself notes "Only matters when
`interactive_object_selection=True`." Unreachable at default.

**Verdict:** refuted (unreachable at default config). revised_severity: none
(low only if the picker is enabled).

---

## L4 — old XFeat estimator GPU weights not explicitly freed on object switch · LOW → **REFUTED-as-unreachable in default config; real-but-trivial if enabled**

**Claim:** `_reseed_tracked_object` nulls `self._motion_estimator` (base:1156) then
builds a fresh `XFeatMotionEstimator` (base:1993); old weights reclaimed only by GC.

**Refutation attempt:** mid-run object switches happen only via the interactive
picker / "Change object" button / bare-Enter (all gated on
`interactive_object_selection=False` by default — live:232, base:1346). In the
default live run D0 reseeds the estimator exactly once at bootstrap; there is no
repeated switch, so no transient double footprint. Even when the picker is on, the
old estimator is dropped (`= None`, base:1156) and GC + later `empty_cache`
reclaim it — a transient, not a true leak. Real-but-trivial, and unreachable at
default.

**Verdict:** refuted (unreachable at default; trivial transient if enabled).
revised_severity: none (low if picker enabled).

---

## B1 — `sam3d_init_target_flags` rebuilt/zeroed but never written · N/A deadbranch → **CONFIRMED (expected per invariant #8)**

**Claim:** the only value-writer `initialize_object_from_sam3d`
(dynamic_gs_model.py:1604) has no live caller; all-zeros is the documented
expected state; load-bearing for state_dict shape + dump scripts.

**Verification:** `grep -rn initialize_object_from_sam3d dynamic_gs/` returns only
the definition (dynamic_gs_model.py:1604) and a docstring mention
(static_gs_model.py:28) — no caller. Matches CLAUDE.md invariant #8 exactly. The
buffer is rebuilt at load (post_fusion_cache rebuild path) and carried by
insert/delete (dynamic_gs_model.py:1140-1141) for shape compatibility. NOT a bug;
do not delete in any purge.

**Verdict:** confirmed-as-expected (not a defect). revised_severity: none.

---

## B2 — per-tick allocations on tracker critical path · LOW → **CONFIRMED (expected, not waste)**

**Claim:** `_batch_from_live_frame` allocs 3 GPU tensors/tick; `_pick_d0_object`
projects all means each deferred D0. Both expected (new frames each tick; D0 runs
only until first lock). No fix.

**Verification:** D0 projection is gated by `is_first`/deferred-D0
(`_d0_completed`, live:279, 294-301) so it stops after bootstrap; the per-tick
batch copy is inherent to a live stream. Not steady-state waste. Confirmed as
benign; do not "optimize" it away in a purge.

**Verdict:** confirmed-as-expected. revised_severity: none.

---

## Summary

| Hazard | Audit sev | Verdict | Revised |
|---|---|---|---|
| H1 viser lock-swap race | MEDIUM | refuted (render thread starts only in attach_model, after the swap; None-guard) | none |
| H2 FF reads live `_latest_tracker_frame` for pose | LOW | confirmed (real pose/frame skew, not a crash) | low |
| H3 buffer reads vs FF resize | MEDIUM | confirmed; audit's atexit LIFO reasoning is backwards — snapshot runs BEFORE bg-drain | low |
| H4 cleanup kill() gap | MEDIUM | confirmed (kill gap real, rare); idempotency claim refuted | low |
| H5 picker flag race | LOW | refuted (gated off by default) | none |
| L1 publisher orphan | MEDIUM | confirmed (process orphan); stale-region/next-run-poison claim refuted | low |
| L2 anysplat worker/ipc leak | LOW | confirmed | low |
| L3 ESAM never freed | LOW | refuted (picker-only, off by default) | none |
| L4 old XFeat weights | LOW | refuted (switch is picker-only; trivial transient) | none |
| B1 sam3d_init_target_flags | N/A | confirmed-as-expected (invariant #8) | none |
| B2 per-tick allocs | LOW | confirmed-as-expected (not waste) | none |

Highest-value real items after skeptical re-check: **H3 atexit ordering** (the
snapshot reads model state *before* the FF bg slot is drained — the opposite of
what the audit reasoned) and **H4 missing `kill()`** (one-line robustness fix; the
publisher's own SIGTERM handler makes the orphan rare). Everything tagged MEDIUM
by the audit downgrades to LOW or NONE under the default live config; H1 fully
refuted.
