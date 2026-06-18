# FIXES — must-fix hazards the rewrite has to design out

Authoritative fix list (operator-prioritized 2026-06-18), from the adversarial-refutation
triage (`code_audit/RUNTIME_TRIAGE.md`). These are REWRITE requirements — design them out
of `dynamic_gs2`, do NOT retrofit the old `dynamic_gs/` (it stays the frozen ground-truth
baseline; see VERIFICATION.md). Old file:line are cited so the rewrite knows the behavior
to NOT reproduce.

## P0 — HIGHEST PRIORITY (operator-named): FF must use ONE frozen pose
**Bug (old `dynamic_gs_pipeline_live.py:323,465`, mediums ff-H4 + ff-H5):** the FF bg thread
back-projects/places inserts using the LIVE `self._latest_tracker_frame['camera']` and
`self._latest_live_rgb_bgr`, which the main thread re-points every tick during the ~200 ms
FF run. So **inserts land against a newer pose than the CDN scored them with** → mm–cm
spatial drift → re-flags the same region → CDN churn → FF over-insertion (the 3M blowup
class). The old docstring *claims* a snapshot but reads live attributes.

**Rewrite fix (INVARIANT — amends `dynamic_feedforward.md`):** the FF dispatch freezes ONE
immutable `FeedforwardDispatch` bundle = {seq, **camera/pose**, rgb_bgr, depth_m, masks,
scene snapshot}. The `FeedforwardWorker` reads **ONLY that bundle** — it must NEVER touch
any `self._latest_*` / live pipeline attribute. The pose the CDN scores against == the pose
inserts are placed against == the dispatch pose, BY CONSTRUCTION. Enforce with a review
check: `grep` the FF worker for `_latest_` → must be zero hits. This is the single change
that makes FF "insert much less" (the operator's expectation) — it removes the skew-driven
re-insertion loop, independent of ICP.

## P1 — the other MEDIUMS (all must-fix)
1. **rospy pose/joint history race** (old `live_ros_publisher.py:751-781`, shm-H3 — the one
   true corrupting race). Two independent `rospy.Subscriber` receive threads mutate the
   pose/joint history lists with no lock; the worker reads them in `_interpolate_c2w`; the
   two-statement `insert` is interruptible → `len(times)!=len(mats)` → IndexError/wrong pose.
   **Rewrite fix:** the ros source owns a **dedicated history `Lock`** held by both callbacks
   (writers) and the interpolation read-window — already specced in `adapters_source.md` §8;
   mark it MUST-DO, separate from any model lock.
2. **`_object_crop_bbox` unlocked means read** (old `base:1866-1876`, ff-dispatch-H1 —
   narrow-window hot-path crash). Reads `model.means[obj_mask]` unlocked every tick; an FF
   Parameter realloc in the read→index gap → size-mismatch crash that kills the tracker
   thread (no try/except at the call site). **Rewrite fix:** the tracker reads the crop bbox
   from the **`GaussianSet.snapshot()`** (immutable, length-consistent) — never the live
   tensors. Falls out of the WRAP/snapshot design for free; assert it.

## P2 — LOW (fold in ONLY because the new design makes them ~free; do NOT bloat, do NOT retrofit old)
These are cheap *by construction* in the rewrite (most are already in the specs). Each is a
few lines; the operator's caution ("could be new sources of error") is respected because in
`dynamic_gs2` they are the DEFAULT correct behavior, not a patch bolted onto working code:
- **`/dev/shm` FF file leak** (old `base:3367,3369,644`, ff-H8 / warmload-L2 — the audit's #1
  purge item): cleanup must unlink the **actual indexed** crop/ipc filenames it wrote. In the
  rewrite, `debug.py`'s writer thread OWNS those files and removes them on `finalize()`; the
  FF IPC path uses a tempfile context that self-cleans. No dangling-name bug possible.
- **SHM unlink on crash** (shm-H1) + **`_closed` guard in peek + `kill()` after `terminate()`**
  (shm-H4 / warmload-H4): already in `adapters_source.md` §7 (`finally` release, SIGTERM
  best-effort unlink, `_closed` early-return, terminate→kill). Confirm implemented.
- **`_anysplat_slot_lock` leak on `Thread.start()` raise** (tracker-H3): the FF worker's
  single-in-flight slot uses `try/except` around thread spawn so a spawn failure releases the
  slot (else FF silently dies forever). One guarded block in `FeedforwardWorker.dispatch`.
- **viser camera-feed copy when headless** (viser-H5): `ViserBridge` skips the `.cpu().numpy()`
  push when `_client_state` is empty. One guard.
- **doc staleness** (shm-H8): not code — fix the CLAUDE.md "median+bilateral on tracker batch
  depth" line (true for recorded, stale for live where the tracker uses RAW depth).

## DO NOT TOUCH / NOT A FIX (confirmed by refutation)
- `sam3d_init_target_flags` all-zeros (invariant #8) — keep the buffer, it is not a leak.
- The seqlock (shm-H2), `_obj_mask_cache` is-None-outside-lock, the `_timing` defaultdict,
  `_on_synced` drop-oldest — REFUTED as benign (GIL-atomic / already-locked / freshest-wins).
  Do not "fix" these; adding locks here is pure risk for zero gain.
- 17 hazards REFUTED entirely (unreachable in live config / already guarded) — see TRIAGE §2.

## Acceptance (ties to VERIFICATION.md)
- FF worker has **zero** `_latest_*` reads (grep check). FF insert total ≤ old, bounded, non-continuous.
- No unlocked `model.means`/identity-buffer read on the tracker hot path (all via snapshot/lock).
- The ros history lock exists; SHM/worker/tmpfiles all have a guaranteed release path.
