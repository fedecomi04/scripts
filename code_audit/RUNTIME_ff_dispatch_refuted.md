# RUNTIME_ff_dispatch — adversarial re-verification (skeptic pass)

Stance: try to REFUTE each hazard in `RUNTIME_ff_dispatch.md` against the actual code on
the **live default config** (`vis="tensorboard"` → NS viewer OFF, viser-direct only;
`enable_feedforward_inpaint="anysplat_decode"`; `xfeat_crop_to_object_bbox=True`;
`xfeat_object_mask_filter=True`). Each verdict cites file:line evidence. "refuted" only when
shown unreachable/guarded/dead; "confirmed" only when shown real+reachable+unguarded;
"uncertain" when neither is provable.

Config evidence anchoring this pass:
- `dynamic_gs_config.py:227` live `vis="tensorboard"`, `:225` `_ZERO_LR_OPTIMIZERS`.
- `dynamic_gs_pipeline_base.py:176` `enable_feedforward_inpaint = "anysplat_decode"` (default).
- `dynamic_gs_model.py:429` `xfeat_crop_to_object_bbox = True`, `:484` `xfeat_object_mask_filter = True`.

---

## H1 — `_object_crop_bbox` reads `model.means` unlocked (RACE) → **CONFIRMED**

Refutation attempt (looked for a guard / GIL-atomicity / unreachability that closes the window):
- Reachable: YES. `_object_crop_bbox` is called from `_apply_motion_estimator` at
  base.py:2083 whenever `xfeat_crop_to_object_bbox` is True — and it IS True by default
  (dynamic_gs_model.py:429). This is the live tracker hot path, every tick.
- The lock: there is NO `_viser_lock_ctx()` anywhere in `_object_crop_bbox`
  (base.py:1855-1933 — verified the whole function body). So the read genuinely bypasses
  the lock that every writer holds.
- The guard at base.py:1874 (`if int(obj_mask.shape[0]) != n_means: return None`) does
  NOT close the window. Sequence: `obj_mask = model._tracked_object_mask()` reads
  `object_instance_ids` (base.py:1866 → model.py:958); `n_means = model.means.shape[0]`
  (1873); index `model.means[obj_mask]` (1876). The FF insert reassigns the `means`
  Parameter at model.py:1219 and resizes the ID buffers separately at model.py:1223 —
  two distinct statements with GIL-yielding torch ops (`torch.cat`, CUDA) between them,
  all under `_model_lock` (base.py:3497) which `_object_crop_bbox` does not contend for.
  If the `means` reassign lands AFTER `n_means` is read (1873) but BEFORE the index op
  (1876), then `means` is the new-N tensor indexed by an old-N bool mask → size-mismatch
  IndexError / CUDA index assert at 1876. The guard ran at 1874, before that reassign,
  so it cannot catch this ordering.
- Severity check: real-but-rare. FF fires every ~10 ticks; the insert/delete critical
  section is short; but the doc-cited 459k→1.29M growth makes the resize non-instant, and
  the index op at 1876 is the one statement that crashes the whole tracker thread (no
  try/except around it — `_apply_motion_estimator` has none at the call site base.py:2083).
- Verdict: CONFIRMED, reachable on live hot path, unguarded in the read-then-index gap.
  Revised severity **medium** (genuine crash, but narrow timing window, low per-tick prob).

## H2 — `_feedforward_delete_in_region` reads cross-thread `model.info` (RACE, rgbd only) → **REFUTED (unreachable on live default)**

Refutation:
- `model.info` is read only in `_feedforward_delete_in_region` (base.py:2750 →
  active_mask.py:618-646), called only from the rgbd-mode `_run_feedforward` body at
  base.py:2681. That body is reached only when `enable_feedforward_inpaint != "anysplat_decode"`
  (base.py:2561 returns to `_run_feedforward_anysplat` before ever reaching line 2681).
- Live default is `anysplat_decode` (base.py:176). The anysplat cull
  `_feedforward_cull_in_front_of_depth` projects `model.means` directly (base.py:2818) and
  reads `object_instance_ids` (2845) — it never touches `model.info`, and the whole call
  runs under `_model_lock` (base.py:2876). So the `model.info` aliasing class does not
  exist on the live path.
- The "coincidental count match → deletes wrong Gaussians" sub-claim: the guard at
  active_mask.py:639-642 raises ValueError on count mismatch, caught at base.py:2752 →
  delete skipped (safe). The wrong-delete only fires when a stale `info` happens to have
  exactly the current `num_points` — narrow, and still rgbd-only.
- Verdict: REFUTED for the live config (dead branch under `anysplat_decode`); real latent
  bug only if someone flips the mode. Revised severity **low**.

## H3 — `_obj_mask_cache` check-then-act shared by main + FF bg (RACE) → **CONFIRMED (low)**

Refutation attempt:
- The `if self._obj_mask_cache is None:` test at base.py:1666 is OUTSIDE the lock; the
  lock is acquired at 1677 and the render+set at 1680, with NO double-check inside the
  locked region. So two threads (main tick + FF bg `_compute_tick_cdn` path at
  base.py:1756 / `_run_feedforward_anysplat` at 3010) can both pass the None test and both
  render — wasteful, not corrupting (each render is individually under the lock).
- The staleness path is real: FF bg caches a mask, main thread moves the object via the
  rigid transform (base.py:2164) + `_invalidate_object_mask_cache` (base.py:2168), but if
  the bg thread already cached, it reuses the pre-move mask for one FF call. The
  invalidate at live.py:275 / base.py:2168 is unlocked, so the None-write can interleave
  with the bg None-test.
- Worst case is a one-call misaligned object-exclusion mask feeding the documented
  "misplacement ring" churn — quality, not a crash or buffer corruption.
- Verdict: CONFIRMED as a real unsynchronized check-then-act, but consequence is cosmetic/
  churn. Revised severity **low**.

## H4 — FF bg decodes against a frame the tracker advanced past (STALENESS) → **CONFIRMED (medium)**

Refutation attempt:
- Tried to find that the live override snapshots the dispatch camera. It does NOT.
  `_scene_c2w_for_frame` (live override) reads `self._latest_tracker_frame["camera"]`
  (live.py:465) — the CURRENT attribute, replaced every tick at live.py:323 — NOT the
  `target_frame` the FF was dispatched with. The base threads a `camera` into bg_args, but
  the live `_scene_c2w_for_frame` ignores it and re-reads the live attribute. The docstring
  at live.py:459-462 even claims the dispatcher snapshots state "so this is the right pose"
  — but the read is of the live attribute, contradicting the comment.
- The main thread runs ~tens of ticks during the ~200 ms FF, so the reproject pose can be
  a later camera than the CDN/depth the inserts were selected against → a few mm–cm spatial
  mismatch on the inserts. Reference swap is atomic under the GIL, so no torn dict (the bg
  thread holds its own `target_frame` ref) — no crash.
- This is plausibly part of the CDN churn loop the project notes chase.
- Verdict: CONFIRMED staleness, by design-omission, lossy not crashing. Revised severity
  **medium** (quality/churn contributor).

## H5 — `_latest_live_rgb_bgr` read on FF bg, written on main (RACE/staleness) → **CONFIRMED (low)**

Refutation attempt:
- Main writes `self._latest_live_rgb_bgr = latest.rgb_bgr` every tick (live.py:321); FF bg
  reads it in `_resolve_anysplat_context_image_paths` (live.py:446) then `cv2.imwrite`s it
  (live.py:455). Bare attribute, same root cause as H4 — the bg thread sees whatever the
  latest write left, which can be one+ tick ahead of the depth/CDN it is decoding.
- Reference rebind is atomic under the GIL → no torn frame, no crash; the AnySplat source
  image is just possibly newer than the depth used to back-project it → small reproject
  error.
- Verdict: CONFIRMED staleness; no corruption. Revised severity **low** (it feeds the same
  drift as H4 but on the RGB-source side only).

## H6 — `self._timing` defaultdict written by two threads (BENIGN) → **REFUTED-as-benign (confirmed benign)**

- Main appends DN.* keys, FF bg appends FF.* keys — disjoint key sets (verified: DN.* at
  base.py:2062-2189 main; FF.* at base.py:2494/3387/3494/3507 bg). CPython GIL makes
  `list.append` and dict-slot insert individually atomic; disjoint keys → no lost update.
- Verdict: matches the audit's own "benign". No action. Revised severity **none**.

## H7 — slot lock acquired on main, released on FF bg; `Thread.start()` failure leaks it → **CONFIRMED (low)**

Refutation attempt:
- The normal release path is sound: `_feedforward_threaded` wraps `_run_feedforward` in
  try/finally and releases unconditionally (base.py:2540-2544), with the release itself in
  try/except RuntimeError (swallows double-release). `_anysplat_bg_run`'s finally
  deliberately does NOT release (base.py:3549-3553). Shutdown drain `_cleanup_anysplat_bg`
  blocks-acquires with 60 s timeout (base.py:658). So exceptions inside the FF body are
  covered.
- The one gap is real: `_dispatch_feedforward_async` acquires the slot at base.py:2509 then
  spawns the thread at base.py:2513 with NO try/except around `Thread(...).start()`. If
  `start()` raises (OS thread-exhaustion), the slot is never released → every future
  dispatch logs "previous FF still in flight" and FF is permanently dead. Unbounded but
  needs OS thread exhaustion (very unlikely in this process).
- Verdict: CONFIRMED gap, extremely low probability. Revised severity **low**.

## H8 — /dev/shm crop/ipc files leak; only non-indexed name cleaned (LEAK) → **CONFIRMED (low)**

Refutation attempt:
- Files written: `anysplat_crop_{pid}_{wi}.png` (base.py:3367), `anysplat_ipc_{pid}_{wi}.npz`
  (base.py:3369) — both indexed by `wi`; plus `dgs_live_ff_frame_{pid}.png` (live.py:454).
  Cleanup `_cleanup_anysplat_ipc_file` unlinks ONLY `anysplat_ipc_{pid}.npz` (no `_wi`,
  base.py:644) — a name this path never writes. So the indexed files + the live frame png
  are never removed at exit.
- BUT: filenames are fixed per `(pid, wi)` and overwritten each call (base.py:3367-3369),
  so it's a bounded handful of stale files (≤ max windows + 1), NOT unbounded growth — the
  audit states this correctly.
- Verdict: CONFIRMED tmpfs litter, bounded. Revised severity **low**.

## H9 — per-FF CPU↔GPU sync + host allocs (PERF, not correctness) → **REFUTED-as-not-a-bug (confirmed perf-only)**

- The reproject path is pure-numpy CPU with several full-N allocs per window
  (anysplat_decode.py reproject), and ICP does a GPU→Open3D dlpack handoff per call. All on
  the FF bg thread, sharing the GPU (no MPS) and serializing against the tracker only inside
  the `_model_lock` windows (frustum-cull read base.py:3281, insert base.py:3497). This
  matches the documented "FF contends with tracker on shared GPU"; `DN.3k_model_lock_wait`
  (base.py:1678) measures exactly that stall.
- Not a correctness bug; flagged so a purge doesn't move these back onto the main thread.
- Verdict: REFUTED as a hazard (it's a known perf characteristic, not a defect). Revised
  severity **none**.

## H10 — `_compute_tick_cdn` swallows render failure → FF silently no-ops (DEAD-ish) → **CONFIRMED-benign**

- `_compute_tick_cdn` returns None on render exception (base.py:1740-1742); `_feedforward_threaded`
  stores `target_frame["cdn"]` = the None; `_run_feedforward` logs "no CDN this tick — skip"
  and returns (base.py:2558-2560). A transient render failure drops one FF firing; the next
  cadence retries. No crash, no corruption.
- Verdict: CONFIRMED real + benign; documented so it isn't mistaken for "FF never runs".
  Revised severity **low**.

---

## Net

- Only **H1** is a real crash on the live hot path, and it's narrow (read-then-index gap;
  needs an FF resize inside a microsecond window). Medium.
- **H4** (and its sibling **H5**) are real, by-design staleness that plausibly feed the
  documented CDN churn — worth a snapshot-the-dispatch-state fix, but quality not crash.
- **H2** is genuinely unreachable on the live default (rgbd-only); refuted for live.
- **H3 / H7 / H8 / H10** are real but low (churn / unlikely / litter / benign-skip).
- **H6 / H9** are correctly self-classified non-issues.
- No hazard was found to be FALSELY cleared: every "refuted" is backed by a dead branch
  (H2) or an explicit non-defect classification (H6, H9). I did not down-clear any real race.
