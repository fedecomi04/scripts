# RUNTIME_tracker_tick — adversarial re-verification (SKEPTIC pass)

Stance: try to REFUTE each hazard in `RUNTIME_tracker_tick.md` against the
ACTUAL live config (`dynamic-gs-live`: `vis="tensorboard"`, viser-direct only,
`_ZERO_LR_OPTIMIZERS`, anysplat FF default, `interactive_object_selection=False`).
Every verdict cites code I read this pass. Source NOT edited.

Repo root: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`

## Ground-truth I re-confirmed (so reachability/guard calls are honest)

- Live config has NO override of `interactive_object_selection` →
  inherits base default **`False`** (`dynamic_gs_pipeline_base.py:152`); neither
  `bootstrap_live.sh` nor `resume_live.sh` passes the flag. So the interactive
  picker and the mid-run *reselect* path are OFF by default in live.
- `_recurring_ff_due` returns **False** when `is_first`
  (`dynamic_gs_pipeline_base.py:950-951`) → no FF thread can be inserting during
  the D0 tick.
- `_pick_d0_object` is reached in live ONLY from `_bootstrap_d0`
  (`dynamic_gs_pipeline_live.py:486`), itself only in the `is_first` branch
  (`:295`). The picker-path reach (`_selection_fallback_id` :1210) is gated by
  `interactive_object_selection` (OFF).
- `insert_inpaint_gaussians` REALLOCATES every `gauss_params[name]` to a fresh
  `Parameter(cat(...))` and grows N (`dynamic_gs_model.py:1216-1228`) — a
  non-atomic, count-changing write. So the "unlocked reader tears" premise of
  H2/H4 is genuinely real for any reader that doesn't hold the lock.
- The shared RLock is plumbed three ways: model render
  (`attach_render_lock`, model.py:741-747), viser server
  (`_viser_direct_server.model_lock = self._model_lock`, base:1480), and every
  pipeline mutation site. Viser render thread reads `model.get_outputs` under
  `with self.model_lock` (`viser_direct.py:689`).
- `xfeat_motion.py` references neither `self.model` nor `gauss_params` (grep
  empty) → the estimate step touches no model state, consistent with the audit.

---

## H1 — `_latest_tracker_frame` published before FF reads it; tracker keeps mutating the model
- Original: medium, race (semantic staleness, not corruption).
- Refutation attempt: the only way this is worse than "stale pose" is torn
  tensors. Both the FF model read/writes (`_compute_tick_cdn` under
  `_viser_lock_ctx` base:2535-2536; cull base:2690/3497; insert) and the tracker
  write (base:2163) take the SAME `_model_lock`. The dict slots it captures
  (`camera`, `batch`) are per-tick immutable tensors. So NO corruption — only
  semantic staleness, which is the documented+accepted "_4_rendered captured a
  few ticks after _6_raw_mask" caveat.
- Verdict: **confirmed** as described (a real, intentional, contained design
  property — not a defect). Revised severity **low**. The audit already
  recommends "no fix"; I agree. Calling it "medium" overstates a documented,
  accepted behavior.

## H2 — Unlocked `gauss_params` reader (`_pick_d0_object`) would tear vs an FF insert
- Original: high (latent), race.
- Refutation attempt: `_pick_d0_object` reads `self.model.means` /
  `object_instance_ids` with NO lock (`dynamic_gs_pipeline_live.py:367,381,398`).
  Is an FF insert reachable concurrently in the LIVE default config?
  - At D0 (`is_first`): NO. `_recurring_ff_due` is False while `is_first`
    (base:950-951), and the oneshot dispatch in `get_train_loss_dict` requires
    `_latest_tracker_frame is not None` (base:991) which is only set AFTER the D0
    path; D0 also `return`s early when deferred (live:296-301). So no FF thread
    exists during any `_pick_d0_object` call on the live tracker thread today.
  - Mid-run reselect path (the audit's "if the purge moves D0 pick to mid-run"):
    reachable only via `_tick_interactive_selection` /`_selection_fallback_id`,
    both gated by `interactive_object_selection` which is **False** by default in
    live and not set by the launch scripts.
  So the tear is UNREACHABLE in the shipped live config — it is a latent
  invariant that depends on (a) the `is_first` FF gate and (b) the picker being
  off. The audit itself says "safe TODAY."
- Verdict: **uncertain** (genuinely: the hazardous read IS unguarded and the
  reallocation IS real, so it is not *refuted* as a bug; but it is **not
  currently reachable** in the default live config, so it is not a live defect
  either). Revised severity **low** (latent / not reachable today). Flag for the
  human: keep the `is_first` FF gate and keep the picker off, OR wrap the two
  reads in `_pick_d0_object` in `with self._viser_lock_ctx():` to make it
  unconditionally safe (cheap — one projection).

## H3 — `_anysplat_slot_lock` acquired on trainer thread, released on FF thread; not released if `Thread.start()` raises
- Original: medium, lock/leak.
- Refutation attempt: code reads exactly as the audit says —
  `acquire(blocking=False)` (base:2509) then `threading.Thread(...).start()`
  (base:2513) with NO try/except around `start()`; release only in
  `_feedforward_threaded`'s `finally` (base:2542), on the bg thread. If `start()`
  raises (`RuntimeError: can't start new thread`) after acquire succeeds, the bg
  thread never runs → `finally` never runs → slot lock held forever → every
  future dispatch hits `acquire(blocking=False)`==False and logs "previous FF
  still in flight" → FF silently dead for the rest of the run. I could not find
  any guard that refutes this. Acquire-here/release-there is legal for a plain
  `threading.Lock` (not owner-checked), so the only hole is the unguarded
  `start()`.
- Severity reality: `Thread.start()` raising is genuinely rare (needs OS
  thread-table exhaustion). Real structure, low firing probability — analogous to
  the "H-CROP race" class.
- Verdict: **confirmed** (real, reachable in principle, unguarded). Revised
  severity **low** (real-but-rare; single-point silent FF death only under
  thread-limit exhaustion). Cheap fix: try/except around `start()` releasing the
  slot on failure.

## H4 — `_obj_mask_cache` pointer R/W without a lock; cross-thread use by the FF thread
- Original: low, race.
- Refutation attempt: I confirmed the cross-thread use is REAL — the FF bg
  thread reaches `_render_object_mask_cached` via `_compute_tick_cdn`
  (base:1756, called under `_feedforward_threaded` :2536) and via
  `_feedforward_clean_cdn` (base:2477, called from the anysplat path :3019),
  while the trainer thread invalidates the pointer at tick start (live:275) and
  after the rigid transform (base:2168). BUT: (1) the only thing shared via the
  pointer is a render-result tensor; (2) each actual *render* is under
  `_model_lock` (base:1676-1682), so no torn gauss_params; (3) pointer
  assignment/read is atomic under the GIL. Worst case is one extra mask render or
  a one-tick-stale coarse (+2%-scaled, dilated) mask — within the documented
  cache tolerance. No corruption path exists.
- Verdict: **confirmed** as a benign race (exactly as the audit rates it).
  Revised severity **low**. No action needed.

## H5 — Per-tick GPU allocations on the tracker critical path
- Original: low, allocation.
- Refutation attempt: `_batch_from_live_frame` does per-tick H2D copies
  (live:531,537,538) and `_render_object_mask_cached` rasterizes the object each
  cold-cache tick — all real, all inherent to a per-frame tracker, all freed each
  tick (no retained references → no leak). `DGS_TRACK_TRAJ_LOG` file-append
  (base:2173-2188) is opt-in (env-gated). Nothing here is a correctness hazard.
- Verdict: **confirmed** as a non-issue (perf note, not a bug). Revised severity
  **low** (arguably **none** as a hazard). No action.

## H6 — `_force_viewer_rerender` iterates NS-viewer state machines — dead branch under the live config
- Original: low, deadbranch.
- Refutation attempt: `_force_viewer_rerender` early-returns when
  `trainer.viewer_state is None` or not `ready` (base:1624-1632). Live config is
  `vis="tensorboard"` (dynamic_gs_config.py:227), so the NS viewer is never
  constructed → `viewer_state` is None → this is a no-op every tick. It is NOT
  live-path logic; it only matters under `--vis viewer`. Cheap guard, breaks no
  invariant.
- Verdict: **refuted** as a live-path hazard (dead/no-op under the documented
  config; reachable only with `--vis viewer`, which Invariant #9 forbids for
  live). Revised severity **low** (informational; harmless to keep).

## H7 — `request_render()` from trainer thread onto a possibly-closing viser server
- Original: low, lifecycle.
- Refutation attempt: `request_render` itself guards: `if
  self._stop_event.is_set(): return` (viser_direct.py:620-622), and `is_closing`
  is exactly `self._stop_event.is_set()` (:609). So a late per-tick call after
  teardown is a no-op, not a raise. `set_background_image` on a closing client is
  caught in `_render_once`. The teardown "cannot schedule new futures" noise is
  the known cosmetic issue, not from this call site.
- Verdict: **refuted** as a hazard (the call is already guarded; the audit even
  notes this). Revised severity **low**/**none**. No action.

---

## Bottom line
- Refuted as a live-path hazard: **H6** (dead branch under tensorboard),
  **H7** (already guarded).
- Confirmed but downgraded to low (real, contained/benign/rare): **H1**
  (documented staleness, not corruption), **H3** (real unguarded `start()`, but
  fires only on OS thread exhaustion), **H4** (benign GIL-atomic pointer race),
  **H5** (perf note, no correctness issue).
- Uncertain / latent-not-reachable: **H2** — the unlocked read + reallocation are
  both real, so I will NOT false-clear it; but it is genuinely UNREACHABLE in the
  default live config (D0-only + picker-off). It becomes a real high-severity tear
  only if a future change removes the `is_first` FF gate or enables the picker.
  Flagged for the human; the one-line lock wrap makes it unconditionally safe.
