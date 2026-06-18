# RUNTIME_viser_push — adversarial re-verification (SKEPTIC pass)

Re-checking each hazard in `RUNTIME_viser_push.md` against the ACTUAL live config
(`dynamic-gs-live`, `vis="tensorboard"`, viser-direct, anysplat FF). Stance: try to
REFUTE. Every verdict carries code evidence (file:line). Source NOT modified.

Live-config facts established up front (they decide most severities):
- `dynamic_gs_config.py:218-220` — live model uses `camera_optimizer=CameraOptimizerConfig(mode="SO3xR3")`
  and `output_depth_during_training=True`.
- `dynamic_gs_pipeline_live.py:128-129` — `model.set_phase("dynamic")` at init, so
  `self.phase == "dynamic"` for the whole live run.
- `dynamic_gs_model.py:1260` — `_should_apply_camera_optimizer` short-circuits to
  `False` when `self.phase != "static"`. So in dynamic phase the camera optimizer is
  NEVER applied, regardless of `self.training`.
- `use_bilateral_grid` default `False` (upstream SplatfactoModelConfig; not overridden
  in `DynamicGSModelConfig`, grep shows no assignment) → `dynamic_gs_model.py:2336`
  branch never taken.
- `crop_box` only READ (`dynamic_gs_model.py:2223-2224`), never assigned in any dynamic
  path (grep: zero writers) → always `None` (it's the NS-viewer crop feature, OFF per
  invariant #9).

---

## H1 — `model.train()/eval()` flipped outside the lock while the render thread reads `self.training` — claimed MEDIUM/HIGH

**Original claim:** `_render_from_camera`/`_render_from_camera_at_scale` toggle `train()`
(`base:1696`) / `eval()` (`base:1701`) OUTSIDE the `with self._viser_lock_ctx()` block,
and the viser render thread reads `self.training` inside `get_outputs` under the lock.
Cross-thread read of an unsynchronized bool → wrong-mode render (camera_optimizer
applied, `RGB+ED` vs `RGB`, etc.).

**Refutation attempt — the RACE is real but every branch it could corrupt is INERT in
the live config:**
- The race STRUCTURE is confirmed: `_render_from_camera` flips train/eval outside the
  lock (`base:1695-1702`), and these run on the tracker thread (`base:1739` via
  `_compute_tick_cdn`) AND the FF bg thread (`base:2536,2677,3044,3065`); the viser
  render thread reads `self.training` inside `get_outputs` (`dynamic_gs_model.py:2219,
  2223,2268,2336,2346`) on an independent thread. So a torn read of the bool genuinely
  CAN happen. I cannot refute the race's existence.
- BUT every `self.training`-dependent branch the audit cites is a no-op here:
  - render_mode (`:2268`): `"RGB+ED" if output_depth_during_training or not self.training`
    — `output_depth_during_training=True` (config:220) makes it `RGB+ED` **regardless of
    training**. No effect.
  - camera_optimizer (`:1260,2221`): gated on `phase=="static"`; live phase is
    `"dynamic"` → never applied. The audit's headline worst case ("camera_optimizer pose
    correction applied during a viewer frame") is UNREACHABLE.
  - bilateral grid (`:2336`): `use_bilateral_grid=False`. No effect.
  - crop_box (`:2223`): always `None`. No effect.
  - `assert camera.shape[0]==1` (`:2219`): viser camera is always batch-1
    (`viser_direct.py:154-163`). Harmless even if observed.
  - background expand (`:2346`): only reshapes the `background` output key; the viser
    path reads only `outputs.get("rgb")` (`viser_direct.py:691`). No visual effect.

**Verdict: uncertain** (race exists, impact inert). The unsynchronized cross-thread bool
read is genuine and unguarded — I will not false-clear it as nonexistent. But in the
LIVE config every consuming branch is a no-op, so the realistic effect is zero, not the
"wrong-mode render" the audit implies. Real-but-inert.
**Revised severity: low** (was medium/high). Fix is still cheap and correct (move the
toggles inside the lock or pass an explicit render-mode), but it is a latent/hygiene
issue, not a live correctness bug. Flagged as uncertain because a future config change
(turning bilateral on, or running this render code while `phase=="static"`) would make
it bite — so it should not be dismissed outright.

---

## H2 — `_initial_camera_applied` set mutated from two threads without a lock — claimed LOW

**Original claim:** `viser_direct.py:486` (tracker via `set_initial_camera`), `:569`
(viser pool via connect) `.add()`; `:578` `.discard()`; `:567/482` read — no lock around
the set, while neighbouring `_client_state` IS locked. Compound check-then-add not atomic.

**Refutation attempt:** Confirmed exactly as described. `set_initial_camera` reads
`_client_state` under `_client_state_lock` (`:478-483`) to build `pending`, then mutates
`_initial_camera_applied` OUTSIDE the lock (`:486`); `_on_client_connect` reads `:567`
and `.add()` `:569` outside any lock; `_on_client_disconnect` `.discard()` `:578` outside
any lock. The check-then-add across threads is genuinely unsynchronized. I cannot refute
it. The blast radius is correctly stated: at worst a client is snapped to the initial
camera twice, or a connect/`set_initial_camera` interleave double-applies/skips one snap
— a one-frame camera jump, never a crash, never a torn data structure (CPython `set.add`/
`discard` are individually GIL-atomic; only the compound check-then-act races).

**Verdict: confirmed** (real, reachable, unguarded) — but cosmetic.
**Revised severity: low** (unchanged). Accurately rated by the audit.

---

## H3 — `self._viser_direct_server` read-then-use across teardown (TOCTOU) — claimed LOW/MEDIUM

**Original claim:** atexit `_cleanup_viser_direct` sets the field to `None` (`base:629`)
while tracker/FF threads read it directly at `base:1525,1579` and call methods on it.

**Refutation attempt:**
- Teardown is `atexit`-registered (`base:476`), i.e. runs at interpreter exit AFTER the
  main ns-train loop (the tracker thread) has already returned. So the tracker-thread
  reader at `:1525/1579` is not actually concurrent with cleanup in the normal exit path
  — the same thread that drives `_push_viser_direct_transforms` is the one heading into
  atexit. The audit's "tracker thread reads while atexit nulls it" interleave does not
  occur on the main thread.
- The only thread that could still be live at atexit is the FF bg daemon
  (`_anysplat_bg_run`). Its call into `_viser_direct_register_ff_insert` is guarded:
  `base:1579` `is None` check and `base:1585` `is_closing` check
  (`is_closing == _stop_event.is_set()`, `viser_direct.py:609`), and `request_render`
  itself no-ops when `_stop_event` is set (`viser_direct.py:620-622`). The render thread
  is a daemon joined first in `close()` (`viser_direct.py:715-716`).
- The residual TOCTOU window the audit names (between the `is None` check at `:1525`/
  `:1579` and the method call) is real in principle, but: (a) `close()` does NOT null the
  field — only `_cleanup_viser_direct` does, at `base:629`, AFTER `srv.close()` already
  set `_stop_event`; so by the time the field could be nulled, `is_closing` is already
  True and the FF path early-returns at `:1585`; (b) a stale local ref calling a stub
  no-op (`add_ff_insert_chunk`/`push_tracker_transform`) or the `_stop_event`-guarded
  `request_render` cannot raise — the documented "cannot schedule new futures after
  shutdown" is fully defused by `_stop_event`/`is_closing`.

**Verdict: refuted** (as a live hazard). The dangerous interleave needs a non-main-thread
reader concurrent with field-nulling; the only such reader (FF bg) is guarded by
`is_closing`, and field-nulling happens strictly after `_stop_event` is set. The direct
`self._viser_direct_server` read at `:1525/1579` is a style wart (other sites snapshot to
a local), not a reachable crash.
**Revised severity: low** (was low/medium). The recommended snapshot-to-local fix is fine
hygiene but closes no real window in this config.

---

## H4 — render loop swallows ALL exceptions; CUDA error retried forever — claimed LOW

**Original claim:** `_render_loop` catches every exception (`viser_direct.py:640-643`),
prints only first 3 + every 50th; `_render_once` per-client `try/except continue`
(`:696-699`). A persistent failure spins / log-floods with no surfaced error.

**Refutation attempt:** Confirmed structurally — `_render_loop` `:638-643` catches `Exception`
and only rate-limits the print; `_render_once` `:696-699` `continue`s per client. But the
"spins forever at the request rate" framing is overstated: the loop is EVENT-DRIVEN, not
polling — it blocks on `self._render_requested.wait(timeout=1.0)` (`:632`) and only renders
when a request was set (`:635-637`). With no new `request_render`, it wakes at most once
per second (the 1.0 s timeout) and `continue`s (`:636`), so a persistent failure log-floods
at ~1 Hz worst case, not "up to the request rate." The lock IS released on exception (the
`with self.model_lock` block at `:689` unwinds), so no deadlock — the audit agrees. This is
a silent-failure / log-noise issue, not a correctness or liveness hazard.

**Verdict: confirmed** (real, reachable) — but benign; the "retried forever" rate is
bounded by the event wait, and there is no deadlock.
**Revised severity: low** (unchanged).

---

## H5 — per-tick `.cpu().numpy()` host copy on the push path even with no clients — claimed LOW (perf)

**Original claim:** `_push_viser_camera_feed` (`base:1546-1560`) does `arr.cpu().numpy()`
+ `np.ascontiguousarray` every tick even when no client connected; ~6.9 MB host copy per
tick at 1920×1200 on the tracker critical path.

**Refutation attempt:** Confirmed: `base:1551-1560` unconditionally builds `rgb_np` and
calls `srv.update_camera_feed(...)` with no `connected-clients` guard;
`update_camera_feed` (`viser_direct.py:342-351`) only ref-swaps under `_feed_lock` — the
copy already happened in the caller. There is no early-out keyed on
`srv._client_state` being empty. So the host copy IS paid every tick regardless of
clients. Pure perf, not correctness; the audit labels it LOW/perf correctly.

**Verdict: confirmed** (real, reachable) — perf only.
**Revised severity: low** (unchanged).

---

## H6 — `request_render()` collapses bursts to one render — explicitly NOT A BUG

**Original claim:** binary `Event` coalesces multiple requests into one render; by design.

**Refutation attempt:** Confirmed by design (`viser_direct.py:622,632,637`); the render
always pulls live model state, so coalescing loses nothing. Nothing to refute — the audit
itself flags this as not-a-bug.

**Verdict: refuted** (it is not a hazard; correctly self-classified).
**Revised severity: none.**

---

## H7 — `_refresh_feed_image` touches `self.server.gui` outside `_feed_lock` — claimed LOW

**Original claim:** `viser_direct.py:378-388` mutates `_feed_gui_image` / calls
`server.gui.add_image` outside any lock, render thread only.

**Refutation attempt:** Confirmed single-writer: `_feed_gui_image` is created and mutated
ONLY inside `_refresh_feed_image` (`viser_direct.py:379-385`), which is called ONLY from
`_render_once` (`viser_direct.py:672`) on the single render thread. The `_feed_rgb`/
`_feed_dirty` cross-thread state IS protected by `_feed_lock` (`:366-368`); the GUI handle
mutation is render-thread-private and viser serializes its own outbound queue. No
intra-field race. The audit's own conclusion ("benign as long as the render thread stays
the sole writer — it is") matches.

**Verdict: refuted** (benign; single-writer, no real race).
**Revised severity: low** (effectively none; audit already rated low).

---

## Dead/legacy-stub claims — spot check

The audit lists `push_tracker_transform`, `add_ff_insert_chunk`, `maybe_flush_ff_handle`,
`flush_pending_ff`, `refresh_static_handle` as no-op stubs. Confirmed: all return
immediately (`viser_direct.py:506-528`). `setup_handles` (`:494`) is partially live — it
forwards to `attach_model` + `set_initial_camera`. `_force_viewer_rerender` (`base:1614`)
early-returns because `_trainer.viewer_state` is None under invariant #9
(`base:1624-1632`). These are accurate; nothing to refute.

---

## Summary of verdicts

| Hazard | Original | Reachable (live) | Guarded | Verdict | Revised |
|---|---|---|---|---|---|
| H1 train/eval bool race | med/high | partial (race yes; effect no) | no | uncertain | low |
| H2 `_initial_camera_applied` | low | yes | no | confirmed | low |
| H3 server-ref TOCTOU | low/med | no (atexit after main; FF guarded) | yes (is_closing) | refuted | low |
| H4 swallow-all + retry | low | yes | n/a | confirmed | low |
| H5 per-tick host copy | low | yes | no | confirmed | low |
| H6 request coalesce | none | n/a | by design | refuted | none |
| H7 feed gui unlocked | low | yes | single-writer | refuted | low |

Net: the high end of the audit (H1 as medium/high; H3 as low/medium) does NOT hold in the
live config. H1's race is real but its every consequence is inert under
`output_depth=True / phase=dynamic / bilateral=off / crop_box=None`. H3's dangerous
interleave is closed by `is_closing` + atexit-after-main ordering. The genuine, reachable,
unguarded items (H2, H4, H5) are all correctly LOW and cosmetic/perf.
