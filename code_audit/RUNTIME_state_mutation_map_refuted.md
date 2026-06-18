# LIVE dynamic-gs — Runtime State Mutation Map: ADVERSARIAL RE-VERIFICATION

Skeptic pass over `RUNTIME_state_mutation_map.md`. Goal: try to **refute** each hazard against the
actual `feedforward_dev` source. Verdict = `confirmed` (real + reachable + unguarded), `refuted`
(unreachable / already-guarded / dead), or `uncertain` (cannot prove either way). Evidence by
`file:line`.

Live config facts established (`dynamic_gs_config.py` + model/base defaults), used throughout:
- `vis="tensorboard"` (config.py:227) → NS viewer OFF (Invariant #9). viser-direct is the only renderer.
- `enable_feedforward_inpaint="anysplat_decode"` (base:176) → anysplat FF path is the live default.
- `feedforward_recurring_every_n_ticks=10` (base:187).
- `interactive_object_selection=False` (base:152) → picker inert.
- `xfeat_pose_filter_enabled=False` (model:313) — KF off by default (operator may flip on; irrelevant to these hazards).
- `_model_lock` is one shared `threading.RLock` (base:474), handed to the model
  (`attach_render_lock`, base:519-520) and the viser server (`server.model_lock = self._model_lock`, base:1480).

---

## H1 — `_obj_mask_cache` invalidated unlocked while FF bg may read/set it  (orig: race, medium)

**Original claim:** T-thread stores `_obj_mask_cache=None` unlocked at tick start (live:275) and after a
rigid transform (base:2168), racing the FF bg thread's `is None` check + locked render
(base:1666-1683). Benign data race that can silently break the "render once per tick" guarantee.

**Refutation attempt:**
- The race is REAL as a Python-attribute data race: invalidate sites at `live:275`, `base:1171`,
  `base:2168` take no lock; the populate/read path at `base:1666-1683` takes the lock only AROUND the
  render, not around the `if self._obj_mask_cache is None:` test.
- Could not refute on "unreachable" — FF runs concurrently with new ticks by design (FF is on a bg
  thread, base:2513; tick loop keeps running).
- Could not refute on "already guarded" — the `None`-store and the `is None`-read are genuinely
  outside the lock.
- BUT the audit's own severity is right-sized down: it CANNOT tear a tensor. The actual render is
  always under `_model_lock` (base:1677-1682), so the worst outcome is FF re-rendering a fresh mask or
  using a mask one tick stale against a slightly different camera. No crash, no torn `gauss_params`.

**Verdict: confirmed** (the race exists and is unguarded) **but severity → low.** It is a benign
attribute race with no memory-safety consequence; the only victim is the "render once, reuse
everywhere" optimization, which degrades to "render twice" in the rare interleave.

---

## H2 — CDN re-render after cull runs WITHOUT `_model_lock`  (orig: race, medium→high)

**Original claim:** `cdn_new = self._compute_tick_cdn(camera, batch)` at base:2885 sits AFTER the
`with _viser_lock_ctx():` cull block closes at base:2880, so the re-render is unlocked.

**Refutation attempt:**
- The re-render at base:2885 IS textually outside the cull's `with` block (cull at base:2876-2880,
  re-render at base:2885) — confirmed.
- BUT `_compute_tick_cdn` → `_render_from_camera` (base:1739) re-acquires the lock internally:
  `with self._viser_lock_ctx(): return self.model.get_outputs(...)` at base:1698-1699. So the render
  kernel itself runs under `_model_lock`. The cull (delete, base:2877) and the re-render are two
  SEPARATE lock acquisitions, but the gap between them holds no torn state — it's just plain Python.
- The only other `gauss_params` writers are (a) the trainer's rigid transform, which is locked
  (base:2163-2164), and (b) another FF insert/delete — impossible here because FF is single-in-flight
  via `_anysplat_slot_lock` (base:2509, held for the whole `_feedforward_threaded`). The viser render
  thread is read-only and also locks (viser_direct:688 `with self.model_lock`).
- So no concurrent writer can interleave in the unlocked gap, and every reader/writer that touches
  `gauss_params` holds the same RLock. The audit itself concedes "Net: actually safe today."

**Verdict: refuted** (as a live correctness hazard) — the re-render re-locks internally and FF
single-in-flight + the locked rigid transform close the gap. **Severity → low** (latent/fragile only:
it would tear ONLY if a future change adds an unlocked `gauss_params` writer; that's a maintenance
note, not a current race).

---

## H3 — `_anysplat_slot_lock` acquire on T thread, release on FF thread; leak if `Thread.start()` raises  (orig: lock, low)

**Original claim:** acquire at base:2509 (T), release in `_feedforward_threaded` finally at base:2542
(FF). If `threading.Thread(...).start()` (base:2513) raises before the thread body runs, the lock is
never released → all future FF dispatches no-op forever (base:2509-2511), FF silently dies; tracking
continues. `_cleanup_anysplat_bg` (base:650-660) then blocks up to 60 s at shutdown.

**Refutation attempt:**
- Cross-thread acquire/release of a plain `threading.Lock` is legal (base:465 `threading.Lock()`),
  and the `finally` release tolerates a stray double-release via the bare `except RuntimeError: pass`
  (base:2543-2544) — so the normal path is fine. Could not refute the legality.
- The leak structure is REAL: between `acquire` (base:2509) and the thread body's `try/finally`
  (base:2530-2544), the ONLY statements are `self._last_feedforward_wall_time = time.time()`
  (base:2512) and `threading.Thread(...).start()` (base:2513). `Thread.start()` raising
  (`RuntimeError: can't start new thread`, an OS thread-table-exhaustion failure) leaves the slot lock
  held with no owner thread to release it. There is no try/except around the `.start()` to release on
  failure.
- Reachability: this requires the process to be unable to spawn an OS thread. In this pipeline FF
  spawns at most one daemon thread at a time (single-in-flight), so thread-table exhaustion from FF
  alone is essentially impossible; it would require an unrelated thread leak elsewhere. So it is real
  but extremely-rare — the H-CROP-class "real structure that almost never fires."
- `_cleanup_anysplat_bg` 60 s shutdown block (base:658) is correctly bounded by `timeout=60.0`, so
  even a wedged lock only delays exit, not hangs it.

**Verdict: confirmed** (the lock-leak-on-start-failure is a real unguarded structure) **but severity →
low** — fires only on OS thread-creation failure, which this single-in-flight FF design makes
near-impossible to trigger on its own. Worth a one-line try/except, not urgent.

---

## H4 — `apply_rigid_object_transform` (legacy variant) takes no lock  (orig: lock, low)

**Original claim:** `dynamic_gs_model.py:922-939` writes `gauss_params["means"/"quats"]` under
`@torch.no_grad` but with NO `_model_lock`. Live uses `apply_rigid_object_transform_from_reference`
(model:987), which the caller wraps (base:2163). Legacy variant appears to have no live caller; purge
candidate.

**Refutation attempt:**
- `grep -rn "apply_rigid_object_transform\b" dynamic_gs/ | grep -v _from_reference` returns ONLY the
  definition at model:923 — zero callers anywhere in the tree (not just live). Confirmed dead.
- The live tick path uses `apply_rigid_object_transform_from_reference` (base:2164), which is wrapped
  by `with self._viser_lock_ctx():` at base:2163. So the LIVE write site is correctly locked
  (model:1006-1007 under that lock).
- Since the legacy method is never called, it cannot race anything — the "takes no lock" property is
  inert.

**Verdict: refuted** (as a hazard) — the method is dead code (no caller), so its missing lock can
never fire. **Severity → none.** It IS a legitimate purge candidate (dead public method superseded by
the `_from_reference` variant), but that is hygiene, not a runtime hazard.

---

## H5 — per-tick & per-FF tensor allocations / Parameter realloc churn  (orig: allocation, medium)

**Original claim:** each FF insert (model:1216-1221) does `torch.cat` over all six gauss_params; each
delete (model:1135-1137) reallocates all six via boolean `[keep]`; `_refresh_gaussian_optimizers`
clears optimizer state + re-registers the means hook (model:1078). With FF every 10 ticks +
accumulating inserts (459k→1.29M), each FF call copies the whole growing param set twice (cull +
insert) under the lock, contending with the tracker render.

**Refutation attempt:**
- The realloc pattern is exactly as described: insert `torch.cat([old.detach(), new])` for all six at
  model:1216-1221; delete `self.gauss_params[name] = torch.nn.Parameter(sliced)` over the six at
  model:1135-1137; both call `_refresh_gaussian_optimizers(reset_means_optimizer=True)` (model:1146,
  1227) which re-registers the grad hook (model:1078). All confirmed.
- These run under `_model_lock` (insert at base:3497-3498, cull at base:2876) on the FF bg thread, so
  the tracker tick's locked reads (object-mask render base:1677, CDN render base:1698) block on them —
  the documented GPU/lock contention.
- Could not refute: this is a real, reachable, by-design allocation cost. But it is NOT a
  correctness/race hazard — it is a performance characteristic. The audit itself scopes it "out of
  scope for a pure race audit."

**Verdict: confirmed** (the allocation churn is real and reachable) **but severity → low** for a state-
mutation/race lens — there is no data corruption or crash, only throughput cost already tracked in
CLAUDE.md (the dynamic-phase purge TODO). Reclassify as perf, not hazard.

---

## H6 — `_latest_tracker_frame` dict shared T→FF without lock  (orig: race, low)

**Original claim:** published unlocked at live:323 (T), read by FF at dispatch; FF mutates
`target_frame["cdn"]` at base:2536. Safe because dispatch passes the current reference, FF is single-
in-flight, and T rebinds `self._latest_tracker_frame` to a new dict next tick (doesn't mutate the in-
flight one). Shared `camera`/`batch` tensors are read-only in FF.

**Refutation attempt:**
- `self._latest_tracker_frame = {...}` at live:323 rebinds the attribute to a freshly-constructed dict
  every tick — it never mutates the previously-published dict in place. Confirmed.
- The dispatch hands FF the current dict reference (`_on_tracker_frame` → `_dispatch_feedforward_async`
  receives `target_frame`), so the in-flight FF holds its own object; a later T rebind is invisible to
  it. FF's one write, `target_frame["cdn"] = ...` (base:2536), is on the dict FF exclusively owns for
  that call (single-in-flight via the slot lock).
- The shared tensors (`camera`, `batch`) are only read by FF (renders, depth filter, decode) — no FF
  write-back into them. So no torn write.

**Verdict: refuted** (effectively safe) — the publish is a single-writer attribute rebind, the
consumer gets an immutable-per-call snapshot, and FF only reads the shared tensors. **Severity → low**
(the audit already rated it low; concur, leaning toward none). Document-only.

---

## H7 — viser GUI / `request_render` from callback threads vs teardown  (orig: lifecycle, low — already guarded)

**Original claim:** `request_render` (viser_direct:611) and `_viser_direct_register_ff_insert`
(base:1571) both check `is_closing`/`_stop_event` before submitting, avoiding "cannot schedule new
futures after shutdown". `add_ff_insert_chunk`/`flush_pending_ff`/`refresh_static_handle`/
`push_tracker_transform`/`maybe_flush_ff_handle` are NO-OP stubs. Guarded correctly.

**Refutation attempt (i.e. try to find an UNGUARDED path):**
- `request_render` guards: `if self._stop_event.is_set(): return` (viser_direct:619-621). Confirmed.
- `_render_once` guards: `if self._stop_event.is_set(): return` (viser_direct:686-687) and renders
  under `with self.model_lock, torch.no_grad():` (viser_direct:688-689). Confirmed.
- `_viser_direct_register_ff_insert` guards on `is_closing` (base:1585-1586) and wraps both the stub
  call and `request_render` in try/except (base:1587-1594). Confirmed.
- The five stubs (viser_direct:506-528) all `return` immediately — verified each body is a no-op.
- Could NOT find an unguarded teardown-race push path.

**Verdict: refuted** (not a hazard — already guarded as the original note itself concluded).
**Severity → low/none.** The dead stubs are harmless purge candidates.

---

## H8 — interactive picker BLOCKS the trainer thread  (orig: lifecycle, low; default OFF)

**Original claim:** `_wait_for_selection` blocks the trainer up to `object_selection_timeout_s`. An
in-flight FF can still insert under the lock during the block (picker doesn't hold `_model_lock` while
waiting). Default `interactive_object_selection=False`, so inert.

**Refutation attempt:**
- Default confirmed `interactive_object_selection: bool = False` (base:152), and the live config never
  overrides it (config.py LiveDynamicGSPipelineConfig has no such field). So `_tick_interactive_selection`
  is never entered in a normal live run (gated at live:287).
- Even when enabled, the block holds `_model_lock` only during the `_reseed_tracked_object` `copy_`
  (base:1164), not during the wait — so a concurrent FF insert is fine.

**Verdict: refuted** (inert in the default live config; benign even when on). **Severity → none** for
the shipped config.

---

## H9 — unreachable / dead branches on the live path  (orig: deadbranch, info)

**Per-item refutation attempt (here "refuted" = confirmed dead/inert as claimed):**
- `_force_viewer_rerender` (base:1614): returns early when `trainer.viewer_state` is None or
  `not ready` (base:1624-1632). With `vis="tensorboard"` there is no NS `viewer_state`, so it no-ops
  every tick. Confirmed inert.
- `get_outputs_for_camera` opacity-hide branch (model:708-739): grep shows the only callers are its
  own internal `super().get_outputs_for_camera` lines (model:724/728/736) + the NS-viewer comment at
  base:512. viser-direct calls `model.get_outputs(camera)` (viser_direct:690), NOT
  `get_outputs_for_camera`. So in live the whole method (and its `_render_lock_ctx`) is never invoked.
  Confirmed inert.
- `_mask_means_grad` (model:825-828): returns early/zeros whenever `enable_cotracker_rigid_motion`
  (default True, model:209) — and the trainer fast path skips backward, so the registered hook
  (model:665/1078) never fires in live. Confirmed inert (still re-registered on every insert/delete,
  harmless).
- `apply_rigid_object_transform` legacy variant (model:923): zero callers (see H4). Confirmed dead.
- viser stubs (viser_direct:506-528): no-ops (see H7). Confirmed dead-but-harmless.

**Verdict: refuted** (these are correctly identified as dead/inert on the live path — not live
hazards). **Severity → none.** Valid purge candidates (informational), not runtime risks.

---

## Summary of revised verdicts

| Hazard | Orig severity | Verdict | Revised severity | One-line reason |
|---|---|---|---|---|
| H1 | medium | confirmed | low | real unlocked attribute race, but render is locked → can't tear; only loses "render once" |
| H2 | medium→high | refuted | low | re-render re-locks internally (base:1698); FF single-in-flight + locked rigid transform close the gap |
| H3 | low | confirmed | low | real lock-leak if `Thread.start()` raises, but needs OS thread-exhaustion → near-impossible here |
| H4 | low | refuted | none | legacy method has zero callers — dead code, lock-absence inert |
| H5 | medium | confirmed | low | real realloc churn but perf-only, no corruption/crash |
| H6 | low | refuted | low | single-writer attribute rebind + per-call snapshot; FF reads shared tensors read-only |
| H7 | low | refuted | low | already guarded (`_stop_event`/`is_closing` + try/except); could not find an unguarded path |
| H8 | low | refuted | none | picker default OFF in live config; benign even when on |
| H9 | info | refuted | none | dead/inert branches correctly identified; purge candidates, not hazards |

**Net:** the one load-bearing safety primitive — every `gauss_params` reader/writer across the three
threads holding the shared `_model_lock` RLock — holds up under scrutiny. No memory-safety race was
confirmed reachable+unguarded. The two surviving "confirmed" items (H1, H3) are both low-severity
real-but-rare or benign-by-construction. H2/H4/H6/H7/H8/H9 are refuted as live hazards. Highest-value
ACTION items unchanged from the audit: a one-line try/except releasing `_anysplat_slot_lock` on
`Thread.start()` failure (H3), and optionally owning `_obj_mask_cache` per-FF-snapshot (H1) — both
hardening, neither urgent.
