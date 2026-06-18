# debug.py — opt-in diagnostic sink (off the hot path)

> Added 2026-06-18 (co-designed). A first-class, opt-in capture sink so a user whose
> pipeline misbehaves can flip one flag, reproduce, and send you a folder. Replaces the
> scattered `if DGS_FF_DEBUG:` inline branches — which the audit found cost **140 ms/tick
> on the hot path** because they wrote synchronously.

## 1. RESPONSIBILITY
When enabled, capture per-tick / per-FF-call diagnostic artifacts (masks, rendered frames,
change masks, scene stats, the timing report) to a single zippable folder — doing **all
I/O off the hot path** so enabling debug never changes tracker timing materially.

## 2. PUBLIC INTERFACE
- `debug.enabled() -> bool` — `os.environ.get("DGS_DEBUG","0") != "0"`. The hot path guards every capture with this (cheap bool).
- `debug.capture(group, name, kind, data)` — non-blocking enqueue onto a bounded queue; a daemon writer thread persists. `kind ∈ {image, array, json, text}`. No-op + ~0 cost when disabled. **Drops oldest if the queue is full** (never blocks the tracker — the 140 ms lesson).
- `debug.tick(seq)` / `debug.ff_call(call_id)` — open a numbered group so files sort in pipeline order (mirrors the old `call_NNNN_frame_MMMMMM_K_NAME.png` ordering that was useful).
- `debug.finalize()` — drain queue, write a `manifest.json` (config snapshot + env + the final `timing_report.txt`), zip the folder to `<data_dir>/debug_<stamp>.zip`.
- **Registry:** `debug.register(name, fn)` — users/devs add a capture function in ONE place; the pipeline calls registered fns at its capture points. New diagnostics = one function, zero hot-path edits.

## 3. DEPENDS ON
`config` (the `DGS_DEBUG` flag + paths), `timing` (bundles the report into the dump). Reads `frame`/`GaussianSnapshot` types it's handed — does not import scene logic.

## 4. CONSUMES / PRODUCES
Consumes: artifacts handed in at capture points (gripper/object masks, real rgb, rendered-pre-cull, rendered-post-cull, raw change mask, clean change mask, gaussian count, tracker inliers, lock-wait spikes).
Produces: `<data_dir>/debug_<stamp>/` (+ `.zip`) with ordered images + `manifest.json` + `timing_report.txt`.

## 5. THREADING
**One daemon writer thread** owns all disk I/O. Producers (tracker-main, FF-bg) only `capture()` = enqueue (µs, lock-free, drop-oldest). The writer is bounded; if it falls behind, captures drop (a gap in the dump) rather than stalling the pipeline. `finalize()` joins the writer at shutdown.

## 6. THE DEFAULT CAPTURE SET (registered, matching the old ff_debug that proved useful)
Per FF call: `1_gripper_mask 2_object_mask 3_real 4_rendered_pre_cull 5_rendered_post_cull 6_raw_change_mask 7_clean_change_mask` + the dispatch/insert counts. Per N ticks (configurable): tracked-frame rgb + the rendered scene + inlier count. Always at finalize: the timing report + config manifest.

## 7. OPEN QUESTIONS
- Default capture cadence when enabled: every FF call (as before) + every Nth tick? Make `N` a config knob (`DGS_DEBUG_TICK_EVERY`, default e.g. 30).
- Cap the dump size (ring of last K FF calls) so a long run doesn't fill disk — recommend yes, keep last ~50 FF calls.
- Should `debug.capture` of a GPU tensor copy to CPU on the producer (adds ~ms to the hot path) or hand the GPU ref to the writer (writer does the copy, but the tensor may be mutated by then)? Recommend: producer does a cheap detached `.cpu()` only when `enabled()` — accept the small cost since debug is opt-in.
