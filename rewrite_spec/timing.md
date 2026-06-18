# timing.py — always-on timing + real-time metrics

> Added 2026-06-18 (co-designed). Resolves coherence-check §B: `timing.py` was cited as a
> dep with no spec. **Default-ON every run** (cheap), because the real-time question is
> answered by *what* and *how* we time, not by raw speed.

## 1. RESPONSIBILITY
Collect low-overhead per-stage timings + the headline real-time metrics across all 3
threads, and render one report at end of run. Never does per-tick I/O.

## 2. PUBLIC INTERFACE
- `with timing.stage(name): ...` — context manager; accumulates wall-time into `name`. Thread-safe (per-thread buffers, merged at render). The 99% path.
- `timing.stage_sync(name)` — same, but `cuda.synchronize()` inside the timer → **true GPU cost**. Opt-in via `DGS_TIME_SYNC=1` (off by default; sync stalls the pipeline). Mark sync'd rows in the report.
- `timing.frame(capture_ts, display_ts)` — records **end-to-end latency = display_ts − capture_ts** (the teleop metric, event-time).
- `timing.gauge(name, value)` — point-in-time value (e.g. gaussian count).
- `timing.event(name)` — counter (e.g. `frame_dropped`, `tracking_failed`, `ff_skipped`).
- `timing.lock_wait(seconds)` — records time spent *waiting* to acquire `_model_lock` (the blocking-vs-contention detector). Called by the lock wrapper, not by hand.
- `timing.render() -> str` / `timing.write(path)` — the report. `timing.reset()`.

## 3. DEPENDS ON
`config` (knobs + `DGS_TIME_SYNC`). Optional `torch` for CUDA-event timing in sync mode. Nothing else (leaf-ish utility).

## 4. CONSUMES / PRODUCES
Consumes: stage durations, frame timestamps, gauges, events, lock-waits.
Produces: `<data_dir>/timing_report.txt` (default), + the JSONL ledger for offline analysis.

## 5. THE 6 HEADLINE METRICS (top of every report)
1. **Effective tracker Hz + p99 tick** — are we real-time?
2. **Glass-to-display latency, p50/p99** (`display − capture`) — the teleop truth.
3. **Tracker lock-wait, p99** — *the* blocking-vs-contention detector: high → FF holds the lock too long (fixable); low → pure GPU contention (hardware).
4. **FF Hz + tracker-ticks-overlapped-per-FF** — proves FF is non-blocking (ticks must keep flowing during the ~400 ms).
5. **Gaussian count over time** — the bounded-growth watchdog (catches the zed_final 3M blowup live).
6. **Frames dropped** (camera frames the tracker skipped) — liveness under load.

Below these: the per-stage drill-down (XFeat extract/match/RANSAC, CDN, AnySplat infer/reproject/ICP/insert, snapshot, viz push) with min/avg/**p90/p99**/max/n.

## 6. HOW TO TIME CORRECTLY (the non-obvious rules — enforce in the impl)
- **Event-time, not processing-time.** Frame carries `capture_ts` (from the source/SHM); latency is measured against it, not against tick-start.
- **Tail, not mean.** Keep percentiles per stage (ring buffer or P²/t-digest). The mean hid the cudnn freeze (754 ms behind a 14 ms mean).
- **GPU is async** — un-synced wall timers measure *launch+queue*, not compute; good for relative trends, NOT absolute GPU cost. Absolute cost needs `stage_sync` (CUDA events). Report must say which a row is.
- **No per-tick I/O.** Accumulate in memory; write once at end. (The `DGS_FF_DEBUG` 140 ms/tick regression was inline I/O — never repeat it; debug I/O lives in `debug.py`'s writer thread.)
- **Cheap.** `time.perf_counter()`, in-memory accumulation, lock-free per-thread. Target < ~5 µs/stage so always-on is free.

## 7. THREADING
Each thread (tracker-main / FF-bg / viz) records into its **own** buffer (no contention); `render()` merges. `lock_wait` is recorded by the `_model_lock` wrapper so every acquisition is measured uniformly. The timer itself never blocks.

## 8. OPEN QUESTIONS
- Percentile method under always-on constraint: fixed ring buffer (simple, bounded memory) vs streaming t-digest (exact-ish tail, more code)? Recommend ring buffer (last N per stage).
- Stream the 6 headline metrics to the live viser panel each second (cheap gauges), or report-at-end only? Recommend a tiny live readout — it's how the operator *sees* real-time degradation as it happens.
- Should `frame_dropped` be counted in the source adapter (knows camera rate) or the loop (knows what it processed)? Both — adapter counts produced-but-overwritten SHM slots; loop counts skipped seqs.
