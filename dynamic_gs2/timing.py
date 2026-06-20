"""timing.py — always-on, low-overhead per-stage timing for the dynamic pipeline.

Per the rewrite_spec (rewrite_spec/timing.md): cheap enough to leave on every run
(time.perf_counter + in-memory ring buffers, lock-free per-thread), percentiles not means
(the mean hid the cudnn 754ms freeze behind a 14ms average), and NO per-tick I/O — the
report is rendered once at the end.

Currently wired into the FEEDFORWARD path only (the FF worker bg thread + the tracker-tick
marker from the main loop), which is the path whose latency we need to see first. The
interface is general so the tracker/viser threads can record into the same ledger later
without rework.

Two things it answers for the FF:
  1. Per-step timing (aggregate mean/p90/p99/max/n + a per-FF-cycle breakdown).
  2. Tracker-tick INTERLEAVING — every tracker tick stamps a point event; at render each
     tick is placed inside whichever FF step interval contained it, so you can see "the
     tracker ticked 3x during the AnySplat decode of FF cycle 7" directly.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ring-buffer cap per stage — bounds memory on a long live session (last N samples kept).
_RING = 4096


@dataclass
class _Interval:
    """One timed FF step: name, [t0, t1] wall window, the FF cycle it belongs to, and the
    thread that ran it. Wall times are perf_counter seconds (monotonic, process-relative).
    `reported` rows (e.g. the worker's self-reported forward time) carry a known DURATION but
    no real wall position — they appear in the aggregate table but are kept OUT of the per-cycle
    timeline + tick-interleave so they can't create phantom overlaps."""
    name: str
    t0: float
    t1: float
    cycle: int
    thread: str
    reported: bool = False


@dataclass
class _Event:
    """A point-in-time marker (e.g. a tracker tick, an ff_skipped). `info` carries the tick
    number / reason for the interleave + counters."""
    name: str
    t: float
    thread: str
    info: dict = field(default_factory=dict)


class TimingLedger:
    """Thread-safe accumulator. Each record is appended under a tiny lock (append is O(1));
    the heavy work — percentiles, interleave placement — happens once in render()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intervals: List[_Interval] = []
        self._events: List[_Event] = []
        self._gauges: Dict[str, List[tuple]] = defaultdict(list)   # name -> [(t, value), ...]
        self._counters: Dict[str, int] = defaultdict(int)
        self._t_origin = time.perf_counter()
        self._cur_cycle = -1                                       # set by cycle(); -1 = outside any cycle

    # ---- recording API (called from any thread; cheap) ----
    def stage(self, name: str, cycle: Optional[int] = None) -> "_StageCtx":
        """Context manager: `with ledger.stage('decode.reproject'): ...` accumulates the wall
        window into `name`, tagged with the current FF cycle (or an explicit one)."""
        return _StageCtx(self, name, self._cur_cycle if cycle is None else cycle)

    def cycle(self, cycle_id: int) -> None:
        """Mark the start of an FF cycle; subsequent stage() calls on THIS thread inherit it.
        (FF is single-in-flight, so one writer of _cur_cycle at a time — no contention.)"""
        self._cur_cycle = cycle_id

    def event(self, name: str, **info) -> None:
        """A point marker at 'now' (tracker tick, ff_skipped, frame_dropped). Also bumps a
        same-named counter so the report can show totals without scanning."""
        t = time.perf_counter()
        with self._lock:
            self._events.append(_Event(name, t, threading.current_thread().name, dict(info)))
            self._counters[name] += 1
            if len(self._events) > _RING * 4:
                self._events = self._events[-_RING * 4:]

    def gauge(self, name: str, value: float) -> None:
        """Record a point-in-time scalar over time (e.g. gaussian count after each insert)."""
        t = time.perf_counter()
        with self._lock:
            g = self._gauges[name]
            g.append((t, float(value)))
            if len(g) > _RING:
                del g[: len(g) - _RING]

    def record_ms(self, name: str, dur_ms: float, cycle: Optional[int] = None) -> None:
        """Record a step whose duration is KNOWN but happened ELSEWHERE (the AnySplat worker's
        self-reported forward time, measured in the subprocess). It feeds the aggregate per-step
        table but is flagged `reported` so it stays OUT of the per-cycle timeline + tick interleave
        (it has no real wall position on this thread — back-dating it would fake overlaps)."""
        t1 = time.perf_counter()
        self._record_interval(name, t1 - dur_ms / 1000.0, t1,
                              self._cur_cycle if cycle is None else cycle,
                              threading.current_thread().name, reported=True)

    def _record_interval(self, name: str, t0: float, t1: float, cycle: int, thread: str,
                         reported: bool = False) -> None:
        with self._lock:
            self._intervals.append(_Interval(name, t0, t1, cycle, thread, reported))
            if len(self._intervals) > _RING * 8:
                self._intervals = self._intervals[-_RING * 8:]

    # ---- report ----
    def render(self) -> str:
        with self._lock:
            intervals = list(self._intervals)
            events = list(self._events)
            counters = dict(self._counters)
            gauges = {k: list(v) for k, v in self._gauges.items()}
        return _render_report(intervals, events, counters, gauges, self._t_origin)

    def write(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(self.render())

    def reset(self) -> None:
        with self._lock:
            self._intervals.clear(); self._events.clear()
            self._gauges.clear(); self._counters.clear()
            self._cur_cycle = -1
            self._t_origin = time.perf_counter()


class _StageCtx:
    __slots__ = ("_led", "_name", "_cycle", "_t0")

    def __init__(self, led: TimingLedger, name: str, cycle: int) -> None:
        self._led = led; self._name = name; self._cycle = cycle

    def __enter__(self) -> "_StageCtx":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        t1 = time.perf_counter()
        self._led._record_interval(self._name, self._t0, t1, self._cycle,
                                   threading.current_thread().name)
        return False                                              # never swallow exceptions


# ----------------------------------------------------------------- rendering helpers
def _pct(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


# The FF step order — defines the row order in the aggregate table + per-cycle breakdown so the
# report reads top-to-bottom in pipeline order regardless of which stages fired.
_FF_STEP_ORDER = [
    "cdn.render", "cdn.clean",
    "cull_infront.compute", "cull_infront.hide",
    "recdn.render", "recdn.gate",
    "decode.icp", "decode.crop_ipc", "decode.worker_forward", "decode.reproject",
    "decode.density_shape", "decode.clamp",
    "enforce_ceiling", "cull_replaced.compute", "surgery.cull_insert",
    "ff_debug.dump",
]


def _render_report(intervals, events, counters, gauges, t_origin) -> str:
    ms = lambda t: (t - t_origin) * 1000.0
    out: List[str] = []
    out.append("=" * 78)
    out.append("FEEDFORWARD TIMING REPORT  (perf_counter, ms; un-synced wall = launch+queue,")
    out.append("not absolute GPU cost — set DGS_TIME_SYNC for true GPU time when implemented)")
    out.append("=" * 78)

    # ---- counters (headline events) ----
    if counters:
        out.append("")
        out.append("EVENTS / COUNTERS")
        for k in sorted(counters):
            out.append(f"  {k:28s} {counters[k]}")

    # ---- tracker Hz (headline metric #1) — from the tracker_tick markers ----
    tk = sorted([e.t for e in events if e.name == "tracker_tick"])
    if len(tk) >= 2:
        span = tk[-1] - tk[0]
        hz = (len(tk) - 1) / span if span > 0 else 0.0
        dts = sorted((tk[i + 1] - tk[i]) * 1000.0 for i in range(len(tk) - 1))   # inter-tick ms
        out.append("")
        out.append("TRACKER RATE  (headline #1 — are we real-time?)")
        out.append(f"  effective Hz                   {hz:.1f}  ({len(tk)} ticks over {span:.1f}s)")
        out.append(f"  inter-tick ms  p50/p90/p99/max  "
                   f"{_pct(dts,0.5):.1f} / {_pct(dts,0.9):.1f} / {_pct(dts,0.99):.1f} / {max(dts):.1f}")

    # ---- aggregate per-step table (ALL intervals incl. worker-reported) ----
    by_step: Dict[str, List[float]] = defaultdict(list)
    for iv in intervals:
        by_step[iv.name].append((iv.t1 - iv.t0) * 1000.0)
    # The per-cycle timeline + tick interleave use only REAL wall-positioned intervals (drop the
    # worker-reported rows, which have a duration but no true position on this thread).
    wall_intervals = [iv for iv in intervals if not iv.reported]
    ff_cycles = sorted({iv.cycle for iv in wall_intervals if iv.cycle >= 0})
    out.append("")
    out.append(f"AGGREGATE PER-STEP   (over {len(ff_cycles)} FF cycle(s))")
    out.append(f"  {'step':24s} {'n':>4s} {'mean':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
    ordered = [s for s in _FF_STEP_ORDER if s in by_step] + \
              [s for s in sorted(by_step) if s not in _FF_STEP_ORDER]
    for step in ordered:
        v = sorted(by_step[step])
        mean = sum(v) / len(v)
        out.append(f"  {step:24s} {len(v):>4d} {mean:>8.1f} {_pct(v,0.9):>8.1f} "
                   f"{_pct(v,0.99):>8.1f} {max(v):>8.1f}")

    # ---- tracker-tick interleave: aggregate (spec metric #4) ----
    tick_ev = sorted([e for e in events if e.name == "tracker_tick"], key=lambda e: e.t)
    cyc_intervals: Dict[int, List[_Interval]] = defaultdict(list)
    for iv in wall_intervals:
        if iv.cycle >= 0:
            cyc_intervals[iv.cycle].append(iv)
    if ff_cycles:
        overlaps = []
        for c in ff_cycles:
            ivs = cyc_intervals[c]
            c0 = min(i.t0 for i in ivs); c1 = max(i.t1 for i in ivs)
            overlaps.append(sum(1 for e in tick_ev if c0 <= e.t <= c1))
        avg_ov = sum(overlaps) / len(overlaps)
        out.append("")
        out.append("TRACKER INTERLEAVE (proves FF is non-blocking — ticks must keep flowing)")
        out.append(f"  tracker ticks total            {len(tick_ev)}")
        out.append(f"  avg tracker-ticks per FF cycle {avg_ov:.1f}  (0 => FF blocks the tracker)")

    # ---- per-cycle breakdown (with tick interleave placed inside each step) ----
    if ff_cycles:
        out.append("")
        out.append("PER-FF-CYCLE BREAKDOWN  (steps in pipeline order; +Xms = offset from cycle start;")
        out.append("  [tick #N @ +Yms] = a tracker tick landed inside that step)")
        for c in ff_cycles:
            ivs = sorted(cyc_intervals[c], key=lambda i: i.t0)
            c0 = ivs[0].t0
            c1 = max(i.t1 for i in ivs)
            total = (c1 - c0) * 1000.0
            n_ticks = sum(1 for e in tick_ev if c0 <= e.t <= c1)
            out.append("")
            out.append(f"  ── FF cycle {c}  (wall {total:.1f}ms, {n_ticks} tracker tick(s) overlapped) ──")
            for iv in ivs:
                dur = (iv.t1 - iv.t0) * 1000.0
                off = (iv.t0 - c0) * 1000.0
                inside = [e for e in tick_ev if iv.t0 <= e.t <= iv.t1]
                tick_str = ""
                if inside:
                    marks = ", ".join(
                        f"#{e.info.get('tick','?')} @ +{(e.t - iv.t0)*1000.0:.0f}ms" for e in inside)
                    tick_str = f"   [tick {marks}]"
                out.append(f"     +{off:7.1f}  {iv.name:24s} {dur:8.1f}ms{tick_str}")

    # ---- gaussian-count watchdog ----
    if "gaussian_count" in gauges and gauges["gaussian_count"]:
        g = gauges["gaussian_count"]
        out.append("")
        out.append("GAUSSIAN COUNT (bounded-growth watchdog)")
        out.append(f"  start {int(g[0][1])}  ->  end {int(g[-1][1])}  "
                   f"(min {int(min(v for _,v in g))}, max {int(max(v for _,v in g))}, n={len(g)})")
    # ---- other gauges (mean/min/max) ----
    other = {k: v for k, v in gauges.items() if k != "gaussian_count" and v}
    if other:
        out.append("")
        out.append("GAUGES (mean / min / max / n)")
        for k in sorted(other):
            vals = [v for _, v in other[k]]
            out.append(f"  {k:20s} {sum(vals)/len(vals):>10.0f} / {min(vals):>8.0f} / "
                       f"{max(vals):>8.0f} / {len(vals)}")
    out.append("=" * 78)
    return "\n".join(out)


# Process-wide default ledger (the FF worker + the loop record into this one).
_LEDGER = TimingLedger()


def get_ledger() -> TimingLedger:
    return _LEDGER
