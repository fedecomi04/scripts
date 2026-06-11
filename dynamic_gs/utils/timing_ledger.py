"""Cross-process timing ledger for the static→teleop pipeline.

Every model load / inference / fusion step appends ONE row recording its own
``t_start``/``t_end`` (pure work — the timer wraps only the call, never the
wait before it). Rows from ALL processes (live_session capture, ns-train
static-gs, dynamic-gs-live) land in one JSONL under the dataset dir, so a single
``render()`` produces the by-phase report with **load vs inference** split.

Why per-op start/end (not just a duration): the renderer can then, per phase,
compute ``wall = max(t_end) − min(t_start)`` and ``work = Σ dur``:
  * ``work > wall`` ⇒ ops overlapped (parallel / a load hidden behind another) — good.
  * ``wall > work`` ⇒ **idle**: the phase spent time NOT doing tracked work
    (GPU/CPU queue wait, disk stall, a blocking sleep). That gap is the
    "stuck in a queue" signal the timing report is meant to surface.

A op's ``dur`` is pure work; inter-op waits are never folded into it, so the
totals stay honest and the wall−work delta localises stalls.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

LEDGER_FILENAME = "timing_ledger.jsonl"

# Canonical phase order + human labels for the bulleted report. Ops tag
# themselves with one of these keys.
PHASE_ORDER = [
    "capture",
    "segmentation",
    "object_3d_gen",
    "pointcloud_fusion",
    "static_training",
    "object_fusion",
    "teleop_init",
    "dynamic_runtime",
]
PHASE_LABELS = {
    "capture": "Static image capturing (operator sweep)",
    "segmentation": "Segmentation (FastSAM / SAM3)",
    "object_3d_gen": "Object 3D generation (SAM3D)",
    "pointcloud_fusion": "Pointcloud fusion (TSDF seed)",
    "static_training": "Static training (Splatfacto)",
    "object_fusion": "Object fusion (NDP / Phase 0b)",
    "teleop_init": "Teleop init (warm-start)",
    "dynamic_runtime": "Dynamic runtime (per-tick)",
}
# kinds: "load" (weights/model construct), "infer" (forward/compute),
# "fusion" (TSDF/NDP geometry), "io" (disk), "train" (optimisation loop).


def ledger_path(data_root) -> Path:
    return Path(data_root) / LEDGER_FILENAME


def reset(data_root) -> None:
    """Delete the ledger (call once at the very start of a fresh run)."""
    try:
        ledger_path(data_root).unlink()
    except FileNotFoundError:
        pass


def record(data_root, phase: str, op: str, kind: str,
           t_start: float, t_end: float, **meta) -> None:
    """Append one timing row. ``t_start``/``t_end`` are ``time.time()`` stamps
    wrapping ONLY the work (no preceding wait)."""
    row = {
        "phase": phase, "op": op, "kind": kind,
        "t_start": float(t_start), "t_end": float(t_end),
        "dur": float(t_end) - float(t_start),
    }
    row.update(meta)
    try:
        with open(ledger_path(data_root), "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass  # timing must never break the pipeline


@contextmanager
def timed(data_root, phase: str, op: str, kind: str, **meta):
    """``with timed(data, 'segmentation', 'FastSAM', 'infer'): ...`` — records
    the wrapped block's wall time as pure work."""
    t0 = time.time()
    try:
        yield
    finally:
        record(data_root, phase, op, kind, t0, time.time(), **meta)


def _fmt(seconds: float) -> str:
    return f"{seconds*1000:.0f}ms" if seconds < 1.0 else f"{seconds:.1f}s"


def render(data_root) -> str:
    """Render the by-phase bulleted load/infer report from the ledger."""
    path = ledger_path(data_root)
    if not path.exists():
        return "(no timing ledger)"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return "(timing ledger empty)"

    by_phase: dict[str, list] = {}
    for r in rows:
        by_phase.setdefault(r.get("phase", "?"), []).append(r)

    out = []
    out.append("=" * 78)
    out.append("PIPELINE TIMING — model load + inference, by phase")
    out.append("  (dur = pure work; per-phase 'idle' = wall−work = queue/stall;")
    out.append("   'overlap' = work−wall = ops ran in parallel / load hidden)")
    out.append("=" * 78)

    total_work = 0.0
    total_load = 0.0
    total_infer = 0.0
    ordered = [p for p in PHASE_ORDER if p in by_phase] + \
              [p for p in by_phase if p not in PHASE_ORDER]
    for phase in ordered:
        items = sorted(by_phase[phase], key=lambda r: r["t_start"])
        wall = max(r["t_end"] for r in items) - min(r["t_start"] for r in items)
        work = sum(r["dur"] for r in items)
        total_work += work
        out.append("")
        out.append(f"● {PHASE_LABELS.get(phase, phase)}")
        for r in items:
            kind = r.get("kind", "")
            extra = ""
            if "gpu_mb" in r and r["gpu_mb"]:
                extra += f"  [{r['gpu_mb']:.0f} MiB]"
            if "n" in r:
                extra += f"  (n={r['n']})"
            out.append(f"    - {r['op']:<26}{kind:<7}{_fmt(r['dur']):>8}{extra}")
            if kind == "load":
                total_load += r["dur"]
            elif kind == "infer":
                total_infer += r["dur"]
        overlap = work - wall
        if overlap > 0.15:
            note = f"overlap {_fmt(overlap)} (parallel ✓)"
        elif wall - work > 0.15:
            note = f"idle {_fmt(wall - work)} (queue/stall)"
        else:
            note = "serial, no stall"
        out.append(f"    └─ phase wall {_fmt(wall)} | work {_fmt(work)} | {note}")

    out.append("")
    out.append("-" * 78)
    out.append(f"  Σ load  {_fmt(total_load)}   Σ infer  {_fmt(total_infer)}   "
               f"Σ all work  {_fmt(total_work)}")
    out.append("  NOTE: compare each phase's wall to the process wall-clock — a large")
    out.append("        gap beyond 'work' means time lost to a queue/stall, not compute.")
    out.append("=" * 78)
    return "\n".join(out)
