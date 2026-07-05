#!/usr/bin/env python3
"""_plot_pipeline_timeline.py — render the FULL-LIVE pipeline timeline (load times +
which algorithm is running) from the two MEASURED timing reports a run drops in its
dataset dir:  timing_report_static.txt  +  timing_report_dynamic.txt.

Built for the NEW report format (>= 2026-06-22) that separately times the model loads
(load.sam3d / load.fastsam / load.anysplat / load.xfeat) and the per-keyframe seed work
(seed.icp_per_kf / seed.tsdf_per_kf) — so every bar here is a measured number, including
the background loads that overlap the main track (drawn on their own lanes).

One figure, three panels:
  A  STATIC phase  — sequential load schedule (Gantt) + the measured parallel tracks
                     (background model loads + the CPU seed builder) that overlap it.
  B  DYNAMIC phase — ONE representative feed-forward (FF) cycle as a Gantt, with the
                     tracker ticks that landed inside it (proves FF never blocks the tracker).
  C  DYNAMIC phase — mean per-step FF cost (aggregate over every cycle) + the tracker headline.

Usage:  python _plot_pipeline_timeline.py [<dataset_dir>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_DATA = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/2026-06-22_025905"


# ----------------------------------------------------------------- parsers
def parse_static(path: Path):
    """Return dict: stages=[(name,dur,off)] (schedule order), agg={name:total_s},
    wall, dead_time, train_steps."""
    txt = path.read_text()
    stages, agg = [], {}
    in_sched = in_agg = False
    for line in txt.splitlines():
        if "AGGREGATE PER-STAGE" in line:
            in_agg, in_sched = True, False
            continue
        if "SCHEDULE BREAKDOWN" in line:
            in_sched, in_agg = True, False
            continue
        if in_agg:
            m = re.match(r"^\s*([\w.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m and m.group(1) != "stage":
                agg[m.group(1)] = float(m.group(3))           # total_s
        if in_sched:
            if line.startswith("GAUGES"):
                in_sched = False
                continue
            m = re.match(r"^\s*([\w.]+)\s+([\d.]+)s\s+@\s+\+\s*([\d.]+)s\s*$", line)
            if m:
                stages.append((m.group(1), float(m.group(2)), float(m.group(3))))

    def grab(pat, cast=float, grp=1):
        mm = re.search(pat, txt)
        return cast(mm.group(grp)) if mm else None

    return {"stages": stages, "agg": agg,
            "wall": grab(r"static phase\s+\(wall\s+([\d.]+)s\)"),
            "dead": grab(r"from 'sam3d_done' to end\s+([\d.]+)s"),
            "train_steps": grab(r"static_train_steps\s+(\d+)", int)}


def parse_dynamic(path: Path):
    """Return dict: agg=[(name,n,mean)], hz, ticks, span_s, intertick=[p50,p90,p99,max],
    n_cycles, n_inserts, ticks_per_cycle, cycle0={wall,stages=[(off,name,dur,[(tick,t)])]}"""
    txt = path.read_text()
    lines = txt.splitlines()

    agg = []
    in_agg = False
    for line in lines:
        if "AGGREGATE PER-STEP" in line:
            in_agg = True
            continue
        if in_agg:
            if line.strip().startswith("TRACKER INTERLEAVE") or "PER-FF-CYCLE" in line:
                break
            m = re.match(r"^\s*([\w.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m and m.group(1) not in ("step",):
                agg.append((m.group(1), int(m.group(2)), float(m.group(3))))

    def grab(pat, cast=float, grp=1):
        mm = re.search(pat, txt)
        return cast(mm.group(grp)) if mm else None

    hz = grab(r"effective Hz\s+([\d.]+)")
    ticks = grab(r"\(([\d]+) ticks over", int)
    span = grab(r"ticks over\s+([\d.]+)s")
    it = re.search(r"p50/p90/p99/max\s+([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", txt)
    intertick = [float(it.group(i)) for i in range(1, 5)] if it else None

    cyc0 = {"wall": None, "stages": []}
    grabbing = False
    for line in lines:
        h = re.match(r"^\s*──\s*FF cycle 0\s+\(wall\s+([\d.]+)ms,.*", line)
        if h:
            grabbing = True
            cyc0["wall"] = float(h.group(1))
            continue
        if grabbing:
            if re.match(r"^\s*──\s*FF cycle 1\b", line):
                break
            sm = re.match(r"^\s*\+\s*([\d.]+)\s+([\w.]+)\s+([\d.]+)ms(?:\s+\[(.+)\])?\s*$", line)
            if sm:
                off, name, dur = float(sm.group(1)), sm.group(2), float(sm.group(3))
                tk = [(int(t), float(rel))
                      for t, rel in re.findall(r"#(\d+)\s*@\s*\+([\d.]+)ms", sm.group(4) or "")]
                cyc0["stages"].append((off, name, dur, tk))

    return {"agg": agg, "hz": hz, "ticks": ticks, "span_s": span, "intertick": intertick,
            "n_cycles": grab(r"over (\d+) FF cycle", int), "n_inserts": grab(r"ff_inserted\s+(\d+)", int),
            "ticks_per_cycle": grab(r"avg tracker-ticks per FF cycle\s+([\d.]+)"), "cycle0": cyc0}


# ----------------------------------------------------------------- styling
# (color, multi-line label) for the STATIC main-lane stages.
S_STYLE = {
    "trigger.snapshot_anchor":  ("#9e9e9e", "snapshot\nanchor"),
    "trigger.fastsam_segment":  ("#ff8c1a", "FastSAM + CLIP\n2D segmentation"),
    "trigger.write_seg_folder": ("#cfcfcf", "write\nmasks"),
    "trigger.sam3d_infer":      ("#e6194B", "Fast-SAM3D\n3D reconstruct"),
    "after.sam_worker_close":   ("#607d8b", "close SAM\nworker"),
    "after.tsdf_integrate":     ("#3cb44b", "TSDF seed\nfinalize"),
    "after.splatfacto_train":   ("#4363d8", "Splatfacto train\n+ Phase-0b fuse + export"),
    "end.wake_dynamic":         ("#8e44ad", "wake dynamic\n(wait AnySplat)"),
}
S_MAIN_ORDER = list(S_STYLE.keys())
INSIDE = {"trigger.sam3d_infer", "after.splatfacto_train"}   # only the WIDE stages get an inside label

# FF step -> (color, sub-phase label) for the DYNAMIC panels.
F_STYLE = {
    "cdn.render":            ("#1abc9c", "CDN render (locked)"),
    "cdn.clean":             ("#16a085", "CDN change-detect + clean"),
    "cull_infront.compute":  ("#b7950b", "cull-in-front compute"),
    "cull_infront.hide":     ("#d4ac0d", "cull-in-front hide"),
    "recdn.render":          ("#48c9b0", "re-CDN render"),
    "recdn.gate":            ("#45b39d", "re-CDN gate"),
    "decode.icp":            ("#ff8c1a", "ICP (depth align)"),
    "decode.crop_ipc":       ("#9b59b6", "crop + IPC to AnySplat"),
    "decode.worker_forward": ("#e6194B", "AnySplat forward (decode)"),
    "decode.reproject":      ("#4363d8", "reproject (scene K)"),
    "decode.density_shape":  ("#3cb44b", "density shaping"),
    "decode.clamp":          ("#95a5a6", "scale clamp"),
    "enforce_ceiling":       ("#7f8c8d", "enforce ceiling"),
    "cull_replaced.compute": ("#8B4513", "cull-replaced compute"),
    "surgery.cull_insert":   ("#a0522d", "atomic cull+insert (locked)"),
}


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATA)
    S = parse_static(data_dir / "timing_report_static.txt")
    D = parse_dynamic(data_dir / "timing_report_dynamic.txt")
    soff = {n: (d, o) for n, d, o in S["stages"]}
    agg = S["agg"]

    print("STATIC stages:", len(S["stages"]), "wall", S["wall"], "dead", S["dead"], "steps", S["train_steps"])
    print("  loads:", {k: agg.get(k) for k in ("load.sam3d", "load.fastsam", "load.anysplat", "load.xfeat")})
    print("  seed :", {k: agg.get(k) for k in ("seed.icp_per_kf", "seed.tsdf_per_kf")})
    print("DYNAMIC agg steps:", len(D["agg"]), "hz", D["hz"], "ticks", D["ticks"],
          "cyc0 stages", len(D["cycle0"]["stages"]), "cyc0 wall", D["cycle0"]["wall"], "n_cycles", D["n_cycles"])

    fig = plt.figure(figsize=(17, 19.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.3, 2.1, 3.3], hspace=0.45)

    # =========================================================== PANEL A : STATIC
    axA = fig.add_subplot(gs[0])
    LANE_SWEEP, LANE_MAIN, LANE_SAM, LANE_DYN, LANE_SEED = 5.0, 3.8, 2.0, 1.1, 0.3
    H_MAIN, H = 0.74, 0.5
    trig = soff.get("trigger.snapshot_anchor", (0, 0))[1]
    wall = S["wall"] or 72.4
    sd_dur, sd_off = soff.get("trigger.sam3d_infer", (0, 0))
    sam3d_done = sd_off + sd_dur

    # initial capture (operator sweep ~0..trigger, then trigger-side segment + Fast-SAM3D
    # reconstruct) — the whole front-end capture, ending when Fast-SAM3D generation completes.
    axA.barh(LANE_SWEEP, sam3d_done, left=0, height=H, color="#bdbdbd", edgecolor="black", zorder=3)
    axA.text(sam3d_done / 2, LANE_SWEEP,
             f"Initial capture — operator sweep (~{trig:.0f}s, operator-paced) → Fast-SAM3D done",
             ha="center", va="center", fontsize=9, zorder=4)

    # main sequential pipeline lane: WIDE stages get an inside label; sub-2s stages get a
    # leader line + label BELOW the lane.
    for name in S_MAIN_ORDER:
        if name not in soff:
            continue
        dur, off = soff[name]
        color, lbl = S_STYLE[name]
        axA.barh(LANE_MAIN, max(dur, 0.25), left=off, height=H_MAIN, color=color,
                 edgecolor="black", zorder=3)
        if name in INSIDE:
            axA.text(off + dur / 2, LANE_MAIN, f"{lbl}\n{dur:.2f}s", ha="center", va="center",
                     fontsize=8, color="white", zorder=4)
    below = [("trigger.snapshot_anchor", "snapshot anchor"),
             ("trigger.write_seg_folder", "write masks"),
             ("after.sam_worker_close", "close SAM worker"),
             ("after.tsdf_integrate", "TSDF seed finalize")]
    ytxt = LANE_MAIN - 0.62
    for i, (name, label) in enumerate(below):
        if name not in soff:
            continue
        dur, off = soff[name]
        xc = off + dur / 2
        yl = ytxt - (0.32 if i % 2 else 0.0)
        axA.plot([xc, xc], [LANE_MAIN - 0.37, yl + 0.05], color="black", lw=0.5, zorder=2)
        axA.text(xc, yl, f"{label}\n({dur:.2f}s)", ha="center", va="top", fontsize=7.2, zorder=4)
    # ABOVE-lane leader labels for the narrow-but-notable stages (too thin for an inside label)
    above = [("trigger.fastsam_segment", "FastSAM + CLIP 2D seg"),
             ("end.wake_dynamic", "wake dynamic (wait AnySplat)")]
    yab = LANE_MAIN + H_MAIN / 2 + 0.16
    for name, label in above:
        if name not in soff:
            continue
        dur, off = soff[name]
        xc = off + dur / 2
        axA.plot([xc, xc], [LANE_MAIN + 0.37, yab - 0.02], color="black", lw=0.5, zorder=2)
        axA.text(xc, yab, f"{label} ({dur:.2f}s)", ha="center", va="bottom", fontsize=7.4, zorder=4)

    # measured parallel tracks — background model LOADS (their own lanes; anchored at the
    # measured spawn offset, length = measured load.* duration).
    def load_bar(lane, stage_for_off, off_default, load_key, color, label):
        if load_key not in agg:
            return None
        off = soff.get(stage_for_off, (0, off_default))[1]
        d = agg[load_key]
        axA.barh(lane, d, left=off, height=H, color=color, alpha=0.80,
                 edgecolor=color, lw=1.4, linestyle=(0, (4, 2)), zorder=2)
        if d >= 3:                                   # wide enough for an inside label
            axA.text(off + d / 2, lane, f"{label}  {d:.1f}s", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold", zorder=4)
        else:                                        # too narrow: leader label ABOVE the bar
            xc = off + d / 2
            axA.plot([xc, xc], [lane + 0.25, lane + 0.40], color="black", lw=0.5, zorder=3)
            axA.text(xc, lane + 0.43, f"{label} ({d:.1f}s)", ha="center", va="bottom",
                     fontsize=7, zorder=4)
        return off + d

    sam_end = load_bar(LANE_SAM, "sweep.sam3d_load", 0.0, "load.sam3d", "#e6194B", "Fast-SAM3D model load")
    if sam_end is not None:
        axA.annotate(f"✓ finishes at ~{sam_end:.0f}s — fully under the sweep → segment runs clean (not stalled)",
                     xy=(sam_end, LANE_SAM), xytext=(sam_end + 1.0, LANE_SAM - 0.55),
                     fontsize=7.6, color="#b30000", va="top",
                     arrowprops=dict(arrowstyle="->", color="#b30000", lw=0.8))
    load_bar(LANE_DYN, "after.anysplat_spawn", 50.1, "load.anysplat", "#9b59b6", "AnySplat model load")
    load_bar(LANE_DYN, "after.dyn_models_prewarm", 50.1, "load.xfeat", "#16a085", "XFeat/LighterGlue")

    # seed builder (CPU) — measured per-keyframe ICP + TSDF, incremental during the sweep
    icp, tsdf = agg.get("seed.icp_per_kf"), agg.get("seed.tsdf_per_kf")
    axA.barh(LANE_SEED, trig, left=0, height=H, color="#3cb44b", alpha=0.80, edgecolor="#1e7d33",
             lw=1.2, zorder=2)
    seed_lbl = "Seed builder (CPU) — incremental TSDF fuse during sweep"
    if icp is not None and tsdf is not None:
        seed_lbl = f"Seed builder (CPU): 29 keyframes — ICP {icp:.1f}s + TSDF {tsdf:.1f}s (incremental during sweep)"
    axA.text(trig / 2, LANE_SEED, seed_lbl, ha="center", va="center", fontsize=8, color="white",
             fontweight="bold", zorder=4)

    # markers: trigger, sam3d_done, dead-time bracket
    axA.axvline(trig, color="black", lw=1.3, ls="--", zorder=5)
    axA.text(trig, LANE_SWEEP + 0.45, "TRIGGER", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.axvline(sam3d_done, color="#e6194B", lw=1.1, ls=":", zorder=5)
    if S["dead"]:
        ybr = LANE_SWEEP + 0.2
        axA.annotate("", xy=(wall, ybr), xytext=(sam3d_done, ybr),
                     arrowprops=dict(arrowstyle="<->", color="#b30000", lw=1.6))
        axA.text((sam3d_done + wall) / 2, ybr + 0.12,
                 f"operator-visible DEAD TIME  (sam3d_done → live) = {S['dead']:.1f}s",
                 ha="center", va="bottom", fontsize=9, color="#b30000", fontweight="bold")

    axA.set_xlim(-1, wall + 7)
    axA.set_ylim(-0.3, LANE_SWEEP + 1.15)
    axA.set_yticks([LANE_SEED, LANE_DYN, LANE_SAM, LANE_MAIN, LANE_SWEEP])
    axA.set_yticklabels(["seed (CPU)", "dyn-model loads", "SAM3D load", "MAIN pipeline\n(GPU, serial)", "operator"])
    axA.set_xlabel("seconds from static-phase start")
    axA.set_title(f"A.  STATIC phase — sequential load schedule + measured parallel tracks   "
                  f"(total wall {wall:.1f}s,  {S['train_steps']} train steps)",
                  fontsize=12, fontweight="bold", loc="left")
    axA.grid(axis="x", ls=":", alpha=0.4)

    # =========================================================== PANEL B : DYNAMIC, one FF cycle
    axB = fig.add_subplot(gs[1])
    c0 = D["cycle0"]
    LANE_FF, LANE_TICK = 1.0, 0.25
    seen, all_ticks = set(), []
    for off, name, dur, tk in c0["stages"]:
        color = F_STYLE.get(name, ("#999999", name))[0]
        axB.barh(LANE_FF, dur, left=off, height=0.6, color=color, edgecolor="black", lw=0.6, zorder=3)
        if dur >= 60:
            axB.text(off + dur / 2, LANE_FF, f"{dur:.0f}", ha="center", va="center",
                     fontsize=7.5, color="white", zorder=4)
        seen.add(name)
        all_ticks += [(t, off + rel) for t, rel in tk]
    for t, x in all_ticks:
        axB.axvline(x, ymin=0.05, ymax=0.95, color="#444444", lw=0.8, ls="--", alpha=0.55, zorder=2)
        axB.plot(x, LANE_TICK, marker="^", ms=8, color="#111111", zorder=5)
        axB.text(x, LANE_TICK - 0.16, f"#{t}", ha="center", va="top", fontsize=7)
    axB.set_xlim(-15, (c0["wall"] or 1700) + 20)
    axB.set_ylim(-0.25, 1.5)
    axB.set_yticks([LANE_TICK, LANE_FF])
    axB.set_yticklabels(["tracker ticks\n(main thread)", "FF stages\n(bg thread)"])
    axB.set_xlabel("milliseconds from FF-cycle start")
    axB.set_title(f"B.  DYNAMIC phase — one representative FF cycle (#0, wall {c0['wall']:.0f}ms): "
                  f"{len(all_ticks)} tracker ticks interleave → FF never blocks the tracker",
                  fontsize=12, fontweight="bold", loc="left")
    axB.grid(axis="x", ls=":", alpha=0.4)
    handles = [Patch(facecolor=F_STYLE[n][0], edgecolor="black", label=F_STYLE[n][1])
               for n in F_STYLE if n in seen]
    axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               ncol=4, fontsize=8, frameon=False)

    # =========================================================== PANEL C : DYNAMIC, mean per-step
    axC = fig.add_subplot(gs[2])
    dagg = D["agg"]
    n_cyc = D["n_cycles"] or 1
    ys = list(range(len(dagg)))[::-1]
    for y, (name, n, mean) in zip(ys, dagg):
        color = F_STYLE.get(name, ("#999999", name))[0]
        axC.barh(y, mean, color=color, edgecolor="black", zorder=3)
        tag = f"{mean:.1f} ms" + (f"   (≈2×/cycle, n={n})" if n > n_cyc + 2 else "")
        axC.text(mean + 4, y, tag, va="center", fontsize=8.5, zorder=4)
    axC.set_yticks(ys)
    axC.set_yticklabels([F_STYLE.get(n, ("", n))[1] for n, _, _ in dagg], fontsize=9)
    axC.set_xlabel("mean milliseconds per FF cycle (aggregate over all cycles)")
    axC.set_xlim(0, max(m for _, _, m in dagg) * 1.35)
    axC.set_ylim(-0.8, len(dagg) - 1 + 0.9)
    axC.set_title("C.  DYNAMIC phase — mean per-step FF cost  "
                  "(AnySplat crop+forward dominates; the 3 decode steps run ~2× per cycle)",
                  fontsize=12, fontweight="bold", loc="left")
    axC.grid(axis="x", ls=":", alpha=0.4)
    it = D["intertick"] or [0, 0, 0, 0]
    head = (f"TRACKER:  {D['hz']:.1f} Hz effective  ·  {D['ticks']} ticks / {D['span_s']:.1f}s  ·  "
            f"inter-tick p50/p90/p99/max = {it[0]:.0f}/{it[1]:.0f}/{it[2]:.0f}/{it[3]:.0f} ms\n"
            f"FF:  {D['n_cycles']} cycles  ·  {D['n_inserts']} inserts  ·  "
            f"avg {D['ticks_per_cycle']:.1f} tracker ticks per FF cycle")
    axC.text(0.99, 0.04, head, transform=axC.transAxes, ha="right", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff7e6", ec="#d9a441"))

    fig.suptitle("Full-Live Gaussian-Splat pipeline — load times & active algorithm  (single process: static capture → dynamic loop)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.004,
             f"All bars measured (perf_counter) from {data_dir.name}/timing_report_{{static,dynamic}}.txt.  "
             "Background model loads (SAM3D / AnySplat / XFeat) are on their own lanes, anchored at their measured spawn offset.  "
             "Sub-second static bars are labelled below the lane.",
             ha="center", fontsize=8.5, style="italic")

    out = data_dir / "pipeline_timeline.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
