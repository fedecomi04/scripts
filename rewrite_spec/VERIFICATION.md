# VERIFICATION — new pipeline vs the old pipeline as ground truth

The old pipeline WORKS on the reference dataset. So we don't verify the rewrite against
an abstract spec — we verify it **reproduces the old pipeline's behavior** on that dataset,
then improves the one thing we expect to improve (FF insert volume).

## Reference dataset (ground truth)
**`data_teleoperation/datasets/screwdriver recorded full`** — the recorded run that behaved
correctly (tracker 9.3 Hz, FF **non-continuous** at ~0.87 Hz, scene stayed lean — the
opposite of zed_final's 3M blowup). 1920×1200, 313 dynamic frames. **No depth noise**
(sim, clean) → run with `source=sim` but ZED-noise OFF; depth handling is not under test here.
> If "the one that worked perfectly" meant a different dataset, swap this path — everything below is dataset-agnostic.

## What to capture from the OLD pipeline (= ground truth)
Add an additive, non-destructive debug dump to the old pipeline (it already logs FF
insert/cull/total per call; add the per-tick rigid transform). Per tick, write a JSONL row:
- `tick`, `frame_seq`
- `R` (3×3) and `t` (3) — the rigid object transform the tracker applied that tick
  (`apply_rigid_object_transform_from_reference` inputs)
- `tracking_ok` (bool), `inliers`
- `ff_fired` (bool), `ff_inserted`, `ff_culled`, `total_gauss` after the tick
→ `screwdriver.../old_trace.jsonl`. (One log line in the old tracker tick + FF callback;
the user explicitly OK'd adding this.)

## What the NEW pipeline emits
The same JSONL (`new_trace.jsonl`) via the always-on `timing`/`debug` modules — it's a
default capture, not bolted on.

## Comparison (a small diff script)
`compare_traces(old_trace, new_trace)` reports, aligned by `frame_seq`:
1. **Pose fidelity** — per tick: rotation-angle diff `|R_old⁻¹R_new|` (deg) and translation
   diff `‖t_old−t_new‖` (mm). Report p50/p99/max. **Accept if p99 ≤ ~1 mm / ~0.5°**
   (the tracker is deterministic given the same frames + same anchors → should match closely;
   larger diffs = a real behavioral divergence to investigate).
2. **FF insert volume** — total inserted gaussians, #FF calls, final scene count.
   **Expectation: new ≤ old, and ideally MUCH less** — the audit flagged FF possibly acting
   on a stale/outdated mask (a hazard); the clean snapshot+contract should cut redundant
   inserts. New must also stay **non-continuous** (FF fires intermittently, not every tick)
   and **bounded** (never approaches the zed_final 3M blowup).
3. **Throughput** — effective tracker Hz, FF Hz. New should be ≥ old (9.3 Hz tracker).

## Acceptance gate (before the new pipeline replaces the old)
- [ ] Pose: p99 rotation ≤ 0.5°, translation ≤ 1 mm vs old, over the full episode.
- [ ] FF: total inserts ≤ old; FF non-continuous; final gaussian count bounded.
- [ ] Throughput: tracker Hz ≥ old; no crash over the full 313 frames.
- [ ] Run via `replay_mode="fast"` (deterministic, frame-exact) for the pose A/B, AND
      `replay_mode="paced"` once to confirm real-time behavior holds.

## Harness (build alongside; never delete the old)
Both pipelines run on the SAME dataset; the old stays the fallback until the new passes the
gate. Keep `old_trace.jsonl` checked in as the frozen reference so regressions are caught by
diffing, not by eyeballing the viewer.
