# LOC — the rewrite's primary KPI

Goal of the rewrite: same behavior, **far fewer lines** (the old package is vibe-coded
and tangled). **Target: < 10,000 LOC** (from the 28,003 baseline — a ~64%+ cut).
Track every module as it lands.

## BASELINE (old package — the number to beat)
- `dynamic_gs/` = **28,003 LOC** (measured 2026-06-18, `find dynamic_gs -name '*.py' | xargs wc -l`).
  Biggest offenders: `dynamic_gs_pipeline_base.py` 3553, `dynamic_gs_model.py` 2411,
  `live_ros_publisher.py` 1541, `xfeat_motion.py` 1515, `sam3d_fusion.py` 1363.

## NEW (`dynamic_gs2/`) — running total
| module | LOC | status |
|---|---:|---|
| `frame.py` (contract + SHM codec) | ~165 | ✅ written + round-trip test PASSES |
| `tests/test_frame.py` | ~70 | ✅ green |
| `config.py` | — | TODO (leaf) |
| `shm_channel.py` | — | TODO (leaf — producer/consumer lifecycle over frame.py codec) |
| `gaussian_set.py` | — | TODO (core — SSOT + locked surgery + snapshot) |
| `scene_model.py` | — | TODO (core — WRAP render/train) |
| static/* dynamic/* pipeline | — | TODO |

> Update this table as each module lands. Re-measure with:
> `find dynamic_gs2 -name '*.py' -not -path '*/tests/*' | xargs wc -l | tail -1`
> (count tests separately; they don't count against the KPI but DO count as deliverables.)

## Build order (per 00_OVERVIEW.md)
leaves (`frame`✅, `config`, `shm_channel`) → core (`gaussian_set`, `scene_model`) →
static (`segment/fuse/fit/persist`) → dynamic (`track/feedforward/viz/scheduler`) →
orchestrator (`pipeline`). Leaf+core are unit-test-verifiable (autonomous-safe); the
dynamic layer needs live/replay verification with the operator.
