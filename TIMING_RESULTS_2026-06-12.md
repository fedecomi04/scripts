# Pipeline timing results — 2026-06-12

Dead-time-to-teleop-ready (operator waits, nothing to do), measured against the
replay harness (`scripts/replay_pipeline.sh`, bag teleop_20260611_192947):

| Phase (after sweep Enter) | Before campaign | Now |
|---|---|---|
| FastSAM infer | 0.97 s | ~1.0 s |
| SAM3D model load | 32.3 s (exposed) | **hidden behind sweep** (preload, 0-OOM, peak 12.65 GB) |
| SAM3D inference | 12.1 s | ~10.2 s |
| TSDF seed | 47.9 s (CPU) + 45.5 s redundant stage-1.5 rebuild | **2.1–2.3 s (GPU) + 0 (skip)** |
| Static training | 24.4 s (500 steps, eval mid-train) | **~3.4 s** (EMA early-stop @0.09, full-res-from-0, eval off; PSNR 19.12 vs 20.10 dB) |
| NDP Phase 0b | 2.4 s | ~2.5 s |
| Teleop init | 7.6 s | ~6.5 s |
| **TOTAL** | **~173 s** | **~26 s** |

Dynamic runtime: tracker ~21–28 Hz (36–47 ms/tick, run variance), AnySplat FF
~1.17 s/call in background (doesn't block ticks).

Tracker oscillation (fixed-frames fixture, identical input):
- Stationary: 0.52→0.30 mm / 0.37→0.15° (raw vs KF+crop-fix), 42 %/59 % lower.
- KF-induced tracking death eliminated (302→616 ticks) — crop bbox now from the
  raw tracker pose (commit 286aa34).
- Moving-segment residual ~7 mm high-pass = correlated match-set error; output-
  side filtering is a measured dead end (≤5 %). Next lever: anchor-policy redesign.

Commits: fbe2d23, a7b5cfb, 4bd8333, 75886a1, 862e70e, b3577db, 286aa34, 2e1a32e, 7af9b40.
