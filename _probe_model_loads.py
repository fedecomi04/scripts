"""Isolated per-model load probe — settles which model owns the ~32s load.

Spawns a FRESH SamWorkerClient for EACH model so there's no serial-queue interleaving
and no shared-import attribution ambiguity. For each: measure both the worker's
self-reported `load_seconds` AND the true client-side wall time (which includes any
import the worker does outside its internal timer). Run from scripts/.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_gs.utils.sam_worker import SamWorkerClient


def probe(name, load_call):
    print(f"\n=== {name}: fresh worker ===", flush=True)
    t_spawn = time.perf_counter()
    c = SamWorkerClient(conda_env="sam3_dynamic_gs")
    spawn_s = time.perf_counter() - t_spawn
    print(f"  worker spawn: {spawn_s:.2f}s", flush=True)
    try:
        t0 = time.perf_counter()
        reported = load_call(c)          # returns the worker's internal load_seconds
        wall = time.perf_counter() - t0
        print(f"  {name} LOAD: wall={wall:.2f}s   worker-reported load_seconds={reported}", flush=True)
        return wall, reported
    finally:
        c.close()


if __name__ == "__main__":
    import os
    os.environ.setdefault("CONDA_PREFIX", os.path.expanduser("~/miniconda3/envs/sam3_dynamic_gs"))
    sam3d_wall, sam3d_rep = probe("Fast-SAM3D (load_sam3d)", lambda c: c.load_sam3d())
    fastsam_wall, fastsam_rep = probe("FastSAM 2D (load_fastsam)", lambda c: c.load_fastsam())
    print("\n================ SUMMARY (isolated, no interleaving) ================")
    print(f"  Fast-SAM3D (sam3d) : wall {sam3d_wall:6.2f}s  (worker reported {sam3d_rep})")
    print(f"  FastSAM 2D         : wall {fastsam_wall:6.2f}s  (worker reported {fastsam_rep})")
    print("====================================================================")
