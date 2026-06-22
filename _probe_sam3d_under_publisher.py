"""Measure SAM3D load WHILE the live publisher is running (contention test).

Isolated SAM3D load = 24.8s (headless). Live run = 49s. Hypothesis: the ROS publisher
(camera decode + mask render into SHM every tick) contends for CPU/GPU during the load.
This spawns the real publisher, confirms frames are streaming, THEN times load_sam3d.
Needs the headless Gazebo/ROS stack up. Run from scripts/ with the dynamic_gs env python.
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CONDA_PREFIX", os.path.expanduser("~/miniconda3/envs/sam3_dynamic_gs"))

from dynamic_gs2 import config as _C
from dynamic_gs2.adapters_source import open_source, ShmRing
from dynamic_gs.utils.sam_worker import SamWorkerClient


def main():
    cfg = _C.load_runtime_config()
    shm = cfg.shm_name
    print(f"[probe] spawning live publisher into SHM '{shm}' ...", flush=True)
    src = open_source("live_bridge", shm_name=shm, attach=True)
    ring = ShmRing(shm)

    # confirm the publisher is actually streaming frames (so the contention is real, not idle)
    t_wait = time.perf_counter()
    seen, last = 0, -1
    while time.perf_counter() - t_wait < 15.0 and seen < 10:
        fr = ring.peek_latest()
        if fr is not None and int(fr.seq) != last:
            last = int(fr.seq); seen += 1
        time.sleep(0.02)
    print(f"[probe] publisher streaming: {seen} distinct frames seen in {time.perf_counter()-t_wait:.1f}s "
          f"(seq={last})", flush=True)
    if seen < 3:
        print("[probe] WARNING: publisher not streaming well — contention result may be unrepresentative", flush=True)

    # now time SAM3D load with the publisher actively contending
    print("[probe] loading SAM3D (publisher running) ...", flush=True)
    c = SamWorkerClient(conda_env="sam3_dynamic_gs")
    t0 = time.perf_counter()
    reported = c.load_sam3d()
    wall = time.perf_counter() - t0

    print("\n================ SAM3D LOAD UNDER PUBLISHER ================")
    print(f"  wall={wall:.2f}s   worker-reported load_seconds={reported}")
    print(f"  (isolated headless baseline was 24.8s; live full run was 49.0s)")
    print("===========================================================")

    c.close(); ring.close(); src.close()


if __name__ == "__main__":
    main()
