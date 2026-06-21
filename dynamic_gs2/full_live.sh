#!/usr/bin/env bash
# dynamic_gs2 MODE 1/4 — FULL LIVE: the whole pipeline (static capture -> dynamic) in ONE process.
# Needs Gazebo/ROS + dVRK up. AnySplat + XFeat/LighterGlue are warm-loaded DURING the static phase
# (hidden under the sweep/train) and carried STRAIGHT into the dynamic loop — no reload, no re-spawn.
#
# Operator: sweep the camera; press "Trigger" in the viser red-box UI (http://localhost:8081) — or
# Enter in this terminal — when the object fills the box. Then it trains/fuses and goes live.
#
# Usage:  dynamic_gs2/full_live.sh <data_dir> [prompt]
#   e.g.  dynamic_gs2/full_live.sh "../data_teleoperation/datasets/$(date +%Y-%m-%d_%H%M%S)" "screwdriver"
set -euo pipefail

DATA="${1:?usage: full_live.sh <data_dir> [prompt]}"
PROMPT="${2:-}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../scripts
ENV="$HOME/miniconda3/envs/dynamic_gs"
PY="$ENV/bin/python"
mkdir -p "$DATA"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
export CUDA_HOME="$ENV"                                              # gsplat JIT needs these on a bare-python
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"    # launch (else cuda_runtime.h not found)
# Sim ZED-X depth-noise injection OFF for live: ~85 ms/frame in the publisher and it's a sim-only
# realism model, irrelevant to live tracking. Set DGS_SIM_ZED_NOISE=1 before launch for a noise study.
export DGS_SIM_ZED_NOISE="${DGS_SIM_ZED_NOISE:-0}"

# dVRK RT protection: reserve cores 0-3 for the dVRK 1kHz loop, cap math threadpools, pin the pipeline
# to 4-23, lock the dVRK ONTO 0-3. Helpers self-skip when no dVRK/cpuset is present. DGS_NO_CPU_PIN=1 off.
source "$SCRIPTS_DIR/scripts/_ros_cleanup.sh"
dgs_ros_cleanup                       # flush any leaked publisher state from a prior run (never touches Gazebo)
dgs_export_thread_caps
DGS_PIN="$(dgs_cpu_pin_prefix)"
dgs_isolate_dvrk

echo "[full_live] whole pipeline (single process): live static capture -> live dynamic. data=$DATA prompt='${PROMPT}' pin=${DGS_PIN:-none}"
exec $DGS_PIN "$PY" -m dynamic_gs2.pipeline --mode full --source live_bridge --data "$DATA" --prompt "$PROMPT" --ff
