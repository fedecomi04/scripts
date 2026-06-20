#!/usr/bin/env bash
# dynamic_gs2 — WHOLE PIPELINE in ONE command + ONE process (needs Gazebo/ROS + dVRK up).
#
# Live static capture (red-box sweep) -> in-process hand-off -> live dynamic loop. The point of the
# single process: AnySplat + XFeat/LighterGlue are warm-loaded DURING the static phase (hidden under
# the sweep/train) and carried STRAIGHT into the dynamic loop — no reload, no ~17s AnySplat re-spawn.
#
# Operator: sweep the camera in Gazebo; press "Trigger" in the viser red-box UI (http://localhost:8081)
# — or Enter in this terminal — when the object fills the box. Then it trains/fuses and goes live.
#
# Usage:  dynamic_gs2/live.sh <data_dir> [prompt]
#   e.g.  dynamic_gs2/live.sh "../data_teleoperation/datasets/$(date +%Y-%m-%d_%H%M%S)" "screwdriver"
set -euo pipefail

DATA="${1:?usage: live.sh <data_dir> [prompt]}"
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

# dVRK RT protection: reserve cores 0-3 for the dVRK 1kHz loop, cap math threadpools, pin the pipeline
# to 4-23, lock the dVRK ONTO 0-3. Helpers self-skip when no dVRK/cpuset is present. DGS_NO_CPU_PIN=1 off.
source "$SCRIPTS_DIR/scripts/_ros_cleanup.sh"
dgs_ros_cleanup                       # flush any leaked publisher state from a prior run (never touches Gazebo)
dgs_export_thread_caps
DGS_PIN="$(dgs_cpu_pin_prefix)"
dgs_isolate_dvrk

echo "[live] WHOLE pipeline (single process): static capture -> dynamic. data=$DATA prompt='${PROMPT}' pin=${DGS_PIN:-none}"
exec $DGS_PIN "$PY" -m dynamic_gs2.pipeline --mode full --data "$DATA" --prompt "$PROMPT" --ff
