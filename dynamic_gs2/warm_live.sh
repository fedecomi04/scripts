#!/usr/bin/env bash
# dynamic_gs2 MODE 3/4 — WARM LIVE: SKIP the static phase; warm-load static_state.pt + AnySplat/XFeat,
# then track on the LIVE camera. Needs Gazebo/ROS + dVRK up. Reuses the proven ROS publisher via the
# LiveBridgeSource (frames forwarded into the new SHM layout).
#
# Usage:  dynamic_gs2/warm_live.sh <data_dir> [--no-ff]
# Needs:  <data_dir>/static_scene/static_state.pt
set -euo pipefail

DATA="${1:?usage: warm_live.sh <data_dir> [--no-ff]}"; shift || true
FF_FLAG="--ff"; for a in "$@"; do [ "$a" = "--no-ff" ] && FF_FLAG="--no-ff"; done

if [ ! -f "$DATA/static_scene/static_state.pt" ]; then
  echo "ERROR: $DATA/static_scene/static_state.pt not found (run full_live.sh or full_recorded.sh first)." >&2
  exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../scripts
ENV="$HOME/miniconda3/envs/dynamic_gs"
PY="$ENV/bin/python"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
export CUDA_HOME="$ENV"                                              # gsplat JIT needs these on a bare-python
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"    # launch (else cuda_runtime.h not found)

# Real-HW ROS networking (see full_live.sh): the Jetson publisher is on another machine, so the PC must
# advertise its REAL IP, not localhost, or frames silently freeze. roscore must use the SAME values.
if [ "${DGS_REAL_HW_CAMERA:-0}" != "0" ]; then
  DGS_PC_IP="${DGS_PC_IP:-192.168.55.100}"
  export ROS_IP="${ROS_IP:-$DGS_PC_IP}"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://$DGS_PC_IP:11311}"
  echo "[warm_live] real-HW ROS net: ROS_MASTER_URI=$ROS_MASTER_URI ROS_IP=$ROS_IP "
  # Cap dynamic publisher PC-side rate to ease USB/CPU contention (see full_live.sh). =0 for full rate.
  export DGS_PUB_MAX_HZ="${DGS_PUB_MAX_HZ:-10}"
fi

# dVRK RT protection: reserve cores 0-3 for the dVRK 1kHz loop, cap math threadpools, pin the pipeline
# to 4-23, lock the dVRK ONTO 0-3. Helpers self-skip when no dVRK/cpuset is present. DGS_NO_CPU_PIN=1 off.
source "$SCRIPTS_DIR/scripts/_ros_cleanup.sh"
dgs_ros_cleanup                       # flush any leaked publisher state from a prior run (never touches Gazebo)
dgs_export_thread_caps
DGS_PIN="$(dgs_cpu_pin_prefix)"
dgs_isolate_dvrk

echo "[warm_live] warm-start dynamic on LIVE camera: $DATA $FF_FLAG (pin: ${DGS_PIN:-none})"
exec $DGS_PIN "$PY" -m dynamic_gs2.pipeline --mode live --source live_bridge --data "$DATA" $FF_FLAG
