#!/usr/bin/env bash
# dynamic_gs2 — go LIVE on a pre-trained dataset (warm-load static_state.pt, track + optional FF).
#
# Reuses the proven ROS publisher via the LiveBridgeSource (default --source live_bridge):
# the validated py3.8 publisher is spawned by the old LiveShmSubscriber and its frames are
# forwarded into the new SHM layout. Requires a live Gazebo/ROS stack (operator step 4).
#
# Usage:  dynamic_gs2/resume_live.sh <data_dir> [--ff]
# Needs:  <data_dir>/static_scene/static_state.pt
set -euo pipefail

DATA="${1:?usage: resume_live.sh <data_dir> [--ff]}"
shift || true
FF_FLAG=""
for a in "$@"; do [ "$a" = "--ff" ] && FF_FLAG="--ff"; done

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../scripts
ENV_LIB="$HOME/miniconda3/envs/dynamic_gs/lib"
PY="$HOME/miniconda3/envs/dynamic_gs/bin/python"

if [ ! -f "$DATA/static_scene/static_state.pt" ]; then
  echo "ERROR: $DATA/static_scene/static_state.pt not found (run the static phase first)." >&2
  exit 1
fi

cd "$SCRIPTS_DIR"
ENV="$HOME/miniconda3/envs/dynamic_gs"
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
export CUDA_HOME="$ENV"                                              # gsplat JIT needs these on a bare-python
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"    # launch (else cuda_runtime.h not found)

# dVRK RT protection (same as the old bootstrap): reserve cores 0-3 for the dVRK 1kHz loop, cap the
# math threadpools, pin the pipeline to 4-23, and lock the dVRK ONTO 0-3. All helpers self-skip when
# no dVRK/cpuset is present, so this is safe in sim-only runs. DGS_NO_CPU_PIN=1 disables.
source "$SCRIPTS_DIR/scripts/_ros_cleanup.sh"
dgs_export_thread_caps
DGS_PIN="$(dgs_cpu_pin_prefix)"
dgs_isolate_dvrk
echo "[resume_live] LIVE dynamic phase on $DATA  $FF_FLAG  (pin: ${DGS_PIN:-none})"
exec $DGS_PIN "$PY" -m dynamic_gs2.pipeline --mode live --data "$DATA" --source live_bridge $FF_FLAG
