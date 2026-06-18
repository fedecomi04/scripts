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
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
echo "[resume_live] LIVE dynamic phase on $DATA  $FF_FLAG"
exec "$PY" -m dynamic_gs2.pipeline --mode live --data "$DATA" --source live_bridge $FF_FLAG
