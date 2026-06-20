#!/usr/bin/env bash
# dynamic_gs2 — REPLAY-AS-LIVE: drive a recorded dataset through the LIVE pipeline.
#
# Unlike view_dynamic.sh (sequential, one frame per tracker tick), this paces the recorded frames
# into SHM on their capture schedule via a producer thread; the tracker reads the FRESHEST frame and
# drops stale ones if it falls behind — an honest real-time test (no Gazebo/ROS stack needed).
# The viser viewer comes up; open the printed URL and orbit. Plays the episode ONCE then stops.
#
# Usage:  dynamic_gs2/replay_live.sh <data_dir> [transforms_name] [--no-ff] [--fps N] [--once]
# Feedforward is ON by default (the pipeline defaults --ff true); pass --no-ff to disable.
# LOOPS the episode forever by default (Ctrl-C to stop, which still writes the FF timing report);
# pass --once to play a single pass and exit. The tracker snap-resets at each wrap.
set -euo pipefail

DATA="${1:?usage: replay_live.sh <data_dir> [transforms_name] [--no-ff] [--fps N] [--once]}"
TJ="${2:-transforms.json}"
FF_FLAG=""
FPS="15"
LOOP_FLAG="--loop"
args=("$@")
for i in "${!args[@]}"; do
  a="${args[$i]}"
  [ "$a" = "--no-ff" ] && FF_FLAG="--no-ff"
  [ "$a" = "--once" ] && LOOP_FLAG=""
  [ "$a" = "--fps" ] && FPS="${args[$((i+1))]}"
done

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_LIB="$HOME/miniconda3/envs/dynamic_gs/lib"
PY="$HOME/miniconda3/envs/dynamic_gs/bin/python"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
echo "[replay_live] paced replay-as-live ($TJ, ${FPS}fps) $FF_FLAG ${LOOP_FLAG:-(once)}"
exec "$PY" -m dynamic_gs2.pipeline --mode live --source replay --data "$DATA" \
  --transforms "$TJ" --fps "$FPS" $FF_FLAG $LOOP_FLAG
