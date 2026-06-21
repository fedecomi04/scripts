#!/usr/bin/env bash
# dynamic_gs2 MODE 4/4 — WARM RECORDED: SKIP the static phase; warm-load static_state.pt + AnySplat/
# XFeat, then replay dynamic_scene/ through SHM (no sim). The fast dev/debug path to see the dynamic
# phase on recorded data WITHOUT retraining the static scene.
#
# Usage:  dynamic_gs2/warm_recorded.sh <data_dir> [transforms_name] [--no-ff] [--fps N] [--loop]
# Needs:  <data_dir>/static_scene/static_state.pt  +  <data_dir>/dynamic_scene/<transforms_name>
set -euo pipefail

DATA="${1:?usage: warm_recorded.sh <data_dir> [transforms_name] [--no-ff] [--fps N] [--loop]}"
TJ="transforms.json"; [ -n "${2:-}" ] && [ "${2#--}" = "${2:-}" ] && TJ="$2"   # 2nd positional = transforms name (not a --flag)
FF_FLAG="--ff"; LOOP=""; FPS=""
args=("$@")
for i in "${!args[@]}"; do
  a="${args[$i]}"
  [ "$a" = "--no-ff" ] && FF_FLAG="--no-ff"
  [ "$a" = "--loop" ] && LOOP="--loop"
  [ "$a" = "--fps" ] && FPS="--fps ${args[$((i+1))]}"
done

if [ ! -f "$DATA/static_scene/static_state.pt" ]; then
  echo "ERROR: $DATA/static_scene/static_state.pt not found (run full_recorded.sh first)." >&2
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
# No sim -> no ROS cleanup / dVRK isolation.

echo "[warm_recorded] warm-start dynamic, replaying $TJ on $DATA $FF_FLAG $FPS $LOOP"
exec "$PY" -m dynamic_gs2.pipeline --mode live --source replay --data "$DATA" --transforms "$TJ" $FF_FLAG $FPS $LOOP
