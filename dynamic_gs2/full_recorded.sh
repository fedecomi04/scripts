#!/usr/bin/env bash
# dynamic_gs2 MODE 2/4 — FULL RECORDED: the whole pipeline (static train+fuse -> dynamic) on a
# RECORDED dataset. No sim, no UI. The static phase reuses static_scene/ on disk (anchor = the LAST
# keyframe, or --trigger-frame N) and reuses its seed PLY; the dynamic phase replays dynamic_scene/
# through SHM exactly like live. Phase boundaries are printed to the terminal.
#
# Usage:  dynamic_gs2/full_recorded.sh <data_dir> [prompt] [--trigger-frame N] [--no-ff]
#   needs: <data_dir>/static_scene/{transforms.json, depth_camera_init_points.ply}
#          <data_dir>/dynamic_scene/transforms.json
set -euo pipefail

DATA="${1:?usage: full_recorded.sh <data_dir> [prompt] [--trigger-frame N] [--no-ff]}"
PROMPT="${2:-}"; [ "${PROMPT#--}" != "$PROMPT" ] && PROMPT=""   # drop a leading --flag mistaken as the prompt
TRIG=""; FF_FLAG="--ff"
args=("$@")
for i in "${!args[@]}"; do
  a="${args[$i]}"
  [ "$a" = "--trigger-frame" ] && TRIG="--trigger-frame ${args[$((i+1))]}"
  [ "$a" = "--no-ff" ] && FF_FLAG="--no-ff"
done

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../scripts
ENV="$HOME/miniconda3/envs/dynamic_gs"
PY="$ENV/bin/python"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
export CUDA_HOME="$ENV"                                              # gsplat JIT needs these on a bare-python
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"    # launch (else cuda_runtime.h not found)
# No sim -> no ROS cleanup / dVRK isolation (they only matter when Gazebo is up).

echo "[full_recorded] whole pipeline on recorded data: static (reuse static_scene/) -> dynamic (replay). data=$DATA prompt='${PROMPT}' ${TRIG}"
exec "$PY" -m dynamic_gs2.pipeline --mode full --source replay --data "$DATA" --prompt "$PROMPT" $TRIG $FF_FLAG
