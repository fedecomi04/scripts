#!/usr/bin/env bash
# dynamic_gs2 — GROUND-TRUTH RECORDER for tracking-accuracy evaluation.
#
# Logs the true Gazebo pose of the manipulated object on the SIM clock, so it can be compared
# against the tracker's estimate. The estimate side needs no recorder: the pipeline already
# writes <data_dir>/object_track_poses.jsonl every tick.
#
# Start this FIRST, in its own terminal, then launch the pipeline with the SAME <data_dir>.
# Ctrl-C it when the run ends. Then analyse with _eval_pose_rmse.py.
#
#   DATA="/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/$(date +%Y-%m-%d_%H%M%S)"
#   dynamic_gs2/record_gt.sh "$DATA"                 # terminal 1  (leave running)
#   dynamic_gs2/full_live.sh "$DATA" "screwdriver"   # terminal 2
#
# NOTE: set DATA ONCE as above. Writing $(date ...) separately in each command produces two
# DIFFERENT directories and the estimate and ground truth would never meet.
#
# Usage:  dynamic_gs2/record_gt.sh <data_dir> [model] [hz]
#   model defaults to auto-pick (the one model that is not the robot/ground/sun); pass it
#   explicitly if the scene holds several objects. hz defaults to 10.
set -euo pipefail

DATA="${1:?usage: record_gt.sh <data_dir> [model] [hz]}"
MODEL="${2:-}"
# Cap on the internal truth buffer, NOT the output rate: one row is written per camera frame, at
# that frame's own stamp, so the log lands exactly on the timestamps the tracker logs against.
# The buffer only has to be dense enough that filling the gap between two truth samples is
# negligible -- 200 Hz puts that at micrometres.
HZ="${3:-200}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../scripts
ROS_ENV="$HOME/miniconda3/envs/dynamic_gs_ros"
ROS_SETUP="/opt/ros/noetic/setup.bash"
OUT="$DATA/gt_object_poses.csv"
mkdir -p "$DATA"

ARGS=(--out "$OUT" --buffer-hz "$HZ")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")

echo "[record_gt] logging gazebo GT -> $OUT (model='${MODEL:-auto}', <=${HZ}Hz). Ctrl-C to stop."
cd "$SCRIPTS_DIR"
# Same env discipline as the live publisher: the ROS env's own python, ROS Noetic sourced, and
# PYTHONNOUSERSITE so the user-local py3.8 site-packages cannot shadow the env's.
exec bash -c "source '$ROS_SETUP' && export PYTHONNOUSERSITE=1 && exec '$ROS_ENV/bin/python' -u \
  '$SCRIPTS_DIR/dynamic_gs2/_record_gazebo_gt.py' $(printf "'%s' " "${ARGS[@]}")"
