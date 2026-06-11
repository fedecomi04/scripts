#!/bin/bash
# replay_pipeline.sh -- run the FULL dynamic-gs pipeline (capture -> static ->
# teleop) deterministically against a RECORDED teleop rosbag, by force-tracking
# the Gazebo sim to the recording (live_joint_replay.py). No robot operator
# needed: drives the camera sweep for capture, fires the capture Enters, then
# drives the dynamic camera+object motion for teleop. Enables autonomous
# iteration on the pipeline.
#
# Prereqs: Gazebo + roscore up (the same world the bag was recorded in).
# Usage:
#   scripts/replay_pipeline.sh <teleop.bag> [object_model] [sam3_prompt]
#
# Tunables (env): DGS_STATIC_SWEEP_S (when to fire capture Enter, def 44),
#   DGS_DYN_START (bag-time of dynamic phase, def 46), DGS_DYN_DUR (def 53).
set -uo pipefail

BAG="${1:?usage: replay_pipeline.sh <teleop.bag> [object_model] [sam3_prompt]}"
OBJ="${2:-Craftsman_Grip_Screwdriver_Phillips_Cushion}"
PROMPT="${3:-screwdriver}"
STATIC_SWEEP_S="${DGS_STATIC_SWEEP_S:-44}"
DYN_START="${DGS_DYN_START:-46}"
DYN_DUR="${DGS_DYN_DUR:-53}"

ROOT=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
DATA=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/replay_$(date +%Y%m%d_%H%M%S)
LOG=/tmp/replay_pipeline.log
FIFO=/tmp/dgs_capture_stdin
ROS_SETUP=/opt/ros/noetic/setup.bash
SYS_PY=/usr/bin/python3
cd "$ROOT"
source "$ROS_SETUP" 2>/dev/null

_wait_log() {  # $1=regex $2=timeout_s
  local re="$1" to="$2" i=0
  while [ "$i" -lt "$to" ]; do
    grep -qiE "$re" "$LOG" 2>/dev/null && return 0
    sleep 1; i=$((i+1))
  done
  return 1
}

echo "==> [replay] cleanup + reset"
pgrep -af "run_live_capture|sam_worker.py|live_ros_publisher|anysplat_worker|live_joint_replay|bootstrap_live|dgs_capture_stdin" \
  | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 2; rm -f /dev/shm/dgs* "$FIFO" "$LOG" 2>/dev/null || true
timeout 6 rosservice call /gazebo/unpause_physics >/dev/null 2>&1 || true  # killed runs can leave physics paused (pose plugin goes silent)
timeout 6 rosservice call /gazebo/reset_world >/dev/null 2>&1 || true
sleep 2

echo "==> [replay] launch bootstrap (capture->static->teleop) data=$DATA"
mkfifo "$FIFO"
setsid bash -c "exec 9>'$FIFO'; sleep 7200" >/dev/null 2>&1 &
nohup bash scripts/bootstrap_live.sh "$DATA" "$PROMPT" < "$FIFO" > "$LOG" 2>&1 &
_wait_log "recording started" 240 || { echo "capture never started"; exit 1; }  # 1200p cold start can exceed 90s

echo "==> [replay] drive static sweep [0->$((STATIC_SWEEP_S+4))s]"
nohup "$SYS_PY" scripts/live_joint_replay.py --bag "$BAG" --start 0 \
  --duration $((STATIC_SWEEP_S+4)) --rate 60 > /tmp/replay_static.log 2>&1 &
sleep "$STATIC_SWEEP_S"
echo "==> [replay] fire capture ENTER #1 (segment)"; printf '\n' > "$FIFO"

_wait_log "sweep the scene to capture more views|SAM3D done" 180 \
  && { echo "==> [replay] fire capture ENTER #2 (done capturing)"; printf '\n' > "$FIFO"; } \
  || echo "(SAM3D/segment marker not seen; continuing)"

echo "==> [replay] waiting for teleop (stage 3)"
_wait_log "spawning ROS publisher|GO-LIVE|recurring call=" 300 || echo "(teleop marker not seen)"
sleep 3
echo "==> [replay] drive dynamic motion [${DYN_START}->$((DYN_START+DYN_DUR))s] (object=$OBJ)"
"$SYS_PY" scripts/live_joint_replay.py --bag "$BAG" --start "$DYN_START" \
  --duration "$DYN_DUR" --rate 60 --objects "$OBJ"

echo "==> [replay] dynamic motion done. Teleop still running (viser :8081)."
echo "    data=$DATA  log=$LOG"
echo "    to stop teleop:  kill -INT \$(pgrep -f 'ns-train dynamic-gs-live')"
