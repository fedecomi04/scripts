#!/usr/bin/env bash
# Deploy + run the ZED Mini -> ROS publisher on the Jetson, from the PC.
# Run this on the PC: ./run_zed_publisher.sh   (Ctrl-C stops the remote node cleanly)
set -euo pipefail

JETSON=shengzhiwang@192.168.55.1
REMOTE_DIR=/home/shengzhiwang/zed_pub
NODE=zed_mini_publisher.py
VENV_PY=/home/shengzhiwang/zed_env/bin/python
MASTER=${ROS_MASTER_URI:-http://localhost:11311}   # set ROS_MASTER_URI to the robot master if needed

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) deploy the node
ssh "$JETSON" "mkdir -p $REMOTE_DIR"
scp "$DIR/$NODE" "$JETSON:$REMOTE_DIR/"

# 2) launch on the Jetson (env inline; -t so Ctrl-C reaches the node)
#    ROS path ONLY: /usr/lib numpy is older and breaks the pyzed ABI. venv has cv2.
ssh -t "$JETSON" "
  source /opt/ros/noetic/setup.bash
  export ROS_MASTER_URI=$MASTER ROS_HOSTNAME=localhost
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\${PYTHONPATH:-}
  rosnode list >/dev/null 2>&1 || { roscore & sleep 4; }
  exec $VENV_PY $REMOTE_DIR/$NODE
"
