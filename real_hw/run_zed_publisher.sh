#!/usr/bin/env bash
# Deploy + run the ZED Mini -> ROS publisher on the Jetson, from the PC.
# Run this on the PC: ./run_zed_publisher.sh   (Ctrl-C stops the remote node cleanly)
set -euo pipefail

JETSON=shengzhiwang@192.168.55.1
JETSON_IP=192.168.55.1          # Jetson's IP on the PC link — it advertises this so the PC can pull frames
PC_IP="${DGS_PC_IP:-192.168.55.100}"   # the PC's IP on the Jetson link (where roscore must advertise)
REMOTE_DIR=/home/shengzhiwang/zed_pub
NODE=zed_mini_publisher.py
VENV_PY=/home/shengzhiwang/zed_env/bin/python
# The Jetson must point at the PC master by its REAL IP. Do NOT inherit a localhost ROS_MASTER_URI from
# the PC shell (a "localhost" master means the JETSON's own localhost -> "master not running" + freeze).
# Only honor an inherited value if it isn't localhost.
MASTER="http://$PC_IP:11311"
case "${ROS_MASTER_URI:-}" in
  ""|*localhost*|*127.0.0.1*) : ;;          # ignore unset / localhost — use the PC IP
  *) MASTER="$ROS_MASTER_URI" ;;            # honor a real explicit override
esac

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_zed_publisher] Jetson -> master $MASTER  (ROS_IP=$JETSON_IP)"

# 1) deploy the node
ssh "$JETSON" "mkdir -p $REMOTE_DIR"
scp "$DIR/$NODE" "$JETSON:$REMOTE_DIR/"

# 2) launch on the Jetson (env inline; -t so Ctrl-C reaches the node)
#    ROS path ONLY: /usr/lib numpy is older and breaks the pyzed ABI. venv has cv2.
#    ROS_IP = the Jetson's real IP (NOT localhost) so the PC master can route TCP back for image/depth.
#    The exports come AFTER source so they win over the Jetson .bashrc (which has a stale master URI).
#    No roscore fallback: the master is the PC's; a local roscore would be a competing master.
ssh -t "$JETSON" "
  source /opt/ros/noetic/setup.bash
  export ROS_MASTER_URI='$MASTER' ROS_IP='$JETSON_IP' ROS_HOSTNAME='$JETSON_IP'
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\${PYTHONPATH:-}
  echo \"[jetson] ROS_MASTER_URI=\$ROS_MASTER_URI ROS_IP=\$ROS_IP\"
  exec $VENV_PY $REMOTE_DIR/$NODE
"
