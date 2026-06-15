#!/usr/bin/env bash
# _dvrk_cpuset_watch.sh -- kernel-pin the dVRK real-time console onto the
# reserved CPU cores, surviving console restarts.
#
# Counterpart to dgs_cpu_pin_prefix in _ros_cleanup.sh: that pins the heavy GS
# PIPELINE (ns-train / gsplat CUDA JIT / AnySplat / SAM3D) OFF the reserved
# cores; THIS pins the dVRK ONTO them. Reserving cores for the pipeline is not
# enough on its own -- the dVRK's mtsRobotIO1394 1 kHz / 1 ms-deadline I/O loop
# runs SCHED_OTHER with affinity 0-23, so the scheduler happily places it on
# the very cores the gsplat compile storm is saturating -> the loop misses its
# deadline -> the firmware watchdog latches the arms "not ready" / cuts power.
#
# A persistent loop run as ROOT is required (not a one-shot taskset) because:
#   * the dVRK console spawns worker processes/threads continuously, and
#   * the operator restarts the console between attempts (new PIDs).
# cgroup v1 cpuset is used so the confinement is kernel-enforced on every
# current AND future thread of each matched process (writing the PID to
# cgroup.procs moves all its threads; children inherit the cgroup).
#
# Usage (as root):  _dvrk_cpuset_watch.sh [reserved_cpus]   # default 0-3
set -u
RESERVED="${1:-0-3}"
CS=/sys/fs/cgroup/cpuset/dvrk

if [ ! -d /sys/fs/cgroup/cpuset ]; then
  echo "[dvrk-watch] cpuset cgroup v1 not mounted; nothing to do" >&2
  exit 0
fi

mkdir -p "$CS"
# cpuset v1 refuses task moves until cpuset.mems is set; inherit the root node's
# memory nodes, then constrain the CPUs.
cat /sys/fs/cgroup/cpuset/cpuset.mems > "$CS/cpuset.mems" 2>/dev/null
echo "$RESERVED" > "$CS/cpuset.cpus"

while true; do
  for p in $(pgrep -f dvrk_console_json 2>/dev/null); do
    echo "$p" > "$CS/cgroup.procs" 2>/dev/null
  done
  sleep 2
done
