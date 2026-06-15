#!/bin/bash
# _ros_cleanup.sh -- scoped pre-run cleanup of leaked ROS publisher state.
#
# Sourced by bootstrap_live.sh / capture_only.sh / resume_live.sh BEFORE
# they spawn a fresh publisher. An unclean previous run (Ctrl-C, crash,
# dVRK power-off mid-session, wrong-PID kill) leaves a stale
# `dynamic_gs_live_pub_*` node + an orphaned `image_transport republish`
# still subscribed to /camera_info + the image topics. The next run's
# publisher then hangs on "waiting for /camera_info" / "spawning ROS
# publisher". Restarting Gazebo "fixes" it only by wiping the whole ROS
# graph -- this does the same scoped flush WITHOUT touching the sim.
#
# SAFETY (per project rules): this NEVER blanket-kills python and NEVER
# touches Gazebo / gzserver / roscore / the dVRK. It targets only our own
# leaked processes by exact name, and prints what it will kill first.
#
# It ALSO provides dgs_cpu_pin_prefix: a taskset+thread-cap prefix that
# keeps the heavy pipeline OFF the cores the dVRK real-time control loop
# needs. The dVRK's mtsRobotIO1394 runs a 1 kHz / 1 ms-deadline I/O loop at
# SCHED_OTHER on shared cores; when the pipeline's gsplat CUDA JIT (cicc
# storm) or CUDA-sync/tracker threads saturate that core for >1 ms, the loop
# misses its deadline and the controller cuts actuator power ("power is
# unexpectedly off" / "detected power loss"). MEASURED 2026-06-14: the dVRK
# + Gazebo + controllers real-time domain peaks at ~2.9 cores during teleop
# motion, so cores 0-3 are reserved for it and the pipeline is pinned to
# cores 4-23 (20 of 24). This is a CPU-latency fix, not GPU (the GPU was
# ~18% busy at fault time and the dVRK never touches it).
#
# The reservation has TWO sides and you need BOTH:
#   * dgs_cpu_pin_prefix  -- keeps the PIPELINE OFF cores 0-3 (taskset 4-23).
#   * dgs_isolate_dvrk    -- locks the dVRK ONTO cores 0-3 (cgroup cpuset).
# Pinning only the pipeline is NOT enough: the dVRK loop is SCHED_OTHER with
# affinity 0-23, so the scheduler still drops it onto 4-23 next to the gsplat
# compile storm and starves it (observed 2026-06-14: arms latched "not ready"
# mid-JIT even with the pipeline correctly pinned). dgs_isolate_dvrk closes it.

# Cores reserved for the dVRK/Gazebo/controllers RT domain. Override with
# DGS_RT_RESERVED_CORES; the pipeline gets every other core.
DGS_RT_RESERVED_CORES="${DGS_RT_RESERVED_CORES:-0-3}"

# Echo a command prefix that pins what follows to the pipeline cores +
# caps the BLAS/OpenMP thread pools so they don't spawn a thread per core
# (which would defeat the pin and oversubscribe). Use as:
#     exec $(dgs_cpu_pin_prefix) ns-train ...
# Set DGS_NO_CPU_PIN=1 to disable (e.g. when no dVRK is connected).
dgs_cpu_pin_prefix() {
  if [[ "${DGS_NO_CPU_PIN:-0}" == "1" ]]; then
    return 0
  fi
  # Pipeline cores = all cores NOT in the reserved range. Compute the
  # complement of DGS_RT_RESERVED_CORES over [0, nproc-1].
  local ncpu
  ncpu=$(nproc --all)
  local last_reserved="${DGS_RT_RESERVED_CORES##*-}"  # "0-3" -> "3"
  local pipe_lo=$((last_reserved + 1))
  local pipe_hi=$((ncpu - 1))
  if (( pipe_lo > pipe_hi )); then
    # Degenerate (reserved >= all cores) -> don't pin, warn to stderr.
    echo "[cpu-pin] WARNING: reserved range $DGS_RT_RESERVED_CORES covers all $ncpu cores; not pinning" >&2
    return 0
  fi
  echo "taskset -c ${pipe_lo}-${pipe_hi}"
}

# Export the thread caps into the environment of whatever the script execs
# next. Caps each math/threadpool lib to a modest count so the pipeline
# doesn't fan out a thread per core (which oversubscribes the pinned set
# and bleeds onto the reserved cores via the scheduler). Tuned for the
# 20-core pipeline slice; override individually if needed.
dgs_export_thread_caps() {
  if [[ "${DGS_NO_CPU_PIN:-0}" == "1" ]]; then
    return 0
  fi
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
  # gsplat's first-run CUDA JIT spawns a cicc compiler per core; cap the
  # nvcc build parallelism so the compile storm can't peg every core.
  export MAX_JOBS="${MAX_JOBS:-8}"
}

# Lock the dVRK real-time console ONTO the reserved cores. This is the
# counterpart to dgs_cpu_pin_prefix: that keeps the pipeline OFF cores 0-3, but
# nothing kept the dVRK ON them -- its mtsRobotIO1394 1 kHz loop runs
# SCHED_OTHER with affinity 0-23, so the scheduler still co-located it with the
# gsplat compile storm on 4-23 and starved it (MEASURED 2026-06-14: arms latched
# "body_servo_cf, arm not ready" during the JIT build even with the pipeline
# pinned). Starts a persistent ROOT watchdog (cgroup v1 cpuset, see
# _dvrk_cpuset_watch.sh) that confines every current + future dvrk_console_json
# thread to DGS_RT_RESERVED_CORES and survives operator restarts of the console.
# Needs sudo (one prompt per launch, cached after). Non-fatal: if sudo/cpuset is
# unavailable it warns and continues. Skipped when no dVRK console is running
# (pure recorded runs) or DGS_NO_CPU_PIN=1.
dgs_isolate_dvrk() {
  if [[ "${DGS_NO_CPU_PIN:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -d /sys/fs/cgroup/cpuset ]]; then
    echo "[dvrk-isolate] cpuset cgroup v1 not present; skipping dVRK isolation" >&2
    return 0
  fi
  if ! pgrep -f dvrk_console_json >/dev/null 2>&1; then
    echo "[dvrk-isolate] no dVRK console running; skipping (start the dVRK first if teleoperating)" >&2
    return 0
  fi
  local watch
  watch="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dvrk_cpuset_watch.sh"
  if [[ ! -x "$watch" ]]; then
    echo "[dvrk-isolate] watchdog missing/not executable: $watch -- skipping" >&2
    return 0
  fi
  echo "==> [dvrk-isolate] locking dVRK RT console to cores ${DGS_RT_RESERVED_CORES} (sudo; one prompt)"
  # Run the watchdog as a persistent ROOT daemon. Prefer a systemd transient
  # unit -- it survives reliably; a bare `setsid ... &` under `sudo bash -c`
  # gets reaped when sudo returns (verified 2026-06-14), so new dVRK threads
  # then escape the cpuset. Fall back to setsid+disown if systemd-run is absent.
  if command -v systemd-run >/dev/null 2>&1; then
    if sudo sh -c "systemctl reset-failed dvrk-cpuset.service 2>/dev/null; systemctl stop dvrk-cpuset.service 2>/dev/null; systemd-run --quiet --unit=dvrk-cpuset --description='Lock dVRK RT console to reserved CPU cores' '$watch' '$DGS_RT_RESERVED_CORES'"; then
      echo "==> [dvrk-isolate] OK -- dVRK confined to ${DGS_RT_RESERVED_CORES} (systemd unit dvrk-cpuset; catches restarts)"
    else
      echo "[dvrk-isolate] WARNING: sudo/systemd-run failed -- dVRK NOT isolated; RT loop may be starved" >&2
    fi
  else
    if sudo bash -c "pkill -f _dvrk_cpuset_watch.sh 2>/dev/null; setsid '$watch' '$DGS_RT_RESERVED_CORES' </dev/null >/dev/null 2>&1 & disown"; then
      echo "==> [dvrk-isolate] OK -- dVRK confined to ${DGS_RT_RESERVED_CORES} (setsid watchdog; catches restarts)"
    else
      echo "[dvrk-isolate] WARNING: sudo failed -- dVRK NOT isolated; RT loop may be starved" >&2
    fi
  fi
}

# Run a ROS CLI command (rostopic/rosnode/rosservice) with the conda
# contamination stripped. The launch scripts put the dynamic_gs conda env
# first on PATH + LD_LIBRARY_PATH, which shadows the system python the ROS
# CLI tools need -> they crash with "ModuleNotFoundError: No module named
# 'cffi'" and every ROS call silently fails (false-negative preflight,
# no-op cleanup). The ROS env (ROS_MASTER_URI / PYTHONPATH / ROS package
# paths) is inherited from the parent shell where ROS was sourced; we only
# remove the conda bits. /usr/bin + /opt/ros/noetic/bin remain on PATH so
# `timeout` and `rostopic` still resolve.
_dgs_ros() {
  env -u LD_LIBRARY_PATH \
    PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -e miniconda -e conda | paste -sd:)" \
    "$@"
}

dgs_ros_cleanup() {
  echo "==> [cleanup] flushing any leaked publisher/worker from a prior run"
  # Our own leaked processes only -- matched by the exact script/module names.
  local pat="live_ros_publisher|sam_worker.py|anysplat_worker|run_live_capture"
  local pids
  # `|| true`: under `set -e` in the caller, an empty pgrep match (the normal
  # clean case) makes this pipeline exit non-zero and would abort the script.
  pids=$(pgrep -af "$pat" 2>/dev/null | grep -v grep | awk '{print $1}' || true)
  if [[ -n "$pids" ]]; then
    echo "    killing: $pids"
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  else
    echo "    (no leaked publisher/worker procs)"
  fi
  # The auto-launched C++ depth republisher leaks separately; scope it tightly
  # to the compressedDepth transport so we never touch unrelated republishers.
  pkill -9 -f "republish.*compressedDepth" 2>/dev/null || true
  # Purge dead/unreachable nodes from the ROS master (the stale
  # dynamic_gs_live_pub_* registration). Harmless if the master is down.
  _dgs_ros timeout 8 rosnode cleanup <<<$'y\n' >/dev/null 2>&1 || true
  # Free the shared-memory frame buffer left behind by a killed publisher.
  rm -f /dev/shm/dgs* 2>/dev/null || true
  # A killed run can leave Gazebo physics PAUSED (live_session pauses it for
  # the SAM3D window) -> the camera-pose plugin goes silent -> the new
  # publisher never gets a first synced frame. Unpause defensively.
  _dgs_ros timeout 6 rosservice call /gazebo/unpause_physics >/dev/null 2>&1 || true
  sleep 1
  echo "==> [cleanup] done"
}

# Pre-flight: verify the sim is actually alive (camera-pose topic publishing)
# BEFORE we spawn the publisher. Without this the publisher just hangs on
# "waiting for /camera_info" for ~90 s before timing out. The camera-pose
# plugin goes silent when Gazebo is dead/paused or the arm controllers
# aren't running -- a state no process cleanup can fix (it's sim-side).
# Returns 0 if a pose message arrives within the window; else prints a
# "Restart Gazebo" message and returns 1 so the caller aborts.
DGS_POSE_TOPIC="/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose"
# Per-attempt timeout (s) and number of attempts. The pose plugin can publish
# at a LOW effective wall-clock rate (sim-time + host load), so a single short
# wait false-negatives on a healthy sim (observed 2026-06-14: a message took
# >8 s but <12 s to arrive). Wait longer + retry. ~25 s total before giving up.
DGS_POSE_WAIT_S="${DGS_POSE_WAIT_S:-12}"
DGS_POSE_TRIES="${DGS_POSE_TRIES:-2}"
dgs_check_sim_alive() {
  echo "==> [preflight] checking sim is alive ($DGS_POSE_TOPIC publishing; up to ${DGS_POSE_TRIES}x${DGS_POSE_WAIT_S}s)"
  # rostopic echo -n1 returns 0 only if a message actually arrives. Run via
  # _dgs_ros so the conda env on PATH doesn't crash rostopic (cffi import)
  # and produce a false "DEAD" reading on a perfectly healthy sim.
  local try
  for try in $(seq 1 "$DGS_POSE_TRIES"); do
    if _dgs_ros timeout "$DGS_POSE_WAIT_S" rostopic echo -n1 "$DGS_POSE_TOPIC" >/dev/null 2>&1; then
      _dgs_check_sim_alive_ok=1
    else
      _dgs_check_sim_alive_ok=0
    fi
    [[ "$_dgs_check_sim_alive_ok" == "1" ]] && break
    (( try < DGS_POSE_TRIES )) && echo "    no message in ${DGS_POSE_WAIT_S}s (attempt $try/${DGS_POSE_TRIES}) — retrying..."
  done
  if [[ "$_dgs_check_sim_alive_ok" == "1" ]]; then
    echo "==> [preflight] OK -- camera pose is publishing"
    return 0
  fi
  echo "" >&2
  echo "############################################################" >&2
  echo " RESTART GAZEBO since the camera-pose topic is DEAD." >&2
  echo "   topic: $DGS_POSE_TOPIC  (no message in ${DGS_POSE_TRIES}x${DGS_POSE_WAIT_S}s)" >&2
  echo "" >&2
  echo " The publisher would just hang on 'waiting for /camera_info'." >&2
  echo " This is sim-side, not a leftover process -- causes:" >&2
  echo "   * Gazebo crashed / not running, or physics is paused" >&2
  echo "   * the dynaarm controllers aren't running (joint_state" >&2
  echo "     controller stuck 'initialized') -> FK/pose plugin silent" >&2
  echo "" >&2
  echo " Fix: relaunch the Gazebo + dvrk_dynaarm sim, confirm" >&2
  echo "   'rostopic hz $DGS_POSE_TOPIC' shows a rate, then re-run." >&2
  echo "############################################################" >&2
  return 1
}
