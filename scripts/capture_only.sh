#!/bin/bash
# capture_only.sh -- record a static+dynamic dataset, no training.
#
# Usage:
#   scripts/capture_only.sh <data_dir>
#
# Two phases, switched by Enter:
#   STATIC  -- publisher dedup-recorder (2cm OR 20°) + concurrent ICP+TSDF
#              fusion (init seed). Press ENTER to end.
#   DYNAMIC -- reader-side 30 fps polling, no dedup. Press ENTER to end + exit.

set -euo pipefail

# Resolve <data_dir>:
#   * absolute or starts with "./" / "../"  → use as-is
#   * otherwise → treat as a name under DATASETS_ROOT (so the user can
#     type `capture_only.sh new_scene_2026-05-31` and it lands in the
#     standard datasets dir without re-typing the path)
DATASETS_ROOT=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets

DATA_DIR="${1:-}"
if [[ -z "$DATA_DIR" ]]; then
  # No arg → default to a timestamped subdir of the datasets root.
  DATA_DIR="$(date +'%Y-%m-%d_%H%M%S')"
fi
case "$DATA_DIR" in
  /*|./*|../*) ;;                          # absolute or explicit relative — leave alone
  *) DATA_DIR="$DATASETS_ROOT/$DATA_DIR" ;; # bare name — prepend the root
esac
DATA_DIR="$(realpath -m "$DATA_DIR")"

CONDA_ROOT=/home/mrc-cuhk/miniconda3
TRAIN_ENV=dynamic_gs
TRAIN_PREFIX="$CONDA_ROOT/envs/$TRAIN_ENV"
PY="$TRAIN_PREFIX/bin/python"

export PATH="$TRAIN_PREFIX/bin:$PATH"
export CONDA_DEFAULT_ENV="$TRAIN_ENV"
export CONDA_PREFIX="$TRAIN_PREFIX"
export LD_LIBRARY_PATH="$TRAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# CUDA build toolchain (sm_120) — see bootstrap_live.sh. Capture-only doesn't
# train, but keep the three launch scripts uniform so any gsplat import works.
export CUDA_HOME="$TRAIN_PREFIX"
export CPATH="$TRAIN_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$TRAIN_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export DGS_LIVE_ROOT="$DATA_DIR"

# Scoped flush of any leaked publisher/worker from a previous unclean run,
# so the fresh publisher doesn't hang on "waiting for /camera_info".
# (Replaces the need to restart Gazebo between runs.)
source "$(dirname "$0")/_ros_cleanup.sh"
dgs_ros_cleanup
dgs_check_sim_alive || exit 1

# Pin the pipeline (+ the publisher it spawns, via inherited affinity) off
# the cores the dVRK 1 kHz control loop needs, and cap the BLAS/OpenMP
# thread pools. Prevents the gsplat CUDA JIT / CUDA-sync threads from
# starving the dVRK loop -> "power is unexpectedly off". See _ros_cleanup.sh.
dgs_export_thread_caps
exec $(dgs_cpu_pin_prefix) "$PY" -u "$(dirname "$0")/capture_only.py" "$DATA_DIR" "${@:2}"
