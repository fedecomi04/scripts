#!/bin/bash
# resume_live.sh -- run dynamic-gs-live against an already-captured +
# already-trained scene (the output of bootstrap_live.sh). Stage 3 only.
#
# Usage:
#   scripts/resume_live.sh <data_dir>
#
# Requires <data_dir>/static_scene/post_fusion_state.pt to exist
# (produced by `bootstrap_live.sh` or by `ns-train static-gs --data <dir>`).

set -euo pipefail

DATA_DIR="${1:-}"
if [[ -z "$DATA_DIR" ]]; then
  echo "usage: $(basename "$0") <data_dir>" >&2
  exit 2
fi
DATA_DIR="$(realpath -m "$DATA_DIR")"

WARM_CACHE="$DATA_DIR/static_scene/static_state.pt"
if [[ ! -f "$WARM_CACHE" ]]; then
  LEGACY="$DATA_DIR/static_scene/post_fusion_state.pt"
  if [[ -f "$LEGACY" ]]; then
    WARM_CACHE="$LEGACY"
  else
    echo "no warm cache at $WARM_CACHE (or legacy $LEGACY)" >&2
    echo "run scripts/bootstrap_live.sh $DATA_DIR  first (or ns-train static-gs[-preseg])" >&2
    exit 1
  fi
fi

CONDA_ROOT=/home/mrc-cuhk/miniconda3
TRAIN_ENV=dynamic_gs
TRAIN_PREFIX="$CONDA_ROOT/envs/$TRAIN_ENV"
NS_TRAIN="$TRAIN_PREFIX/bin/ns-train"
OUTPUT_DIR=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/outputs

export PATH="$TRAIN_PREFIX/bin:$PATH"
export CONDA_DEFAULT_ENV="$TRAIN_ENV"
export CONDA_PREFIX="$TRAIN_PREFIX"
export LD_LIBRARY_PATH="$TRAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$TRAIN_PREFIX"
export CPATH="$TRAIN_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$TRAIN_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export DGS_LIVE_ROOT="$DATA_DIR"

echo
echo "===> resume_live :: data_dir=$DATA_DIR"
echo "     warm cache: $WARM_CACHE ($(du -h "$WARM_CACHE" | cut -f1))"
echo "     viser at http://localhost:8081 -- Ctrl+C or 'stop' to end"
echo

# Scoped flush of any leaked publisher/worker from a previous unclean run,
# so the fresh live publisher doesn't hang on "waiting for /camera_info".
# (Replaces the need to restart Gazebo between runs.)
source "$(dirname "$0")/_ros_cleanup.sh"
dgs_ros_cleanup
dgs_check_sim_alive || exit 1

# Pin the pipeline (+ publisher) off the dVRK's RT cores + cap thread pools
# so the tracker/AnySplat/CUDA-sync load can't starve the dVRK 1 kHz loop
# ("power is unexpectedly off"). See _ros_cleanup.sh.
dgs_export_thread_caps
# Lock the dVRK RT console ONTO the reserved cores (the pin only keeps the
# pipeline OFF them). See dgs_isolate_dvrk / _dvrk_cpuset_watch.sh.
dgs_isolate_dvrk
# Per-FF-call CDN debug dump (-> <data>/dynamic_scene/_ff_debug/, the ordered
# _1..._7 frames). OFF by default (extra renders on the FF thread); enable for
# diagnosing change-detection churn:  DGS_FF_DEBUG=1 scripts/resume_live.sh <dir>
[[ "${DGS_FF_DEBUG:-0}" == "1" ]] && SAVE_DBG=True || SAVE_DBG=False
exec $(dgs_cpu_pin_prefix) "$NS_TRAIN" dynamic-gs-live \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --logging.local-writer.enable False \
  --logging.profiler none \
  --pipeline.enable_viser_direct=True \
  --pipeline.enable-feedforward-inpaint=anysplat_decode \
  --pipeline.save-debug-images="$SAVE_DBG" \
  --pipeline.live-wipe-root=False
