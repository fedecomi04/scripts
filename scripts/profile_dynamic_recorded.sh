#!/bin/bash
# profile_dynamic_recorded.sh -- profile the DYNAMIC loop on an already-built
# RECORDED dataset with torch.profiler (CPU+CUDA op-level), dVRK-SAFE.
#
# Why this script exists: torch.profiler is the ONLY profiler safe to run while
# the dVRK is live (it is in-process CUDA-event timing; NO nsys, NO `-s cpu`
# CPU-sampling, NO perf_event_paranoid change -> no kernel-wide perf interrupts
# on the dVRK's reserved RT cores 0-3). It STILL pins the pipeline off cores 0-3
# (dgs_cpu_pin_prefix + dgs_isolate_dvrk), exactly like resume_live.sh — a bare
# `ns-train dynamic-gs` would run unpinned on all cores and can starve the dVRK
# 1 kHz loop. DO NOT run ns-train directly while the dVRK is up.
#
# Recorded (not live): no ROS publisher, no sim needed -> the only GPU/CPU load
# is the pipeline itself, which is also the cleanest profile.
#
# Usage:
#   scripts/profile_dynamic_recorded.sh <data_dir>
#     DGS_TPROF_WAIT (default 30)  ticks to skip before recording
#     DGS_TPROF_ACTIVE (default 120) ticks to record
#
# Output: <data_dir>/profiling/dynamic_torch_profile.txt  (top CPU/CUDA ops)
#         <data_dir>/profiling/dynamic_torch_trace.json    (chrome trace)
#         <data_dir>/timing_report.txt                      (wall-clock buckets)

set -euo pipefail

DATA_DIR="${1:-}"
if [[ -z "$DATA_DIR" ]]; then
  echo "usage: $(basename "$0") <data_dir>" >&2
  exit 2
fi
DATA_DIR="$(realpath -m "$DATA_DIR")"

WARM_CACHE="$DATA_DIR/static_scene/static_state.pt"
[[ -f "$WARM_CACHE" ]] || WARM_CACHE="$DATA_DIR/static_scene/post_fusion_state.pt"
if [[ ! -f "$WARM_CACHE" ]]; then
  echo "no warm cache at $DATA_DIR/static_scene/{static_state,post_fusion_state}.pt" >&2
  echo "run ns-train static-gs --data $DATA_DIR  first" >&2
  exit 1
fi
if [[ "$(ls "$DATA_DIR/dynamic_scene/rgb" 2>/dev/null | wc -l)" -lt 2 ]]; then
  echo "no recorded dynamic episode in $DATA_DIR/dynamic_scene/rgb (need a recorded run)" >&2
  exit 1
fi

CONDA_ROOT=/home/mrc-cuhk/miniconda3
TRAIN_PREFIX="$CONDA_ROOT/envs/dynamic_gs"
NS_TRAIN="$TRAIN_PREFIX/bin/ns-train"
OUTPUT_DIR=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/outputs

export PATH="$TRAIN_PREFIX/bin:$PATH"
export CONDA_DEFAULT_ENV=dynamic_gs CONDA_PREFIX="$TRAIN_PREFIX"
export LD_LIBRARY_PATH="$TRAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$TRAIN_PREFIX"
export CPATH="$TRAIN_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$TRAIN_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

# torch.profiler ON (the safe profiler).
export DGS_TORCH_PROFILE=1
export DGS_TPROF_WAIT="${DGS_TPROF_WAIT:-30}"
export DGS_TPROF_ACTIVE="${DGS_TPROF_ACTIVE:-120}"

echo "===> profile_dynamic_recorded :: $DATA_DIR"
echo "     warm cache: $WARM_CACHE"
echo "     torch.profiler: wait=$DGS_TPROF_WAIT active=$DGS_TPROF_ACTIVE -> $DATA_DIR/profiling/"
echo "     dVRK-safe: pinned off cores 0-3, NO nsys/perf sampling"

# dVRK isolation — identical to resume_live.sh. Keeps the pipeline OFF cores 0-3
# and locks the dVRK RT console ONTO them.
source "$(dirname "$0")/_ros_cleanup.sh"
dgs_export_thread_caps
dgs_isolate_dvrk

exec $(dgs_cpu_pin_prefix) "$NS_TRAIN" dynamic-gs \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --pipeline.enable_viser_direct=True \
  --pipeline.enable-feedforward-inpaint=anysplat_decode
