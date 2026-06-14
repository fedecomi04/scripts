#!/bin/bash
# bootstrap_live.sh -- one-shot fresh-scene capture + train + go-live.
#
# Usage:
#   scripts/bootstrap_live.sh <data_dir> [sam3_prompt]
#
# Stages (sequential, each blocks until done):
#   1. CAPTURE  -- spawn ROS publisher under <data_dir>, interactive scene
#                  sweep + SAM3 segmentation + SAM3D 3D-object generation.
#                  All artifacts saved under <data_dir>/static_scene/ and
#                  <data_dir>/dynamic_scene/initialization_{debug,artifacts}/.
#   2. FIT      -- ns-train static-gs trains the Splatfacto scene + runs
#                  Phase 0b CPD fusion of the SAM3D objects, writes
#                  <data_dir>/static_scene/post_fusion_state.pt (the warm
#                  cache with per-object instance_ids preserved).
#   3. GO-LIVE  -- ns-train dynamic-gs-live warm-loads that .pt and starts
#                  the XFeat tracker + AnySplat FF against fresh live frames.
#                  Viser at http://localhost:8081. Ctrl+C or 'stop' to end.
#
# After a successful bootstrap, <data_dir> is a fully reusable bundle:
# re-run just stage 3 with `scripts/resume_live.sh <data_dir>` to come
# back up without re-capturing or re-training.

set -euo pipefail

DATA_DIR="${1:-}"
SAM3_PROMPT="${2:-the can of coke on the table}"

if [[ -z "$DATA_DIR" ]]; then
  echo "usage: $(basename "$0") <data_dir> [sam3_prompt]" >&2
  echo "example: $(basename "$0") /home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/fresh_test 'red coke can'" >&2
  exit 2
fi
DATA_DIR="$(realpath -m "$DATA_DIR")"

# Cross-process "static sequence ran from scratch" stopwatch. This shell is the
# only place that sees the whole capture→train→go-live span across THREE
# separate processes (each resets its own in-process timing ledger), so it
# stamps real wall-clock boundaries into a sidecar. The dynamic-gs-live process
# reads this on the FIRST tracked frame, emits the end-to-end section in
# timing_report.txt, then DELETES the sidecar (so a later resume_live.sh on the
# same dir — which never stamps it — correctly skips the section). resume_live
# / capture_only never write this file, so the section is bootstrap-only.
SEQ_T0="$DATA_DIR/.static_sequence_t0"
mkdir -p "$DATA_DIR"
: > "$SEQ_T0"
dgs_stamp_seq() { echo "$1=$(date +%s.%N)" >> "$SEQ_T0"; }
dgs_stamp_seq t0_command

# Env setup -- pin the train env on PATH + LD_LIBRARY_PATH so the
# subprocesses inherit a consistent toolchain.
CONDA_ROOT=/home/mrc-cuhk/miniconda3
TRAIN_ENV=dynamic_gs
TRAIN_PREFIX="$CONDA_ROOT/envs/$TRAIN_ENV"
NS_TRAIN="$TRAIN_PREFIX/bin/ns-train"
PY="$TRAIN_PREFIX/bin/python"
OUTPUT_DIR=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/outputs

export PATH="$TRAIN_PREFIX/bin:$PATH"
export CONDA_DEFAULT_ENV="$TRAIN_ENV"
export CONDA_PREFIX="$TRAIN_PREFIX"
export LD_LIBRARY_PATH="$TRAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# CUDA build toolchain for gsplat's JIT (sm_120): the env ships nvcc 12.8
# under bin/ + headers under targets/ + cicc under nvvm/. CUDA_HOME must be
# the env prefix (so nvcc finds nvvm/bin/cicc), and CPATH must expose
# cuda_runtime.h, or gsplat's first rasterization JIT-compile fails with
# "cuda_runtime.h: No such file" / "cicc: not found" / "Unsupported gpu
# architecture compute_120" (the latter if /usr/local/cuda-12.1 wins PATH).
# Mirrors etc/conda/activate.d/dynamic_gs.sh + resume_live.sh.
export CUDA_HOME="$TRAIN_PREFIX"
export CPATH="$TRAIN_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$TRAIN_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

# Tell live_session + LiveShmSubscriber to use this dir (env override of
# the hardcoded LIVE_ROOT in dynamic_gs/utils/live_shm_reader.py).
export DGS_LIVE_ROOT="$DATA_DIR"

# Pass the SAM3 prompt via env so `run_live_capture_session` picks it
# up without re-asking the user interactively. The same prompt is then
# passed to the `static-gs` train command below so the pipeline's
# cached SAM3 re-read uses the matching text.
export DGS_SAM3_PROMPT="$SAM3_PROMPT"

# Eager AnySplat: stage 1 pre-spawns the detached AnySplat worker right
# after SAM3D finishes, so its ~17 s model load overlaps the sweep +
# static training. Stage 3 (dynamic-gs-live, anysplat FF) adopts it
# instead of paying the load at go-live.
export DGS_EAGER_ANYSPLAT=1

echo
echo "============================================================"
echo " bootstrap_live :: data_dir=$DATA_DIR"
echo " sam3_prompt=\"$SAM3_PROMPT\""
echo "============================================================"
echo

# Scoped flush of any leaked publisher/worker from a previous unclean run,
# so the fresh publisher below doesn't hang on "waiting for /camera_info".
# (Replaces the need to restart Gazebo between runs.)
source "$(dirname "$0")/_ros_cleanup.sh"
dgs_ros_cleanup
dgs_check_sim_alive || exit 1

# Pin every pipeline stage (+ the publisher they spawn, via inherited
# affinity) off the cores the dVRK 1 kHz control loop needs, and cap the
# BLAS/OpenMP/nvcc thread pools. Without this the gsplat CUDA JIT (cicc
# storm at first training step) + tracker/AnySplat/CUDA-sync threads
# saturate the dVRK loop's core for >1 ms -> missed deadline -> the
# controller cuts actuator power ("power is unexpectedly off"). MEASURED:
# the dVRK+Gazebo+controllers RT domain peaks ~2.9 cores, so 0-3 are
# reserved and the pipeline runs on 4-23. See _ros_cleanup.sh.
dgs_export_thread_caps
DGS_PIN="$(dgs_cpu_pin_prefix)"

# ---------------------------------------------------------------- 1/3
echo
echo "===> [1/3] CAPTURE -- live_session.run_live_capture_session()"
echo "     (sweeps the scene + SAM3 + SAM3D; follow the on-screen prompts)"
echo
dgs_stamp_seq t1_capture_start
$DGS_PIN "$PY" -u -c "
import os
from dynamic_gs.utils.live_session import run_live_capture_session
out = run_live_capture_session(sam3_prompt_text=os.environ.get('DGS_SAM3_PROMPT'))
print(f'\n[bootstrap] capture session complete -> {out}', flush=True)
"

# Safety net: live_session already runs the ICP+TSDF refinement at the
# end of capture, but if you reach stage 2 from any other path (manual
# recording, half-aborted session, legacy dataset without refinement)
# this idempotent call ensures the seed PLY is the refined version
# before static-gs reads it.
echo
echo "===> [1.5/3] RGBD-FUSION INIT -- ICP-refined TSDF seed (idempotent)"
dgs_stamp_seq t2_capture_end
$DGS_PIN "$PY" -u -m dynamic_gs.utils.rgbd_fusion_init "$DATA_DIR"

# ---------------------------------------------------------------- 2/3
echo
echo "===> [2/3] FIT -- ns-train static-gs"
echo "     (trains the Splatfacto scene + Phase 0b CPD fusion)"
echo "     SAM3 prompt: \"$SAM3_PROMPT\""
echo
dgs_stamp_seq t3_fit_start
$DGS_PIN "$NS_TRAIN" static-gs \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --pipeline.model.sam3_prompt_text "$SAM3_PROMPT"

# Verify the cache landed. static-gs writes static_state.pt (new name); fall
# back to the legacy post_fusion_state.pt for old datasets.
WARM_CACHE="$DATA_DIR/static_scene/static_state.pt"
if [[ ! -f "$WARM_CACHE" ]]; then
  LEGACY="$DATA_DIR/static_scene/post_fusion_state.pt"
  if [[ -f "$LEGACY" ]]; then
    WARM_CACHE="$LEGACY"
  else
    echo "BOOTSTRAP FAILED: static-gs finished but $WARM_CACHE (or legacy $LEGACY) is missing." >&2
    exit 1
  fi
fi
echo "     [ok] warm cache written: $WARM_CACHE ($(du -h "$WARM_CACHE" | cut -f1))"

# ---------------------------------------------------------------- 3/3
echo
echo "===> [3/3] GO-LIVE -- ns-train dynamic-gs-live"
echo "     (XFeat tracker + AnySplat FF, viser at http://localhost:8081)"
echo "     Ctrl+C or type 'stop' to end."
echo
dgs_stamp_seq t4_golive_start
# Re-run dynamic-gs-live in non-destructive mode (live_wipe_root=False is
# already the default, but we set it explicitly so this script documents
# the intent -- we just spent stage 1 + 2 building this dir, don't wipe it).
exec $DGS_PIN "$NS_TRAIN" dynamic-gs-live \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --pipeline.enable_viser_direct=True \
  --pipeline.enable-feedforward-inpaint=anysplat_decode \
  --pipeline.save-debug-images=False \
  --pipeline.live-wipe-root=False
