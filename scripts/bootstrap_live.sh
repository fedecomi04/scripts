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

# ---------------------------------------------------------------- 1/3
echo
echo "===> [1/3] CAPTURE -- live_session.run_live_capture_session()"
echo "     (sweeps the scene + SAM3 + SAM3D; follow the on-screen prompts)"
echo
"$PY" -u -c "
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
"$PY" -u -m dynamic_gs.utils.rgbd_fusion_init "$DATA_DIR"

# ---------------------------------------------------------------- 2/3
echo
echo "===> [2/3] FIT -- ns-train static-gs"
echo "     (trains the Splatfacto scene + Phase 0b CPD fusion)"
echo "     SAM3 prompt: \"$SAM3_PROMPT\""
echo
"$NS_TRAIN" static-gs \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --pipeline.model.sam3_prompt_text "$SAM3_PROMPT"

# Verify the cache landed.
WARM_CACHE="$DATA_DIR/static_scene/post_fusion_state.pt"
if [[ ! -f "$WARM_CACHE" ]]; then
  echo "BOOTSTRAP FAILED: static-gs finished but $WARM_CACHE is missing." >&2
  exit 1
fi
echo "     [ok] warm cache written: $WARM_CACHE ($(du -h "$WARM_CACHE" | cut -f1))"

# ---------------------------------------------------------------- 3/3
echo
echo "===> [3/3] GO-LIVE -- ns-train dynamic-gs-live"
echo "     (XFeat tracker + AnySplat FF, viser at http://localhost:8081)"
echo "     Ctrl+C or type 'stop' to end."
echo
# Re-run dynamic-gs-live in non-destructive mode (live_wipe_root=False is
# already the default, but we set it explicitly so this script documents
# the intent -- we just spent stage 1 + 2 building this dir, don't wipe it).
exec "$NS_TRAIN" dynamic-gs-live \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vis tensorboard \
  --pipeline.enable_viser_direct=True \
  --pipeline.enable-feedforward-inpaint=anysplat_decode \
  --pipeline.save-debug-images=False \
  --pipeline.live-wipe-root=False
