#!/usr/bin/env bash
# Closed-loop Mode B inpaint runner. All tyro syntax baked in.
#
# Usage:
#   bash scripts/run_mode_b.sh                       # uses pinned dataset + sensible defaults
#   bash scripts/run_mode_b.sh /path/to/dataset      # override dataset
#   ZOFFSET=1.2 bash scripts/run_mode_b.sh           # zoom-out distance (metres)
#   DS=2 bash scripts/run_mode_b.sh                  # CDN downsample factor (0=auto, 1=off, 2, 4, ...)
#   SCALE=1.0 bash scripts/run_mode_b.sh             # per-pixel Gaussian scale multiplier (default 5.0)
#   CLIFF=0.02 bash scripts/run_mode_b.sh            # depth-cliff threshold (m). 0 disables. default 0.05
#   LEAK=0.05 bash scripts/run_mode_b.sh             # leak threshold (m). drop pixels behind rendered. 0 disables. default 0.02
#   PROMPT="plate" bash scripts/run_mode_b.sh        # SAM3 text prompt (short noun phrase works best)
#   CULL=False bash scripts/run_mode_b.sh            # disable cull-in-front filter (default True)
#   CULL_TOL=0.01 bash scripts/run_mode_b.sh         # cull depth tolerance (m), default 0.005
#   THRESH=0.05 bash scripts/run_mode_b.sh           # MSSIM change threshold (lower = more sensitive). default 0.07
#   LG_CONF=0.02 bash scripts/run_mode_b.sh          # LighterGlue match confidence floor (lower = more matches, default 0.1)

set -e

ENV=/home/mrc-cuhk/miniconda3/envs/dynamic_gs
# Include /home/mrc-cuhk/miniconda3/bin so `conda run -n sam3_dynamic_gs ...`
# (used by SAM3 text-segmentation subprocess) can find the conda binary.
export PATH="$ENV/bin:/home/mrc-cuhk/miniconda3/bin:$PATH"
export CUDA_HOME=$ENV
export CPATH=$ENV/targets/x86_64-linux/include
export LIBRARY_PATH=$ENV/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$ENV/lib
export PYTHONNOUSERSITE=1
# SAM3D worker (third_party/sam-3d-objects/notebook/inference.py) reads
# $CONDA_PREFIX at import time to find CUDA_HOME. The worker runs in the
# `sam3_dynamic_gs` env, so we point CONDA_PREFIX at THAT env's path — not
# the dynamic_gs main env path. Without this, the subprocess dies with
# KeyError on Phase 0a SAM3D generation.
export CONDA_PREFIX=/home/mrc-cuhk/miniconda3/envs/sam3_dynamic_gs

# Pinned dataset (per project memory — only this dataset is in scope for tuning).
DATA="${1:-/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynamic_gs_test_2026-03-28_19-49-45_w_background}"
ZOFFSET="${ZOFFSET:-0.5}"
DS="${DS:-0}"
SCALE="${SCALE:-5.0}"
CLIFF="${CLIFF:-0.05}"
LEAK="${LEAK:-0.01}"
PROMPT="${PROMPT:-coke can}"  # bare noun only — no articles/prepositions (CLIP/SAM3 prompt rule)
CULL="${CULL:-True}"
CULL_TOL="${CULL_TOL:-0.005}"
THRESH="${THRESH:-0.07}"
OUT="${OUT:-/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/outputs/feedforward/mode_b_closedloop.mp4}"

echo "[run_mode_b] dataset = $DATA"
echo "[run_mode_b] z-offset = ${ZOFFSET} m"
echo "[run_mode_b] CDN downsample factor = ${DS}"
echo "[run_mode_b] scale multiplier = ${SCALE}"
echo "[run_mode_b] cliff threshold = ${CLIFF} m"
echo "[run_mode_b] leak threshold = ${LEAK} m"
echo "[run_mode_b] SAM3 prompt = ${PROMPT}"
echo "[run_mode_b] cull-in-front = ${CULL}, tol = ${CULL_TOL} m"
echo "[run_mode_b] MSSIM change threshold = ${THRESH}"
echo "[run_mode_b] output = $OUT"

ns-train dynamic-gs \
  --data "$DATA" \
  --pipeline.enable-feedforward-inpaint=rgbd_decode \
  --pipeline.feedforward-recurring-every-n-ticks=1 \
  --pipeline.feedforward-skip-delete=True \
  --pipeline.feedforward-anchor-z-offset-m="$ZOFFSET" \
  --pipeline.feedforward-cdn-downsample-factor="$DS" \
  --pipeline.feedforward-cdn-keep-largest-only=False \
  --pipeline.feedforward-rgbd-scale-multiplier="$SCALE" \
  --pipeline.feedforward-rgbd-cliff-threshold-m="$CLIFF" \
  --pipeline.feedforward-rgbd-leak-threshold-m="$LEAK" \
  --pipeline.model.sam3-prompt-text="$PROMPT" \
  --pipeline.feedforward-cull-in-front="$CULL" \
  --pipeline.feedforward-cull-in-front-depth-tol-m="$CULL_TOL" \
  --pipeline.model.change-mask-rgb-threshold="$THRESH" \
  --pipeline.feedforward-save-debug-pair=True \
  --pipeline.enable-dynamic-keyframe-filter=False \
  --pipeline.tracker-tick-every-steps=1 \
  --pipeline.feedforward-video-out="$OUT" \
  --pipeline.save-debug-images=False \
  --vis=tensorboard

# Re-encode the raw mp4v output to H.264 with a clean env (conda libffi clashes
# with system ffmpeg).
if [ -f "$OUT" ]; then
  TMP="${OUT}.raw.mp4"
  cp "$OUT" "$TMP"
  env -i HOME="$HOME" PATH=/usr/bin:/usr/local/bin \
    ffmpeg -y -loglevel error -i "$TMP" -c:v libx264 -pix_fmt yuv420p "$OUT"
  rm -f "$TMP"
  echo "[run_mode_b] H.264 re-encoded → $OUT"
fi
