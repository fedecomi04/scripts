#!/usr/bin/env bash
# dynamic_gs2 — recorded A/B: replay a dataset through the NEW pipeline + compare to the OLD run.
#
# This is the UNATTENDED-VALIDATED path (no live sim needed). Replays the recorded dynamic
# frames through the new pipeline (warm-load -> tracker -> write-pose), writes new_trace.jsonl,
# and diffs it against the old pipeline's per-frame motion logs (dynamic_scene/debug/*_motion.txt).
#
# Usage:  dynamic_gs2/replay_ab.sh <data_dir> [transforms_name] [--ff]
set -euo pipefail

DATA="${1:?usage: replay_ab.sh <data_dir> [transforms_name] [--ff]}"
TJ="${2:-transforms.json}"
FF_FLAG=""
for a in "$@"; do [ "$a" = "--ff" ] && FF_FLAG="--ff"; done

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_LIB="$HOME/miniconda3/envs/dynamic_gs/lib"
PY="$HOME/miniconda3/envs/dynamic_gs/bin/python"
OUT="$DATA/new_trace.jsonl"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
echo "[replay_ab] new-pipeline replay ($TJ) $FF_FLAG -> $OUT"
"$PY" -m dynamic_gs2.pipeline --mode recorded --data "$DATA" --transforms "$TJ" --out-trace "$OUT" $FF_FLAG
echo "[replay_ab] comparing vs old motion logs"
"$PY" -m dynamic_gs2.verify.compare_traces "$OUT" "$DATA"
