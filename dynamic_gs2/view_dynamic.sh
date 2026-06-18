#!/usr/bin/env bash
# dynamic_gs2 — VISUAL VALIDATION with the viser viewer, NO live sim needed.
# Replays a recorded dataset through the new dynamic pipeline (warm-load -> tracker ->
# write-pose) at ~fps with the viser-direct viewer up. Open the printed URL and orbit
# to watch the tracker drive the gaussian object. Loops until Ctrl-C.
#
# Usage:  dynamic_gs2/view_dynamic.sh <data_dir> [transforms_name] [--ff] [--once]
set -euo pipefail

DATA="${1:?usage: view_dynamic.sh <data_dir> [transforms_name] [--ff] [--once]}"
TJ="${2:-transforms.json}"
EXTRA=""
for a in "$@"; do { [ "$a" = "--ff" ] || [ "$a" = "--once" ]; } && EXTRA="$EXTRA $a"; done

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_LIB="$HOME/miniconda3/envs/dynamic_gs/lib"
PY="$HOME/miniconda3/envs/dynamic_gs/bin/python"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
exec "$PY" -m dynamic_gs2.pipeline --mode view --data "$DATA" --transforms "$TJ" $EXTRA
