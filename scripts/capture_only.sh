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
export DGS_LIVE_ROOT="$DATA_DIR"

exec "$PY" -u "$(dirname "$0")/capture_only.py" "$DATA_DIR" "${@:2}"
