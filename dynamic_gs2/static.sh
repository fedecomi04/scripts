#!/usr/bin/env bash
# dynamic_gs2 — static phase: train + Phase-0a/0b fuse (proven old static-gs) -> convert to the
# dynamic_gs2 warm-cache (static_scene/static_state.pt) + prewarm the dynamic models.
#
#   recorded (default, UNATTENDED): a static_scene/ dataset is already on disk.
#   live (needs the sim): sweep + red-box trigger UI -> segment -> SAM3D -> seed -> train -> convert.
#
# After this, go live with:  dynamic_gs2/resume_live.sh <data_dir>   (warm-loads the .pt)
#
# Usage:  dynamic_gs2/static.sh <data_dir> [prompt] [--live]
set -euo pipefail

DATA="${1:?usage: static.sh <data_dir> [prompt] [--live]}"
PROMPT="${2:-}"
MODE="recorded"
for a in "$@"; do [ "$a" = "--live" ] && MODE="live"; done
[ "$PROMPT" = "--live" ] && PROMPT=""

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="$HOME/miniconda3/envs/dynamic_gs"
PY="$ENV/bin/python"

cd "$SCRIPTS_DIR"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # shared-GPU defrag
export CUDA_HOME="$ENV"                                                  # gsplat JIT needs these for a bare-python
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"        # launch (conda-activate sets them otherwise)

# dVRK RT protection (live mode only): the gsplat JIT compile storm + training would otherwise
# co-locate on the dVRK's reserved cores 0-3 and starve its 1kHz loop. Reserve 0-3, cap math
# threadpools, pin the pipeline to 4-23, lock the dVRK ONTO 0-3. Helpers self-skip when no
# dVRK/cpuset is present. DGS_NO_CPU_PIN=1 disables. (recorded mode = no sim -> not applied.)
DGS_PIN=""
if [ "$MODE" = "live" ]; then
  source "$SCRIPTS_DIR/scripts/_ros_cleanup.sh"
  dgs_export_thread_caps
  DGS_PIN="$(dgs_cpu_pin_prefix)"
  dgs_isolate_dvrk
fi
echo "[static] mode=$MODE data=$DATA prompt='${PROMPT}'  (pin: ${DGS_PIN:-none})"
exec $DGS_PIN "$PY" -m dynamic_gs2.static_pipeline --mode "$MODE" --data "$DATA" --prompt "$PROMPT"
