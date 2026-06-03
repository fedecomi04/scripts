#!/usr/bin/env bash
# setup_third_party.sh — clone the required external repos into third_party/.
# Run from the repo root. Idempotent: skips any dir that already exists.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

# REQUIRED FOR RUNTIME
clone_or_skip() {
  local name="$1" url="$2"
  if [ -d "$TP/$name" ]; then
    echo "[$name] already present — skipping"
  else
    echo "[$name] cloning from $url"
    git clone --depth 1 "$url" "$TP/$name"
  fi
}

# Dynamic tracker (XFeat + LighterGlue)
clone_or_skip xfeat https://github.com/verlab/accelerated_features.git
# Note: also rename to xfeat after clone if needed:
[ -d "$TP/accelerated_features" ] && [ ! -d "$TP/xfeat" ] && mv "$TP/accelerated_features" "$TP/xfeat"

# AnySplat (feedforward decoder)
clone_or_skip AnySplat https://github.com/InternRobotics/AnySplat.git

# SAM3 (text-prompted segmentation, used by static-gs-preseg)
clone_or_skip sam3 https://github.com/facebookresearch/sam3.git

# OPTIONAL: only needed for legacy `static-gs` Phase 0b SAM3D path.
# `static-gs-preseg` (the recommended default) does NOT use SAM3D.
# clone_or_skip sam-3d-objects https://github.com/facebookresearch/sam-3d-objects.git

# OPTIONAL: only if you set sam3d_registration_backend='teaser'
# clone_or_skip TEASER-plusplus https://github.com/MIT-SPARK/TEASER-plusplus.git

# Nerfstudio (vendored editable so we can patch the dataparser conventions
# the dynamic-gs pipeline relies on — see CLAUDE.md monkeypatches in
# dynamic_gs/__init__.py).
clone_or_skip nerfstudio https://github.com/nerfstudio-project/nerfstudio.git

echo ""
echo "[setup_third_party] done. Next:"
echo "  conda activate dynamic_gs"
echo "  pip install -e third_party/nerfstudio"
echo "  pip install -e ."
