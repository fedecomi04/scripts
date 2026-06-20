"""Quick check: is the inserted SAM3D object full-size or nested-small inside the real-surface shell?
Loads static_state.pt, measures the inserted-core extent vs the real-surface-shell extent, and renders
the object silhouette. Run after a static rerun to see if the 0.82x under-scale was a one-off."""
import sys
from pathlib import Path
import numpy as np
import torch

pt = sys.argv[1] if len(sys.argv) > 1 else \
    "../data_teleoperation/datasets/screwdriver recorded full/static_scene/static_state.pt"
sd = torch.load(pt, map_location="cpu", weights_only=False)["model_state_dict"]
ids = sd["object_instance_ids"].squeeze(-1)
ins = sd["inserted_flags"].squeeze(-1)
means = sd["gauss_params.means"].float()

shell = (ids == 1) & (ins < 0.5)          # real-surface gaussians (trained/flagged)
core = (ids == 1) & (ins > 0.5)           # SAM3D-inserted object


def ext(m):
    return float((m.max(0).values - m.min(0).values).norm()) if len(m) else 0.0


sm = means[shell]; cm = means[core]
print(f"real-surface SHELL: N={int(shell.sum())} extent={ext(sm)*100:.1f}cm centroid={sm.mean(0).numpy().round(3) if len(sm) else None}")
print(f"inserted CORE:      N={int(core.sum())} extent={ext(cm)*100:.1f}cm centroid={cm.mean(0).numpy().round(3) if len(cm) else None}")
if len(sm) and len(cm):
    off = float((cm.mean(0) - sm.mean(0)).norm()) * 1000
    ratio = ext(cm) / max(ext(sm), 1e-6)
    print(f"core/shell extent ratio = {ratio:.2f}  centroid offset = {off:.1f}mm")
    if ratio < 0.85 and off < 20:
        print("VERDICT: NESTED-SMALL (core sits shrunken inside the shell) — the bug REPRODUCED")
    elif ratio >= 0.9:
        print("VERDICT: FULL-SIZE (core fills the shell) — the 0.82x was a ONE-OFF, this run is good")
    else:
        print("VERDICT: borderline — inspect the render")
