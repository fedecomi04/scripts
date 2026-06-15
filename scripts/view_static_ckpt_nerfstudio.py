#!/usr/bin/env python
"""View a static-gs ``static_state.pt`` (post-fusion warm-cache) as REAL
Gaussian splats in Nerfstudio's viewer (server-side gsplat rasterization),
not a point cloud.

Loads the ckpt's ``gauss_params`` into a vanilla Splatfacto model and opens a
one-frame Nerfstudio dataset built from a ``static_scene`` camera (the last /
operator-final view) so the viewer starts looking at the scene. Reuses
``build_pipeline`` + ``write_one_frame_dataset`` from
``view_anysplat_nerfstudio.py``.

Safe to use the NS viewer here (unlike the live pipeline, invariant #9): this is
a standalone static scene with no tracker / feedforward threads mutating
gauss_params, so there's no render-vs-mutation race.

Usage (dynamic_gs env — has nerfstudio + gsplat):
    python scripts/view_static_ckpt_nerfstudio.py \\
        <dataset>/static_scene/static_state.pt [--port 7007] [--opacity-min 0.0]
Then open http://localhost:<port>.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# splatfacto's get_viewmat is torch.compiled; Nerfstudio's render-interrupt raises
# IOChangeException mid-render, which dynamo otherwise wraps as a FATAL
# InternalTorchDynamoError. suppress_errors → dynamo falls back to eager on any
# trace error, so IOChangeException propagates normally and the render state machine
# catches it (clean camera-move restarts instead of a crash).
import torch._dynamo  # noqa: E402
torch._dynamo.config.suppress_errors = True

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from view_anysplat_nerfstudio import build_pipeline, write_one_frame_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path, help="path to static_state.pt / post_fusion_state.pt")
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--opacity-min", type=float, default=0.0,
                    help="drop splats with sigmoid(opacity) < this (0 = keep all)")
    ap.add_argument("--background", default="white", help="viewer bg: white / black / random")
    args = ap.parse_args()
    device = "cuda"

    # --- load ckpt gauss_params ---
    blob = torch.load(args.state, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]

    def g(name):
        for k in (f"gauss_params.{name}", name):
            if k in sd:
                return sd[k]
        raise KeyError(f"{name} not in checkpoint (keys e.g. {list(sd)[:6]})")

    means = g("means").float()
    scales = g("scales").float()           # log-scale
    quats = g("quats").float()
    opac = g("opacities").float().reshape(-1, 1)
    fdc = g("features_dc").float()
    frest = g("features_rest").float()
    if fdc.ndim == 3 and fdc.shape[1] == 1:    # (N,1,3) -> (N,3)
        fdc = fdc.reshape(fdc.shape[0], 3)

    if args.opacity_min > 0.0:
        keep = torch.sigmoid(opac.squeeze(1)) >= args.opacity_min
        means, scales, quats, opac, fdc, frest = (
            t[keep] for t in (means, scales, quats, opac, fdc, frest)
        )
        print(f"[view-ckpt] opacity filter >= {args.opacity_min}: kept {int(keep.sum())}/{keep.numel()}")
    n = means.shape[0]
    inst = sd.get("object_instance_ids")
    n_obj = int((inst.float() > 0).sum()) if inst is not None else 0
    print(f"[view-ckpt] loaded {n:,} gaussians ({n_obj:,} inserted object) from {args.state.name}")

    # --- camera from static_scene transforms (last frame = operator's final view) ---
    sd_dir = args.state.parent
    meta = json.loads((sd_dir / "transforms.json").read_text())
    frame = sorted(meta["frames"], key=lambda f: int(re.findall(r"\d+", f["file_path"])[-1]))[-1]
    c2w = np.asarray(frame["transform_matrix"], dtype=float)   # OpenGL c2w (NS convention)
    img = sd_dir / frame["file_path"].lstrip("./")
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    w, h = int(meta["w"]), int(meta["h"])

    out_dir = Path("/tmp/view_static_ckpt")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_one_frame_dataset(data_dir, img, c2w, fx, fy, cx, cy, w, h)

    config, pipeline = build_pipeline(data_dir, out_dir, args.background, args.port, device)

    m = pipeline.model
    m.gauss_params = nn.ParameterDict({
        "means": nn.Parameter(means.to(device), requires_grad=False),
        "scales": nn.Parameter(scales.to(device), requires_grad=False),
        "quats": nn.Parameter(quats.to(device), requires_grad=False),
        "features_dc": nn.Parameter(fdc.to(device), requires_grad=False),
        "features_rest": nn.Parameter(frest.to(device), requires_grad=False),
        "opacities": nn.Parameter(opac.to(device), requires_grad=False),
    }).to(device)
    m.step = 30000          # activate all SH bands at render time
    m.crop_box = None

    from nerfstudio.scripts.viewer.run_viewer import _start_viewer
    print(f"\n[view-ckpt] Nerfstudio viewer → http://localhost:{args.port}  (Ctrl-C to quit)\n", flush=True)
    _start_viewer(config, pipeline, step=30000)


if __name__ == "__main__":
    main()
