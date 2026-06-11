#!/usr/bin/env python
"""Render training views from a static_state.pt via gsplat and report PSNR vs the
robot-masked ground-truth RGB. Used to A/B static-training step counts: cut steps
only if PSNR holds. Usage: static_render_psnr.py <data_dir> <static_state.pt> [n_views]
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import cv2
import gsplat

BG = torch.tensor([0.86, 0.92, 1.0])


def main():
    data = Path(sys.argv[1])
    state = Path(sys.argv[2])
    n_views = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    static = data / "static_scene"
    meta = json.loads((static / "transforms.json").read_text())

    sd = torch.load(state, map_location="cuda")
    msd = sd.get("model_state_dict", sd)
    g = lambda n: msd[f"gauss_params.{n}"].cuda().float()
    means = g("means")
    quats = g("quats")
    scales = torch.exp(g("scales"))
    opac = torch.sigmoid(g("opacities")).squeeze(-1)
    fdc = g("features_dc")
    frest = g("features_rest")
    colors = torch.cat([fdc[:, None, :], frest], dim=1)
    sh_degree = int(round(colors.shape[1] ** 0.5)) - 1

    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    W, H = int(meta["w"]), int(meta["h"])
    Kmat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device="cuda").float()
    bg = BG.cuda()
    gl2cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device="cuda"))

    frames = meta["frames"]
    step = max(1, len(frames) // n_views)
    sample = frames[::step][:n_views]
    psnrs = []
    for fr in sample:
        c2w = torch.tensor(fr["transform_matrix"], device="cuda").float() @ gl2cv
        viewmat = torch.linalg.inv(c2w)
        out, _, _ = gsplat.rasterization(
            means, quats, scales, opac, colors,
            viewmat[None], Kmat[None], W, H,
            sh_degree=sh_degree, render_mode="RGB", backgrounds=bg[None],
        )
        pred = out[0].clamp(0, 1)
        gt = cv2.imread(str(static / fr["file_path"].lstrip("./")))[:, :, ::-1].astype(np.float32) / 255.0
        gt = torch.tensor(np.ascontiguousarray(gt), device="cuda")
        mrel = fr.get("mask_path")
        if mrel:
            mp = static / mrel.lstrip("./")
            if mp.exists():
                m = (cv2.imread(str(mp), 0) > 0).astype(np.float32)
                m = torch.tensor(m, device="cuda")[..., None]
                pred = pred * m + bg * (1 - m)
                gt = gt * m + bg * (1 - m)
        mse = ((pred - gt) ** 2).mean()
        psnrs.append(float(-10 * torch.log10(mse)))
    print(f"PSNR mean={np.mean(psnrs):.2f}dB min={np.min(psnrs):.2f} max={np.max(psnrs):.2f} "
          f"(n={len(psnrs)} views, {means.shape[0]} gaussians)")


if __name__ == "__main__":
    main()
