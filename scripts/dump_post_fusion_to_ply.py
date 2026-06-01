"""Convert a `post_fusion_state.pt` to a standard 3D Gaussian Splatting
.ply file (the same format that the SAM3D output PLYs use). Lets you
view the post-fusion scene with any 3DGS viewer (Open3D, our existing
`scripts/view_sam3d_output.py`, SuperSplat, etc).

Usage:
    python scripts/dump_post_fusion_to_ply.py \\
        /path/to/dataset/static_scene/post_fusion_state.pt \\
        [--out /path/to/output.ply] \\
        [--opacity-min 0.05]   # drop near-transparent splats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opacity-min", type=float, default=0.0,
                    help="drop splats with sigmoid(opacity) < this (default 0 = keep all)")
    args = ap.parse_args()

    blob = torch.load(args.state, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"]

    def _get(name):
        for k in (name, f"gauss_params.{name}", f"_buffers.{name}"):
            if k in state:
                return state[k]
        raise KeyError(name)

    means = _get("means").float().numpy().astype(np.float32)
    scales = _get("scales").float().numpy().astype(np.float32)  # log-scale
    quats = _get("quats").float()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    quats = quats.numpy().astype(np.float32)  # wxyz
    features_dc = _get("features_dc").float().numpy().astype(np.float32)
    if features_dc.ndim == 3:
        features_dc = features_dc[:, 0, :]  # (N, 3)
    opacities_logit = _get("opacities").float().numpy().reshape(-1).astype(np.float32)

    N = means.shape[0]

    # Optional opacity filter (sigmoid).
    if args.opacity_min > 0:
        opacities_sig = 1.0 / (1.0 + np.exp(-opacities_logit))
        keep = opacities_sig >= float(args.opacity_min)
        n_kept = int(keep.sum())
        print(f"opacity filter: kept {n_kept}/{N} splats (>= {args.opacity_min})")
        means, scales, quats = means[keep], scales[keep], quats[keep]
        features_dc, opacities_logit = features_dc[keep], opacities_logit[keep]
        N = n_kept

    # Build the standard 3DGS PLY field layout (matches gsplat / SuperSplat).
    # Per-row: x, y, z, nx, ny, nz (unused), f_dc_0/1/2, opacity, scale_0/1/2,
    # rot_0/1/2/3 (wxyz).
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    rows = np.empty(N, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = means[:, 0], means[:, 1], means[:, 2]
    rows["nx"] = rows["ny"] = rows["nz"] = 0.0
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = features_dc[:, 0], features_dc[:, 1], features_dc[:, 2]
    rows["opacity"] = opacities_logit  # logit (standard 3DGS PLY stores raw activations)
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    out_path = args.out or args.state.with_name("post_fusion.ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(out_path))
    print(f"wrote {N} splats → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
