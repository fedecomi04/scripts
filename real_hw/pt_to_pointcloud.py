#!/usr/bin/env python3
"""Export a dynamic_gs2 state .pt (static_state.pt / post_dynamic_state.pt) to a colored PLY point
cloud (gaussian means as xyz, features_dc -> RGB) for quick visual inspection in MeshLab/CloudCompare.

Usage:
  python real_hw/pt_to_pointcloud.py <state.pt> [out.ply] [--min-opacity 0.05]
Default out: <state.pt>.ply next to the input.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

_SH_C0 = 0.28209479177387814   # SH band-0 <-> RGB (matches gaussian_set / static_persist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--min-opacity", type=float, default=0.05,
                    help="drop gaussians below this (sigmoid) opacity (default 0.05)")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    out = Path(args.out) if args.out else pt_path.with_suffix(pt_path.suffix + ".ply")

    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    m = d["model_state_dict"]
    means = m["gauss_params.means"].float()
    fdc = m["gauss_params.features_dc"].float().reshape(-1, 3)
    op = torch.sigmoid(m["gauss_params.opacities"].float().reshape(-1))

    rgb = (fdc * _SH_C0 + 0.5).clamp(0, 1)

    keep = op >= args.min_opacity
    means, rgb, op = means[keep], rgb[keep], op[keep]
    n = means.shape[0]
    print(f"[pt2ply] {pt_path.name}: {len(keep)} gaussians -> {n} kept "
          f"(opacity>={args.min_opacity})")

    xyz = means.numpy().astype(np.float32)
    col = (rgb.numpy() * 255).astype(np.uint8)

    with open(out, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        buf = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        buf["r"], buf["g"], buf["b"] = col[:, 0], col[:, 1], col[:, 2]
        f.write(buf.tobytes())

    bb_min = xyz.min(0); bb_max = xyz.max(0)
    print(f"[pt2ply] wrote {out}")
    print(f"[pt2ply] bbox min={bb_min.round(3).tolist()} max={bb_max.round(3).tolist()} "
          f"(extent {(bb_max-bb_min).round(3).tolist()} m)")


if __name__ == "__main__":
    main()
