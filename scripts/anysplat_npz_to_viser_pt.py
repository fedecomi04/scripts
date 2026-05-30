"""Convert an AnySplat worker .npz to a .pt blob that view_splats_viser.py reads.

Output keys: means (N,3), covariances (N,3,3), rgbs (N,3) in [0,1], opacities (N,).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR))
from dynamic_gs.utils.anysplat_decode import quat_wxyz_to_rotmat  # noqa: E402

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opacity-min", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    means = d["means_canonical"]
    log_scales = d["log_scales"]
    quats = d["quats_wxyz"]
    opacity_logits = d["opacity_logits"]
    features_dc = d["features_dc"]

    opac = 1.0 / (1.0 + np.exp(-opacity_logits))
    keep = opac >= args.opacity_min
    means, log_scales, quats, opac = means[keep], log_scales[keep], quats[keep], opac[keep]
    features_dc = features_dc[keep]
    print(f"opacity filter kept {keep.sum()}/{keep.size}")

    R = quat_wxyz_to_rotmat(quats)                  # (N, 3, 3)
    S = np.exp(log_scales)                           # (N, 3) linear scales (std-dev per axis)
    cov = R @ (S[:, :, None] * S[:, None, :] * np.eye(3)) @ R.transpose(0, 2, 1)
    rgbs = np.clip(features_dc * SH_C0 + 0.5, 0.0, 1.0)

    out_path = args.out or args.npz.with_name(args.npz.stem + ".pt")
    torch.save({
        "means": torch.from_numpy(means.astype(np.float32)),
        "covariances": torch.from_numpy(cov.astype(np.float32)),
        "rgbs": torch.from_numpy(rgbs.astype(np.float32)),
        "opacities": torch.from_numpy(opac.astype(np.float32)),
    }, out_path)
    print(f"wrote {out_path}  (N={means.shape[0]})")


if __name__ == "__main__":
    main()
