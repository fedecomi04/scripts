"""Dump ONLY the raw AnySplat output (canonical frame) as a 3DGS PLY.
No Umeyama, no scene merge, no alignment, no scale multiplier.

Usage (one-liner from any cwd):

    python /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/dump_anysplat_raw_ply.py \\
        <npz_path> [--out raw.ply] [--opacity-min 0.05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR / "third_party" / "AnySplat"))

from src.model.ply_export import export_ply  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opacity-min", type=float, default=0.0,
                    help="Drop splats with sigmoid(opacity) < this (0 = keep all)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    means = d["means_canonical"]
    log_scales = d["log_scales"]
    quats = d["quats_wxyz"]
    opacity_logits = d["opacity_logits"]
    features_dc = d["features_dc"]                  # (N, 3)
    features_rest = d["features_rest"]              # (N, 15, 3)

    if args.opacity_min > 0:
        opac = 1.0 / (1.0 + np.exp(-opacity_logits))
        keep = opac >= args.opacity_min
        means, log_scales, quats = means[keep], log_scales[keep], quats[keep]
        opacity_logits = opacity_logits[keep]
        features_dc, features_rest = features_dc[keep], features_rest[keep]
        print(f"opacity filter (>= {args.opacity_min}): kept {int(keep.sum())} / {keep.size}")

    out_path = args.out or args.npz.with_name(args.npz.stem + "_raw.ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_sh = np.concatenate([features_dc[:, None, :], features_rest], axis=1)  # (N, 16, 3)
    harm = torch.from_numpy(np.transpose(full_sh, (0, 2, 1)).astype(np.float32))   # (N, 3, 16)
    export_ply(
        means=torch.from_numpy(means.astype(np.float32)),
        scales=torch.from_numpy(np.exp(log_scales).astype(np.float32)),
        rotations=torch.from_numpy(quats.astype(np.float32)),
        harmonics=harm,
        opacities=torch.from_numpy((1.0 / (1.0 + np.exp(-opacity_logits))).astype(np.float32)),
        path=out_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
    )
    print(f"wrote {out_path}  (N={means.shape[0]})")


if __name__ == "__main__":
    main()
