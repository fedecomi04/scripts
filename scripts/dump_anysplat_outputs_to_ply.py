"""Dump the AnySplat feedforward worker output as 3DGS PLY files.

Reads a ``.npz`` written by ``scripts/anysplat_worker.py`` (e.g. one of
``<data_root>/dynamic_scene/debug/feedforward_anysplat/call_*.npz``) and
exports up to three companion PLYs next to it:

    raw_canonical.ply   — the worker's full output in AnySplat canonical frame
    aligned_world.ply   — the full output after Umeyama (canonical → scene world)
                          + optional scale multiplier (matches what the pipeline
                          inserts when ``feedforward_anysplat_scale_multiplier > 1``)

The PLYs are in the standard 3DGS format (same one ``view_sam3d_output.py``
and SuperSplat read). Open both in any 3DGS viewer to compare the raw model
output against the version inserted into the scene.

Usage:
    python scripts/dump_anysplat_outputs_to_ply.py \\
        <npz_path> \\
        --scene-transforms <data_root>/dynamic_scene/transforms.json \\
        [--scale-multiplier 5.0] \\
        [--opacity-min 0.05] \\
        [--out-dir <override>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
# scripts/scripts/dump_anysplat_outputs_to_ply.py → parents[1] is the scripts/ dir
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "third_party" / "AnySplat"))

from dynamic_gs.utils.anysplat_decode import (
    apply_similarity_to_gaussians,
    umeyama_similarity,
)
from src.model.ply_export import export_ply  # noqa: E402


def _build_anysplat_harmonics(features_dc: np.ndarray, features_rest: np.ndarray) -> torch.Tensor:
    """Reconstruct AnySplat-format (G, 3, d_sh) harmonics from the npz's
    (G, 3) DC + (G, 15, 3) rest tensors stored in Splatfacto convention."""

    dc = features_dc[:, None, :]      # (G, 1, 3)
    full = np.concatenate([dc, features_rest], axis=1)  # (G, 16, 3)
    return torch.from_numpy(np.transpose(full, (0, 2, 1)).astype(np.float32))  # (G, 3, 16)


def _scene_c2w_from_transforms(transforms_path: Path, image_names: list[str]) -> np.ndarray:
    with open(transforms_path) as f:
        tfs = json.load(f)
    fmap = {Path(str(fr["file_path"])).stem: np.array(fr["transform_matrix"]) for fr in tfs["frames"]}
    out = []
    for name in image_names:
        if name not in fmap:
            raise KeyError(f"Frame {name!r} not found in {transforms_path}")
        out.append(fmap[name])
    return np.stack(out, 0).astype(np.float32)


def _opacity_mask(opacity_logits: np.ndarray, opacity_min: float) -> np.ndarray:
    if opacity_min <= 0.0:
        return np.ones(opacity_logits.shape[0], dtype=bool)
    opac = 1.0 / (1.0 + np.exp(-opacity_logits))
    return opac >= opacity_min


def _save_ply(
    means: np.ndarray, log_scales: np.ndarray, quats_wxyz: np.ndarray,
    opacity_logits: np.ndarray, harmonics: torch.Tensor, out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # AnySplat's exporter wants LINEAR scales, sigmoid opacities, harmonics (G, 3, d_sh).
    linear_scales = np.exp(log_scales).astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logits.astype(np.float32)))
    export_ply(
        means=torch.from_numpy(means.astype(np.float32)),
        scales=torch.from_numpy(linear_scales),
        rotations=torch.from_numpy(quats_wxyz.astype(np.float32)),
        harmonics=harmonics,
        opacities=torch.from_numpy(opacities),
        path=out_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
    )
    print(f"  wrote {out_path}  (N={means.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_path", type=Path)
    ap.add_argument("--scene-transforms", type=Path, required=True,
                    help="Path to <data_root>/dynamic_scene/transforms.json for Umeyama target poses.")
    ap.add_argument("--scale-multiplier", type=float, default=5.0,
                    help="Multiplier applied to log_scales for aligned_world.ply (default 5.0).")
    ap.add_argument("--opacity-min", type=float, default=0.05,
                    help="Drop splats with sigmoid(opacity) < this. Default 0.05.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output dir. Default: same dir as npz_path.")
    args = ap.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    out_dir = args.out_dir or args.npz_path.parent
    stem = args.npz_path.stem

    means_can = data["means_canonical"]
    log_scales = data["log_scales"]
    quats = data["quats_wxyz"]
    opacity_logits = data["opacity_logits"]
    features_dc = data["features_dc"]
    features_rest = data["features_rest"]
    pred_c2w = data["pred_extrinsic_c2w"]
    input_paths = [str(p) for p in data["input_image_paths"]]

    print(f"Loaded {args.npz_path}")
    print(f"  N gaussians: {means_can.shape[0]}")
    print(f"  K context views: {pred_c2w.shape[0]}")
    print(f"  input frames: {[Path(p).stem for p in input_paths]}")

    keep = _opacity_mask(opacity_logits, args.opacity_min)
    if keep.sum() < means_can.shape[0]:
        print(f"  opacity filter (>= {args.opacity_min}): {keep.sum()} / {means_can.shape[0]} kept")

    harm = _build_anysplat_harmonics(features_dc[keep], features_rest[keep])

    # 1. Raw canonical PLY.
    print("[1/2] Exporting raw canonical PLY...")
    _save_ply(
        means_can[keep], log_scales[keep], quats[keep], opacity_logits[keep],
        harm, out_dir / f"{stem}_raw_canonical.ply",
    )

    # 2. Aligned world PLY — apply Umeyama + scale multiplier.
    print("[2/2] Exporting aligned world PLY...")
    image_names = [Path(p).stem for p in input_paths]
    scene_c2w = _scene_c2w_from_transforms(args.scene_transforms, image_names)
    src_centres = pred_c2w[:, :3, 3]
    dst_centres = scene_c2w[:, :3, 3]
    s_um, R_um, t_um = umeyama_similarity(src_centres, dst_centres)
    print(f"  Umeyama: s={s_um:.4f}, |t|={np.linalg.norm(t_um):.4f}")

    means_world, log_scales_world, quats_world = apply_similarity_to_gaussians(
        means_canonical=means_can[keep], log_scales=log_scales[keep], quats_wxyz=quats[keep],
        similarity_s=s_um, similarity_R=R_um, similarity_t=t_um,
    )
    if args.scale_multiplier > 0 and args.scale_multiplier != 1.0:
        log_scales_world = log_scales_world + np.log(args.scale_multiplier).astype(np.float32)
        print(f"  scale multiplier: ×{args.scale_multiplier}")

    _save_ply(
        means_world, log_scales_world, quats_world, opacity_logits[keep],
        harm, out_dir / f"{stem}_aligned_world.ply",
    )

    print(f"\nView with: python scripts/view_sam3d_output.py {out_dir/(stem + '_aligned_world.ply')}")


if __name__ == "__main__":
    main()
