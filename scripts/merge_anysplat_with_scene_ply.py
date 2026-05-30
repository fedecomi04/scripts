"""Merge the static-scene gaussians with AnySplat's aligned-world gaussians
into a single 3DGS PLY, tinting the AnySplat splats magenta to make them
visually distinct from the scene.

Inputs:
    - post_fusion_state.pt (the Splatfacto checkpoint after Phase 0b fusion)
    - the AnySplat worker .npz (we apply Umeyama + scale-multiplier on the fly,
      same math as the pipeline runs)
    - the dynamic_scene/transforms.json (needed for Umeyama target poses)

Output: one .ply containing all scene gaussians (true colors) + all AnySplat
gaussians (magenta SH-DC). Open in SuperSplat / view_sam3d_output.py.

Usage (one-liner, from any cwd):

    python /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/merge_anysplat_with_scene_ply.py \\
        --post-fusion /path/to/static_scene/post_fusion_state.pt \\
        --npz /path/to/dynamic_scene/debug/feedforward_anysplat/call_0000_step_*_frame_*.npz \\
        --scene-transforms /path/to/dynamic_scene/transforms.json \\
        --out /tmp/anysplat_merged.ply \\
        [--scale-multiplier 5.0] [--opacity-min 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "third_party" / "AnySplat"))

from dynamic_gs.utils.anysplat_decode import (  # noqa: E402
    apply_similarity_to_gaussians,
    umeyama_similarity,
)
from src.model.ply_export import export_ply  # noqa: E402

# Magenta in RGB → SH degree-0 (DC band): RGB ≈ SH_dc × 0.28209479 + 0.5
SH_C0 = 0.28209479177387814
MAGENTA_RGB = (1.0, 0.0, 1.0)
MAGENTA_DC = tuple((c - 0.5) / SH_C0 for c in MAGENTA_RGB)  # ≈ (+1.772, -1.772, +1.772)


def _load_post_fusion_params(state_path: Path) -> dict[str, np.ndarray]:
    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    # post_fusion_state.pt stores a Splatfacto model_state_dict with
    # flat "gauss_params.<key>" entries. Older variants may store a
    # nested dict under "gauss_params".
    sd = blob.get("model_state_dict", blob)
    out = {}
    for key in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
        flat_key = f"gauss_params.{key}"
        if flat_key in sd:
            out[key] = sd[flat_key].detach().cpu().numpy().astype(np.float32)
        elif "gauss_params" in sd and key in sd["gauss_params"]:
            out[key] = sd["gauss_params"][key].detach().cpu().numpy().astype(np.float32)
        else:
            raise KeyError(f"{state_path}: missing gauss_params/{key}")
    return out


def _scene_c2w_from_transforms(transforms_path: Path, image_names: list[str]) -> np.ndarray:
    with open(transforms_path) as f:
        tfs = json.load(f)
    fmap = {Path(str(fr["file_path"])).stem: np.array(fr["transform_matrix"]) for fr in tfs["frames"]}
    return np.stack([fmap[n] for n in image_names], 0).astype(np.float32)


def _opacity_mask(opacity_logits: np.ndarray, opacity_min: float) -> np.ndarray:
    if opacity_min <= 0.0:
        return np.ones(opacity_logits.shape[0], dtype=bool)
    opac = 1.0 / (1.0 + np.exp(-opacity_logits))
    return opac >= opacity_min


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--post-fusion", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--scene-transforms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale-multiplier", type=float, default=5.0)
    ap.add_argument("--opacity-min", type=float, default=0.05)
    ap.add_argument("--drop-background-rgb", type=float, nargs=3, default=[0.86, 0.92, 1.0],
                    metavar=("R", "G", "B"),
                    help="RGB color of the background to drop (default = Gazebo sky 0.86 0.92 1.0).")
    ap.add_argument("--background-tolerance", type=float, default=0.08,
                    help="Per-channel tolerance for the background match (default 0.08).")
    args = ap.parse_args()

    # --- Static scene side ---
    scene = _load_post_fusion_params(args.post_fusion)
    Ns = scene["means"].shape[0]
    print(f"[merge] scene gaussians:    N={Ns}")

    # --- AnySplat side: load npz, apply Umeyama + scale multiplier + opacity filter ---
    d = np.load(args.npz, allow_pickle=True)
    N_raw = d["means_canonical"].shape[0]

    keep = _opacity_mask(d["opacity_logits"], args.opacity_min)
    n_op = int(keep.sum())

    # Background-color filter: AnySplat can't predict depth for uniform-color
    # regions (sky/background), so those splats land at arbitrary depths and
    # pollute the cloud. Drop any splat whose predicted DC-band RGB matches the
    # background color within ``--background-tolerance`` per channel.
    if args.drop_background_rgb is not None:
        rgb_pred = d["features_dc"] * SH_C0 + 0.5      # (N, 3)
        bg_rgb = np.asarray(args.drop_background_rgb, dtype=np.float32)
        bg_match = np.all(np.abs(rgb_pred - bg_rgb) <= args.background_tolerance, axis=-1)
        keep = keep & (~bg_match)
        print(f"[merge] background-color filter dropped {int(bg_match.sum())} splats "
              f"(rgb≈{tuple(round(x,2) for x in bg_rgb)}, tol={args.background_tolerance})")

    means_can = d["means_canonical"][keep]
    log_scales = d["log_scales"][keep]
    quats = d["quats_wxyz"][keep]
    opacity_logits = d["opacity_logits"][keep]
    features_dc = d["features_dc"][keep]            # (N, 3)
    features_rest = d["features_rest"][keep]        # (N, 15, 3)
    pred_c2w = d["pred_extrinsic_c2w"]
    input_names = [Path(str(p)).stem for p in d["input_image_paths"]]
    print(f"[merge] anysplat after filters: kept {int(keep.sum())}/{N_raw} "
          f"(opacity→{n_op}, then background→{int(keep.sum())})")

    scene_c2w = _scene_c2w_from_transforms(args.scene_transforms, input_names)
    s_um, R_um, t_um = umeyama_similarity(pred_c2w[:, :3, 3], scene_c2w[:, :3, 3])
    print(f"[merge] Umeyama: s={s_um:.4f}, |t|={np.linalg.norm(t_um):.4f}")

    means_world, log_scales_world, quats_world = apply_similarity_to_gaussians(
        means_canonical=means_can, log_scales=log_scales, quats_wxyz=quats,
        similarity_s=s_um, similarity_R=R_um, similarity_t=t_um,
    )
    if args.scale_multiplier > 0 and args.scale_multiplier != 1.0:
        log_scales_world = log_scales_world + np.log(args.scale_multiplier).astype(np.float32)
        print(f"[merge] scale multiplier: ×{args.scale_multiplier}")

    Na = means_world.shape[0]
    print(f"[merge] anysplat gaussians (final): N={Na}")

    # Override AnySplat features_dc with magenta; zero out features_rest.
    fdc_magenta = np.tile(np.asarray(MAGENTA_DC, dtype=np.float32), (Na, 1))   # (Na, 3)
    frest_zero = np.zeros((Na, 15, 3), dtype=np.float32)

    # --- Concat both sets ---
    means_all = np.concatenate([scene["means"], means_world], axis=0)
    scales_all = np.concatenate([np.exp(scene["scales"]), np.exp(log_scales_world)], axis=0)
    quats_all = np.concatenate([scene["quats"], quats_world], axis=0)
    op_all = np.concatenate([
        1.0 / (1.0 + np.exp(-scene["opacities"].reshape(-1))),
        1.0 / (1.0 + np.exp(-opacity_logits.reshape(-1))),
    ], axis=0)
    fdc_all = np.concatenate([scene["features_dc"], fdc_magenta], axis=0)            # (N, 3)
    frest_all = np.concatenate([scene["features_rest"], frest_zero], axis=0)         # (N, 15, 3)

    # Build harmonics (G, 3, d_sh) for AnySplat's exporter.
    full_sh = np.concatenate([fdc_all[:, None, :], frest_all], axis=1)               # (N, 16, 3)
    harmonics = torch.from_numpy(np.transpose(full_sh, (0, 2, 1)).astype(np.float32)) # (N, 3, 16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_ply(
        means=torch.from_numpy(means_all.astype(np.float32)),
        scales=torch.from_numpy(scales_all.astype(np.float32)),
        rotations=torch.from_numpy(quats_all.astype(np.float32)),
        harmonics=harmonics,
        opacities=torch.from_numpy(op_all.astype(np.float32)),
        path=args.out,
        shift_and_scale=False,
        save_sh_dc_only=True,
    )
    print(f"[merge] wrote {args.out}  (total N={means_all.shape[0]} = {Ns} scene + {Na} magenta-AnySplat)")


if __name__ == "__main__":
    main()
