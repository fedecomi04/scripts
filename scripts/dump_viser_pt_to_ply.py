"""Convert a `feedforward_scene_splats.pt` (or any viser-blob .pt produced
by ``DynamicGSPipeline._dump_scene_splats``) into a standard 3DGS .ply.

Drop into SuperSplat (https://superspl.at/editor), Open3D viewers, or any
3DGS viewer — no nerfstudio runtime required.

Usage:
    python /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/dump_viser_pt_to_ply.py \\
        <pt_path> [--out <ply_path>] [--opacity-min 0.05] [--only-inserted]

The blob is the dict with keys
``{means, covariances, rgbs, opacities, inserted_flags, object_instance_ids}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def _construct_attributes() -> list[str]:
    attrs = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    for i in range(45):  # 15 SH coeffs × 3 channels
        attrs.append(f"f_rest_{i}")
    attrs += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    return attrs


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) wxyz unit quaternion, Shepperd's method."""
    m00 = R[:, 0, 0]; m11 = R[:, 1, 1]; m22 = R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((R.shape[0], 4), dtype=np.float32)
    s = np.zeros(R.shape[0], dtype=np.float64)
    # case 1: tr > 0
    case1 = tr > 0
    s1 = np.sqrt(np.where(case1, tr + 1.0, 1.0)) * 2.0
    q[case1, 0] = 0.25 * s1[case1]
    q[case1, 1] = (R[case1, 2, 1] - R[case1, 1, 2]) / s1[case1]
    q[case1, 2] = (R[case1, 0, 2] - R[case1, 2, 0]) / s1[case1]
    q[case1, 3] = (R[case1, 1, 0] - R[case1, 0, 1]) / s1[case1]
    rest = ~case1
    if rest.any():
        diag = np.stack([m00, m11, m22], axis=-1)
        big = np.argmax(diag, axis=-1)
        for j in range(3):
            sel = rest & (big == j)
            if not sel.any():
                continue
            i = j; k = (j + 1) % 3; l = (j + 2) % 3
            s_j = np.sqrt(R[sel, i, i] - R[sel, k, k] - R[sel, l, l] + 1.0) * 2.0
            qw = (R[sel, l, k] - R[sel, k, l]) / s_j
            q[sel, 0] = qw
            comp_i = 0.25 * s_j
            comp_k = (R[sel, i, k] + R[sel, k, i]) / s_j
            comp_l = (R[sel, i, l] + R[sel, l, i]) / s_j
            q[sel, 1 + i] = comp_i
            q[sel, 1 + k] = comp_k
            q[sel, 1 + l] = comp_l
    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    return q.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pt", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opacity-min", type=float, default=0.05,
                    help="Drop splats with sigmoid(opacity) < this (default 0.05).")
    ap.add_argument("--only-inserted", action="store_true",
                    help="Keep only splats with inserted_flag==1.")
    args = ap.parse_args()

    blob = torch.load(args.pt, map_location="cpu", weights_only=False)
    means = np.asarray(blob["means"], dtype=np.float32)
    covs = np.asarray(blob["covariances"], dtype=np.float32)
    rgbs = np.asarray(blob["rgbs"], dtype=np.float32)
    opac = np.asarray(blob["opacities"], dtype=np.float32).reshape(-1)
    ins = blob.get("inserted_flags")
    ids = blob.get("object_instance_ids")
    N0 = means.shape[0]
    print(f"loaded {N0} splats from {args.pt}")

    # Opacity floor.
    keep = opac >= args.opacity_min
    if keep.sum() < N0:
        print(f"  opacity ≥ {args.opacity_min}: {int(keep.sum())} / {N0}")

    # Optional inserted-only filter — prefer instance_id==999 (FF-only) when present.
    if args.only_inserted:
        if ids is not None:
            ids_arr = np.asarray(ids).reshape(-1)
            ff_mask = ids_arr == 999
            keep = keep & ff_mask
            print(f"  --only-inserted (instance_id==999): {int(keep.sum())} kept")
        elif ins is not None:
            ins_arr = np.asarray(ins).reshape(-1)
            keep = keep & (ins_arr > 0)
            print(f"  --only-inserted (inserted_flag==1): {int(keep.sum())} kept")
        else:
            print("  WARN --only-inserted but blob has no inserted_flags/object_instance_ids; ignoring")

    means = means[keep]; covs = covs[keep]; rgbs = rgbs[keep]; opac = opac[keep]
    N = means.shape[0]
    if N == 0:
        raise RuntimeError("Filters removed all splats; nothing to export.")

    # Eigendecompose covariances → scales (sqrt of eigenvalues), rotmat → quat.
    # eigh returns eigenvalues in ascending order; we want scales as the per-axis
    # sigma, so sqrt the eigenvalues and treat eigvecs as the rotation matrix.
    print(f"  eigendecomposing {N} covariances...")
    eigvals, eigvecs = np.linalg.eigh(covs.astype(np.float64))  # eigvals (N,3), eigvecs (N,3,3)
    eigvals = np.clip(eigvals, 1e-12, None)
    scales = np.sqrt(eigvals).astype(np.float32)              # (N, 3) linear
    # Ensure a right-handed rotation matrix (eigvecs may be left-handed → flip last col).
    det = np.linalg.det(eigvecs)
    flip = det < 0
    eigvecs[flip, :, 2] *= -1.0
    quats = _rotmat_to_quat_wxyz(eigvecs.astype(np.float32))  # (N, 4) wxyz

    # Convert RGB → SH-DC, sigmoid opacity → logit, linear scale → log scale.
    rgbs_clipped = rgbs.clip(1e-6, 1 - 1e-6)
    features_dc = ((rgbs_clipped - 0.5) / C0).astype(np.float32)
    opac_clipped = opac.clip(1e-6, 1 - 1e-6)
    opacity_logits = np.log(opac_clipped / (1 - opac_clipped)).astype(np.float32)
    log_scales = np.log(scales).astype(np.float32)

    # Assemble PLY vertex record.
    attrs = _construct_attributes()
    dtypes = [(name, "f4") for name in attrs]
    arr = np.zeros(N, dtype=dtypes)
    arr["x"] = means[:, 0]; arr["y"] = means[:, 1]; arr["z"] = means[:, 2]
    arr["nx"] = 0.0; arr["ny"] = 0.0; arr["nz"] = 0.0
    arr["f_dc_0"] = features_dc[:, 0]
    arr["f_dc_1"] = features_dc[:, 1]
    arr["f_dc_2"] = features_dc[:, 2]
    # f_rest_* left zero (no SH > 0).
    arr["opacity"] = opacity_logits
    arr["scale_0"] = log_scales[:, 0]
    arr["scale_1"] = log_scales[:, 1]
    arr["scale_2"] = log_scales[:, 2]
    arr["rot_0"] = quats[:, 0]
    arr["rot_1"] = quats[:, 1]
    arr["rot_2"] = quats[:, 2]
    arr["rot_3"] = quats[:, 3]

    el = PlyElement.describe(arr, "vertex")
    out = args.out or args.pt.with_suffix(".ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(out)
    print(f"\nwrote {out}  ({N} splats)")
    print(f"Open in SuperSplat: https://superspl.at/editor  (drag-and-drop)")


if __name__ == "__main__":
    main()
