"""Convert a `post_fusion_state.pt` (saved at the static→dynamic boundary)
into the viser-ready blob format that `scripts/view_splats_viser.py`
opens. Lets you inspect the scene + fused object cloud without re-running
the pipeline.

Usage:
    python scripts/dump_post_fusion_to_viser.py \\
        /path/to/dataset/static_scene/post_fusion_state.pt \\
        [--out /path/to/output.pt]

If --out is omitted, the file is written next to the input as
``post_fusion_viser.pt``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path, help="Path to post_fusion_state.pt")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    blob = torch.load(args.state, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"]

    def _get(name):
        for k in (name, f"_buffers.{name}", f"gauss_params.{name}"):
            if k in state:
                return state[k]
        raise KeyError(f"{name} not in state_dict; available: {sorted(state.keys())[:15]}...")

    means = _get("means").float()
    scales_lin = _get("scales").float().exp()
    quats = _get("quats").float()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = quats.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    S2 = (scales_lin ** 2)[:, :, None] * torch.eye(3, dtype=R.dtype)[None]
    cov = (R @ S2 @ R.transpose(-1, -2)).numpy().astype(np.float32)

    dc = _get("features_dc").float().numpy()
    if dc.ndim == 3:
        dc = dc[:, 0, :]
    rgbs = (C0 * dc + 0.5).clip(0.0, 1.0).astype(np.float32)
    opacities = torch.sigmoid(_get("opacities").float()).numpy().reshape(-1, 1).astype(np.float32)

    inserted = state.get("inserted_flags")
    inserted_np = inserted.numpy().squeeze(-1).astype(np.uint8) if inserted is not None else None
    ids = state.get("object_instance_ids")
    ids_np = ids.numpy().squeeze(-1).astype(np.int64) if ids is not None else None

    out_blob = {
        "means": means.numpy().astype(np.float32),
        "covariances": cov,
        "rgbs": rgbs,
        "opacities": opacities,
        "inserted_flags": inserted_np,
        "object_instance_ids": ids_np,
        "anchor_frame": "post_fusion",
        "selected_frames": [],
    }
    out_path = args.out or args.state.with_name("post_fusion_viser.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, out_path)
    n = means.shape[0]
    n_obj = int((ids_np > 0).sum()) if ids_np is not None else -1
    print(f"wrote {n} splats ({n_obj} with non-zero instance_id) → {out_path}")


if __name__ == "__main__":
    main()
