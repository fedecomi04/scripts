"""Simple near/far adaptive downsample.

Strategy (per user spec 2026-06-01):
    near  — within 1.0 m of the LAST camera pose → kept at native density
    far   — beyond 1.0 m → voxel-downsampled to 5 mm

No curvature, no color gradient, no feature score. Two regions, two
densities, concatenate.

Input  : a fused PLY (e.g. voxel_015_x0p1mm.ply produced by online fusion)
Output : <stem>_adaptive.ply  — concatenation of (near_full, far_5mm)
         and on-screen breakdown of point counts.

Usage:
    python scripts/adaptive_downsample.py [ply_path]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d


NEAR_RADIUS_M = 1.0    # hardcoded by user request
FAR_VOXEL_M = 0.005    # hardcoded by user request (5 mm)


def _last_camera_position(ply_path: Path):
    """Resolve the last frame's camera world position from a sibling
    transforms.json. Walks up from the PLY's dir."""
    for parent in [ply_path.parent, *ply_path.parents]:
        for cand in [parent / "static_scene" / "transforms.json",
                     parent / "transforms.json"]:
            if cand.exists():
                meta = json.loads(cand.read_text())
                frames = meta.get("frames", [])
                if frames:
                    T = np.asarray(frames[-1]["transform_matrix"], dtype=np.float64)
                    return T[:3, 3]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "ply_path", type=Path, nargs="?",
        default=Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/validate_run_1/_voxel_sweep/voxel_015_x0p1mm.ply"),
    )
    args = ap.parse_args()

    ply_path = args.ply_path.resolve()
    print(f"[downsample] loading {ply_path}")
    pc = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pc.points)
    N = pts.shape[0]
    print(f"[downsample] {N:,} points")

    cam = _last_camera_position(ply_path)
    if cam is None:
        raise SystemExit("ERROR: could not find transforms.json near PLY — cannot locate last camera pose")
    print(f"[downsample] last camera position: {cam.round(3).tolist()}")

    # Boolean near/far split.
    t = time.time()
    depth = np.linalg.norm(pts - cam, axis=1).astype(np.float32)
    near_mask = depth <= NEAR_RADIUS_M
    n_near = int(near_mask.sum())
    n_far = N - n_near
    print(f"[downsample] split  ({time.time()-t:.2f}s):  near={n_near:,}  ({100*n_near/N:.1f}%)   far={n_far:,}")

    # Split into two o3d clouds.
    near_pc = pc.select_by_index(np.where(near_mask)[0])
    far_pc = pc.select_by_index(np.where(~near_mask)[0])

    # Voxel-downsample far to FAR_VOXEL_M.
    t = time.time()
    far_down = far_pc.voxel_down_sample(FAR_VOXEL_M)
    n_far_down = int(np.asarray(far_down.points).shape[0])
    print(f"[downsample] far voxel-down to {FAR_VOXEL_M*1000:.1f} mm  ({time.time()-t:.2f}s):  "
          f"{n_far:,} → {n_far_down:,}  ({100*n_far_down/max(n_far,1):.1f}% kept)")

    # Concatenate.
    out = near_pc + far_down
    n_out = int(np.asarray(out.points).shape[0])
    reduction = N / max(n_out, 1)
    print(f"[downsample] TOTAL: {N:,} → {n_out:,}  ({reduction:.1f}× reduction)")

    out_path = ply_path.with_name(ply_path.stem + "_adaptive.ply")
    o3d.io.write_point_cloud(str(out_path), out)
    print(f"[downsample] wrote {out_path}")

    # Diagnostic: also write a recoloured version (red = near, original = far).
    near_colored = pc.select_by_index(np.where(near_mask)[0])
    near_cols = np.asarray(near_colored.colors).copy()
    near_cols[:] = [1.0, 0.0, 0.0]
    near_colored.colors = o3d.utility.Vector3dVector(near_cols)
    diag = near_colored + far_down
    diag_path = ply_path.with_name(ply_path.stem + "_adaptive_diag.ply")
    o3d.io.write_point_cloud(str(diag_path), diag)
    print(f"[downsample] wrote {diag_path}  (red = near-full, original-colour = far-downsampled)")


if __name__ == "__main__":
    main()
