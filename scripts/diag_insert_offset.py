#!/usr/bin/env python3
"""Diagnose the SAM3D insert offset: is the back-projected TARGET offset from
the scene, or is the INSERTED cloud offset from the target?

Loads, all in world frame:
  * inserted object   = static_state.pt means where inserted_flags==1   (GREEN)
  * back-proj target  = static0_obj_00_sam3d_target_reg_ref.ply         (BLUE)
  * NDP source ref    = static0_obj_00_sam3d_source_reg_ref.ply         (MAGENTA)
  * scene context     = static_state.pt means where inserted_flags==0   (GRAY, subsampled)

Prints centroid offsets and writes a single colored PLY into the dataset's
initialization_debug dir for the user to judge in a viewer.

Usage: diag_insert_offset.py <dataset_dir>
"""
import sys
import numpy as np
import torch
from pathlib import Path


def read_ply_xyz(path: Path) -> np.ndarray:
    """Minimal binary/ascii PLY xyz reader (no open3d dependency needed)."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(path))
    return np.asarray(pc.points, dtype=np.float64)


def write_ply_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector((rgb.astype(np.float64) / 255.0).clip(0, 1))
    o3d.io.write_point_cloud(str(path), pc)


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene" / "static_state.pt"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"

    blob = torch.load(ss, map_location="cpu")
    sd = blob["model_state_dict"]
    means = sd["gauss_params.means"].numpy().astype(np.float64)
    inserted = sd["inserted_flags"].squeeze(-1).numpy() > 0.5
    inst = sd["object_instance_ids"].squeeze(-1).numpy()

    ins_xyz = means[inserted]                       # GREEN  (the SAM3D insert)
    scene_xyz = means[~inserted]                    # scene (TSDF seed + trained)

    target = read_ply_xyz(art / "static0_obj_00_sam3d_target_reg_ref.ply")   # BLUE
    src_ref_p = art / "static0_obj_00_sam3d_source_reg_ref.ply"
    source = read_ply_xyz(src_ref_p) if src_ref_p.exists() else np.zeros((0, 3))

    def c(a):
        return a.mean(axis=0) if len(a) else np.full(3, np.nan)

    C_ins, C_tgt, C_src = c(ins_xyz), c(target), c(source)

    # Scene gaussians physically near the back-projected target = the scene's
    # own representation of the real object (independent of the flag-propagation,
    # which is biased toward the insert).
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(target)
    d_scene, _ = nn.kneighbors(scene_xyz)
    near = d_scene[:, 0] < 0.08            # 8 cm of the real object surface
    scene_obj = scene_xyz[near]
    C_sobj = c(scene_obj)

    print(f"\ndataset: {ddir.name}")
    print(f"  inserted (GREEN)      N={len(ins_xyz):>7d}  centroid={np.round(C_ins,4)}")
    print(f"  target   (BLUE)       N={len(target):>7d}  centroid={np.round(C_tgt,4)}")
    print(f"  source-ref (MAGENTA)  N={len(source):>7d}  centroid={np.round(C_src,4)}")
    print(f"  scene<8cm of target   N={len(scene_obj):>7d}  centroid={np.round(C_sobj,4)}")
    print()
    print("  OFFSETS (meters, world frame XYZ):")
    print(f"    inserted - target        = {np.round(C_ins - C_tgt,4)}  |{np.linalg.norm(C_ins-C_tgt)*100:.1f} cm|")
    print(f"    target   - scene_obj     = {np.round(C_tgt - C_sobj,4)}  |{np.linalg.norm(C_tgt-C_sobj)*100:.1f} cm|")
    print(f"    inserted - scene_obj     = {np.round(C_ins - C_sobj,4)}  |{np.linalg.norm(C_ins-C_sobj)*100:.1f} cm|")
    print()

    # Build combined colored PLY (subsample scene context for size).
    rng = np.random.default_rng(0)
    if len(scene_xyz) > 200_000:
        scene_xyz = scene_xyz[rng.choice(len(scene_xyz), 200_000, replace=False)]

    def col(a, rgb):
        return np.tile(np.array(rgb, np.float64), (len(a), 1))

    parts_xyz = [scene_xyz, target, ins_xyz]
    parts_rgb = [col(scene_xyz, (140, 140, 140)),   # scene = gray
                 col(target, (0, 90, 255)),          # target = blue
                 col(ins_xyz, (0, 220, 0))]          # inserted = green
    if len(source):
        parts_xyz.append(source)
        parts_rgb.append(col(source, (230, 0, 230))) # source-ref = magenta

    out = dbg / "DIAG_offset_scene-gray_target-blue_inserted-green_source-magenta.ply"
    write_ply_rgb(out, np.concatenate(parts_xyz), np.concatenate(parts_rgb))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
