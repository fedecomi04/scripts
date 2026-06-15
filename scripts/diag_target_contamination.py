#!/usr/bin/env python3
"""Is the registration target contaminated by table/background points (from a
loose mask), biasing the centroid-based rigid init that seeds NDP?

Splits the back-projected target into a dominant plane (table) vs the rest
(object) via RANSAC, reports the table fraction and the centroid bias it
induces, and checks where the inserted object actually sits.

Usage: diag_target_contamination.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
import torch
from pathlib import Path


def c(a):
    return a.mean(0) if len(a) else np.full(3, np.nan)


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    art = ddir / "dynamic_scene" / "initialization_artifacts"
    dbg = ddir / "dynamic_scene" / "initialization_debug"

    tgt_pc = o3d.io.read_point_cloud(str(art / "static0_obj_00_sam3d_target_reg_ref.ply"))
    tgt = np.asarray(tgt_pc.points)
    print(f"target: {len(tgt)} pts   extent(cm)={np.round((tgt.max(0)-tgt.min(0))*100,1)}")

    # RANSAC plane (table) on the target.
    plane, inl = tgt_pc.segment_plane(distance_threshold=0.006, ransac_n=3, num_iterations=400)
    inl = np.asarray(inl)
    is_table = np.zeros(len(tgt), bool); is_table[inl] = True
    table, obj = tgt[is_table], tgt[~is_table]
    print(f"  plane (table) pts = {len(table)} ({100*len(table)/len(tgt):.0f}%)   "
          f"object pts = {len(obj)} ({100*len(obj)/len(tgt):.0f}%)")
    print(f"  full-target centroid = {np.round(c(tgt),4)}")
    print(f"  object-only centroid = {np.round(c(obj),4)}")
    print(f"  centroid BIAS from table contamination = {np.linalg.norm(c(tgt)-c(obj))*100:.2f} cm")
    print(f"     (bias XYZ = {np.round((c(tgt)-c(obj))*100,2)} cm)\n")

    # Inserted object from static_state.
    blob = torch.load(ss / "static_state.pt", map_location="cpu")
    sd = blob["model_state_dict"]
    means = sd["gauss_params.means"].numpy()
    ins = means[sd["inserted_flags"].squeeze(-1).numpy() > 0.5]
    print(f"inserted: {len(ins)} pts")
    print(f"  inserted centroid          = {np.round(c(ins),4)}")
    print(f"  inserted - full_target ctr = {np.linalg.norm(c(ins)-c(tgt))*100:.2f} cm")
    print(f"  inserted - object_only ctr = {np.linalg.norm(c(ins)-c(obj))*100:.2f} cm")

    # How close does the inserted object sit to the OBJECT part vs TABLE part of target?
    from sklearn.neighbors import NearestNeighbors
    nn_obj = NearestNeighbors(n_neighbors=1).fit(obj)
    d_obj, _ = nn_obj.kneighbors(ins)
    print(f"  median NN(inserted -> target_object) = {np.median(d_obj[:,0])*100:.2f} cm")

    # Write a PLY the user can judge.
    def col(a, rgb): return np.tile(np.array(rgb, float), (len(a), 1))
    seed = o3d.io.read_point_cloud(str(ss / "depth_camera_init_points.ply"))
    seed_pts = np.asarray(seed.points)
    nn_t = NearestNeighbors(n_neighbors=1).fit(tgt)
    dn, _ = nn_t.kneighbors(seed_pts)
    seed_loc = seed_pts[dn[:, 0] < 0.12]
    out = dbg / "DIAG_contam_seedGRAY_objBLUE_tableRED_insertedGREEN.ply"
    P = np.concatenate([seed_loc, obj, table, ins])
    C = np.concatenate([col(seed_loc, (150, 150, 150)), col(obj, (0, 90, 255)),
                        col(table, (255, 0, 0)), col(ins, (0, 220, 0))])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector((C / 255.0).clip(0, 1))
    o3d.io.write_point_cloud(str(out), pc)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
