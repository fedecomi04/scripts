#!/usr/bin/env python3
"""Prove the fix: isolating the object in the back-projected target (remove the
supporting plane + keep the largest cluster) snaps the registration centroid
from the table-biased location onto the object.

Usage: diag_fix_isolate.py <dataset_dir>
"""
import sys, json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image


def isolate_object_in_target(points, colors, min_plane_frac=0.30):
    """Remove the dominant supporting plane (table) if it's a large fraction of
    the cloud, then keep the largest spatial cluster (the object)."""
    if len(points) < 50:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    # 1) drop the supporting plane if it dominates (table under the object)
    try:
        _, inl = pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=300)
        inl = np.asarray(inl)
        if len(inl) >= min_plane_frac * len(points):
            keep = np.ones(len(points), bool); keep[inl] = False
            points, colors = points[keep], colors[keep]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    except Exception:
        pass
    if len(points) < 20:
        return points, colors
    # 2) keep the largest connected cluster (the object body)
    nn = o3d.geometry.KDTreeFlann(pcd)
    d = []
    for i in range(0, len(points), max(1, len(points) // 2000)):
        _, idx, dist = nn.search_knn_vector_3d(pcd.points[i], 2)
        if len(dist) > 1:
            d.append(dist[1] ** 0.5)
    spacing = float(np.median(d)) if d else 0.004
    eps = max(3.0 * spacing, 0.01)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
    if labels.max() >= 0:
        biggest = np.argmax(np.bincount(labels[labels >= 0]))
        keep = labels == biggest
        points, colors = points[keep], colors[keep]
    return points, colors


def main():
    ddir = Path(sys.argv[1])
    ss = ddir / "static_scene"
    dbg = ddir / "dynamic_scene" / "initialization_debug"
    tj = json.load(open(ss / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    fr = tj["frames"][-1]
    c2w = np.array(fr["transform_matrix"], np.float64)
    depth = np.array(Image.open(ss / "depth" / f"{Path(fr['file_path']).stem}.tiff")).astype(np.float32) * 1e-3
    mask = np.array(Image.open(dbg / "static0_obj_00_mask.png").convert("L")) > 127

    ys, xs = np.where(mask & (depth > 1e-4))
    z = depth[ys, xs]
    med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
    k = np.abs(z - med) < 5.0 * 1.4826 * mad
    ys, xs, z = ys[k], xs[k], z[k]
    x = (xs - cx) / fx * z; y = -(ys - cy) / fy * z
    tgt = (np.stack([x, y, -z], -1)) @ c2w[:3, :3].T + c2w[:3, 3]
    col = np.full_like(tgt, 0.5)

    clean, _ = isolate_object_in_target(tgt.copy(), col.copy())

    def ext(a): return np.round((a.max(0) - a.min(0)) * 100, 1)
    print(f"contaminated target : N={len(tgt):>6}  centroid={np.round(tgt.mean(0),4)}  extent(cm)={ext(tgt)}")
    print(f"isolated object     : N={len(clean):>6}  centroid={np.round(clean.mean(0),4)}  extent(cm)={ext(clean)}")
    print(f"centroid moved      : {np.linalg.norm(clean.mean(0)-tgt.mean(0))*100:.2f} cm "
          f"({np.round((clean.mean(0)-tgt.mean(0))*100,2)} cm XYZ)")

    out = dbg / "DIAG_isolate_contamGRAY_isolatedGREEN.ply"
    P = np.concatenate([tgt, clean])
    C = np.concatenate([np.tile([150, 150, 150], (len(tgt), 1)),
                        np.tile([0, 220, 0], (len(clean), 1))]) / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(str(out), pc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
