#!/usr/bin/env python3
"""Open a generated SAM 3D Objects `.ply` output with Open3D.

Run:
    python view_sam3d_output.py /path/to/sam3d_object_gs.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to the generated .ply file")
    return parser.parse_args()


def _decode_colors_from_vertex(vertex) -> np.ndarray | None:
    names = set(vertex.dtype.names or ())
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        return np.clip(features_dc * C0 + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)
    return None


def _build_gaussian_pcd(path: Path) -> o3d.geometry.PointCloud:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    colors = _decode_colors_from_vertex(vertex)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return point_cloud


def _draw_point_cloud(point_cloud: o3d.geometry.PointCloud, window_name: str) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(point_cloud)
    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.asarray([0.0, 0.0, 0.0])
    vis.run()
    vis.destroy_window()


def main() -> int:
    args = parse_args()
    path = args.path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=str(path.name))
        return 0

    point_cloud = _build_gaussian_pcd(path)
    if len(point_cloud.points) == 0:
        raise RuntimeError(f"No mesh or point cloud geometry found in {path}")
    print(f"[view_sam3d_output] points={len(point_cloud.points)} has_colors={point_cloud.has_colors()}")
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        print(
            "[view_sam3d_output] color stats:",
            f"min={colors.min():.4f}",
            f"max={colors.max():.4f}",
            f"mean={colors.mean():.4f}",
            f"std={colors.std():.4f}",
        )
    _draw_point_cloud(point_cloud, window_name=str(path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
