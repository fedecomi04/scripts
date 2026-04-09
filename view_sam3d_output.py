#!/usr/bin/env python3
"""Open a generated SAM 3D Objects `.ply` output with Open3D.

Run:
    python view_sam3d_output.py /path/to/sam3d_object_gs.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to the generated .ply file")
    return parser.parse_args()


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

    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) == 0:
        raise RuntimeError(f"No mesh or point cloud geometry found in {path}")

    o3d.visualization.draw_geometries([point_cloud], window_name=str(path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
