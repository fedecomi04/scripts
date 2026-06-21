#!/usr/bin/env python3
"""Bake the dynaarm_ELBOW mesh + a camera-mount reference marker into one file for MeshLab/Blender.

Reference point (ELBOW link frame), per the requested construction:
  X = most-POSITIVE-X mesh vertex   = +0.04125
  Y = most-NEGATIVE-Y mesh vertex   = -0.14000
  Z = 0
=> REF_XYZ = (0.04125, -0.14000, 0.0)  metres

Outputs (next to this script):
  elbow_cam_marker.glb  -- mesh (grey) + ELBOW link triad (RGB) + red marker sphere at REF_XYZ.
                           Open in MeshLab (File>Import Mesh) or Blender (File>Import>glTF 2.0).
  elbow_cam_marker.ply  -- JUST the marker sphere, to drag onto the raw mesh in MeshLab if preferred.

Run: /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python real_hw/export_elbow_cam_marker.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

MESH_STL = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/stl/300_elbow_mesh.stl")

# Built from the mesh extents (recomputed below and asserted, so it can't silently drift).
REF_XYZ = np.array([0.04125, -0.14000, 0.0])
MARKER_RADIUS = 0.012      # 1.2 cm sphere, visible against the ~8 cm mesh
AXIS_LEN = 0.08
AXIS_RAD = 0.0025


def triad(length: float, radius: float) -> trimesh.Trimesh:
    """One mesh holding 3 colored cylinders (X red / Y green / Z blue) from the origin."""
    parts = []
    specs = [
        (np.pi / 2, [0, 1, 0], [230, 40, 40, 255]),    # +X red
        (-np.pi / 2, [1, 0, 0], [40, 200, 40, 255]),   # +Y green
        (0.0, [1, 0, 0], [40, 90, 230, 255]),          # +Z blue
    ]
    for ang, axis, color in specs:
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=20)
        cyl.apply_translation([0, 0, length / 2.0])
        cyl.apply_transform(trimesh.transformations.rotation_matrix(ang, axis))
        cyl.visual.face_colors = color
        parts.append(cyl)
    return trimesh.util.concatenate(parts)


def main() -> None:
    mesh = trimesh.load(MESH_STL, force="mesh")
    v = mesh.vertices
    # Re-derive + assert the reference point so this file is self-checking.
    ref = np.array([v[:, 0].max(), v[:, 1].min(), 0.0])
    assert np.allclose(ref, REF_XYZ, atol=1e-4), f"mesh extents changed: {ref} vs {REF_XYZ}"
    print(f"[export] mesh {MESH_STL.name}: {len(v)} verts")
    print(f"[export]   X {v[:,0].min():+.5f}..{v[:,0].max():+.5f}  "
          f"Y {v[:,1].min():+.5f}..{v[:,1].max():+.5f}  Z {v[:,2].min():+.5f}..{v[:,2].max():+.5f}")
    print(f"[export] REF point (most+X, most-Y, Z=0) = {np.round(ref,5).tolist()}")

    mesh.visual.face_colors = [185, 185, 195, 255]

    marker = trimesh.creation.uv_sphere(radius=MARKER_RADIUS)
    marker.apply_translation(ref)
    marker.visual.face_colors = [235, 30, 30, 255]

    here = Path(__file__).parent
    # standalone marker for drag-onto-mesh-in-MeshLab
    ply_path = here / "elbow_cam_marker.ply"
    marker.export(ply_path)
    print(f"[export] wrote {ply_path}  (marker sphere only)")

    # combined scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="elbow_mesh")
    scene.add_geometry(triad(AXIS_LEN, AXIS_RAD), node_name="elbow_link_frame")
    scene.add_geometry(marker, node_name="cam_ref_marker")
    glb_path = here / "elbow_cam_marker.glb"
    scene.export(glb_path)
    print(f"[export] wrote {glb_path}  (mesh + triad + marker)")
    print("[export] open in MeshLab:  meshlab " + str(glb_path))
    print("[export] open in Blender:  File > Import > glTF 2.0")


if __name__ == "__main__":
    main()
