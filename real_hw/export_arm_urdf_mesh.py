#!/usr/bin/env python3
"""Export the WHOLE dynaarm URDF (all link visual meshes at their FK poses) to a single mesh for
MeshLab/CloudCompare. Reuses the URDF path / STL substitution from dynamic_gs2/ros_mask.py.

Run (urdfpy lives in the ROS env):
  /home/mrc-cuhk/miniconda3/envs/dynamic_gs_ros/bin/python real_hw/export_arm_urdf_mesh.py
Out: real_hw/arm_urdf.obj  (+ .ply). Open in MeshLab.
Optional: pass joint values to pose it; default = all zeros (URDF home pose).
"""
from __future__ import annotations
import os, re, sys, tempfile
from pathlib import Path
import numpy as np
for _a, _r in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _a):
        setattr(np, _a, _r)
import trimesh
from urdfpy import URDF

# --- pulled verbatim from dynamic_gs2/ros_mask.py so this matches the mask renderer's geometry ---
URDF_PATH = Path(
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
    "active_camera_arm_examples/dynaarm_description/urdf/dynamic_gaussian_splat/"
    "dynaarm_with_gripper_for_gazebo_only_no_wrist_collision.urdf"
)
STL_DIR = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/stl")
PACKAGE_MAP = {
    "dynaarm_description": (
        "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
        "active_camera_arm_examples/dynaarm_description"
    ),
    "robotiq_2f_85_gripper_visualization": (
        "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
        "active_camera_arm_examples/robotiq/robotiq_2f_85_gripper_visualization"
    ),
}


def make_temp_resolved_urdf() -> str:
    text = URDF_PATH.read_text()

    def repl(match):
        pkg, rest = match.group(1), match.group(2)
        stl = STL_DIR / (Path(rest).stem + ".stl")
        if stl.exists():
            return str(stl)
        if pkg not in PACKAGE_MAP:
            raise RuntimeError(f"Missing package root for '{pkg}'")
        return str(Path(PACKAGE_MAP[pkg]) / rest)

    text = re.sub(r"package://([^/]+)/([^\"'<> ]+)", repl, text)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(text); tmp.flush(); tmp.close()
    return tmp.name


def main():
    tmp = make_temp_resolved_urdf()
    robot = URDF.load(tmp)
    link_fk = robot.link_fk(use_names=True)          # all zeros = home pose
    print(f"[urdf] {len(robot.links)} links")

    parts = []
    for link in robot.links:
        base_to_link = link_fk.get(link.name)
        if base_to_link is None:
            continue
        for visual in link.visuals:
            geom = visual.geometry
            tri = None
            inner = None
            for attr in ("mesh", "box", "cylinder", "sphere"):
                inner = getattr(geom, attr, None)
                if inner is not None:
                    break
            if inner is None:
                continue
            if getattr(inner, "filename", None):
                loaded = trimesh.load(inner.filename, force="scene")
                tri = trimesh.util.concatenate(
                    [g.copy() for g in loaded.geometry.values()]) if isinstance(loaded, trimesh.Scene) else loaded.copy()
                if getattr(inner, "scale", None) is not None:
                    tri.apply_scale(np.array(inner.scale, dtype=np.float64))
            elif getattr(inner, "size", None) is not None:
                tri = trimesh.creation.box(extents=np.array(inner.size, dtype=np.float64))
            elif hasattr(inner, "radius") and hasattr(inner, "length"):
                tri = trimesh.creation.cylinder(radius=float(inner.radius), height=float(inner.length))
            elif hasattr(inner, "radius"):
                tri = trimesh.creation.icosphere(radius=float(inner.radius))
            if tri is None:
                continue
            origin = visual.origin if visual.origin is not None else np.eye(4)
            tri.apply_transform(np.asarray(base_to_link, float) @ np.asarray(origin, float))
            parts.append(tri)
            print(f"  + {link.name}: {len(tri.vertices)} v")

    combined = trimesh.util.concatenate(parts)
    out = Path(__file__).resolve().parent
    combined.export(out / "arm_urdf.obj")
    combined.export(out / "arm_urdf.ply")
    print(f"[urdf] wrote {out/'arm_urdf.obj'} + .ply  ({len(combined.vertices)} verts, {len(parts)} meshes)")
    print(f"[urdf] open:  meshlab {out/'arm_urdf.ply'}")
    try:
        os.unlink(tmp)
    except Exception:
        pass


if __name__ == "__main__":
    main()
