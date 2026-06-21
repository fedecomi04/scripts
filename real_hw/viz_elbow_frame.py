#!/usr/bin/env python3
"""Visualize the dynaarm_ELBOW link with its frames, to place the real-HW camera mount.

Renders, all in the ELBOW *link* frame:
  - the elbow mesh (300_elbow_mesh)
  - the ELBOW link-frame axis triad at the origin (where URDF FK puts the link)
  - the collision cylinder (length 0.46, radius 0.047) at its URDF origin
    (xyz="0.185 -0.0935 0" rpy="0 1.57 0"), plus a triad at the cylinder origin
  - both cylinder-axis END caps marked (red = +axis end, blue = -axis end), since the
    camera sits "at the limit of the collision box + an offset" — these are the faces
    you measured from
  - OPTIONAL: a candidate camera position (edit CAMERA_* below) drawn as a green triad

Axis colors for every triad: X=red, Y=green, Z=blue. Units = metres.

Run:  /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python real_hw/viz_elbow_frame.py
GUI (interactive):  add  --gui   (trimesh scene viewer, needs the GL display)
Static PNG always written next to this script: viz_elbow_frame.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

# ---- URDF facts for dynaarm_ELBOW (from the gazebo URDF) ----
MESH_STL = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/stl/300_elbow_mesh.stl")
MESH_DAE = Path(
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
    "active_camera_arm_examples/dynaarm_description/meshes/300_elbow_mesh.dae"
)
COLL_XYZ = np.array([0.185, -0.0935, 0.0])     # collision <origin xyz>
COLL_RPY = np.array([0.0, 1.57, 0.0])          # collision <origin rpy>
COLL_LEN = 0.46                                 # cylinder length (along its local Z)
COLL_RAD = 0.047                                # cylinder radius

AXIS_LEN = 0.08          # triad arm length (m)
AXIS_RAD = 0.0025        # triad arm radius (m)

# ---- OPTIONAL candidate camera pose, expressed in the ELBOW link frame ----
# Leave as None until you've worked out the offset; then set xyz (m) + rpy (rad).
CAMERA_XYZ: tuple[float, float, float] | None = None
CAMERA_RPY: tuple[float, float, float] = (0.0, 0.0, 0.0)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw (Rz @ Ry @ Rx) -> 3x3."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def triad(T: np.ndarray, length: float = AXIS_LEN, radius: float = AXIS_RAD) -> list[trimesh.Trimesh]:
    """Three colored cylinders (X red, Y green, Z blue) for the frame T."""
    colors = [[230, 40, 40, 255], [40, 200, 40, 255], [40, 90, 230, 255]]
    out = []
    for axis in range(3):
        # cylinder built along +Z, then rotated so it points along `axis`, then placed by T
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        cyl.apply_translation([0, 0, length / 2.0])  # base at origin, grows along +Z
        if axis == 0:      # -> +X
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        elif axis == 1:    # -> +Y
            rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        else:              # +Z
            rot = np.eye(4)
        cyl.apply_transform(rot)
        cyl.apply_transform(T)
        cyl.visual.face_colors = colors[axis]
        out.append(cyl)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="open interactive trimesh viewer")
    args = ap.parse_args()

    scene = trimesh.Scene()

    # elbow mesh (prefer the trimmed STL; fall back to the DAE)
    mesh_path = MESH_STL if MESH_STL.exists() else MESH_DAE
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.visual.face_colors = [180, 180, 190, 160]   # translucent grey so triads show through
    scene.add_geometry(mesh, node_name="elbow_mesh")
    print(f"[viz] mesh: {mesh_path.name}  bounds(min..max) =\n{mesh.bounds}")

    # ELBOW link-frame triad at origin
    for g in triad(np.eye(4)):
        scene.add_geometry(g)
    print("[viz] link-frame triad at ELBOW origin (0,0,0)")

    # collision cylinder + its triad
    T_coll = transform(COLL_XYZ, COLL_RPY)
    cyl = trimesh.creation.cylinder(radius=COLL_RAD, height=COLL_LEN, sections=32)
    cyl.apply_transform(T_coll)
    cyl.visual.face_colors = [60, 200, 220, 70]      # translucent cyan
    scene.add_geometry(cyl, node_name="collision_cyl")
    for g in triad(T_coll, length=0.05):
        scene.add_geometry(g)
    # the two cylinder end-cap centres (camera is at "the limit" of this box)
    end_plus = (T_coll @ np.array([0, 0, +COLL_LEN / 2, 1]))[:3]
    end_minus = (T_coll @ np.array([0, 0, -COLL_LEN / 2, 1]))[:3]
    for c, color in ((end_plus, [230, 40, 40, 255]), (end_minus, [40, 90, 230, 255])):
        s = trimesh.creation.uv_sphere(radius=0.012)
        s.apply_translation(c)
        s.visual.face_colors = color
        scene.add_geometry(s)
    print(f"[viz] collision cylinder origin={COLL_XYZ.tolist()} rpy={COLL_RPY.tolist()}")
    print(f"[viz]   +axis end (red sphere)  in ELBOW frame = {np.round(end_plus, 4).tolist()}")
    print(f"[viz]   -axis end (blue sphere) in ELBOW frame = {np.round(end_minus, 4).tolist()}")

    # optional candidate camera
    if CAMERA_XYZ is not None:
        T_cam = transform(np.array(CAMERA_XYZ), np.array(CAMERA_RPY))
        for g in triad(T_cam, length=0.10, radius=0.004):
            g.visual.face_colors = [40, 230, 120, 255]   # all-green so it stands out
            scene.add_geometry(g)
        print(f"[viz] CAMERA candidate xyz={CAMERA_XYZ} rpy={CAMERA_RPY}")

    out_png = Path(__file__).with_suffix(".png")
    try:
        png = scene.save_image(resolution=(1400, 1000), visible=True)
        out_png.write_bytes(png)
        print(f"[viz] wrote {out_png} (trimesh/pyglet)")
    except Exception as e:
        print(f"[viz] pyglet offscreen failed ({e}); using matplotlib fallback")
        _matplotlib_png(mesh, T_coll, end_plus, end_minus, out_png)

    if args.gui:
        scene.show()


def _matplotlib_png(mesh, T_coll, end_plus, end_minus, out_png: Path) -> None:
    """3-view static render (no GL): mesh wireframe + link triad + collision cylinder + ends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    views = [("XY_top", 90, -90), ("XZ_front", 0, -90), ("3D", 22, -60)]
    cyl = trimesh.creation.cylinder(radius=COLL_RAD, height=COLL_LEN, sections=24)
    cyl.apply_transform(T_coll)
    allpts = np.vstack([mesh.vertices, cyl.vertices, [end_plus, end_minus, [0, 0, 0]]])
    c = allpts.mean(0)
    r = (allpts.max(0) - allpts.min(0)).max() / 2 * 1.05

    stem = out_png.with_suffix("")           # drop .png; we append per-view
    written = []
    for title, elev, azim in views:
        fig = plt.figure(figsize=(13, 12))
        ax = fig.add_subplot(111, projection="3d")
        # mesh
        ax.add_collection3d(Poly3DCollection(
            mesh.vertices[mesh.faces], alpha=0.30, facecolor="#b0b0be", edgecolor="none"))
        # link-frame triad at origin
        for vec, col in ([0.08, 0, 0], "r"), ([0, 0.08, 0], "g"), ([0, 0, 0.08], "b"):
            ax.quiver(0, 0, 0, *vec, color=col, linewidth=3.0)
        # collision-origin triad (shorter, dotted)
        o = T_coll[:3, 3]
        for axj, col in enumerate("rgb"):
            v = T_coll[:3, axj] * 0.05
            ax.quiver(*o, *v, color=col, linewidth=2.0, linestyle=":")
        # end caps + labels
        ax.scatter(*end_plus, color="red", s=140)
        ax.scatter(*end_minus, color="blue", s=140)
        ax.text(*end_plus, "  +X end\n  %s" % np.round(end_plus, 3).tolist(), color="red", fontsize=9)
        ax.text(*end_minus, "  -X end\n  %s" % np.round(end_minus, 3).tolist(), color="blue", fontsize=9)
        ax.text(0, 0, 0, "  ELBOW origin", color="k", fontsize=9)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title("dynaarm_ELBOW  —  %s view\n"
                     "grey=mesh  solid RGB=link frame  dotted RGB=collision-cyl origin  "
                     "red=+X cyl end  blue=-X cyl end" % title, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        path = Path("%s_%s.png" % (stem, title))
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
        print("[viz] wrote %s (matplotlib, %s view)" % (path, title))
    print("[viz] %d images: %s" % (len(written), ", ".join(p.name for p in written)))


if __name__ == "__main__":
    main()
