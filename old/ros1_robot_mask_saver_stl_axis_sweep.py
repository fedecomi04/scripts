#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image
from urdfpy import URDF

# Do NOT force EGL here.
# Let the environment decide (or leave it unset for desktop OpenGL).
import pyrender
import trimesh


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class RobotMaskSaver:
    def __init__(
        self,
        urdf_path: str,
        dataset_dir: str,
        stl_dir: str,
        base_frame: str,
        camera_frame: str,
        image_topic: str,
        info_topic: str,
        camera_name: str,
        package_map: Dict[str, str],
        tf_timeout: float = 0.1,
        overwrite: bool = False,
        link_frame_prefix: str = "dynaarm_arm_tf",
        sweep_step_deg: float = 10.0,
        sweep_start_deg: float = -180.0,
        sweep_end_deg: float = 180.0,
        save_axis_sweep: bool = True,
        queue_size: int = 200,
    ) -> None:
        self.urdf_path = urdf_path
        self.dataset_dir = Path(dataset_dir)
        self.stl_dir = Path(stl_dir)
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.image_topic = image_topic
        self.info_topic = info_topic
        self.camera_name = camera_name
        self.tf_timeout = tf_timeout
        self.overwrite = overwrite
        self.link_frame_prefix = link_frame_prefix.strip("/")
        self.sweep_step_deg = float(sweep_step_deg)
        self.sweep_start_deg = float(sweep_start_deg)
        self.sweep_end_deg = float(sweep_end_deg)
        self.save_axis_sweep = bool(save_axis_sweep)
        self.queue_size = queue_size

        self.renders_dir = self.dataset_dir / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.renders_sweep_dir = self.dataset_dir / "renders_sweep"
        self.renders_sweep_dir.mkdir(parents=True, exist_ok=True)
        
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.intrinsics: Optional[CameraIntrinsics] = None
        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        self.scene: Optional[pyrender.Scene] = None
        self.camera_node = None
        self.robot_nodes: list[Tuple[str, object, object]] = []
        self.mesh_cache: Dict[str, trimesh.Trimesh] = {}

        # Queue of (seq, stamp)
        self.pending_frames: Deque[Tuple[int, rospy.Time]] = deque(maxlen=self.queue_size)

        rospy.loginfo("Loading URDF from %s", urdf_path)
        patched_urdf = self._make_temp_resolved_urdf(urdf_path, package_map, self.stl_dir)
        self.robot = URDF.load(patched_urdf)

        self.camera_info_sub = rospy.Subscriber(info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        self.image_sub = rospy.Subscriber(image_topic, Image, self._image_cb, queue_size=20)

    # -------------------------
    # ROS callbacks: NO rendering here
    # -------------------------

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.intrinsics is not None:
            return

        self.intrinsics = CameraIntrinsics(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(msg.K[0]),
            fy=float(msg.K[4]),
            cx=float(msg.K[2]),
            cy=float(msg.K[5]),
        )
        rospy.loginfo(
            "Camera intrinsics: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
            self.intrinsics.width,
            self.intrinsics.height,
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
        )

    def _image_cb(self, msg: Image) -> None:
        # Only queue work here. Do not touch pyrender/OpenGL in callbacks.
        seq = int(msg.header.seq)
        if not self.overwrite:
            if self.save_axis_sweep:
                if self._all_sweep_outputs_exist(seq):
                    return
            else:
                render_path = self._render_path_for_seq(seq)
                if render_path.exists():
                    return
        self.pending_frames.append((seq, msg.header.stamp))

    # -------------------------
    # URDF / mesh prep
    # -------------------------

    def _make_temp_resolved_urdf(self, urdf_path: str, package_map: Dict[str, str], stl_dir: Path) -> str:
        text = Path(urdf_path).read_text()

        def repl(match):
            pkg = match.group(1)
            rest = match.group(2)

            # Prefer converted STL by basename
            basename = Path(rest).stem + ".stl"
            stl_path = stl_dir / basename
            if stl_path.exists():
                rospy.loginfo("Using converted STL for %s -> %s", rest, stl_path)
                return str(stl_path)

            if pkg not in package_map:
                raise RuntimeError(f"Missing package root for '{pkg}'")
            return str(Path(package_map[pkg]) / rest)

        text = re.sub(r"package://([^/]+)/([^\"'<> ]+)", repl, text)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tmp.write(text)
        tmp.flush()
        tmp.close()
        rospy.loginfo("Wrote patched URDF to %s", tmp.name)
        return tmp.name

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        if self.intrinsics is None:
            return

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.intrinsics.width,
            viewport_height=self.intrinsics.height,
        )

        self.scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([0.2, 0.2, 0.2]),
        )

        camera = pyrender.IntrinsicsCamera(
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            znear=0.001,
            zfar=100.0,
        )
        self.camera_node = self.scene.add(camera, pose=np.eye(4, dtype=np.float32))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self.scene.add(light, pose=np.eye(4, dtype=np.float32))

        light2_pose = np.eye(4, dtype=np.float32)
        light2_pose[:3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add(pyrender.PointLight(color=np.ones(3), intensity=20.0), pose=light2_pose)
        self._build_scene()
        rospy.loginfo("Renderer initialized")

    def _build_scene(self) -> None:
        assert self.scene is not None

        for link in self.robot.links:
            for visual in link.visuals:
                geom = visual.geometry
                tri = self._geometry_to_trimesh(geom)
                if tri is None:
                    rospy.logwarn("Skipping unsupported visual geometry on link '%s'", link.name)
                    continue

                render_mesh = self._make_white_mesh(tri)
                pose = np.eye(4, dtype=np.float32)
                if visual.origin is not None:
                    pose = visual.origin.astype(np.float32)

                node = self.scene.add(render_mesh, pose=pose)
                self.robot_nodes.append((link.name, visual, node))

        rospy.loginfo("Scene built with %d visual nodes", len(self.robot_nodes))

    def _geometry_to_trimesh(self, geom) -> Optional[trimesh.Trimesh]:
        inner = None

        if hasattr(geom, "mesh") and geom.mesh is not None:
            inner = geom.mesh
        elif hasattr(geom, "box") and geom.box is not None:
            inner = geom.box
        elif hasattr(geom, "cylinder") and geom.cylinder is not None:
            inner = geom.cylinder
        elif hasattr(geom, "sphere") and geom.sphere is not None:
            inner = geom.sphere
        else:
            return None

        # Mesh geometry
        if hasattr(inner, "filename") and inner.filename is not None:
            scale = np.array(inner.scale, dtype=np.float32) if getattr(inner, "scale", None) is not None else None
            return self._load_trimesh(inner.filename, scale)

        # Box
        if hasattr(inner, "size") and inner.size is not None:
            size = np.array(inner.size, dtype=np.float32)
            return trimesh.creation.box(extents=size)

        # Cylinder
        if hasattr(inner, "radius") and hasattr(inner, "length"):
            radius = getattr(inner, "radius", None)
            length = getattr(inner, "length", None)
            if radius is not None and length is not None:
                return trimesh.creation.cylinder(radius=float(radius), height=float(length), sections=32)

        # Sphere
        if hasattr(inner, "radius") and not hasattr(inner, "length"):
            radius = getattr(inner, "radius", None)
            if radius is not None:
                return trimesh.creation.icosphere(radius=float(radius), subdivisions=2)

        return None

    def _load_trimesh(self, path: str, scale: Optional[np.ndarray]) -> trimesh.Trimesh:
        key = f"{path}|{None if scale is None else tuple(scale.tolist())}"
        if key in self.mesh_cache:
            return self.mesh_cache[key].copy()

        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = [g.copy() for g in loaded.geometry.values()]
            if not meshes:
                raise RuntimeError(f"No geometry found in mesh file: {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded.copy()

        if scale is not None:
            mesh.apply_scale(scale)

        self.mesh_cache[key] = mesh.copy()
        return mesh

    def _make_white_mesh(self, mesh: trimesh.Trimesh) -> pyrender.Mesh:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.9, 0.2, 0.2, 1.0),  # red
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
        )
        mesh = mesh.copy()
        mesh.vertex_normals
        return pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)

    # -------------------------
    # TF / pose / render
    # -------------------------

    @staticmethod
    def _transform_to_matrix(transform_msg) -> np.ndarray:
        t = transform_msg.transform.translation
        q = transform_msg.transform.rotation

        x, y, z, w = q.x, q.y, q.z, q.w
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float32)
        return mat

    def _lookup_camera_pose(self, stamp: rospy.Time) -> np.ndarray:
        tf_msg = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.camera_frame,
            stamp,
            timeout=rospy.Duration(self.tf_timeout),
        )
        return self._transform_to_matrix(tf_msg)

    def _tf_link_frame(self, link_name: str) -> str:
        if self.link_frame_prefix:
            return f"{self.link_frame_prefix}/{link_name}"
        return link_name

    def _update_robot_poses(self, stamp: rospy.Time) -> None:
        assert self.scene is not None

        for link_name, visual, node in self.robot_nodes:
            link_frame = self._tf_link_frame(link_name)

            if not self.tf_buffer.can_transform(
                self.base_frame,
                link_frame,
                stamp,
                timeout=rospy.Duration(self.tf_timeout),
            ):
                rospy.logwarn_throttle(2.0, "Skipping link without TF connection: %s", link_frame)
                continue

            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                link_frame,
                stamp,
                timeout=rospy.Duration(self.tf_timeout),
            )
            base_T_link = self._transform_to_matrix(tf_msg)

            link_T_visual = np.eye(4, dtype=np.float32)
            if visual.origin is not None:
                link_T_visual = visual.origin.astype(np.float32)

            base_T_visual = base_T_link @ link_T_visual
            self.scene.set_pose(node, pose=base_T_visual)

    # -------------------------
    # File naming / main loop
    # -------------------------

    def _rgb_name_for_seq(self, seq: int) -> str:
        return f"{self.camera_name}_{seq:05d}.png"

    def _render_path_for_seq(self, seq: int) -> Path:
        return self.renders_dir / self._rgb_name_for_seq(seq)

    def _angle_token(self, angle_deg: float) -> str:
        rounded = int(round(angle_deg))
        if rounded < 0:
            return f"m{abs(rounded):03d}"
        return f"p{rounded:03d}"

    def _sweep_render_path_for_seq(self, seq: int, axis: str, angle_deg: float) -> Path:
        return self.renders_sweep_dir / f"{self.camera_name}_{seq:05d}_{axis}_{self._angle_token(angle_deg)}.png"

    def _sweep_angles(self) -> list[float]:
        angles: list[float] = []
        angle = self.sweep_start_deg
        while angle <= self.sweep_end_deg + 1e-6:
            angles.append(round(angle, 6))
            angle += self.sweep_step_deg
        return angles

    def _all_sweep_outputs_exist(self, seq: int) -> bool:
        for axis in ("x", "y", "z"):
            for angle_deg in self._sweep_angles():
                if not self._sweep_render_path_for_seq(seq, axis, angle_deg).exists():
                    return False
        return True

    def _rotation_4x4(self, axis: str, angle_deg: float) -> np.ndarray:
        theta = np.deg2rad(angle_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        T = np.eye(4, dtype=np.float32)

        if axis == "x":
            T[:3, :3] = np.array([
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ], dtype=np.float32)
        elif axis == "y":
            T[:3, :3] = np.array([
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ], dtype=np.float32)
        elif axis == "z":
            T[:3, :3] = np.array([
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported axis '{axis}'")

        return T

    def _save_color_render(self, path: Path) -> bool:
        try:
            color, depth = self.renderer.render(self.scene)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Render failed: %s", exc)
            return False

        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), color_bgr)
        if not ok:
            rospy.logerr("Failed to save render to %s", path)
        return ok

    def process_once(self) -> None:
        self._ensure_renderer()
        if self.renderer is None or self.scene is None:
            return

        if not self.pending_frames:
            return

        seq, stamp = self.pending_frames.popleft()
        if not self.overwrite:
            if self.save_axis_sweep:
                if self._all_sweep_outputs_exist(seq):
                    return
            else:
                render_path = self._render_path_for_seq(seq)
                if render_path.exists():
                    return

        try:
            cam_pose = self._lookup_camera_pose(stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "TF lookup failed for camera pose: %s", exc)
            return

        try:
            self._update_robot_poses(stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Could not update robot collision poses: %s", exc)
            return

        T_optical_to_opengl = np.eye(4, dtype=np.float32)
        T_optical_to_opengl[:3, :3] = np.array([
            [1,  0,  0],
            [0, 1,  0],
            [0,  0, 1],
        ], dtype=np.float32)

        theta_x = np.deg2rad(0.0)
        T_rot_x = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x), 0.0],
            [0.0, np.sin(theta_x),  np.cos(theta_x), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        T_base_correction = T_optical_to_opengl @ T_rot_x

        if self.save_axis_sweep:
            saved = 0
            for axis in ("x", "y", "z"):
                for angle_deg in self._sweep_angles():
                    render_path = self._sweep_render_path_for_seq(seq, axis, angle_deg)
                    if render_path.exists() and not self.overwrite:
                        continue

                    T_sweep = self._rotation_4x4(axis, angle_deg)
                    cam_pose_sweep = cam_pose @ T_base_correction @ T_sweep
                    self.scene.set_pose(self.camera_node, pose=cam_pose_sweep)

                    if self._save_color_render(render_path):
                        saved += 1

            if saved > 0:
                rospy.loginfo("Saved %d sweep renders for seq %05d", saved, seq)
            return

        cam_pose_single = cam_pose @ T_base_correction
        self.scene.set_pose(self.camera_node, pose=cam_pose_single)
        render_path = self._render_path_for_seq(seq)
        if self._save_color_render(render_path):
            rospy.loginfo_throttle(1.0, "Saved render %s", render_path.name)



    def spin(self, rate_hz: float = 30.0) -> None:
        rate = rospy.Rate(rate_hz)
        while not rospy.is_shutdown():
            self.process_once()
            rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and save robot images using converted STL meshes.")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--stl-dir", required=True)
    parser.add_argument("--base-frame", default="dynaarm_arm_tf/world")
    parser.add_argument("--camera-frame", default="dynaarm_arm_tf/camera_link_optical")
    parser.add_argument("--image-topic", default="/dynaarm_arm/dynaarm_arm/camera1/image_raw")
    parser.add_argument("--info-topic", default="/dynaarm_arm/dynaarm_arm/camera1/camera_info")
    parser.add_argument("--camera-name", default="arm")
    parser.add_argument("--link-frame-prefix", default="dynaarm_arm_tf")
    parser.add_argument("--pkg", action="append", default=[], help="Package mapping: name=/abs/path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tf-timeout", type=float, default=0.1)
    parser.add_argument("--queue-size", type=int, default=200)
    parser.add_argument("--loop-rate", type=float, default=30.0)
    parser.add_argument("--sweep-step-deg", type=float, default=10.0)
    parser.add_argument("--sweep-start-deg", type=float, default=-180.0)
    parser.add_argument("--sweep-end-deg", type=float, default=180.0)
    parser.add_argument("--no-axis-sweep", action="store_true", help="Disable per-axis rotation sweep renders")
    return parser.parse_args(rospy.myargv()[1:])


def parse_package_map(items) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --pkg entry '{item}', expected name=/abs/path")
        name, path = item.split("=", 1)
        result[name] = path
    return result


def main() -> None:
    rospy.init_node("robot_mask_saver_stl", anonymous=False)
    args = parse_args()
    package_map = parse_package_map(args.pkg)

    saver = RobotMaskSaver(
        urdf_path=args.urdf,
        dataset_dir=args.dataset_dir,
        stl_dir=args.stl_dir,
        base_frame=args.base_frame,
        camera_frame=args.camera_frame,
        image_topic=args.image_topic,
        info_topic=args.info_topic,
        camera_name=args.camera_name,
        package_map=package_map,
        tf_timeout=args.tf_timeout,
        overwrite=args.overwrite,
        link_frame_prefix=args.link_frame_prefix,
        sweep_step_deg=args.sweep_step_deg,
        sweep_start_deg=args.sweep_start_deg,
        sweep_end_deg=args.sweep_end_deg,
        save_axis_sweep=not args.no_axis_sweep,
        queue_size=args.queue_size,
    )

    rospy.loginfo("robot_mask_saver_stl is running")
    saver.spin(rate_hz=args.loop_rate)


if __name__ == "__main__":
    main()