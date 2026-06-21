#!/usr/bin/env python3
# Live ROS mask + RGB/depth/pose helpers shared by the dynamic_gs2 publisher.
#
# This module is the LIVE half of the old `save_data_img_depth_mask_pose.py` recorder: the topic/path
# constants, the small ROS<->numpy converters, and `RobotMaskGenerator` (the per-frame robot-exclusion
# silhouette renderer). The standalone `CaptureSession` recorder + `main()` from the old file are NOT
# carried over — dynamic_gs2 captures via `static_capture.StaticRecorder` + the live publisher's SHM
# write, never this script's `main()`.
#
# The live publisher (`dynamic_gs/utils/live_ros_publisher.py`) importlib-loads this module by PATH and
# pulls the symbols below; it runs in the minimal `dynamic_gs_ros` py3.8 env, so keep the imports to
# what that env provides (ROS bindings + cv2/numpy/pyrender/trimesh/urdfpy).
from __future__ import annotations

from bisect import bisect_left
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import re
import tempfile

import cv2
import numpy as np
import pyrender
import rospy
from sensor_msgs.msg import Image
from tf.transformations import quaternion_matrix
import trimesh
from urdfpy import URDF

IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
DEPTH_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw"
CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
GAZEBO_JOINT_STATES_TOPIC = "/dynaarm_arm/joint_states_full"
GAZEBO_CAMERA_POSE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose"

WORLD_FRAME = "dynaarm_arm_tf/world"
CAMERA_POSE_SAVE_FRAME = "dynaarm_arm_tf/camera_link_optical"
MASK_RENDER_CAMERA_FRAME = "dynaarm_arm_tf/camera_pose_link"

URDF_PATH = Path(
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
    "active_camera_arm_examples/dynaarm_description/urdf/dynamic_gaussian_splat/"
    "dynaarm_with_gripper_for_gazebo_only_no_wrist_collision.urdf"
)
STL_DIR = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/stl")
WORLD_FILE = Path(
    "/home/mrc-cuhk/dev/teleop/catkin_ws/src/active_camera_arm_control/"
    "active_camera_arm_gazebo/worlds/dynamic_gaussian_splat/empty_world.world"
)
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

SYNC_QUEUE_SIZE = 20
SYNC_SLOP_SEC = 0.1
IMAGE_NAME_PREFIX = "arm"

INIT_CLOUD_NAME = "depth_camera_init_points.ply"
MAX_INIT_CLOUD_POINTS = 600000
BACKGROUND_COLOR_THRESHOLD = 10.0
MASK_KEEP_ERODE_RADIUS_PX = 4
MASK_MIN_KEEP_COMPONENT_AREA_PX = 64
TIME_EPS_SEC = 1e-6

# ROS optical / OpenCV camera frame:
#   +X right, +Y down, +Z forward
#
# Nerfstudio / OpenGL camera frame:
#   +X right, +Y up, +Z back
ROS_OPTICAL_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "bgr8":
        image = data.reshape(msg.height, msg.step)[:, : msg.width * 3]
        return image.reshape(msg.height, msg.width, 3)

    if msg.encoding == "rgb8":
        image = data.reshape(msg.height, msg.step)[:, : msg.width * 3]
        image = image.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(image[:, :, ::-1])

    if msg.encoding == "mono8":
        image = data.reshape(msg.height, msg.step)[:, : msg.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def ros_depth_to_uint16_mm(msg: Image) -> np.ndarray:
    if msg.encoding == "32FC1":
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape(msg.height, msg.step // 4)[:, : msg.width]
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        depth_mm = np.round(depth * 1000.0)
        depth_mm[invalid] = 0.0
        return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

    if msg.encoding in {"16UC1", "mono16"}:
        depth = np.frombuffer(msg.data, dtype=np.uint16)
        depth = depth.reshape(msg.height, msg.step // 2)[:, : msg.width]
        return depth.copy()

    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def rotate_camera_frame_only(transform_ros: np.ndarray) -> np.ndarray:
    transform_output = transform_ros.copy()
    transform_output[:3, :3] = transform_ros[:3, :3] @ ROS_OPTICAL_TO_NERFSTUDIO
    transform_output[:3, 3] = transform_ros[:3, 3]
    return transform_output


def pose_msg_to_matrix(pose_msg) -> np.ndarray:
    rotation = pose_msg.orientation
    transform = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
    transform[0, 3] = pose_msg.position.x
    transform[1, 3] = pose_msg.position.y
    transform[2, 3] = pose_msg.position.z
    return transform


def compose_transform_matrix(translation_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    transform = quaternion_matrix(quaternion_xyzw)
    transform[0, 3] = float(translation_xyz[0])
    transform[1, 3] = float(translation_xyz[1])
    transform[2, 3] = float(translation_xyz[2])
    return transform


def normalize_frame_id(frame_id: str | None) -> str:
    return (frame_id or "").strip().lstrip("/")


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def load_saved_depth_mm(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image from {depth_path}")
    if depth.ndim != 2:
        raise RuntimeError(f"Depth image must be HxW, got {depth.shape} for {depth_path}")

    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32) * 1000.0
    if depth.dtype == np.uint16:
        return depth.astype(np.float32)
    raise RuntimeError(f"Unsupported depth dtype {depth.dtype} for {depth_path}")


def load_saved_mask(mask_path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask from {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != expected_hw:
        raise RuntimeError(
            f"Mask shape mismatch for {mask_path}: got {mask.shape}, expected {expected_hw}"
        )
    return mask > 0


def load_saved_rgb(rgb_path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read RGB image from {rgb_path}")
    if image_bgr.shape[:2] != expected_hw:
        raise RuntimeError(
            f"RGB shape mismatch for {rgb_path}: got {image_bgr.shape[:2]}, expected {expected_hw}"
        )
    return image_bgr


def distribute_point_budget_evenly(capacities: list[int], total_budget: int) -> list[int]:
    capped_budget = min(max(0, int(total_budget)), int(sum(capacities)))
    quotas = [0] * len(capacities)
    remaining = [idx for idx, capacity in enumerate(capacities) if capacity > 0]
    remaining_budget = capped_budget

    while remaining and remaining_budget > 0:
        share, remainder = divmod(remaining_budget, len(remaining))
        allocated_this_round = 0
        next_remaining = []

        for order, idx in enumerate(remaining):
            target = share + (1 if order < remainder else 0)
            if target <= 0:
                next_remaining.append(idx)
                continue

            available = capacities[idx] - quotas[idx]
            allocation = min(available, target)
            quotas[idx] += allocation
            allocated_this_round += allocation
            if quotas[idx] < capacities[idx]:
                next_remaining.append(idx)

        if allocated_this_round == 0:
            break

        remaining_budget -= allocated_this_round
        remaining = next_remaining

    return quotas


class RobotMaskGenerator:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        joint_state_times_sec: list[float],
        joint_state_positions: list[dict[str, float]],
    ) -> None:
        self.intrinsics = intrinsics
        self.joint_state_times_sec = joint_state_times_sec
        self.joint_state_positions = joint_state_positions

        self.renderer = None
        self.scene = None
        self.camera_node = None
        self.robot_nodes: list[tuple[str, object, object]] = []
        self.mesh_cache: dict[str, trimesh.Trimesh] = {}
        self.temp_urdf_path = None

        rospy.loginfo("Loading URDF from %s", URDF_PATH)
        self.temp_urdf_path = self._make_temp_resolved_urdf(URDF_PATH, PACKAGE_MAP, STL_DIR)
        self.robot = URDF.load(self.temp_urdf_path)
        self.zero_link_fk_by_name = self.robot.link_fk(use_names=True)
        self.actuated_joint_names = set(self.robot.actuated_joint_names)
        self.frame_prefix = f"{normalize_frame_id(WORLD_FRAME).rsplit('/', 1)[0]}/"
        self.background_rgb_colors = self._load_background_rgb_colors()

    def cleanup(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.delete()
            except Exception as exc:
                rospy.logwarn("Renderer cleanup failed: %s", exc)
        self.renderer = None
        self.scene = None
        self.camera_node = None
        self.robot_nodes = []

        if self.temp_urdf_path is not None:
            try:
                temp_urdf = Path(self.temp_urdf_path)
                if temp_urdf.exists():
                    temp_urdf.unlink()
            except Exception:
                pass
            self.temp_urdf_path = None

    def _make_temp_resolved_urdf(
        self,
        urdf_path: Path,
        package_map: dict[str, str],
        stl_dir: Path,
    ) -> str:
        text = urdf_path.read_text()

        def repl(match):
            pkg = match.group(1)
            rest = match.group(2)

            basename = Path(rest).stem + ".stl"
            stl_path = stl_dir / basename
            if stl_path.exists():
                return str(stl_path)
            if pkg not in package_map:
                raise RuntimeError(f"Missing package root for '{pkg}'")
            return str(Path(package_map[pkg]) / rest)

        text = re.sub(r"package://([^/]+)/([^\"'<> ]+)", repl, text)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tmp.write(text)
        tmp.flush()
        tmp.close()
        return tmp.name

    def _ensure_renderer(self, scale: float | None = None) -> None:
        if self.renderer is not None and scale is None:
            return
        if scale is not None:
            # Sweep mode rebuild: tear down the existing renderer so we rebuild at the new scale.
            if self.renderer is not None:
                try:
                    self.renderer.delete()
                except Exception:
                    pass
                self.renderer = None
                self.scene = None
                self.camera_node = None
                self.robot_nodes = []

        # Quarter-resolution render: pyrender is the publisher's single biggest per-frame cost
        # (GL render ~53 ms @1920x1200, full mask ~66 ms). The robot-exclusion keep-mask is a coarse
        # silhouette, so rendering small and NEAREST-upscaling the binary result back to full res is
        # visually identical for the gripper cut-out. A live scale sweep (n=58/scale, 3 cycles) found
        # mask render time is fill-bound DOWN TO ~480x300 and then hits a ~14-15 ms FLOOR (fixed cost:
        # URDF FK + per-mesh pyrender pose updates + GL setup + the full-res upscale) — 0.25, 0.20, and
        # 0.15 all measure ~14-17 ms, so going below 0.25 buys nothing but jaggier edges. 0.25 sits at
        # the floor (median ~15 ms, inside the 20 ms budget) with a clean silhouette. The full output
        # shape (intrinsics.width x height) is unchanged — both call sites get full res.
        # DGS_MASK_RENDER_SCALE=1.0 restores full-res rendering.
        if scale is not None:
            self._render_scale = scale
        else:
            try:
                self._render_scale = float(os.environ.get("DGS_MASK_RENDER_SCALE", "0.25"))
            except ValueError:
                self._render_scale = 0.25
        self._render_scale = min(max(self._render_scale, 0.1), 1.0)
        self._render_w = max(1, int(round(self.intrinsics.width * self._render_scale)))
        self._render_h = max(1, int(round(self.intrinsics.height * self._render_scale)))
        # Actual achieved scale per axis (integer rounding may differ slightly from the request).
        sx = self._render_w / float(self.intrinsics.width)
        sy = self._render_h / float(self.intrinsics.height)

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self._render_w,
            viewport_height=self._render_h,
        )
        self.scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([0.8, 0.8, 0.8]),
        )
        camera = pyrender.IntrinsicsCamera(
            fx=self.intrinsics.fx * sx,
            fy=self.intrinsics.fy * sy,
            cx=self.intrinsics.cx * sx,
            cy=self.intrinsics.cy * sy,
            znear=0.001,
            zfar=100.0,
        )
        self.camera_node = self.scene.add(camera, pose=np.eye(4, dtype=np.float32))
        self.scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
            pose=np.eye(4, dtype=np.float32),
        )
        light_pose = np.eye(4, dtype=np.float32)
        light_pose[:3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add(pyrender.PointLight(color=np.ones(3), intensity=20.0), pose=light_pose)
        self._build_scene()
        rospy.loginfo("Mask renderer initialized")

    def _build_scene(self) -> None:
        assert self.scene is not None
        for link in self.robot.links:
            for visual in link.visuals:
                tri = self._geometry_to_trimesh(visual.geometry)
                if tri is None:
                    rospy.logwarn("Skipping unsupported visual geometry on link '%s'", link.name)
                    continue
                pose = np.eye(4, dtype=np.float32)
                if visual.origin is not None:
                    pose = visual.origin.astype(np.float32)
                node = self.scene.add(self._make_render_mesh(tri), pose=pose)
                self.robot_nodes.append((link.name, visual, node))

    def _geometry_to_trimesh(self, geom):
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

        if hasattr(inner, "filename") and inner.filename is not None:
            scale = np.array(inner.scale, dtype=np.float32) if getattr(inner, "scale", None) is not None else None
            return self._load_trimesh(inner.filename, scale)

        if hasattr(inner, "size") and inner.size is not None:
            return trimesh.creation.box(extents=np.array(inner.size, dtype=np.float32))

        if hasattr(inner, "radius") and hasattr(inner, "length"):
            radius = getattr(inner, "radius", None)
            length = getattr(inner, "length", None)
            if radius is not None and length is not None:
                return trimesh.creation.cylinder(radius=float(radius), height=float(length), sections=32)

        if hasattr(inner, "radius") and not hasattr(inner, "length"):
            radius = getattr(inner, "radius", None)
            if radius is not None:
                return trimesh.creation.icosphere(radius=float(radius), subdivisions=2)

        return None

    def _load_trimesh(self, path: str, scale: np.ndarray | None) -> trimesh.Trimesh:
        key = f"{path}|{None if scale is None else tuple(scale.tolist())}"
        if key in self.mesh_cache:
            return self.mesh_cache[key].copy()

        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = [geometry.copy() for geometry in loaded.geometry.values()]
            if not meshes:
                raise RuntimeError(f"No geometry found in mesh file: {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded.copy()

        if scale is not None:
            mesh.apply_scale(scale)

        self.mesh_cache[key] = mesh.copy()
        return mesh

    def _make_render_mesh(self, mesh: trimesh.Trimesh) -> pyrender.Mesh:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.9, 0.2, 0.2, 1.0),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
        )
        return pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material)

    def _frame_to_link_name(self, frame_id: str | None) -> str | None:
        normalized = normalize_frame_id(frame_id)
        if not normalized.startswith(self.frame_prefix):
            return None
        return normalized[len(self.frame_prefix):]

    def _invert_rigid_transform(self, transform: np.ndarray) -> np.ndarray:
        transform_inv = np.eye(4, dtype=np.float32)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        transform_inv[:3, :3] = rotation.T
        transform_inv[:3, 3] = -rotation.T @ translation
        return transform_inv

    def _static_link_offset(self, source_frame: str, target_frame: str) -> np.ndarray:
        source_link = self._frame_to_link_name(source_frame)
        target_link = self._frame_to_link_name(target_frame)
        if source_link is None or target_link is None:
            raise RuntimeError(f"Unable to resolve link names from '{source_frame}' -> '{target_frame}'")

        source_pose = self.zero_link_fk_by_name.get(source_link)
        target_pose = self.zero_link_fk_by_name.get(target_link)
        if source_pose is None or target_pose is None:
            raise RuntimeError(f"Missing URDF FK pose for '{source_link}' or '{target_link}'")
        return self._invert_rigid_transform(source_pose.astype(np.float32)) @ target_pose.astype(np.float32)

    def _sample_joint_positions(self, stamp: rospy.Time) -> dict[str, float]:
        if not self.joint_state_times_sec:
            raise RuntimeError("No Gazebo joint state samples were received")

        query_time_sec = float(stamp.to_sec())
        insert_at = bisect_left(self.joint_state_times_sec, query_time_sec)

        if insert_at < len(self.joint_state_times_sec) and abs(self.joint_state_times_sec[insert_at] - query_time_sec) <= TIME_EPS_SEC:
            sample = self.joint_state_positions[insert_at]
            return {name: value for name, value in sample.items() if name in self.actuated_joint_names}
        if insert_at > 0 and abs(self.joint_state_times_sec[insert_at - 1] - query_time_sec) <= TIME_EPS_SEC:
            sample = self.joint_state_positions[insert_at - 1]
            return {name: value for name, value in sample.items() if name in self.actuated_joint_names}

        prev_idx = insert_at - 1 if insert_at > 0 else None
        next_idx = insert_at if insert_at < len(self.joint_state_times_sec) else None

        if prev_idx is not None and next_idx is not None:
            prev_time_sec = self.joint_state_times_sec[prev_idx]
            next_time_sec = self.joint_state_times_sec[next_idx]
            alpha = (query_time_sec - prev_time_sec) / (next_time_sec - prev_time_sec)
            prev_positions = self.joint_state_positions[prev_idx]
            next_positions = self.joint_state_positions[next_idx]
            joint_names = (set(prev_positions.keys()) | set(next_positions.keys())) & self.actuated_joint_names

            interpolated = {}
            for joint_name in joint_names:
                prev_value = prev_positions.get(joint_name)
                next_value = next_positions.get(joint_name)
                if prev_value is None:
                    interpolated[joint_name] = float(next_value)
                    continue
                if next_value is None:
                    interpolated[joint_name] = float(prev_value)
                    continue
                interpolated[joint_name] = float(prev_value * (1.0 - alpha) + next_value * alpha)
            return interpolated

        sample_idx = prev_idx if prev_idx is not None else next_idx
        if sample_idx is None:
            raise RuntimeError("No Gazebo joint state samples are available")
        sample = self.joint_state_positions[sample_idx]
        return {name: value for name, value in sample.items() if name in self.actuated_joint_names}

    def _camera_pose_from_link_fk(self, link_fk: dict[str, np.ndarray], camera_frame: str) -> np.ndarray:
        resolved_camera_frame = normalize_frame_id(camera_frame) or MASK_RENDER_CAMERA_FRAME
        link_name = self._frame_to_link_name(resolved_camera_frame)
        if link_name is not None and link_name in link_fk:
            return link_fk[link_name].astype(np.float32)

        default_link_name = self._frame_to_link_name(MASK_RENDER_CAMERA_FRAME)
        if default_link_name is None or default_link_name not in link_fk:
            raise RuntimeError(f"Camera link '{MASK_RENDER_CAMERA_FRAME}' is missing from FK results")

        default_pose = link_fk[default_link_name].astype(np.float32)
        return default_pose @ self._static_link_offset(MASK_RENDER_CAMERA_FRAME, resolved_camera_frame)

    def _update_robot_poses(self, link_fk: dict[str, np.ndarray]) -> None:
        assert self.scene is not None
        for link_name, visual, node in self.robot_nodes:
            base_to_link = link_fk.get(link_name)
            if base_to_link is None:
                rospy.logwarn_throttle(2.0, "Skipping link without FK pose: %s", link_name)
                continue
            link_to_visual = np.eye(4, dtype=np.float32)
            if visual.origin is not None:
                link_to_visual = visual.origin.astype(np.float32)
            self.scene.set_pose(node, pose=base_to_link.astype(np.float32) @ link_to_visual)

    def _build_render_camera_pose(self, ros_pose: np.ndarray) -> np.ndarray:
        optical_to_opengl = np.eye(4, dtype=np.float32)
        optical_to_opengl[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )
        rot_y_m90 = np.eye(4, dtype=np.float32)
        rot_y_m90[:3, :3] = np.array(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )
        rot_z_90 = np.eye(4, dtype=np.float32)
        rot_z_90[:3, :3] = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return ros_pose @ optical_to_opengl @ rot_y_m90 @ rot_z_90

    def _render_robot_exclusion_mask(self, stamp: rospy.Time, camera_frame: str) -> np.ndarray:
        # Scale-sweep diagnostic: DGS_MASK_SCALE_SWEEP="0.5,0.35,0.25,0.2" cycles render scales,
        # rebuilding the renderer every DGS_MASK_SWEEP_EVERY frames (default 40), so one live run
        # measures mask render time vs resolution. Off unless the env is set.
        if not hasattr(self, "_sweep_scales"):
            raw = os.environ.get("DGS_MASK_SCALE_SWEEP", "").strip()
            self._sweep_scales = [float(x) for x in raw.split(",") if x] if raw else []
            self._sweep_every = int(os.environ.get("DGS_MASK_SWEEP_EVERY", "60"))
            self._sweep_n = 0
            self._sweep_idx = -1
            self._sweep_t = []        # per-render ms for the current window
            self._sweep_t_render = []  # GL render() only, isolates fill from FK/upscale overhead
        sweep = bool(self._sweep_scales)
        if sweep:
            phase = self._sweep_n // self._sweep_every
            idx = phase % len(self._sweep_scales)
            if idx != self._sweep_idx:
                # flush the window we just finished
                if self._sweep_t:
                    a = sorted(self._sweep_t[2:]); g = sorted(self._sweep_t_render[2:])  # drop 2 warmups
                    if a:
                        rospy.loginfo("[mask-sweep] scale=%.2f %dx%d  total med=%.1f p90=%.1f  "
                                      "GLrender med=%.1f  n=%d",
                                      self._render_scale, self._render_w, self._render_h,
                                      a[len(a)//2], a[min(len(a)-1, int(len(a)*0.9))],
                                      g[len(g)//2], len(a))
                self._sweep_idx = idx
                self._sweep_t = []; self._sweep_t_render = []
                self._ensure_renderer(scale=self._sweep_scales[idx])
            self._sweep_n += 1
        else:
            self._ensure_renderer()
        assert self.renderer is not None
        assert self.scene is not None
        assert self.camera_node is not None

        _t0 = time.time() if sweep else 0.0   # timing only when the diagnostic sweep is on
        sampled_joint_positions = self._sample_joint_positions(stamp)
        link_fk = self.robot.link_fk(cfg=sampled_joint_positions, use_names=True)
        camera_pose = self._camera_pose_from_link_fk(link_fk, camera_frame)
        self._update_robot_poses(link_fk)
        self.scene.set_pose(self.camera_node, pose=self._build_render_camera_pose(camera_pose))
        _tr = time.time() if sweep else 0.0
        _, depth = self.renderer.render(self.scene)
        _gl = (time.time() - _tr) * 1000.0 if sweep else 0.0

        # 0 where the robot is rendered, 255 elsewhere (rendered at self._render_w x _render_h).
        keep = cv2.flip((depth == 0).astype(np.uint8) * 255, 0)
        if keep.shape[1] != self.intrinsics.width or keep.shape[0] != self.intrinsics.height:
            keep = cv2.resize(
                keep,
                (self.intrinsics.width, self.intrinsics.height),
                interpolation=cv2.INTER_NEAREST,
            )
        if sweep:
            self._sweep_t.append((time.time() - _t0) * 1000.0)
            self._sweep_t_render.append(_gl)
        return keep

    def _compute_background_keep_mask(self, rgb_path: Path) -> np.ndarray | None:
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image from {rgb_path}")

        if not self.background_rgb_colors:
            return None

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        background_like = np.zeros(rgb.shape[:2], dtype=bool)
        for color_rgb in self.background_rgb_colors:
            diff = rgb - color_rgb[None, None, :]
            color_dist = np.linalg.norm(diff, axis=2)
            background_like |= color_dist <= BACKGROUND_COLOR_THRESHOLD
        return (~background_like).astype(np.uint8) * 255

    def _parse_rgba_text_to_rgb255(self, text: str | None) -> np.ndarray | None:
        if text is None:
            return None
        parts = [part for part in text.strip().split() if part]
        if len(parts) < 3:
            return None
        try:
            rgb = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
        except ValueError:
            return None
        if np.any(rgb > 1.0):
            return np.clip(rgb, 0.0, 255.0).astype(np.float32)
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.float32)

    def _load_background_rgb_colors(self) -> list[np.ndarray]:
        colors = []
        if not WORLD_FILE.exists():
            rospy.logwarn("World file not found for background masking: %s", WORLD_FILE)
            return colors

        try:
            root = ET.parse(WORLD_FILE).getroot()
        except Exception as exc:
            rospy.logwarn("Failed to parse world file %s: %s", WORLD_FILE, exc)
            return colors

        for scene in root.findall(".//scene"):
            color = self._parse_rgba_text_to_rgb255(scene.findtext("background"))
            if color is not None:
                colors.append(color)

        for model in root.findall(".//model"):
            name = (model.get("name") or "").lower()
            if not any(token in name for token in ("wall", "floor", "background")):
                continue
            for material in model.findall(".//material"):
                color = self._parse_rgba_text_to_rgb255(material.findtext("emissive"))
                if color is not None:
                    colors.append(color)

        deduped = []
        seen = set()
        for color in colors:
            key = tuple(int(round(value)) for value in color.tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(np.array(key, dtype=np.float32))
        return deduped

    def _refine_keep_mask(self, mask: np.ndarray) -> np.ndarray:
        refined = mask.copy()

        if MASK_KEEP_ERODE_RADIUS_PX > 0:
            kernel_size = MASK_KEEP_ERODE_RADIUS_PX * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            refined = cv2.erode(refined, kernel, iterations=1)

        if MASK_MIN_KEEP_COMPONENT_AREA_PX > 0:
            keep_binary = (refined > 0).astype(np.uint8)
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(keep_binary, connectivity=8)
            cleaned = np.zeros_like(refined)
            for component_idx in range(1, component_count):
                area = stats[component_idx, cv2.CC_STAT_AREA]
                if area < MASK_MIN_KEEP_COMPONENT_AREA_PX:
                    continue
                cleaned[labels == component_idx] = 255
            refined = cleaned

        return refined

    def save_mask(self, stamp: rospy.Time, rgb_path: Path, mask_path: Path) -> None:
        robot_exclusion_mask = self._render_robot_exclusion_mask(stamp, MASK_RENDER_CAMERA_FRAME)
        # Background masking disabled: keep the full background and only exclude the robot.
        keep_mask = robot_exclusion_mask
        keep_mask = self._refine_keep_mask(keep_mask)

        if not cv2.imwrite(str(mask_path), keep_mask):
            raise RuntimeError(f"Failed to save mask to {mask_path}")
