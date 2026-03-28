#!/usr/bin/env python3
from __future__ import annotations

from bisect import bisect_left
import datetime
import json
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import pyrender
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp
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

DATASET_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/"
    "datasets/dynaarm_gs_depth_mask_01"
)
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

SAVE_HZ = 5.0
MAX_IMAGES = 60
SYNC_QUEUE_SIZE = 20
SYNC_SLOP_SEC = 0.1
IMAGE_NAME_PREFIX = "arm"

INIT_CLOUD_NAME = "depth_camera_init_points.ply"
MAX_INIT_CLOUD_POINTS = 300000
BACKGROUND_COLOR_THRESHOLD = 10.0
MASK_KEEP_ERODE_RADIUS_PX = 4
MASK_MIN_KEEP_COMPONENT_AREA_PX = 64
TIME_EPS_SEC = 1e-6
MAX_GAZEBO_POSE_HISTORY_SEC = 30.0

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


@dataclass
class SavedFrameRecord:
    seq: int
    stamp: rospy.Time
    frame: dict


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


def resolve_relpath(base_dir: Path, rel_path: str) -> Path:
    return (base_dir / rel_path).resolve()


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


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


def delete_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        rospy.logwarn("Failed to delete %s: %s", path, exc)


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

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.intrinsics.width,
            viewport_height=self.intrinsics.height,
        )
        self.scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([0.8, 0.8, 0.8]),
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
        self._ensure_renderer()
        assert self.renderer is not None
        assert self.scene is not None
        assert self.camera_node is not None

        sampled_joint_positions = self._sample_joint_positions(stamp)
        link_fk = self.robot.link_fk(cfg=sampled_joint_positions, use_names=True)
        camera_pose = self._camera_pose_from_link_fk(link_fk, camera_frame)
        self._update_robot_poses(link_fk)
        self.scene.set_pose(self.camera_node, pose=self._build_render_camera_pose(camera_pose))
        _, depth = self.renderer.render(self.scene)

        # 0 where the robot is rendered, 255 elsewhere.
        return cv2.flip((depth == 0).astype(np.uint8) * 255, 0)

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
        background_keep_mask = self._compute_background_keep_mask(rgb_path)
        if background_keep_mask is None:
            keep_mask = robot_exclusion_mask
        else:
            keep_mask = cv2.bitwise_and(robot_exclusion_mask, background_keep_mask)
        keep_mask = self._refine_keep_mask(keep_mask)

        if not cv2.imwrite(str(mask_path), keep_mask):
            raise RuntimeError(f"Failed to save mask to {mask_path}")


class CaptureSession:
    def __init__(self) -> None:
        self.intrinsics = None
        self.robot_model = None

        self.run_dir = None
        self.rgb_dir = None
        self.depth_dir = None
        self.masks_dir = None
        self.transforms_path = None
        self.ply_path = None

        self.metadata = None
        self.saved_records: list[SavedFrameRecord] = []

        self.frame_index = 0
        self.last_saved_stamp = None
        self.warned_unexpected_image_frame = False

        self.joint_state_times_sec: list[float] = []
        self.joint_state_positions: list[dict[str, float]] = []
        self.gazebo_pose_times_sec: list[float] = []
        self.gazebo_pose_matrices: list[np.ndarray] = []

    def initialize(self) -> None:
        camera_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=5.0)
        self.intrinsics = CameraIntrinsics(
            width=int(camera_info.width),
            height=int(camera_info.height),
            fx=float(camera_info.K[0]),
            fy=float(camera_info.K[4]),
            cx=float(camera_info.K[2]),
            cy=float(camera_info.K[5]),
        )

        self.run_dir = self._make_unique_run_dir()
        self.rgb_dir = self.run_dir / "rgb"
        self.depth_dir = self.run_dir / "depth"
        self.masks_dir = self.run_dir / "masks"
        self.transforms_path = self.run_dir / "transforms.json"
        self.ply_path = self.run_dir / INIT_CLOUD_NAME

        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)
        self.masks_dir.mkdir(parents=True, exist_ok=False)

        self.metadata = {
            "fl_x": self.intrinsics.fx,
            "fl_y": self.intrinsics.fy,
            "cx": self.intrinsics.cx,
            "cy": self.intrinsics.cy,
            "w": self.intrinsics.width,
            "h": self.intrinsics.height,
            "frames": [],
        }
        self.robot_model = RobotMaskGenerator(
            intrinsics=self.intrinsics,
            joint_state_times_sec=self.joint_state_times_sec,
            joint_state_positions=self.joint_state_positions,
        )
        self.write_transforms()
        rospy.loginfo("Saving dataset to %s", self.run_dir)

    def _make_unique_run_dir(self) -> Path:
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        base_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = DATASET_ROOT / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = DATASET_ROOT / f"{base_name}_{suffix:02d}"
        return candidate

    def joint_state_callback(self, msg: JointState) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0 or not msg.name or not msg.position:
            return

        positions = {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
        }
        if not positions:
            return

        insert_at = bisect_left(self.joint_state_times_sec, stamp_sec)
        if insert_at < len(self.joint_state_times_sec) and abs(self.joint_state_times_sec[insert_at] - stamp_sec) <= TIME_EPS_SEC:
            self.joint_state_positions[insert_at] = positions
        else:
            self.joint_state_times_sec.insert(insert_at, stamp_sec)
            self.joint_state_positions.insert(insert_at, positions)

    def gazebo_pose_callback(self, msg: PoseStamped) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0:
            return

        pose_matrix = pose_msg_to_matrix(msg.pose).astype(np.float64)
        insert_at = bisect_left(self.gazebo_pose_times_sec, stamp_sec)
        if (
            insert_at < len(self.gazebo_pose_times_sec)
            and abs(self.gazebo_pose_times_sec[insert_at] - stamp_sec) <= TIME_EPS_SEC
        ):
            self.gazebo_pose_matrices[insert_at] = pose_matrix
        else:
            self.gazebo_pose_times_sec.insert(insert_at, stamp_sec)
            self.gazebo_pose_matrices.insert(insert_at, pose_matrix)

        newest_time_sec = self.gazebo_pose_times_sec[-1]
        while (
            len(self.gazebo_pose_times_sec) > 1
            and newest_time_sec - self.gazebo_pose_times_sec[0] > MAX_GAZEBO_POSE_HISTORY_SEC
        ):
            self.gazebo_pose_times_sec.pop(0)
            self.gazebo_pose_matrices.pop(0)

    def write_transforms(self) -> None:
        if self.metadata is None or self.transforms_path is None:
            return
        write_json_atomic(self.transforms_path, self.metadata)

    def _warn_if_unexpected_image_frame(self, frame_id: str) -> None:
        normalized = normalize_frame_id(frame_id)
        if normalized == CAMERA_POSE_SAVE_FRAME or self.warned_unexpected_image_frame:
            return
        rospy.logwarn(
            "Image header frame_id is '%s' but poses are saved from '%s'.",
            normalized,
            CAMERA_POSE_SAVE_FRAME,
        )
        self.warned_unexpected_image_frame = True

    def _lookup_pose_matrix(self, stamp: rospy.Time) -> np.ndarray:
        if self.robot_model is None:
            raise RuntimeError("Gazebo pose model is not initialized")

        if not self.gazebo_pose_times_sec:
            raise RuntimeError("No Gazebo camera pose samples were received")

        query_time_sec = float(stamp.to_sec())
        insert_at = bisect_left(self.gazebo_pose_times_sec, query_time_sec)

        if (
            insert_at < len(self.gazebo_pose_times_sec)
            and abs(self.gazebo_pose_times_sec[insert_at] - query_time_sec) <= TIME_EPS_SEC
        ):
            pose_matrix = self.gazebo_pose_matrices[insert_at]
        elif (
            insert_at > 0
            and abs(self.gazebo_pose_times_sec[insert_at - 1] - query_time_sec) <= TIME_EPS_SEC
        ):
            pose_matrix = self.gazebo_pose_matrices[insert_at - 1]
        else:
            prev_idx = insert_at - 1 if insert_at > 0 else None
            next_idx = insert_at if insert_at < len(self.gazebo_pose_times_sec) else None

            if prev_idx is None and next_idx is None:
                raise RuntimeError("No Gazebo camera pose samples are available")
            if prev_idx is None:
                pose_matrix = self.gazebo_pose_matrices[next_idx]
            elif next_idx is None:
                pose_matrix = self.gazebo_pose_matrices[prev_idx]
            else:
                prev_time_sec = self.gazebo_pose_times_sec[prev_idx]
                next_time_sec = self.gazebo_pose_times_sec[next_idx]
                alpha = (query_time_sec - prev_time_sec) / (next_time_sec - prev_time_sec)
                prev_matrix = self.gazebo_pose_matrices[prev_idx]
                next_matrix = self.gazebo_pose_matrices[next_idx]
                prev_quat = quaternion_from_matrix(prev_matrix)
                next_quat = quaternion_from_matrix(next_matrix)
                interp_quat = quaternion_slerp(prev_quat, next_quat, alpha)
                interp_xyz = prev_matrix[:3, 3] * (1.0 - alpha) + next_matrix[:3, 3] * alpha
                pose_matrix = compose_transform_matrix(interp_xyz, interp_quat)

        optical_offset = self.robot_model._static_link_offset(
            MASK_RENDER_CAMERA_FRAME,
            CAMERA_POSE_SAVE_FRAME,
        ).astype(np.float64)
        return rotate_camera_frame_only(pose_matrix @ optical_offset)

    def image_callback(self, image_msg: Image, depth_msg: Image) -> None:
        if self.metadata is None or self.rgb_dir is None or self.depth_dir is None:
            return
        if self.frame_index >= MAX_IMAGES:
            return

        if self.last_saved_stamp is not None:
            dt = (image_msg.header.stamp - self.last_saved_stamp).to_sec()
            if dt < 1.0 / SAVE_HZ:
                return

        self._warn_if_unexpected_image_frame(image_msg.header.frame_id)

        try:
            transform_matrix = self._lookup_pose_matrix(image_msg.header.stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Skipping frame because Gazebo camera pose is unavailable: %s", exc)
            return

        image_bgr = ros_image_to_bgr(image_msg)
        depth_mm = ros_depth_to_uint16_mm(depth_msg)

        seq = int(image_msg.header.seq)
        file_stem = f"{IMAGE_NAME_PREFIX}_{seq:05d}"
        rgb_name = f"{file_stem}.png"
        depth_name = f"{file_stem}.tiff"

        rgb_path = self.rgb_dir / rgb_name
        depth_path = self.depth_dir / depth_name
        if not cv2.imwrite(str(rgb_path), image_bgr):
            raise RuntimeError(f"Failed to save RGB image to {rgb_path}")
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise RuntimeError(f"Failed to save depth image to {depth_path}")

        frame_entry = {
            "file_path": f"./rgb/{rgb_name}",
            "depth_file_path": f"./depth/{depth_name}",
            "transform_matrix": transform_matrix.tolist(),
        }
        self.metadata["frames"].append(frame_entry)
        self.saved_records.append(
            SavedFrameRecord(
                seq=seq,
                stamp=image_msg.header.stamp,
                frame=frame_entry,
            )
        )
        self.write_transforms()

        self.frame_index += 1
        self.last_saved_stamp = image_msg.header.stamp
        rospy.loginfo("Saved frame %d/%d", self.frame_index, MAX_IMAGES)

        if self.frame_index >= MAX_IMAGES:
            rospy.signal_shutdown(f"Saved {MAX_IMAGES} images")

    def _path_from_frame_entry(self, frame: dict, key: str) -> Path:
        assert self.run_dir is not None
        return resolve_relpath(self.run_dir.resolve(), frame[key])

    def _remove_failed_frame_files(self, frame: dict) -> None:
        for key in ("file_path", "depth_file_path", "mask_path"):
            rel = frame.get(key)
            if not rel:
                continue
            delete_if_exists(self._path_from_frame_entry(frame, key))

    def generate_masks(self) -> None:
        if self.metadata is None or self.masks_dir is None or self.robot_model is None:
            return
        if not self.saved_records:
            return

        kept_records = []
        kept_frames = []
        removed_count = 0

        for record in self.saved_records:
            rgb_name = Path(record.frame["file_path"]).name
            rgb_path = self._path_from_frame_entry(record.frame, "file_path")
            mask_path = self.masks_dir / rgb_name

            try:
                self.robot_model.save_mask(record.stamp, rgb_path, mask_path)
            except Exception as exc:
                removed_count += 1
                rospy.logwarn(
                    "Dropping frame %s (seq %d) because mask generation failed: %s",
                    rgb_name,
                    record.seq,
                    exc,
                )
                self._remove_failed_frame_files(record.frame)
                continue

            record.frame["mask_path"] = f"./masks/{rgb_name}"
            kept_records.append(record)
            kept_frames.append(record.frame)

        self.saved_records = kept_records
        self.metadata["frames"] = kept_frames
        self.frame_index = len(kept_records)
        self.write_transforms()

        if removed_count > 0:
            rospy.loginfo("Dropped %d frame(s) without valid masks", removed_count)

    def write_init_cloud_from_saved_frames(self) -> None:
        if self.metadata is None or self.run_dir is None or self.ply_path is None:
            return
        if not self.metadata["frames"]:
            return

        dataset_dir = self.run_dir.resolve()
        rng = np.random.default_rng(0)
        frame_infos = []
        valid_counts = []

        for frame in self.metadata["frames"]:
            if "mask_path" not in frame:
                continue

            depth_path = resolve_relpath(dataset_dir, frame["depth_file_path"])
            rgb_path = resolve_relpath(dataset_dir, frame["file_path"])
            mask_path = resolve_relpath(dataset_dir, frame["mask_path"])

            depth_mm = load_saved_depth_mm(depth_path)
            valid_mask = load_saved_mask(mask_path, depth_mm.shape)
            valid_count = int(np.count_nonzero(valid_mask & (depth_mm > 0.0)))
            if valid_count == 0:
                continue

            frame_infos.append(
                {
                    "frame": frame,
                    "depth_path": depth_path,
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,
                }
            )
            valid_counts.append(valid_count)

        if not frame_infos:
            raise RuntimeError("No valid depth points remained after applying masks across saved frames")

        quotas = distribute_point_budget_evenly(valid_counts, MAX_INIT_CLOUD_POINTS)
        all_xyz = []
        all_rgb = []

        for frame_info, n_sample in zip(frame_infos, quotas):
            if n_sample <= 0:
                continue

            frame = frame_info["frame"]
            depth_mm = load_saved_depth_mm(frame_info["depth_path"])
            valid_mask = load_saved_mask(frame_info["mask_path"], depth_mm.shape)
            rgb_bgr = load_saved_rgb(frame_info["rgb_path"], depth_mm.shape)

            ys, xs = np.where(valid_mask & (depth_mm > 0.0))
            if ys.size == 0:
                continue
            if n_sample < ys.size:
                choice = rng.choice(ys.size, size=n_sample, replace=False)
                ys = ys[choice]
                xs = xs[choice]

            depth_m = depth_mm[ys, xs] / 1000.0
            x = (xs.astype(np.float32) - self.intrinsics.cx) * depth_m / self.intrinsics.fx
            y = -(ys.astype(np.float32) - self.intrinsics.cy) * depth_m / self.intrinsics.fy
            xyz_cam = np.stack([x, y, -depth_m], axis=1)
            hom = np.concatenate(
                [xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            transform_matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
            xyz_world = (transform_matrix @ hom.T).T[:, :3]
            rgb = rgb_bgr[ys, xs][:, ::-1].astype(np.uint8)

            all_xyz.append(xyz_world.astype(np.float32))
            all_rgb.append(rgb)

        if not all_xyz:
            raise RuntimeError("No valid depth points remained after applying masks across saved frames")

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        write_ascii_ply(self.ply_path, xyz, rgb)
        self.metadata["ply_file_path"] = INIT_CLOUD_NAME
        self.write_transforms()

    def cleanup(self) -> None:
        if self.robot_model is not None:
            self.robot_model.cleanup()
            self.robot_model = None


def main() -> None:
    rospy.init_node("save_dynaarm_camera1_rgb_tf")
    session = CaptureSession()

    try:
        session.initialize()

        joint_state_sub = rospy.Subscriber(
            GAZEBO_JOINT_STATES_TOPIC,
            JointState,
            session.joint_state_callback,
            queue_size=200,
        )
        gazebo_pose_sub = rospy.Subscriber(
            GAZEBO_CAMERA_POSE_TOPIC,
            PoseStamped,
            session.gazebo_pose_callback,
            queue_size=200,
        )
        rgb_sub = Subscriber(IMAGE_TOPIC, Image)
        depth_sub = Subscriber(DEPTH_TOPIC, Image)
        sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        sync.registerCallback(session.image_callback)

        rospy.spin()

        session.generate_masks()
        session.write_init_cloud_from_saved_frames()
        session.write_transforms()
    finally:
        session.cleanup()


if __name__ == "__main__":
    main()
