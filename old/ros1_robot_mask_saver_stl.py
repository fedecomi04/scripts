#!/usr/bin/env python3
from __future__ import annotations

from bisect import bisect_left
import fcntl
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pyrender
import rospy
import trimesh
from sensor_msgs.msg import CameraInfo, Image, JointState
from urdfpy import URDF

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

IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
BASE_FRAME = "dynaarm_arm_tf/world"
DEFAULT_CAMERA_FRAME = "dynaarm_arm_tf/camera_pose_link"
GAZEBO_JOINT_STATES_TOPIC = "/dynaarm_arm/joint_states_full"
CAMERA_NAME = "arm"

INIT_CLOUD_NAME = "depth_camera_init_points.ply"
READY_SENTINEL_NAME = ".robot_mask_saver_ready"
CAPTURE_COMPLETE_SENTINEL_NAME = ".capture_complete.json"
TRANSFORMS_METADATA_NAMES = ("transforms.json", "transforms_tf.json", "transform_gazebo.json")

QUEUE_SIZE = 200
POINTS_PER_FRAME = 200000
BACKGROUND_COLOR_THRESHOLD = 10.0
COMPLETE_DRAIN_SEC = 1.0
LOOP_RATE_HZ = 30.0
HISTORY_TIMEOUT_SEC = 10.0
TIME_EPS_SEC = 1e-6
MASK_KEEP_ERODE_RADIUS_PX = 4
MASK_MIN_KEEP_COMPONENT_AREA_PX = 64

RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d{2})?$")


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def read_json_with_retry(path: Path, retries: int = 5, delay_sec: float = 0.05) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(delay_sec)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read JSON from {path}")


def write_json_atomic(path: Path, payload: dict, indent: int = 2) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    os.replace(tmp_path, path)

@contextmanager
def locked_file(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


class RobotMaskSaver:
    def __init__(self) -> None:
        self.dataset_root = DATASET_ROOT.expanduser().resolve()
        self.ready_sentinel_path = self.dataset_root / READY_SENTINEL_NAME
        self.known_run_names = {run.name for run in self._list_run_dirs(self.dataset_root)}

        self.dataset_dir: Optional[Path] = None
        self.rgb_dir: Optional[Path] = None
        self.masks_dir: Optional[Path] = None
        self.transforms_path: Optional[Path] = None
        self.transforms_lock_path: Optional[Path] = None
        self.init_cloud_path: Optional[Path] = None
        self.capture_complete_path: Optional[Path] = None

        self.intrinsics: Optional[CameraIntrinsics] = None
        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        self.scene: Optional[pyrender.Scene] = None
        self.camera_node = None
        self.robot_nodes: list[Tuple[str, object, object]] = []
        self.mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self.joint_state_times_sec: list[float] = []
        self.joint_state_positions: list[Dict[str, float]] = []

        self.stamps_by_seq: "OrderedDict[int, rospy.Time]" = OrderedDict()
        self.processed_seqs: set[int] = set()
        self.rgb_name_by_seq: Dict[int, str] = {}
        self.init_cloud_target_seq: Optional[int] = None
        self.init_cloud_saved = False
        self.completion_detected_at: Optional[float] = None
        self.finalized = False

        self.background_rgb_colors = self._load_background_rgb_colors()

        rospy.loginfo("Loading URDF from %s", URDF_PATH)
        patched_urdf = self._make_temp_resolved_urdf(URDF_PATH, PACKAGE_MAP, STL_DIR)
        self.robot = URDF.load(patched_urdf)
        self.zero_link_fk_by_name = self.robot.link_fk(use_names=True)
        self.actuated_joint_names = set(self.robot.actuated_joint_names)
        self.frame_prefix = f"{self._normalize_frame_id(BASE_FRAME).rsplit('/', 1)[0]}/"

        self.camera_info_sub = rospy.Subscriber(INFO_TOPIC, CameraInfo, self._camera_info_cb, queue_size=1)
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self._image_cb, queue_size=20)
        self.joint_states_sub = rospy.Subscriber(
            GAZEBO_JOINT_STATES_TOPIC,
            JointState,
            self._joint_states_cb,
            queue_size=200,
        )

    def write_ready_sentinel(self) -> None:
        payload = {"pid": os.getpid(), "ready_time": time.time()}
        write_json_atomic(self.ready_sentinel_path, payload, indent=2)

    def _list_run_dirs(self, dataset_dir: Path) -> list[Path]:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return []
        return [path for path in dataset_dir.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)]

    def _refresh_dataset_dir(self) -> None:
        if self.dataset_dir is not None:
            return

        new_runs = [run for run in self._list_run_dirs(self.dataset_root) if run.name not in self.known_run_names]
        if not new_runs:
            return

        dataset_dir = max(new_runs, key=lambda path: path.stat().st_mtime).resolve()
        self.known_run_names.add(dataset_dir.name)
        self.dataset_dir = dataset_dir
        self.rgb_dir = dataset_dir / "rgb"
        self.masks_dir = dataset_dir / "masks"
        self.transforms_path = dataset_dir / "transforms.json"
        self.transforms_lock_path = dataset_dir / "transforms.json.lock"
        self.init_cloud_path = dataset_dir / INIT_CLOUD_NAME
        self.capture_complete_path = dataset_dir / CAPTURE_COMPLETE_SENTINEL_NAME
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_processed_seqs()
        rospy.loginfo("Detected new dataset run: %s", dataset_dir)

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
        seq = int(msg.header.seq)
        if seq in self.stamps_by_seq:
            del self.stamps_by_seq[seq]
        self.stamps_by_seq[seq] = msg.header.stamp
        while len(self.stamps_by_seq) > QUEUE_SIZE:
            self.stamps_by_seq.popitem(last=False)

    def _make_temp_resolved_urdf(self, urdf_path: Path, package_map: Dict[str, str], stl_dir: Path) -> str:
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
        if self.renderer is not None or self.intrinsics is None:
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
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.0), pose=np.eye(4, dtype=np.float32))
        light2_pose = np.eye(4, dtype=np.float32)
        light2_pose[:3, 3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add(pyrender.PointLight(color=np.ones(3), intensity=20.0), pose=light2_pose)
        self._build_scene()
        rospy.loginfo("Renderer initialized")

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

    def _load_trimesh(self, path: str, scale: Optional[np.ndarray]) -> trimesh.Trimesh:
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

    @staticmethod
    def _normalize_frame_id(frame_id: Optional[str]) -> str:
        return (frame_id or "").strip().lstrip("/")

    @staticmethod
    def _invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
        transform_inv = np.eye(4, dtype=np.float32)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        transform_inv[:3, :3] = rotation.T
        transform_inv[:3, 3] = -rotation.T @ translation
        return transform_inv

    def _trim_history(self, times_sec: list[float], payloads: list, newest_time_sec: float) -> None:
        while len(times_sec) > 1 and newest_time_sec - times_sec[0] > HISTORY_TIMEOUT_SEC:
            times_sec.pop(0)
            payloads.pop(0)

    def _joint_states_cb(self, msg: JointState) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0 or not msg.name or not msg.position:
            return

        positions = {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
            if name in self.actuated_joint_names
        }
        if not positions:
            return

        insert_at = bisect_left(self.joint_state_times_sec, stamp_sec)
        if insert_at < len(self.joint_state_times_sec) and abs(self.joint_state_times_sec[insert_at] - stamp_sec) <= TIME_EPS_SEC:
            self.joint_state_positions[insert_at] = positions
        else:
            self.joint_state_times_sec.insert(insert_at, stamp_sec)
            self.joint_state_positions.insert(insert_at, positions)
        self._trim_history(self.joint_state_times_sec, self.joint_state_positions, stamp_sec)

    def _frame_to_link_name(self, frame_id: Optional[str]) -> Optional[str]:
        normalized = self._normalize_frame_id(frame_id)
        if not normalized.startswith(self.frame_prefix):
            return None
        return normalized[len(self.frame_prefix):]

    def _static_link_offset(self, source_frame: str, target_frame: str) -> np.ndarray:
        source_link = self._frame_to_link_name(source_frame)
        target_link = self._frame_to_link_name(target_frame)
        if source_link is None or target_link is None:
            raise RuntimeError(
                f"Unable to resolve link names from frames '{source_frame}' -> '{target_frame}'"
            )

        source_pose = self.zero_link_fk_by_name.get(source_link)
        target_pose = self.zero_link_fk_by_name.get(target_link)
        if source_pose is None or target_pose is None:
            raise RuntimeError(
                f"URDF FK is missing required link pose for '{source_link}' or '{target_link}'"
            )
        return self._invert_rigid_transform(source_pose.astype(np.float32)) @ target_pose.astype(np.float32)

    def _sample_joint_positions(self, stamp: rospy.Time) -> Dict[str, float]:
        if not self.joint_state_times_sec:
            raise RuntimeError("No Gazebo joint state samples have been received yet")

        query_time_sec = float(stamp.to_sec())
        insert_at = bisect_left(self.joint_state_times_sec, query_time_sec)

        if insert_at < len(self.joint_state_times_sec) and abs(self.joint_state_times_sec[insert_at] - query_time_sec) <= TIME_EPS_SEC:
            return dict(self.joint_state_positions[insert_at])
        if insert_at > 0 and abs(self.joint_state_times_sec[insert_at - 1] - query_time_sec) <= TIME_EPS_SEC:
            return dict(self.joint_state_positions[insert_at - 1])

        prev_idx = insert_at - 1 if insert_at > 0 else None
        next_idx = insert_at if insert_at < len(self.joint_state_times_sec) else None

        if prev_idx is not None and next_idx is not None:
            prev_time_sec = self.joint_state_times_sec[prev_idx]
            next_time_sec = self.joint_state_times_sec[next_idx]
            alpha = (query_time_sec - prev_time_sec) / (next_time_sec - prev_time_sec)
            prev_positions = self.joint_state_positions[prev_idx]
            next_positions = self.joint_state_positions[next_idx]
            joint_names = set(prev_positions.keys()) | set(next_positions.keys())
            interpolated_positions: Dict[str, float] = {}
            for joint_name in joint_names:
                prev_value = prev_positions.get(joint_name)
                next_value = next_positions.get(joint_name)
                if prev_value is None:
                    interpolated_positions[joint_name] = float(next_value)
                    continue
                if next_value is None:
                    interpolated_positions[joint_name] = float(prev_value)
                    continue
                interpolated_positions[joint_name] = float(prev_value * (1.0 - alpha) + next_value * alpha)
            return interpolated_positions

        sample_idx = prev_idx if prev_idx is not None else next_idx
        if sample_idx is None:
            raise RuntimeError("No Gazebo joint state samples are available for interpolation")
        return dict(self.joint_state_positions[sample_idx])

    def _camera_pose_from_link_fk(
        self,
        link_fk: Dict[str, np.ndarray],
        camera_frame: Optional[str],
    ) -> np.ndarray:
        resolved_camera_frame = self._normalize_frame_id(camera_frame) or DEFAULT_CAMERA_FRAME
        link_name = self._frame_to_link_name(resolved_camera_frame)
        if link_name is not None and link_name in link_fk:
            return link_fk[link_name].astype(np.float32)

        default_link_name = self._frame_to_link_name(DEFAULT_CAMERA_FRAME)
        if default_link_name is None or default_link_name not in link_fk:
            raise RuntimeError(f"Camera link '{DEFAULT_CAMERA_FRAME}' is missing from FK results")

        default_pose = link_fk[default_link_name].astype(np.float32)
        return default_pose @ self._static_link_offset(DEFAULT_CAMERA_FRAME, resolved_camera_frame)

    def _update_robot_poses(self, link_fk: Dict[str, np.ndarray]) -> None:
        assert self.scene is not None
        for link_name, visual, node in self.robot_nodes:
            base_T_link = link_fk.get(link_name)
            if base_T_link is None:
                rospy.logwarn_throttle(2.0, "Skipping link without FK pose: %s", link_name)
                continue
            link_T_visual = np.eye(4, dtype=np.float32)
            if visual.origin is not None:
                link_T_visual = visual.origin.astype(np.float32)
            self.scene.set_pose(node, pose=base_T_link.astype(np.float32) @ link_T_visual)

    def _build_render_camera_pose(self, ros_pose: np.ndarray) -> np.ndarray:
        T_optical_to_opengl = np.eye(4, dtype=np.float32)
        T_optical_to_opengl[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )
        T_rot_y_m90 = np.eye(4, dtype=np.float32)
        T_rot_y_m90[:3, :3] = np.array(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )
        T_rot_z_90 = np.eye(4, dtype=np.float32)
        T_rot_z_90[:3, :3] = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return ros_pose @ T_optical_to_opengl @ T_rot_y_m90 @ T_rot_z_90

    def _render_robot_black_mask(self, stamp: rospy.Time, camera_frame: str) -> np.ndarray:
        sampled_joint_positions = self._sample_joint_positions(stamp)
        link_fk = self.robot.link_fk(cfg=sampled_joint_positions, use_names=True)
        cam_pose = self._camera_pose_from_link_fk(link_fk, camera_frame)
        self._update_robot_poses(link_fk)
        render_cam_pose = self._build_render_camera_pose(cam_pose)
        self.scene.set_pose(self.camera_node, pose=render_cam_pose)
        _, depth = self.renderer.render(self.scene)
        robot_black_mask = (depth == 0).astype(np.uint8) * 255
        return cv2.flip(robot_black_mask, 0)

    def _rgb_name_for_seq(self, seq: int) -> str:
        return self.rgb_name_by_seq.get(seq, f"{CAMERA_NAME}_{seq:05d}.png")

    def _mask_path_for_seq(self, seq: int) -> Path:
        assert self.masks_dir is not None
        return self.masks_dir / self._rgb_name_for_seq(seq)

    def _rgb_path_for_seq(self, seq: int) -> Path:
        assert self.rgb_dir is not None
        return self.rgb_dir / self._rgb_name_for_seq(seq)

    def _seq_from_camera_filename(self, path: Path) -> Optional[int]:
        match = re.fullmatch(rf"{re.escape(CAMERA_NAME)}_(\d+)\.png", path.name)
        if match is not None:
            return int(match.group(1))
        stem = path.stem
        return int(stem) if stem.isdigit() else None

    def _initialize_processed_seqs(self) -> None:
        self.processed_seqs = set()
        if self.masks_dir is None or not self.masks_dir.exists():
            return

        for mask_path in self.masks_dir.glob("*.png"):
            seq = self._seq_from_camera_filename(mask_path)
            if seq is None:
                continue
            self.rgb_name_by_seq[seq] = mask_path.name
            self.processed_seqs.add(seq)

    def _load_frame_record(self, seq: int) -> Optional[dict]:
        if self.transforms_path is None or not self.transforms_path.exists():
            return None

        try:
            data = read_json_with_retry(self.transforms_path)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to read transforms.json for seq %d: %s", seq, exc)
            return None

        rgb_name = self._rgb_name_for_seq(seq)
        for frame in data.get("frames", []):
            if Path(frame.get("file_path", "")).name != rgb_name:
                continue
            return frame
        return None

    def _resolve_dataset_path(self, rel_or_abs: str) -> Optional[Path]:
        if not rel_or_abs or self.dataset_dir is None:
            return None
        path = Path(rel_or_abs)
        if path.is_absolute():
            return path
        return (self.dataset_dir / path).resolve()

    def _load_frame_pose(self, seq: int) -> Optional[np.ndarray]:
        frame = self._load_frame_record(seq)
        if frame is None:
            return None

        matrix = np.array(frame.get("transform_matrix"), dtype=np.float32)
        if matrix.shape != (4, 4):
            rospy.logwarn("Invalid transform_matrix shape for seq %d: %s", seq, matrix.shape)
            return None
        return matrix

    def _load_depth_mm(self, frame: dict, seq: int, expected_hw: tuple[int, int]) -> Optional[np.ndarray]:
        depth_path = self._resolve_dataset_path(frame.get("depth_file_path", ""))
        if depth_path is None or not depth_path.exists():
            return None

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            rospy.logwarn("Failed to read depth image for seq %d from %s", seq, depth_path)
            return None

        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape != expected_hw:
            rospy.logwarn(
                "Depth shape mismatch for seq %d: got %s expected %s",
                seq,
                depth.shape,
                expected_hw,
            )
            return None

        if np.issubdtype(depth.dtype, np.floating):
            invalid = ~np.isfinite(depth) | (depth <= 0.0)
            depth_mm = np.round(depth * 1000.0)
            depth_mm[invalid] = 0.0
            return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

        return depth.astype(np.uint16)

    def _load_rgb(self, frame: dict, seq: int, expected_hw: tuple[int, int]) -> Optional[np.ndarray]:
        rgb_path = self._resolve_dataset_path(frame.get("file_path", ""))
        if rgb_path is None or not rgb_path.exists():
            return None

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            rospy.logwarn("Failed to read RGB image for seq %d from %s", seq, rgb_path)
            return None
        if rgb_bgr.shape[:2] != expected_hw:
            rospy.logwarn(
                "RGB shape mismatch for seq %d: got %s expected %s",
                seq,
                rgb_bgr.shape[:2],
                expected_hw,
            )
            return None
        return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    def _save_init_cloud(self, seq: int, mask: Optional[np.ndarray] = None) -> bool:
        if self.init_cloud_saved or self.init_cloud_target_seq != seq or self.intrinsics is None:
            return False

        if mask is None:
            mask_path = self._mask_path_for_seq(seq)
            if not mask_path.exists():
                return False
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                rospy.logwarn("Failed to read mask for seq %d from %s", seq, mask_path)
                return False
            if mask.ndim == 3:
                mask = mask[..., 0]

        frame = self._load_frame_record(seq)
        if frame is None:
            return False

        depth_mm = self._load_depth_mm(frame, seq, mask.shape)
        if depth_mm is None:
            return False

        c2w = self._load_frame_pose(seq)
        if c2w is None:
            return False

        rgb = self._load_rgb(frame, seq, mask.shape)
        if rgb is None:
            return False

        valid = (mask > 0) & (depth_mm > 0)
        ys, xs = np.where(valid)
        if ys.size == 0:
            rospy.logwarn("No scene points remained after masking for seq %d", seq)
            return False

        if ys.size > POINTS_PER_FRAME:
            choice = np.random.default_rng(seq).choice(ys.size, size=POINTS_PER_FRAME, replace=False)
            ys = ys[choice]
            xs = xs[choice]

        depth_m = depth_mm[ys, xs].astype(np.float32) / 1000.0
        x = (xs.astype(np.float32) - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y = -(ys.astype(np.float32) - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        xyz_cam_ns = np.stack([x, y, -depth_m], axis=1)
        rgb = rgb[ys, xs].reshape(-1, 3).astype(np.uint8)

        hom = np.concatenate([xyz_cam_ns, np.ones((xyz_cam_ns.shape[0], 1), dtype=np.float32)], axis=1)
        xyz_world = (c2w @ hom.T).T[:, :3]

        assert self.init_cloud_path is not None
        write_ascii_ply(self.init_cloud_path, xyz_world.astype(np.float32), rgb)
        self.init_cloud_saved = True
        self._patch_transforms()
        rospy.loginfo("Saved init cloud for seq %d to %s", seq, self.init_cloud_path)
        return True

    def _patch_transforms(self) -> None:
        if self.transforms_path is None or self.transforms_lock_path is None or not self.transforms_path.exists():
            return

        with locked_file(self.transforms_lock_path):
            for transforms_path in self._existing_transforms_paths():
                data = read_json_with_retry(transforms_path)
                updated = False

                for frame in data.get("frames", []):
                    rgb_name = Path(frame.get("file_path", "")).name
                    if not rgb_name or self.masks_dir is None:
                        continue
                    mask_file = self.masks_dir / rgb_name
                    if not mask_file.exists():
                        continue
                    mask_rel = f"./masks/{rgb_name}"
                    if frame.get("mask_path") != mask_rel:
                        frame["mask_path"] = mask_rel
                        updated = True

                if self.init_cloud_path is not None and self.init_cloud_path.exists():
                    if data.get("ply_file_path") != self.init_cloud_path.name:
                        data["ply_file_path"] = self.init_cloud_path.name
                        updated = True

                if updated:
                    write_json_atomic(transforms_path, data, indent=2)

    def _compute_background_black_mask(self, seq: int) -> Optional[np.ndarray]:
        rgb_path = self._rgb_path_for_seq(seq)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            rospy.logwarn("Failed to read RGB image for seq %d from %s", seq, rgb_path)
            return None

        if not self.background_rgb_colors:
            rospy.logwarn_throttle(
                2.0,
                "No background colors were loaded from %s; keeping full scene mask",
                WORLD_FILE,
            )
            return None

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        background_like = np.zeros(rgb.shape[:2], dtype=bool)
        for color_rgb in self.background_rgb_colors:
            diff = rgb - color_rgb[None, None, :]
            color_dist = np.linalg.norm(diff, axis=2)
            background_like |= color_dist <= BACKGROUND_COLOR_THRESHOLD
        return (~background_like).astype(np.uint8) * 255

    def _parse_rgba_text_to_rgb255(self, text: Optional[str]) -> Optional[np.ndarray]:
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
        colors: list[np.ndarray] = []
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

        deduped: list[np.ndarray] = []
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
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
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

    def _process_seq(self, seq: int, stamp: rospy.Time, camera_frame: Optional[str]) -> bool:
        self._ensure_renderer()
        if self.renderer is None or self.scene is None:
            return False

        selected_camera_frame = (camera_frame or "").strip().lstrip("/") or DEFAULT_CAMERA_FRAME
        try:
            robot_black_mask = self._render_robot_black_mask(stamp, selected_camera_frame)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Gazebo/render pose update failed for camera frame %s: %s",
                selected_camera_frame,
                exc,
            )
            return False

        robot_pixels = int(np.count_nonzero(robot_black_mask == 0))
        if robot_pixels <= 0:
            rospy.logwarn_throttle(
                2.0,
                "Rendered robot mask is empty for seq %d with camera frame %s",
                seq,
                selected_camera_frame,
            )
        else:
            rospy.loginfo(
                "Using camera frame %s for seq %d gripper mask (%d robot pixels)",
                selected_camera_frame,
                seq,
                robot_pixels,
            )

        # Background masking disabled: keep the full background and only exclude the robot.
        # background_black_mask = self._compute_background_black_mask(seq)
        # if background_black_mask is None:
        #     background_black_mask = np.full(robot_black_mask.shape, 255, dtype=np.uint8)
        #     mask = robot_black_mask
        # else:
        #     mask = cv2.bitwise_and(robot_black_mask, background_black_mask)
        mask = robot_black_mask
        mask = self._refine_keep_mask(mask)

        mask_path = self._mask_path_for_seq(seq)
        if not cv2.imwrite(str(mask_path), mask):
            rospy.logerr("Failed to save mask to %s", mask_path)
            return False

        self.processed_seqs.add(seq)
        if self.init_cloud_target_seq is None:
            self.init_cloud_target_seq = seq
        self._save_init_cloud(seq, mask=mask)
        self._patch_transforms()
        rospy.loginfo("Saved mask for seq %d", seq)
        return True

    def _prune_frames_missing_masks(self) -> None:
        if self.transforms_path is None or self.transforms_lock_path is None or not self.transforms_path.exists():
            return

        with locked_file(self.transforms_lock_path):
            primary_data = read_json_with_retry(self.transforms_path)
            frames = primary_data.get("frames", [])
            kept_frames = []
            removed_frames = []

            for frame in frames:
                rgb_name = Path(frame.get("file_path", "")).name
                if not rgb_name or self.masks_dir is None:
                    kept_frames.append(frame)
                    continue
                mask_path = (self.masks_dir / rgb_name).resolve()
                if mask_path.exists():
                    kept_frames.append(frame)
                    continue
                removed_frames.append(frame)

            if not removed_frames:
                return

            removed_rgb_names = {
                Path(frame.get("file_path", "")).name
                for frame in removed_frames
                if frame.get("file_path")
            }
            for frame in removed_frames:
                for key in ("file_path", "depth_file_path", "mask_path"):
                    rel = frame.get(key)
                    if not rel or self.dataset_dir is None:
                        continue
                    path = (self.dataset_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception as exc:
                        rospy.logwarn("Failed to delete %s while pruning missing-mask frame: %s", path, exc)

            primary_data["frames"] = kept_frames
            write_json_atomic(self.transforms_path, primary_data, indent=2)
            for transforms_path in self._existing_transforms_paths():
                if transforms_path == self.transforms_path:
                    continue
                data = read_json_with_retry(transforms_path)
                data["frames"] = [
                    frame
                    for frame in data.get("frames", [])
                    if Path(frame.get("file_path", "")).name not in removed_rgb_names
                ]
                write_json_atomic(transforms_path, data, indent=2)
            rospy.loginfo("Pruned %d frames without masks from dataset %s", len(removed_frames), self.dataset_dir)

    def _existing_transforms_paths(self) -> list[Path]:
        if self.dataset_dir is None:
            return []

        paths: list[Path] = []
        seen = set()
        for name in TRANSFORMS_METADATA_NAMES:
            path = (self.dataset_dir / name).resolve()
            if not path.exists():
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
        return paths

    def _cleanup_renderer(self) -> None:
        if self.renderer is None:
            return
        try:
            self.renderer.delete()
        except Exception as exc:
            rospy.logwarn("Renderer cleanup failed: %s", exc)
        finally:
            self.renderer = None
            self.scene = None
            self.camera_node = None
            self.robot_nodes = []

    def _cleanup_sentinels(self) -> None:
        for path in (self.ready_sentinel_path, self.capture_complete_path):
            if path is None:
                continue
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        try:
            if self.init_cloud_target_seq is not None:
                self._save_init_cloud(self.init_cloud_target_seq)
            self._patch_transforms()
            self._prune_frames_missing_masks()
        finally:
            self._cleanup_renderer()
            self._cleanup_sentinels()

    def process_once(self) -> None:
        self._refresh_dataset_dir()
        if self.rgb_dir is None or not self.rgb_dir.exists():
            return

        for rgb_path in sorted(self.rgb_dir.glob("*.png")):
            seq = self._seq_from_camera_filename(rgb_path)
            if seq is None:
                continue
            self.rgb_name_by_seq[seq] = rgb_path.name

            if seq in self.processed_seqs:
                continue

            mask_path = self._mask_path_for_seq(seq)
            if mask_path.exists():
                self.processed_seqs.add(seq)
                if self.init_cloud_target_seq is None:
                    self.init_cloud_target_seq = seq
                continue

            stamp = self.stamps_by_seq.get(seq)
            if stamp is None:
                continue
            self._process_seq(seq, stamp, DEFAULT_CAMERA_FRAME)

        if self.init_cloud_target_seq is not None:
            self._save_init_cloud(self.init_cloud_target_seq)

        if self.capture_complete_path is None or not self.capture_complete_path.exists():
            return

        if self.completion_detected_at is None:
            self.completion_detected_at = time.time()
            rospy.loginfo("Capture complete sentinel detected for %s", self.dataset_dir)

        self._patch_transforms()
        if time.time() - self.completion_detected_at >= COMPLETE_DRAIN_SEC:
            self.finalize()
            rospy.signal_shutdown("Mask generation finished")

    def spin(self) -> None:
        rate = rospy.Rate(LOOP_RATE_HZ)
        while not rospy.is_shutdown():
            self.process_once()
            rate.sleep()


def main() -> None:
    rospy.init_node("robot_mask_saver_stl", anonymous=False)
    saver = RobotMaskSaver()
    rospy.on_shutdown(saver.finalize)
    saver.write_ready_sentinel()
    rospy.loginfo("robot_mask_saver_stl is waiting for a new dataset run under %s", DATASET_ROOT)
    saver.spin()


if __name__ == "__main__":
    main()
