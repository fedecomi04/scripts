#!/usr/bin/env python3
from __future__ import annotations

import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import cv2
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp
from urdfpy import URDF

IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
DEPTH_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw"
CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
GAZEBO_CAMERA_POSE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose"
WORLD_FRAME = "dynaarm_arm_tf/world"
GAZEBO_CAMERA_FRAME = "dynaarm_arm_tf/camera_pose_link"
CAMERA_POSE_SAVE_FRAME = "dynaarm_arm_tf/camera_link_optical"

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
MASK_SAVER_SCRIPT = Path(__file__).with_name("ros1_robot_mask_saver_stl.py")
MASK_SAVER_READY_SENTINEL = DATASET_ROOT / ".robot_mask_saver_ready"
CAPTURE_COMPLETE_SENTINEL_NAME = ".capture_complete.json"

SAVE_HZ = 5.0
MAX_IMAGES = 60
IMAGE_NAME_PREFIX = "arm"
MASK_SAVER_READY_TIMEOUT_SEC = 20.0
MASK_SAVER_EXIT_TIMEOUT_SEC = 180.0
SYNC_QUEUE_SIZE = 20
SYNC_SLOP_SEC = 0.1

INIT_CLOUD_NAME = "depth_camera_init_points.ply"
MAX_INIT_CLOUD_POINTS = 300000
MAX_GAZEBO_POSE_HISTORY_SEC = 30.0

# ROS optical / OpenCV camera frame:
#   +X right, +Y down, +Z forward
#
# Nerfstudio / OpenGL camera frame:
#   +X right, +Y up, +Z back
ROS_OPTICAL_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0])


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


def rotate_camera_frame_only(transform_ros: np.ndarray) -> np.ndarray:
    transform_output = transform_ros.copy()
    transform_output[:3, :3] = transform_ros[:3, :3] @ ROS_OPTICAL_TO_NERFSTUDIO
    transform_output[:3, 3] = transform_ros[:3, 3]
    return transform_output


def normalize_frame_id(frame_id: str | None) -> str:
    return (frame_id or "").strip().lstrip("/")


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    transform_inv = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    transform_inv[:3, :3] = rotation.T
    transform_inv[:3, 3] = -rotation.T @ translation
    return transform_inv


def load_static_transform_from_urdf(source_frame: str, target_frame: str) -> np.ndarray:
    urdf_text = URDF_PATH.read_text()

    def repl(match):
        pkg = match.group(1)
        rest = match.group(2)

        basename = Path(rest).stem + ".stl"
        stl_path = STL_DIR / basename
        if stl_path.exists():
            return str(stl_path)
        if pkg not in PACKAGE_MAP:
            raise RuntimeError(f"Missing package root for '{pkg}'")
        return str(Path(PACKAGE_MAP[pkg]) / rest)

    urdf_text = re.sub(r"package://([^/]+)/([^\"'<> ]+)", repl, urdf_text)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as handle:
            handle.write(urdf_text)
            temp_path = Path(handle.name)

        robot = URDF.load(str(temp_path))
        zero_link_fk_by_name = robot.link_fk(use_names=True)
        frame_prefix = f"{normalize_frame_id(WORLD_FRAME).rsplit('/', 1)[0]}/"

        def frame_to_link_name(frame_id: str) -> str | None:
            normalized = normalize_frame_id(frame_id)
            if not normalized.startswith(frame_prefix):
                return None
            return normalized[len(frame_prefix):]

        source_link = frame_to_link_name(source_frame)
        target_link = frame_to_link_name(target_frame)
        if source_link is None or target_link is None:
            raise RuntimeError(f"Unable to resolve link names from '{source_frame}' -> '{target_frame}'")

        source_pose = zero_link_fk_by_name.get(source_link)
        target_pose = zero_link_fk_by_name.get(target_link)
        if source_pose is None or target_pose is None:
            raise RuntimeError(f"Missing URDF FK pose for '{source_link}' or '{target_link}'")

        return invert_rigid_transform(source_pose.astype(np.float64)) @ target_pose.astype(np.float64)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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


def resolve_relpath(base_dir: Path, rel_path: str) -> Path:
    return (base_dir / rel_path).resolve()


def resolve_frame_mask_path(dataset_dir: Path, frame: dict) -> Path | None:
    mask_rel = frame.get("mask_path")
    if mask_rel:
        mask_path = resolve_relpath(dataset_dir, mask_rel)
        if mask_path.exists():
            return mask_path

    rgb_rel = frame.get("file_path")
    if not rgb_rel:
        return None

    fallback_mask_path = (dataset_dir / "masks" / Path(rgb_rel).name).resolve()
    if fallback_mask_path.exists():
        return fallback_mask_path
    return None


def load_saved_mask(mask_path: Path | None, expected_hw: tuple[int, int]) -> np.ndarray:
    if mask_path is None or not mask_path.exists():
        raise RuntimeError(f"Missing mask at {mask_path}")

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


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


class LockedFile:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.handle = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("w", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


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


def launch_mask_saver() -> subprocess.Popen:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    if MASK_SAVER_READY_SENTINEL.exists():
        MASK_SAVER_READY_SENTINEL.unlink()

    process = subprocess.Popen([sys.executable, str(MASK_SAVER_SCRIPT)])
    deadline = time.time() + MASK_SAVER_READY_TIMEOUT_SEC
    while time.time() < deadline:
        if MASK_SAVER_READY_SENTINEL.exists():
            return process
        if process.poll() is not None:
            raise RuntimeError(f"Mask saver exited early with code {process.returncode}")
        time.sleep(0.1)

    process.terminate()
    process.wait(timeout=5.0)
    raise RuntimeError("Timed out waiting for mask saver ready sentinel")


def terminate_mask_saver(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def wait_for_mask_saver(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=MASK_SAVER_EXIT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        terminate_mask_saver(process)
        raise RuntimeError("Timed out waiting for mask saver to finish") from exc
    if process.returncode != 0:
        raise RuntimeError(f"Mask saver exited with code {process.returncode}")


class CaptureSession:
    def __init__(self) -> None:
        self.run_dir: Path | None = None
        self.rgb_dir: Path | None = None
        self.depth_dir: Path | None = None
        self.masks_dir: Path | None = None
        self.transforms_path: Path | None = None
        self.transforms_lock_path: Path | None = None
        self.capture_complete_path: Path | None = None
        self.metadata: dict | None = None
        self.frame_index = 0
        self.last_saved_stamp = None
        self.warned_unexpected_image_frame = False
        self.camera_pose_offset = load_static_transform_from_urdf(
            GAZEBO_CAMERA_FRAME,
            CAMERA_POSE_SAVE_FRAME,
        )
        self.gazebo_pose_times_sec: list[float] = []
        self.gazebo_pose_matrices: list[np.ndarray] = []

    def initialize(self) -> None:
        camera_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=5.0)
        self.run_dir = self._make_unique_run_dir()
        self.rgb_dir = self.run_dir / "rgb"
        self.depth_dir = self.run_dir / "depth"
        self.masks_dir = self.run_dir / "masks"
        self.transforms_path = self.run_dir / "transforms.json"
        self.transforms_lock_path = self.run_dir / "transforms.json.lock"
        self.capture_complete_path = self.run_dir / CAPTURE_COMPLETE_SENTINEL_NAME

        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "fl_x": float(camera_info.K[0]),
            "fl_y": float(camera_info.K[4]),
            "cx": float(camera_info.K[2]),
            "cy": float(camera_info.K[5]),
            "w": int(camera_info.width),
            "h": int(camera_info.height),
            "frames": [],
        }
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

    def _warn_if_unexpected_image_frame(self, frame_id: str) -> None:
        normalized = normalize_frame_id(frame_id)
        expected = CAMERA_POSE_SAVE_FRAME
        if normalized == expected or self.warned_unexpected_image_frame:
            return
        rospy.logwarn(
            "Image header frame_id is '%s' but poses are saved from '%s'.",
            normalized,
            expected,
        )
        self.warned_unexpected_image_frame = True

    def gazebo_pose_callback(self, msg: PoseStamped) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0:
            return

        pose_matrix = pose_msg_to_matrix(msg.pose).astype(np.float64)
        insert_at = bisect_left(self.gazebo_pose_times_sec, stamp_sec)
        if (
            insert_at < len(self.gazebo_pose_times_sec)
            and abs(self.gazebo_pose_times_sec[insert_at] - stamp_sec) <= 1e-6
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

    def _lookup_pose_matrix(self, stamp) -> np.ndarray:
        if not self.gazebo_pose_times_sec:
            raise RuntimeError("No Gazebo camera pose samples were received")

        query_time_sec = float(stamp.to_sec())
        insert_at = bisect_left(self.gazebo_pose_times_sec, query_time_sec)

        if (
            insert_at < len(self.gazebo_pose_times_sec)
            and abs(self.gazebo_pose_times_sec[insert_at] - query_time_sec) <= 1e-6
        ):
            pose_matrix = self.gazebo_pose_matrices[insert_at]
        elif (
            insert_at > 0
            and abs(self.gazebo_pose_times_sec[insert_at - 1] - query_time_sec) <= 1e-6
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

        return rotate_camera_frame_only(pose_matrix @ self.camera_pose_offset)

    def _build_transforms_payload(self, mask_paths: dict[str, str], ply_file_path: str | None) -> dict:
        assert self.metadata is not None
        payload = {key: value for key, value in self.metadata.items() if key != "frames"}
        payload_frames = []
        for frame in self.metadata["frames"]:
            payload_frame = dict(frame)
            mask_path = mask_paths.get(payload_frame["file_path"])
            if mask_path:
                payload_frame["mask_path"] = mask_path
            payload_frames.append(payload_frame)
        payload["frames"] = payload_frames
        if ply_file_path:
            payload["ply_file_path"] = ply_file_path
        return payload

    def write_transforms(self) -> None:
        if self.metadata is None or self.transforms_path is None or self.transforms_lock_path is None:
            return

        with LockedFile(self.transforms_lock_path):
            mask_paths = {}
            ply_file_path = None
            if self.transforms_path.exists():
                try:
                    current = read_json_with_retry(self.transforms_path)
                except Exception:
                    current = {}
                for frame in current.get("frames", []):
                    file_path = frame.get("file_path")
                    mask_path = frame.get("mask_path")
                    if file_path and mask_path:
                        mask_paths[file_path] = mask_path
                ply_file_path = current.get("ply_file_path")

            payload = self._build_transforms_payload(mask_paths, ply_file_path)
            write_json_atomic(self.transforms_path, payload)

    def write_capture_complete_sentinel(self) -> None:
        if self.capture_complete_path is None or self.metadata is None:
            return
        payload = {
            "frame_count": len(self.metadata["frames"]),
            "finished_at": time.time(),
        }
        write_json_atomic(self.capture_complete_path, payload)

    def patch_ply_file_path(self) -> None:
        if self.transforms_path is None or self.transforms_lock_path is None or not self.transforms_path.exists():
            return

        with LockedFile(self.transforms_lock_path):
            data = read_json_with_retry(self.transforms_path)
            data["ply_file_path"] = INIT_CLOUD_NAME
            write_json_atomic(self.transforms_path, data)

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
            rospy.logwarn_throttle(
                2.0,
                "Skipping frame because camera pose is unavailable: %s",
                exc,
            )
            return

        image_bgr = ros_image_to_bgr(image_msg)
        depth_mm = ros_depth_to_uint16_mm(depth_msg)

        seq = int(image_msg.header.seq)
        file_stem = f"{IMAGE_NAME_PREFIX}_{seq:05d}"
        image_file_name = f"{file_stem}.png"
        depth_file_name = f"{file_stem}.tiff"
        image_path = self.rgb_dir / image_file_name
        depth_path = self.depth_dir / depth_file_name

        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"Failed to save RGB image to {image_path}")
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise RuntimeError(f"Failed to save depth image to {depth_path}")

        self.metadata["frames"].append(
            {
                "file_path": f"./rgb/{image_file_name}",
                "depth_file_path": f"./depth/{depth_file_name}",
                "transform_matrix": transform_matrix.tolist(),
            }
        )
        self.write_transforms()

        self.frame_index += 1
        self.last_saved_stamp = image_msg.header.stamp
        rospy.loginfo("Saved frame %d/%d", self.frame_index, MAX_IMAGES)

        if self.frame_index >= MAX_IMAGES:
            rospy.signal_shutdown(f"Saved {MAX_IMAGES} images")

    def write_init_cloud_from_saved_frames(self) -> None:
        if self.run_dir is None or self.transforms_path is None or not self.transforms_path.exists():
            return

        ply_path = self.run_dir / INIT_CLOUD_NAME
        current_transforms = read_json_with_retry(self.transforms_path)
        frames = current_transforms.get("frames", [])
        if not frames:
            return

        dataset_dir = self.run_dir.resolve()
        rng = np.random.default_rng(0)
        frame_infos = []
        valid_counts = []

        for frame in frames:
            depth_rel = frame.get("depth_file_path")
            rgb_rel = frame.get("file_path")
            if not depth_rel or not rgb_rel:
                continue

            depth_path = resolve_relpath(dataset_dir, depth_rel)
            rgb_path = resolve_relpath(dataset_dir, rgb_rel)
            mask_path = resolve_frame_mask_path(dataset_dir, frame)
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
            x = (xs.astype(np.float32) - float(current_transforms["cx"])) * depth_m / float(current_transforms["fl_x"])
            y = -(ys.astype(np.float32) - float(current_transforms["cy"])) * depth_m / float(current_transforms["fl_y"])
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
        write_ascii_ply(ply_path, xyz, rgb)
        self.patch_ply_file_path()


def main() -> None:
    rospy.init_node("save_dynaarm_camera1_rgb_gazebo")
    mask_process = None
    session = CaptureSession()

    try:
        mask_process = launch_mask_saver()
        session.initialize()

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

        session.write_transforms()
        session.write_capture_complete_sentinel()
        wait_for_mask_saver(mask_process)
        session.write_init_cloud_from_saved_frames()
        session.write_transforms()
    finally:
        terminate_mask_saver(mask_process)
        _ = gazebo_pose_sub if "gazebo_pose_sub" in locals() else None


if __name__ == "__main__":
    main()
