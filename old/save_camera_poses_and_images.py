#!/usr/bin/env python3
"""
Save RGB or synchronized RGB+depth images and camera poses from ROS1 into a
Nerfstudio-style dataset run.

This recorder queries tf2 at each saved image timestamp so the recorded pose is
looked up directly for the sensor sample being written.

Compared with the current ROS saver, it records extra timing information per
frame:

- RGB image timestamp
- depth timestamp when available
- TF pose timestamp selected for RGB and depth
- pose minus sensor time deltas
- raw ROS optical-frame camera poses
- Nerfstudio-convention camera poses written to transform_matrix fields

Output layout in RGBD mode:
  <output-root>/<timestamp>/
    rgb/
      arm_00001.png
      ...
    depth/
      arm_00001.tiff
      ...
    transforms.json
    pose_log.csv
    capture_metadata.json
    capture_report.json

By default it follows data_teleoperation/configs/dynaarm_gs_depth_mask.yaml and
writes into:
/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynaarm_gs_depth_mask_01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import ros_numpy
import rospy
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf.transformations import quaternion_from_matrix
import tf2_ros
from tf2_msgs.msg import TFMessage


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "dynaarm_gs_depth_mask.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "datasets" / "dynaarm_gs_depth_mask_01"
DEFAULT_CAMERA_TOPIC_BASE = "/dynaarm_arm/dynaarm_arm/camera1"
CAMERA_PROFILES = {
    "manual": {},
    "basic_depth": {
        "image_topic": "/camera/color/image_raw",
        "info_topic": "/camera/color/camera_info",
        "depth_topic": "/camera/depth/image_raw",
        "depth_info_topic": "/camera/depth/camera_info",
        "camera_frame": "camera_link",
    },
    "kinect": {
        "image_topic": "/dynaarm_arm/dynaarm_arm/camera1/image_raw",
        "info_topic": "/dynaarm_arm/dynaarm_arm/camera1/camera_info",
        "depth_topic": "/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw",
        "depth_info_topic": "/dynaarm_arm/dynaarm_arm/camera1/depth/camera_info",
        "camera_frame": "camera_link_optical",
    },
}
# Current Dynaarm camera_link_optical frame is already aligned with the
# Nerfstudio / OpenGL camera frame, so no extra camera-frame rotation is needed
# before writing transform_matrix.
CAMERA_FRAME_TO_NERFSTUDIO = np.eye(4, dtype=np.float64)
POSE_LOG_COLUMNS = [
    "index",
    "rgb_seq",
    "depth_seq",
    "file_path",
    "depth_file_path",
    "rgb_timestamp_sec",
    "depth_timestamp_sec",
    "sync_dt_sec",
    "rgb_tf_timestamp_sec",
    "depth_tf_timestamp_sec",
    "rgb_tf_dt_sec",
    "depth_tf_dt_sec",
    "rgb_tf_lookup_mode",
    "depth_tf_lookup_mode",
    "rgb_pose_chain_span_sec",
    "depth_pose_chain_span_sec",
    "blur_score",
    "rgb_tx",
    "rgb_ty",
    "rgb_tz",
    "rgb_qx",
    "rgb_qy",
    "rgb_qz",
    "rgb_qw",
    "depth_tx",
    "depth_ty",
    "depth_tz",
    "depth_qx",
    "depth_qy",
    "depth_qz",
    "depth_qw",
]


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    distortion_model: str = "plumb_bob"
    camera_model: str = "OPENCV"

    @classmethod
    def from_msg(cls, msg: CameraInfo) -> "CameraIntrinsics":
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        D = [float(x) for x in list(msg.D)]
        return cls(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            k1=float(D[0]) if len(D) > 0 else 0.0,
            k2=float(D[1]) if len(D) > 1 else 0.0,
            p1=float(D[2]) if len(D) > 2 else 0.0,
            p2=float(D[3]) if len(D) > 3 else 0.0,
            distortion_model=str(getattr(msg, "distortion_model", "") or "plumb_bob"),
        )

    def transforms_fields(self) -> Dict[str, Any]:
        return {
            "camera_model": self.camera_model,
            "w": self.width,
            "h": self.height,
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
        }


@dataclass
class EdgeState:
    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    stamp_sec: float
    translation: list[float]
    quaternion_xyzw: list[float]
    is_static: bool = False


@dataclass
class PoseSample:
    stamp_sec: float
    matrix: np.ndarray
    translation: list[float]
    quaternion_xyzw: list[float]
    chain_stamp_span_sec: float
    dynamic_edge_count: int
    static_edge_count: int


@dataclass
class PendingFrame:
    rgb_seq: int
    depth_seq: Optional[int]
    rgb_stamp_sec: float
    depth_stamp_sec: Optional[float]
    rgb_frame_id: str
    depth_frame_id: str
    camera_frame: str
    rgb_image: np.ndarray
    depth_mm: Optional[np.ndarray]
    blur_score: float
    sync_dt_sec: Optional[float]


def read_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sanitize_frame(frame: str) -> str:
    return frame.lstrip("/")


def normalize_ros_topic(topic_name: str) -> str:
    topic_name = (topic_name or "").strip()
    if not topic_name:
        return ""
    return "/" + topic_name.strip("/")


def normalize_topic_base(topic_base: str) -> str:
    return normalize_ros_topic(topic_base)


def normalize_frame_prefix(frame_prefix: str) -> str:
    return sanitize_frame((frame_prefix or "").strip("/"))


def derive_frame_prefix(base_frame: str) -> str:
    parts = sanitize_frame(base_frame).split("/")
    if len(parts) <= 1:
        return ""
    return "/".join(parts[:-1])


def resolve_frame_name(frame_name: str, frame_prefix: str) -> str:
    frame_name = sanitize_frame(frame_name or "")
    frame_prefix = normalize_frame_prefix(frame_prefix)
    if not frame_name:
        return ""
    if not frame_prefix or "/" in frame_name:
        return frame_name
    return f"{frame_prefix}/{frame_name}"


def build_camera_topics(topic_base: str) -> Dict[str, str]:
    topic_base = normalize_topic_base(topic_base)
    if not topic_base:
        return {}
    return {
        "image_topic": f"{topic_base}/image_raw",
        "info_topic": f"{topic_base}/camera_info",
        "depth_topic": f"{topic_base}/depth/image_raw",
    }


def get_camera_profile(camera_profile: str) -> Dict[str, str]:
    if camera_profile not in CAMERA_PROFILES:
        available = ", ".join(sorted(CAMERA_PROFILES))
        raise ValueError(f"Unknown camera profile '{camera_profile}'. Available: {available}")

    profile = dict(CAMERA_PROFILES[camera_profile])
    for key in ("image_topic", "info_topic", "depth_topic", "depth_info_topic"):
        if key in profile:
            profile[key] = normalize_ros_topic(profile[key])
    return profile


def pick_camera_config(config: Dict[str, Any], camera_name: Optional[str]) -> Dict[str, Any]:
    cameras = config.get("cameras") or []
    if not cameras:
        raise ValueError(f"No cameras found in config: {config}")

    if camera_name is None:
        return cameras[0]

    for camera in cameras:
        if camera.get("name") == camera_name:
            return camera
    available = ", ".join(str(camera.get("name")) for camera in cameras)
    raise ValueError(f"Camera '{camera_name}' not found in config. Available: {available}")


def image_msg_to_rgb(msg: Any) -> np.ndarray:
    if isinstance(msg, CompressedImage):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to decode compressed image")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        elif image.ndim == 3:
            image = image[:, :, ::-1].copy()
        return image

    image = ros_numpy.numpify(msg)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Keep the same channel handling as the existing saver pipeline.
    return image[:, :, ::-1].copy()


def depth_msg_to_uint16_mm(msg: Image) -> np.ndarray:
    depth = np.asarray(ros_numpy.numpify(msg))
    encoding = str(getattr(msg, "encoding", ""))

    if encoding == "32FC1":
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        depth_mm = np.round(depth * 1000.0)
        depth_mm[invalid] = 0.0
        return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

    if encoding in {"16UC1", "mono16"}:
        return depth.astype(np.uint16)

    raise ValueError(f"Unsupported depth encoding: {encoding}")


def ros_transform_to_matrix(transform_msg: Any) -> np.ndarray:
    return np.asarray(ros_numpy.numpify(transform_msg.transform), dtype=np.float64)


def ros_to_nerfstudio_c2w(world_T_ros_camera: np.ndarray) -> np.ndarray:
    return world_T_ros_camera @ CAMERA_FRAME_TO_NERFSTUDIO


def matrix_to_translation_quaternion(matrix: np.ndarray) -> Tuple[list[float], list[float]]:
    translation = [float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])]
    quaternion_xyzw = [float(value) for value in quaternion_from_matrix(matrix)]
    return translation, quaternion_xyzw


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def summarize_values(values: list[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    abs_values = [abs(value) for value in values]
    return {
        "mean": float(sum(values) / len(values)),
        "mean_abs": float(sum(abs_values) / len(abs_values)),
        "max_abs": max(abs_values),
        "min": min(values),
        "max": max(values),
    }


def read_json_with_retry(path: Path, retries: int = 5, delay_sec: float = 0.05) -> Optional[Dict[str, Any]]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(delay_sec)

    if last_exc is not None:
        raise last_exc
    return None


def write_json_atomic(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    os.replace(tmp_path, path)


class CameraDatasetRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = make_run_dir(args.output_root)
        self.rgb_dir = self.run_dir / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir = self.run_dir / "depth" if args.depth_topic else None
        if self.depth_dir is not None:
            self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.transforms_path = self.run_dir / "transforms.json"
        self.pose_log_path = self.run_dir / "pose_log.csv"
        self.metadata_path = self.run_dir / "capture_metadata.json"
        self.report_path = self.run_dir / "capture_report.json"

        self.lock = threading.RLock()
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.transforms: Dict[str, Any] = {
            "camera_model": "OPENCV",
            "frames": [],
        }
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(args.tf_cache_sec))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.static_edges: Dict[str, EdgeState] = {}
        self.dynamic_edges: Dict[str, EdgeState] = {}
        self.pose_samples: list[PoseSample] = []
        self.pending_frames: deque[PendingFrame] = deque()
        self.active_camera_frame = args.camera_frame
        self.saved_frames = 0
        self.saved_rgb_seqs: set[int] = set()
        self.saved_depth_seqs: set[int] = set()
        self.saved_pairs: set[Tuple[int, int]] = set()
        self.skipped_blur = 0
        self.skipped_rate = 0
        self.skipped_tf = 0
        self.skipped_duplicate_seq = 0
        self.waiting_for_info_warned = False
        self.last_saved_stamp_sec: Optional[float] = None
        self.first_saved_stamp_sec: Optional[float] = None
        self.last_saved_seq: Optional[int] = None
        self.last_kept_stamp_sec: Optional[float] = None
        self.lookup_counts: Dict[str, int] = {}
        self.depth_lookup_counts: Dict[str, int] = {}
        self.tf_dt_sec_values: list[float] = []
        self.depth_tf_dt_sec_values: list[float] = []
        self.sync_dt_sec_values: list[float] = []
        self.pose_chain_span_sec_values: list[float] = []
        self.depth_pose_chain_span_sec_values: list[float] = []
        self.shutdown_requested = False
        self.camera_frame_mismatch_warned = False
        self.non_optical_camera_frame_warned = False

        self._write_pose_log_header()
        self._write_metadata()
        self._write_transforms()

        image_msg_type = CompressedImage if args.image_topic.endswith("compressed") else Image
        self.info_sub = rospy.Subscriber(args.info_topic, CameraInfo, self._camera_info_cb, queue_size=1)

        if args.depth_topic:
            self.rgb_sub = Subscriber(args.image_topic, image_msg_type)
            self.depth_sub = Subscriber(args.depth_topic, Image)
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub],
                queue_size=args.sync_queue_size,
                slop=args.sync_slop,
            )
            self.sync.registerCallback(self._rgb_depth_cb)
            self.image_sub = None
        else:
            self.image_sub = rospy.Subscriber(
                args.image_topic,
                image_msg_type,
                self._image_cb,
                queue_size=20,
                buff_size=2**24,
            )
            self.rgb_sub = None
            self.depth_sub = None
            self.sync = None

    def _write_pose_log_header(self) -> None:
        with open(self.pose_log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=POSE_LOG_COLUMNS)
            writer.writeheader()

    def _metadata_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": str(self.args.config),
            "output_root": str(self.args.output_root),
            "run_dir": str(self.run_dir),
            "camera_name": self.args.camera_name,
            "camera_profile": self.args.camera_profile,
            "camera_topic_base": self.args.camera_topic_base or None,
            "frame_prefix": self.args.frame_prefix or None,
            "image_topic": self.args.image_topic,
            "info_topic": self.args.info_topic,
            "depth_topic": self.args.depth_topic or None,
            "depth_enabled": bool(self.args.depth_topic),
            "base_frame": self.args.base_frame,
            "camera_frame": self.args.camera_frame,
            "max_images": self.args.max_images,
            "hz": self.args.hz,
            "blur_threshold": self.args.blur_threshold,
            "allow_latest": self.args.allow_latest,
            "pose_matching_mode": "direct_tf_lookup",
            "tf_topic": self.args.tf_topic,
            "tf_static_topic": self.args.tf_static_topic,
            "tf_queue_size": self.args.tf_queue_size,
            "lookup_timeout_sec": self.args.lookup_timeout_sec,
            "tf_cache_sec": self.args.tf_cache_sec,
            "sync_slop": self.args.sync_slop,
            "sync_queue_size": self.args.sync_queue_size,
            "warn_sync_dt_sec": self.args.warn_sync_dt_sec,
            "saved_frames": self.saved_frames,
            "pending_frames": len(self.pending_frames),
            "last_saved_seq": self.last_saved_seq,
            "command": sys.argv,
        }
        if self.intrinsics is not None:
            payload["camera_info"] = asdict(self.intrinsics)
        return payload

    def _write_metadata(self) -> None:
        write_json_atomic(self.metadata_path, self._metadata_payload(), indent=2)

    def _merge_existing_transform_annotations(self, transforms: Dict[str, Any]) -> Dict[str, Any]:
        try:
            existing = read_json_with_retry(self.transforms_path)
        except Exception:
            return transforms

        if not existing:
            return transforms

        existing_frames = {}
        for frame in existing.get("frames", []):
            file_path = frame.get("file_path")
            if file_path:
                existing_frames[file_path] = frame

        for frame in transforms.get("frames", []):
            previous = existing_frames.get(frame.get("file_path", ""))
            if not previous:
                continue
            if "mask_path" in previous and "mask_path" not in frame:
                frame["mask_path"] = previous["mask_path"]

        merged = dict(existing)
        merged.update(transforms)
        for key in ("ply_file_path", "train_filenames", "val_filenames", "test_filenames"):
            if key in existing and key not in transforms:
                merged[key] = existing[key]
        return merged

    def _write_transforms(self) -> None:
        transforms = dict(self.transforms)
        if self.intrinsics is not None:
            transforms.update(self.intrinsics.transforms_fields())
        write_json_atomic(self.transforms_path, self._merge_existing_transform_annotations(transforms), indent=2)

    def _write_report(self) -> None:
        report = {
            "run_dir": str(self.run_dir),
            "saved_frames": self.saved_frames,
            "skipped_blur": self.skipped_blur,
            "skipped_rate": self.skipped_rate,
            "skipped_tf": self.skipped_tf,
            "skipped_duplicate_seq": self.skipped_duplicate_seq,
            "lookup_counts": self.lookup_counts,
            "depth_lookup_counts": self.depth_lookup_counts,
            "first_saved_stamp_sec": self.first_saved_stamp_sec,
            "last_saved_stamp_sec": self.last_saved_stamp_sec,
            "capture_duration_sec": (
                self.last_saved_stamp_sec - self.first_saved_stamp_sec
                if self.first_saved_stamp_sec is not None and self.last_saved_stamp_sec is not None
                else 0.0
            ),
            "tf_dt_sec": summarize_values(self.tf_dt_sec_values),
            "depth_tf_dt_sec": summarize_values(self.depth_tf_dt_sec_values),
            "sync_dt_sec": summarize_values(self.sync_dt_sec_values),
            "pose_chain_span_sec": summarize_values(self.pose_chain_span_sec_values),
            "depth_pose_chain_span_sec": summarize_values(self.depth_pose_chain_span_sec_values),
        }
        write_json_atomic(self.report_path, report, indent=2)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.intrinsics is not None:
            return
        self.intrinsics = CameraIntrinsics.from_msg(msg)
        rospy.loginfo(
            "Camera intrinsics ready: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
            self.intrinsics.width,
            self.intrinsics.height,
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
        )
        self._write_metadata()
        self._write_transforms()

    def _store_edge_locked(self, transform_msg: Any, is_static: bool) -> None:
        parent_frame = sanitize_frame(getattr(transform_msg.header, "frame_id", ""))
        child_frame = sanitize_frame(getattr(transform_msg, "child_frame_id", ""))
        if not parent_frame or not child_frame:
            return

        matrix = ros_transform_to_matrix(transform_msg)
        translation = [
            float(transform_msg.transform.translation.x),
            float(transform_msg.transform.translation.y),
            float(transform_msg.transform.translation.z),
        ]
        quaternion_xyzw = [
            float(transform_msg.transform.rotation.x),
            float(transform_msg.transform.rotation.y),
            float(transform_msg.transform.rotation.z),
            float(transform_msg.transform.rotation.w),
        ]
        edge = EdgeState(
            parent_frame=parent_frame,
            child_frame=child_frame,
            matrix=matrix,
            stamp_sec=float(transform_msg.header.stamp.to_sec()),
            translation=translation,
            quaternion_xyzw=quaternion_xyzw,
            is_static=is_static,
        )
        target = self.static_edges if is_static else self.dynamic_edges
        target[child_frame] = edge

    def _build_pose_sample_locked(self, sample_stamp_sec: float) -> Optional[PoseSample]:
        camera_frame = self.active_camera_frame
        if not camera_frame:
            return None

        chain: list[EdgeState] = []
        dynamic_stamps: list[float] = []
        dynamic_edge_count = 0
        static_edge_count = 0
        current_frame = camera_frame
        visited: set[str] = set()

        while current_frame != self.args.base_frame:
            if current_frame in visited:
                rospy.logwarn_throttle(
                    2.0,
                    "Detected a TF cycle while building %s -> %s pose chain",
                    self.args.base_frame,
                    camera_frame,
                )
                return None
            visited.add(current_frame)

            edge = self.dynamic_edges.get(current_frame)
            if edge is None:
                edge = self.static_edges.get(current_frame)
            if edge is None:
                return None

            chain.append(edge)
            if edge.is_static:
                static_edge_count += 1
            else:
                dynamic_edge_count += 1
                dynamic_stamps.append(edge.stamp_sec)
            current_frame = edge.parent_frame

        world_T_camera = np.eye(4, dtype=np.float64)
        for edge in reversed(chain):
            world_T_camera = world_T_camera @ edge.matrix

        translation, quaternion_xyzw = matrix_to_translation_quaternion(world_T_camera)
        chain_stamp_span_sec = float(max(dynamic_stamps) - min(dynamic_stamps)) if dynamic_stamps else 0.0
        return PoseSample(
            stamp_sec=float(sample_stamp_sec),
            matrix=world_T_camera,
            translation=translation,
            quaternion_xyzw=quaternion_xyzw,
            chain_stamp_span_sec=chain_stamp_span_sec,
            dynamic_edge_count=dynamic_edge_count,
            static_edge_count=static_edge_count,
        )

    def _append_pose_sample_locked(self, sample: PoseSample) -> None:
        sample_stamps = [pose.stamp_sec for pose in self.pose_samples]
        insert_idx = bisect_left(sample_stamps, sample.stamp_sec)
        if insert_idx < len(self.pose_samples) and abs(self.pose_samples[insert_idx].stamp_sec - sample.stamp_sec) < 1e-9:
            self.pose_samples[insert_idx] = sample
        else:
            self.pose_samples.insert(insert_idx, sample)

    def _trim_pose_samples_locked(self) -> None:
        if not self.pose_samples:
            return
        newest_stamp_sec = self.pose_samples[-1].stamp_sec
        while self.pose_samples and (newest_stamp_sec - self.pose_samples[0].stamp_sec) > self.args.tf_cache_sec:
            self.pose_samples.pop(0)

    def _select_pose_sample_locked(self, stamp_sec: float, force: bool) -> Optional[Tuple[PoseSample, str]]:
        if not self.pose_samples:
            return None

        sample_stamps = [pose.stamp_sec for pose in self.pose_samples]
        insert_idx = bisect_left(sample_stamps, stamp_sec)

        if insert_idx < len(self.pose_samples) and abs(self.pose_samples[insert_idx].stamp_sec - stamp_sec) < 1e-9:
            return self.pose_samples[insert_idx], "exact_topic"

        prev_sample = self.pose_samples[insert_idx - 1] if insert_idx > 0 else None
        next_sample = self.pose_samples[insert_idx] if insert_idx < len(self.pose_samples) else None

        if next_sample is not None:
            if prev_sample is None:
                return next_sample, "next_topic"
            prev_dt = abs(prev_sample.stamp_sec - stamp_sec)
            next_dt = abs(next_sample.stamp_sec - stamp_sec)
            if prev_dt <= next_dt:
                return prev_sample, "nearest_prev_topic"
            return next_sample, "nearest_next_topic"

        if prev_sample is not None and (force or self.args.allow_latest):
            if force and not self.args.allow_latest:
                return prev_sample, "prev_only_shutdown"
            return prev_sample, "prev_only_topic"

        return None

    def _tf_static_cb(self, msg: TFMessage) -> None:
        with self.lock:
            for transform_msg in msg.transforms:
                self._store_edge_locked(transform_msg, is_static=True)

    def _tf_cb(self, msg: TFMessage) -> None:
        with self.lock:
            transforms_by_stamp: Dict[float, list[Any]] = {}
            for transform_msg in msg.transforms:
                stamp_sec = float(transform_msg.header.stamp.to_sec())
                transforms_by_stamp.setdefault(stamp_sec, []).append(transform_msg)

            for stamp_sec in sorted(transforms_by_stamp.keys()):
                for transform_msg in transforms_by_stamp[stamp_sec]:
                    self._store_edge_locked(transform_msg, is_static=False)
                pose_sample = self._build_pose_sample_locked(stamp_sec)
                if pose_sample is not None:
                    self._append_pose_sample_locked(pose_sample)

            self._trim_pose_samples_locked()
            self._finalize_pending_frames_locked(force=False)

    def _lookup_pose_direct_locked(self, stamp_sec: float, camera_frame: str) -> Tuple[PoseSample, str]:
        requested_time = rospy.Time.from_sec(float(stamp_sec))
        tf_msg = None
        lookup_mode = "exact_tf"
        lookup_error: Optional[Exception] = None

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                camera_frame,
                requested_time,
                timeout=rospy.Duration(self.args.lookup_timeout_sec),
            )
        except Exception as exc:
            lookup_error = exc
            if not self.args.allow_latest:
                raise

        if tf_msg is None:
            tf_msg = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                camera_frame,
                rospy.Time(0),
                timeout=rospy.Duration(self.args.lookup_timeout_sec),
            )
            lookup_mode = "latest_tf"
            if lookup_error is not None:
                rospy.logwarn_throttle(
                    2.0,
                    "Exact TF lookup failed for %s -> %s at %.6f; falling back to latest TF: %s",
                    self.args.base_frame,
                    camera_frame,
                    stamp_sec,
                    lookup_error,
                )

        world_T_camera = ros_transform_to_matrix(tf_msg)
        translation, quaternion_xyzw = matrix_to_translation_quaternion(world_T_camera)
        resolved_stamp_sec = float(tf_msg.header.stamp.to_sec())
        if resolved_stamp_sec <= 0.0:
            resolved_stamp_sec = float(stamp_sec)

        return (
            PoseSample(
                stamp_sec=resolved_stamp_sec,
                matrix=world_T_camera,
                translation=translation,
                quaternion_xyzw=quaternion_xyzw,
                chain_stamp_span_sec=0.0,
                dynamic_edge_count=0,
                static_edge_count=0,
            ),
            lookup_mode,
        )

    def _should_keep_frame(self, rgb_stamp_sec: float, rgb_seq: int, depth_seq: Optional[int]) -> bool:
        with self.lock:
            if (self.saved_frames + len(self.pending_frames)) >= self.args.max_images:
                self.request_shutdown("Reached requested number of images")
                return False

            if rgb_seq in self.saved_rgb_seqs:
                self.skipped_duplicate_seq += 1
                return False

            if depth_seq is not None and depth_seq in self.saved_depth_seqs:
                self.skipped_duplicate_seq += 1
                return False

            if depth_seq is not None and (rgb_seq, depth_seq) in self.saved_pairs:
                self.skipped_duplicate_seq += 1
                return False

            if self.args.hz > 0.0 and self.last_kept_stamp_sec is not None:
                if (rgb_stamp_sec - self.last_kept_stamp_sec) < (1.0 / self.args.hz):
                    self.skipped_rate += 1
                    return False

            return True

    def _append_pose_log(self, row: Dict[str, Any]) -> None:
        with open(self.pose_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=POSE_LOG_COLUMNS)
            writer.writerow({key: row.get(key, "") for key in POSE_LOG_COLUMNS})

    def _image_cb(self, msg: Any) -> None:
        self._save_frame(msg, None)

    def _rgb_depth_cb(self, rgb_msg: Any, depth_msg: Image) -> None:
        self._save_frame(rgb_msg, depth_msg)

    def _save_frame(self, rgb_msg: Any, depth_msg: Optional[Image]) -> None:
        if self.shutdown_requested:
            return

        if self.intrinsics is None:
            if not self.waiting_for_info_warned:
                rospy.logwarn("Waiting for CameraInfo before saving frames...")
                self.waiting_for_info_warned = True
            return

        rgb_seq = int(getattr(rgb_msg.header, "seq", 0))
        rgb_stamp_sec = rgb_msg.header.stamp.to_sec()
        depth_seq = int(getattr(depth_msg.header, "seq", 0)) if depth_msg is not None else None
        depth_stamp_sec = depth_msg.header.stamp.to_sec() if depth_msg is not None else None

        if not self._should_keep_frame(rgb_stamp_sec, rgb_seq, depth_seq):
            return

        try:
            rgb_image = image_msg_to_rgb(rgb_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to decode RGB image: %s", exc)
            return

        blur_score = float(cv2.Laplacian(rgb_image, cv2.CV_64F).var())
        if blur_score < self.args.blur_threshold:
            self.skipped_blur += 1
            return

        depth_mm = None
        sync_dt_sec = None
        if depth_msg is not None:
            try:
                depth_mm = depth_msg_to_uint16_mm(depth_msg)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "Failed to decode depth image: %s", exc)
                return
            sync_dt_sec = float(depth_stamp_sec - rgb_stamp_sec)
            if abs(sync_dt_sec) > self.args.warn_sync_dt_sec:
                rospy.logwarn(
                    "RGB-depth sync offset %.3f ms for rgb_seq=%d depth_seq=%d",
                    sync_dt_sec * 1000.0,
                    rgb_seq,
                    depth_seq,
                )

        rgb_frame_id = sanitize_frame(getattr(rgb_msg.header, "frame_id", ""))
        depth_frame_id = sanitize_frame(getattr(depth_msg.header, "frame_id", "")) if depth_msg is not None else ""
        configured_camera_frame = self.args.camera_frame
        header_camera_frame = rgb_frame_id or depth_frame_id
        camera_frame = configured_camera_frame or header_camera_frame
        if not camera_frame:
            rospy.logwarn_throttle(2.0, "Image header.frame_id is empty; set --camera-frame explicitly.")
            return

        if self.args.camera_frame_explicit:
            if self.args.camera_frame and rgb_frame_id and self.args.camera_frame != rgb_frame_id and not self.camera_frame_mismatch_warned:
                rospy.logwarn(
                    "Explicit camera_frame=%s differs from RGB header.frame_id=%s; using the explicit override.",
                    self.args.camera_frame,
                    rgb_frame_id,
                )
                self.camera_frame_mismatch_warned = True
        elif header_camera_frame:
            if configured_camera_frame and configured_camera_frame != header_camera_frame and not self.camera_frame_mismatch_warned:
                rospy.logwarn(
                    "Default camera_frame=%s differs from image header.frame_id=%s; using the image frame so poses match the captured sensor.",
                    configured_camera_frame,
                    header_camera_frame,
                )
                self.camera_frame_mismatch_warned = True
            camera_frame = header_camera_frame

        if "optical" not in camera_frame and not self.non_optical_camera_frame_warned:
            rospy.logwarn(
                "Camera frame %s is not the Nerfstudio-aligned optical frame expected by transform_matrix and may be misaligned.",
                camera_frame,
            )
            self.non_optical_camera_frame_warned = True

        pending_frame = PendingFrame(
            rgb_seq=rgb_seq,
            depth_seq=depth_seq,
            rgb_stamp_sec=float(rgb_stamp_sec),
            depth_stamp_sec=float(depth_stamp_sec) if depth_stamp_sec is not None else None,
            rgb_frame_id=rgb_frame_id,
            depth_frame_id=depth_frame_id,
            camera_frame=camera_frame,
            rgb_image=rgb_image,
            depth_mm=depth_mm,
            blur_score=blur_score,
            sync_dt_sec=sync_dt_sec,
        )

        with self.lock:
            if not self.active_camera_frame:
                self.active_camera_frame = camera_frame
            try:
                rgb_pose, rgb_lookup_mode = self._lookup_pose_direct_locked(
                    pending_frame.rgb_stamp_sec,
                    pending_frame.camera_frame,
                )
                depth_pose = None
                depth_lookup_mode = ""
                if pending_frame.depth_stamp_sec is not None:
                    depth_pose, depth_lookup_mode = self._lookup_pose_direct_locked(
                        pending_frame.depth_stamp_sec,
                        pending_frame.camera_frame,
                    )
            except Exception as exc:
                self.skipped_tf += 1
                rospy.logwarn_throttle(
                    2.0,
                    "Skipping frame rgb_seq=%s depth_seq=%s because TF lookup failed: %s",
                    pending_frame.rgb_seq,
                    pending_frame.depth_seq,
                    exc,
                )
                return

            self._commit_frame_locked(
                pending_frame,
                rgb_pose,
                rgb_lookup_mode,
                depth_pose,
                depth_lookup_mode,
            )

    def _finalize_pending_frames_locked(self, force: bool) -> None:
        while self.pending_frames:
            pending_frame = self.pending_frames[0]
            rgb_selection = self._select_pose_sample_locked(pending_frame.rgb_stamp_sec, force=force)
            if rgb_selection is None:
                break

            depth_selection: Optional[Tuple[PoseSample, str]] = None
            if pending_frame.depth_stamp_sec is not None:
                depth_selection = self._select_pose_sample_locked(pending_frame.depth_stamp_sec, force=force)
                if depth_selection is None:
                    break

            self.pending_frames.popleft()
            rgb_pose, rgb_lookup_mode = rgb_selection
            depth_pose = None
            depth_lookup_mode = ""
            if depth_selection is not None:
                depth_pose, depth_lookup_mode = depth_selection
            self._commit_frame_locked(pending_frame, rgb_pose, rgb_lookup_mode, depth_pose, depth_lookup_mode)

    def _commit_frame_locked(
        self,
        pending_frame: PendingFrame,
        rgb_pose: PoseSample,
        rgb_lookup_mode: str,
        depth_pose: Optional[PoseSample],
        depth_lookup_mode: str,
    ) -> None:
        rgb_raw_ros_c2w = rgb_pose.matrix
        rgb_ns_c2w = ros_to_nerfstudio_c2w(rgb_raw_ros_c2w)
        rgb_translation = rgb_pose.translation
        rgb_quaternion_xyzw = rgb_pose.quaternion_xyzw
        rgb_tf_timestamp_sec = rgb_pose.stamp_sec
        rgb_tf_dt_sec = float(rgb_tf_timestamp_sec - pending_frame.rgb_stamp_sec)

        depth_raw_ros_c2w = None
        depth_ns_c2w = None
        depth_translation = None
        depth_quaternion_xyzw = None
        depth_tf_timestamp_sec = None
        depth_tf_dt_sec = None
        if depth_pose is not None:
            depth_raw_ros_c2w = depth_pose.matrix
            depth_ns_c2w = ros_to_nerfstudio_c2w(depth_raw_ros_c2w)
            depth_translation = depth_pose.translation
            depth_quaternion_xyzw = depth_pose.quaternion_xyzw
            depth_tf_timestamp_sec = depth_pose.stamp_sec
            depth_tf_dt_sec = float(depth_tf_timestamp_sec - float(pending_frame.depth_stamp_sec))

        rgb_name = f"{self.args.camera_name}_{pending_frame.rgb_seq:05d}.png"
        rgb_path = self.rgb_dir / rgb_name
        rgb_rel_path = f"./rgb/{rgb_name}"
        if not cv2.imwrite(str(rgb_path), pending_frame.rgb_image):
            rospy.logwarn("Failed to write RGB image %s", rgb_path)
            self.skipped_tf += 1
            return

        depth_rel_path = None
        if pending_frame.depth_mm is not None and self.depth_dir is not None and pending_frame.depth_seq is not None:
            depth_name = f"{self.args.camera_name}_{pending_frame.depth_seq:05d}.tiff"
            depth_path = self.depth_dir / depth_name
            depth_rel_path = f"./depth/{depth_name}"
            if not cv2.imwrite(str(depth_path), pending_frame.depth_mm):
                rospy.logwarn("Failed to write depth image %s", depth_path)
                self.skipped_tf += 1
                return

        frame_index = self.saved_frames
        frame_record = {
            "file_path": rgb_rel_path,
            "transform_matrix": rgb_ns_c2w.tolist(),
            "ros_transform_matrix": rgb_raw_ros_c2w.tolist(),
            "rgb_timestamp_sec": pending_frame.rgb_stamp_sec,
            "rgb_tf_timestamp_sec": rgb_tf_timestamp_sec,
            "tf_timestamp_sec": rgb_tf_timestamp_sec,
            "rgb_tf_dt_sec": rgb_tf_dt_sec,
            "tf_dt_sec": rgb_tf_dt_sec,
            "rgb_tf_lookup_mode": rgb_lookup_mode,
            "tf_lookup_mode": rgb_lookup_mode,
            "rgb_pose_chain_span_sec": rgb_pose.chain_stamp_span_sec,
            "seq": pending_frame.rgb_seq,
            "rgb_seq": pending_frame.rgb_seq,
            "image_frame_id": pending_frame.rgb_frame_id,
            "camera_frame": pending_frame.camera_frame,
            "base_frame": self.args.base_frame,
            "blur_score": pending_frame.blur_score,
            "ros_translation_xyz": rgb_translation,
            "ros_quaternion_xyzw": rgb_quaternion_xyzw,
            "pose_source": "tf_lookup",
            "rgb_dynamic_edge_count": rgb_pose.dynamic_edge_count,
            "rgb_static_edge_count": rgb_pose.static_edge_count,
        }
        if depth_rel_path is not None:
            frame_record.update(
                {
                    "depth_file_path": depth_rel_path,
                    "depth_transform_matrix": depth_ns_c2w.tolist(),
                    "ros_depth_transform_matrix": depth_raw_ros_c2w.tolist(),
                    "depth_timestamp_sec": pending_frame.depth_stamp_sec,
                    "depth_tf_timestamp_sec": depth_tf_timestamp_sec,
                    "depth_tf_dt_sec": depth_tf_dt_sec,
                    "depth_tf_lookup_mode": depth_lookup_mode,
                    "depth_pose_chain_span_sec": depth_pose.chain_stamp_span_sec if depth_pose is not None else 0.0,
                    "sync_dt_sec": pending_frame.sync_dt_sec,
                    "depth_seq": pending_frame.depth_seq,
                    "depth_frame_id": pending_frame.depth_frame_id,
                    "depth_ros_translation_xyz": depth_translation,
                    "depth_ros_quaternion_xyzw": depth_quaternion_xyzw,
                    "depth_dynamic_edge_count": depth_pose.dynamic_edge_count if depth_pose is not None else 0,
                    "depth_static_edge_count": depth_pose.static_edge_count if depth_pose is not None else 0,
                }
            )

        self.transforms["frames"].append(frame_record)
        self._append_pose_log(
            {
                "index": frame_index,
                "rgb_seq": pending_frame.rgb_seq,
                "depth_seq": pending_frame.depth_seq if pending_frame.depth_seq is not None else "",
                "file_path": rgb_rel_path,
                "depth_file_path": depth_rel_path or "",
                "rgb_timestamp_sec": pending_frame.rgb_stamp_sec,
                "depth_timestamp_sec": pending_frame.depth_stamp_sec if pending_frame.depth_stamp_sec is not None else "",
                "sync_dt_sec": pending_frame.sync_dt_sec if pending_frame.sync_dt_sec is not None else "",
                "rgb_tf_timestamp_sec": rgb_tf_timestamp_sec,
                "depth_tf_timestamp_sec": depth_tf_timestamp_sec if depth_tf_timestamp_sec is not None else "",
                "rgb_tf_dt_sec": rgb_tf_dt_sec,
                "depth_tf_dt_sec": depth_tf_dt_sec if depth_tf_dt_sec is not None else "",
                "rgb_tf_lookup_mode": rgb_lookup_mode,
                "depth_tf_lookup_mode": depth_lookup_mode,
                "rgb_pose_chain_span_sec": rgb_pose.chain_stamp_span_sec,
                "depth_pose_chain_span_sec": depth_pose.chain_stamp_span_sec if depth_pose is not None else "",
                "blur_score": pending_frame.blur_score,
                "rgb_tx": rgb_translation[0],
                "rgb_ty": rgb_translation[1],
                "rgb_tz": rgb_translation[2],
                "rgb_qx": rgb_quaternion_xyzw[0],
                "rgb_qy": rgb_quaternion_xyzw[1],
                "rgb_qz": rgb_quaternion_xyzw[2],
                "rgb_qw": rgb_quaternion_xyzw[3],
                "depth_tx": depth_translation[0] if depth_translation is not None else "",
                "depth_ty": depth_translation[1] if depth_translation is not None else "",
                "depth_tz": depth_translation[2] if depth_translation is not None else "",
                "depth_qx": depth_quaternion_xyzw[0] if depth_quaternion_xyzw is not None else "",
                "depth_qy": depth_quaternion_xyzw[1] if depth_quaternion_xyzw is not None else "",
                "depth_qz": depth_quaternion_xyzw[2] if depth_quaternion_xyzw is not None else "",
                "depth_qw": depth_quaternion_xyzw[3] if depth_quaternion_xyzw is not None else "",
            }
        )

        self.saved_rgb_seqs.add(pending_frame.rgb_seq)
        if pending_frame.depth_seq is not None:
            self.saved_depth_seqs.add(pending_frame.depth_seq)
            self.saved_pairs.add((pending_frame.rgb_seq, pending_frame.depth_seq))

        self.saved_frames += 1
        self.lookup_counts[rgb_lookup_mode] = self.lookup_counts.get(rgb_lookup_mode, 0) + 1
        if depth_lookup_mode:
            self.depth_lookup_counts[depth_lookup_mode] = self.depth_lookup_counts.get(depth_lookup_mode, 0) + 1
        self.tf_dt_sec_values.append(rgb_tf_dt_sec)
        self.pose_chain_span_sec_values.append(rgb_pose.chain_stamp_span_sec)
        if depth_tf_dt_sec is not None:
            self.depth_tf_dt_sec_values.append(depth_tf_dt_sec)
        if depth_pose is not None:
            self.depth_pose_chain_span_sec_values.append(depth_pose.chain_stamp_span_sec)
        if pending_frame.sync_dt_sec is not None:
            self.sync_dt_sec_values.append(pending_frame.sync_dt_sec)
        self.last_saved_stamp_sec = pending_frame.rgb_stamp_sec
        self.last_saved_seq = pending_frame.rgb_seq
        self.last_kept_stamp_sec = pending_frame.rgb_stamp_sec
        if self.first_saved_stamp_sec is None:
            self.first_saved_stamp_sec = pending_frame.rgb_stamp_sec

        self._write_transforms()
        self._write_metadata()
        self._write_report()

        if pending_frame.depth_seq is None:
            rospy.loginfo(
                "Saved RGB frame %03d/%03d seq=%d rgb=%.6f pose=%.6f dt=%+.6f mode=%s blur=%.2f",
                self.saved_frames,
                self.args.max_images,
                pending_frame.rgb_seq,
                pending_frame.rgb_stamp_sec,
                rgb_tf_timestamp_sec,
                rgb_tf_dt_sec,
                rgb_lookup_mode,
                pending_frame.blur_score,
            )
        else:
            rospy.loginfo(
                "Saved RGBD frame %03d/%03d rgb_seq=%d depth_seq=%d rgb=%.6f depth=%.6f sync=%+.6f rgb_pose=%+.6f depth_pose=%+.6f modes=%s/%s",
                self.saved_frames,
                self.args.max_images,
                pending_frame.rgb_seq,
                pending_frame.depth_seq,
                pending_frame.rgb_stamp_sec,
                pending_frame.depth_stamp_sec,
                pending_frame.sync_dt_sec,
                rgb_tf_dt_sec,
                depth_tf_dt_sec,
                rgb_lookup_mode,
                depth_lookup_mode,
            )

        if self.saved_frames >= self.args.max_images:
            self.request_shutdown("Reached requested number of images")

    def request_shutdown(self, reason: str) -> None:
        with self.lock:
            if self.shutdown_requested:
                return
            self.shutdown_requested = True
            if self.pending_frames:
                dropped = len(self.pending_frames)
                self.skipped_tf += dropped
                self.pending_frames.clear()
                rospy.logwarn("Dropped %d pending frame(s) during shutdown.", dropped)
            self._write_transforms()
            self._write_metadata()
            self._write_report()
            rospy.loginfo("Capture finished: %s", reason)
            rospy.loginfo("Dataset run written to %s", self.run_dir)
        rospy.signal_shutdown(reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save images and camera poses from ROS1 into a new run folder under dynaarm_gs_depth_mask_01, "
            "with optional synchronized depth and direct-tf-lookup timing diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file to read defaults from.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Dataset root. A timestamped run folder is created inside it.")
    parser.add_argument("--camera-name", default=None, help="Camera entry name in the config. Defaults to the first camera.")
    parser.add_argument(
        "--camera-profile",
        choices=sorted(CAMERA_PROFILES.keys()),
        default="manual",
        help="Named camera preset that fills in topic names and camera frame automatically.",
    )
    parser.add_argument(
        "--camera-topic-base",
        default=None,
        help=(
            "Optional camera topic base used to derive topics automatically, for example "
            f"'{DEFAULT_CAMERA_TOPIC_BASE}' or '/camera'."
        ),
    )
    parser.add_argument("--image-topic", default=None, help="Override RGB image topic.")
    parser.add_argument("--info-topic", default=None, help="Override CameraInfo topic.")
    parser.add_argument("--depth-topic", default=None, help="Override depth image topic. Leave empty to run RGB-only mode.")
    parser.add_argument("--base-frame", default=None, help="Override base/world TF frame.")
    parser.add_argument(
        "--frame-prefix",
        default=None,
        help="Prefix applied to profile camera frames. Defaults to the namespace of --base-frame, for example 'dynaarm_arm_tf'.",
    )
    parser.add_argument("--camera-frame", default=None, help="Override camera TF frame.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of frames to save. Defaults to config num_images.")
    parser.add_argument("--hz", type=float, default=None, help="Maximum saved frame rate. Defaults to config hz.")
    parser.add_argument("--blur-threshold", type=float, default=None, help="Minimum Laplacian variance required to keep a frame.")
    parser.add_argument(
        "--allow-latest",
        action="store_true",
        help="If exact tf lookup fails, fall back to the latest available tf transform.",
    )
    parser.add_argument(
        "--lookup-timeout-sec",
        type=float,
        default=0.2,
        help="Timeout in seconds for each tf lookup.",
    )
    parser.add_argument("--tf-cache-sec", type=float, default=120.0, help="Seconds of tf history to keep in the tf2 buffer.")
    parser.add_argument("--tf-topic", default="/tf", help="Deprecated compatibility option. Exact tf lookup uses the default tf2 listener topics.")
    parser.add_argument("--tf-static-topic", default="/tf_static", help="Deprecated compatibility option. Exact tf lookup uses the default tf2 listener topics.")
    parser.add_argument("--tf-queue-size", type=int, default=200, help="Compatibility option kept for older saver configurations.")
    parser.add_argument("--sync-slop", type=float, default=0.1, help="ApproximateTimeSynchronizer slop in seconds for RGB-depth pairing.")
    parser.add_argument("--sync-queue-size", type=int, default=20, help="ApproximateTimeSynchronizer queue size.")
    parser.add_argument("--warn-sync-dt-sec", type=float, default=0.01, help="Warn when |depth_time - rgb_time| exceeds this threshold.")
    args = parser.parse_args()
    args.camera_frame_explicit = args.camera_frame is not None

    config = read_config(args.config.expanduser().resolve())
    camera_cfg = pick_camera_config(config, args.camera_name)
    profile_overrides = get_camera_profile(args.camera_profile)

    args.config = args.config.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.camera_topic_base = normalize_topic_base(args.camera_topic_base)
    topic_overrides = build_camera_topics(args.camera_topic_base)
    args.camera_name = args.camera_name or str(camera_cfg.get("name", "camera"))
    args.image_topic = args.image_topic or profile_overrides.get(
        "image_topic",
        topic_overrides.get("image_topic", str(camera_cfg.get("image_topic", ""))),
    )
    args.info_topic = args.info_topic or profile_overrides.get(
        "info_topic",
        topic_overrides.get("info_topic", str(camera_cfg.get("info_topic", ""))),
    )

    depth_cfg = camera_cfg.get("depth_topic", "")
    if args.depth_topic is None:
        args.depth_topic = profile_overrides.get("depth_topic", topic_overrides.get("depth_topic", str(depth_cfg)))
    elif args.depth_topic == "":
        args.depth_topic = ""

    args.base_frame = sanitize_frame(args.base_frame or str(config.get("base_frame", "world")))
    args.frame_prefix = normalize_frame_prefix(args.frame_prefix or derive_frame_prefix(args.base_frame))
    args.camera_frame = sanitize_frame(
        resolve_frame_name(
            args.camera_frame or profile_overrides.get("camera_frame", str(camera_cfg.get("camera_frame", ""))),
            args.frame_prefix,
        )
    )
    args.max_images = int(args.max_images if args.max_images is not None else config.get("num_images", 150))
    args.hz = float(args.hz if args.hz is not None else config.get("hz", 5.0))
    args.blur_threshold = float(
        args.blur_threshold if args.blur_threshold is not None else config.get("blur_threshold", 0.0)
    )
    args.image_topic = normalize_ros_topic(args.image_topic)
    args.info_topic = normalize_ros_topic(args.info_topic)
    args.depth_topic = normalize_ros_topic(args.depth_topic) if args.depth_topic else ""

    if not args.image_topic:
        parser.error("No image topic provided and config did not define one.")
    if not args.info_topic:
        parser.error("No info topic provided and config did not define one.")
    if args.max_images <= 0:
        parser.error("--max-images must be > 0")
    if args.hz < 0.0:
        parser.error("--hz must be >= 0")
    if args.blur_threshold < 0.0:
        parser.error("--blur-threshold must be >= 0")
    if args.lookup_timeout_sec < 0.0:
        parser.error("--lookup-timeout-sec must be >= 0")
    if args.tf_cache_sec < 0.0:
        parser.error("--tf-cache-sec must be >= 0")
    if args.tf_queue_size < 1:
        parser.error("--tf-queue-size must be >= 1")
    if args.sync_slop < 0.0:
        parser.error("--sync-slop must be >= 0")
    if args.sync_queue_size < 1:
        parser.error("--sync-queue-size must be >= 1")
    if args.warn_sync_dt_sec < 0.0:
        parser.error("--warn-sync-dt-sec must be >= 0")

    return args


def main() -> int:
    args = parse_args()
    rospy.init_node("save_camera_poses_and_images", anonymous=True)

    recorder = CameraDatasetRecorder(args)

    def _shutdown_handler(sig: int, _frame: Any) -> None:
        recorder.request_shutdown(f"Signal {sig} received")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    rospy.loginfo("Writing dataset run to %s", recorder.run_dir)
    rospy.loginfo("Camera profile: %s", args.camera_profile)
    if args.camera_topic_base:
        rospy.loginfo("Camera topic base: %s", args.camera_topic_base)
    if args.frame_prefix:
        rospy.loginfo("Frame prefix: %s", args.frame_prefix)
    rospy.loginfo("Image topic: %s", args.image_topic)
    rospy.loginfo("CameraInfo topic: %s", args.info_topic)
    rospy.loginfo("Depth topic: %s", args.depth_topic or "<disabled>")
    rospy.loginfo("Base frame: %s", args.base_frame)
    rospy.loginfo("Camera frame: %s", args.camera_frame or "<from image header>")
    rospy.loginfo("Pose source: direct tf lookup at image/depth timestamp")
    rospy.loginfo("Allow latest-tf fallback: %s", args.allow_latest)
    rospy.loginfo("TF lookup timeout: %.3f s", args.lookup_timeout_sec)
    rospy.loginfo("Target save rate: %.3f Hz", args.hz)
    rospy.loginfo("Max images: %d", args.max_images)
    if args.depth_topic:
        rospy.loginfo("RGB-depth sync slop: %.3f s", args.sync_slop)

    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
