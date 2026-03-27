#!/usr/bin/env python3

import datetime
from bisect import bisect_left
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
DEPTH_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw"
CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
WORLD_FRAME = "dynaarm_arm_tf/world"
CAMERA_POSE_SAVE_FRAME = "dynaarm_arm_tf/camera_link_optical"
GAZEBO_CAMERA_POSE_FRAME = "dynaarm_arm_tf/camera_pose_link"
DATASET_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/"
    "datasets/dynaarm_gs_depth_mask_01"
)
MASK_SAVER_SCRIPT = Path(__file__).with_name("ros1_robot_mask_saver_stl.py")
MASK_SAVER_READY_SENTINEL = DATASET_ROOT / ".robot_mask_saver_ready"
CAPTURE_COMPLETE_SENTINEL_NAME = ".capture_complete.json"
SAVE_HZ = 5.0
MAX_IMAGES = 60
TF_TIMEOUT = 0.2
IMAGE_NAME_PREFIX = "arm"
MASK_SAVER_READY_TIMEOUT_SEC = 20.0
MASK_SAVER_EXIT_TIMEOUT_SEC = 180.0
SYNC_QUEUE_SIZE = 20
SYNC_SLOP_SEC = 0.1
INIT_CLOUD_NAME = "depth_camera_init_points.ply"
MAX_INIT_CLOUD_POINTS = 400000
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
GAZEBO_LINK_STATES_TOPIC = "/gazebo/link_states"
GAZEBO_CAMERA_POSE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose"
GAZEBO_WORLD_FRAME = "world"
POSE_DEBUG_NAME = "pose_debug.json"
GAZEBO_TRANSFORMS_NAME = "transform_gazebo.json"
GAZEBO_POSE_DIFF_NAME = "pose_difference_gazebo_vs_tf.json"
TF_TRANSFORMS_NAME = "transforms_tf.json"
TF_TIME_EPS_SEC = 1e-6
MAX_SKIPPED_NO_TF_EVENTS = 200
MAX_GAZEBO_HISTORY_SEC = 30.0
PRIMARY_POSE_SOURCE = "gazebo_aligned_first_tf_frame"
SECONDARY_TF_POSE_SOURCE = "tf_camera_link_optical"

# ROS optical / OpenCV camera frame:
#   +X right, +Y down, +Z forward
#
# Nerfstudio / OpenGL camera frame:
#   +X right, +Y up, +Z back
ROS_OPTICAL_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0])

ROS_CAMERA_TO_OUTPUT_FRAME = ROS_OPTICAL_TO_NERFSTUDIO 

tf_buffer = None
tf_listener = None
mask_process = None
run_dir = None
rgb_dir = None
depth_dir = None
masks_dir = None
transforms_path = None
transforms_lock_path = None
capture_complete_path = None
meta = None
meta_tf = None
frame_index = 0
last_saved_stamp = None
pose_debug_path = None
gazebo_transforms_path = None
gazebo_pose_diff_path = None
transforms_tf_path = None
transforms_tf_lock_path = None
tf_edge_samples = None
tf_parent_by_child = None
pose_debug_records = None
skipped_no_tf_events = None
capture_stats = None
run_completed = False
gazebo_link_histories = None
warned_unexpected_image_frame = False
gazebo_alignment_transform = None
gazebo_alignment_metadata = None


def ros_image_to_bgr(msg):
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


def ros_depth_to_uint16_mm(msg):
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


def transform_stamped_to_matrix(tf_msg)->quaternion_matrix:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def normalize_frame_id(frame_id):
    return str(frame_id).lstrip("/")


def pose_msg_to_matrix(pose_msg):
    q = pose_msg.orientation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0, 3] = pose_msg.position.x
    T[1, 3] = pose_msg.position.y
    T[2, 3] = pose_msg.position.z
    return T


def compose_transform_matrix(translation_xyz, quaternion_xyzw):
    T = quaternion_matrix(quaternion_xyzw)
    T[0, 3] = translation_xyz[0]
    T[1, 3] = translation_xyz[1]
    T[2, 3] = translation_xyz[2]
    return T


def invert_rigid_transform(T):
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rotation_angle_deg(R_a, R_b):
    R_rel = R_a.T @ R_b
    trace_value = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_value)))


def write_ascii_ply(path, xyz, rgb):
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


def resolve_relpath(base_dir, rel_path):
    return (base_dir / rel_path).resolve()


def resolve_frame_mask_path(dataset_dir, frame):
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


def load_saved_mask(mask_path, expected_hw):
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


def load_saved_rgb(rgb_path, expected_hw):
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read RGB image from {rgb_path}")
    if image_bgr.shape[:2] != expected_hw:
        raise RuntimeError(
            f"RGB shape mismatch for {rgb_path}: got {image_bgr.shape[:2]}, expected {expected_hw}"
        )
    return image_bgr


def gazebo_link_name_to_tf_frame(link_name):
    if "::" not in link_name:
        return None
    model_name, link_suffix = link_name.split("::", 1)
    if model_name != "Dynaarm_Arm":
        return None
    return f"dynaarm_arm_tf/{link_suffix}"


def gazebo_link_states_callback(msg):
    for link_name, pose_msg in zip(msg.name, msg.pose):
        tf_frame = gazebo_link_name_to_tf_frame(link_name)
        if tf_frame is None:
            continue
        if tf_frame == GAZEBO_CAMERA_POSE_FRAME:
            continue

        stamp_sec = float(rospy.Time.now().to_sec())

        entry = gazebo_link_histories.setdefault(
            tf_frame,
            {
                "gazebo_link_name": link_name,
                "reference_frame": GAZEBO_WORLD_FRAME,
                "times_sec": [],
                "matrices": [],
            },
        )
        times_sec = entry["times_sec"]
        matrices = entry["matrices"]

        if times_sec and abs(times_sec[-1] - stamp_sec) <= TF_TIME_EPS_SEC:
            continue

        times_sec.append(stamp_sec)
        matrices.append(pose_msg_to_matrix(pose_msg))

        while len(times_sec) > 1 and stamp_sec - times_sec[0] > MAX_GAZEBO_HISTORY_SEC:
            times_sec.pop(0)
            matrices.pop(0)


def gazebo_camera_pose_callback(msg):
    stamp_sec = float(msg.header.stamp.to_sec())
    if stamp_sec <= 0.0:
        return

    tf_frame = GAZEBO_CAMERA_POSE_FRAME
    entry = gazebo_link_histories.setdefault(
        tf_frame,
        {
            "gazebo_link_name": GAZEBO_CAMERA_POSE_TOPIC,
            "reference_frame": normalize_frame_id(msg.header.frame_id),
            "times_sec": [],
            "matrices": [],
        },
    )
    times_sec = entry["times_sec"]
    matrices = entry["matrices"]

    if times_sec and abs(times_sec[-1] - stamp_sec) <= TF_TIME_EPS_SEC:
        return

    times_sec.append(stamp_sec)
    matrices.append(pose_msg_to_matrix(msg.pose))

    while len(times_sec) > 1 and stamp_sec - times_sec[0] > MAX_GAZEBO_HISTORY_SEC:
        times_sec.pop(0)
        matrices.pop(0)


def tf_message_callback(msg, is_static):
    for transform in msg.transforms:
        parent = normalize_frame_id(transform.header.frame_id)
        child = normalize_frame_id(transform.child_frame_id)
        if not parent or not child:
            continue

        tf_parent_by_child[child] = parent
        entry = tf_edge_samples.setdefault(
            (parent, child),
            {
                "static": is_static,
                "times_sec": [],
            },
        )
        if is_static:
            entry["static"] = True

        stamp_sec = float(transform.header.stamp.to_sec())
        times_sec = entry["times_sec"]
        insert_at = bisect_left(times_sec, stamp_sec)
        if insert_at < len(times_sec) and abs(times_sec[insert_at] - stamp_sec) <= TF_TIME_EPS_SEC:
            continue
        times_sec.insert(insert_at, stamp_sec)


def tf_dynamic_callback(msg):
    tf_message_callback(msg, is_static=False)


def tf_static_callback(msg):
    tf_message_callback(msg, is_static=True)


def resolve_tf_chain(source_frame, target_frame):
    source_frame = normalize_frame_id(source_frame)
    target_frame = normalize_frame_id(target_frame)
    chain = []
    seen = set()
    current = source_frame

    while current != target_frame:
        if current in seen:
            return chain, False, current
        seen.add(current)
        parent = tf_parent_by_child.get(current)
        if parent is None:
            return chain, False, current
        chain.append((parent, current))
        current = parent

    return chain, True, target_frame


def describe_tf_edge(parent, child, query_time_sec):
    entry = tf_edge_samples.get((parent, child))
    info = {
        "parent_frame": parent,
        "child_frame": child,
        "query_time_sec": float(query_time_sec),
    }

    if entry is None:
        info["status"] = "missing_from_debug_cache"
        return info

    times_sec = entry["times_sec"]
    info["static"] = bool(entry["static"])
    info["sample_count"] = len(times_sec)
    if times_sec:
        info["first_sample_time_sec"] = float(times_sec[0])
        info["last_sample_time_sec"] = float(times_sec[-1])

    if entry["static"]:
        info["status"] = "static"
        info["interpolated"] = False
        return info

    insert_at = bisect_left(times_sec, query_time_sec)
    prev_time_sec = times_sec[insert_at - 1] if insert_at > 0 else None
    next_time_sec = times_sec[insert_at] if insert_at < len(times_sec) else None

    if prev_time_sec is not None:
        info["previous_sample_time_sec"] = float(prev_time_sec)
        info["previous_sample_dt_sec"] = float(query_time_sec - prev_time_sec)
    if next_time_sec is not None:
        info["next_sample_time_sec"] = float(next_time_sec)
        info["next_sample_dt_sec"] = float(next_time_sec - query_time_sec)

    prev_exact = prev_time_sec is not None and abs(query_time_sec - prev_time_sec) <= TF_TIME_EPS_SEC
    next_exact = next_time_sec is not None and abs(next_time_sec - query_time_sec) <= TF_TIME_EPS_SEC

    if prev_exact or next_exact:
        info["status"] = "exact"
        info["interpolated"] = False
        info["exact_sample_time_sec"] = float(next_time_sec if next_exact else prev_time_sec)
    elif prev_time_sec is not None and next_time_sec is not None:
        info["status"] = "interpolated"
        info["interpolated"] = True
        info["interpolation_window_sec"] = float(next_time_sec - prev_time_sec)
    elif prev_time_sec is not None:
        info["status"] = "after_latest_sample"
        info["interpolated"] = False
    elif next_time_sec is not None:
        info["status"] = "before_first_sample"
        info["interpolated"] = False
    else:
        info["status"] = "no_samples"
        info["interpolated"] = False

    return info


def make_pose_debug_record(
    image_msg,
    depth_msg,
    source_frame,
    tf_msg,
    transform_matrix,
    file_path,
    depth_file_path,
):
    query_time_sec = float(image_msg.header.stamp.to_sec())
    chain, chain_complete, chain_stop_frame = resolve_tf_chain(source_frame, WORLD_FRAME)
    edge_debug = [describe_tf_edge(parent, child, query_time_sec) for parent, child in chain]
    interpolated_edges = [
        {
            "parent_frame": edge["parent_frame"],
            "child_frame": edge["child_frame"],
            "query_time_sec": edge["query_time_sec"],
            "previous_sample_time_sec": edge.get("previous_sample_time_sec"),
            "next_sample_time_sec": edge.get("next_sample_time_sec"),
            "interpolation_window_sec": edge.get("interpolation_window_sec"),
        }
        for edge in edge_debug
        if edge.get("interpolated")
    ]

    translation = tf_msg.transform.translation
    rotation = tf_msg.transform.rotation
    return {
        "file_path": file_path,
        "depth_file_path": depth_file_path,
        "rgb_seq": int(image_msg.header.seq),
        "depth_seq": int(depth_msg.header.seq),
        "rgb_stamp_sec": query_time_sec,
        "depth_stamp_sec": float(depth_msg.header.stamp.to_sec()),
        "sync_dt_sec": float(depth_msg.header.stamp.to_sec() - image_msg.header.stamp.to_sec()),
        "image_header_frame": normalize_frame_id(image_msg.header.frame_id),
        "source_frame": normalize_frame_id(source_frame),
        "target_frame": WORLD_FRAME,
        "lookup_return_stamp_sec": float(tf_msg.header.stamp.to_sec()),
        "lookup_return_parent_frame": normalize_frame_id(tf_msg.header.frame_id),
        "lookup_return_child_frame": normalize_frame_id(tf_msg.child_frame_id),
        "pose_interpolated": bool(interpolated_edges),
        "interpolated_edges": interpolated_edges,
        "chain_complete": bool(chain_complete),
        "chain_stop_frame": normalize_frame_id(chain_stop_frame) if chain_stop_frame else None,
        "tf_chain": edge_debug,
        "translation_xyz": [float(translation.x), float(translation.y), float(translation.z)],
        "quaternion_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
        "transform_matrix": transform_matrix.tolist(),
    }


def lookup_interpolated_transform(source_frame, stamp):
    # tf2 lookup_transform interpolates when the requested image stamp lies inside the TF buffer.
    return tf_buffer.lookup_transform(
        WORLD_FRAME,
        source_frame,
        stamp,
        rospy.Duration(TF_TIMEOUT),
    )


def warn_if_unexpected_image_frame(frame_id):
    global warned_unexpected_image_frame

    normalized_frame = normalize_frame_id(frame_id)
    if normalized_frame == CAMERA_POSE_SAVE_FRAME or warned_unexpected_image_frame:
        return

    rospy.logwarn(
        "Image header frame_id is '%s' but poses are saved from '%s'.",
        normalized_frame,
        CAMERA_POSE_SAVE_FRAME,
    )
    warned_unexpected_image_frame = True


def read_json_with_retry(path, retries=5, delay_sec=0.05):
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


def write_json_atomic(path, payload):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


class _LockedFile:
    def __init__(self, lock_path):
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


def _current_mask_paths():
    if transforms_path is None or not transforms_path.exists():
        return {}, None

    try:
        current = read_json_with_retry(transforms_path)
    except Exception:
        return {}, None

    mask_paths = {}
    for frame in current.get("frames", []):
        file_path = frame.get("file_path")
        mask_path = frame.get("mask_path")
        if file_path and mask_path:
            mask_paths[file_path] = mask_path
    return mask_paths, current.get("ply_file_path")


def build_transforms_payload(meta_source, pose_source):
    payload = dict(meta_source)
    payload["frames"] = [dict(frame) for frame in meta_source["frames"]]
    payload["pose_source"] = pose_source
    return payload


def write_dataset_payload(path, lock_path, payload, mask_paths, ply_file_path):
    with _LockedFile(lock_path):
        for frame in payload["frames"]:
            mask_path = mask_paths.get(frame.get("file_path"))
            if mask_path:
                frame["mask_path"] = mask_path
        if ply_file_path:
            payload["ply_file_path"] = ply_file_path
        write_json_atomic(path, payload)


def write_transforms():
    if meta is None or transforms_path is None or transforms_lock_path is None:
        return

    mask_paths, ply_file_path = _current_mask_paths()
    primary_payload = build_transforms_payload(meta, PRIMARY_POSE_SOURCE)
    write_dataset_payload(
        transforms_path,
        transforms_lock_path,
        primary_payload,
        mask_paths,
        ply_file_path,
    )

    if gazebo_transforms_path is not None:
        write_json_atomic(gazebo_transforms_path, primary_payload)

    if meta_tf is None or transforms_tf_path is None or transforms_tf_lock_path is None:
        return

    tf_payload = build_transforms_payload(meta_tf, SECONDARY_TF_POSE_SOURCE)
    write_dataset_payload(
        transforms_tf_path,
        transforms_tf_lock_path,
        tf_payload,
        mask_paths,
        ply_file_path,
    )


def patch_ply_file_path_on_outputs():
    targets = []
    if transforms_path is not None and transforms_lock_path is not None and transforms_path.exists():
        targets.append((transforms_path, transforms_lock_path))
    if transforms_tf_path is not None and transforms_tf_lock_path is not None and transforms_tf_path.exists():
        targets.append((transforms_tf_path, transforms_tf_lock_path))
    if gazebo_transforms_path is not None and gazebo_transforms_path.exists():
        targets.append((gazebo_transforms_path, None))

    for path, lock_path in targets:
        if lock_path is None:
            data = read_json_with_retry(path)
            data["ply_file_path"] = INIT_CLOUD_NAME
            write_json_atomic(path, data)
            continue
        with _LockedFile(lock_path):
            data = read_json_with_retry(path)
            data["ply_file_path"] = INIT_CLOUD_NAME
            write_json_atomic(path, data)


def write_pose_debug_report():
    if pose_debug_path is None:
        return

    per_edge_summary = {}
    interpolated_pose_count = 0
    exact_pose_count = 0

    for record in pose_debug_records:
        if record.get("pose_interpolated"):
            interpolated_pose_count += 1
        else:
            exact_pose_count += 1

        for edge in record.get("tf_chain", []):
            key = (edge.get("parent_frame"), edge.get("child_frame"))
            summary = per_edge_summary.setdefault(
                key,
                {
                    "parent_frame": edge.get("parent_frame"),
                    "child_frame": edge.get("child_frame"),
                    "exact_count": 0,
                    "interpolated_count": 0,
                    "static_count": 0,
                    "missing_count": 0,
                    "other_count": 0,
                },
            )
            status = edge.get("status")
            if status == "exact":
                summary["exact_count"] += 1
            elif status == "interpolated":
                summary["interpolated_count"] += 1
            elif status == "static":
                summary["static_count"] += 1
            elif status == "missing_from_debug_cache":
                summary["missing_count"] += 1
            else:
                summary["other_count"] += 1

    report = {
        "world_frame": WORLD_FRAME,
        "image_topic": IMAGE_TOPIC,
        "depth_topic": DEPTH_TOPIC,
        "camera_info_topic": CAMERA_INFO_TOPIC,
        "tf_topic": TF_TOPIC,
        "tf_static_topic": TF_STATIC_TOPIC,
        "gazebo_link_states_topic": GAZEBO_LINK_STATES_TOPIC,
        "gazebo_camera_pose_topic": GAZEBO_CAMERA_POSE_TOPIC,
        "camera_pose_save_frame": CAMERA_POSE_SAVE_FRAME,
        "gazebo_camera_pose_frame": GAZEBO_CAMERA_POSE_FRAME,
        "primary_dataset_pose_source": PRIMARY_POSE_SOURCE,
        "secondary_tf_pose_source": SECONDARY_TF_POSE_SOURCE,
        "tf_timeout_sec": TF_TIMEOUT,
        "sync_slop_sec": SYNC_SLOP_SEC,
        "save_hz": SAVE_HZ,
        "run_completed": bool(run_completed),
        "capture_stats": dict(capture_stats),
        "pose_summary": {
            "saved_pose_count": len(pose_debug_records),
            "interpolated_pose_count": interpolated_pose_count,
            "exact_pose_count": exact_pose_count,
            "per_edge_summary": list(per_edge_summary.values()),
        },
        "saved_pose_debug": pose_debug_records,
        "skipped_no_tf_events": skipped_no_tf_events,
        "tf_edge_histories": [
            {
                "parent_frame": parent,
                "child_frame": child,
                "static": bool(entry["static"]),
                "sample_count": len(entry["times_sec"]),
                "sample_times_sec": [float(t) for t in entry["times_sec"]],
            }
            for (parent, child), entry in sorted(tf_edge_samples.items())
        ],
    }

    if hasattr(tf_buffer, "all_frames_as_yaml"):
        try:
            report["tf_buffer_frames_yaml"] = tf_buffer.all_frames_as_yaml()
        except Exception as exc:
            report["tf_buffer_frames_yaml_error"] = str(exc)

    write_json_atomic(pose_debug_path, report)


def sample_gazebo_link_pose(tf_frame, query_time_sec):
    history = gazebo_link_histories.get(tf_frame)
    if history is None or not history["times_sec"]:
        return None

    times_sec = history["times_sec"]
    matrices = history["matrices"]
    insert_at = bisect_left(times_sec, query_time_sec)
    prev_idx = insert_at - 1 if insert_at > 0 else None
    next_idx = insert_at if insert_at < len(times_sec) else None

    if prev_idx is not None and abs(times_sec[prev_idx] - query_time_sec) <= TF_TIME_EPS_SEC:
        return {
            "status": "exact",
            "interpolated": False,
            "previous_sample_time_sec": float(times_sec[prev_idx]),
            "next_sample_time_sec": float(times_sec[prev_idx]),
            "transform_matrix": matrices[prev_idx],
            "gazebo_link_name": history["gazebo_link_name"],
            "reference_frame": history.get("reference_frame", GAZEBO_WORLD_FRAME),
        }

    if next_idx is not None and abs(times_sec[next_idx] - query_time_sec) <= TF_TIME_EPS_SEC:
        return {
            "status": "exact",
            "interpolated": False,
            "previous_sample_time_sec": float(times_sec[next_idx]),
            "next_sample_time_sec": float(times_sec[next_idx]),
            "transform_matrix": matrices[next_idx],
            "gazebo_link_name": history["gazebo_link_name"],
            "reference_frame": history.get("reference_frame", GAZEBO_WORLD_FRAME),
        }

    if prev_idx is not None and next_idx is not None:
        prev_time_sec = times_sec[prev_idx]
        next_time_sec = times_sec[next_idx]
        alpha = (query_time_sec - prev_time_sec) / (next_time_sec - prev_time_sec)
        prev_matrix = matrices[prev_idx]
        next_matrix = matrices[next_idx]
        prev_quat = quaternion_from_matrix(prev_matrix)
        next_quat = quaternion_from_matrix(next_matrix)
        interp_quat = quaternion_slerp(prev_quat, next_quat, alpha)
        interp_xyz = prev_matrix[:3, 3] * (1.0 - alpha) + next_matrix[:3, 3] * alpha
        return {
            "status": "interpolated",
            "interpolated": True,
            "previous_sample_time_sec": float(prev_time_sec),
            "next_sample_time_sec": float(next_time_sec),
            "interpolation_alpha": float(alpha),
            "transform_matrix": compose_transform_matrix(interp_xyz, interp_quat),
            "gazebo_link_name": history["gazebo_link_name"],
            "reference_frame": history.get("reference_frame", GAZEBO_WORLD_FRAME),
        }

    sample_idx = prev_idx if prev_idx is not None else next_idx
    sample_time_sec = times_sec[sample_idx]
    return {
        "status": "nearest_only",
        "interpolated": False,
        "previous_sample_time_sec": float(sample_time_sec),
        "next_sample_time_sec": float(sample_time_sec),
        "transform_matrix": matrices[sample_idx],
        "gazebo_link_name": history["gazebo_link_name"],
        "reference_frame": history.get("reference_frame", GAZEBO_WORLD_FRAME),
    }


def build_gazebo_camera_pose(record):
    source_frame = record["source_frame"]
    query_time_sec = float(record["rgb_stamp_sec"])
    candidate_frames = []
    if GAZEBO_CAMERA_POSE_FRAME:
        candidate_frames.append(GAZEBO_CAMERA_POSE_FRAME)

    chain, _, _ = resolve_tf_chain(source_frame, WORLD_FRAME)
    candidate_frames.extend([source_frame] + [parent for parent, _ in chain])

    seen_frames = set()
    gazebo_frame = None
    for frame in candidate_frames:
        normalized_frame = normalize_frame_id(frame)
        if normalized_frame in seen_frames:
            continue
        seen_frames.add(normalized_frame)
        if normalized_frame in gazebo_link_histories:
            gazebo_frame = normalized_frame
            break

    if gazebo_frame is None:
        return {
            "available": False,
            "reason": "no_gazebo_link_found_in_camera_chain",
        }

    sampled_link_pose = sample_gazebo_link_pose(gazebo_frame, query_time_sec)
    if sampled_link_pose is None:
        return {
            "available": False,
            "reason": "no_gazebo_pose_samples",
            "gazebo_source_frame": gazebo_frame,
        }

    if gazebo_frame == source_frame:
        T_gazebo_frame_to_camera = np.eye(4, dtype=np.float64)
    else:
        try:
            offset_tf = tf_buffer.lookup_transform(
                gazebo_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(TF_TIMEOUT),
            )
        except Exception as exc:
            return {
                "available": False,
                "reason": f"static_offset_lookup_failed: {exc}",
                "gazebo_source_frame": gazebo_frame,
            }
        T_gazebo_frame_to_camera = transform_stamped_to_matrix(offset_tf)

    T_world_camera_ros = sampled_link_pose["transform_matrix"] @ T_gazebo_frame_to_camera
    T_world_camera_output = rotate_camera_frame_only(T_world_camera_ros)

    return {
        "available": True,
        "query_time_sec": query_time_sec,
        "gazebo_world_frame": sampled_link_pose["reference_frame"],
        "gazebo_source_frame": gazebo_frame,
        "gazebo_link_name": sampled_link_pose["gazebo_link_name"],
        "gazebo_sample_status": sampled_link_pose["status"],
        "gazebo_pose_interpolated": bool(sampled_link_pose["interpolated"]),
        "previous_sample_time_sec": sampled_link_pose["previous_sample_time_sec"],
        "next_sample_time_sec": sampled_link_pose["next_sample_time_sec"],
        "interpolation_alpha": sampled_link_pose.get("interpolation_alpha"),
        "transform_matrix": T_world_camera_output,
    }


def build_alignment_from_reference(tf_matrix, gazebo_matrix, pose_record, gazebo_pose, file_path, timestamp_sec):
    alignment_transform = tf_matrix @ invert_rigid_transform(gazebo_matrix)
    raw_translation_delta = (tf_matrix[:3, 3] - gazebo_matrix[:3, 3]).astype(np.float64)
    return {
        "available": True,
        "mode": "first_saved_frame_tf_alignment",
        "reference_file_path": file_path,
        "reference_timestamp_sec": float(timestamp_sec),
        "reference_source_frame": pose_record.get("source_frame"),
        "reference_image_header_frame": pose_record.get("image_header_frame"),
        "reference_gazebo_source_frame": gazebo_pose.get("gazebo_source_frame"),
        "reference_gazebo_link_name": gazebo_pose.get("gazebo_link_name"),
        "reference_gazebo_world_frame": gazebo_pose.get("gazebo_world_frame"),
        "reference_gazebo_sample_status": gazebo_pose.get("gazebo_sample_status"),
        "reference_gazebo_pose_interpolated": gazebo_pose.get("gazebo_pose_interpolated"),
        "raw_reference_translation_delta_xyz": raw_translation_delta.tolist(),
        "raw_reference_translation_error_norm": float(np.linalg.norm(raw_translation_delta)),
        "raw_reference_rotation_error_deg": rotation_angle_deg(
            gazebo_matrix[:3, :3],
            tf_matrix[:3, :3],
        ),
        "alignment_transform_matrix": alignment_transform.tolist(),
    }


def build_gazebo_world_alignment(comparisons):
    if not comparisons:
        return {
            "available": False,
            "reason": "no_comparable_gazebo_frames",
        }

    reference = comparisons[0]
    tf_matrix = np.asarray(reference["frame"]["transform_matrix"], dtype=np.float64)
    gazebo_matrix = np.asarray(reference["gazebo_pose"]["transform_matrix"], dtype=np.float64)
    return build_alignment_from_reference(
        tf_matrix,
        gazebo_matrix,
        reference["pose_record"],
        reference["gazebo_pose"],
        reference["frame"].get("file_path"),
        reference["frame"].get("timestamp_sec"),
    )


def write_gazebo_pose_reports():
    if run_dir is None or transforms_tf_path is None or not transforms_tf_path.exists():
        return

    current_transforms = read_json_with_retry(transforms_tf_path)
    pose_debug_by_file_path = {
        record["file_path"]: record
        for record in pose_debug_records
        if "file_path" in record
    }

    gazebo_frames = []
    difference_records = []
    missing_frames = []
    comparisons = []

    for frame in current_transforms.get("frames", []):
        file_path = frame.get("file_path")
        if not file_path:
            continue

        pose_record = pose_debug_by_file_path.get(file_path)
        if pose_record is None:
            missing_frames.append(
                {
                    "file_path": file_path,
                    "reason": "missing_saved_tf_record",
                }
            )
            continue

        gazebo_pose = build_gazebo_camera_pose(pose_record)
        if not gazebo_pose.get("available"):
            missing_frames.append(
                {
                    "file_path": file_path,
                    "reason": gazebo_pose.get("reason", "gazebo_pose_unavailable"),
                }
            )
            continue

        comparisons.append(
            {
                "frame": frame,
                "pose_record": pose_record,
                "gazebo_pose": gazebo_pose,
            }
        )

    alignment = build_gazebo_world_alignment(comparisons)
    alignment_transform = None
    if alignment.get("available"):
        alignment_transform = np.asarray(alignment["alignment_transform_matrix"], dtype=np.float64)

    actual_gazebo_world_frame = None
    if comparisons:
        actual_gazebo_world_frame = comparisons[0]["gazebo_pose"].get("gazebo_world_frame")

    for comparison in comparisons:
        frame = comparison["frame"]
        pose_record = comparison["pose_record"]
        gazebo_pose = comparison["gazebo_pose"]
        tf_matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        raw_gazebo_matrix = np.asarray(gazebo_pose["transform_matrix"], dtype=np.float64)
        aligned_gazebo_matrix = (
            alignment_transform @ raw_gazebo_matrix
            if alignment_transform is not None
            else raw_gazebo_matrix
        )

        gazebo_frame = dict(frame)
        gazebo_frame["transform_matrix"] = aligned_gazebo_matrix.tolist()
        gazebo_frames.append(gazebo_frame)

        translation_delta = (tf_matrix[:3, 3] - aligned_gazebo_matrix[:3, 3]).astype(np.float64)
        difference_records.append(
            {
                "file_path": frame.get("file_path"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "source_frame": pose_record["source_frame"],
                "image_header_frame": pose_record.get("image_header_frame"),
                "tf_world_frame": WORLD_FRAME,
                "gazebo_world_frame": gazebo_pose.get("gazebo_world_frame"),
                "gazebo_world_aligned_to_tf": bool(alignment_transform is not None),
                "gazebo_source_frame": gazebo_pose["gazebo_source_frame"],
                "gazebo_link_name": gazebo_pose["gazebo_link_name"],
                "gazebo_sample_status": gazebo_pose["gazebo_sample_status"],
                "gazebo_pose_interpolated": gazebo_pose["gazebo_pose_interpolated"],
                "gazebo_previous_sample_time_sec": gazebo_pose["previous_sample_time_sec"],
                "gazebo_next_sample_time_sec": gazebo_pose["next_sample_time_sec"],
                "translation_delta_xyz": translation_delta.tolist(),
                "translation_error_norm": float(np.linalg.norm(translation_delta)),
                "rotation_error_deg": rotation_angle_deg(
                    aligned_gazebo_matrix[:3, :3],
                    tf_matrix[:3, :3],
                ),
                "tf_transform_matrix": tf_matrix.tolist(),
                "gazebo_raw_transform_matrix": raw_gazebo_matrix.tolist(),
                "gazebo_transform_matrix": aligned_gazebo_matrix.tolist(),
            }
        )

    gazebo_payload = dict(current_transforms)
    gazebo_payload["frames"] = gazebo_frames
    gazebo_payload["pose_source"] = PRIMARY_POSE_SOURCE
    gazebo_payload["gazebo_world_alignment"] = alignment
    if actual_gazebo_world_frame is not None:
        gazebo_payload["gazebo_world_frame"] = actual_gazebo_world_frame
    if transforms_path is not None:
        write_json_atomic(transforms_path, gazebo_payload)
    write_json_atomic(gazebo_transforms_path, gazebo_payload)

    translation_errors = [entry["translation_error_norm"] for entry in difference_records]
    rotation_errors = [entry["rotation_error_deg"] for entry in difference_records]
    difference_payload = {
        "tf_world_frame": WORLD_FRAME,
        "gazebo_world_frame": actual_gazebo_world_frame,
        "gazebo_world_alignment": alignment,
        "frame_count_compared": len(difference_records),
        "frame_count_missing": len(missing_frames),
        "translation_error_mean": float(np.mean(translation_errors)) if translation_errors else None,
        "translation_error_median": float(np.median(translation_errors)) if translation_errors else None,
        "translation_error_max": float(np.max(translation_errors)) if translation_errors else None,
        "rotation_error_deg_mean": float(np.mean(rotation_errors)) if rotation_errors else None,
        "rotation_error_deg_median": float(np.median(rotation_errors)) if rotation_errors else None,
        "rotation_error_deg_max": float(np.max(rotation_errors)) if rotation_errors else None,
        "missing_frames": missing_frames,
        "differences": difference_records,
    }
    write_json_atomic(gazebo_pose_diff_path, difference_payload)


def write_init_cloud_from_saved_frames():
    if run_dir is None or transforms_path is None or not transforms_path.exists():
        return

    ply_path = run_dir / INIT_CLOUD_NAME
    current_transforms = read_json_with_retry(transforms_path)
    frames = current_transforms.get("frames", [])
    if not frames:
        return

    dataset_dir = run_dir.resolve()
    samples_per_frame = max(1, (MAX_INIT_CLOUD_POINTS + len(frames) - 1) // len(frames))
    rng = np.random.default_rng(0)
    all_xyz = []
    all_rgb = []

    for frame in frames:
        depth_rel = frame.get("depth_file_path")
        rgb_rel = frame.get("file_path")
        if not depth_rel or not rgb_rel:
            continue

        depth_path = resolve_relpath(dataset_dir, depth_rel)
        rgb_path = resolve_relpath(dataset_dir, rgb_rel)
        mask_path = resolve_frame_mask_path(dataset_dir, frame)

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth image from {depth_path}")
        if depth.ndim != 2:
            raise RuntimeError(f"Depth image must be HxW, got {depth.shape} for {depth_path}")

        if np.issubdtype(depth.dtype, np.floating):
            depth_mm = depth.astype(np.float32) * 1000.0
        elif depth.dtype == np.uint16:
            depth_mm = depth.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported depth dtype {depth.dtype} for {depth_path}")

        expected_hw = depth_mm.shape
        valid_mask = load_saved_mask(mask_path, expected_hw)
        rgb_bgr = load_saved_rgb(rgb_path, expected_hw)

        valid = valid_mask & (depth_mm > 0.0)
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue

        n_sample = min(samples_per_frame, ys.size)
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

    if xyz.shape[0] > MAX_INIT_CLOUD_POINTS:
        keep = rng.choice(xyz.shape[0], size=MAX_INIT_CLOUD_POINTS, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]

    write_ascii_ply(ply_path, xyz.astype(np.float32), rgb)
    meta["ply_file_path"] = INIT_CLOUD_NAME
    if meta_tf is not None:
        meta_tf["ply_file_path"] = INIT_CLOUD_NAME
    patch_ply_file_path_on_outputs()


def rotate_camera_frame_only(T_ros):
    T_output = T_ros.copy()
    T_output[:3, :3] = T_ros[:3, :3] @ ROS_CAMERA_TO_OUTPUT_FRAME
    T_output[:3, 3] = T_ros[:3, 3]
    return T_output


def launch_mask_saver():
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


def write_capture_complete_sentinel():
    payload = {
        "frame_count": len(meta["frames"]),
        "finished_at": time.time(),
    }
    write_json_atomic(capture_complete_path, payload)


def terminate_mask_saver(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def wait_for_mask_saver(process):
    if process is None:
        return
    try:
        process.wait(timeout=MASK_SAVER_EXIT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        terminate_mask_saver(process)
        raise RuntimeError("Timed out waiting for mask saver to finish") from exc
    if process.returncode != 0:
        raise RuntimeError(f"Mask saver exited with code {process.returncode}")


def image_callback(image_msg, depth_msg):
    global frame_index
    global last_saved_stamp
    global gazebo_alignment_transform
    global gazebo_alignment_metadata

    capture_stats["rgb_depth_callbacks"] += 1

    if frame_index >= MAX_IMAGES:
        return

    if last_saved_stamp is not None:
        if (image_msg.header.stamp - last_saved_stamp).to_sec() < 1.0 / SAVE_HZ:
            capture_stats["skipped_rate"] += 1
            return

    pose_source_frame = CAMERA_POSE_SAVE_FRAME
    warn_if_unexpected_image_frame(image_msg.header.frame_id)

    if not tf_buffer.can_transform(
        WORLD_FRAME,
        pose_source_frame,
        image_msg.header.stamp,
        rospy.Duration(TF_TIMEOUT),
    ):
        capture_stats["skipped_no_tf"] += 1
        if len(skipped_no_tf_events) < MAX_SKIPPED_NO_TF_EVENTS:
            skipped_no_tf_events.append(
                {
                    "rgb_seq": int(image_msg.header.seq),
                    "depth_seq": int(depth_msg.header.seq),
                    "rgb_stamp_sec": float(image_msg.header.stamp.to_sec()),
                    "depth_stamp_sec": float(depth_msg.header.stamp.to_sec()),
                    "image_header_frame": normalize_frame_id(image_msg.header.frame_id),
                    "source_frame": normalize_frame_id(pose_source_frame),
                }
            )
        return

    image = ros_image_to_bgr(image_msg)
    depth_mm = ros_depth_to_uint16_mm(depth_msg)
    tf_msg = lookup_interpolated_transform(pose_source_frame, image_msg.header.stamp)
    T_tf_ros = transform_stamped_to_matrix(tf_msg)
    T_tf_output = rotate_camera_frame_only(T_tf_ros)

    ros_seq = int(image_msg.header.seq)
    file_stem = f"{IMAGE_NAME_PREFIX}_{ros_seq:05d}"
    image_file_name = f"{file_stem}.png"
    depth_file_name = f"{file_stem}.tiff"
    file_path = f"./rgb/{image_file_name}"
    depth_file_path = f"./depth/{depth_file_name}"

    pose_debug_record = make_pose_debug_record(
        image_msg,
        depth_msg,
        pose_source_frame,
        tf_msg,
        T_tf_output,
        file_path,
        depth_file_path,
    )
    gazebo_pose = build_gazebo_camera_pose(pose_debug_record)
    if not gazebo_pose.get("available"):
        capture_stats["skipped_no_gazebo"] += 1
        rospy.logwarn_throttle(
            2.0,
            "Skipping frame because gazebo pose is unavailable: %s",
            gazebo_pose.get("reason", "unknown_reason"),
        )
        return

    raw_gazebo_matrix = np.asarray(gazebo_pose["transform_matrix"], dtype=np.float64)
    if gazebo_alignment_transform is None:
        gazebo_alignment_metadata = build_alignment_from_reference(
            T_tf_output,
            raw_gazebo_matrix,
            pose_debug_record,
            gazebo_pose,
            file_path,
            float(image_msg.header.stamp.to_sec()),
        )
        gazebo_alignment_transform = np.asarray(
            gazebo_alignment_metadata["alignment_transform_matrix"],
            dtype=np.float64,
        )

    T_primary_output = gazebo_alignment_transform @ raw_gazebo_matrix

    cv2.imwrite(str(rgb_dir / image_file_name), image)
    cv2.imwrite(str(depth_dir / depth_file_name), depth_mm)

    meta["frames"].append(
        {
            "file_path": file_path,
            "depth_file_path": depth_file_path,
            "timestamp_sec": float(image_msg.header.stamp.to_sec()),
            "transform_matrix": T_primary_output.tolist(),
        }
    )
    meta_tf["frames"].append(
        {
            "file_path": file_path,
            "depth_file_path": depth_file_path,
            "timestamp_sec": float(image_msg.header.stamp.to_sec()),
            "transform_matrix": T_tf_output.tolist(),
        }
    )
    pose_debug_record["primary_pose_source"] = PRIMARY_POSE_SOURCE
    pose_debug_record["primary_transform_matrix"] = T_primary_output.tolist()
    pose_debug_record["gazebo_world_frame"] = gazebo_pose.get("gazebo_world_frame")
    pose_debug_record["gazebo_source_frame"] = gazebo_pose.get("gazebo_source_frame")
    pose_debug_record["gazebo_link_name"] = gazebo_pose.get("gazebo_link_name")
    pose_debug_record["gazebo_sample_status"] = gazebo_pose.get("gazebo_sample_status")
    pose_debug_record["gazebo_pose_interpolated"] = gazebo_pose.get("gazebo_pose_interpolated")
    pose_debug_record["gazebo_raw_transform_matrix"] = raw_gazebo_matrix.tolist()
    pose_debug_record["gazebo_aligned_transform_matrix"] = T_primary_output.tolist()
    if gazebo_alignment_metadata is not None:
        pose_debug_record["gazebo_world_alignment"] = gazebo_alignment_metadata
    pose_debug_records.append(pose_debug_record)
    write_transforms()
    frame_index += 1
    capture_stats["saved_frames"] = frame_index
    last_saved_stamp = image_msg.header.stamp

    if frame_index >= MAX_IMAGES:
        rospy.signal_shutdown(f"Saved {MAX_IMAGES} images")


def main():
    global tf_buffer
    global tf_listener
    global mask_process
    global run_dir
    global rgb_dir
    global depth_dir
    global masks_dir
    global pose_debug_path
    global gazebo_transforms_path
    global gazebo_pose_diff_path
    global transforms_path
    global transforms_lock_path
    global transforms_tf_path
    global transforms_tf_lock_path
    global capture_complete_path
    global meta
    global meta_tf
    global tf_edge_samples
    global tf_parent_by_child
    global pose_debug_records
    global skipped_no_tf_events
    global capture_stats
    global run_completed
    global gazebo_link_histories
    global gazebo_alignment_transform
    global gazebo_alignment_metadata

    rospy.init_node("save_dynaarm_camera1_rgb_tf")
    tf_buffer = Buffer()
    mask_process = launch_mask_saver()

    try:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = DATASET_ROOT / run_name
        rgb_dir = run_dir / "rgb"
        depth_dir = run_dir / "depth"
        masks_dir = run_dir / "masks"
        pose_debug_path = run_dir / POSE_DEBUG_NAME
        gazebo_transforms_path = run_dir / GAZEBO_TRANSFORMS_NAME
        gazebo_pose_diff_path = run_dir / GAZEBO_POSE_DIFF_NAME
        transforms_path = run_dir / "transforms.json"
        transforms_lock_path = run_dir / "transforms.json.lock"
        transforms_tf_path = run_dir / TF_TRANSFORMS_NAME
        transforms_tf_lock_path = run_dir / f"{TF_TRANSFORMS_NAME}.lock"
        capture_complete_path = run_dir / CAPTURE_COMPLETE_SENTINEL_NAME
        tf_edge_samples = {}
        tf_parent_by_child = {}
        pose_debug_records = []
        skipped_no_tf_events = []
        gazebo_link_histories = {}
        gazebo_alignment_transform = None
        gazebo_alignment_metadata = None
        run_completed = False
        capture_stats = {
            "rgb_depth_callbacks": 0,
            "skipped_rate": 0,
            "skipped_no_tf": 0,
            "skipped_no_gazebo": 0,
            "saved_frames": 0,
        }
        rgb_dir.mkdir(parents=True, exist_ok=False)
        depth_dir.mkdir(parents=True, exist_ok=False)

        tf_listener = TransformListener(tf_buffer)
        tf_sub = rospy.Subscriber(TF_TOPIC, TFMessage, tf_dynamic_callback, queue_size=200)
        tf_static_sub = rospy.Subscriber(TF_STATIC_TOPIC, TFMessage, tf_static_callback, queue_size=10)
        gazebo_camera_pose_sub = rospy.Subscriber(
            GAZEBO_CAMERA_POSE_TOPIC,
            PoseStamped,
            gazebo_camera_pose_callback,
            queue_size=200,
        )
        gazebo_link_states_sub = rospy.Subscriber(
            GAZEBO_LINK_STATES_TOPIC,
            LinkStates,
            gazebo_link_states_callback,
            queue_size=50,
        )
        rospy.sleep(0.5)
        info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=5.0)

        meta = {
            "fl_x": info.K[0],
            "fl_y": info.K[4],
            "cx": info.K[2],
            "cy": info.K[5],
            "w": info.width,
            "h": info.height,
            "frames": [],
        }
        meta_tf = dict(meta)
        meta_tf["frames"] = []
        write_transforms()

        rgb_sub = Subscriber(IMAGE_TOPIC, Image)
        depth_sub = Subscriber(DEPTH_TOPIC, Image)
        sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        sync.registerCallback(image_callback)
        rospy.spin()
        write_transforms()
        write_capture_complete_sentinel()
        wait_for_mask_saver(mask_process)
        write_transforms()
        write_init_cloud_from_saved_frames()
        write_gazebo_pose_reports()
        run_completed = True
    finally:
        if gazebo_transforms_path is not None and gazebo_pose_diff_path is not None:
            try:
                write_gazebo_pose_reports()
            except Exception as exc:
                rospy.logwarn("Failed to write Gazebo pose reports: %s", exc)
        if pose_debug_path is not None:
            try:
                write_pose_debug_report()
            except Exception as exc:
                rospy.logwarn("Failed to write pose debug report: %s", exc)
        terminate_mask_saver(mask_process)


if __name__ == "__main__":
    main()
