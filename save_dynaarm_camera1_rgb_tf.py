#!/usr/bin/env python3

import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_matrix
from tf2_ros import Buffer, TransformListener

IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
WORLD_FRAME = "dynaarm_arm_tf/world"
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

# ROS optical / OpenCV camera frame:
#   +X right, +Y down, +Z forward
#
# Nerfstudio / OpenGL camera frame:
#   +X right, +Y up, +Z back
#
# Keep the original optical->OpenGL basis change first, then apply the extra
# fixed-axis rotation requested by the user:
#   1. +90 deg about the initial Y axis
#   2. +90 deg about the initial X axis
# This fixed-axis composition is equivalent to +90 deg about Y followed by
# +90 deg about the rotated Z axis.
ROS_OPTICAL_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0])
EXTRA_FIXED_ROTATION = np.array([
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
], dtype=np.float64)

ROS_CAMERA_TO_OUTPUT_FRAME = ROS_OPTICAL_TO_NERFSTUDIO @ EXTRA_FIXED_ROTATION

tf_buffer = None
tf_listener = None
mask_process = None
rgb_dir = None
transforms_path = None
transforms_lock_path = None
capture_complete_path = None
meta = None
frame_index = 0
last_saved_stamp = None


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


def transform_stamped_to_matrix(tf_msg)->quaternion_matrix:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def lookup_interpolated_transform(msg):
    # tf2 lookup_transform interpolates when the requested image stamp lies inside the TF buffer.
    return tf_buffer.lookup_transform(
        WORLD_FRAME,
        msg.header.frame_id,
        msg.header.stamp,
        rospy.Duration(TF_TIMEOUT),
    )


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


def write_transforms():
    payload = dict(meta)
    payload["frames"] = [dict(frame) for frame in meta["frames"]]

    with _LockedFile(transforms_lock_path):
        mask_paths, ply_file_path = _current_mask_paths()
        for frame in payload["frames"]:
            mask_path = mask_paths.get(frame.get("file_path"))
            if mask_path:
                frame["mask_path"] = mask_path
        if ply_file_path:
            payload["ply_file_path"] = ply_file_path
        write_json_atomic(transforms_path, payload)


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


def image_callback(msg):
    global frame_index
    global last_saved_stamp

    if frame_index >= MAX_IMAGES:
        return

    if last_saved_stamp is not None:
        if (msg.header.stamp - last_saved_stamp).to_sec() < 1.0 / SAVE_HZ:
            return

    if not tf_buffer.can_transform(
        WORLD_FRAME,
        msg.header.frame_id,
        msg.header.stamp,
        rospy.Duration(TF_TIMEOUT),
    ):
        return

    image = ros_image_to_bgr(msg)
    tf_msg = lookup_interpolated_transform(msg)
    T_ros = transform_stamped_to_matrix(tf_msg)
    T_output = rotate_camera_frame_only(T_ros)

    ros_seq = int(msg.header.seq)
    file_name = f"{IMAGE_NAME_PREFIX}_{ros_seq:05d}.png"
    cv2.imwrite(str(rgb_dir / file_name), image)

    meta["frames"].append(
        {
            "file_path": f"./rgb/{file_name}",
            "timestamp_sec": float(msg.header.stamp.to_sec()),
            "transform_matrix": T_output.tolist(),
        }
    )
    write_transforms()
    frame_index += 1
    last_saved_stamp = msg.header.stamp

    if frame_index >= MAX_IMAGES:
        rospy.signal_shutdown(f"Saved {MAX_IMAGES} images")


def main():
    global tf_buffer
    global tf_listener
    global mask_process
    global rgb_dir
    global transforms_path
    global transforms_lock_path
    global capture_complete_path
    global meta

    rospy.init_node("save_dynaarm_camera1_rgb_tf")
    tf_buffer = Buffer()
    mask_process = launch_mask_saver()

    try:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = DATASET_ROOT / run_name
        rgb_dir = run_dir / "rgb"
        transforms_path = run_dir / "transforms.json"
        transforms_lock_path = run_dir / "transforms.json.lock"
        capture_complete_path = run_dir / CAPTURE_COMPLETE_SENTINEL_NAME
        rgb_dir.mkdir(parents=True, exist_ok=False)

        tf_listener = TransformListener(tf_buffer)
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
        write_transforms()

        rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)
        rospy.spin()
        write_transforms()
        write_capture_complete_sentinel()
        wait_for_mask_saver(mask_process)
    finally:
        if capture_complete_path is None or not capture_complete_path.exists():
            terminate_mask_saver(mask_process)


if __name__ == "__main__":
    main()
