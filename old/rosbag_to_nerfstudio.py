
#!/usr/bin/env python3
"""
Convert a ROS1 bag containing RGB images, depth images, CameraInfo, and TF
into a Nerfstudio-compatible dataset:

output_dir/
  images/
    frame_000000.png
    ...
  depth/
    frame_000000.png   # uint16 depth in millimeters
    ...
  transforms.json
  conversion_report.json

Requirements:
  pip install rosbag tf2_ros cv_bridge opencv-python numpy

Typical usage:
  python rosbag_to_nerfstudio.py \
    --bag /path/to/run.bag \
    --output-dir /path/to/ns_dataset \
    --rgb-topic /camera/color/image_raw \
    --camera-info-topic /camera/color/camera_info \
    --depth-topic /camera/depth/image_raw \
    --target-frame world \
    --camera-frame camera_link_optical
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import rosbag
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_msgs.msg import TFMessage
except Exception as e:
    raise SystemExit(
        "This script must run in a ROS1 Python environment with rosbag, rospy, "
        "sensor_msgs, tf2_msgs, and cv_bridge available.\n"
        f"Import error: {e}"
    )


def stamp_to_sec(stamp):
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def transform_to_matrix(translation, quaternion):
    tx, ty, tz = translation
    qx, qy, qz, qw = quaternion
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def make_ns_camera_to_world_from_ros_optical(T_world_ros_camera):
    # ROS optical frame / OpenCV camera convention:
    # x right, y down, z forward
    #
    # Nerfstudio/OpenGL convention:
    # x right, y up, z back
    #
    # Convert camera basis by flipping Y and Z.
    cv_to_gl = np.eye(4, dtype=np.float64)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0
    return T_world_ros_camera @ cv_to_gl


class TfBufferSimple:
    def __init__(self):
        self.static_edges = {}  # (parent, child) -> T_parent_child
        self.dynamic_edges = {} # (parent, child) -> list[(time, T_parent_child)]

    def add_tf_message(self, msg, is_static=False):
        for tr in msg.transforms:
            parent = tr.header.frame_id.lstrip("/")
            child = tr.child_frame_id.lstrip("/")
            ts = stamp_to_sec(tr.header.stamp)
            T = transform_to_matrix(
                (tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z),
                (tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w),
            )
            key = (parent, child)
            if is_static:
                self.static_edges[key] = T
            else:
                self.dynamic_edges.setdefault(key, []).append((ts, T))

    def finalize(self):
        for key in self.dynamic_edges:
            self.dynamic_edges[key].sort(key=lambda x: x[0])

    def _get_edge_at_time(self, parent, child, t):
        key = (parent, child)
        if key in self.static_edges:
            return self.static_edges[key]
        if key in self.dynamic_edges:
            seq = self.dynamic_edges[key]
            idx = None
            lo, hi = 0, len(seq) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if seq[mid][0] <= t:
                    idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if idx is None:
                return None
            return seq[idx][1]
        return None

    def lookup_transform(self, target_frame, source_frame, t):
        target_frame = target_frame.lstrip("/")
        source_frame = source_frame.lstrip("/")

        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        # BFS over forward and reverse edges
        from collections import deque
        q = deque()
        q.append((target_frame, np.eye(4, dtype=np.float64)))
        visited = {target_frame}

        neighbors = set()
        for p, c in self.static_edges.keys():
            neighbors.add(p); neighbors.add(c)
        for p, c in self.dynamic_edges.keys():
            neighbors.add(p); neighbors.add(c)
        if target_frame not in neighbors or source_frame not in neighbors:
            raise KeyError(f"Frame not found in TF graph: {target_frame} or {source_frame}")

        all_edges = list(self.static_edges.keys()) + list(self.dynamic_edges.keys())

        while q:
            cur, T_target_cur = q.popleft()
            if cur == source_frame:
                return T_target_cur

            for p, c in all_edges:
                if p == cur and c not in visited:
                    T_p_c = self._get_edge_at_time(p, c, t)
                    if T_p_c is not None:
                        q.append((c, T_target_cur @ T_p_c))
                        visited.add(c)
                elif c == cur and p not in visited:
                    T_p_c = self._get_edge_at_time(p, c, t)
                    if T_p_c is not None:
                        q_c_p = invert_transform(T_p_c)
                        q.append((p, T_target_cur @ q_c_p))
                        visited.add(p)

        raise KeyError(f"No TF path from {target_frame} to {source_frame} at time {t:.6f}")


def sanitize_depth_to_uint16_mm(depth_m):
    depth = np.array(depth_m, dtype=np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    depth_mm = np.round(depth * 1000.0)
    depth_mm[invalid] = 0.0
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    return depth_mm


def read_messages(bag, topics):
    for topic, msg, t in bag.read_messages(topics=topics):
        yield topic, msg, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Input ROS1 bag file")
    ap.add_argument("--output-dir", required=True, help="Output Nerfstudio dataset directory")
    ap.add_argument("--rgb-topic", required=True)
    ap.add_argument("--camera-info-topic", required=True)
    ap.add_argument("--depth-topic", required=True)
    ap.add_argument("--target-frame", required=True, help="Global frame, e.g. world/map/odom")
    ap.add_argument("--camera-frame", required=True, help="Camera optical frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--depth-nearest", action="store_true",
                    help="Use nearest depth message in time instead of latest depth at or before RGB timestamp")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    depth_dir = out_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()

    # Pass 1: collect TF and intrinsics
    tfbuf = TfBufferSimple()
    camera_info = None
    num_tf = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in read_messages(bag, ["/tf", "/tf_static", args.camera_info_topic]):
            if topic == "/tf":
                tfbuf.add_tf_message(msg, is_static=False)
                num_tf += len(msg.transforms)
            elif topic == "/tf_static":
                tfbuf.add_tf_message(msg, is_static=True)
                num_tf += len(msg.transforms)
            elif topic == args.camera_info_topic and camera_info is None:
                camera_info = msg

    tfbuf.finalize()

    if camera_info is None:
        raise SystemExit(f"No CameraInfo found on topic {args.camera_info_topic}")

    w = int(camera_info.width)
    h = int(camera_info.height)
    K = np.array(camera_info.K, dtype=np.float64).reshape(3, 3)
    fl_x = float(K[0, 0])
    fl_y = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    D = list(camera_info.D)
    distortion_model = getattr(camera_info, "distortion_model", "") or "plumb_bob"
    if distortion_model in ("plumb_bob", "rational_polynomial"):
        camera_model = "OPENCV"
        k1 = float(D[0]) if len(D) > 0 else 0.0
        k2 = float(D[1]) if len(D) > 1 else 0.0
        p1 = float(D[2]) if len(D) > 2 else 0.0
        p2 = float(D[3]) if len(D) > 3 else 0.0
        k3 = float(D[4]) if len(D) > 4 else 0.0
    else:
        camera_model = "OPENCV"
        k1 = k2 = p1 = p2 = k3 = 0.0

    # Pass 2: collect depth messages
    depth_msgs = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in read_messages(bag, [args.depth_topic]):
            ts = stamp_to_sec(msg.header.stamp)
            depth_msgs.append((ts, msg))
    if not depth_msgs:
        raise SystemExit(f"No depth messages found on topic {args.depth_topic}")
    depth_times = [x[0] for x in depth_msgs]

    def get_depth_msg_for_time(ts):
        import bisect
        idx = bisect.bisect_left(depth_times, ts)
        if args.depth_nearest:
            cands = []
            if idx < len(depth_times):
                cands.append((abs(depth_times[idx] - ts), idx))
            if idx - 1 >= 0:
                cands.append((abs(depth_times[idx - 1] - ts), idx - 1))
            if not cands:
                return None, None
            _, best = min(cands, key=lambda x: x[0])
            return depth_msgs[best]
        else:
            best = idx - 1
            if best < 0:
                return None, None
            return depth_msgs[best]

    frames = []
    num_rgb = 0
    num_written = 0
    tf_failures = 0
    depth_failures = 0

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in read_messages(bag, [args.rgb_topic]):
            num_rgb += 1
            ts = stamp_to_sec(msg.header.stamp)

            # RGB conversion
            try:
                rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                try:
                    rgb = bridge.imgmsg_to_cv2(msg)
                    if rgb.ndim == 2:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"Skipping RGB frame at {ts:.6f}: cv_bridge failed: {e}")
                    continue

            if rgb.shape[0] != h or rgb.shape[1] != w:
                # Accept CameraInfo from a slightly different topic only if dimensions still compatible.
                h_i, w_i = rgb.shape[:2]
                print(f"Warning: RGB frame size {w_i}x{h_i} differs from CameraInfo {w}x{h}")

            # Depth matching
            depth_ts, depth_msg = get_depth_msg_for_time(ts)
            if depth_msg is None:
                depth_failures += 1
                continue

            try:
                if depth_msg.encoding == "32FC1":
                    depth_m = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                elif depth_msg.encoding == "16UC1":
                    depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                    depth_m = depth_raw.astype(np.float32) * 1e-3
                else:
                    depth_m = bridge.imgmsg_to_cv2(depth_msg)
                    depth_m = depth_m.astype(np.float32)
            except Exception as e:
                print(f"Skipping frame at {ts:.6f}: depth conversion failed: {e}")
                depth_failures += 1
                continue

            # Pose lookup at RGB timestamp
            try:
                T_target_camera_ros = tfbuf.lookup_transform(args.target_frame, args.camera_frame, ts)
            except Exception as e:
                print(f"Skipping frame at {ts:.6f}: TF lookup failed: {e}")
                tf_failures += 1
                continue

            T_target_camera_ns = make_ns_camera_to_world_from_ros_optical(T_target_camera_ros)

            frame_name = f"frame_{num_written:06d}"
            rgb_path = images_dir / f"{frame_name}.png"
            depth_path = depth_dir / f"{frame_name}.png"

            ok = cv2.imwrite(str(rgb_path), rgb)
            if not ok:
                print(f"Failed to write RGB image: {rgb_path}")
                continue

            depth_mm = sanitize_depth_to_uint16_mm(depth_m)
            ok = cv2.imwrite(str(depth_path), depth_mm)
            if not ok:
                print(f"Failed to write depth image: {depth_path}")
                rgb_path.unlink(missing_ok=True)
                continue

            frames.append({
                "file_path": f"images/{frame_name}.png",
                "depth_file_path": f"depth/{frame_name}.png",
                "transform_matrix": T_target_camera_ns.tolist(),
            })

            num_written += 1
            if args.max_frames is not None and num_written >= args.max_frames:
                break

    transforms = {
        "camera_model": camera_model,
        "orientation_override": "none",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "frames": frames,
    }

    with open(out_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    report = {
        "bag": str(bag_path),
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "camera_info_topic": args.camera_info_topic,
        "target_frame": args.target_frame,
        "camera_frame": args.camera_frame,
        "num_tf_messages_or_transforms": num_tf,
        "num_rgb_messages": num_rgb,
        "num_frames_written": num_written,
        "tf_failures": tf_failures,
        "depth_failures": depth_failures,
        "camera_info": {
            "width": w,
            "height": h,
            "fx": fl_x,
            "fy": fl_y,
            "cx": cx,
            "cy": cy,
            "distortion_model": distortion_model,
            "D": D,
        },
        "notes": [
            "Depth images are written as uint16 PNG in millimeters.",
            "Nerfstudio defaults to depth_unit_scale_factor=1e-3, so these PNG depths should load correctly.",
            "transform_matrix is camera-to-world using Nerfstudio/OpenGL camera convention.",
        ],
    }
    with open(out_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
