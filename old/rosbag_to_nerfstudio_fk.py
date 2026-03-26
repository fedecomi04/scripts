#!/usr/bin/env python3
"""
Convert a ROS1 bag to a Nerfstudio dataset using URDF forward kinematics
instead of TF. This is useful when the camera frame is not connected in /tf
but the robot arm joint states are available.

Outputs:
  output_dir/
    images/
    depth/
    transforms.json
    conversion_report.json

Requirements:
  ROS1 Python env with rosbag, rospy, cv_bridge, sensor_msgs available.
  Python packages: numpy, opencv-python
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
except Exception as e:
    raise SystemExit(
        "Run this inside a ROS1 Python environment with rosbag and cv_bridge.\n"
        f"Import error: {e}"
    )


def stamp_to_sec(stamp):
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def make_transform(R=None, t=None):
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def origin_transform(xyz, rpy):
    R = rpy_to_matrix(*rpy)
    return make_transform(R, xyz)


def make_ns_camera_to_world_from_ros_optical(T_world_ros_camera):
    cv_to_gl = np.eye(4, dtype=np.float64)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0
    return T_world_ros_camera @ cv_to_gl


def sanitize_depth_to_uint16_mm(depth_m):
    depth = np.array(depth_m, dtype=np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    depth_mm = np.round(depth * 1000.0)
    depth_mm[invalid] = 0.0
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    return depth_mm


def parse_xyz_rpy(elem):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if elem is None:
        return xyz, rpy
    if "xyz" in elem.attrib:
        xyz = [float(v) for v in elem.attrib["xyz"].split()]
    if "rpy" in elem.attrib:
        rpy = [float(v) for v in elem.attrib["rpy"].split()]
    return xyz, rpy


class Joint:
    def __init__(self, name, jtype, parent, child, origin_xyz, origin_rpy, axis):
        self.name = name
        self.jtype = jtype
        self.parent = parent
        self.child = child
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        self.axis = axis


def load_urdf_chain(urdf_path, base_link, camera_link):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_by_child = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        xyz, rpy = parse_xyz_rpy(j.find("origin"))
        axis_elem = j.find("axis")
        axis = [0.0, 0.0, 1.0]
        if axis_elem is not None and "xyz" in axis_elem.attrib:
            axis = [float(v) for v in axis_elem.attrib["xyz"].split()]
        joints_by_child[child] = Joint(name, jtype, parent, child, xyz, rpy, axis)

    chain = []
    cur = camera_link
    while cur != base_link:
        if cur not in joints_by_child:
            raise RuntimeError(f"No parent joint found for link '{cur}' while walking toward '{base_link}'")
        joint = joints_by_child[cur]
        chain.append(joint)
        cur = joint.parent

    chain.reverse()
    return chain


def joint_transform(joint, joint_position):
    T = origin_transform(joint.origin_xyz, joint.origin_rpy)
    if joint.jtype in ("fixed",):
        return T
    if joint.jtype in ("revolute", "continuous"):
        R = axis_angle_to_matrix(joint.axis, joint_position)
        return T @ make_transform(R, [0, 0, 0])
    if joint.jtype == "prismatic":
        disp = np.asarray(joint.axis, dtype=np.float64) * joint_position
        return T @ make_transform(np.eye(3), disp)
    raise RuntimeError(f"Unsupported joint type '{joint.jtype}' for joint '{joint.name}'")


def read_messages(bag, topics):
    for topic, msg, t in bag.read_messages(topics=topics):
        yield topic, msg, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--rgb-topic", required=True)
    ap.add_argument("--camera-info-topic", required=True)
    ap.add_argument("--depth-topic", required=True)
    ap.add_argument("--joint-states-topic", required=True)
    ap.add_argument("--base-link", required=True, help="Fixed reference link, e.g. dynaarm_base")
    ap.add_argument("--camera-link", required=True, help="Camera optical link, e.g. camera_link_optical")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    depth_dir = out_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()

    chain = load_urdf_chain(args.urdf, args.base_link, args.camera_link)

    camera_info = None
    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, t in read_messages(bag, [args.camera_info_topic]):
            camera_info = msg
            break
    if camera_info is None:
        raise SystemExit(f"No CameraInfo found on {args.camera_info_topic}")

    w = int(camera_info.width)
    h = int(camera_info.height)
    K = np.array(camera_info.K, dtype=np.float64).reshape(3, 3)
    fl_x, fl_y = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    D = list(camera_info.D)
    k1 = float(D[0]) if len(D) > 0 else 0.0
    k2 = float(D[1]) if len(D) > 1 else 0.0
    p1 = float(D[2]) if len(D) > 2 else 0.0
    p2 = float(D[3]) if len(D) > 3 else 0.0
    k3 = float(D[4]) if len(D) > 4 else 0.0

    depth_msgs = []
    joint_msgs = []
    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, t in read_messages(bag, [args.depth_topic, args.joint_states_topic]):
            ts = stamp_to_sec(msg.header.stamp)
            if topic == args.depth_topic:
                depth_msgs.append((ts, msg))
            elif topic == args.joint_states_topic:
                joint_msgs.append((ts, msg))

    if not depth_msgs:
        raise SystemExit(f"No depth messages found on {args.depth_topic}")
    if not joint_msgs:
        raise SystemExit(f"No joint states found on {args.joint_states_topic}")

    depth_times = [x[0] for x in depth_msgs]
    joint_times = [x[0] for x in joint_msgs]

    def latest_at_or_before(times, msgs, ts):
        import bisect
        idx = bisect.bisect_right(times, ts) - 1
        if idx < 0:
            return None, None
        return msgs[idx]

    def fk_base_to_camera(joint_state_msg):
        joint_map = dict(zip(joint_state_msg.name, joint_state_msg.position))
        T = np.eye(4, dtype=np.float64)
        for joint in chain:
            q = 0.0
            if joint.jtype in ("revolute", "continuous", "prismatic"):
                if joint.name not in joint_map:
                    raise RuntimeError(f"Joint '{joint.name}' missing from joint state message")
                q = float(joint_map[joint.name])
            T = T @ joint_transform(joint, q)
        return T

    frames = []
    num_rgb = 0
    num_written = 0
    depth_failures = 0
    joint_failures = 0

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, t in read_messages(bag, [args.rgb_topic]):
            num_rgb += 1
            ts = stamp_to_sec(msg.header.stamp)

            try:
                rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"Skipping RGB frame at {ts:.6f}: {e}")
                continue

            _, depth_msg = latest_at_or_before(depth_times, depth_msgs, ts)
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
                    depth_m = bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
            except Exception as e:
                print(f"Skipping depth at {ts:.6f}: {e}")
                depth_failures += 1
                continue

            _, joint_msg = latest_at_or_before(joint_times, joint_msgs, ts)
            if joint_msg is None:
                joint_failures += 1
                continue
            try:
                T_base_cam_ros = fk_base_to_camera(joint_msg)
            except Exception as e:
                print(f"Skipping frame at {ts:.6f}: FK failed: {e}")
                joint_failures += 1
                continue

            T_ns = make_ns_camera_to_world_from_ros_optical(T_base_cam_ros)

            frame_name = f"frame_{num_written:06d}"
            rgb_path = images_dir / f"{frame_name}.png"
            depth_path = depth_dir / f"{frame_name}.png"

            if not cv2.imwrite(str(rgb_path), rgb):
                continue
            depth_mm = sanitize_depth_to_uint16_mm(depth_m)
            if not cv2.imwrite(str(depth_path), depth_mm):
                rgb_path.unlink(missing_ok=True)
                continue

            frames.append({
                "file_path": f"images/{frame_name}.png",
                "depth_file_path": f"depth/{frame_name}.png",
                "transform_matrix": T_ns.tolist(),
            })
            num_written += 1
            if args.max_frames is not None and num_written >= args.max_frames:
                break

    transforms = {
        "camera_model": "OPENCV",
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
        "bag": args.bag,
        "urdf": args.urdf,
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "camera_info_topic": args.camera_info_topic,
        "joint_states_topic": args.joint_states_topic,
        "base_link": args.base_link,
        "camera_link": args.camera_link,
        "num_rgb_messages": num_rgb,
        "num_frames_written": num_written,
        "depth_failures": depth_failures,
        "joint_failures": joint_failures,
        "chain_joints": [j.name for j in chain],
        "notes": [
            "Pose is computed by URDF forward kinematics from base_link to camera_link using /joint_states.",
            "Depth images are written as uint16 PNG in millimeters.",
            "transform_matrix is camera-to-world with world := base_link and Nerfstudio/OpenGL camera convention.",
        ],
    }
    with open(out_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
