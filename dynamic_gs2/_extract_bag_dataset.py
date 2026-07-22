#!/usr/bin/env python
"""Extract a GT-annotated RGB dataset from the teleop bag (OFFLINE, no sim).

The bag recorded the object teleop but NOT depth (the sim was meant to
regenerate it on replay). This pulls out everything that IS in the bag, all on
the bag's single sim clock so frames + GT are exactly synced:

  dynamic_scene/rgb/frame_NNNNNN.png   RGB (BGR on disk, from camera1/image_raw)
  dynamic_scene/transforms.json        per-frame GT camera pose + intrinsics
  gt_object_trajectory.csv             per-frame GT screwdriver pose (t,x,y,z,qx,qy,qz,qw)
  gt_camera_trajectory.csv             per-frame GT camera  pose (t,x,y,z,qx,qy,qz,qw)
  intrinsics.json                      K, image size

Run with the dynamic_gs_ros env python (has rosbag + cv2).
NOTE: no depth -> the standard tracker cannot run on this as-is; depth must be
synthesized (mono-depth) or re-rendered in sim. This script only extracts.
"""
import json
import os

import numpy as np
import cv2
import rosbag

BAG = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/replays/teleop_20260611_192947.bag"
OUT = "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver_from_bag_20260611"
IMG_T = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
INFO_T = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
CAMPOSE_T = "/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose"
OBJ = "Craftsman_Grip_Screwdriver_Phillips_Cushion"


def quat_to_R(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def nearest(times, arr, t):
    i = int(np.searchsorted(times, t))
    i = max(0, min(i, len(arr) - 1))
    if i > 0 and abs(times[i - 1] - t) < abs(times[i] - t):
        i -= 1
    return arr[i]


def main():
    os.makedirs(os.path.join(OUT, "dynamic_scene", "rgb"), exist_ok=True)
    bag = rosbag.Bag(BAG)

    # intrinsics
    K = None
    for _, m, _ in bag.read_messages(topics=[INFO_T]):
        K = dict(fx=m.K[0], fy=m.K[4], cx=m.K[2], cy=m.K[5], w=m.width, h=m.height)
        break
    json.dump(K, open(os.path.join(OUT, "intrinsics.json"), "w"), indent=2)

    # camera GT poses (PoseStamped) and object GT poses (ModelStates)
    ct, cpose = [], []
    for _, m, t in bag.read_messages(topics=[CAMPOSE_T]):
        p, q = m.pose.position, m.pose.orientation
        ct.append(t.to_sec())
        cpose.append((p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    ct = np.asarray(ct); cpose = np.asarray(cpose)

    ot, opose = [], []
    for _, m, t in bag.read_messages(topics=["/gazebo/model_states"]):
        try:
            i = m.name.index(OBJ)
        except ValueError:
            continue
        p, q = m.pose[i].position, m.pose[i].orientation
        ot.append(t.to_sec())
        opose.append((p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    ot = np.asarray(ot); opose = np.asarray(opose)

    # RGB frames + build transforms + per-frame GT csvs
    frames = []
    gt_obj = open(os.path.join(OUT, "gt_object_trajectory.csv"), "w")
    gt_cam = open(os.path.join(OUT, "gt_camera_trajectory.csv"), "w")
    n = 0
    for _, m, t in bag.read_messages(topics=[IMG_T]):
        ts = t.to_sec()
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)  # rgb8
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = "frame_%06d.png" % n
        cv2.imwrite(os.path.join(OUT, "dynamic_scene", "rgb", name), bgr)

        cx = nearest(ct, cpose, ts)      # camera world pose (GT)
        ox = nearest(ot, opose, ts)      # object world pose (GT)
        R = quat_to_R(cx[3], cx[4], cx[5], cx[6])
        c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = cx[:3]   # camera link->world (raw)
        frames.append({"file_path": "rgb/%s" % name,
                       "transform_matrix": c2w.tolist(),
                       "stamp_wall": ts})
        gt_obj.write(",".join("%.6f" % v for v in (ts, *ox[:7])) + "\n")
        gt_cam.write(",".join("%.6f" % v for v in (ts, *cx[:7])) + "\n")
        n += 1
    gt_obj.close(); gt_cam.close()

    tj = {"fl_x": K["fx"], "fl_y": K["fy"], "cx": K["cx"], "cy": K["cy"],
          "w": K["w"], "h": K["h"], "frames": frames,
          "_note": "transform_matrix = raw Gazebo camera link->world pose; "
                   "may need OpenGL/OpenCV convention fix before training. NO DEPTH in bag."}
    json.dump(tj, open(os.path.join(OUT, "dynamic_scene", "transforms.json"), "w"))
    print("extracted %d RGB frames -> %s" % (n, OUT))
    print("intrinsics:", K)
    print("GT object rows:", len(ot), " GT camera rows:", len(ct))


if __name__ == "__main__":
    main()
