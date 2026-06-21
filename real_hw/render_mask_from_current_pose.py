#!/usr/bin/env python3
"""One-shot: grab the CURRENT gazebo joints + RGB + camera_info, render robot-exclusion masks.

Renders two masks from the SAME current joint configuration:
  A) the current SIM camera frame  (camera_pose_link, the mask frame the pipeline uses today)
  B) the proposed ELBOW-mounted camera (dynaarm_ELBOW FK  *  measured fixed offset)

Both poses come from URDF FK (urdfpy) on the live joints -- NO gazebo camera is needed for (B),
which is exactly how the real-HW elbow camera will resolve its pose. We also dump gazebo's own
published camera pose for cross-checking the FK of the *sim* camera.

Run in the ROS env:
  /home/mrc-cuhk/miniconda3/envs/dynamic_gs_ros/bin/python real_hw/render_mask_from_current_pose.py

Outputs -> real_hw/mask_out/:
  rgb_live.png            live gazebo RGB at this instant
  mask_simcam.png         robot-exclusion mask from the current sim camera frame
  mask_elbowcam.png       robot-exclusion mask from the proposed elbow camera
  overlay_simcam.png      rgb with sim-cam mask edge (green) -- should hug the robot in the live image
  poses.txt               the FK + gazebo poses, side by side
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Use the same GL path as the live publisher. A real X display (:1) is present, so let pyrender
# use the default platform unless the caller forces one via PYOPENGL_PLATFORM. EGL offscreen
# segfaulted on this box; the publisher renders fine on the default path.

import numpy as np
# urdfpy (py3.8) still uses removed aliases np.float/np.int; shim them for modern numpy.
for _alias, _real in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped

# import the pipeline's mask module by path (same as the publisher does)
ROS_MASK = Path(__file__).resolve().parent.parent / "dynamic_gs2" / "ros_mask.py"
import importlib.util
spec = importlib.util.spec_from_file_location("ros_mask", ROS_MASK)
rm = importlib.util.module_from_spec(spec)
sys.modules["ros_mask"] = rm          # dataclass needs the module visible in sys.modules
spec.loader.exec_module(rm)

OUT = Path(__file__).resolve().parent / "mask_out"
OUT.mkdir(exist_ok=True)

# ---- the measured elbow camera mount (ELBOW link frame) ----
ELBOW_LINK = "dynaarm_ELBOW"
ELBOW_CAM_XYZ = np.array([0.05000, -0.16330, 0.03150])
ELBOW_CAM_R = np.array([[0, 0, 1.0],
                        [0, 1, 0],
                        [-1, 0, 0]])   # ELBOW_from_camera (looks +X, up=-Y), pitch +90 about Y


def grab_one():
    """Block for one synchronized-ish snapshot of the live topics + a short joint buffer."""
    state = {"img": None, "info": None, "gz": None, "js_t": [], "js_p": []}

    def on_img(m): state["img"] = m
    def on_info(m): state["info"] = m
    def on_gz(m): state["gz"] = m
    def on_js(m):
        state["js_t"].append(m.header.stamp.to_sec())
        state["js_p"].append({n: p for n, p in zip(m.name, m.position)})

    rospy.Subscriber(rm.IMAGE_TOPIC, Image, on_img, queue_size=2)
    rospy.Subscriber(rm.CAMERA_INFO_TOPIC, CameraInfo, on_info, queue_size=2)
    rospy.Subscriber(rm.GAZEBO_CAMERA_POSE_TOPIC, PoseStamped, on_gz, queue_size=2)
    rospy.Subscriber(rm.GAZEBO_JOINT_STATES_TOPIC, JointState, on_js, queue_size=50)

    rospy.loginfo("waiting for image/info/joints ...")
    t0 = rospy.Time.now()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if state["img"] and state["info"] and len(state["js_t"]) >= 5:
            break
        if (rospy.Time.now() - t0).to_sec() > 20:
            raise RuntimeError(f"timeout. have img={bool(state['img'])} "
                               f"info={bool(state['info'])} njs={len(state['js_t'])}")
        rate.sleep()
    # collect a little more joint history so interpolation around the image stamp is clean
    while len(state["js_t"]) < 12 and not rospy.is_shutdown():
        rate.sleep()
    return state


def main():
    rospy.init_node("render_mask_oneshot", anonymous=True, disable_signals=True)
    s = grab_one()

    info = s["info"]
    intr = rm.CameraIntrinsics(
        width=info.width, height=info.height,
        fx=info.K[0], fy=info.K[4], cx=info.K[2], cy=info.K[5],
    )
    rospy.loginfo("intrinsics %dx%d fx=%.2f cx=%.2f cy=%.2f", intr.width, intr.height, intr.fx, intr.cx, intr.cy)

    # sort joint buffer by time (RobotMaskGenerator assumes ascending)
    order = np.argsort(s["js_t"])
    js_t = [s["js_t"][i] for i in order]
    js_p = [s["js_p"][i] for i in order]

    gen = rm.RobotMaskGenerator(intr, js_t, js_p)

    # use the image stamp as the query time
    stamp = s["img"].header.stamp

    # save the live RGB
    bgr = rm.ros_image_to_bgr(s["img"])
    cv2.imwrite(str(OUT / "rgb_live.png"), bgr)

    # --- (A) mask from the current SIM camera frame ---
    mask_sim = gen._render_robot_exclusion_mask(stamp, rm.MASK_RENDER_CAMERA_FRAME)
    cv2.imwrite(str(OUT / "mask_simcam.png"), mask_sim)

    # FK pose of the sim camera (for the report) + gazebo's published pose
    joints = gen._sample_joint_positions(stamp)
    link_fk = gen.robot.link_fk(cfg=joints, use_names=True)
    sim_cam_pose = gen._camera_pose_from_link_fk(link_fk, rm.MASK_RENDER_CAMERA_FRAME)

    # --- (B) mask from the proposed ELBOW camera (FK of elbow * measured offset) ---
    T_elbow = link_fk[ELBOW_LINK].astype(np.float32)
    T_off = np.eye(4, dtype=np.float32)
    T_off[:3, :3] = ELBOW_CAM_R
    T_off[:3, 3] = ELBOW_CAM_XYZ
    elbow_cam_pose = T_elbow @ T_off
    # render by temporarily driving the camera node with this pose
    gen._ensure_renderer()
    gen._update_robot_poses(link_fk)
    # elbow_cam_pose is a true OPTICAL frame (+Z = viewing dir), so build WITHOUT the body-frame remaps.
    gen.scene.set_pose(gen.camera_node, pose=gen._build_render_camera_pose(elbow_cam_pose, optical=True))
    _, depth = gen.renderer.render(gen.scene)
    # No vertical flip here: with the optical build-pose (+Z forward, +Y down = image-down) the
    # render already comes out upright. The sim path's cv2.flip(...,0) is for its body-frame convention.
    mask_elbow = (depth == 0).astype(np.uint8) * 255
    if mask_elbow.shape[1] != intr.width or mask_elbow.shape[0] != intr.height:
        mask_elbow = cv2.resize(mask_elbow, (intr.width, intr.height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "mask_elbowcam.png"), mask_elbow)

    # --- overlay sim-cam mask edge on the live rgb (alignment sanity) ---
    edge = cv2.morphologyEx((mask_sim == 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ov = bgr.copy(); ov[edge > 0] = (0, 255, 0)
    cv2.imwrite(str(OUT / "overlay_simcam.png"), ov)

    # --- report poses ---
    gz = s["gz"].pose if s["gz"] else None
    with open(OUT / "poses.txt", "w") as f:
        f.write("intrinsics: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n\n" %
                (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy))
        f.write("joints used:\n  " + "\n  ".join(f"{k}={v:+.4f}" for k, v in sorted(joints.items())) + "\n\n")
        f.write("SIM camera FK pose (camera_pose_link), c2w:\n%s\n\n" % np.array2string(sim_cam_pose, precision=4))
        if gz is not None:
            f.write("GAZEBO published camera pose: pos=(%.4f, %.4f, %.4f) quat=(%.4f,%.4f,%.4f,%.4f)\n\n" %
                    (gz.position.x, gz.position.y, gz.position.z,
                     gz.orientation.x, gz.orientation.y, gz.orientation.z, gz.orientation.w))
        f.write("ELBOW FK pose, c2w:\n%s\n\n" % np.array2string(T_elbow, precision=4))
        f.write("ELBOW camera FK pose (ELBOW * offset), c2w:\n%s\n" % np.array2string(elbow_cam_pose, precision=4))

    rospy.loginfo("mask_sim keep%%=%.1f  mask_elbow keep%%=%.1f",
                  100 * (mask_sim > 0).mean(), 100 * (mask_elbow > 0).mean())
    rospy.loginfo("wrote outputs to %s", OUT)
    print("\n".join(str(p) for p in sorted(OUT.glob("*"))))


if __name__ == "__main__":
    main()
