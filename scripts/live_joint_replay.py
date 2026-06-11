#!/usr/bin/env python
"""Replay a recorded teleop rosbag by FORCING the Gazebo sim state to track it.

The camera arm is driven by a stateful teleop (MTM masters + state machine +
clutch) whose inputs can't be faithfully replayed, and the controller overrides
one-shot joint sets. But forcing the camera-arm joints via
/gazebo/set_model_configuration at ~60 Hz OUT-PACES the controller (measured:
holds the recorded pose to ~20 mm) — so we can reproduce the exact camera
trajectory WITHOUT killing the live teleop. Object poses (free bodies, no
controller) are forced via /gazebo/set_model_state.

This drives the LIVE pipeline: run this alongside the capture / dynamic-gs-live
publisher; the publisher reads the camera (rendering the forced poses) + the
live /joint_states (which reflect the forced arm) and runs normally — so the
whole pipeline replays deterministically against the recorded motion, no arm
operator needed.

Usage (ROS env):
  <ros_py> scripts/live_joint_replay.py --bag <bag> --start 0   --duration 46   # static sweep
  <ros_py> scripts/live_joint_replay.py --bag <bag> --start 46             \
           --objects Craftsman_Grip_Screwdriver_Phillips_Cushion              # dynamic teleop
Reset Gazebo (Ctrl+R / rosservice call /gazebo/reset_world) before the STATIC
segment so the arm starts at the same home pose the recording started from.
"""
from __future__ import print_function
import argparse
import bisect
import time

import rospy
import rosbag
from gazebo_msgs.srv import SetModelConfiguration, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import JointState

CAMERA_ARM_MODEL = "Dynaarm_Arm"
# 6-DOF arm + the 2 gripper joints (joint1/joint2). Forcing only the arm left
# the gripper open; include the gripper so grasps reproduce.
CAMERA_ARM_JOINTS = ["SH_ROT", "SH_FLE", "EL_FLE", "FA_ROT", "WRIST_1", "WRIST_2", "joint1", "joint2"]


def load_bag(bag_path, object_models):
    b = rosbag.Bag(bag_path)
    t0 = b.get_start_time()
    joints = []   # (rel_t, [positions for CAMERA_ARM_JOINTS present])
    js_msgs = []  # (rel_t, raw JointState) for republishing (mask FK source)
    for _, m, t in b.read_messages(topics=["/joint_states"]):
        rt = t.to_sec() - t0
        js_msgs.append((rt, m))
        n2p = dict(zip(m.name, m.position))
        present = [j for j in CAMERA_ARM_JOINTS if j in n2p]
        if present:
            joints.append((rt, present, [n2p[j] for j in present]))
    objs = {om: [] for om in object_models}
    if object_models:
        for _, m, t in b.read_messages(topics=["/gazebo/model_states"]):
            rt = t.to_sec() - t0
            for om in object_models:
                if om in m.name:
                    objs[om].append((rt, m.pose[m.name.index(om)]))
    b.close()
    return joints, objs, js_msgs


def _nearest(samples, times, t):
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return samples[0]
    if i >= len(samples):
        return samples[-1]
    return samples[i] if abs(times[i] - t) < abs(times[i - 1] - t) else samples[i - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=1e9)
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--objects", type=str, default="")
    ap.add_argument("--republish-joints", action="store_true",
                    help="Publish recorded /joint_states (use when the teleop controller "
                         "is killed, so the pipeline mask FK still gets joints).")
    args = ap.parse_args()

    object_models = [o for o in args.objects.split(",") if o.strip()]
    joints, objs, js_msgs = load_bag(args.bag, object_models)
    if not joints:
        print("[replay] no /joint_states with camera-arm joints in bag", flush=True)
        return 1
    jt = [s[0] for s in joints]
    jst = [s[0] for s in js_msgs]
    ot = {om: [s[0] for s in objs[om]] for om in object_models}
    span = (jt[0], jt[-1])
    print("[replay] %d joint samples (bag t=[%.1f,%.1f]); objects=%s; start=%.1f dur=%.1f"
          % (len(joints), span[0], span[1], object_models, args.start, args.duration), flush=True)

    rospy.init_node("live_joint_replay", anonymous=True, disable_signals=True)
    rospy.wait_for_service("/gazebo/set_model_configuration", timeout=10)
    setcfg = rospy.ServiceProxy("/gazebo/set_model_configuration", SetModelConfiguration)
    setstate = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    jpub = rospy.Publisher("/joint_states", JointState, queue_size=10) if args.republish_joints else None

    t_wall0 = time.time()
    r = rospy.Rate(args.rate)
    n = 0
    last_log = 0.0
    while not rospy.is_shutdown():
        e = time.time() - t_wall0
        if e > args.duration:
            break
        bt = args.start + e
        if bt > span[1]:
            break
        js = _nearest(joints, jt, bt)
        try:
            setcfg(CAMERA_ARM_MODEL, "", js[1], js[2])
        except Exception:
            pass
        if jpub is not None:
            jm = _nearest(js_msgs, jst, bt)[1]
            jm.header.stamp = rospy.Time.now()
            try:
                jpub.publish(jm)
            except Exception:
                pass
        for om in object_models:
            if objs[om]:
                os_ = _nearest(objs[om], ot[om], bt)
                ms = ModelState()
                ms.model_name = om
                ms.pose = os_[1]
                ms.reference_frame = "world"
                try:
                    setstate(ms)
                except Exception:
                    pass
        n += 1
        if e - last_log >= 5.0:
            print("[replay] t=+%.1fs (bag %.1fs), %d forces" % (e, bt, n), flush=True)
            last_log = e
        r.sleep()
    print("[replay] done: %d forces over %.1fs" % (n, time.time() - t_wall0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
