# Hardware test — one-shot runbook (ZED Mini on the elbow)

Real camera = ZED Mini on `dynaarm_ELBOW`. Camera **frames + saved pose** come from URDF FK on the
PC (NOT from Gazebo), gated by `DGS_REAL_HW_CAMERA=1`. The Jetson publishes ONLY the camera topics.

## The one switch that changes everything
```bash
export DGS_REAL_HW_CAMERA=1
```
With it set, the PC pipeline:
- renders the robot-exclusion MASK from the elbow optical camera (FK of dynaarm_ELBOW × measured offset),
- writes the saved camera POSE (transforms.json) from that SAME FK, converted to OpenGL c2w,
- does NOT read `/dynaarm_arm/.../camera1/gazebo_pose` at all (see "rigid link" note below).

Unset (default) = sim behaviour, byte-for-byte.

---

## STEP 1 — robot bringup (arm + joints + TF), NO sim camera needed
The arm/joint_states_full + TF must be up (same as sim, the arm is unchanged). Real robot driver
replaces Gazebo. You still need:
- `/dynaarm_arm/joint_states_full`   (from joint_state_merger — set gripper_source=topic for the real gripper)
- ROS master reachable at `ROS_MASTER_URI`

joint_state_merger for the REAL gripper (finger angle from a topic, not the Gazebo service):
```bash
rosrun <pkg> joint_state_merger.py _gripper_source:=topic \
    _gripper_topic:=/arm_1/gripper/joint_states _gripper_joint_name:=finger_joint
```

## STEP 2 — Jetson: ZED Mini → ROS camera topics (run from the PC)
```bash
cd /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
# point at the robot master if not localhost:
export ROS_MASTER_URI=http://<PC-or-robot-master>:11311
./real_hw/run_zed_publisher.sh
```
Publishes (MUST match the PC subscriber — verified identical):
- `/dynaarm_arm/dynaarm_arm/camera1/image_raw(/compressed)`  RGB
- `/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw`         32FC1 metres
- `/dynaarm_arm/dynaarm_arm/camera1/camera_info`             K from the SDK (NEVER sim values)

Note: the Jetson docstring still says "pose comes from gazebo_pose" — that line is now STALE for the
elbow build. With DGS_REAL_HW_CAMERA=1 the pose is FK on the PC; nothing needs to publish gazebo_pose.

## STEP 3 — sanity-check topics BEFORE launching the pipeline
```bash
rostopic hz   /dynaarm_arm/dynaarm_arm/camera1/image_raw      # ZED RGB alive
rostopic echo -n1 /dynaarm_arm/dynaarm_arm/camera1/camera_info | grep -E "width|height|K:"   # REAL K, real res
rostopic echo -n1 /dynaarm_arm/joint_states_full | head -3    # joints alive
# camera_info width/height + cx must be the ZED's, NOT 1920x1200 cx=960.5 (that's the sim fallback bug).
```

## STEP 4 — launch the pipeline (PC)
```bash
cd /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
export DGS_REAL_HW_CAMERA=1
# FULL (capture static -> dynamic), live:
dynamic_gs2/full_live.sh "../data_teleoperation/datasets/$(date +%Y-%m-%d_%H%M%S)" "<prompt>"
# OR WARM (skip static, reuse static_state.pt):
# dynamic_gs2/full_live.sh writes static_state.pt; later:
# DGS_REAL_HW_CAMERA=1 dynamic_gs2/warm_live.sh "<data_dir>"
```
Viser live view: http://localhost:8081  (NOT :7007).

---

## The "rigid link stuff" in the URDF — what it is, and is it useless now?
In the gazebo URDF there is a Gazebo plugin:
```xml
<plugin filename="libactive_camera_arm_link_pose_publisher.so" name="camera_pose_link_publisher">
  <linkName>dynaarm_WRIST_2_base</linkName>
  <referenceLinkName>dynaarm_base</referenceLinkName>
  <linkPoseOffset>0.1 0 0.0 0 -1.39626 -3.14159</linkPoseOffset>
  <topicName>/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose</topicName>
</plugin>
```
This is the "rigid link" pose publisher: a GAZEBO-ONLY plugin that took WRIST_2_base, applied the
fixed camera offset, and published the camera pose on `gazebo_pose` at 250 Hz. The old real-HW plan
reused that topic for the pose.

- On hardware there is NO Gazebo, so this plugin does not run — it produced the sim pose only.
- With DGS_REAL_HW_CAMERA=1 the PC computes the pose by FK itself (`elbow_camera_optical_pose`), so
  `gazebo_pose` is NOT consumed. The plugin is therefore **useless for the hardware test** (it was the
  sim's way of doing the same FK the PC now does directly). It is NOT useless for SIM — leave it in the
  URDF; sim still uses it when DGS_REAL_HW_CAMERA is unset.
- The offset it bakes (`0.1 0 0.0 0 -1.39626 -3.14159` off WRIST_2_base) is the OLD wrist mount, NOT the
  elbow mount. Do not copy it for the elbow camera — the elbow values live in ros_mask.py
  (REAL_HW_CAMERA_XYZ / REAL_HW_CAMERA_ROT).
```
```

## Verified wiring (2026-06-21)
- Topic names identical in zed_mini_publisher.py and dynamic_gs2/ros_mask.py (image/depth/camera_info).
- Under DGS_REAL_HW_CAMERA=1: mask + saved pose both from FK(dynaarm_ELBOW)×offset; gazebo_pose unused.
- Saved pose convention: ROS optical → OpenGL c2w via rotate_camera_frame_only (diag(1,-1,-1)).
- Production render path runs EXIT 0; mask upright; example saved cam world pos for a test config.
- NOT yet validated against a live ZED frame (no real capture run yet) — orientation rpy unconfirmed
  on real data; refine with eye-in-hand hand-eye calib if the camera image disagrees with the mask.
