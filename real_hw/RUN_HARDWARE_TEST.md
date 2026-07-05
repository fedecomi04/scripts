# Hardware test — commands

Runnable from any folder (absolute paths). Jetson SSH + ZED camera + deps verified working.

```bash
# every shell
export ROS_MASTER_URI=http://192.168.55.100:11311   # PC master (direct single master)

# 1. robot bringup (your side, NimbRo) -> needs /dynaarm_arm/joint_states_full alive (gripper_source=topic)

# 2. Jetson ZED -> ROS camera topics (deploys + runs the node on the Jetson)
/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/real_hw/run_zed_publisher.sh

# 3. check topics
rostopic hz   /dynaarm_arm/dynaarm_arm/camera1/image_raw
rostopic echo -n1 /dynaarm_arm/dynaarm_arm/camera1/camera_info | grep -E "width|height|K:"
rostopic echo -n1 /dynaarm_arm/joint_states_full | head -3

# 4a. RECORD dataset (static + dynamic, no training) — Enter ends STATIC, Enter ends DYNAMIC
export DGS_REAL_HW_CAMERA=1
export DGS_POSE_TOPIC=/dynaarm_arm/joint_states_full   # skip the dead gazebo_pose preflight
/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/capture_only.sh real_dynaarm_recording

# 4b. OR run the live pipeline
export DGS_REAL_HW_CAMERA=1
# warm (reuse static_state.pt):
# DGS_REAL_HW_CAMERA=1 /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/dynamic_gs2/warm_live.sh "<data_dir>"

# viser: http://localhost:8081
```

## Cabling / physical rig (matters for mask + reliability)
- IF ZEDMINI GIVES ERRORS JUST UNPLUG BOTH ENDS OF THE USB CABLE WAIT 5  TO 10 SECONDS AND REPLUG   
- The ZED USB cable is NOT in the URDF, so cables drooping into the camera view are NOT masked out and
  leak into the scene. FIX physically: tape the cables flush along the arm links running BACK toward the
  base (camera looks along ELBOW +X — keep cables on the -X/back side, out of the frustum).
- Leave a small SERVICE LOOP at the camera mount (slack clamped to the link) so arm rotation flexes the
  loop, not the connector. This also prevents the mid-motion USB disconnects.
- ZED video needs a real USB 3.0 port: `lsusb -t` must show the camera (2b03 Video/uvcvideo) at 5000M
  under the 10000M root, NOT 12M/480M. A marginal USB3 link causes "CAMERA NOT DETECTED" on open AND can
  deliver torn left/right-stereo-in-one-frame artifacts. The 12M HID leg (2b03 IMU) is normal.
- NEVER unplug the camera while the publisher is streaming — it wedges the ZED firmware (needs power
  cycle / reboot to recover). Stop the publisher first, then unplug.

## Jetson clock sync (IMPORTANT — else the dynamic camera pose FREEZES)
The ZED camera (Jetson) and the robot joints publish on different machine clocks. A big skew makes the
dynamic-phase camera pose freeze (render stays still, FF inserts misaligned). The pipeline now:
- re-stamps camera+joints with the PC clock on receipt (robust to skew),
- runs a LAUNCH-TIME skew check: if camera<->joint stamps differ > 20 ms it warns and asks
  `[s]top or default to [l]atest-joints pose?`. Headless: set `DGS_CLOCK_SKEW=stop|latest`.

Permanent fix (Jetson chrony -> PC; re-do if Jetson is reflashed / chrony reset):
```bash
# PC: serve NTP to the Jetson subnet
echo "allow 192.168.55.0/24" | sudo tee -a /etc/chrony/chrony.conf
echo "local stratum 10"      | sudo tee -a /etc/chrony/chrony.conf
sudo systemctl restart chrony
# Jetson: add the PC as a source (ADDITIVE) + step now
ssh -t shengzhiwang@192.168.55.1 '
  echo "server 192.168.55.100 iburst minpoll 1 maxpoll 2" | sudo tee -a /etc/chrony/chrony.conf
  sudo systemctl restart chrony && sudo chronyc -a makestep'
# verify: ssh ...jetson "chronyc tracking"  -> Leap=Normal, "synchronized: yes", Ref=192-168-55-100
```
One-shot fallback (non-permanent, resets on reboot):
`ssh -t shengzhiwang@192.168.55.1 "sudo date -u -s @$(date -u +%s.%N)"`
