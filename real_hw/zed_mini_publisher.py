#!/usr/bin/env python3
"""ZED Mini -> ROS publisher for the live dynamic-gaussian-splat pipeline.

Runs ON THE JETSON. It replaces the Gazebo sim camera: it publishes the same
three camera topics the PC live pipeline already subscribes to, so the PC-side
live_ros_publisher.py consumes them unchanged.

    RGB    <IMAGE_TOPIC>/compressed   sensor_msgs/CompressedImage  (JPEG, BGR)
    DEPTH  <DEPTH_TOPIC>              sensor_msgs/Image  32FC1 float32 METRES
    INFO   <CAMERA_INFO_TOPIC>        sensor_msgs/CameraInfo (rectified K @ streamed res)

RGB and DEPTH carry one shared timestamp per frame; the PC syncs them with an
approximate-time filter. The ZED depth map is the LEFT-rectified depth and is
already pixel-aligned to the left image, so no registration step is needed.

Camera config: HD1080 (1920x1080) NEURAL, confidence_threshold=60, texture_confidence=100
(measured on this Jetson: ~6 ms depth compute, ~18 ms grab+retrieve end-to-end,
~54 Hz). NEURAL fills textureless surfaces that the stereo modes leave holed.

INTRINSICS are read FROM THE SDK at the streamed resolution at startup and
written verbatim into CameraInfo.K. They are never hardcoded and never the sim
values -- a mismatch here is what put inserted objects in the wrong place.

This node only handles the CAMERA. The camera POSE comes separately from robot
FK (a PoseStamped on GAZEBO_CAMERA_POSE_TOPIC), and the joints come from the
joint_state_merger -- both live outside this file.

Launch (on the Jetson -- rospy lives in the system ROS env, pyzed in the venv):
    source /opt/ros/noetic/setup.bash
    export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
    /home/shengzhiwang/zed_env/bin/python zed_mini_publisher.py

Note: do NOT add /usr/lib/python3/dist-packages to PYTHONPATH -- its older numpy
shadows the venv numpy and breaks the pyzed Cython ABI. The venv already has cv2.
"""
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import pyzed.sl as sl

# Topic names -- MUST match dynamic_gs2/ros_mask.py (IMAGE_TOPIC / DEPTH_TOPIC / CAMERA_INFO_TOPIC).
IMAGE_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/image_raw"
DEPTH_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/depth/image_raw"
CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"
CAMERA_FRAME_ID = "camera_link_optical"
JPEG_QUALITY = 80


def build_camera_info(left_cam, width, height, frame_id):
    """CameraInfo for the rectified left image: K from the SDK, zero distortion."""
    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.D = [0.0, 0.0, 0.0, 0.0, 0.0]            # rectified left image -> no distortion
    fx, fy, cx, cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
    info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info


# Torn-frame (frame-tearing / stereo-desync) rejection. The ZED occasionally delivers a frame stitched
# from two captures (half old + half new) — visible as a horizontal SEAM where content jumps. It happens
# at the SDK/USB level (seen on custom pyzed code too, not just this publisher), and only LOOKS like a bug
# when the scene moves (static frames are identical so the seam is invisible). We detect the seam as a
# single row whose row-to-row brightness jump is a large outlier vs the rest of the frame, and drop it.
_TEAR_REJECT = os.environ.get("DGS_ZED_TEAR_REJECT", "1") != "0"
_TEAR_RATIO = float(os.environ.get("DGS_ZED_TEAR_RATIO", "8.0"))   # seam must exceed this x the median row-jump
_TEAR_ABS = float(os.environ.get("DGS_ZED_TEAR_ABS", "18.0"))      # AND exceed this absolute (8-bit) jump


def _is_torn_frame(bgr) -> bool:
    """True if the frame has a tear seam (one row with an outlier brightness discontinuity)."""
    if not _TEAR_REJECT:
        return False
    g = bgr[:, :, 1].astype(np.float32)          # green channel ~ luma, cheap
    row_mean = g.mean(axis=1)                     # (H,)
    jumps = np.abs(np.diff(row_mean))             # (H-1,) row-to-row change
    if jumps.size < 8:
        return False
    med = float(np.median(jumps)) + 1e-3
    mx = float(jumps.max())
    return mx > _TEAR_ABS and mx > _TEAR_RATIO * med


def open_camera():
    """Open the ZED Mini at HD1080 (1920x1080) NEURAL with metric depth. Returns the camera."""
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080   # 1920x1080 (keep resolution for segmentation quality)
    # 10 fps not 30: at 1080p@30 the USB3 link dropped ~96% of grabs (0.8 Hz). The marginal ZED<->Jetson
    # USB3 link gets worse under the dynamic phase's full-rate + PC compute load -> grab FAILURE + heavy
    # frame tearing. Lower fps = less USB3 data = fewer grab failures + tears. 10 fps is plenty for slow
    # manipulation. Override with DGS_ZED_FPS (e.g. =15 if the link is solid).
    init.camera_fps = int(os.environ.get("DGS_ZED_FPS", "10"))
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER          # depth comes out in metres directly
    init.depth_minimum_distance = 0.1
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed: %s" % status)
    return cam, init       # return init too so main() can reopen the camera on a grab stall


def main():
    rospy.init_node("zed_mini_publisher")

    cam, init = open_camera()
    cam_info = cam.get_camera_information()
    left_cam = cam_info.camera_configuration.calibration_parameters.left_cam
    res = cam_info.camera_configuration.resolution
    width, height = int(res.width), int(res.height)
    rospy.loginfo("[zed_mini] HD1080 NEURAL up: %dx%d  fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                  width, height, left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 60
    runtime.texture_confidence_threshold = 100

    pub_rgb = rospy.Publisher(IMAGE_TOPIC + "/compressed", CompressedImage, queue_size=2)
    pub_depth = rospy.Publisher(DEPTH_TOPIC, Image, queue_size=2)
    pub_info = rospy.Publisher(CAMERA_INFO_TOPIC, CameraInfo, queue_size=2, latch=True)

    info_msg = build_camera_info(left_cam, width, height, CAMERA_FRAME_ID)

    left_mat = sl.Mat()
    depth_mat = sl.Mat()

    # Grab-stall recovery. A marginal USB link can make grab() fail FOREVER mid-run -> the old loop spun
    # on `continue` silently and BOTH rqt and the SHM froze (live process, zero frames). Detect a run of
    # consecutive failures, log it, and reopen the camera to re-establish the link; if reopening keeps
    # failing, exit loudly so the launcher restarts cleanly instead of hanging.
    consecutive_fail = 0
    GRAB_FAIL_REOPEN = int(os.environ.get("DGS_ZED_GRAB_FAIL_REOPEN", "30"))   # ~consecutive fails -> reopen
    REOPEN_MAX = int(os.environ.get("DGS_ZED_REOPEN_MAX", "5"))                # give up after N reopen tries
    reopen_tries = 0

    while not rospy.is_shutdown():
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            consecutive_fail += 1
            if consecutive_fail % 10 == 0:
                rospy.logwarn("[zed_mini] grab failing (%d consecutive) — camera/USB link may be dropping",
                              consecutive_fail)
            if consecutive_fail >= GRAB_FAIL_REOPEN:
                reopen_tries += 1
                if reopen_tries > REOPEN_MAX:
                    rospy.logfatal("[zed_mini] camera unrecoverable after %d reopen attempts — exiting "
                                   "(check USB link/cable; relaunch run_zed_publisher.sh)", REOPEN_MAX)
                    break
                rospy.logwarn("[zed_mini] reopening camera (attempt %d/%d) to recover the link",
                              reopen_tries, REOPEN_MAX)
                try:
                    cam.close()
                except Exception:
                    pass
                rospy.sleep(1.0)
                if cam.open(init) == sl.ERROR_CODE.SUCCESS:
                    rospy.logwarn("[zed_mini] camera reopened OK")
                    consecutive_fail = 0
                else:
                    rospy.logwarn("[zed_mini] reopen failed; will retry")
                    rospy.sleep(2.0)
                consecutive_fail = 0   # reset the counter either way so we don't reopen every frame
            continue
        if consecutive_fail:
            rospy.loginfo("[zed_mini] grab recovered after %d failures", consecutive_fail)
            consecutive_fail = 0
            reopen_tries = 0
        stamp = rospy.Time.now()

        # RGB: ZED left image is BGRA; drop alpha -> BGR, JPEG-encode.
        # cv2 round-trips BGR, and the PC decoder expects BGR, so colours stay correct.
        cam.retrieve_image(left_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(left_mat.get_data()[:, :, :3])
        # Drop torn/stereo-desync frames before they reach the pipeline (the "half of both images" bug).
        if _is_torn_frame(bgr):
            rospy.logwarn_throttle(2.0, "[zed_mini] dropped a torn frame (tear seam detected)")
            continue
        ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        rgb_msg = CompressedImage()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = CAMERA_FRAME_ID
        rgb_msg.format = "jpeg"
        rgb_msg.data = jpg.tobytes()

        # DEPTH: float32 metres, invalid (nan/+-inf) -> 0, raw 32FC1 (never compressed transport).
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = np.nan_to_num(np.asarray(depth_mat.get_data(), dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.ascontiguousarray(depth)
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = CAMERA_FRAME_ID
        depth_msg.height = height
        depth_msg.width = width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = width * 4
        depth_msg.data = depth.tobytes()

        # Same stamp on all three so the PC sync matches RGB to depth.
        info_msg.header.stamp = stamp
        pub_info.publish(info_msg)
        pub_rgb.publish(rgb_msg)
        pub_depth.publish(depth_msg)

    cam.close()


if __name__ == "__main__":
    main()
