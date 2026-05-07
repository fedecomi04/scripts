"""Live ROS subscriber for `ns-train dynamic-gs --live`.

Subscribes once to RGB / depth / camera_info / joint_states / gazebo_pose,
holds the most recent synchronised tuple under a lock, and exposes a
small set of methods used by the pre-training session (recording to
disk during the static capture window) and by the dynamic loop
(`peek_latest()` per tracker tick).

The URDF mask generator + ROS message helpers are pulled directly from
`save_data_img_depth_mask_pose.py` via `importlib` — that file is the
authority for how this rig produces masks and intrinsics, and we don't
want to keep two copies of that logic in sync.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nerfstudio.cameras.cameras import Cameras, CameraType
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf.transformations import (
    quaternion_from_matrix,
    quaternion_slerp,
)


# Hardcoded data root for live runs. Wiped + recreated at the start of
# every session.
LIVE_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/live"
)

# Sibling recording script — single source of truth for ROS topics,
# intrinsics handling, URDF mask rendering, and pose interpolation.
_RECORDER_SCRIPT = Path(__file__).resolve().parents[2] / "save_data_img_depth_mask_pose.py"


def _load_recorder_module():
    """Import save_data_img_depth_mask_pose.py as a regular module.

    The file lives outside any package, so a normal `import` would fail.
    `importlib.util` runs the module top-level (defining classes and
    constants) without triggering its `if __name__ == "__main__"` block.
    """
    spec = importlib.util.spec_from_file_location("_dgs_live_recorder", _RECORDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load recorder module from {_RECORDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dgs_live_recorder"] = module
    spec.loader.exec_module(module)
    return module
    # If the ROS env is misconfigured (PYTHONPATH missing the catkin overlay)
    # this raises ImportError on the cv_bridge / message_filters chain rather
    # than failing later — catch by sourcing the workspace before launching.


_REC = _load_recorder_module()

# Republished for convenience.
ros_image_to_bgr = _REC.ros_image_to_bgr
ros_depth_to_uint16_mm = _REC.ros_depth_to_uint16_mm
pose_msg_to_matrix = _REC.pose_msg_to_matrix
compose_transform_matrix = _REC.compose_transform_matrix
rotate_camera_frame_only = _REC.rotate_camera_frame_only
write_ascii_ply = _REC.write_ascii_ply
distribute_point_budget_evenly = _REC.distribute_point_budget_evenly
load_saved_depth_mm = _REC.load_saved_depth_mm
load_saved_mask = _REC.load_saved_mask
load_saved_rgb = _REC.load_saved_rgb
CameraIntrinsics = _REC.CameraIntrinsics
RobotMaskGenerator = _REC.RobotMaskGenerator
IMAGE_TOPIC = _REC.IMAGE_TOPIC
DEPTH_TOPIC = _REC.DEPTH_TOPIC
CAMERA_INFO_TOPIC = _REC.CAMERA_INFO_TOPIC
GAZEBO_JOINT_STATES_TOPIC = _REC.GAZEBO_JOINT_STATES_TOPIC
GAZEBO_CAMERA_POSE_TOPIC = _REC.GAZEBO_CAMERA_POSE_TOPIC
MASK_RENDER_CAMERA_FRAME = _REC.MASK_RENDER_CAMERA_FRAME
CAMERA_POSE_SAVE_FRAME = _REC.CAMERA_POSE_SAVE_FRAME
INIT_CLOUD_NAME = _REC.INIT_CLOUD_NAME
MAX_INIT_CLOUD_POINTS = _REC.MAX_INIT_CLOUD_POINTS
TIME_EPS_SEC = _REC.TIME_EPS_SEC
SYNC_QUEUE_SIZE = _REC.SYNC_QUEUE_SIZE
SYNC_SLOP_SEC = _REC.SYNC_SLOP_SEC
IMAGE_NAME_PREFIX = _REC.IMAGE_NAME_PREFIX


@dataclass
class LiveFrame:
    """One synchronised live tuple. Everything is on CPU as numpy/host
    types; the consumer moves what it needs to GPU. ``stamp_sec`` is
    monotonic-ish (gazebo sim time) and is used as the dedup key by
    `_tracker_tick_live`."""

    seq: int
    stamp_sec: float
    rgb_bgr: np.ndarray  # (H, W, 3) uint8
    depth_mm: np.ndarray  # (H, W) uint16, scale 1e-3 m/unit
    mask_keep: np.ndarray  # (H, W) uint8, 255 = keep, 0 = robot/background
    c2w_4x4: np.ndarray  # (4, 4) float64, Nerfstudio convention (y-up, z-back)


class LiveRosSubscriber:
    """Single-process holder for the latest live tuple.

    Threading model: rospy spins callbacks on its own background thread
    (we never call ``rospy.spin()``); we hand them a lock and a single
    output slot. Any consumer just calls ``peek_latest()``.
    """

    _singleton: Optional["LiveRosSubscriber"] = None

    def __init__(self) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node("dynamic_gs_live", disable_signals=True, anonymous=True)

        info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, timeout=10.0)
        self.intrinsics = CameraIntrinsics(
            width=int(info_msg.width),
            height=int(info_msg.height),
            fx=float(info_msg.K[0]),
            fy=float(info_msg.K[4]),
            cx=float(info_msg.K[2]),
            cy=float(info_msg.K[5]),
        )

        self._joint_state_times_sec: list[float] = []
        self._joint_state_positions: list[dict] = []
        self._gazebo_pose_times_sec: list[float] = []
        self._gazebo_pose_matrices: list[np.ndarray] = []
        self._mask_gen: Optional[RobotMaskGenerator] = None

        self._lock = threading.Lock()
        self._latest: Optional[LiveFrame] = None
        self._frame_seq = 0

        # Disk capture (static window only).
        self._record_lock = threading.Lock()
        self._record_active = False
        self._record_dir: Optional[Path] = None
        self._record_meta: Optional[dict] = None
        self._record_frames_written: list[dict] = []
        self._record_stamps: list[rospy.Time] = []

        # Subscribers (kept on the instance so they aren't GC'd).
        self._joint_sub = rospy.Subscriber(
            GAZEBO_JOINT_STATES_TOPIC,
            JointState,
            self._on_joint_state,
            queue_size=200,
        )
        self._pose_sub = rospy.Subscriber(
            GAZEBO_CAMERA_POSE_TOPIC,
            PoseStamped,
            self._on_gazebo_pose,
            queue_size=200,
        )
        self._rgb_sub = Subscriber(IMAGE_TOPIC, Image)
        self._depth_sub = Subscriber(DEPTH_TOPIC, Image)
        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        self._sync.registerCallback(self._on_synced)

        LiveRosSubscriber._singleton = self
        # PROBLEM: if the ROS clock is `/use_sim_time=True` but no /clock
        # is publishing, ApproximateTimeSynchronizer will hold messages
        # forever waiting for a stamp pair — `wait_until_ready` will hang
        # silently. Verify gazebo is actually running before launching.

    @classmethod
    def get_singleton(cls) -> "LiveRosSubscriber":
        if cls._singleton is None:
            return cls()
        return cls._singleton

    def _on_joint_state(self, msg: JointState) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0 or not msg.name or not msg.position:
            return
        positions = {n: float(p) for n, p in zip(msg.name, msg.position)}
        insert_at = bisect_left(self._joint_state_times_sec, stamp_sec)
        if (
            insert_at < len(self._joint_state_times_sec)
            and abs(self._joint_state_times_sec[insert_at] - stamp_sec) <= TIME_EPS_SEC
        ):
            self._joint_state_positions[insert_at] = positions
        else:
            self._joint_state_times_sec.insert(insert_at, stamp_sec)
            self._joint_state_positions.insert(insert_at, positions)
        # PROBLEM: this list grows unboundedly. For long live sessions
        # the bisect cost grows logarithmically and the memory
        # linearly — fine for the 5-30 min sessions we run, but not for
        # multi-hour deployments without a periodic prune.

    def _on_gazebo_pose(self, msg: PoseStamped) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0:
            return
        pose_matrix = pose_msg_to_matrix(msg.pose).astype(np.float64)
        insert_at = bisect_left(self._gazebo_pose_times_sec, stamp_sec)
        if (
            insert_at < len(self._gazebo_pose_times_sec)
            and abs(self._gazebo_pose_times_sec[insert_at] - stamp_sec) <= TIME_EPS_SEC
        ):
            self._gazebo_pose_matrices[insert_at] = pose_matrix
        else:
            self._gazebo_pose_times_sec.insert(insert_at, stamp_sec)
            self._gazebo_pose_matrices.insert(insert_at, pose_matrix)
        # PROBLEM: same unbounded-growth caveat as joint states. Also,
        # if gazebo republishes an old stamp (rare but possible after a
        # /reset), the bisect comparison flags it as duplicate and
        # silently overwrites — we'd lose the new sample. Acceptable.

    def _interpolate_c2w(self, stamp_sec: float) -> Optional[np.ndarray]:
        """Stamp-aligned camera_to_world matrix in Nerfstudio convention."""
        times = self._gazebo_pose_times_sec
        mats = self._gazebo_pose_matrices
        if not times:
            return None
        insert_at = bisect_left(times, stamp_sec)
        if insert_at < len(times) and abs(times[insert_at] - stamp_sec) <= TIME_EPS_SEC:
            base = mats[insert_at]
        elif insert_at > 0 and abs(times[insert_at - 1] - stamp_sec) <= TIME_EPS_SEC:
            base = mats[insert_at - 1]
        else:
            prev_idx = insert_at - 1 if insert_at > 0 else None
            next_idx = insert_at if insert_at < len(times) else None
            if prev_idx is None and next_idx is None:
                return None
            if prev_idx is None:
                base = mats[next_idx]
            elif next_idx is None:
                base = mats[prev_idx]
            else:
                t_prev = times[prev_idx]
                t_next = times[next_idx]
                alpha = (stamp_sec - t_prev) / (t_next - t_prev)
                q_prev = quaternion_from_matrix(mats[prev_idx])
                q_next = quaternion_from_matrix(mats[next_idx])
                q_interp = quaternion_slerp(q_prev, q_next, alpha)
                t_interp = mats[prev_idx][:3, 3] * (1.0 - alpha) + mats[next_idx][:3, 3] * alpha
                base = compose_transform_matrix(t_interp, q_interp)

        if self._mask_gen is None:
            # Mask gen owns the static link offsets; lazily build it on
            # the first interpolation request so single-frame consumers
            # don't pay the URDF parse upfront.
            self._mask_gen = RobotMaskGenerator(
                intrinsics=self.intrinsics,
                joint_state_times_sec=self._joint_state_times_sec,
                joint_state_positions=self._joint_state_positions,
            )
        optical_offset = self._mask_gen._static_link_offset(
            MASK_RENDER_CAMERA_FRAME, CAMERA_POSE_SAVE_FRAME
        ).astype(np.float64)
        return rotate_camera_frame_only(base @ optical_offset)
        # PROBLEM: when the gazebo stamp is older than the oldest cached
        # pose (e.g. the simulator restarted), we silently fall through to
        # using the oldest pose. Detecting this would need a bounded
        # history with explicit "stale" sentinel — punted for now.

    def _on_synced(self, image_msg: Image, depth_msg: Image) -> None:
        try:
            c2w = self._interpolate_c2w(float(image_msg.header.stamp.to_sec()))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"[live] pose interp failed: {exc}")
            return
        if c2w is None:
            return
        if self._mask_gen is None:
            self._mask_gen = RobotMaskGenerator(
                intrinsics=self.intrinsics,
                joint_state_times_sec=self._joint_state_times_sec,
                joint_state_positions=self._joint_state_positions,
            )
        try:
            mask_keep = self._mask_gen._render_robot_exclusion_mask(
                image_msg.header.stamp, MASK_RENDER_CAMERA_FRAME
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"[live] mask render failed: {exc}")
            return

        rgb_bgr = ros_image_to_bgr(image_msg)
        depth_mm = ros_depth_to_uint16_mm(depth_msg)
        with self._lock:
            self._frame_seq += 1
            seq = self._frame_seq
            frame = LiveFrame(
                seq=seq,
                stamp_sec=float(image_msg.header.stamp.to_sec()),
                rgb_bgr=rgb_bgr,
                depth_mm=depth_mm,
                mask_keep=mask_keep,
                c2w_4x4=c2w,
            )
            self._latest = frame

        # Disk write happens outside the latest-tuple lock so a slow disk
        # never holds up `peek_latest`.
        if self._record_active:
            self._write_frame_to_disk(frame, image_msg.header.stamp)
        # PROBLEM: pyrender / OpenGL contexts are single-thread-affine,
        # so the mask renderer only works on the thread that built it.
        # We build it lazily here in the rospy callback thread and we
        # never touch `_mask_gen` from anywhere else — keep it that way.

    def peek_latest(self) -> Optional[LiveFrame]:
        with self._lock:
            return self._latest
        # PROBLEM: this returns a reference to the mutable LiveFrame in
        # the slot. If the consumer holds onto the numpy arrays past the
        # next callback, fine — they're new arrays each time, not views.

    def wait_for_first_frame(self, timeout_s: float = 30.0) -> LiveFrame:
        """Block until the first synced tuple arrives; raise on timeout."""
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)
        while not rospy.is_shutdown():
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if rospy.Time.now() > deadline:
                raise TimeoutError(
                    f"no synced (rgb, depth, pose) tuple within {timeout_s}s — check that "
                    f"{IMAGE_TOPIC}, {DEPTH_TOPIC}, {GAZEBO_CAMERA_POSE_TOPIC} are publishing"
                )
            rospy.sleep(0.05)
        raise RuntimeError("rospy shut down while waiting for first frame")
        # PROBLEM: rospy.Time.now() reads sim time when /use_sim_time
        # is True, so timeout is sim-time too — pause Gazebo and this
        # never expires. Fine: pausing the sim before training starts
        # is itself a misconfiguration the operator will notice.

    def capture_anchor(self) -> LiveFrame:
        """Return the next synced tuple strictly newer than the call instant.

        Used to lock SAM3's input frame to "the frame after the user
        pressed enter", not whatever happened to be in the buffer.
        """
        with self._lock:
            baseline_seq = self._frame_seq
        while not rospy.is_shutdown():
            with self._lock:
                if self._latest is not None and self._latest.seq > baseline_seq:
                    return self._latest
            rospy.sleep(0.02)
        raise RuntimeError("rospy shut down while waiting for anchor frame")
        # PROBLEM: if no new tuple arrives (camera streaming halted),
        # this loops forever. Operators on this rig press Enter while
        # the camera is live, so we accept that and skip the timeout.

    # ---- Static-window disk recorder ----

    def start_recording(self, anchor_frame: LiveFrame) -> None:
        """Begin writing every new synced tuple under LIVE_ROOT/static_scene/.

        ``anchor_frame`` is the SAM3 input frame; it is written first
        (frame 0) before any subsequent live tuples.
        """
        record_dir = LIVE_ROOT / "static_scene"
        (record_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (record_dir / "depth").mkdir(parents=True, exist_ok=True)
        (record_dir / "masks").mkdir(parents=True, exist_ok=True)

        meta = {
            "fl_x": self.intrinsics.fx,
            "fl_y": self.intrinsics.fy,
            "cx": self.intrinsics.cx,
            "cy": self.intrinsics.cy,
            "w": self.intrinsics.width,
            "h": self.intrinsics.height,
            "frames": [],
        }
        with self._record_lock:
            self._record_dir = record_dir
            self._record_meta = meta
            self._record_frames_written = []
            self._record_stamps = []
            self._record_active = True
        self._write_frame_to_disk(anchor_frame, rospy.Time.from_sec(anchor_frame.stamp_sec))
        # PROBLEM: any frames that arrive between `anchor_frame` capture
        # and `start_recording` are dropped (subscriber callback runs
        # without the record_active flag). At 30 Hz / a few ms gap that's
        # negligible for static reconstruction.

    def _write_frame_to_disk(self, frame: LiveFrame, stamp: rospy.Time) -> None:
        with self._record_lock:
            if not self._record_active or self._record_dir is None:
                return
            record_dir = self._record_dir
            meta = self._record_meta
            frame_index = len(self._record_frames_written)

        stem = f"{IMAGE_NAME_PREFIX}_{frame_index:05d}"
        rgb_path = record_dir / "rgb" / f"{stem}.png"
        depth_path = record_dir / "depth" / f"{stem}.tiff"
        mask_path = record_dir / "masks" / f"{stem}.png"

        cv2.imwrite(str(rgb_path), frame.rgb_bgr)
        cv2.imwrite(str(depth_path), frame.depth_mm)
        cv2.imwrite(str(mask_path), frame.mask_keep)

        frame_entry = {
            "file_path": f"./rgb/{stem}.png",
            "depth_file_path": f"./depth/{stem}.tiff",
            "mask_path": f"./masks/{stem}.png",
            "transform_matrix": frame.c2w_4x4.tolist(),
        }
        with self._record_lock:
            self._record_frames_written.append(frame_entry)
            self._record_stamps.append(stamp)
            meta["frames"] = self._record_frames_written
            transforms_path = record_dir / "transforms.json"
            tmp_path = transforms_path.with_name(f".{transforms_path.name}.tmp")
            tmp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, transforms_path)
        # PROBLEM: cv2.imwrite return is not checked; on a full disk it
        # silently writes nothing and `transforms.json` ends up listing a
        # frame whose files don't exist, breaking the dataparser later.
        # Fine for our single-disk dev setup; would be a real bug on
        # network storage.

    def stop_recording(self) -> None:
        with self._record_lock:
            self._record_active = False

    def num_recorded_frames(self) -> int:
        with self._record_lock:
            return len(self._record_frames_written)

    def build_static_init_pointcloud(self) -> Path:
        """Write LIVE_ROOT/static_scene/depth_camera_init_points.ply.

        Mirrors `CaptureSession.write_init_cloud_from_saved_frames` —
        evenly distributes a budget of points across the recorded
        frames, back-projects through depth + camera intrinsics +
        c2w, persists to PLY, and registers the path in
        transforms.json.
        """
        with self._record_lock:
            record_dir = self._record_dir
            meta = self._record_meta
            frames_written = list(self._record_frames_written)
        if record_dir is None or meta is None or not frames_written:
            raise RuntimeError("start_recording() was not called or no frames were recorded")

        rng = np.random.default_rng(0)
        frame_infos = []
        valid_counts = []
        for frame in frames_written:
            depth_path = (record_dir / frame["depth_file_path"]).resolve()
            rgb_path = (record_dir / frame["file_path"]).resolve()
            mask_path = (record_dir / frame["mask_path"]).resolve()
            depth_mm = load_saved_depth_mm(depth_path)
            valid_mask = load_saved_mask(mask_path, depth_mm.shape)
            valid_count = int(np.count_nonzero(valid_mask & (depth_mm > 0.0)))
            if valid_count == 0:
                continue
            frame_infos.append(
                {"frame": frame, "depth_path": depth_path, "rgb_path": rgb_path, "mask_path": mask_path}
            )
            valid_counts.append(valid_count)
        if not frame_infos:
            raise RuntimeError("no valid (depth & mask) pixels found in recorded static frames")

        quotas = distribute_point_budget_evenly(valid_counts, MAX_INIT_CLOUD_POINTS)
        all_xyz = []
        all_rgb = []
        for frame_info, n_sample in zip(frame_infos, quotas):
            if n_sample <= 0:
                continue
            depth_mm = load_saved_depth_mm(frame_info["depth_path"])
            valid_mask = load_saved_mask(frame_info["mask_path"], depth_mm.shape)
            rgb_bgr = load_saved_rgb(frame_info["rgb_path"], depth_mm.shape)
            ys, xs = np.where(valid_mask & (depth_mm > 0.0))
            if ys.size == 0:
                continue
            if n_sample < ys.size:
                choice = rng.choice(ys.size, size=n_sample, replace=False)
                ys = ys[choice]
                xs = xs[choice]
            depth_m = depth_mm[ys, xs] / 1000.0
            x = (xs.astype(np.float32) - self.intrinsics.cx) * depth_m / self.intrinsics.fx
            y = -(ys.astype(np.float32) - self.intrinsics.cy) * depth_m / self.intrinsics.fy
            xyz_cam = np.stack([x, y, -depth_m], axis=1)
            hom = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)], axis=1)
            transform_matrix = np.asarray(frame_info["frame"]["transform_matrix"], dtype=np.float32)
            xyz_world = (transform_matrix @ hom.T).T[:, :3]
            rgb = rgb_bgr[ys, xs][:, ::-1].astype(np.uint8)
            all_xyz.append(xyz_world.astype(np.float32))
            all_rgb.append(rgb)
        if not all_xyz:
            raise RuntimeError("init pointcloud built from 0 points")
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        ply_path = record_dir / INIT_CLOUD_NAME
        write_ascii_ply(ply_path, xyz, rgb)

        meta["ply_file_path"] = INIT_CLOUD_NAME
        transforms_path = record_dir / "transforms.json"
        tmp_path = transforms_path.with_name(f".{transforms_path.name}.tmp")
        tmp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, transforms_path)
        return ply_path
        # PROBLEM: this loads each captured frame from disk twice (once
        # to count valid pixels, once to sample). Not a hot path — runs
        # exactly once per session, after the second Enter. If a frame
        # was written without a mask file (not happening today but a
        # latent risk), `load_saved_mask` will raise and we'll abort
        # before any PLY is written.


def cameras_from_live_frame(
    frame: LiveFrame,
    intrinsics: CameraIntrinsics,
    device: torch.device,
    cam_idx: int = 0,
) -> Cameras:
    """Build a single-frame Nerfstudio Cameras object from a LiveFrame.

    The c2w is float32 and trimmed to (3, 4) — Cameras drops the homo
    row internally. We attach `metadata["cam_idx"]` so any model code
    that branches on cam_idx (camera optimizer, debug logs) gets a
    stable, well-defined value.
    """
    c2w_3x4 = torch.from_numpy(frame.c2w_4x4[:3, :4].astype(np.float32)).unsqueeze(0)
    cam = Cameras(
        camera_to_worlds=c2w_3x4,
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.cx),
        cy=float(intrinsics.cy),
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    cam.metadata = {"cam_idx": int(cam_idx)}
    return cam
    # PROBLEM: nerfstudio's CameraOptimizer pose-adjustment tensor is
    # sized to `num_train_data` (i.e. the number of static keyframes).
    # If `cam_idx` exceeds that range, `apply_to_camera` indexes past
    # the end. We pass cam_idx=0 by default so live-mode FP / change
    # mask lookups always hit a valid optimizer slot.
