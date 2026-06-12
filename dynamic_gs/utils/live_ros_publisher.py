"""Standalone ROS publisher: pumps live (rgb, depth, mask, pose) tuples
into POSIX shared memory; handles disk recording + init-PLY build on
control commands.

Runs in the ROS env (python 3.8, NO torch, NO nerfstudio) via:

    conda run -n radiance_ros_4060 python \
        /home/.../scripts/dynamic_gs/utils/live_ros_publisher.py

IPC contract (both directions):
  * High-frequency frame data → POSIX shared memory
    Layout: header (4096B) + N slots of fixed size. Publisher writes
    slot[(seq+1) % N], then atomically bumps header.latest_seq. Reader
    samples header.latest_seq, reads that slot, re-checks slot.seq
    matches. 4 slots @ 30 Hz gives the reader >100 ms before the slot
    is reused.
  * Control + low-frequency events → stdin / stdout, line-delimited
    JSON. Reader sends commands (capture_anchor, start_recording, ...)
    and waits for the matching reply.

This file is invoked as a script (not as a package module) because
the ROS env doesn't have ``dynamic_gs`` on its python path. Anything
``dynamic_gs/utils/`` would normally provide (keyframe filter math)
is inlined below.

PROBLEM: this whole script runs in python 3.8. Use only 3.8-compatible
syntax; in particular no PEP 604 ``X | Y`` types in annotations, no
``match`` statements, no ``dict[str, int]`` parameterized builtins.
"""

from __future__ import annotations

import os
import sys

# Force-add the env's site-packages to sys.path early. Belt + suspenders:
# PYTHONNOUSERSITE=1 is set in the spawn wrapper to keep user-local Python 3.8
# site-packages from shadowing env packages, but if anything in this process
# (ROS setup, openrobots site init, etc.) re-orders or shortens sys.path,
# we want our env site to be unconditionally findable for the catkin_pkg /
# pyrender / urdfpy import chain.
_ENV_SITE = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                         f"python{sys.version_info.major}.{sys.version_info.minor}",
                         "site-packages")
_ENV_SITE = os.path.normpath(_ENV_SITE)
if os.path.isdir(_ENV_SITE) and _ENV_SITE not in sys.path:
    sys.path.insert(0, _ENV_SITE)

# IPC channel setup, done BEFORE any other import that might log.
# rospy / message_filters / urdfpy will happily ``rospy.loginfo`` or
# ``print`` straight to stdout (fd 1), which corrupts our line-delimited
# JSON over the parent's stdout pipe. To survive:
#
#   1. ``dup`` fd 1 to a new fd → that becomes our private IPC writer.
#   2. ``dup2`` stderr (fd 2) onto fd 1 so anything writing to "stdout"
#      (rospy.loginfo, raw print(), C-side fprintf to stdout) ends up
#      in the parent's stderr log file instead.
#   3. Wrap the saved fd in a line-buffered text file ``_IPC_OUT`` and
#      have ``_send_response`` write through it.
#
# Without this, the first urdfpy/pyrender/rospy info log corrupts the
# stream and the reader explodes with JSONDecodeError mid-handshake.
_IPC_FD = os.dup(1)
os.dup2(2, 1)  # fd 1 now points at the same file as fd 2 (parent's stderr log)
_IPC_OUT = os.fdopen(_IPC_FD, "w", buffering=1)
# sys.stdout still points at the new fd 1 (= stderr), so Python prints
# from third-party code land in the log file. Good.

import argparse
import atexit
import signal
import importlib.util
import json
import shutil
import struct
import subprocess
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import shared_memory
from typing import Dict, List, Optional

import cv2
import numpy as np

# Restore deprecated numpy aliases removed in numpy 1.24. urdfpy 0.0.22
# (in ~/.local/lib/python3.8/site-packages/urdfpy/urdf.py) still uses
# ``np.float`` via ``np.asanyarray(value).astype(np.float)``. Without
# this, the lazy RobotMaskGenerator constructor in _interpolate_c2w
# throws AttributeError on every synced frame and no shm slot is ever
# published. Cheaper than patching the global urdfpy install.
for _name, _alias in (
    ("float", float), ("int", int), ("bool", bool), ("object", object),
    ("str", str), ("long", int), ("complex", complex), ("unicode", str),
):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

# DEBUG: dump env + sys.path so we can see why pyrender comes from user-local
import sys as _sys, os as _os
print("[publisher-debug] sys.executable:", _sys.executable, file=_sys.stderr, flush=True)
print("[publisher-debug] PYTHONNOUSERSITE:", _os.environ.get("PYTHONNOUSERSITE", "(unset)"), file=_sys.stderr, flush=True)
print("[publisher-debug] PYTHONPATH:", _os.environ.get("PYTHONPATH", "(unset)"), file=_sys.stderr, flush=True)
print("[publisher-debug] sys.path[:10]:", file=_sys.stderr, flush=True)
for _i, _p in enumerate(_sys.path[:10]):
    print(f"[publisher-debug]   [{_i}] {_p}", file=_sys.stderr, flush=True)
# Eagerly test pyparsing import here in the publisher process to make
# 100% sure it's findable BEFORE rospy starts the catkin_pkg chain.
try:
    import pyparsing as _pp
    print(f"[publisher-debug] pyparsing OK: {_pp.__file__}", file=_sys.stderr, flush=True)
except Exception as _e:
    print(f"[publisher-debug] pyparsing IMPORT FAILED: {type(_e).__name__}: {_e}", file=_sys.stderr, flush=True)
    print("[publisher-debug] env site contents (pypars*):", file=_sys.stderr, flush=True)
    _site = "/home/mrc-cuhk/miniconda3/envs/dynamic_gs_ros/lib/python3.8/site-packages"
    if _os.path.isdir(_site):
        for _f in _os.listdir(_site):
            if "pypars" in _f.lower():
                print(f"  {_f}", file=_sys.stderr, flush=True)

import rospy
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from tf.transformations import quaternion_from_matrix, quaternion_slerp


# ---------------------------------------------------------------------------
# Recorder module loader (single source of truth for ROS topics, mask render
# logic, intrinsics). Loaded via importlib because the recorder lives outside
# any package.
# ---------------------------------------------------------------------------

_RECORDER_SCRIPT = Path(__file__).resolve().parents[2] / "save_data_img_depth_mask_pose.py"


def _load_recorder_module():
    spec = importlib.util.spec_from_file_location("_dgs_live_recorder", _RECORDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load recorder module from {_RECORDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dgs_live_recorder"] = module
    spec.loader.exec_module(module)
    return module


_REC = _load_recorder_module()

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
# Override the disk-recorder's 100 ms slop. RGB and depth from Gazebo
# are co-stamped within microseconds (both come from the same sim step),
# so a 100 ms slop forces the ApproximateTimeSynchronizer to spend
# ~slop seconds per match on its pivot-wait policy, capping throughput at
# 1/(slop + callback_ms) ≈ 10 Hz. 20 ms is still 600× the actual stamp
# divergence and lifts the throughput cap to ~25 Hz.
SYNC_SLOP_SEC = 0.02
IMAGE_NAME_PREFIX = _REC.IMAGE_NAME_PREFIX


# ---------------------------------------------------------------------------
# Shared-memory layout
# ---------------------------------------------------------------------------

# Default shm name. Reader passes the same one back via --shm-name.
DEFAULT_SHM_NAME = "/dgs_live_shm"
NUM_SLOTS = 4
HEADER_BYTES = 4096

# Header struct, little-endian, in order:
#   magic[4], version u32, height u32, width u32, num_slots u32,
#   fx f64, fy f64, cx f64, cy f64,
#   latest_seq u64,
#   slot_bytes u64,
#   rgb_off u32, depth_off u32, mask_off u32, pose_off u32, seq_off u32, stamp_off u32,
#   ready u32, shutdown u32
_HDR_FMT = "<4sIIII4xdddd Q Q IIIIII II"
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # well under HEADER_BYTES so we have padding


def _slot_layout(height: int, width: int):
    """Return (slot_bytes, offsets dict). Per-slot offsets are relative
    to slot start. All multi-byte fields are 8-byte aligned by ordering
    largest first."""
    rgb_bytes = height * width * 3
    depth_bytes = height * width * 4
    mask_bytes = height * width
    pose_bytes = 16 * 8
    seq_bytes = 8
    stamp_bytes = 8
    # Order: pose (8-aligned), seq (8), stamp (8), rgb (1-aligned),
    # depth (4-aligned), mask (1). Round slot_bytes up to 64.
    off = 0
    pose_off = off; off += pose_bytes
    seq_off = off; off += seq_bytes
    stamp_off = off; off += stamp_bytes
    rgb_off = off; off += rgb_bytes
    # 4-byte align before depth
    if off % 4 != 0:
        off += 4 - (off % 4)
    depth_off = off; off += depth_bytes
    mask_off = off; off += mask_bytes
    if off % 64 != 0:
        off += 64 - (off % 64)
    slot_bytes = off
    return slot_bytes, dict(
        rgb_off=rgb_off, depth_off=depth_off, mask_off=mask_off,
        pose_off=pose_off, seq_off=seq_off, stamp_off=stamp_off,
    )


def _total_shm_bytes(height: int, width: int) -> int:
    slot_bytes, _ = _slot_layout(height, width)
    return HEADER_BYTES + NUM_SLOTS * slot_bytes


def _write_header(
    shm_buf, height, width, fx, fy, cx, cy, slot_bytes, offsets, latest_seq, ready, shutdown
):
    packed = struct.pack(
        _HDR_FMT,
        b"DGS\0", 1, int(height), int(width), int(NUM_SLOTS),
        float(fx), float(fy), float(cx), float(cy),
        int(latest_seq),
        int(slot_bytes),
        int(offsets["rgb_off"]), int(offsets["depth_off"]), int(offsets["mask_off"]),
        int(offsets["pose_off"]), int(offsets["seq_off"]), int(offsets["stamp_off"]),
        int(ready), int(shutdown),
    )
    shm_buf[:len(packed)] = packed
    # PROBLEM: struct.pack returns a bytes object; we assign-slice the
    # mmap-backed buffer. That works because shared_memory.buf is a
    # writable memoryview-of-bytes.


# Header field offsets we mutate after the initial pack — for atomic-ish
# updates without re-packing the whole header. These are computed once
# below from _HDR_FMT and verified to be 8-byte aligned where needed.

def _compute_header_field_offsets():
    """Compute byte offsets of fields we update post-init."""
    # _HDR_FMT layout, walk through:
    # 4s I I I I 4x d d d d Q Q I I I I I I I I
    # Use struct.calcsize on prefixes.
    prefixes = [
        ("magic", "<4s"),
        ("version", "<4sI"),
        ("height", "<4sII"),
        ("width", "<4sIII"),
        ("num_slots", "<4sIIII"),
        # 4 bytes pad
        ("fx", "<4sIIII4x"),
        ("fy", "<4sIIII4xd"),
        ("cx", "<4sIIII4xdd"),
        ("cy", "<4sIIII4xddd"),
        ("latest_seq", "<4sIIII4xdddd"),
        ("slot_bytes", "<4sIIII4xddddQ"),
        ("rgb_off", "<4sIIII4xddddQQ"),
        ("depth_off", "<4sIIII4xddddQQI"),
        ("mask_off", "<4sIIII4xddddQQII"),
        ("pose_off", "<4sIIII4xddddQQIII"),
        ("seq_off", "<4sIIII4xddddQQIIII"),
        ("stamp_off", "<4sIIII4xddddQQIIIII"),
        ("ready", "<4sIIII4xddddQQIIIIII"),
        ("shutdown", "<4sIIII4xddddQQIIIIIII"),
    ]
    out = {}
    base = struct.calcsize(prefixes[0][1])
    out["magic"] = 0
    for i in range(1, len(prefixes)):
        # Offset of field i is the calcsize of the prefix that includes
        # everything strictly before field i — i.e. prefixes[i-1][1].
        name, prev_fmt = prefixes[i][0], prefixes[i - 1][1]
        out[name] = struct.calcsize(prev_fmt)
    return out


HDR_OFFSETS = _compute_header_field_offsets()


# ---------------------------------------------------------------------------
# Inlined keyframe filter (avoid `dynamic_gs.utils.keyframe_filter` import).
# ---------------------------------------------------------------------------


class _KeyframeFilter:
    """ORB-SLAM-style greedy keyframe filter. NumPy only; no torch."""

    def __init__(self, translation_thresh_m: float, rotation_thresh_deg: float):
        self.t_thresh = float(translation_thresh_m)
        self.r_thresh_rad = float(np.deg2rad(rotation_thresh_deg))
        self._kept_R = []  # type: List[np.ndarray]
        self._kept_t = []  # type: List[np.ndarray]

    @property
    def num_kept(self) -> int:
        return len(self._kept_R)

    def reset(self) -> None:
        self._kept_R.clear()
        self._kept_t.clear()

    def accept(self, c2w_3x4) -> bool:
        c2w = np.asarray(c2w_3x4, dtype=np.float64)
        if c2w.shape != (3, 4):
            raise ValueError("expected (3,4) c2w, got {}".format(c2w.shape))
        R_i = c2w[:, :3]
        t_i = c2w[:, 3]
        if not self._kept_R:
            self._kept_R.append(R_i)
            self._kept_t.append(t_i)
            return True
        K_R = np.stack(self._kept_R, axis=0)
        K_t = np.stack(self._kept_t, axis=0)
        dt = np.linalg.norm(t_i - K_t, axis=1)
        traces = np.einsum("ab,kab->k", R_i, K_R)
        cos_theta = np.clip(0.5 * (traces - 1.0), -1.0, 1.0)
        dr = np.arccos(cos_theta)
        near = (dt <= self.t_thresh) & (dr <= self.r_thresh_rad)
        if near.any():
            return False
        self._kept_R.append(R_i)
        self._kept_t.append(t_i)
        return True


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


@dataclass
class _StoredFrame:
    """Per-rospy-callback latched copy of a synchronised tuple.

    Mirrors the original LiveFrame from live_ros_subscriber.py but is
    stored on the publisher side; consumers see the bytes via shm.
    """
    seq: int
    stamp_sec: float
    rgb_bgr: np.ndarray   # (H, W, 3) uint8
    depth_m: np.ndarray   # (H, W) float32 metres
    mask_keep: np.ndarray # (H, W) uint8, 255 = keep
    c2w_4x4: np.ndarray   # (4, 4) float64


class LivePublisher:
    """Single-process publisher: ROS subscribers + shm slots + recorder."""

    @staticmethod
    def _wait_for_camera_info_primed(timeout_s: float = 20.0) -> Optional["CameraInfo"]:
        """Resolve camera_info, working around Gazebo's lazy-publish.

        Gazebo's camera plugin only publishes ``camera_info`` / ``image_raw``
        while something is subscribed to the image stream. A bare
        ``wait_for_message(camera_info)`` races that warm-up and intermittently
        times out even though the camera is perfectly healthy (the
        "stuck at spawning ROS publisher" / "camera_info wait timed out"
        symptom). We fix it deterministically by:

          1. Holding a real subscriber on the RGB image topic (``/compressed``)
             for the whole wait — this is what wakes Gazebo's lazy publisher
             and keeps the camera_info pipeline emitting.
          2. Polling ``wait_for_message(camera_info)`` in short slices until
             ``timeout_s`` elapses, instead of one all-or-nothing 5 s wait.

        Returns the CameraInfo on success, or ``None`` on timeout (caller
        falls back to cached intrinsics).
        """
        primer = None
        try:
            # The message type doesn't matter — we only subscribe to force
            # Gazebo to start publishing the camera pipeline. CompressedImage
            # is the lightest RGB transport on this camera.
            primer = rospy.Subscriber(
                IMAGE_TOPIC + "/compressed", CompressedImage,
                lambda _msg: None, queue_size=1,
            )
            deadline = time.time() + float(timeout_s)
            attempt = 0
            while time.time() < deadline:
                attempt += 1
                try:
                    remaining = max(0.5, deadline - time.time())
                    return rospy.wait_for_message(
                        CAMERA_INFO_TOPIC, CameraInfo,
                        timeout=min(2.0, remaining),
                    )
                except rospy.ROSException:
                    if attempt == 1:
                        print(
                            f"[publisher] priming camera (subscribed {IMAGE_TOPIC}/compressed); "
                            f"waiting for {CAMERA_INFO_TOPIC} ...",
                            flush=True,
                        )
                    continue
            return None
        finally:
            if primer is not None:
                try:
                    primer.unregister()
                except Exception:
                    pass

    def __init__(self, live_root: Path, shm_name: str, keyframe_translation_m: float,
                 keyframe_rotation_deg: float, record_replay_dir: Optional[str] = None):
        if not rospy.core.is_initialized():
            rospy.init_node("dynamic_gs_live_pub", disable_signals=True, anonymous=True)
        # Replay-recording tap (off unless --record-replay): captures the full
        # SHM frame stream + control events to disk so a fake publisher can
        # replay the whole capture without ROS/Gazebo. Set up after intrinsics.
        self._replay_dir = None
        self._replay_record_dir_arg = record_replay_dir

        # Try ROS first; if Gazebo's camera_info lazy-publish is stuck
        # (happens after Ctrl-R world-reset or a subscriber crash), fall
        # back to a cached intrinsics source. The fallback order:
        #   1. <live_root>/static_scene/transforms.json
        #   2. <live_root>/dynamic_scene/transforms.json
        #   3. ~/.cache/dgs_camera_intrinsics.json  (sticky, written on
        #      every successful resolve so a fresh dataset dir can still
        #      bootstrap when Gazebo refuses to publish camera_info).
        #   4. Any transforms.json under the standard datasets root.
        import json as _json
        _CACHE_PATH = Path.home() / ".cache" / "dgs_camera_intrinsics.json"
        _DATASETS_ROOT = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets")

        def _write_cache(intr: CameraIntrinsics) -> None:
            try:
                _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CACHE_PATH.write_text(_json.dumps({
                    "w": intr.width, "h": intr.height,
                    "fl_x": intr.fx, "fl_y": intr.fy,
                    "cx": intr.cx, "cy": intr.cy,
                }, indent=2))
            except Exception:
                pass

        info_msg = self._wait_for_camera_info_primed(timeout_s=20.0)
        if info_msg is not None:
            self.intrinsics = CameraIntrinsics(
                width=int(info_msg.width),
                height=int(info_msg.height),
                fx=float(info_msg.K[0]),
                fy=float(info_msg.K[4]),
                cx=float(info_msg.K[2]),
                cy=float(info_msg.K[5]),
            )
            _write_cache(self.intrinsics)  # refresh the sticky fallback
        else:
            # Build the ordered list of fallback candidates.
            candidates = [
                Path(live_root) / "static_scene" / "transforms.json",
                Path(live_root) / "dynamic_scene" / "transforms.json",
                _CACHE_PATH,
            ]
            # Then any transforms.json under the standard datasets root.
            try:
                candidates.extend(sorted(_DATASETS_ROOT.glob("*/static_scene/transforms.json")))
                candidates.extend(sorted(_DATASETS_ROOT.glob("*/dynamic_scene/transforms.json")))
            except Exception:
                pass
            tj = next((p for p in candidates if p.is_file()), None)
            if tj is None:
                raise RuntimeError(
                    f"camera_info never arrived on {CAMERA_INFO_TOPIC} after priming + "
                    f"retry, and no cached intrinsics fallback exists (checked "
                    f"static/dynamic transforms.json, {_CACHE_PATH}, and the datasets root). "
                    f"Is Gazebo running and the camera topic alive?"
                )
            meta = _json.loads(tj.read_text())
            self.intrinsics = CameraIntrinsics(
                width=int(meta["w"]),
                height=int(meta["h"]),
                fx=float(meta["fl_x"]),
                fy=float(meta["fl_y"]),
                cx=float(meta["cx"]),
                cy=float(meta["cy"]),
            )
            print(
                f"[publisher] camera_info wait timed out — falling back to "
                f"intrinsics from {tj} "
                f"(w={self.intrinsics.width} h={self.intrinsics.height} "
                f"fx={self.intrinsics.fx:.1f})",
                flush=True,
            )

        self.live_root = Path(live_root)
        self._joint_state_times_sec = []  # type: List[float]
        self._joint_state_positions = []  # type: List[dict]
        self._gazebo_pose_times_sec = []  # type: List[float]
        self._gazebo_pose_matrices = []   # type: List[np.ndarray]
        self._mask_gen = None  # type: Optional[RobotMaskGenerator]

        self._state_lock = threading.Lock()
        self._latest = None  # type: Optional[_StoredFrame]
        self._frame_seq = 0
        self._first_frame_event = threading.Event()

        # Worker thread + queue for off-receive-thread heavy work. The
        # rospy callback (_on_synced) only enqueues; the worker drains
        # the queue and does cv_bridge decode + pose interp + mask
        # render + shm write. This decouples the receive path from the
        # ~17 ms mask render, so the rospy reader can drain the socket
        # at full source rate (no per-subscriber TCP backpressure from
        # gazebo). Queue is bounded with drop-oldest semantics so a
        # falling-behind worker doesn't accumulate stale frames.
        import queue as _q
        self._frame_queue: "_q.Queue" = _q.Queue(maxsize=4)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_shutdown = threading.Event()

        # Disk recorder
        self._record_lock = threading.Lock()
        self._record_active = False
        self._record_dir = None        # type: Optional[Path]
        self._record_meta = None       # type: Optional[Dict]
        self._record_frames_written = []  # type: List[Dict]
        # M3: track in-flight _write_frame_to_disk callbacks so stop_recording
        # can quiesce them before returning. Without this, a callback that's
        # already past the _record_active check may still be doing cv2.imwrite
        # when stop_recording returns; the fusion watcher then misses that frame.
        self._inflight_writes = 0
        self._inflight_cv = threading.Condition(self._record_lock)
        self._record_keyframe_filter = _KeyframeFilter(
            translation_thresh_m=keyframe_translation_m,
            rotation_thresh_deg=keyframe_rotation_deg,
        )

        # Shared memory allocation. Sized once H/W are known.
        slot_bytes, offsets = _slot_layout(self.intrinsics.height, self.intrinsics.width)
        total = HEADER_BYTES + NUM_SLOTS * slot_bytes
        # Unlink any stale region with the same name (previous crash).
        try:
            existing = shared_memory.SharedMemory(name=shm_name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            # Some OSes raise non-FileNotFoundError for missing regions; ignore.
            pass
        self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total)
        self.shm_name = shm_name
        self.slot_bytes = slot_bytes
        self.offsets = offsets
        _write_header(
            self.shm.buf,
            self.intrinsics.height, self.intrinsics.width,
            self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy,
            slot_bytes, offsets,
            latest_seq=0, ready=1, shutdown=0,
        )

        # Pre-build slot numpy views (zero-copy into shm.buf).
        self._slot_views = []
        for i in range(NUM_SLOTS):
            base = HEADER_BYTES + i * slot_bytes
            self._slot_views.append({
                "pose": np.frombuffer(self.shm.buf, dtype=np.float64,
                                      count=16, offset=base + offsets["pose_off"]).reshape(4, 4),
                "seq":  np.frombuffer(self.shm.buf, dtype=np.uint64,
                                      count=1, offset=base + offsets["seq_off"]),
                "stamp": np.frombuffer(self.shm.buf, dtype=np.float64,
                                       count=1, offset=base + offsets["stamp_off"]),
                "rgb":  np.frombuffer(self.shm.buf, dtype=np.uint8,
                                      count=self.intrinsics.height * self.intrinsics.width * 3,
                                      offset=base + offsets["rgb_off"]).reshape(
                                          self.intrinsics.height, self.intrinsics.width, 3),
                "depth": np.frombuffer(self.shm.buf, dtype=np.float32,
                                       count=self.intrinsics.height * self.intrinsics.width,
                                       offset=base + offsets["depth_off"]).reshape(
                                           self.intrinsics.height, self.intrinsics.width),
                "mask": np.frombuffer(self.shm.buf, dtype=np.uint8,
                                      count=self.intrinsics.height * self.intrinsics.width,
                                      offset=base + offsets["mask_off"]).reshape(
                                          self.intrinsics.height, self.intrinsics.width),
            })
        # PROBLEM: np.frombuffer returns a writable view only when shm.buf
        # is writable (it is — SharedMemory is). The reshape preserves
        # this. The pose array IS writable; we assign in-place below.

        # Subscribers
        self._joint_sub = rospy.Subscriber(
            GAZEBO_JOINT_STATES_TOPIC, JointState, self._on_joint_state, queue_size=200,
        )
        self._pose_sub = rospy.Subscriber(
            GAZEBO_CAMERA_POSE_TOPIC, PoseStamped, self._on_gazebo_pose, queue_size=200,
        )
        # RGB uses the JPEG /compressed transport (~50 KB vs 1.9 MB raw),
        # but depth is subscribed RAW. Gazebo's openni_kinect publishes
        # depth as 32FC1 (float32 metres). When that's republished through
        # compressed_depth_image_transport, the encoder uses INVERSE-DEPTH
        # log compression (format=0 INV_DEPTH, depthQuantA/B in the
        # ConfigHeader) rather than raw uint16 mm — so the naive PNG-as-
        # uint16-mm decoder we previously used returned values ~32× too
        # large (e.g. 17.5 m where truth was 0.55 m). That silently
        # destroyed every RANSAC pose fit because back-projected 3D points
        # were in fantasy metres.
        # Subscribing to the raw 32FC1 topic costs ~80 MB/s on loopback at
        # 30 Hz (manageable) and gives us the exact metres the sim
        # produced, with zero decode ambiguity.
        self._depth_republisher_proc: Optional[subprocess.Popen] = None
        self._rgb_sub = Subscriber(
            IMAGE_TOPIC + "/compressed", CompressedImage,
            queue_size=30, buff_size=2 * 1024 * 1024, tcp_nodelay=True,
        )
        self._depth_sub = Subscriber(
            DEPTH_TOPIC, Image,
            queue_size=30, buff_size=16 * 1024 * 1024, tcp_nodelay=True,
        )
        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        self._sync.registerCallback(self._on_synced)

        # Spawn the worker thread that drains _frame_queue. Done last
        # so all member state (shm, _state_lock, _mask_gen=None, etc.)
        # is fully initialised before the worker can pull a frame.
        if self._replay_record_dir_arg:
            self._setup_replay_recording(self._replay_record_dir_arg)

        self.start_worker()

    def _spawn_depth_republisher(self) -> None:
        """Auto-launch a C++ image_transport republish for /compressedDepth.

        Gazebo's openni_kinect plugin publishes raw depth on DEPTH_TOPIC but
        does NOT advertise the /compressedDepth transport. The C++ republish
        node fixes that with no Python GIL involvement.

        We ALWAYS spawn — we don't check ``rospy.get_published_topics()`` for
        an existing republisher because the master often holds stale topic
        registrations after an unclean kill (publisher dies without
        unregistering, registration sits at the master indefinitely). A
        stale entry would short-circuit the spawn here and the depth
        subscriber would then wait forever for messages that never arrive.
        If a real republisher happens to be running too, both will publish
        to the same topic — the ApproximateTimeSynchronizer downstream
        handles the duplicates by stamp.
        """
        compressed_topic = DEPTH_TOPIC + "/compressedDepth"
        cmd = [
            "rosrun", "image_transport", "republish",
            "raw", f"in:={DEPTH_TOPIC}",
            "compressedDepth", f"out:={DEPTH_TOPIC}",
        ]
        try:
            # New process group so a SIGINT to the publisher doesn't propagate
            # to the republisher mid-shutdown (we kill it explicitly below).
            self._depth_republisher_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            rospy.logwarn(f"[live] failed to spawn depth republish: {exc}")
            return
        # Wait briefly for the republisher to start advertising. If it never
        # does, the depth Subscriber below will sit idle and ApproximateTime
        # will never produce a synced pair — better to log it loudly here.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                if compressed_topic in {name for name, _ in rospy.get_published_topics()}:
                    rospy.loginfo(
                        f"[live] depth republish ready: {compressed_topic} (pid={self._depth_republisher_proc.pid})"
                    )
                    return
            except Exception:
                pass
            time.sleep(0.1)
        rospy.logwarn(
            f"[live] depth republish spawned (pid={self._depth_republisher_proc.pid}) "
            f"but {compressed_topic} did not appear within 5 s"
        )

    # ---- ROS callbacks ----

    # Throttle threshold (seconds). Drop callbacks whose stamp is within
    # 20 ms of the last accepted one — we only need pose/joint samples
    # that bracket each ~33 ms image stamp, so 50 Hz storage is more
    # than enough for _interpolate_c2w.
    _POSE_JOINT_MIN_DT_SEC = 0.02

    def _on_joint_state(self, msg: JointState) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0 or not msg.name or not msg.position:
            return
        if (self._joint_state_times_sec and
                stamp_sec - self._joint_state_times_sec[-1] < self._POSE_JOINT_MIN_DT_SEC):
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

    def _on_gazebo_pose(self, msg: PoseStamped) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if stamp_sec <= 0.0:
            return
        if (self._gazebo_pose_times_sec and
                stamp_sec - self._gazebo_pose_times_sec[-1] < self._POSE_JOINT_MIN_DT_SEC):
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

    def _interpolate_c2w(self, stamp_sec: float):
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
            self._mask_gen = RobotMaskGenerator(
                intrinsics=self.intrinsics,
                joint_state_times_sec=self._joint_state_times_sec,
                joint_state_positions=self._joint_state_positions,
            )
        optical_offset = self._mask_gen._static_link_offset(
            MASK_RENDER_CAMERA_FRAME, CAMERA_POSE_SAVE_FRAME
        ).astype(np.float64)
        return rotate_camera_frame_only(base @ optical_offset)

    @staticmethod
    def _decode_compressed_rgb(msg: CompressedImage) -> np.ndarray:
        """JPEG-decode a sensor_msgs/CompressedImage to BGR uint8 HxWx3."""
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _decode_raw_depth(msg: Image) -> np.ndarray:
        """Decode a sensor_msgs/Image. Returns depth in metres (float32).

        Gazebo's openni_kinect publishes 32FC1 (float32 metres). Some sensors
        publish 16UC1 (uint16 millimetres) — handle both. We deliberately do
        NOT subscribe to the compressedDepth transport here: for 32FC1 sources
        ``image_transport republish`` switches to inverse-depth log encoding
        (format=0 INV_DEPTH, depthQuantA/B in the ConfigHeader) and a naive
        PNG-as-mm decoder returns values ~32× too large, destroying every
        downstream RANSAC pose fit.
        """
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        if msg.encoding == "16UC1":
            arr_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            return arr_mm.astype(np.float32) * 1e-3
        raise ValueError(f"Unsupported depth encoding: {msg.encoding!r}")

    def _on_synced(self, image_msg: CompressedImage, depth_msg: Image) -> None:
        """rospy callback fires on a sync match. Keep this MINIMAL: just
        push the raw messages to the worker queue and return. The mask
        render and other heavy work runs on the worker thread so the
        rospy receive thread can drain the socket at full source rate.
        """
        import time as _t

        # Enqueue. If the worker is falling behind, drop the OLDEST and
        # keep the newest — for tracking we always want the freshest
        # frame, not a stale backlog.
        try:
            self._frame_queue.put_nowait((image_msg, depth_msg, _t.time()))
        except Exception:
            # Queue full → drop oldest then push.
            try:
                _ = self._frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self._frame_queue.put_nowait((image_msg, depth_msg, _t.time()))
            except Exception:
                pass

    def start_worker(self) -> None:
        """Spawn the worker thread that drains _frame_queue. Call once
        after subscriptions are set up."""
        if self._worker_thread is not None:
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="dgs-publisher-worker", daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Drain frame_queue and do cv_bridge + pose interp + mask render
        + shm write. Runs in its own thread so the rospy receive thread
        is never blocked on heavy work."""
        import queue as _q
        while not self._worker_shutdown.is_set():
            try:
                image_msg, depth_msg, _enq_t = self._frame_queue.get(timeout=0.1)
            except _q.Empty:
                continue
            try:
                self._process_synced_pair(image_msg, depth_msg)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "[live] worker exc: {}".format(exc))

    # ---- Replay recording (--record-replay) ----

    def _setup_replay_recording(self, record_replay_dir: str) -> None:
        """Record the full SHM frame stream + control events to disk so a fake
        publisher (live_replay_publisher) can replay the whole session without
        ROS/Gazebo. Fixed-size raw records (fast, no compression, no drops)."""
        import queue as _q
        # A bootstrap spawns TWO publishers (capture, then teleop). Give each
        # spawn its own timestamped subdir so they don't clobber stream.bin;
        # the replay driver reads subdirs in chronological (lexical) order →
        # [capture, teleop]. The phase tag (env, best-effort) aids readability.
        _base = Path(record_replay_dir)
        _base.mkdir(parents=True, exist_ok=True)
        _phase = os.environ.get("DGS_REPLAY_PHASE", "seg")
        self._replay_dir = _base / f"{int(time.time()*1000):013d}_{_phase}_{os.getpid()}"
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        H, W = self.intrinsics.height, self.intrinsics.width
        # record = c2w(16 f64) + seq(i64) + stamp(f64) + rgb(HxWx3 u8) + depth(HxW f32) + mask(HxW u8)
        self._replay_record_size = 16 * 8 + 8 + 8 + H * W * 3 + H * W * 4 + H * W
        self._replay_stream = open(self._replay_dir / "stream.bin", "wb", buffering=4 * 1024 * 1024)
        self._replay_queue = _q.Queue(maxsize=1200)
        self._replay_count = 0
        self._replay_dropped = 0
        self._replay_control_events: List[Dict] = []
        self._replay_stop = threading.Event()
        self._replay_writer = threading.Thread(
            target=self._replay_writer_loop, name="replay_writer", daemon=True)
        self._replay_writer.start()
        self._write_replay_meta(finalized=False)
        print(f"[publisher] replay-recording ON → {self._replay_dir} "
              f"({self._replay_record_size} B/frame, {H}x{W})", file=sys.stderr, flush=True)

    def _write_replay_meta(self, finalized: bool) -> None:
        H, W = self.intrinsics.height, self.intrinsics.width
        meta = {
            "shm_name": self.shm_name, "height": H, "width": W,
            "fx": self.intrinsics.fx, "fy": self.intrinsics.fy,
            "cx": self.intrinsics.cx, "cy": self.intrinsics.cy,
            "record_size": self._replay_record_size,
            "layout": ["c2w_f64x16", "seq_i64", "stamp_f64",
                       f"rgb_u8_{H}x{W}x3", f"depth_f32_{H}x{W}", f"mask_u8_{H}x{W}"],
            "num_frames": int(self._replay_count),
            "dropped": int(self._replay_dropped),
            "control_events": list(self._replay_control_events),
            "finalized": bool(finalized),
        }
        (self._replay_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _replay_writer_loop(self) -> None:
        import queue as _q
        while not (self._replay_stop.is_set() and self._replay_queue.empty()):
            try:
                seq, stamp, c2w, rgb, depth, mask = self._replay_queue.get(timeout=0.2)
            except _q.Empty:
                continue
            try:
                self._replay_stream.write(
                    np.ascontiguousarray(c2w, dtype="<f8").tobytes()
                    + struct.pack("<qd", int(seq), float(stamp))
                    + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
                    + np.ascontiguousarray(depth, dtype="<f4").tobytes()
                    + np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
                )
                self._replay_count += 1
            except Exception as exc:
                print(f"[publisher] replay write failed: {exc}", file=sys.stderr, flush=True)

    def record_control_event(self, op: str, **kw) -> None:
        """Log a reader control op (capture_anchor / start_recording / ...)
        mapped to the current frame seq, so replay can re-fire it (auto-Enter)."""
        if self._replay_dir is None:
            return
        with self._state_lock:
            seq = self._frame_seq
        self._replay_control_events.append({"op": op, "seq": int(seq), "t": time.time(), **kw})

    def _process_synced_pair(self, image_msg: CompressedImage, depth_msg: Image) -> None:
        """The original _on_synced body, now running on the worker thread."""
        try:
            c2w = self._interpolate_c2w(float(image_msg.header.stamp.to_sec()))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[live] pose interp failed: {}".format(exc))
            return
        if c2w is None:
            return

        rgb_bgr = self._decode_compressed_rgb(image_msg)
        depth_m = self._decode_raw_depth(depth_msg)

        should_write = False
        if self._record_active:
            with self._record_lock:
                should_write = self._record_keyframe_filter.accept(c2w[:3, :4])

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
            rospy.logwarn_throttle(2.0, "[live] mask render failed: {}".format(exc))
            return

        with self._state_lock:
            self._frame_seq += 1
            seq = self._frame_seq
            stamp = float(image_msg.header.stamp.to_sec())
            self._latest = _StoredFrame(
                seq=seq, stamp_sec=stamp,
                rgb_bgr=rgb_bgr, depth_m=depth_m, mask_keep=mask_keep, c2w_4x4=c2w,
            )

        slot_idx = seq % NUM_SLOTS
        slot_views_snapshot = self._slot_views
        if not slot_views_snapshot:
            return
        slot = slot_views_snapshot[slot_idx]
        slot["seq"][0] = seq
        slot["pose"][:] = c2w
        slot["stamp"][0] = stamp
        slot["rgb"][:] = rgb_bgr
        slot["depth"][:] = depth_m
        slot["mask"][:] = mask_keep
        latest_off = HDR_OFFSETS["latest_seq"]
        struct.pack_into("<Q", self.shm.buf, latest_off, seq)

        # Replay tap: enqueue the fresh decoded arrays (NOT the SHM views) for
        # the background writer. They're per-frame fresh, so refs are safe; the
        # writer serialises them off the worker thread → no per-frame copy here.
        if self._replay_dir is not None:
            try:
                self._replay_queue.put_nowait((seq, stamp, c2w, rgb_bgr, depth_m, mask_keep))
            except Exception:
                self._replay_dropped += 1

        if not self._first_frame_event.is_set():
            self._first_frame_event.set()

        if should_write:
            self._write_frame_to_disk(self._latest, image_msg.header.stamp)

    # ---- Reader-facing operations ----

    def wait_first_frame(self, timeout_s: float) -> bool:
        return self._first_frame_event.wait(timeout=timeout_s)

    def capture_anchor(self, timeout_s: float = 30.0) -> Optional[_StoredFrame]:
        with self._state_lock:
            baseline = self._frame_seq
        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            with self._state_lock:
                if self._latest is not None and self._latest.seq > baseline:
                    return self._latest
            time.sleep(0.01)
        return None

    def start_recording(self, anchor: _StoredFrame) -> int:
        record_dir = self.live_root / "static_scene"
        (record_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (record_dir / "depth").mkdir(parents=True, exist_ok=True)
        (record_dir / "masks").mkdir(parents=True, exist_ok=True)
        meta = {
            "fl_x": self.intrinsics.fx, "fl_y": self.intrinsics.fy,
            "cx": self.intrinsics.cx, "cy": self.intrinsics.cy,
            "w": self.intrinsics.width, "h": self.intrinsics.height,
            "frames": [],
        }
        with self._record_lock:
            self._record_dir = record_dir
            self._record_meta = meta
            self._record_frames_written = []
            self._record_keyframe_filter.reset()
            self._record_keyframe_filter.accept(anchor.c2w_4x4[:3, :4])
            self._record_active = True
        self._write_frame_to_disk(anchor, rospy.Time.from_sec(anchor.stamp_sec))
        return 1

    def stop_recording(self) -> int:
        """Flip recording off and wait for any in-flight writes to finish.

        M3: returning before quiesce caused the fusion watcher's final sweep
        to race against an in-progress _write_frame_to_disk, silently losing
        the last keyframe. Now we block here (max 2 s) until callbacks past
        the active-flag check complete their cv2.imwrite + transforms.json swap.
        """
        with self._record_lock:
            self._record_active = False
            written = len(self._record_frames_written)
            deadline_s = 2.0
            t0 = time.time()
            while self._inflight_writes > 0:
                if time.time() - t0 > deadline_s:
                    rospy.logwarn(
                        "[publisher] stop_recording: %d in-flight writes did not finish in %.1f s",
                        self._inflight_writes, deadline_s,
                    )
                    break
                self._inflight_cv.wait(timeout=0.05)
            # The count after wait may include frames that landed JUST as we
            # flipped active=False — those are still on disk + in transforms.json,
            # which is correct; the fusion watcher will pick them up.
            return len(self._record_frames_written)

    def _write_frame_to_disk(self, frame: _StoredFrame, stamp) -> None:
        with self._record_lock:
            if not self._record_active or self._record_dir is None:
                return
            record_dir = self._record_dir
            meta = self._record_meta
            frame_index = len(self._record_frames_written)
            self._inflight_writes += 1
        try:
            stem = "{}_{:05d}".format(IMAGE_NAME_PREFIX, frame_index)
            rgb_path = record_dir / "rgb" / "{}.png".format(stem)
            # uint16 mm depth on disk (matches recorded-mode convention).
            depth_path = record_dir / "depth" / "{}.tiff".format(stem)
            mask_path = record_dir / "masks" / "{}.png".format(stem)

            cv2.imwrite(str(rgb_path), frame.rgb_bgr)
            depth_mm_u16 = np.clip(frame.depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm_u16)
            cv2.imwrite(str(mask_path), frame.mask_keep)

            frame_entry = {
                "file_path": "./rgb/{}.png".format(stem),
                "depth_file_path": "./depth/{}.tiff".format(stem),
                "mask_path": "./masks/{}.png".format(stem),
                "transform_matrix": frame.c2w_4x4.tolist(),
            }
            with self._record_lock:
                self._record_frames_written.append(frame_entry)
                meta["frames"] = self._record_frames_written
                transforms_path = record_dir / "transforms.json"
                tmp_path = transforms_path.with_name(".{}.tmp".format(transforms_path.name))
                tmp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp_path, transforms_path)
        finally:
            # M3: decrement and notify so stop_recording's quiesce wakes.
            with self._record_lock:
                self._inflight_writes -= 1
                if self._inflight_writes <= 0:
                    self._inflight_cv.notify_all()

    def num_recorded_frames(self) -> int:
        with self._record_lock:
            return len(self._record_frames_written)

    def build_static_init_pointcloud(self) -> Path:
        with self._record_lock:
            record_dir = self._record_dir
            meta = self._record_meta
            frames_written = list(self._record_frames_written)
        if record_dir is None or meta is None or not frames_written:
            raise RuntimeError("start_recording was not called or no frames were recorded")

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
            frame_infos.append({
                "frame": frame, "depth_path": depth_path,
                "rgb_path": rgb_path, "mask_path": mask_path,
            })
            valid_counts.append(valid_count)
        if not frame_infos:
            raise RuntimeError("no valid (depth & mask) pixels in recorded static frames")

        quotas = distribute_point_budget_evenly(valid_counts, MAX_INIT_CLOUD_POINTS)
        all_xyz = []
        all_rgb = []
        for info, n_sample in zip(frame_infos, quotas):
            if n_sample <= 0:
                continue
            depth_mm = load_saved_depth_mm(info["depth_path"])
            valid_mask = load_saved_mask(info["mask_path"], depth_mm.shape)
            rgb_bgr = load_saved_rgb(info["rgb_path"], depth_mm.shape)
            ys, xs = np.where(valid_mask & (depth_mm > 0.0))
            if ys.size == 0:
                continue
            if n_sample < ys.size:
                choice = rng.choice(ys.size, size=n_sample, replace=False)
                ys = ys[choice]
                xs = xs[choice]
            depth_m_pix = depth_mm[ys, xs] / 1000.0
            x = (xs.astype(np.float32) - self.intrinsics.cx) * depth_m_pix / self.intrinsics.fx
            y = -(ys.astype(np.float32) - self.intrinsics.cy) * depth_m_pix / self.intrinsics.fy
            xyz_cam = np.stack([x, y, -depth_m_pix], axis=1)
            hom = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)], axis=1)
            transform_matrix = np.asarray(info["frame"]["transform_matrix"], dtype=np.float32)
            xyz_world = (transform_matrix @ hom.T).T[:, :3]
            rgb_arr = rgb_bgr[ys, xs][:, ::-1].astype(np.uint8)
            all_xyz.append(xyz_world.astype(np.float32))
            all_rgb.append(rgb_arr)
        if not all_xyz:
            raise RuntimeError("init pointcloud built from 0 points")
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        ply_path = record_dir / INIT_CLOUD_NAME
        write_ascii_ply(ply_path, xyz, rgb)

        meta["ply_file_path"] = INIT_CLOUD_NAME
        transforms_path = record_dir / "transforms.json"
        tmp_path = transforms_path.with_name(".{}.tmp".format(transforms_path.name))
        tmp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, transforms_path)
        return ply_path

    # ---- Anchor save for SAM3 / SAM3D side effects ----

    def save_anchor_for_sam3(self, anchor: _StoredFrame, debug_dir: Path) -> str:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        rgb_rgb = cv2.cvtColor(anchor.rgb_bgr, cv2.COLOR_BGR2RGB).copy()
        keep = anchor.mask_keep > 0
        if keep.shape != rgb_rgb.shape[:2]:
            keep = cv2.resize(
                keep.astype(np.uint8) * 255,
                (rgb_rgb.shape[1], rgb_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 127
        rgb_rgb[~keep] = 0
        # Save as PNG via cv2 (avoid Pillow dependency).
        out_path = debug_dir / "static0_rgb.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR))
        return str(out_path)

    def save_anchor_intrinsics_and_depth(self, anchor: _StoredFrame, artifact_dir: Path):
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        depth_path = artifact_dir / "static0_full_depth_meters.tiff"
        cv2.imwrite(str(depth_path), anchor.depth_m.astype(np.float32))
        intrinsics_path = artifact_dir / "static0_full_intrinsics.json"
        intrinsics_payload = {
            "fx": float(self.intrinsics.fx), "fy": float(self.intrinsics.fy),
            "cx": float(self.intrinsics.cx), "cy": float(self.intrinsics.cy),
            "width": int(self.intrinsics.width), "height": int(self.intrinsics.height),
        }
        intrinsics_path.write_text(json.dumps(intrinsics_payload, indent=2) + "\n")
        return str(depth_path), str(intrinsics_path)

    # ---- Gazebo physics service helpers ----

    def pause_gazebo(self) -> bool:
        try:
            from std_srvs.srv import Empty
            rospy.wait_for_service("/gazebo/pause_physics", timeout=2.0)
            rospy.ServiceProxy("/gazebo/pause_physics", Empty)()
            return True
        except Exception:
            return False

    def unpause_gazebo(self) -> bool:
        try:
            from std_srvs.srv import Empty
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)
            rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Best-effort shutdown.

        Crucially, we do NOT clear ``self._slot_views`` here. ROS
        callbacks run on rospy's background threadpool and can fire
        between when ``_main`` returns and when the process actually
        exits. Clearing the views in this window would IndexError on
        ``self._slot_views[slot_idx]`` inside ``_on_synced``, get logged
        by rospy as a "bad callback", and look exactly like a publisher
        crash to the reader. We rely on process exit to release the
        memoryview refs.

        We also do NOT call ``self.shm.unlink()`` here. The reader may
        still be reading slots when we're tearing down; unlinking
        deletes the POSIX shm name, but the OS keeps the underlying
        memory alive while any fd references exist. Process exit cleans
        this up. The header.shutdown flag below is the polite signal.
        """
        # Flush + finalize the replay recording (drain the writer, write meta).
        if getattr(self, "_replay_dir", None) is not None:
            try:
                self._replay_stop.set()
                self._replay_writer.join(timeout=10.0)
                self._replay_stream.flush()
                self._replay_stream.close()
                self._write_replay_meta(finalized=True)
                print(f"[publisher] replay-recording finalized: {self._replay_count} frames "
                      f"({self._replay_dropped} dropped) → {self._replay_dir}",
                      file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[publisher] replay finalize failed: {exc}", file=sys.stderr, flush=True)
        # Signal the worker thread to stop draining the queue.
        try:
            self._worker_shutdown.set()
        except Exception:
            pass
        # Kill the depth republisher we spawned (if any). Skipped if a
        # pre-existing republisher was detected at startup (we didn't spawn
        # one in that case, so there's nothing to clean up).
        if self._depth_republisher_proc is not None:
            try:
                self._depth_republisher_proc.terminate()
                self._depth_republisher_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self._depth_republisher_proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
        # Mark header.shutdown so any future reader sees we're done.
        try:
            struct.pack_into("<I", self.shm.buf, HDR_OFFSETS["shutdown"], 1)
        except Exception:
            pass
        # Stop accepting new ROS messages so callbacks don't pile up
        # during the python interpreter teardown. unregister() is
        # idempotent and safe to call from any thread.
        try:
            self._joint_sub.unregister()
        except Exception:
            pass
        try:
            self._pose_sub.unregister()
        except Exception:
            pass
        try:
            self._rgb_sub.unregister()
        except Exception:
            pass
        try:
            self._depth_sub.unregister()
        except Exception:
            pass
        # Tell rosmaster to drop our node registration. Without this the
        # node entry stays in `rosnode list` after we exit, and subsequent
        # publisher subprocesses get unpredictable topic-wiring behaviour
        # (e.g. "timeout waiting for /camera_info" on a topic that IS being
        # published). rospy.signal_shutdown is idempotent + thread-safe.
        try:
            import rospy as _rospy
            if not _rospy.is_shutdown():
                _rospy.signal_shutdown("publisher.shutdown() called")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# JSON command loop (stdin → publisher → stdout). Each command is one
# line; each response is one line. Reader is single-threaded against
# this stream, so we don't need correlation IDs.
# ---------------------------------------------------------------------------


def _send_response(payload):
    """Write one JSON line to the saved IPC channel (was fd 1)."""
    _IPC_OUT.write(json.dumps(payload) + "\n")
    _IPC_OUT.flush()


def _wipe_live_root(live_root: Path) -> None:
    if live_root.exists():
        shutil.rmtree(live_root)
    (live_root / "static_scene").mkdir(parents=True, exist_ok=True)
    (live_root / "dynamic_scene").mkdir(parents=True, exist_ok=True)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--shm-name", default=DEFAULT_SHM_NAME)
    parser.add_argument("--keyframe-translation-m", type=float, default=0.02)
    parser.add_argument("--keyframe-rotation-deg", type=float, default=20.0)
    parser.add_argument("--wipe-live-root", action="store_true")
    parser.add_argument("--record-replay", type=str, default=None,
                        help="Directory to record the full SHM frame stream + control "
                             "events for deterministic replay (live_replay_publisher).")
    args = parser.parse_args()

    if args.wipe_live_root:
        _wipe_live_root(args.live_root)

    try:
        pub = LivePublisher(
            live_root=args.live_root,
            shm_name=args.shm_name,
            keyframe_translation_m=args.keyframe_translation_m,
            keyframe_rotation_deg=args.keyframe_rotation_deg,
            record_replay_dir=args.record_replay,
        )
    except Exception as exc:
        _send_response({"event": "init_error", "error": "{}: {}".format(type(exc).__name__, exc)})
        return 1

    # First "ready" message — reader uses this to know shm is open.
    _send_response({
        "event": "ready",
        "shm_name": args.shm_name,
        "width": pub.intrinsics.width,
        "height": pub.intrinsics.height,
        "fx": pub.intrinsics.fx,
        "fy": pub.intrinsics.fy,
        "cx": pub.intrinsics.cx,
        "cy": pub.intrinsics.cy,
        "num_slots": NUM_SLOTS,
        "slot_bytes": pub.slot_bytes,
        "header_bytes": HEADER_BYTES,
    })

    def _atexit_shutdown():
        try:
            pub.shutdown()
        except Exception:
            pass
    atexit.register(_atexit_shutdown)

    # SIGINT/SIGTERM from the parent must drop the rospy node registration
    # so the next publisher run doesn't fight with a zombie /camera_info
    # subscriber entry in rosmaster. rospy was initialised with
    # disable_signals=True, so we install our own minimal handler here.
    def _on_signal(signum, _frame):
        try:
            pub.shutdown()
        finally:
            os._exit(128 + signum)
    try:
        signal.signal(signal.SIGINT,  _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP,  _on_signal)
    except (ValueError, AttributeError):
        # signal.signal only works from the main thread; fall back to atexit.
        pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as exc:
            _send_response({"ok": False, "error": "bad_json: {}".format(exc)})
            continue
        op = cmd.get("op")
        try:
            if op == "wait_first_frame":
                ok = pub.wait_first_frame(timeout_s=float(cmd.get("timeout_s", 30.0)))
                _send_response({"ok": bool(ok)})
            elif op == "capture_anchor":
                anchor = pub.capture_anchor(timeout_s=float(cmd.get("timeout_s", 30.0)))
                if anchor is None:
                    _send_response({"ok": False, "error": "no_anchor"})
                else:
                    pub.record_control_event("capture_anchor", anchor_seq=int(anchor.seq))
                    _send_response({"ok": True, "seq": int(anchor.seq), "stamp": float(anchor.stamp_sec)})
            elif op == "start_recording":
                anchor_seq = int(cmd.get("anchor_seq", -1))
                with pub._state_lock:
                    cur = pub._latest
                if cur is None or cur.seq != anchor_seq:
                    _send_response({"ok": False, "error": "anchor_not_latest"})
                else:
                    pub.start_recording(cur)
                    pub.record_control_event("start_recording", anchor_seq=anchor_seq)
                    _send_response({"ok": True, "recorded": pub.num_recorded_frames()})
            elif op == "stop_recording":
                pub.record_control_event("stop_recording")
                n = pub.stop_recording()
                _send_response({"ok": True, "recorded": int(n)})
            elif op == "num_recorded":
                _send_response({"ok": True, "recorded": pub.num_recorded_frames()})
            elif op == "save_anchor_for_sam3":
                with pub._state_lock:
                    anchor = pub._latest
                if anchor is None or anchor.seq != int(cmd.get("anchor_seq", -1)):
                    _send_response({"ok": False, "error": "anchor_not_latest"})
                else:
                    path = pub.save_anchor_for_sam3(anchor, Path(cmd["debug_dir"]))
                    _send_response({"ok": True, "path": path})
            elif op == "save_anchor_depth_intrinsics":
                with pub._state_lock:
                    anchor = pub._latest
                if anchor is None or anchor.seq != int(cmd.get("anchor_seq", -1)):
                    _send_response({"ok": False, "error": "anchor_not_latest"})
                else:
                    dp, ip = pub.save_anchor_intrinsics_and_depth(anchor, Path(cmd["artifact_dir"]))
                    _send_response({"ok": True, "depth_path": dp, "intrinsics_path": ip})
            elif op == "build_init_pcd":
                ply = pub.build_static_init_pointcloud()
                _send_response({"ok": True, "ply_path": str(ply)})
            elif op == "pause_gazebo":
                _send_response({"ok": bool(pub.pause_gazebo())})
            elif op == "unpause_gazebo":
                _send_response({"ok": bool(pub.unpause_gazebo())})
            elif op == "shutdown":
                _send_response({"ok": True})
                break
            else:
                _send_response({"ok": False, "error": "unknown_op: {}".format(op)})
        except Exception as exc:
            _send_response({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)})

    return 0


if __name__ == "__main__":
    sys.exit(_main())
