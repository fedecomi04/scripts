"""POSIX-shm reader + control-pipe client for the live ROS publisher.

Runs in the `dynamic_gs` env (python 3.12+, torch 2.7+/2.11+, cu128).
Spawns ``dynamic_gs/utils/live_ros_publisher.py`` as a subprocess in the
ROS env (`radiance_ros_4060`) and communicates with it via:

* POSIX shared memory — every synced (rgb, depth, mask, pose, stamp)
  tuple lands in a fixed-size slot. Reader-side reads are lock-free
  using the publisher's seqlock-style write order (slot.seq tagged
  before payload, header.latest_seq bumped after).
* stdin / stdout JSON lines — control commands (capture_anchor,
  start_recording, stop_recording, build_init_pcd, pause_gazebo, ...)
  and their matching responses.

The class ``LiveShmSubscriber`` mirrors the old ``LiveRosSubscriber``
API surface (``peek_latest``, ``intrinsics``, ``wait_for_first_frame``,
``capture_anchor``, ``start_recording``, ``stop_recording``,
``num_recorded_frames``, ``build_static_init_pointcloud``) so the
pipeline call site only changes the import.
"""

from __future__ import annotations

import atexit
import json
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType


# Hardcoded data root for live runs. Wiped + recreated at session start by
# the publisher (`--wipe-live-root`).
LIVE_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/live"
)

DEFAULT_SHM_NAME = "/dgs_live_shm"
NUM_SLOTS = 4
HEADER_BYTES = 4096

# Mirror of publisher's _HDR_FMT — keep these in sync.
_HDR_FMT = "<4sIIII4xdddd Q Q IIIIII II"
_HDR_SIZE = struct.calcsize(_HDR_FMT)


def _decode_header(buf: memoryview):
    fields = struct.unpack_from(_HDR_FMT, buf, 0)
    return {
        "magic": fields[0],
        "version": fields[1],
        "height": fields[2],
        "width": fields[3],
        "num_slots": fields[4],
        "fx": fields[5], "fy": fields[6], "cx": fields[7], "cy": fields[8],
        "latest_seq": fields[9],
        "slot_bytes": fields[10],
        "rgb_off": fields[11], "depth_off": fields[12], "mask_off": fields[13],
        "pose_off": fields[14], "seq_off": fields[15], "stamp_off": fields[16],
        "ready": fields[17], "shutdown": fields[18],
    }


def _compute_header_field_offsets():
    prefixes = [
        ("magic", "<4s"),
        ("version", "<4sI"),
        ("height", "<4sII"),
        ("width", "<4sIII"),
        ("num_slots", "<4sIIII"),
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
    out = {"magic": 0}
    for i in range(1, len(prefixes)):
        out[prefixes[i][0]] = struct.calcsize(prefixes[i - 1][1])
    return out


HDR_OFFSETS = _compute_header_field_offsets()


# ---------------------------------------------------------------------------
# LiveFrame: same shape the pipeline/live_session already operate on
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsicsLite:
    """Minimal CameraIntrinsics replica (no recorder import needed)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class LiveFrame:
    """One synced live tuple. Same shape live_session/pipeline use.

    Note vs the legacy LiveRosSubscriber: ``depth_mm`` (uint16) was
    replaced by ``depth_m`` (float32, metres) because the IPC layer
    already converts depth at the publisher to avoid a second
    conversion on the consumer side. The pipeline reads ``.depth_m``
    directly.
    """
    seq: int
    stamp_sec: float
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    mask_keep: np.ndarray
    c2w_4x4: np.ndarray


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------


# Hardcoded env names and paths. The ROS env is `dynamic_gs_ros` (as of
# 2026-05-14, replacing the deleted `radiance_ros_4060`): a minimal
# python 3.8 + numpy + opencv env that gets its ROS Noetic bindings by
# sourcing /opt/ros/noetic/setup.bash before launch (rospy etc. live at
# /opt/ros/noetic/lib/python3/dist-packages). SAM3 stays in
# `sam3_dynamic_gs`; SAM3D was migrated into `sam3_dynamic_gs` too.
ROS_PUBLISHER_CONDA_ENV = "dynamic_gs_ros"
ROS_SETUP_SCRIPT = "/opt/ros/noetic/setup.bash"

_PUBLISHER_SCRIPT = Path(__file__).resolve().parent / "live_ros_publisher.py"
_CONDA_ROOT = Path("/home/mrc-cuhk/miniconda3")
_ROS_ENV_PYTHON = _CONDA_ROOT / "envs" / ROS_PUBLISHER_CONDA_ENV / "bin" / "python"


def _spawn_publisher(
    live_root: Path,
    shm_name: str,
    keyframe_translation_m: float,
    keyframe_rotation_deg: float,
    wipe_live_root: bool,
) -> subprocess.Popen:
    """Start the publisher in the ROS env. Returns the Popen handle.

    The subprocess is launched by direct invocation of the ROS env's
    python interpreter — bypassing ``conda run`` so there's no extra
    process between us and the publisher, and so stdin/stdout stay
    clean. ROS Python bindings already live in the env's site-packages
    so no sourcing of /opt/ros is needed.
    """
    if not _ROS_ENV_PYTHON.exists():
        raise RuntimeError(
            f"ROS env python not found at {_ROS_ENV_PYTHON}. "
            f"Expected env '{ROS_PUBLISHER_CONDA_ENV}' under {_CONDA_ROOT}/envs/."
        )
    if not Path(ROS_SETUP_SCRIPT).exists():
        raise RuntimeError(
            f"ROS setup script not found at {ROS_SETUP_SCRIPT}. "
            f"Install ros-noetic-desktop-full or update ROS_SETUP_SCRIPT."
        )
    # Build the inner command (env python + publisher script + args).
    inner_args = [
        "-u", str(_PUBLISHER_SCRIPT),
        "--live-root", str(live_root),
        "--shm-name", shm_name,
        "--keyframe-translation-m", str(keyframe_translation_m),
        "--keyframe-rotation-deg", str(keyframe_rotation_deg),
    ]
    if wipe_live_root:
        inner_args.append("--wipe-live-root")
    # Wrap in a bash shell that sources ROS Noetic first so the env's
    # python finds rospy / sensor_msgs / tf at /opt/ros/noetic/lib/python3/
    # dist-packages. Pin PYTHONNOUSERSITE=1 so the user-local Python 3.8
    # site-packages (~/.local/lib/python3.8/site-packages) — which is
    # added implicitly by Python's site.py with shadowing precedence —
    # does NOT mask the env's own packages (notably pyrender, urdfpy,
    # OpenGL, trimesh which are deliberately pinned in this env).
    # `exec` so the python process becomes the bash PID's only child.
    inner_quoted = " ".join(f"'{a}'" for a in inner_args)
    cmd = [
        "bash", "-c",
        f"source '{ROS_SETUP_SCRIPT}' && export PYTHONNOUSERSITE=1 && exec '{_ROS_ENV_PYTHON}' {inner_quoted}",
    ]

    # Inherit current env vars (ROS_MASTER_URI etc.) but pin PYTHONUNBUFFERED.
    # CRITICAL: do NOT pass dynamic_gs's LD_LIBRARY_PATH / CPATH / CUDA_HOME
    # through to the ROS-env subprocess. Those env vars target the conda
    # libstdc++/cuda layout in dynamic_gs (libstdc++.so.6.0.34 etc.); the
    # ROS env subprocess loads C-extensions linked against an older
    # libstdc++ + the system OpenGL stack, and a forwarded path makes
    # rospy / pyrender pick up incompatible .so's at load time.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    for _var in ("LD_LIBRARY_PATH", "CPATH", "LIBRARY_PATH", "CUDA_HOME"):
        env.pop(_var, None)

    # Forward publisher stderr to a log file. Important: log lives in
    # /tmp so the publisher's --wipe-live-root doesn't blow it away
    # mid-run (the publisher rmtrees live_root before constructing the
    # ROS subscribers, which would invalidate our open file descriptor).
    log_dir = Path("/tmp/dgs_live_publisher")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "publisher.stderr.log"
    # Truncate per session so it's easy to read.
    log_fd = open(log_path, "wb")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log_fd,
        bufsize=0,  # we'll do line-buffering by reading bytes
        env=env,
    )
    # Stash the log path on the Popen handle so the caller can surface
    # it in error messages.
    proc._dgs_log_path = log_path  # type: ignore[attr-defined]
    return proc


# ---------------------------------------------------------------------------
# Subscriber: shared-memory reader + control pipe client
# ---------------------------------------------------------------------------


class LiveShmSubscriber:
    """Reader for the publisher's shm region + control-pipe client.

    Lifetime: one per training session. Spawns the publisher subprocess
    in ``__init__`` (blocking until first "ready" message), and signals
    shutdown in ``close()`` / atexit.
    """

    _singleton: Optional["LiveShmSubscriber"] = None

    def __init__(
        self,
        live_root: Path = LIVE_ROOT,
        shm_name: str = DEFAULT_SHM_NAME,
        keyframe_translation_m: float = 0.02,
        keyframe_rotation_deg: float = 20.0,
        wipe_live_root: bool = True,
        ready_timeout_s: float = 30.0,
    ):
        self.live_root = Path(live_root)
        self.shm_name = shm_name
        self._proc = _spawn_publisher(
            live_root=self.live_root,
            shm_name=shm_name,
            keyframe_translation_m=keyframe_translation_m,
            keyframe_rotation_deg=keyframe_rotation_deg,
            wipe_live_root=wipe_live_root,
        )
        self._proc_lock = threading.Lock()  # for stdin/stdout serialization
        self._closed = False

        # Read first line — expect {"event": "ready", ...}. The publisher
        # may take a few seconds to wait_for_message on /camera_info before
        # this fires.
        ready = self._read_response(timeout_s=ready_timeout_s)
        if ready.get("event") != "ready":
            self._proc.terminate()
            raise RuntimeError(f"publisher init failed: {ready}")

        self._intrinsics = CameraIntrinsicsLite(
            width=int(ready["width"]),
            height=int(ready["height"]),
            fx=float(ready["fx"]),
            fy=float(ready["fy"]),
            cx=float(ready["cx"]),
            cy=float(ready["cy"]),
        )
        self._num_slots = int(ready["num_slots"])
        self._slot_bytes = int(ready["slot_bytes"])
        self._header_bytes = int(ready["header_bytes"])

        # Attach to the publisher's shm region. We unregister with the
        # multiprocessing resource_tracker right after attaching — the
        # publisher subprocess is the sole owner of the shm name, and
        # without this the reader's atexit cleanup tries to unlink a
        # name it never created, producing the "leaked shared_memory
        # objects" warning. See https://bugs.python.org/issue38119.
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        try:
            from multiprocessing import resource_tracker as _rt
            _rt.unregister(self._shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        header = _decode_header(self._shm.buf)
        if header["magic"] != b"DGS\0":
            raise RuntimeError(f"shm magic mismatch: {header['magic']!r}")
        self._offsets = {
            "rgb_off": header["rgb_off"], "depth_off": header["depth_off"],
            "mask_off": header["mask_off"], "pose_off": header["pose_off"],
            "seq_off": header["seq_off"], "stamp_off": header["stamp_off"],
        }
        self._slot_views = self._build_slot_views()

        LiveShmSubscriber._singleton = self
        atexit.register(self._atexit_close)

    @classmethod
    def get_singleton(cls) -> "LiveShmSubscriber":
        if cls._singleton is None:
            return cls()
        return cls._singleton

    @property
    def intrinsics(self) -> CameraIntrinsicsLite:
        return self._intrinsics

    # ---- Slot views ----

    def _build_slot_views(self):
        H = self._intrinsics.height
        W = self._intrinsics.width
        views = []
        for i in range(self._num_slots):
            base = self._header_bytes + i * self._slot_bytes
            v = {
                "pose": np.frombuffer(self._shm.buf, dtype=np.float64,
                                      count=16, offset=base + self._offsets["pose_off"]).reshape(4, 4),
                "seq":  np.frombuffer(self._shm.buf, dtype=np.uint64,
                                      count=1, offset=base + self._offsets["seq_off"]),
                "stamp": np.frombuffer(self._shm.buf, dtype=np.float64,
                                       count=1, offset=base + self._offsets["stamp_off"]),
                "rgb":  np.frombuffer(self._shm.buf, dtype=np.uint8,
                                      count=H * W * 3,
                                      offset=base + self._offsets["rgb_off"]).reshape(H, W, 3),
                "depth": np.frombuffer(self._shm.buf, dtype=np.float32,
                                       count=H * W,
                                       offset=base + self._offsets["depth_off"]).reshape(H, W),
                "mask": np.frombuffer(self._shm.buf, dtype=np.uint8,
                                      count=H * W,
                                      offset=base + self._offsets["mask_off"]).reshape(H, W),
            }
            views.append(v)
        return views

    # ---- High-frequency frame read (lock-free) ----

    def peek_latest(self) -> Optional[LiveFrame]:
        """Return a copy of the latest published frame, or None.

        The reader does:
          1. Read header.latest_seq (= S_pre)
          2. Read slot[S_pre % NUM_SLOTS] (copy into private arrays)
          3. Read header.latest_seq again (= S_post)
          4. If slot.seq == S_pre AND S_pre == S_post → consistent.
          5. Else retry up to a few times before giving up.
        """
        # Fast path: peek into shm. Each per-slot field is a numpy view
        # over the mmap buffer; `np.array(view)` copies into a fresh
        # owned array.
        for _attempt in range(5):
            s_pre = struct.unpack_from("<Q", self._shm.buf, HDR_OFFSETS["latest_seq"])[0]
            if s_pre == 0:
                return None  # no frame published yet
            slot_idx = int(s_pre) % self._num_slots
            slot = self._slot_views[slot_idx]
            slot_seq = int(slot["seq"][0])
            if slot_seq != s_pre:
                # Publisher is mid-write into this slot; retry.
                time.sleep(0)
                continue
            # Snapshot (copy out of shm).
            rgb = np.array(slot["rgb"], copy=True)
            depth = np.array(slot["depth"], copy=True)
            mask = np.array(slot["mask"], copy=True)
            pose = np.array(slot["pose"], copy=True)
            stamp = float(slot["stamp"][0])
            # Verify after copy.
            s_post = struct.unpack_from("<Q", self._shm.buf, HDR_OFFSETS["latest_seq"])[0]
            slot_seq_post = int(slot["seq"][0])
            if s_post == s_pre and slot_seq_post == s_pre:
                return LiveFrame(
                    seq=int(s_pre), stamp_sec=stamp,
                    rgb_bgr=rgb, depth_m=depth, mask_keep=mask, c2w_4x4=pose,
                )
        # Took too many tries — the publisher is hot. Return None so the
        # caller re-queries next tick. This is exceedingly rare at 30 Hz
        # with NUM_SLOTS=4 and per-read ms timing.
        return None

    def wait_for_first_frame(self, timeout_s: float = 30.0) -> LiveFrame:
        """Spin until peek_latest returns a frame, with a stamp-or-die deadline."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = self.peek_latest()
            if frame is not None:
                return frame
            # Cheap sleep, won't block ROS callbacks (they're in the
            # publisher subprocess, unaffected).
            time.sleep(0.05)
        raise TimeoutError(
            f"no synced (rgb, depth, pose) tuple within {timeout_s}s — check that the "
            f"publisher subprocess is running and ROS topics are publishing"
        )

    # ---- Control-pipe commands ----

    def _send_command(self, op: str, **kwargs) -> dict:
        with self._proc_lock:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"publisher subprocess exited (code={self._proc.returncode}); "
                    f"see /tmp/dgs_live_publisher/publisher.stderr.log"
                )
            payload = {"op": op, **kwargs}
            line = (json.dumps(payload) + "\n").encode("utf-8")
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            return self._read_response()

    def _read_response(self, timeout_s: float = 600.0) -> dict:
        """Read one JSON line from the publisher's stdout (binary mode).

        Tees every byte to /tmp/dgs_live_publisher/publisher.stdout.log
        so we can post-hoc inspect what the publisher actually emitted
        on a parse failure.
        """
        buf = bytearray()
        deadline = time.time() + timeout_s
        log_dir = Path("/tmp/dgs_live_publisher")
        log_dir.mkdir(parents=True, exist_ok=True)
        debug_path = log_dir / "publisher.stdout.log"
        with open(debug_path, "ab") as dbg:
            while True:
                byte = self._proc.stdout.read(1)
                if not byte:
                    raise RuntimeError(
                        f"publisher EOF on stdout; see /tmp/dgs_live_publisher/publisher.stderr.log"
                    )
                dbg.write(byte)
                dbg.flush()
                if byte == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    if not line:
                        buf.clear()
                        continue
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"publisher emitted non-JSON line: {line!r} "
                            f"(parse error: {exc})"
                        )
                buf.extend(byte)
                if time.time() > deadline:
                    raise TimeoutError(
                        f"publisher response timeout after {timeout_s}s; partial={bytes(buf)!r}"
                    )

    def capture_anchor(self) -> LiveFrame:
        """Block for the next strictly-newer synced tuple.

        Returns a fresh LiveFrame — the publisher confirms the seq via
        the control pipe, and we then peek it out of shm.
        """
        resp = self._send_command("capture_anchor", timeout_s=30.0)
        if not resp.get("ok"):
            raise RuntimeError(f"capture_anchor failed: {resp}")
        target_seq = int(resp["seq"])
        # The slot is fresh — read it.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            frame = self.peek_latest()
            if frame is not None and frame.seq >= target_seq:
                return frame
            time.sleep(0.005)
        raise RuntimeError(f"anchor seq {target_seq} disappeared from shm")

    def start_recording(self, anchor: LiveFrame) -> None:
        resp = self._send_command("start_recording", anchor_seq=int(anchor.seq))
        if not resp.get("ok"):
            raise RuntimeError(f"start_recording failed: {resp}")

    def stop_recording(self) -> None:
        resp = self._send_command("stop_recording")
        if not resp.get("ok"):
            raise RuntimeError(f"stop_recording failed: {resp}")

    def num_recorded_frames(self) -> int:
        resp = self._send_command("num_recorded")
        if not resp.get("ok"):
            raise RuntimeError(f"num_recorded failed: {resp}")
        return int(resp["recorded"])

    def save_anchor_for_sam3(self, anchor: LiveFrame, debug_dir: Path) -> Path:
        resp = self._send_command(
            "save_anchor_for_sam3",
            anchor_seq=int(anchor.seq), debug_dir=str(debug_dir),
        )
        if not resp.get("ok"):
            raise RuntimeError(f"save_anchor_for_sam3 failed: {resp}")
        return Path(resp["path"])

    def save_anchor_intrinsics_and_depth(
        self, anchor: LiveFrame, artifact_dir: Path
    ) -> tuple[Path, Path]:
        resp = self._send_command(
            "save_anchor_depth_intrinsics",
            anchor_seq=int(anchor.seq), artifact_dir=str(artifact_dir),
        )
        if not resp.get("ok"):
            raise RuntimeError(f"save_anchor_intrinsics_and_depth failed: {resp}")
        return Path(resp["depth_path"]), Path(resp["intrinsics_path"])

    def build_static_init_pointcloud(self) -> Path:
        resp = self._send_command("build_init_pcd", timeout_s=300.0)
        # Allow longer for the PLY build (~tens of seconds for many frames).
        if not resp.get("ok"):
            raise RuntimeError(f"build_init_pcd failed: {resp}")
        return Path(resp["ply_path"])

    def pause_gazebo_physics(self) -> bool:
        try:
            resp = self._send_command("pause_gazebo")
            return bool(resp.get("ok"))
        except Exception:
            return False

    def unpause_gazebo_physics(self) -> bool:
        try:
            resp = self._send_command("unpause_gazebo")
            return bool(resp.get("ok"))
        except Exception:
            return False

    # ---- Shutdown ----

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Best-effort graceful shutdown — send "shutdown" then wait.
        try:
            with self._proc_lock:
                if self._proc.poll() is None:
                    payload = (json.dumps({"op": "shutdown"}) + "\n").encode("utf-8")
                    try:
                        self._proc.stdin.write(payload)
                        self._proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass
        # Drop shm views before closing.
        self._slot_views = []
        try:
            self._shm.close()
        except Exception:
            pass

    def _atexit_close(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Camera builder (matches old cameras_from_live_frame)
# ---------------------------------------------------------------------------


def cameras_from_live_frame(
    frame: LiveFrame,
    intrinsics: CameraIntrinsicsLite,
    device: torch.device,
    cam_idx: int = 0,
) -> Cameras:
    """Build a single-frame Nerfstudio Cameras object from a LiveFrame."""
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
