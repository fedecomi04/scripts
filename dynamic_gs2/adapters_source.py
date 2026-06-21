"""adapters_source.py — the ONE file a new user customizes for their hardware/middleware.

Owns the PRODUCER end of the Frame contract: fills the SHM ring. The rest of the
pipeline only ever READS the ring (ShmRing.peek_latest). One ingest path for both
recorded (ReplaySource) and live (Ros1Source) → live + replay are identical
downstream. (rewrite_spec/adapters_source.md; imports frame.py + shm_channel.py per D1.)

Sources:
  - ReplaySource  : disk dataset -> SHM. paced (real-time proxy, default) or fast
                    (lock-step, frame-exact). In-process, dynamic_gs env. FULLY TESTED.
  - Ros1Source    : spawns the ROS publisher subprocess (dynamic_gs_ros, py3.8) that
                    writes the new-layout SHM. Structural here; publisher body = ros_publisher.py.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from .frame import DEFAULT_NUM_SLOTS, DEFAULT_SHM_NAME, Frame, Intrinsics
from .shm_channel import ShmConsumer, ShmProducer


# --------------------------------------------------------------- interface
class FrameSource(Protocol):
    def intrinsics(self) -> Intrinsics: ...
    def attach(self, shm_name: str) -> None: ...
    def next_frame(self) -> Optional[Frame]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------- consumer
class ShmRing:
    """Reader-side attach to the producer's ring. NOT a FrameSource. Lock-free peek."""

    def __init__(self, shm_name: str = DEFAULT_SHM_NAME):
        self._c = ShmConsumer(name=shm_name)

    def intrinsics(self) -> Intrinsics:
        return self._c.intrinsics

    def peek_latest(self) -> Optional[Frame]:
        return self._c.read_latest()

    def is_shutdown(self) -> bool:
        return self._c.is_shutdown()

    def close(self) -> None:
        self._c.close()


# --------------------------------------------------------------- camera build
def camera_from_frame(frame: Frame, intr: Intrinsics, device, cam_idx: int = 0):
    """Single nerfstudio Cameras from a Frame's OpenGL c2w + intrinsics. torch imported lazily."""
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    c2w = torch.from_numpy(np.ascontiguousarray(frame.c2w_4x4[:3, :4], dtype=np.float32)).unsqueeze(0)
    cam = Cameras(
        camera_to_worlds=c2w,
        fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.cx), cy=float(intr.cy),
        width=int(intr.width), height=int(intr.height),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    cam.metadata = {"cam_idx": int(cam_idx)}
    return cam


# --------------------------------------------------------------- replay
def _rel(data_dir: Path, p: str) -> Path:
    return (data_dir / p.lstrip("./")) if not os.path.isabs(p) else Path(p)


class ReplaySource:
    """Drive a recorded dynamic dataset through SHM so it 'feels live'.

    paced (DEFAULT): a producer thread writes EVERY frame on its capture-stamp schedule;
        the consumer (peek_latest) drops/trails like live -> honest real-time test.
    fast: NO thread; the harness calls next_frame() each step (lock-step, frame-exact,
        deterministic) -> the pose A/B path. next_frame()->None at end-of-dataset.
    """

    def __init__(self, data_dir, *, mode: str = "paced", replay_fps: float = 15.0,
                 transforms_name: str = "transforms.json", num_slots: int = DEFAULT_NUM_SLOTS,
                 loop: bool = False):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.replay_fps = float(replay_fps)
        self._slots = num_slots
        self._loop = bool(loop)   # paced: replay the episode forever (tracker snap-resets at each wrap)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prod: Optional[ShmProducer] = None
        self._seq = 0
        self._idx = 0

        dd = self.data_dir / "dynamic_scene" if (self.data_dir / "dynamic_scene").is_dir() else self.data_dir
        self._frame_dir = dd
        meta = json.loads((dd / transforms_name).read_text())
        self._intr = Intrinsics(width=int(meta["w"]), height=int(meta["h"]),
                                fx=float(meta["fl_x"]), fy=float(meta["fl_y"]),
                                cx=float(meta["cx"]), cy=float(meta["cy"]))
        frames = meta["frames"]

        def _fidx(f):
            nums = re.findall(r"\d+", f["file_path"])
            return int(nums[-1]) if nums else 0
        frames = sorted(frames, key=_fidx)
        self._frames = frames
        # capture stamps -> relative seconds; synth from fps if absent (principle #5)
        stamps = [f.get("stamp_wall") for f in frames]
        # Pace on the REAL capture stamps when present (honest variable-dt replay). DGS_REPLAY_UNIFORM_FPS=1
        # overrides them with a fixed replay_fps cadence (--fps), e.g. to remove the capture jitter.
        force_fps = os.environ.get("DGS_REPLAY_UNIFORM_FPS") == "1"
        if (not force_fps) and all(s is not None for s in stamps) and len(stamps) > 1:
            s0 = stamps[0]
            self._rel_t = [float(s - s0) for s in stamps]
        else:
            self._rel_t = [i / self.replay_fps for i in range(len(frames))]

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def __len__(self) -> int:
        return len(self._frames)

    def _load(self, i: int) -> Frame:
        f = self._frames[i]
        rgb = cv2.imread(str(_rel(self._frame_dir, f["file_path"])), cv2.IMREAD_COLOR)  # BGR uint8
        dpath = f.get("depth_file_path") or f["file_path"].replace("rgb", "depth").replace(".png", ".tiff")
        d = cv2.imread(str(_rel(self._frame_dir, dpath)), cv2.IMREAD_UNCHANGED)
        depth_m = (d.astype(np.float32) * 1e-3) if d is not None else \
            np.zeros((self._intr.height, self._intr.width), np.float32)
        mpath = f.get("mask_path")
        if mpath:
            m = cv2.imread(str(_rel(self._frame_dir, mpath)), cv2.IMREAD_GRAYSCALE)
            mask = (m > 0).astype(np.uint8) if m is not None else np.ones_like(depth_m, np.uint8)
        else:
            mask = np.ones(depth_m.shape, np.uint8)
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        self._seq += 1
        return Frame(seq=self._seq, stamp_sec=self._rel_t[i],
                     rgb_bgr=np.ascontiguousarray(rgb), depth_m=np.ascontiguousarray(depth_m),
                     mask_keep=np.ascontiguousarray(mask), c2w_4x4=c2w)

    def attach(self, shm_name: str = DEFAULT_SHM_NAME) -> None:
        self._prod = ShmProducer(self._intr, name=shm_name, num_slots=self._slots)
        if self.mode == "paced":
            # publish frame 0 immediately so the reader can peek, then auto-drive the rest.
            if self._frames:
                self._prod.write(self._load(0))
                self._idx = 1
            self._thread = threading.Thread(target=self._paced_loop, daemon=True)
            self._thread.start()

    def _paced_loop(self) -> None:
        t0 = self._rel_t[0] if self._rel_t else 0.0
        n = len(self._frames)
        while not self._stop.is_set():
            start_wall = time.time()   # restart the pacing clock at the top of each pass
            while not self._stop.is_set() and self._idx < n:
                target = start_wall + (self._rel_t[self._idx] - t0)
                dt = target - time.time()
                if dt > 0:
                    self._stop.wait(dt)
                if self._stop.is_set():
                    break
                self._prod.write(self._load(self._idx))
                self._idx += 1
            if not self._loop or self._stop.is_set():
                break
            self._idx = 0              # wrap: republish from frame 0 (object jumps back to its D0 pose)
        if self._prod is not None and not self._stop.is_set():
            self._prod.mark_shutdown()

    def next_frame(self) -> Optional[Frame]:
        """fast mode: produce + publish ONE frame, lock-step. paced: thread drives; this is a no-op->None."""
        if self.mode == "paced":
            return None
        if self._idx >= len(self._frames):
            if self._prod is not None:
                self._prod.mark_shutdown()
            return None
        fr = self._load(self._idx)
        self._idx += 1
        self._prod.write(fr)
        return fr

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._prod is not None:
            self._prod.close(unlink=True)
            self._prod = None


# --------------------------------------------------------------- live (ROS)
# Load-bearing env-strip (H10): the ROS publisher runs in the py3.8 dynamic_gs_ros
# env via a bash wrapper that sources ROS; the parent (dynamic_gs) env's
# LD_LIBRARY_PATH/CPATH/CUDA_HOME would otherwise shadow it.
_STRIP_ENV_VARS = ("LD_LIBRARY_PATH", "CPATH", "CUDA_HOME", "PYTHONPATH")


def _ros_publisher_env() -> dict:
    env = dict(os.environ)
    for k in _STRIP_ENV_VARS:
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class Ros1Source:
    """Spawn the ROS1 (Noetic) publisher subprocess that fills the new-layout SHM ring.

    The publisher body lives in dynamic_gs2/ros_publisher.py (run inside dynamic_gs_ros).
    This class owns the subprocess lifecycle (spawn with env-strip + ROS source,
    terminate->kill). It does NOT import rospy itself (stays env-portable). The reader
    side is a plain ShmRing on the same shm_name.

    NOTE: this path requires a live Gazebo/ROS stack and is validated by the OPERATOR
    (pipeline step 4), not in unattended tests.
    """

    def __init__(self, *, shm_name: str = DEFAULT_SHM_NAME, ros_env: str = "dynamic_gs_ros",
                 ros_setup: str = "/opt/ros/noetic/setup.bash",
                 conda_sh: str = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh"),
                 publisher_args: Optional[list] = None, log_path: Optional[str] = None):
        self.shm_name = shm_name
        self.ros_env = ros_env
        self.ros_setup = ros_setup
        self.conda_sh = conda_sh
        self.publisher_args = publisher_args or []
        self.log_path = log_path
        self._proc: Optional[subprocess.Popen] = None
        self._log_fd = None
        self._intr: Optional[Intrinsics] = None

    def attach(self, shm_name: Optional[str] = None) -> None:
        shm_name = shm_name or self.shm_name
        mod = "dynamic_gs2.ros_publisher"
        inner = (f"source {self.conda_sh} && conda activate {self.ros_env} && "
                 f"source {self.ros_setup} && "
                 f"exec python -m {mod} --shm-name {shm_name} " + " ".join(self.publisher_args))
        self._log_fd = open(self.log_path, "ab") if self.log_path else None
        self._proc = subprocess.Popen(
            ["bash", "-lc", inner], env=_ros_publisher_env(),
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=self._log_fd or None, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def intrinsics(self) -> Intrinsics:
        # Live intrinsics are published into the SHM header by the subprocess; read via ShmRing.
        if self._intr is None:
            raise RuntimeError("Ros1Source intrinsics come from the SHM header — read via ShmRing(shm_name).intrinsics()")
        return self._intr

    def next_frame(self) -> Optional[Frame]:
        # The subprocess produces frames straight into SHM; the pipeline reads via ShmRing.
        return None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    self._proc.wait(timeout=5.0)
            except ProcessLookupError:
                pass
            self._proc = None
        if self._log_fd is not None:
            self._log_fd.close()
            self._log_fd = None


# --------------------------------------------------------------- live (proven publisher, DIRECT new-layout)
class LiveBridgeSource:
    """RECOMMENDED live producer: spawns the proven py3.8 ROS publisher with --new-layout so it writes
    THIS SHM segment (frame.py layout) DIRECTLY. The tracker's ShmRing reads it — NO forwarder thread,
    NO LiveFrame copy, NO SHM-A->SHM-B copy (the old double-write is gone). All the proven
    rospy/FK/mask/decode code is reused unchanged; only the publisher's SHM-write target changed.

    NOTE: requires a live Gazebo/ROS stack; validated by the OPERATOR (pipeline step 4).
    """

    def __init__(self, *, old_shm_name: str = "dgs_live_shm", live_root=None,
                 ready_timeout_s: float = 30.0, **sub_kwargs):
        self.old_shm_name = old_shm_name                  # kept for API-compat; unused in the direct path
        self.live_root = live_root
        self.ready_timeout_s = ready_timeout_s
        self._sub_kwargs = sub_kwargs
        self._proc = None
        self._intr: Optional[Intrinsics] = None

    def attach(self, shm_name: str = DEFAULT_SHM_NAME) -> None:
        import json as _json
        from dynamic_gs.utils.live_shm_reader import _spawn_publisher, LIVE_ROOT
        live_root = self.live_root if self.live_root is not None else LIVE_ROOT
        # Spawn the proven publisher writing the NEW layout straight into `shm_name` (what ShmRing reads).
        self._proc = _spawn_publisher(
            live_root=live_root, shm_name=shm_name,
            keyframe_translation_m=self._sub_kwargs.get("keyframe_translation_m", 0.02),
            keyframe_rotation_deg=self._sub_kwargs.get("keyframe_rotation_deg", 20.0),
            wipe_live_root=self._sub_kwargs.get("wipe_live_root", True),
            new_layout=True)   # the ONLY live path: publisher writes the frame.py SHM directly
        # Read the publisher's one-line "ready" handshake (intrinsics + shm open confirmation).
        import time as _t
        deadline = _t.time() + self.ready_timeout_s
        ready = None
        while _t.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError("live publisher exited during startup; see /tmp/dgs_live_publisher/")
                continue
            try:
                msg = _json.loads(line.decode().strip() if isinstance(line, bytes) else line.strip())
            except Exception:
                continue
            if msg.get("event") == "ready":
                ready = msg; break
            if msg.get("event") == "init_error":
                raise RuntimeError("live publisher init failed: %s" % msg.get("error"))
        if ready is None:
            raise RuntimeError("live publisher did not become ready in %.0fs" % self.ready_timeout_s)
        self._intr = Intrinsics(width=int(ready["width"]), height=int(ready["height"]),
                                fx=float(ready["fx"]), fy=float(ready["fy"]),
                                cx=float(ready["cx"]), cy=float(ready["cy"]))

    def intrinsics(self) -> Intrinsics:
        if self._intr is None:
            raise RuntimeError("LiveBridgeSource not attached")
        return self._intr

    def next_frame(self) -> Optional[Frame]:
        return None                                       # publisher writes SHM directly; read via ShmRing

    def close(self) -> None:
        # _spawn_publisher does NOT start a new session, so the proc shares our process group —
        # never killpg here (it would kill us). The cmd is `bash -c "... exec python ..."`, so exec
        # replaced bash: SIGTERM to proc.pid hits the python publisher directly (its SIGTERM handler
        # drops the rospy node + unlinks the SHM).
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None


# --------------------------------------------------------------- factory
def open_source(kind: str, data_dir=None, shm_name: str = DEFAULT_SHM_NAME,
                replay_mode: str = "paced", attach: bool = True, **opts) -> FrameSource:
    """Construct (+optionally attach) the requested source. data_dir required for 'replay'."""
    if kind == "replay":
        if data_dir is None:
            raise ValueError("replay source needs data_dir")
        src = ReplaySource(data_dir, mode=replay_mode, **opts)
    elif kind == "live_bridge":
        src = LiveBridgeSource(**opts)
    elif kind == "ros1":
        src = Ros1Source(shm_name=shm_name, **opts)
    elif kind == "ros2":
        raise NotImplementedError("ros2 source not implemented (ros1/Noetic only — adapters_source.md OQ#4)")
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    if attach:
        src.attach(shm_name)
    return src
