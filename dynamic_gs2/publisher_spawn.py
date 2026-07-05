"""publisher_spawn.py — spawn the py3.8 ROS publisher subprocess (LiveBridgeSource + static sweep).

Extracted VERBATIM from dynamic_gs.utils.live_shm_reader: the LIVE_ROOT data-root constant, the
launcher constants, and `_spawn_publisher` (the cross-env process launcher). The rest of that
632-line module (the old SHM-layout mirror, LiveShmSubscriber, LiveFrame) is dead from dynamic_gs2's
perspective — dynamic_gs2 replaced it with frame.py + shm_channel.py (ShmRing).

Detach complete (Phase 6): _PUBLISHER_SCRIPT points at the dynamic_gs2-local copy of the py3.8 ROS
publisher (dynamic_gs2/live_ros_publisher.py), which path-loads its ros_mask.py / frame.py /
depth_filter.py / zed_depth_noise.py siblings from this same dir. No dynamic_gs dependency remains.
"""

import os
import subprocess
from pathlib import Path

# Live data root. Override via DGS_LIVE_ROOT env var (set by
# scripts/bootstrap_live.sh so a fresh scene gets its own dir without
# stomping prior recordings). Default kept for back-compat with the
# legacy hard-coded location. Wiped + recreated at session start by
# the publisher when --wipe-live-root is passed.
LIVE_ROOT = Path(os.environ.get(
    "DGS_LIVE_ROOT",
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/live",
))

# Hardcoded env names and paths. The ROS env is `dynamic_gs_ros` (as of
# 2026-05-14, replacing the deleted `radiance_ros_4060`): a minimal
# python 3.8 + numpy + opencv env that gets its ROS Noetic bindings by
# sourcing /opt/ros/noetic/setup.bash before launch (rospy etc. live at
# /opt/ros/noetic/lib/python3/dist-packages). SAM3 stays in
# `sam3_dynamic_gs`; SAM3D was migrated into `sam3_dynamic_gs` too.
ROS_PUBLISHER_CONDA_ENV = "dynamic_gs_ros"
ROS_SETUP_SCRIPT = "/opt/ros/noetic/setup.bash"

# The publisher body is the dynamic_gs2-local py3.8 ROS script (plain sibling).
_PUBLISHER_SCRIPT = Path(__file__).resolve().parent / "live_ros_publisher.py"
_CONDA_ROOT = Path("/home/mrc-cuhk/miniconda3")
_ROS_ENV_PYTHON = _CONDA_ROOT / "envs" / ROS_PUBLISHER_CONDA_ENV / "bin" / "python"


def _spawn_publisher(
    live_root: Path,
    shm_name: str,
    keyframe_translation_m: float,
    keyframe_rotation_deg: float,
    wipe_live_root: bool,
    new_layout: bool = False,
    max_hz: float = 0.0,
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
    if new_layout:
        inner_args.append("--new-layout")     # publisher writes the dynamic_gs2 frame.py SHM directly
    # Replay recording: if DGS_RECORD_REPLAY=<dir> is set, the publisher records
    # the full SHM frame stream + control events there for deterministic replay.
    _rec_dir = os.environ.get("DGS_RECORD_REPLAY")
    if _rec_dir:
        inner_args += ["--record-replay", str(_rec_dir)]
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
    # Per-spawn output-rate cap (default 0 = OFF). Injected into THIS subprocess's env only, so the
    # static-sweep publisher can be throttled without touching the dynamic publisher (separate spawn).
    if max_hz and max_hz > 0:
        env["DGS_PUB_MAX_HZ"] = str(max_hz)

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
