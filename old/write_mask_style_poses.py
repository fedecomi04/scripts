#!/usr/bin/env python3
"""
Write an alternate transforms file whose poses come from exact tf2 lookups at the
saved image timestamps.

This mirrors the important timing behavior of ros1_robot_mask_saver_stl.py:
it uses the image message timestamp itself, not the nearest /tf topic sample.

The current dataset saver already records those timestamps in transforms.json as
rgb_timestamp_sec and depth_timestamp_sec. This script reuses them and writes a
sidecar transforms file, by default transforms_mask_pose.json, so the original
dataset is left untouched.

Important limitation:
The exact TF lookups only work while the tf2 buffer still contains the relevant
history. For old completed runs, start this while the simulation is still
running or shortly after capture.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from tf.transformations import euler_matrix


DEFAULT_DATASET_ROOT = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/"
    "data_teleoperation/datasets/dynaarm_gs_depth_mask_01"
)
RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d{2})?$")
ROS_TO_NERFSTUDIO = np.array(
    euler_matrix(np.pi / 2.0, 0.0, -np.pi / 2.0),
    dtype=np.float64,
)
CAMERA_PROFILES = {
    "manual": {},
    "basic_depth": {"camera_frame": "camera_link"},
    "kinect": {"camera_frame": "camera_link_optical"},
}


def sanitize_frame(frame: Optional[str]) -> str:
    return (frame or "").strip().lstrip("/")


def normalize_frame_prefix(frame_prefix: Optional[str]) -> str:
    return sanitize_frame((frame_prefix or "").strip("/"))


def resolve_frame_name(frame_name: Optional[str], frame_prefix: Optional[str]) -> str:
    frame_name = sanitize_frame(frame_name)
    frame_prefix = normalize_frame_prefix(frame_prefix)
    if not frame_name:
        return ""
    if not frame_prefix or "/" in frame_name:
        return frame_name
    return f"{frame_prefix}/{frame_name}"


def get_camera_profile(camera_profile: str) -> Dict[str, str]:
    if camera_profile not in CAMERA_PROFILES:
        available = ", ".join(sorted(CAMERA_PROFILES))
        raise ValueError(f"Unknown camera profile '{camera_profile}'. Available: {available}")
    return dict(CAMERA_PROFILES[camera_profile])


def read_json_with_retry(path: Path, retries: int = 5, delay_sec: float = 0.05) -> Optional[Dict[str, Any]]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(delay_sec)

    if last_exc is not None:
        raise last_exc
    return None


def write_json_atomic(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    os.replace(tmp_path, path)


def ros_transform_to_matrix(transform_msg: Any) -> np.ndarray:
    t = transform_msg.transform.translation
    q = transform_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return mat


def ros_to_nerfstudio_c2w(world_T_ros_camera: np.ndarray) -> np.ndarray:
    return world_T_ros_camera @ ROS_TO_NERFSTUDIO


def is_dataset_run_dir(dataset_dir: Path) -> bool:
    dataset_dir = dataset_dir.expanduser().resolve()
    return (dataset_dir / "rgb").exists() or (dataset_dir / "transforms.json").exists()


def list_run_dirs(dataset_dir: Path) -> list[Path]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []
    return [
        path for path in dataset_dir.iterdir()
        if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
    ]


def ros_time_from_sec(stamp_sec: float) -> rospy.Time:
    secs = int(math.floor(stamp_sec))
    nsecs = int(round((stamp_sec - secs) * 1e9))
    if nsecs >= 1_000_000_000:
        secs += 1
        nsecs -= 1_000_000_000
    return rospy.Time(secs=secs, nsecs=nsecs)


class MaskStylePoseWriter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_root = Path(args.dataset_dir).expanduser().resolve()
        self.wait_for_new_run = bool(args.wait_for_new_run and not is_dataset_run_dir(self.dataset_root))
        self.known_run_names = (
            {run.name for run in list_run_dirs(self.dataset_root)}
            if self.wait_for_new_run
            else set()
        )
        self.current_run_dir: Optional[Path] = None
        self.lookup_cache: Dict[Tuple[str, str, float], Dict[str, Any]] = {}

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(args.tf_cache_sec))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        if self.wait_for_new_run:
            rospy.loginfo(
                "Waiting for a new dataset run under %s; ignoring %d existing run(s)",
                self.dataset_root,
                len(self.known_run_names),
            )
        elif is_dataset_run_dir(self.dataset_root):
            self.current_run_dir = self.dataset_root
        else:
            runs = list_run_dirs(self.dataset_root)
            if runs:
                self.current_run_dir = max(runs, key=lambda path: path.name)

    def _resolve_run_dir(self) -> Optional[Path]:
        if is_dataset_run_dir(self.dataset_root):
            return self.dataset_root

        runs = list_run_dirs(self.dataset_root)
        if not runs:
            return None

        if self.wait_for_new_run:
            new_runs = [run for run in runs if run.name not in self.known_run_names]
            if not new_runs:
                return None
            run_dir = max(new_runs, key=lambda path: path.name)
            self.known_run_names.add(run_dir.name)
            return run_dir

        return max(runs, key=lambda path: path.name)

    def _switch_run_if_needed(self) -> Optional[Path]:
        run_dir = self._resolve_run_dir()
        if run_dir is None:
            return None

        run_dir = run_dir.expanduser().resolve()
        if run_dir != self.current_run_dir:
            self.current_run_dir = run_dir
            self.lookup_cache.clear()
            rospy.loginfo("Active dataset run: %s", self.current_run_dir)
        return self.current_run_dir

    def _load_metadata(self, run_dir: Path) -> Dict[str, Any]:
        return read_json_with_retry(run_dir / "capture_metadata.json") or {}

    def _default_frames(self, metadata: Dict[str, Any]) -> Tuple[str, str]:
        base_frame = sanitize_frame(self.args.base_frame or metadata.get("base_frame"))
        if self.args.camera_frame:
            camera_frame = sanitize_frame(self.args.camera_frame)
        else:
            metadata_frame = sanitize_frame(metadata.get("camera_frame"))
            if metadata_frame:
                camera_frame = metadata_frame
            else:
                profile = get_camera_profile(self.args.camera_profile)
                camera_frame = resolve_frame_name(
                    profile.get("camera_frame"),
                    self.args.link_frame_prefix,
                )
        return base_frame, sanitize_frame(camera_frame)

    def _lookup_pose(self, base_frame: str, camera_frame: str, stamp_sec: float) -> Dict[str, Any]:
        cache_key = (base_frame, camera_frame, float(stamp_sec))
        cached = self.lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        tf_msg = self.tf_buffer.lookup_transform(
            base_frame,
            camera_frame,
            ros_time_from_sec(float(stamp_sec)),
            timeout=rospy.Duration(self.args.tf_timeout),
        )
        ros_matrix = ros_transform_to_matrix(tf_msg)
        ns_matrix = ros_to_nerfstudio_c2w(ros_matrix)
        result = {
            "ros_transform_matrix": ros_matrix.tolist(),
            "transform_matrix": ns_matrix.tolist(),
        }
        self.lookup_cache[cache_key] = result
        return result

    def _rewrite_frame(self, frame: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        out = dict(frame)
        default_base_frame, default_camera_frame = self._default_frames(metadata)
        base_frame = sanitize_frame(out.get("base_frame") or default_base_frame)
        camera_frame = sanitize_frame(out.get("camera_frame") or default_camera_frame)
        rgb_stamp_sec = out.get("rgb_timestamp_sec")

        if rgb_stamp_sec is None:
            return out, "missing rgb_timestamp_sec"
        if not base_frame:
            return out, "missing base_frame"
        if not camera_frame:
            return out, "missing camera_frame"

        try:
            rgb_pose = self._lookup_pose(base_frame, camera_frame, float(rgb_stamp_sec))
        except Exception as exc:
            return out, f"rgb lookup failed: {exc}"

        if self.args.preserve_source_fields:
            if "source_transform_matrix" not in out and "transform_matrix" in frame:
                out["source_transform_matrix"] = frame["transform_matrix"]
            if "source_ros_transform_matrix" not in out and "ros_transform_matrix" in frame:
                out["source_ros_transform_matrix"] = frame["ros_transform_matrix"]

        out["transform_matrix"] = rgb_pose["transform_matrix"]
        out["ros_transform_matrix"] = rgb_pose["ros_transform_matrix"]
        out["rgb_tf_timestamp_sec"] = float(rgb_stamp_sec)
        out["tf_timestamp_sec"] = float(rgb_stamp_sec)
        out["rgb_tf_dt_sec"] = 0.0
        out["tf_dt_sec"] = 0.0
        out["rgb_tf_lookup_mode"] = "exact_tf_lookup"
        out["tf_lookup_mode"] = "exact_tf_lookup"
        out["pose_source"] = "mask_style_exact_tf"
        out["base_frame"] = base_frame
        out["camera_frame"] = camera_frame

        depth_stamp_sec = out.get("depth_timestamp_sec")
        if depth_stamp_sec is not None and not self.args.skip_depth:
            try:
                depth_pose = self._lookup_pose(base_frame, camera_frame, float(depth_stamp_sec))
            except Exception as exc:
                return out, f"depth lookup failed: {exc}"

            if self.args.preserve_source_fields:
                if "source_depth_transform_matrix" not in out and "depth_transform_matrix" in frame:
                    out["source_depth_transform_matrix"] = frame["depth_transform_matrix"]
                if "source_ros_depth_transform_matrix" not in out and "ros_depth_transform_matrix" in frame:
                    out["source_ros_depth_transform_matrix"] = frame["ros_depth_transform_matrix"]

            out["depth_transform_matrix"] = depth_pose["transform_matrix"]
            out["ros_depth_transform_matrix"] = depth_pose["ros_transform_matrix"]
            out["depth_tf_timestamp_sec"] = float(depth_stamp_sec)
            out["depth_tf_dt_sec"] = 0.0
            out["depth_tf_lookup_mode"] = "exact_tf_lookup"

        return out, None

    def process_once(self) -> bool:
        run_dir = self._switch_run_if_needed()
        if run_dir is None:
            return False

        source_path = run_dir / self.args.source_name
        if not source_path.exists():
            return False

        data = read_json_with_retry(source_path)
        if not data:
            return False
        metadata = self._load_metadata(run_dir)

        frames = data.get("frames", [])
        rewritten_frames = []
        failures = []
        resolved = 0
        for idx, frame in enumerate(frames):
            rewritten, error = self._rewrite_frame(frame, metadata)
            rewritten_frames.append(rewritten)
            if error is None:
                resolved += 1
            else:
                failures.append(
                    {
                        "index": idx,
                        "file_path": frame.get("file_path"),
                        "seq": frame.get("seq", frame.get("rgb_seq")),
                        "error": error,
                    }
                )

        output = dict(data)
        output["frames"] = rewritten_frames
        output["mask_style_pose_patch"] = {
            "generated_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_transforms": self.args.source_name,
            "base_frame_override": sanitize_frame(self.args.base_frame),
            "camera_frame_override": sanitize_frame(self.args.camera_frame),
            "camera_profile_fallback": self.args.camera_profile,
            "resolved_frames": resolved,
            "failed_frames": len(failures),
            "preserve_source_fields": bool(self.args.preserve_source_fields),
            "skip_depth": bool(self.args.skip_depth),
            "pose_source": "mask_style_exact_tf",
        }

        output_path = run_dir / self.args.output_name
        write_json_atomic(output_path, output, indent=2)

        report = {
            "generated_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir),
            "source_transforms_path": str(source_path),
            "output_transforms_path": str(output_path),
            "frame_count": len(frames),
            "resolved_frames": resolved,
            "failed_frames": len(failures),
            "failures": failures[: self.args.max_report_failures],
        }
        write_json_atomic(run_dir / self.args.report_name, report, indent=2)

        if failures:
            rospy.logwarn_throttle(
                2.0,
                "mask-style pose patch resolved %d/%d frame(s) for %s",
                resolved,
                len(frames),
                run_dir.name,
            )
        else:
            rospy.loginfo(
                "mask-style pose patch resolved %d/%d frame(s) for %s",
                resolved,
                len(frames),
                run_dir.name,
            )
        return True

    def spin(self) -> None:
        rate = rospy.Rate(self.args.loop_rate)
        while not rospy.is_shutdown():
            self.process_once()
            if self.args.once:
                return
            rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a transforms sidecar whose poses come from exact tf2 lookups at "
            "the saved image timestamps, matching the mask script's timing semantics."
        )
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-name", default="transforms.json")
    parser.add_argument("--output-name", default="transforms_mask_pose.json")
    parser.add_argument("--report-name", default="mask_pose_report.json")
    parser.add_argument("--base-frame", default="")
    parser.add_argument("--camera-frame", default="")
    parser.add_argument(
        "--camera-profile",
        choices=sorted(CAMERA_PROFILES.keys()),
        default="manual",
        help="Fallback camera-frame preset if neither metadata nor --camera-frame provides one.",
    )
    parser.add_argument("--link-frame-prefix", default="dynaarm_arm_tf")
    parser.add_argument("--tf-timeout", type=float, default=0.1)
    parser.add_argument("--tf-cache-sec", type=float, default=120.0)
    parser.add_argument("--loop-rate", type=float, default=5.0)
    parser.add_argument("--max-report-failures", type=int, default=20)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--once", action="store_true", help="Process the current active run once, then exit.")
    parser.add_argument(
        "--preserve-source-fields",
        dest="preserve_source_fields",
        action="store_true",
        help="Keep the original saved poses in source_* fields inside the sidecar.",
    )
    parser.add_argument(
        "--no-preserve-source-fields",
        dest="preserve_source_fields",
        action="store_false",
        help="Do not copy the original poses into source_* fields.",
    )
    parser.set_defaults(preserve_source_fields=True, wait_for_new_run=True)
    parser.add_argument(
        "--wait-for-new-run",
        dest="wait_for_new_run",
        action="store_true",
        help=(
            "If --dataset-dir points to a dataset root containing timestamped runs, ignore the current latest "
            "run and wait for a newly created run. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-wait-for-new-run",
        dest="wait_for_new_run",
        action="store_false",
        help="Attach immediately to the current latest run instead of waiting.",
    )
    cli_args = rospy.myargv()[1:]
    args = parser.parse_args(cli_args)
    if args.tf_timeout < 0.0:
        parser.error("--tf-timeout must be >= 0")
    if args.tf_cache_sec <= 0.0:
        parser.error("--tf-cache-sec must be > 0")
    if args.loop_rate <= 0.0:
        parser.error("--loop-rate must be > 0")
    return args


def main() -> None:
    rospy.init_node("write_mask_style_poses", anonymous=False)
    args = parse_args()
    writer = MaskStylePoseWriter(args)
    rospy.loginfo("Dataset root: %s", writer.dataset_root)
    rospy.loginfo("Source transforms: %s", args.source_name)
    rospy.loginfo("Output transforms: %s", args.output_name)
    rospy.loginfo("wait_for_new_run=%s", args.wait_for_new_run)
    writer.spin()


if __name__ == "__main__":
    main()
