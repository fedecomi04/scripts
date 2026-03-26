#!/usr/bin/env python3
"""
Capture a merged initialization point cloud directly from a ROS depth camera
PointCloud2 stream.

This bypasses saved depth TIFF backprojection entirely. It uses the point data
already produced by the depth camera plugin, transforms each accepted frame into
the configured world frame, samples points, and writes a merged PLY.
"""

import argparse
import json
import re
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import ros_numpy
import rospy
import tf2_ros
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CompressedImage, Image, PointCloud2


def read_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_pointcloud_topic(depth_topic: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if not depth_topic:
        raise ValueError("Need either --pointcloud-topic or a config camera depth_topic")
    if depth_topic.endswith("/image_raw"):
        return depth_topic[: -len("/image_raw")] + "/points"
    return depth_topic.rsplit("/", 1)[0] + "/points"


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if (dataset_dir / "rgb").exists() or (dataset_dir / "transforms.json").exists():
        return dataset_dir

    runs = [
        p for p in dataset_dir.iterdir()
        if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", p.name)
    ]
    if not runs:
        return dataset_dir
    return max(runs, key=lambda p: p.stat().st_mtime)


def image_msg_to_rgb(msg) -> np.ndarray:
    if isinstance(msg, CompressedImage):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to decode compressed RGB image")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        elif image.ndim == 3:
            image = image[:, :, ::-1].copy()
        return image

    image = ros_numpy.numpify(msg)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported RGB image shape {image.shape}")

    # ROS image buffers in this pipeline are BGR-like; keep consistency with saver.
    return image[:, :, ::-1].copy()


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def pointcloud_struct_to_xyz(cloud_struct: np.ndarray) -> np.ndarray:
    if not all(name in cloud_struct.dtype.names for name in ("x", "y", "z")):
        raise ValueError(f"PointCloud2 missing x/y/z fields: {cloud_struct.dtype.names}")
    return np.stack(
        [
            np.asarray(cloud_struct["x"], dtype=np.float32),
            np.asarray(cloud_struct["y"], dtype=np.float32),
            np.asarray(cloud_struct["z"], dtype=np.float32),
        ],
        axis=-1,
    )


def pointcloud_rgb_from_struct(cloud_struct: np.ndarray) -> Optional[np.ndarray]:
    names = set(cloud_struct.dtype.names or [])
    if {"r", "g", "b"}.issubset(names):
        return np.stack(
            [
                np.asarray(cloud_struct["r"], dtype=np.uint8),
                np.asarray(cloud_struct["g"], dtype=np.uint8),
                np.asarray(cloud_struct["b"], dtype=np.uint8),
            ],
            axis=-1,
        )

    if "rgb" in names:
        split = ros_numpy.point_cloud2.split_rgb_field(cloud_struct.copy())
        return np.stack(
            [
                np.asarray(split["r"], dtype=np.uint8),
                np.asarray(split["g"], dtype=np.uint8),
                np.asarray(split["b"], dtype=np.uint8),
            ],
            axis=-1,
        )

    return None


class DepthPointCloudCapture:
    def __init__(self, args):
        self.args = args
        self.frames_captured = 0
        self.last_capture_t = None
        self.world_xyz = []
        self.world_rgb = []
        self.frame_log = []
        self.shutdown_requested = False

        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_ply = (self.output_dir / args.output_name).resolve()
        self.dataset_root = None
        self.dataset_dir = None
        self.rgb_dir = None
        self.masks_dir = None
        self.transforms_path = None
        self.processed_seqs = set()
        self.pending_by_seq = OrderedDict()
        self.pending_limit = None if args.pending_history <= 0 else max(1, args.pending_history)
        self.last_buffer_wall_time = time.time()

        if args.dataset_dir:
            self.dataset_root = Path(args.dataset_dir).expanduser().resolve()
            self._refresh_dataset_paths()

        self.buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        tf2_ros.TransformListener(self.buffer)

        msg_type = Image if not args.rgb_topic.endswith("compressed") else CompressedImage
        rgb_sub = Subscriber(args.rgb_topic, msg_type)
        points_sub = Subscriber(args.pointcloud_topic, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, points_sub],
            queue_size=args.queue_size,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self.callback)

    def dataset_mode(self) -> bool:
        return self.dataset_dir is not None

    def _reset_accumulated_state(self):
        self.world_xyz = []
        self.world_rgb = []
        self.frame_log = []
        self.frames_captured = 0
        self.last_capture_t = None
        self.shutdown_requested = False
        self.processed_seqs.clear()
        self.pending_by_seq.clear()

    def _refresh_dataset_paths(self):
        if self.dataset_root is None:
            return

        new_dataset_dir = resolve_dataset_dir(self.dataset_root)
        if new_dataset_dir == self.dataset_dir:
            return

        if self.dataset_dir is not None and (
            self.frames_captured > 0 or self.pending_by_seq or self.world_xyz
        ):
            rospy.logwarn(
                "Dataset run changed from %s to %s; resetting buffered and merged point-cloud state",
                self.dataset_dir,
                new_dataset_dir,
            )
            self._reset_accumulated_state()

        self.dataset_dir = new_dataset_dir
        self.rgb_dir = self.dataset_dir / "rgb"
        self.masks_dir = self.dataset_dir / "masks"
        self.transforms_path = self.dataset_dir / "transforms.json"
        self.output_dir = self.dataset_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_ply = (self.output_dir / self.args.output_name).resolve()
        rospy.loginfo("Watching dataset run %s", self.dataset_dir)

    def _rgb_name_for_seq(self, seq: int) -> str:
        return f"{self.args.camera_name}_{seq:05d}.png"

    def _rgb_path_for_seq(self, seq: int) -> Path:
        return self.rgb_dir / self._rgb_name_for_seq(seq)

    def _mask_path_for_seq(self, seq: int) -> Path:
        return self.masks_dir / self._rgb_name_for_seq(seq)

    def _seq_from_camera_filename(self, path: Path) -> Optional[int]:
        match = re.fullmatch(rf"{re.escape(self.args.camera_name)}_(\d+)\.png", path.name)
        if match is None:
            return None
        return int(match.group(1))

    def should_capture(self) -> bool:
        if self.shutdown_requested:
            return False
        if self.frames_captured >= self.args.num_frames:
            return False
        now = rospy.get_rostime().to_sec()
        if self.last_capture_t is not None and (now - self.last_capture_t) < self.args.update_period:
            return False
        self.last_capture_t = now
        return True

    def callback(self, rgb_msg, points_msg: PointCloud2):
        if not self.should_capture():
            return

        try:
            rgb = image_msg_to_rgb(rgb_msg)
        except Exception as exc:
            rospy.logerr("RGB decode failed: %s", exc)
            return

        blur = cv2.Laplacian(rgb, cv2.CV_64F).var()
        if blur < self.args.blur_threshold:
            rospy.logwarn("Skipping frame %d due to blur %.3f < %.3f", self.frames_captured, blur, self.args.blur_threshold)
            return

        try:
            cloud_struct = ros_numpy.numpify(points_msg)
            xyz_cam = pointcloud_struct_to_xyz(cloud_struct)
        except Exception as exc:
            rospy.logerr("PointCloud2 decode failed: %s", exc)
            return

        colors = None
        if xyz_cam.ndim == 3 and rgb.shape[:2] == xyz_cam.shape[:2]:
            colors = rgb
        else:
            colors = pointcloud_rgb_from_struct(cloud_struct)
            if colors is None:
                flat_n = int(np.prod(xyz_cam.shape[:-1]))
                colors = np.full((flat_n, 3), 255, dtype=np.uint8)

        valid = np.isfinite(xyz_cam[..., 0]) & np.isfinite(xyz_cam[..., 1]) & np.isfinite(xyz_cam[..., 2])
        valid &= xyz_cam[..., 2] > 0.0
        if self.args.z_min > 0:
            valid &= xyz_cam[..., 2] >= self.args.z_min
        if self.args.z_max > 0:
            valid &= xyz_cam[..., 2] <= self.args.z_max

        if not np.any(valid):
            rospy.logwarn("Skipping frame %d: no valid points", self.frames_captured)
            return

        uv_valid = None
        if xyz_cam.ndim == 3:
            ys, xs = np.where(valid)
            xyz_valid = xyz_cam[ys, xs].reshape(-1, 3).astype(np.float32)
            rgb_valid = colors[ys, xs].reshape(-1, 3).astype(np.uint8)
            uv_valid = np.stack([ys, xs], axis=1).astype(np.int32)
        else:
            xyz_valid = xyz_cam[valid].reshape(-1, 3).astype(np.float32)
            rgb_valid = colors[valid].reshape(-1, 3).astype(np.uint8)

        if xyz_valid.shape[0] > self.args.points_per_frame:
            choice = np.random.default_rng(self.args.seed + self.frames_captured).choice(
                xyz_valid.shape[0], size=self.args.points_per_frame, replace=False
            )
            xyz_valid = xyz_valid[choice]
            rgb_valid = rgb_valid[choice]
            if uv_valid is not None:
                uv_valid = uv_valid[choice]

        cloud_frame = points_msg.header.frame_id.lstrip("/")
        try:
            tf_msg = self.buffer.lookup_transform(
                self.args.base_frame,
                cloud_frame,
                points_msg.header.stamp,
                rospy.Duration(1.0),
            )
        except Exception as exc:
            rospy.logerr("TF lookup failed for %s -> %s: %s", cloud_frame, self.args.base_frame, exc)
            return

        world_T_cam = ros_numpy.numpify(tf_msg.transform).astype(np.float32)
        hom = np.concatenate(
            [xyz_valid, np.ones((xyz_valid.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        xyz_world = (world_T_cam @ hom.T).T[:, :3]
        record = {
            "rgb_seq": int(rgb_msg.header.seq),
            "points_seq": int(points_msg.header.seq),
            "rgb_stamp_sec": float(rgb_msg.header.stamp.to_sec()),
            "points_stamp_sec": float(points_msg.header.stamp.to_sec()),
            "sync_dt_sec": float(points_msg.header.stamp.to_sec() - rgb_msg.header.stamp.to_sec()),
            "points_before_sampling": int(np.count_nonzero(valid)),
            "points_after_sampling": int(xyz_world.shape[0]),
            "cloud_frame": cloud_frame,
            "blur": float(blur),
            "xyz_world": xyz_world.astype(np.float32),
            "rgb": rgb_valid,
            "uv": uv_valid,
        }

        if self.dataset_mode():
            seq = record["rgb_seq"]
            if seq in self.pending_by_seq:
                del self.pending_by_seq[seq]
            self.pending_by_seq[seq] = record
            self.last_buffer_wall_time = time.time()
            while self.pending_limit is not None and len(self.pending_by_seq) > self.pending_limit:
                dropped_seq, _ = self.pending_by_seq.popitem(last=False)
                rospy.logwarn(
                    "Dropping buffered point cloud for seq %d because pending_history=%d was exceeded",
                    dropped_seq,
                    self.pending_limit,
                )
            rospy.loginfo(
                "Buffered point cloud for seq %d with %d sampled points",
                seq,
                xyz_world.shape[0],
            )
            return

        self._commit_record(record)

    def _commit_record(self, record, mask: Optional[np.ndarray] = None):
        xyz = record["xyz_world"]
        rgb = record["rgb"]

        if mask is not None and record["uv"] is not None:
            uv = record["uv"]
            keep = mask[uv[:, 0], uv[:, 1]] > 0
            xyz = xyz[keep]
            rgb = rgb[keep]

        if xyz.shape[0] == 0:
            rospy.logwarn("Skipping seq %d after masking: no remaining points", record["rgb_seq"])
            self.processed_seqs.add(record["rgb_seq"])
            return

        self.world_xyz.append(xyz)
        self.world_rgb.append(rgb)
        self.frame_log.append(
            {
                "index": self.frames_captured,
                "rgb_seq": record["rgb_seq"],
                "points_seq": record["points_seq"],
                "rgb_stamp_sec": record["rgb_stamp_sec"],
                "points_stamp_sec": record["points_stamp_sec"],
                "sync_dt_sec": record["sync_dt_sec"],
                "points_before_sampling": record["points_before_sampling"],
                "points_after_sampling": int(xyz.shape[0]),
                "cloud_frame": record["cloud_frame"],
                "blur": record["blur"],
            }
        )
        self.processed_seqs.add(record["rgb_seq"])
        self.frames_captured += 1
        rospy.loginfo(
            "Merged frame %d / %d from seq %d with %d points",
            self.frames_captured,
            self.args.num_frames,
            record["rgb_seq"],
            xyz.shape[0],
        )
        if self.frames_captured >= self.args.num_frames:
            self.shutdown_requested = True
            rospy.signal_shutdown("Reached requested frame count")

    def process_once(self):
        if not self.dataset_mode():
            return
        self._refresh_dataset_paths()
        if self.rgb_dir is None or not self.rgb_dir.exists():
            return

        trigger_dir = self.masks_dir if self.args.require_mask else self.rgb_dir
        if trigger_dir is None or not trigger_dir.exists():
            return

        trigger_files = sorted(trigger_dir.glob(f"{self.args.camera_name}_*.png"))
        for trigger_path in trigger_files:
            seq = self._seq_from_camera_filename(trigger_path)
            if seq is None or seq in self.processed_seqs:
                continue

            rgb_path = self._rgb_path_for_seq(seq)
            if not rgb_path.exists():
                continue

            record = self.pending_by_seq.get(seq)
            if record is None:
                continue

            mask = None
            if self.masks_dir is not None:
                mask_path = self._mask_path_for_seq(seq)
                if self.args.require_mask and not mask_path.exists():
                    continue
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is not None and mask.ndim == 3:
                        mask = mask[..., 0]
                    if mask is None:
                        rospy.logwarn("Failed to read mask for seq %d from %s", seq, mask_path)

            self._commit_record(record, mask=mask)
            del self.pending_by_seq[seq]

    def drain_pending(self, timeout_sec: float):
        if not self.dataset_mode() or timeout_sec <= 0 or not self.pending_by_seq:
            return

        deadline = time.time() + timeout_sec
        while time.time() < deadline and self.pending_by_seq:
            before = len(self.pending_by_seq)
            self.process_once()
            after = len(self.pending_by_seq)
            if after < before:
                deadline = time.time() + timeout_sec
                continue
            time.sleep(0.05)

    def dataset_target_reached(self) -> bool:
        if not self.dataset_mode() or self.transforms_path is None or not self.transforms_path.exists():
            return False
        try:
            meta = json.loads(self.transforms_path.read_text())
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to read %s: %s", self.transforms_path, exc)
            return False
        frames = meta.get("frames", [])
        return len(frames) >= self.args.num_frames

    def finalize(self):
        self.drain_pending(self.args.drain_timeout)
        if not self.world_xyz:
            raise RuntimeError("No point-cloud frames were captured")

        xyz = np.concatenate(self.world_xyz, axis=0)
        rgb = np.concatenate(self.world_rgb, axis=0)
        write_ascii_ply(self.output_ply, xyz, rgb)

        transforms_path = self.output_dir / "transforms.json"
        patched_transforms = False
        if transforms_path.exists():
            try:
                meta = json.loads(transforms_path.read_text())
                meta["ply_file_path"] = self.output_ply.name
                transforms_path.write_text(json.dumps(meta, indent=2))
                patched_transforms = True
            except Exception as exc:
                rospy.logwarn("Failed to patch %s with ply_file_path: %s", transforms_path, exc)

        report = {
            "base_frame": self.args.base_frame,
            "rgb_topic": self.args.rgb_topic,
            "pointcloud_topic": self.args.pointcloud_topic,
            "num_frames_captured": self.frames_captured,
            "points_total": int(xyz.shape[0]),
            "points_per_frame_cap": self.args.points_per_frame,
            "z_min": self.args.z_min,
            "z_max": self.args.z_max,
            "output_ply": str(self.output_ply),
            "patched_transforms": patched_transforms,
            "frames": self.frame_log,
        }
        report_path = self.output_dir / "capture_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        return xyz.shape[0], report_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/configs/dynaarm_gs_depth_mask.yaml"),
        help="Sensor config YAML used for the RGB/depth dataset capture",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/pointcloud_captures") / time.strftime("%Y-%m-%d_%H-%M-%S"),
        help="Directory to write the merged initialization point cloud and report",
    )
    ap.add_argument("--output-name", type=str, default="depth_camera_init_points.ply")
    ap.add_argument("--rgb-topic", type=str, default="")
    ap.add_argument("--pointcloud-topic", type=str, default="")
    ap.add_argument("--base-frame", type=str, default="")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--camera-name", type=str, default="arm")
    ap.add_argument("--require-mask", action="store_true")
    ap.add_argument(
        "--pending-history",
        type=int,
        default=0,
        help="Number of buffered seqs waiting for RGB/mask files; 0 keeps all pending seqs",
    )
    ap.add_argument(
        "--drain-timeout",
        type=float,
        default=2.0,
        help="Seconds to keep polling for late mask/RGB files before writing the final merged PLY",
    )
    ap.add_argument(
        "--idle-stop-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait after the dataset reaches its target frame count before auto-finalizing",
    )
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--hz", type=float, default=None)
    ap.add_argument("--blur-threshold", type=float, default=None)
    ap.add_argument("--points-per-frame", type=int, default=2000)
    ap.add_argument("--queue-size", type=int, default=10)
    ap.add_argument("--sync-slop", type=float, default=0.1)
    ap.add_argument("--z-min", type=float, default=0.0, help="Minimum camera-frame z in meters")
    ap.add_argument("--z-max", type=float, default=0.0, help="Maximum camera-frame z in meters, 0 disables max")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    camera = config["cameras"][0]

    args.rgb_topic = args.rgb_topic or camera["image_topic"]
    args.base_frame = args.base_frame or config["base_frame"]
    args.camera_name = args.camera_name or camera.get("name", "arm")
    args.num_frames = int(config["num_images"]) if args.num_frames is None else int(args.num_frames)
    capture_hz = float(config["hz"]) if args.hz is None else float(args.hz)
    args.update_period = 1.0 / capture_hz if capture_hz > 0 else 0.0
    args.blur_threshold = float(config.get("blur_threshold", 0.0)) if args.blur_threshold is None else float(args.blur_threshold)
    args.pointcloud_topic = resolve_pointcloud_topic(camera.get("depth_topic"), args.pointcloud_topic or None)

    rospy.init_node("capture_depth_points_init_cloud", anonymous=True)

    capture = DepthPointCloudCapture(args)

    def handle_shutdown(sig=None, frame=None):
        capture.shutdown_requested = True
        rospy.signal_shutdown("Signal received")

    signal.signal(signal.SIGINT, handle_shutdown)

    rospy.loginfo("Capturing merged init cloud from %s", args.pointcloud_topic)
    while not rospy.is_shutdown() and capture.frames_captured < args.num_frames:
        capture.process_once()
        if capture.dataset_target_reached():
            idle_sec = time.time() - capture.last_buffer_wall_time
            if idle_sec >= args.idle_stop_timeout:
                rospy.loginfo(
                    "Dataset reached %d frames and point-cloud capture has been idle for %.2f s; finalizing",
                    args.num_frames,
                    idle_sec,
                )
                break
        time.sleep(0.1)

    count, report_path = capture.finalize()
    rospy.loginfo("Wrote merged init cloud %s with %d points", capture.output_ply, count)
    rospy.loginfo("Wrote capture report %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
