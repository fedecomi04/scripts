#!/usr/bin/env python3
"""
Interactive TF vs Gazebo camera pose comparison.

What it shows:
1. 3D camera trajectories
2. The six pose components over frame index:
   x, y, z, roll, pitch, yaw
3. A second window with translation / rotation errors

Defaults:
- If no dataset is provided, use the latest dataset under
  data_teleoperation/datasets/dynaarm_gs_depth_mask_01
- If Gazebo poses are already aligned in transform_gazebo.json, reuse them
- Otherwise, align the first Gazebo pose to the first TF pose for plotting

Examples:
  python plot_tf_vs_gazebo_camera_poses.py
  python plot_tf_vs_gazebo_camera_poses.py --dataset-root /path/to/dataset
  python plot_tf_vs_gazebo_camera_poses.py --mode raw
  python plot_tf_vs_gazebo_camera_poses.py --mode align-first-frame
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat")
DEFAULT_DATASET_PARENT = (
    REPO_ROOT / "data_teleoperation" / "datasets" / "dynaarm_gs_depth_mask_01"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TF and Gazebo camera poses interactively."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset directory containing transforms.json and transform_gazebo.json",
    )
    parser.add_argument(
        "--datasets-parent",
        type=Path,
        default=DEFAULT_DATASET_PARENT,
        help="Parent folder used to auto-pick the latest dataset when --dataset-root is omitted",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "raw", "align-first-frame"),
        default="auto",
        help=(
            "Which Gazebo poses to plot: raw report poses, first-frame aligned poses, "
            "or auto-detect aligned report output"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Plot one viewing-direction arrow every N frames in the 3D view",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.05,
        help="Length of camera viewing-direction arrows",
    )
    parser.add_argument(
        "--show-raw-overlay",
        action="store_true",
        help="Overlay raw Gazebo poses even when plotting aligned Gazebo poses",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_dataset(datasets_parent: Path) -> Path:
    candidates = sorted(
        path for path in datasets_parent.iterdir()
        if path.is_dir() and (path / "transforms.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No datasets with transforms.json found in {datasets_parent}")
    return candidates[-1]


def invert_rigid_transform(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def load_pose_series(path: Path) -> Tuple[List[str], np.ndarray]:
    data = load_json(path)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {path}")

    file_paths = []
    poses = []
    for index, frame in enumerate(frames):
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"{path} frame {index} has shape {matrix.shape}, expected (4, 4)"
            )
        file_paths.append(frame.get("file_path", f"frame_{index:06d}"))
        poses.append(matrix)
    return file_paths, np.stack(poses, axis=0)


def load_raw_gazebo_poses_from_diff(path: Path) -> Tuple[List[str], np.ndarray] | None:
    if not path.exists():
        return None

    data = load_json(path)
    differences = data.get("differences", [])
    if not differences:
        return None

    file_paths = []
    poses = []
    for index, item in enumerate(differences):
        matrix = item.get("gazebo_raw_transform_matrix")
        if matrix is None:
            return None
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"{path} difference {index} has raw Gazebo shape {matrix.shape}, expected (4, 4)"
            )
        file_paths.append(item.get("file_path", f"frame_{index:06d}"))
        poses.append(matrix)

    return file_paths, np.stack(poses, axis=0)


def compute_first_frame_alignment(
    tf_poses: np.ndarray,
    gazebo_poses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    alignment = tf_poses[0] @ invert_rigid_transform(gazebo_poses[0])
    aligned = np.einsum("ij,njk->nik", alignment, gazebo_poses)
    return aligned, alignment


def extract_centers_and_dirs(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = poses[:, :3, 3]
    forward_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    dirs = np.einsum("nij,j->ni", poses[:, :3, :3], forward_cam)
    return centers, dirs


def unwrap_deg(values_deg: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values_deg), axis=0))


def rotation_matrix_to_rpy_zyx_deg(R: np.ndarray) -> np.ndarray:
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def pose_series_to_rpy_deg(poses: np.ndarray) -> np.ndarray:
    angles = np.stack(
        [rotation_matrix_to_rpy_zyx_deg(T[:3, :3]) for T in poses],
        axis=0,
    )
    return unwrap_deg(angles)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    trace_value = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_value)))


def compute_errors(
    tf_poses: np.ndarray,
    gazebo_poses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_xyz = tf_poses[:, :3, 3] - gazebo_poses[:, :3, 3]
    translation_norm = np.linalg.norm(delta_xyz, axis=1)
    rotation_deg = np.asarray(
        [
            rotation_angle_deg(gazebo_pose[:3, :3], tf_pose[:3, :3])
            for tf_pose, gazebo_pose in zip(tf_poses, gazebo_poses)
        ],
        dtype=np.float64,
    )
    return delta_xyz, translation_norm, rotation_deg


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def summarize_series(
    name: str,
    centers: np.ndarray,
    translation_norm: np.ndarray | None = None,
    rotation_deg: np.ndarray | None = None,
) -> None:
    print(name)
    print(f"  x range: [{centers[:, 0].min():.6f}, {centers[:, 0].max():.6f}]")
    print(f"  y range: [{centers[:, 1].min():.6f}, {centers[:, 1].max():.6f}]")
    print(f"  z range: [{centers[:, 2].min():.6f}, {centers[:, 2].max():.6f}]")
    if translation_norm is not None:
        print(f"  translation error mean:   {translation_norm.mean():.6f}")
        print(f"  translation error median: {np.median(translation_norm):.6f}")
        print(f"  translation error max:    {translation_norm.max():.6f}")
    if rotation_deg is not None:
        print(f"  rotation error mean:      {rotation_deg.mean():.6f} deg")
        print(f"  rotation error median:    {np.median(rotation_deg):.6f} deg")
        print(f"  rotation error max:       {rotation_deg.max():.6f} deg")
    print()


def build_plot_data(dataset_root: Path, mode: str) -> Dict[str, object]:
    transforms_path = dataset_root / "transforms.json"
    gazebo_path = dataset_root / "transform_gazebo.json"
    diff_path = dataset_root / "pose_difference_gazebo_vs_tf.json"

    tf_file_paths, tf_poses = load_pose_series(transforms_path)
    gz_file_paths, gz_poses_report = load_pose_series(gazebo_path)

    if tf_file_paths != gz_file_paths:
        raise ValueError("transforms.json and transform_gazebo.json frame orders do not match")

    raw_from_diff = load_raw_gazebo_poses_from_diff(diff_path)
    gz_poses_raw = raw_from_diff[1] if raw_from_diff is not None else gz_poses_report
    if raw_from_diff is not None and raw_from_diff[0] != tf_file_paths:
        raise ValueError("pose_difference_gazebo_vs_tf.json frame order does not match transforms.json")

    gazebo_report = load_json(gazebo_path)
    report_alignment = gazebo_report.get("gazebo_world_alignment")
    report_is_aligned = bool(report_alignment and report_alignment.get("available"))

    if mode == "raw":
        gz_poses_selected = gz_poses_raw
        selected_label = "Gazebo raw"
        alignment_mode = "raw"
    elif mode == "align-first-frame":
        gz_poses_selected, alignment_matrix = compute_first_frame_alignment(tf_poses, gz_poses_raw)
        selected_label = "Gazebo aligned (first frame)"
        alignment_mode = "first-frame"
        report_alignment = {
            "available": True,
            "mode": "first_saved_frame_tf_alignment",
            "alignment_transform_matrix": alignment_matrix.tolist(),
        }
    else:
        if report_is_aligned:
            gz_poses_selected = gz_poses_report
            selected_label = "Gazebo aligned (report)"
            alignment_mode = "report"
        else:
            gz_poses_selected, alignment_matrix = compute_first_frame_alignment(tf_poses, gz_poses_raw)
            selected_label = "Gazebo aligned (first frame)"
            alignment_mode = "first-frame"
            report_alignment = {
                "available": True,
                "mode": "first_saved_frame_tf_alignment",
                "alignment_transform_matrix": alignment_matrix.tolist(),
            }

    tf_centers, tf_dirs = extract_centers_and_dirs(tf_poses)
    gz_centers, gz_dirs = extract_centers_and_dirs(gz_poses_selected)
    raw_centers, raw_dirs = extract_centers_and_dirs(gz_poses_raw)

    tf_rpy = pose_series_to_rpy_deg(tf_poses)
    gz_rpy = pose_series_to_rpy_deg(gz_poses_selected)
    raw_rpy = pose_series_to_rpy_deg(gz_poses_raw)

    delta_xyz, translation_norm, rotation_deg = compute_errors(tf_poses, gz_poses_selected)
    raw_delta_xyz, raw_translation_norm, raw_rotation_deg = compute_errors(tf_poses, gz_poses_raw)

    return {
        "dataset_root": dataset_root,
        "file_paths": tf_file_paths,
        "tf_poses": tf_poses,
        "tf_centers": tf_centers,
        "tf_dirs": tf_dirs,
        "tf_rpy": tf_rpy,
        "gz_poses_selected": gz_poses_selected,
        "gz_centers": gz_centers,
        "gz_dirs": gz_dirs,
        "gz_rpy": gz_rpy,
        "gz_poses_raw": gz_poses_raw,
        "raw_centers": raw_centers,
        "raw_dirs": raw_dirs,
        "raw_rpy": raw_rpy,
        "delta_xyz": delta_xyz,
        "translation_norm": translation_norm,
        "rotation_deg": rotation_deg,
        "raw_delta_xyz": raw_delta_xyz,
        "raw_translation_norm": raw_translation_norm,
        "raw_rotation_deg": raw_rotation_deg,
        "selected_label": selected_label,
        "alignment_mode": alignment_mode,
        "report_alignment": report_alignment,
    }


def plot_pose_comparison(plot_data: Dict[str, object], stride: int, arrow_scale: float, show_raw_overlay: bool) -> None:
    tf_centers = plot_data["tf_centers"]
    gz_centers = plot_data["gz_centers"]
    raw_centers = plot_data["raw_centers"]
    tf_dirs = plot_data["tf_dirs"]
    gz_dirs = plot_data["gz_dirs"]
    raw_dirs = plot_data["raw_dirs"]
    tf_rpy = plot_data["tf_rpy"]
    gz_rpy = plot_data["gz_rpy"]
    raw_rpy = plot_data["raw_rpy"]
    selected_label = plot_data["selected_label"]
    file_paths = plot_data["file_paths"]

    frame_indices = np.arange(len(file_paths))
    arrow_indices = np.arange(0, len(file_paths), max(1, stride))
    component_labels = ("x", "y", "z")
    angle_labels = ("roll", "pitch", "yaw")

    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(4, 2)

    ax_3d = fig.add_subplot(grid[0, :], projection="3d")
    ax_3d.plot(tf_centers[:, 0], tf_centers[:, 1], tf_centers[:, 2], label="TF", linewidth=2)
    ax_3d.plot(
        gz_centers[:, 0],
        gz_centers[:, 1],
        gz_centers[:, 2],
        label=selected_label,
        linewidth=2,
    )
    if show_raw_overlay and selected_label != "Gazebo raw":
        ax_3d.plot(
            raw_centers[:, 0],
            raw_centers[:, 1],
            raw_centers[:, 2],
            label="Gazebo raw",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    ax_3d.scatter(*tf_centers[0], s=60, marker="o")
    ax_3d.scatter(*tf_centers[-1], s=60, marker="^")
    ax_3d.scatter(*gz_centers[0], s=60, marker="o")
    ax_3d.scatter(*gz_centers[-1], s=60, marker="^")

    tf_arrow_centers = tf_centers[arrow_indices]
    gz_arrow_centers = gz_centers[arrow_indices]
    tf_arrow_dirs = tf_dirs[arrow_indices]
    gz_arrow_dirs = gz_dirs[arrow_indices]
    ax_3d.quiver(
        tf_arrow_centers[:, 0],
        tf_arrow_centers[:, 1],
        tf_arrow_centers[:, 2],
        tf_arrow_dirs[:, 0],
        tf_arrow_dirs[:, 1],
        tf_arrow_dirs[:, 2],
        length=arrow_scale,
        normalize=True,
        alpha=0.6,
    )
    ax_3d.quiver(
        gz_arrow_centers[:, 0],
        gz_arrow_centers[:, 1],
        gz_arrow_centers[:, 2],
        gz_arrow_dirs[:, 0],
        gz_arrow_dirs[:, 1],
        gz_arrow_dirs[:, 2],
        length=arrow_scale,
        normalize=True,
        alpha=0.6,
    )
    ax_3d.set_title("3D camera trajectory")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.legend()
    set_axes_equal(ax_3d, np.vstack([tf_centers, gz_centers]))

    axes_left = [fig.add_subplot(grid[i, 0]) for i in range(1, 4)]
    for axis_index, ax in enumerate(axes_left):
        ax.plot(frame_indices, tf_centers[:, axis_index], label="TF", linewidth=2)
        ax.plot(frame_indices, gz_centers[:, axis_index], label=selected_label, linewidth=2)
        if show_raw_overlay and selected_label != "Gazebo raw":
            ax.plot(
                frame_indices,
                raw_centers[:, axis_index],
                label="Gazebo raw",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )
        ax.set_title(f"{component_labels[axis_index]} position")
        ax.set_xlabel("frame index")
        ax.set_ylabel(component_labels[axis_index])
        ax.grid(True, alpha=0.3)
        if axis_index == 0:
            ax.legend()

    axes_right = [fig.add_subplot(grid[i, 1]) for i in range(1, 4)]
    for axis_index, ax in enumerate(axes_right):
        ax.plot(frame_indices, tf_rpy[:, axis_index], label="TF", linewidth=2)
        ax.plot(frame_indices, gz_rpy[:, axis_index], label=selected_label, linewidth=2)
        if show_raw_overlay and selected_label != "Gazebo raw":
            ax.plot(
                frame_indices,
                raw_rpy[:, axis_index],
                label="Gazebo raw",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )
        ax.set_title(f"{angle_labels[axis_index]} (ZYX deg)")
        ax.set_xlabel("frame index")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        if axis_index == 0:
            ax.legend()

    fig.suptitle(
        f"TF vs Gazebo camera poses\n{plot_data['dataset_root']}  [{selected_label}]",
        fontsize=14,
    )

    err_fig, err_axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    delta_xyz = plot_data["delta_xyz"]
    raw_delta_xyz = plot_data["raw_delta_xyz"]
    translation_norm = plot_data["translation_norm"]
    raw_translation_norm = plot_data["raw_translation_norm"]
    rotation_deg = plot_data["rotation_deg"]
    raw_rotation_deg = plot_data["raw_rotation_deg"]

    for axis_index, label in enumerate(component_labels):
        err_axes[0, 0].plot(frame_indices, delta_xyz[:, axis_index], label=label)
    err_axes[0, 0].set_title("Selected Gazebo translation delta (TF - Gazebo)")
    err_axes[0, 0].set_xlabel("frame index")
    err_axes[0, 0].set_ylabel("delta")
    err_axes[0, 0].grid(True, alpha=0.3)
    err_axes[0, 0].legend()

    err_axes[0, 1].plot(frame_indices, translation_norm, label=selected_label, linewidth=2)
    if show_raw_overlay and selected_label != "Gazebo raw":
        err_axes[0, 1].plot(
            frame_indices,
            raw_translation_norm,
            label="Gazebo raw",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    err_axes[0, 1].set_title("Translation error norm")
    err_axes[0, 1].set_xlabel("frame index")
    err_axes[0, 1].set_ylabel("norm")
    err_axes[0, 1].grid(True, alpha=0.3)
    err_axes[0, 1].legend()

    if show_raw_overlay and selected_label != "Gazebo raw":
        for axis_index, label in enumerate(component_labels):
            err_axes[1, 0].plot(
                frame_indices,
                raw_delta_xyz[:, axis_index],
                label=label,
                linestyle="--",
                alpha=0.8,
            )
        err_axes[1, 0].set_title("Raw Gazebo translation delta (TF - Gazebo)")
    else:
        for axis_index, label in enumerate(component_labels):
            err_axes[1, 0].plot(frame_indices, delta_xyz[:, axis_index], label=label)
        err_axes[1, 0].set_title("Selected Gazebo translation delta copy")
    err_axes[1, 0].set_xlabel("frame index")
    err_axes[1, 0].set_ylabel("delta")
    err_axes[1, 0].grid(True, alpha=0.3)
    err_axes[1, 0].legend()

    err_axes[1, 1].plot(frame_indices, rotation_deg, label=selected_label, linewidth=2)
    if show_raw_overlay and selected_label != "Gazebo raw":
        err_axes[1, 1].plot(
            frame_indices,
            raw_rotation_deg,
            label="Gazebo raw",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    err_axes[1, 1].set_title("Rotation error")
    err_axes[1, 1].set_xlabel("frame index")
    err_axes[1, 1].set_ylabel("deg")
    err_axes[1, 1].grid(True, alpha=0.3)
    err_axes[1, 1].legend()

    err_fig.suptitle("TF vs Gazebo pose errors", fontsize=14)
    plt.show()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or find_latest_dataset(args.datasets_parent)
    plot_data = build_plot_data(dataset_root=dataset_root, mode=args.mode)

    print(f"Dataset: {dataset_root}")
    print(f"Frames: {len(plot_data['file_paths'])}")
    print(f"Plot mode: {plot_data['alignment_mode']}")
    if plot_data["report_alignment"] is not None:
        print("Gazebo world alignment metadata:")
        print(json.dumps(plot_data["report_alignment"], indent=2))
        print()

    summarize_series("TF poses", plot_data["tf_centers"])
    summarize_series(
        plot_data["selected_label"],
        plot_data["gz_centers"],
        plot_data["translation_norm"],
        plot_data["rotation_deg"],
    )
    if args.show_raw_overlay and plot_data["selected_label"] != "Gazebo raw":
        summarize_series(
            "Gazebo raw",
            plot_data["raw_centers"],
            plot_data["raw_translation_norm"],
            plot_data["raw_rotation_deg"],
        )

    plot_pose_comparison(
        plot_data=plot_data,
        stride=args.stride,
        arrow_scale=args.arrow_scale,
        show_raw_overlay=args.show_raw_overlay,
    )


if __name__ == "__main__":
    main()
