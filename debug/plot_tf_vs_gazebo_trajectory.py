#!/usr/bin/env python3
"""
Interactive 3D trajectory comparison between TF camera poses and Gazebo camera poses.

What it shows:
- TF camera centers
- Gazebo camera centers
- Optional raw Gazebo trajectory overlay

Usage:
  conda run -n radiance_ros python plot_tf_vs_gazebo_trajectory.py

  conda run -n radiance_ros python plot_tf_vs_gazebo_trajectory.py \
    --dataset-root /path/to/dataset \
    --mode align-first-frame \
    --show-raw-overlay
"""

from __future__ import annotations

import argparse
import json
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
        description="Interactive 3D trajectory comparison for TF vs Gazebo camera poses."
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
            "Which Gazebo trajectory to plot: raw report poses, first-frame aligned poses, "
            "or auto-detect aligned report output"
        ),
    )
    parser.add_argument(
        "--show-raw-overlay",
        action="store_true",
        help="Overlay raw Gazebo trajectory even when plotting aligned Gazebo poses",
    )
    parser.add_argument(
        "--marker-every",
        type=int,
        default=5,
        help="Place a point marker every N frames along each trajectory",
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


def extract_centers(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3, 3]


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def build_plot_data(dataset_root: Path, mode: str) -> Dict[str, object]:
    transforms_path = dataset_root / "transforms.json"
    gazebo_path = dataset_root / "transform_gazebo.json"
    diff_path = dataset_root / "pose_difference_gazebo_vs_tf.json"

    file_paths, tf_poses = load_pose_series(transforms_path)
    gz_file_paths, gz_poses_report = load_pose_series(gazebo_path)
    if gz_file_paths != file_paths:
        raise ValueError("transforms.json and transform_gazebo.json frame orders do not match")

    raw_from_diff = load_raw_gazebo_poses_from_diff(diff_path)
    gz_poses_raw = raw_from_diff[1] if raw_from_diff is not None else gz_poses_report
    if raw_from_diff is not None and raw_from_diff[0] != file_paths:
        raise ValueError("pose_difference_gazebo_vs_tf.json frame order does not match transforms.json")

    gazebo_report = load_json(gazebo_path)
    report_alignment = gazebo_report.get("gazebo_world_alignment")
    report_is_aligned = bool(report_alignment and report_alignment.get("available"))

    if mode == "raw":
        gz_poses_selected = gz_poses_raw
        selected_label = "Gazebo raw"
        alignment_mode = "raw"
        alignment_metadata = None
    elif mode == "align-first-frame":
        gz_poses_selected, alignment_matrix = compute_first_frame_alignment(tf_poses, gz_poses_raw)
        selected_label = "Gazebo aligned (first frame)"
        alignment_mode = "first-frame"
        alignment_metadata = {
            "available": True,
            "mode": "first_saved_frame_tf_alignment",
            "alignment_transform_matrix": alignment_matrix.tolist(),
        }
    else:
        if report_is_aligned:
            gz_poses_selected = gz_poses_report
            selected_label = "Gazebo aligned (report)"
            alignment_mode = "report"
            alignment_metadata = report_alignment
        else:
            gz_poses_selected, alignment_matrix = compute_first_frame_alignment(tf_poses, gz_poses_raw)
            selected_label = "Gazebo aligned (first frame)"
            alignment_mode = "first-frame"
            alignment_metadata = {
                "available": True,
                "mode": "first_saved_frame_tf_alignment",
                "alignment_transform_matrix": alignment_matrix.tolist(),
            }

    tf_centers = extract_centers(tf_poses)
    gz_centers = extract_centers(gz_poses_selected)
    raw_centers = extract_centers(gz_poses_raw)
    delta = tf_centers - gz_centers
    delta_norm = np.linalg.norm(delta, axis=1)

    return {
        "dataset_root": dataset_root,
        "file_paths": file_paths,
        "tf_centers": tf_centers,
        "gz_centers": gz_centers,
        "raw_centers": raw_centers,
        "delta_norm": delta_norm,
        "selected_label": selected_label,
        "alignment_mode": alignment_mode,
        "alignment_metadata": alignment_metadata,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or find_latest_dataset(args.datasets_parent)
    plot_data = build_plot_data(dataset_root=dataset_root, mode=args.mode)

    tf_centers = plot_data["tf_centers"]
    gz_centers = plot_data["gz_centers"]
    raw_centers = plot_data["raw_centers"]
    selected_label = plot_data["selected_label"]
    file_paths = plot_data["file_paths"]
    marker_every = max(1, args.marker_every)
    marker_indices = np.arange(0, len(file_paths), marker_every)

    print(f"Dataset: {dataset_root}")
    print(f"Frames: {len(file_paths)}")
    print(f"Plot mode: {plot_data['alignment_mode']}")
    if plot_data["alignment_metadata"] is not None:
        print("Gazebo alignment metadata:")
        print(json.dumps(plot_data["alignment_metadata"], indent=2))
    print(f"Mean center distance: {plot_data['delta_norm'].mean():.6f}")
    print(f"Median center distance: {np.median(plot_data['delta_norm']):.6f}")
    print(f"Max center distance: {plot_data['delta_norm'].max():.6f}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        tf_centers[:, 0],
        tf_centers[:, 1],
        tf_centers[:, 2],
        label="TF / sim",
        linewidth=2.5,
        color="tab:blue",
    )
    ax.plot(
        gz_centers[:, 0],
        gz_centers[:, 1],
        gz_centers[:, 2],
        label=selected_label,
        linewidth=2.5,
        color="tab:orange",
    )

    if args.show_raw_overlay and selected_label != "Gazebo raw":
        ax.plot(
            raw_centers[:, 0],
            raw_centers[:, 1],
            raw_centers[:, 2],
            label="Gazebo raw",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            color="tab:green",
        )

    ax.scatter(
        tf_centers[0, 0],
        tf_centers[0, 1],
        tf_centers[0, 2],
        s=70,
        marker="o",
        color="tab:blue",
    )
    ax.scatter(
        tf_centers[-1, 0],
        tf_centers[-1, 1],
        tf_centers[-1, 2],
        s=90,
        marker="^",
        color="tab:blue",
    )
    ax.scatter(
        gz_centers[0, 0],
        gz_centers[0, 1],
        gz_centers[0, 2],
        s=70,
        marker="o",
        color="tab:orange",
    )
    ax.scatter(
        gz_centers[-1, 0],
        gz_centers[-1, 1],
        gz_centers[-1, 2],
        s=90,
        marker="^",
        color="tab:orange",
    )

    ax.scatter(
        tf_centers[marker_indices, 0],
        tf_centers[marker_indices, 1],
        tf_centers[marker_indices, 2],
        s=18,
        color="tab:blue",
        alpha=0.8,
    )
    ax.scatter(
        gz_centers[marker_indices, 0],
        gz_centers[marker_indices, 1],
        gz_centers[marker_indices, 2],
        s=18,
        color="tab:orange",
        alpha=0.8,
    )

    all_points = np.vstack([tf_centers, gz_centers])
    if args.show_raw_overlay and selected_label != "Gazebo raw":
        all_points = np.vstack([all_points, raw_centers])
    set_axes_equal(ax, all_points)

    ax.set_title(f"Camera trajectory comparison\n{dataset_root.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
