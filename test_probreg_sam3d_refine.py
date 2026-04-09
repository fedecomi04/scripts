#!/usr/bin/env python3
"""Standalone probreg refinement test on saved SAM3D registration clouds.

This script does not touch dynamic-gs training code. It reuses the saved
arm_05460 fusion artifacts, starts from the saved FGR transform, and runs
probreg rigid CPD with scale update enabled as an offline refinement test.

Outputs are saved next to the existing SAM3D debug artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from probreg import cpd


DEBUG_DIR = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_masks_esam"
)
STEM = "arm_05460_sam3d"


def _read_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        values[key.strip()] = value.strip()
    return values


def _load_points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points, dtype=np.float32)


def _save_points(path: Path, points: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homog @ transform.T)[:, :3].astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    down = cloud.voxel_down_sample(float(voxel_size))
    down_points = np.asarray(down.points, dtype=np.float32)
    return down_points if len(down_points) > 0 else points


def _nn_stats(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    distances = np.asarray(source_cloud.compute_point_cloud_distance(target_cloud), dtype=np.float32)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.median(finite))


def _sample_rows(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    keep_positions = np.linspace(0, len(points) - 1, num=max_points)
    keep_indices = np.unique(np.round(keep_positions).astype(np.int64))
    return points[keep_indices]


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _make_transform_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (float(scale) * rotation).astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform


def main() -> None:
    log_path = DEBUG_DIR / f"{STEM}_fusion_log.txt"
    values = _read_log(log_path)

    source_ref_path = Path(values["source_reg_ref_ply"])
    target_ref_path = Path(values["target_reg_ref_ply"])
    fgr_transform = np.asarray(ast.literal_eval(values["fgr_transformation"]), dtype=np.float32)
    voxel_size = float(values["voxel_size"])

    source_ref = _load_points(source_ref_path)
    target_ref = _load_points(target_ref_path)
    source_init = _transform_points(source_ref, fgr_transform)

    probreg_voxel = max(2.0 * voxel_size, 1e-3)
    source_probreg = _voxel_downsample(source_init, probreg_voxel)
    target_probreg = _voxel_downsample(target_ref, probreg_voxel)

    result = cpd.registration_cpd(
        source_probreg,
        target_probreg,
        tf_type_name="rigid",
        update_scale=True,
        maxiter=80,
        tol=1e-6,
        w=0.0,
    )

    probreg_tf = result.transformation
    refined_down = probreg_tf.transform(source_probreg)
    refined_full = probreg_tf.transform(source_init)

    total_transform = _make_transform_matrix(
        probreg_tf.scale,
        probreg_tf.rot,
        probreg_tf.t,
    ) @ fgr_transform

    before_mean, before_median = _nn_stats(source_init, target_ref)
    after_mean, after_median = _nn_stats(refined_full, target_ref)

    refined_path = DEBUG_DIR / f"{STEM}_probreg_refined_source_reg.ply"
    refined_down_path = DEBUG_DIR / f"{STEM}_probreg_refined_source_reg_downsampled.ply"
    plot_path = DEBUG_DIR / f"{STEM}_probreg_comparison.png"
    info_path = DEBUG_DIR / f"{STEM}_probreg_run_info.txt"

    _save_points(refined_path, refined_full)
    _save_points(refined_down_path, refined_down)

    source_plot = _sample_rows(source_init, 6000)
    refined_plot = _sample_rows(refined_full, 6000)
    target_plot = _sample_rows(target_ref, 6000)
    all_points = np.concatenate([source_plot, refined_plot, target_plot], axis=0)

    fig = plt.figure(figsize=(16, 8))
    ax0 = fig.add_subplot(121, projection="3d")
    ax0.scatter(target_plot[:, 0], target_plot[:, 1], target_plot[:, 2], s=2.0, c="royalblue", alpha=0.7, label="target")
    ax0.scatter(source_plot[:, 0], source_plot[:, 1], source_plot[:, 2], s=1.0, c="crimson", alpha=0.45, label="source after FGR")
    _set_equal_axes(ax0, all_points)
    ax0.set_title(f"Before probreg\nmean={before_mean:.6f}, median={before_median:.6f}")
    ax0.legend(loc="upper right")

    ax1 = fig.add_subplot(122, projection="3d")
    ax1.scatter(target_plot[:, 0], target_plot[:, 1], target_plot[:, 2], s=2.0, c="royalblue", alpha=0.7, label="target")
    ax1.scatter(refined_plot[:, 0], refined_plot[:, 1], refined_plot[:, 2], s=1.0, c="darkorange", alpha=0.45, label="source after probreg")
    _set_equal_axes(ax1, all_points)
    ax1.set_title(
        f"After probreg\nmean={after_mean:.6f}, median={after_median:.6f}, "
        f"scale={float(probreg_tf.scale):.6f}"
    )
    ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    info_path.write_text(
        "\n".join(
            [
                f"log_path: {log_path}",
                f"source_ref_path: {source_ref_path}",
                f"target_ref_path: {target_ref_path}",
                f"source_ref_count: {len(source_ref)}",
                f"target_ref_count: {len(target_ref)}",
                f"probreg_voxel: {probreg_voxel}",
                f"source_probreg_count: {len(source_probreg)}",
                f"target_probreg_count: {len(target_probreg)}",
                f"initial_fgr_transform: {fgr_transform.tolist()}",
                f"probreg_scale: {float(probreg_tf.scale)}",
                f"probreg_rotation: {np.asarray(probreg_tf.rot).tolist()}",
                f"probreg_translation: {np.asarray(probreg_tf.t).tolist()}",
                f"total_transform_from_source_ref: {total_transform.tolist()}",
                f"before_mean_nn: {before_mean}",
                f"before_median_nn: {before_median}",
                f"after_mean_nn: {after_mean}",
                f"after_median_nn: {after_median}",
                f"sigma2: {float(result.sigma2)}",
                f"q: {float(result.q)}",
                f"refined_path: {refined_path}",
                f"refined_down_path: {refined_down_path}",
                f"plot_path: {plot_path}",
            ]
        )
        + "\n"
    )

    print(refined_path)
    print(plot_path)
    print(info_path)


if __name__ == "__main__":
    main()
