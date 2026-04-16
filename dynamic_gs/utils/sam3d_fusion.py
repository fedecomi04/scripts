from __future__ import annotations
"""Geometry-only SAM3D object fusion for dynamic-gs.

The active fusion path is:
1. load the raw SAM3D point cloud
2. estimate an isotropic scale from source/target extents
3. translate the scaled SAM3D source to the target centroid
4. voxel-downsample both clouds
5. refine the pose with probreg CPD similarity (scale + rigid)
6. append only non-overlapping SAM3D points back into the Gaussian scene

The final insertion still uses append-with-dedup only. Existing scene/object
Gaussians are kept. This file intentionally uses geometry-first registration with
RGB used only for the CPD refinement and the final Gaussian insertion.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from nerfstudio.utils.spherical_harmonics import SH2RGB

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - training env dependency
    o3d = None

try:
    from plyfile import PlyData
except ImportError:  # pragma: no cover - training env dependency
    PlyData = None

try:
    from probreg import cpd
except ImportError:  # pragma: no cover - training env dependency
    cpd = None


@dataclass
class Sam3DInsertionResult:
    aligned_points: np.ndarray
    aligned_colors: np.ndarray
    kept_points: np.ndarray
    kept_colors: np.ndarray
    chosen_scale: float
    dedup_threshold: float
    voxel_size: float
    source_point_count: int
    target_point_count: int
    visible_source_point_count: int
    registration_source_point_count: int
    kept_point_count: int
    similarity_transform: np.ndarray
    similarity_correspondence_count: int
    similarity_scale: float
    correspondence_threshold: float
    correspondence_plot_path: str


def _require_open3d():
    if o3d is None:
        raise ImportError("Open3D is required for SAM3D alignment and fusion.")
    return o3d


def _require_plyfile():
    if PlyData is None:
        raise ImportError("plyfile is required to read SAM3D gaussian outputs.")
    return PlyData


def load_sam3d_gaussian_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = _require_plyfile().read(str(ply_path))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if all(name in vertex.dtype.names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        rgb = SH2RGB(torch.from_numpy(features_dc)).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    return xyz, rgb


def _to_pcd(points: np.ndarray, colors: np.ndarray | None = None):
    o3d_mod = _require_open3d()
    pcd = o3d_mod.geometry.PointCloud()
    pcd.points = o3d_mod.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d_mod.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    return pcd


def _ensure_rgb_colors(colors: np.ndarray, point_count: int) -> np.ndarray:
    if colors is None or len(colors) != point_count:
        return np.full((point_count, 3), 0.5, dtype=np.float32)
    colors_np = np.asarray(colors, dtype=np.float32)
    if colors_np.ndim != 2 or colors_np.shape[1] != 3:
        return np.full((point_count, 3), 0.5, dtype=np.float32)

    # Registration backends expect RGB in [0, 1]. If values are outside this
    # range, treat input as SH-DC coefficients and decode to RGB.
    if float(np.min(colors_np)) < 0.0 or float(np.max(colors_np)) > 1.0:
        return SH2RGB(torch.from_numpy(colors_np)).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return np.clip(colors_np, 0.0, 1.0).astype(np.float32)


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 0\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        return
    _require_open3d().io.write_point_cloud(str(path), _to_pcd(points, colors))


def _sample_rows_for_plot(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    keep_positions = np.linspace(0, len(points) - 1, num=max_points)
    keep_indices = np.unique(np.round(keep_positions).astype(np.int64))
    return points[keep_indices]


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _save_correspondence_plot(
    debug_dir: Path,
    output_stem: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    correspondences,
    threshold: float,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plot_source = _sample_rows_for_plot(source_points, max_points=5000)
    plot_target = _sample_rows_for_plot(target_points, max_points=5000)
    correspondence_array = np.asarray(correspondences, dtype=np.int32)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(
        plot_target[:, 0],
        plot_target[:, 1],
        plot_target[:, 2],
        s=2.0,
        c="royalblue",
        alpha=0.70,
        label=f"target splat object ({len(target_points)})",
    )
    ax.scatter(
        plot_source[:, 0],
        plot_source[:, 1],
        plot_source[:, 2],
        s=1.0,
        c="crimson",
        alpha=0.45,
        label=f"aligned SAM3D source ({len(source_points)})",
    )

    if len(correspondence_array) > 0:
        segments = np.stack(
            [
                source_points[correspondence_array[:, 0]],
                target_points[correspondence_array[:, 1]],
            ],
            axis=1,
        )
        if len(segments) > 400:
            keep_positions = np.linspace(0, len(segments) - 1, num=400)
            keep_indices = np.unique(np.round(keep_positions).astype(np.int64))
            segments = segments[keep_indices]
        ax.add_collection3d(Line3DCollection(segments, colors="black", linewidths=0.8, alpha=0.45))

    all_points = np.concatenate([plot_source, plot_target], axis=0)
    _set_equal_axes(ax, all_points)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    ax.set_title(
        f"{output_stem} full context\n"
        f"source points={len(source_points)}, target points={len(target_points)}, "
        f"pairs={len(correspondence_array)}, tau={threshold:.6f}"
    )

    ax_pairs = fig.add_subplot(122, projection="3d")
    if len(correspondence_array) > 0:
        pair_points_source = source_points[correspondence_array[:, 0]]
        pair_points_target = target_points[correspondence_array[:, 1]]
        if len(correspondence_array) > 600:
            keep_positions = np.linspace(0, len(correspondence_array) - 1, num=600)
            keep_indices = np.unique(np.round(keep_positions).astype(np.int64))
            pair_points_source = pair_points_source[keep_indices]
            pair_points_target = pair_points_target[keep_indices]
        ax_pairs.scatter(
            pair_points_target[:, 0],
            pair_points_target[:, 1],
            pair_points_target[:, 2],
            s=8.0,
            c="royalblue",
            alpha=0.90,
            label=f"matched target ({len(pair_points_target)})",
        )
        ax_pairs.scatter(
            pair_points_source[:, 0],
            pair_points_source[:, 1],
            pair_points_source[:, 2],
            s=8.0,
            c="crimson",
            alpha=0.90,
            label=f"matched source ({len(pair_points_source)})",
        )
        pair_segments = np.stack([pair_points_source, pair_points_target], axis=1)
        ax_pairs.add_collection3d(Line3DCollection(pair_segments, colors="darkgreen", linewidths=1.0, alpha=0.55))
        _set_equal_axes(ax_pairs, np.concatenate([pair_points_source, pair_points_target], axis=0))
        ax_pairs.legend(loc="upper right")
    else:
        ax_pairs.text2D(0.1, 0.5, "No correspondences", transform=ax_pairs.transAxes)
        _set_equal_axes(ax_pairs, all_points)
    ax_pairs.set_xlabel("x")
    ax_pairs.set_ylabel("y")
    ax_pairs.set_zlabel("z")
    ax_pairs.set_title("Matched pairs only")

    out_path = Path(debug_dir) / f"{output_stem}_correspondence_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0) if len(points) > 0 else np.zeros(3, dtype=np.float32)


def _bbox_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e-3
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extents).clip(min=1e-6))


def _largest_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e-3
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.max(extents).clip(min=1e-6))


def _median_nn_distance(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 1e-3
    distances = np.asarray(_to_pcd(points).compute_nearest_neighbor_distance(), dtype=np.float32)
    positive = distances[np.isfinite(distances) & (distances > 0)]
    if len(positive) == 0:
        return 1e-3
    return float(np.median(positive))


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    down = _to_pcd(points, colors).voxel_down_sample(voxel_size=float(voxel_size))
    down_points = np.asarray(down.points, dtype=np.float32)
    if len(down_points) == 0:
        return points.astype(np.float32), colors.astype(np.float32)
    down_colors = np.asarray(down.colors, dtype=np.float32)
    return down_points, down_colors


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homog @ transform.T)[:, :3].astype(np.float32)


def _extract_isotropic_scale(transform: np.ndarray) -> float:
    linear = np.asarray(transform[:3, :3], dtype=np.float32)
    norms = np.linalg.norm(linear, axis=0)
    finite = norms[np.isfinite(norms)]
    if len(finite) == 0:
        return 1.0
    return float(np.mean(finite).clip(min=1e-6))


def _compose_similarity_transform(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (float(scale) * np.asarray(rotation, dtype=np.float32)).astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _build_explicit_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float,
):
    o3d_mod = _require_open3d()
    if len(source_points) == 0 or len(target_points) == 0:
        return o3d_mod.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32)), 0

    target_pcd = _to_pcd(target_points)
    target_kdtree = o3d_mod.geometry.KDTreeFlann(target_pcd)
    max_distance_sq = float(max_distance) * float(max_distance)
    pairs = []
    for source_idx, point in enumerate(source_points.astype(np.float64)):
        count, indices, sq_dists = target_kdtree.search_knn_vector_3d(point, 1)
        if count <= 0 or len(indices) == 0 or len(sq_dists) == 0:
            continue
        if not np.isfinite(sq_dists[0]) or sq_dists[0] > max_distance_sq:
            continue
        pairs.append([source_idx, int(indices[0])])

    if len(pairs) == 0:
        pair_array = np.zeros((0, 2), dtype=np.int32)
    else:
        pair_array = np.asarray(pairs, dtype=np.int32)
    return o3d_mod.utility.Vector2iVector(pair_array), int(len(pair_array))


def _run_probreg_similarity_refinement(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    init_transform: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, int]:
    if cpd is None:
        return init_transform.astype(np.float32), 0
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0

    transformed_source = _transform_points(source_points, init_transform)
    source_probreg_points = transformed_source
    source_probreg_colors = source_colors
    target_probreg_points = target_points
    target_probreg_colors = target_colors
    if len(source_probreg_points) < 3 or len(target_probreg_points) < 3:
        return init_transform.astype(np.float32), 0

    if len(source_probreg_colors) != len(source_probreg_points):
        source_probreg_colors = np.full((len(source_probreg_points), 3), 0.5, dtype=np.float32)
    if len(target_probreg_colors) != len(target_probreg_points):
        target_probreg_colors = np.full((len(target_probreg_points), 3), 0.5, dtype=np.float32)

    source_probreg_pcd = _to_pcd(source_probreg_points, source_probreg_colors)
    target_probreg_pcd = _to_pcd(target_probreg_points, target_probreg_colors)
    try:
        probreg_result = cpd.registration_cpd(
            source_probreg_pcd,
            target_probreg_pcd,
            tf_type_name="rigid",
            update_scale=True,
            maxiter=80,
            tol=1e-6,
            w=0.5,
            use_color=True,
        )
    except Exception:
        return init_transform.astype(np.float32), 0

    probreg_transform = _compose_similarity_transform(
        probreg_result.transformation.scale,
        probreg_result.transformation.rot,
        probreg_result.transformation.t,
    )
    refined_transform = probreg_transform @ init_transform.astype(np.float32)

    correspondence_count = int(min(len(source_probreg_points), len(target_probreg_points)))
    return refined_transform.astype(np.float32), correspondence_count


def register_and_fuse_sam3d_object(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    debug_dir: Path | None = None,
    artifact_dir: Path | None = None,
    output_stem: str | None = None,
) -> Sam3DInsertionResult:
    if len(source_points) == 0:
        raise ValueError("SAM3D source point cloud is empty.")
    if len(target_points) < 3:
        raise ValueError("Need at least 3 existing object Gaussians for SAM3D registration.")

    source_points = source_points.astype(np.float32)
    target_points = target_points.astype(np.float32)
    source_colors = _ensure_rgb_colors(source_colors, len(source_points))
    target_colors = _ensure_rgb_colors(target_colors, len(target_points))
    source_point_count = int(len(source_points))
    target_point_count = int(len(target_points))

    source_diag = _bbox_diagonal(source_points)
    target_diag = _bbox_diagonal(target_points)
    base_scale = target_diag / max(source_diag, 1e-6)
    chosen_scale = base_scale

    source_centroid = _centroid(source_points)
    target_centroid = _centroid(target_points)
    scaled_source = target_centroid[None, :] + chosen_scale * (source_points - source_centroid[None, :])
    scaled_source_colors = source_colors

    target_spacing = _median_nn_distance(target_points)
    source_spacing = _median_nn_distance(scaled_source)
    voxel_size = max(3.0 * max(target_spacing, source_spacing), 1e-3)
    source_down_points, source_down_colors = _voxel_downsample(scaled_source, scaled_source_colors, voxel_size)
    target_down_points, target_down_colors = _voxel_downsample(target_points, target_colors, voxel_size)

    similarity_transform = np.eye(4, dtype=np.float32)
    similarity_transform, similarity_correspondence_count = _run_probreg_similarity_refinement(
        source_down_points,
        source_down_colors,
        target_down_points,
        target_down_colors,
        similarity_transform,
        voxel_size,
    )
    similarity_correspondence_threshold = max(2.0 * _median_nn_distance(target_down_points), 1e-3)
    similarity_scale = float(chosen_scale * _extract_isotropic_scale(similarity_transform))
    source_visible_for_plot = _transform_points(source_down_points, similarity_transform)
    similarity_correspondences, _ = _build_explicit_correspondences(
        source_visible_for_plot,
        target_down_points,
        max_distance=similarity_correspondence_threshold,
    )

    aligned_points = _transform_points(scaled_source, similarity_transform)
    aligned_colors = scaled_source_colors.astype(np.float32)
    final_scale = float(chosen_scale * _extract_isotropic_scale(similarity_transform))

    dedup_threshold = 1.5 * target_spacing
    target_pcd = _to_pcd(target_points, target_colors)
    distances = np.asarray(_to_pcd(aligned_points).compute_point_cloud_distance(target_pcd), dtype=np.float32)
    keep_mask = np.isfinite(distances) & (distances >= dedup_threshold)
    kept_points = aligned_points[keep_mask].astype(np.float32)
    kept_colors = aligned_colors[keep_mask].astype(np.float32)

    correspondence_plot_path = ""
    if debug_dir is not None and output_stem is not None:
        debug_dir = Path(debug_dir)
        correspondence_plot_path = str(
            _save_correspondence_plot(
                debug_dir,
                output_stem,
                source_visible_for_plot,
                target_down_points,
                similarity_correspondences,
                similarity_correspondence_threshold,
            )
        )
    if artifact_dir is not None and output_stem is not None:
        artifact_dir = Path(artifact_dir)
        save_point_cloud(artifact_dir / f"{output_stem}_source_reg_ref.ply", source_down_points, source_down_colors)
        save_point_cloud(artifact_dir / f"{output_stem}_target_reg_ref.ply", target_down_points, target_down_colors)
        save_point_cloud(artifact_dir / f"{output_stem}_source_visible_work_iter_00.ply", source_visible_for_plot, source_down_colors)

    return Sam3DInsertionResult(
        aligned_points=aligned_points,
        aligned_colors=aligned_colors,
        kept_points=kept_points,
        kept_colors=kept_colors,
        chosen_scale=float(final_scale),
        dedup_threshold=float(dedup_threshold),
        voxel_size=float(voxel_size),
        source_point_count=source_point_count,
        target_point_count=target_point_count,
        # Visibility filtering is disabled in this experiment path, so report
        # the full downsampled source count here.
        visible_source_point_count=int(len(source_down_points)),
        registration_source_point_count=int(len(source_down_points)),
        kept_point_count=int(len(kept_points)),
        similarity_transform=similarity_transform,
        similarity_correspondence_count=int(similarity_correspondence_count),
        similarity_scale=float(similarity_scale),
        correspondence_threshold=float(similarity_correspondence_threshold),
        correspondence_plot_path=correspondence_plot_path,
    )
