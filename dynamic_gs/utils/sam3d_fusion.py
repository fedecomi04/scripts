from __future__ import annotations
"""Geometry-only SAM3D object fusion for dynamic-gs.

The active fusion path is:
1. load the raw SAM3D point cloud and the current visible object subset from the splat
2. estimate an isotropic scale from source/target extents
3. translate the scaled SAM3D source to the target centroid
4. voxel-downsample both clouds and run Fast Global Registration for coarse pose
5. keep the post-FGR source points that remain visible in the rendered object mask
6. refine the coarse pose with probreg CPD similarity (scale + rigid)
7. run a short color-aware ICP cleanup
8. append only non-overlapping SAM3D points back into the Gaussian scene

The final insertion still uses append-with-dedup only. Existing scene/object
Gaussians are kept. This file intentionally uses geometry-first registration with
RGB used only in the final colored ICP cleanup.
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
    fgr_transformation: np.ndarray
    icp_transformation: np.ndarray
    icp_fitness: float
    icp_rmse: float
    robust_transform: np.ndarray
    robust_inlier_count: int
    robust_truncation_distance: float
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


def _project_points(
    points: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    depths = camera_points[:, 2]
    valid = np.isfinite(depths) & (depths > 1e-6)
    pixels = np.full((len(points), 2), -1, dtype=np.int32)
    if not np.any(valid):
        return pixels, depths, valid

    xy = camera_points[valid, :2] / depths[valid, None]
    u = intrinsics[0, 0] * xy[:, 0] + intrinsics[0, 2]
    v = intrinsics[1, 1] * xy[:, 1] + intrinsics[1, 2]
    valid_indices = np.flatnonzero(valid)
    pixels[valid_indices, 0] = np.rint(u).astype(np.int32)
    pixels[valid_indices, 1] = np.rint(v).astype(np.int32)
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, depths, valid & in_frame


def _visible_source_indices(
    points: np.ndarray,
    render_object_mask: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    pixels, depths, valid = _project_points(points, viewmat, intrinsics, width, height)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return np.zeros((0,), dtype=np.int64)

    inside_mask = render_object_mask[pixels[valid_indices, 1], pixels[valid_indices, 0]] > 0
    valid_indices = valid_indices[inside_mask]
    if len(valid_indices) == 0:
        return np.zeros((0,), dtype=np.int64)

    pixel_ids = pixels[valid_indices, 1] * width + pixels[valid_indices, 0]
    order = np.lexsort((depths[valid_indices], pixel_ids))
    sorted_indices = valid_indices[order]
    sorted_pixel_ids = pixel_ids[order]
    keep = np.ones(len(sorted_indices), dtype=bool)
    keep[1:] = sorted_pixel_ids[1:] != sorted_pixel_ids[:-1]
    return sorted_indices[keep].astype(np.int64)


def _estimate_normals_and_fpfh(point_cloud, voxel_size: float):
    o3d_mod = _require_open3d()
    if len(point_cloud.points) == 0:
        raise ValueError("Cannot estimate normals on an empty point cloud.")
    normal_radius = max(voxel_size * 2.0, 1e-3)
    feature_radius = max(voxel_size * 5.0, normal_radius)
    point_cloud.estimate_normals(
        o3d_mod.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=30)
    )
    features = o3d_mod.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d_mod.geometry.KDTreeSearchParamHybrid(radius=float(feature_radius), max_nn=100),
    )
    return point_cloud, features


def _run_fgr(source_points: np.ndarray, source_colors: np.ndarray, target_points: np.ndarray, target_colors: np.ndarray, voxel_size: float):
    o3d_mod = _require_open3d()
    source_pcd = _to_pcd(source_points, source_colors)
    target_pcd = _to_pcd(target_points, target_colors)
    source_pcd, source_fpfh = _estimate_normals_and_fpfh(source_pcd, voxel_size)
    target_pcd, target_fpfh = _estimate_normals_and_fpfh(target_pcd, voxel_size)
    threshold = max(voxel_size * 1.5, 1e-3)
    registration_mod = o3d_mod.pipelines.registration
    fgr_fn = getattr(
        registration_mod,
        "registration_fgr_based_on_feature_matching",
        getattr(registration_mod, "registration_fast_based_on_feature_matching"),
    )
    result = fgr_fn(
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
        registration_mod.FastGlobalRegistrationOption(
            maximum_correspondence_distance=float(threshold),
            iteration_number=64,
        ),
    )
    return result, threshold


def _run_icp(source_points: np.ndarray, source_colors: np.ndarray, target_points: np.ndarray, target_colors: np.ndarray, init_transform: np.ndarray, voxel_size: float):
    o3d_mod = _require_open3d()
    source_pcd = _to_pcd(source_points, source_colors)
    target_pcd = _to_pcd(target_points, target_colors)
    threshold = max(voxel_size * 1.0, 1e-3)
    registration_mod = o3d_mod.pipelines.registration
    use_colored_icp = (
        hasattr(registration_mod, "registration_colored_icp")
        and source_pcd.has_colors()
        and target_pcd.has_colors()
    )
    if use_colored_icp:
        normal_radius = max(voxel_size * 2.0, 1e-3)
        normal_param = o3d_mod.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=30)
        source_pcd.estimate_normals(normal_param)
        target_pcd.estimate_normals(normal_param)
        try:
            result = registration_mod.registration_colored_icp(
                source_pcd,
                target_pcd,
                max_correspondence_distance=float(threshold),
                init=init_transform.astype(np.float64),
                estimation_method=registration_mod.TransformationEstimationForColoredICP(lambda_geometric=0.968),
                criteria=registration_mod.ICPConvergenceCriteria(max_iteration=20),
            )
        except RuntimeError:
            result = registration_mod.registration_icp(
                source_pcd,
                target_pcd,
                max_correspondence_distance=float(threshold),
                init=init_transform.astype(np.float64),
                estimation_method=registration_mod.TransformationEstimationPointToPoint(),
                criteria=registration_mod.ICPConvergenceCriteria(max_iteration=20),
            )
    else:
        result = registration_mod.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=float(threshold),
            init=init_transform.astype(np.float64),
            estimation_method=registration_mod.TransformationEstimationPointToPoint(),
            criteria=registration_mod.ICPConvergenceCriteria(max_iteration=20),
        )
    return result, threshold


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


def _run_similarity_correspondence_refinement(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    init_transform: np.ndarray,
    correspondence_distance: float,
) -> tuple[np.ndarray, int]:
    o3d_mod = _require_open3d()
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0

    transformed_source = _transform_points(source_points, init_transform)
    correspondences, correspondence_count = _build_explicit_correspondences(
        transformed_source,
        target_points,
        max_distance=correspondence_distance,
    )
    if correspondence_count < 3:
        return init_transform.astype(np.float32), correspondence_count

    estimator = o3d_mod.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    delta_transform = estimator.compute_transformation(
        _to_pcd(transformed_source, source_colors),
        _to_pcd(target_points, target_colors),
        correspondences,
    )
    refined_transform = np.asarray(delta_transform, dtype=np.float32) @ init_transform.astype(np.float32)
    return refined_transform.astype(np.float32), correspondence_count


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


def _estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if len(source_points) < 3 or len(target_points) < 3:
        return np.eye(4, dtype=np.float32)

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid[None, :]
    target_centered = target_points - target_centroid[None, :]

    covariance = source_centered.T @ target_centered
    try:
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float32)

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = (target_centroid - rotation @ source_centroid).astype(np.float32)
    return transform


def _run_truncated_rigid_refinement(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    truncation_distance: float,
    trim_keep_ratio: float = 0.8,
    max_iterations: int = 10,
) -> tuple[np.ndarray, int]:
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0

    from sklearn.neighbors import NearestNeighbors

    transform = init_transform.astype(np.float32).copy()
    best_inlier_count = 0
    for _ in range(max_iterations):
        transformed_source = _transform_points(source_points, transform)
        nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean").fit(transformed_source)
        distances, indices = nn_model.kneighbors(target_points)
        distances = distances[:, 0].astype(np.float32)
        indices = indices[:, 0].astype(np.int64)

        valid = np.isfinite(distances) & (distances <= truncation_distance)
        if int(valid.sum()) < 3:
            break

        valid_indices = np.flatnonzero(valid)
        valid_distances = distances[valid]
        keep_count = max(3, int(np.ceil(trim_keep_ratio * len(valid_indices))))
        keep_order = np.argsort(valid_distances)[:keep_count]
        target_keep = valid_indices[keep_order]
        source_keep = indices[target_keep]

        delta = _estimate_rigid_transform(transformed_source[source_keep], target_points[target_keep])
        transform = (delta @ transform).astype(np.float32)
        best_inlier_count = max(best_inlier_count, keep_count)

        delta_translation = float(np.linalg.norm(delta[:3, 3]))
        delta_rotation = float(np.linalg.norm(delta[:3, :3] - np.eye(3, dtype=np.float32)))
        if delta_translation < 1e-5 and delta_rotation < 1e-4:
            break

    return transform, best_inlier_count


def register_and_fuse_sam3d_object(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    render_object_mask: np.ndarray,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    debug_dir: Path | None = None,
    output_stem: str | None = None,
) -> Sam3DInsertionResult:
    if len(source_points) == 0:
        raise ValueError("SAM3D source point cloud is empty.")
    if len(target_points) < 3:
        raise ValueError("Need at least 3 existing object Gaussians for SAM3D registration.")

    render_object_mask = render_object_mask.astype(bool)
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
    # Initialization-only experiment path: disable FGR/CPD/ICP and keep only
    # the initial scale + centroid alignment.
    fgr_transform = np.eye(4, dtype=np.float32)
    fgr_aligned_points = source_down_points.astype(np.float32)

    icp_source_points = source_down_points
    icp_source_colors = source_down_colors

    target_largest_extent = _largest_extent(target_down_points)
    robust_truncation_distance = max(target_largest_extent, 6.0 * voxel_size)
    robust_transform = np.eye(4, dtype=np.float32)
    robust_inlier_count = 0

    similarity_transform = np.eye(4, dtype=np.float32)
    similarity_transform, similarity_correspondence_count = _run_probreg_similarity_refinement(
        icp_source_points,
        icp_source_colors,
        target_down_points,
        target_down_colors,
        similarity_transform,
        voxel_size,
    )
    similarity_correspondence_threshold = max(2.0 * _median_nn_distance(target_down_points), 1e-3)
    similarity_scale = float(chosen_scale * _extract_isotropic_scale(similarity_transform))
    source_visible_for_plot = _transform_points(icp_source_points, similarity_transform)
    similarity_correspondences, _ = _build_explicit_correspondences(
        source_visible_for_plot,
        target_down_points,
        max_distance=similarity_correspondence_threshold,
    )

    # icp_result, _ = _run_icp(
    #     icp_source_points,
    #     icp_source_colors,
    #     target_down_points,
    #     target_down_colors,
    #     similarity_transform,
    #     voxel_size,
    # )
    # icp_transform = np.asarray(icp_result.transformation, dtype=np.float32)
    icp_transform = similarity_transform.astype(np.float32)
    icp_fitness = 0.0
    icp_rmse = 0.0

    aligned_points = _transform_points(scaled_source, icp_transform)
    aligned_colors = scaled_source_colors.astype(np.float32)
    final_scale = float(chosen_scale * _extract_isotropic_scale(icp_transform))

    dedup_threshold = 1.5 * target_spacing
    target_pcd = _to_pcd(target_points, target_colors)
    distances = np.asarray(_to_pcd(aligned_points).compute_point_cloud_distance(target_pcd), dtype=np.float32)
    keep_mask = np.isfinite(distances) & (distances >= dedup_threshold)
    kept_points = aligned_points[keep_mask].astype(np.float32)
    kept_colors = aligned_colors[keep_mask].astype(np.float32)

    correspondence_plot_path = ""
    if debug_dir is not None and output_stem is not None:
        debug_dir = Path(debug_dir)
        save_point_cloud(debug_dir / f"{output_stem}_source_reg_ref.ply", icp_source_points, icp_source_colors)
        save_point_cloud(debug_dir / f"{output_stem}_target_reg_ref.ply", target_down_points, target_down_colors)
        save_point_cloud(debug_dir / f"{output_stem}_source_visible_work_iter_00.ply", source_visible_for_plot, icp_source_colors)
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
        registration_source_point_count=int(len(icp_source_points)),
        kept_point_count=int(len(kept_points)),
        fgr_transformation=fgr_transform,
        icp_transformation=icp_transform,
        icp_fitness=float(icp_fitness),
        icp_rmse=float(icp_rmse),
        robust_transform=robust_transform,
        robust_inlier_count=int(robust_inlier_count),
        robust_truncation_distance=float(robust_truncation_distance),
        similarity_transform=similarity_transform,
        similarity_correspondence_count=int(similarity_correspondence_count),
        similarity_scale=float(similarity_scale),
        correspondence_threshold=float(similarity_correspondence_threshold),
        correspondence_plot_path=correspondence_plot_path,
    )
