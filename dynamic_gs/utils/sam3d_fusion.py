from __future__ import annotations
"""Geometry-only SAM3D object fusion for dynamic-gs.

The pipeline is:
1. load the SAM3D point cloud
2. estimate an isotropic scale from source/target extents
3. translate the scaled source to the target centroid
4. run Fast Global Registration on full downsampled geometry
5. keep only the post-FGR source points that are visible in the rendered object mask
6. refine the pose with a truncated nearest-neighbor rigid step, then point-to-point ICP
7. append only non-overlapping SAM3D points back into the Gaussian scene

This file intentionally uses only geometry. No RGB matching, colored ICP, or
object-specific axis heuristics are used here.
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
    result = o3d_mod.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance=float(threshold),
        init=init_transform.astype(np.float64),
        estimation_method=o3d_mod.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d_mod.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )
    return result, threshold


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
) -> Sam3DInsertionResult:
    if len(source_points) == 0:
        raise ValueError("SAM3D source point cloud is empty.")
    if len(target_points) < 3:
        raise ValueError("Need at least 3 existing object Gaussians for SAM3D registration.")

    render_object_mask = render_object_mask.astype(bool)
    source_point_count = int(len(source_points))
    target_point_count = int(len(target_points))

    source_diag = _bbox_diagonal(source_points)
    target_diag = _bbox_diagonal(target_points)
    base_scale = target_diag / max(source_diag, 1e-6)
    chosen_scale = base_scale

    source_centroid = _centroid(source_points)
    target_centroid = _centroid(target_points)
    scaled_source = target_centroid[None, :] + chosen_scale * (source_points - source_centroid[None, :])
    scaled_source_colors = source_colors.astype(np.float32)

    target_spacing = _median_nn_distance(target_points)
    source_spacing = _median_nn_distance(scaled_source)
    voxel_size = max(1.5 * max(target_spacing, source_spacing), 1e-3)
    source_down_points, source_down_colors = _voxel_downsample(scaled_source, scaled_source_colors, voxel_size)
    target_down_points, target_down_colors = _voxel_downsample(target_points, target_colors, voxel_size)
    fgr_result, _ = _run_fgr(
        source_down_points,
        source_down_colors,
        target_down_points,
        target_down_colors,
        voxel_size,
    )
    fgr_transform = np.asarray(fgr_result.transformation, dtype=np.float32)
    fgr_aligned_points = _transform_points(source_down_points, fgr_transform)

    visible_source_indices = _visible_source_indices(
        fgr_aligned_points,
        render_object_mask,
        viewmat,
        intrinsics,
        width,
        height,
    )
    visible_source_points = fgr_aligned_points[visible_source_indices].astype(np.float32)

    icp_source_points = source_down_points
    icp_source_colors = source_down_colors
    if len(visible_source_indices) >= 32:
        icp_source_points = source_down_points[visible_source_indices]
        icp_source_colors = source_down_colors[visible_source_indices]

    target_largest_extent = _largest_extent(target_down_points)
    robust_truncation_distance = max(target_largest_extent, 6.0 * voxel_size)
    robust_transform, robust_inlier_count = _run_truncated_rigid_refinement(
        icp_source_points,
        target_down_points,
        fgr_transform,
        truncation_distance=robust_truncation_distance,
    )

    icp_result, _ = _run_icp(
        icp_source_points,
        icp_source_colors,
        target_down_points,
        target_down_colors,
        robust_transform,
        voxel_size,
    )
    icp_transform = np.asarray(icp_result.transformation, dtype=np.float32)

    aligned_points = _transform_points(scaled_source, icp_transform)
    aligned_colors = scaled_source_colors.astype(np.float32)

    dedup_threshold = 1.5 * target_spacing
    target_pcd = _to_pcd(target_points, target_colors)
    distances = np.asarray(_to_pcd(aligned_points).compute_point_cloud_distance(target_pcd), dtype=np.float32)
    keep_mask = np.isfinite(distances) & (distances >= dedup_threshold)
    kept_points = aligned_points[keep_mask].astype(np.float32)
    kept_colors = aligned_colors[keep_mask].astype(np.float32)

    return Sam3DInsertionResult(
        aligned_points=aligned_points,
        aligned_colors=aligned_colors,
        kept_points=kept_points,
        kept_colors=kept_colors,
        chosen_scale=float(chosen_scale),
        dedup_threshold=float(dedup_threshold),
        voxel_size=float(voxel_size),
        source_point_count=source_point_count,
        target_point_count=target_point_count,
        visible_source_point_count=int(len(visible_source_points)),
        registration_source_point_count=int(len(icp_source_points)),
        kept_point_count=int(len(kept_points)),
        fgr_transformation=fgr_transform,
        icp_transformation=icp_transform,
        icp_fitness=float(icp_result.fitness),
        icp_rmse=float(icp_result.inlier_rmse),
        robust_transform=robust_transform,
        robust_inlier_count=int(robust_inlier_count),
        robust_truncation_distance=float(robust_truncation_distance),
    )
