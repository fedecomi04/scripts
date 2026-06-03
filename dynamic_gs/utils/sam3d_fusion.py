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

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from nerfstudio.utils.spherical_harmonics import SH2RGB
from .sam3d import load_sam3d_pose

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

try:
    import teaserpp_python as _teaserpp
except ImportError:  # pragma: no cover - optional registration backend
    _teaserpp = None

SAM3D_P3D_TO_NS_CAMERA = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)


@dataclass
class Sam3DInsertionResult:
    aligned_points: np.ndarray
    aligned_colors: np.ndarray
    kept_points: np.ndarray
    kept_colors: np.ndarray
    chosen_scale: float
    dedup_threshold: float
    source_spacing: float
    target_spacing: float
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
    # Full canonical-source-frame -> world 4x4 (rotation*scale | translation).
    # Composes the bbox-scale recentering with ``similarity_transform``:
    #   aligned_world = canonical_to_world_4x4 @ canonical_source_homog
    # This is the transform FoundationPose needs as ``mesh_to_world``: it
    # places a mesh given in the same canonical frame as the SAM3D Gaussian
    # PLY (and the Open3D Poisson reconstruction of it) into world space at
    # the same location as the inserted Gaussians.
    canonical_to_world_4x4: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    timing: dict = field(default_factory=dict)
    used_sam3d_rotation_init: bool = False


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


def load_sam3d_rotation_wxyz(pose_path: Path) -> np.ndarray:
    pose = load_sam3d_pose(pose_path)
    rotation = pose.get("rotation")
    if rotation is None or rotation.size != 4 or not np.isfinite(rotation).all():
        raise ValueError(f"SAM3D pose sidecar does not contain a valid rotation quaternion: {pose_path}")
    return rotation.astype(np.float32)


def _quaternion_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _apply_sam3d_rotation_init(
    source_points: np.ndarray,
    rotation_wxyz: np.ndarray,
    camera_to_world_rotation: np.ndarray,
) -> np.ndarray:
    rotation_p3d = _quaternion_wxyz_to_rotation_matrix(rotation_wxyz)
    rotated_p3d = source_points.astype(np.float32) @ rotation_p3d
    rotated_ns = rotated_p3d @ SAM3D_P3D_TO_NS_CAMERA
    camera_to_world_rotation = np.asarray(camera_to_world_rotation, dtype=np.float32).reshape(3, 3)
    return (rotated_ns @ camera_to_world_rotation.T).astype(np.float32)


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


def reconstruct_mesh_from_points(
    points: np.ndarray,
    mesh_ply_path: Path,
    voxel_size: float = 0.005,
    poisson_depth: int = 8,
    density_quantile_trim: float = 0.05,
) -> bool:
    """Build a triangle mesh PLY from an arbitrary point cloud using Open3D Poisson.

    Voxel-downsample, estimate normals via tangent-plane propagation, run
    Poisson surface reconstruction, and trim the lowest-density vertices to
    remove Poisson's convex-hull skirt. Returns True if a non-empty mesh was
    written.

    The input ``points`` is expected to be (N, 3) float in whatever frame the
    caller wants the mesh expressed in (e.g., world frame for the fused-result
    path, canonical SAM3D frame for the SAM3D-only path).
    """
    o3d = _require_open3d()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    if len(pcd.points) < 16:
        return False

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(2.0 * float(voxel_size), 0.01), max_nn=30
        )
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(k=20)
    except Exception:
        # Fallback: orient toward the centroid (gives a stable inward/outward).
        centroid = np.asarray(pcd.points).mean(axis=0)
        pcd.orient_normals_towards_camera_location(centroid)

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth), scale=1.1, linear_fit=False
        )
    )
    if len(mesh.triangles) == 0:
        return False

    # Trim the bottom-quantile density vertices (Poisson's hallucinated skirt).
    densities_np = np.asarray(densities)
    if densities_np.size > 0 and 0.0 < density_quantile_trim < 1.0:
        threshold = np.quantile(densities_np, density_quantile_trim)
        mask = densities_np < threshold
        if mask.any():
            mesh.remove_vertices_by_mask(mask)

    if len(mesh.triangles) == 0:
        return False

    mesh.compute_vertex_normals()
    Path(mesh_ply_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
    return True


def reconstruct_mesh_from_gaussian_ply(
    gaussian_ply_path: Path,
    mesh_ply_path: Path,
    voxel_size: float = 0.005,
    poisson_depth: int = 8,
    density_quantile_trim: float = 0.05,
) -> bool:
    """Build a triangle mesh PLY from a SAM3D Gaussian-splat PLY using Open3D Poisson.

    The SAM3D triangle-mesh decoder OOMs on 8 GiB GPUs (its 256^3 FlexiCubes
    grid pins ~740 MB on GPU). FoundationPose still needs a triangle mesh,
    so we reconstruct one from the Gaussian *centers*. Returns True if a
    non-empty mesh was written. Mesh is in the canonical SAM3D frame (matches
    the input PLY).
    """
    plyfile_mod = _require_plyfile()
    ply = plyfile_mod.read(str(gaussian_ply_path))
    vertex = ply["vertex"].data
    if vertex.size == 0:
        return False
    points = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float64)
    return reconstruct_mesh_from_points(
        points=points,
        mesh_ply_path=mesh_ply_path,
        voxel_size=voxel_size,
        poisson_depth=poisson_depth,
        density_quantile_trim=density_quantile_trim,
    )


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


# Robust statistics — needed because back-projected SAM3 masks routinely include
# a few depth-outlier pixels at the mask boundary (sensor noise / hole-fill drift).
# On a real mug-sized mask, ONE 2.2 m outlier blows up the bbox 10× and the
# centroid by tens of cm. Use the 5-95 percentile per axis to ignore the tails.
_ROBUST_PCT_LOW = 5.0
_ROBUST_PCT_HIGH = 95.0
_ROBUST_MIN_POINTS = 50  # below this, fall back to plain mean / min-max


def _centroid(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    if len(points) < _ROBUST_MIN_POINTS:
        return points.mean(axis=0)
    # Trimmed centroid: mean of points whose per-axis values all fall inside the
    # 5-95 percentile band. Equivalent to mean after rejecting outliers; preserves
    # the "centre of the bulk" while ignoring isolated stragglers.
    lo = np.percentile(points, _ROBUST_PCT_LOW, axis=0)
    hi = np.percentile(points, _ROBUST_PCT_HIGH, axis=0)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    if int(inside.sum()) < _ROBUST_MIN_POINTS:
        return points.mean(axis=0)
    return points[inside].mean(axis=0)


def _bbox_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e-3
    if len(points) < _ROBUST_MIN_POINTS:
        extents = points.max(axis=0) - points.min(axis=0)
    else:
        # Robust extent: 95th - 5th percentile per axis, ignoring tail outliers.
        hi = np.percentile(points, _ROBUST_PCT_HIGH, axis=0)
        lo = np.percentile(points, _ROBUST_PCT_LOW, axis=0)
        extents = hi - lo
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


def _l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def _multiscale_fpfh_descriptors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    normal_radius_mult: float,
    feature_radius_mults: list[float],
    color_weight: float,
    fpfh_max_nn: int = 500,
    normal_max_nn: int = 30,
) -> np.ndarray:
    """Concatenate FPFH descriptors at multiple radii (each L2-normalised) with optional color block."""
    o3d = _require_open3d()
    pcd = _to_pcd(points, colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_mult * voxel_size, max_nn=normal_max_nn
        )
    )
    blocks: list[np.ndarray] = []
    geom_weight = (1.0 - color_weight) / max(len(feature_radius_mults), 1)
    for mult in feature_radius_mults:
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=mult * voxel_size, max_nn=fpfh_max_nn
            ),
        )
        fpfh_arr = np.asarray(fpfh.data).T.astype(np.float32)
        blocks.append(_l2_normalize_rows(fpfh_arr) * geom_weight)
    if color_weight > 0:
        blocks.append(_l2_normalize_rows(colors.astype(np.float32)) * color_weight)
    return np.concatenate(blocks, axis=1)


def _euclidean_nn_correspondences(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual NN in 3D space with a distance threshold. Used for post-TEASER reprojection."""
    from scipy.spatial import cKDTree

    tree_tgt = cKDTree(tgt_pts)
    tree_src = cKDTree(src_pts)
    d_st, nn_t_for_s = tree_tgt.query(src_pts, k=1)
    _, nn_s_for_t = tree_src.query(tgt_pts, k=1)
    idx_s = np.arange(len(src_pts))
    keep = (nn_s_for_t[nn_t_for_s] == idx_s) & (d_st < max_dist)
    return idx_s[keep], nn_t_for_s[keep]


def _estimate_normals_np(points: np.ndarray, voxel_size: float, normal_max_nn: int = 30) -> np.ndarray:
    """Open3D normal estimation -> (N, 3) float32 numpy array. Orientation is unconstrained."""
    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2.0 * voxel_size, max_nn=normal_max_nn
        )
    )
    return np.asarray(pcd.normals, dtype=np.float32)


def _normal_consistency_filter(
    src_normals: np.ndarray,
    tgt_normals: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop correspondences whose source/target normals disagree by more than max_deg degrees.

    Uses |cos_sim| because Open3D normal orientation is ambiguous (no consistent
    tangent-plane orientation is applied for speed).
    """
    cos_thresh = float(np.cos(np.deg2rad(max_deg)))
    sn = src_normals[src_idx]
    tn = tgt_normals[tgt_idx]
    cos_sim = np.abs(np.einsum("ij,ij->i", sn, tn))
    keep = cos_sim > cos_thresh
    return src_idx[keep], tgt_idx[keep]


def _teaser_solve_xyz(
    src_xyz: np.ndarray, dst_xyz: np.ndarray, noise_bound: float,
) -> tuple[np.ndarray, float]:
    """Run TEASER on already-paired (3, N) correspondences. Returns (similarity_4x4, scale)."""
    if _teaserpp is None:
        return np.eye(4, dtype=np.float32), 1.0
    params = _teaserpp.RobustRegistrationSolver.Params()
    params.noise_bound = float(noise_bound)
    params.cbar2 = 1.0
    params.estimate_scaling = True
    params.rotation_estimation_algorithm = (
        _teaserpp.RotationEstimationAlgorithm.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-6
    solver = _teaserpp.RobustRegistrationSolver(params)
    solver.solve(src_xyz, dst_xyz)
    sol = solver.getSolution()
    T = _compose_similarity_transform(
        float(sol.scale),
        np.asarray(sol.rotation, dtype=np.float64),
        np.asarray(sol.translation, dtype=np.float64),
    )
    return T.astype(np.float32), float(sol.scale)


def _run_icp_polish(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    init_T: np.ndarray,
    voxel_size: float,
    max_corr_dist_mult: float = 2.0,
    iterations: int = 50,
) -> tuple[np.ndarray, dict]:
    """Point-to-plane ICP polish, scale-aware and guarded.

    ``src_pts``/``tgt_pts`` are the downsampled clouds; ``init_T`` is the full
    similarity transform (scale + rotation + translation) mapping ``src_pts`` into
    the target frame, as produced by the upstream TEASER/reproject stages.

    Two robustness properties — both REQUIRED because the upstream transform
    carries a non-unit scale and Open3D ICP only solves rigid SE(3):

    1. Scale handling. Open3D point-to-plane ICP cannot estimate scale and silently
       diverges when handed a scaled init (the rigid Jacobian is wrong in a scaled
       frame). We therefore PRE-APPLY ``init_T`` to the source, then run ICP with an
       identity init on the already-transformed cloud (a pure rigid residual at
       unit scale), and compose: ``final = icp_rigid @ init_T``.
    2. Improvement guard. We measure both correspondence fitness (inlier count
       ratio) and inlier RMSE of ``init_T`` and of the ICP result at the same
       threshold, and KEEP ICP only if it strictly increases the inlier count,
       or ties the count without worsening RMSE. Because Open3D fitness ignores
       RMSE, a count-only guard could accept a looser (higher-RMSE) fit that
       merely grazes more points — the classic ICP-sliding mode on a symmetric /
       low-texture object. The count+RMSE guard makes the stage never reduce
       either overlap or fit tightness relative to its input.

    Returns ``(final_T, meta)``. ``meta`` reports the pre/post fitness and which
    transform was accepted so the timing report and smoke test can see the decision.
    """
    o3d = _require_open3d()
    init_T = init_T.astype(np.float64)
    max_corr = float(max_corr_dist_mult) * float(voxel_size)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))
    nrm = o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel_size, max_nn=30)
    tgt.estimate_normals(nrm)

    # Pre-apply the (possibly scaled) init to the source so ICP sees a unit-scale
    # rigid residual. Build the cloud from the already-transformed points.
    pre_aligned = _transform_points(src_pts, init_T.astype(np.float32)).astype(np.float64)
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pre_aligned)
    src.estimate_normals(nrm)

    # Fitness of the init itself (identity transform on the pre-aligned cloud).
    eval_init = o3d.pipelines.registration.evaluate_registration(
        src, tgt, max_corr, np.eye(4, dtype=np.float64),
    )
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_corr,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
    )

    init_fitness = float(eval_init.fitness)
    icp_fitness = float(icp.fitness)
    init_rmse = float(eval_init.inlier_rmse)
    icp_rmse = float(icp.inlier_rmse)
    # Acceptance guard. Open3D fitness = (#inliers within max_corr) / (#source
    # points) and does NOT account for inlier_rmse, so a count-only guard can
    # accept a pose that grazes MORE points at a LOOSER (higher-RMSE) fit — the
    # classic ICP-sliding-on-a-symmetric-surface mode. To honour the documented
    # "never worse than input" invariant we require ICP to either strictly
    # increase the inlier count, OR tie the count (fitness is quantized to 1/N
    # on the small downsampled clouds, so ties are common) without worsening
    # RMSE. icp_fitness > 0 also rejects the divergence-to-zero case.
    accepted = icp_fitness > 0.0 and (
        icp_fitness > init_fitness
        or (icp_fitness >= init_fitness and icp_rmse <= init_rmse)
    )
    if accepted:
        final_T = (np.asarray(icp.transformation, dtype=np.float64) @ init_T).astype(np.float32)
        reported_fitness = icp_fitness
        reported_rmse = icp_rmse
    else:
        # ICP did not help (or diverged); fall back to the upstream transform.
        final_T = init_T.astype(np.float32)
        reported_fitness = init_fitness
        reported_rmse = init_rmse

    return final_T, {
        "fitness": reported_fitness,
        "inlier_rmse": reported_rmse,
        "init_fitness": init_fitness,
        "icp_fitness": icp_fitness,
        "init_rmse": init_rmse,
        "icp_rmse": icp_rmse,
        "accepted": bool(accepted),
        "max_corr_dist_mult": float(max_corr_dist_mult),
        "iterations": int(iterations),
    }


def _color_aware_fpfh_descriptors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    normal_radius_mult: float,
    feature_radius_mult: float,
    color_weight: float,
    fpfh_max_nn: int = 100,
    normal_max_nn: int = 30,
) -> np.ndarray:
    """Returns an (N, 33+3) descriptor: L2-normalised FPFH * (1-w) concatenated with L2-normalised RGB * w.

    ``fpfh_max_nn`` caps how many neighbors FPFH actually integrates over. Open3D
    uses Hybrid kNN+radius search, so if the cloud is dense the radius is silently
    capped at max_nn (default 100 was hiding the radius knob on dense clouds).
    """
    o3d = _require_open3d()
    pcd = _to_pcd(points, colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_mult * voxel_size, max_nn=normal_max_nn
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius_mult * voxel_size, max_nn=fpfh_max_nn
        ),
    )
    fpfh_arr = np.asarray(fpfh.data).T.astype(np.float32)  # (N, 33)
    fpfh_norm = _l2_normalize_rows(fpfh_arr) * (1.0 - color_weight)
    rgb_norm = _l2_normalize_rows(colors.astype(np.float32)) * color_weight
    return np.concatenate([fpfh_norm, rgb_norm], axis=1)


def _mutual_nearest_neighbor(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    ratio_thresh: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (idx_a, idx_b) of mutual nearest-neighbor matches between rows of feat_a and feat_b.

    If ``ratio_thresh`` is given, applies Lowe's ratio test on the a->b side:
    a query is only kept when its 1st-NN distance is < ratio_thresh * 2nd-NN distance.
    """
    from scipy.spatial import cKDTree

    tree_b = cKDTree(feat_b)
    tree_a = cKDTree(feat_a)

    if ratio_thresh is None:
        _, nn_b_for_a = tree_b.query(feat_a, k=1)
        keep_a = np.ones(len(feat_a), dtype=bool)
    else:
        dists, top2 = tree_b.query(feat_a, k=2)
        nn_b_for_a = top2[:, 0]
        keep_a = dists[:, 0] < ratio_thresh * np.maximum(dists[:, 1], 1e-12)

    _, nn_a_for_b = tree_a.query(feat_b, k=1)
    idx_a = np.arange(len(feat_a))
    mutual = (nn_a_for_b[nn_b_for_a] == idx_a) & keep_a
    return idx_a[mutual], nn_b_for_a[mutual]


def _run_teaser_similarity_refinement(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    init_transform: np.ndarray,
    voxel_size: float,
    *,
    noise_bound: float = 0.02,
    max_correspondences: int = 5000,
    normal_radius_mult: float = 2.0,
    feature_radius_mult: float = 5.0,
    color_weight: float = 0.3,
    fpfh_max_nn: int = 100,
    normal_max_nn: int = 30,
    ratio_thresh: float | None = None,
    multi_scale_radii: list[float] | None = None,
    normal_filter_deg: float | None = None,
) -> tuple[np.ndarray, int, dict]:
    """Returns (refined_transform, correspondence_count, teaser_meta).

    teaser_meta keys:
      noise_bound          – configured per-correspondence inlier tolerance (m)
      fpfh_correspondences – mutual-NN match count before subsampling
      used_correspondences – actual count fed to the TEASER solver
      scale                – recovered similarity scale (1.0 if scale estimation off)
      stop_reason          – "ok" if solve returned, "skipped" if backend unavailable
                             or too few correspondences, "exception" on solver error
    """
    _empty_meta: dict = {
        "noise_bound": float(noise_bound),
        "fpfh_correspondences": 0,
        "used_correspondences": 0,
        "scale": 1.0,
        "stop_reason": "skipped",
    }

    if _teaserpp is None:
        return init_transform.astype(np.float32), 0, _empty_meta
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0, _empty_meta

    transformed_source = _transform_points(source_points, init_transform)
    if len(source_colors) != len(transformed_source):
        source_colors = np.full((len(transformed_source), 3), 0.5, dtype=np.float32)
    if len(target_colors) != len(target_points):
        target_colors = np.full((len(target_points), 3), 0.5, dtype=np.float32)

    if multi_scale_radii is not None and len(multi_scale_radii) > 1:
        src_feat = _multiscale_fpfh_descriptors(
            transformed_source, source_colors, voxel_size,
            normal_radius_mult, multi_scale_radii, color_weight,
            fpfh_max_nn=fpfh_max_nn, normal_max_nn=normal_max_nn,
        )
        dst_feat = _multiscale_fpfh_descriptors(
            target_points, target_colors, voxel_size,
            normal_radius_mult, multi_scale_radii, color_weight,
            fpfh_max_nn=fpfh_max_nn, normal_max_nn=normal_max_nn,
        )
    else:
        src_feat = _color_aware_fpfh_descriptors(
            transformed_source, source_colors, voxel_size,
            normal_radius_mult, feature_radius_mult, color_weight,
            fpfh_max_nn=fpfh_max_nn, normal_max_nn=normal_max_nn,
        )
        dst_feat = _color_aware_fpfh_descriptors(
            target_points, target_colors, voxel_size,
            normal_radius_mult, feature_radius_mult, color_weight,
            fpfh_max_nn=fpfh_max_nn, normal_max_nn=normal_max_nn,
        )

    src_idx, dst_idx = _mutual_nearest_neighbor(src_feat, dst_feat, ratio_thresh=ratio_thresh)
    n_after_mutual = int(len(src_idx))

    if normal_filter_deg is not None and len(src_idx) >= 3:
        src_normals = _estimate_normals_np(transformed_source, voxel_size, normal_max_nn=normal_max_nn)
        tgt_normals = _estimate_normals_np(target_points, voxel_size, normal_max_nn=normal_max_nn)
        src_idx, dst_idx = _normal_consistency_filter(
            src_normals, tgt_normals, src_idx, dst_idx, max_deg=normal_filter_deg,
        )
    fpfh_correspondences = int(len(src_idx))
    if fpfh_correspondences < 3:
        meta = dict(_empty_meta, fpfh_correspondences=fpfh_correspondences, stop_reason="too_few_correspondences")
        return init_transform.astype(np.float32), fpfh_correspondences, meta

    if fpfh_correspondences > max_correspondences:
        rng = np.random.default_rng(0)
        sub = rng.choice(fpfh_correspondences, size=max_correspondences, replace=False)
        src_idx = src_idx[sub]
        dst_idx = dst_idx[sub]

    src_xyz = transformed_source[src_idx].T.astype(np.float64)  # (3, N)
    dst_xyz = target_points[dst_idx].T.astype(np.float64)

    params = _teaserpp.RobustRegistrationSolver.Params()
    params.noise_bound = float(noise_bound)
    params.cbar2 = 1.0
    params.estimate_scaling = True
    params.rotation_estimation_algorithm = (
        _teaserpp.RotationEstimationAlgorithm.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-6

    try:
        solver = _teaserpp.RobustRegistrationSolver(params)
        solver.solve(src_xyz, dst_xyz)
        solution = solver.getSolution()
    except Exception:  # noqa: BLE001 - solver wraps C++ exceptions
        return (
            init_transform.astype(np.float32),
            fpfh_correspondences,
            dict(_empty_meta, fpfh_correspondences=fpfh_correspondences, stop_reason="exception"),
        )

    teaser_transform = _compose_similarity_transform(
        float(solution.scale),
        np.asarray(solution.rotation, dtype=np.float64),
        np.asarray(solution.translation, dtype=np.float64),
    )
    refined_transform = teaser_transform @ init_transform.astype(np.float32)

    meta = {
        "noise_bound": float(noise_bound),
        "fpfh_correspondences": fpfh_correspondences,
        "after_mutual_match": n_after_mutual,
        "used_correspondences": int(len(src_idx)),
        "scale": float(solution.scale),
        "stop_reason": "ok",
        "fpfh_max_nn": int(fpfh_max_nn),
        "ratio_thresh": None if ratio_thresh is None else float(ratio_thresh),
        "multi_scale_radii": list(multi_scale_radii) if multi_scale_radii else None,
        "normal_filter_deg": None if normal_filter_deg is None else float(normal_filter_deg),
    }
    return refined_transform.astype(np.float32), int(len(src_idx)), meta


def _run_teaser_reproject_refinement(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    voxel_size: float,
    *,
    noise_bound: float = 0.005,
    max_corr_dist_mult: float = 3.0,
    max_correspondences: int = 5000,
) -> tuple[np.ndarray, int, dict]:
    """Second TEASER pass using Euclidean NN correspondences in 3D.

    Use this AFTER a first TEASER pass has produced an init_transform that's
    already close. We apply init_transform to source, then build mutual nearest
    neighbor correspondences in 3D space within ``max_corr_dist_mult * voxel_size``,
    then re-solve with TEASER on those geometric pairs.
    """
    empty_meta = {
        "noise_bound": float(noise_bound),
        "geom_correspondences": 0,
        "used_correspondences": 0,
        "max_corr_dist_mult": float(max_corr_dist_mult),
        "stop_reason": "skipped",
    }
    if _teaserpp is None:
        return init_transform.astype(np.float32), 0, empty_meta
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0, empty_meta

    aligned_src = _transform_points(source_points, init_transform)
    max_dist = max_corr_dist_mult * voxel_size
    src_idx, tgt_idx = _euclidean_nn_correspondences(aligned_src, target_points, max_dist)
    n_corr = int(len(src_idx))
    if n_corr < 3:
        meta = dict(empty_meta, geom_correspondences=n_corr, stop_reason="too_few_correspondences")
        return init_transform.astype(np.float32), n_corr, meta

    if n_corr > max_correspondences:
        rng = np.random.default_rng(0)
        sub = rng.choice(n_corr, size=max_correspondences, replace=False)
        src_idx = src_idx[sub]
        tgt_idx = tgt_idx[sub]

    src_xyz = aligned_src[src_idx].T.astype(np.float64)
    dst_xyz = target_points[tgt_idx].T.astype(np.float64)
    try:
        delta_T, scale = _teaser_solve_xyz(src_xyz, dst_xyz, noise_bound)
    except Exception:
        return (
            init_transform.astype(np.float32),
            n_corr,
            dict(empty_meta, geom_correspondences=n_corr, stop_reason="exception"),
        )

    final_T = delta_T @ init_transform.astype(np.float32)
    meta = {
        "noise_bound": float(noise_bound),
        "geom_correspondences": n_corr,
        "used_correspondences": int(len(src_idx)),
        "max_corr_dist_mult": float(max_corr_dist_mult),
        "delta_scale": float(scale),
        "stop_reason": "ok",
    }
    return final_T.astype(np.float32), int(len(src_idx)), meta


def _run_probreg_similarity_refinement(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    init_transform: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, int, dict]:
    """Returns (refined_transform, correspondence_count, cpd_meta).

    cpd_meta keys:
      iterations     – EM iterations actually executed
      stop_reason    – "tol" if tolerance was met early, "maxiter" if hard cap was hit
      maxiter        – configured hard cap
      tol            – configured convergence tolerance
      last_q_delta   – |q - q_prev| at the last iteration (None if unavailable); used
                       to show the actual convergence value when maxiter is reached
    """
    _MAXITER = 70
    _TOL = 1e-2
    _empty_meta: dict = {"iterations": 0, "stop_reason": "skipped", "maxiter": _MAXITER, "tol": _TOL, "last_q_delta": None}

    if cpd is None:
        return init_transform.astype(np.float32), 0, _empty_meta
    if len(source_points) < 3 or len(target_points) < 3:
        return init_transform.astype(np.float32), 0, _empty_meta

    transformed_source = _transform_points(source_points, init_transform)
    source_probreg_points = transformed_source
    source_probreg_colors = source_colors
    target_probreg_points = target_points
    target_probreg_colors = target_colors
    if len(source_probreg_points) < 3 or len(target_probreg_points) < 3:
        return init_transform.astype(np.float32), 0, _empty_meta

    if len(source_probreg_colors) != len(source_probreg_points):
        source_probreg_colors = np.full((len(source_probreg_points), 3), 0.5, dtype=np.float32)
    if len(target_probreg_colors) != len(target_probreg_points):
        target_probreg_colors = np.full((len(target_probreg_points), 3), 0.5, dtype=np.float32)

    source_probreg_pcd = _to_pcd(source_probreg_points, source_probreg_colors)
    target_probreg_pcd = _to_pcd(target_probreg_points, target_probreg_colors)

    # Callback fires once per completed EM step with the current result object.
    # probreg exposes the Q-function value as result.q in supported versions;
    # we track consecutive Q deltas to report the actual convergence value when
    # maxiter is hit instead of just saying "not reached".
    _iter_count = [0]
    _q_values: list = []
    def _iter_cb(result):
        _iter_count[0] += 1
        q = getattr(result, "q", None)
        if q is not None:
            try:
                _q_values.append(float(q))
            except (TypeError, ValueError):
                pass

    try:
        probreg_result = cpd.registration_cpd(
            source_probreg_pcd,
            target_probreg_pcd,
            tf_type_name="rigid",
            update_scale=True,
            maxiter=_MAXITER,
            tol=_TOL,
            w=0.5,
            use_color=True,
            callbacks=[_iter_cb],
        )
    except Exception:
        return init_transform.astype(np.float32), 0, _empty_meta

    iterations_run = _iter_count[0]
    stop_reason = "maxiter" if iterations_run >= _MAXITER else "tol"
    last_q_delta = abs(_q_values[-1] - _q_values[-2]) if len(_q_values) >= 2 else None
    cpd_meta = {
        "iterations": iterations_run,
        "stop_reason": stop_reason,
        "maxiter": _MAXITER,
        "tol": _TOL,
        "last_q_delta": last_q_delta,
    }

    probreg_transform = _compose_similarity_transform(
        probreg_result.transformation.scale,
        probreg_result.transformation.rot,
        probreg_result.transformation.t,
    )
    refined_transform = probreg_transform @ init_transform.astype(np.float32)

    correspondence_count = int(min(len(source_probreg_points), len(target_probreg_points)))
    return refined_transform.astype(np.float32), correspondence_count, cpd_meta


def register_and_fuse_sam3d_object(
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    source_rotation_wxyz: np.ndarray | None = None,
    camera_to_world_rotation: np.ndarray | None = None,
    debug_dir: Path | None = None,
    artifact_dir: Path | None = None,
    output_stem: str | None = None,
    registration_backend: str = "cpd",
    teaser_params: dict | None = None,
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
    used_sam3d_rotation_init = False

    if source_rotation_wxyz is not None:
        if camera_to_world_rotation is None:
            raise ValueError("camera_to_world_rotation is required when using SAM3D rotation initialization.")
        source_points = _apply_sam3d_rotation_init(source_points, source_rotation_wxyz, camera_to_world_rotation)
        used_sam3d_rotation_init = True

    source_diag = _bbox_diagonal(source_points)
    target_diag = _bbox_diagonal(target_points)
    base_scale = target_diag / max(source_diag, 1e-6)
    chosen_scale = base_scale

    source_centroid = _centroid(source_points)
    target_centroid = _centroid(target_points)
    scaled_source = target_centroid[None, :] + chosen_scale * (source_points - source_centroid[None, :])
    scaled_source_colors = source_colors

    # --- TIMING: D0.3b1 median NN distances (Open3D nearest-neighbor for voxel size estimation) ---
    _t = time.time()
    target_spacing = _median_nn_distance(target_points)
    source_spacing = _median_nn_distance(scaled_source)
    t_nn_distances = time.time() - _t

    # --- TIMING: D0.3b2 voxel downsampling (both source and target clouds) ---
    _t = time.time()
    voxel_size = max(3.0 * max(target_spacing, source_spacing), 1e-3)
    source_down_points, source_down_colors = _voxel_downsample(scaled_source, scaled_source_colors, voxel_size)
    target_down_points, target_down_colors = _voxel_downsample(target_points, target_colors, voxel_size)
    t_voxel_downsample = time.time() - _t

    # --- TIMING: D0.3b3 similarity refinement (backend = "cpd" probreg CPD, or "teaser" color-aware FPFH + TEASER++) ---
    _t = time.time()
    similarity_transform = np.eye(4, dtype=np.float32)
    # The reproject + ICP stages are TEASER-only; left None for the CPD path so
    # the timing dict reports them as not-run.
    reproject_meta = None
    icp_meta = None
    t_reproject = 0.0
    t_icp = 0.0
    if registration_backend == "teaser":
        tp = teaser_params or {}
        # --- Stage 1: FPFH + mutual-NN + TEASER ---
        similarity_transform, similarity_correspondence_count, refinement_meta = _run_teaser_similarity_refinement(
            source_down_points,
            source_down_colors,
            target_down_points,
            target_down_colors,
            similarity_transform,
            voxel_size,
            noise_bound=float(tp.get("noise_bound", 0.02)),
            max_correspondences=int(tp.get("max_correspondences", 5000)),
            normal_radius_mult=float(tp.get("normal_radius_mult", 2.0)),
            feature_radius_mult=float(tp.get("feature_radius_mult", 5.0)),
            color_weight=float(tp.get("color_weight", 0.0)),
            fpfh_max_nn=int(tp.get("fpfh_max_nn", 500)),
            normal_max_nn=int(tp.get("normal_max_nn", 30)),
            ratio_thresh=tp.get("ratio_thresh", None),
            multi_scale_radii=tp.get("multi_scale_radii", None),
            normal_filter_deg=tp.get("normal_filter_deg", None),
        )
        refinement_meta_key = "D0.3b3_teaser_meta"
        t_cpd_refinement = time.time() - _t

        # --- Stage 2: Euclidean-NN reproject + TEASER (composes onto similarity_transform).
        # This stage and ICP both operate on the SAME downsampled clouds and
        # thread the transform, mirroring scripts/run_teaser_registration_only.py
        # exactly. Each helper returns gracefully (empty meta) if teaserpp/open3d
        # is missing or too few correspondences survive, in which case the
        # transform passes through unchanged. ---
        if bool(tp.get("enable_reproject", True)):
            _tr = time.time()
            similarity_transform, _reproj_n, reproject_meta = _run_teaser_reproject_refinement(
                source_down_points,
                target_down_points,
                similarity_transform,
                voxel_size,
                noise_bound=float(tp.get("reproject_noise_bound", 0.005)),
                max_corr_dist_mult=float(tp.get("reproject_max_corr_mult", 3.0)),
                max_correspondences=int(tp.get("max_correspondences", 5000)),
            )
            t_reproject = time.time() - _tr

        # --- Stage 3: point-to-plane ICP polish (composes onto similarity_transform) ---
        if bool(tp.get("enable_post_icp", True)):
            _ti = time.time()
            similarity_transform, icp_meta = _run_icp_polish(
                source_down_points,
                target_down_points,
                similarity_transform,
                voxel_size,
                max_corr_dist_mult=float(tp.get("post_icp_max_corr_mult", 2.0)),
                iterations=int(tp.get("post_icp_iterations", 50)),
            )
            t_icp = time.time() - _ti
    elif registration_backend == "cpd":
        similarity_transform, similarity_correspondence_count, refinement_meta = _run_probreg_similarity_refinement(
            source_down_points,
            source_down_colors,
            target_down_points,
            target_down_colors,
            similarity_transform,
            voxel_size,
        )
        refinement_meta_key = "D0.3b3_cpd_meta"
        t_cpd_refinement = time.time() - _t
    else:
        raise ValueError(
            f"Unknown SAM3D registration backend: {registration_backend!r}. Expected 'cpd' or 'teaser'."
        )

    # --- TIMING: D0.3b4 explicit correspondence building (KD-tree per-point search) ---
    _t = time.time()
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

    # Compose the full canonical-source-frame -> world 4x4. The fusion
    # pipeline does:
    #   scaled_source = target_centroid + chosen_scale * (canonical - source_centroid)
    #   world         = similarity_transform @ scaled_source
    # Equivalently:
    #   world = similarity_transform @ T(target_centroid - chosen_scale * source_centroid) @ S(chosen_scale) @ canonical
    align_trans = np.eye(4, dtype=np.float64)
    align_trans[:3, 3] = (
        target_centroid.astype(np.float64) - float(chosen_scale) * source_centroid.astype(np.float64)
    )
    bbox_scale = np.eye(4, dtype=np.float64) * float(chosen_scale)
    bbox_scale[3, 3] = 1.0
    canonical_to_world_4x4 = (
        similarity_transform.astype(np.float64) @ align_trans @ bbox_scale
    )
    t_correspondences = time.time() - _t

    # --- TIMING: D0.3b5 dedup (disabled — keep all generated SAM3D points; existing-splat pruning is done in the pipeline using source_spacing) ---
    _t = time.time()
    dedup_threshold = 0.0
    kept_points = aligned_points.astype(np.float32)
    kept_colors = aligned_colors.astype(np.float32)
    t_dedup = time.time() - _t

    # --- TIMING: D0.3b6 correspondence plot + artifact PLY saves ---
    _t = time.time()
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
    t_plot_and_save = time.time() - _t

    return Sam3DInsertionResult(
        aligned_points=aligned_points,
        aligned_colors=aligned_colors,
        kept_points=kept_points,
        kept_colors=kept_colors,
        chosen_scale=float(final_scale),
        dedup_threshold=float(dedup_threshold),
        source_spacing=float(source_spacing),
        target_spacing=float(target_spacing),
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
        canonical_to_world_4x4=canonical_to_world_4x4,
        timing={
            "D0.3b1_nn_distances": t_nn_distances,
            "D0.3b2_voxel_downsample": t_voxel_downsample,
            "D0.3b3_refinement": t_cpd_refinement,
            "D0.3b3_backend": registration_backend,
            refinement_meta_key: refinement_meta,
            "D0.3b3_reproject": t_reproject,
            "D0.3b3_reproject_meta": reproject_meta,
            "D0.3b3_icp": t_icp,
            "D0.3b3_icp_meta": icp_meta,
            "D0.3b4_correspondences": t_correspondences,
            "D0.3b5_dedup": t_dedup,
            "D0.3b6_plot_and_save": t_plot_and_save,
        },
        used_sam3d_rotation_init=used_sam3d_rotation_init,
    )
