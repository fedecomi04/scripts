#!/usr/bin/env python3
"""
Build a Nerfstudio-compatible colored point cloud from a saved dataset.

Required input:
    --depth-dir /path/to/run_folder/depth

Assumed dataset layout:
    <run_folder>/
        rgb/
        depth/
        masks/              # optional
        transforms.json

Sampling policy:
    - valid depth range: [100, 3000] mm
    - total cap: configurable with --max-total-points
    - per-image cap: configurable with --max-points-per-image
    - near-equal sampling across images with a final global cap:
          samples_per_image = min(max_points_per_image, ceil(max_total_points / N))
      where N = number of usable depth frames

Mask policy:
    - mask > 0  => valid scene pixel
    - mask == 0 => ignored

Depth convention:
    - assumes input depth comes from a ROS optical camera:
        x right, y down, z forward
    - converts to Nerfstudio/OpenGL-style camera coordinates before applying
      transform_matrix:
        x_ns =  x_ros
        y_ns = -y_ros
        z_ns = -z_ros

Output:
    ASCII PLY with fields:
        x y z red green blue

Example:
    python build_seed_pointcloud.py \
        --depth-dir /home/.../dataset_run/depth \
        --output /home/.../dataset_run/depth_seed_points.ply
"""

import argparse
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        try:
            import tifffile
            return tifffile.imread(str(path))
        except Exception:
            pass

    try:
        import imageio.v3 as iio
        return iio.imread(str(path))
    except Exception:
        pass

    from PIL import Image
    return np.array(Image.open(path))


def resolve_relpath(dataset_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (dataset_root / p).resolve()


def get_intrinsics(frame: Dict, meta: Dict) -> Tuple[float, float, float, float]:
    fx = frame.get("fl_x", meta.get("fl_x"))
    fy = frame.get("fl_y", meta.get("fl_y"))
    cx = frame.get("cx", meta.get("cx"))
    cy = frame.get("cy", meta.get("cy"))

    missing = [k for k, v in [("fl_x", fx), ("fl_y", fy), ("cx", cx), ("cy", cy)] if v is None]
    if missing:
        raise ValueError(f"Missing intrinsics in transforms.json: {missing}")

    return float(fx), float(fy), float(cx), float(cy)


def maybe_fix_gazebo_openni_principal_point(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    enabled: bool,
) -> Tuple[float, float, float, float]:
    if not enabled:
        return fx, fy, cx, cy

    # Gazebo Classic's gazebo_ros_openni_kinect depth ray model is centered at
    # (width - 1) / 2, (height - 1) / 2, while the default published CameraInfo
    # principal point is often one pixel higher on even-sized images.
    fixed_cx = 0.5 * (float(width) - 1.0)
    fixed_cy = 0.5 * (float(height) - 1.0)
    return fx, fy, fixed_cx, fixed_cy


def get_frame_c2w(frame: Dict, fix_live_optical_pose: bool = False) -> np.ndarray:
    pose_key = "depth_transform_matrix" if "depth_transform_matrix" in frame else "transform_matrix"
    c2w = np.array(frame[pose_key], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"{pose_key} must be 4x4, got {c2w.shape}")
    if fix_live_optical_pose:
        legacy_fix = np.eye(4, dtype=np.float32)
        legacy_fix[:3, :3] = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        c2w = c2w @ legacy_fix
    return c2w


def load_mask(mask_path: Optional[Path], expected_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_path is None or not mask_path.exists():
        return None

    mask = read_image(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape != expected_hw:
        raise ValueError(
            f"Mask shape mismatch for {mask_path}: got {mask.shape}, expected {expected_hw}"
        )

    # Valid scene pixel = nonzero
    return mask > 0


def resolve_frame_mask_path(dataset_root: Path, frame: Dict) -> Optional[Path]:
    mask_rel = frame.get("mask_path")
    if mask_rel is not None:
        mask_path = resolve_relpath(dataset_root, mask_rel)
        if mask_path.exists():
            return mask_path

    rgb_rel = frame.get("file_path")
    if rgb_rel is None:
        return None

    rgb_name = Path(rgb_rel).name
    fallback_mask_path = (dataset_root / "masks" / rgb_name).resolve()
    if fallback_mask_path.exists():
        return fallback_mask_path
    return None


def load_rgb(rgb_path: Optional[Path], expected_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if rgb_path is None or not rgb_path.exists():
        return None

    rgb = read_image(rgb_path)

    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[..., :3]
    else:
        raise ValueError(f"Unsupported RGB image shape for {rgb_path}: {rgb.shape}")

    if rgb.shape[:2] != expected_hw:
        raise ValueError(
            f"RGB shape mismatch for {rgb_path}: got {rgb.shape[:2]}, expected {expected_hw}"
        )

    return rgb.astype(np.uint8)

def backproject_variant_to_world(
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
    mode: str = "A",
) -> np.ndarray:
    z = depth_mm / 1000.0
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    if mode == "A":
        # current behavior
        cam_xyz = np.stack([x, -y, -z], axis=1)
    elif mode == "B":
        cam_xyz = np.stack([x,  y,  z], axis=1)
    elif mode == "C":
        cam_xyz = np.stack([x, -y,  z], axis=1)
    elif mode == "D":
        cam_xyz = np.stack([x,  y, -z], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cam_points = np.concatenate(
        [cam_xyz, np.ones((cam_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    world = (c2w @ cam_points.T).T[:, :3]
    return world.astype(np.float32)

def backproject_ns_camera_to_world(
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
) -> np.ndarray:
    """
    Backproject directly into Nerfstudio camera coordinates.

    Assumes:
    - transform_matrix is already correct in Nerfstudio convention
    - depth is metric distance along the camera forward axis
    - image coordinates are standard raster coordinates:
        u right, v down

    Nerfstudio camera convention:
        +X right
        +Y up
        +Z back
        camera looks along -Z
    """
    z = depth_mm.astype(np.float32) / 1000.0
    x = (u.astype(np.float32) - cx) * z / fx
    y = -(v.astype(np.float32) - cy) * z / fy

    cam_points = np.stack(
        [
            x,
            y,
            -z,
            np.ones_like(z, dtype=np.float32),
        ],
        axis=1,
    )  # [N, 4]

    world = (c2w @ cam_points.T).T[:, :3]
    return world.astype(np.float32)


def camera_points_from_mode(
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mode: str,
) -> np.ndarray:
    z = depth_mm.astype(np.float32) / 1000.0
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    if mode == "A":
        return np.stack([x, -y, -z], axis=1)
    if mode == "B":
        return np.stack([x, y, z], axis=1)
    if mode == "C":
        return np.stack([x, -y, z], axis=1)
    if mode == "D":
        return np.stack([x, y, -z], axis=1)
    if mode == "NS":
        return np.stack([x, -y, -z], axis=1)

    raise ValueError(f"Unknown projection mode: {mode}")


def backproject_mode_to_world(
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
    mode: str,
) -> np.ndarray:
    cam_xyz = camera_points_from_mode(u, v, depth_mm, fx, fy, cx, cy, mode)
    cam_points = np.concatenate(
        [cam_xyz, np.ones((cam_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (c2w @ cam_points.T).T[:, :3]
    return world.astype(np.float32)

def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of points")

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


def collect_usable_frames(
    dataset_root: Path,
    meta: Dict,
    source_frame_range: Optional[Tuple[int, int]] = None,
) -> List[Dict]:
    usable = []
    frames = meta.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms.json")

    if source_frame_range is not None:
        start_idx, end_idx = source_frame_range
        if start_idx < 0 or start_idx >= len(frames):
            raise ValueError(
                f"source-frame-range start {start_idx} is outside source frames ({len(frames)})"
            )
        end_idx = min(end_idx, len(frames) - 1)
        frames = frames[start_idx : end_idx + 1]

    prev_pose = None
    skipped_repeated_pose = 0

    for frame in frames:
        depth_rel = frame.get("depth_file_path")
        if depth_rel is None:
            continue

        depth_path = resolve_relpath(dataset_root, depth_rel)
        if not depth_path.exists():
            continue

        c2w = get_frame_c2w(frame)
        if prev_pose is not None and np.allclose(c2w, prev_pose, atol=1e-12, rtol=0.0):
            skipped_repeated_pose += 1
            continue

        usable.append(frame)
        prev_pose = c2w

    if not usable:
        raise ValueError("No usable frames with existing depth_file_path were found")

    if skipped_repeated_pose:
        print(f"Skipped {skipped_repeated_pose} consecutive repeated-pose frames")

    return usable


def parse_frame_range(spec: str) -> Optional[Tuple[int, int]]:
    if not spec:
        return None

    cleaned = spec.replace(",", ":").strip()
    parts = [p.strip() for p in cleaned.split(":") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid --debug-frame-range '{spec}'. Use start:end or start,end")

    start, end = int(parts[0]), int(parts[1])
    if start > end:
        start, end = end, start
    return start, end


def estimate_plane_normal(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return centroid, normal, singular_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Path to the depth/ folder inside a Nerfstudio dataset run folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. Default: <dataset_root>/depth_seed_points.ply",
    )
    parser.add_argument(
        "--depth-min-mm",
        type=int,
        default=100,
        help="Minimum valid depth in millimeters",
    )
    parser.add_argument(
        "--depth-max-mm",
        type=int,
        default=3000,
        help="Maximum valid depth in millimeters",
    )
    parser.add_argument(
        "--max-total-points",
        type=int,
        default=300000,
        help="Global point cap across the full dataset",
    )
    parser.add_argument(
        "--max-points-per-image",
        type=int,
        default=10000,
        help="Upper bound on points sampled from one image",
    )
    parser.add_argument(
        "--source-frame-range",
        type=str,
        default="",
        help="Optional source-frame index range to use before repeated-pose filtering, e.g. 0:0 or 0,0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--max-visualize-frames",
        type=int,
        default=0,
        help="Maximum number of used camera poses to visualize. 0 means all used frames.",
    )
    parser.add_argument(
        "--debug-poses",
        action="store_true",
        help="Print compact pose diagnostics for used frames.",
    )
    parser.add_argument(
        "--debug-batch-visualize",
        action="store_true",
        help="Visualize used-frame point clouds in batches to isolate bad frame ranges.",
    )
    parser.add_argument(
        "--debug-batch-size",
        type=int,
        default=15,
        help="Number of used frames per debug visualization batch.",
    )
    parser.add_argument(
        "--debug-batch-points-per-image",
        type=int,
        default=5000,
        help="Maximum sampled points per image in debug batch visualization mode.",
    )
    parser.add_argument(
        "--debug-frame-range",
        type=str,
        default="",
        help="Used-frame index range to inspect in detail, e.g. 120:134 or 120,134.",
    )
    parser.add_argument(
        "--debug-projection-modes",
        type=str,
        default="",
        help="Comma-separated projection modes to compare on debug-frame-range, e.g. NS,A,C,D.",
    )
    parser.add_argument(
        "--fix-live-optical-pose",
        action="store_true",
        help="Apply a constant correction for legacy live-saver datasets that stored "
             "camera_link_optical poses with a non-optical camera-frame conversion.",
    )
    parser.add_argument(
        "--gazebo-openni-principal-point-fix",
        action="store_true",
        help="Override cx/cy with (width-1)/2 and (height-1)/2 when reconstructing "
             "Gazebo Classic gazebo_ros_openni_kinect depth datasets.",
    )
    parser.add_argument(
        "--skip-outlier-removal",
        action="store_true",
        help="Write the sampled points directly without Open3D statistical outlier removal.",
    )
    args = parser.parse_args()

    depth_dir = args.depth_dir.resolve()
    if not depth_dir.exists() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    dataset_root = depth_dir.parent
    transforms_path = dataset_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at: {transforms_path}")

    output_path = args.output.resolve() if args.output else (dataset_root / "depth_seed_points.ply")

    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Check for missing depth/mask entries in the original transforms
    missing = []
    for i, f in enumerate(meta.get("frames", [])):
        if "depth_file_path" not in f:
            missing.append((i, "no_depth"))
        if "mask_path" not in f:
            missing.append((i, "no_mask"))

    if missing:
        print(f"WARNING: {len(missing)} frames missing depth/mask")

    source_frame_range = parse_frame_range(args.source_frame_range)
    usable_frames = collect_usable_frames(dataset_root, meta, source_frame_range=source_frame_range)
    num_frames = len(usable_frames)
    if source_frame_range is not None:
        print(
            f"Using source frames {source_frame_range[0]}..{source_frame_range[1]} "
            f"before repeated-pose filtering"
        )

    if args.debug_poses:
        centers = []
        bad_rotations = 0
        for i, frame in enumerate(usable_frames):
            try:
                c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)
            except ValueError as exc:
                print(f"POSE ERROR frame {i}: {exc}")
                continue
            rot = c2w[:3, :3]
            center = c2w[:3, 3]
            ortho_err = np.linalg.norm(rot.T @ rot - np.eye(3, dtype=np.float32))
            det = np.linalg.det(rot)
            if not np.isfinite(det) or abs(det) < 1e-6:
                bad_rotations += 1
            centers.append(center)
            if i < 10:
                print(
                    f"POSE {i}: center={center.tolist()} det={det:.6f} ortho_err={ortho_err:.6f}"
                )

        if centers:
            centers_np = np.stack(centers, axis=0)
            mins = centers_np.min(axis=0)
            maxs = centers_np.max(axis=0)
            unique_rounded = np.unique(np.round(centers_np, 4), axis=0).shape[0]
            print(f"POSE SUMMARY: used_frames={len(usable_frames)}")
            print(f"  center_min={mins.tolist()}")
            print(f"  center_max={maxs.tolist()}")
            print(f"  unique_centers_rounded_1e-4={unique_rounded}")
            print(f"  bad_rotations={bad_rotations}")

    effective_max_points_per_image = args.max_points_per_image
    if num_frames == 1:
        effective_max_points_per_image = max(effective_max_points_per_image, args.max_total_points)

    samples_per_image = min(
        effective_max_points_per_image,
        math.ceil(args.max_total_points / num_frames),
    )

    if samples_per_image <= 0:
        raise ValueError(
            f"Computed samples_per_image = {samples_per_image}. "
            f"Increase max-total-points or reduce number of depth frames."
        )

    rng = np.random.default_rng(args.seed)

    all_xyz = []
    all_rgb = []

    for i, frame in enumerate(usable_frames):
        depth_path = resolve_relpath(dataset_root, frame["depth_file_path"])
        rgb_path = None
        if "file_path" in frame:
            rgb_path = resolve_relpath(dataset_root, frame["file_path"])

        mask_path = resolve_frame_mask_path(dataset_root, frame)

        depth = read_image(depth_path)
        if depth.ndim != 2:
            raise ValueError(f"Depth image must be HxW, got {depth.shape} for {depth_path}")

        # Keep depth in float32 millimeters internally.
        if np.issubdtype(depth.dtype, np.floating):
            # Assume floating TIFF depth is in meters -> convert to mm
            depth = depth.astype(np.float32) * 1000.0
        elif depth.dtype == np.uint16:
            depth = depth.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported depth dtype for {depth_path}: {depth.dtype}. "
                f"Expected uint16 (mm) or floating type (meters)."
            )

        if i < 3:
            nonzero = depth[depth > 0]
            print(
                f"DEBUG {depth_path.name}: "
                f"dtype={depth.dtype}, "
                f"min_nonzero_mm={nonzero.min() if nonzero.size else 'none'}, "
                f"max_mm={depth.max()}"
            )

        h, w = depth.shape
        fx, fy, cx, cy = get_intrinsics(frame, meta)
        fx, fy, cx, cy = maybe_fix_gazebo_openni_principal_point(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            enabled=args.gazebo_openni_principal_point_fix,
        )

        rgb = load_rgb(rgb_path, (h, w))
        if rgb is None:
            raise ValueError(
                f"RGB image not found for frame {i}. "
                f"file_path is required to write colored PLY points."
            )

        valid_mask = load_mask(mask_path, (h, w))
        if valid_mask is None:
            raise ValueError(f"Missing mask for frame {i}: {mask_path}")

        valid_depth = (
            (depth > 0)
            & (depth >= args.depth_min_mm)
            & (depth <= args.depth_max_mm)
        )

        valid = valid_mask & valid_depth
        ys, xs = np.where(valid)
        n_valid = xs.shape[0]

        # Print detailed mask/depth/debug info for the first few frames
        if i < 5:
            print(f"FRAME {i}")
            print(f"  mask_path: {mask_path}")
            print(f"  intrinsics_used: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            print(f"  valid_mask_pixels: {valid_mask.sum()} / {valid_mask.size}")
            print(f"  valid_depth_pixels: {valid_depth.sum()}")
            print(f"  final_valid_pixels: {n_valid}")

        # Skip frames with too few valid points
        if n_valid < 500:
            continue

        if i < 3:
            print(
                f"DEBUG frame {i}: "
                f"depth_min_valid={depth[depth > 0].min() if np.any(depth > 0) else 'none'}, "
                f"depth_max_valid={depth[depth > 0].max() if np.any(depth > 0) else 'none'}, "
                f"masked_valid_pixels={n_valid}"
            )

        if n_valid == 0:
            continue

        n_sample = min(samples_per_image, n_valid)
        choice = rng.choice(n_valid, size=n_sample, replace=False)

        u = xs[choice]
        v = ys[choice]
        d = depth[v, u]

        c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)

        xyz_world = backproject_ns_camera_to_world(
            u=u,
            v=v,
            depth_mm=d,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            c2w=c2w,
        )
        rgb_sel = rgb[v, u]

        all_xyz.append(xyz_world)
        all_rgb.append(rgb_sel)

        print(
            f"[{i+1:04d}/{num_frames:04d}] "
            f"{depth_path.name}: valid={n_valid}, sampled={n_sample}"
        )

    if not all_xyz:
        raise ValueError("No points were generated. Check depth range, masks, and transforms.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # Final safety cap, though the equal-per-image rule should already enforce it.
    if xyz.shape[0] > args.max_total_points:
        keep = rng.choice(xyz.shape[0], size=args.max_total_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]

    # Remove statistical outlier points (planes/noise) before writing unless explicitly disabled.
    if not args.skip_outlier_removal:
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            xyz = np.asarray(pcd.points)
            rgb = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
        except Exception as exc:
            print(f"Warning: open3d outlier removal failed or open3d not installed: {exc}")

    write_ascii_ply(output_path, xyz, rgb)

    # Add top-level ply_file_path to transforms.json (use a relative path)
    try:
        rel_ply = output_path.relative_to(dataset_root).as_posix()
    except Exception:
        rel_ply = output_path.name

    meta["ply_file_path"] = rel_ply
    try:
        with open(transforms_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated transforms.json with ply_file_path: {rel_ply}")
    except Exception as exc:
        print(f"Warning: failed to write transforms.json with ply_file_path: {exc}")

    debug_frame_range = parse_frame_range(args.debug_frame_range)
    if debug_frame_range is not None:
        try:
            import open3d as o3d

            start_idx, end_idx = debug_frame_range
            if start_idx >= len(usable_frames):
                raise ValueError(
                    f"debug-frame-range start {start_idx} is outside used frames ({len(usable_frames)})"
                )

            end_idx = min(end_idx, len(usable_frames) - 1)
            inspect_frames = usable_frames[start_idx : end_idx + 1]
            inspect_geometries = []
            inspect_colors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )

            print(f"DEBUG FRAME RANGE inspecting used frames {start_idx}..{end_idx}")

            for local_idx, frame in enumerate(inspect_frames):
                global_idx = start_idx + local_idx
                depth_path = resolve_relpath(dataset_root, frame["depth_file_path"])
                rgb_path = resolve_relpath(dataset_root, frame["file_path"]) if "file_path" in frame else None
                mask_path = resolve_relpath(dataset_root, frame["mask_path"]) if "mask_path" in frame else None

                depth = read_image(depth_path)
                if np.issubdtype(depth.dtype, np.floating):
                    depth = depth.astype(np.float32) * 1000.0
                elif depth.dtype == np.uint16:
                    depth = depth.astype(np.float32)
                else:
                    print(f"DEBUG FRAME {global_idx}: unsupported depth dtype {depth.dtype}")
                    continue

                h, w = depth.shape
                fx, fy, cx, cy = get_intrinsics(frame, meta)
                rgb_img = load_rgb(rgb_path, (h, w))
                valid_mask = load_mask(mask_path, (h, w))
                if rgb_img is None or valid_mask is None:
                    print(f"DEBUG FRAME {global_idx}: missing rgb or mask")
                    continue

                valid_depth = (
                    (depth > 0)
                    & (depth >= args.depth_min_mm)
                    & (depth <= args.depth_max_mm)
                )
                valid = valid_mask & valid_depth
                ys, xs = np.where(valid)
                n_valid = xs.shape[0]
                if n_valid < 3:
                    print(f"DEBUG FRAME {global_idx}: too few valid points ({n_valid})")
                    continue

                n_sample = min(args.debug_batch_points_per_image, n_valid)
                choice = rng.choice(n_valid, size=n_sample, replace=False)
                u = xs[choice]
                v = ys[choice]
                d = depth[v, u]

                c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)

                z = d.astype(np.float32) / 1000.0
                x = (u.astype(np.float32) - cx) * z / fx
                y = -(v.astype(np.float32) - cy) * z / fy
                cam_xyz = np.stack([x, y, -z], axis=1)
                world_xyz = backproject_ns_camera_to_world(
                    u=u,
                    v=v,
                    depth_mm=d,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                )

                cam_centroid, cam_normal, cam_singular = estimate_plane_normal(cam_xyz)
                world_centroid, world_normal, world_singular = estimate_plane_normal(world_xyz)

                world_cam_forward = c2w[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
                cos_angle = np.clip(
                    np.abs(np.dot(world_normal, world_cam_forward))
                    / (np.linalg.norm(world_normal) * np.linalg.norm(world_cam_forward) + 1e-8),
                    0.0,
                    1.0,
                )
                angle_deg = float(np.degrees(np.arccos(cos_angle)))

                print(
                    f"DEBUG FRAME {global_idx}: valid={n_valid}, sampled={n_sample}, "
                    f"cam_centroid={cam_centroid.tolist()}, world_centroid={world_centroid.tolist()}, "
                    f"cam_plane_sv={cam_singular.tolist()}, world_plane_sv={world_singular.tolist()}, "
                    f"normal_vs_forward_deg={angle_deg:.2f}"
                )

                frame_color = inspect_colors[local_idx % len(inspect_colors)]

                world_pcd = o3d.geometry.PointCloud()
                world_pcd.points = o3d.utility.Vector3dVector(world_xyz)
                world_pcd.paint_uniform_color(frame_color.tolist())
                inspect_geometries.append(world_pcd)

                cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam.transform(c2w)
                inspect_geometries.append(cam)

            if inspect_geometries:
                o3d.visualization.draw_geometries(inspect_geometries)
        except Exception as exc:
            print(f"Warning: debug frame range visualization failed: {exc}")

    if debug_frame_range is not None and args.debug_projection_modes:
        try:
            import open3d as o3d

            modes = [m.strip() for m in args.debug_projection_modes.split(",") if m.strip()]
            start_idx, end_idx = debug_frame_range
            end_idx = min(end_idx, len(usable_frames) - 1)
            inspect_frames = usable_frames[start_idx : end_idx + 1]
            mode_colors = {
                "NS": [1.0, 0.0, 0.0],
                "A": [0.0, 1.0, 0.0],
                "B": [0.0, 0.0, 1.0],
                "C": [1.0, 1.0, 0.0],
                "D": [1.0, 0.0, 1.0],
            }

            print(
                f"DEBUG PROJECTION MODES for used frames {start_idx}..{end_idx}: {modes}"
            )

            for global_idx, frame in enumerate(inspect_frames, start=start_idx):
                depth_path = resolve_relpath(dataset_root, frame["depth_file_path"])
                mask_path = resolve_relpath(dataset_root, frame["mask_path"]) if "mask_path" in frame else None

                depth = read_image(depth_path)
                if np.issubdtype(depth.dtype, np.floating):
                    depth = depth.astype(np.float32) * 1000.0
                elif depth.dtype == np.uint16:
                    depth = depth.astype(np.float32)
                else:
                    print(f"DEBUG PROJECTION frame {global_idx}: unsupported depth dtype {depth.dtype}")
                    continue

                h, w = depth.shape
                fx, fy, cx, cy = get_intrinsics(frame, meta)
                valid_mask = load_mask(mask_path, (h, w))
                if valid_mask is None:
                    print(f"DEBUG PROJECTION frame {global_idx}: missing mask")
                    continue

                valid_depth = (
                    (depth > 0)
                    & (depth >= args.depth_min_mm)
                    & (depth <= args.depth_max_mm)
                )
                valid = valid_mask & valid_depth
                ys, xs = np.where(valid)
                n_valid = xs.shape[0]
                if n_valid < 3:
                    print(f"DEBUG PROJECTION frame {global_idx}: too few valid points ({n_valid})")
                    continue

                n_sample = min(args.debug_batch_points_per_image, n_valid)
                choice = rng.choice(n_valid, size=n_sample, replace=False)
                u = xs[choice]
                v = ys[choice]
                d = depth[v, u]
                c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)

                geometries = []
                print(f"DEBUG PROJECTION frame {global_idx}")

                for mode in modes:
                    world_xyz = backproject_mode_to_world(
                        u=u,
                        v=v,
                        depth_mm=d,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        c2w=c2w,
                        mode=mode,
                    )
                    _, world_normal, world_singular = estimate_plane_normal(world_xyz)
                    world_cam_forward = c2w[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
                    cos_angle = np.clip(
                        np.abs(np.dot(world_normal, world_cam_forward))
                        / (np.linalg.norm(world_normal) * np.linalg.norm(world_cam_forward) + 1e-8),
                        0.0,
                        1.0,
                    )
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))

                    print(
                        f"  mode={mode}: centroid={world_xyz.mean(axis=0).tolist()}, "
                        f"plane_sv={world_singular.tolist()}, normal_vs_forward_deg={angle_deg:.2f}"
                    )

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(world_xyz)
                    pcd.paint_uniform_color(mode_colors.get(mode, [0.7, 0.7, 0.7]))
                    geometries.append(pcd)

                cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam.transform(c2w)
                geometries.append(cam)
                o3d.visualization.draw_geometries(geometries)

        except Exception as exc:
            print(f"Warning: debug projection comparison failed: {exc}")

    if args.debug_batch_visualize:
        try:
            import open3d as o3d

            batch_size = max(1, args.debug_batch_size)
            debug_points_per_image = max(1, args.debug_batch_points_per_image)

            for batch_start in range(0, len(usable_frames), batch_size):
                batch_frames = usable_frames[batch_start : batch_start + batch_size]
                batch_xyz = []
                batch_rgb = []
                geometries = []

                for local_idx, frame in enumerate(batch_frames):
                    global_idx = batch_start + local_idx
                    depth_path = resolve_relpath(dataset_root, frame["depth_file_path"])
                    rgb_path = None
                    if "file_path" in frame:
                        rgb_path = resolve_relpath(dataset_root, frame["file_path"])

                    mask_path = None
                    if "mask_path" in frame:
                        mask_path = resolve_relpath(dataset_root, frame["mask_path"])

                    depth = read_image(depth_path)
                    if np.issubdtype(depth.dtype, np.floating):
                        depth = depth.astype(np.float32) * 1000.0
                    elif depth.dtype == np.uint16:
                        depth = depth.astype(np.float32)
                    else:
                        print(
                            f"DEBUG BATCH skip frame {global_idx}: unsupported depth dtype {depth.dtype}"
                        )
                        continue

                    h, w = depth.shape
                    fx, fy, cx, cy = get_intrinsics(frame, meta)
                    rgb_img = load_rgb(rgb_path, (h, w))
                    valid_mask = load_mask(mask_path, (h, w))

                    if rgb_img is None or valid_mask is None:
                        print(f"DEBUG BATCH skip frame {global_idx}: missing rgb or mask")
                        continue

                    valid_depth = (
                        (depth > 0)
                        & (depth >= args.depth_min_mm)
                        & (depth <= args.depth_max_mm)
                    )
                    valid = valid_mask & valid_depth
                    ys, xs = np.where(valid)
                    n_valid = xs.shape[0]
                    if n_valid == 0:
                        print(f"DEBUG BATCH skip frame {global_idx}: no valid points")
                        continue

                    n_sample = min(debug_points_per_image, n_valid)
                    choice = rng.choice(n_valid, size=n_sample, replace=False)
                    u = xs[choice]
                    v = ys[choice]
                    d = depth[v, u]

                    c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)
                    xyz_world = backproject_ns_camera_to_world(
                        u=u,
                        v=v,
                        depth_mm=d,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        c2w=c2w,
                    )
                    rgb_sel = rgb_img[v, u]

                    batch_xyz.append(xyz_world)
                    batch_rgb.append(rgb_sel)

                    cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    cam.transform(c2w)
                    geometries.append(cam)

                if not batch_xyz:
                    print(
                        f"DEBUG BATCH {batch_start}:{batch_start + len(batch_frames)} has no valid reconstructed points"
                    )
                    continue

                batch_xyz_np = np.concatenate(batch_xyz, axis=0)
                batch_rgb_np = np.concatenate(batch_rgb, axis=0)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(batch_xyz_np)
                pcd.colors = o3d.utility.Vector3dVector(batch_rgb_np.astype(np.float32) / 255.0)
                geometries.insert(0, pcd)

                print(
                    f"DEBUG BATCH frames {batch_start}..{batch_start + len(batch_frames) - 1}: "
                    f"frames={len(batch_frames)}, points={batch_xyz_np.shape[0]}"
                )
                o3d.visualization.draw_geometries(geometries)
        except Exception as exc:
            print(f"Warning: debug batch visualization failed: {exc}")

    if args.visualize:
        try:
            import open3d as o3d

            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)

            geometries = [pcd]

            # add camera frames
            frames_to_draw = usable_frames
            if args.max_visualize_frames > 0:
                frames_to_draw = usable_frames[: args.max_visualize_frames]

            pose_centers = []
            for frame in frames_to_draw:
                c2w = get_frame_c2w(frame, fix_live_optical_pose=args.fix_live_optical_pose)
                pose_centers.append(c2w[:3, 3])

                cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam.transform(c2w)
                geometries.append(cam)

            if pose_centers:
                pose_centers_np = np.stack(pose_centers, axis=0)
                pose_cloud = o3d.geometry.PointCloud()
                pose_cloud.points = o3d.utility.Vector3dVector(pose_centers_np)
                pose_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(pose_cloud)

            print(
                f"Visualizing {len(frames_to_draw)} camera poses out of {len(usable_frames)} used frames"
            )

            o3d.visualization.draw_geometries(geometries)
        except Exception as exc:
            print(f"Warning: open3d visualization failed or open3d not installed: {exc}")

    print(f"\nWrote point cloud: {output_path}")
    print(f"Total points: {xyz.shape[0]}")
    print(f"Samples per image: {samples_per_image}")
    # Report how many frames actually contributed points
    print(f"Frames used (with depth): {len(all_xyz)}")


if __name__ == "__main__":
    main()
