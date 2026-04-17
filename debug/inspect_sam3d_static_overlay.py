#!/usr/bin/env python3
"""Reconstruct the static scene and insert SAM3D using pose + D0 depth only.

This is an offline inspection utility for trying a shorter SAM3D insertion
path than the full fusion pipeline:
1. load the raw SAM3D gaussian output
2. apply SAM3D's own predicted pose
3. metric-correct it with the D0 live depth image and live object mask
4. place the corrected object into the static scene

Inputs:
    --data /path/to/dataset_root

Expected dataset layout:
    <dataset_root>/
        static_scene/
            rgb/
            depth/
            masks/               # optional but preferred
            transforms.json
        dynamic_scene/
            initialization_artifacts/   # new layout
            render_masks_esam/          # legacy layout

Outputs (default: <dataset_root>/sam3d_static_overlay/):
    - <frame>_static_scene_with_depth_corrected_sam3d_inserted.ply
    - manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 255], dtype=np.uint8)
ORANGE = np.array([255, 140, 0], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)
SAM3D_P3D_TO_NS_CAMERA = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing static_scene/ and dynamic_scene/")
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Dynamic frame stem to inspect. Default: first dynamic frame with a SAM3D raw output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data>/sam3d_static_overlay",
    )
    parser.add_argument(
        "--sam3d-dir",
        type=Path,
        default=None,
        help="Optional override directory containing a custom SAM3D raw_output.ply and matching pose.json.",
    )
    parser.add_argument("--max-total-points", type=int, default=800_000)
    parser.add_argument("--max-points-per-image", type=int, default=20_000)
    parser.add_argument("--depth-min-mm", type=float, default=100.0)
    parser.add_argument("--depth-max-mm", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--placement",
        type=str,
        choices=("depth_corrected", "sam3d_pose_only"),
        default="depth_corrected",
        help="How to place the SAM3D object into the scene.",
    )
    parser.add_argument(
        "--sam3d-translation-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the SAM3D pose translation before placement.",
    )
    parser.add_argument(
        "--sam3d-object-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the SAM3D object scale before placement.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def resolve_relpath(scene_root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return (scene_root / path).resolve()


def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if image is not None:
            return image
    return np.array(Image.open(path))


def load_rgb(path: Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if rgb.shape[:2] != expected_hw:
        raise ValueError(f"RGB shape mismatch for {path}: got {rgb.shape[:2]}, expected {expected_hw}")
    return rgb.astype(np.uint8)


def load_mask(path: Optional[Path], expected_hw: Tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones(expected_hw, dtype=bool)
    mask = np.array(Image.open(path).convert("L"))
    if mask.shape != expected_hw:
        raise ValueError(f"Mask shape mismatch for {path}: got {mask.shape}, expected {expected_hw}")
    return mask > 0


def load_depth_mm(path: Path) -> np.ndarray:
    depth = read_image(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if finite.size == 0:
            return depth.astype(np.float32)
        if float(np.nanmax(finite)) <= 10.0:
            depth = depth * 1000.0
        return depth.astype(np.float32)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32)
    raise ValueError(f"Unsupported depth dtype for {path}: {depth.dtype}")


def get_intrinsics(frame: Dict, meta: Dict) -> Tuple[float, float, float, float]:
    fx = frame.get("fl_x", meta.get("fl_x"))
    fy = frame.get("fl_y", meta.get("fl_y"))
    cx = frame.get("cx", meta.get("cx"))
    cy = frame.get("cy", meta.get("cy"))
    missing = [k for k, v in (("fl_x", fx), ("fl_y", fy), ("cx", cx), ("cy", cy)) if v is None]
    if missing:
        raise ValueError(f"Missing intrinsics in transforms.json: {missing}")
    return float(fx), float(fy), float(cx), float(cy)


def get_frame_c2w(frame: Dict) -> np.ndarray:
    pose_key = "depth_transform_matrix" if "depth_transform_matrix" in frame else "transform_matrix"
    c2w = np.asarray(frame[pose_key], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"{pose_key} must be 4x4, got {c2w.shape}")
    return c2w


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
    z = depth_mm.astype(np.float32) / 1000.0
    x = (u.astype(np.float32) - cx) * z / fx
    y = -(v.astype(np.float32) - cy) * z / fy
    cam_points = np.stack([x, y, -z, np.ones_like(z, dtype=np.float32)], axis=1)
    return (c2w @ cam_points.T).T[:, :3].astype(np.float32)


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        xyz.shape[0],
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertex["x"] = xyz[:, 0].astype(np.float32)
    vertex["y"] = xyz[:, 1].astype(np.float32)
    vertex["z"] = xyz[:, 2].astype(np.float32)
    vertex["red"] = rgb[:, 0].astype(np.uint8)
    vertex["green"] = rgb[:, 1].astype(np.uint8)
    vertex["blue"] = rgb[:, 2].astype(np.uint8)

    with path.open("wb") as handle:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertex)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def inspect_vertex_ply(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            header_lines.append(line.decode("ascii").rstrip())
            if line == b"end_header\n":
                break

        fmt = None
        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise RuntimeError(f"PLY list properties are not supported: {path}")
                ply_type = parts[1]
                prop_name = parts[2]
                if ply_type not in PLY_DTYPE_MAP:
                    raise RuntimeError(f"Unsupported PLY property type `{ply_type}` in {path}")
                properties.append((prop_name, PLY_DTYPE_MAP[ply_type]))

        if fmt is None or vertex_count is None:
            raise RuntimeError(f"Could not parse PLY header: {path}")

        meta = {
            "format": fmt,
            "vertex_count": int(vertex_count),
            "vertex_properties": [{"name": name, "dtype": dtype_code} for name, dtype_code in properties],
            "header_lines": header_lines,
        }

        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, f"<{dtype_code}") for name, dtype_code in properties])
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
            return {name: data[name] for name, _ in properties}, meta

        if fmt == "ascii":
            rows = np.loadtxt(handle, ndmin=2, max_rows=vertex_count)
            out: Dict[str, np.ndarray] = {}
            for idx, (name, dtype_code) in enumerate(properties):
                out[name] = rows[:, idx].astype(np.dtype(dtype_code))
            return out, meta

        raise RuntimeError(f"Unsupported PLY format `{fmt}` in {path}")


def _decode_sh_dc_to_rgb8(features_dc: np.ndarray) -> np.ndarray:
    rgb = np.clip(features_dc.astype(np.float32) * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def _extract_vertex_rgb8(vertex: Dict[str, np.ndarray]) -> tuple[Optional[np.ndarray], str]:
    if all(name in vertex for name in ("red", "green", "blue")):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)
        if np.issubdtype(rgb.dtype, np.floating):
            if float(np.nanmax(rgb)) <= 1.0:
                rgb = np.round(np.clip(rgb, 0.0, 1.0) * 255.0)
            else:
                rgb = np.round(np.clip(rgb, 0.0, 255.0))
        return rgb.astype(np.uint8), "rgb_fields"

    if all(name in vertex for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        return _decode_sh_dc_to_rgb8(features_dc), "f_dc_decode"

    return None, "none"


def _axis_like_mask(rgb8: Optional[np.ndarray], color_mode: str) -> np.ndarray:
    if rgb8 is None or len(rgb8) == 0 or color_mode != "rgb_fields":
        return np.zeros((0,), dtype=bool) if rgb8 is None else np.zeros((rgb8.shape[0],), dtype=bool)
    red = (rgb8[:, 0] >= 240) & (rgb8[:, 1] <= 20) & (rgb8[:, 2] <= 20)
    green = (rgb8[:, 1] >= 240) & (rgb8[:, 0] <= 20) & (rgb8[:, 2] <= 20)
    blue = (rgb8[:, 2] >= 240) & (rgb8[:, 0] <= 20) & (rgb8[:, 1] <= 20)
    return red | green | blue


def _tint_colors(rgb8: Optional[np.ndarray], tint: np.ndarray, color_mode: str) -> np.ndarray:
    if rgb8 is None:
        return np.repeat(tint[None, :], 0, axis=0).astype(np.uint8)
    tinted = np.repeat(tint[None, :], rgb8.shape[0], axis=0).astype(np.uint8)
    axis_mask = _axis_like_mask(rgb8, color_mode)
    if axis_mask.any():
        tinted[axis_mask] = rgb8[axis_mask]
    return tinted


def load_cloud_from_ply(path: Path) -> Dict[str, object]:
    vertex, meta = inspect_vertex_ply(path)
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb8, color_mode = _extract_vertex_rgb8(vertex)
    axis_mask = _axis_like_mask(rgb8, color_mode)
    return {
        "xyz": xyz,
        "rgb8": rgb8,
        "color_mode": color_mode,
        "vertex_meta": meta,
        "vertex_names": list(vertex.keys()),
        "axis_like_count": int(axis_mask.sum()) if axis_mask.size > 0 else 0,
    }


def _default_cloud_rgb(point_count: int, color: np.ndarray) -> np.ndarray:
    return np.repeat(color[None, :], point_count, axis=0).astype(np.uint8)


def quaternion_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion.astype(np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
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


def apply_similarity_pose(
    xyz: np.ndarray,
    scale: float,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    rotation = quaternion_wxyz_to_rotation_matrix(rotation_wxyz)
    return (float(scale) * (xyz @ rotation) + translation.reshape(1, 3)).astype(np.float32)


def sam3d_pose_camera_to_ns_camera(xyz: np.ndarray) -> np.ndarray:
    """Convert SAM3D/PyTorch3D camera coordinates to this repo's NS camera frame.

    SAM3D's pose/gaussian stack uses PyTorch3D camera coordinates after the
    R3/OpenCV point map is rotated into PyTorch3D. The dataset cameras used in
    dynamic_gs backprojection instead expect:
    - +X right
    - +Y up
    - -Z forward

    PyTorch3D camera coordinates are:
    - +X left
    - +Y up
    - +Z forward

    So the fixed conversion is a flip on X and Z.
    """
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (xyz.astype(np.float32) @ SAM3D_P3D_TO_NS_CAMERA).astype(np.float32)


def transform_points_homogeneous(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    homog = np.concatenate([xyz.astype(np.float32), np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    return (transform.astype(np.float32) @ homog.T).T[:, :3].astype(np.float32)


def resolve_sam3d_pose_path(raw_ply_path: Path, frame_name: str) -> Optional[Path]:
    candidates = []
    if raw_ply_path.name.endswith("_raw_output.ply"):
        candidates.append(raw_ply_path.with_name(raw_ply_path.name[: -len("_raw_output.ply")] + "_pose.json"))
    candidates.extend(
        [
            raw_ply_path.parent / f"{frame_name}_sam3d_pose.json",
            raw_ply_path.parent / f"{frame_name}_d0_true_sam3d_pose.json",
        ]
    )
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def load_sam3d_pose(path: Path) -> Dict[str, np.ndarray]:
    payload = read_json(path)
    return {
        "translation": np.asarray(payload["translation"], dtype=np.float32).reshape(3),
        "rotation": np.asarray(payload["rotation"], dtype=np.float32).reshape(4),
        "scale": np.asarray(payload["scale"], dtype=np.float32).reshape(-1),
    }


def get_dynamic_frame_c2w(dataset_root: Path, frame_name: str) -> np.ndarray:
    dynamic_meta = read_json(dataset_root / "dynamic_scene" / "transforms.json")
    for frame in dynamic_meta.get("frames", []):
        file_path = frame.get("file_path")
        if file_path is None:
            continue
        if Path(file_path).stem == frame_name:
            return get_frame_c2w(frame)
    raise FileNotFoundError(f"Could not find dynamic frame `{frame_name}` in dynamic_scene/transforms.json")


def compare_cloud_geometry(reference_xyz: np.ndarray, query_xyz: np.ndarray) -> Dict[str, object]:
    if reference_xyz.size == 0 or query_xyz.size == 0:
        return {
            "centroid_delta": None,
            "centroid_distance": None,
            "extent_ratio_query_over_reference": None,
        }
    ref_centroid = reference_xyz.mean(axis=0)
    query_centroid = query_xyz.mean(axis=0)
    ref_extent = np.maximum(reference_xyz.max(axis=0) - reference_xyz.min(axis=0), 1e-8)
    query_extent = query_xyz.max(axis=0) - query_xyz.min(axis=0)
    delta = query_centroid - ref_centroid
    return {
        "centroid_delta": delta.round(6).tolist(),
        "centroid_distance": float(np.linalg.norm(delta)),
        "extent_ratio_query_over_reference": (query_extent / ref_extent).round(6).tolist(),
    }


def resolve_static_mask_path(scene_root: Path, frame: Dict) -> Optional[Path]:
    mask_rel = frame.get("mask_path")
    if mask_rel is not None:
        path = resolve_relpath(scene_root, mask_rel)
        if path.exists():
            return path
    file_rel = frame.get("file_path")
    if file_rel is None:
        return None
    fallback = scene_root / "masks" / Path(file_rel).name
    return fallback if fallback.exists() else None


def reconstruct_static_scene(
    scene_root: Path,
    max_total_points: int,
    max_points_per_image: int,
    depth_min_mm: float,
    depth_max_mm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = read_json(scene_root / "transforms.json")
    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames found in {scene_root / 'transforms.json'}")

    usable: list[Dict] = []
    for frame in frames:
        if frame.get("file_path") is None or frame.get("depth_file_path") is None:
            continue
        usable.append(frame)
    if not usable:
        raise RuntimeError("No usable static frames with rgb+depth paths were found.")

    samples_per_image = min(max_points_per_image, max(1, math.ceil(max_total_points / max(len(usable), 1))))
    rng = np.random.default_rng(seed)
    all_xyz = []
    all_rgb = []
    used_frames = 0

    for frame in usable:
        rgb_path = resolve_relpath(scene_root, frame["file_path"])
        depth_path = resolve_relpath(scene_root, frame["depth_file_path"])
        mask_path = resolve_static_mask_path(scene_root, frame)

        depth = load_depth_mm(depth_path)
        h, w = depth.shape
        rgb = load_rgb(rgb_path, (h, w))
        valid_mask = load_mask(mask_path, (h, w))

        fx, fy, cx, cy = get_intrinsics(frame, meta)
        c2w = get_frame_c2w(frame)

        valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth >= depth_min_mm) & (depth <= depth_max_mm)
        valid = valid_mask & valid_depth
        ys, xs = np.where(valid)
        if xs.size == 0:
            continue

        n_sample = min(samples_per_image, xs.size)
        choice = rng.choice(xs.size, size=n_sample, replace=False)
        u = xs[choice]
        v = ys[choice]
        d = depth[v, u]

        xyz_world = backproject_ns_camera_to_world(u, v, d, fx, fy, cx, cy, c2w)
        rgb_sel = rgb[v, u]
        all_xyz.append(xyz_world)
        all_rgb.append(rgb_sel)
        used_frames += 1

    if not all_xyz:
        raise RuntimeError("No valid static-scene points were generated.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    if xyz.shape[0] > max_total_points:
        keep = rng.choice(xyz.shape[0], size=max_total_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]

    stats = {
        "usable_frames": int(len(usable)),
        "used_frames": int(used_frames),
        "samples_per_image": int(samples_per_image),
        "point_count": int(xyz.shape[0]),
    }
    return xyz.astype(np.float32), rgb.astype(np.uint8), stats


def first_dynamic_frame_name(dataset_root: Path) -> str:
    dynamic_meta = read_json(dataset_root / "dynamic_scene" / "transforms.json")
    frames = dynamic_meta.get("frames", [])
    if not frames:
        raise RuntimeError("No dynamic frames found.")
    return Path(frames[0]["file_path"]).stem


def find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_artifact_dir(dataset_root: Path) -> Path:
    dynamic_scene = dataset_root / "dynamic_scene"
    for candidate in [
        dynamic_scene / "initialization_artifacts",
        dynamic_scene / "render_masks_esam",
        dynamic_scene / "initialization_debug",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find SAM3D artifact directory in dynamic_scene/")


def resolve_sam3d_frame_and_paths(dataset_root: Path, frame_name: Optional[str]) -> tuple[str, Path, Optional[Path]]:
    artifact_dir = resolve_artifact_dir(dataset_root)

    def raw_candidates(frame: str) -> list[Path]:
        return [
            artifact_dir / f"{frame}_d0_true_sam3d_raw_output.ply",
            artifact_dir / f"{frame}_sam3d_raw_output.ply",
        ]

    if frame_name is not None:
        raw_path = find_first_existing(raw_candidates(frame_name))
        if raw_path is None:
            raise FileNotFoundError(f"Could not find SAM3D raw output for frame `{frame_name}` in {artifact_dir}")
        aligned_path = artifact_dir / f"{frame_name}_sam3d_aligned_output.ply"
        return frame_name, raw_path, aligned_path if aligned_path.exists() else None

    default_frame = first_dynamic_frame_name(dataset_root)
    raw_path = find_first_existing(raw_candidates(default_frame))
    if raw_path is not None:
        aligned_path = artifact_dir / f"{default_frame}_sam3d_aligned_output.ply"
        return default_frame, raw_path, aligned_path if aligned_path.exists() else None

    for candidate in sorted(artifact_dir.glob("*_d0_true_sam3d_raw_output.ply")):
        stem = candidate.name[: -len("_d0_true_sam3d_raw_output.ply")]
        aligned_path = artifact_dir / f"{stem}_sam3d_aligned_output.ply"
        return stem, candidate, aligned_path if aligned_path.exists() else None
    for candidate in sorted(artifact_dir.glob("*_sam3d_raw_output.ply")):
        stem = candidate.name[: -len("_sam3d_raw_output.ply")]
        aligned_path = artifact_dir / f"{stem}_sam3d_aligned_output.ply"
        return stem, candidate, aligned_path if aligned_path.exists() else None

    raise FileNotFoundError(f"No SAM3D raw output .ply found in {artifact_dir}")


def resolve_custom_sam3d_frame_and_paths(sam3d_dir: Path, frame_name: Optional[str]) -> tuple[str, Path, Optional[Path]]:
    sam3d_dir = Path(sam3d_dir).expanduser().resolve()
    if not sam3d_dir.exists():
        raise FileNotFoundError(sam3d_dir)

    raw_candidates = sorted(sam3d_dir.glob("*_raw_output.ply"))
    if not raw_candidates:
        raise FileNotFoundError(f"No `*_raw_output.ply` found in {sam3d_dir}")

    selected_raw: Optional[Path] = None
    selected_frame: Optional[str] = None
    for candidate in raw_candidates:
        stem = candidate.name[: -len("_raw_output.ply")]
        frame_prefix = stem.split("_sam3d", 1)[0]
        if frame_name is None or frame_prefix == frame_name:
            selected_raw = candidate
            selected_frame = frame_prefix
            break

    if selected_raw is None or selected_frame is None:
        raise FileNotFoundError(f"Could not find a SAM3D raw output matching frame `{frame_name}` in {sam3d_dir}")

    aligned_path = selected_raw.with_name(f"{selected_frame}_sam3d_aligned_output.ply")
    return selected_frame, selected_raw, aligned_path if aligned_path.exists() else None


def resolve_dynamic_frame(dataset_root: Path, frame_name: str) -> tuple[Dict, Dict]:
    dynamic_meta = read_json(dataset_root / "dynamic_scene" / "transforms.json")
    for frame in dynamic_meta.get("frames", []):
        file_path = frame.get("file_path")
        if file_path is None:
            continue
        if Path(file_path).stem == frame_name:
            return frame, dynamic_meta
    raise FileNotFoundError(f"Could not find dynamic frame `{frame_name}` in dynamic_scene/transforms.json")


def resolve_live_object_mask_path(dataset_root: Path, artifact_dir: Path, frame_name: str) -> Path:
    dynamic_scene = dataset_root / "dynamic_scene"
    candidates = [
        artifact_dir / f"{frame_name}_live_object_mask.png",
        dynamic_scene / "initialization_debug" / f"{frame_name}_live_object_mask.png",
        dynamic_scene / "render_masks_esam" / f"{frame_name}_live_object_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find `{frame_name}_live_object_mask.png` in initialization/debug artifact folders."
    )


def robust_bounds(points: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return zeros, zeros
    lower = np.percentile(points, lower_q, axis=0).astype(np.float32)
    upper = np.percentile(points, upper_q, axis=0).astype(np.float32)
    return lower, upper


def robust_extent(points: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
    lower, upper = robust_bounds(points, lower_q=lower_q, upper_q=upper_q)
    return (upper - lower).astype(np.float32)


def robust_centroid(points: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    lower, upper = robust_bounds(points, lower_q=lower_q, upper_q=upper_q)
    keep = np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)
    if not np.any(keep):
        return points.mean(axis=0).astype(np.float32)
    return points[keep].mean(axis=0).astype(np.float32)


def fusion_bbox_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e-3
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extents).clip(min=1e-6))


def fusion_centroid(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return points.mean(axis=0).astype(np.float32)


def fit_bbox_centroid_initialized_sam3d_in_camera(
    source_points_local: np.ndarray,
    target_points_camera_ns: np.ndarray,
) -> Dict[str, object]:
    if len(source_points_local) == 0:
        raise ValueError("SAM3D raw source cloud is empty.")
    if len(target_points_camera_ns) < 3:
        raise ValueError("Need at least 3 target points for bbox/centroid initialization.")

    source_diag = fusion_bbox_diagonal(source_points_local)
    target_diag = fusion_bbox_diagonal(target_points_camera_ns)
    chosen_scale = float(target_diag / max(source_diag, 1e-6))

    source_centroid = fusion_centroid(source_points_local)
    target_centroid = fusion_centroid(target_points_camera_ns)
    aligned_source = target_centroid[None, :] + chosen_scale * (source_points_local - source_centroid[None, :])
    return {
        "aligned_source_camera_ns": aligned_source.astype(np.float32),
        "chosen_scale": chosen_scale,
        "source_centroid": source_centroid.astype(np.float32),
        "target_centroid": target_centroid.astype(np.float32),
        "source_diag": float(source_diag),
        "target_diag": float(target_diag),
    }


def project_ns_points_to_image(
    points_ns: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_hw
    if len(points_ns) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=bool)
    z_forward = -points_ns[:, 2]
    valid = np.isfinite(z_forward) & (z_forward > 1e-6)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), valid
    x = points_ns[valid, 0]
    y = points_ns[valid, 1]
    z = z_forward[valid]
    u = fx * (x / z) + cx
    v = cy - fy * (y / z)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_image = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui[in_image], vi[in_image], valid


def fit_depth_corrected_sam3d_in_camera(
    source_pose_camera: np.ndarray,
    target_depth_camera_ns: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Dict[str, object]:
    if len(source_pose_camera) == 0:
        raise ValueError("SAM3D source cloud is empty after pose application.")
    if len(target_depth_camera_ns) < 16:
        raise ValueError("Not enough D0 object depth points for depth correction.")

    target_extent = robust_extent(target_depth_camera_ns)
    target_centroid = robust_centroid(target_depth_camera_ns)
    image_hw = mask.shape
    best: Optional[Dict[str, object]] = None

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                sign_flip = np.array([sx, sy, sz], dtype=np.float32)
                source_ns = source_pose_camera * sign_flip[None, :]
                source_extent = robust_extent(source_ns)

                scale_candidates = [
                    float(target_extent[axis] / source_extent[axis])
                    for axis in (0, 1)
                    if float(abs(source_extent[axis])) > 1e-6 and float(abs(target_extent[axis])) > 1e-6
                ]
                if not scale_candidates:
                    continue

                depth_scale = float(np.median(scale_candidates))
                scaled_source = depth_scale * source_ns
                translation_delta = (target_centroid - robust_centroid(scaled_source)).astype(np.float32)
                corrected_source = scaled_source + translation_delta[None, :]

                ui, vi, _ = project_ns_points_to_image(corrected_source, fx, fy, cx, cy, image_hw)
                if ui.size == 0:
                    in_mask_ratio = 0.0
                    bbox = None
                else:
                    in_mask_ratio = float(mask[vi, ui].mean())
                    bbox = [int(ui.min()), int(vi.min()), int(ui.max()), int(vi.max())]

                candidate = {
                    "corrected_source_camera_ns": corrected_source.astype(np.float32),
                    "sign_flip_camera_to_ns": sign_flip.astype(np.float32),
                    "depth_scale_correction": depth_scale,
                    "translation_delta_camera_ns": translation_delta.astype(np.float32),
                    "projected_in_mask_ratio": in_mask_ratio,
                    "projected_bbox_xyxy": bbox,
                }
                if best is None or candidate["projected_in_mask_ratio"] > best["projected_in_mask_ratio"]:
                    best = candidate

    if best is None:
        raise RuntimeError("Failed to find a valid SAM3D depth-correction candidate.")
    return best


def summarize_cloud(xyz: np.ndarray) -> Dict[str, object]:
    if xyz.size == 0:
        return {
            "point_count": 0,
            "centroid": None,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
        }

    xyz = xyz.astype(np.float64)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    return {
        "point_count": int(xyz.shape[0]),
        "centroid": xyz.mean(axis=0).round(6).tolist(),
        "bbox_min": bbox_min.round(6).tolist(),
        "bbox_max": bbox_max.round(6).tolist(),
        "bbox_extent": (bbox_max - bbox_min).round(6).tolist(),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.data.expanduser().resolve()
    static_scene = dataset_root / "static_scene"
    dynamic_scene = dataset_root / "dynamic_scene"
    if not static_scene.exists():
        raise FileNotFoundError(static_scene)
    if not dynamic_scene.exists():
        raise FileNotFoundError(dynamic_scene)

    output_dir = (args.output_dir or (dataset_root / "sam3d_static_overlay")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_ply in output_dir.glob("*.ply"):
        stale_ply.unlink()

    if args.sam3d_dir is not None:
        frame_name, raw_ply_path, aligned_ply_path = resolve_custom_sam3d_frame_and_paths(args.sam3d_dir, args.frame)
        artifact_dir = Path(args.sam3d_dir).expanduser().resolve()
    else:
        frame_name, raw_ply_path, aligned_ply_path = resolve_sam3d_frame_and_paths(dataset_root, args.frame)
        artifact_dir = resolve_artifact_dir(dataset_root)
    del aligned_ply_path
    dynamic_frame, dynamic_meta = resolve_dynamic_frame(dataset_root, frame_name)
    pose_path = resolve_sam3d_pose_path(raw_ply_path, frame_name)
    if pose_path is None:
        raise FileNotFoundError(f"Could not find SAM3D pose sidecar for `{frame_name}` next to {raw_ply_path}")
    live_object_mask_path = resolve_live_object_mask_path(dataset_root, artifact_dir, frame_name)

    depth_path = resolve_relpath(dataset_root / "dynamic_scene", dynamic_frame["depth_file_path"])
    depth_mm = load_depth_mm(depth_path)
    live_object_mask = load_mask(live_object_mask_path, depth_mm.shape)
    fx, fy, cx, cy = get_intrinsics(dynamic_frame, dynamic_meta)
    dynamic_c2w = get_frame_c2w(dynamic_frame)
    valid_depth = (
        np.isfinite(depth_mm)
        & (depth_mm > 0.0)
        & (depth_mm >= args.depth_min_mm)
        & (depth_mm <= args.depth_max_mm)
    )
    live_depth_valid = live_object_mask & valid_depth
    ys, xs = np.where(live_depth_valid)
    if xs.size == 0:
        raise RuntimeError("No valid D0 live object depth pixels were found.")
    target_depth_camera_ns = backproject_ns_camera_to_world(
        xs,
        ys,
        depth_mm[ys, xs],
        fx,
        fy,
        cx,
        cy,
        np.eye(4, dtype=np.float32),
    )

    static_xyz, static_rgb, static_stats = reconstruct_static_scene(
        static_scene,
        max_total_points=args.max_total_points,
        max_points_per_image=args.max_points_per_image,
        depth_min_mm=args.depth_min_mm,
        depth_max_mm=args.depth_max_mm,
        seed=args.seed,
    )

    raw_cloud = load_cloud_from_ply(raw_ply_path)
    raw_xyz = raw_cloud["xyz"]
    raw_original_rgb = raw_cloud["rgb8"]
    pose_payload = load_sam3d_pose(pose_path)
    pose_scale_raw = float(pose_payload["scale"][0]) if pose_payload["scale"].size > 0 else 1.0
    pose_scale = pose_scale_raw * float(args.sam3d_object_scale)
    sam3d_translation = pose_payload["translation"] * float(args.sam3d_translation_scale)
    source_pose_camera_p3d = apply_similarity_pose(
        raw_xyz,
        scale=pose_scale,
        rotation_wxyz=pose_payload["rotation"],
        translation=sam3d_translation,
    )
    source_pose_camera_ns = sam3d_pose_camera_to_ns_camera(source_pose_camera_p3d)
    pose_only_world = transform_points_homogeneous(source_pose_camera_ns, dynamic_c2w)

    bbox_init_fit = fit_bbox_centroid_initialized_sam3d_in_camera(
        source_points_local=raw_xyz,
        target_points_camera_ns=target_depth_camera_ns,
    )
    bbox_init_camera_ns = bbox_init_fit["aligned_source_camera_ns"]
    bbox_init_world = transform_points_homogeneous(bbox_init_camera_ns, dynamic_c2w)

    source_orientation_only_camera_p3d = apply_similarity_pose(
        raw_xyz,
        scale=1.0,
        rotation_wxyz=pose_payload["rotation"],
        translation=np.zeros(3, dtype=np.float32),
    )
    source_orientation_only_camera_ns = sam3d_pose_camera_to_ns_camera(source_orientation_only_camera_p3d)
    mixed_init_fit = fit_bbox_centroid_initialized_sam3d_in_camera(
        source_points_local=source_orientation_only_camera_ns,
        target_points_camera_ns=target_depth_camera_ns,
    )
    mixed_init_camera_ns = mixed_init_fit["aligned_source_camera_ns"]
    mixed_init_world = transform_points_homogeneous(mixed_init_camera_ns, dynamic_c2w)

    scalar_metric_scale = float(bbox_init_fit["chosen_scale"])
    sam3d_translation_metric_ns = sam3d_pose_camera_to_ns_camera(
        (sam3d_translation * scalar_metric_scale).reshape(1, 3)
    )[0]
    sam3d_scaled_pose_camera_ns = (
        scalar_metric_scale * source_orientation_only_camera_ns + sam3d_translation_metric_ns[None, :]
    ).astype(np.float32)
    sam3d_scaled_pose_world = transform_points_homogeneous(sam3d_scaled_pose_camera_ns, dynamic_c2w)

    depth_fit = None
    corrected_source_camera_ns = source_pose_camera_ns
    corrected_source_world = pose_only_world
    placement_mode = args.placement
    if placement_mode == "depth_corrected":
        depth_fit = fit_depth_corrected_sam3d_in_camera(
            source_pose_camera=source_pose_camera_p3d,
            target_depth_camera_ns=target_depth_camera_ns,
            mask=live_object_mask,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        corrected_source_camera_ns = depth_fit["corrected_source_camera_ns"]
        corrected_source_world = transform_points_homogeneous(corrected_source_camera_ns, dynamic_c2w)

    object_rgb = raw_original_rgb if raw_original_rgb is not None else _default_cloud_rgb(raw_xyz.shape[0], ORANGE)
    bbox_init_rgb = _tint_colors(raw_original_rgb, GREEN, raw_cloud["color_mode"])
    if bbox_init_rgb.shape[0] != raw_xyz.shape[0]:
        bbox_init_rgb = _default_cloud_rgb(raw_xyz.shape[0], GREEN)
    mixed_init_rgb = _tint_colors(raw_original_rgb, RED, raw_cloud["color_mode"])
    if mixed_init_rgb.shape[0] != raw_xyz.shape[0]:
        mixed_init_rgb = _default_cloud_rgb(raw_xyz.shape[0], RED)
    scaled_pose_rgb = _tint_colors(raw_original_rgb, BLUE, raw_cloud["color_mode"])
    if scaled_pose_rgb.shape[0] != raw_xyz.shape[0]:
        scaled_pose_rgb = _default_cloud_rgb(raw_xyz.shape[0], BLUE)
    if placement_mode == "depth_corrected":
        inserted_scene_name = f"{frame_name}_static_scene_with_depth_corrected_sam3d_inserted.ply"
    else:
        inserted_scene_name = f"{frame_name}_static_scene_with_sam3d_pose_only_inserted.ply"
    inserted_scene_path = output_dir / inserted_scene_name
    combined_scene_xyz = np.concatenate(
        [static_xyz, corrected_source_world, bbox_init_world, mixed_init_world, sam3d_scaled_pose_world], axis=0
    )
    combined_scene_rgb = np.concatenate(
        [static_rgb, object_rgb, bbox_init_rgb, mixed_init_rgb, scaled_pose_rgb], axis=0
    )
    write_binary_ply(
        inserted_scene_path,
        combined_scene_xyz,
        combined_scene_rgb,
    )

    manifest = {
        "dataset_root": str(dataset_root),
        "frame_name": frame_name,
        "placement_mode": placement_mode,
        "inserted_scene_ply": str(inserted_scene_path),
        "live_object_mask_path": str(live_object_mask_path),
        "dynamic_depth_path": str(depth_path),
        "sam3d_raw_source_ply": str(raw_ply_path),
        "sam3d_pose_source_json": str(pose_path),
        "counts": {
            "static_scene_points": int(static_xyz.shape[0]),
            "sam3d_raw_points": int(raw_xyz.shape[0]),
            "d0_live_depth_points": int(target_depth_camera_ns.shape[0]),
            "sam3d_pose_inserted_points": int(corrected_source_world.shape[0]),
            "bbox_initialized_inserted_points": int(bbox_init_world.shape[0]),
            "sam3d_orientation_bbox_inserted_points": int(mixed_init_world.shape[0]),
            "sam3d_scaled_translation_inserted_points": int(sam3d_scaled_pose_world.shape[0]),
            "combined_points": int(combined_scene_xyz.shape[0]),
        },
        "static_scene_stats": static_stats,
        "sam3d_structures": {
            "raw_output": {
                "color_mode": raw_cloud["color_mode"],
                "axis_like_count": int(raw_cloud["axis_like_count"]),
                "vertex_names": raw_cloud["vertex_names"],
                "vertex_meta": raw_cloud["vertex_meta"],
            },
        },
        "sam3d_pose_sidecar": {
            "path": str(pose_path),
            "translation": pose_payload["translation"].round(6).tolist(),
            "translation_after_scale": sam3d_translation.round(6).tolist(),
            "translation_scale_factor": float(args.sam3d_translation_scale),
            "rotation_wxyz": pose_payload["rotation"].round(6).tolist(),
            "scale": pose_payload["scale"].round(6).tolist(),
            "uniform_scale_after_factor": float(round(pose_scale, 6)),
            "object_scale_factor": float(args.sam3d_object_scale),
            "camera_convention_after_pose": "pytorch3d_camera",
            "dataset_camera_conversion_applied": SAM3D_P3D_TO_NS_CAMERA.round(6).tolist(),
        },
        "depth_correction": None
        if depth_fit is None
        else {
            "sign_flip_camera_to_ns": np.asarray(depth_fit["sign_flip_camera_to_ns"]).round(6).tolist(),
            "depth_scale_correction": float(depth_fit["depth_scale_correction"]),
            "translation_delta_camera_ns": np.asarray(depth_fit["translation_delta_camera_ns"]).round(6).tolist(),
            "projected_in_mask_ratio": float(depth_fit["projected_in_mask_ratio"]),
            "projected_bbox_xyxy": depth_fit["projected_bbox_xyxy"],
        },
        "bbox_initialization": {
            "description": "Current fusion initialization before CPD: bbox-diagonal scale plus centroid alignment.",
            "chosen_scale": float(bbox_init_fit["chosen_scale"]),
            "source_centroid": np.asarray(bbox_init_fit["source_centroid"]).round(6).tolist(),
            "target_centroid": np.asarray(bbox_init_fit["target_centroid"]).round(6).tolist(),
            "source_diag": float(bbox_init_fit["source_diag"]),
            "target_diag": float(bbox_init_fit["target_diag"]),
            "bbox_init_color_rgb": GREEN.tolist(),
        },
        "sam3d_orientation_bbox_initialization": {
            "description": "SAM3D orientation only, then current fusion initialization before CPD: bbox-diagonal scale plus centroid alignment.",
            "chosen_scale": float(mixed_init_fit["chosen_scale"]),
            "source_centroid": np.asarray(mixed_init_fit["source_centroid"]).round(6).tolist(),
            "target_centroid": np.asarray(mixed_init_fit["target_centroid"]).round(6).tolist(),
            "source_diag": float(mixed_init_fit["source_diag"]),
            "target_diag": float(mixed_init_fit["target_diag"]),
            "mixed_init_color_rgb": RED.tolist(),
        },
        "sam3d_scaled_translation_initialization": {
            "description": "Scalar metric scale from current bbox initialization, with SAM3D rotation and SAM3D translation scaled by the same factor.",
            "scalar_metric_scale": float(scalar_metric_scale),
            "scaled_translation_p3d": (sam3d_translation * scalar_metric_scale).round(6).tolist(),
            "scaled_translation_ns": sam3d_translation_metric_ns.round(6).tolist(),
            "scaled_translation_color_rgb": BLUE.tolist(),
        },
        "geometry": {
            "static_scene": summarize_cloud(static_xyz),
            "sam3d_pose_camera_p3d_raw": summarize_cloud(source_pose_camera_p3d),
            "sam3d_pose_camera_ns_raw": summarize_cloud(source_pose_camera_ns),
            "sam3d_pose_world_raw": summarize_cloud(pose_only_world),
            "sam3d_bbox_initialized_camera_ns": summarize_cloud(bbox_init_camera_ns),
            "sam3d_bbox_initialized_world": summarize_cloud(bbox_init_world),
            "sam3d_orientation_only_camera_p3d": summarize_cloud(source_orientation_only_camera_p3d),
            "sam3d_orientation_only_camera_ns": summarize_cloud(source_orientation_only_camera_ns),
            "sam3d_orientation_bbox_initialized_camera_ns": summarize_cloud(mixed_init_camera_ns),
            "sam3d_orientation_bbox_initialized_world": summarize_cloud(mixed_init_world),
            "sam3d_scaled_translation_camera_ns": summarize_cloud(sam3d_scaled_pose_camera_ns),
            "sam3d_scaled_translation_world": summarize_cloud(sam3d_scaled_pose_world),
            "d0_live_depth_camera_ns": summarize_cloud(target_depth_camera_ns),
            "sam3d_depth_corrected_camera_ns": summarize_cloud(corrected_source_camera_ns),
            "sam3d_depth_corrected_world": summarize_cloud(corrected_source_world),
            "combined_inserted_scene": summarize_cloud(combined_scene_xyz),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"frame={frame_name}")
    print(f"inserted_scene={inserted_scene_path}")
    print(f"manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
