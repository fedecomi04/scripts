"""Regenerate the static-scene initial point cloud from depth + masks only.

The shipped ``depth_camera_init_points.ply`` for some datasets was built
from depth across BOTH static and dynamic frames, so the moved object
appears multiple times in the seed cloud (a "ghost trail"). This script
rebuilds the seed cloud from **only** the static-scene depth images,
applies the existing gripper mask AND a simulator-background mask
(pixels matching the Gazebo bg color), and downsamples to a fixed
total point count.

Usage
-----
    python regenerate_static_init_ply.py \
        --root /path/to/dataset_root \
        --num-points 500000

The script overwrites
``<root>/static_scene/depth_camera_init_points.ply`` after first
backing up the existing file to ``...points.ply.bak``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from plyfile import PlyData, PlyElement


_FRAME_ID_RE = re.compile(r"(\d+)")


def _frame_id(file_path: str) -> int | None:
    """Extract the numeric ID from filenames like ``arm_14864.png``."""
    stem = Path(file_path).stem
    m = _FRAME_ID_RE.search(stem)
    return int(m.group(1)) if m else None


# Simulator background color (uint8): matches DynamicGSModelConfig.simulator_background_rgb
# = (0.86, 0.92, 1.0). Tolerance accounts for JPEG/PNG round-trip noise.
SIMULATOR_BG_RGB = np.array([0.86, 0.92, 1.0], dtype=np.float32) * 255.0
DEFAULT_BG_TOLERANCE = 8.0


def _load_depth_meters(path: Path, depth_unit_scale: float = 1e-3) -> np.ndarray:
    raw = tifffile.imread(path)
    if raw.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth, got {raw.dtype} for {path}")
    return raw.astype(np.float32) * depth_unit_scale


def _load_mask(path: Path) -> np.ndarray:
    """Returns boolean mask: True where the pixel is *kept* (not gripper)."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def _bg_mask(rgb_uint8: np.ndarray, tol: float) -> np.ndarray:
    """Returns boolean mask: True where pixel is background (to be excluded)."""
    diff = rgb_uint8.astype(np.float32) - SIMULATOR_BG_RGB[None, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist <= tol


def _backproject(
    depth_m: np.ndarray,
    valid: np.ndarray,
    rgb_uint8: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Same OpenGL convention used by the live pipeline (see
    DynamicGSPipeline._backproject_mask_to_world): x_cam = (u-cx)/fx*z,
    y_cam = -(v-cy)/fy*z, z_cam = -depth.
    """
    ys, xs = np.where(valid)
    if xs.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    z = depth_m[ys, xs]
    x_cam = (xs.astype(np.float32) - cx) / fx * z
    y_cam = -(ys.astype(np.float32) - cy) / fy * z
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    pts_world = pts_cam @ R.T + t[None, :]
    colors = rgb_uint8[ys, xs]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def _save_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    n = points.shape[0]
    arr = np.zeros(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(out_path))


def regenerate(
    root: Path,
    num_points: int = 500_000,
    bg_tolerance: float = DEFAULT_BG_TOLERANCE,
    depth_unit_scale: float = 1e-3,
    seed: int = 0,
    max_frame_id: int | None = None,
    min_frame_id: int | None = None,
    backup_existing: bool = True,
) -> Path:
    """Rebuild ``<root>/static_scene/depth_camera_init_points.ply`` from depth
    + masks of the static-scene frames only. Returns the output PLY path.

    Used both as a standalone CLI (see ``main`` below) and from
    ``prepare_pipeline_split_datasets.py`` so newly split datasets get a
    static-only init cloud automatically.
    """
    root = Path(root).expanduser().resolve()
    static_root = root / "static_scene"
    transforms_path = static_root / "transforms.json"
    out_path = static_root / "depth_camera_init_points.ply"
    backup_path = out_path.with_suffix(out_path.suffix + ".bak")

    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)

    meta = json.loads(transforms_path.read_text())
    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    frames = meta["frames"]
    raw_count = len(frames)
    if max_frame_id is not None or min_frame_id is not None:
        kept_frames = []
        skipped = []
        for f in frames:
            fid = _frame_id(f["file_path"])
            if fid is None:
                kept_frames.append(f)
                continue
            if max_frame_id is not None and fid > max_frame_id:
                skipped.append(fid)
                continue
            if min_frame_id is not None and fid < min_frame_id:
                skipped.append(fid)
                continue
            kept_frames.append(f)
        frames = kept_frames
        print(
            f"Frame-id filter: kept {len(frames)}/{raw_count} "
            f"(min={min_frame_id}, max={max_frame_id}); "
            f"dropped {len(skipped)} ids: "
            f"{skipped[:3]}{'...' if len(skipped) > 6 else ''}{skipped[-3:] if len(skipped) > 3 else ''}"
        )
    print(f"Static frames: {len(frames)}; intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    rng = np.random.default_rng(seed)
    all_pts_chunks: list[np.ndarray] = []
    all_col_chunks: list[np.ndarray] = []
    n_total_valid = 0
    n_dropped_gripper = 0
    n_dropped_bg = 0
    n_dropped_zero = 0

    for fi, frame in enumerate(frames):
        depth_path = static_root / frame["depth_file_path"].lstrip("./")
        rgb_path = static_root / frame["file_path"].lstrip("./")
        mask_path = static_root / frame["mask_path"].lstrip("./")
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"frame {fi} has bad transform_matrix shape {c2w.shape}")

        depth_m = _load_depth_meters(depth_path, depth_unit_scale)
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        keep_gripper = _load_mask(mask_path)
        bg_mask = _bg_mask(rgb, bg_tolerance)
        depth_valid = depth_m > 0

        valid = keep_gripper & (~bg_mask) & depth_valid

        n_dropped_gripper += int((~keep_gripper).sum())
        n_dropped_bg += int(bg_mask.sum())
        n_dropped_zero += int((~depth_valid).sum())
        n_total_valid += int(valid.sum())

        pts, cols = _backproject(depth_m, valid, rgb, fx, fy, cx, cy, c2w)
        all_pts_chunks.append(pts)
        all_col_chunks.append(cols)

        if (fi + 1) % 20 == 0 or fi == len(frames) - 1:
            print(
                f"  frame {fi + 1:>3}/{len(frames)}: kept {pts.shape[0]:>6} px"
                f" (running total {sum(p.shape[0] for p in all_pts_chunks):>10})"
            )

    points = np.concatenate(all_pts_chunks, axis=0)
    colors = np.concatenate(all_col_chunks, axis=0)
    print(
        f"\nValid pixel counts (summed across {len(frames)} frames):\n"
        f"  total kept     : {n_total_valid:>12}\n"
        f"  dropped gripper: {n_dropped_gripper:>12}\n"
        f"  dropped bg     : {n_dropped_bg:>12}\n"
        f"  dropped depth=0: {n_dropped_zero:>12}\n"
        f"  total points   : {points.shape[0]:>12}\n"
    )

    target = int(num_points)
    if points.shape[0] > target:
        idx = rng.choice(points.shape[0], size=target, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {target} points (random without replacement).")
    else:
        print(f"Total points {points.shape[0]} <= target {target}; keeping all.")

    if backup_existing and out_path.exists() and not backup_path.exists():
        shutil.copy2(out_path, backup_path)
        print(f"Backed up existing PLY -> {backup_path.name}")
    _save_ply(points, colors, out_path)
    print(f"Wrote {points.shape[0]} points -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing static_scene/",
    )
    parser.add_argument("--num-points", type=int, default=500_000)
    parser.add_argument(
        "--bg-tolerance",
        type=float,
        default=DEFAULT_BG_TOLERANCE,
        help="L2 RGB distance from the simulator bg color (217,235,255) to drop.",
    )
    parser.add_argument(
        "--depth-unit-scale",
        type=float,
        default=1e-3,
        help="Multiplier from raw uint16 depth to meters (default 1e-3).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frame-id",
        type=int,
        default=None,
        help=(
            "Drop static frames whose numeric ID (parsed from the filename, "
            "e.g. arm_14864.png -> 14864) exceeds this value."
        ),
    )
    parser.add_argument(
        "--min-frame-id",
        type=int,
        default=None,
        help="Symmetric counterpart: drop frames with ID below this value.",
    )
    args = parser.parse_args()

    regenerate(
        root=args.root,
        num_points=args.num_points,
        bg_tolerance=args.bg_tolerance,
        depth_unit_scale=args.depth_unit_scale,
        seed=args.seed,
        max_frame_id=args.max_frame_id,
        min_frame_id=args.min_frame_id,
    )


if __name__ == "__main__":
    main()
