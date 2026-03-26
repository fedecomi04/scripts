
#!/usr/bin/env python3
"""
Plot Nerfstudio camera poses from transforms.json.

What it shows:
1. 3D camera trajectory (camera centers)
2. Camera centers over frame index (x, y, z)
3. Step distance between consecutive poses
4. A sparse set of camera viewing directions as arrows

Assumptions:
- transforms.json uses Nerfstudio/OpenGL camera convention
- each frame has a 4x4 camera-to-world transform_matrix

Usage:
  python plot_nerfstudio_poses.py /path/to/transforms.json

Optional:
  python plot_nerfstudio_poses.py /path/to/transforms.json --stride 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_transforms(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms.json")

    poses = []
    file_paths = []
    for i, frame in enumerate(frames):
        T = np.array(frame["transform_matrix"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Frame {i} has transform_matrix shape {T.shape}, expected (4,4)")
        poses.append(T)
        file_paths.append(frame.get("file_path", f"frame_{i:06d}"))

    poses = np.stack(poses, axis=0)
    return data, poses, file_paths


def extract_camera_centers_and_dirs(poses: np.ndarray):
    centers = poses[:, :3, 3]

    # Nerfstudio/OpenGL convention:
    # camera looks along negative Z axis in camera coordinates.
    # World viewing direction = R * [0, 0, -1]
    forward_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    dirs = np.einsum("nij,j->ni", poses[:, :3, :3], forward_cam)

    up_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ups = np.einsum("nij,j->ni", poses[:, :3, :3], up_cam)
    return centers, dirs, ups


def compute_step_distances(centers: np.ndarray):
    if len(centers) < 2:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(np.diff(centers, axis=0), axis=1)


def set_axes_equal(ax, pts):
    pts = np.asarray(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transforms_json", type=Path)
    parser.add_argument("--stride", type=int, default=10,
                        help="Plot one camera direction arrow every N frames")
    parser.add_argument("--arrow-scale", type=float, default=0.05,
                        help="Length of camera direction arrows")
    args = parser.parse_args()

    data, poses, file_paths = load_transforms(args.transforms_json)
    centers, dirs, ups = extract_camera_centers_and_dirs(poses)
    step_dists = compute_step_distances(centers)

    num_frames = len(centers)
    print(f"Loaded {num_frames} frames")
    print(f"First image: {file_paths[0]}")
    print(f"Last image:  {file_paths[-1]}")
    print()
    print("Camera center ranges:")
    print(f"  x: [{centers[:,0].min():.6f}, {centers[:,0].max():.6f}]")
    print(f"  y: [{centers[:,1].min():.6f}, {centers[:,1].max():.6f}]")
    print(f"  z: [{centers[:,2].min():.6f}, {centers[:,2].max():.6f}]")
    if len(step_dists) > 0:
        print()
        print("Step distance stats:")
        print(f"  min:  {step_dists.min():.6e}")
        print(f"  max:  {step_dists.max():.6e}")
        print(f"  mean: {step_dists.mean():.6e}")
        print(f"  std:  {step_dists.std():.6e}")

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot(centers[:, 0], centers[:, 1], centers[:, 2], marker="o", linewidth=1, markersize=2)
    ax1.scatter(centers[0, 0], centers[0, 1], centers[0, 2], s=60, label="start")
    ax1.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], s=60, label="end")

    idxs = np.arange(0, num_frames, max(args.stride, 1))
    arrow_dirs = dirs[idxs]
    arrow_centers = centers[idxs]

    ax1.quiver(
        arrow_centers[:, 0], arrow_centers[:, 1], arrow_centers[:, 2],
        arrow_dirs[:, 0], arrow_dirs[:, 1], arrow_dirs[:, 2],
        length=args.arrow_scale, normalize=True
    )

    ax1.set_title("3D camera trajectory and viewing directions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()
    set_axes_equal(ax1, centers)

    ax2 = fig.add_subplot(222)
    frames_idx = np.arange(num_frames)
    ax2.plot(frames_idx, centers[:, 0], label="x")
    ax2.plot(frames_idx, centers[:, 1], label="y")
    ax2.plot(frames_idx, centers[:, 2], label="z")
    ax2.set_title("Camera center coordinates")
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(223)
    if len(step_dists) > 0:
        ax3.plot(np.arange(1, num_frames), step_dists)
    ax3.set_title("Distance between consecutive camera centers")
    ax3.set_xlabel("frame index")
    ax3.set_ylabel("step distance")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.plot(centers[:, 0], centers[:, 1], marker="o", linewidth=1, markersize=2)
    ax4.scatter(centers[0, 0], centers[0, 1], s=60, label="start")
    ax4.scatter(centers[-1, 0], centers[-1, 1], s=60, label="end")
    for i in idxs:
        d = dirs[i]
        c = centers[i]
        ax4.arrow(c[0], c[1], d[0] * args.arrow_scale, d[1] * args.arrow_scale,
                  head_width=args.arrow_scale * 0.15, length_includes_head=True)
    ax4.set_title("XY projection of trajectory")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.axis("equal")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Nerfstudio camera pose sanity check", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
