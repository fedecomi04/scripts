#!/usr/bin/env python3
"""Back-project N random static frames (rgb + depth) into colored point clouds
so the depth-noise can be inspected reprojected on RGB.

One PLY per frame, written into <static_dir>/reproj_check/. Each cloud is in the
CAMERA frame (z forward), so it's a standalone view of that frame's depth — no
multi-frame alignment, exactly the per-frame reprojection asked for.

Usage:
  python scripts/reproject_static_frames.py <data_dir> [n] [seed]
    <data_dir>  dataset root (uses <data_dir>/static_scene)
    n           number of random frames (default 5)
    seed        RNG seed (default 0)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """xyz (M,3) float, rgb (M,3) uint8 -> binary little-endian PLY."""
    m = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {m}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    verts = np.empty(
        m,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


def main():
    data_dir = Path(sys.argv[1]).resolve()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    static = data_dir / "static_scene"
    tf = json.loads((static / "transforms.json").read_text())

    fx = float(tf["fl_x"]); fy = float(tf["fl_y"])
    cx = float(tf["cx"]); cy = float(tf["cy"])

    frames = tf["frames"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frames), size=min(n, len(frames)), replace=False)
    out_dir = static / "reproj_check"
    out_dir.mkdir(exist_ok=True)
    print(f"[reproj] {data_dir.name}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"[reproj] {len(idx)} random frames -> {out_dir}")

    for i in sorted(idx.tolist()):
        fr = frames[i]
        rgb_p = static / fr["file_path"].replace("./", "")
        dep_p = static / fr["depth_file_path"].replace("./", "")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
        depth_mm = cv2.imread(str(dep_p), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            print(f"  [{i}] MISSING depth {dep_p}")
            continue
        H, W = depth_mm.shape[:2]
        z = depth_mm.astype(np.float32) / 1000.0  # mm -> m

        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        valid = z > 0
        zf = z[valid]
        xf = (uu[valid] - cx) / fx * zf
        yf = (vv[valid] - cy) / fy * zf
        # Camera frame, OpenCV convention (x right, y down, z forward).
        xyz = np.stack([xf, yf, zf], axis=1)
        col = rgb[valid]

        stem = Path(fr["file_path"]).stem
        out = out_dir / f"{stem}_reproj.ply"
        write_ply(out, xyz, col)
        print(f"  [{i:>3}] {stem}: {xyz.shape[0]:>8d} pts  "
              f"z[min={zf.min():.3f} max={zf.max():.3f} med={np.median(zf):.3f}] -> {out.name}")


if __name__ == "__main__":
    main()
