"""Gate — verify the depth + ray convention used by the feedforward decode path.

Re-runnable standalone script (no Nerfstudio import). Loads frame 0 of the
static scene, builds a back-projection ray for a known table pixel using
two competing conventions (OpenCV-style and Nerfstudio/OpenGL-style with
+z OpenCV depth), reprojects through gt_depth, and reports which convention
lands within 5 cm of the nearest SfM init point.

The convention that passes is the one the decoder must use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


DEFAULT_DATASET = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45_w_background"
)
DEPTH_UNIT_SCALE = 1e-3


def load_transforms_first_frame(transforms_path: Path):
    with transforms_path.open() as f:
        meta = json.load(f)
    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    w, h = int(meta["w"]), int(meta["h"])
    frame = meta["frames"][0]
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
    rgb_rel = frame["file_path"]
    depth_rel = frame["depth_file_path"]
    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h,
        "c2w": c2w, "rgb_rel": rgb_rel, "depth_rel": depth_rel,
    }


def load_depth_metres(depth_path: Path) -> np.ndarray:
    arr = np.array(Image.open(depth_path))
    if arr.dtype != np.uint16:
        raise RuntimeError(f"expected uint16 depth, got {arr.dtype} at {depth_path}")
    return arr.astype(np.float32) * DEPTH_UNIT_SCALE


def load_sfm_points(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)


def backproject_opencv(u, v, depth_z, K, c2w):
    """OpenCV: dir_cam = K^-1 @ [u, v, 1]; depth is z-component along +z."""
    fx, fy, cx, cy = K
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    R, t = c2w[:3, :3], c2w[:3, 3]
    rays_d_un_world = R @ dir_cam
    xyz_world = t + rays_d_un_world * depth_z
    return xyz_world, rays_d_un_world


def backproject_opengl(u, v, depth_z, K, c2w):
    """Nerfstudio/OpenGL convention: camera forward = -z, up = +y.

    The depth TIFF stores +z (OpenCV-style sensor depth). Convert the ray
    direction to the OpenGL frame (flip y and z), then multiply by the
    same positive depth value.
    """
    fx, fy, cx, cy = K
    dir_cam_opengl = np.array(
        [(u - cx) / fx, -(v - cy) / fy, -1.0],
        dtype=np.float64,
    )
    R, t = c2w[:3, :3], c2w[:3, 3]
    rays_d_un_world = R @ dir_cam_opengl
    xyz_world = t + rays_d_un_world * depth_z
    return xyz_world, rays_d_un_world


def find_nearest(query: np.ndarray, sfm: np.ndarray) -> tuple[float, np.ndarray]:
    d = np.linalg.norm(sfm - query[None, :], axis=1)
    i = int(np.argmin(d))
    return float(d[i]), sfm[i]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--test-pixels",
        type=str,
        default="400,400;600,400;200,400;400,600;400,200",
        help="Semicolon-separated u,v pairs to back-project.",
    )
    args = parser.parse_args()

    transforms_path = args.dataset / "static_scene" / "transforms.json"
    sfm_path = args.dataset / "static_scene" / "depth_camera_init_points.ply"

    if not transforms_path.exists():
        print(f"[gate] FAIL: {transforms_path} missing", file=sys.stderr)
        return 2
    if not sfm_path.exists():
        print(f"[gate] FAIL: {sfm_path} missing", file=sys.stderr)
        return 2

    meta = load_transforms_first_frame(transforms_path)
    depth_path = (transforms_path.parent / meta["depth_rel"]).resolve()
    depth_m = load_depth_metres(depth_path)
    sfm = load_sfm_points(sfm_path)

    print(f"[gate] static frame: {meta['rgb_rel']}")
    print(f"[gate] depth map shape={depth_m.shape} dtype={depth_m.dtype} "
          f"min={depth_m.min():.4f} max={depth_m.max():.4f} mean={depth_m.mean():.4f}")
    print(f"[gate] SfM init: {len(sfm)} points, range "
          f"x=[{sfm[:,0].min():.3f},{sfm[:,0].max():.3f}] "
          f"y=[{sfm[:,1].min():.3f},{sfm[:,1].max():.3f}] "
          f"z=[{sfm[:,2].min():.3f},{sfm[:,2].max():.3f}]")
    print()

    K = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    c2w = meta["c2w"]
    print(f"[gate] K=(fx={K[0]:.3f}, fy={K[1]:.3f}, cx={K[2]:.3f}, cy={K[3]:.3f})")
    print(f"[gate] c2w translation: {c2w[:3, 3]}")
    print()

    results = {"opencv": [], "opengl": []}
    for pix_str in args.test_pixels.split(";"):
        u_s, v_s = pix_str.strip().split(",")
        u, v = int(u_s), int(v_s)
        d = float(depth_m[v, u])
        if d <= 0:
            print(f"[gate] pixel ({u},{v}): depth=0 (invalid) — skipping")
            continue

        xyz_cv, _ = backproject_opencv(u, v, d, K, c2w)
        xyz_gl, _ = backproject_opengl(u, v, d, K, c2w)
        dist_cv, nn_cv = find_nearest(xyz_cv, sfm)
        dist_gl, nn_gl = find_nearest(xyz_gl, sfm)

        results["opencv"].append(dist_cv)
        results["opengl"].append(dist_gl)

        print(f"[gate] pixel ({u},{v}) depth={d:.4f} m")
        print(f"        opencv → xyz={xyz_cv}  nearest_sfm={nn_cv}  dist={dist_cv*100:.2f} cm")
        print(f"        opengl → xyz={xyz_gl}  nearest_sfm={nn_gl}  dist={dist_gl*100:.2f} cm")
        print()

    if not results["opencv"]:
        print("[gate] FAIL: no valid pixels tested", file=sys.stderr)
        return 2

    cv_median = float(np.median(results["opencv"]))
    gl_median = float(np.median(results["opengl"]))
    print(f"[gate] median back-projection distance to nearest SfM point:")
    print(f"        opencv convention: {cv_median*100:.2f} cm")
    print(f"        opengl convention: {gl_median*100:.2f} cm")

    THRESH_M = 0.05
    winner = None
    if cv_median < THRESH_M and gl_median < THRESH_M:
        winner = "opencv" if cv_median <= gl_median else "opengl"
        print(f"[gate] both conventions within 5 cm; preferring {winner} (smaller median)")
    elif cv_median < THRESH_M:
        winner = "opencv"
    elif gl_median < THRESH_M:
        winner = "opengl"
    else:
        print(f"[gate] FAIL: NEITHER convention is within 5 cm.", file=sys.stderr)
        print(f"        Likely causes: (a) depth scale wrong; (b) camera frame mismatch; "
              f"(c) SfM ply in a different world frame.", file=sys.stderr)
        return 1

    print()
    print(f"[gate] PASS — winning convention: {winner}")
    print(f"[gate] decoder must build rays as:")
    if winner == "opencv":
        print("        dir_cam = ((u-cx)/fx, (v-cy)/fy, 1)")
        print("        ray_d_un_world = R_c2w @ dir_cam")
        print("        xyz_world = c2w_translation + ray_d_un_world * z_depth")
    else:
        print("        dir_cam = ((u-cx)/fx, -(v-cy)/fy, -1)  # nerfstudio/opengl c2w + opencv +z depth")
        print("        ray_d_un_world = R_c2w @ dir_cam")
        print("        xyz_world = c2w_translation + ray_d_un_world * z_depth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
