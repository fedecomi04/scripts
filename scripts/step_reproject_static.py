#!/usr/bin/env python3
"""Step through a static dataset's per-frame reprojected point clouds.

For each frame: back-project the (masked) depth through the camera intrinsics,
transform by the frame's c2w (OpenGL -> world), color by RGB, and show it in an
Open3D window. Press a key to advance to the next frame.

Purpose: visually decide whether the "doubled walls" in the fused seed are a
POSE problem (consecutive frames' clouds don't overlap -> drift) or a DEPTH
problem (each frame's cloud is individually clean but thick/smeared -> sensor
depth floor).

Controls (in the Open3D window):
    ESC / Q   -> next frame   (close current, open next)
    N         -> next frame   (alias)
    A         -> toggle "accumulate" mode: keep ALL frames so far on screen
                 (the closest thing to the fused result; doubling shows here)
    P         -> toggle "pair" mode: show current + previous frame together,
                 current = RGB color, previous = solid red. If a wall is
                 doubled, you'll see a red ghost offset from the live wall.

Usage:
    python scripts/step_reproject_static.py <static_scene_dir>
    python scripts/step_reproject_static.py            # defaults to zed_scene
    # options:
    #   --stride N      subsample frames (default 1 = every frame)
    #   --px-stride N   subsample pixels for speed (default 2)
    #   --max-depth M   ignore depth beyond M metres (default 3.0)
    #   --start I       start at frame index I
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

_DEFAULT = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/"
    "datasets/ZED/zed_scene/static_scene"
)

# OpenGL c2w -> OpenCV c2w (flip y,z of the camera axes).
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _resolve(static_dir: Path, rel: str) -> Path:
    return (static_dir / rel.lstrip("./")).resolve()


def backproject(depth_m, rgb, fx, fy, cx, cy, c2w_gl, mask, px_stride, max_depth):
    """Return (Nx3 world xyz, Nx3 rgb float [0,1]) for one frame."""
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:px_stride, 0:w:px_stride]
    z = depth_m[vs, us]
    valid = (z > 0.05) & (z < max_depth)
    if mask is not None:
        valid &= mask[vs, us] > 0
    us, vs, z = us[valid], vs[valid], z[valid]
    if us.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # OpenCV cam coords
    c2w_cv = c2w_gl @ _GL2CV
    world = (c2w_cv @ cam.T).T[:, :3]
    col = rgb[vs, us].astype(np.float32) / 255.0  # already RGB
    return world, col


def make_pcd(xyz, rgb=None, paint=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if paint is not None:
        pcd.paint_uniform_color(paint)
    elif rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("static_dir", nargs="?", default=_DEFAULT)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--px-stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    static_dir = Path(args.static_dir)
    tj = json.load(open(static_dir / "transforms.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    frames = tj["frames"][args.start :: args.stride]
    n = len(frames)
    print(f"[step-reproject] {n} frames from {static_dir}")
    print("  controls: ESC/Q/N = next frame | A = accumulate toggle | "
          "P = pair-with-previous toggle")

    # mutable view state shared with key callbacks
    state = {"i": 0, "accumulate": False, "pair": False, "advance": False,
             "accum_xyz": [], "accum_rgb": [], "prev_xyz": None,
             "cur_xyz": None}

    def load_frame(idx):
        f = frames[idx]
        depth = cv2.imread(str(_resolve(static_dir, f["depth_file_path"])),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        bgr = cv2.imread(str(_resolve(static_dir, f["file_path"])), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = None
        mp = f.get("mask_path")
        if mp:
            mpath = _resolve(static_dir, mp)
            if mpath.exists():
                mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        c2w = np.array(f["transform_matrix"], dtype=np.float64)
        return backproject(depth, rgb, fx, fy, cx, cy, c2w, mask,
                           args.px_stride, args.max_depth)

    while state["i"] < n:
        i = state["i"]
        xyz, col = load_frame(i)

        geoms = []
        if state["accumulate"]:
            state["accum_xyz"].append(xyz)
            state["accum_rgb"].append(col)
            axyz = np.concatenate(state["accum_xyz"], 0)
            acol = np.concatenate(state["accum_rgb"], 0)
            geoms.append(make_pcd(axyz, acol))
            title = f"ACCUMULATE  frames 0..{i}/{n-1}  ({len(axyz):,} pts)"
        elif state["pair"] and state["prev_xyz"] is not None:
            geoms.append(make_pcd(xyz, col))  # current = RGB
            geoms.append(make_pcd(state["prev_xyz"], paint=[1.0, 0.0, 0.0]))  # prev = red
            title = f"PAIR  frame {i}/{n-1} (RGB) + prev (RED)  [{len(xyz):,} pts]"
        else:
            geoms.append(make_pcd(xyz, col))
            title = f"frame {i}/{n-1}  ({len(xyz):,} pts)"

        state["cur_xyz"] = xyz
        state["advance"] = False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title, width=1400, height=900)
        for g in geoms:
            vis.add_geometry(g)

        def _next(v):
            # remember the frame we're leaving as "previous" for pair mode
            state["prev_xyz"] = state["cur_xyz"]
            state["i"] += 1
            state["advance"] = True
            v.close()
            return False

        def _toggle_accum(v):
            state["accumulate"] = not state["accumulate"]
            if not state["accumulate"]:
                state["accum_xyz"].clear()
                state["accum_rgb"].clear()
            print(f"  accumulate = {state['accumulate']}")
            v.close()  # reopen same frame in new mode
            return False

        def _toggle_pair(v):
            state["pair"] = not state["pair"]
            print(f"  pair = {state['pair']}")
            v.close()
            return False

        # ESC(256) Q(81) N(78) advance; A(65) accum; P(80) pair
        vis.register_key_callback(256, _next)
        vis.register_key_callback(81, _next)
        vis.register_key_callback(78, _next)
        vis.register_key_callback(65, _toggle_accum)
        vis.register_key_callback(80, _toggle_pair)
        vis.run()
        vis.destroy_window()

        # If a toggle closed the window (advance not set), redisplay SAME frame.
        if not state["advance"]:
            # toggles set prev_xyz to current; undo so pair still shows real prev
            continue

    print("[step-reproject] done.")


if __name__ == "__main__":
    main()
