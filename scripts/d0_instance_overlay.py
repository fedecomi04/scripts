"""Render each prefused object instance projected onto the D0 (dynamic frame 0)
RGB image, color-coded with its instance_id labelled at the centroid, and print
a terminal legend naming each instance by the dominant colour it actually has in
the image (so the operator can say "track the green one" = instance N).

Reads the per-Gaussian instance buffer from
``static_scene/depth_camera_init_points.instance_ids.npy`` (1:1 with the seed
PLY) and the D0 camera pose from ``dynamic_scene/transforms.json``.

Usage:
    python scripts/d0_instance_overlay.py <data_dir> [frame_index]

Output: <data_dir>/dynamic_scene/d0_instance_overlay.png
Prints:  a legend "instance N -> <dominant colour>" for the terminal picker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2
from plyfile import PlyData

# Distinct overlay BGR colors for instances 1..8 (instance 0 = background).
_OVERLAY = [
    (0, 0, 0),
    (60, 60, 255), (60, 220, 60), (255, 120, 60), (60, 220, 255),
    (255, 60, 220), (255, 220, 60), (60, 140, 255), (200, 60, 200),
]

# Named colour anchors in RGB for naming the *actual* object colour.
_NAMED = {
    "red": (200, 40, 40), "orange": (230, 140, 30), "yellow": (220, 210, 40),
    "green": (40, 180, 60), "cyan": (40, 200, 200), "blue": (40, 60, 220),
    "purple": (140, 40, 200), "magenta": (220, 40, 200), "pink": (235, 130, 180),
    "brown": (130, 80, 40), "white": (235, 235, 235), "gray": (130, 130, 130),
    "black": (25, 25, 25),
}


def _color_name(rgb: np.ndarray) -> str:
    best, bd = "?", 1e18
    for name, ref in _NAMED.items():
        d = float(np.sum((rgb.astype(float) - np.array(ref, float)) ** 2))
        if d < bd:
            bd, best = d, name
    return best


def _load_seed_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def main() -> None:
    data = Path(sys.argv[1])
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ss, ds = data / "static_scene", data / "dynamic_scene"

    xyz = _load_seed_xyz(ss / "depth_camera_init_points.ply")
    ids = np.load(ss / "depth_camera_init_points.instance_ids.npy")
    assert len(ids) == len(xyz)

    tf = json.loads((ds / "transforms.json").read_text())
    fr = sorted(tf["frames"], key=lambda f: f["file_path"])[frame_idx]
    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    R, t = c2w[:3, :3], c2w[:3, 3]
    fx, fy = float(tf["fl_x"]), float(tf["fl_y"])
    cx, cy = float(tf["cx"]), float(tf["cy"])
    W = int(tf.get("w") or fr.get("w"))
    H = int(tf.get("h") or fr.get("h"))

    rgb_path = ds / "rgb" / Path(fr["file_path"]).name
    img = cv2.imread(str(rgb_path))
    if img is None:
        img = np.full((H, W, 3), 200, np.uint8)
    img = cv2.resize(img, (W, H))
    overlay = img.copy()

    cam = (xyz - t) @ R
    z = -cam[:, 2]
    valid = z > 1e-6
    u = fx * (cam[:, 0] / np.where(valid, z, 1)) + cx
    v = fy * (-cam[:, 1] / np.where(valid, z, 1)) + cy

    print(f"D0 frame: {rgb_path.name}  {W}x{H}\n")
    print("instance  in-frame  pixel-bbox(u,v)         dominant-colour")
    legend = []
    for iid in range(1, len(_OVERLAY)):
        m = (ids == iid) & valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n = int(m.sum())
        if n == 0:
            print(f"   {iid}        0       (off-frame)")
            continue
        uu, vv = u[m].astype(int), v[m].astype(int)
        col = _OVERLAY[iid]
        overlay[vv, uu] = col
        # Sample the object's actual image colour at its pixels (BGR->RGB).
        sampled_bgr = img[vv, uu].astype(float).mean(0)
        rgb = sampled_bgr[::-1]
        cname = _color_name(rgb)
        lx, ly = int(np.median(uu)), int(np.median(vv))
        cv2.circle(overlay, (lx, ly), 7, (255, 255, 255), -1)
        cv2.putText(overlay, str(iid), (lx + 9, ly + 7), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(overlay, str(iid), (lx + 9, ly + 7), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2, cv2.LINE_AA)
        print(f"   {iid}     {n:6d}    u[{uu.min():3d}-{uu.max():3d}] v[{vv.min():3d}-{vv.max():3d}]   {cname}")
        legend.append((iid, cname))

    blended = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    out = ds / "d0_instance_overlay.png"
    cv2.imwrite(str(out), blended)
    print("\nLEGEND:  " + "   ".join(f"{i} ({c})" for i, c in legend))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
