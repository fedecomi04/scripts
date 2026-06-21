#!/usr/bin/env python3
"""Sweep the camera optical-center DEPTH (along the viewing axis = ELBOW +X) and overlay each
elbow mask on the real ZED RGB, so the ~24mm "how deep is the optical center" placement can be
hand-tuned by eye.

The camera looks along ELBOW +X, so optical-depth = the X component of REAL_HW_CAMERA_XYZ.
We sweep X by deltas (mm) and render a mask at the SAME joint config that produced the current
overlay (read from poses.txt), full-res (no quarter-res edge fattening), and overlay on the ZED RGB
at the RGB's native resolution.

Run:  DISPLAY=:1 /home/mrc-cuhk/miniconda3/envs/dynamic_gs_ros/bin/python real_hw/sweep_optical_depth.py
Out:  real_hw/mask_out/sweep/overlay_dX_{-5..+5}mm.png  + a montage.
"""
from __future__ import annotations
import os, sys, importlib.util
from pathlib import Path
import numpy as np
for _a, _r in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _a):
        setattr(np, _a, _r)
import cv2

os.environ["DGS_MASK_RENDER_SCALE"] = "1.0"   # full-res render: no edge fattening from upscale

ROS_MASK = Path(__file__).resolve().parent.parent / "dynamic_gs2" / "ros_mask.py"
spec = importlib.util.spec_from_file_location("ros_mask", ROS_MASK)
rm = importlib.util.module_from_spec(spec); sys.modules["ros_mask"] = rm; spec.loader.exec_module(rm)

OUT = Path(__file__).resolve().parent / "mask_out" / "sweep"
OUT.mkdir(parents=True, exist_ok=True)
RGB_PATH = Path(__file__).resolve().parent.parent / "dynamic_gs2" / "verify" / "zedm_rgb_realcam.png"

# the joint config that produced the current elbow mask (from poses.txt)
JOINTS = {
    "EL_FLE": 2.3118, "FA_ROT": -1.3555, "SH_FLE": 0.2388, "SH_ROT": -1.1818,
    "WRIST_1": -1.2689, "WRIST_2": 0.6095, "finger_joint": 0.0625,
}
DELTAS_MM = list(range(-5, 6))   # -5 .. +5


class T:
    def __init__(self, v): self.v = float(v)
    def to_sec(self): return self.v


def main():
    rgb = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
    H, W = rgb.shape[:2]
    print(f"RGB {RGB_PATH.name} {W}x{H}")

    # REAL ZED Mini HD720 intrinsics (S/N 13902435, queried from the SDK 2026-06-21).
    intr = rm.CameraIntrinsics(width=W, height=H, fx=732.544, fy=732.544, cx=633.254, cy=362.258)

    gen = rm.RobotMaskGenerator(intr, [0.0, 1.0], [JOINTS, JOINTS])
    base_xyz = rm.REAL_HW_CAMERA_XYZ.copy()
    R = rm.REAL_HW_CAMERA_ROT
    link_fk = gen.robot.link_fk(cfg={k: v for k, v in JOINTS.items() if k in gen.actuated_joint_names},
                                use_names=True)
    T_elbow = link_fk[rm.REAL_HW_CAMERA_PARENT_LINK].astype(np.float32)

    tiles = []
    for d_mm in DELTAS_MM:
        xyz = base_xyz.copy()
        xyz[0] = base_xyz[0] + d_mm / 1000.0          # X = optical depth along viewing axis
        off = np.eye(4, dtype=np.float32); off[:3, :3] = R; off[:3, 3] = xyz
        cam_pose = T_elbow @ off
        gen._ensure_renderer()
        gen._update_robot_poses(link_fk)
        gen.scene.set_pose(gen.camera_node, pose=gen._build_render_camera_pose(cam_pose, optical=True))
        _, depth = gen.renderer.render(gen.scene)
        mask = (depth == 0).astype(np.uint8) * 255       # 255 keep, 0 robot; upright (optical)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        robot = (mask == 0).astype(np.uint8) * 255
        edge = cv2.morphologyEx(robot, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ov = rgb.copy()
        fill = ov.copy(); fill[robot > 0] = (0, 0, 255)
        ov = cv2.addWeighted(ov, 0.75, fill, 0.25, 0)
        ov[edge > 0] = (0, 255, 0)
        label = f"dX={d_mm:+d}mm  X={xyz[0]*1000:.1f}mm"
        cv2.putText(ov, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        p = OUT / f"overlay_dX_{d_mm:+d}mm.png"
        cv2.imwrite(str(p), ov)
        tiles.append(ov)
        print(f"wrote {p.name}  robot-cover%={100*(robot>0).mean():.1f}")

    # montage: 11 tiles in a grid (3 cols)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sc = 520.0 / tw
    small = [cv2.resize(t, (int(tw*sc), int(th*sc))) for t in tiles]
    sh, sw = small[0].shape[:2]
    grid = np.full((rows*sh, cols*sw, 3), 30, np.uint8)
    for i, t in enumerate(small):
        r, c = divmod(i, cols)
        grid[r*sh:(r+1)*sh, c*sw:(c+1)*sw] = t
    cv2.imwrite(str(OUT / "montage.png"), grid)
    print("wrote montage.png")


if __name__ == "__main__":
    main()
