"""Single-frame depth reprojection: back-project one RGB+depth frame into a
colored point cloud. Writes a .ply (open in Open3D) AND a PNG montage
(RGB | depth heatmap | 3D scatter of the reprojection) for inline viewing.

Usage: reproject_one_frame.py <data_dir> [frame_index]
  data_dir = dataset root (uses static_scene/)
"""
import json
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

DATA = sys.argv[1]
STATIC = os.path.join(DATA, "static_scene")
FI = int(sys.argv[2]) if len(sys.argv) > 2 else None
Z_MAX, Z_MIN = 2.0, 0.05

j = json.load(open(os.path.join(STATIC, "transforms.json")))
fx, fy, cx, cy = j["fl_x"], j["fl_y"], j["cx"], j["cy"]
frames = j["frames"]
if FI is None:
    FI = len(frames) // 2
fr = frames[FI]

rgb = cv2.imread(os.path.join(STATIC, fr["file_path"]), cv2.IMREAD_COLOR)  # BGR
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(STATIC, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
h, w = depth.shape
print(f"[reproj] frame {FI}: {fr['file_path']}  {w}x{h}  "
      f"depth valid {(depth>0).mean()*100:.1f}%  "
      f"range {depth[depth>0].min():.3f}-{depth[depth>0].max():.3f} m")

valid = (depth > Z_MIN) & (depth < Z_MAX)
mpath = os.path.join(STATIC, fr.get("mask_path", ""))
if os.path.exists(mpath):
    m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
    if m is not None and m.shape == depth.shape:
        valid &= m > 0
vs, us = np.where(valid)
z = depth[vs, us]
# OpenGL camera frame
x = (us - cx) / fx * z
y = -(vs - cy) / fy * z
pts_cam = np.stack([x, y, -z, np.ones_like(z)], 1).astype(np.float64)
c2w = np.array(fr["transform_matrix"], dtype=np.float64)
pts_w = (c2w @ pts_cam.T).T[:, :3]
cols = rgb[vs, us].astype(np.float32) / 255.0
print(f"[reproj] {len(pts_w):,} points back-projected")

# --- write .ply (world coords) ---
ply_path = os.path.join(STATIC, f"reproj_frame_{FI:03d}.ply")
with open(ply_path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(pts_w)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    step = max(1, len(pts_w) // 2_000_000)  # cap file size
    for p, c in zip(pts_w[::step], (cols[::step] * 255).astype(np.uint8)):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
print(f"[reproj] wrote {ply_path}")

# --- PNG montage ---
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(rgb); ax1.set_title(f"RGB (frame {FI})"); ax1.axis("off")
ax2 = fig.add_subplot(1, 3, 2)
dvis = np.where(depth > 0, depth, np.nan)
im = ax2.imshow(dvis, cmap="turbo", vmin=Z_MIN, vmax=Z_MAX)
ax2.set_title("Depth (m)"); ax2.axis("off"); fig.colorbar(im, ax=ax2, fraction=0.03)
# 3D scatter from camera-ish viewpoint, colored by RGB
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ss = max(1, len(pts_cam) // 120_000)
P = pts_cam[::ss]
C = cols[::ss]
ax3.scatter(P[:, 0], P[:, 2], P[:, 1], c=C, s=0.4, marker=".", linewidths=0)
ax3.set_title("Reprojected (cam frame)")
ax3.set_xlabel("x"); ax3.set_ylabel("z(depth)"); ax3.set_zlabel("y")
ax3.view_init(elev=-70, azim=-90)
png_path = os.path.join(STATIC, f"reproj_frame_{FI:03d}.png")
plt.tight_layout(); plt.savefig(png_path, dpi=90, bbox_inches="tight"); plt.close()
print(f"[reproj] wrote {png_path}")
