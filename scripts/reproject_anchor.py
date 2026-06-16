"""Reproject the anchor_ref head-on frame with its depth. Bigger 3D view.

Usage: reproject_anchor.py <data_dir>
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
AR = os.path.join(DATA, "static_scene", "anchor_ref")
Z_MAX, Z_MIN = 2.0, 0.05

K = json.load(open(os.path.join(AR, "intrinsics.json")))
fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
rgb = cv2.cvtColor(cv2.imread(os.path.join(AR, "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(AR, "depth.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
objmask = cv2.imread(os.path.join(AR, "mask_00.png"), cv2.IMREAD_UNCHANGED)
h, w = depth.shape
print(f"[anchor] {w}x{h} depth valid {(depth>0).mean()*100:.1f}% "
      f"range {depth[depth>0].min():.3f}-{depth[depth>0].max():.3f} m  "
      f"obj-mask px {(objmask>0).sum():,}")

valid = (depth > Z_MIN) & (depth < Z_MAX)
vs, us = np.where(valid)
z = depth[vs, us]
x = (us - cx) / fx * z
y = -(vs - cy) / fy * z
pts_cam = np.stack([x, y, -z], 1).astype(np.float64)
cols = rgb[vs, us].astype(np.float32) / 255.0
on_obj = (objmask[vs, us] > 0) if objmask is not None else np.zeros(len(z), bool)
print(f"[anchor] {len(pts_cam):,} points ({on_obj.sum():,} on banana)")

# ---- .ply (world) ----
c2w = np.array(json.load(open(os.path.join(AR, "c2w.json"))), dtype=np.float64)
pts_w = (c2w @ np.c_[pts_cam, np.ones(len(pts_cam))].T).T[:, :3]
ply = os.path.join(AR, "reproj_anchor.ply")
step = max(1, len(pts_w) // 2_000_000)
with open(ply, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts_w[::step])}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(pts_w[::step], (cols[::step] * 255).astype(np.uint8)):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
print(f"[anchor] wrote {ply}")

# ---- montage: RGB+mask outline | depth | BIG 3D ----
fig = plt.figure(figsize=(22, 9))
ax1 = fig.add_subplot(1, 3, 1)
rgb_ov = rgb.copy()
if objmask is not None:
    cnts, _ = cv2.findContours((objmask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_ov, cnts, -1, (0, 255, 0), 3)
ax1.imshow(rgb_ov); ax1.set_title("anchor RGB (banana mask = green)"); ax1.axis("off")
ax2 = fig.add_subplot(1, 3, 2)
im = ax2.imshow(np.where(depth > 0, depth, np.nan), cmap="turbo", vmin=Z_MIN, vmax=Z_MAX)
ax2.set_title("depth (m)"); ax2.axis("off"); fig.colorbar(im, ax=ax2, fraction=0.03)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ss = max(1, len(pts_cam) // 200_000)
P, C = pts_cam[::ss], cols[::ss]
ax3.scatter(P[:, 0], P[:, 2], P[:, 1], c=C, s=0.5, marker=".", linewidths=0)
ax3.set_title("reprojected (cam frame)")
ax3.set_xlabel("x"); ax3.set_ylabel("depth z"); ax3.set_zlabel("y")
ax3.view_init(elev=-72, azim=-90)
png = os.path.join(AR, "reproj_anchor.png")
plt.tight_layout(); plt.savefig(png, dpi=95, bbox_inches="tight"); plt.close()
print(f"[anchor] wrote {png}")

# ---- standalone BIG 3D, two angles, banana highlighted ----
fig = plt.figure(figsize=(20, 10))
for i, (el, az) in enumerate([(-72, -90), (-55, -60)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=C, s=0.5, marker=".", linewidths=0)
    Pb = pts_cam[on_obj][::max(1, on_obj.sum() // 40000)]
    if len(Pb):
        ax.scatter(Pb[:, 0], Pb[:, 2], Pb[:, 1], c="lime", s=1.2, marker=".", linewidths=0)
    ax.set_title(f"view {i+1} (banana=lime)  elev={el} az={az}")
    ax.set_xlabel("x"); ax.set_ylabel("depth z"); ax.set_zlabel("y")
    ax.view_init(elev=el, azim=az)
png2 = os.path.join(AR, "reproj_anchor_3d.png")
plt.tight_layout(); plt.savefig(png2, dpi=95, bbox_inches="tight"); plt.close()
print(f"[anchor] wrote {png2}")
