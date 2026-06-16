"""Preview what DOUBLED ZED depth noise looks like on an already-captured frame.

Frame depth on disk already carries 1x baked noise (sigma_orig = 0.00005 + 0.000477*z^2 m).
To show 2x total noise we ADD a layer with sigma_extra = sqrt(3)*sigma_orig, because
variances add: sqrt(sigma^2 + 3*sigma^2) = 2*sigma  -> identical to a fresh 2x capture.

Renders current(1x) | doubled(2x) side-by-side from the same head-on camera.
Usage: preview_double_noise.py <data_dir> [frame_index]
"""
import json, os, sys
import numpy as np
import cv2
import open3d as o3d

DATA = sys.argv[1]
STATIC = os.path.join(DATA, "static_scene")
j = json.load(open(os.path.join(STATIC, "transforms.json")))
fx, fy, cx, cy = j["fl_x"], j["fl_y"], j["cx"], j["cy"]
frames = j["frames"]
FI = int(sys.argv[2]) if len(sys.argv) > 2 else 11
fr = frames[FI]
Z_MAX, Z_MIN = 2.0, 0.05

# ORIGINAL (baked) model constants — what the capture used.
SIGMA0, K = 0.00005, 0.000477

rgb = cv2.cvtColor(cv2.imread(os.path.join(STATIC, fr["file_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth0 = cv2.imread(os.path.join(STATIC, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
valid = (depth0 > Z_MIN) & (depth0 < Z_MAX)
mpath = os.path.join(STATIC, fr.get("mask_path", ""))
if os.path.exists(mpath):
    m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
    if m is not None and m.shape == depth0.shape:
        valid &= m > 0

rng = np.random.default_rng(0)
# doubled-noise depth: add sqrt(3)*sigma_orig(z)
z_all = depth0.copy()
sigma_orig = SIGMA0 + K * (z_all ** 2)
extra = rng.normal(0.0, 1.0, size=z_all.shape).astype(np.float32) * (np.sqrt(3.0) * sigma_orig)
depth2 = np.where(valid, z_all + extra, z_all)

def cloud(depth):
    vs, us = np.where(valid)
    z = depth[vs, us]
    keep = (z > Z_MIN) & (z < Z_MAX)
    vs, us, z = vs[keep], us[keep], z[keep]
    x = (us - cx) / fx * z
    y = -(vs - cy) / fy * z
    pts = np.stack([x, y, z], 1).astype(np.float64)
    cols = rgb[vs, us].astype(np.float64) / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc, pts

W, H = 1200, 1000
def render(pc, pts):
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.5
    r.scene.add_geometry("pc", pc, mat)
    center = np.median(pts, axis=0)
    eye = np.array([0.0, 0.0, -0.15])
    r.scene.camera.look_at(center, eye, [0, 1, 0])
    img = np.asarray(r.render_to_image())
    return img

pc1, p1 = cloud(depth0)
pc2, p2 = cloud(depth2)
img1, img2 = render(pc1, p1), render(pc2, p2)

# report per-surface roughness change on a flat table patch (sanity)
def plane_rms(pts):
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    pl, inl = pc.segment_plane(0.01, 3, 300)
    n = np.array(pl[:3]); d = pl[3]
    sd = (pts[inl] @ n + d) / np.linalg.norm(n)
    return np.sqrt((sd ** 2).mean()) * 1000
print(f"[preview] frame {FI}: table-plane RMS  1x={plane_rms(p1):.2f}mm  2x={plane_rms(p2):.2f}mm")

lab = img1.copy()
def annotate(img, text):
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
annotate(img1, "current (1x noise)")
annotate(img2, "DOUBLED (2x noise)")
combo = np.hstack([img1, img2])
out = os.path.join(STATIC, f"preview_2x_noise_{FI:03d}.png")
cv2.imwrite(out, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
print(f"[preview] wrote {out}")
