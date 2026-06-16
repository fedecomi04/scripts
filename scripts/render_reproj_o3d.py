"""Render a single-frame depth reprojection as a clean Open3D point-cloud image (offscreen).

Usage: render_reproj_o3d.py <data_dir> [frame_index]
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
FI = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames) // 2
fr = frames[FI]
Z_MAX, Z_MIN = 2.0, 0.05

rgb = cv2.cvtColor(cv2.imread(os.path.join(STATIC, fr["file_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(STATIC, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
valid = (depth > Z_MIN) & (depth < Z_MAX)
mpath = os.path.join(STATIC, fr.get("mask_path", ""))
if os.path.exists(mpath):
    m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
    if m is not None and m.shape == depth.shape:
        valid &= m > 0
vs, us = np.where(valid)
z = depth[vs, us]
x = (us - cx) / fx * z
y = -(vs - cy) / fy * z
pts = np.stack([x, y, z], 1).astype(np.float64)  # +z forward for a natural camera-looking view
cols = rgb[vs, us].astype(np.float64) / 255.0

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(pts)
pc.colors = o3d.utility.Vector3dVector(cols)
print(f"[render] frame {FI}: {len(pts):,} pts, depth {z.min():.3f}-{z.max():.3f} m")

out = os.path.join(STATIC, f"reproj_o3d_{FI:03d}.png")
W, H = 1600, 1000
try:
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.5
    r.scene.add_geometry("pc", pc, mat)
    # Natural reprojection view: sit at the original camera (origin), look into +z.
    pmed = np.median(pts, axis=0)
    center = pmed                       # look at the scene centroid (ahead in +z)
    eye = np.array([0.0, 0.0, -0.15])   # ~at the camera, pulled back slightly
    r.scene.camera.look_at(center, eye, [0, 1, 0])
    img = r.render_to_image()
    o3d.io.write_image(out, img)
    print(f"[render] wrote {out} (OffscreenRenderer)")
except Exception as e:
    print(f"[render] OffscreenRenderer failed ({type(e).__name__}: {e}); trying legacy Visualizer")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(pc)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([1, 1, 1])
    vc = vis.get_view_control()
    vc.set_front([0, -0.4, -0.9]); vc.set_up([0, -1, 0]); vc.set_zoom(0.7)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(out, do_render=True)
    vis.destroy_window()
    print(f"[render] wrote {out} (legacy Visualizer)")
