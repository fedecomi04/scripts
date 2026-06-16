"""4-way depth-filter comparison on a REAL ZED frame's reprojection (flat dataset
layout: rgb/ depth/ transforms.json at root). Tests the SHARED depth_filter module
(the exact code the pipeline runs), not a reimplementation.

  1. raw              -- real ZED depth, no filter
  2. median           -- median-only (the TRACKER path: median=True, bilateral=False)
  3. bilateral        -- bilateral-only (median=False, bilateral=True)
  4. median+bilateral -- the full filter (FF path)

Opens all four as Open3D point clouds (DISPLAY=:1), one window at a time.
Usage: compare_depth_filters_zed.py <zed_dataset_dir> [frame_index]
"""
import json, os, sys
import numpy as np, cv2, torch, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_gs.utils import depth_filter as df

DATA = sys.argv[1]
j = json.load(open(os.path.join(DATA, "transforms.json")))
fx, fy, cx, cy = j["fl_x"], j["fl_y"], j["cx"], j["cy"]
frames = j["frames"]
FI = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames) // 2
fr = frames[FI]
Z_MIN, Z_MAX = 0.05, 3.0   # real ZED range gate (matches online_fusion DEPTH_MIN/MAX for ZED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

rgb = cv2.cvtColor(cv2.imread(os.path.join(DATA, fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(DATA, fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
valid0 = depth > 0.0
print(f"[zed] {os.path.basename(DATA)} frame {FI} ({fr['file_path']})  valid {(valid0).mean()*100:.1f}%  "
      f"range {depth[valid0].min():.3f}-{depth[valid0].max():.3f} m")


def filt(median, bilateral):
    t = torch.from_numpy(np.ascontiguousarray(depth)).to(dev).float().unsqueeze(-1)
    out = df.filter_depth_torch(t, median=median, bilateral=bilateral)
    return out.squeeze(-1).cpu().numpy()


variants = {
    "1_raw": depth,
    "2_median (tracker)": filt(True, False),
    "3_bilateral": filt(False, True),
    "4_median+bilateral (FF)": filt(True, True),
}


def cloud(d):
    v = valid0 & (d > Z_MIN) & (d < Z_MAX)
    vs, us = np.where(v)
    z = d[vs, us]
    x = (us - cx) / fx * z
    y = -(vs - cy) / fy * z
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.stack([x, y, z], 1).astype(float))
    pc.colors = o3d.utility.Vector3dVector(rgb[vs, us].astype(float) / 255.0)
    return pc, np.stack([x, y, z], 1)


def plane_rms(pts):
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    pl, inl = pc.segment_plane(0.01, 3, 400)
    n = np.array(pl[:3]); d = pl[3]
    return np.sqrt((((pts[inl] @ n + d) / np.linalg.norm(n)) ** 2).mean()) * 1000


clouds = {}
print(f"{'variant':>26} | {'pts':>9} | table-plane RMS")
for name, d in variants.items():
    pc, pts = cloud(d)
    clouds[name] = pc
    o3d.io.write_point_cloud(os.path.join(DATA, f"filter_{name.split()[0]}_{FI:03d}.ply"), pc)
    print(f"{name:>26} | {len(pts):>9,} | {plane_rms(pts):.2f} mm")

print("\n[zed] opening 4 Open3D windows in order; close (Q) each to advance:")
for name, pc in clouds.items():
    print(f"   -> {name}")
    o3d.visualization.draw_geometries([pc], window_name=f"ZED {os.path.basename(DATA)} f{FI}  {name}", width=1500, height=1000)
