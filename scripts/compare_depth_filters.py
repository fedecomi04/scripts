"""4-way depth-filter comparison on one static frame's reprojection.

Back-projects the SAME static frame's (already upper-bound-noised) depth four ways:
  1. raw            -- no filter (what FF currently back-projects without the fix)
  2. median(k)      -- the FF flying-pixel fix (isolated-outlier removal, edge-safe)
  3. bilateral      -- edge-preserving smoothing of within-surface jitter
  4. median+bilateral -- flying-pixels removed THEN surface smoothed

Opens all four as Open3D point clouds (DISPLAY=:1), one window at a time, and
also writes each as a .ply. Filters hold out invalid (==0) pixels so holes never
bleed into valid depth and the filters never invent depth.

Usage: compare_depth_filters.py <data_dir> [frame_index] [median_k] [bilat_d] [bilat_sigmaColor_m] [bilat_sigmaSpace]
"""
import json, os, sys
import numpy as np, cv2, open3d as o3d

DATA = sys.argv[1]
STATIC = os.path.join(DATA, "static_scene")
FI = int(sys.argv[2]) if len(sys.argv) > 2 else 11
MED_K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
BI_D = int(sys.argv[4]) if len(sys.argv) > 4 else 5
BI_SC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01   # colour sigma in METRES (depth units)
BI_SS = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
Z_MIN, Z_MAX = 0.05, 2.0

j = json.load(open(os.path.join(STATIC, "transforms.json")))
fx, fy, cx, cy = j["fl_x"], j["fl_y"], j["cx"], j["cy"]
fr = j["frames"][FI]
rgb = cv2.cvtColor(cv2.imread(os.path.join(STATIC, fr["file_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth = cv2.imread(os.path.join(STATIC, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
valid0 = depth > 0.0


def median_filt(d, k):
    if k < 3:
        return d
    if k % 2 == 0:
        k += 1
    k = min(k, 5)                       # cv2 float32 medianBlur supports 3/5 only
    f = cv2.medianBlur(d, k)
    return np.where(valid0 & (f > 0.0), f, d)


def bilateral_filt(d, dia, sc, ss):
    # zero out invalid so they don't pull valid pixels; bilateral, then restore holes.
    f = cv2.bilateralFilter(d, dia, sc, ss)
    # re-normalize: bilateral with zeros in the window biases valid pixels toward 0.
    # weight-correct by filtering the validity mask the same way.
    wmask = valid0.astype(np.float32)
    wf = cv2.bilateralFilter(d * wmask, dia, sc, ss)
    ww = cv2.bilateralFilter(wmask, dia, sc, ss)
    corrected = np.where(ww > 1e-6, wf / np.maximum(ww, 1e-6), d)
    return np.where(valid0, corrected, 0.0)


variants = {
    "1_raw": depth,
    "2_median": median_filt(depth, MED_K),
    "3_bilateral": bilateral_filt(depth, BI_D, BI_SC, BI_SS),
    "4_median+bilateral": bilateral_filt(median_filt(depth, MED_K), BI_D, BI_SC, BI_SS),
}


def cloud(d):
    valid = valid0 & (d > Z_MIN) & (d < Z_MAX)
    vs, us = np.where(valid)
    z = d[vs, us]
    x = (us - cx) / fx * z
    y = -(vs - cy) / fy * z
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.stack([x, y, z], 1).astype(float))
    pc.colors = o3d.utility.Vector3dVector(rgb[vs, us].astype(float) / 255.0)
    return pc, np.stack([x, y, z], 1)


# per-variant flat-table roughness (RANSAC plane RMS) as a quantitative readout
def plane_rms(pts):
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    pl, inl = pc.segment_plane(0.01, 3, 400)
    n = np.array(pl[:3]); d = pl[3]
    return np.sqrt((((pts[inl] @ n + d) / np.linalg.norm(n)) ** 2).mean()) * 1000

print(f"[compare] frame {FI}  median k={min(MED_K,5)}  bilateral d={BI_D} sigmaColor={BI_SC*1000:.0f}mm sigmaSpace={BI_SS}")
clouds = {}
for name, d in variants.items():
    pc, pts = cloud(d)
    clouds[name] = pc
    o3d.io.write_point_cloud(os.path.join(STATIC, f"filter_{name.replace('+','_')}_{FI:03d}.ply"), pc)
    print(f"  {name:>20}:  {len(pts):>8,} pts   table-plane RMS = {plane_rms(pts):.2f} mm")

print("\n[compare] opening 4 Open3D windows in order; close (Q) each to advance:")
for name, pc in clouds.items():
    print(f"   -> {name}")
    o3d.visualization.draw_geometries([pc], window_name=f"frame{FI}  {name}", width=1500, height=1000)
