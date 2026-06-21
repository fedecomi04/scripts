"""Quality A/B: ICP_SRC_STRIDE=4 (baseline) vs 8 (lighter) on the SAME live dataset.
Checks two things the seed depends on:
  (1) POSES — per-frame ICP-refined c2w: translation + rotation delta between stride 4 and 8.
      (also reports each stride's pose delta vs raw FK, i.e. how much ICP actually moves the pose)
  (2) GEOMETRY — final seed point cloud: count, bbox extent, and symmetric nearest-neighbour
      surface distance between the stride-4 and stride-8 clouds (are the fused surfaces the same?).
Run: python -m dynamic_gs2.verify._probe_stride_quality <dataset_dir>
"""
import sys, json, re, os
from pathlib import Path
import numpy as np, cv2, open3d as o3d
os.environ["DGS_FUSION_DEVICE"] = "cpu"
os.environ.setdefault("DGS_TSDF_VOXEL_M", "0.003")
import importlib
import dynamic_gs.utils.online_fusion as OF
importlib.reload(OF)   # pick up the 3mm voxel from env

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_170319_live").resolve()
sd = DS if DS.name == "static_scene" else DS / "static_scene"
meta = json.loads((sd / "transforms.json").read_text())
fx, fy, cx, cy = float(meta["fl_x"]), float(meta["fl_y"]), float(meta["cx"]), float(meta["cy"])
W, H = int(meta["w"]), int(meta["h"])
frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))
print(f"dataset={sd.parent.name} frames={len(frames)} res={W}x{H} voxel={OF.TSDF_VOXEL_M*1000:.0f}mm")

def load(fr):
    d = cv2.imread(str(sd / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
    mp = fr.get("mask_path") or fr.get("mask_file_path")
    if mp:
        m = cv2.imread(str(sd / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE)
        if m is not None: d[m == 0] = 0
    return d, np.asarray(fr["transform_matrix"], dtype=np.float64)
loaded = [load(fr) for fr in frames]

def rot_angle_deg(Ra, Rb):
    R = Ra[:3, :3].T @ Rb[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def run_stride(stride):
    OF.ICP_SRC_STRIDE = stride
    f = OF._CpuOnlineFusion(fx, fy, cx, cy, W, H)
    refined_poses, fk_poses = [], []
    for d, c2w_gl in loaded:
        c2w_cv = OF.OnlineFusion._cv_c2w(c2w_gl)
        fk_poses.append(c2w_cv.copy())
        refined_poses.append(f.add_frame(d, c2w_cv, None).copy())
    pc = f.finalize()
    if len(pc.points) > 0:
        pc = OF.adaptive_downsample(pc, fk_poses[-1][:3, 3])
    return refined_poses, fk_poses, pc

print("running stride=4 (baseline)..."); p4, fk, pc4 = run_stride(4)
print("running stride=8 (lighter)...");  p8, _,  pc8 = run_stride(8)

# (1) POSES
print("\n=== POSE A/B (per-frame ICP-refined c2w) ===")
dt48 = [np.linalg.norm((p4[i][:3,3]-p8[i][:3,3]))*1000 for i in range(len(p4))]   # mm
dr48 = [rot_angle_deg(p4[i], p8[i]) for i in range(len(p4))]
icp4 = [np.linalg.norm((p4[i][:3,3]-fk[i][:3,3]))*1000 for i in range(len(p4))]   # how much ICP@4 moves vs FK
icp8 = [np.linalg.norm((p8[i][:3,3]-fk[i][:3,3]))*1000 for i in range(len(p8))]
print(f"  stride4 vs stride8  : trans median={np.median(dt48):.3f}mm max={np.max(dt48):.3f}mm | "
      f"rot median={np.median(dr48):.4f}deg max={np.max(dr48):.4f}deg")
print(f"  ICP move vs FK @str4 : trans median={np.median(icp4):.3f}mm max={np.max(icp4):.3f}mm")
print(f"  ICP move vs FK @str8 : trans median={np.median(icp8):.3f}mm max={np.max(icp8):.3f}mm")

# (2) GEOMETRY
print("\n=== SEED GEOMETRY A/B ===")
def bbox(pc):
    a=np.asarray(pc.points); return (a.max(0)-a.min(0)) if len(a) else np.zeros(3)
b4, b8 = bbox(pc4), bbox(pc8)
print(f"  stride4: {len(pc4.points):,} pts  bbox(m)={np.round(b4,3)}")
print(f"  stride8: {len(pc8.points):,} pts  bbox(m)={np.round(b8,3)}")
d_pc8_to_pc4 = np.asarray(pc8.compute_point_cloud_distance(pc4))
d_pc4_to_pc8 = np.asarray(pc4.compute_point_cloud_distance(pc8))
print(f"  surface dist stride8->stride4: median={np.median(d_pc8_to_pc4)*1000:.2f}mm  p95={np.percentile(d_pc8_to_pc4,95)*1000:.2f}mm")
print(f"  surface dist stride4->stride8: median={np.median(d_pc4_to_pc8)*1000:.2f}mm  p95={np.percentile(d_pc4_to_pc8,95)*1000:.2f}mm")
print("\nVERDICT: stride 8 is safe if pose-delta is sub-mm/sub-0.05deg AND surface dist << voxel (3mm).")
