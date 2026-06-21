"""Where does the CPU fusion 890ms/frame go? Per-stage timing of the CPU path:
  src_cloud (backproject + voxel + estimate_normals) | icp (2-stage) | integrate (full-res TSDF) | model_refresh
so we know which lever to pull. Also sweeps a few cheaper-ICP configs to show the time/quality knob.
Run: python -m dynamic_gs2.verify._probe_cpu_stage_breakdown [dataset_dir]"""
import sys, time, json, re, os
from pathlib import Path
import numpy as np, cv2, open3d as o3d
os.environ["DGS_FUSION_DEVICE"] = "cpu"
import dynamic_gs.utils.online_fusion as OF

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_170319_live").resolve()
sd = DS / "static_scene"
meta = json.loads((sd / "transforms.json").read_text())
fx, fy, cx, cy = float(meta["fl_x"]), float(meta["fl_y"]), float(meta["cx"]), float(meta["cy"])
W, H = int(meta["w"]), int(meta["h"])
frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", Path(fr["file_path"]).name)[-1]))

def load(fr):
    d = cv2.imread(str(sd / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
    mp = fr.get("mask_path") or fr.get("mask_file_path")
    if mp:
        m = cv2.imread(str(sd / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE)
        if m is not None: d[m == 0] = 0
    return d, np.asarray(fr["transform_matrix"], dtype=np.float64)

loaded = [load(fr) for fr in frames]

def run(src_stride, icp_voxel, normal_radius, refresh_every, label):
    """Instrument the CPU fusion with these knobs and time each stage."""
    OF.ICP_SRC_STRIDE = src_stride
    OF.ICP_VOXEL_M = icp_voxel
    OF.NORMAL_RADIUS_M = normal_radius
    OF.MODEL_REFRESH_EVERY = refresh_every
    f = OF._CpuOnlineFusion(fx, fy, cx, cy, W, H)
    T_src = T_icp = T_int = T_ref = 0.0
    for k, (d, c2w_gl) in enumerate(loaded):
        c2w_cv = OF.OnlineFusion._cv_c2w(c2w_gl)
        t0 = time.time(); src = f._src_cloud(d, c2w_cv); T_src += time.time() - t0
        if f.model is None:
            f.model = src
            t0 = time.time(); f._integrate(d, None, c2w_cv); T_int += time.time() - t0
            f.idx += 1; continue
        t0 = time.time()
        Tm = np.eye(4); reg = None
        for dist, iters in OF.ICP_STAGES:
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            reg = o3d.pipelines.registration.registration_icp(src, f.model, dist, Tm, f.estim, crit)
            Tm = reg.transformation
        T_icp += time.time() - t0
        refined = Tm @ c2w_cv if reg.fitness >= OF.ICP_FITNESS_MIN else c2w_cv
        t0 = time.time(); f._integrate(d, None, refined); T_int += time.time() - t0
        src.transform(refined @ np.linalg.inv(c2w_cv)); f._pend.append(src); f.idx += 1
        if f.idx % OF.MODEL_REFRESH_EVERY == 0:
            t0 = time.time()
            for s in f._pend: f.model += s
            f.model = f.model.voxel_down_sample(OF.ICP_VOXEL_M)
            f.model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=OF.NORMAL_RADIUS_M, max_nn=30))
            f._pend = []; T_ref += time.time() - t0
    n = len(loaded)
    tot = T_src + T_icp + T_int + T_ref
    print(f"\n[{label}] stride={src_stride} icp_voxel={icp_voxel*100:.0f}cm normR={normal_radius*100:.0f}cm refresh={refresh_every}")
    print(f"  src_cloud(backproj+voxel+normals) = {T_src/n*1000:6.1f} ms/f   ({T_src:.2f}s)")
    print(f"  icp (2-stage)                     = {T_icp/n*1000:6.1f} ms/f   ({T_icp:.2f}s)")
    print(f"  integrate (FULL-RES TSDF)         = {T_int/n*1000:6.1f} ms/f   ({T_int:.2f}s)")
    print(f"  model_refresh                     = {T_ref/n*1000:6.1f} ms/f   ({T_ref:.2f}s)")
    print(f"  TOTAL                             = {tot/n*1000:6.1f} ms/f   ({tot:.2f}s over {n} frames)")

# baseline (current production constants)
run(4, 0.01, 0.03, 5, "BASELINE")
# lever 1: decimate ICP source harder (stride 8) + coarser ICP voxel (2cm)
run(8, 0.02, 0.04, 5, "lighter-ICP")
# lever 2: very light ICP (stride 12, 3cm) — how cheap can pose-refine get
run(12, 0.03, 0.05, 8, "min-ICP")
