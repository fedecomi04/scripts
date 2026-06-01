#!/usr/bin/env python3
# =============================================================================
# ICP-refined TSDF fusion of the SAM-labelled RGB-D  --  throwaway experiment.
#
# PROBLEM it fixes: the earlier MVP clouds back-project every frame and
# concatenate with NO fusion, so each surface is ~136 slightly-misaligned noisy
# copies (small pose errors) -> "thick" surfaces. This script:
#
#   1. Refines every frame's camera pose with frame-to-model ICP, EXCLUDING the
#      gripper (it moves with the arm, independently of the static scene, so it
#      would corrupt a rigid alignment) -- this is where the SAM/gripper mask
#      pays off for fusion.
#   2. Fuses the gripper-free RGB-D into 3 TSDF volumes (averages out per-view
#      depth noise -> one clean surface): REAL-RGB (seed color), id-color
#      (per-point SAM object label), and IMAGE-GRADIENT (per-frame Sobel
#      magnitude -> the adaptive-density signal).
#   3. ADAPTIVE ("active") density: keeps more points where the fused image
#      gradient is high and fewer on flat regions. Image gradient already lights
#      up on appearance detail (the coke can's printed text/logos) AND geometric
#      contours (object silhouettes, edges, shaded folds), so one signal covers
#      both. Flat table -> sparse, text/edges -> dense.
#   4. Also writes naive-concat (init poses) and refined-concat (ICP poses) for
#      a direct before/after comparison, and a table-flatness denoising metric.
#
# ISOLATION: reuses ONLY the sibling experiments (sam2_static_mvp for loaders /
#   palette / convention, sam3_merge_mvp for the merge remap) and their CACHED
#   outputs. No SAM/GPU re-run. Open3D does ICP + TSDF on CPU.
#
# OUTPUTS (./output/):
#   fused_seed.ply          - real-RGB, denoised, adaptive-density = the GS init seed
#   fused_seed_idcolor.ply  - same points, SAM-id colored (object semantics viz)
#   fused_seed_ids.npy      - per-point merged SAM id (int, parallel to the seed)
#   refined_concat.ply      - ICP-pose concatenation (comparison)
#   naive_concat.ply        - init-pose concatenation (noisy baseline)
#   timing_report.txt       - per-step wall-clock timing
#
# RUN:  conda activate dynamic_gs
#       python experiments/icp_fusion_mvp/icp_fusion_mvp.py
# =============================================================================

import os
import sys
import json
import time

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.dirname(THIS_DIR)
PREV_DIR = os.path.join(EXPS, "sam2_static_mvp")
MERGE_DIR = os.path.join(EXPS, "sam3_merge_mvp")
MERGE_OUT = os.path.join(MERGE_DIR, "output")
OUTPUT_DIR = os.path.join(THIS_DIR, "output")

# ICP (frame-to-model, point-to-plane, initialised from transforms.json poses)
ICP_VOXEL_M = 0.01            # downsample each frame to this for ICP
MODEL_VOXEL_M = 0.01          # keep the running model at this resolution
NORMAL_RADIUS_M = 0.03
ICP_COARSE_DIST = 0.05        # 2-stage correspondence distance
ICP_FINE_DIST = 0.02
ICP_MAX_ITER = 40
ICP_FITNESS_MIN = 0.30        # below this, distrust ICP and keep the init pose

# TSDF fusion. Two volumes are fused from the same ICP poses: a REAL-RGB volume
# (the GS seed color + the appearance signal that sees the can's printed text)
# and an id-color volume (per-point SAM object label). The base voxel is fine;
# adaptive density then coarsens flat regions back (see below).
TSDF_VOXEL_M = 0.0025         # 2.5 mm fine base extraction
TSDF_SDF_TRUNC_M = 0.0125     # ~5 x voxel
DEPTH_TRUNC_M = 3.0
DEPTH_SCALE = 1000.0          # uint16 mm -> m

# Adaptive ("active") density: keep more points where local detail is high and
# fewer on flat regions. The detail signal is the fused 2D IMAGE GRADIENT: a
# per-frame Sobel magnitude is fused into its own TSDF channel, so every point
# carries its multi-view-averaged image-gradient. Image gradient already lights
# up on BOTH appearance detail (the coke text/logos) AND geometric contours
# (object silhouettes, edges, shaded folds) -- so it subsumes a separate
# curvature term. The fine base cloud is split into 3 gradient tiers and each is
# grid-decimated at its own voxel: flat table -> sparse, text/edges -> dense.
# Set ADAPTIVE_DENSITY=False for a plain uniform seed at TSDF_VOXEL_UNIFORM.
ADAPTIVE_DENSITY = True
TSDF_VOXEL_UNIFORM = 0.0028   # used only when ADAPTIVE_DENSITY=False (~0.8M uniform)
GRAD_SCALE = 0.25            # Sobel-magnitude -> uint8 scale (fixed, cross-frame stable)
GRAD_BLUR_KSIZE = 5          # blur the gradient so detailed REGIONS (not 1-px edges) go dense
ADAPT_Q = (0.50, 0.85)        # gradient quantiles -> the two tier boundaries
ADAPT_VOXEL = (0.009, 0.0045, 0.0025)   # voxel per tier: flat / mid / detailed (m)

# point-cloud cleanup (applied to every saved cloud)
SOR_NB = 20
SOR_STD = 2.0
CONCAT_VOXEL_M = 0.005

NAIVE_PLY = os.path.join(OUTPUT_DIR, "naive_concat.ply")          # init poses (baseline)
REFINED_PLY = os.path.join(OUTPUT_DIR, "refined_concat.ply")      # ICP poses
SEED_PLY = os.path.join(OUTPUT_DIR, "fused_seed.ply")             # real-RGB adaptive GS seed
ID_PLY = os.path.join(OUTPUT_DIR, "fused_seed_idcolor.ply")       # same pts, SAM-id colored
IDS_NPY = os.path.join(OUTPUT_DIR, "fused_seed_ids.npy")          # per-point merged SAM id
TIMING_TXT = os.path.join(OUTPUT_DIR, "timing_report.txt")

TIMING = []   # list of (step_label, seconds) in execution order


# ----------------------------------------------------------------------------
def cv_c2w(c2w_opengl: np.ndarray) -> np.ndarray:
    """nerfstudio OpenGL camera-to-world -> OpenCV camera-to-world."""
    return c2w_opengl @ np.diag([1.0, -1.0, -1.0, 1.0])


def backproject_world(depth_m, valid, c2w_cv, fx, fy, cx, cy):
    vv, uu = np.where(valid)
    zz = depth_m[vv, uu]
    x = (uu - cx) * zz / fx
    y = (vv - cy) * zz / fy
    cam = np.stack([x, y, zz, np.ones_like(zz)], axis=1)
    world = (c2w_cv @ cam.T).T[:, :3]
    return world, vv, uu


def make_o3d(points, normals=False):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if normals:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
    return pc


def save_cloud(points, colors, path, do_sor=True):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pc = pc.voxel_down_sample(CONCAT_VOXEL_M)
    if do_sor:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=SOR_NB, std_ratio=SOR_STD)
    o3d.io.write_point_cloud(path, pc)
    return len(pc.points)


# ----------------------------------------------------------------------------
# Adaptive ("active") density helpers
# ----------------------------------------------------------------------------
def image_gradient(rgb_u8):
    """Per-pixel image-gradient magnitude (Sobel) as a uint8 grayscale-in-RGB
    image, on a fixed cross-frame scale and lightly blurred so detailed regions
    (not just 1-px edges) read as high."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if GRAD_BLUR_KSIZE > 1:
        mag = cv2.GaussianBlur(mag, (GRAD_BLUR_KSIZE, GRAD_BLUR_KSIZE), 0)
    mag8 = np.clip(mag * GRAD_SCALE, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(mag8[:, :, None], 3, axis=2))


def recover_nn(pos, src_pc):
    """Nearest-neighbour lookup of src_pc's colors for each point in pos."""
    nn = cKDTree(np.asarray(src_pc.points)).query(pos, k=1)[1]
    return np.asarray(src_pc.colors)[nn]


def recover_ids(pos, id_pc, palette):
    """Per-point merged SAM id: nearest id-cloud color -> nearest palette row."""
    return cKDTree(palette).query(recover_nn(pos, id_pc), k=1)[1].astype(np.int64)


def _grid_keep(pos, idx, vx):
    """Keep one representative point per voxel (preserves colors -> sharp text)."""
    if len(idx) == 0:
        return idx
    key = np.floor((pos[idx] - pos[idx].min(0)) / vx).astype(np.int64)
    _, u = np.unique(key, axis=0, return_index=True)
    return idx[u]


def adaptive_keep(pos, detail):
    """3-tier decimation: flat regions coarse, detailed regions fine."""
    qlo, qhi = np.quantile(detail, ADAPT_Q)
    tiers = [(detail < qlo, ADAPT_VOXEL[0]),
             ((detail >= qlo) & (detail < qhi), ADAPT_VOXEL[1]),
             (detail >= qhi, ADAPT_VOXEL[2])]
    return np.concatenate([_grid_keep(pos, np.where(m)[0], vx) for m, vx in tiers if m.any()])


# ----------------------------------------------------------------------------
def write_timing_report(total_wall, extra_lines):
    w = max(len(lbl) for lbl, _ in TIMING) + 2
    lines = ["ICP + TSDF fusion -- timing report", "=" * (w + 20),
             f"{'step':<{w}}{'time(s)':>10}{'%':>8}", "-" * (w + 18)]
    for lbl, sec in TIMING:
        lines.append(f"{lbl:<{w}}{sec:>10.2f}{100 * sec / total_wall:>8.1f}")
    lines.append("-" * (w + 18))
    lines.append(f"{'TOTAL (wall clock)':<{w}}{total_wall:>10.2f}{100.0:>8.1f}")
    lines += ["", *extra_lines]
    txt = "\n".join(lines)
    with open(TIMING_TXT, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\n[out] timing report -> {TIMING_TXT}")


def main():
    t_all = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t = time.time()
    sys.path.insert(0, PREV_DIR)
    sys.path.insert(0, MERGE_DIR)
    import sam2_static_mvp as prev
    import sam3_merge_mvp as merge
    TIMING.append(("0_imports (incl. torch)", time.time() - t))

    t = time.time()
    frames, intr = prev.load_dataset()
    root = prev.DATASET_DIR
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W, H = intr["w"], intr["h"]
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    N = len(frames)
    TIMING.append(("1_load_dataset", time.time() - t))

    # --- merged SAM IDs from the cached merge step (no SAM re-run) ---
    t = time.time()
    seg_ids = np.load(os.path.join(MERGE_OUT, "seg_ids.npz"))["seg_ids"]   # gripper-aware
    sam3_masks = np.load(os.path.join(MERGE_OUT, "_sam3_masks.npz"))["masks"].astype(bool)
    meta = json.load(open(os.path.join(MERGE_OUT, "_sam3_meta.json")))
    remap, _bt, _cov, target_for_newid, _present = merge.compute_merge(seg_ids[0], sam3_masks, meta)
    remap_lut = np.arange(int(seg_ids.max()) + 1, dtype=seg_ids.dtype)
    for k, v in remap.items():
        if k < len(remap_lut):
            remap_lut[k] = v
    merged = remap_lut[seg_ids]                                    # (N,H,W) merged ids
    table_id = None
    if target_for_newid:
        table_id = max(target_for_newid, key=lambda nid: meta["instances"][target_for_newid[nid]]["area"])
    TIMING.append(("2_merge_ids", time.time() - t))
    print(f"[merge] reused cached IDs; table merged-id = {table_id}")

    # --- precompute per-frame depth (gripper-zeroed) + valid masks + id-color ---
    t = time.time()
    print("[prep] loading depth + gripper masks ...")
    depth_u16, valid, idcol, c2w_cv = [], [], [], []
    for i, fr in enumerate(frames):
        z16 = cv2.imread(prev.abspath(root, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        grip = prev.load_gripper_mask(root, fr)
        z16 = z16.copy(); z16[grip] = 0                           # exclude gripper from fusion
        zf = z16.astype(np.float32) / DEPTH_SCALE
        v = (zf > 0.05) & (zf < DEPTH_TRUNC_M)
        depth_u16.append(z16)
        valid.append(v)
        idcol.append((prev.PALETTE[merged[i]] * 255.0).astype(np.uint8))   # RGB by merged id
        c2w_cv.append(cv_c2w(np.asarray(fr["transform_matrix"], dtype=np.float64)))
    TIMING.append(("3_preload_depth_masks", time.time() - t))

    # --- frame-to-model ICP (gripper already excluded via valid mask) ---
    print("[icp] refining poses (frame-to-model, point-to-plane) ...")
    p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER)

    def frame_cloud(i, c2w):
        world, _, _ = backproject_world(depth_u16[i].astype(np.float32) / DEPTH_SCALE, valid[i], c2w, fx, fy, cx, cy)
        pc = make_o3d(world).voxel_down_sample(ICP_VOXEL_M)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
        return pc

    t = time.time(); icp_only = 0.0
    refined = [c2w_cv[0].copy()]
    model = frame_cloud(0, c2w_cv[0])
    n_trusted = 0
    for i in range(1, N):
        src = frame_cloud(i, c2w_cv[i])                          # in world via init pose
        T = np.eye(4)
        ti = time.time()
        for dist in (ICP_COARSE_DIST, ICP_FINE_DIST):
            reg = o3d.pipelines.registration.registration_icp(src, model, dist, T, p2pl, crit)
            T = reg.transformation
        icp_only += time.time() - ti
        if reg.fitness >= ICP_FITNESS_MIN:
            refined.append(T @ c2w_cv[i]); src.transform(T); n_trusted += 1
        else:
            refined.append(c2w_cv[i].copy())                     # distrust -> keep init
        model += src
        model = model.voxel_down_sample(MODEL_VOXEL_M)
        model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
        if i % 20 == 0 or i == N - 1:
            print(f"[icp]   frame {i+1}/{N}  fitness={reg.fitness:.2f}  trusted={n_trusted}/{i}")
    TIMING.append((f"4_icp_refine ({N-1} frames)", time.time() - t))
    TIMING.append(("4a_icp_register_only", icp_only))
    print(f"[icp] avg {1000*icp_only/(N-1):.0f} ms/frame on registration ({n_trusted}/{N-1} trusted)")

    # --- build the comparison clouds ---
    counts = {}

    def concat(poses, path, tag):
        pts, cols = [], []
        for i in range(N):
            zf = depth_u16[i].astype(np.float32) / DEPTH_SCALE
            world, vv, uu = backproject_world(zf, valid[i], poses[i], fx, fy, cx, cy)
            pts.append(world.astype(np.float32))
            cols.append(prev.PALETTE[merged[i][vv, uu]])
        n = save_cloud(np.concatenate(pts), np.concatenate(cols), path)
        print(f"[out] {tag}: {n:,} pts -> {path}")
        return n

    print("[fuse] writing naive (init-pose) + refined (ICP-pose) concatenations ...")
    t = time.time(); counts["naive_concat"] = concat(c2w_cv, NAIVE_PLY, "naive_concat ")
    TIMING.append(("6_naive_concat", time.time() - t))
    t = time.time(); counts["refined_concat"] = concat(refined, REFINED_PLY, "refined_concat")
    TIMING.append(("7_refined_concat", time.time() - t))

    # --- TSDF fusion: real-RGB (seed color) + id (labels) + image-gradient (detail) ---
    print(f"[fuse] integrating TSDF (real-RGB + id + gradient) at {TSDF_VOXEL_M*1000:.1f} mm ...")
    t = time.time()

    def new_vol():
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=TSDF_VOXEL_M, sdf_trunc=TSDF_SDF_TRUNC_M,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def integrate(vol, color_u8, depth_img, ext):
        vol.integrate(o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_u8)), depth_img,
            depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC_M,
            convert_rgb_to_intensity=False), intr_o3d, ext)

    rgbvol, idvol, gradvol = new_vol(), new_vol(), new_vol()
    for i in range(N):
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_u16[i]))
        ext = np.linalg.inv(refined[i])                          # world->cam
        rgb_np = prev.load_rgb(root, frames[i])
        integrate(rgbvol, rgb_np, depth, ext)
        integrate(idvol, idcol[i], depth, ext)
        integrate(gradvol, image_gradient(rgb_np), depth, ext)   # fused image-gradient detail
    TIMING.append(("8_tsdf_integrate(x3)", time.time() - t))

    t = time.time()
    rgb_pc = rgbvol.extract_point_cloud()
    pos = np.asarray(rgb_pc.points)
    rgb = np.asarray(rgb_pc.colors)
    ids = recover_ids(pos, idvol.extract_point_cloud(), prev.PALETTE)   # per-point SAM id
    grad = recover_nn(pos, gradvol.extract_point_cloud())[:, 0]         # per-point image gradient
    TIMING.append(("9_extract+per_point_id+grad", time.time() - t))
    print(f"[fuse] fine TSDF cloud: {len(pos):,} pts")

    # --- adaptive ("active") density, driven by the fused image gradient ---
    t = time.time()
    if ADAPTIVE_DENSITY:
        keep = adaptive_keep(pos, grad)
        print(f"[adapt] {len(pos):,} -> {len(keep):,} pts "
              f"(high-gradient top {100*(1-ADAPT_Q[1]):.0f}% kept at {ADAPT_VOXEL[2]*1000:.1f}mm, "
              f"flat bottom {100*ADAPT_Q[0]:.0f}% at {ADAPT_VOXEL[0]*1000:.1f}mm)")
    else:
        keep = _grid_keep(pos, np.arange(len(pos)), TSDF_VOXEL_UNIFORM)
    TIMING.append(("10_adaptive_decimate", time.time() - t))

    # --- statistical outlier removal + save (real-RGB seed, id-colored, ids) ---
    t = time.time()
    P, C, ID = pos[keep], rgb[keep], ids[keep]
    seed = o3d.geometry.PointCloud()
    seed.points = o3d.utility.Vector3dVector(P)
    seed.colors = o3d.utility.Vector3dVector(C)
    seed, ind = seed.remove_statistical_outlier(nb_neighbors=SOR_NB, std_ratio=SOR_STD)
    ind = np.asarray(ind)
    o3d.io.write_point_cloud(SEED_PLY, seed)
    idseed = o3d.geometry.PointCloud()
    idseed.points = o3d.utility.Vector3dVector(P[ind])
    idseed.colors = o3d.utility.Vector3dVector(prev.PALETTE[ID[ind]])
    o3d.io.write_point_cloud(ID_PLY, idseed)
    np.save(IDS_NPY, ID[ind])
    counts["fused_seed"] = len(seed.points)
    TIMING.append(("11_sor+save_seed", time.time() - t))
    print(f"[out] fused_seed: {len(seed.points):,} pts (real-RGB) -> {SEED_PLY}")
    print(f"[out]   + id-colored {os.path.basename(ID_PLY)} + per-point ids {os.path.basename(IDS_NPY)}")

    compaction = counts["naive_concat"] / max(counts["refined_concat"], 1)
    extra = [f"point counts:  naive={counts['naive_concat']:,}  refined={counts['refined_concat']:,}  "
             f"fused_seed={counts['fused_seed']:,}  (adaptive={ADAPTIVE_DENSITY}, base voxel={TSDF_VOXEL_M*1000:.1f} mm)",
             f"icp: {n_trusted}/{N-1} frames trusted, {1000*icp_only/(N-1):.0f} ms/frame on registration",
             f"denoising: ICP collapses multi-view spread -> refined_concat has {compaction:.1f}x fewer "
             f"points than naive_concat at {CONCAT_VOXEL_M*1000:.0f} mm voxel (tighter surfaces)"]
    write_timing_report(time.time() - t_all, extra)

    print("\n[done] outputs in", OUTPUT_DIR)
    print("       fused_seed.ply (real-RGB GS seed)  fused_seed_idcolor.ply  fused_seed_ids.npy")
    print("       refined_concat.ply / naive_concat.ply (comparison)  timing_report.txt")


if __name__ == "__main__":
    main()
