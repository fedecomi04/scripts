"""ICP-refined TSDF fusion of the static-scene RGB-D frames -> the Gaussian
Splatting init seed point cloud.

Replaces the naive multi-view back-projection that
``live_ros_publisher.build_static_init_pointcloud`` writes as
``<data_root>/static_scene/depth_camera_init_points.ply`` (transforms.json's
``ply_file_path``). The naive seed back-projects every frame and concatenates,
producing ~N slightly-misaligned copies of every surface (small pose +
depth errors -> visibly "thick" walls and floor). This module:

1. Per-frame: depth = ``static_scene/depth/<file>.tiff`` (uint16 mm / 1000 -> m)
   with the gripper-keep mask zeroing out the robot arm so it does not
   corrupt either ICP or TSDF fusion. Objects on the table are static and
   are NOT masked.
2. Frame-to-model point-to-plane ICP refines each frame's pose, initialised
   from transforms.json's ``transform_matrix`` (OpenGL c2w convention).
3. Open3D ``ScalableTSDFVolume`` fuses real RGB into a single averaged
   surface (per-view depth noise cancels).
4. Adaptive density: a second TSDF channel fuses per-frame Sobel image-gradient
   magnitude; the final cloud is decimated in 3 gradient quantile tiers so
   flat regions get a coarse voxel and high-detail regions stay fine.
5. Statistical outlier removal, write as the seed PLY (same path the
   transforms.json points at, so Splatfacto / nerfstudio's
   ``load_3D_points=True`` picks it up unchanged).

Conventions:
* nerfstudio OpenGL c2w -> Open3D OpenCV extrinsic = ``inv(c2w @ diag(1,-1,-1,1))``.
* Dataparser uses ``orientation_method="none"``, ``center_method="none"``,
  ``auto_scale_poses=False``, so the seed must be in the SAME unrecentered
  world frame as transforms.json (no rescaling, no re-centering).
* PLY MUST carry RGB colors (Splatfacto inits ``features_dc`` from them).

SAM-free port of ``experiments/icp_fusion_mvp/icp_fusion_mvp.py``: the
prototype additionally produced an id-colored cloud + per-point SAM ids,
which the init seed does not need.

Idempotent: if the seed PLY is newer than ``transforms.json`` and the
``rgb/`` / ``depth/`` / ``masks/`` dirs, the fusion is skipped (call with
``force=True`` to override).

CLI usage:
    python -m dynamic_gs.utils.rgbd_fusion_init /path/to/data_root [--force]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# -------------------------------------------------------------------------
# Tunables. Kept as module-level constants (not a config dataclass) so the
# values stay readable in one place; the reference prototype validated
# these on the recorded teleop dataset.
# -------------------------------------------------------------------------

# ICP (frame-to-model, point-to-plane)
ICP_VOXEL_M = 0.01
MODEL_VOXEL_M = 0.01
NORMAL_RADIUS_M = 0.03
ICP_COARSE_DIST = 0.05
ICP_FINE_DIST = 0.02
ICP_MAX_ITER = 40
ICP_FITNESS_MIN = 0.30  # below this, distrust ICP and keep the init pose

# TSDF fusion
TSDF_VOXEL_M = 0.0025          # 2.5 mm fine base extraction
TSDF_SDF_TRUNC_M = 0.0125      # ~5 x voxel
DEPTH_TRUNC_M = 3.0
DEPTH_SCALE = 1000.0           # uint16 mm -> m

# Adaptive ("active") density
ADAPTIVE_DENSITY = True
TSDF_VOXEL_UNIFORM = 0.0028    # used when ADAPTIVE_DENSITY=False
GRAD_SCALE = 0.25              # Sobel magnitude -> uint8 scale (cross-frame stable)
GRAD_BLUR_KSIZE = 5            # blur so detailed regions (not 1-px edges) read as high
ADAPT_Q = (0.50, 0.85)         # gradient quantiles -> tier boundaries
ADAPT_VOXEL = (0.009, 0.0045, 0.0025)  # flat / mid / detailed (m)

# Cleanup
SOR_NB = 20
SOR_STD = 2.0

INIT_PLY_NAME = "depth_camera_init_points.ply"  # matches publisher / transforms.json


# -------------------------------------------------------------------------
# Conventions
# -------------------------------------------------------------------------

def cv_c2w_from_opengl(c2w_opengl: np.ndarray) -> np.ndarray:
    """nerfstudio OpenGL camera-to-world -> OpenCV camera-to-world."""
    return c2w_opengl @ np.diag([1.0, -1.0, -1.0, 1.0])


def _backproject_world(depth_m, valid, c2w_cv, fx, fy, cx, cy):
    vv, uu = np.where(valid)
    zz = depth_m[vv, uu]
    x = (uu - cx) * zz / fx
    y = (vv - cy) * zz / fy
    cam = np.stack([x, y, zz, np.ones_like(zz)], axis=1)
    world = (c2w_cv @ cam.T).T[:, :3]
    return world


def _make_o3d_with_normals(points: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS_M, max_nn=30))
    return pc


# -------------------------------------------------------------------------
# Adaptive density
# -------------------------------------------------------------------------

def _image_gradient(rgb_u8: np.ndarray) -> np.ndarray:
    """Per-pixel Sobel magnitude -> a uint8 grayscale-in-RGB image on a
    fixed cross-frame scale, lightly blurred so detail REGIONS (not 1-px
    edges) read as high. The output shape (H, W, 3) is what Open3D's TSDF
    color path expects."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if GRAD_BLUR_KSIZE > 1:
        mag = cv2.GaussianBlur(mag, (GRAD_BLUR_KSIZE, GRAD_BLUR_KSIZE), 0)
    mag8 = np.clip(mag * GRAD_SCALE, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(mag8[:, :, None], 3, axis=2))


def _nn_lookup(pos: np.ndarray, src_pc: o3d.geometry.PointCloud) -> np.ndarray:
    """Nearest-neighbour color/value lookup of ``src_pc`` for each pos."""
    nn = cKDTree(np.asarray(src_pc.points)).query(pos, k=1)[1]
    return np.asarray(src_pc.colors)[nn]


def _grid_keep(pos: np.ndarray, idx: np.ndarray, vx: float) -> np.ndarray:
    """One representative point per voxel (preserves colors -> sharp text)."""
    if len(idx) == 0:
        return idx
    key = np.floor((pos[idx] - pos[idx].min(0)) / vx).astype(np.int64)
    _, u = np.unique(key, axis=0, return_index=True)
    return idx[u]


def _adaptive_keep(pos: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """3-tier decimation: flat regions coarse, detailed regions fine."""
    qlo, qhi = np.quantile(detail, ADAPT_Q)
    tiers = [
        (detail < qlo, ADAPT_VOXEL[0]),
        ((detail >= qlo) & (detail < qhi), ADAPT_VOXEL[1]),
        (detail >= qhi, ADAPT_VOXEL[2]),
    ]
    keeps = [_grid_keep(pos, np.where(m)[0], vx) for m, vx in tiers if m.any()]
    return np.concatenate(keeps) if keeps else np.empty(0, dtype=np.int64)


# -------------------------------------------------------------------------
# Dataset loader -- matches the publisher's transforms.json schema
# -------------------------------------------------------------------------

@dataclass
class _Frame:
    rgb_path: Path
    depth_path: Path
    mask_path: Optional[Path]
    c2w_opengl: np.ndarray  # (4,4)


def _load_static_dataset(static_dir: Path) -> tuple[list[_Frame], dict]:
    """Read static_scene/transforms.json + resolve per-frame paths.

    transforms.json schema (from live_ros_publisher.start_recording):
        {"fl_x", "fl_y", "cx", "cy", "w", "h", "frames": [...]}
    Per-frame:
        {"file_path": "./rgb/arm_NNNNN.png",
         "depth_file_path": "./depth/arm_NNNNN.tiff",
         "mask_path": "./masks/arm_NNNNN.png",
         "transform_matrix": [[..4..]*4]}    # OpenGL c2w
    """
    tj_path = static_dir / "transforms.json"
    if not tj_path.exists():
        raise FileNotFoundError(f"missing {tj_path}")
    tj = json.loads(tj_path.read_text())

    intr = {
        "fx": float(tj["fl_x"]), "fy": float(tj["fl_y"]),
        "cx": float(tj["cx"]),   "cy": float(tj["cy"]),
        "w":  int(tj["w"]),      "h":  int(tj["h"]),
    }

    def _resolve(p: str) -> Path:
        return (static_dir / p).resolve()

    frames: list[_Frame] = []
    for fr in tj["frames"]:
        rgb_path = _resolve(fr["file_path"])
        depth_path = _resolve(fr["depth_file_path"])
        mask_path = _resolve(fr["mask_path"]) if fr.get("mask_path") else None
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"bad transform_matrix shape {c2w.shape} in frame {fr}")
        frames.append(_Frame(rgb_path, depth_path, mask_path, c2w))
    return frames, intr


def _load_rgb_rgb_order(path: Path) -> np.ndarray:
    """Load PNG as uint8 (H,W,3) in RGB order. Publisher writes BGR via
    cv2.imwrite, so we read as BGR then swap."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth_u16_mm(path: Path) -> np.ndarray:
    z = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if z is None:
        raise FileNotFoundError(f"could not read {path}")
    return z.astype(np.uint16)


def _load_gripper_keep_mask(path: Optional[Path], h: int, w: int) -> np.ndarray:
    """Load the per-frame gripper-keep mask. Returns a bool array of shape
    (H, W) where True = robot arm (drop), False = scene (keep).

    Mask convention on disk: 255 = keep (background), 0 = drop (robot arm).
    This function returns the INVERSION (True = the arm, to be zeroed out)
    so callers can do ``depth[mask] = 0``.

    If no mask is provided, returns all-False (don't drop anything)."""
    if path is None or not path.exists():
        return np.zeros((h, w), dtype=bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), dtype=bool)
    return m == 0


# -------------------------------------------------------------------------
# Idempotency: skip fusion if the output is newer than all inputs
# -------------------------------------------------------------------------

def _output_is_fresh(out_ply: Path, static_dir: Path) -> bool:
    if not out_ply.exists():
        return False
    out_mtime = out_ply.stat().st_mtime
    sentinels = [
        static_dir / "transforms.json",
        static_dir / "rgb",
        static_dir / "depth",
        static_dir / "masks",
    ]
    for s in sentinels:
        if not s.exists():
            continue
        if s.is_dir():
            for f in s.iterdir():
                if f.stat().st_mtime > out_mtime:
                    return False
        else:
            if s.stat().st_mtime > out_mtime:
                return False
    return True


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def build_tsdf_seed(
    data_root: Path | str,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Run ICP+TSDF fusion on ``<data_root>/static_scene/`` and write the
    resulting RGB-colored point cloud as the init seed PLY (overwrites
    the naive seed produced by the publisher). Returns the output path.

    Idempotent: returns immediately if the existing seed is newer than
    every input file. Pass ``force=True`` to recompute regardless."""

    data_root = Path(data_root).resolve()
    static_dir = data_root / "static_scene"
    out_ply = static_dir / INIT_PLY_NAME

    if not force and _output_is_fresh(out_ply, static_dir):
        if verbose:
            print(f"[rgbd-fusion-init] up to date: {out_ply} (use force=True to rebuild)")
        return out_ply

    t_all = time.time()
    if verbose:
        print(f"[rgbd-fusion-init] starting ICP + TSDF fusion on {static_dir}")

    frames, intr = _load_static_dataset(static_dir)
    N = len(frames)
    if N == 0:
        raise RuntimeError(f"no frames listed in {static_dir / 'transforms.json'}")
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W, H = intr["w"], intr["h"]
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    if verbose:
        print(f"[rgbd-fusion-init] {N} frames, {W}x{H}, "
              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # --- Preload depth (gripper-zeroed) + valid masks + c2w_cv ---
    t = time.time()
    depth_u16: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    c2w_cv: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        z16 = _load_depth_u16_mm(fr.depth_path).copy()
        drop = _load_gripper_keep_mask(fr.mask_path, z16.shape[0], z16.shape[1])
        z16[drop] = 0
        zf = z16.astype(np.float32) / DEPTH_SCALE
        v = (zf > 0.05) & (zf < DEPTH_TRUNC_M)
        depth_u16.append(z16)
        valid.append(v)
        c2w_cv.append(cv_c2w_from_opengl(fr.c2w_opengl))
    if verbose:
        print(f"[rgbd-fusion-init] preload depth+masks: {time.time()-t:.2f}s")

    # --- Frame-to-model ICP refinement ---
    t = time.time()
    p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER)

    def _frame_cloud(i: int, c2w: np.ndarray) -> o3d.geometry.PointCloud:
        world = _backproject_world(
            depth_u16[i].astype(np.float32) / DEPTH_SCALE,
            valid[i], c2w, fx, fy, cx, cy,
        )
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(world.astype(np.float64))
        pc = pc.voxel_down_sample(ICP_VOXEL_M)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_M, max_nn=30))
        return pc

    refined: list[np.ndarray] = [c2w_cv[0].copy()]
    model = _frame_cloud(0, c2w_cv[0])
    n_trusted = 0
    icp_only = 0.0
    for i in range(1, N):
        src = _frame_cloud(i, c2w_cv[i])  # in world via init pose
        T = np.eye(4)
        ti = time.time()
        last_fitness = 0.0
        for dist in (ICP_COARSE_DIST, ICP_FINE_DIST):
            reg = o3d.pipelines.registration.registration_icp(src, model, dist, T, p2pl, crit)
            T = reg.transformation
            last_fitness = reg.fitness
        icp_only += time.time() - ti
        if last_fitness >= ICP_FITNESS_MIN:
            refined.append(T @ c2w_cv[i])
            src.transform(T)
            n_trusted += 1
        else:
            refined.append(c2w_cv[i].copy())  # distrust -> keep init
        model += src
        model = model.voxel_down_sample(MODEL_VOXEL_M)
        model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_M, max_nn=30))
        if verbose and (i % 20 == 0 or i == N - 1):
            print(f"[rgbd-fusion-init]   icp frame {i+1}/{N}  "
                  f"fitness={last_fitness:.2f}  trusted={n_trusted}/{i}")
    if verbose:
        print(f"[rgbd-fusion-init] icp total: {time.time()-t:.2f}s "
              f"({1000*icp_only/max(N-1,1):.0f} ms/frame avg, "
              f"{n_trusted}/{N-1} trusted)")

    # --- TSDF fusion: real RGB + per-frame image gradient ---
    t = time.time()

    def _new_vol() -> o3d.pipelines.integration.ScalableTSDFVolume:
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=TSDF_VOXEL_M,
            sdf_trunc=TSDF_SDF_TRUNC_M,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def _integrate(vol, color_u8, depth_img, ext):
        vol.integrate(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color_u8)),
                depth_img,
                depth_scale=DEPTH_SCALE,
                depth_trunc=DEPTH_TRUNC_M,
                convert_rgb_to_intensity=False,
            ),
            intr_o3d,
            ext,
        )

    rgbvol = _new_vol()
    gradvol = _new_vol() if ADAPTIVE_DENSITY else None
    for i, fr in enumerate(frames):
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_u16[i]))
        ext = np.linalg.inv(refined[i])  # world -> cam
        rgb_np = _load_rgb_rgb_order(fr.rgb_path)
        _integrate(rgbvol, rgb_np, depth_img, ext)
        if gradvol is not None:
            _integrate(gradvol, _image_gradient(rgb_np), depth_img, ext)
    if verbose:
        n_vols = 2 if gradvol is not None else 1
        print(f"[rgbd-fusion-init] tsdf integrate ({n_vols} volume(s)): {time.time()-t:.2f}s")

    # --- Extract + adaptive decimate + SOR ---
    t = time.time()
    rgb_pc = rgbvol.extract_point_cloud()
    pos = np.asarray(rgb_pc.points)
    rgb = np.asarray(rgb_pc.colors)
    if verbose:
        print(f"[rgbd-fusion-init] fine tsdf cloud: {len(pos):,} pts")

    if ADAPTIVE_DENSITY and gradvol is not None:
        grad = _nn_lookup(pos, gradvol.extract_point_cloud())[:, 0]
        keep = _adaptive_keep(pos, grad)
        if verbose:
            print(f"[rgbd-fusion-init] adaptive decimate: {len(pos):,} -> {len(keep):,} pts "
                  f"(top {100*(1-ADAPT_Q[1]):.0f}% @ {ADAPT_VOXEL[2]*1000:.1f}mm, "
                  f"bottom {100*ADAPT_Q[0]:.0f}% @ {ADAPT_VOXEL[0]*1000:.1f}mm)")
    else:
        keep = _grid_keep(pos, np.arange(len(pos)), TSDF_VOXEL_UNIFORM)
        if verbose:
            print(f"[rgbd-fusion-init] uniform decimate @ {TSDF_VOXEL_UNIFORM*1000:.1f}mm: "
                  f"{len(pos):,} -> {len(keep):,} pts")

    P, C = pos[keep], rgb[keep]
    seed = o3d.geometry.PointCloud()
    seed.points = o3d.utility.Vector3dVector(P)
    seed.colors = o3d.utility.Vector3dVector(C)
    seed, _ = seed.remove_statistical_outlier(nb_neighbors=SOR_NB, std_ratio=SOR_STD)
    n_final = len(seed.points)
    if verbose:
        print(f"[rgbd-fusion-init] sor: {len(P):,} -> {n_final:,} pts")

    # --- Write (overwrites the naive seed in place) ---
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), seed)
    if verbose:
        print(f"[rgbd-fusion-init] wrote {out_ply} ({n_final:,} pts) "
              f"in {time.time()-t_all:.1f}s total")
    return out_ply


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def _main_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("data_root", type=Path,
                    help="Dataset root containing static_scene/{transforms.json, rgb/, depth/, masks/}.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the seed PLY is newer than every input.")
    args = ap.parse_args()
    build_tsdf_seed(args.data_root, force=args.force)


if __name__ == "__main__":
    _main_cli()
