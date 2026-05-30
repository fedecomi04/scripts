"""K-image AnySplat → world frame via scene-intrinsics back-projection (+ICP).

Geometry is owned entirely by the scene: each gaussian is placed at the world point
obtained by back-projecting (u, v, sensor_depth) through the **scene's** intrinsics
and scene_c2w. AnySplat is used ONLY for:
    * per-pixel color (features_dc, features_rest)
    * per-pixel splat scale + orientation (relative to the predicted camera)
    * the (u, v) → gaussian mapping (via projection through pred_K of canonical xyz)

This eliminates the lateral-offset bug caused by AnySplat predicting wrong intrinsics
(pred fx ≈ 190 vs scene fx ≈ 267 at 448×448).

Per gaussian:
    1. Project canonical position through pred_K → pred-pixel (u, v)
    2. Look up sensor depth at (u, v)
    3. Back-project (u, v, sensor_depth) through scene_K (OpenGL convention) → camera-frame xyz
    4. world_xyz = R_scene @ camera_xyz + t_scene
    5. log_scale_world = log_scale + log(d_sensor / z_canonical)
    6. quat_world = (R_scene @ flip @ R_pred^T) @ quat_canonical (as rotmat then back to wxyz)

For pixels with no valid sensor depth: fall back to AnySplat's predicted depth scaled
by the global s_fit (median sensor/pred ratio over a center patch). Pass
--drop-no-sensor-depth to drop them instead.

ICP refinement against the static-scene gaussian centres runs at the end (if --post-fusion
is given), cleaning up residual rotation/translation slop.

Usage (one-liner from any cwd):

    python /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/anysplat_patch_to_scene.py \\
        --image /path/to/static/rgb/arm_05011.png \\
        --depth /path/to/static/depth/arm_05011.tiff \\
        --scene-transforms /path/to/static/transforms.json \\
        --post-fusion /path/to/static/post_fusion_state.pt \\
        --out /home/mrc-cuhk/Desktop/anysplat_world_sceneK.ply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "third_party" / "AnySplat"))

from dynamic_gs.utils.anysplat_decode import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz  # noqa: E402
from src.model.model.anysplat import AnySplat  # noqa: E402
from src.model.ply_export import export_ply    # noqa: E402
from src.utils.image import process_image      # noqa: E402

SH_C0 = 0.28209479177387814
BG_RGB = np.array([0.86, 0.92, 1.0], dtype=np.float32)
BG_TOL = 0.08
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])     # for rotations (positions handled by formula)


def _load_scene_camera(transforms_path: Path, image_path: Path) -> tuple[np.ndarray, dict]:
    """Return (scene_c2w_4x4, scene_intrinsics_dict)."""
    with open(transforms_path) as f:
        tfs = json.load(f)
    stem = image_path.stem
    c2w = None
    for fr in tfs["frames"]:
        if Path(str(fr["file_path"])).stem == stem:
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            break
    if c2w is None:
        raise KeyError(f"frame {stem} not in {transforms_path}")
    intr = {
        "w": int(tfs["w"]), "h": int(tfs["h"]),
        "fl_x": float(tfs["fl_x"]), "fl_y": float(tfs["fl_y"]),
        "cx": float(tfs["cx"]), "cy": float(tfs["cy"]),
    }
    return c2w, intr


def _save_ply(means, log_scales, quats_wxyz, opacities_sig, features_dc, features_rest, out_path: Path):
    full_sh = np.concatenate([features_dc[:, None, :], features_rest], axis=1)
    harm = torch.from_numpy(np.transpose(full_sh, (0, 2, 1)).astype(np.float32))
    export_ply(
        means=torch.from_numpy(means.astype(np.float32)),
        scales=torch.from_numpy(np.exp(log_scales).astype(np.float32)),
        rotations=torch.from_numpy(quats_wxyz.astype(np.float32)),
        harmonics=harm,
        opacities=torch.from_numpy(opacities_sig.astype(np.float32)),
        path=out_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append", required=True, type=Path,
                    help="Input image. First is TARGET; extras are context views.")
    ap.add_argument("--depth", type=Path, required=True,
                    help="Sensor depth for TARGET image (uint16 PNG/TIFF × 1e-3 = m).")
    ap.add_argument("--scene-transforms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=50,
                    help="Center patch for the global-s fallback.")
    ap.add_argument("--opacity-min", type=float, default=0.05)
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=10.0)
    ap.add_argument("--drop-no-sensor-depth", action="store_true",
                    help="Drop gaussians without valid sensor depth instead of falling back to global s.")
    ap.add_argument("--post-fusion", type=Path, default=None,
                    help="Static-scene checkpoint. If given, run ICP refinement.")
    ap.add_argument("--icp-max-distance", type=float, default=0.05)
    ap.add_argument("--icp-max-iter", type=int, default=30)
    ap.add_argument("--icp-scene-subsample", type=int, default=200_000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = len(args.image)
    print(f"[anysplat→world] K = {K}  target = {args.image[0].name}")

    # --- AnySplat inference ---
    model = AnySplat.from_pretrained("lhjiang/anysplat").to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    imgs = torch.stack([process_image(str(p)) for p in args.image], dim=0).unsqueeze(0).to(device)
    H, W = 448, 448
    with torch.no_grad():
        gaussians, pred = model.inference((imgs + 1) * 0.5)

    # --- Predicted depth at target view (for global-s fallback only) ---
    near_t = torch.full((1, 1), args.near, device=device)
    far_t  = torch.full((1, 1), args.far,  device=device)
    with torch.no_grad():
        out0 = model.decoder(gaussians, pred["extrinsic"][:, :1], pred["intrinsic"][:, :1],
                             near_t, far_t, image_shape=(H, W))
    pred_depth = out0.depth.detach().cpu().numpy()
    while pred_depth.ndim > 2: pred_depth = pred_depth.squeeze(0)

    # --- Sensor depth + global s fallback ---
    sensor = np.array(Image.open(args.depth)).astype(np.float32) * 1e-3
    sensor_r = np.array(Image.fromarray(sensor).resize((W, H), Image.NEAREST))
    cy0, cx0 = H // 2 - args.patch_size // 2, W // 2 - args.patch_size // 2
    cy1, cx1 = cy0 + args.patch_size, cx0 + args.patch_size
    sp = sensor_r[cy0:cy1, cx0:cx1]; pp = pred_depth[cy0:cy1, cx0:cx1]
    valid_patch = (sp > 0.01) & (pp > 1e-4)
    s_fit_global = float(np.median(sp[valid_patch] / pp[valid_patch]))
    print(f"[anysplat→world] global-s fallback = {s_fit_global:.4f}")

    # --- Gaussian params + opacity + background filter ---
    means_can = gaussians.means[0].detach().cpu().numpy().astype(np.float64)
    log_scales = torch.log(gaussians.scales[0].clamp(min=1e-12)).detach().cpu().numpy().astype(np.float32)
    quats = gaussians.rotations[0].detach().cpu().numpy().astype(np.float32)
    opacities_sig = gaussians.opacities[0].detach().cpu().numpy().astype(np.float32)
    features_dc = gaussians.harmonics[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
    features_rest = np.transpose(
        gaussians.harmonics[0, :, :, 1:16].detach().cpu().numpy().astype(np.float32), (0, 2, 1)
    )
    keep = opacities_sig >= args.opacity_min
    means_can, log_scales, quats = means_can[keep], log_scales[keep], quats[keep]
    opacities_sig, features_dc, features_rest = opacities_sig[keep], features_dc[keep], features_rest[keep]
    rgb_pred = features_dc * SH_C0 + 0.5
    keep_bg = ~np.all(np.abs(rgb_pred - BG_RGB) <= BG_TOL, axis=-1)
    means_can = means_can[keep_bg]; log_scales = log_scales[keep_bg]; quats = quats[keep_bg]
    opacities_sig = opacities_sig[keep_bg]; features_dc = features_dc[keep_bg]; features_rest = features_rest[keep_bg]
    N = means_can.shape[0]
    print(f"[anysplat→world] kept {N} gaussians after filters")

    # --- Compute pred-camera frame position for each gaussian ---
    pred_c2w_0 = pred["extrinsic"][0, 0].detach().cpu().numpy().astype(np.float64)
    R_pred = pred_c2w_0[:3, :3]; t_pred = pred_c2w_0[:3, 3]
    p_cam_cv = ((means_can - t_pred) @ R_pred).astype(np.float64)   # R_pred^T @ (means - t_pred)
    z_cam = p_cam_cv[:, 2]

    # --- (u, v) via PRED intrinsics (this is where AnySplat thinks the gaussian is) ---
    pred_K_norm = pred["intrinsic"][0, 0].detach().cpu().numpy().astype(np.float64)
    fx_pred = pred_K_norm[0, 0] * W; fy_pred = pred_K_norm[1, 1] * H
    cx_pred = pred_K_norm[0, 2] * W; cy_pred = pred_K_norm[1, 2] * H
    safe_z = np.where(z_cam > 1e-6, z_cam, 1.0)
    u = fx_pred * p_cam_cv[:, 0] / safe_z + cx_pred
    v = fy_pred * p_cam_cv[:, 1] / safe_z + cy_pred
    in_image = (z_cam > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_idx = np.clip(u.astype(np.int64), 0, W - 1)
    v_idx = np.clip(v.astype(np.int64), 0, H - 1)

    # --- Sensor depth per gaussian ---
    sensor_per_gauss = np.where(in_image, sensor_r[v_idx, u_idx], 0.0).astype(np.float64)
    valid_sensor = in_image & (sensor_per_gauss > 0.01)
    n_valid = int(valid_sensor.sum())
    print(f"[anysplat→world] valid sensor depth: {n_valid} / {N} ({100*n_valid/N:.1f}%)")

    # Fill in fallback depth where sensor invalid (= AnySplat's predicted depth × global s)
    fallback_depth = z_cam * s_fit_global
    d_per_gauss = np.where(valid_sensor, sensor_per_gauss, fallback_depth)

    if args.drop_no_sensor_depth:
        means_can = means_can[valid_sensor]; log_scales = log_scales[valid_sensor]; quats = quats[valid_sensor]
        opacities_sig = opacities_sig[valid_sensor]; features_dc = features_dc[valid_sensor]; features_rest = features_rest[valid_sensor]
        u = u[valid_sensor]; v = v[valid_sensor]; z_cam = z_cam[valid_sensor]
        d_per_gauss = sensor_per_gauss[valid_sensor]
        N = means_can.shape[0]
        print(f"[anysplat→world] --drop-no-sensor-depth: kept {N}")

    # --- Back-project (u, v, d) through SCENE intrinsics in OpenGL convention ---
    # Scene K applied at the (448, 448) image grid (sensor depth is resized to this).
    scene_c2w, scene_intr = _load_scene_camera(args.scene_transforms, args.image[0])
    fx_s = scene_intr["fl_x"] * W / scene_intr["w"]
    fy_s = scene_intr["fl_y"] * H / scene_intr["h"]
    cx_s = scene_intr["cx"]   * W / scene_intr["w"]
    cy_s = scene_intr["cy"]   * H / scene_intr["h"]
    print(f"[anysplat→world] scene K @ {W}x{H}: fx={fx_s:.1f} fy={fy_s:.1f} cx={cx_s:.1f} cy={cy_s:.1f}")

    # OpenGL camera frame: x = (u-cx)/fx * d,  y = -(v-cy)/fy * d,  z = -d
    p_cam_gl = np.stack([
        d_per_gauss * (u - cx_s) / fx_s,
        -d_per_gauss * (v - cy_s) / fy_s,
        -d_per_gauss,
    ], axis=-1)  # (N, 3)

    R_scene = scene_c2w[:3, :3]; t_scene = scene_c2w[:3, 3]
    means_world = (R_scene @ p_cam_gl.T).T + t_scene

    # --- Per-gauss scale: keep same image-space footprint at new depth ---
    safe_z2 = np.where(z_cam > 1e-6, z_cam, 1.0)
    s_per_gauss = d_per_gauss / safe_z2
    log_scales_world = log_scales + np.log(np.clip(s_per_gauss, 1e-9, None))[:, None].astype(np.float32)

    # --- Rotation: canonical-CV → scene-GL = R_scene @ flip @ R_pred^T applied to R_g_can ---
    M_rot = R_scene @ OPENCV_TO_OPENGL @ R_pred.T                     # (3, 3)
    Rg_can = quat_wxyz_to_rotmat(quats).astype(np.float64)            # (N, 3, 3)
    Rg_world = (M_rot[None, :, :] @ Rg_can).astype(np.float32)
    quats_world = rotmat_to_quat_wxyz(Rg_world).astype(np.float32)

    print(f"[anysplat→world] pre-ICP bbox: x[{means_world[:,0].min():.2f},{means_world[:,0].max():.2f}] "
          f"y[{means_world[:,1].min():.2f},{means_world[:,1].max():.2f}] "
          f"z[{means_world[:,2].min():.2f},{means_world[:,2].max():.2f}]")

    # --- Optional ICP refine ---
    if args.post_fusion is not None:
        blob = torch.load(args.post_fusion, map_location="cpu", weights_only=False)
        sd = blob.get("model_state_dict", blob)
        scene_means_full = sd["gauss_params.means"].detach().cpu().numpy().astype(np.float64)
        scene_opac_logit = sd["gauss_params.opacities"].detach().cpu().numpy().reshape(-1)
        scene_means_full = scene_means_full[1.0 / (1.0 + np.exp(-scene_opac_logit)) >= args.opacity_min]
        if args.icp_scene_subsample and scene_means_full.shape[0] > args.icp_scene_subsample:
            idx = np.random.default_rng(0).choice(scene_means_full.shape[0], args.icp_scene_subsample, replace=False)
            scene_means_full = scene_means_full[idx]
        print(f"[anysplat→world] ICP: src={N} target={scene_means_full.shape[0]} max_dist={args.icp_max_distance}m")

        src_pcd = o3d.geometry.PointCloud(); src_pcd.points = o3d.utility.Vector3dVector(means_world)
        tgt_pcd = o3d.geometry.PointCloud(); tgt_pcd.points = o3d.utility.Vector3dVector(scene_means_full)
        result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            max_correspondence_distance=float(args.icp_max_distance),
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(args.icp_max_iter)),
        )
        T_icp = np.asarray(result.transformation)
        R_icp = T_icp[:3, :3]; t_icp = T_icp[:3, 3]
        print(f"[anysplat→world] ICP: fitness={result.fitness:.4f} inlier_rmse={result.inlier_rmse:.5f}  "
              f"|t_icp|={np.linalg.norm(t_icp):.4f}  |R_icp-I|={np.linalg.norm(R_icp - np.eye(3)):.4f}")
        means_world = (R_icp @ means_world.T).T + t_icp
        Rg_refined = (R_icp[None, :, :] @ Rg_world).astype(np.float32)
        quats_world = rotmat_to_quat_wxyz(Rg_refined).astype(np.float32)
        print(f"[anysplat→world] post-ICP bbox: x[{means_world[:,0].min():.2f},{means_world[:,0].max():.2f}] "
              f"y[{means_world[:,1].min():.2f},{means_world[:,1].max():.2f}] "
              f"z[{means_world[:,2].min():.2f},{means_world[:,2].max():.2f}]")

    _save_ply(
        means=means_world, log_scales=log_scales_world, quats_wxyz=quats_world,
        opacities_sig=opacities_sig, features_dc=features_dc, features_rest=features_rest,
        out_path=args.out,
    )
    print(f"[anysplat→world] wrote {args.out}  (N={N})")


if __name__ == "__main__":
    main()
