"""Direct RGB-D feedforward decode for hole-fill.

Closed-form per-pixel back-projection over a 2D component mask in a target
frame. No model, no checkpoint, no subprocess — runs entirely on the GPU
inside the main env in well under a millisecond per typical component.

Convention (validated by ``scripts/feedforward_depth_convention_check.py``):

    dir_cam_opengl = ((u - cx) / fx, -(v - cy) / fy, -1)
    ray_d_un_world = R_c2w @ dir_cam_opengl
    xyz_world      = c2w_translation + ray_d_un_world * z_depth

where z_depth is the +z (OpenCV-style) sensor depth in metres. The
Nerfstudio Cameras object stores c2w in the OpenGL convention
(forward = -z, up = +y).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 is a hard dep elsewhere in this repo
    _HAS_CV2 = False

from nerfstudio.utils.spherical_harmonics import RGB2SH

try:
    from scipy.ndimage import label as _scipy_label  # type: ignore
    from scipy.ndimage import binary_erosion as _scipy_binary_erosion  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _camera_intrinsics(target_camera) -> tuple[float, float, float, float, int, int]:
    """Extract (fx, fy, cx, cy, width, height) from a single-frame Cameras."""

    def _as_scalar(x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().reshape(-1)[0].item())
        return float(x)

    fx = _as_scalar(target_camera.fx)
    fy = _as_scalar(target_camera.fy)
    cx = _as_scalar(target_camera.cx)
    cy = _as_scalar(target_camera.cy)
    width = int(_as_scalar(target_camera.width))
    height = int(_as_scalar(target_camera.height))
    return fx, fy, cx, cy, width, height


def _camera_c2w(target_camera) -> torch.Tensor:
    """Return c2w as a (3, 4) torch tensor on the camera's device."""

    c2w = target_camera.camera_to_worlds
    if c2w.ndim == 3:
        c2w = c2w[0]
    return c2w


def _bilateral_smooth_depth(
    depth_m: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    sigma_space: float = 2.0,
    sigma_depth: float = 0.02,
) -> torch.Tensor:
    """Bilateral filter depth in metres, ignoring invalid (==0) pixels.

    Run on CPU via ``cv2.bilateralFilter`` (sub-millisecond at 800x800).
    """

    if not _HAS_CV2 or radius <= 0:
        return depth_m

    depth_np = depth_m.detach().cpu().numpy().astype(np.float32)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    depth_np = np.where(valid_np, depth_np, 0.0)
    kernel = 2 * int(radius) + 1
    smoothed = cv2.bilateralFilter(depth_np, d=kernel, sigmaColor=float(sigma_depth), sigmaSpace=float(sigma_space))
    smoothed = np.where(valid_np, smoothed, depth_np)
    return torch.from_numpy(smoothed).to(depth_m.device, dtype=depth_m.dtype)


def _backproject_world(
    u: torch.Tensor, v: torch.Tensor, depth_z: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    c2w: torch.Tensor,
) -> torch.Tensor:
    """OpenGL/Nerfstudio convention back-projection (verified by Gate).

    Inputs are 1D tensors of length N (one entry per pixel). Returns (N, 3).
    """

    u = u.to(c2w.dtype)
    v = v.to(c2w.dtype)
    depth_z = depth_z.to(c2w.dtype)
    dir_cam = torch.stack(
        [
            (u - cx) / fx,
            -(v - cy) / fy,
            -torch.ones_like(u),
        ],
        dim=-1,
    )  # (N, 3)
    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]   # (3,)
    rays_d_un = dir_cam @ R.T  # (N, 3)
    xyz = t[None, :] + rays_d_un * depth_z[:, None]
    return xyz


def _rotmat_to_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (N, 4) wxyz unit quaternions.

    Uses the standard four-cases-of-largest-component formulation for
    numerical stability when the trace is small.
    """

    m00 = R[..., 0, 0]; m01 = R[..., 0, 1]; m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]; m11 = R[..., 1, 1]; m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]; m21 = R[..., 2, 1]; m22 = R[..., 2, 2]

    trace = m00 + m11 + m22
    abs_w_sq = (1.0 + trace).clamp(min=0)
    abs_x_sq = (1.0 + m00 - m11 - m22).clamp(min=0)
    abs_y_sq = (1.0 - m00 + m11 - m22).clamp(min=0)
    abs_z_sq = (1.0 - m00 - m11 + m22).clamp(min=0)

    q_abs = 0.5 * torch.sqrt(
        torch.stack([abs_w_sq, abs_x_sq, abs_y_sq, abs_z_sq], dim=-1)
    )

    # Pick the branch with the largest absolute component for numerical stability.
    largest = q_abs.argmax(dim=-1)
    q = torch.zeros_like(q_abs)

    # Branch 0 (w largest)
    mask = largest == 0
    s = (4.0 * q_abs[..., 0].clamp(min=1e-8))
    q[mask, 0] = q_abs[mask, 0]
    q[mask, 1] = (m21[mask] - m12[mask]) / s[mask]
    q[mask, 2] = (m02[mask] - m20[mask]) / s[mask]
    q[mask, 3] = (m10[mask] - m01[mask]) / s[mask]

    # Branch 1 (x largest)
    mask = largest == 1
    s = (4.0 * q_abs[..., 1].clamp(min=1e-8))
    q[mask, 0] = (m21[mask] - m12[mask]) / s[mask]
    q[mask, 1] = q_abs[mask, 1]
    q[mask, 2] = (m01[mask] + m10[mask]) / s[mask]
    q[mask, 3] = (m02[mask] + m20[mask]) / s[mask]

    # Branch 2 (y largest)
    mask = largest == 2
    s = (4.0 * q_abs[..., 2].clamp(min=1e-8))
    q[mask, 0] = (m02[mask] - m20[mask]) / s[mask]
    q[mask, 1] = (m01[mask] + m10[mask]) / s[mask]
    q[mask, 2] = q_abs[mask, 2]
    q[mask, 3] = (m12[mask] + m21[mask]) / s[mask]

    # Branch 3 (z largest)
    mask = largest == 3
    s = (4.0 * q_abs[..., 3].clamp(min=1e-8))
    q[mask, 0] = (m10[mask] - m01[mask]) / s[mask]
    q[mask, 1] = (m02[mask] + m20[mask]) / s[mask]
    q[mask, 2] = (m12[mask] + m21[mask]) / s[mask]
    q[mask, 3] = q_abs[mask, 3]

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return q


def _normals_from_xyz_grid(xyz_grid: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normal via central differences on a per-pixel XYZ grid.

    ``xyz_grid``: (H, W, 3). Returns (H, W, 3) unit normals (zeros where
    finite difference is undefined at the boundary).
    """

    H, W, _ = xyz_grid.shape
    dxdu = torch.zeros_like(xyz_grid)
    dxdv = torch.zeros_like(xyz_grid)
    dxdu[:, 1:-1, :] = (xyz_grid[:, 2:, :] - xyz_grid[:, :-2, :]) * 0.5
    dxdv[1:-1, :, :] = (xyz_grid[2:, :, :] - xyz_grid[:-2, :, :]) * 0.5
    normals = torch.cross(dxdu, dxdv, dim=-1)
    norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / norm
    # Orient towards camera (assumes camera at +Z relative to surface won't
    # matter at insert time; we just pick a consistent sign so the third
    # scale axis isn't flipped randomly):
    # Use the average +z direction as a hint.
    flip = (normals[..., 2:3].mean() < 0).float()
    normals = normals * (1.0 - 2.0 * flip)
    return normals


def _rotation_from_normal(normals: torch.Tensor) -> torch.Tensor:
    """Build per-pixel rotation matrices whose third (z) column is the normal.

    Picks tangent axes via Gram-Schmidt against a world up hint. Returns
    (N, 3, 3) orthonormal matrices [tangent_u, tangent_v, normal].
    """

    n = normals  # (N, 3) unit
    # Choose an "up" hint that's not collinear with the normal.
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    parallel = (n @ z_axis).abs() > 0.95  # (N,)
    up = torch.where(parallel[:, None], y_axis[None, :], z_axis[None, :])
    tu = torch.cross(up, n, dim=-1)
    tu = tu / tu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tv = torch.cross(n, tu, dim=-1)
    tv = tv / tv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    R = torch.stack([tu, tv, n], dim=-1)  # columns: tu, tv, n → (N, 3, 3)
    return R


def decode_component_to_gaussians(
    target_camera,
    live_rgb: torch.Tensor,
    gt_depth_m: torch.Tensor,
    component_mask: torch.Tensor,
    *,
    opacity: float = 0.99,
    normal_smoothing_radius: int = 3,
    min_valid_fraction: float = 0.95,
    thin_axis_ratio: float = 0.25,
    scale_multiplier: float = 5.0,
    cliff_threshold_m: float = 0.05,
    post_cliff_erode_px: int = 1,
    rendered_depth_m: Optional[torch.Tensor] = None,
    leak_threshold_m: float = 0.02,
) -> Optional[dict]:
    """Decode the pixels of one component mask into per-pixel frozen Gaussians.

    Inputs are all on the same device. Returns a dict with the same tensor
    layout that ``DynamicGSModel.insert_inpaint_gaussians`` accepts, or
    ``None`` if the component is skipped (valid_fraction < threshold or
    no pixels at all).

    Returned dict keys: ``xyz`` (N,3), ``features_dc`` (N,3),
    ``features_rest`` (N, dim_sh-1, 3), ``opacities`` (N,1), ``scales`` (N,3),
    ``quats`` (N,4), ``diagnostics`` (dict).
    """

    if live_rgb.ndim != 3 or live_rgb.shape[-1] != 3:
        raise ValueError(f"live_rgb must be (H, W, 3); got {tuple(live_rgb.shape)}")
    H, W = live_rgb.shape[:2]
    device = live_rgb.device
    dtype = live_rgb.dtype if live_rgb.is_floating_point() else torch.float32

    if gt_depth_m.shape[-1] == 1:
        gt_depth_m = gt_depth_m[..., 0]
    if gt_depth_m.shape != (H, W):
        raise ValueError(
            f"gt_depth_m must be (H, W) or (H, W, 1); got {tuple(gt_depth_m.shape)}"
        )
    gt_depth_m = gt_depth_m.to(device=device, dtype=dtype)

    mask = component_mask
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != (H, W):
        raise ValueError(
            f"component_mask must be (H, W) or (H, W, 1); got {tuple(component_mask.shape)}"
        )
    mask_bool = (mask > 0.5).to(device)

    total_count = int(mask_bool.sum().item())
    if total_count == 0:
        return None

    valid_mask = mask_bool & (gt_depth_m > 0)
    valid_count = int(valid_mask.sum().item())
    valid_fraction = float(valid_count) / float(total_count)
    if valid_fraction < float(min_valid_fraction) or valid_count == 0:
        return {
            "skipped": True,
            "diagnostics": {
                "valid_fraction": valid_fraction,
                "total_pixels": total_count,
                "valid_pixels": valid_count,
            },
        }

    fx, fy, cx, cy, _, _ = _camera_intrinsics(target_camera)
    c2w = _camera_c2w(target_camera).to(device=device, dtype=dtype)

    # ---- Leak cut: drop pixels whose sensor depth is behind the rendered
    # scene depth by more than `leak_threshold_m`. The sensor depth at table
    # edges leaks into the floor behind (1-5 cm typical for RealSense/TOF),
    # which back-projects "table" pixels behind the actual table. The
    # rendered depth is the trusted "scene-as-it-currently-stands" surface;
    # a legitimate new occluder appears IN FRONT of it, never behind, so
    # we only drop pixels that are behind. 0 disables.
    leak_dropped = 0
    if (
        rendered_depth_m is not None
        and float(leak_threshold_m) > 0
    ):
        rd = rendered_depth_m
        if rd.ndim == 3 and rd.shape[-1] == 1:
            rd = rd[..., 0]
        rd = rd.to(device=device, dtype=dtype)
        if rd.shape != (H, W):
            raise ValueError(
                f"rendered_depth_m must be (H, W) or (H, W, 1); got {tuple(rendered_depth_m.shape)}"
            )
        # Compare ONLY where rendered depth is itself valid (rd > 0). Where
        # rendered depth is 0 (e.g. background gaps in the splat scene), we
        # have no reference, so we keep the sensor pixel.
        rd_valid = rd > 0
        behind = (gt_depth_m > rd + float(leak_threshold_m)) & rd_valid
        new_valid = valid_mask & ~behind
        leak_dropped = int(valid_mask.sum().item() - new_valid.sum().item())
        valid_mask = new_valid
        valid_count = int(valid_mask.sum().item())
        valid_fraction = float(valid_count) / float(total_count) if total_count > 0 else 0.0
        if valid_count == 0:
            return {
                "skipped": True,
                "diagnostics": {
                    "valid_fraction": valid_fraction,
                    "total_pixels": total_count,
                    "valid_pixels": valid_count,
                    "leak_dropped": leak_dropped,
                },
            }

    # ---- Cliff cut: drop pixels adjacent to a depth discontinuity, then
    # re-segment so each remaining sub-component sits on one surface ----
    # Without this, finite-difference normals at the cliff cross a 10-100 cm
    # depth jump and the splat ends up oriented edge-on to the camera (visible
    # as streaks). The 3D position of the cliff-adjacent splat is also bad
    # because the smoothed depth used for normals leaks the cliff into the
    # neighbouring pixels' orientation.
    cliff_th = float(cliff_threshold_m)
    erode_px = int(post_cliff_erode_px)
    sub_component_masks: list[torch.Tensor] = []
    cliff_dropped = 0
    if cliff_th > 0 and _HAS_SCIPY:
        valid_np = valid_mask.detach().cpu().numpy().astype(bool)
        depth_np = gt_depth_m.detach().cpu().numpy()
        # |∂depth/∂u| and |∂depth/∂v| via 1-px shifts; depth=0 (invalid) is
        # treated as cliff to keep dropouts out of finite-diff windows.
        du = np.zeros_like(depth_np)
        dv = np.zeros_like(depth_np)
        du[:, 1:] = np.abs(depth_np[:, 1:] - depth_np[:, :-1])
        dv[1:, :] = np.abs(depth_np[1:, :] - depth_np[:-1, :])
        # Invalid-neighbour pixels: any neighbour ≤0 → mark as cliff.
        invalid_left = np.zeros_like(valid_np); invalid_left[:, 1:] = ~valid_np[:, :-1]
        invalid_up = np.zeros_like(valid_np); invalid_up[1:, :] = ~valid_np[:-1, :]
        cliff = (du > cliff_th) | (dv > cliff_th) | invalid_left | invalid_up
        # Mirror onto the OTHER neighbour so the cliff "thickness" is 2 px (the
        # pixel before AND after the jump are dropped — both have wrong normals).
        cliff[:, :-1] |= cliff[:, 1:]
        cliff[:-1, :] |= cliff[1:, :]
        clean_np = valid_np & ~cliff
        cliff_dropped = int(valid_np.sum() - clean_np.sum())
        if erode_px > 0 and clean_np.any():
            structure = np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=bool)
            clean_np = _scipy_binary_erosion(clean_np, structure=structure)
        if not clean_np.any():
            return {
                "skipped": True,
                "diagnostics": {
                    "valid_fraction": valid_fraction,
                    "total_pixels": total_count,
                    "valid_pixels": valid_count,
                    "cliff_dropped": cliff_dropped,
                    "after_cliff_pixels": 0,
                },
            }
        labels_np, n_sub = _scipy_label(clean_np)
        for k in range(1, n_sub + 1):
            sub_np = labels_np == k
            if int(sub_np.sum()) < 8:
                continue  # too few pixels to define a meaningful surface
            sub_component_masks.append(
                torch.from_numpy(sub_np).to(device=device)
            )
        if not sub_component_masks:
            return {
                "skipped": True,
                "diagnostics": {
                    "valid_fraction": valid_fraction,
                    "total_pixels": total_count,
                    "valid_pixels": valid_count,
                    "cliff_dropped": cliff_dropped,
                    "n_sub_components": 0,
                },
            }
    else:
        sub_component_masks = [valid_mask]

    # ---- Surface normals from bilateral-smoothed depth on the whole frame ----
    # Smoothing is done once; the smoothing kernel itself won't bridge cliffs
    # because of σ_depth=2 cm, AND the cliff pixels were already dropped above.
    depth_smoothed = _bilateral_smooth_depth(
        gt_depth_m, valid_mask=(gt_depth_m > 0),
        radius=int(normal_smoothing_radius),
    )
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dir_grid = torch.stack(
        [(us - cx) / fx, -(vs - cy) / fy, -torch.ones_like(us)], dim=-1
    )  # (H, W, 3)
    rays_d_un_grid = dir_grid @ R.T  # (H, W, 3)
    xyz_smooth_grid = t[None, None, :] + rays_d_un_grid * depth_smoothed[..., None]
    normals_grid = _normals_from_xyz_grid(xyz_smooth_grid)  # (H, W, 3)

    # ---- Per-sub-component decode + merge ----
    xyz_list, dc_list, rest_list, op_list, sc_list, q_list = [], [], [], [], [], []
    sub_diag = []
    for sub_mask in sub_component_masks:
        coords = torch.nonzero(sub_mask, as_tuple=False)
        if coords.shape[0] == 0:
            continue
        v_idx = coords[:, 0]
        u_idx = coords[:, 1]
        d_pix = gt_depth_m[v_idx, u_idx]
        xyz_pix = _backproject_world(u_idx, v_idx, d_pix, fx, fy, cx, cy, c2w)

        rgb_pix = live_rgb[v_idx, u_idx].to(dtype).clamp(0.0, 1.0)
        features_dc = RGB2SH(rgb_pix)
        sh_degree_max_coeffs = 15  # matches DynamicGSModelConfig.sh_degree=3
        features_rest = torch.zeros(
            (xyz_pix.shape[0], sh_degree_max_coeffs, 3), device=device, dtype=dtype
        )

        scale_mul = float(scale_multiplier)
        scale_u = d_pix * (scale_mul / float(fx))
        scale_v = d_pix * (scale_mul / float(fy))
        scale_n = float(thin_axis_ratio) * torch.minimum(scale_u, scale_v)
        scales_lin = torch.stack([scale_u, scale_v, scale_n], dim=-1).clamp(min=1e-6)
        scales = torch.log(scales_lin)

        normals_pix = normals_grid[v_idx, u_idx]
        bad = normals_pix.norm(dim=-1) < 1e-6
        if bad.any():
            normals_pix = normals_pix.clone()
            normals_pix[bad] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        R_pix = _rotation_from_normal(normals_pix)
        quats = _rotmat_to_wxyz(R_pix)

        opacity_t = torch.full(
            (xyz_pix.shape[0], 1), float(opacity),
            device=device, dtype=dtype,
        ).clamp(1e-4, 1 - 1e-4)
        opacities = torch.logit(opacity_t)

        xyz_list.append(xyz_pix)
        dc_list.append(features_dc)
        rest_list.append(features_rest)
        op_list.append(opacities)
        sc_list.append(scales)
        q_list.append(quats)
        sub_diag.append({
            "pixels": int(xyz_pix.shape[0]),
            "depth_min_m": float(d_pix.min().item()),
            "depth_max_m": float(d_pix.max().item()),
        })

    if not xyz_list:
        return {
            "skipped": True,
            "diagnostics": {
                "valid_fraction": valid_fraction,
                "total_pixels": total_count,
                "valid_pixels": valid_count,
                "cliff_dropped": cliff_dropped,
                "n_sub_components": 0,
            },
        }

    return {
        "skipped": False,
        "xyz": torch.cat(xyz_list, dim=0),
        "features_dc": torch.cat(dc_list, dim=0),
        "features_rest": torch.cat(rest_list, dim=0),
        "opacities": torch.cat(op_list, dim=0),
        "scales": torch.cat(sc_list, dim=0),
        "quats": torch.cat(q_list, dim=0),
        "diagnostics": {
            "valid_fraction": valid_fraction,
            "total_pixels": total_count,
            "valid_pixels": valid_count,
            "leak_dropped": leak_dropped,
            "cliff_dropped": cliff_dropped,
            "n_sub_components": len(sub_diag),
            "sub": sub_diag,
        },
    }
