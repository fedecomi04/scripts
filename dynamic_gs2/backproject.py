"""backproject.py — pure back-projection / cull geometry helpers.

The two functions here are lifted VERBATIM from the frozen dynamic_gs.fusion.phase0
(``backproject_mask_to_world`` + ``cull_points_in_front``); they reference nothing
else in that module. Copied into dynamic_gs2 so the static Phase-0b path does not
import the old package. Nerfstudio/OpenGL camera frame (x right, y up, z back).
"""

from __future__ import annotations

import numpy as np
import torch  # noqa: F401  (kept for parity with the source; used in annotations)
from PIL import Image


def backproject_mask_to_world(
    mask_bool_np: np.ndarray,
    depth_image: torch.Tensor,
    rgb_image: torch.Tensor,
    camera,
    max_object_slope_deg: float = 70.0,
    near_surface_window_frac: float = 0.012,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an image-plane mask through a depth image into world 3D points.

    Args:
        mask_bool_np: (H_mask, W_mask) boolean array on CPU.
        depth_image: (H, W) depth in meters (CPU tensor).
        rgb_image: (H, W, 3) uint8 or float RGB (CPU or GPU tensor).
        camera: ``Cameras`` with at least one element (we use index 0).
        max_object_slope_deg: local near-surface filter — the steepest object
            surface (tilt from frontal) to PRESERVE. The drop tolerance is
            derived per pixel from this angle + the window + depth + intrinsics
            (no hardcoded distance); anything farther behind the local object
            surface than a ``max_object_slope_deg`` surface could account for is
            treated as table/see-through and dropped. >= 90 disables.
        near_surface_window_frac: filter window size as a fraction of the image
            short side (sets the spatial reach for finding the local surface).

    Returns:
        ``(points_np, colors_np)`` where ``points_np`` is ``(N, 3)`` float32
        in world coordinates and ``colors_np`` is ``(N, 3)`` float32 in [0, 1].
        Points with missing/zero depth are filtered out.
    """
    H, W = int(depth_image.shape[0]), int(depth_image.shape[1])

    if mask_bool_np.shape != (H, W):
        mask_resized = np.array(
            Image.fromarray(mask_bool_np.astype(np.uint8) * 255).resize((W, H), Image.NEAREST),
            dtype=np.uint8,
        ) > 127
    else:
        mask_resized = mask_bool_np

    depth_np = depth_image.detach().cpu().numpy().astype(np.float32)

    if hasattr(rgb_image, "detach"):
        rgb_cpu = rgb_image.detach().cpu()
    else:
        rgb_cpu = rgb_image
    rgb_np = rgb_cpu.numpy() if hasattr(rgb_cpu, "numpy") else np.asarray(rgb_cpu)
    if rgb_np.dtype == np.uint8:
        rgb_np = rgb_np.astype(np.float32) / 255.0
    else:
        rgb_np = rgb_np.astype(np.float32)
    if rgb_np.shape[:2] != (H, W):
        rgb_np = np.array(
            Image.fromarray((rgb_np * 255).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    ys, xs = np.where(mask_resized & (depth_np > 1e-4))
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    z = depth_np[ys, xs]

    # MAD-based depth outlier scrub. Mask-boundary pixels frequently hit
    # the background/table behind the object (silhouette-edge depth bleed);
    # without this scrub CPD's similarity fit is pulled toward a smaller
    # scale and a shifted centroid. 5.0 × 1.4826 ≈ 7.4 MAD is intentionally
    # permissive so legitimate object depth extent is preserved.
    if z.size >= 10:
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med))) + 1e-6
        keep = np.abs(z - med) < 5.0 * 1.4826 * mad
        if keep.sum() >= 3:
            ys = ys[keep]
            xs = xs[keep]
            z = z[keep]

    fx = float(camera.fx[0].item())
    fy = float(camera.fy[0].item())
    cx = float(camera.cx[0].item())
    cy = float(camera.cy[0].item())
    c2w = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float32)

    src_H = int(camera.height[0].item()) if hasattr(camera.height[0], "item") else int(camera.height[0])
    src_W = int(camera.width[0].item()) if hasattr(camera.width[0], "item") else int(camera.width[0])
    if (H, W) != (src_H, src_W):
        sx = W / float(src_W)
        sy = H / float(src_H)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    # Local near-surface filter (registration-target border cleanup). The mask
    # silhouette overshoots onto the table/background, and at the edge the depth
    # sensor sees PAST the object to far surfaces; both pollute the target and
    # bias the centroid/bbox the registration init keys on. Within a window the
    # object's own surface is the local MINIMUM depth, so a masked pixel sitting
    # FARTHER than the local min is a depth step (object→table edge) or see-
    # through. The keep tolerance is NOT a hardcoded distance: a genuine object
    # surface tilted up to ``max_object_slope_deg`` can only drop by
    # (half_window_px · z / fx · tan(slope)) across the half-window at this
    # pixel's depth, so anything beyond that is not the object. Derived per
    # pixel from window+depth+intrinsics (only the slope angle is a choice).
    # Safe for thin/smooth objects (interior == local min → kept) and leaves
    # benign coplanar leakage (a flat object on the table has no depth step).
    # Asymmetric complement to the MAD scrub above; >= 90° disables.
    if max_object_slope_deg < 90.0 and ys.size >= 3:
        from scipy.ndimage import minimum_filter

        win = max(7, (int(round(near_surface_window_frac * min(H, W))) | 1))
        d_masked = np.full((H, W), np.inf, dtype=np.float32)
        d_masked[ys, xs] = z
        local_min = minimum_filter(d_masked, size=win, mode="nearest")
        allow = (win // 2) * (z / fx) * float(np.tan(np.radians(max_object_slope_deg)))
        near_keep = (z - local_min[ys, xs]) <= allow
        if near_keep.sum() >= 3:
            ys = ys[near_keep]
            xs = xs[near_keep]
            z = z[near_keep]

    # Back-project in Nerfstudio/OpenGL camera frame (x right, y up, z back).
    x_cam = (xs.astype(np.float32) - cx) / fx * z
    y_cam = -(ys.astype(np.float32) - cy) / fy * z
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = pts_cam @ R.T + t[None, :]
    colors = rgb_np[ys, xs]
    return pts_world.astype(np.float32), colors.astype(np.float32)


def cull_points_in_front(
    points_world: np.ndarray,
    target_points_world: np.ndarray,
    camera,
    render_hw: tuple[int, int],
    band_m: float = 0.0,
    radius_px: int = 2,
) -> np.ndarray:
    """Boolean keep-mask: drop ``points_world`` that lie in FRONT of the trusted
    real surface from the camera viewpoint (between the camera and the surface).

    Builds a front-surface depth buffer by projecting ``target_points_world``
    (the back-projected real/GT depth) into the image, then removes any inserted
    point whose forward-depth is closer than that surface by more than ``band_m``.
    Points with no surface along their ray (outside the silhouette) are kept.
    Inverse of :func:`backproject_mask_to_world` (Nerfstudio/OpenGL camera frame).
    Mirrors the tuned cull in scripts/experiments/nonrigid_bench/.
    """
    from scipy.ndimage import minimum_filter

    H, W = int(render_hw[0]), int(render_hw[1])
    fx = float(camera.fx[0].item())
    fy = float(camera.fy[0].item())
    cx = float(camera.cx[0].item())
    cy = float(camera.cy[0].item())
    c2w = camera.camera_to_worlds[0].detach().cpu().numpy().astype(np.float64)
    src_H = int(camera.height[0].item()) if hasattr(camera.height[0], "item") else int(camera.height[0])
    src_W = int(camera.width[0].item()) if hasattr(camera.width[0], "item") else int(camera.width[0])
    if (H, W) != (src_H, src_W):
        fx *= W / float(src_W)
        fy *= H / float(src_H)
        cx *= W / float(src_W)
        cy *= H / float(src_H)
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    def _project(pts: np.ndarray):
        cam = (pts.astype(np.float64) - t) @ R
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx + fx * cam[:, 0] / z
            v = cy - fy * cam[:, 1] / z
        return u, v, z

    tu, tv, tz = _project(target_points_world)
    tui, tvi = np.round(tu).astype(int), np.round(tv).astype(int)
    tvalid = np.isfinite(tz) & (tz > 0) & (tui >= 0) & (tui < W) & (tvi >= 0) & (tvi < H)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    np.minimum.at(depth_buf, (tvi[tvalid], tui[tvalid]), tz[tvalid])
    if radius_px > 0:
        depth_buf = minimum_filter(depth_buf, size=2 * int(radius_px) + 1)

    u, v, z = _project(points_world)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    in_img = np.isfinite(z) & (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    d_surf = np.full(points_world.shape[0], np.inf, dtype=np.float64)
    d_surf[in_img] = depth_buf[vi[in_img], ui[in_img]]
    has_surface = np.isfinite(d_surf)
    in_front = has_surface & (z < d_surf - float(band_m))
    return ~in_front
