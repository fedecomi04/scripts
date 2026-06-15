"""Change-mask computation (MS-SSIM render-vs-live with gripper/object exclusion).

Pure function — no pipeline state, no model state. Callers pass in everything
needed: the two rendered/live RGB tensors, optional depth tensors, exclusion
masks, and the cleanup thresholds.

The function is on-demand only: callers fire it when they need a CDN (e.g. the
feedforward dispatcher before each FF call, or the static-convergence check).
The dynamic tick loop does NOT call it every tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as TF

from ..utils.active_mask import build_change_mask, dilate_binary_mask


@dataclass
class ChangeMaskConfig:
    """Thresholds + cleanup knobs for ``compute_change_mask``.

    Mirrors the ``change_mask_*`` fields on ``DynamicGSModelConfig`` so the
    function does not depend on the model config dataclass directly.
    """

    depth_threshold: float = 0.02
    rgb_threshold: float = 0.07
    use_rgb: bool = False
    mode: str = "rgb"
    scene_coverage_threshold: float = 0.5
    outlier_median_multiplier: float = 10.0
    """For ``mode='depth_outlier'``: error must exceed this × median to fire."""
    outlier_min_threshold_m: float = 0.01
    """For ``mode='depth_outlier'``: absolute error floor (m)."""
    """Minimum rendered cumulative alpha for a pixel to be 'real scene'. Below
    this, the rasterizer has no Gaussians covering the pixel and the rendered
    depth falls back to ``depth_im.detach().max()`` (model.get_outputs line
    2202), which makes CDN compare junk against the live sensor and fire on
    every uncovered pixel — this manifests as a huge 'change' band above the
    object on viewpoints where the camera looks beyond the warm-cache scene.
    Pixels with ``accumulation < threshold`` are treated as 'don't know' and
    AND'd out of valid_mask, so CDN never flags them — UNLESS the live sensor
    sees a real near surface there (``live_depth_min_m..live_depth_max_m``), in
    which case the pixel is a fillable HOLE and is kept. ``0.0`` disables this
    gating."""
    """CDN comparison mode for ``build_change_mask``: 'rgb' (MS-SSIM luminance)
    or 'depth' (per-pixel |pred-gt| in metres). Mirrors
    ``DynamicGSModelConfig.change_mask_mode``."""
    live_depth_min_m: float = 0.05
    live_depth_max_m: float = 3.0
    """Live-sensor valid-depth band (m). An UNCOVERED pixel (rendered alpha below
    ``scene_coverage_threshold`` → no Gaussians → renders background) is kept as a
    fillable HOLE when the live depth falls in this band (a real near surface the
    static scene is missing). Outside the band (sky / no return → 0, or beyond
    range) it stays the genuine void the coverage gate was built to drop. Matches
    ``DEPTH_MIN_M``/``DEPTH_MAX_M`` in ``utils/online_fusion.py``."""
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    filter_radius: int = 1
    min_component_size: int = 64
    dilate_radius: int = 0
    gripper_erode_px: int = 0
    """Erode the gripper KEEP mask by this many px (= grow the gripper
    exclusion) so the leak ring of gripper-coloured pixels just outside the
    silhouette is dropped from CDN. 0 = off."""
    block_valid_min_frac: float = 0.5
    """Downsample block-validity threshold. A pooled MS-SSIM block is kept when at
    least this FRACTION of its source pixels were valid (not gripper/object). The
    block colour is already the masked mean of only the valid pixels, so a
    mostly-valid block is uncontaminated. Replaces the old strict rule (drop the
    whole block if ANY source pixel was excluded), which carved a ~downsample-px
    dead halo around the object/gripper and hid change right next to the tracked
    object. ``1.0`` restores the strict behaviour; lower keeps more boundary blocks."""


def _resize_mask_to(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a (H,W,C) mask to (target_h, target_w, C) via nearest interpolation."""
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    return TF.interpolate(
        mask.permute(2, 0, 1).unsqueeze(0),
        size=(target_h, target_w),
        mode="nearest",
    ).squeeze(0).permute(1, 2, 0)


def resolve_downsample_factor(
    rgb_or_shape,
    configured_factor: int,
    target_side: int,
) -> int:
    """Resolve the MSSIM downsample factor.

    When ``configured_factor`` is 0 (auto), scale with ``sqrt(H*W) / target_side``
    so MSSIM always runs on ~``target_side * target_side`` pixels regardless of
    native resolution or aspect ratio. Otherwise return ``max(1, configured_factor)``.
    """
    if configured_factor != 0:
        return max(1, int(configured_factor))
    if hasattr(rgb_or_shape, "shape"):
        H, W = int(rgb_or_shape.shape[0]), int(rgb_or_shape.shape[1])
    else:
        H, W = int(rgb_or_shape[0]), int(rgb_or_shape[1])
    ds = int(math.sqrt(float(H) * float(W)) / float(max(1, target_side)))
    return max(1, ds)


def compute_change_mask(
    *,
    rendered_rgb: torch.Tensor,
    rendered_depth: Optional[torch.Tensor],
    live_rgb: torch.Tensor,
    gt_depth: Optional[torch.Tensor],
    gripper_mask: Optional[torch.Tensor],
    object_mask: Optional[torch.Tensor],
    config: ChangeMaskConfig,
    downsample_factor: int = 1,
    keep_largest_only: bool = True,
    rendered_alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render-vs-live change mask, excluding gripper + object regions.

    All RGB tensors are (H, W, 3) in [0, 1]. Depth tensors are (H, W) or (H, W, 1)
    in metres. ``gripper_mask`` is a KEEP mask (1 = keep, 0 = gripper / drop).
    ``object_mask`` is an EXCLUDE mask (1 = exclude). Either may be None.

    ``downsample_factor`` > 1 masked-avg-pools the RGB+depth before MS-SSIM, then
    nearest-upsamples the result back to native resolution. Use to ignore small
    details (specular shimmer, sub-pixel tracker jitter).

    ``keep_largest_only=False`` keeps every connected component above the
    min-area threshold (multi-blob output for the feedforward decode path).
    Returns a (H, W, 1) float mask in {0, 1} on the same device as ``rendered_rgb``.
    """
    target_h, target_w = rendered_rgb.shape[:2]
    device = rendered_rgb.device

    valid_mask = None
    if object_mask is not None:
        obj = object_mask.float()
        if obj.ndim == 2:
            obj = obj[..., None]
        obj = _resize_mask_to(obj.to(device), target_h, target_w)
        valid_mask = 1.0 - obj
    if gripper_mask is not None:
        grip = gripper_mask.float().to(device)
        if grip.ndim == 2:
            grip = grip[..., None]
        grip = _resize_mask_to(grip, target_h, target_w)
        # Grow the gripper EXCLUSION by ``gripper_erode_px`` (= erode the KEEP
        # mask): pixels just outside the gripper silhouette still carry gripper
        # colour (anti-aliasing, an imperfect/under-tight mask, motion blur) and
        # would otherwise read as 'change'. Eroding the keep-mask drops that
        # leak ring so it can't fire. Works in BOTH the full-res and the
        # downsampled paths (the latter then also drops any pooled block that
        # touches the eroded gripper via strict block-validity below).
        erode_px = int(getattr(config, "gripper_erode_px", 0))
        if erode_px > 0:
            from ..utils.active_mask import erode_binary_mask
            grip = erode_binary_mask(grip, erode_px)
        valid_mask = grip * valid_mask if valid_mask is not None else grip

    # Scene-coverage gate: pixels with rendered alpha below threshold are
    # "uncovered" (the rasterizer fell back to depth_im.max() at line 2202 of
    # dynamic_gs_model.py), so the depth comparison is junk there. AND them out
    # of valid_mask so CDN never flags them.
    cov_thr = float(getattr(config, "scene_coverage_threshold", 0.0))
    if rendered_alpha is not None and cov_thr > 0.0:
        cov = rendered_alpha.float().to(device)
        if cov.ndim == 2:
            cov = cov[..., None]
        cov = _resize_mask_to(cov, target_h, target_w)
        coverage_keep = cov > cov_thr
        # An uncovered pixel (no Gaussians → rendered = background) is a FILLABLE
        # HOLE — not junk — when the live sensor sees a real near surface there
        # (e.g. table revealed once the grasped object lifts off it; the static
        # scene never saw under the object). Keep those so the CDN flags them for
        # feedforward. Only the genuine void (camera looking past the scene → live
        # ALSO has no depth: sky / 0 / beyond range) stays dropped — which is the
        # spurious-band case the coverage gate was built for.
        if gt_depth is not None:
            d = gt_depth.float().to(device)
            if d.ndim == 2:
                d = d[..., None]
            d = _resize_mask_to(d, target_h, target_w)
            live_valid = (d > float(config.live_depth_min_m)) & (d < float(config.live_depth_max_m))
            coverage_keep = coverage_keep | live_valid
        coverage_keep = coverage_keep.float()
        valid_mask = coverage_keep * valid_mask if valid_mask is not None else coverage_keep

    # Optional masked-avg-pool downsample of inputs. Invalid pixels (gripper,
    # object) contribute 0 to both numerator and denominator so the downsampled
    # block colour is the clean mean of valid pixels only — preventing
    # gripper texture from bleeding into neighbouring blocks via bilinear
    # interpolation and flagging that as a false change.
    ds = max(1, int(downsample_factor))
    if ds > 1:
        def _avg_pool(t):
            return TF.avg_pool2d(t, kernel_size=ds, stride=ds, ceil_mode=False)

        if valid_mask is not None:
            valid_chw = valid_mask.permute(2, 0, 1).unsqueeze(0)
        else:
            valid_chw = torch.ones(
                (1, 1, target_h, target_w), device=device, dtype=rendered_rgb.dtype
            )

        def _masked_avg_rgb(rgb):
            rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
            num = _avg_pool(rgb_chw * valid_chw)
            den = _avg_pool(valid_chw).clamp(min=1e-8)
            return (num / den).squeeze(0).permute(1, 2, 0)

        def _masked_depth(d):
            if d is None:
                return None
            d_in = d if d.ndim == 3 else d[..., None]
            d_chw = d_in.permute(2, 0, 1).unsqueeze(0)
            num = _avg_pool(d_chw * valid_chw)
            den = _avg_pool(valid_chw).clamp(min=1e-8)
            return (num / den).squeeze(0).permute(1, 2, 0)

        rendered_rgb_use = _masked_avg_rgb(rendered_rgb)
        live_rgb_use = _masked_avg_rgb(live_rgb)
        rendered_depth_use = _masked_depth(rendered_depth)
        gt_depth_use = _masked_depth(gt_depth)
        # Block validity: keep a pooled block when at least ``block_valid_min_frac``
        # of its source pixels were valid — NOT the old strict "drop if ANY source
        # pixel is excluded" (= max_pool), which discarded a full block, a
        # ~downsample-px halo around the object/gripper, whenever a single pixel
        # touched the mask and hid change right beside the tracked object. The
        # block colour above is the masked mean of the VALID pixels only, so a
        # mostly-valid block is already uncontaminated; only mostly-excluded blocks
        # drop. (``block_valid_min_frac=1.0`` reproduces the old strict rule.)
        valid_frac = _avg_pool(valid_chw)
        thr = float(getattr(config, "block_valid_min_frac", 0.5))
        valid_block = (valid_frac >= thr).float()
        valid_mask_use = valid_block.squeeze(0).permute(1, 2, 0)
    else:
        rendered_rgb_use = rendered_rgb
        live_rgb_use = live_rgb
        rendered_depth_use = rendered_depth
        gt_depth_use = gt_depth
        valid_mask_use = valid_mask

    change_mask = build_change_mask(
        rendered_depth_use,
        gt_depth_use,
        pred_rgb=rendered_rgb_use,
        gt_rgb=live_rgb_use,
        valid_mask=valid_mask_use,
        depth_threshold=config.depth_threshold,
        rgb_threshold=config.rgb_threshold,
        use_rgb=config.use_rgb,
        blur_kernel_size=config.blur_kernel_size,
        blur_sigma=config.blur_sigma,
        filter_radius=config.filter_radius,
        min_component_size=config.min_component_size,
        keep_largest_only=keep_largest_only,
        mode=config.mode,
        outlier_median_multiplier=config.outlier_median_multiplier,
        outlier_min_threshold_m=config.outlier_min_threshold_m,
    )

    if ds > 1:
        m = change_mask
        if m.ndim == 2:
            m = m[..., None]
        change_mask = TF.interpolate(
            m.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0).permute(1, 2, 0)

    if config.dilate_radius > 0:
        change_mask = dilate_binary_mask(change_mask, config.dilate_radius)

    # Re-clip dilated result to valid_mask so the closing operation cannot bleed
    # back into the excluded gripper/object regions.
    if valid_mask is not None:
        change_mask = change_mask * valid_mask

    return change_mask
