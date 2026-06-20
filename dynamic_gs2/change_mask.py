"""change_mask.py — render-vs-live change detection (CDN). Self-contained; NO dynamic_gs import.

The CDN finds where the live camera disagrees with the rendered scene, so feedforward knows
where to decode new geometry. Clean rewrite of the old dynamic_gs CDN (single-scale, RGB-only).

Pipeline (compute_change_mask):
  1. Build a VALID mask (drop tracked object + gripper + uncovered-void pixels). Depth is used
     ONLY here as a validity gate — a low-coverage pixel is kept as a fillable HOLE iff the live
     sensor sees a real near surface there; the genuine void (no live return) stays dropped.
  2. Masked-avg-pool BOTH images down to ~downsample_target_side (the speed + jitter-ignore lever).
     Excluded pixels contribute 0 to num+den so a block's colour is the clean mean of valid pixels.
  3. SINGLE-SCALE SSIM (one Gaussian-windowed pass) on the blurred grayscale -> per-pixel
     dissimilarity = 1 - SSIM. (No multi-scale pyramid: the avg-pool already coarsens; the pyramid
     was redundant and is removed.)
  4. Threshold (rgb_threshold) -> binary; component cleanup (close/open/min-area, keep-all or
     largest); upsample back to native; final AND with the valid mask.
Returns a (H,W,1) {0,1} float mask. Empty (all-zero) means "no change" — downstream skips FF.

Why mask AFTER SSIM, never black-out before: SSIM is a sliding window; blacking pixels creates a
fake edge that fires as change at the mask border, and it saves no compute (cost is grid-size).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as TF
from scipy.ndimage import label as _scipy_label


# ----------------------------------------------------------------- small mask ops
def _to_hw1(m: torch.Tensor) -> torch.Tensor:
    return (m if m.ndim == 3 else m[..., None]).float()


def _resize_to(m: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Nearest-resize a (H,W,C) tensor to (h,w,C). No-op if already that size."""
    if m.shape[0] == h and m.shape[1] == w:
        return m
    return TF.interpolate(m.permute(2, 0, 1)[None], size=(h, w), mode="nearest")[0].permute(1, 2, 0)


def _dilate(binary_hw1: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return binary_hw1
    x = (binary_hw1[..., 0] > 0.5).float()[None, None]
    x = TF.max_pool2d(x, 2 * r + 1, stride=1, padding=r)
    return x[0, 0, ..., None]


def _erode(binary_hw1: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return binary_hw1
    x = (binary_hw1[..., 0] > 0.5).float()[None, None]
    x = 1.0 - TF.max_pool2d(1.0 - x, 2 * r + 1, stride=1, padding=r)
    return x[0, 0, ..., None]


def _close(m, r):   # fill tiny holes
    return _erode(_dilate(m, r), r)


def _open(m, r):    # remove speckle
    return _dilate(_erode(m, r), r)


def _keep_components(binary_hw1: torch.Tensor, min_area: int, largest_only: bool) -> torch.Tensor:
    """scipy connected-components: keep all >= min_area, or just the largest if it clears min_area."""
    binary = binary_hw1[..., 0] > 0.5
    if not torch.any(binary):
        return torch.zeros_like(binary_hw1)
    labels, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return torch.zeros_like(binary_hw1)
    areas = np.bincount(labels.ravel(), minlength=num + 1)
    areas[0] = 0
    if largest_only:
        best = int(np.argmax(areas))
        keep = labels == best if int(areas[best]) >= int(min_area) else np.zeros_like(labels, bool)
    else:
        keep_lbls = np.where(areas >= int(min_area))[0]
        keep = np.isin(labels, keep_lbls) if keep_lbls.size else np.zeros_like(labels, bool)
    return torch.from_numpy(keep).to(binary_hw1.device).float()[..., None]


# ----------------------------------------------------------------- SSIM (single scale)
_SSIM_KERNEL_CACHE: dict = {}


def _ssim_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    key = (size, round(float(sigma), 4), str(device), str(dtype))
    k = _SSIM_KERNEL_CACHE.get(key)
    if k is None:
        c = torch.arange(size, device=device, dtype=dtype) - size // 2
        g = torch.exp(-(c * c) / (2 * sigma * sigma))
        g = g / g.sum().clamp_min(1e-8)
        k = torch.outer(g, g).view(1, 1, size, size)
        _SSIM_KERNEL_CACHE[key] = k
    return k


def _ssim_dissim(pred_gray: torch.Tensor, gt_gray: torch.Tensor, window: int) -> torch.Tensor:
    """Per-pixel 1 - SSIM on two (H,W) grayscale tensors (windowed local stats)."""
    ker = _ssim_kernel(window, 1.5, pred_gray.device, pred_gray.dtype)
    pad = window // 2
    x, y = pred_gray[None, None], gt_gray[None, None]
    mu_x = TF.conv2d(x, ker, padding=pad)
    mu_y = TF.conv2d(y, ker, padding=pad)
    sx = TF.conv2d(x * x, ker, padding=pad) - mu_x * mu_x
    sy = TF.conv2d(y * y, ker, padding=pad) - mu_y * mu_y
    sxy = TF.conv2d(x * y, ker, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
           (((mu_x * mu_x + mu_y * mu_y + c1) * (sx + sy + c2)).clamp_min(1e-6))
    return (1.0 - ssim[0, 0].clamp(0.0, 1.0))


def _masked_blur(img: torch.Tensor, size: int, sigma: float, valid_hw: Optional[torch.Tensor]) -> torch.Tensor:
    """Gaussian blur a (H,W,3) image; if valid_hw given, normalize by valid weight (no bleed from
    excluded pixels). sigma<=0 or size<=1 disables."""
    if sigma <= 0 or size <= 1:
        return img.float()
    if size % 2 == 0:
        size += 1
    x = img.float().permute(2, 0, 1)[None]
    ch = x.shape[1]
    if valid_hw is None:
        w = torch.ones((1, 1, img.shape[0], img.shape[1]), dtype=x.dtype, device=x.device)
    else:
        w = valid_hw.float()[None, None].to(x.device)
    ker1 = _ssim_kernel(size, sigma, x.device, x.dtype)
    x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
    num = TF.conv2d(x * w, ker1.expand(ch, 1, -1, -1), padding=size // 2, groups=ch)
    den = TF.conv2d(w, ker1, padding=size // 2)
    return (num / den.clamp_min(1e-6))[0].permute(1, 2, 0)


# ----------------------------------------------------------------- public API
def resolve_downsample_factor(rgb_or_shape, target_side: int) -> int:
    """ds = sqrt(H*W)/target_side (>=1), so SSIM runs on ~target_side^2 px at any resolution/aspect."""
    if hasattr(rgb_or_shape, "shape"):
        h, w = int(rgb_or_shape.shape[0]), int(rgb_or_shape.shape[1])
    else:
        h, w = int(rgb_or_shape[0]), int(rgb_or_shape[1])
    return max(1, int(math.sqrt(float(h) * float(w)) / float(max(1, target_side))))


def compute_change_mask(*, rendered_rgb: torch.Tensor, live_rgb: torch.Tensor,
                        rendered_alpha: Optional[torch.Tensor], gt_depth: Optional[torch.Tensor],
                        gripper_keep: Optional[torch.Tensor], object_mask: Optional[torch.Tensor],
                        cfg, keep_largest_only: bool = False,
                        debug_out: Optional[dict] = None) -> torch.Tensor:
    """Single-scale RGB CDN. All RGB (H,W,3) in [0,1]; depth/alpha (H,W) or (H,W,1).
    gripper_keep = KEEP mask (1=keep). object_mask = EXCLUDE mask (1=exclude). cfg = ChangeMaskConfig.
    Returns (H,W,1) {0,1} float on rendered_rgb's device. If debug_out (a dict) is given, stashes the
    per-pixel SSIM dissimilarity `score` (H,W native-res, [0,1]) under key 'score' for a heatmap dump."""
    dev = rendered_rgb.device
    H, W = rendered_rgb.shape[:2]

    # ---- 1. valid mask: exclude tracked object + gripper, gate uncovered void (depth = validity only)
    valid = None
    if object_mask is not None:
        valid = 1.0 - _resize_to(_to_hw1(object_mask).to(dev), H, W)
    if gripper_keep is not None:
        grip = _resize_to(_to_hw1(gripper_keep).to(dev), H, W)
        er = int(getattr(cfg, "gripper_erode_px", 0))
        if er > 0:
            grip = _erode(grip, er)
        valid = grip * valid if valid is not None else grip
    cov_thr = float(cfg.scene_coverage_threshold)
    if rendered_alpha is not None and cov_thr > 0.0:
        cov = _resize_to(_to_hw1(rendered_alpha).to(dev), H, W)
        keep = cov > cov_thr
        if gt_depth is not None:                              # uncovered pixel is a fillable HOLE iff
            d = _resize_to(_to_hw1(gt_depth).to(dev), H, W)   # the live sensor sees a real near surface
            keep = keep | ((d > float(cfg.live_depth_min_m)) & (d < float(cfg.live_depth_max_m)))
        keep = keep.float()
        valid = keep * valid if valid is not None else keep

    # ---- 2. masked-avg-pool downsample (clean per-block mean of valid pixels only)
    ds = resolve_downsample_factor(rendered_rgb, int(cfg.downsample_target_side))
    if ds > 1:
        valid_chw = (valid.permute(2, 0, 1)[None] if valid is not None
                     else torch.ones((1, 1, H, W), device=dev, dtype=rendered_rgb.dtype))

        def mavg(rgb):
            r = rgb.permute(2, 0, 1)[None]
            num = TF.avg_pool2d(r * valid_chw, ds, ds)
            den = TF.avg_pool2d(valid_chw, ds, ds).clamp_min(1e-8)
            return (num / den)[0].permute(1, 2, 0)

        pred_use, live_use = mavg(rendered_rgb), mavg(live_rgb)
        # block-validity: keep a pooled block when >= block_valid_min_frac of its source px were valid
        vfrac = TF.avg_pool2d(valid_chw, ds, ds)
        valid_use = (vfrac >= float(cfg.block_valid_min_frac)).float()[0].permute(1, 2, 0)
    else:
        pred_use, live_use, valid_use = rendered_rgb, live_rgb, valid

    # ---- 3. single-scale SSIM dissimilarity on blurred grayscale
    region = (valid_use[..., 0] > 0.5) if valid_use is not None else None
    pred_b = _masked_blur(pred_use, int(cfg.blur_kernel_size), float(cfg.blur_sigma), region)
    live_b = _masked_blur(live_use, int(cfg.blur_kernel_size), float(cfg.blur_sigma), region)
    pg = 0.2989 * pred_b[..., 0] + 0.5870 * pred_b[..., 1] + 0.1140 * pred_b[..., 2]
    lg = 0.2989 * live_b[..., 0] + 0.5870 * live_b[..., 1] + 0.1140 * live_b[..., 2]
    score = _ssim_dissim(pg, lg, int(getattr(cfg, "ssim_window", 11)))
    if region is not None:
        score = score * region.float()
    if debug_out is not None:                             # per-pixel 1-SSIM, upsampled to native for the heatmap
        s = score[..., None]
        debug_out["score"] = (_resize_to(s, H, W) if ds > 1 else s)[..., 0].detach()

    # ---- 4. threshold + cleanup (close/open/min-area, keep-all or largest) + AND-to-valid
    binary = (torch.isfinite(score) & (score > float(cfg.rgb_threshold))).float()[..., None]
    if valid_use is not None:
        binary = binary * valid_use
    cleaned = _close(binary, int(getattr(cfg, "morph_close_px", 2)))
    cleaned = _open(cleaned, int(getattr(cfg, "morph_open_px", 1)))
    cleaned = _keep_components(cleaned, int(cfg.min_component_area), largest_only=keep_largest_only)
    if valid_use is not None:
        cleaned = cleaned * valid_use

    # upsample back to native + final AND with the full-res valid mask
    if ds > 1:
        cleaned = TF.interpolate(cleaned.permute(2, 0, 1)[None], size=(H, W), mode="nearest")[0].permute(1, 2, 0)
    if valid is not None:
        cleaned = cleaned * valid
    return cleaned
