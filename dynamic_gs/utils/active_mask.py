from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as _scipy_label

# Fallback per-pixel MS-SSIM dissimilarity threshold for `build_change_mask`.
# Used when the caller does not pass `rgb_threshold` explicitly.
# 0.07 matches the validated runtime config (DynamicGSModelConfig.
# change_mask_rgb_threshold) used in the 2026-06-12 sharpness-mismatch A/B —
# see `_rgb_msssim_score`'s docstring for the measured numbers.
OFFICIAL_RGB_MSSSIM_THRESHOLD = 0.07
OFFICIAL_FILTER_CLOSE_RADIUS = 10
OFFICIAL_FILTER_OPEN_RADIUS = 3
OFFICIAL_FILTER_MIN_AREA = 760

# Default per-pixel absolute depth-difference threshold (metres) for the
# depth-only CDN mode of `build_change_mask`. 2 cm matches the sensor depth
# noise floor at ~1 m range — pixels exceeding this are genuinely different
# geometry, not just noise.
OFFICIAL_DEPTH_DIFF_THRESHOLD_M = 0.02


def _to_hw1(mask):
    if mask.ndim == 2:
        mask = mask[..., None]
    return mask.float()


def dilate_binary_mask(mask, radius):
    """Dilate a [H, W, 1] or [H, W] binary mask."""

    mask = _to_hw1(mask)
    if radius <= 0:
        return mask

    x = (mask[..., 0] > 0.5).float()[None, None, ...]
    x = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return x[0, 0, ..., None]


def erode_binary_mask(mask, radius):
    """Erode a [H, W, 1] or [H, W] binary mask."""

    mask = _to_hw1(mask)
    if radius <= 0:
        return mask

    x = (mask[..., 0] > 0.5).float()[None, None, ...]
    x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return x[0, 0, ..., None]


def open_binary_mask(mask, radius):
    """Binary opening for small speckle removal."""

    return dilate_binary_mask(erode_binary_mask(mask, radius), radius)


def close_binary_mask(mask, radius):
    """Binary closing for filling tiny holes."""

    return erode_binary_mask(dilate_binary_mask(mask, radius), radius)


def remove_small_components(mask, min_area):
    """Remove connected components smaller than ``min_area`` pixels."""

    mask = _to_hw1(mask)
    if min_area <= 1:
        return (mask > 0.5).float()

    binary = mask[..., 0] > 0.5
    if not torch.any(binary):
        return torch.zeros_like(mask)

    labels_np, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return torch.zeros_like(mask)

    areas = np.bincount(labels_np.ravel(), minlength=num + 1)
    keep_labels = areas >= min_area
    keep_labels[0] = False
    keep_np = keep_labels[labels_np]
    return torch.from_numpy(keep_np).to(device=mask.device).float()[..., None]


def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask."""

    mask = _to_hw1(mask)
    binary = mask[..., 0] > 0.5
    if not torch.any(binary):
        return torch.zeros_like(mask)

    labels_np, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return torch.zeros_like(mask)

    areas = np.bincount(labels_np.ravel(), minlength=num + 1)
    areas[0] = 0
    best = int(np.argmax(areas))
    keep_np = labels_np == best
    return torch.from_numpy(keep_np).to(device=mask.device).float()[..., None]


def keep_all_components_above_min_area(mask, min_area):
    """Drop connected components below ``min_area`` but keep ALL surviving ones.

    Sibling of ``keep_largest_component_with_min_area`` for callers that need
    multi-component output (e.g. the feedforward closed-loop path, where every
    changed region should be back-projected, not just the largest).
    """
    mask = _to_hw1(mask)
    binary = mask[..., 0] > 0.5
    if not torch.any(binary):
        return torch.zeros_like(mask)
    labels_np, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return torch.zeros_like(mask)
    areas = np.bincount(labels_np.ravel(), minlength=num + 1)
    areas[0] = 0
    keep_labels = np.where(areas >= int(min_area))[0]
    if keep_labels.size == 0:
        return torch.zeros_like(mask)
    keep_np = np.isin(labels_np, keep_labels)
    return torch.from_numpy(keep_np).to(device=mask.device).float()[..., None]


def keep_largest_component_with_min_area(mask, min_area):
    """Combined `remove_small_components(min_area)` + `keep_largest_component`.

    Returns the single largest connected component of *mask*, but only if
    its area is at least *min_area* pixels; otherwise returns an empty
    mask. Replaces the prior two-scipy-label-call sequence (one inside
    the cleanup recipe + one in the caller) with a single CPU
    round-trip, which is the dominant cost on 800×800 masks.
    """

    mask = _to_hw1(mask)
    if min_area <= 1:
        return keep_largest_component(mask)

    binary = mask[..., 0] > 0.5
    if not torch.any(binary):
        return torch.zeros_like(mask)

    labels_np, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return torch.zeros_like(mask)

    areas = np.bincount(labels_np.ravel(), minlength=num + 1)
    areas[0] = 0
    best = int(np.argmax(areas))
    if int(areas[best]) < int(min_area):
        return torch.zeros_like(mask)

    keep_np = labels_np == best
    return torch.from_numpy(keep_np).to(device=mask.device).float()[..., None]


def select_top_n_components_filtered(
    mask,
    n: int = 3,
    area_ratio: float = 0.3,
    min_area: int = 1500,
):
    """Return up to ``n`` largest connected components of *mask* as a list of
    per-component binary masks (``[H, W, 1]`` float).

    Sorted by area descending. Components smaller than ``min_area`` pixels
    are dropped regardless. After taking the top ``n``, additionally drop
    any whose area is below ``area_ratio * largest_area`` (use
    ``area_ratio=0.0`` to disable the dominance filter).
    """

    mask = _to_hw1(mask)
    binary = mask[..., 0] > 0.5
    if not torch.any(binary):
        return []

    labels_np, num = _scipy_label(binary.detach().cpu().numpy())
    if num == 0:
        return []

    areas = np.bincount(labels_np.ravel(), minlength=num + 1)
    areas[0] = 0  # background

    # Sort by area descending, indices in labels_np start at 1.
    order = np.argsort(-areas[1:]) + 1  # (num,) labels sorted by area desc
    kept_labels: list[int] = []
    largest_area = int(areas[order[0]]) if len(order) > 0 else 0
    if largest_area < int(min_area):
        return []
    dominance_floor = max(int(min_area), int(round(float(area_ratio) * largest_area)))
    for lbl in order[: int(n)]:
        a = int(areas[lbl])
        if a < dominance_floor:
            break
        kept_labels.append(int(lbl))

    out = []
    for lbl in kept_labels:
        keep_np = labels_np == lbl
        out.append(
            torch.from_numpy(keep_np).to(device=mask.device).float()[..., None]
        )
    return out


def combine_object_masks(render_mask, live_mask, valid_mask=None):
    """Build the optimization mask from rendered and live object masks."""

    raw_union = ((_to_hw1(render_mask) > 0.5) | (_to_hw1(live_mask) > 0.5)).float()
    combined = close_binary_mask(raw_union, OFFICIAL_FILTER_OPEN_RADIUS)
    combined = open_binary_mask(combined, 1)
    combined = remove_small_components(combined, OFFICIAL_FILTER_MIN_AREA)
    if not torch.any(combined > 0.5):
        combined = raw_union
    if valid_mask is not None:
        combined = combined * _to_hw1(valid_mask)
    return combined


def _gaussian_blur_image(image, kernel_size, sigma, valid_mask=None):
    """Apply a light Gaussian blur while respecting an optional valid mask."""

    if sigma <= 0 or kernel_size <= 1:
        return image.float()

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    original_ndim = image.ndim
    if image.ndim == 2:
        image = image[..., None]

    x = image.float().permute(2, 0, 1).unsqueeze(0)
    channels = x.shape[1]

    if valid_mask is None:
        weights = torch.ones(
            1,
            1,
            image.shape[0],
            image.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
    else:
        weights = _to_hw1(valid_mask)[..., :1].permute(2, 0, 1).unsqueeze(0).to(device=x.device, dtype=x.dtype)

    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    weighted = x * weights
    blurred = F.conv2d(weighted, kernel.expand(channels, 1, -1, -1), padding=kernel_size // 2, groups=channels)
    norm = F.conv2d(weights, kernel, padding=kernel_size // 2)
    blurred = blurred / norm.clamp_min(1e-6)

    if original_ndim == 2:
        return blurred[0, 0]
    return blurred.squeeze(0).permute(1, 2, 0)


def _ssim_map(gray_pred, gray_gt, kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size, device=gray_pred.device, dtype=gray_pred.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)

    x = gray_pred[None, None]
    y = gray_gt[None, None]
    mu_x = F.conv2d(x, kernel, padding=kernel_size // 2)
    mu_y = F.conv2d(y, kernel, padding=kernel_size // 2)
    sigma_x = F.conv2d(x * x, kernel, padding=kernel_size // 2) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, kernel, padding=kernel_size // 2) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, kernel, padding=kernel_size // 2) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-6))[0, 0].clamp(0.0, 1.0)


def _rgb_msssim_score(
    pred_rgb,
    gt_rgb,
    valid_mask=None,
    blur_kernel_size=5,
    blur_sigma=1.0,
    pyramid_weights=(0.15, 0.30, 0.55),
):
    """Per-pixel MS-SSIM dissimilarity (3-level pyramid on luminance).

    ``pyramid_weights`` are (full-res, 1/2-res, 1/4-res). Default is
    COARSE-weighted (0.15, 0.30, 0.55): the rendered scene is inherently
    softer than the live camera image (static training early-stops, SH
    degree 0, downsampled seed), so the full-res SSIM band reads the
    texture-sharpness mismatch as "change" across the whole frame.
    Down-weighting the full-res band suppresses that while real content
    changes survive at the coarse scales.

    Measured (2026-06-12 A/B, recording_15fps_2026-06-11_115107, 6 static +
    6 motion 800x800 pairs through the runtime CDN path: gripper-erode 3 px,
    object exclusion, masked avg-pool ds=8, blur k=5 sigma=1.0, thr=0.07):
      - fine (0.55, 0.30, 0.15) [old default]: static-segment false positives
        1408-1984 px on EVERY pair; motion pairs 576-44736 px.
      - coarse (0.15, 0.30, 0.55) [new default]: static FPs 0-128 px
        (mean 0.01 % of valid pixels, was 0.31 %); motion pairs retain
        384-43328 px on 5/6 pairs (frame-155 pair: 576 px -> 0; its actual
        runtime CDN was already near-empty at 64 px).
      - Increasing the shared blur instead (sigma=2, coarse) made static FPs
        WORSE (0-1600 px) — reweighting, not more blur, is the fix.
    """
    region_mask = None
    if valid_mask is not None:
        region_mask = _to_hw1(valid_mask)[..., 0] > 0.5

    pred_rgb = _gaussian_blur_image(
        pred_rgb,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma,
        valid_mask=region_mask,
    )
    gt_rgb = _gaussian_blur_image(
        gt_rgb,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma,
        valid_mask=region_mask,
    )

    pred_gray = 0.2989 * pred_rgb[..., 0] + 0.5870 * pred_rgb[..., 1] + 0.1140 * pred_rgb[..., 2]
    gt_gray = 0.2989 * gt_rgb[..., 0] + 0.5870 * gt_rgb[..., 1] + 0.1140 * gt_rgb[..., 2]

    original_height, original_width = pred_gray.shape
    total = torch.zeros_like(pred_gray)
    weights = tuple(pyramid_weights)
    current_mask = None if region_mask is None else region_mask.float()

    for level, weight in enumerate(weights):
        score = 1.0 - _ssim_map(pred_gray, gt_gray)
        if current_mask is not None:
            score = score * (current_mask > 0.5).float()
        if score.shape != (original_height, original_width):
            score = F.interpolate(
                score[None, None],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        total = total + weight * score
        if level < len(weights) - 1:
            pred_gray = F.avg_pool2d(pred_gray[None, None], kernel_size=2, stride=2, ceil_mode=True)[0, 0]
            gt_gray = F.avg_pool2d(gt_gray[None, None], kernel_size=2, stride=2, ceil_mode=True)[0, 0]
            if current_mask is not None:
                current_mask = F.avg_pool2d(current_mask[None, None], kernel_size=2, stride=2, ceil_mode=True)[0, 0]

    return total


def _depth_diff_score(pred_depth, gt_depth, valid_mask=None, blur_kernel_size=5, blur_sigma=1.0):
    """Per-pixel absolute depth difference in metres.

    Mirrors ``_rgb_msssim_score``'s interface so it slots into the same
    threshold + cleanup pipeline. Both inputs may be (H, W) or (H, W, 1).
    Result is (H, W) float metres.

    The Gaussian blur is **mask-weighted via** ``_gaussian_blur_image`` — pixels
    where ``valid_mask`` is 0 (gripper, robot, tracked object) do NOT contribute
    to the blurred value of their neighbours, so the gripper's depth
    discontinuity cannot bleed across the silhouette into adjacent keep-region
    pixels. This matches what ``_rgb_msssim_score`` already does for RGB.

    Zero-depth pixels (sensor dropouts) get a zero score regardless of mask;
    the threshold step then re-applies ``valid_mask`` for the final binary.
    """
    def _squeeze(d):
        if d.ndim == 3:
            d = d[..., 0]
        return d.float()

    p = _squeeze(pred_depth)
    g = _squeeze(gt_depth)

    region_mask = None
    if valid_mask is not None:
        region_mask = _to_hw1(valid_mask)[..., 0] > 0.5

    if blur_kernel_size and blur_kernel_size > 1:
        p = _gaussian_blur_image(
            p, kernel_size=blur_kernel_size, sigma=blur_sigma, valid_mask=region_mask
        )
        g = _gaussian_blur_image(
            g, kernel_size=blur_kernel_size, sigma=blur_sigma, valid_mask=region_mask
        )
        if p.ndim == 3:
            p = p[..., 0]
        if g.ndim == 3:
            g = g[..., 0]

    valid_pix = (p > 1e-4) & (g > 1e-4)
    diff = (p - g).abs()
    diff = torch.where(valid_pix, diff, torch.zeros_like(diff))
    return diff


def _depth_outlier_score(
    pred_depth,
    gt_depth,
    valid_mask=None,
    blur_kernel_size=5,
    blur_sigma=1.0,
    median_multiplier: float = 10.0,
    min_threshold_m: float = 0.01,
):
    """Robust depth-outlier score, BIDIRECTIONAL (both ``pred > gt`` and
    ``pred < gt`` fire — we want CDN to catch both "new object in front" and
    "object moved away leaving a stale splat behind").

    Inspired by GaME (https://github.com/VladimirYugay/GaME,
    ``src/entities/game.py:_add_gaussians`` lines 574-575):

        depth_error > 40 * depth_error.median()

    GaME uses a one-sided gate (``rendered > sensor``) because they have a
    separate ``_remove_gaussians`` path for the opposite case. Our pipeline's
    FF dispatcher handles both via cull-in-front + insert, so we DROP the
    one-sided gate and fire on absolute error symmetrically.

    The ``median_multiplier`` factor (10× by default — lower than GaME's 40
    because our ICP-refined poses keep drift below ~5 mm, and we want to
    detect small-but-real object motion that 40× would miss). It's
    self-calibrating: whatever median pose-drift / sensor-noise the frame
    inherently has becomes the baseline, only outliers fire.

    A ``min_threshold_m`` floor (default 1 cm) prevents an extremely
    noise-free frame (median ≈ 0) from firing on every microscopic
    discrepancy.

    Returns a float score in [0, +inf): zero where the pixel is NOT a depth
    outlier, ``|err|`` where it IS. The downstream ``_threshold_mask``
    thresholds at 0. Caller passes ``depth_threshold=0.0``.
    """
    def _squeeze(d):
        if d.ndim == 3:
            d = d[..., 0]
        return d.float()

    p = _squeeze(pred_depth)
    g = _squeeze(gt_depth)

    region_mask = None
    if valid_mask is not None:
        region_mask = _to_hw1(valid_mask)[..., 0] > 0.5

    if blur_kernel_size and blur_kernel_size > 1:
        p = _gaussian_blur_image(p, kernel_size=blur_kernel_size, sigma=blur_sigma, valid_mask=region_mask)
        g = _gaussian_blur_image(g, kernel_size=blur_kernel_size, sigma=blur_sigma, valid_mask=region_mask)
        if p.ndim == 3:
            p = p[..., 0]
        if g.ndim == 3:
            g = g[..., 0]

    valid_pix = (p > 1e-4) & (g > 1e-4)
    err = (p - g).abs()
    err = torch.where(valid_pix, err, torch.zeros_like(err))
    if region_mask is not None:
        valid_pix = valid_pix & region_mask.to(valid_pix.device)
    if int(valid_pix.sum()) > 0:
        med = float(err[valid_pix].median().item())
    else:
        med = 0.0
    outlier_thr = max(float(median_multiplier) * med, float(min_threshold_m))
    outlier = valid_pix & (err > outlier_thr)
    return torch.where(outlier, err, torch.zeros_like(err))


def _threshold_mask(score, valid_mask, threshold):
    mask = torch.isfinite(score) & (score > threshold)
    if valid_mask is not None:
        region_mask = _to_hw1(valid_mask)[..., 0] > 0.5
        mask = mask & region_mask
    return mask.float()[..., None]


def _apply_cleanup_recipe(mask, valid_mask=None, close_radius=0, open_radius=0, min_area=1, keep_largest_only=True):
    cleaned = _to_hw1(mask)
    if close_radius > 0:
        cleaned = close_binary_mask(cleaned, close_radius)
    if open_radius > 0:
        cleaned = open_binary_mask(cleaned, open_radius)
    if keep_largest_only:
        cleaned = keep_largest_component_with_min_area(cleaned, min_area)
    else:
        cleaned = keep_all_components_above_min_area(cleaned, min_area)
    if valid_mask is not None:
        cleaned = cleaned * _to_hw1(valid_mask)
    # Cleanup-empty means NO real change — return empty. The old fallback
    # returned the RAW thresholded mask here (the very noise specks the
    # cleanup just rejected), so every "nothing changed" tick fed specks to
    # the feedforward, which inserted garbage there, which rendered worse
    # than the scene it replaced, which the next tick re-flagged BIGGER — a
    # compounding insert loop (measured: static-frame inserts ramping
    # 47 -> 849/call at 1920x1200; bounded but wasteful 18-83/call at
    # 800x800). Downstream handles an empty mask fine ("decode skipped").
    return cleaned


def build_change_mask(
    pred_depth,
    gt_depth,
    pred_rgb=None,
    gt_rgb=None,
    valid_mask=None,
    depth_threshold=0.02,
    rgb_threshold=None,
    use_rgb=True,
    blur_kernel_size=5,
    blur_sigma=1.0,
    filter_radius=1,
    min_component_size=64,
    keep_largest_only=True,
    mode="rgb",
    outlier_median_multiplier=10.0,
    outlier_min_threshold_m=0.01,
):
    """Build the dynamic-gs change mask.

    ``mode="rgb"`` (default): RGB MS-SSIM dissimilarity score, thresholded at
    ``rgb_threshold`` (falls back to ``OFFICIAL_RGB_MSSSIM_THRESHOLD``).
    ``mode="depth"``: per-pixel absolute depth diff (metres), thresholded at
    ``depth_threshold`` (default 2 cm, see ``OFFICIAL_DEPTH_DIFF_THRESHOLD_M``).
    In both modes the cleanup recipe (c10_o3_a760) and ``valid_mask`` re-apply
    the same way.
    """
    del use_rgb, filter_radius, min_component_size

    if mode == "rgb":
        if pred_rgb is None or gt_rgb is None:
            raise ValueError("mode='rgb' requires both pred_rgb and gt_rgb.")
        threshold = (
            OFFICIAL_RGB_MSSSIM_THRESHOLD if rgb_threshold is None else float(rgb_threshold)
        )
        score = _rgb_msssim_score(
            pred_rgb,
            gt_rgb,
            valid_mask=valid_mask,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        )
    elif mode == "depth":
        if pred_depth is None or gt_depth is None:
            raise ValueError("mode='depth' requires both pred_depth and gt_depth.")
        threshold = (
            OFFICIAL_DEPTH_DIFF_THRESHOLD_M if depth_threshold is None else float(depth_threshold)
        )
        score = _depth_diff_score(
            pred_depth,
            gt_depth,
            valid_mask=valid_mask,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        )
    elif mode == "depth_outlier":
        # Robust outlier-based score: self-calibrates against per-frame median
        # depth error, so bulk pose drift / sensor noise is absorbed and only
        # statistical outliers fire. Threshold is 0 (any positive score is an
        # outlier; the gating is inside _depth_outlier_score). Ported from
        # GaME (https://github.com/VladimirYugay/GaME, src/entities/game.py
        # lines 574-575: depth_error > 40 * depth_error.median()).
        if pred_depth is None or gt_depth is None:
            raise ValueError("mode='depth_outlier' requires both pred_depth and gt_depth.")
        threshold = 0.0
        score = _depth_outlier_score(
            pred_depth,
            gt_depth,
            valid_mask=valid_mask,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            median_multiplier=float(outlier_median_multiplier),
            min_threshold_m=float(outlier_min_threshold_m),
        )
    else:
        raise ValueError(f"build_change_mask: unknown mode '{mode}' (expected 'rgb' / 'depth' / 'depth_outlier')")

    basic_mask = _threshold_mask(score, valid_mask=valid_mask, threshold=threshold)
    filtered_mask = _apply_cleanup_recipe(
        basic_mask,
        valid_mask=valid_mask,
        close_radius=OFFICIAL_FILTER_CLOSE_RADIUS,
        open_radius=OFFICIAL_FILTER_OPEN_RADIUS,
        min_area=OFFICIAL_FILTER_MIN_AREA,
        keep_largest_only=keep_largest_only,
    )
    final_mask = filtered_mask
    if valid_mask is not None:
        final_mask = final_mask * _to_hw1(valid_mask)

    if torch.any(final_mask):
        return final_mask
    if torch.any(filtered_mask):
        return filtered_mask
    return basic_mask


def extract_projected_centers_and_radii(info, num_points):
    """Read projected centers and radii from gsplat rasterization metadata."""

    if "means2d" not in info:
        raise KeyError("'means2d' not found in rasterization info.")
    if "radii" not in info:
        raise KeyError("'radii' not found in rasterization info.")

    centers = info["means2d"]
    radii = info["radii"]

    if centers.ndim == 3:
        centers = centers[0]
    if centers.ndim != 2:
        centers = centers.reshape(-1, 2)
    if radii.ndim > 1:
        radii = radii.reshape(-1)

    centers = centers.float()
    radii = radii.float()

    if centers.shape[0] != num_points:
        raise ValueError("Projected center count does not match the Gaussian count.")
    if radii.shape[0] != num_points:
        raise ValueError("Projected radius count does not match the Gaussian count.")
    if centers.shape[-1] != 2:
        raise ValueError("Projected centers must have shape [N, 2].")

    return centers, radii


def build_active_mask(mask, centers_2d, radii):
    """Mark a Gaussian active if its projected footprint overlaps the binary mask."""

    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0.5

    height, width = mask.shape
    integral = torch.cumsum(torch.cumsum(mask.to(torch.int32), dim=0), dim=1)

    x = centers_2d[:, 0]
    y = centers_2d[:, 1]
    r = radii.reshape(-1).clamp_min(1.0)

    x0 = torch.floor(x - r).long().clamp(0, width - 1)
    x1 = torch.ceil(x + r).long().clamp(0, width - 1)
    y0 = torch.floor(y - r).long().clamp(0, height - 1)
    y1 = torch.ceil(y + r).long().clamp(0, height - 1)

    def rect_sum(xx0, yy0, xx1, yy1):
        a = integral[yy1, xx1]
        b = torch.where(xx0 > 0, integral[yy1, xx0 - 1], torch.zeros_like(a))
        c = torch.where(yy0 > 0, integral[yy0 - 1, xx1], torch.zeros_like(a))
        d = torch.where((xx0 > 0) & (yy0 > 0), integral[yy0 - 1, xx0 - 1], torch.zeros_like(a))
        return a - b - c + d

    overlap = rect_sum(x0, y0, x1, y1) > 0
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(r)
    return overlap & finite & (r > 0)


def build_active_mask_center_only(mask, centers_2d, dilate_px: int = 0):
    """Mark a Gaussian active iff its projected 2D center falls inside ``mask``.

    Unlike ``build_active_mask`` (footprint-overlap via integral image), this
    samples ``mask`` at the integer pixel under each Gaussian center, so a
    Gaussian whose rendered footprint extends into the mask but whose actual
    3D centre projects outside is excluded.

    ``dilate_px`` morphologically dilates ``mask`` before sampling, so the
    rule becomes "centre inside the mask OR within ``dilate_px`` of its
    border". 0 = strict centre test.
    """

    if mask.ndim == 3:
        mask = mask[..., 0]
    bool_mask = mask > 0.5

    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        dil = torch.nn.functional.max_pool2d(
            bool_mask.float().unsqueeze(0).unsqueeze(0),
            kernel_size=k, stride=1, padding=int(dilate_px),
        )
        bool_mask = (dil.squeeze(0).squeeze(0) > 0.5)

    height, width = bool_mask.shape
    x = centers_2d[:, 0]
    y = centers_2d[:, 1]
    finite = torch.isfinite(x) & torch.isfinite(y)
    xi = torch.where(finite, x.round().long(), torch.zeros_like(x, dtype=torch.long))
    yi = torch.where(finite, y.round().long(), torch.zeros_like(y, dtype=torch.long))
    in_bounds = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi = xi.clamp(0, width - 1)
    yi = yi.clamp(0, height - 1)
    inside = bool_mask[yi, xi]
    return inside & finite & in_bounds
