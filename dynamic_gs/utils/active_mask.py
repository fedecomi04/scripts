from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F

OFFICIAL_RGB_MSSSIM_THRESHOLD = 0.10
OFFICIAL_FILTER_CLOSE_RADIUS = 10
OFFICIAL_FILTER_OPEN_RADIUS = 3
OFFICIAL_FILTER_MIN_AREA = 760


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

    mask_cpu = (mask[..., 0] > 0.5).detach().cpu()
    visited = torch.zeros_like(mask_cpu, dtype=torch.bool)
    keep = torch.zeros_like(mask_cpu, dtype=torch.bool)
    height, width = mask_cpu.shape

    for start_y, start_x in torch.nonzero(mask_cpu, as_tuple=False).tolist():
        if visited[start_y, start_x]:
            continue

        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        component = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask_cpu[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        if len(component) >= min_area:
            ys, xs = zip(*component)
            keep[list(ys), list(xs)] = True

    return keep.to(mask.device).float()[..., None]


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


def _rgb_msssim_score(pred_rgb, gt_rgb, valid_mask=None, blur_kernel_size=5, blur_sigma=1.0):
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
    weights = (0.55, 0.30, 0.15)
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


def _threshold_mask(score, valid_mask, threshold):
    mask = torch.isfinite(score) & (score > threshold)
    if valid_mask is not None:
        region_mask = _to_hw1(valid_mask)[..., 0] > 0.5
        mask = mask & region_mask
    return mask.float()[..., None]


def _apply_cleanup_recipe(mask, valid_mask=None, close_radius=0, open_radius=0, min_area=1):
    cleaned = _to_hw1(mask)
    if close_radius > 0:
        cleaned = close_binary_mask(cleaned, close_radius)
    if open_radius > 0:
        cleaned = open_binary_mask(cleaned, open_radius)
    cleaned = remove_small_components(cleaned, min_area)
    if valid_mask is not None:
        cleaned = cleaned * _to_hw1(valid_mask)
    if torch.any(cleaned):
        return cleaned
    return (mask > 0.5).float()


def build_change_mask(
    pred_depth,
    gt_depth,
    pred_rgb=None,
    gt_rgb=None,
    valid_mask=None,
    depth_threshold=0.02,
    rgb_threshold=0.15,
    use_rgb=True,
    blur_kernel_size=5,
    blur_sigma=1.0,
    filter_radius=1,
    min_component_size=64,
):
    """Build the official dynamic-gs change mask.

    Uses the RGB MS-SSIM score, thresholds it at 0.10, applies the selected
    lvl10 cleanup recipe (c10_o3_a760), then re-applies the dataset image mask
    from ``batch["mask"]`` as the safety constraint.
    """
    del pred_depth, gt_depth, depth_threshold, rgb_threshold, use_rgb, filter_radius, min_component_size

    if pred_rgb is None or gt_rgb is None:
        raise ValueError("dynamic-gs change mask requires both pred_rgb and gt_rgb.")

    basic_mask = _threshold_mask(
        _rgb_msssim_score(
            pred_rgb,
            gt_rgb,
            valid_mask=valid_mask,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        ),
        valid_mask=valid_mask,
        threshold=OFFICIAL_RGB_MSSSIM_THRESHOLD,
    )
    filtered_mask = _apply_cleanup_recipe(
        basic_mask,
        valid_mask=valid_mask,
        close_radius=OFFICIAL_FILTER_CLOSE_RADIUS,
        open_radius=OFFICIAL_FILTER_OPEN_RADIUS,
        min_area=OFFICIAL_FILTER_MIN_AREA,
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
