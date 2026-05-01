from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

try:
    from efficient_sam.efficient_sam import build_efficient_sam
except ImportError:  # pragma: no cover - optional dependency at import time
    build_efficient_sam = None


ESAM_CHECKPOINT_URL = "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"
ESAM_CHECKPOINT_PATH = Path.home() / ".cache" / "efficient_sam" / "efficient_sam_vitt.pt"
ESAM_NUM_PROMPT_POINTS = 8
ESAM_PROMPT_KEEP_RATIO = 0.8
# Cap the longest side of the ESAM input. ViT-Tiny's encoder cost scales
# with input area, so 800x800 → 512xR (R≈512) cuts inference cost by ~60%.
# The mask returned at low resolution is upsampled with nearest-neighbor
# back to the input resolution.
ESAM_MAX_SIDE = 512


def _to_mask_numpy(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach()
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return (mask.float().cpu().numpy() > 0.5)


def compute_prompt_interior(mask: torch.Tensor, keep_ratio: float = ESAM_PROMPT_KEEP_RATIO) -> tuple[torch.Tensor, torch.Tensor]:
    mask_np = _to_mask_numpy(mask)
    if not np.any(mask_np):
        empty = torch.zeros(mask_np.shape, dtype=torch.bool, device=mask.device)
        dist = torch.zeros(mask_np.shape, dtype=torch.float32, device=mask.device)
        return empty, dist

    # Restrict the EDT to the mask's bounding box (with a 1-px margin so
    # boundary pixels still see "outside"). For typical small object masks,
    # this drops EDT cost from O(H*W) on 800x800 → O(bbox area).
    ys, xs = np.where(mask_np)
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, mask_np.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, mask_np.shape[1])
    crop = mask_np[y0:y1, x0:x1]
    crop_dist = distance_transform_edt(crop)
    dist_np = np.zeros(mask_np.shape, dtype=np.float32)
    dist_np[y0:y1, x0:x1] = crop_dist

    threshold = float(np.quantile(dist_np[mask_np], 1.0 - keep_ratio))
    inner_np = mask_np & (dist_np >= threshold)
    if not np.any(inner_np):
        inner_np = mask_np

    inner = torch.from_numpy(inner_np).to(device=mask.device, dtype=torch.bool)
    dist = torch.from_numpy(dist_np).to(device=mask.device, dtype=torch.float32)
    return inner, dist


def sample_interior_points(inner_mask: torch.Tensor, distance_map: torch.Tensor, num_points: int = ESAM_NUM_PROMPT_POINTS) -> torch.Tensor:
    coords = torch.nonzero(inner_mask, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=inner_mask.device)

    num_points = min(max(1, int(num_points)), ESAM_NUM_PROMPT_POINTS, coords.shape[0])
    distances = distance_map[coords[:, 0], coords[:, 1]]

    cutoff = torch.quantile(distances, 0.5) if coords.shape[0] > num_points else distances.min()
    safer_mask = distances >= cutoff
    safer_coords = coords[safer_mask]
    safer_distances = distances[safer_mask]
    if safer_coords.shape[0] < num_points:
        safer_coords = coords
        safer_distances = distances

    selected_indices = [int(torch.argmax(safer_distances).item())]
    while len(selected_indices) < num_points:
        selected = safer_coords[selected_indices]
        deltas = safer_coords[:, None, :] - selected[None, :, :]
        min_sq_dist = (deltas * deltas).sum(dim=2).min(dim=1).values.float()
        score = min_sq_dist * (1.0 + safer_distances)
        score[selected_indices] = -1.0
        next_index = int(torch.argmax(score).item())
        if score[next_index] < 0:
            break
        selected_indices.append(next_index)

    points_rc = safer_coords[selected_indices]
    return points_rc[:, [1, 0]]


def ensure_esam_checkpoint(checkpoint_path: Path = ESAM_CHECKPOINT_PATH) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return checkpoint_path
    urllib.request.urlretrieve(ESAM_CHECKPOINT_URL, checkpoint_path)
    return checkpoint_path


def build_esam_ti(device: torch.device, checkpoint_path: Path = ESAM_CHECKPOINT_PATH):
    if build_efficient_sam is None:
        raise ImportError(
            "EfficientSAM is required for dynamic-gs dynamic masking. "
            "Install it with `pip install git+https://github.com/yformer/EfficientSAM.git`."
        )

    checkpoint_path = ensure_esam_checkpoint(checkpoint_path)
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=str(checkpoint_path),
    )
    return model.to(device).eval()


def _select_esam_mask(predicted_logits: torch.Tensor, predicted_iou: torch.Tensor | None, prompt_region: torch.Tensor) -> torch.Tensor:
    candidate_masks = predicted_logits[0, 0] >= 0
    if predicted_iou is not None and torch.isfinite(predicted_iou).any():
        best_index = int(torch.argmax(predicted_iou[0, 0]).item())
        return candidate_masks[best_index]

    prompt_region = prompt_region.to(device=candidate_masks.device, dtype=torch.bool)
    overlaps = (candidate_masks & prompt_region[None, ...]).flatten(1).sum(dim=1)
    best_index = int(torch.argmax(overlaps).item())
    return candidate_masks[best_index]


def _run_esam_query(
    model,
    image_tensor: torch.Tensor,
    prompt_region: torch.Tensor,
    num_points: int,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner_mask, distance_map = compute_prompt_interior(prompt_region, keep_ratio=keep_ratio)
    points_xy = sample_interior_points(inner_mask, distance_map, num_points=num_points)
    if points_xy.shape[0] == 0:
        empty = torch.zeros_like(inner_mask)
        return empty, inner_mask, points_xy

    point_tensor = points_xy.float().view(1, 1, -1, 2).to(image_tensor.device)
    label_tensor = torch.ones((1, 1, points_xy.shape[0]), dtype=torch.float32, device=image_tensor.device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(image_tensor, point_tensor, label_tensor)

    esam_mask = _select_esam_mask(predicted_logits, predicted_iou, inner_mask)
    return esam_mask.to(prompt_region.device), inner_mask, points_xy


def query_esam_mask(
    model,
    rendered_rgb: torch.Tensor,
    change_mask: torch.Tensor,
    num_points: int = ESAM_NUM_PROMPT_POINTS,
    keep_ratio: float = ESAM_PROMPT_KEEP_RATIO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = rendered_rgb.detach().float()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Rendered RGB image must have shape [H, W, 3].")

    H_orig, W_orig = int(image.shape[0]), int(image.shape[1])
    cm = change_mask.float()
    if cm.ndim == 2:
        cm = cm[..., None]

    # Downsample image and prompt region to ESAM_MAX_SIDE on the longest
    # side. ViT-Tiny encoder cost is quadratic in input area, so this is
    # the single biggest knob on ESAM latency. The output mask is
    # upsampled with nearest-neighbor; prompt points are re-scaled.
    max_side = max(H_orig, W_orig)
    if max_side > ESAM_MAX_SIDE:
        scale = float(ESAM_MAX_SIDE) / float(max_side)
        H_lo = max(1, int(round(H_orig * scale)))
        W_lo = max(1, int(round(W_orig * scale)))
        image_lo = F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(H_lo, W_lo),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        cm_lo = F.interpolate(
            cm.permute(2, 0, 1).unsqueeze(0),
            size=(H_lo, W_lo),
            mode="nearest",
        ).squeeze(0).permute(1, 2, 0)
    else:
        H_lo, W_lo = H_orig, W_orig
        image_lo = image
        cm_lo = cm

    image_tensor = image_lo.permute(2, 0, 1).unsqueeze(0).to(rendered_rgb.device)
    prompt_region = cm_lo[..., 0] if cm_lo.ndim == 3 else cm_lo

    # Single ESAM pass — the prior 3-iteration convergence loop rarely
    # changed the result by more than a few pixels and tripled latency.
    esam_mask, inner_mask, points_xy = _run_esam_query(
        model,
        image_tensor,
        prompt_region,
        num_points=num_points,
        keep_ratio=keep_ratio,
    )

    # Upsample mask back to the original image resolution (binary →
    # nearest-neighbor) and rescale point coordinates.
    if (H_lo, W_lo) != (H_orig, W_orig):
        esam_mask_hi = F.interpolate(
            esam_mask.float()[None, None, ...],
            size=(H_orig, W_orig),
            mode="nearest",
        )[0, 0].bool()
        inner_mask_hi = F.interpolate(
            inner_mask.float()[None, None, ...],
            size=(H_orig, W_orig),
            mode="nearest",
        )[0, 0].bool()
        if points_xy.shape[0] > 0:
            sx = float(W_orig) / float(W_lo)
            sy = float(H_orig) / float(H_lo)
            scaled = points_xy.float()
            scaled = torch.stack([scaled[:, 0] * sx, scaled[:, 1] * sy], dim=-1)
            points_xy = scaled.long()
    else:
        esam_mask_hi = esam_mask
        inner_mask_hi = inner_mask

    return esam_mask_hi.to(change_mask.device), inner_mask_hi, points_xy


def query_esam_mask_pair(
    model,
    rendered_rgb_a: torch.Tensor,
    rendered_rgb_b: torch.Tensor,
    change_mask: torch.Tensor,
    num_points: int = ESAM_NUM_PROMPT_POINTS,
    keep_ratio: float = ESAM_PROMPT_KEEP_RATIO,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Run ESAM on two images that share the same prompt mask, in a single
    batched forward pass.

    The change mask is identical for the render and live calls in
    ``prepare_dynamic_update``, so the prompt-interior EDT and point
    sampling only need to be computed once. Stacking the two images into
    a batch=2 forward pass also gives the GPU better utilization than
    two sequential forwards. Returns
    ``((mask_a, inner_a, points_a), (mask_b, inner_b, points_b))``.
    """
    image_a = rendered_rgb_a.detach().float()
    image_b = rendered_rgb_b.detach().float()
    if image_a.ndim != 3 or image_a.shape[-1] != 3:
        raise ValueError("rendered_rgb_a must have shape [H, W, 3].")
    if image_b.shape != image_a.shape:
        raise ValueError("rendered_rgb_a and rendered_rgb_b must have identical shape.")

    H_orig, W_orig = int(image_a.shape[0]), int(image_a.shape[1])
    cm = change_mask.float()
    if cm.ndim == 2:
        cm = cm[..., None]

    max_side = max(H_orig, W_orig)
    if max_side > ESAM_MAX_SIDE:
        scale = float(ESAM_MAX_SIDE) / float(max_side)
        H_lo = max(1, int(round(H_orig * scale)))
        W_lo = max(1, int(round(W_orig * scale)))
        image_a_lo = F.interpolate(
            image_a.permute(2, 0, 1).unsqueeze(0),
            size=(H_lo, W_lo),
            mode="bilinear",
            align_corners=False,
        )
        image_b_lo = F.interpolate(
            image_b.permute(2, 0, 1).unsqueeze(0),
            size=(H_lo, W_lo),
            mode="bilinear",
            align_corners=False,
        )
        cm_lo_3 = F.interpolate(
            cm.permute(2, 0, 1).unsqueeze(0),
            size=(H_lo, W_lo),
            mode="nearest",
        ).squeeze(0).permute(1, 2, 0)
        prompt_lo = cm_lo_3[..., 0]
    else:
        H_lo, W_lo = H_orig, W_orig
        image_a_lo = image_a.permute(2, 0, 1).unsqueeze(0)
        image_b_lo = image_b.permute(2, 0, 1).unsqueeze(0)
        prompt_lo = cm[..., 0]

    # Compute interior mask + prompt points ONCE (the prompt is shared).
    inner_mask, distance_map = compute_prompt_interior(prompt_lo, keep_ratio=keep_ratio)
    points_xy = sample_interior_points(inner_mask, distance_map, num_points=num_points)

    if points_xy.shape[0] == 0:
        empty = torch.zeros((H_orig, W_orig), dtype=torch.bool, device=change_mask.device)
        empty_pts = torch.zeros((0, 2), dtype=torch.long, device=change_mask.device)
        return (empty, empty, empty_pts), (empty, empty, empty_pts)

    device = rendered_rgb_a.device
    image_batch = torch.cat([image_a_lo, image_b_lo], dim=0).to(device)
    point_tensor = (
        points_xy.float().view(1, 1, -1, 2).expand(2, 1, -1, 2).contiguous().to(device)
    )
    label_tensor = torch.ones((2, 1, points_xy.shape[0]), dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(image_batch, point_tensor, label_tensor)

    inner_mask_dev = inner_mask.to(device=predicted_logits.device, dtype=torch.bool)

    def _pick(idx: int) -> torch.Tensor:
        candidate_masks = predicted_logits[idx, 0] >= 0
        if predicted_iou is not None and torch.isfinite(predicted_iou[idx]).any():
            best = int(torch.argmax(predicted_iou[idx, 0]).item())
            return candidate_masks[best]
        overlaps = (candidate_masks & inner_mask_dev[None, ...]).flatten(1).sum(dim=1)
        best = int(torch.argmax(overlaps).item())
        return candidate_masks[best]

    mask_a_lo = _pick(0)
    mask_b_lo = _pick(1)

    if (H_lo, W_lo) != (H_orig, W_orig):
        def _up(m):
            return F.interpolate(
                m.float()[None, None, ...],
                size=(H_orig, W_orig),
                mode="nearest",
            )[0, 0].bool()
        mask_a_hi = _up(mask_a_lo)
        mask_b_hi = _up(mask_b_lo)
        inner_hi = _up(inner_mask)
        sx = float(W_orig) / float(W_lo)
        sy = float(H_orig) / float(H_lo)
        scaled = torch.stack([points_xy[:, 0].float() * sx, points_xy[:, 1].float() * sy], dim=-1).long()
        points_hi = scaled
    else:
        mask_a_hi = mask_a_lo
        mask_b_hi = mask_b_lo
        inner_hi = inner_mask
        points_hi = points_xy

    return (
        (mask_a_hi.to(change_mask.device), inner_hi, points_hi),
        (mask_b_hi.to(change_mask.device), inner_hi, points_hi),
    )
