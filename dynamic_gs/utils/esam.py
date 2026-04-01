from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

try:
    from efficient_sam.efficient_sam import build_efficient_sam
except ImportError:  # pragma: no cover - optional dependency at import time
    build_efficient_sam = None


ESAM_CHECKPOINT_URL = "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"
ESAM_CHECKPOINT_PATH = Path.home() / ".cache" / "efficient_sam" / "efficient_sam_vitt.pt"
ESAM_NUM_PROMPT_POINTS = 8


def _to_mask_numpy(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach()
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return (mask.float().cpu().numpy() > 0.5)


def compute_prompt_interior(mask: torch.Tensor, keep_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    mask_np = _to_mask_numpy(mask)
    if not np.any(mask_np):
        empty = torch.zeros(mask_np.shape, dtype=torch.bool, device=mask.device)
        dist = torch.zeros(mask_np.shape, dtype=torch.float32, device=mask.device)
        return empty, dist

    dist_np = distance_transform_edt(mask_np)
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


def query_esam_mask(
    model,
    rendered_rgb: torch.Tensor,
    change_mask: torch.Tensor,
    num_points: int = ESAM_NUM_PROMPT_POINTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner_mask, distance_map = compute_prompt_interior(change_mask, keep_ratio=0.9)
    points_xy = sample_interior_points(inner_mask, distance_map, num_points=num_points)
    if points_xy.shape[0] == 0:
        empty = torch.zeros_like(inner_mask)
        return empty, inner_mask, points_xy

    image = rendered_rgb.detach().float()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Rendered RGB image must have shape [H, W, 3].")

    image_tensor = image.permute(2, 0, 1).unsqueeze(0).to(rendered_rgb.device)
    point_tensor = points_xy.float().view(1, 1, -1, 2).to(rendered_rgb.device)
    label_tensor = torch.ones((1, 1, points_xy.shape[0]), dtype=torch.float32, device=rendered_rgb.device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(image_tensor, point_tensor, label_tensor)

    esam_mask = _select_esam_mask(predicted_logits, predicted_iou, inner_mask)
    return esam_mask.to(change_mask.device), inner_mask, points_xy
