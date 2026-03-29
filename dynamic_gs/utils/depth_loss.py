from __future__ import annotations

import torch


def masked_l1_depth_loss(pred_depth, gt_depth, mask=None):
    """Masked finite-value L1 depth loss."""

    if pred_depth.ndim == 3 and pred_depth.shape[-1] == 1:
        pred_depth = pred_depth[..., 0]
    if gt_depth.ndim == 3 and gt_depth.shape[-1] == 1:
        gt_depth = gt_depth[..., 0]

    valid = torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    if mask is not None:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        valid = valid & (mask > 0.5)

    if not torch.any(valid):
        return pred_depth.new_tensor(0.0)

    return torch.abs(pred_depth[valid] - gt_depth[valid]).mean()
