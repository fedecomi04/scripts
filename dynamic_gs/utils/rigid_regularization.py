from __future__ import annotations

import torch
from torch import Tensor


def estimate_rigid_transform_kabsch(reference_points: Tensor, current_points: Tensor) -> tuple[Tensor, Tensor] | None:
    """Return the best-fit rigid transform mapping reference_points to current_points."""

    if reference_points.ndim != 2 or current_points.ndim != 2:
        return None
    if reference_points.shape != current_points.shape or reference_points.shape[0] < 3:
        return None

    ref_center = reference_points.mean(dim=0, keepdim=True)
    cur_center = current_points.mean(dim=0, keepdim=True)
    ref_zero = reference_points - ref_center
    cur_zero = current_points - cur_center

    try:
        u, _, vh = torch.linalg.svd(ref_zero.transpose(0, 1) @ cur_zero)
    except RuntimeError:
        return None

    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1, :] *= -1
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)

    translation = cur_center[0] - rotation @ ref_center[0]
    if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
        return None
    return rotation, translation


def rigid_or_static_loss(
    reference_points: Tensor | None,
    current_points: Tensor | None,
    rigid_inlier_threshold: float,
) -> Tensor:
    """Encourage each point to either follow the common rigid motion or stay still."""

    if reference_points is None and current_points is None:
        return torch.zeros(())
    if reference_points is None:
        return current_points.new_zeros(())  # type: ignore[union-attr]
    if current_points is None:
        return reference_points.new_zeros(())
    if reference_points.shape != current_points.shape or reference_points.shape[0] < 3:
        return current_points.new_zeros(())

    transform = estimate_rigid_transform_kabsch(reference_points, current_points)
    if transform is None:
        return current_points.new_zeros(())

    rotation, translation = transform
    rigid_target = reference_points @ rotation.transpose(0, 1) + translation
    rigid_residual = torch.sum((current_points - rigid_target) ** 2, dim=-1)
    static_residual = torch.sum((current_points - reference_points) ** 2, dim=-1)
    rigid_inlier = (rigid_residual.detach() <= rigid_inlier_threshold).float()
    return torch.mean(rigid_inlier * rigid_residual + (1.0 - rigid_inlier) * static_residual)
