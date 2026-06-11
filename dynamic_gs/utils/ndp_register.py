"""NDP (Neural Deformation Pyramid) non-rigid registration wrapper.

Deforms a complete (approximate) SAM3D source cloud onto a partial (accurate)
target cloud using the no-learned hierarchical Sim3 deformation pyramid. Runs
in-process on GPU in the main `dynamic_gs` env — no pytorch3d, no subprocess.

This is the default Phase-0b registration backend, replacing the rigid
CPD / TEASER++ similarity refinement. The prototype + tuning lives in
`scripts/experiments/nonrigid_bench/` (see `02_run_ndp.py`).

Public entrypoint: :func:`deform_source_to_target`.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from .ndp.nets import Deformation_Pyramid

# Defaults mirror DeformationPyramid/shape_transfer.py (validated in the bench).
_NDP_CONFIG = dict(
    iters=500,
    lr=0.01,
    max_break_count=15,
    break_threshold_ratio=0.001,
    samples=6000,
    motion_type="Sim3",
    rotation_format="euler",
    m=9,
    k0=-8,
    depth=3,
    width=128,
    w_reg=0.0,
)


def _setup_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _truncated_chamfer(a: torch.Tensor, b: torch.Tensor, trunc: float = 1e9) -> torch.Tensor:
    """Symmetric truncated Chamfer for (1,N,3) tensors via dense cdist.

    Equivalent to DeformationPyramid/model/loss.compute_truncated_chamfer_distance
    for the few-thousand-point subsample NDP optimizes on (avoids the pytorch3d
    KNN dependency).
    """
    a2, b2 = a.squeeze(0), b.squeeze(0)
    d = torch.cdist(a2, b2)
    d_a2b = torch.clamp(d.min(dim=1).values, max=trunc)
    d_b2a = torch.clamp(d.min(dim=0).values, max=trunc)
    return d_a2b.mean() + d_b2a.mean()


def deform_source_to_target(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    device: torch.device | str | None = None,
    config: dict | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Non-rigidly warp ``source_xyz`` (N,3) onto ``target_xyz`` (M,3).

    Both clouds are expected in the SAME metric world frame (the source should
    already be roughly placed/scaled onto the target — e.g. by the SAM3D pose
    init + bbox-scale + centroid translation; NDP handles the residual local
    deformation). Returns ``(warped_source_xyz (N,3) float32, meta)``.

    The warp is applied to ALL source points; the optimization itself runs on a
    random ``config['samples']``-point subsample of each cloud for speed.
    """
    cfg = dict(_NDP_CONFIG)
    if config:
        cfg.update(config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    _setup_seed(seed)
    t0 = time.perf_counter()

    src = torch.from_numpy(np.ascontiguousarray(source_xyz, dtype=np.float32)).to(device)
    tgt = torch.from_numpy(np.ascontiguousarray(target_xyz, dtype=np.float32)).to(device)

    # Random subsample for the optimization (full cloud is warped at the end).
    if src.shape[0] > cfg["samples"]:
        sel = torch.randperm(src.shape[0], device=device)[: cfg["samples"]]
        src_s = src[sel]
    else:
        src_s = src
    if tgt.shape[0] > cfg["samples"]:
        sel = torch.randperm(tgt.shape[0], device=device)[: cfg["samples"]]
        tgt_s = tgt[sel]
    else:
        tgt_s = tgt

    ndp = Deformation_Pyramid(
        depth=cfg["depth"], width=cfg["width"], device=device,
        k0=cfg["k0"], m=cfg["m"], nonrigidity_est=cfg["w_reg"] > 0,
        rotation_format=cfg["rotation_format"], motion=cfg["motion_type"],
    )

    # Cancel global translation (matches shape_transfer.py); re-add target
    # centroid at the end so the warped cloud lands in the world frame.
    src_mean = src_s.mean(dim=0, keepdim=True)
    tgt_mean = tgt_s.mean(dim=0, keepdim=True)
    s_sample = src_s - src_mean
    t_sample = tgt_s - tgt_mean

    for level in range(ndp.n_hierarchy):
        ndp.gradient_setup(optimized_level=level)
        optimizer = torch.optim.Adam(ndp.pyramid[level].parameters(), lr=cfg["lr"])
        break_counter = 0
        loss_prev = 1e6
        for _ in range(cfg["iters"]):
            s_warped, _ = ndp.warp(s_sample, max_level=level, min_level=level)
            loss = _truncated_chamfer(s_warped[None], t_sample[None], trunc=1e9)
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * cfg["break_threshold_ratio"]:
                break_counter += 1
            if break_counter >= cfg["max_break_count"]:
                break
            loss_prev = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        s_sample = s_warped.detach()

    # Warp the FULL source cloud through the fitted pyramid.
    ndp.gradient_setup(optimized_level=-1)
    with torch.no_grad():
        warped_full, _ = ndp.warp(src - src_mean)
    warped_world = (warped_full + tgt_mean).cpu().numpy().astype(np.float32)

    if device.type == "cuda":
        torch.cuda.synchronize()
    meta = {
        "ndp_seconds": float(time.perf_counter() - t0),
        "ndp_samples": int(min(src.shape[0], cfg["samples"])),
        "ndp_levels": int(ndp.n_hierarchy),
        "source_count": int(src.shape[0]),
        "target_count": int(tgt.shape[0]),
    }
    return warped_world, meta
