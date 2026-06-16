"""Median + bilateral depth filter, shared by the static and dynamic phases.

The real ZED-X (and the upper-bound sim noise model in zed_depth_noise.py) leaves
two artifacts on the depth map:
  * stereo FLYING-PIXELS at depth discontinuities — a silhouette pixel whose depth
    jumped to the background behind it (back-projects to a point floating off the
    surface; operator-observed as AnySplat gaussians inserted in the air);
  * per-pixel axial JITTER on otherwise-flat surfaces (~mm scale).

A MEDIAN kill the flying-pixels (isolated outliers → replaced by a true neighbour
value, NO edge-smear), then a weight-corrected BILATERAL smooths the residual
surface jitter while preserving the real >cm depth steps at object edges. This
median+bilateral pair was A/B'd against raw / median-only / bilateral-only on a
static banana reprojection (2026-06-16) and chosen by the operator: lowest flat-
plane RMS (1.75mm vs raw 1.93) AND no floating inserts.

Invalid (==0) pixels are held out throughout: the filters never invent depth where
the sensor returned none, and holes never bleed their zeros into valid neighbours
(the bilateral is weight-normalized by a validity mask filtered the same way).

Knobs are env-overridable for A/B (no relaunch); the filter is ON by default.
``DGS_DEPTH_FILTER=0`` disables it everywhere.
"""

import os
import numpy as np

# Median kernel (px). cv2.medianBlur on float32 supports only 3 or 5.
_MED_K = int(float(os.environ.get("DGS_DEPTH_MEDIAN_KSIZE", "5")))
# Bilateral: diameter (px), colour sigma (METRES, depth units), space sigma (px).
_BI_D = int(float(os.environ.get("DGS_DEPTH_BILATERAL_D", "5")))
_BI_SIGMA_COLOR_M = float(os.environ.get("DGS_DEPTH_BILATERAL_SIGMA_COLOR_M", "0.01"))  # 1 cm
_BI_SIGMA_SPACE = float(os.environ.get("DGS_DEPTH_BILATERAL_SIGMA_SPACE", "5.0"))


def enabled() -> bool:
    """True iff the median+bilateral depth filter should be applied."""
    return os.environ.get("DGS_DEPTH_FILTER", "1") != "0"


def filter_depth(depth_m: np.ndarray) -> np.ndarray:
    """Median+bilateral filter a float32 depth map (metres, 0 = invalid).

    Returns a new array; does not mutate the input. Invalid pixels stay 0 and never
    contaminate valid neighbours. No-op (returns the input array) if disabled.
    """
    if not enabled():
        return depth_m
    import cv2  # local import: keep module import cheap / numpy-only when unused

    d = depth_m.astype(np.float32, copy=True)
    if d.ndim == 3:
        d2 = d[..., 0]
    else:
        d2 = d
    valid0 = d2 > 0.0

    # (1) median — kill isolated flying-pixels, keep edges/holes.
    k = _MED_K
    if k >= 3:
        if k % 2 == 0:
            k += 1
        k = min(k, 5)                       # cv2 float32 medianBlur supports 3/5 only
        f = cv2.medianBlur(d2, k)
        d2 = np.where(valid0 & (f > 0.0), f, d2)

    # (2) bilateral — smooth surface jitter, preserve depth steps. Weight-correct by
    # the validity mask so zeros in the window don't pull valid pixels toward 0.
    if _BI_D >= 1:
        wmask = valid0.astype(np.float32)
        wf = cv2.bilateralFilter(d2 * wmask, _BI_D, _BI_SIGMA_COLOR_M, _BI_SIGMA_SPACE)
        ww = cv2.bilateralFilter(wmask, _BI_D, _BI_SIGMA_COLOR_M, _BI_SIGMA_SPACE)
        corrected = np.where(ww > 1e-6, wf / np.maximum(ww, 1e-6), d2)
        d2 = np.where(valid0, corrected, 0.0)

    if d.ndim == 3:
        d[..., 0] = d2
        return d
    return d2


# ---------------------------------------------------------------------------
# GPU path (torch) — sub-ms on the depth tensor that's already on CUDA, so the
# live tracker tick doesn't pay the ~60 ms/frame CPU cost. Numerically matches
# the cv2 CPU path (median 3/5, weight-corrected bilateral) within depth-quant.
# ---------------------------------------------------------------------------
def filter_depth_torch(depth_t, *, median: bool = True, bilateral: bool = True):
    """Median and/or bilateral filter a depth tensor (H,W) or (H,W,1) float, metres,
    0 = invalid. Runs on the tensor's own device (GPU if it's a CUDA tensor).
    Returns a new tensor; same shape/dtype/device. No-op if disabled.

    The two stages are independently selectable so the TRACKER can run
    median-only (kills flying-pixels → bad RANSAC correspondences; ~7.6 ms) while
    FEEDFORWARD runs the full median+bilateral (also smooths surface jitter for
    clean inserts; off-thread so the extra ~8.5 ms bilateral is free). Pass
    ``median=False, bilateral=True`` to add ONLY the bilateral on top of an
    already-median'd depth (FF, to avoid a double-median).
    """
    if not enabled():
        return depth_t
    import torch
    import torch.nn.functional as F

    squeeze_last = (depth_t.dim() == 3 and depth_t.shape[-1] == 1)
    d = depth_t[..., 0] if squeeze_last else depth_t
    d = d.float()
    H, W = d.shape
    x = d.view(1, 1, H, W)
    valid0 = (x > 0.0).float()

    # (1) median over a k×k window, holes held out. cv2 supports 3/5; clamp to 5.
    k = max(0, min(_MED_K, 5))
    if median and k >= 3:
        if k % 2 == 0:
            k += 1
        pad = k // 2
        # Replicate cv2.medianBlur EXACTLY: median over the FULL k×k window with
        # invalid pixels counted as their stored 0 (NOT held out), then restore
        # holes afterwards. cv2 medians the raw window including the 0s; the lower-
        # median element is taken for the even case. (Differs from a valid-only
        # median only at true depth edges where the window is ~half near / half far —
        # both picks are real surface, not a flying-pixel; we match cv2 so the GPU
        # path is identical to the A/B'd CPU result.)
        cols = F.unfold(x, kernel_size=k, padding=pad)               # (1, k*k, HW)
        srt, _ = cols.sort(dim=1)
        kk = srt.shape[1]
        med = srt[:, (kk - 1) // 2, :].view(1, 1, H, W)              # lower-median (cv2)
        x = torch.where((valid0 > 0.5) & (med > 0.0), med, x)        # restore holes/keep raw where median fell on a 0

    # (2) weight-corrected bilateral over a d×d window. Spatial + range gaussian,
    # validity folded into the weight so holes never bleed (matches the cv2 path).
    dia = max(1, _BI_D)
    if bilateral and dia >= 2:
        if dia % 2 == 0:
            dia += 1
        pad = dia // 2
        valid = (x > 0.0).float()
        # spatial gaussian (precomputed on the kernel grid)
        ax = torch.arange(dia, device=d.device, dtype=torch.float32) - pad
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        sp = torch.exp(-(xx * xx + yy * yy) / (2.0 * _BI_SIGMA_SPACE ** 2)).reshape(1, dia * dia, 1)
        cols = F.unfold(x, kernel_size=dia, padding=pad)             # (1, d*d, H*W)
        vcols = F.unfold(valid, kernel_size=dia, padding=pad)
        center = x.view(1, 1, H * W)
        rng = torch.exp(-((cols - center) ** 2) / (2.0 * _BI_SIGMA_COLOR_M ** 2))
        w = sp * rng * vcols                                         # zero weight on invalid neighbours
        num = (w * cols).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1e-6)
        bil = (num / den).view(1, 1, H, W)
        x = torch.where(valid > 0.5, bil, torch.zeros_like(x))

    out = x.view(H, W)
    if squeeze_last:
        out = out.unsqueeze(-1)
    return out.to(depth_t.dtype)
