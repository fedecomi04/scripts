"""ZED-X depth-error model for sim-to-real realism on Gazebo sim depth.

The Gazebo `libgazebo_ros_openni_kinect.so` depth sensor returns the GPU
z-buffer essentially noise-free (its `<noise>` block applies to RGB, not depth).
That perfect depth is the sim-to-real gap this module closes: it corrupts clean
sim depth so the pipeline sees ZED-X-realistic per-pixel depth jitter.

Model: per-pixel AXIAL noise, zero-mean Gaussian, sigma growing with range as
the stereo physics predicts (sigma_z proportional to z^2, from Z = f*B/disparity
=> dZ = (z^2 / (f*B)) * d_disp):

        sigma_z(z) = sigma0 + k * z^2      [metres, z in metres]

**Calibrated to our ACTUAL ZED-X as an UPPER BOUND**, not a generic paper, and
deliberately tuned to the NOISIEST real capture so the sim trains the pipeline
against worst-case sensor noise (robustness). Measured per-pixel roughness
(residual std to a local plane fit over flat 9x9 windows, the SAME metric) on
three real HD1200 NEURAL captures showed the rough one, ZED/scene_dataset, runs
~2-3x noisier than ZED/zed_validate2:
    scene_dataset mean  : 0.4m 0.50mm  0.6m 0.72mm  1.0m 1.25mm  1.4m 1.94mm  2.4m 2.49mm
    scene_dataset p90   : 0.4m 0.84mm  0.6m 1.26mm  1.0m 2.20mm  1.4m 3.25mm  2.4m 3.66mm
    zed_validate2 mean  : 0.4m 0.29mm  0.6m 0.32mm  1.0m 0.48mm  1.6m 0.81mm  2.4m 1.15mm
Least-squares z^2 fit to the scene_dataset **p90** (the upper envelope) ->
sigma0 = 1.47mm (a FLAT term so noise is visible even at near range, unlike the
old pure-z^2 fit which was invisible <1.5m), k = 0.500 mm/m^2. So sigma is
~1.5mm at 0.3m, ~2.0mm at 1.0m, ~3mm at 1.75m.
*History: the first real fit (2026-06-15) was the zed_validate2 MEAN -> sigma0~=0,
k=0.477; that was the typical-case low end and near-invisible at tabletop range.
Switched 2026-06-16 to the scene_dataset p90 upper bound for robustness.*

WHY NOT the Ortiz et al. 2018 exponential f(Z)=a*exp(b*Z): that is the paper's
WHOLE-FRAME RMS *error* (bias + calibration + edge/flying-pixel effects averaged
over a checkerboard at range), NOT clean-surface per-pixel jitter. Sampling it
per-pixel injected ~12mm sigma at 0.5m -> ~5cm visible point spread, ~40x noisier
than the real sensor. Replaced 2026-06-15 after measuring the real camera.

NOT modeled (deliberate): edge/discontinuity flying-pixels (the real sensor IS
rough there, but on flat surfaces it is sub-mm as measured). Holes: a small
random null-pixel rate. Range gate [0.05, 3.0] m matches online_fusion.py
DEPTH_MIN_M/DEPTH_MAX_M (depth outside it nulled, as the camera doesn't deliver
reliable depth there).

All knobs are env-overridable for A/B. The model is ON by default (it is the
realistic sim default); set DGS_SIM_ZED_NOISE=0 to disable (clean sim depth).
"""

import os
import numpy as np

# Per-pixel axial-noise upper-bound fit to the real ZED-X (ZED/scene_dataset p90,
# see module doc):  sigma_z(z) = SIGMA0 + K * z^2   [metres]
_SIGMA0 = float(os.environ.get("DGS_SIM_ZED_SIGMA0_M", "0.00147"))  # 1.47 mm flat term (near-range visible)
_K = float(os.environ.get("DGS_SIM_ZED_K_M", "0.000500"))          # 0.500 mm/m^2 (p90 upper bound)
# Random invalid/null-pixel rate (clean-surface holes; edges add more, not modeled).
_HOLE_RATE = float(os.environ.get("DGS_SIM_ZED_HOLE_RATE", "0.01"))
# Range gate (metres) — matches online_fusion.py DEPTH_MIN_M / DEPTH_MAX_M
_Z_MIN = float(os.environ.get("DGS_SIM_ZED_Z_MIN", "0.05"))
_Z_MAX = float(os.environ.get("DGS_SIM_ZED_Z_MAX", "3.0"))


def enabled() -> bool:
    """True iff the sim ZED depth-noise model should be applied.

    ON by default (the measured ZED-X model is the realistic sim default);
    set ``DGS_SIM_ZED_NOISE=0`` to disable (clean sim depth)."""
    return os.environ.get("DGS_SIM_ZED_NOISE", "1") != "0"


def sigma_z(z):
    """Per-pixel axial noise std (metres) at range z (metres)."""
    return _SIGMA0 + _K * z * z


def apply_zed_depth_noise(depth_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt clean sim depth (float32 metres, 0 = invalid) into ZED-realistic depth.

    Returns a new array; does not mutate the input. `rng` is passed in so the
    caller owns the stream (reproducible / per-publisher seeding).
    """
    out = depth_m.astype(np.float32, copy=True)
    valid = out > 0.0

    # (1) Axial noise: per-pixel sigma = sigma0 + k*z^2, zero-mean Gaussian.
    z = out[valid]
    sigma = sigma_z(z)
    out[valid] = z + rng.normal(0.0, 1.0, size=z.shape).astype(np.float32) * sigma

    # (2) Range gate — null depth outside the camera's reliable band.
    out[(out > 0.0) & ((out < _Z_MIN) | (out > _Z_MAX))] = 0.0

    # (3) Random holes — null a fraction of the still-valid pixels.
    if _HOLE_RATE > 0.0:
        still_valid = out > 0.0
        holes = rng.random(out.shape, dtype=np.float32) < _HOLE_RATE
        out[still_valid & holes] = 0.0

    return out
