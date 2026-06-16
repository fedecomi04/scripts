"""ZED-X depth-error model for sim-to-real realism on Gazebo sim depth.

The Gazebo `libgazebo_ros_openni_kinect.so` depth sensor returns the GPU
z-buffer essentially noise-free (its `<noise>` block applies to RGB, not depth).
That perfect depth is the sim-to-real gap this module closes: it corrupts clean
sim depth so the pipeline sees ZED-X-realistic per-pixel depth jitter.

Model: per-pixel AXIAL noise, zero-mean Gaussian, sigma growing with range as
the stereo physics predicts (sigma_z proportional to z^2, from Z = f*B/disparity
=> dZ = (z^2 / (f*B)) * d_disp):

        sigma_z(z) = sigma0 + k * z^2      [metres, z in metres]

**Calibrated to our ACTUAL ZED-X**, not a generic paper. Measured per-pixel
roughness on dataset ZED/zed_validate2 (HD1200 NEURAL conf=50): residual std to
a local plane fit over 41,927 flat 9x9 windows across 30 frames:
    0.4m: 0.29mm   0.6m: 0.32mm   0.8m: 0.39mm   1.0m: 0.48mm
    1.25m: 0.44mm  1.55m: 0.73mm  1.9m: 2.15mm
Least-squares fit -> sigma0 ~= 0 (we keep a tiny 0.05mm quantization floor),
k = 0.477 mm/m^2. So sigma is SUB-MILLIMETRE out to ~1.5m and ~1.7mm at 1.9m.

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

# Per-pixel axial-noise fit to the real ZED-X (ZED/zed_validate2, see module doc):
#   sigma_z(z) = SIGMA0 + K * z^2   [metres]
_SIGMA0 = float(os.environ.get("DGS_SIM_ZED_SIGMA0_M", "0.00005"))  # 0.05 mm quantization floor
_K = float(os.environ.get("DGS_SIM_ZED_K_M", "0.000477"))          # 0.477 mm/m^2 (measured)
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
