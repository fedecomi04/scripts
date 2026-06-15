"""ZED-X depth-error model for sim-to-real realism on Gazebo sim depth.

The Gazebo `libgazebo_ros_openni_kinect.so` depth sensor returns the GPU
z-buffer essentially noise-free (its `<noise>` block applies to RGB, not depth).
That perfect depth is the sim-to-real gap this module closes: it corrupts clean
sim depth so the pipeline sees ZED-X-realistic error.

Model (paper-faithful):
  Ortiz, Cabrera, Goncalves, "Depth Data Error Modeling of the ZED 3D Vision
  Sensor from Stereolabs", ELCVIA 17(1):1-15, 2018.

  - Axial (z) RMS error grows EXPONENTIALLY with range (their Eq. 8):
        f(Z) = a * exp(b * Z)        [metres, Z in metres]
    Fitted coefficients per resolution (Table 2):
        hd2k   (2208x1242): a=0.01805 b=0.1746  R^2=0.968
        hd1080 (1920x1080): a=0.0106  b=0.2215  R^2=0.96
        hd720  (1280x720):  a=0.0184  b=0.2106  R^2=0.964
        wvga   (672x376):   a=0.0115  b=0.2986  R^2=0.97
    Select via DGS_SIM_ZED_RES (default hd1080, the closest fit to our
    recorded HD1200 = 1920x1200).
  - Random invalid/null pixels (Section 4.1): hd2k 2.30%, hd1080 2.43%,
    hd720 2.06%, wvga 2.09% — selected with the same resolution key.

Range gate [0.05, 3.0] m matches the TSDF ingest band (online_fusion.py
DEPTH_MIN_M/DEPTH_MAX_M); depth outside it is nulled, consistent with the
camera not delivering reliable depth there.

NOT modeled (deliberately, per "paper-faithful" decision): edge/discontinuity
"flying pixels". The paper measured a flat checkerboard, so it has no edge term.

All knobs are env-overridable for A/B; the whole thing is OFF unless
DGS_SIM_ZED_NOISE=1.
"""

import os
import numpy as np

# --- Ortiz et al. 2018, per-resolution fits ---
# Exponential RMS-error coefficients (Table 2) + null-pixel rates (Section 4.1),
# for all four ZED resolutions. Pick a row via DGS_SIM_ZED_RES (default hd1080,
# closest to our recorded HD1200). a/b/hole_rate can still be overridden
# individually below.
#   key       resolution      a         b        null-rate   R^2
_ZED_RES = {
    "hd2k":   dict(a=0.01805, b=0.1746, hole=0.0230),  # 2208x1242, R^2=0.968
    "hd1080": dict(a=0.0106,  b=0.2215, hole=0.0243),  # 1920x1080, R^2=0.96
    "hd720":  dict(a=0.0184,  b=0.2106, hole=0.0206),  # 1280x720,  R^2=0.964
    "wvga":   dict(a=0.0115,  b=0.2986, hole=0.0209),  # 672x376,   R^2=0.97
}
_RES = os.environ.get("DGS_SIM_ZED_RES", "hd1080").lower()
if _RES not in _ZED_RES:
    raise ValueError(
        f"DGS_SIM_ZED_RES={_RES!r} not in {sorted(_ZED_RES)}"
    )
_FIT = _ZED_RES[_RES]

_A = float(os.environ.get("DGS_SIM_ZED_A", str(_FIT["a"])))
_B = float(os.environ.get("DGS_SIM_ZED_B", str(_FIT["b"])))
_HOLE_RATE = float(os.environ.get("DGS_SIM_ZED_HOLE_RATE", str(_FIT["hole"])))
# Range gate (metres) — matches online_fusion.py DEPTH_MIN_M / DEPTH_MAX_M
_Z_MIN = float(os.environ.get("DGS_SIM_ZED_Z_MIN", "0.05"))
_Z_MAX = float(os.environ.get("DGS_SIM_ZED_Z_MAX", "3.0"))


def enabled() -> bool:
    """True iff the sim ZED depth-noise model should be applied."""
    return os.environ.get("DGS_SIM_ZED_NOISE", "0") == "1"


def apply_zed_depth_noise(depth_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt clean sim depth (float32 metres, 0 = invalid) into ZED-realistic depth.

    Returns a new array; does not mutate the input. `rng` is passed in so the
    caller owns the stream (reproducible / per-publisher seeding).
    """
    out = depth_m.astype(np.float32, copy=True)
    valid = out > 0.0

    # (1) Axial exponential noise: per-pixel sigma = a * exp(b * z), zero-mean Gaussian.
    z = out[valid]
    sigma = _A * np.exp(_B * z)
    out[valid] = z + rng.normal(0.0, 1.0, size=z.shape).astype(np.float32) * sigma

    # (2) Range gate — null depth outside the camera's reliable band.
    out[(out > 0.0) & ((out < _Z_MIN) | (out > _Z_MAX))] = 0.0

    # (3) Random holes — null a fraction of the still-valid pixels.
    if _HOLE_RATE > 0.0:
        still_valid = out > 0.0
        holes = rng.random(out.shape, dtype=np.float32) < _HOLE_RATE
        out[still_valid & holes] = 0.0

    return out
