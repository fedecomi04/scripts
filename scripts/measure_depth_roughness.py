#!/usr/bin/env python3
"""Measure per-pixel depth roughness vs depth to verify the ZED noise model.

Samples many small flat windows, back-projects each to 3D, fits a local plane
(SVD), and takes the point-to-plane residual std = physical roughness (the SAME
metric zed_depth_noise.py was calibrated with). Bins by depth and compares to
the model σ_z(z)=SIGMA0+K·z² and the uint16-mm quantization floor (~0.29mm).

If roughness GROWS as z² toward the table → noise applied + follows the model.
If FLAT ~0.29mm at all depths → noise NOT applied (only disk quantization).

Usage: python scripts/measure_depth_roughness.py <data_dir> [n_frames] [n_windows_per_frame]
"""
import json, sys, re
from pathlib import Path
import cv2, numpy as np

SIGMA0, K = 0.00005, 0.000477          # the model's constants
WIN = 9                                # 9x9 windows (matches calibration)
FLAT_RESID_MAX_M = 0.004               # reject edge windows: residual > 4mm = not flat


def main():
    static = Path(sys.argv[1]).resolve()
    if static.name != "static_scene":
        static = static / "static_scene"
    nfr = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    nwin = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    tf = json.loads((static / "transforms.json").read_text())
    fx, fy, cx, cy = (float(tf[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    frames = tf["frames"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(frames), size=min(nfr, len(frames)), replace=False)

    # (depth, roughness) samples
    zs, rough = [], []
    for i in idx:
        dmm = cv2.imread(str(static / frames[i]["depth_file_path"].replace("./", "")),
                         cv2.IMREAD_UNCHANGED)
        if dmm is None:
            continue
        H, W = dmm.shape
        dm = dmm.astype(np.float32) / 1000.0
        for _ in range(nwin):
            r = rng.integers(0, H - WIN); c = rng.integers(0, W - WIN)
            patch = dm[r:r + WIN, c:c + WIN]
            if (patch <= 0.01).any():        # need a fully-valid window
                continue
            uu, vv = np.meshgrid(np.arange(c, c + WIN), np.arange(r, r + WIN))
            z = patch.ravel()
            x = (uu.ravel() - cx) * z / fx
            y = (vv.ravel() - cy) * z / fy
            P = np.stack([x, y, z], 1)
            Pc = P - P.mean(0)
            # plane normal = smallest singular vector; residual = |Pc·n|
            _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
            resid = Pc @ Vt[2]
            rstd = float(resid.std())
            if rstd > FLAT_RESID_MAX_M:       # edge / non-flat window → skip
                continue
            zs.append(float(z.mean())); rough.append(rstd)

    zs, rough = np.array(zs), np.array(rough)
    print(f"[roughness] {static.parent.name}: {len(zs)} flat windows over {len(idx)} frames")
    print(f"  quantization floor (uint16 mm rounding) ≈ 0.29 mm")
    print(f"  {'z bin':>12} {'n':>6} {'measured':>10} {'model σ_z':>10}")
    edges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 3.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (zs >= lo) & (zs < hi)
        if m.sum() < 5:
            continue
        zmid = zs[m].mean()
        meas = rough[m].mean() * 1000.0
        model = (SIGMA0 + K * zmid * zmid) * 1000.0
        print(f"  [{lo:.1f},{hi:.1f})m {m.sum():>6} {meas:>8.2f}mm {model:>8.2f}mm")


if __name__ == "__main__":
    main()
