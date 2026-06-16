"""Re-noise a dataset's static depth TIFFs to the CURRENT zed_depth_noise model,
then rebuild the TSDF seed point cloud.

The depth on disk already carries the OLD baked noise (sigma_old = 0.05 + 0.477 z^2 mm).
We RAISE it to the current model (sigma_new = SIGMA0 + K z^2, read live from
zed_depth_noise.py) by adding a layer sigma_add = sqrt(max(sigma_new^2 - sigma_old^2, 0))
per valid pixel — total becomes sigma_new (variances add). Pixels where the new
model is BELOW the baked level are left unchanged (can't remove baked noise).

Backs up original depth/ + seed PLY to static_scene/_orig_depth_backup/ first.
CPU-only TSDF rebuild (avoids the 1200p GPU OOM).

Usage: renoise_static_depth.py <data_dir>
"""
import json, os, shutil, sys, glob, importlib.util
from pathlib import Path
import numpy as np, cv2

DATA = Path(sys.argv[1]).resolve()
STATIC = DATA / "static_scene"
Z_MIN, Z_MAX = 0.05, 2.0
# The OLD baked-noise constants (what these TIFFs were captured with).
S0_OLD, K_OLD = 0.00005, 0.000477

# Load the CURRENT model live (so we always re-noise to today's default).
spec = importlib.util.spec_from_file_location(
    "_zed", str(Path(__file__).resolve().parent.parent / "dynamic_gs/utils/zed_depth_noise.py"))
zed = importlib.util.module_from_spec(spec); spec.loader.exec_module(zed)
S0_NEW, K_NEW = zed._SIGMA0, zed._K
print(f"[renoise] OLD baked sigma = {S0_OLD*1000:.3f} + {K_OLD*1e6:.3f} z^2 mm")
print(f"[renoise] NEW target sigma = {S0_NEW*1000:.3f} + {K_NEW*1e6:.3f} z^2 mm  (live from zed_depth_noise.py)")

# --- backup ---
bk = STATIC / "_orig_depth_backup"
if bk.exists():
    print(f"[renoise] backup already exists at {bk} — refusing to overwrite it (originals are safe). "
          f"Re-noising from CURRENT on-disk depth would double-apply; aborting.")
    sys.exit(1)
bk.mkdir()
shutil.copytree(STATIC / "depth", bk / "depth")
seed = STATIC / "depth_camera_init_points.ply"
if seed.exists():
    shutil.copy2(seed, bk / seed.name)
print(f"[renoise] backed up original depth/ ({len(list((bk/'depth').glob('*')))} files) + seed -> {bk}")

# --- re-noise each TIFF in place ---
rng = np.random.default_rng(12345)
tiffs = sorted(glob.glob(str(STATIC / "depth" / "*.tiff")))
n_changed = 0
for f in tiffs:
    dmm = cv2.imread(f, cv2.IMREAD_UNCHANGED)          # uint16 mm
    d = dmm.astype(np.float32) / 1000.0
    valid = (d > Z_MIN) & (d < Z_MAX)
    s_old = S0_OLD + K_OLD * d * d
    s_new = S0_NEW + K_NEW * d * d
    s_add = np.sqrt(np.clip(s_new * s_new - s_old * s_old, 0.0, None))
    noise = rng.normal(0.0, 1.0, size=d.shape).astype(np.float32) * s_add
    d_out = np.where(valid, d + noise, d)
    out_mm = np.clip(d_out * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(f, out_mm)                              # TIFF, lossless uint16
    n_changed += 1
print(f"[renoise] re-noised {n_changed} depth TIFFs in place")

# --- rebuild seed (CPU TSDF) ---
os.environ["DGS_FUSION_DEVICE"] = "cpu"
os.environ.setdefault("DGS_TSDF_DEPTH_MAX_M", "2.0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamic_gs.utils.online_fusion import fuse_recorded_dataset
print("[renoise] rebuilding TSDF seed on CPU (2mm voxel, 2m cap)...")
out_ply = fuse_recorded_dataset(STATIC)
print(f"[renoise] seed rebuilt -> {out_ply}")
