"""Is the static scene blurry because it's undertrained, or because the viewer downsamples?
Loads the saved static_state.pt scene OR retrains, renders at FULL 1920x1200, reports masked PSNR
per keyframe + dumps a full-res render PNG so we can eyeball trained quality (not the 960x600 viewer).
Run: python -m dynamic_gs2.verify._probe_static_quality [dataset_dir]"""
import sys, threading
from pathlib import Path
import numpy as np, torch, cv2
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_162315_live").resolve()
cfg = C.load_runtime_config(); dev = "cuda"

seed_xyz, seed_rgb = static_fuse.load_seed_ply(seed_ply_path(DS))
lock = threading.RLock()
sm = SceneModel(cfg, dev, seed_xyz=seed_xyz, seed_rgb=seed_rgb, phase="static",
                num_downscales=2, resolution_schedule=50)
sm.attach_render_lock(lock)
gset = GaussianSet(sm, lock, freelist=False)
cams, batches = static_fuse.load_static_cameras(DS / "static_scene", dev)
dmax = float(cfg.depth.scene_depth_max_m)
def _dk(b):
    d = b.get("depth_image")
    if dmax <= 0 or d is None: return None
    d = d.to(dev).float(); d = d[..., None] if d.ndim == 2 else d
    return ((d > 0.05) & (d < dmax)).float()
sm.set_mask_provider(_dk)

n = int(sys.argv[2]) if len(sys.argv) > 2 else 352
print(f"training {n} steps (nd=2/rs=50, fp32, early-stop OFF)...")
static_train.train_static(sm, gset, cams, batches, num_steps=n, means_lr=0.0,
                          mixed_precision=False, early_stop_loss=0.0)

sm.model.step = 30000; sm.model.eval()       # full SH, downscale=1 (eval) => FULL res render
psnrs = []
for k, (cam, b) in enumerate(zip(cams, batches)):
    with lock, torch.no_grad():
        rgb, _, _ = sm.render(cam)
    gt = b["image"].to(dev)
    if gt.shape != rgb.shape: continue
    m = _dk(b)
    if m is not None:
        m = m.expand_as(rgb)
        mse = (((rgb - gt) ** 2) * m).sum() / m.sum().clamp_min(1)
    else:
        mse = ((rgb - gt) ** 2).mean()
    psnrs.append(float(-10 * torch.log10(mse.clamp_min(1e-8))))
    if k == 0:
        out = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        p = DS / "_probe_fullres_render.png"; cv2.imwrite(str(p), out)
        print(f"full-res render WxH={out.shape[1]}x{out.shape[0]} -> {p}")
print(f"masked PSNR: mean={np.mean(psnrs):.2f}dB  min={np.min(psnrs):.2f}  max={np.max(psnrs):.2f}  over {len(psnrs)} kf  N={gset.num_points}")
