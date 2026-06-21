"""Per-resolution train-step timing probe (fp32). Loads the existing static scene and times N
warm steps at downscale 4 (1/4), 2 (1/2), 1 (full) to see exactly where the static-train cost is.
Run: python -m dynamic_gs2.verify._probe_train_step_time [dataset_dir]"""
import sys, threading, time
from pathlib import Path
import torch
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

# Build the optimizer once (same as train_static does).
opt = static_train._build_optimizers(sm, 0.0)
sm.enforce_phase_lr()
n = len(cams)
print(f"frames={n} res={tuple(batches[0]['image'].shape[:2])}")

def time_at(downscale_step, label, iters=30):
    """downscale_step picks the model.step that yields the wanted downscale under nd=2/rs=50:
       step 0->1/4, step 50->1/2, step 100->full."""
    sm.model.train()
    times = []
    for k in range(iters + 5):              # 5 warmup
        i = k % n
        sm.model.step = downscale_step
        opt.zero_grad_all()
        out = sm.model.get_outputs(cams[i])
        loss = sum(sm.get_loss_dict(out, batches[i]).values())
        loss.backward()
        opt.optimizer_step_all()
        torch.cuda.synchronize()
        t0 = time.time()
        if k >= 5:
            times.append(0.0)               # placeholder; real timing below
    # real timed pass
    times = []
    for k in range(iters):
        i = k % n
        sm.model.step = downscale_step
        opt.zero_grad_all()
        torch.cuda.synchronize(); t0 = time.time()
        out = sm.model.get_outputs(cams[i])
        loss = sum(sm.get_loss_dict(out, batches[i]).values())
        loss.backward()
        opt.optimizer_step_all()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    ms = sorted(times)[len(times) // 2] * 1000
    print(f"{label:12s} downscale={sm.model._get_downscale_factor()}  median={ms:6.1f} ms/step")
    return ms

q = time_at(0,   "1/4 res")
h = time_at(50,  "1/2 res")
f = time_at(100, "full res")
print(f"\nSCHEDULE COST (50/50/rest, early-stop@352):")
print(f"  50*{q:.0f} + 50*{h:.0f} + 252*{f:.0f} ms = {(50*q + 50*h + 252*f)/1000:.1f} s")
print(f"  OLD (100 half + 252 full) = {(100*h + 252*f)/1000:.1f} s")
