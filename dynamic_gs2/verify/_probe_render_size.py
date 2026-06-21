"""Measure viser-direct render cost at 960x600 vs 1280x800 vs 1920x1200 on the trained static scene,
so we know the per-frame render budget hit from the viewer-res bump. Builds a camera at each size and
times model.get_outputs (the rasterize the render thread does). Run:
python -m dynamic_gs2.verify._probe_render_size [dataset_dir]"""
import sys, threading, time
from pathlib import Path
import torch
from dynamic_gs2 import config as C, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "../data_teleoperation/datasets/2026-06-21_162315_live").resolve()
cfg = C.load_runtime_config(); dev = "cuda"
seed_xyz, seed_rgb = static_fuse.load_seed_ply(seed_ply_path(DS))
lock = threading.RLock()
sm = SceneModel(cfg, dev, seed_xyz=seed_xyz, seed_rgb=seed_rgb, phase="static")
sm.attach_render_lock(lock)
gset = GaussianSet(sm, lock, freelist=False)
cams, batches = static_fuse.load_static_cameras(DS / "static_scene", dev)
sm.model.step = 30000; sm.model.eval()
base = cams[0]

def cam_at(W, H):
    """Rescale the first keyframe camera to WxH (fx/fy/cx/cy scale with the image)."""
    from copy import deepcopy
    c = deepcopy(base)
    sw = W / float(c.width.item()); sh = H / float(c.height.item())
    c.fx = c.fx * sw; c.fy = c.fy * sh; c.cx = c.cx * sw; c.cy = c.cy * sh
    c.width = torch.tensor([[W]]); c.height = torch.tensor([[H]])
    return c.to(dev)

for (W, H) in [(960, 600), (1280, 800), (1920, 1200)]:
    c = cam_at(W, H)
    for _ in range(5):  # warmup
        with lock, torch.no_grad(): sm.model.get_outputs(c)
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.time()
        with lock, torch.no_grad(): sm.model.get_outputs(c)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    ms = sorted(ts)[len(ts) // 2] * 1000
    print(f"{W}x{H}: median render = {ms:6.1f} ms  ({1000/ms:.0f} fps render-bound)")
