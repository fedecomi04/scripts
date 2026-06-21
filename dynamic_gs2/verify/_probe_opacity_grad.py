"""Does opacity actually train? Track median sigmoid(opacity) at step 0 vs end + grad norm.
Run: python -m dynamic_gs2.verify._probe_opacity_grad <dataset_dir>"""
import sys, threading
import numpy as np, torch
from pathlib import Path
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path
DS=Path(sys.argv[1]); cfg=C.load_runtime_config(); dev="cuda"
sx,sr=static_fuse.load_seed_ply(seed_ply_path(DS))
lock=threading.RLock()
sm=SceneModel(cfg,dev,seed_xyz=sx,seed_rgb=sr,phase="static",num_downscales=2,resolution_schedule=50)
sm.attach_render_lock(lock); gset=GaussianSet(sm,lock,freelist=False)
cams,batches=static_fuse.load_static_cameras(DS/"static_scene",dev)
dmax=float(cfg.depth.scene_depth_max_m)
def dk(b):
    d=b.get("depth_image")
    if dmax<=0 or d is None: return None
    d=d.to(dev).float(); d=d[...,None] if d.ndim==2 else d
    return ((d>0.05)&(d<dmax)).float()
sm.set_mask_provider(dk)
op0=torch.sigmoid(sm.model.gauss_params["opacities"].detach().float().squeeze())
print(f"step0 opacity median={op0.median():.3f}")
static_train.train_static(sm,gset,cams,batches,num_steps=500,means_lr=0.0,mixed_precision=False,early_stop_loss=0.0)
op1=torch.sigmoid(sm.model.gauss_params["opacities"].detach().float().squeeze())
print(f"END   opacity median={op1.median():.3f}  >0.5={(op1>0.5).float().mean()*100:.0f}%  <0.05={(op1<0.05).float().mean()*100:.0f}%")
