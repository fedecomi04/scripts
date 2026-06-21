"""Isolation: GOOD capture (162315), train with (A) its shipped seed vs (B) the new CPU SweepSeedBuilder
seed. Same RGB/poses. If B collapses opacity too -> my CPU seed path is the bug. Else -> the 173849
capture is the problem. Run: python -m dynamic_gs2.verify._probe_isolate_seedpath <dataset_dir> <seed_ply>"""
import sys, threading, torch
from pathlib import Path
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
DS=Path(sys.argv[1]); SEED=Path(sys.argv[2]); cfg=C.load_runtime_config(); dev="cuda"
sx,sr=static_fuse.load_seed_ply(SEED)
lock=threading.RLock()
sm=SceneModel(cfg,dev,seed_xyz=sx,seed_rgb=sr,phase="static",num_downscales=2,resolution_schedule=50)
sm.attach_render_lock(lock); g=GaussianSet(sm,lock,freelist=False)
cams,b=static_fuse.load_static_cameras(DS/"static_scene",dev)
dmax=float(cfg.depth.scene_depth_max_m)
def dk(x):
    d=x.get("depth_image")
    if dmax<=0 or d is None: return None
    d=d.to(dev).float(); d=d[...,None] if d.ndim==2 else d
    return ((d>0.05)&(d<dmax)).float()
sm.set_mask_provider(dk)
print(f"seed N={len(sx):,}")
static_train.train_static(sm,g,cams,b,num_steps=500,means_lr=0.0,mixed_precision=False,early_stop_loss=0.0)
op=torch.sigmoid(sm.model.gauss_params["opacities"].detach().float().squeeze())
print(f"RESULT end opacity median={op.median():.3f}  >0.5={(op>0.5).float().mean()*100:.0f}%  <0.05(purged)={(op<0.05).float().mean()*100:.0f}%")
