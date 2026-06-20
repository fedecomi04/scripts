"""Probe native-train scene quality vs config knobs: render PSNR over ALL static keyframes after
500 steps, under variants (scale-reset on/off, fp16 on/off). Isolates the train-quality regression."""
import sys, json, re, threading
from pathlib import Path
import numpy as np, torch, cv2
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path

DS=Path("../data_teleoperation/datasets/screwdriver recorded full").resolve()
cfg=C.load_runtime_config(); dev="cuda"
variant = sys.argv[1] if len(sys.argv)>1 else "baseline"

seed_xyz, seed_rgb = static_fuse.load_seed_ply(seed_ply_path(DS))
lock=threading.RLock()
sm=SceneModel(cfg,dev,seed_xyz=seed_xyz,seed_rgb=seed_rgb,phase="static",num_downscales=1,resolution_schedule=100)
sm.attach_render_lock(lock)
gset=GaussianSet(sm,lock,freelist=False)
cams,batches=static_fuse.load_static_cameras(DS/"static_scene",dev)
# depth-keep mask
dmax=float(cfg.depth.scene_depth_max_m)
def _dk(b):
    d=b.get("depth_image")
    if dmax<=0 or d is None: return None
    d=d.to(dev).float(); d=d[...,None] if d.ndim==2 else d
    return ((d>0.05)&(d<dmax)).float()
sm.set_mask_provider(_dk)

kw=dict(num_steps=500,means_lr=0.0,early_stop_loss=0.0)  # disable early-stop to always run 500
if variant=="no_scalereset": kw["scale_clamp_every_n"]=0
if variant=="fp32": kw["mixed_precision"]=False
if variant=="no_scalereset_fp32": kw.update(scale_clamp_every_n=0, mixed_precision=False)
static_train.train_static(sm,gset,cams,batches,**kw)

# PSNR over all keyframes
sm.model.step=30000; sm.model.eval()
psnrs=[]
for cam,b in zip(cams,batches):
    with lock,torch.no_grad(): rgb,_,_=sm.render(cam)
    gt=b["image"].to(dev)  # already [0,1] RGB
    if gt.shape!=rgb.shape: continue
    mse=((rgb-gt)**2).mean(); psnrs.append(float(-10*torch.log10(mse)))
sc=torch.exp(gset.snapshot().params["scales"]).max(1).values
print(f"VARIANT={variant}: mean PSNR={np.mean(psnrs):.2f}dB over {len(psnrs)} kf | maxscale={float(sc.max())*100:.1f}cm N={gset.num_points}")
