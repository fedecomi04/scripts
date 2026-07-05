"""Does CPU-TSDF-NO-ICP (integrate at FK poses) fix the opacity collapse? Build seed both ways on the
bad data, train, report end opacity. Run: python -m dynamic_gs2.verify._probe_noicp_fix <dataset_dir>"""
import sys, threading, os, json, re
import numpy as np, cv2, torch, open3d as o3d
from pathlib import Path
os.environ["DGS_FUSION_DEVICE"]="cpu"; os.environ.setdefault("DGS_TSDF_VOXEL_M","0.003")
import dynamic_gs2.online_fusion as OF
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
DS=Path(sys.argv[1]); sd=DS/"static_scene"
cfg=C.load_runtime_config(); dev="cuda"
meta=json.loads((sd/"transforms.json").read_text())
fx,fy,cx,cy=float(meta["fl_x"]),float(meta["fl_y"]),float(meta["cx"]),float(meta["cy"])
W,H=int(meta["w"]),int(meta["h"])
frames=sorted(meta["frames"],key=lambda fr:int(re.findall(r"\d+",Path(fr["file_path"]).name)[-1]))
def load(fr):
    d=cv2.imread(str(sd/fr["depth_file_path"].lstrip("./")),cv2.IMREAD_UNCHANGED).astype(np.uint16).copy()
    mp=fr.get("mask_path") or fr.get("mask_file_path")
    if mp:
        m=cv2.imread(str(sd/mp.lstrip("./")),cv2.IMREAD_GRAYSCALE)
        if m is not None: d[m==0]=0
    return d,np.asarray(fr["transform_matrix"],dtype=np.float64)
loaded=[load(fr) for fr in frames]

def build(noicp):
    OF.ICP_SRC_STRIDE=8
    f=OF.OnlineFusion(fx,fy,cx,cy,W,H)
    for d,c2w in loaded:
        if noicp:
            cv=OF.OnlineFusion._cv_c2w(c2w); f._impl._integrate(d,None,cv); f._impl.idx+=1
        else:
            f.add_frame(d,c2w,None)
    pc=f.finalize()
    pc=OF.adaptive_downsample(pc, OF.OnlineFusion._cv_c2w(loaded[-1][1])[:3,3])
    p=sd/"_probe_seed.ply"; o3d.io.write_point_cloud(str(p),pc); return p

def train_seed(ply):
    sx,sr=static_fuse.load_seed_ply(ply)
    lock=threading.RLock()
    sm=SceneModel(cfg,dev,seed_xyz=sx,seed_rgb=sr,phase="static",num_downscales=2,resolution_schedule=50)
    sm.attach_render_lock(lock); g=GaussianSet(sm,lock,freelist=False)
    cams,batches=static_fuse.load_static_cameras(sd,dev)
    dmax=float(cfg.depth.scene_depth_max_m)
    def dk(b):
        dd=b.get("depth_image")
        if dmax<=0 or dd is None: return None
        dd=dd.to(dev).float(); dd=dd[...,None] if dd.ndim==2 else dd
        return ((dd>0.05)&(dd<dmax)).float()
    sm.set_mask_provider(dk)
    static_train.train_static(sm,g,cams,batches,num_steps=500,means_lr=0.0,mixed_precision=False,early_stop_loss=0.0)
    op=torch.sigmoid(sm.model.gauss_params["opacities"].detach().float().squeeze())
    return op.median().item(),(op>0.5).float().mean().item()*100,(op<0.05).float().mean().item()*100

for noicp,lbl in [(False,"CPU-WITH-ICP (current/buggy)"),(True,"CPU-NO-ICP (FK poses)")]:
    ply=build(noicp); med,hi,lo=train_seed(ply)
    print(f"{lbl:30s}: end opacity median={med:.3f}  >0.5={hi:.0f}%  <0.05(purged)={lo:.0f}%")
