"""Per-method A/B: OLD StaticGSModel queries vs NATIVE static_phase0b queries on the SAME
trained scene + same render (info,depth) + same object mask. Asserts identical index SETS."""
import json, numpy as np, torch, threading
from pathlib import Path
import cv2
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

DS = Path("../data_teleoperation/datasets/screwdriver recorded full").resolve()
st = DS/"static_scene"
pt = st/"static_state.pt"
meta = json.loads((st/"transforms.json").read_text())
fx,fy,cx,cy = meta["fl_x"],meta["fl_y"],meta["cx"],meta["cy"]; W,H=int(meta["w"]),int(meta["h"])
import re
fr = sorted(meta["frames"], key=lambda f:int(re.findall(r"\d+",Path(f["file_path"]).name)[-1]))[-1]
c2w = torch.tensor(np.asarray(fr["transform_matrix"],dtype=np.float32)[:3,:4]).unsqueeze(0)
def mkcam(dev):
    cam = Cameras(camera_to_worlds=c2w, fx=fx,fy=fy,cx=cx,cy=cy,width=W,height=H,
                  camera_type=CameraType.PERSPECTIVE).to(dev)
    cam.metadata={"cam_idx":0}; return cam
dev="cuda"
blob = torch.load(pt, map_location="cpu", weights_only=False); sd=blob["model_state_dict"]
n = int(blob["num_points"])
means = sd["gauss_params.means"].float()
rgb = (sd["gauss_params.features_dc"].float().reshape(n,3)*0.28209479177387814+0.5).clamp(0,1)

# object mask
mask_np = np.array(cv2.imread(str(DS/"dynamic_scene/initialization_debug/static0_obj_00_mask.png"),
                              cv2.IMREAD_GRAYSCALE))
obj_mask = torch.from_numpy((mask_np>127).astype(np.float32))[...,None].to(dev)
if obj_mask.shape[0]!=H or obj_mask.shape[1]!=W:
    obj_mask = torch.nn.functional.interpolate(obj_mask.permute(2,0,1)[None], size=(H,W),
                                               mode="nearest")[0].permute(1,2,0)

# ---------- NATIVE ----------
from dynamic_gs2 import config as C, static_phase0b
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
cfg = C.load_runtime_config()
lock = threading.RLock()
sm = SceneModel(cfg, dev, seed_xyz=means, seed_rgb=rgb, phase="static"); sm.attach_render_lock(lock)
gset = GaussianSet(sm, lock, freelist=False); gset.reload_from_state_dict(sd, num_points=n)
cam = mkcam(dev)
with lock:
    out_n, info_n = sm.get_outputs_with_info(cam)
sub_n,_,_ = static_phase0b.get_existing_object_subset(gset, info_n, obj_mask, out_n["depth"])
slab_n_15 = static_phase0b.get_object_mask_slab_indices(gset, info_n, obj_mask, out_n["depth"], 0.015)
slab_n_02 = static_phase0b.get_object_mask_slab_indices(gset, info_n, obj_mask, out_n["depth"], 0.02)
e_pts_n = gset.snapshot().params["means"][slab_n_15].cpu().numpy().astype(np.float32)
spac_n = static_phase0b.estimate_spacing(e_pts_n) if len(e_pts_n)>=2 else -1.0
print(f"[native] subset={sub_n.numel()} slab015={slab_n_15.numel()} slab02={slab_n_02.numel()} spacing={spac_n:.6f}")

# ---------- OLD ----------
from dynamic_gs.static_gs_model import StaticGSModel, StaticGSModelConfig
ocfg = StaticGSModelConfig()
sb = SceneBox(aabb=torch.tensor([[-2.,-2,-2],[2.,2,2]]))
om = StaticGSModel(config=ocfg, scene_box=sb, num_train_data=len(meta["frames"]),
                   seed_points=(means.to(dev), rgb.to(dev))).to(dev)
om.optimizers={}; om.step=30000
# load the exact .pt (its load_state_dict resizes buffers)
om.load_state_dict(sd, strict=False)
om.eval()
ocam = mkcam(dev)
with torch.no_grad():
    oout = om.get_outputs(ocam)   # sets om.info
sub_o,_,_ = om._get_existing_object_subset(obj_mask, oout["depth"])
slab_o_15 = om._get_object_mask_slab_indices(obj_mask, oout["depth"], depth_tol_m=0.015)
slab_o_02 = om._get_object_mask_slab_indices(obj_mask, oout["depth"], depth_tol_m=0.02)
e_pts_o = om.means[slab_o_15].detach().cpu().numpy().astype(np.float32)
spac_o = om._estimate_spacing(e_pts_o) if len(e_pts_o)>=2 else -1.0
print(f"[old]    subset={sub_o.numel()} slab015={slab_o_15.numel()} slab02={slab_o_02.numel()} spacing={spac_o:.6f}")

def setcmp(a,b,name):
    sa,sb_=set(a.cpu().tolist()),set(b.cpu().tolist())
    inter=len(sa&sb_); union=len(sa|sb_) or 1
    print(f"  {name}: |old|={len(sb_)} |new|={len(sa)} jaccard={inter/union:.4f} identical={sa==sb_}")
    return sa==sb_
print("=== A/B (index SETS) ===")
ok1=setcmp(sub_n,sub_o,"existing_object_subset")
ok2=setcmp(slab_n_15,slab_o_15,"slab@0.015")
ok3=setcmp(slab_n_02,slab_o_02,"slab@0.02")
ok4=abs(spac_n-spac_o)<1e-6
print(f"  spacing identical={ok4} (old={spac_o:.6f} new={spac_n:.6f})")
print("RESULT:", "PASS" if (ok1 and ok2 and ok3 and ok4) else "MISMATCH")
