"""native TRAIN + native FUSE but with the OLD SAM3D PLY (eliminate fresh-SAM3D variance).
Isolates: is the 60/313 from the native-trained scene, or from fresh-SAM3D run-to-run variance?"""
import json,re,threading
from pathlib import Path
import numpy as np,torch,cv2
from dynamic_gs2 import config as C, static_train, static_fuse, static_phase0b, static_segment
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path, save_warm_cache
from dynamic_gs2.frame import Frame, Intrinsics
DS=Path("../data_teleoperation/datasets/screwdriver recorded full").resolve(); st=DS/"static_scene"
cfg=C.load_runtime_config(); dev="cuda"
OLD_PLY=DS/"dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_raw_output.ply"
OLD_POSE=DS/"dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_pose.json"
MASK=DS/"dynamic_scene/initialization_debug/static0_obj_00_mask.png"
meta=json.loads((st/"transforms.json").read_text())
fr=sorted(meta["frames"],key=lambda f:int(re.findall(r"\d+",Path(f["file_path"]).name)[-1]))[-1]
sx,sr=static_fuse.load_seed_ply(seed_ply_path(DS)); lock=threading.RLock()
sm=SceneModel(cfg,dev,seed_xyz=sx,seed_rgb=sr,phase="static",num_downscales=1,resolution_schedule=100)
sm.attach_render_lock(lock); gset=GaussianSet(sm,lock,freelist=False)
cams,batches=static_fuse.load_static_cameras(st,dev); dmax=float(cfg.depth.scene_depth_max_m)
sm.set_mask_provider(lambda b:(((b["depth_image"].to(dev)>0.05)&(b["depth_image"].to(dev)<dmax)).float()[...,None]) if b.get("depth_image") is not None else None)
static_train.train_static(sm,gset,cams,batches,num_steps=500,means_lr=0.0,mixed_precision=False,
    early_stop_loss=cfg.static_train.early_stop_loss,scale_clamp_max_m=0.05,scale_reset_value_m=0.01,scale_clamp_every_n=10)
static_fuse.purge_low_opacity(gset,cfg.budget.static_opacity_purge_threshold)
# anchor from last frame
depth=cv2.imread(str(st/fr["depth_file_path"].lstrip("./")),cv2.IMREAD_UNCHANGED).astype(np.float32)*1e-3
intr=Intrinsics(width=int(meta["w"]),height=int(meta["h"]),fx=meta["fl_x"],fy=meta["fl_y"],cx=meta["cx"],cy=meta["cy"])
rgb_a=cv2.imread(str(st/fr["file_path"].lstrip("./")),cv2.IMREAD_COLOR); keep=cv2.imread(str(st/fr["mask_path"].lstrip("./")),cv2.IMREAD_GRAYSCALE)
frame=Frame(seq=1,stamp_sec=0.0,rgb_bgr=rgb_a,depth_m=depth,mask_keep=(keep>0).astype(np.uint8),c2w_4x4=np.asarray(fr["transform_matrix"],np.float64))
import tempfile; tmp=Path(tempfile.mkdtemp()); (tmp/"static_scene").mkdir(parents=True)
anchor=static_segment.snapshot_anchor(frame,intr,tmp)
static_phase0b.run_phase0b_native(sm,gset,lock,anchor=anchor,
    sam3_objects=[{"object_index":0,"mask_path":str(MASK),"score":1.0}],
    sam3d_results=[{"ply_path":OLD_PLY,"pose_path":OLD_POSE}],registration_backend="ndp",device=dev)
out=save_warm_cache(gset,DS,cfg=cfg)
print(f"[ntos] native-train + OLD-SAM3D fuse -> N={gset.num_points} obj={int((gset.snapshot().buffers['object_instance_ids'][:,0]==1).sum())}")
