"""Decisive test: native Phase-0b on the OLD well-trained scene (no native train) -> save cache.
If this tracks well, the regression is the NATIVE TRAIN (not the fuse). Uses old .pt scene + old SAM3D."""
import json, re, threading
from pathlib import Path
import numpy as np, torch, cv2
from dynamic_gs2 import config as C, static_phase0b, static_segment
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import save_warm_cache
from dynamic_gs2.frame import Frame, Intrinsics

DS = Path("../data_teleoperation/datasets/screwdriver recorded full").resolve()
st = DS/"static_scene"
OLD_PT = Path("/tmp/dgs2_native_ab_backup/old_static_state.pt")
SAM3D_PLY = DS/"dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_raw_output.ply"
SAM3D_POSE = DS/"dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_pose.json"
MASK = DS/"dynamic_scene/initialization_debug/static0_obj_00_mask.png"
meta = json.loads((st/"transforms.json").read_text())
fr = sorted(meta["frames"], key=lambda f:int(re.findall(r"\d+",Path(f["file_path"]).name)[-1]))[-1]
dev="cuda"; cfg=C.load_runtime_config()

blob=torch.load(OLD_PT,map_location="cpu",weights_only=False); sd=blob["model_state_dict"]; n=int(blob["num_points"])
means=sd["gauss_params.means"].float()
rgb=(sd["gauss_params.features_dc"].float().reshape(n,3)*0.28209479177387814+0.5).clamp(0,1)
lock=threading.RLock()
sm=SceneModel(cfg,dev,seed_xyz=means,seed_rgb=rgb,phase="static"); sm.attach_render_lock(lock)
gset=GaussianSet(sm,lock,freelist=False); gset.reload_from_state_dict(sd,num_points=n); sm.model.step=30000
# IMPORTANT: clear the OLD object's instance/inserted flags so we re-fuse cleanly (else double object)
import torch as _t
with lock:
    gset._buffers["object_instance_ids"].zero_(); gset._buffers["inserted_flags"].zero_()
    gset._sync_buffer_attr()
# build anchor from last frame
depth=cv2.imread(str(st/fr["depth_file_path"].lstrip("./")),cv2.IMREAD_UNCHANGED).astype(np.float32)*1e-3
intr=Intrinsics(width=int(meta["w"]),height=int(meta["h"]),fx=meta["fl_x"],fy=meta["fl_y"],cx=meta["cx"],cy=meta["cy"])
rgb_a=cv2.imread(str(st/fr["file_path"].lstrip("./")),cv2.IMREAD_COLOR)
keep=cv2.imread(str(st/fr["mask_path"].lstrip("./")),cv2.IMREAD_GRAYSCALE)
frame=Frame(seq=1,stamp_sec=0.0,rgb_bgr=rgb_a,depth_m=depth,mask_keep=(keep>0).astype(np.uint8),
            c2w_4x4=np.asarray(fr["transform_matrix"],np.float64))
import tempfile; tmp=Path(tempfile.mkdtemp(prefix="dgs2_nfo_")); (tmp/"static_scene").mkdir(parents=True)
anchor=static_segment.snapshot_anchor(frame,intr,tmp)
static_phase0b.run_phase0b_native(sm,gset,lock,anchor=anchor,
    sam3_objects=[{"object_index":0,"mask_path":str(MASK),"score":1.0}],
    sam3d_results=[{"ply_path":SAM3D_PLY,"pose_path":SAM3D_POSE}],
    registration_backend="ndp",device=dev)
out=save_warm_cache(gset,DS,cfg=cfg)
print(f"[nfo] native Phase-0b on OLD scene -> {out} N={gset.num_points} "
      f"obj={int((gset.snapshot().buffers['object_instance_ids'][:,0]==1).sum())}")
