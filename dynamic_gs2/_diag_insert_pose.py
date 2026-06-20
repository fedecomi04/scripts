"""_diag_insert_pose.py — settle the 'object inserted in wrong place' bug with DATA.

Compares, inside the fused static_state.pt:
  (A) the fused object subset (object_instance_ids == d0) centroid + extent in WORLD coords
  (B) the back-projected anchor-mask target centroid + extent (where the object SHOULD be)
  (C) the existing-scene-under-mask centroid (the real visible surface)
If A ~= B ~= C the static fuse placed it right; a big A-vs-(B,C) gap = the insertion bug.
Run from scripts/ in the dynamic_gs conda env:
    python -m dynamic_gs2._diag_insert_pose ../data_teleoperation/datasets/2026-06-20_224240
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main(data_dir):
    data_dir = Path(data_dir)
    st = data_dir / "static_scene"
    sd = torch.load(st / "static_state.pt", map_location="cpu")
    # find the gauss_params + object_instance_ids in whatever layout save_warm_cache used
    print("== static_state.pt top-level keys ==", list(sd.keys()))
    msd = sd["model_state_dict"]
    print("== model_state_dict keys ==", list(msd.keys()))
    means = msd["gauss_params.means"].float()
    inst = msd["object_instance_ids"].reshape(-1).long()
    print("means:", None if means is None else tuple(means.shape),
          "inst:", None if inst is None else tuple(inst.shape))
    assert means is not None and inst is not None

    ids, counts = torch.unique(inst[inst > 0], return_counts=True)
    if ids.numel() == 0:
        print("!! NO object_instance_ids > 0 in the fused scene — object never tagged !!")
        return
    d0 = int(ids[int(counts.argmax())].item())
    obj = means[inst == d0]
    print(f"\nd0 instance id = {d0}, fused object gaussians = {obj.shape[0]}")
    cA = obj.mean(0).numpy()
    extA = (obj.max(0).values - obj.min(0).values).numpy()
    print(f"(A) FUSED OBJECT  centroid = {cA}  extent(m) = {extA}  diag = {np.linalg.norm(extA):.4f}")

    # (B) back-project anchor mask through anchor depth+camera
    import cv2
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from dynamic_gs.fusion.phase0 import backproject_mask_to_world
    seg = st / "segmentation"
    man = json.loads((seg / "manifest.json").read_text())
    pose = json.loads((seg / "anchor" / "pose.json").read_text())
    intr = json.loads((seg / "anchor" / "intrinsics.json").read_text())
    c2w = torch.tensor(np.asarray(pose, np.float32)[:3, :4]).unsqueeze(0)
    cam = Cameras(camera_to_worlds=c2w, fx=intr["fx"], fy=intr["fy"], cx=intr["cx"],
                  cy=intr["cy"], width=intr["w"], height=intr["h"],
                  camera_type=CameraType.PERSPECTIVE)
    cam.metadata = {"cam_idx": 0}
    depth = cv2.imread(str(seg / "anchor" / "depth.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    image = cv2.imread(str(seg / "anchor" / "rgb.png"), cv2.IMREAD_COLOR)[..., ::-1].copy()
    print(f"\nanchor depth: dtype={depth.dtype} min={depth[depth>0].min():.4f} "
          f"max={depth.max():.4f} median(nonzero)={np.median(depth[depth>0]):.4f} (metres expected)")
    obj0 = man["objects"][0]
    mpath = seg / obj0["mask"]
    m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127
    if m.shape != depth.shape:
        m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
    tgt_pts, _ = backproject_mask_to_world(m, torch.from_numpy(depth),
                                           torch.from_numpy(image), cam)
    tgt = np.asarray(tgt_pts, np.float32)
    print(f"mask px = {int(m.sum())}, target points = {tgt.shape[0]}")
    if tgt.shape[0] >= 3:
        cB = tgt.mean(0)
        extB = tgt.max(0) - tgt.min(0)
        print(f"(B) ANCHOR TARGET centroid = {cB}  extent(m) = {extB}  diag = {np.linalg.norm(extB):.4f}")
        print(f"\n>>> |A - B| centroid offset = {np.linalg.norm(cA - cB)*1000:.1f} mm")
        print(f">>> A diag / B diag = {np.linalg.norm(extA)/max(np.linalg.norm(extB),1e-6):.3f} "
              f"(1.0 = same size; <1 = fused object too small)")


if __name__ == "__main__":
    main(sys.argv[1])
