"""ISOLATED full Phase-0b A/B: drive the OLD run_phase0b_fusion vs the NATIVE run_phase0b_native
on the SAME trained scene (old static_state.pt) + SAME SAM3D PLY + SAME mask + SAME anchor camera.

This removes train-variance + SAM3D-variance, isolating purely the FUSION logic (register + cull +
insert + flag). Compares the inserted-object point count, centroid, bbox, and instance-id count.
NDP is stochastic (seeded) so exact equality isn't guaranteed, but the two should land within a few
mm and a few % count — far tighter than two independent full runs.
"""
import json, re, threading
from pathlib import Path
import numpy as np
import torch
import cv2
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

DS = Path("../data_teleoperation/datasets/screwdriver recorded full").resolve()
st = DS / "static_scene"
OLD_PT = Path("/tmp/dgs2_native_ab_backup/old_static_state.pt")       # the pristine OLD trained scene
SAM3D_PLY = DS / "dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_raw_output.ply"
SAM3D_POSE = DS / "dynamic_scene/initialization_artifacts/static0_obj_00_sam3d_pose.json"
MASK = DS / "dynamic_scene/initialization_debug/static0_obj_00_mask.png"

meta = json.loads((st / "transforms.json").read_text())
fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
W, H = int(meta["w"]), int(meta["h"])
fr = sorted(meta["frames"], key=lambda f: int(re.findall(r"\d+", Path(f["file_path"]).name)[-1]))[-1]
c2w_np = np.asarray(fr["transform_matrix"], dtype=np.float32)
dev = "cuda"


def mkcam():
    cam = Cameras(camera_to_worlds=torch.tensor(c2w_np[:3, :]).unsqueeze(0),
                  fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
                  camera_type=CameraType.PERSPECTIVE).to(dev)
    cam.metadata = {"cam_idx": 0}
    return cam


blob = torch.load(OLD_PT, map_location="cpu", weights_only=False)
sd = blob["model_state_dict"]
n = int(blob["num_points"])
means = sd["gauss_params.means"].float()
rgb = (sd["gauss_params.features_dc"].float().reshape(n, 3) * 0.28209479177387814 + 0.5).clamp(0, 1)

mask_np = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)


def stats(obj_means: torch.Tensor):
    m = obj_means.detach().cpu().numpy()
    return dict(n=len(m), centroid=m.mean(0).tolist() if len(m) else None,
                bmin=m.min(0).tolist() if len(m) else None, bmax=m.max(0).tolist() if len(m) else None)


# ---------------- NATIVE ----------------
def run_native():
    from dynamic_gs2 import config as C, static_phase0b, static_segment
    from dynamic_gs2.gaussian_set import GaussianSet
    from dynamic_gs2.scene_model import SceneModel
    cfg = C.load_runtime_config()
    lock = threading.RLock()
    sm = SceneModel(cfg, dev, seed_xyz=means, seed_rgb=rgb, phase="static")
    sm.attach_render_lock(lock)
    gset = GaussianSet(sm, lock, freelist=False)
    gset.reload_from_state_dict(sd, num_points=n)
    sm.model.step = 30000
    # build an AnchorRef whose camera/rgb/depth come from this anchor frame
    depth = cv2.imread(str(st / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
    from dynamic_gs2.frame import Frame, Intrinsics
    intr = Intrinsics(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
    rgb_anchor = cv2.imread(str(st / fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
    keep = cv2.imread(str(st / fr["mask_path"].lstrip("./")), cv2.IMREAD_GRAYSCALE)
    frame = Frame(seq=1, stamp_sec=0.0, rgb_bgr=rgb_anchor, depth_m=depth,
                  mask_keep=(keep > 0).astype(np.uint8), c2w_4x4=np.asarray(fr["transform_matrix"], np.float64))
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="dgs2_iso_native_"))
    (tmp / "static_scene").mkdir(parents=True)
    anchor = static_segment.snapshot_anchor(frame, intr, tmp)
    sam3_objects = [{"object_index": 0, "mask_path": str(MASK), "score": 1.0}]
    sam3d_results = [{"ply_path": SAM3D_PLY, "pose_path": SAM3D_POSE}]
    before = gset.num_points
    static_phase0b.run_phase0b_native(sm, gset, lock, anchor=anchor, sam3_objects=sam3_objects,
                                      sam3d_results=sam3d_results, registration_backend="ndp", device=dev)
    snap = gset.snapshot()
    ins = snap.buffers["inserted_flags"].squeeze(-1) > 0.5
    return stats(snap.params["means"][ins]), int((snap.buffers["object_instance_ids"].squeeze(-1) > 0).sum())


# ---------------- OLD ----------------
def run_old():
    from dynamic_gs.static_gs_model import StaticGSModel, StaticGSModelConfig
    from dynamic_gs.utils.sam3d_fusion import (load_sam3d_gaussian_ply, load_sam3d_rotation_wxyz,
                                               register_and_fuse_sam3d_object)
    from dynamic_gs.fusion.phase0 import backproject_mask_to_world, cull_points_in_front
    ocfg = StaticGSModelConfig()
    sb = SceneBox(aabb=torch.tensor([[-2., -2, -2], [2., 2, 2]]))
    om = StaticGSModel(config=ocfg, scene_box=sb, num_train_data=len(meta["frames"]),
                       seed_points=(means.to(dev), rgb.to(dev))).to(dev)
    om.optimizers = {}
    om.step = 30000
    om.load_state_dict(sd, strict=False)
    om.eval()
    cam = mkcam()
    with torch.no_grad():
        out = om.get_outputs(cam)
    rh, rw = out["rgb"].shape[:2]
    obj_mask = torch.from_numpy((mask_np > 127).astype(np.float32))[..., None].to(dev)
    src_pts, src_cols = load_sam3d_gaussian_ply(SAM3D_PLY)
    src_rot = load_sam3d_rotation_wxyz(SAM3D_POSE)
    ei, em, ec = om._get_existing_object_subset(obj_mask, out["depth"])
    em_np, ec_np = em.cpu().numpy(), ec.cpu().numpy()
    depth = cv2.imread(str(st / fr["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
    img = torch.from_numpy(cv2.imread(str(st / fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR)[..., ::-1].copy()).to(dev)
    tgt_pts, tgt_cols = backproject_mask_to_world(obj_mask.squeeze(-1).cpu().numpy() > 0.5,
                                                  torch.from_numpy(depth), img, cam)
    c2w_rot = cam.camera_to_worlds[0, :3, :3].cpu().numpy().astype(np.float32)
    res = register_and_fuse_sam3d_object(source_points=src_pts, source_colors=src_cols,
        target_points=tgt_pts, target_colors=tgt_cols, source_rotation_wxyz=src_rot,
        camera_to_world_rotation=c2w_rot, registration_backend="ndp", output_stem="iso_old")
    e_cull = om._get_object_mask_slab_indices(obj_mask, out["depth"], depth_tol_m=0.015)
    cp, cc = res.kept_points.astype(np.float32), res.kept_colors.astype(np.float32)
    if cp.shape[0] > 0 and e_cull.numel() >= 2:
        from sklearn.neighbors import NearestNeighbors
        e_pts = om.means[e_cull].detach().cpu().numpy().astype(np.float32)
        tau = max(om._estimate_spacing(e_pts) * 1.3, 0.003)
        d, _ = NearestNeighbors(n_neighbors=1).fit(e_pts).kneighbors(cp)
        k = ~(np.isfinite(d[:, 0]) & (d[:, 0] <= tau)); cp, cc = cp[k], cc[k]
    if cp.shape[0] > 0 and tgt_pts.shape[0] >= 3:
        kf = cull_points_in_front(cp, tgt_pts, cam, (rh, rw), band_m=0.0, radius_px=2); cp, cc = cp[kf], cc[kf]
    return stats(torch.from_numpy(cp)), None   # inserted-object stats (pre-insert survivors == what gets inserted)


if __name__ == "__main__":
    print("[iso-A/B] running NATIVE phase0b on old scene + old SAM3D...")
    nat, nat_inst = run_native()
    torch.cuda.empty_cache()
    print("[iso-A/B] running OLD phase0b on old scene + old SAM3D...")
    old, _ = run_old()

    def d3(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b))) * 1000.0 if (a and b) else float("nan")
    print("\n=== ISOLATED Phase-0b A/B (same scene + same SAM3D + same mask) ===")
    print(f"  inserted count   OLD={old['n']:6d}  NATIVE={nat['n']:6d}  "
          f"delta={nat['n']-old['n']:+d} ({100.0*abs(nat['n']-old['n'])/max(1,old['n']):.1f}%)")
    print(f"  inserted centroid delta = {d3(old['centroid'], nat['centroid']):.1f} mm")
    print(f"  inserted bbox_min delta = {d3(old['bmin'], nat['bmin']):.1f} mm")
    print(f"  inserted bbox_max delta = {d3(old['bmax'], nat['bmax']):.1f} mm")
    ok = (abs(nat['n'] - old['n']) <= max(50, int(0.05 * old['n']))
          and d3(old['centroid'], nat['centroid']) <= 5.0)
    print("RESULT:", "PASS (native fusion matches old within tolerance)" if ok else "REVIEW (delta exceeds tolerance)")
