"""static_fuse.py — NATIVE static train + Phase-0b fuse + warm-cache export (no subprocess).

Replaces the old wrap (ns-train static-gs subprocess + .pt convert). Now in-process on the
dynamic_gs2 SSOT: load the TSDF seed PLY -> SceneModel (static phase, fp16 / half-res-first-100
config) -> GaussianSet -> native 500-step train (static_train) -> one-shot opacity purge ->
native Phase-0b register+cull+insert+flag (static_phase0b) -> save_warm_cache. The risky
register/cull/backproject/NDP math is still WRAPPED unchanged; only the model-method coupling
(the 8 StaticGSModel methods + the nerfstudio train loop) is reimplemented natively.

Phase-0a (FastSAM segment + SAM3D generate) is run by the orchestrator (static_pipeline) and
its outputs (anchor + masks + object PLYs) are passed in.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from . import static_phase0b, static_train
from .gaussian_set import GaussianSet
from .scene_model import SceneModel
from .static_persist import save_warm_cache, seed_ply_path


def load_seed_ply(ply_path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read the TSDF seed PLY -> (xyz[N,3] float32, rgb[N,3] float32 in [0,1]). Falls back to
    grey if the PLY has no colours."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(ply_path))
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else np.full_like(xyz, 0.5)
    return torch.from_numpy(xyz), torch.from_numpy(rgb)


def load_static_cameras(static_dir, device, *, keyframe_filter: bool = True,
                        trans_thresh_m: float = 0.02, rot_thresh_deg: float = 20.0):
    """Build per-keyframe (Cameras, batch{image,depth_image,mask}) from static_scene/transforms.json,
    on `device`. Mirrors the dataparser contract (OpenGL c2w, uint16-mm depth -> metres, keep-mask).
    An ORB-SLAM 2cm/20deg keyframe filter drops near-duplicate views (matches the old datamanager)."""
    import cv2
    from nerfstudio.cameras.cameras import Cameras, CameraType
    static_dir = Path(static_dir)
    meta = json.loads((static_dir / "transforms.json").read_text())
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    W, H = int(meta["w"]), int(meta["h"])
    frames = sorted(meta["frames"], key=lambda f: int(re.findall(r"\d+", Path(f["file_path"]).name)[-1]))

    kept_R: List[np.ndarray] = []
    kept_t: List[np.ndarray] = []
    rt = float(np.deg2rad(rot_thresh_deg))
    cameras, batches = [], []
    for fr in frames:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        if keyframe_filter and kept_R:                       # greedy 2cm/20deg dedup (far enough in T OR R)
            R, t = c2w[:3, :3], c2w[:3, 3]
            near = any(np.linalg.norm(t - tk) <= trans_thresh_m and
                       np.arccos(np.clip(0.5 * (np.trace(R.T @ Rk) - 1.0), -1, 1)) <= rt
                       for Rk, tk in zip(kept_R, kept_t))
            if near:
                continue
        rgb = cv2.imread(str(static_dir / fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
        image = torch.from_numpy(rgb[..., ::-1].copy()).float().to(device) / 255.0   # RGB [0,1]
        dp = fr.get("depth_file_path") or fr["file_path"].replace("rgb", "depth").replace(".png", ".tiff")
        d = cv2.imread(str(static_dir / dp.lstrip("./")), cv2.IMREAD_UNCHANGED)
        depth_m = torch.from_numpy((d.astype(np.float32) * 1e-3)).to(device) if d is not None else None
        batch = {"image": image}
        if depth_m is not None:
            batch["depth_image"] = depth_m
        mp = fr.get("mask_path")
        if mp:
            m = cv2.imread(str(static_dir / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                batch["mask"] = torch.from_numpy((m > 0).astype(np.float32))[..., None].to(device)
        c2w_t = torch.from_numpy(c2w[:3, :4].astype(np.float32)).unsqueeze(0)
        cam = Cameras(camera_to_worlds=c2w_t, fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
                      camera_type=CameraType.PERSPECTIVE).to(device)
        cam.metadata = {"cam_idx": len(cameras)}
        cameras.append(cam); batches.append(batch)
        kept_R.append(c2w[:3, :3]); kept_t.append(c2w[:3, 3])
    return cameras, batches


def purge_low_opacity(gset: GaussianSet, threshold: float) -> int:
    """One-shot opacity purge BEFORE Phase-0b (matches old static_phase_opacity_purge_threshold):
    delete scene gaussians with sigmoid(opacity) < threshold. cull() takes DROP indices."""
    if threshold <= 0.0:
        return 0
    snap = gset.snapshot()
    low = torch.nonzero(torch.sigmoid(snap.params["opacities"].squeeze(-1)) < threshold,
                        as_tuple=False).flatten()
    return gset.cull(low)


def train_fuse_and_export(data_dir, cfg, device, *, anchor, sam3_objects: List[dict],
                          sam3d_results: List[dict], tm=None, return_scene: bool = False,
                          on_fuse=None):
    """Native static-fuse: seed -> SceneModel -> train -> purge -> Phase-0b -> warm-cache .pt.
    anchor/sam3_objects/sam3d_results come from Phase-0a (the orchestrator). `tm` = the static
    TimingLedger (gauges train steps). `on_fuse` (optional) is called once at the train->fuse
    boundary (lets the UI checklist flip train->done / fuse->doing at the real boundary, since
    train + Phase-0b both live in this one function). Returns the .pt path, OR (path, sm, gset,
    lock) when return_scene=True (the single-process hand-off carries the scene straight to dynamic)."""
    data_dir = Path(data_dir)
    seed_xyz, seed_rgb = load_seed_ply(seed_ply_path(data_dir))
    lock = threading.RLock()
    sm = SceneModel(cfg, device, seed_xyz=seed_xyz, seed_rgb=seed_rgb, phase="static",
                    num_downscales=1, resolution_schedule=100)   # half-res first 100, then full
    sm.attach_render_lock(lock)
    gset = GaussianSet(sm, lock, freelist=False)                 # static = exact reallocating path

    cameras, batches = load_static_cameras(data_dir / "static_scene", device)

    # Depth-cap loss mask (reproduces old StaticGSModel.get_loss_dict): exclude pixels with sensor
    # depth outside (0.05, scene_depth_max_m] so the loss never fits color to far/no-return pixels.
    dmax = float(cfg.depth.scene_depth_max_m)
    def _depth_keep(batch):
        d = batch.get("depth_image")
        if dmax <= 0.0 or d is None:
            return None
        d = d.to(device).float()
        if d.ndim == 2:
            d = d[..., None]
        return ((d > 0.05) & (d < dmax)).float()
    sm.set_mask_provider(_depth_keep)

    # fp32 (NOT fp16): measured the static fit at fp16 autocast tops out ~17-18 dB masked PSNR vs
    # ~26-27 dB at fp32 — the custom gsplat rasterizer backward + SSIM lose too much precision under
    # autocast, leaving the scene blurry (which then starves the XFeat tracker: 70/313 vs 312/313).
    # The static phase is a one-time ~20s cost, so correctness wins over the (broken) fp16 speedup.
    static_train.train_static(
        sm, gset, cameras, batches,
        num_steps=cfg.static_train.num_steps, means_lr=0.0, mixed_precision=False,
        early_stop_loss=cfg.static_train.early_stop_loss,
        early_stop_patience=cfg.static_train.early_stop_patience,
        early_stop_min_steps=cfg.static_train.early_stop_min_steps,
        scale_clamp_max_m=cfg.budget.static_scale_clamp_max_m,
        scale_reset_value_m=cfg.budget.static_scale_reset_value_m,
        scale_clamp_every_n=cfg.budget.static_scale_clamp_every_n, tm=tm)

    if on_fuse is not None:                                   # train done; Phase-0b (fusion) begins now
        try:
            on_fuse()
        except Exception:
            pass

    purged = purge_low_opacity(gset, cfg.budget.static_opacity_purge_threshold)
    print(f"[static-fuse] opacity purge dropped {purged} -> {gset.num_points} gaussians", flush=True)

    # Phase-0b register writes debug/artifact PNGs + the manifest into these dirs; a fresh from-live
    # dataset has neither (only recorded datasets pre-created them) -> FileNotFoundError on the first
    # correspondence-plot write. Create them up front.
    art = data_dir / "dynamic_scene" / "initialization_artifacts"
    dbg = data_dir / "dynamic_scene" / "initialization_debug"
    art.mkdir(parents=True, exist_ok=True)
    dbg.mkdir(parents=True, exist_ok=True)
    static_phase0b.run_phase0b_native(
        sm, gset, lock, anchor=anchor, sam3_objects=sam3_objects, sam3d_results=sam3d_results,
        registration_backend=cfg.segmentation.sam3d_registration_backend,
        device=device, debug_dir=dbg, artifact_dir=art)

    out = save_warm_cache(gset, data_dir, cfg=cfg)
    print(f"[static-fuse] warm-cache -> {out} (N={gset.num_points})", flush=True)
    if return_scene:
        return out, sm, gset, lock                       # carry the in-memory scene to the dynamic loop
    return out
