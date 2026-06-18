"""dynamic_ff_backends.py — real CDN + AnySplat decode callbacks for FeedforwardWorker.

WRAPS the proven dynamic_gs utils (change_mask CDN, anysplat_decode reproject, the
persistent AnySplat worker). The FF orchestration / P0 frozen-dispatch / load-shed live
in dynamic_feedforward.py; these are the injected cdn_fn / decode_fn it calls.

- make_cdn_fn: SMOKE-TESTABLE (renders the loaded scene + runs compute_change_mask).
- AnysplatHandle + make_decode_fn: need the AnySplat subprocess (anysplat_dynamic_gs env)
  + a live decode — VALIDATED BY THE OPERATOR (the unattended-validated path is FF-off).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from .gaussian_set import GaussTensors

_H_ANY = _W_ANY = 448            # AnySplat fixed crop resolution


def _clean_object_footprint(obj_bool: np.ndarray, scale: float, dilate_px: int) -> np.ndarray:
    """Enlarge the tracked-object footprint by `scale` about its OWN centroid (+1.02 swallows
    the rendered-vs-live misplacement ring) then dilate by `dilate_px`. Ported from the old
    _scale_mask_about_centroid + dilate_binary_mask. Returns a bool mask (H,W)."""
    import cv2
    m = obj_bool.astype(np.uint8)
    if scale is not None and abs(float(scale) - 1.0) > 1e-6:
        ys, xs = np.where(m > 0)
        if xs.size:
            cx, cy = float(xs.mean()), float(ys.mean())
            H, W = m.shape
            s = float(scale)
            M = np.array([[s, 0.0, cx * (1.0 - s)], [0.0, s, cy * (1.0 - s)]], np.float32)
            warped = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
            m = np.maximum(m, warped)          # union so the enlarged mask CONTAINS the original
    if dilate_px and dilate_px > 0:
        k = np.ones((2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1), np.uint8)
        m = cv2.dilate(m, k)
    return m > 0


def _old_change_mask_config(cm):
    from dynamic_gs.change_detection.change_mask import ChangeMaskConfig as OldCMC
    return OldCMC(
        rgb_threshold=float(cm.rgb_threshold), mode="rgb",
        scene_coverage_threshold=float(cm.scene_coverage_threshold),
        live_depth_min_m=float(cm.live_depth_min_m), live_depth_max_m=float(cm.live_depth_max_m),
        min_component_size=int(cm.min_component_area), block_valid_min_frac=float(cm.block_valid_min_frac),
    )


def make_cdn_fn(scene_model, lock, cfg, intr) -> Callable:
    """Return cdn_fn(dispatch) -> [cleaned_cdn_mask_np] (HxW bool) or [] if no change.
    Renders the scene at the dispatch camera UNDER LOCK; scores MS-SSIM lock-free."""
    from dynamic_gs.change_detection.change_mask import compute_change_mask, resolve_downsample_factor
    old_cfg = _old_change_mask_config(cfg.change_mask)
    om_scale = float(cfg.feedforward.object_mask_scale)
    om_dilate = int(cfg.feedforward.object_mask_dilate_px)

    def cdn_fn(d) -> List[np.ndarray]:
        dev = scene_model.device
        with lock:
            rgb_r, depth_r, alpha_r = scene_model.render(d.camera)
        live_rgb = torch.from_numpy(np.ascontiguousarray(d.rgb_bgr[..., ::-1])).float().to(dev) / 255.0
        gt_depth = d.depth_m if torch.is_tensor(d.depth_m) else torch.from_numpy(d.depth_m).to(dev)
        ds = resolve_downsample_factor(rgb_r, 0, int(cfg.change_mask.downsample_target_side))
        cdn = compute_change_mask(
            rendered_rgb=rgb_r, rendered_depth=depth_r, rendered_alpha=alpha_r,
            live_rgb=live_rgb, gt_depth=gt_depth,
            gripper_mask=d.gripper_keep, object_mask=d.object_mask,
            config=old_cfg, downsample_factor=ds,
            keep_largest_only=bool(cfg.change_mask.keep_largest_only))
        m = cdn.squeeze(-1) if cdn.ndim == 3 else cdn
        m_np = m.detach().cpu().numpy().astype(bool)
        # CLEAN (ported _feedforward_clean_cdn): subtract the tracked object's footprint so FF
        # never re-decodes a flat copy ONTO the tracked 3D object (the documented churn/smear).
        if d.object_mask is not None:
            obj = d.object_mask
            obj = obj.squeeze(-1) if obj.ndim == 3 else obj
            obj_np = obj.detach().cpu().numpy() > 0.5
            obj_np = _clean_object_footprint(obj_np, om_scale, om_dilate)
            m_np = m_np & ~obj_np
        return [m_np] if m_np.any() else []

    return cdn_fn


def _crop_windows(cdn_np: np.ndarray, pad: int) -> List[tuple]:
    """Square (left,top,size) window(s) encompassing the change bbox (1, or 2 tiled when
    wider than the image short side). Mirrors old _anysplat_crop_windows."""
    ys, xs = np.where(cdn_np)
    if xs.size == 0:
        return []
    H, W = cdn_np.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0, y1 - y0
    size = min(max(bw, bh) + 2 * pad, min(H, W))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if bw > min(H, W):                       # too wide for one square -> 2 tiled
        wins = []
        for c in (x0 + size // 2, x1 - size // 2):
            left = max(0, min(c - size // 2, W - size)); top = max(0, min(cy - size // 2, H - size))
            wins.append((left, top, size))
        return wins[:2]
    left = max(0, min(cx - size // 2, W - size)); top = max(0, min(cy - size // 2, H - size))
    return [(left, top, size)]


class AnysplatHandle:
    """Lazy wrapper over the persistent AnySplat worker (anysplat_dynamic_gs env)."""

    def __init__(self, device, conda_env: str = "anysplat_dynamic_gs", timeout_s: float = 60.0):
        self.device = device
        self.conda_env = conda_env
        self.timeout_s = timeout_s
        self._worker = None

    def _ensure(self):
        if self._worker is None:
            from dynamic_gs.utils.anysplat_decode import PersistentAnysplatWorker
            self._worker = PersistentAnysplatWorker(conda_env=self.conda_env,
                                                    startup_timeout_s=self.timeout_s)
        return self._worker

    def inference(self, crop_png: Path, out_npz: Path) -> dict:
        return self._ensure().inference([crop_png], out_npz, timeout_s=self.timeout_s)

    def close(self):
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception:
                pass
            self._worker = None


def make_decode_fn(anysplat: AnysplatHandle, cfg, intr) -> Callable:
    """Return decode_fn(dispatch, regions, snapshot) -> GaussTensors. OPERATOR-VALIDATED:
    runs the AnySplat subprocess + reproject (the proven anysplat_decode path)."""
    from dynamic_gs.utils.anysplat_decode import reproject_anysplat_to_scene, icp_refine_scene_c2w
    ff = cfg.feedforward
    scene_intr = {"fl_x": intr.fx, "fl_y": intr.fy, "cx": intr.cx, "cy": intr.cy, "w": intr.width, "h": intr.height}

    def decode_fn(d, regions, snap) -> Optional[GaussTensors]:
        cdn_np = regions[0]
        wins = _crop_windows(cdn_np, int(ff.crop_pad_px))
        if not wins:
            return None
        src_bgr = d.rgb_bgr
        depth_np = d.depth_m.detach().cpu().numpy() if torch.is_tensor(d.depth_m) else np.asarray(d.depth_m)
        c2w = d.camera.camera_to_worlds[0].detach().cpu().numpy() if hasattr(d.camera, "camera_to_worlds") else np.eye(4)
        scene_c2w = np.eye(4, dtype=np.float64); scene_c2w[:3, :4] = c2w
        if ff.icp_refine:
            try:
                tgt = snap.params["means"].to(anysplat.device)
                scene_c2w, _ = icp_refine_scene_c2w(sensor_depth_m=depth_np, scene_c2w=scene_c2w,
                                                    scene_intr=scene_intr, target_xyz_gpu=tgt)
            except Exception as e:
                print(f"[ff-decode] ICP skipped: {e}")
        import cv2
        pid = os.getpid()
        parts = []
        for wi, (left, top, size) in enumerate(wins):
            crop_png = Path(f"/dev/shm/dgs2_ff_crop_{pid}_{wi}.png")
            out_npz = Path(f"/dev/shm/dgs2_ff_ipc_{pid}_{wi}.npz")
            try:
                cv2.imwrite(str(crop_png), src_bgr[top:top + size, left:left + size])
                anysplat.inference(crop_png, out_npz)
                with open(out_npz, "rb") as f:
                    data = pickle.load(f)
                dec = reproject_anysplat_to_scene(
                    means_canonical=data["means_canonical"], log_scales=data["log_scales"],
                    quats_wxyz=data["quats_wxyz"], opacity_logits=data["opacity_logits"],
                    features_dc=data["features_dc"], features_rest=data["features_rest"],
                    pred_c2w_0=data["pred_extrinsic_c2w"][0], pred_K_norm=data["pred_intrinsic_norm"][0],
                    pred_image_hw=(_H_ANY, _W_ANY), sensor_depth_m=depth_np, scene_c2w=scene_c2w,
                    scene_intr=scene_intr, opacity_min=float(ff.opacity_min), component_mask=cdn_np,
                    scene_crop=(left, top, size), scale_multiplier=float(ff.scale_multiplier),
                    max_scale_m=float(ff.max_scale_m), min_scale_m=float(ff.min_scale_m))
                if int(dec["xyz"].shape[0]) > 0:
                    parts.append(dec)
            except Exception as e:
                print(f"[ff-decode] window {wi} failed: {e}")
            finally:
                for p in (crop_png, out_npz):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
        if not parts:
            return None
        dev = anysplat.device
        cat = lambda k: torch.cat([torch.from_numpy(p[k]).to(dev) for p in parts], 0)
        return GaussTensors(means=cat("xyz"), features_dc=cat("features_dc"),
                            features_rest=cat("features_rest"), scales=cat("scales"),
                            quats=cat("quats"), opacities=cat("opacities"))

    return decode_fn
