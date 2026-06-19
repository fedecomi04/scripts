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
    """Enlarge the tracked-object footprint by `scale` about its OWN centroid (e.g. 1.1 swallows
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


def make_cdn_fn(scene_model, lock, cfg, intr, data_dir=None) -> Callable:
    """Return cdn_fn(dispatch, restrict_to=None) -> [cleaned_cdn_mask_np] (HxW bool) or [] if no change.
    Renders the scene at the dispatch camera UNDER LOCK; scores single-scale SSIM lock-free.

    restrict_to (HxW bool, the first CDN region): when given, the render rasterizes ONLY gaussians whose
    mean projects into that region's padded bbox (saves projection on the rest of the frame), AND the
    resulting change mask is AND-gated to `restrict_to`. The two are COUPLED: a restricted render leaves
    everything outside the bbox as background, so the re-CDN would flag it as change — the AND-gate
    discards exactly that spurious outside-region change. So the replaced-cull keyed on this re-CDN can
    only ever delete geometry inside the originally-detected change region.

    When cfg.debug.ff_debug_images is on, every CDN render dumps the live RGB / rendered scene / change
    mask (raw + cleaned) to <data_dir>/dynamic_scene/_ff_debug/ for visual inspection."""
    from .change_mask import compute_change_mask
    cm = cfg.change_mask
    om_scale = float(cfg.feedforward.object_mask_scale)
    om_dilate = int(cfg.feedforward.object_mask_dilate_px)
    pad = int(getattr(cfg.feedforward, "crop_pad_px", 50))
    dbg = bool(getattr(cfg.debug, "ff_debug_images", False))
    dbg_dir = (Path(data_dir) / "dynamic_scene" / "_ff_debug") if (dbg and data_dir) else None
    if dbg_dir is not None:
        dbg_dir.mkdir(parents=True, exist_ok=True)
    dbg_n = [0]                                          # CDN-call counter (closure-mutable)

    def cdn_fn(d, restrict_to: Optional[np.ndarray] = None) -> List[np.ndarray]:
        dev = scene_model.device
        restrict_idx = None
        if restrict_to is not None:
            ys, xs = np.where(restrict_to)
            if xs.size == 0:
                return []                                # first region empty -> nothing to re-check
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        with lock:
            if restrict_to is not None:                  # restrict the render to the changed-region bbox
                restrict_idx = scene_model.means_in_bbox_idx(d.camera, bbox, pad=pad)
            rgb_r, _depth_r, alpha_r = scene_model.render(d.camera, restrict_idx=restrict_idx)
        live_rgb = torch.from_numpy(np.ascontiguousarray(d.rgb_bgr[..., ::-1])).float().to(dev) / 255.0
        gt_depth = d.depth_m if torch.is_tensor(d.depth_m) else torch.from_numpy(d.depth_m).to(dev)
        cdn = compute_change_mask(
            rendered_rgb=rgb_r, rendered_alpha=alpha_r, live_rgb=live_rgb, gt_depth=gt_depth,
            gripper_keep=d.gripper_keep, object_mask=d.object_mask, cfg=cm,
            keep_largest_only=bool(cm.keep_largest_only))
        m = cdn.squeeze(-1) if cdn.ndim == 3 else cdn
        raw_np = m.detach().cpu().numpy().astype(bool)
        m_np = raw_np
        # CLEAN: subtract the tracked object's footprint so a flat copy is never re-decoded ONTO the
        # tracked 3D object (which would smear it).
        if d.object_mask is not None:
            obj = d.object_mask
            obj = obj.squeeze(-1) if obj.ndim == 3 else obj
            obj_np = obj.detach().cpu().numpy() > 0.5
            obj_np = _clean_object_footprint(obj_np, om_scale, om_dilate)
            m_np = m_np & ~obj_np
        if restrict_to is not None:                      # AND-gate: keep ONLY change inside the first region
            m_np = m_np & restrict_to
        if dbg_dir is not None:
            _dump_ff_debug(dbg_dir, dbg_n[0], d, rgb_r, raw_np, m_np, restrict_to is not None)
            dbg_n[0] += 1
        return [m_np] if m_np.any() else []

    return cdn_fn


def _dump_ff_debug(dbg_dir, n, d, rgb_r, raw_mask, clean_mask, is_recdn) -> None:
    """Write one CDN call's images for visual inspection (when cfg.debug.ff_debug_images is on):
    live RGB, rendered scene, raw change mask, cleaned change mask. `cdnB` = the second (region-
    restricted, AND-gated) re-CDN; `cdnA` = the first full CDN. Numbered so they sort in call order."""
    import cv2
    tag = "cdnB" if is_recdn else "cdnA"
    stem = f"{n:04d}_{tag}"
    def _u8_rgb(t):
        return cv2.cvtColor((t.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dbg_dir / f"{stem}_1_live.png"), d.rgb_bgr)
    cv2.imwrite(str(dbg_dir / f"{stem}_2_rendered.png"), _u8_rgb(rgb_r))
    cv2.imwrite(str(dbg_dir / f"{stem}_3_mask_raw.png"), raw_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(dbg_dir / f"{stem}_4_mask_clean.png"), clean_mask.astype(np.uint8) * 255)


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
                    max_scale_m=float(ff.max_scale_m), min_scale_m=float(ff.min_scale_m),
                    voxel_dedup_m=0.0)   # density shaping (merge/grow/corner) is applied below, not here
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
        g = {"means": cat("xyz"), "features_dc": cat("features_dc"),
             "features_rest": cat("features_rest"), "scales": cat("scales"),
             "quats": cat("quats"), "opacities": cat("opacities")}
        if os.environ.get("DGS_DENSITY_DEBUG") == "1":
            _density_report(g["means"], g["scales"])
        if os.environ.get("DGS_DUMP_INSERT") and not os.path.exists(os.environ["DGS_DUMP_INSERT"]):
            np.savez(os.environ["DGS_DUMP_INSERT"],          # one-shot: scene-frame means+scales of 1 insert
                     means=g["means"].detach().cpu().numpy(), scales=g["scales"].detach().cpu().numpy())
            print(f"[dump] wrote insert to {os.environ['DGS_DUMP_INSERT']} (N={g['means'].shape[0]})", flush=True)
        # Cornerness (once): flat points (~0) get voxel-merged + grown; CORNER points (>= threshold)
        # are passed RAW — NOT merged (a 1mm voxel straddling an edge would fuse two surfaces into one
        # splat that pokes past the corner) and NOT grown. So edges keep their raw correct geometry;
        # only flat surfaces are downsampled + hole-filled.
        vm = float(getattr(ff, "voxel_merge_m", 0.0))
        gf = float(getattr(ff, "grow_inplane_factor", 1.0))
        if (vm > 0.0 or gf > 1.0) and g["means"].shape[0] > int(ff.corner_knn_k):
            ck = int(ff.corner_knn_k); halo_k = int(getattr(ff, "corner_halo_k", 0))
            knn = _knn_indices(g["means"], max(ck, halo_k))        # ONE cKDTree build, shared below
            cs = _corner_score(g["means"], ck, float(ff.corner_var_scale),
                               float(getattr(ff, "corner_boundary_scale", 3.0)), knn=knn)
            is_corner = cs >= float(ff.corner_merge_threshold)
            # HALO (decoupled from detection): a point is also treated as corner if any of its
            # corner_halo_k nearest neighbours is a detected corner. ONE non-iterative hop (never fed
            # back -> no cascade). Lets the halo width be tuned independently of the detection k.
            if halo_k > 0 and bool(is_corner.any()) and not bool(is_corner.all()):
                is_corner = _dilate_corner_mask(g["means"], is_corner, halo_k, knn=knn)
            if os.environ.get("DGS_CORNER_DEBUG") == "1":
                csf = cs.detach().float()
                print(f"[corner-dbg] N={cs.shape[0]} score p50/p90/max="
                      f"{float(csf.quantile(0.5)):.2f}/{float(csf.quantile(0.9)):.2f}/{float(csf.max()):.2f} "
                      f"flagged_corner={float(is_corner.float().mean())*100:.0f}% (thr={float(ff.corner_merge_threshold)})",
                      flush=True)
            flat = {k: v[~is_corner] for k, v in g.items()}        # merge + grow these
            corner = {k: v[is_corner] for k, v in g.items()}       # pass raw, untouched
            if vm > 0.0 and flat["means"].shape[0] > 0:
                flat = voxel_merge(flat, vm)
            if gf > 1.0 and flat["means"].shape[0] > 0:
                flat["scales"] = _grow_inplane(flat["scales"], gf)  # uniform — all flat now
            g = {k: torch.cat([flat[k], corner[k]], 0) for k in g} if corner["means"].shape[0] else flat
        elif vm > 0.0 and g["means"].shape[0] > 0:                 # too few pts for cornerness -> just merge
            g = voxel_merge(g, vm)
        # Final HARD clamp at the insert boundary: guarantee no gaussian exceeds max_scale_m,
        # whatever upstream produced (definitive cap; warns once if anything was over).
        g["scales"] = _clamp_log_scale(g["scales"], float(ff.max_scale_m))
        if g["means"].shape[0] == 0:
            return None
        return GaussTensors(means=g["means"], features_dc=g["features_dc"],
                            features_rest=g["features_rest"], scales=g["scales"],
                            quats=g["quats"], opacities=g["opacities"])

    return decode_fn


def _grow_inplane(log_scales: torch.Tensor, factor: float) -> torch.Tensor:
    """Grow each splat's TWO LARGEST (in-plane) axes by `factor`, leaving the SMALLEST (surface
    normal) axis unchanged — fills sub-splat surface gaps without thickening the surface (no blur).
    log_scales (N,3) log-space, so growing an axis = adding log(factor). factor<=1 is a no-op."""
    if factor is None or factor <= 1.0 or log_scales.numel() == 0:
        return log_scales
    add = float(np.log(factor))
    smallest = log_scales.argmin(dim=1, keepdim=True)             # index of the normal axis
    grow = torch.full_like(log_scales, add)
    grow.scatter_(1, smallest, 0.0)                              # don't grow the normal axis
    return log_scales + grow


def _knn_indices(means: torch.Tensor, k: int) -> torch.Tensor:
    """ONE cKDTree build + query -> (N, k+1) neighbour indices (incl. self at col 0). Both _corner_score
    and _dilate_corner_mask take a SLICE of this so the tree is built ONCE per insert. CPU, O(N log N)."""
    from scipy.spatial import cKDTree
    n = means.shape[0]
    pts = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
    idx = cKDTree(pts).query(pts, k=min(k + 1, n))[1]
    return torch.as_tensor(idx, device=means.device, dtype=torch.long)


def _corner_score(means: torch.Tensor, k: int, var_scale: float,
                  boundary_scale: float = 0.80, knn: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cornerness in [0,1] per point = MAX of two signals (both from ONE shared kNN query):

      (1) CREASE — local PCA surface-variation l0/(l0+l1+l2): ~0 on a flat patch, large where two
          surfaces MEET (a fold/edge). Detects creases.
      (2) BOUNDARY — neighbour-centroid offset ‖mean(neighbours) - point‖ / mean_neighbour_dist:
          ~0 for an interior point (neighbours surround it), LARGE for a SILHOUETTE/boundary point
          (the surface just ENDS — all neighbours sit on one side, so the centroid shifts inward).
          Detects edges-with-nothing-behind, which the crease/PCA metric is BLIND to (a boundary's
          local patch is still planar -> low surf_var) and which leak when grown into the void.

    crease normalized by var_scale, boundary by boundary_scale. `knn` (N, >=k+1) neighbour indices may
    be PRECOMPUTED + shared with the halo (sliced to k+1 here); else built fresh. cov sanitized + ridged
    + CPU eigh fallback (no cusolver NaN crash). Finite [0,1]."""
    n = means.shape[0]
    if n <= max(3, k):
        return torch.zeros(n, device=means.device, dtype=means.dtype)
    kk = min(k + 1, n)
    if knn is None:
        knn = _knn_indices(means, k)
    knn = knn[:, :kk]                                             # slice the shared query to this k
    nbr = means[knn].float()                                     # (N,kk,3) incl. self at col 0
    # (1) crease
    centered = nbr - nbr.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / float(kk - 1)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.transpose(1, 2)) + 1e-10 * torch.eye(3, device=cov.device)
    try:
        evals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except Exception:
        evals = torch.linalg.eigvalsh(cov.cpu()).clamp_min(0.0).to(cov.device)
    surf_var = evals[:, 0] / evals.sum(dim=1).clamp_min(1e-12)
    crease = (surf_var / float(max(var_scale, 1e-9)))
    # (2) boundary: offset of the NEIGHBOUR centroid (exclude self at col 0) from the point itself,
    # normalized by the mean neighbour distance -> scale-invariant.
    self_xyz = nbr[:, :1, :]
    nbr_only = nbr[:, 1:, :]
    centroid = nbr_only.mean(dim=1, keepdim=True)
    offset = (centroid - self_xyz).squeeze(1).norm(dim=1)         # (N,) centroid shift
    spacing = (nbr_only - self_xyz).norm(dim=2).mean(dim=1).clamp_min(1e-9)  # mean nbr dist
    boundary = (offset / spacing) / float(max(boundary_scale, 1e-9))
    score = torch.maximum(crease, boundary)
    return torch.nan_to_num(score, nan=0.0).clamp(0.0, 1.0)


def _dilate_corner_mask(means: torch.Tensor, is_corner: torch.Tensor, halo_k: int,
                        knn: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Grow the corner set by ONE non-iterative neighbour hop: a point becomes corner if ANY of its
    halo_k nearest neighbours is already a detected corner. Decoupled from the detection k so the halo
    width is tunable on its own. Done ONCE on the ORIGINAL mask (never fed back) so it cannot cascade.
    `knn` (N, >=halo_k+1) neighbour indices may be PRECOMPUTED + shared with _corner_score (sliced to
    halo_k+1 here — the first halo_k+1 of a larger query ARE the halo_k nearest); else built fresh."""
    n = means.shape[0]
    if halo_k <= 0 or n <= halo_k:
        return is_corner
    if knn is None:
        knn = _knn_indices(means, halo_k)
    knn = knn[:, :min(halo_k + 1, n)].detach().cpu().numpy()      # (N,halo_k+1) incl. self
    corner_np = is_corner.detach().cpu().numpy()
    grown = corner_np[knn].any(axis=1)                           # any neighbour (or self) is a corner
    return torch.as_tensor(grown, device=is_corner.device, dtype=torch.bool)


_clamp_warned = False


def _clamp_log_scale(log_scales: torch.Tensor, max_scale_m: float) -> torch.Tensor:
    """Definitive insert-boundary cap: uniformly shrink any gaussian whose LARGEST axis exceeds
    max_scale_m (all 3 axes ÷ same factor so shape is preserved). max_scale_m<=0 disables.
    Some oversized inserts are NORMAL (AnySplat over-predicts a few splats every decode), so the
    notice prints ONCE per process to confirm the cap is active without spamming the log."""
    global _clamp_warned
    if max_scale_m is None or max_scale_m <= 0.0 or log_scales.numel() == 0:
        return log_scales
    log_cap = float(np.log(max_scale_m))
    log_max = log_scales.max(dim=1, keepdim=True).values            # (N,1) largest axis per row
    over = (log_max > log_cap)
    if bool(over.any()) and not _clamp_warned:
        biggest = float(torch.exp(log_max[over]).max())
        print(f"[ff-decode] insert scale cap active: clamping splats over max_scale_m={max_scale_m} "
              f"(first hit: {int(over.sum())} over, biggest {biggest*1e2:.1f}cm). Silenced hereafter.",
              flush=True)
        _clamp_warned = True
    shift = (log_max - log_cap).clamp_min(0.0)                      # >=0, uniform divide
    return log_scales - shift


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3) rotation matrices (normalized)."""
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def voxel_merge(g: dict, voxel_m: float) -> dict:
    """Downsample the AnySplat insert cloud by MOMENT-MATCH MERGE: fuse all gaussians sharing a
    `voxel_m`^3 voxel into ONE gaussian whose covariance = mean(member covariances) + covariance OF the
    member means. The second term sizes the merged splat to the cluster's spatial EXTENT, so it fills
    the voxel (no sub-voxel holes) without inflating the scale. Opacity-weighted means/colour; opacity
    = cluster max (preserve coverage); merged orientation+scale from the merged covariance eigvecs.
    Empty/sparse voxels (1 member) pass through unchanged. O(N)."""
    means, log_scales, quats = g["means"], g["scales"], g["quats"]
    n = means.shape[0]
    if voxel_m <= 0.0 or n == 0:
        return g
    dev = means.device
    # Sanitize inputs: a NaN/inf mean/scale/quat from a bad AnySplat row would poison the cluster
    # covariance -> eigh crash. nan_to_num at entry (means->0, log_scale->small, quat->finite).
    means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    log_scales = torch.nan_to_num(log_scales, nan=-7.0, posinf=-7.0, neginf=-7.0)   # ~0.9mm
    quats = torch.nan_to_num(quats, nan=0.0, posinf=0.0, neginf=0.0)
    g = {**g, "means": means, "scales": log_scales, "quats": quats}
    vidx = torch.floor(means / voxel_m).to(torch.int64)
    # cluster id per point (compact 0..M-1) via unique on voxel rows
    _, inv, counts = torch.unique(vidx, dim=0, return_inverse=True, return_counts=True)
    m = int(counts.shape[0])
    if m == n:                                                     # every voxel has 1 point -> nothing to merge
        return g
    w = torch.sigmoid(g["opacities"][:, 0]).clamp_min(1e-6)        # opacity weight per point
    wsum = torch.zeros(m, device=dev).index_add_(0, inv, w)        # (M,)
    def wmean(x):                                                  # opacity-weighted per-cluster mean of (N,...)
        flat = x.reshape(n, -1)
        acc = torch.zeros(m, flat.shape[1], device=dev, dtype=flat.dtype)
        acc.index_add_(0, inv, flat * w[:, None])
        return (acc / wsum[:, None]).reshape(m, *x.shape[1:])
    mu = wmean(means)                                              # (M,3) merged means
    # per-point covariance Sigma_i = R diag(s^2) R^T
    s2 = torch.exp(log_scales) ** 2                                # (N,3) linear scale^2
    R = _quat_to_rotmat(quats)                                    # (N,3,3)
    cov_i = R @ torch.diag_embed(s2) @ R.transpose(1, 2)          # (N,3,3)
    cov_mean = torch.zeros(m, 3, 3, device=dev).index_add_(0, inv, cov_i * w[:, None, None]) / wsum[:, None, None]
    # covariance OF the member means about their cluster mean (the EXTENT term that fills the voxel)
    d = means - mu[inv]                                           # (N,3) member offset from its cluster mean
    outer = d[:, :, None] * d[:, None, :]                        # (N,3,3)
    cov_spread = torch.zeros(m, 3, 3, device=dev).index_add_(0, inv, outer * w[:, None, None]) / wsum[:, None, None]
    cov = cov_mean + cov_spread
    # Robustness: a degenerate/collinear cluster (or a NaN from a bad input splat) makes cov
    # non-PSD/NaN -> eigh throws CUSOLVER_STATUS_INVALID_VALUE and kills the whole FF call.
    # Sanitize + a real isotropic ridge so every cov is solvable; CPU fallback as a backstop.
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.transpose(1, 2))                       # force symmetry
    cov = cov + 1e-10 * torch.eye(3, device=dev)                  # ridge (>= float32 noise floor)
    try:
        evals, evecs = torch.linalg.eigh(cov)                    # (M,3),(M,3,3) ascending
    except Exception:
        evals, evecs = torch.linalg.eigh(cov.cpu())
        evals, evecs = evals.to(dev), evecs.to(dev)
    merged_log_scales = 0.5 * torch.log(evals.clamp_min(1e-12))   # sqrt of eigenvalue, in log-space
    merged_quats = _rotmat_to_quat_wxyz(evecs)                    # (M,4)
    # opacity = cluster max (coverage); colour = opacity-weighted mean
    op_max = torch.full((m,), -1e9, device=dev)
    op_max.scatter_reduce_(0, inv, g["opacities"][:, 0], reduce="amax", include_self=True)
    out = {
        "means": mu,
        "scales": merged_log_scales,
        "quats": merged_quats,
        "opacities": op_max[:, None],
        "features_dc": wmean(g["features_dc"]),
        "features_rest": wmean(g["features_rest"]),
    }
    return out


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """(M,3,3) rotation -> (M,4) wxyz quaternion (batched). Branch on the largest of {trace, R00, R11,
    R22} so it is numerically stable AND correct for ALL rotation angles — the trace-only method gives a
    degenerate sqrt for angles >=120deg (trace <= 0), so those rows take a diagonal branch instead."""
    m = R.shape[0]
    r00, r11, r22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = r00 + r11 + r22
    q = torch.zeros(m, 4, device=R.device, dtype=R.dtype)
    # choose the branch (0=trace, 1=R00, 2=R11, 3=R22) whose under-sqrt term is largest -> stable.
    cand = torch.stack([t, r00, r11, r22], dim=1)
    branch = cand.argmax(dim=1)
    b0, b1, b2, b3 = branch == 0, branch == 1, branch == 2, branch == 3

    def _fill(mask, sexpr, w, x, y, z):
        if not bool(mask.any()):
            return
        s = torch.sqrt(sexpr[mask].clamp_min(1e-12)) * 2.0   # s = 4*component
        q[mask, 0] = w(mask, s); q[mask, 1] = x(mask, s)
        q[mask, 2] = y(mask, s); q[mask, 3] = z(mask, s)

    _fill(b0, t + 1.0,
          lambda mk, s: 0.25 * s,
          lambda mk, s: (R[mk, 2, 1] - R[mk, 1, 2]) / s,
          lambda mk, s: (R[mk, 0, 2] - R[mk, 2, 0]) / s,
          lambda mk, s: (R[mk, 1, 0] - R[mk, 0, 1]) / s)
    _fill(b1, 1.0 + r00 - r11 - r22,
          lambda mk, s: (R[mk, 2, 1] - R[mk, 1, 2]) / s,
          lambda mk, s: 0.25 * s,
          lambda mk, s: (R[mk, 0, 1] + R[mk, 1, 0]) / s,
          lambda mk, s: (R[mk, 0, 2] + R[mk, 2, 0]) / s)
    _fill(b2, 1.0 - r00 + r11 - r22,
          lambda mk, s: (R[mk, 0, 2] - R[mk, 2, 0]) / s,
          lambda mk, s: (R[mk, 0, 1] + R[mk, 1, 0]) / s,
          lambda mk, s: 0.25 * s,
          lambda mk, s: (R[mk, 1, 2] + R[mk, 2, 1]) / s)
    _fill(b3, 1.0 - r00 - r11 + r22,
          lambda mk, s: (R[mk, 1, 0] - R[mk, 0, 1]) / s,
          lambda mk, s: (R[mk, 0, 2] + R[mk, 2, 0]) / s,
          lambda mk, s: (R[mk, 1, 2] + R[mk, 2, 1]) / s,
          lambda mk, s: 0.25 * s)
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)


def _density_report(means: torch.Tensor, log_scales: torch.Tensor) -> None:
    """One-shot diagnostic (DGS_DENSITY_DEBUG=1): N, bbox, mean nearest-neighbour spacing, splat
    size, and how many points SURVIVE a 1/2/3mm voxel-dedup. Read-only — does not change inserts.
    Tells whether plain voxel-dedup (no grow) would thin a lot without leaving holes."""
    pts = means.detach().cpu().numpy()
    n = pts.shape[0]
    if n < 4:
        print(f"[density] N={n} (too few)", flush=True)
        return
    bb = pts.max(0) - pts.min(0)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nn = tree.query(pts, k=2)[0][:, 1]                            # nearest-neighbour distance per point
    splat = float(torch.exp(log_scales).max(dim=1).values.median())  # median largest-axis (m)
    def survive(v):
        vidx = np.floor(pts / v).astype(np.int64)
        return np.unique(vidx, axis=0).shape[0]
    s1, s2, s3 = survive(0.001), survive(0.002), survive(0.003)
    print(f"[density] N={n} bbox(mm)={bb[0]*1e3:.0f}x{bb[1]*1e3:.0f}x{bb[2]*1e3:.0f} "
          f"nn_spacing(mm) p10/50/90={np.quantile(nn,0.1)*1e3:.2f}/{np.median(nn)*1e3:.2f}/"
          f"{np.quantile(nn,0.9)*1e3:.2f} splat_med(mm)={splat*1e3:.2f} | "
          f"dedup survivors 1mm={s1}({100*s1/n:.0f}%) 2mm={s2}({100*s2/n:.0f}%) 3mm={s3}({100*s3/n:.0f}%)",
          flush=True)


