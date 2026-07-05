"""static_phase0b.py — NATIVE Phase-0b on GaussianSet/SceneModel (no old StaticGSModel).

Reproduces the old run_phase0b_fusion behavior 1:1, but on the dynamic_gs2 SSOT: the two
projected-gaussian queries (existing-object-subset + slab-indices) read gsplat's `info`
(means2d/radii/depths) via SceneModel.get_outputs_with_info + the PROVEN
extract_projected_centers_and_radii; spacing/insert/instance-id-flag run on GaussianSet; and
the register + cull + back-project math is WRAPPED unchanged from dynamic_gs (register_and_
fuse_sam3d_object, backproject_mask_to_world, cull_points_in_front, load_sam3d_*).

Every numbered comment below pins which old symbol the block ports, so the A/B (old vs native
on the same (info,mask,depth)) can be checked method-by-method. The 13 flagged correctness
risks from the audit are enforced inline (lexsort key order, top_k=1, greedy-max early-break,
RAISE-vs-empty on depths None, length-count instance mask, RNG seed 42, SH<->RGB constants).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.math import k_nearest_sklearn

# --- wrapped-unchanged pure utilities from the proven old pipeline ---
from .sam3d_fusion import (
    load_sam3d_gaussian_ply, load_sam3d_rotation_wxyz, register_and_fuse_sam3d_object,
)
from .backproject import backproject_mask_to_world, cull_points_in_front

from .gaussian_set import GaussTensors

_SH_C0 = 0.28209479177387814            # SH band-0 <-> RGB constant (must match insert/read)


def extract_projected_centers_and_radii(info, num_points):
    """Read projected centers and radii from gsplat rasterization metadata.

    Inlined verbatim from the proven dynamic_gs.utils.active_mask.
    """

    if "means2d" not in info:
        raise KeyError("'means2d' not found in rasterization info.")
    if "radii" not in info:
        raise KeyError("'radii' not found in rasterization info.")

    centers = info["means2d"]
    radii = info["radii"]

    if centers.ndim == 3:
        centers = centers[0]
    if centers.ndim != 2:
        centers = centers.reshape(-1, 2)
    if radii.ndim > 1:
        radii = radii.reshape(-1)

    centers = centers.float()
    radii = radii.float()

    if centers.shape[0] != num_points:
        raise ValueError("Projected center count does not match the Gaussian count.")
    if radii.shape[0] != num_points:
        raise ValueError("Projected radius count does not match the Gaussian count.")
    if centers.shape[-1] != 2:
        raise ValueError("Projected centers must have shape [N, 2].")

    return centers, radii


# ----------------------------------------------------------------- spacing (port of #3)
def estimate_spacing(points: np.ndarray, max_samples: int = 50_000) -> float:
    """Median of per-point mean-kNN(k=3) distance. RNG seed 42 is LOAD-BEARING (deterministic A/B)."""
    if len(points) <= 1:
        return 1e-3
    if len(points) > max_samples:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), size=max_samples, replace=False)]
    k = min(3, len(points) - 1)
    distances, _ = k_nearest_sklearn(torch.from_numpy(points.astype(np.float32)), k)
    return float(distances.mean(dim=-1).median().item())


# ----------------------------------------------------------------- projected-gaussian queries
def _projected(info: dict, count: int):
    """(centers_2d[N,2], radii[N], depths[N]) from gsplat info via the proven extractor.
    depths may be None (slab query tolerates it; subset query RAISES — caller decides)."""
    centers_2d, radii = extract_projected_centers_and_radii(info, count)
    depths = info.get("depths")
    if depths is not None:
        depths = depths.reshape(-1).float()
    return centers_2d, radii, depths


def get_existing_object_subset(gset, info: dict, render_object_mask, rendered_depth):
    """Port of StaticGSModel._get_existing_object_subset -> (indices[K], means[K,3], colors[K,3]).
    The CPD/NDP registration target: frontmost-per-pixel + depth-tolerance-visible + half-downsampled
    subset of scene gaussians under the object mask. RAISES if info has no depths (matches old)."""
    count = gset.count
    centers_2d, radii, projected_depths = _projected(info, count)
    if projected_depths is None:
        raise RuntimeError("SAM3D init requires projected Gaussian depths in rasterization info.")
    mask = render_object_mask[..., 0] if render_object_mask.ndim == 3 else render_object_mask
    depth_image = rendered_depth[..., 0] if rendered_depth.ndim == 3 else rendered_depth
    height, width = mask.shape

    cx = torch.round(centers_2d[:, 0]).long()
    cy = torch.round(centers_2d[:, 1]).long()
    candidate_mask = (
        torch.isfinite(centers_2d).all(dim=-1) & torch.isfinite(radii)
        & torch.isfinite(projected_depths) & (radii > 0)
        & (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height))
    cand = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if cand.numel() > 0:
        candidate_mask[cand] &= (mask[cy[cand], cx[cand]] > 0.5)
        cand = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

    means_all = gset.snapshot().params["means"]            # [count,3] detached, live rows

    # frontmost Gaussian per masked pixel (top_k_per_pixel = 1; lexsort: pixel_id primary, depth 2nd)
    if cand.numel() >= 2:
        pixel_ids = (cy[cand] * width + cx[cand]).detach().cpu().numpy()
        cand_depths = projected_depths[cand].detach().cpu().numpy()
        order = np.lexsort((cand_depths, pixel_ids))       # LAST key (pixel_ids) is PRIMARY
        sorted_idx = cand[torch.from_numpy(order).to(cand.device)]
        sorted_pix = pixel_ids[order]
        keep = np.zeros(len(sorted_idx), dtype=bool)
        rank = 0
        top_k_per_pixel = 1                                 # HARDCODE — loosening leaks table geometry
        for i in range(len(sorted_idx)):
            rank = 0 if (i == 0 or sorted_pix[i] != sorted_pix[i - 1]) else rank + 1
            keep[i] = rank < top_k_per_pixel
        cand = sorted_idx[torch.from_numpy(keep).to(cand.device)]

    # depth-tolerance multiplier search: greedy-MAX visible, early-break at >= 50% of pre-depth count
    if cand.numel() >= 3:
        sampled = depth_image[cy[cand], cx[cand]]
        n_before = int(cand.numel())
        if cand.numel() > 1:
            nn_k = min(3, cand.numel() - 1)
            nn_d, _ = k_nearest_sklearn(means_all[cand].detach().cpu(), nn_k)
            target_spacing = float(nn_d.mean(dim=-1).median().item())
        else:
            target_spacing = 1e-3
        depth_tol = max(0.008, 5.0 * target_spacing)
        desired_min_keep = max(3, int(0.50 * n_before))
        best, best_n = None, 0
        for mult in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            vis = torch.isfinite(sampled) & (
                (projected_depths[cand] - sampled).abs() <= mult * depth_tol)
            vn = int(vis.sum().item())
            if vn > best_n:
                best, best_n = vis, vn
            if vn >= desired_min_keep:
                break
        if best is not None and best_n >= 3:
            cand = cand[best]

    # half-downsample (linspace pick, deduped)
    if cand.numel() >= 6:
        keep_count = max(3, cand.numel() // 2)
        pos = torch.linspace(0, cand.numel() - 1, steps=keep_count, device=cand.device)
        pos = torch.round(pos).long().unique(sorted=True)
        cand = cand[pos]

    fdc = gset.snapshot().params["features_dc"][cand]       # [K,3] band-0 SH
    colors = (fdc * _SH_C0 + 0.5)                           # SH2RGB (NOT clamped here, matches old)
    return cand, means_all[cand].detach(), colors.detach()


def get_object_mask_slab_indices(gset, info: dict, render_object_mask, rendered_depth,
                                 depth_tol_m: float = 0.01):
    """Port of StaticGSModel._get_object_mask_slab_indices -> indices[M]. In-mask gaussians whose
    projected depth is within depth_tol_m of the rendered front surface. Returns EMPTY (not raise)
    if info has no depths (the key asymmetry vs get_existing_object_subset)."""
    count = gset.count
    dev = gset.device
    centers_2d, radii, projected_depths = _projected(info, count)
    if projected_depths is None:
        return torch.zeros((0,), dtype=torch.long, device=dev)
    mask = render_object_mask[..., 0] if render_object_mask.ndim == 3 else render_object_mask
    depth_image = rendered_depth[..., 0] if rendered_depth.ndim == 3 else rendered_depth
    height, width = mask.shape
    cx = torch.round(centers_2d[:, 0]).long()
    cy = torch.round(centers_2d[:, 1]).long()
    in_bounds = (
        torch.isfinite(centers_2d).all(dim=-1) & torch.isfinite(radii)
        & torch.isfinite(projected_depths) & (radii > 0)
        & (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height))
    idx = torch.nonzero(in_bounds, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return idx
    in_mask = mask[cy[idx], cx[idx]] > 0.5
    sampled = depth_image[cy[idx], cx[idx]]
    near = torch.isfinite(sampled) & ((projected_depths[idx] - sampled).abs() <= float(depth_tol_m))
    return idx[in_mask & near]


# ----------------------------------------------------------------- build insert tensors (port of #5)
def _build_insert_tensors(gset, xyz: np.ndarray, rgb: np.ndarray) -> GaussTensors:
    """kNN-spacing log-scales + RGB2SH features_dc + zero rest + identity quats + logit(0.1) opacity
    (matches StaticGSModel._build_new_gaussian_tensors). sh_rest width auto-coerced by GaussTensors."""
    dev = gset.device
    x = torch.as_tensor(xyz, device=dev, dtype=torch.float32)
    c = torch.as_tensor(rgb, device=dev, dtype=torch.float32).clamp(0, 1)
    m = x.shape[0]
    if m > 1:
        nn_d, _ = k_nearest_sklearn(x.detach().cpu(), min(3, m - 1))
        avg = nn_d.mean(dim=-1, keepdim=True).to(dev).clamp_min(1e-6)
    else:
        avg = torch.full((m, 1), 1e-3, device=dev)
    features_dc = (c - 0.5) / _SH_C0                        # RGB2SH band-0
    scales = torch.log(avg.repeat(1, 3))
    quats = torch.zeros((m, 4), device=dev); quats[:, 0] = 1.0
    opac = torch.full((m, 1), float(np.log(0.1 / 0.9)), device=dev)   # logit(0.1)
    rest = torch.zeros((m, gset.sh_rest_dim, 3), device=dev)
    return GaussTensors(means=x, features_dc=features_dc, features_rest=rest,
                        scales=scales, quats=quats, opacities=opac)


# ----------------------------------------------------------------- anchor -> camera + dense tensors
def load_anchor_for_fusion(anchor, device):
    """Build a nerfstudio Cameras(1) + (rgb tensor, depth-metres tensor) from a static_segment
    AnchorRef's saved pose/intrinsics/rgb/depth — the exact frame the masks were segmented on.
    Returns (camera, image_uint8_tensor, depth_m_tensor)."""
    import cv2
    from nerfstudio.cameras.cameras import Cameras, CameraType
    pose = json.loads(Path(anchor.pose_path).read_text())
    intr = json.loads(Path(anchor.intrinsics_path).read_text())
    c2w = torch.tensor(np.asarray(pose, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam = Cameras(camera_to_worlds=c2w, fx=float(intr["fx"]), fy=float(intr["fy"]),
                  cx=float(intr["cx"]), cy=float(intr["cy"]),
                  width=int(intr["w"]), height=int(intr["h"]),
                  camera_type=CameraType.PERSPECTIVE).to(device)
    cam.metadata = {"cam_idx": 0}
    rgb = cv2.imread(str(anchor.rgb_path), cv2.IMREAD_COLOR)             # BGR
    image = torch.from_numpy(rgb[..., ::-1].copy()).to(device)          # RGB uint8 (for backproject colors)
    depth = cv2.imread(str(anchor.depth_path), cv2.IMREAD_UNCHANGED)    # float32 metres
    depth_m = torch.from_numpy(depth.astype(np.float32))
    return cam, image, depth_m


# ----------------------------------------------------------------- the driver (port of run_phase0b_fusion)
def run_phase0b_native(scene_model, gset, lock, *, anchor, sam3_objects: List[dict],
                       sam3d_results: List[dict], registration_backend: str = "ndp",
                       device=None, debug_dir=None, artifact_dir=None,
                       timing: Optional[dict] = None) -> dict:
    """Native Phase-0b: register + cull + insert each SAM3D object into the trained scene + propagate
    instance ids, all on the GaussianSet SSOT. anchor = static_segment.AnchorRef (the frame the masks
    belong to; its camera/rgb/depth are loaded here). Returns the manifest; writes phase0_manifest.json."""
    if timing is None:
        timing = {}
    device = device or gset.device
    camera, static_image, static_depth_m = load_anchor_for_fusion(anchor, device)
    manifest: dict = {}
    n_objs = len(sam3_objects)
    if n_objs == 0:
        return {}

    # Phase-0b's slab/subset indices + the Stage-C instance-id scatter assume STABLE row indices
    # across the per-object inserts (free-list swap-remove would reorder rows -> wrong-row writes).
    # Static is built freelist=False; assert it loudly so a future freelist flip can't silently corrupt.
    assert not getattr(gset, "_freelist", False), "Phase-0b requires a freelist=False GaussianSet (row-index stability)"

    # cull/flag recipe constants (verbatim from run_phase0b_fusion)
    CULL_STRENGTH, TAU_FLOOR_M = 1.3, 0.003
    CULL_DEPTH_TOL_M, FLAG_DEPTH_TOL_M = 0.015, 0.02
    IN_FRONT_BAND_M, MAX_RADIUS_M = 0.0, 0.02

    for obj_idx, (sam3_obj, sam3d_out) in enumerate(zip(sam3_objects, sam3d_results)):
        instance_id = obj_idx + 1
        if not sam3d_out:
            continue
        t_obj = time.time()
        # fresh render+info (insert() between objects changes N -> info must be regenerated)
        with lock:
            outputs, info = scene_model.get_outputs_with_info(camera)
        render_h, render_w = outputs["rgb"].shape[:2]

        try:
            source_points, source_colors = load_sam3d_gaussian_ply(sam3d_out["ply_path"])
            source_rotation_wxyz = load_sam3d_rotation_wxyz(sam3d_out["pose_path"])
        except Exception as exc:
            print(f"[phase0b] obj {obj_idx}: load failed: {exc}", flush=True)
            continue

        obj_mask_np = np.array(Image.open(sam3_obj["mask_path"]).convert("L"))
        obj_mask = torch.from_numpy((obj_mask_np > 127).astype(np.float32))[..., None].to(device)
        if obj_mask.shape[0] != render_h or obj_mask.shape[1] != render_w:
            obj_mask = torch.nn.functional.interpolate(
                obj_mask.permute(2, 0, 1).unsqueeze(0), size=(render_h, render_w),
                mode="nearest").squeeze(0).permute(1, 2, 0)

        existing_idx, existing_means, existing_colors = get_existing_object_subset(
            gset, info, obj_mask, outputs["depth"])
        existing_means_np = existing_means.cpu().numpy().astype(np.float32)
        existing_colors_np = existing_colors.cpu().numpy().astype(np.float32)

        # dense registration target via back-projection through the anchor depth (preferred)
        target_pts_np, target_colors_np = existing_means_np, existing_colors_np
        if static_depth_m is not None:
            target_pts_np, target_colors_np = backproject_mask_to_world(
                obj_mask.squeeze(-1).cpu().numpy() > 0.5, static_depth_m, static_image, camera)
        if target_pts_np.shape[0] < 3:
            print(f"[phase0b] obj {obj_idx}: <3 target points; skipping", flush=True)
            continue

        c2w_rot = camera.camera_to_worlds[0, :3, :3].detach().cpu().numpy().astype(np.float32)
        result = register_and_fuse_sam3d_object(
            source_points=source_points, source_colors=source_colors,
            target_points=target_pts_np, target_colors=target_colors_np,
            source_rotation_wxyz=source_rotation_wxyz, camera_to_world_rotation=c2w_rot,
            debug_dir=debug_dir, artifact_dir=artifact_dir,
            output_stem=f"static0_obj_{obj_idx:02d}_sam3d", registration_backend=registration_backend)

        # two slabs (cull-tight / flag-loose)
        e_idx_cull = get_object_mask_slab_indices(gset, info, obj_mask, outputs["depth"], CULL_DEPTH_TOL_M)
        e_idx_flag = get_object_mask_slab_indices(gset, info, obj_mask, outputs["depth"], FLAG_DEPTH_TOL_M)

        cull_pts = result.kept_points.astype(np.float32)
        cull_colors = result.kept_colors.astype(np.float32)
        n_culled, tau = 0, 0.0
        # Stage A — proximity de-dup vs the existing visible surface
        if cull_pts.shape[0] > 0 and e_idx_cull.numel() >= 2:
            e_pts = gset.snapshot().params["means"][e_idx_cull].cpu().numpy().astype(np.float32)
            tau = max(estimate_spacing(e_pts) * CULL_STRENGTH, TAU_FLOOR_M)
            from sklearn.neighbors import NearestNeighbors
            d, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(e_pts).kneighbors(cull_pts)
            keep = ~(np.isfinite(d[:, 0]) & (d[:, 0] <= tau))
            n_culled = int((~keep).sum())
            cull_pts, cull_colors = cull_pts[keep], cull_colors[keep]
        # Stage B — in-front (occlusion) cull vs the trusted real front surface
        n_culled_front = 0
        if cull_pts.shape[0] > 0 and target_pts_np.shape[0] >= 3:
            keep_front = cull_points_in_front(cull_pts, target_pts_np, camera,
                                              (render_h, render_w), band_m=IN_FRONT_BAND_M, radius_px=2)
            n_culled_front = int((~keep_front).sum())
            cull_pts, cull_colors = cull_pts[keep_front], cull_colors[keep_front]

        # insert survivors (object_flag=0; insert auto-stamps instance_id + inserted_flags on new rows)
        if cull_pts.shape[0] > 0:
            tensors = _build_insert_tensors(gset, cull_pts, cull_colors)
            inserted = gset.insert(tensors, object_flag=0.0, instance_id=instance_id)
        else:
            inserted = torch.zeros((0,), dtype=torch.long, device=device)

        # Stage C — instance-id flag propagation onto EXISTING rows (port of phase0.py:1065-1104)
        n_flagged = 0
        if e_idx_flag.numel() > 0 and result.kept_point_count > 0:
            ids_flat = gset.snapshot().buffers["object_instance_ids"].squeeze(-1)
            slab_owners = ids_flat[e_idx_flag]
            eligible = (slab_owners == 0) | (slab_owners == instance_id)
            cand_idx = e_idx_flag.to(device)[eligible]
            if cand_idx.numel() > 0:
                from sklearn.neighbors import NearestNeighbors
                cand_pts = gset.snapshot().params["means"][cand_idx].cpu().numpy().astype(np.float32)
                proxy_pts = result.kept_points.astype(np.float32)
                proxy_r = min(MAX_RADIUS_M, max(0.003, 1.5 * estimate_spacing(proxy_pts)))
                pd, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(proxy_pts).kneighbors(cand_pts)
                near_proxy = np.isfinite(pd[:, 0]) & (pd[:, 0] <= proxy_r)
                near_target = np.zeros(len(cand_pts), dtype=bool)
                if len(existing_means_np) > 0:
                    target_r = min(MAX_RADIUS_M, max(0.002, 6.0 * estimate_spacing(existing_means_np)))
                    td, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(existing_means_np).kneighbors(cand_pts)
                    near_target = np.isfinite(td[:, 0]) & (td[:, 0] <= target_r)
                match = near_proxy | near_target
                if match.any():
                    # write_instance_ids needs a FULL-length (==count) bool mask -> scatter the matches
                    full = torch.zeros(gset.count, dtype=torch.bool, device=device)
                    full[cand_idx[torch.from_numpy(match).to(device)]] = True
                    gset.write_instance_ids(full, instance_id)
                    n_flagged = int(match.sum())

        inst_count = int((gset.snapshot().buffers["object_instance_ids"].squeeze(-1) == instance_id).sum())
        manifest[instance_id] = {
            "object_index": obj_idx, "existing_gaussians": int(existing_idx.numel()),
            "sam3d_pre_cull": int(result.kept_point_count), "sam3d_culled": n_culled,
            "sam3d_culled_in_front": n_culled_front, "cull_tau_m": float(tau),
            "inserted_gaussians": int(inserted.numel()), "flagged_existing": n_flagged,
            "instance_count": inst_count, "registration_backend": registration_backend,
        }
        timing.setdefault(f"phase0b_obj_{obj_idx}", []).append(time.time() - t_obj)
        print(f"[phase0b] obj {obj_idx} (id={instance_id}): existing={existing_idx.numel()} "
              f"sam3d={result.kept_point_count}->{inserted.numel()} "
              f"(prox_cull={n_culled} tau={tau*1000:.1f}mm front_cull={n_culled_front}) "
              f"flagged={n_flagged} inst_total={inst_count}", flush=True)

    if artifact_dir is not None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        (Path(artifact_dir) / "phase0_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n")

    # Fail fast with a clear retry message if NO object ended up tagged (object_instance_ids all-zero).
    # Each object can drop silently above (SAM3D returned {}, PLY load failed, <3 depth-backproject
    # target points). Without this, the pipeline trains + hands off and only crashes ~30 s later in the
    # tracker with the opaque "no tracked object" error. The <3-points case is usually sparse/holed
    # depth on a thin or textureless object — retry with a bigger, more textured object.
    total_tagged = int((gset.snapshot().buffers["object_instance_ids"].squeeze(-1) > 0).sum())
    if total_tagged == 0:
        raise RuntimeError(
            f"Phase-0b fused 0 object Gaussians from {len(sam3d_results)} segmented object(s) — nothing "
            "to track. The object was found by segmentation but dropped during fusion (see the "
            "'[phase0b] obj N: ...' lines above: SAM3D failure, PLY load fail, or '<3 target points' "
            "from sparse/holed depth on a thin/textureless object). RETRY with a bigger, more textured "
            "object and make sure the depth over it is clean.")
    return manifest
