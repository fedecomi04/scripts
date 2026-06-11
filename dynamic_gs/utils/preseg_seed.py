"""preseg_seed.py -- factored SAM3-grouped, SAM2-AMG-tight, SAM2-video-propagated
per-Gaussian instance-id labeling for a TSDF-fused PLY.

Recipe (verbatim port of experiments/sam3_seed_sam2_mvp/{amg_merge_propagate,
color_precise_cloud}.py):
  1. SAM2-AMG on frame 0  -> tight edge masks
  2. SAM3 on frame 0 via sam_worker.sam3_infer_raw -> grouping targets
  3. Coverage-merge SAM2-AMG masks onto SAM3 instances (|amg n sam3| / |amg|);
     drop AMG masks that don't clear coverage_threshold (kills table fragments).
  4. SAM2-video propagate merged seeds across all frames -> seg_ids (F,H,W).
  5. Visibility-aware vote transfer onto PLY points using projected depth +
     occ_tol_m. Decision rule kept character-for-character with the MVP.
  6. Write <ply_stem>.instance_ids.npy sidecar.

Public API: AmgConfig, LabeledSeed, build_labeled_seed.
"""

from __future__ import annotations

import colorsys
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import open3d as o3d
import torch


# ----------------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------------
@dataclass
class AmgConfig:
    points_per_side: int = 32
    min_area_px: int = 400
    max_area_frac: float = 0.60
    max_masks: int = 32
    black_out_gripper: bool = True
    sam2_hf_model: str = "facebook/sam2.1-hiera-large"


@dataclass
class LabeledSeed:
    ply_path: Path                 # input PLY (unchanged)
    instance_ids_path: Path        # sidecar .npy with (N,) int64 -- written next to PLY
    seg_ids_path: Path             # (F, H, W) uint8 .npz -- kept for QA
    num_instances: int
    num_labeled_points: int        # number of points with instance_id > 0


# ----------------------------------------------------------------------------
# Internal helpers (verbatim from MVP, adapted to be module-local)
# ----------------------------------------------------------------------------
_DEVICE = "cuda"
_OFFLOAD_VIDEO_TO_CPU = True
_OFFLOAD_STATE_TO_CPU = True


def _log(msg: str) -> None:
    print(f"[preseg-seed] {msg}", flush=True)


def _abspath(dataset_dir: Path, rel: str) -> str:
    return os.path.join(str(dataset_dir), rel.lstrip("./"))


def _load_transforms(dataset_dir: Path):
    """Load static_scene/transforms.json from dataset_dir.

    Accepts either a path pointing at the dataset root (containing
    static_scene/) OR directly at the static_scene dir itself, mirroring how
    the MVP scripts pointed at static_scene/.
    """
    candidate_a = Path(dataset_dir) / "transforms.json"
    candidate_b = Path(dataset_dir) / "static_scene" / "transforms.json"
    if candidate_a.exists():
        meta_path = candidate_a
        base = Path(dataset_dir)
    elif candidate_b.exists():
        meta_path = candidate_b
        base = Path(dataset_dir) / "static_scene"
    else:
        raise FileNotFoundError(
            f"Could not find transforms.json under {dataset_dir} (tried "
            f"{candidate_a} and {candidate_b})"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    intr = dict(
        fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
        w=int(meta["w"]), h=int(meta["h"]),
    )
    frames = sorted(
        meta["frames"],
        key=lambda fr: int(re.findall(r"\d+", os.path.basename(fr["file_path"]))[-1]),
    )
    depth_unit_scale = float(meta.get("depth_unit_scale_factor", 1e-3))
    _log(f"loaded {len(frames)} frames, fx={intr['fx']:.2f} {intr['w']}x{intr['h']}")
    return frames, intr, base, depth_unit_scale


def _load_gripper_mask(base: Path, fr: dict) -> np.ndarray:
    m = cv2.imread(_abspath(base, fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
    return m == 0


def _load_rgb(base: Path, fr: dict, black_out_gripper: bool) -> np.ndarray:
    rgb = cv2.imread(_abspath(base, fr["file_path"]), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    if black_out_gripper and fr.get("mask_path"):
        rgb[_load_gripper_mask(base, fr)] = 0
    return rgb


def _load_depth_m(base: Path, fr: dict, depth_unit_scale: float) -> np.ndarray:
    raw = cv2.imread(_abspath(base, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) * depth_unit_scale


# ----------------------------------------------------------------------------
# Step 1: SAM2 automatic masks on frame 0 (edge-tight)
# ----------------------------------------------------------------------------
def _load_sam2_predictor(sam2_hf_model: str):
    """Load the SAM2 video predictor ONCE. It is a strict superset of the
    image model (``SAM2VideoPredictor`` subclasses ``SAM2Base``), so the same
    instance serves both the frame-0 AMG and the video propagation — loading
    ``build_sam2_hf`` separately for AMG would put the identical hiera-large
    weights on the GPU twice in sequence."""
    from sam2.build_sam import build_sam2_video_predictor_hf

    _log(f"loading SAM2 ({sam2_hf_model}) — shared by AMG + propagation ...")
    t0 = time.time()
    predictor = build_sam2_video_predictor_hf(sam2_hf_model, device=_DEVICE)
    _log(f"SAM2 loaded in {time.time() - t0:.1f}s")
    return predictor


def _sam2_amg_frame0(frame0_rgb: np.ndarray, cfg: AmgConfig, sam2_model):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    _log("SAM2 automatic mask generation on frame 0 ...")
    amg = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=cfg.points_per_side,
        min_mask_region_area=cfg.min_area_px,
        output_mode="binary_mask",
    )
    with torch.inference_mode(), torch.autocast(_DEVICE, dtype=torch.bfloat16):
        raw = amg.generate(frame0_rgb)

    H, W = frame0_rgb.shape[:2]
    kept = []
    for m in raw:
        area = int(m["area"])
        if area < cfg.min_area_px or area > cfg.max_area_frac * H * W:
            continue
        if cfg.black_out_gripper and float(
            frame0_rgb[np.asarray(m["segmentation"], bool)].mean()
        ) < 10.0:
            continue
        kept.append(m)
    kept.sort(key=lambda m: m["area"], reverse=True)
    kept = kept[: cfg.max_masks]
    masks = [np.asarray(m["segmentation"], bool) for m in kept]
    _log(f"AMG kept {len(masks)} masks")

    seg0 = np.zeros((H, W), np.int32)
    for k, m in enumerate(masks, start=1):
        seg0[m] = k

    del amg  # the shared sam2_model stays loaded for video propagation
    torch.cuda.empty_cache()
    return masks, seg0


# ----------------------------------------------------------------------------
# Step 3+4: assign each tight AMG mask to its best SAM3 instance, union per
# instance, DROP AMG masks that belong to no SAM3 instance.
# ----------------------------------------------------------------------------
def _assign_and_merge(
    amg_masks: Sequence[np.ndarray],
    sam3_masks: np.ndarray,
    coverage_threshold: float,
):
    from collections import defaultdict

    T = sam3_masks.shape[0]
    target: list = []
    for am in amg_masks:
        a = int(am.sum())
        best_t, best_c = None, coverage_threshold
        for t in range(T):
            cov = float(np.logical_and(am, sam3_masks[t]).sum()) / max(a, 1)
            if cov > best_c:
                best_c, best_t = cov, t
        target.append(best_t)

    groups = defaultdict(list)
    for i, t in enumerate(target):
        if t is not None:
            groups[t].append(i)
    # IMPORTANT: keep each merged object tagged with its ORIGINAL SAM3 instance
    # index `t` (sorted only for deterministic iteration order). Downstream the
    # SAM2-video `obj_id` is seeded as `t + 1`, so the final per-Gaussian
    # `object_instance_ids` equals the SAM3 mask number + 1 — the invariant the
    # interactive object picker relies on. (Previously this compacted to
    # 1..len(merged) via positional enumerate, silently renumbering instances
    # whenever a SAM3 mask had no AMG coverage.)
    T = sam3_masks.shape[0]
    merged: list[tuple[int, np.ndarray]] = []
    fallback_ts: list[int] = []
    for t in range(T):
        if t in groups:
            # AMG-refined boundary: union of the AMG masks assigned to this
            # SAM3 instance.
            m = np.zeros_like(amg_masks[0], bool)
            for j in groups[t]:
                m |= amg_masks[j]
        else:
            # No AMG mask cleared the coverage threshold for this SAM3
            # instance — DON'T drop it (that loses a whole object + leaves a
            # gap in the id sequence the picker exposes). Fall back to the raw
            # SAM3 mask: it's complete and clean, just not AMG-edge-tight.
            m = sam3_masks[t].astype(bool)
            fallback_ts.append(t)
        merged.append((t, m))
    n_drop = sum(1 for t in target if t is None)
    _log(
        f"merge: {len(amg_masks)} AMG masks -> {len(merged)} objects "
        f"(SAM3-backed, ids preserved); dropped {n_drop} background/table "
        f"fragments; {len(fallback_ts)} SAM3 mask(s) used raw (no AMG coverage): "
        f"{[t + 1 for t in fallback_ts]}"
    )
    return merged, target


# ----------------------------------------------------------------------------
# Step 5: SAM2 video propagation of the merged seeds
# ----------------------------------------------------------------------------
def _write_video_frames(
    frames: Iterable[dict],
    base: Path,
    black_out_gripper: bool,
    video_frames_dir: Path,
):
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    rgbs = []
    for i, fr in enumerate(frames):
        rgb = _load_rgb(base, fr, black_out_gripper)
        rgbs.append(rgb)
        cv2.imwrite(str(video_frames_dir / f"{i:05d}.jpg"), rgb[:, :, ::-1])
    return rgbs


def _propagate(
    seed_masks: Sequence[tuple[int, np.ndarray]],
    num_frames: int,
    H: int,
    W: int,
    video_frames_dir: Path,
    predictor,
) -> np.ndarray:
    """Propagate the seeded objects across the video.

    ``seed_masks`` is a list of ``(sam3_index, mask)`` pairs. Each object is
    seeded with ``obj_id = sam3_index + 1`` so the propagated label (and thus
    the final per-Gaussian instance id) equals the SAM3 mask number + 1.

    ``predictor`` is the shared SAM2 video predictor from
    :func:`_load_sam2_predictor` (also used by the frame-0 AMG).
    """
    seg_ids = np.zeros((num_frames, H, W), np.uint8)
    with torch.inference_mode(), torch.autocast(_DEVICE, dtype=torch.bfloat16):
        state = predictor.init_state(
            str(video_frames_dir),
            offload_video_to_cpu=_OFFLOAD_VIDEO_TO_CPU,
            offload_state_to_cpu=_OFFLOAD_STATE_TO_CPU,
        )
        for sam3_index, mask in seed_masks:
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=int(sam3_index) + 1, mask=mask
            )
        _log(f"seeded {len(seed_masks)} merged objects; propagating ...")
        t = time.time()
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            logits = mask_logits.squeeze(1).float().cpu().numpy()
            best = np.full((logits.shape[1], logits.shape[2]), -1e9, np.float32)
            seg = np.zeros_like(best, np.uint8)
            for j, oid in enumerate(obj_ids):
                lg = logits[j]
                upd = (lg > 0.0) & (lg > best)
                seg[upd] = oid
                best[upd] = lg[upd]
            if seg.shape != (H, W):
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            seg_ids[fidx] = seg
            if fidx % 20 == 0 or fidx == num_frames - 1:
                _log(f"propagated frame {fidx + 1}/{num_frames}")
        dt = time.time() - t
        _log(
            f"propagation {dt:.1f}s | {1000 * dt / max(num_frames, 1):.0f} ms/frame "
            f"({len(seed_masks)} objects)"
        )
    # Ownership of the shared predictor stays with the caller
    # (build_labeled_seed), which frees it after this returns.
    torch.cuda.empty_cache()
    return seg_ids


# ----------------------------------------------------------------------------
# Step 6: visibility-aware vote transfer onto a PLY
# ----------------------------------------------------------------------------
def _transfer_labels_to_ply(
    ply_path: Path,
    frames: Sequence[dict],
    intr: dict,
    base: Path,
    seg_ids: np.ndarray,
    depth_unit_scale: float,
    min_obj_votes: int,
    occ_tol_m: float,
) -> tuple[np.ndarray, int]:
    pc = o3d.io.read_point_cloud(str(ply_path))
    world = np.asarray(pc.points, np.float64)
    N = len(world)
    _log(f"precise cloud: {N:,} points")

    F, H, W = seg_ids.shape
    K = int(seg_ids.max())
    assert len(frames) == F, f"seg_ids has {F} frames but transforms has {len(frames)}"
    _log(f"labels: {F} frames, {K} object ids, {H}x{W}")

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    homog = np.concatenate([world, np.ones((N, 1))], 1).T  # (4,N)
    votes = np.zeros((N, K + 1), np.int32)                 # incl. id 0 = background

    for i, fr in enumerate(frames):
        c2w_cv = np.asarray(fr["transform_matrix"], np.float64) @ flip
        cam = (np.linalg.inv(c2w_cv) @ homog).T[:, :3]
        z = cam[:, 2]
        front = z > 1e-6
        u = np.full(N, -1.0)
        v = np.full(N, -1.0)
        u[front] = fx * cam[front, 0] / z[front] + cx
        v[front] = fy * cam[front, 1] / z[front] + cy
        in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(in_img)[0]
        ui = u[idx].astype(np.int32)
        vi = v[idx].astype(np.int32)
        zi = z[idx]

        depth = _load_depth_m(base, fr, depth_unit_scale)
        dmap = depth[vi, ui]
        vis = (dmap > 0) & (np.abs(zi - dmap) < occ_tol_m)
        idx_v = idx[vis]
        lbl = seg_ids[i][vi[vis], ui[vis]].astype(np.int64)
        np.add.at(votes, (idx_v, lbl), 1)
        if i % 20 == 0 or i == F - 1:
            _log(f"vote frame {i + 1}/{F}: {len(idx_v):,} visible votes")

    # Decision rule (verbatim from color_precise_cloud.py):
    #   bg_cnt  = votes[:, 0]
    #   obj_votes = votes.copy(); obj_votes[:, 0] = 0
    #   best_obj = obj_votes.argmax(1)
    #   best_cnt = obj_votes[np.arange(N), best_obj]
    #   ids = np.where((best_cnt >= MIN_OBJ_VOTES) & (best_cnt > bg_cnt), best_obj, 0)
    bg_cnt = votes[:, 0]
    obj_votes = votes.copy()
    obj_votes[:, 0] = 0
    best_obj = obj_votes.argmax(1)
    best_cnt = obj_votes[np.arange(N), best_obj]
    ids = np.where(
        (best_cnt >= min_obj_votes) & (best_cnt > bg_cnt), best_obj, 0
    ).astype(np.int64)

    uniq, cnt = np.unique(ids, return_counts=True)
    _log("points per id (0=background):")
    for k, c in zip(uniq, cnt):
        _log(f"  id {int(k):>2}: {int(c):>9,}  ({100 * c / max(N, 1):4.1f}%)")

    return ids, K


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def build_labeled_seed(
    *,
    dataset_dir: Path,
    ply_path: Path,
    out_dir: Path,
    sam_worker,
    text_prompts: Sequence[str] = ("objects",),
    sam3_confidence_threshold: float = 0.40,
    coverage_threshold: float = 0.80,
    amg_cfg: AmgConfig = AmgConfig(),
    min_obj_votes: int = 2,
    occ_tol_m: float = 0.02,
) -> LabeledSeed:
    """Build a per-point instance_id sidecar for ``ply_path``.

    See module docstring for the recipe. Returns a LabeledSeed describing
    the artifacts written.
    """
    dataset_dir = Path(dataset_dir)
    ply_path = Path(ply_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    frames, intr, base, depth_unit_scale = _load_transforms(dataset_dir)
    H, W = intr["h"], intr["w"]

    # Frame 0 RGB (gripper blacked out per the MVP).
    frame0 = _load_rgb(base, frames[0], amg_cfg.black_out_gripper)
    frame0_png = out_dir / "_frame0_input.png"
    cv2.imwrite(str(frame0_png), frame0[:, :, ::-1])

    # SAM2 loads ONCE — the video predictor serves both the frame-0 AMG
    # (step 1) and the propagation (step 5).
    sam2_predictor = _load_sam2_predictor(amg_cfg.sam2_hf_model)

    # Step 1: SAM2-AMG on frame 0.
    amg_masks, _seg0 = _sam2_amg_frame0(frame0, amg_cfg, sam2_predictor)
    if not amg_masks:
        raise RuntimeError(
            "SAM2-AMG returned 0 masks on frame 0; lower min_area_px or check input."
        )

    # Step 2: SAM3 on frame 0 via the shared worker. The caller's SamWorkerClient
    # must already have SAM3 loaded. sam3_infer_raw returns the boolean mask
    # stack shape (M, H, W).
    _log(
        f"SAM3 prompts={list(text_prompts)} ct={sam3_confidence_threshold} "
        f"via sam_worker.sam3_infer_raw ..."
    )
    # sam3_infer_raw takes ONE prompt at a time + an image_path. Loop over
    # prompts and concatenate the resulting boolean (M, H, W) mask stacks.
    # The worker's confidence_threshold is set at load_sam3 time; here
    # min_score=0.0 lets through everything the SAM3 backbone emits.
    sam3_mask_chunks: list[np.ndarray] = []
    for pi, prompt in enumerate(text_prompts):
        resp = sam_worker.sam3_infer_raw(
            image_path=frame0_png,
            text_prompt=prompt,
            output_dir=out_dir,
            output_stem=f"_sam3_prompt_{pi:02d}",
            min_score=0.0,
        )
        if int(resp.get("num_masks", 0)) == 0:
            continue
        masks_path = Path(resp["masks_path"])
        npz = np.load(masks_path)
        sam3_mask_chunks.append(np.asarray(npz["masks"], dtype=bool))
    if not sam3_mask_chunks:
        raise RuntimeError(
            f"SAM3 returned 0 instances for prompts={list(text_prompts)} "
            f"at ct={sam3_confidence_threshold}. Lower ct or change prompt."
        )
    sam3_masks = np.concatenate(sam3_mask_chunks, axis=0)
    if sam3_masks.ndim == 2:
        sam3_masks = sam3_masks[None]
    _log(f"SAM3 returned {sam3_masks.shape[0]} instance(s) across "
         f"{len(text_prompts)} prompt(s)")

    # Save a human-readable SAM3 overlay PNG (input image + per-instance
    # colored masks). Lands BOTH in preseg_artifacts/ and at the static_scene
    # root so it's obvious to anyone opening the dataset folder.
    try:
        overlay_paths = [
            out_dir / "sam3_overlay.png",
            out_dir.parent / "sam3_overlay.png",
        ]
        rgb = frame0.copy().astype(np.float32)
        K = int(sam3_masks.shape[0])
        # Distinct, evenly-spaced hues (HSV->BGR) so adjacent instance ids
        # always look distinguishable, regardless of K.
        hues = np.linspace(0, 179, max(K, 1), endpoint=False).astype(np.uint8)
        for i in range(K):
            hsv = np.zeros((1, 1, 3), dtype=np.uint8)
            hsv[0, 0] = (hues[i], 220, 255)
            color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color_rgb = color_bgr[::-1].astype(np.float32)
            m = sam3_masks[i].astype(bool)
            # 0.55 mask alpha keeps the underlying texture readable while
            # making colours unmistakable.
            rgb[m] = 0.45 * rgb[m] + 0.55 * color_rgb[None, :]
        overlay = np.clip(rgb, 0, 255).astype(np.uint8)
        for p in overlay_paths:
            cv2.imwrite(str(p), overlay[:, :, ::-1])
        _log(f"SAM3 overlay -> {overlay_paths[1]} (and a copy in preseg_artifacts/)")
    except Exception as exc:
        _log(f"WARNING: failed to write SAM3 overlay: {exc}")

    # Steps 3+4: coverage-merge.
    merged, _target = _assign_and_merge(amg_masks, sam3_masks, coverage_threshold)
    if not merged:
        raise RuntimeError(
            "No AMG mask cleared the coverage threshold into a SAM3 instance; "
            "try lowering coverage_threshold."
        )

    # Step 5: SAM2-video propagation.
    video_frames_dir = out_dir / "_video_frames"
    _write_video_frames(frames, base, amg_cfg.black_out_gripper, video_frames_dir)
    seg_ids = _propagate(
        merged, len(frames), H, W, video_frames_dir, sam2_predictor
    )
    del sam2_predictor
    torch.cuda.empty_cache()
    seg_ids_path = out_dir / "seg_ids.npz"
    np.savez_compressed(seg_ids_path, seg_ids=seg_ids)
    _log(f"seg_ids -> {seg_ids_path}")

    # Step 6: vote transfer onto the PLY.
    ids, num_instances = _transfer_labels_to_ply(
        ply_path=ply_path,
        frames=frames,
        intr=intr,
        base=base,
        seg_ids=seg_ids,
        depth_unit_scale=depth_unit_scale,
        min_obj_votes=min_obj_votes,
        occ_tol_m=occ_tol_m,
    )

    sidecar = ply_path.with_name(f"{ply_path.stem}.instance_ids.npy")
    np.save(sidecar, ids.astype(np.int64))
    num_labeled = int((ids > 0).sum())
    _log(
        f"wrote {sidecar} ({num_labeled:,}/{ids.size:,} points labeled, "
        f"{num_instances} instance ids)"
    )

    return LabeledSeed(
        ply_path=ply_path,
        instance_ids_path=sidecar,
        seg_ids_path=seg_ids_path,
        num_instances=num_instances,
        num_labeled_points=num_labeled,
    )
