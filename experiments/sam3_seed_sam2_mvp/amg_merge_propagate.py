#!/usr/bin/env python3
# =============================================================================
# amg_merge_propagate.py  --  "FLAVOR 1, reordered" to fix the SAM3-seed border
# leakage.
#
# WHY: seeding the SAM2 video predictor with SAM3 masks propagates SAM3's LOOSE
# boundaries (a few px of table at every object edge) -> border colour leakage.
# SAM2's OWN automatic masks are edge-tight. So we use SAM2 for boundaries and
# SAM3 only for the grouping decision:
#
#   1. SAM2-AMG on frame 0            -> fine masks, SAM2-tight boundaries
#   2. SAM3 on frame 0               -> object instances (grouping targets only)
#   3. compute_merge (union-find)    -> which AMG masks share one object
#   4. union AMG masks per object    -> merged seed masks, STILL SAM2-tight edges
#   5. SAM2 video propagate the merged seeds -> tight borders + merged IDs,
#      fewer objects than the raw 16 AMG masks
#
# Outputs (output/amg_merge/):
#   flipbook/         side-by-side RGB / overlay
#   seg_ids.npz       per-frame merged IDs (feed color_precise_cloud.py via SEG_IDS_NPZ)
#   pointcloud.ply    naive back-projection, coloured by merged ID
#   merge_compare.png frame 0: AMG pre-merge | merged
#
# RUN: /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python \
#        experiments/sam3_seed_sam2_mvp/amg_merge_propagate.py
# =============================================================================

import os
import re
import sys
import json
import time
import colorsys
import subprocess
from collections import defaultdict

import numpy as np
import cv2
import open3d as o3d
import torch

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATASET_DIR = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "new_env/static_scene"
)

# --- SAM2 automatic masks (frame 0) ---
SAM2_HF_MODEL = "facebook/sam2.1-hiera-large"
DEVICE = "cuda"
AMG_POINTS_PER_SIDE = 32
MIN_MASK_AREA_PX = 400
MAX_MASK_AREA_FRAC = 0.60         # drop the table/background blob -> stays unsegmented
MAX_MASKS = 32
OFFLOAD_VIDEO_TO_CPU = True
OFFLOAD_STATE_TO_CPU = True

# --- SAM3 (grouping targets) ---
TEXT_PROMPTS = ["objects"]
SAM3_CONFIDENCE_THRESHOLD = 0.40
SAM3_CONDA_ENV = "sam3_dynamic_gs"
CONDA_EXE = "/home/mrc-cuhk/miniconda3/bin/conda"
COVERAGE_THRESHOLD = 0.8          # |amg ∩ sam3| / |amg| on frame 0 to merge

# --- misc ---
DEPTH_SCALE = 1000.0
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0
VOXEL_SIZE_M = 0.005
SOR_NB_NEIGHBORS, SOR_STD_RATIO = 20, 2.0
BLACK_OUT_GRIPPER = True
OVERLAY_ALPHA = 0.55
MAX_IDS = 64

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output", "amg_merge")
VIDEO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "_video_frames")
FLIPBOOK_DIR = os.path.join(OUTPUT_DIR, "flipbook")
PLY_PATH = os.path.join(OUTPUT_DIR, "pointcloud.ply")
SEG_IDS_NPZ = os.path.join(OUTPUT_DIR, "seg_ids.npz")
COMPARE_PNG = os.path.join(OUTPUT_DIR, "merge_compare.png")
FRAME0_PNG = os.path.join(OUTPUT_DIR, "_frame0_input.png")
WORKER_PATH = os.path.join(OUTPUT_DIR, "_sam3_worker.py")
SAM3_NPZ = os.path.join(OUTPUT_DIR, "_sam3_masks.npz")
SAM3_META = os.path.join(OUTPUT_DIR, "_sam3_meta.json")
PROMPTS_JSON = os.path.join(OUTPUT_DIR, "_sam3_prompts.json")


# ----------------------------------------------------------------------------
# palette + loaders
# ----------------------------------------------------------------------------
def color_for_id(k):
    if k <= 0:
        return (0.62, 0.62, 0.62)
    h = (k * 0.6180339887498949) % 1.0
    s = 0.55 + 0.35 * ((k * 2) % 3) / 2.0
    v = 0.75 + 0.25 * ((k * 5) % 2)
    return colorsys.hsv_to_rgb(h, s, v)


PALETTE = np.array([color_for_id(k) for k in range(MAX_IDS + 1)], dtype=np.float32)


def load_dataset():
    with open(os.path.join(DATASET_DIR, "transforms.json")) as f:
        meta = json.load(f)
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=int(meta["w"]), h=int(meta["h"]))
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", os.path.basename(fr["file_path"]))[-1]))
    print(f"[data] {len(frames)} frames, fx={intr['fx']:.2f} {intr['w']}x{intr['h']}")
    return frames, intr


def abspath(rel):
    return os.path.join(DATASET_DIR, rel.lstrip("./"))


def load_gripper_mask(fr):
    m = cv2.imread(abspath(fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
    return m == 0


def load_rgb(fr):
    rgb = cv2.imread(abspath(fr["file_path"]), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    if BLACK_OUT_GRIPPER and fr.get("mask_path"):
        rgb[load_gripper_mask(fr)] = 0
    return rgb


def load_depth_m(fr):
    raw = cv2.imread(abspath(fr["depth_file_path"]), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / DEPTH_SCALE


# ----------------------------------------------------------------------------
# Step 1: SAM2 automatic masks on frame 0 (edge-tight)
# ----------------------------------------------------------------------------
def sam2_amg_frame0(frame0_rgb):
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print("[amg] SAM2 automatic mask generation on frame 0 ...")
    image_model = build_sam2_hf(SAM2_HF_MODEL, device=DEVICE)
    amg = SAM2AutomaticMaskGenerator(image_model, points_per_side=AMG_POINTS_PER_SIDE,
                                     min_mask_region_area=MIN_MASK_AREA_PX, output_mode="binary_mask")
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        raw = amg.generate(frame0_rgb)

    H, W = frame0_rgb.shape[:2]
    kept = []
    for m in raw:
        area = int(m["area"])
        if area < MIN_MASK_AREA_PX or area > MAX_MASK_AREA_FRAC * H * W:
            continue
        if BLACK_OUT_GRIPPER and float(frame0_rgb[np.asarray(m["segmentation"], bool)].mean()) < 10.0:
            continue
        kept.append(m)
    kept.sort(key=lambda m: m["area"], reverse=True)
    kept = kept[:MAX_MASKS]
    masks = [np.asarray(m["segmentation"], bool) for m in kept]
    print(f"[amg] kept {len(masks)} masks")

    # seg0 label image: paint largest first, smaller overwrite -> small objects win
    seg0 = np.zeros((H, W), np.int32)
    for k, m in enumerate(masks, start=1):
        seg0[m] = k

    del amg, image_model
    torch.cuda.empty_cache()
    return masks, seg0


# ----------------------------------------------------------------------------
# Step 2: SAM3 instances on frame 0 (grouping targets)
# ----------------------------------------------------------------------------
SAM3_WORKER_SRC = r'''#!/usr/bin/env python3
import argparse, json, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--ct", type=float, default=0.3)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    prompts = json.load(open(a.prompts_json))
    image = Image.open(a.image).convert("RGB")
    W, H = image.width, image.height
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=a.ct)

    all_masks, instances = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pi, prompt in enumerate(prompts):
            try:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)
            except Exception as e:
                print(f"[sam3-worker] prompt '{prompt}' FAILED: {type(e).__name__}: {e}")
                continue
            masks = output["masks"]; scores = output["scores"]
            masks = masks.float().cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
            scores = scores.float().cpu().numpy().reshape(-1) if hasattr(scores, "cpu") else np.asarray(scores).reshape(-1)
            if masks.ndim == 2: masks = masks[None]
            if masks.ndim == 4 and masks.shape[1] == 1: masks = masks[:, 0]
            n = 0
            for i in range(masks.shape[0]):
                m = (masks[i] > 0.5).astype(np.uint8); area = int(m.sum())
                if area == 0 or area > 0.95 * H * W: continue
                all_masks.append(m)
                instances.append({"prompt_idx": pi, "prompt": prompt, "inst": n,
                                  "score": float(scores[i]) if i < len(scores) else 0.0, "area": area})
                n += 1
            print(f"[sam3-worker] prompt '{prompt}': kept {n} instance(s)")

    arr = np.stack(all_masks, 0).astype(np.uint8) if all_masks else np.zeros((0, H, W), np.uint8)
    np.savez_compressed(a.out_npz, masks=arr)
    json.dump({"prompts": prompts, "instances": instances, "H": H, "W": W}, open(a.out_json, "w"), indent=2)
    print(f"[sam3-worker] total {len(instances)} instance(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def run_sam3(frame0_png):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    open(WORKER_PATH, "w").write(SAM3_WORKER_SRC)
    json.dump(TEXT_PROMPTS, open(PROMPTS_JSON, "w"))
    cmd = [CONDA_EXE, "run", "--no-capture-output", "-n", SAM3_CONDA_ENV, "python",
           WORKER_PATH, "--image", frame0_png, "--prompts-json", PROMPTS_JSON,
           "--ct", str(SAM3_CONFIDENCE_THRESHOLD), "--out-npz", SAM3_NPZ, "--out-json", SAM3_META]
    print(f"[sam3] launching prompts={TEXT_PROMPTS} ct={SAM3_CONFIDENCE_THRESHOLD} ...")
    subprocess.run(cmd, check=True)
    masks = np.load(SAM3_NPZ)["masks"].astype(bool)
    meta = json.load(open(SAM3_META))
    print(f"[sam3] {masks.shape[0]} instance(s)")
    return masks, meta


# ----------------------------------------------------------------------------
# Step 3+4: assign each tight AMG mask to its best SAM3 object, union per object,
# DROP AMG masks that belong to no SAM3 object (table/background fragments).
#   coverage(amg, sam3) = |amg ∩ sam3| / |amg|   (AMG is tight & inside SAM3's
#   looser object mask -> high coverage for real object parts, low for table).
# ----------------------------------------------------------------------------
def assign_and_merge(amg_masks, sam3_masks):
    T = sam3_masks.shape[0]
    target = []                                   # per AMG mask: best SAM3 instance or None
    for am in amg_masks:
        a = int(am.sum())
        best_t, best_c = None, COVERAGE_THRESHOLD
        for t in range(T):
            cov = float(np.logical_and(am, sam3_masks[t]).sum()) / max(a, 1)
            if cov > best_c:
                best_c, best_t = cov, t
        target.append(best_t)

    groups = defaultdict(list)                    # sam3 instance -> [amg indices]
    for i, t in enumerate(target):
        if t is not None:
            groups[t].append(i)
    merged = []
    for t in sorted(groups):
        m = np.zeros_like(amg_masks[0], bool)
        for j in groups[t]:
            m |= amg_masks[j]
        merged.append(m)
    n_drop = sum(1 for t in target if t is None)
    print(f"[merge] {len(amg_masks)} AMG masks -> {len(merged)} objects "
          f"(SAM3-backed); dropped {n_drop} background/table fragments")
    return merged, target


# ----------------------------------------------------------------------------
# Step 5: SAM2 video propagation of the merged seeds
# ----------------------------------------------------------------------------
def write_video_frames(frames):
    os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)
    rgbs = []
    for i, fr in enumerate(frames):
        rgb = load_rgb(fr); rgbs.append(rgb)
        cv2.imwrite(os.path.join(VIDEO_FRAMES_DIR, f"{i:05d}.jpg"), rgb[:, :, ::-1])
    return rgbs


def propagate(seed_masks, num_frames, H, W):
    from sam2.build_sam import build_sam2_video_predictor_hf
    print(f"[sam2] loading video predictor ({SAM2_HF_MODEL}) ...")
    predictor = build_sam2_video_predictor_hf(SAM2_HF_MODEL, device=DEVICE)
    seg_ids = np.zeros((num_frames, H, W), np.uint8)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = predictor.init_state(VIDEO_FRAMES_DIR, offload_video_to_cpu=OFFLOAD_VIDEO_TO_CPU,
                                     offload_state_to_cpu=OFFLOAD_STATE_TO_CPU)
        for k, mask in enumerate(seed_masks, start=1):
            predictor.add_new_mask(state, frame_idx=0, obj_id=k, mask=mask)
        print(f"[sam2] seeded {len(seed_masks)} merged objects; propagating ...")
        t = time.time()
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            logits = mask_logits.squeeze(1).float().cpu().numpy()
            best = np.full((logits.shape[1], logits.shape[2]), -1e9, np.float32)
            seg = np.zeros_like(best, np.uint8)
            for j, oid in enumerate(obj_ids):
                lg = logits[j]; upd = (lg > 0.0) & (lg > best)
                seg[upd] = oid; best[upd] = lg[upd]
            if seg.shape != (H, W):
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            seg_ids[fidx] = seg
            if fidx % 20 == 0 or fidx == num_frames - 1:
                print(f"[sam2]   propagated frame {fidx + 1}/{num_frames}")
        dt = time.time() - t
        print(f"[sam2] propagation {dt:.1f}s | {1000*dt/num_frames:.0f} ms/frame ({len(seed_masks)} objects)")
    del predictor; torch.cuda.empty_cache()
    return seg_ids


# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------
def write_overlay(rgb, seg, title):
    lut = (PALETTE * 255.0).astype(np.float32)
    out = rgb.astype(np.float32).copy(); fg = seg > 0
    out[fg] = (1 - OVERLAY_ALPHA) * out[fg] + OVERLAY_ALPHA * lut[seg][fg]
    out = out.astype(np.uint8)
    for oid in np.unique(seg):
        if oid == 0: continue
        ys, xs = np.where(seg == oid); cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(out, str(int(oid)), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(int(oid)), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_flipbook(rgbs, seg_ids):
    os.makedirs(FLIPBOOK_DIR, exist_ok=True)
    for i, rgb in enumerate(rgbs):
        side = np.concatenate([rgb, write_overlay(rgb, seg_ids[i], "")], 1)
        cv2.imwrite(os.path.join(FLIPBOOK_DIR, f"{i:04d}.png"), side[:, :, ::-1])
    print(f"[out] flipbook -> {FLIPBOOK_DIR}")


def build_pointcloud(frames, seg_ids, intr):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    pts_all, col_all = [], []
    for i, fr in enumerate(frames):
        z = load_depth_m(fr)
        valid = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
        if BLACK_OUT_GRIPPER and fr.get("mask_path"):
            valid &= ~load_gripper_mask(fr)
        vv, uu = np.where(valid); zz = z[vv, uu]
        x = (uu - cx) * zz / fx; y = (vv - cy) * zz / fy
        cam = np.stack([x, y, zz, np.ones_like(zz)], 1)
        c2w = np.asarray(fr["transform_matrix"], np.float64) @ flip
        world = (c2w @ cam.T).T[:, :3]
        pts_all.append(world.astype(np.float32)); col_all.append(PALETTE[seg_ids[i][vv, uu]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all).astype(np.float64))
    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    o3d.io.write_point_cloud(PLY_PATH, pcd)
    print(f"[out] point cloud ({len(pcd.points):,} pts) -> {PLY_PATH}")


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames, intr = load_dataset()
    H, W = intr["h"], intr["w"]
    frame0 = load_rgb(frames[0])
    cv2.imwrite(FRAME0_PNG, frame0[:, :, ::-1])

    amg_masks, seg0 = sam2_amg_frame0(frame0)            # 1. SAM2 tight masks
    sam3_masks, meta = run_sam3(FRAME0_PNG)              # 2. SAM3 instances
    if sam3_masks.shape[0] == 0:
        print("[FATAL] SAM3 returned 0 instances. Lower ct or change prompt.")
        sys.exit(1)
    merged, _ = assign_and_merge(amg_masks, sam3_masks)   # 3+4. assign -> union -> tight seeds
    if not merged:
        print("[FATAL] no AMG mask cleared the coverage threshold into a SAM3 object.")
        sys.exit(1)

    # frame-0 compare: AMG pre-merge | merged
    merged_seg0 = np.zeros((H, W), np.int32)
    for k, m in enumerate(merged, start=1):
        merged_seg0[m] = k
    cmp = np.concatenate([write_overlay(frame0, seg0, "SAM2-AMG pre-merge"),
                          write_overlay(frame0, merged_seg0, "merged seeds")], 1)
    cv2.imwrite(COMPARE_PNG, cmp[:, :, ::-1]); print(f"[out] {COMPARE_PNG}")

    rgbs = write_video_frames(frames)
    seg_ids = propagate(merged, len(frames), H, W)        # 5. propagate tight merged seeds
    np.savez_compressed(SEG_IDS_NPZ, seg_ids=seg_ids)
    print(f"[out] seg_ids -> {SEG_IDS_NPZ}")
    write_flipbook(rgbs, seg_ids)
    build_pointcloud(frames, seg_ids, intr)
    print(f"\n[done] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
