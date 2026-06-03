#!/usr/bin/env python3
# =============================================================================
# sam3_seed_sam2_mvp.py  --  throwaway MVP: SAM3 segments, SAM2 propagates.
#
# "FLAVOR 2": skip SAM2's automatic-mask-generator AND the union-find merge.
# Instead:
#   1. Run SAM3 on FRAME 0 with a generic text prompt ("objects", ct=0.10) ->
#      a handful of clean per-object instance masks.
#   2. Seed the SAM2 *video predictor* DIRECTLY with those SAM3 masks (obj_id
#      1..M) and propagate across all frames. SAM2 is used purely as a tracker;
#      SAM3 is the segmenter. Per-frame cost scales with #objects, so seeding
#      ~5 SAM3 instances is much cheaper than propagating ~16 AMG masks.
#   3. Outputs: a side-by-side RGB / mask-overlay flipbook, a colored point
#      cloud (mask ID -> color, real geometry via depth back-projection), and a
#      cached seg_ids.npz.
#
# ISOLATION: self-contained. Imports NOTHING from the main dynamic-gs pipeline
#   or the sibling MVPs. SAM3 runs in its own conda env (sam3_dynamic_gs) via a
#   worker written to ./output/_sam3_worker.py at runtime; SAM2 + open3d run
#   here in dynamic_gs.
#
# RUN:
#   /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python \
#       experiments/sam3_seed_sam2_mvp/sam3_seed_sam2_mvp.py
# =============================================================================

import os
import re
import sys
import json
import time
import colorsys
import subprocess

import numpy as np
import cv2
import open3d as o3d
import torch

# ----------------------------------------------------------------------------
# CONFIG  (hardcoded on purpose -- no CLI args)
# ----------------------------------------------------------------------------
DATASET_DIR = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "new_env/static_scene"
)

TEXT_PROMPTS = ["objects"]              # generic objectness prompt
SAM3_CONFIDENCE_THRESHOLD = 0.40        # discrete objects only (table NOT detected -> stays background)
SAM3_CONDA_ENV = "sam3_dynamic_gs"
CONDA_EXE = "/home/mrc-cuhk/miniconda3/bin/conda"   # bare `conda` is not on PATH here

SAM2_HF_MODEL = "facebook/sam2.1-hiera-large"
DEVICE = "cuda"
OFFLOAD_VIDEO_TO_CPU = True             # safe defaults; flip to False to test speed
OFFLOAD_STATE_TO_CPU = True

DEPTH_SCALE = 1000.0                    # uint16 mm -> m
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0
VOXEL_SIZE_M = 0.005                    # 5 mm point-cloud downsample
SOR_NB_NEIGHBORS, SOR_STD_RATIO = 20, 2.0

BLACK_OUT_GRIPPER = True
OVERLAY_ALPHA = 0.55
DRAW_ID_LABELS = True
MAX_IDS = 64                            # palette capacity (SAM3 returns far fewer)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
VIDEO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "_sam2_video_frames")
FLIPBOOK_DIR = os.path.join(OUTPUT_DIR, "flipbook")
PLY_PATH = os.path.join(OUTPUT_DIR, "sam3_seeded_pointcloud.ply")
SEG_IDS_NPZ = os.path.join(OUTPUT_DIR, "seg_ids.npz")
FRAME0_PNG = os.path.join(OUTPUT_DIR, "_frame0_input.png")
WORKER_PATH = os.path.join(OUTPUT_DIR, "_sam3_worker.py")
SAM3_NPZ = os.path.join(OUTPUT_DIR, "_sam3_masks.npz")
SAM3_META = os.path.join(OUTPUT_DIR, "_sam3_meta.json")
PROMPTS_JSON = os.path.join(OUTPUT_DIR, "_sam3_prompts.json")


# ----------------------------------------------------------------------------
# Color palette (golden-ratio hues; id 0 = gray background)
# ----------------------------------------------------------------------------
def color_for_id(k: int) -> tuple:
    if k <= 0:
        return (0.6, 0.6, 0.6)
    h = (k * 0.6180339887498949) % 1.0
    s = 0.55 + 0.35 * ((k * 2) % 3) / 2.0
    v = 0.75 + 0.25 * ((k * 5) % 2)
    return colorsys.hsv_to_rgb(h, s, v)


PALETTE = np.array([color_for_id(k) for k in range(MAX_IDS + 1)], dtype=np.float32)


# ----------------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------------
def _frame_sort_key(path: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def load_dataset():
    with open(os.path.join(DATASET_DIR, "transforms.json")) as f:
        meta = json.load(f)
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=int(meta["w"]), h=int(meta["h"]))
    frames = sorted(meta["frames"], key=lambda fr: _frame_sort_key(fr["file_path"]))
    print(f"[data] {len(frames)} frames, fx={intr['fx']:.2f} {intr['w']}x{intr['h']}")
    return frames, intr


def abspath(rel: str) -> str:
    return os.path.join(DATASET_DIR, rel.lstrip("./"))


def load_gripper_mask(fr: dict) -> np.ndarray:
    """True where the gripper is (transforms.json mask is 0 on the gripper)."""
    m = cv2.imread(abspath(fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(fr["mask_path"])
    return m == 0


def load_rgb(fr: dict) -> np.ndarray:
    bgr = cv2.imread(abspath(fr["file_path"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(fr["file_path"])
    rgb = bgr[:, :, ::-1].copy()
    if BLACK_OUT_GRIPPER and fr.get("mask_path"):
        rgb[load_gripper_mask(fr)] = 0
    return rgb


def load_depth_m(fr: dict) -> np.ndarray:
    raw = cv2.imread(abspath(fr["depth_file_path"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(fr["depth_file_path"])
    return raw.astype(np.float32) / DEPTH_SCALE


# ----------------------------------------------------------------------------
# SAM3 subprocess worker  (runs inside sam3_dynamic_gs; written to disk at runtime)
# ----------------------------------------------------------------------------
SAM3_WORKER_SRC = r'''#!/usr/bin/env python3
# Auto-generated SAM3 worker -- runs in the sam3_dynamic_gs conda env.
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
            masks = output["masks"]
            scores = output["scores"]
            masks = masks.float().cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
            scores = scores.float().cpu().numpy().reshape(-1) if hasattr(scores, "cpu") \
                else np.asarray(scores).reshape(-1)
            if masks.ndim == 2:
                masks = masks[None]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            n_kept_for_prompt = 0
            for i in range(masks.shape[0]):
                m = (masks[i] > 0.5).astype(np.uint8)
                area = int(m.sum())
                if area == 0 or area > 0.95 * H * W:
                    continue
                all_masks.append(m)
                instances.append({
                    "prompt_idx": pi, "prompt": prompt, "inst": n_kept_for_prompt,
                    "score": float(scores[i]) if i < len(scores) else 0.0, "area": area,
                })
                n_kept_for_prompt += 1
            print(f"[sam3-worker] prompt '{prompt}': kept {n_kept_for_prompt} instance(s)")

    masks_arr = np.stack(all_masks, 0).astype(np.uint8) if all_masks \
        else np.zeros((0, H, W), np.uint8)
    np.savez_compressed(a.out_npz, masks=masks_arr)
    json.dump({"prompts": prompts, "instances": instances, "H": H, "W": W},
              open(a.out_json, "w"), indent=2)
    print(f"[sam3-worker] total {len(instances)} instance(s) over {len(prompts)} prompt(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def run_sam3(frame0_png: str):
    """Run SAM3 on frame 0 in its own conda env; return (masks (M,H,W) bool, meta)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(WORKER_PATH, "w") as f:
        f.write(SAM3_WORKER_SRC)
    with open(PROMPTS_JSON, "w") as f:
        json.dump(TEXT_PROMPTS, f)

    cmd = [CONDA_EXE, "run", "--no-capture-output", "-n", SAM3_CONDA_ENV, "python",
           WORKER_PATH, "--image", frame0_png, "--prompts-json", PROMPTS_JSON,
           "--ct", str(SAM3_CONFIDENCE_THRESHOLD),
           "--out-npz", SAM3_NPZ, "--out-json", SAM3_META]
    print(f"[sam3] launching in env '{SAM3_CONDA_ENV}' prompts={TEXT_PROMPTS} ct={SAM3_CONFIDENCE_THRESHOLD} ...")
    t = time.time()
    subprocess.run(cmd, check=True)
    print(f"[sam3] done in {time.time() - t:.1f}s")

    masks = np.load(SAM3_NPZ)["masks"].astype(bool)          # (M,H,W)
    meta = json.load(open(SAM3_META))
    print(f"[sam3] got {masks.shape[0]} instance mask(s)")
    for inst in meta["instances"]:
        print(f"         inst {inst['inst']:>2}  score={inst['score']:.3f}  area={inst['area']:,}")
    return masks, meta


# ----------------------------------------------------------------------------
# SAM2 video propagation, seeded with the SAM3 masks
# ----------------------------------------------------------------------------
def write_video_frames(frames: list) -> list:
    os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)
    rgbs = []
    for i, fr in enumerate(frames):
        rgb = load_rgb(fr)
        rgbs.append(rgb)
        cv2.imwrite(os.path.join(VIDEO_FRAMES_DIR, f"{i:05d}.jpg"), rgb[:, :, ::-1])
    return rgbs


def propagate(seed_masks: list, num_frames: int, H: int, W: int) -> np.ndarray:
    from sam2.build_sam import build_sam2_video_predictor_hf

    print(f"[sam2] loading video predictor ({SAM2_HF_MODEL}) ...")
    predictor = build_sam2_video_predictor_hf(SAM2_HF_MODEL, device=DEVICE)

    seg_ids = np.zeros((num_frames, H, W), dtype=np.uint8)
    try:
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = predictor.init_state(
                VIDEO_FRAMES_DIR,
                offload_video_to_cpu=OFFLOAD_VIDEO_TO_CPU,
                offload_state_to_cpu=OFFLOAD_STATE_TO_CPU,
            )
            for k, mask in enumerate(seed_masks, start=1):
                predictor.add_new_mask(state, frame_idx=0, obj_id=k, mask=mask)
            print(f"[sam2] seeded {len(seed_masks)} SAM3 objects on frame 0; propagating ...")

            t = time.time()
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                logits = mask_logits.squeeze(1).float().cpu().numpy()    # (n_obj,h,w)
                best = np.full((logits.shape[1], logits.shape[2]), -1e9, dtype=np.float32)
                seg = np.zeros_like(best, dtype=np.uint8)
                for j, oid in enumerate(obj_ids):
                    lg = logits[j]
                    upd = (lg > 0.0) & (lg > best)
                    seg[upd] = oid
                    best[upd] = lg[upd]
                if seg.shape != (H, W):
                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                seg_ids[fidx] = seg
                if fidx % 20 == 0 or fidx == num_frames - 1:
                    print(f"[sam2]   propagated frame {fidx + 1}/{num_frames}")
            dt = time.time() - t
            print(f"[sam2] propagation: {dt:.1f}s total  |  {1000*dt/num_frames:.0f} ms/frame "
                  f"({len(seed_masks)} objects)")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[FATAL] SAM2 OOM during propagation: {e}")
        print("        Try a smaller model or set OFFLOAD_*_TO_CPU=True.")
        sys.exit(1)

    del predictor
    torch.cuda.empty_cache()
    return seg_ids


# ----------------------------------------------------------------------------
# Outputs: flipbook + colored point cloud
# ----------------------------------------------------------------------------
def write_flipbook(rgbs: list, seg_ids: np.ndarray):
    os.makedirs(FLIPBOOK_DIR, exist_ok=True)
    color_lut = (PALETTE * 255.0).astype(np.float32)
    a = OVERLAY_ALPHA
    for i, rgb in enumerate(rgbs):
        seg = seg_ids[i]
        color_img = color_lut[seg]
        overlay = rgb.astype(np.float32).copy()
        fg = seg > 0
        overlay[fg] = (1.0 - a) * overlay[fg] + a * color_img[fg]
        overlay = overlay.astype(np.uint8)
        if DRAW_ID_LABELS:
            for oid in np.unique(seg):
                if oid == 0:
                    continue
                ys, xs = np.where(seg == oid)
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(overlay, str(int(oid)), (cx - 6, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, str(int(oid)), (cx - 6, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        side = np.concatenate([rgb, overlay], axis=1)
        cv2.imwrite(os.path.join(FLIPBOOK_DIR, f"{i:04d}.png"), side[:, :, ::-1])
    print(f"[out] wrote {len(rgbs)} flipbook frames -> {FLIPBOOK_DIR}")


def build_pointcloud(frames: list, seg_ids: np.ndarray, intr: dict):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])   # nerfstudio OpenGL c2w -> OpenCV c2w
    pts_all, col_all = [], []
    for i, fr in enumerate(frames):
        z = load_depth_m(fr)
        valid = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
        if BLACK_OUT_GRIPPER and fr.get("mask_path"):
            valid &= ~load_gripper_mask(fr)
        vv, uu = np.where(valid)
        zz = z[vv, uu]
        x = (uu - cx) * zz / fx
        y = (vv - cy) * zz / fy
        cam = np.stack([x, y, zz, np.ones_like(zz)], axis=1)
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64) @ flip
        world = (c2w @ cam.T).T[:, :3]
        col = PALETTE[seg_ids[i][vv, uu]]
        pts_all.append(world.astype(np.float32))
        col_all.append(col.astype(np.float32))
        if i % 20 == 0 or i == len(frames) - 1:
            print(f"[pcd]   back-projected frame {i + 1}/{len(frames)} ({world.shape[0]} pts)")

    pts = np.concatenate(pts_all, 0)
    cols = np.concatenate(col_all, 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    print(f"[pcd] concatenated {len(pcd.points):,} points")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
    print(f"[pcd] after {VOXEL_SIZE_M*1000:.0f} mm voxel downsample: {len(pcd.points):,}")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    print(f"[pcd] after SOR: {len(pcd.points):,}")
    o3d.io.write_point_cloud(PLY_PATH, pcd)
    print(f"[out] wrote point cloud -> {PLY_PATH}")


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames, intr = load_dataset()
    H, W = intr["h"], intr["w"]

    # frame 0 (gripper blacked) -> write PNG for the SAM3 worker
    frame0 = load_rgb(frames[0])
    cv2.imwrite(FRAME0_PNG, frame0[:, :, ::-1])

    # 1. SAM3 segments frame 0
    sam3_masks, meta = run_sam3(FRAME0_PNG)
    if sam3_masks.shape[0] == 0:
        print("[FATAL] SAM3 returned 0 masks. Lower ct or change the prompt.")
        sys.exit(1)
    seed_masks = [sam3_masks[i] for i in range(sam3_masks.shape[0])]

    # 2. SAM2 propagates those masks across all frames
    rgbs = write_video_frames(frames)
    seg_ids = propagate(seed_masks, len(frames), H, W)
    np.savez_compressed(SEG_IDS_NPZ, seg_ids=seg_ids)
    print(f"[out] cached per-frame IDs -> {SEG_IDS_NPZ}")

    # 3. outputs
    write_flipbook(rgbs, seg_ids)
    build_pointcloud(frames, seg_ids, intr)

    print(f"\n[done] outputs in {OUTPUT_DIR}")
    print(f"       flipbook : {FLIPBOOK_DIR}/0000.png ... ({len(frames)} frames)")
    print(f"       pointcloud: {PLY_PATH}")


if __name__ == "__main__":
    main()
