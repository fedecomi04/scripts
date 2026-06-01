#!/usr/bin/env python3
# =============================================================================
# SAM2 static RGB-D MVP  --  throwaway, self-contained experiment.
#
# WHAT IT DOES
#   1. Runs SAM2's automatic ("everything") mask generator on frame 0 of a
#      static RGB-D capture (camera moving around a static scene).
#   2. Feeds those frame-0 masks as prompts into the SAM2 *video* predictor and
#      propagates them across every frame (cross-frame memory => consistent IDs).
#   3. Writes a side-by-side flipbook (left: RGB, right: RGB + per-ID colored
#      mask overlay), zero-padded so fast scrolling looks like a video.
#   4. Back-projects every valid depth pixel of every frame to world space using
#      transforms.json, colors each point by its SAM2 mask ID, concatenates all
#      frames (no fusion), voxel-downsamples + removes outliers, saves one .ply.
#
# This is an ISOLATED experiment. It imports nothing from the dynamic-gs
# pipeline and shares no modules with the main repo. Everything lives here.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT / INSTALL
#   Uses the `dynamic_gs` conda env (python 3.12, torch 2.11+cu128, sm_120),
#   which already has open3d / opencv / torch. SAM2 was added with:
#
#     # clone outside the conda env (conda's libffi breaks git https):
#     cd /tmp && env -u LD_LIBRARY_PATH git clone --depth 1 \
#         https://github.com/facebookresearch/sam2.git sam2_src
#     conda activate dynamic_gs
#     # --no-deps protects the existing sm_120 torch build;
#     # SAM2_BUILD_CUDA=0 skips the optional connected-components CUDA ext.
#     SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 \
#         pip install --no-deps --no-build-isolation /tmp/sam2_src
#
#   Checkpoint + config are auto-downloaded from the HF hub on first run
#   (facebook/sam2.1-hiera-large) and cached under ~/.cache/huggingface.
#
#   Run:
#     conda activate dynamic_gs
#     python experiments/sam2_static_mvp/sam2_static_mvp.py
# =============================================================================

import os
import re
import sys
import json
import colorsys

import numpy as np
import cv2
import torch
import open3d as o3d

# ----------------------------------------------------------------------------
# CONFIG  (hardcoded on purpose -- no CLI args)
# ----------------------------------------------------------------------------
DATASET_DIR = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45_w_background/static_scene"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SAM2_HF_MODEL = "facebook/sam2.1-hiera-large"
DEVICE = "cuda"

# point cloud cleanup
VOXEL_SIZE_M = 0.005          # 5 mm voxel downsample
SOR_NB_NEIGHBORS = 20         # open3d remove_statistical_outlier defaults
SOR_STD_RATIO = 2.0

# depth handling
DEPTH_SCALE = 1000.0          # uint16 millimeters -> meters
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0             # observed max ~1.4 m; clip absurd values

# gripper masking: black out the robot gripper (transforms.json mask_path, where
# mask==0) in every RGB frame so SAM2/SAM3 don't segment it, and color its
# points solid black in the cloud.
BLACK_OUT_GRIPPER = True

# automatic mask generation (frame 0)
AMG_POINTS_PER_SIDE = 32
MIN_MASK_AREA_PX = 400        # drop AMG specks
MAX_MASK_AREA_FRAC = 0.60     # drop near-full-frame masks (table/wall) -> stay gray
MAX_MASKS = 32               # cap # of propagated objects (keeps colors distinct + bounds memory)

# flipbook overlay
OVERLAY_ALPHA = 0.55
DRAW_ID_LABELS = True

# SAM2 memory offloading (keeps GPU memory bounded; not perf tuning)
OFFLOAD_VIDEO_TO_CPU = True
OFFLOAD_STATE_TO_CPU = True

VIDEO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "_sam2_video_frames")
FLIPBOOK_DIR = os.path.join(OUTPUT_DIR, "flipbook")
PLY_PATH = os.path.join(OUTPUT_DIR, "mask_pointcloud.ply")


# ----------------------------------------------------------------------------
# Color palette: golden-ratio hue spacing => visually distinct adjacent IDs.
# ID 0 = background / no mask = neutral gray. Same palette is used for both the
# flipbook overlay and the point cloud, so colors match between outputs.
# ----------------------------------------------------------------------------
def color_for_id(k: int) -> tuple:
    if k <= 0:
        return (0.6, 0.6, 0.6)
    h = (k * 0.6180339887498949) % 1.0
    s = 0.55 + 0.35 * ((k * 2) % 3) / 2.0   # jitter sat/val so similar hues separate
    v = 0.75 + 0.25 * ((k * 5) % 2)
    return colorsys.hsv_to_rgb(h, s, v)


PALETTE = np.array([color_for_id(k) for k in range(MAX_MASKS + 1)], dtype=np.float32)  # (K+1, 3) RGB 0..1


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------
def _frame_sort_key(path: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def load_dataset():
    tj = os.path.join(DATASET_DIR, "transforms.json")
    with open(tj) as f:
        meta = json.load(f)
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=int(meta["w"]), h=int(meta["h"]))
    # sort frames by trailing index in filename -> smooth camera motion for SAM2
    frames = sorted(meta["frames"], key=lambda fr: _frame_sort_key(fr["file_path"]))
    print(f"[data] {len(frames)} frames, intrinsics fx={intr['fx']:.2f} "
          f"cx={intr['cx']:.1f} {intr['w']}x{intr['h']}")
    return frames, intr


def abspath(root: str, rel: str) -> str:
    return os.path.join(root, rel.lstrip("./"))


def load_gripper_mask(root: str, fr: dict) -> np.ndarray:
    """True where the gripper is (transforms.json mask is 0 on the gripper)."""
    mp = abspath(root, fr["mask_path"])
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(fr["mask_path"])
    return m == 0


def load_rgb(root: str, fr: dict) -> np.ndarray:
    bgr = cv2.imread(abspath(root, fr["file_path"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(fr["file_path"])
    rgb = bgr[:, :, ::-1].copy()  # -> RGB uint8
    if BLACK_OUT_GRIPPER and fr.get("mask_path"):
        rgb[load_gripper_mask(root, fr)] = 0
    return rgb


def blackout_gripper_seg(seg_ids: np.ndarray, frames: list, root: str) -> np.ndarray:
    """Force gripper pixels to background id 0 in every frame (in place)."""
    if not BLACK_OUT_GRIPPER:
        return seg_ids
    for i, fr in enumerate(frames):
        if fr.get("mask_path"):
            seg_ids[i][load_gripper_mask(root, fr)] = 0
    return seg_ids


def load_depth_m(root: str, fr: dict) -> np.ndarray:
    raw = cv2.imread(abspath(root, fr["depth_file_path"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(fr["depth_file_path"])
    return raw.astype(np.float32) / DEPTH_SCALE


# ----------------------------------------------------------------------------
# SAM2 step 1: automatic masks on frame 0
# ----------------------------------------------------------------------------
def generate_frame0_masks(frame0_rgb: np.ndarray) -> list:
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print("[sam2] loading image model for automatic mask generation ...")
    image_model = build_sam2_hf(SAM2_HF_MODEL, device=DEVICE)
    amg = SAM2AutomaticMaskGenerator(
        image_model,
        points_per_side=AMG_POINTS_PER_SIDE,
        min_mask_region_area=MIN_MASK_AREA_PX,
        output_mode="binary_mask",
    )
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        raw = amg.generate(frame0_rgb)
    print(f"[sam2] automatic generator returned {len(raw)} raw masks")

    H, W = frame0_rgb.shape[:2]
    img_area = H * W
    kept = []
    for m in raw:
        area = int(m["area"])
        if area < MIN_MASK_AREA_PX:
            continue
        if area > MAX_MASK_AREA_FRAC * img_area:   # background blob -> leave as gray
            continue
        # gripper was blacked out -> drop any mask that sits on the black region
        if BLACK_OUT_GRIPPER and float(frame0_rgb[np.asarray(m["segmentation"], bool)].mean()) < 10.0:
            continue
        kept.append(m)
    kept.sort(key=lambda m: m["area"], reverse=True)
    kept = kept[:MAX_MASKS]
    print(f"[sam2] keeping {len(kept)} masks "
          f"(area in [{MIN_MASK_AREA_PX}, {MAX_MASK_AREA_FRAC:.0%} of frame], top {MAX_MASKS})")

    seed_masks = [np.asarray(m["segmentation"], dtype=bool) for m in kept]

    # free the image model before the memory-heavy video propagation
    del amg, image_model
    torch.cuda.empty_cache()
    return seed_masks


# ----------------------------------------------------------------------------
# SAM2 step 2: seed the video predictor and propagate across all frames
# ----------------------------------------------------------------------------
def write_video_frames(frames: list, root: str):
    os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)
    rgbs = []
    for i, fr in enumerate(frames):
        rgb = load_rgb(root, fr)
        rgbs.append(rgb)
        # SAM2's load_video_frames sorts by int(filename stem) and reads RGB JPEGs
        cv2.imwrite(os.path.join(VIDEO_FRAMES_DIR, f"{i:05d}.jpg"), rgb[:, :, ::-1])
    return rgbs


def propagate(seed_masks: list, num_frames: int, H: int, W: int) -> np.ndarray:
    from sam2.build_sam import build_sam2_video_predictor_hf

    print("[sam2] loading video predictor ...")
    predictor = build_sam2_video_predictor_hf(SAM2_HF_MODEL, device=DEVICE)

    # seg_ids[frame] in {0=bg, 1..K}; uint8 is fine for <=255 ids
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
            print(f"[sam2] seeded {len(seed_masks)} objects on frame 0; propagating ...")

            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                logits = mask_logits.squeeze(1).float().cpu().numpy()  # (n_obj, h, w)
                # per-pixel: assign the object with the highest positive logit
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
    except torch.cuda.OutOfMemoryError as e:
        print("\n[FATAL] SAM2 ran out of GPU memory during video propagation.")
        print(f"        {e}")
        print(f"        Try a smaller model (sam2.1-hiera-small / base-plus) or fewer "
              f"masks (MAX_MASKS={MAX_MASKS}).")
        sys.exit(1)

    del predictor
    torch.cuda.empty_cache()
    return seg_ids


# ----------------------------------------------------------------------------
# Output 1: side-by-side flipbook
# ----------------------------------------------------------------------------
def write_flipbook(rgbs: list, seg_ids: np.ndarray):
    os.makedirs(FLIPBOOK_DIR, exist_ok=True)
    color_lut = (PALETTE * 255.0).astype(np.float32)  # (K+1,3) RGB
    a = OVERLAY_ALPHA
    for i, rgb in enumerate(rgbs):
        seg = seg_ids[i]
        color_img = color_lut[seg]                    # (H,W,3) RGB
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

        side = np.concatenate([rgb, overlay], axis=1)         # RGB
        cv2.imwrite(os.path.join(FLIPBOOK_DIR, f"{i:04d}.png"), side[:, :, ::-1])
    print(f"[out] wrote {len(rgbs)} flipbook frames -> {FLIPBOOK_DIR}")


# ----------------------------------------------------------------------------
# Output 2: colored point cloud (mask ID -> color)
# ----------------------------------------------------------------------------
def build_pointcloud(frames: list, root: str, seg_ids: np.ndarray, intr: dict):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # nerfstudio OpenGL c2w -> OpenCV c2w

    pts_all, col_all = [], []
    for i, fr in enumerate(frames):
        z = load_depth_m(root, fr)
        H, W = z.shape
        valid = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
        vv, uu = np.where(valid)
        zz = z[vv, uu]
        # back-project in OpenCV camera frame
        x = (uu - cx) * zz / fx
        y = (vv - cy) * zz / fy
        cam = np.stack([x, y, zz, np.ones_like(zz)], axis=1)         # (N,4)
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64) @ flip
        world = (c2w @ cam.T).T[:, :3]

        seg = seg_ids[i][vv, uu]
        col = PALETTE[seg].copy()
        if BLACK_OUT_GRIPPER and fr.get("mask_path"):
            col[load_gripper_mask(root, fr)[vv, uu]] = 0.0   # gripper -> solid black
        pts_all.append(world.astype(np.float32))
        col_all.append(col)
        if i % 20 == 0 or i == len(frames) - 1:
            print(f"[pcd]   back-projected frame {i + 1}/{len(frames)} "
                  f"({world.shape[0]} pts)")

    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    print(f"[pcd] concatenated {pts.shape[0]:,} points (no fusion)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
    print(f"[pcd] after {VOXEL_SIZE_M*1000:.0f} mm voxel downsample: {len(pcd.points):,} points")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS,
                                            std_ratio=SOR_STD_RATIO)
    print(f"[pcd] after statistical outlier removal: {len(pcd.points):,} points")

    o3d.io.write_point_cloud(PLY_PATH, pcd)
    print(f"[out] wrote point cloud -> {PLY_PATH}")


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available."); sys.exit(1)

    frames, intr = load_dataset()
    H, W = intr["h"], intr["w"]

    print("[data] writing video frames + loading RGB into memory ...")
    rgbs = write_video_frames(frames, DATASET_DIR)

    seed_masks = generate_frame0_masks(rgbs[0])
    if not seed_masks:
        print("[FATAL] automatic mask generator produced no usable masks on frame 0.")
        sys.exit(1)

    seg_ids = propagate(seed_masks, len(frames), H, W)
    blackout_gripper_seg(seg_ids, frames, DATASET_DIR)   # gripper -> id 0 (not overlaid/segmented)

    write_flipbook(rgbs, seg_ids)
    build_pointcloud(frames, DATASET_DIR, seg_ids, intr)

    print("\n[done] outputs in", OUTPUT_DIR)
    print(f"       flipbook : {FLIPBOOK_DIR}/0000.png ... ({len(frames)} frames)")
    print(f"       pointcloud: {PLY_PATH}")


if __name__ == "__main__":
    main()
