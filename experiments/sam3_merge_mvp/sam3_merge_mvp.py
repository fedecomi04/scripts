#!/usr/bin/env python3
# =============================================================================
# SAM3 -> SAM2-ID merge MVP  --  throwaway, follow-up to sam2_static_mvp.
#
# WHAT IT DOES
#   1. Runs SAM3 with hardcoded TEXT PROMPTS on the FIRST frame to find each
#      graspable object instance (two mugs => two distinct targets).
#   2. Re-derives the SAM2 tracked mask IDs (the previous MVP did NOT save raw
#      masks, so we RE-RUN the SAM2 propagation once -- see DESIGN NOTE below --
#      and cache the per-frame IDs to seg_ids.npz so future tweaks skip it).
#   3. Merges over-segmented SAM2 IDs that fall inside the same SAM3 instance:
#      for each SAM3 instance, every SAM2 ID with
#          |sam2_id_mask  ∩  sam3_mask| / |sam2_id_mask|  > COVERAGE_THRESHOLD
#      (measured on frame 0) is grouped together via union-find.
#   4. Rebuilds the colored point cloud from scratch (same backprojection as the
#      previous MVP) using the REMAPPED IDs as the color source, then voxel
#      downsample + statistical outlier removal.
#
#   Outputs (in ./output/):
#     - merged_pointcloud.ply   : cloud colored by merged IDs
#     - remap_log.txt           : old_id -> new_id, + which prompt/instance each
#                                 merged new-id corresponds to, + unique-ID counts
#     - merge_compare.png       : frame 0, left = SAM2 (pre-merge) overlay,
#                                 right = merged (post-merge) overlay
#     - seg_ids.npz             : cached per-frame SAM2 IDs (re-run avoidance)
#
# DESIGN NOTE (per user decision, 2026-05-30): the previous MVP saved only the
#   voxel-downsampled .ply, whose colors are ~39% boundary-blended and recover
#   IDs poorly. So instead of the lossy color-recovery fallback, we RE-RUN the
#   SAM2 propagation once to get clean per-frame IDs and rebuild the cloud. The
#   SAM2 propagation LOGIC/PARAMS are untouched -- we import and call the
#   previous MVP's own functions, guaranteeing identical IDs/palette/geometry.
#
# ISOLATION: imports ONLY the sibling experiment (../sam2_static_mvp), never the
#   main dynamic-gs pipeline. SAM3 is incompatible with the SAM2 env, so it runs
#   in a subprocess (conda run -n sam3_dynamic_gs) via a worker this script
#   writes to ./output/_sam3_worker.py at runtime (keeps everything one file).
#
# -----------------------------------------------------------------------------
# ENVIRONMENT / RUN
#   No install needed. Two existing conda envs are used:
#     - dynamic_gs       : has SAM2 + open3d + torch  -> run THIS script here
#     - sam3_dynamic_gs  : has SAM3                    -> spawned automatically
#
#     conda activate dynamic_gs
#     python experiments/sam3_merge_mvp/sam3_merge_mvp.py
# =============================================================================

import os
import sys
import json
import subprocess
from collections import defaultdict

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# CONFIG  (hardcoded on purpose -- no CLI args)
# ----------------------------------------------------------------------------
# >>> EDIT ME: SAM3 text prompt(s). <<<
# Single GENERIC "objectness" prompt -- segments every human-recognizable object
# WITHOUT naming it. Each returned instance becomes its own merge target.
# Score tiers probed on frame 0 with "object" (see output/_objects_probe/object_grid.png):
#   ~0.25 : the 3 discrete items (can / cube / block)
#   ~0.12 : the FULL table top                       <- captured at ct=0.10
#   <0.08 : noise (table-leg / edge fragments)        <- excluded
# So ct=0.10 keeps {3 items + table}; raise to ~0.18 for items only.
# Background sky/wall is never returned. You may also hand-name objects, e.g.
# ["coke can", "red cube", "blue block"], or use multiple prompts (each is a target).
TEXT_PROMPTS = ["object"]

COVERAGE_THRESHOLD = 0.8            # |sam2 ∩ sam3| / |sam2| on frame 0 to merge
SAM3_CONFIDENCE_THRESHOLD = 0.10    # keeps the 3 items + the table; cuts the noise tail
SAM3_CONDA_ENV = "sam3_dynamic_gs"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
PREV_DIR = os.path.join(os.path.dirname(THIS_DIR), "sam2_static_mvp")   # sibling MVP

SEG_IDS_NPZ = os.path.join(OUTPUT_DIR, "seg_ids.npz")
MERGED_PLY = os.path.join(OUTPUT_DIR, "merged_pointcloud.ply")
REMAP_LOG = os.path.join(OUTPUT_DIR, "remap_log.txt")
COMPARE_PNG = os.path.join(OUTPUT_DIR, "merge_compare.png")
WORKER_PATH = os.path.join(OUTPUT_DIR, "_sam3_worker.py")
SAM3_NPZ = os.path.join(OUTPUT_DIR, "_sam3_masks.npz")
SAM3_META = os.path.join(OUTPUT_DIR, "_sam3_meta.json")
PROMPTS_JSON = os.path.join(OUTPUT_DIR, "_sam3_prompts.json")

REUSE_CACHED_SEG_IDS = True          # skip SAM2 re-run if seg_ids.npz already exists
OVERLAY_ALPHA = 0.55


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
            # fresh set_image per prompt to avoid cross-prompt state contamination
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


# ----------------------------------------------------------------------------
def run_sam3(frame0_png: str):
    """Run SAM3 on frame 0 in its own conda env; return (masks (M,H,W) bool, meta)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(WORKER_PATH, "w") as f:
        f.write(SAM3_WORKER_SRC)
    with open(PROMPTS_JSON, "w") as f:
        json.dump(TEXT_PROMPTS, f)

    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [conda, "run", "--no-capture-output", "-n", SAM3_CONDA_ENV, "python", WORKER_PATH,
           "--image", frame0_png, "--prompts-json", PROMPTS_JSON,
           "--ct", str(SAM3_CONFIDENCE_THRESHOLD),
           "--out-npz", SAM3_NPZ, "--out-json", SAM3_META]
    print(f"[sam3] launching in env '{SAM3_CONDA_ENV}' with prompts {TEXT_PROMPTS} ...")
    subprocess.run(cmd, check=True)

    masks = np.load(SAM3_NPZ)["masks"].astype(bool)          # (M,H,W)
    meta = json.load(open(SAM3_META))
    print(f"[sam3] got {masks.shape[0]} instance mask(s)")
    return masks, meta


# ----------------------------------------------------------------------------
# Merge: SAM2 ids whose frame-0 footprint is mostly inside a SAM3 instance group
# ----------------------------------------------------------------------------
def compute_merge(seg0: np.ndarray, sam3_masks: np.ndarray, meta: dict):
    present_ids = [int(k) for k in np.unique(seg0) if k != 0]
    n_inst = sam3_masks.shape[0]

    # per (sam2 id, instance) coverage = |sam2 ∩ inst| / |sam2|
    best_target = {}            # sam2 id -> instance idx (or None)
    coverage_of = {}            # sam2 id -> best coverage value
    for k in present_ids:
        mk = (seg0 == k)
        ak = int(mk.sum())
        best_t, best_c = None, COVERAGE_THRESHOLD
        for t in range(n_inst):
            cov = float(np.logical_and(mk, sam3_masks[t]).sum()) / max(ak, 1)
            if cov > best_c:
                best_c, best_t = cov, t
        best_target[k] = best_t
        coverage_of[k] = best_c if best_t is not None else 0.0

    # union-find over sam2 ids + per-instance anchors ("T<t>")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k in present_ids:
        find(k)
        if best_target[k] is not None:
            union(k, f"T{best_target[k]}")

    comp = defaultdict(list)
    for k in present_ids:
        comp[find(k)].append(k)

    remap = {0: 0}
    target_for_newid = {}       # new_id -> instance idx (only for merged-into-target groups)
    for ids in comp.values():
        rep = min(ids)
        for k in ids:
            remap[k] = rep
        ts = {best_target[k] for k in ids if best_target[k] is not None}
        if ts:
            target_for_newid[rep] = sorted(ts)[0]   # at most one target per component by construction

    return remap, best_target, coverage_of, target_for_newid, present_ids


# ----------------------------------------------------------------------------
# Overlay rendering (mirrors the previous MVP's flipbook overlay style)
# ----------------------------------------------------------------------------
def overlay(rgb, seg, palette, alpha, title, prompt_for_id=None):
    color_lut = (palette * 255.0).astype(np.float32)
    out = rgb.astype(np.float32).copy()
    fg = seg > 0
    out[fg] = (1.0 - alpha) * out[fg] + alpha * color_lut[seg][fg]
    out = out.astype(np.uint8)
    for oid in np.unique(seg):
        if oid == 0:
            continue
        ys, xs = np.where(seg == oid)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(int(oid))
        if prompt_for_id and int(oid) in prompt_for_id:
            label = f"{int(oid)}:{prompt_for_id[int(oid)]}"
        cv2.putText(out, label, (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_compare(rgb0, seg0, remap, target_for_newid, meta, palette):
    remap_lut = np.arange(int(seg0.max()) + 1, dtype=np.int64)
    for k, v in remap.items():
        if k < len(remap_lut):
            remap_lut[k] = v
    merged_seg0 = remap_lut[seg0].astype(seg0.dtype)

    prompt_for_id = {nid: meta["instances"][t]["prompt"] for nid, t in target_for_newid.items()
                     if t < len(meta["instances"])}
    left = overlay(rgb0, seg0, palette, OVERLAY_ALPHA, "SAM2 pre-merge")
    right = overlay(rgb0, merged_seg0, palette, OVERLAY_ALPHA, "merged post-merge",
                    prompt_for_id=prompt_for_id)
    side = np.concatenate([left, right], axis=1)
    cv2.imwrite(COMPARE_PNG, side[:, :, ::-1])
    print(f"[out] wrote {COMPARE_PNG}")


def write_remap_log(remap, best_target, coverage_of, target_for_newid, meta, present_ids):
    before = len(present_ids) + 1                      # + background
    after = len(set(remap.values()))                   # includes background (0)
    lines = []
    lines.append("SAM3 -> SAM2 merge remap log")
    lines.append(f"coverage_threshold = {COVERAGE_THRESHOLD}")
    lines.append(f"sam3_confidence_threshold = {SAM3_CONFIDENCE_THRESHOLD}")
    lines.append(f"prompts = {meta['prompts']}")
    lines.append("")
    lines.append(f"unique IDs before merge: {before}")
    lines.append(f"unique IDs after  merge: {after}")
    lines.append(f"IDs eliminated by merge: {before - after}")
    lines.append("")
    lines.append("SAM3 instances:")
    for t, inst in enumerate(meta["instances"]):
        lines.append(f"  target T{t}: prompt='{inst['prompt']}' instance#{inst['inst']} "
                     f"score={inst['score']:.3f} area={inst['area']}")
    if not meta["instances"]:
        lines.append("  (none)")
    lines.append("")
    lines.append("Merged groups (new_id <- which SAM3 target, + member sam2 ids):")
    groups = defaultdict(list)
    for old, new in remap.items():
        if old == 0:
            continue
        groups[new].append(old)
    for new in sorted(groups):
        members = sorted(groups[new])
        if new in target_for_newid:
            t = target_for_newid[new]
            inst = meta["instances"][t]
            tag = f"<- SAM3 T{t} '{inst['prompt']}' inst#{inst['inst']}"
        else:
            tag = "(unmatched, kept original id)"
        merged_flag = "  *MERGED*" if len(members) > 1 else ""
        lines.append(f"  new_id {new:3d} {tag}: sam2 ids {members}{merged_flag}")
    lines.append("")
    lines.append("Full remap table (old_id -> new_id  [coverage into its target]):")
    for old in sorted(k for k in remap if k != 0):
        new = remap[old]
        bt = best_target[old]
        cov = f"cov={coverage_of[old]:.3f} -> T{bt}" if bt is not None else "no target"
        lines.append(f"  {old:3d} -> {new:3d}   ({cov})")
    with open(REMAP_LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] wrote {REMAP_LOG}  (IDs {before} -> {after})")


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # import the sibling SAM2 MVP for identical AMG / palette / propagation / geometry
    sys.path.insert(0, PREV_DIR)
    import sam2_static_mvp as prev
    # redirect the previous MVP's file I/O into THIS experiment's output folder
    prev.VIDEO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "_sam2_video_frames")
    prev.PLY_PATH = MERGED_PLY

    frames, intr = prev.load_dataset()
    H, W = intr["h"], intr["w"]
    root = prev.DATASET_DIR

    # frame-0 RGB (gripper blacked out when prev.BLACK_OUT_GRIPPER) -> SAM3 input + overlays
    rgb0 = prev.load_rgb(root, frames[0])
    frame0_for_sam3 = os.path.join(OUTPUT_DIR, "_frame0_input.png")
    cv2.imwrite(frame0_for_sam3, rgb0[:, :, ::-1])

    # ---- SAM3 first (fast, fail-fast on env issues) ----
    sam3_masks, meta = run_sam3(frame0_for_sam3)

    # ---- SAM2 per-frame IDs: reuse cache or re-run propagation once ----
    if REUSE_CACHED_SEG_IDS and os.path.exists(SEG_IDS_NPZ):
        seg_ids = np.load(SEG_IDS_NPZ)["seg_ids"]
        print(f"[sam2] loaded cached seg_ids {seg_ids.shape} from {SEG_IDS_NPZ}")
    else:
        print("[sam2] writing video frames + loading RGB ...")
        rgbs = prev.write_video_frames(frames, root)
        seed_masks = prev.generate_frame0_masks(rgbs[0])
        if not seed_masks:
            print("[FATAL] SAM2 produced no usable masks on frame 0."); sys.exit(1)
        seg_ids = prev.propagate(seed_masks, len(frames), H, W)
        prev.blackout_gripper_seg(seg_ids, frames, root)
        np.savez_compressed(SEG_IDS_NPZ, seg_ids=seg_ids)
        print(f"[sam2] cached seg_ids -> {SEG_IDS_NPZ}")

    prev.blackout_gripper_seg(seg_ids, frames, root)   # idempotent; cleans stale caches too
    seg0 = seg_ids[0]

    # ---- merge ----
    if sam3_masks.shape[0] == 0:
        print("[merge] SAM3 returned no instances -> no merges; cloud == previous MVP.")
    remap, best_target, coverage_of, target_for_newid, present_ids = compute_merge(
        seg0, sam3_masks, meta)

    # ---- outputs: log + side-by-side + rebuilt cloud ----
    write_remap_log(remap, best_target, coverage_of, target_for_newid, meta, present_ids)
    write_compare(rgb0, seg0, remap, target_for_newid, meta, prev.PALETTE)

    # apply remap to every frame, then rebuild the cloud with the previous MVP's logic
    remap_lut = np.arange(int(seg_ids.max()) + 1, dtype=seg_ids.dtype)
    for k, v in remap.items():
        if k < len(remap_lut):
            remap_lut[k] = v
    merged_seg_ids = remap_lut[seg_ids]
    print("[pcd] rebuilding point cloud with merged IDs ...")
    prev.build_pointcloud(frames, root, merged_seg_ids, intr)

    print("\n[done] outputs in", OUTPUT_DIR)
    print(f"       {os.path.basename(MERGED_PLY)}, {os.path.basename(REMAP_LOG)}, "
          f"{os.path.basename(COMPARE_PNG)}")


if __name__ == "__main__":
    main()
