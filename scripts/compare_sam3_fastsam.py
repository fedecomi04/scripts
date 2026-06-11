"""Quality gate: SAM3 vs FastSAM text-prompted masks on the last static frame.

Runs IN the sam3_dynamic_gs env. Reproduces phase0a's input exactly (last
static frame, gripper black-out via masks/<frame>.png keep-mask), runs BOTH
backends with the same prompt + filter params, and reports:
  - per-backend mask count + top-1 score/area/bbox
  - IoU between the two top-1 masks
  - greedy best-match IoU across ALL candidates
  - an overlay PNG (SAM3=green, FastSAM=red, overlap=yellow)
  - PASS/FAIL vs --iou-threshold (default 0.75 = "almost the same").

Usage:
  <sam3_env_python> scripts/compare_sam3_fastsam.py \
      --data <dataset>/static_scene --prompt screwdriver
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

_THIS = Path(__file__).resolve()
_UTILS = _THIS.parents[1] / "dynamic_gs" / "utils"


def _load_module(name: str):
    path = _UTILS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_cmp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def _load_mask_png(p) -> np.ndarray:
    return np.array(Image.open(p).convert("L")) > 127


def _gripper_blacked_frame(data: Path, image_path: Path) -> Path:
    """Replicate phase0a's gripper black-out (masks/<stem>.png is a KEEP mask)."""
    rgb = np.array(Image.open(image_path).convert("RGB"))
    mask_path = data / "masks" / (image_path.stem + ".png")
    if mask_path.exists():
        keep = np.array(Image.open(mask_path).convert("L")) > 127
        if keep.shape != rgb.shape[:2]:
            keep = np.array(Image.fromarray(keep.astype(np.uint8) * 255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
        rgb = rgb.copy()
        rgb[~keep] = 0
    out = data / "_vram_scratch" / "compare_input.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    return out


def run_sam3(image_path: Path, prompt: str, out_dir: Path, **flt):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    proc = Sam3Processor(model, confidence_threshold=flt.get("confidence_threshold", 0.1))
    image = Image.open(image_path).convert("RGB")
    image_area = image.width * image.height
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(image)
        out = proc.set_text_prompt(state=state, prompt=prompt)
    masks = out["masks"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy().reshape(-1)
    boxes = out["boxes"].float().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    objs = _filter(masks > 0.5, scores, boxes, image_area, **flt)
    del proc, model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return objs


def run_fastsam(image_path: Path, prompt: str, out_dir: Path, **flt):
    fsm = _load_module("fastsam_segmentation")
    seg = fsm.FastSamTextSegmenter()
    image = Image.open(image_path).convert("RGB")
    image_area = image.width * image.height
    masks, boxes = seg._run_fastsam(str(image_path), flt.get("fastsam_conf", 0.4),
                                    flt.get("fastsam_iou", 0.9), flt.get("imgsz", 1024))
    scores = seg._clip_scores(np.array(image), masks, prompt)
    objs = _filter(masks, scores, boxes, image_area, **flt)
    del seg
    import gc; gc.collect(); torch.cuda.empty_cache()
    return objs


def _filter(masks_bool, scores, boxes, image_area, *,
            min_area_ratio=0.002, max_area_ratio=0.25, dedup_iou=0.6,
            max_objects=8, min_score=0.2, **_ignore):
    cands = []
    for i in range(masks_bool.shape[0]):
        m = masks_bool[i].astype(np.uint8)
        area = int(m.sum())
        if area == 0 or area < min_area_ratio * image_area or area > max_area_ratio * image_area:
            continue
        # border filter (>=2 borders)
        b = int(np.any(m[0])) + int(np.any(m[-1])) + int(np.any(m[:, 0])) + int(np.any(m[:, -1]))
        if b >= 2:
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        if score < min_score:
            continue
        bbox = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
        cands.append({"mask": m, "score": score, "bbox": bbox, "area": area})
    cands.sort(key=lambda c: c["score"], reverse=True)
    out = []
    for c in cands:
        if any(_iou(c["mask"], k["mask"]) > dedup_iou for k in out):
            continue
        out.append(c)
    return out[:max_objects]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="screwdriver")
    ap.add_argument("--iou-threshold", type=float, default=0.75)
    ap.add_argument("--min-score", type=float, default=0.2)
    args = ap.parse_args()

    frames = sorted((args.data / "rgb").glob("*.png"))
    image_path = frames[-1]
    inp = _gripper_blacked_frame(args.data, image_path)
    out_dir = args.data / "_vram_scratch"
    print(f"[compare] frame={image_path.name}  prompt='{args.prompt}'  input={inp}", flush=True)

    print("[compare] running SAM3 ...", flush=True)
    sam3 = run_sam3(inp, args.prompt, out_dir, min_score=args.min_score)
    print("[compare] running FastSAM ...", flush=True)
    fast = run_fastsam(inp, args.prompt, out_dir, min_score=args.min_score)

    def desc(name, objs):
        print(f"\n[{name}] {len(objs)} masks after filtering", flush=True)
        for j, o in enumerate(objs[:5]):
            print(f"  #{j} score={o['score']:.3f} area={o['area']} bbox={[round(x,1) for x in o['bbox']]}", flush=True)
    desc("SAM3", sam3)
    desc("FastSAM", fast)

    if not sam3 or not fast:
        print(f"\n=== GATE: FAIL — a backend returned 0 masks (SAM3={len(sam3)}, FastSAM={len(fast)}) ===", flush=True)
        return 0

    top_iou = _iou(sam3[0]["mask"], fast[0]["mask"])
    # greedy best-match across all candidates
    best = []
    for s in sam3:
        ious = [_iou(s["mask"], f["mask"]) for f in fast]
        best.append(max(ious) if ious else 0.0)
    mean_best = float(np.mean(best))

    H, W = sam3[0]["mask"].shape
    base = np.array(Image.open(inp).convert("RGB"))
    overlay = base.copy().astype(np.float32)
    s0 = sam3[0]["mask"] > 0
    f0 = fast[0]["mask"] > 0
    overlay[s0] = 0.5 * overlay[s0] + 0.5 * np.array([0, 255, 0])    # SAM3 green
    overlay[f0] = 0.5 * overlay[f0] + 0.5 * np.array([255, 0, 0])    # FastSAM red
    overlay[s0 & f0] = 0.5 * base[s0 & f0] + 0.5 * np.array([255, 255, 0])  # overlap yellow
    ov_path = out_dir / "compare_overlay.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(ov_path)

    print(f"\n==================== SAM3 vs FastSAM ====================", flush=True)
    print(f"top-1 IoU              = {top_iou:.3f}", flush=True)
    print(f"SAM3->FastSAM best IoU = {mean_best:.3f} (mean over {len(sam3)} SAM3 masks)", flush=True)
    print(f"overlay -> {ov_path}", flush=True)
    verdict = "PASS" if top_iou >= args.iou_threshold else "FAIL"
    print(f"=== GATE: {verdict} (top-1 IoU {top_iou:.3f} vs threshold {args.iou_threshold}) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
