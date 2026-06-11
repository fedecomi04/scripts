"""FastSAM + CLIP text-prompted segmentation — a lightweight drop-in for SAM3.

FastSAM (Ultralytics YOLOv8-seg) produces *class-agnostic* instance masks for
the whole image in one forward pass; CLIP (open-clip) then scores each mask crop
against the text prompt. Unlike FastSAM's built-in ``text_prompt`` (top-1 only),
this keeps ALL candidate masks with their CLIP scores so the SAME
area/border/dedup/max_objects filters SAM3 uses apply unchanged. The output
contract is byte-identical to ``sam_worker.sam3_infer`` /
``sam3_segmentation.run_sam3_segmentation``:

    objects = [{mask_path, score, bbox(xyxy), mask_area, object_index}, ...]

and ``infer_raw`` writes the same ``<stem>_raw_masks.npz`` ({masks, scores, boxes}).

Why: SAM3 holds ~3.8 GB resident; FastSAM-x + CLIP ViT-B-32 are far lighter, so
segmentation can co-reside with SAM3D (~12 GB) and SAM3D can load from the start
of the static pipeline. CLIP-on-crop text matching is weaker than SAM3's
dedicated grounding — quality is gated by scripts/compare_sam3_fastsam.py.

Runs inside the ``sam3_dynamic_gs`` env (ultralytics + open-clip-torch installed
there). Imports torch/ultralytics/open_clip lazily so importing this module on
the parent (dynamic_gs) side is cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_CONDA_ROOT = Path(os.environ.get("DYNAMIC_GS_CONDA_ROOT", str(Path.home() / "miniconda3")))


def _resolve_env_python(conda_env: str):
    py = _CONDA_ROOT / "envs" / conda_env / "bin" / "python"
    return py if py.exists() else None


# --- mask filter helpers (inlined to avoid importing the package __init__,
#     which pulls nerfstudio — absent in the sam3 env) -----------------------

def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def _touches_n_borders(mask: np.ndarray, n: int = 2) -> bool:
    b = 0
    if np.any(mask[0, :]):
        b += 1
    if np.any(mask[-1, :]):
        b += 1
    if np.any(mask[:, 0]):
        b += 1
    if np.any(mask[:, -1]):
        b += 1
    return b >= n


def _default_weights_path(weights: str) -> str:
    """Resolve a bare weights name to a stable cache path so ultralytics
    downloads once and reuses it (instead of dumping into cwd)."""
    p = Path(weights)
    if p.is_absolute() or p.exists():
        return str(p)
    cache = Path.home() / ".cache" / "dynamic_gs" / "fastsam"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache / p.name)


class FastSamTextSegmenter:
    """Persistent FastSAM + CLIP segmenter. Construct once, call ``infer`` /
    ``infer_raw`` many times (warm)."""

    def __init__(self,
                 weights: str = "FastSAM-x.pt",
                 clip_model: str = "ViT-B-32-quickgelu",
                 clip_pretrained: str = "openai",
                 device: str = "cuda",
                 clip_logit_scale: float = 100.0) -> None:
        # NOTE: OpenAI CLIP weights were trained with QuickGELU; use the
        # "-quickgelu" model variant or open_clip warns + silently degrades
        # match quality (wrong activation). This matters for the text gate.
        import torch
        from ultralytics import FastSAM
        import open_clip

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_logit_scale = float(clip_logit_scale)

        self.model = FastSAM(_default_weights_path(weights))
        # ultralytics keeps its own device handling; we pass device per-call.

        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip = self.clip.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)

    # -- core --------------------------------------------------------------

    def _run_fastsam(self,
                     image_path: str,
                     conf: float,
                     iou: float,
                     imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (masks (N,H,W) bool at native resolution, boxes (N,4) xyxy)."""
        results = self.model(
            str(image_path), device=self.device, retina_masks=True,
            imgsz=imgsz, conf=conf, iou=iou, verbose=False,
        )
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            # fall back to native HxW from the image
            with Image.open(image_path) as im:
                W, H = im.size
            return np.zeros((0, H, W), dtype=bool), np.zeros((0, 4), dtype=np.float32)
        masks = (r.masks.data.detach().cpu().numpy() > 0.5)
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32) if r.boxes is not None else \
            np.zeros((masks.shape[0], 4), dtype=np.float32)
        return masks, boxes

    def _clip_scores(self,
                     image_rgb: np.ndarray,
                     masks: np.ndarray,
                     text_prompt: str) -> np.ndarray:
        """Per-mask CLIP match score in [0,1] (softmax over crops, FastSAM-style).

        Each mask's bbox is cropped, background within the bbox is whited-out so
        CLIP sees the object, then crops are encoded and matched to the text.
        Softmax over crops makes the best-matching object dominant."""
        torch = self.torch
        N = masks.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=np.float32)
        crops = []
        valid_idx = []
        for i in range(N):
            m = masks[i]
            ys, xs = np.nonzero(m)
            if xs.size == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            sub = image_rgb[y0:y1, x0:x1].copy()
            subm = m[y0:y1, x0:x1]
            sub[~subm] = 255  # white-out background inside the bbox
            crops.append(self.clip_preprocess(Image.fromarray(sub)))
            valid_idx.append(i)
        scores = np.zeros((N,), dtype=np.float32)
        if not crops:
            return scores
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            img_f = self.clip.encode_image(batch)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt = self.clip_tokenizer([text_prompt]).to(self.device)
            txt_f = self.clip.encode_text(txt)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sims = (img_f @ txt_f.T).squeeze(-1)  # cosine, (Nvalid,)
            probs = (self.clip_logit_scale * sims).softmax(dim=0).float().cpu().numpy()
        for j, i in enumerate(valid_idx):
            scores[i] = float(probs[j])
        return scores

    # -- public API (mirrors sam_worker.sam3_infer / sam3_infer_raw) --------

    def infer(self, *,
              image_path: str,
              text_prompt: str,
              output_dir: str,
              output_stem: str,
              min_area_ratio: float = 0.002,
              max_area_ratio: float = 0.25,
              dedup_iou: float = 0.6,
              max_objects: int = 8,
              min_score: float = 0.2,
              fastsam_conf: float = 0.4,
              fastsam_iou: float = 0.9,
              imgsz: int = 1024) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image)
        image_area = image.width * image.height

        masks, boxes = self._run_fastsam(image_path, fastsam_conf, fastsam_iou, imgsz)
        scores = self._clip_scores(image_rgb, masks, text_prompt)

        candidates = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8)
            area = int(m.sum())
            if area == 0:
                continue
            if area < min_area_ratio * image_area or area > max_area_ratio * image_area:
                continue
            if _touches_n_borders(m, n=2):
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < min_score:
                continue
            bbox = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
            candidates.append({"mask": m, "score": score, "bbox": bbox, "mask_area": area})

        candidates.sort(key=lambda c: c["score"], reverse=True)
        deduped: List[Dict[str, Any]] = []
        for c in candidates:
            if any(_compute_iou(c["mask"], k["mask"]) > dedup_iou for k in deduped):
                continue
            deduped.append(c)
        deduped = deduped[:max_objects]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for obj_idx, c in enumerate(deduped):
            mask_path = out_dir / f"{output_stem}_obj_{obj_idx:02d}_mask.png"
            Image.fromarray(c["mask"] * 255).save(mask_path)
            results.append({
                "mask_path": str(mask_path),
                "score": c["score"],
                "bbox": c["bbox"],
                "mask_area": c["mask_area"],
                "object_index": obj_idx,
            })

        # Reuse the SAM3 summary filename so downstream load_sam3_masks +
        # sam3_reuse_cached work unchanged; tag the backend so the artifact is
        # self-describing.
        summary_path = out_dir / f"{output_stem}_sam3_results.json"
        summary_path.write_text(json.dumps({
            "image_path": str(image_path),
            "text_prompt": text_prompt,
            "segmentation_backend": "fastsam",
            "total_raw_masks": int(masks.shape[0]),
            "total_after_filtering": len(results),
            "filter_params": {
                "min_area_ratio": min_area_ratio, "max_area_ratio": max_area_ratio,
                "dedup_iou": dedup_iou, "max_objects": max_objects, "min_score": min_score,
                "fastsam_conf": fastsam_conf, "fastsam_iou": fastsam_iou, "imgsz": imgsz,
            },
            "objects": results,
        }, indent=2) + "\n")

        return results

    def infer_raw(self, *,
                  image_path: str,
                  text_prompt: str,
                  output_dir: str,
                  output_stem: str,
                  min_score: float = 0.0,
                  fastsam_conf: float = 0.4,
                  fastsam_iou: float = 0.9,
                  imgsz: int = 1024) -> Dict[str, Any]:
        """Write all masks with CLIP score >= min_score to an NPZ, no other filters.

        NOTE: with min_score=0.0 this keeps the FULL class-agnostic FastSAM set
        (every instance), which is what the preseg AMG-merge path expects; raise
        min_score to keep only prompt-matched instances."""
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image)
        masks, boxes = self._run_fastsam(image_path, fastsam_conf, fastsam_iou, imgsz)
        scores = self._clip_scores(image_rgb, masks, text_prompt)

        keep = [i for i in range(masks.shape[0])
                if (float(scores[i]) if i < len(scores) else 0.0) >= min_score]
        if keep:
            kept_masks = masks[keep].astype(np.bool_)
            kept_scores = scores[keep].astype(np.float32)
            kept_boxes = boxes[keep].astype(np.float32) if len(boxes) >= (max(keep) + 1) \
                else np.zeros((len(keep), 4), dtype=np.float32)
        else:
            H, W = (masks.shape[1], masks.shape[2]) if masks.ndim == 3 else (image.height, image.width)
            kept_masks = np.zeros((0, H, W), dtype=np.bool_)
            kept_scores = np.zeros((0,), dtype=np.float32)
            kept_boxes = np.zeros((0, 4), dtype=np.float32)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        masks_path = out_dir / f"{output_stem}_raw_masks.npz"
        np.savez(str(masks_path), masks=kept_masks, scores=kept_scores, boxes=kept_boxes)
        return {"masks_path": str(masks_path), "num_masks": int(kept_masks.shape[0])}


# ---------------------------------------------------------------------------
# Subprocess launcher (runs in the training env; spawns the sam3 env) +  CLI.
# Mirrors sam3_segmentation.run_sam3_subprocess so phase0a can branch backends
# with a one-line swap. The persistent SamWorkerClient path is preferred for
# the live/orchestrated flow (FastSAM + SAM3D co-resident); this subprocess is
# the simple recorded-flow path and the no-worker fallback.
# ---------------------------------------------------------------------------


def run_fastsam_subprocess(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    output_stem: str,
    sam3_conda_env: str = "sam3_dynamic_gs",
    **filter_kwargs,
) -> List[Dict]:
    """Launch FastSAM+CLIP segmentation in the sam3 env. Returns the same
    ``{mask_path, score, bbox, mask_area, object_index}`` contract as SAM3."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_python = _resolve_env_python(sam3_conda_env)
    if env_python is not None:
        command = [str(env_python), str(Path(__file__).resolve())]
    else:
        command = ["conda", "run", "--no-capture-output", "-n", sam3_conda_env,
                   "python", str(Path(__file__).resolve())]
    command += [
        "--image", str(image_path),
        "--text-prompt", text_prompt,
        "--output-dir", str(output_dir),
        "--output-stem", output_stem,
    ]
    # Only forward kwargs the CLI understands (drop SAM3-only keys like
    # confidence_threshold so the same phase0a call site works for both).
    _cli_keys = {"min_area_ratio", "max_area_ratio", "dedup_iou", "max_objects",
                 "min_score", "fastsam_conf", "fastsam_iou", "imgsz",
                 "fastsam_weights", "clip_model", "clip_pretrained"}
    for key, value in filter_kwargs.items():
        if key not in _cli_keys:
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])

    sub_env = os.environ.copy()
    if env_python is not None:
        env_lib = str(env_python.parent.parent / "lib")
        sub_env["LD_LIBRARY_PATH"] = (env_lib + ":" + sub_env.get("LD_LIBRARY_PATH", "")).rstrip(":")
        sub_env["PYTHONNOUSERSITE"] = "1"
        sub_env.setdefault("CONDA_PREFIX", str(env_python.parent.parent))

    completed = subprocess.run(
        command, cwd=str(Path(__file__).resolve().parents[2]),
        env=sub_env, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FastSAM subprocess failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    summary_path = output_dir / f"{output_stem}_sam3_results.json"
    if not summary_path.exists():
        raise RuntimeError(f"FastSAM subprocess completed but results JSON not found: {summary_path}")
    return json.loads(summary_path.read_text()).get("objects", [])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastSAM+CLIP text-prompted segmentation worker")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text-prompt", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-stem", type=str, required=True)
    p.add_argument("--min-area-ratio", type=float, default=0.002)
    p.add_argument("--max-area-ratio", type=float, default=0.25)
    p.add_argument("--dedup-iou", type=float, default=0.6)
    p.add_argument("--max-objects", type=int, default=8)
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--fastsam-conf", type=float, default=0.4)
    p.add_argument("--fastsam-iou", type=float, default=0.9)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--fastsam-weights", type=str, default="FastSAM-x.pt")
    p.add_argument("--clip-model", type=str, default="ViT-B-32-quickgelu")
    p.add_argument("--clip-pretrained", type=str, default="openai")
    return p.parse_args()


def _main() -> int:
    a = _parse_args()
    try:
        import time as _time
        _t0 = _time.time()
        seg = FastSamTextSegmenter(weights=a.fastsam_weights, clip_model=a.clip_model,
                                   clip_pretrained=a.clip_pretrained)
        _t_load = _time.time() - _t0
        _t1 = _time.time()
        seg.infer(
            image_path=str(a.image), text_prompt=a.text_prompt,
            output_dir=str(a.output_dir), output_stem=a.output_stem,
            min_area_ratio=a.min_area_ratio, max_area_ratio=a.max_area_ratio,
            dedup_iou=a.dedup_iou, max_objects=a.max_objects, min_score=a.min_score,
            fastsam_conf=a.fastsam_conf, fastsam_iou=a.fastsam_iou, imgsz=a.imgsz,
        )
        # Timing sidecar (load vs infer split) for the unified ledger.
        try:
            (Path(a.output_dir) / "_fastsam_timing.json").write_text(
                json.dumps({"load": _t_load, "infer": _time.time() - _t1}) + "\n")
        except Exception:
            pass
        return 0
    except Exception as exc:
        import traceback
        print(f"FastSAM worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
