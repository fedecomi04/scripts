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


def select_kept_indices(
    probs: np.ndarray,
    cosines: np.ndarray,
    *,
    min_ratio: float = 2.5,
    mad_k: float = 3.0,
    margin_min: float = 0.04,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Automatic replacement for the hardcoded ``score < min_score`` cut.

    Two decisions in two score spaces (a single threshold cannot do both):
      (A) HOW MANY objects -> largest consecutive log-prob (== ratio) gap in the
          softmax ``probs``. A real winner->tail jump is a >=``min_ratio`` ratio
          (measured 13-40x across scenes) vs <2.5x tail-internal steps; the ratio
          (log) space is N-invariant where the raw-prob gap is not (a raw-prob
          gap mis-cuts uneven multi-object sets).
      (B) WHETHER ANY object exists -> a raw-COSINE presence gate. Softmax discards
          absolute scale (a flat zero-object field can softmax to a HIGHER max than
          a weak true match), so "is anything here at all" can only be judged on raw
          cosine: keep cos >= median + mad_k*1.4826*MAD AND cos - median >= margin_min.
          The second (MAD-invariant) clause kills homogeneous-background phantoms
          where MAD->0 explodes a chance specular patch.

    Operates on the survivors of the area/border filter, with ``probs`` re-softmaxed
    over just those survivors. Returns (kept_indices_sorted, diagnostics).
    """
    n = int(probs.shape[0])
    diag = {"n": float(n), "cliff_ratio": 0.0, "cos_floor": 0.0, "top_margin": 0.0}
    if n == 0:
        return np.empty(0, dtype=np.intp), diag
    order = np.argsort(-probs, kind="stable")  # descending
    s = probs[order]

    # --- (A) "how many": largest consecutive log-prob (== ratio) gap ---
    if n >= 2:
        log_s = np.log(s + eps)
        gaps = log_s[:-1] - log_s[1:]            # >= 0 (descending)
        k = int(np.argmax(gaps))
        diag["cliff_ratio"] = float(np.exp(gaps[k]))
        cliff_keep = order[: k + 1] if gaps[k] >= np.log(min_ratio) else order[:0]
    else:
        cliff_keep = order[:1]

    # --- (B) "whether any": raw-cosine presence gate (per-scene, N-invariant) ---
    med = float(np.median(cosines))
    mad = float(np.median(np.abs(cosines - med)))
    cos_floor = med + mad_k * (1.4826 * mad)
    diag["cos_floor"] = float(cos_floor)
    diag["top_margin"] = float(np.max(cosines) - med)

    # If the cliff produced nothing (n==1 or no real cliff) fall back to gating the
    # single top candidate, so a lone clean object with a weak softmax (large N) can
    # still survive on its absolute cosine.
    cand = cliff_keep if cliff_keep.size else order[:1]
    passes = (cosines[cand] >= cos_floor) & ((cosines[cand] - med) >= margin_min)
    return np.sort(cand[passes]), diag


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
                     text_prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Per-mask (softmax_prob, raw_cosine) vs the text prompt.

        Each mask's bbox is cropped, background within the bbox is whited-out so
        CLIP sees the object, then crops are encoded and matched to the text.
        Softmax over crops makes the best-matching object dominant; the RAW cosine
        is returned alongside because the automatic threshold needs the absolute,
        N-invariant cosine for its presence gate (softmax discards that scale)."""
        torch = self.torch
        N = masks.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
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
        cosines = np.zeros((N,), dtype=np.float32)
        if not crops:
            return scores, cosines
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            img_f = self.clip.encode_image(batch)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt = self.clip_tokenizer([text_prompt]).to(self.device)
            txt_f = self.clip.encode_text(txt)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sims = (img_f @ txt_f.T).squeeze(-1)  # cosine, (Nvalid,)
            probs = (self.clip_logit_scale * sims).softmax(dim=0).float().cpu().numpy()
            sims_np = sims.float().cpu().numpy()
        for j, i in enumerate(valid_idx):
            scores[i] = float(probs[j])
            cosines[i] = float(sims_np[j])
        return scores, cosines

    def _split_into_components(self, masks: np.ndarray, min_area_px: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Split each FastSAM instance mask into its connected components (each
        >= ``min_area_px``) so CLIP scores every object independently.

        A single FastSAM mask can span two DISJOINT objects (measured: a
        'screwdriver' mask carrying a 2716px blob on a neighbouring object, 19%
        of the mask). Bundled into one crop, the dominant object's appearance
        carries the contaminant past CLIP. Split, the contaminant becomes its own
        candidate that scores low on its own and the threshold drops it.
        Single-component masks pass through unchanged (one component == itself)."""
        import cv2
        H, W = masks.shape[1], masks.shape[2]
        comp_masks: List[np.ndarray] = []
        comp_boxes: List[List[int]] = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8)
            ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for c in range(1, ncomp):  # 0 is background
                if stats[c, cv2.CC_STAT_AREA] < min_area_px:
                    continue
                cm = lab == c
                ys, xs = np.nonzero(cm)
                comp_masks.append(cm)
                comp_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        if not comp_masks:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0, 4), dtype=np.float32)
        return np.array(comp_masks, dtype=bool), np.array(comp_boxes, dtype=np.float32)

    def _save_components_debug(self, image_rgb: np.ndarray, masks: np.ndarray,
                               cosines: np.ndarray, surv_set: set, keep_set: set,
                               out_path: "Path", topk: int = 20) -> None:
        """Overlay the top-K post-split components (by CLIP cosine) on the RGB,
        each colored + labeled ``rank:cosX.XX:<KEPT|surv|filt>``, so a partial
        mask (object split into multiple components, only the best KEPT) is
        visible at a glance. KEPT = made the final cut; surv = passed the
        area/border filter but lost the score gate; filt = filtered earlier."""
        import cv2
        if masks.shape[0] == 0:
            return
        order = np.argsort(-np.asarray(cosines, dtype=np.float64))[:topk]
        palette = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
                   (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
        vis = image_rgb.copy()
        for rank, i in enumerate(order.tolist()):
            m = masks[i].astype(bool)
            if not m.any():
                continue
            col = palette[rank % len(palette)]
            vis[m] = (0.55 * vis[m] + 0.45 * np.array(col)).astype(np.uint8)
            ys, xs = np.where(m)
            x0, y0 = int(xs.min()), int(ys.min())
            tag = "KEPT" if i in keep_set else ("surv" if i in surv_set else "filt")
            cv2.putText(vis, f"{rank}:cos{float(cosines[i]):.2f}:{tag}",
                        (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        Image.fromarray(vis).save(str(out_path))
        # Machine-readable sidecar: per-component raw cosine + area + bbox + tag,
        # so the merge-by-cosine decision can be judged on numbers (the PNG labels
        # are hard to read in a busy scene).
        try:
            rows = []
            for rank, i in enumerate(order.tolist()):
                m = masks[i].astype(bool)
                if not m.any():
                    continue
                ys, xs = np.where(m)
                rows.append({"rank": rank, "cosine": round(float(cosines[i]), 4),
                             "area": int(m.sum()),
                             "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                             "tag": "KEPT" if i in keep_set else ("surv" if i in surv_set else "filt")})
            Path(str(out_path).replace(".png", ".json")).write_text(json.dumps(rows, indent=2))
        except Exception:
            pass

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
              imgsz: int = 1024,
              split_components: bool = True,
              auto_threshold: bool = True,
              auto_min_ratio: float = 2.5,
              auto_mad_k: float = 3.0,
              auto_margin_min: float = 0.04,
              promote_to_container: bool = True,
              promote_contain_frac: float = 0.8,
              promote_cos_margin: float = 0.03) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image)
        image_area = image.width * image.height

        masks, boxes = self._run_fastsam(image_path, fastsam_conf, fastsam_iou, imgsz)
        n_raw = int(masks.shape[0])
        # Split disjoint components so CLIP scores each object on its own (a single
        # FastSAM mask can merge two objects; see _split_into_components).
        if split_components and masks.shape[0]:
            masks, boxes = self._split_into_components(masks, min_area_ratio * image_area)

        scores, cosines = self._clip_scores(image_rgb, masks, text_prompt)

        # Survivors of the area/border filter — the set the threshold judges.
        surv = []
        for i in range(masks.shape[0]):
            area = int(masks[i].sum())
            if area == 0:
                continue
            if area < min_area_ratio * image_area or area > max_area_ratio * image_area:
                continue
            if _touches_n_borders(masks[i].astype(np.uint8), n=2):
                continue
            surv.append(i)

        diag: Dict[str, float] = {}
        surv_score: Dict[int, float] = {}
        if not surv:
            keep: List[int] = []
        elif auto_threshold:
            cos_s = np.asarray([cosines[i] for i in surv], dtype=np.float64)
            # Re-softmax over survivors only: _clip_scores softmaxes over the RAW
            # FastSAM set (often 100-300 masks), so its probs do NOT sum to 1 over
            # the survivors. Re-normalise so the cliff is measured on what's judged.
            p = np.exp(self.clip_logit_scale * cos_s)
            p = p / (p.sum() + 1e-12)
            surv_score = {surv[j]: float(p[j]) for j in range(len(surv))}
            keep_local, diag = select_kept_indices(
                p, cos_s, min_ratio=auto_min_ratio, mad_k=auto_mad_k, margin_min=auto_margin_min)
            keep = [surv[j] for j in keep_local.tolist()]
            # Optional hard floor (default 0.2): an AND-guard on the survivor-softmax
            # prob, kept for backward-compat. The auto gate already owns the decision.
            if min_score > 0.0:
                keep = [i for i in keep if surv_score[i] >= min_score]
        else:  # legacy fixed-threshold path (auto_threshold=False)
            surv_score = {i: float(scores[i]) for i in surv}
            keep = [i for i in surv if surv_score[i] >= min_score]

        # PROMOTE-TO-CONTAINER: split_components can carve an object into parts (e.g. a
        # screwdriver -> metal shaft + handle), and CLIP often scores the small distinctive
        # PART (the shaft) marginally higher than the whole, so the cliff gate keeps the PART
        # and drops the fuller mask BEFORE the containment-dedup below can englobe it (measured:
        # shaft cos 0.267 area 5112 KEPT vs full-screwdriver cos 0.256 area 42598 dropped, the
        # shaft 100% inside the full mask). Fix: for each kept mask, if a LARGER SURVIVOR
        # contains it (>= promote_contain_frac) within promote_cos_margin cosine, swap the kept
        # PART for that fuller survivor. Runs on SURVIVORS (pre-cliff) so the fuller mask is
        # still available. The dedup below then collapses any duplicates the promotion creates.
        if promote_to_container and keep and surv:
            promoted: List[int] = []
            for i in keep:
                mi = masks[i] > 0
                ai = float(mi.sum())
                best = i
                best_area = ai
                for j in surv:
                    if j == i:
                        continue
                    aj = float(masks[j].sum())
                    if aj <= best_area:
                        continue                       # only promote to a LARGER mask
                    inter = float(np.logical_and(mi, masks[j] > 0).sum())
                    if inter / (ai + 1e-9) >= promote_contain_frac \
                            and (cosines[i] - cosines[j]) <= promote_cos_margin:
                        best, best_area = j, aj          # this larger survivor englobes i
                if best != i and best not in promoted:
                    surv_score.setdefault(best, surv_score.get(i, float(scores[best])))
                    promoted.append(best)
                elif best == i:
                    promoted.append(i)
            keep = list(dict.fromkeys(promoted))         # de-dup, preserve order

        candidates = []
        for i in keep:
            candidates.append({
                "mask": masks[i].astype(np.uint8),
                "score": surv_score.get(i, float(scores[i])),
                "bbox": boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0],
                "mask_area": int(masks[i].sum()),
            })

        # Containment-aware dedup. FastSAM emits the SAME object at multiple
        # extents — e.g. a screwdriver as a tight shaft mask (high CLIP score,
        # 0.76 softmax) AND a full shaft+handle mask (lower, 0.20). We want the
        # FULLEST. Process LARGEST-area first and drop any later candidate that is
        # (a) high-IoU with, or (b) >=80% CONTAINED in, an already-kept larger one.
        # So the full screwdriver is kept and the contained sub-part is dropped.
        # This is also the JUNK discriminator the lowered min_score floor gave up:
        # a real fuller mask CONTAINS the confident sub-part; an unrelated false
        # positive is a disjoint blob that contains nothing kept, so it is NOT
        # collapsed here — it stays a separate candidate (rejected upstream only
        # if its own score is below the floor). Disjoint distinct objects (the
        # split-contamination case) don't contain each other → both survive,
        # exactly as before.
        candidates.sort(key=lambda c: c["mask_area"], reverse=True)
        deduped: List[Dict[str, Any]] = []
        for c in candidates:
            cm = c["mask"] > 0
            c_area = float(c["mask_area"])
            redundant = False
            for k in deduped:  # k came first → k is larger-or-equal area
                km = k["mask"] > 0
                inter = float(np.logical_and(cm, km).sum())
                iou = inter / (float(np.logical_or(cm, km).sum()) + 1e-9)
                contained = inter / (c_area + 1e-9)  # fraction of c inside k
                if iou > dedup_iou or contained >= 0.8:
                    redundant = True
                    break
            if not redundant:
                deduped.append(c)
        deduped.sort(key=lambda c: c["score"], reverse=True)  # output ordered by score
        deduped = deduped[:max_objects]

        if auto_threshold and diag:
            print(f"[fastsam-thr] raw={n_raw} cand={masks.shape[0]} surv={len(surv)} "
                  f"cliff_ratio={diag.get('cliff_ratio', 0):.1f} cos_floor={diag.get('cos_floor', 0):.3f} "
                  f"top_margin={diag.get('top_margin', 0):.3f} kept={len(deduped)}", flush=True)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # DEBUG: dump every post-split component (top-K by CLIP cosine) so a
        # PARTIAL mask is visible — e.g. a screwdriver split by _split_into_components
        # into handle + shaft where only the best-scoring component is KEPT. Shows
        # each component colored + labeled with its cosine + KEPT/surv/filt tag, so
        # you can see what was split off and why it lost. Best-effort.
        try:
            self._save_components_debug(
                image_rgb, masks, cosines, set(surv), set(keep),
                out_dir / f"{output_stem}_fastsam_components.png")
        except Exception as _exc:
            print(f"[fastsam] component debug dump failed: {_exc}", flush=True)

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
            "total_raw_masks": n_raw,
            "total_candidates_after_split": int(masks.shape[0]),
            "total_after_filtering": len(results),
            "filter_params": {
                "min_area_ratio": min_area_ratio, "max_area_ratio": max_area_ratio,
                "dedup_iou": dedup_iou, "max_objects": max_objects, "min_score": min_score,
                "fastsam_conf": fastsam_conf, "fastsam_iou": fastsam_iou, "imgsz": imgsz,
                "split_components": split_components, "auto_threshold": auto_threshold,
                "auto_min_ratio": auto_min_ratio, "auto_mad_k": auto_mad_k,
                "auto_margin_min": auto_margin_min,
                "auto_cos_floor": diag.get("cos_floor"), "auto_cliff_ratio": diag.get("cliff_ratio"),
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
        scores, _ = self._clip_scores(image_rgb, masks, text_prompt)

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
                 "fastsam_weights", "clip_model", "clip_pretrained",
                 "split_components", "auto_threshold", "auto_min_ratio",
                 "auto_mad_k", "auto_margin_min"}
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
        command, cwd=str(Path(__file__).resolve().parents[1]),
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
    _b = lambda v: str(v).lower() in ("1", "true", "yes", "on")
    p.add_argument("--split-components", type=_b, default=True)
    p.add_argument("--auto-threshold", type=_b, default=True)
    p.add_argument("--auto-min-ratio", type=float, default=2.5)
    p.add_argument("--auto-mad-k", type=float, default=3.0)
    p.add_argument("--auto-margin-min", type=float, default=0.04)
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
            split_components=a.split_components, auto_threshold=a.auto_threshold,
            auto_min_ratio=a.auto_min_ratio, auto_mad_k=a.auto_mad_k,
            auto_margin_min=a.auto_margin_min,
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
