"""SAM3 text-prompted segmentation: subprocess worker + launcher.

The worker function runs inside the ``sam3_dynamic_gs`` conda environment
(Python 3.12+, PyTorch 2.7+, CUDA 12.6+) because SAM3 is incompatible
with the training env (``radiance_ros``: Python 3.8, PyTorch 2.1.2,
CUDA 11.8).

The subprocess launcher is called from the training process and uses
``conda run -n <env> python`` to invoke this file's ``__main__`` block.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Cross-env python resolver
# ---------------------------------------------------------------------------
# Mirror sam3d.py / sam_worker.py / anysplat_decode.py: resolve the conda
# env's python binary DIRECTLY rather than relying on ``conda run``. ``conda``
# is frequently NOT on PATH for the process that launches ns-train (only an
# activated shell or bootstrap_live.sh puts it there), which made the legacy
# ``["conda", "run", ...]`` path crash with ``[Errno 2] ... 'conda'`` and
# silently skip SAM3 prefusion. Resolving the env python is launch-independent.
_CONDA_ROOT = Path(os.environ.get("DYNAMIC_GS_CONDA_ROOT", str(Path.home() / "miniconda3")))


def _resolve_env_python(conda_env: str):
    """Return the env's python binary (Path) if it exists, else None
    (caller falls back to ``conda run``)."""
    py = _CONDA_ROOT / "envs" / conda_env / "bin" / "python"
    return py if py.exists() else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _touches_n_borders(mask: np.ndarray, n: int = 2) -> bool:
    """Check if *mask* touches >= *n* image borders."""
    h, w = mask.shape[:2]
    borders_touched = 0
    if np.any(mask[0, :]):       # top
        borders_touched += 1
    if np.any(mask[-1, :]):      # bottom
        borders_touched += 1
    if np.any(mask[:, 0]):       # left
        borders_touched += 1
    if np.any(mask[:, -1]):      # right
        borders_touched += 1
    return borders_touched >= n


def load_sam3_masks(results_json_path: Path) -> List[Dict]:
    """Load SAM3 results from the summary JSON written by the worker."""
    with open(results_json_path, "r") as f:
        data = json.load(f)
    return data.get("objects", [])


# ---------------------------------------------------------------------------
# Worker (runs inside sam3_dynamic_gs conda env)
# ---------------------------------------------------------------------------


def run_sam3_segmentation(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    output_stem: str,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.25,
    dedup_iou: float = 0.6,
    max_objects: int = 8,
    confidence_threshold: float = 0.3,
    min_score: float = 0.2,
) -> List[Dict]:
    """Run SAM3 text-prompted segmentation and return filtered object masks.

    Returns a list of per-object dicts::

        {mask_path, score, bbox, mask_area, object_index}

    Both the worker and the subprocess launcher share this return type.
    """
    from sam3.model_builder import build_sam3_image_model  # type: ignore
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_area = image.width * image.height

    # Build model and run inference (autocast to bf16 — SAM3 checkpoint is bf16)
    import torch

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output["masks"]   # list or tensor of binary masks
    scores = output["scores"]
    boxes = output["boxes"]

    # Convert to numpy arrays for filtering (cast bf16→fp32 first)
    if hasattr(masks, "cpu"):
        masks = masks.float().cpu().numpy()
    else:
        masks = np.asarray(masks)
    if hasattr(scores, "cpu"):
        scores = scores.float().cpu().numpy().reshape(-1)
    else:
        scores = np.asarray(scores).reshape(-1)
    if hasattr(boxes, "cpu"):
        boxes = boxes.float().cpu().numpy()
    else:
        boxes = np.asarray(boxes)

    # Ensure masks are 3D: (N, H, W)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    # --- Filter masks ---
    candidates = []
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        area = int(m.sum())

        # Reject empty
        if area == 0:
            continue
        # Reject too small
        if area < min_area_ratio * image_area:
            continue
        # Reject too large
        if area > max_area_ratio * image_area:
            continue
        # Reject if touching >= 2 borders
        if _touches_n_borders(m, n=2):
            continue

        score = float(scores[i]) if i < len(scores) else 0.0
        # Reject below min_score
        if score < min_score:
            continue
        bbox = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
        candidates.append({
            "mask": m,
            "score": score,
            "bbox": bbox,
            "mask_area": area,
            "original_index": i,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Deduplicate overlapping masks (IoU > threshold), keep higher-score
    deduped = []
    for c in candidates:
        is_dup = False
        for kept in deduped:
            if _compute_iou(c["mask"], kept["mask"]) > dedup_iou:
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)

    # Keep at most max_objects
    deduped = deduped[:max_objects]

    # Save masks and build results
    results = []
    for obj_idx, c in enumerate(deduped):
        mask_filename = f"{output_stem}_obj_{obj_idx:02d}_mask.png"
        mask_path = output_dir / mask_filename
        Image.fromarray(c["mask"] * 255).save(mask_path)
        results.append({
            "mask_path": str(mask_path),
            "score": c["score"],
            "bbox": c["bbox"],
            "mask_area": c["mask_area"],
            "object_index": obj_idx,
        })

    # Write summary JSON
    summary = {
        "image_path": str(image_path),
        "text_prompt": text_prompt,
        "total_raw_masks": int(masks.shape[0]),
        "total_after_filtering": len(results),
        "filter_params": {
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
            "dedup_iou": dedup_iou,
            "max_objects": max_objects,
            "confidence_threshold": confidence_threshold,
            "min_score": min_score,
        },
        "objects": results,
    }
    summary_path = output_dir / f"{output_stem}_sam3_results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Clean up GPU
    del model, processor
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Subprocess launcher (runs in training env)
# ---------------------------------------------------------------------------


def run_sam3_subprocess(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    output_stem: str,
    sam3_conda_env: str = "sam3_dynamic_gs",
    **filter_kwargs,
) -> List[Dict]:
    """Launch SAM3 segmentation in a separate conda environment.

    Returns list of ``{mask_path, score, bbox, mask_area, object_index}`` dicts.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the env python directly (launch-independent); fall back to
    # ``conda run`` only if the binary can't be found.
    env_python = _resolve_env_python(sam3_conda_env)
    if env_python is not None:
        command = [str(env_python), str(Path(__file__).resolve())]
    else:
        command = [
            "conda", "run", "--no-capture-output", "-n", sam3_conda_env,
            "python", str(Path(__file__).resolve()),
        ]
    command += [
        "--image", str(image_path),
        "--text-prompt", text_prompt,
        "--output-dir", str(output_dir),
        "--output-stem", output_stem,
    ]
    for key, value in filter_kwargs.items():
        arg_name = f"--{key.replace('_', '-')}"
        command.extend([arg_name, str(value)])

    # Prepend the env's lib to LD_LIBRARY_PATH + PYTHONNOUSERSITE=1 so the
    # subprocess uses the env's own native libs / site-packages, not the
    # parent dynamic_gs env's (same hardening as sam_worker / anysplat).
    sub_env = os.environ.copy()
    if env_python is not None:
        env_lib = str(env_python.parent.parent / "lib")
        sub_env["LD_LIBRARY_PATH"] = (env_lib + ":" + sub_env.get("LD_LIBRARY_PATH", "")).rstrip(":")
        sub_env["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=sub_env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SAM3 subprocess failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    summary_path = output_dir / f"{output_stem}_sam3_results.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"SAM3 subprocess completed but results JSON not found: {summary_path}"
        )
    return load_sam3_masks(summary_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 text-prompted segmentation worker")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text-prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=str, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-area-ratio", type=float, default=0.25)
    parser.add_argument("--dedup-iou", type=float, default=0.6)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-score", type=float, default=0.44)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    try:
        run_sam3_segmentation(
            image_path=args.image,
            text_prompt=args.text_prompt,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            dedup_iou=args.dedup_iou,
            max_objects=args.max_objects,
            confidence_threshold=args.confidence_threshold,
            min_score=args.min_score,
        )
        return 0
    except Exception as exc:
        print(f"SAM3 worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
