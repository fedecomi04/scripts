#!/usr/bin/env python3
"""Test EfficientSAM from a dynamic-gs change mask.

This script:
- finds a matching render/change-mask pair
- builds safe positive point prompts from the interior of the change mask
- runs EfficientSAM on the rendered image with point prompts only
- saves the ESAM mask, an overlay image, prompt-point visualization, and the inner-90% mask

Run:
    python test_esam_from_change_mask.py --folder <render_and_masks_folder>

Expected files:
- <folder>/<stem>_render.png
- <folder>/<stem>_change_mask.png
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyTorch is required to run this script.") from exc

try:
    from efficient_sam.efficient_sam import build_efficient_sam
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "EfficientSAM is not installed. Install it with:\n"
        "pip install git+https://github.com/yformer/EfficientSAM.git"
    ) from exc


DEFAULT_FOLDER = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_and_masks"
)
CHECKPOINT_URL = "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"
CHECKPOINT_PATH = Path.home() / ".cache" / "efficient_sam" / "efficient_sam_vitt.pt"
MAX_POINTS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--stem", type=str, default=None, help="Optional frame stem such as arm_05460")
    parser.add_argument("--num-points", type=int, default=MAX_POINTS)
    return parser.parse_args()


def ensure_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return checkpoint_path
    print(f"Downloading EfficientSAM-Ti checkpoint to {checkpoint_path}")
    urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint_path)
    return checkpoint_path


def discover_pair(folder: Path, requested_stem: str | None) -> tuple[str, Path, Path]:
    render_files = {p.name.replace("_render.png", ""): p for p in folder.glob("*_render.png")}
    mask_files = {p.name.replace("_change_mask.png", ""): p for p in folder.glob("*_change_mask.png")}
    common_stems = sorted(set(render_files) & set(mask_files))
    if not common_stems:
        raise FileNotFoundError(f"No matching render/change-mask pair found in {folder}")

    if requested_stem is not None:
        if requested_stem not in common_stems:
            raise FileNotFoundError(f"Requested stem {requested_stem!r} not found. Choices: {common_stems}")
        stem = requested_stem
    else:
        stem = common_stems[0]

    return stem, render_files[stem], mask_files[stem]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_binary_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return mask > 127


def compute_inner90_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        return mask.copy(), np.zeros(mask.shape, dtype=np.float32)

    dist = distance_transform_edt(mask)
    dist_values = dist[mask]
    threshold = float(np.quantile(dist_values, 0.10))
    inner_mask = mask & (dist >= threshold)
    if not np.any(inner_mask):
        inner_mask = mask.copy()
    return inner_mask, dist


def sample_spread_points(inner_mask: np.ndarray, dist_map: np.ndarray, num_points: int) -> np.ndarray:
    coords = np.argwhere(inner_mask)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    num_points = max(1, min(int(num_points), MAX_POINTS, len(coords)))
    distances = dist_map[coords[:, 0], coords[:, 1]]
    safer_coords = coords
    safer_distances = distances
    distance_cutoff = float(np.quantile(distances, 0.50))
    if np.any(distances >= distance_cutoff):
        candidate_coords = coords[distances >= distance_cutoff]
        candidate_distances = distances[distances >= distance_cutoff]
        if len(candidate_coords) >= num_points:
            safer_coords = candidate_coords
            safer_distances = candidate_distances

    selected_indices = [int(np.argmax(safer_distances))]
    while len(selected_indices) < num_points:
        selected = safer_coords[selected_indices]
        deltas = safer_coords[:, None, :] - selected[None, :, :]
        min_sq_dist = np.sum(deltas * deltas, axis=2).min(axis=1)
        score = min_sq_dist * (1.0 + safer_distances)
        score[selected_indices] = -1.0
        next_index = int(np.argmax(score))
        if score[next_index] < 0:
            break
        selected_indices.append(next_index)

    points_rc = safer_coords[selected_indices]
    return points_rc[:, ::-1].copy()


def build_model(checkpoint_path: Path, device: torch.device):
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=str(checkpoint_path),
    )
    model = model.to(device)
    model.eval()
    return model


def run_esam(model, image_rgb: np.ndarray, points_xy: np.ndarray, device: torch.device) -> np.ndarray:
    if len(points_xy) == 0:
        raise ValueError("No positive prompt points were generated.")

    image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    point_tensor = torch.from_numpy(points_xy).float().view(1, 1, -1, 2).to(device)
    label_tensor = torch.ones((1, 1, len(points_xy)), dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(image_tensor, point_tensor, label_tensor)

    scores = predicted_iou[0, 0].detach().cpu().numpy()
    best_index = int(np.argmax(scores))
    mask = predicted_logits[0, 0, best_index] >= 0
    return mask.detach().cpu().numpy()


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    base = base_rgb.astype(np.float32).copy()
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    base[mask] = (1.0 - alpha) * base[mask] + alpha * overlay[mask]
    return np.clip(base, 0, 255).astype(np.uint8)


def make_prompt_visualization(image_rgb: np.ndarray, inner_mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    vis = make_overlay(image_rgb, inner_mask, color=(0, 255, 0), alpha=0.20)
    image = Image.fromarray(vis)
    draw = ImageDraw.Draw(image)
    for x, y in points_xy:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0), outline=(255, 255, 255))
    return np.array(image)


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)

    stem, render_path, change_mask_path = discover_pair(folder, args.stem)
    render_rgb = load_rgb(render_path)
    change_mask = load_binary_mask(change_mask_path)
    inner90_mask, dist_map = compute_inner90_mask(change_mask)
    points_xy = sample_spread_points(inner90_mask, dist_map, args.num_points)

    checkpoint_path = ensure_checkpoint(CHECKPOINT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint_path, device)
    esam_mask = run_esam(model, render_rgb, points_xy, device)

    overlay = make_overlay(render_rgb, esam_mask, color=(255, 0, 0), alpha=0.45)
    prompt_vis = make_prompt_visualization(render_rgb, inner90_mask, points_xy)

    save_mask(esam_mask, folder / "esam_mask.png")
    Image.fromarray(overlay).save(folder / "esam_overlay.png")
    Image.fromarray(prompt_vis).save(folder / "esam_prompt_points.png")
    save_mask(inner90_mask, folder / "esam_inner90_mask.png")

    print(f"Used stem: {stem}")
    print(f"Render image: {render_path.name}")
    print(f"Change mask: {change_mask_path.name}")
    print(f"Prompt points: {points_xy.tolist()}")
    print(f"Saved: {folder / 'esam_mask.png'}")
    print(f"Saved: {folder / 'esam_overlay.png'}")
    print(f"Saved: {folder / 'esam_prompt_points.png'}")
    print(f"Saved: {folder / 'esam_inner90_mask.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
