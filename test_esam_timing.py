#!/usr/bin/env python3
"""Measure ESAM load time vs inference time separately."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

RENDER_IMAGE = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_masks_esam/arm_05597_render.png"
)
CHANGE_MASK = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_masks_esam/arm_05594_change_mask.png"
)
CHECKPOINT_PATH = Path.home() / ".cache" / "efficient_sam" / "efficient_sam_vitt.pt"
NUM_INFERENCE_RUNS = 5


def load_images():
    rgb = np.array(Image.open(RENDER_IMAGE).convert("RGB"))
    mask_gray = np.array(Image.open(CHANGE_MASK).convert("L"))
    mask = mask_gray > 127
    return rgb, mask


def get_prompt_points(mask_np, num_points=8):
    if not np.any(mask_np):
        return np.zeros((0, 2), dtype=np.int64)
    dist = distance_transform_edt(mask_np)
    threshold = float(np.quantile(dist[mask_np], 0.10))
    inner = mask_np & (dist >= threshold)
    if not np.any(inner):
        inner = mask_np
    coords = np.argwhere(inner)
    distances = dist[coords[:, 0], coords[:, 1]]
    selected = [int(np.argmax(distances))]
    while len(selected) < min(num_points, len(coords)):
        sel_coords = coords[selected]
        deltas = coords[:, None, :] - sel_coords[None, :, :]
        min_sq = np.sum(deltas ** 2, axis=2).min(axis=1)
        score = min_sq * (1.0 + distances)
        score[selected] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] < 0:
            break
        selected.append(nxt)
    points_rc = coords[selected]
    return points_rc[:, ::-1].copy()  # xy


def run_inference(model, rgb_np, points_xy, device):
    image_tensor = torch.from_numpy(rgb_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    point_tensor = torch.from_numpy(points_xy).float().view(1, 1, -1, 2).to(device)
    label_tensor = torch.ones((1, 1, len(points_xy)), dtype=torch.float32, device=device)
    with torch.no_grad():
        predicted_logits, predicted_iou = model(image_tensor, point_tensor, label_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    best = int(predicted_iou[0, 0].argmax().item())
    return (predicted_logits[0, 0, best] >= 0).cpu().numpy()


def main():
    from efficient_sam.efficient_sam import build_efficient_sam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rgb_np, mask_np = load_images()
    points_xy = get_prompt_points(mask_np)
    print(f"Image size: {rgb_np.shape}, prompt points: {len(points_xy)}")
    print()

    # --- Measure model load time ---
    t0 = time.perf_counter()
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=str(CHECKPOINT_PATH),
    )
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    print(f"Model load time:  {load_time*1000:.1f} ms")

    # --- Warmup run (not counted) ---
    run_inference(model, rgb_np, points_xy, device)

    # --- Measure inference time (repeated) ---
    times = []
    for i in range(NUM_INFERENCE_RUNS):
        t0 = time.perf_counter()
        run_inference(model, rgb_np, points_xy, device)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"Inference time:   avg={avg_ms:.1f} ms  min={min_ms:.1f} ms  max={max_ms:.1f} ms  (over {NUM_INFERENCE_RUNS} runs)")
    print()
    print(f"Summary:")
    print(f"  Load:      {load_time*1000:.0f} ms  ({load_time/(load_time + avg_ms/1000)*100:.0f}% of first call)")
    print(f"  Inference: {avg_ms:.0f} ms  ({avg_ms/1000/(load_time + avg_ms/1000)*100:.0f}% of first call)")
    print(f"  First call total (load + inference): {load_time*1000 + avg_ms:.0f} ms")


if __name__ == "__main__":
    main()
