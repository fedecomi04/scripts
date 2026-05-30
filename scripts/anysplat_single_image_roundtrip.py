"""Single-image AnySplat roundtrip.

Run AnySplat on ONE image. Render the predicted gaussians back from the
predicted camera pose (in AnySplat's canonical frame). Save the input and
the rendered re-projection side-by-side. If they match, AnySplat is happy
with K=1 and we can use a single-image input mode for the FF path.

Usage (one-liner from any cwd):

    python /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/anysplat_single_image_roundtrip.py \\
        --image /path/to/some.png \\
        --out-dir /tmp/anysplat_roundtrip
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR / "third_party" / "AnySplat"))

from src.model.model.anysplat import AnySplat  # noqa: E402
from src.utils.image import process_image       # noqa: E402


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0, 1] → (H, W, 3) uint8."""
    arr = t.detach().cpu().clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=10.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[roundtrip] loading AnySplat on {device}...")
    model = AnySplat.from_pretrained("lhjiang/anysplat").to(device).eval()
    for p in model.parameters(): p.requires_grad = False

    img = process_image(str(args.image))         # (3, 448, 448) in [-1, 1]
    images = img.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 448, 448)
    print(f"[roundtrip] input shape {tuple(images.shape)}")

    with torch.no_grad():
        gaussians, pred = model.inference((images + 1) * 0.5)
    pred_ext = pred["extrinsic"]   # (1, 1, 4, 4) — predicted c2w
    pred_int = pred["intrinsic"]   # (1, 1, 3, 3) — normalized intrinsics
    B, V = pred_ext.shape[:2]
    H, W = 448, 448
    print(f"[roundtrip] gaussians.means: {tuple(gaussians.means.shape)}, "
          f"predicted cam centre: {pred_ext[0, 0, :3, 3].cpu().tolist()}")

    near = torch.full((B, V), args.near, device=device)
    far  = torch.full((B, V), args.far,  device=device)
    with torch.no_grad():
        out = model.decoder(gaussians, pred_ext, pred_int, near, far, image_shape=(H, W))
    rendered = out.color[0, 0]   # (3, H, W) in [0, 1]
    print(f"[roundtrip] rendered shape: {tuple(rendered.shape)}, "
          f"range: [{rendered.min():.3f}, {rendered.max():.3f}]")

    # Save the input (re-decoded from normalised tensor for fair comparison) + rendered + side-by-side.
    input_uint8 = _to_uint8((img + 1) * 0.5)
    rendered_uint8 = _to_uint8(rendered)
    side = np.concatenate([input_uint8, rendered_uint8], axis=1)

    Image.fromarray(input_uint8).save(args.out_dir / "input_448.png")
    Image.fromarray(rendered_uint8).save(args.out_dir / "rendered_from_predicted_camera.png")
    Image.fromarray(side).save(args.out_dir / "side_by_side.png")
    print(f"[roundtrip] wrote {args.out_dir}/side_by_side.png  (left=input, right=rendered)")


if __name__ == "__main__":
    main()
