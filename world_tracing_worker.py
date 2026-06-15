"""World-Tracing object-model worker — runs in the reused ``dynamic_gs`` env.

The worker self-inserts the WT repo + vendored ``structlog`` onto ``sys.path``
(see below), so ``dynamic_gs`` is reused without a dedicated env and without
modifying it. Run it with the ``dynamic_gs`` python.

Mirrors ``third_party/world-tracing/examples/infer_rgba.py`` but, instead of
writing a rerun ``.rrd``, dumps the raw prediction to a pickle file for the
``dynamic_gs``-env consumer (``scripts/view_object_reconstruction.py``) to
back-project + visualize. Same env-isolation pattern as ``anysplat_worker.py``.

Invocation (single deterministic seed = fastest / lowest VRAM):
    <world_tracing python> world_tracing_worker.py \
        --image <obj_rgba.png> --out <wt_out.pkl> --seed 42

Output pickle dict:
    xyz        : (L, S, S, 3) float32  camera-space XYZ, RDF (+x right, +y down, +z fwd); invalid = 0
    mask       : (L, S, S)    bool     per-layer foreground (AND-accumulated across layers)
    rgb        : (S, S, 3)    float32  preprocessed model-input image in [0, 1] (for per-pixel color)
    K_solved   : (3, 3)       float32  pinhole K fit from layer-0 XYZ (or None)
    fov_x, image_size, config, seed
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

_TIMINGS: dict[str, float] = {}


@contextmanager
def _timed(name: str, device: "torch.device"):
    """Wall-clock a substep; CUDA-synced so GPU time is attributed correctly."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        _TIMINGS[name] = dt
        print(f"[wt-worker]   ⏱ {name:<16s} {dt:7.2f}s", flush=True)

# Reuse the `dynamic_gs` env (it already has torch+cu128/sm_120 and every WT
# dep except structlog). Put the WT repo + the vendored structlog on the path
# so this worker runs without a dedicated env and without modifying any env.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "third_party" / "world-tracing", _HERE / "third_party" / "_wt_vendor"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="RGBA object cutout (alpha = mask)")
    ap.add_argument("--out", required=True, type=Path, help="Output pickle path")
    ap.add_argument("--config", default="r75b", help="WT config (r75b = object model)")
    ap.add_argument("--ckpt", default="hf://haoz19/object-model-6layer", help="hf:// URI, bare config name, or local .pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--bg-color", default="128,128,128", help="RGB the foreground is alpha-blended onto pre-encoder")
    args = ap.parse_args()

    from wt import inference_diffusion, solve_intrinsics_from_xyz
    from wt.checkpoint import build_model_and_load_ckpt, resolve_ckpt_path
    from wt.data import load_rgba_image, preprocess_rgba_for_model
    from wt.inference import _bypass_activation_checkpointing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wt-worker] config={args.config} device={device} ckpt={args.ckpt}", flush=True)

    t_all = time.perf_counter()

    # Split the one-time 6.2 GB download from the model build+load so the
    # cached-vs-first-run cost is visible. resolve_ckpt_path returns a local
    # path (cache hit = ~instant); passing it to build_*_ckpt skips re-download.
    with _timed("weights_fetch", device):
        local_ckpt = resolve_ckpt_path(args.ckpt)
    with _timed("model_load", device):
        model, cfg = build_model_and_load_ckpt(args.config, local_ckpt, device)

    bg = tuple(int(x) for x in args.bg_color.split(","))
    with _timed("preprocess", device):
        rgba = load_rgba_image(args.image, auto_alpha=True)  # alpha present in our PNG → used verbatim
        print(f"[wt-worker] input image {rgba.shape}", flush=True)
        rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
            rgba,
            image_size=cfg["image_size"],
            num_layers=cfg["model_kwargs"]["num_layers"],
            center_crop=not args.no_center_crop,
            bg_color=bg,
        )
        rgb_t = rgb_t.to(device)
        mask_t = mask_t.to(device)
        intr_t = intr_t.to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    with _timed("diffusion", device):
        with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
            xyz_pred, mask_pred, _ = inference_diffusion(
                model,
                rgb_t,
                gt_mask=mask_t,
                use_gt_mask=True,
                intrinsics=intr_t,
                invalid_fill_mode="noise",
                **cfg["inference_kwargs"],
            )

    with _timed("postproc", device):
        xyz = xyz_pred[0].float().cpu().numpy()                       # (L, S, S, 3)
        mask = mask_pred[0].cpu().numpy().astype(bool)                # (L, S, S)
        rgb = rgb_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (S, S, 3) in [0,1]
        K_solved, fov_x = solve_intrinsics_from_xyz(xyz[0], mask[0], image_size=cfg["image_size"])

    _TIMINGS["total"] = time.perf_counter() - t_all

    out = {
        "xyz": xyz,
        "mask": mask,
        "rgb": rgb,
        "K_solved": None if K_solved is None else np.asarray(K_solved, np.float32),
        "fov_x": float(fov_x),
        "image_size": int(cfg["image_size"]),
        "config": args.config,
        "seed": args.seed,
        "num_steps": int(cfg["inference_kwargs"].get("num_steps", -1)),
        "timings_s": dict(_TIMINGS),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[wt-worker] wrote {args.out}  (valid pts across {xyz.shape[0]} layers = {int(mask.sum())}, fov_x≈{fov_x:.1f}°)", flush=True)

    # ---- timing summary ----
    print("[wt-worker] ── timing summary ─────────────────", flush=True)
    for k in ("weights_fetch", "model_load", "preprocess", "diffusion", "postproc", "total"):
        if k in _TIMINGS:
            print(f"[wt-worker]   {k:<16s} {_TIMINGS[k]:7.2f}s", flush=True)
    print(f"[wt-worker]   (num_steps={out['num_steps']}, seed={args.seed}, device={device.type})", flush=True)


if __name__ == "__main__":
    main()
