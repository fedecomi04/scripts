"""AnySplat feedforward worker — file-based IPC, invoked via subprocess.

Loads AnySplat once, runs inference on N images (target first), converts the
output to Splatfacto conventions, and writes the result + predicted cameras
to a single `.npz` file.

Invocation pattern (matches SAM3D pattern):

    conda run -n anysplat_dynamic_gs python scripts/anysplat_worker.py \
        --image <path1.png> --image <path2.png> ... \
        --output <out.npz> \
        [--anysplat-repo third_party/AnySplat]

Output `.npz` keys:
    means_canonical     : (N, 3)   float32  — gaussian xyz in AnySplat canonical frame
    log_scales          : (N, 3)   float32  — natural log of per-axis std (Splatfacto convention)
    quats_wxyz          : (N, 4)   float32  — wxyz unit quaternion
    opacity_logits      : (N,)     float32  — torch.logit of [0,1] opacity (Splatfacto convention)
    features_dc         : (N, 1, 3)  float32  — SH degree-0 (DC) band
    features_rest       : (N, 15, 3) float32  — SH degree 1-3 coefficients
    pred_extrinsic_c2w  : (K, 4, 4) float32  — predicted camera-to-world per input view
    pred_intrinsic_norm : (K, 3, 3) float32  — predicted intrinsics, NORMALIZED to [0,1] coords
    pixel_to_world_dropped: bool    — True (voxelization destroys per-pixel mapping)
    voxel_count         : int       — N (after voxelization)
    input_image_paths   : list[str] — paths in the same order as pred_extrinsic_c2w
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _add_anysplat_to_path(anysplat_repo: Path) -> None:
    repo = anysplat_repo.resolve()
    if not (repo / "src" / "model" / "model" / "anysplat.py").exists():
        raise FileNotFoundError(
            f"AnySplat repo not found at {repo} (expected src/model/model/anysplat.py)"
        )
    sys.path.insert(0, str(repo))


def _load_model(device: torch.device):
    from src.model.model.anysplat import AnySplat  # type: ignore
    # TF32 matmul (Blackwell+) + cuDNN benchmark = free speedup on fp32 ops,
    # no quality impact for inference. Set BEFORE the model loads so any
    # weight-conversion or kernel-selection picks the fast path.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Warm-up forward: absorbs the CUDA JIT + cuDNN benchmark search cost
    # at startup so the first real FF call is at steady-state latency
    # (~110 ms, not ~440 ms). Adds ~0.5 s to model-load wall-clock.
    if device.type == "cuda":
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 3, 448, 448), device=device)
            try:
                _ = model.inference((dummy + 1) * 0.5)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[anysplat_worker] warm-up forward failed (non-fatal): {e}",
                      flush=True, file=sys.stderr)
    return model


def _load_and_preprocess(image_paths: list[Path], device: torch.device) -> torch.Tensor:
    from src.utils.image import process_image  # type: ignore
    imgs = [process_image(str(p)) for p in image_paths]
    stacked = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # (1, K, 3, 448, 448)
    return stacked


def _convert_gaussians_to_splatfacto(g) -> dict[str, np.ndarray]:
    """Drop SH-4 band; transpose harmonics; log-scale; logit-opacity.

    All slice/permute/log/logit math runs on GPU so the H2D transfer carries
    only the data we actually keep — the SH-4 truncation alone removes 36 %
    of the harmonics bandwidth (the dominant payload)."""
    sh_total = g.harmonics.shape[-1]
    if sh_total < 16:
        raise RuntimeError(f"AnySplat returned only {sh_total} SH coeffs; need >= 16")

    # GPU-side prep: slice [..., :16] + permute (3,16) -> (16,3), contiguous so
    # the .cpu() below is a single dense copy of N*16*3 floats (not N*25*3).
    harm_t = g.harmonics[0].detach()[..., :16].permute(0, 2, 1).contiguous()  # (N, 16, 3) GPU
    log_scales_t = torch.log(g.scales[0].detach().clamp(min=1e-12))           # (N, 3) GPU
    opac_clamped = g.opacities[0].detach().clamp(min=1e-6, max=1 - 1e-6)
    opac_logits_t = torch.log(opac_clamped / (1.0 - opac_clamped))            # (N,) GPU

    # Single H2D per tensor. PyTorch queues them on the default stream; the
    # first .numpy() triggers one cudaStreamSynchronize for all.
    harm_np = harm_t.cpu().numpy().astype(np.float32, copy=False)
    return {
        "means_canonical": g.means[0].detach().cpu().numpy().astype(np.float32, copy=False),
        "log_scales":      log_scales_t.cpu().numpy().astype(np.float32, copy=False),
        "quats_wxyz":      g.rotations[0].detach().cpu().numpy().astype(np.float32, copy=False),
        "opacity_logits":  opac_logits_t.cpu().numpy().astype(np.float32, copy=False),
        "features_dc":     harm_np[:, 0, :],     # (N, 3)
        "features_rest":   harm_np[:, 1:, :],    # (N, 15, 3)
    }


def _convert_pred_cameras(pred: dict) -> dict[str, np.ndarray]:
    return {
        "pred_extrinsic_c2w": pred["extrinsic"][0].detach().cpu().numpy().astype(np.float32),
        "pred_intrinsic_norm": pred["intrinsic"][0].detach().cpu().numpy().astype(np.float32),
    }


def _run_one(model, image_paths: list[Path], output_npz: Path, device: torch.device) -> dict:
    is_cuda = device.type == "cuda"

    t_img0 = time.time()
    images = _load_and_preprocess(image_paths, device)
    if is_cuda:
        torch.cuda.synchronize()  # fold H2D copy into images_load
    t_images_load_ms = (time.time() - t_img0) * 1000.0

    t_fwd0 = time.time()
    with torch.no_grad():
        gaussians, pred = model.inference((images + 1) * 0.5)
    if is_cuda:
        torch.cuda.synchronize()
    t_forward_ms = (time.time() - t_fwd0) * 1000.0

    t_conv0 = time.time()
    out: dict = {}
    out.update(_convert_gaussians_to_splatfacto(gaussians))
    out.update(_convert_pred_cameras(pred))
    t_convert_ms = (time.time() - t_conv0) * 1000.0

    t_save0 = time.time()
    out["voxel_count"] = int(out["means_canonical"].shape[0])
    out["pixel_to_world_dropped"] = True
    out["input_image_paths"] = [str(p) for p in image_paths]
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    # Pickle (not np.savez) — np.savez wraps each array in a ZIP entry with a
    # CRC32, which on the ~54 MB features_rest array costs ~50 ms even on
    # tmpfs. Pickle is a flat memcpy of the dict-of-ndarrays.
    import pickle
    with open(output_npz, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_npz_save_ms = (time.time() - t_save0) * 1000.0

    return {
        "status": "ok",
        "output": str(output_npz),
        "n_gaussians": int(out["voxel_count"]),
        "k_views": int(out["pred_extrinsic_c2w"].shape[0]),
        "inference_s": t_forward_ms / 1000.0,  # kept for back-compat
        "t_images_load_ms": t_images_load_ms,
        "t_forward_ms": t_forward_ms,
        "t_convert_ms": t_convert_ms,
        "t_npz_save_ms": t_npz_save_ms,
    }


def _persistent_loop(model, device: torch.device) -> None:
    """Read JSON requests from stdin, write JSON responses to stdout.

    Requests: ``{"images": ["p1", "p2", ...], "output": "/tmp/out.npz"}`` OR ``{"cmd": "quit"}``.
    Responses: ``{"status": "ok"|"error", ...}`` on one line each.
    A leading ``{"status": "ready"}`` line is emitted once the model has finished loading.
    """
    import json
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"status": "error", "msg": f"bad json: {e}"}), flush=True)
            continue
        if req.get("cmd") == "quit":
            print(json.dumps({"status": "bye"}), flush=True)
            break
        try:
            image_paths = [Path(p) for p in req["images"]]
            for p in image_paths:
                if not p.exists():
                    raise FileNotFoundError(f"image not found: {p}")
            output_npz = Path(req["output"])
            result = _run_one(model, image_paths, output_npz, device)
            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "msg": str(e)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append", help="Input image path (repeat). First is the target view.")
    ap.add_argument("--output", type=Path, help="Output .npz path (single-shot mode)")
    ap.add_argument("--anysplat-repo", type=Path, default=Path(__file__).parent / "third_party" / "AnySplat")
    ap.add_argument("--persistent", action="store_true",
                    help="Persistent mode: load model once, then handle JSON inference requests from stdin.")
    args = ap.parse_args()

    _add_anysplat_to_path(args.anysplat_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    print(f"[anysplat_worker] loading model on {device}...", flush=True, file=sys.stderr if args.persistent else sys.stdout)
    model = _load_model(device)
    print(f"[anysplat_worker]   loaded in {time.time()-t0:.1f}s", flush=True,
          file=sys.stderr if args.persistent else sys.stdout)

    if args.persistent:
        _persistent_loop(model, device)
        return

    if not args.image or not args.output:
        raise SystemExit("Single-shot mode requires --image and --output")
    image_paths = [Path(p) for p in args.image]
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
    print(f"[anysplat_worker] input: {len(image_paths)} image(s)", flush=True)
    result = _run_one(model, image_paths, args.output, device)
    print(f"[anysplat_worker] wrote {args.output} (N={result['n_gaussians']}, "
          f"K={result['k_views']}, inference={result['inference_s']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
