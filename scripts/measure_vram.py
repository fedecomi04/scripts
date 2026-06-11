"""Measure real GPU resident footprint of the static-pipeline models.

Runs IN the sam3_dynamic_gs env (it imports sam3 + the SAM3D Inference API).
Loads each model in-process and reports both torch.cuda allocator stats AND
nvidia-smi per-PID used_memory (which includes the CUDA context + any
non-torch / spconv allocations) so the numbers reflect what the GPU actually
holds, not just what the torch caching allocator tracks.

Stages (select with --stages, default "sam3,sam3d"):
  sam3       : load SAM3, infer "<prompt>" on the last static frame, unload.
  sam3d      : load SAM3D (full), infer on the SAM3 top mask, unload.
  fastsam    : load FastSAM(+CLIP), infer, unload.   (needs ultralytics+clip)
  coresident : load FastSAM + SAM3D together (co-residence feasibility).

Usage (from scripts/):
  <sam3_env_python> scripts/measure_vram.py \
      --data <dataset>/static_scene --prompt screwdriver --stages sam3,sam3d
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

_THIS = Path(__file__).resolve()
_UTILS = _THIS.parents[1] / "dynamic_gs" / "utils"


def _load_module(name: str):
    """Import dynamic_gs/utils/<name>.py WITHOUT the package __init__ (which
    pulls nerfstudio, absent in the sam3 env)."""
    path = _UTILS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_mv_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _smi_pid_mib() -> int:
    """nvidia-smi used_memory (MiB) for THIS process pid (0 if not listed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return 0
    pid = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def _torch_mib():
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    p = torch.cuda.max_memory_allocated() / 2**20
    return (a, r, p)


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _empty():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_ROWS = []


def snap(label: str):
    a, r, p = _torch_mib()
    smi = _smi_pid_mib()
    _ROWS.append((label, a, r, p, smi))
    print(f"[mem] {label:42s} torch alloc={a:8.0f}  reserved={r:8.0f}  "
          f"peak={p:8.0f}  nvidia-smi(pid)={smi:8d} MiB", flush=True)


def print_table():
    print("\n==================== VRAM MEASUREMENT TABLE (MiB) ====================")
    print(f"{'stage':44s}{'alloc':>9}{'reserved':>10}{'peak':>9}{'smi_pid':>10}")
    for (label, a, r, p, smi) in _ROWS:
        print(f"{label:44s}{a:9.0f}{r:10.0f}{p:9.0f}{smi:10d}")
    print("======================================================================")


# --------------------------------------------------------------------------


def stage_sam3(args, scratch: Path) -> Path | None:
    """Load SAM3, infer prompt, return path to top-score mask PNG."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _reset_peak(); snap("baseline (pre-SAM3)")
    t0 = time.perf_counter()
    model = build_sam3_image_model()
    proc = Sam3Processor(model, confidence_threshold=0.1)
    print(f"[sam3] load {time.perf_counter()-t0:.1f}s", flush=True)
    snap("SAM3 loaded (resident)")

    img = Image.open(args.image).convert("RGB")
    _reset_peak()
    t1 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(img)
        out = proc.set_text_prompt(state=state, prompt=args.prompt)
    torch.cuda.synchronize()
    print(f"[sam3] infer {time.perf_counter()-t1:.2f}s", flush=True)
    snap("SAM3 after infer (peak)")

    masks = out["masks"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy().reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    mask_path = None
    if masks.shape[0] > 0:
        top = int(np.argmax(scores))
        m = (masks[top] > 0.5).astype(np.uint8) * 255
        mask_path = scratch / "sam3_top_mask.png"
        Image.fromarray(m).save(mask_path)
        print(f"[sam3] {masks.shape[0]} masks, top score={scores[top]:.3f}, "
              f"area={int((m>0).sum())} px -> {mask_path}", flush=True)
    else:
        print("[sam3] WARNING: 0 masks", flush=True)

    del proc, model, out
    _empty(); snap("SAM3 unloaded")
    return mask_path


def stage_sam3d(args, scratch: Path, mask_path: Path | None, keep_loaded=False):
    _sd = _load_module("sam3d")
    from omegaconf import OmegaConf

    _reset_peak(); snap("baseline (pre-SAM3D)")
    t0 = time.perf_counter()
    cfg_path = _sd._write_runtime_config()
    Inference = _sd._import_official_api()
    cfg = OmegaConf.load(str(cfg_path))
    inf = Inference(cfg, compile=False)
    inf.hfer_2d = 0
    inf._pipeline.hfer_2d = 0
    inf._pipeline.ss_params = {"ss_faster_stride": 3, "ss_warmup": 2, "ss_order": 1, "ss_momentum_beta": 0.5}
    inf._pipeline.slat_params = {"slat_thresh": 0.5, "slat_warmup": 2, "slat_token_ratio": 0.15}
    inf._pipeline.mesh_params = {"mesh_spectral_threshold_low": 0.5, "mesh_spectral_threshold_high": 0.7}
    inf._pipeline.enable_mesh = False
    print(f"[sam3d] load {time.perf_counter()-t0:.1f}s", flush=True)
    snap("SAM3D loaded (resident)")

    if mask_path is not None and Path(mask_path).exists():
        img = np.array(Image.open(args.image).convert("RGB"))
        m = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        ri, rm = _sd._resize_image_and_mask(img, m, max_side=518)
        # Build a metric pytorch3d pointmap (production always passes one; without
        # it SAM3D falls back to MoGe-on-CPU and bmm device-mismatches).
        pm = None
        tj = json.load(open(args.data / "transforms.json"))
        intr = {"fx": tj["fl_x"], "fy": tj.get("fl_y", tj["fl_x"]),
                "cx": tj["cx"], "cy": tj.get("cy", tj["cy"])}
        depth_file = (args.data / "depth" / (Path(args.image).stem + ".tiff"))
        if depth_file.exists():
            depth_m = np.array(Image.open(depth_file)).astype(np.float32) * 1e-3
            pm_full = _sd._build_pytorch3d_pointmap(depth_m, intr)  # (H,W,3)
            th, tw = ri.shape[:2]
            pm_t = torch.from_numpy(pm_full).permute(2, 0, 1).unsqueeze(0)
            pm_t = torch.nn.functional.interpolate(pm_t, size=(th, tw), mode="nearest")
            pm = pm_t.squeeze(0).permute(1, 2, 0).contiguous()
        _reset_peak()
        t1 = time.perf_counter()
        try:
            out = inf(ri, rm, seed=42, pointmap=pm) if pm is not None else inf(ri, rm, seed=42)
            torch.cuda.synchronize()
            print(f"[sam3d] infer {time.perf_counter()-t1:.1f}s; "
                  f"gs_pts={out['gs'].get_xyz.shape[0] if 'gs' in out else 'NA'}", flush=True)
        except Exception as exc:
            print(f"[sam3d] infer FAILED: {type(exc).__name__}: {exc}", flush=True)
        snap("SAM3D after infer (peak)")
    else:
        print("[sam3d] no mask -> skipping infer (load-only footprint)", flush=True)

    if keep_loaded:
        return inf
    del inf
    _empty(); snap("SAM3D unloaded")
    return None


def stage_fastsam(args, scratch: Path, keep_loaded=False):
    fsm = _load_module("fastsam_segmentation")
    _reset_peak(); snap("baseline (pre-FastSAM)")
    t0 = time.perf_counter()
    seg = fsm.FastSamTextSegmenter(
        weights=args.fastsam_weights,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    print(f"[fastsam] load {time.perf_counter()-t0:.1f}s", flush=True)
    snap("FastSAM+CLIP loaded (resident)")

    _reset_peak()
    t1 = time.perf_counter()
    objs = seg.infer(image_path=args.image, text_prompt=args.prompt,
                     output_dir=scratch, output_stem="fastsam_meas")
    print(f"[fastsam] infer {time.perf_counter()-t1:.2f}s -> {len(objs)} objs", flush=True)
    snap("FastSAM after infer (peak)")

    if keep_loaded:
        return seg
    del seg
    _empty(); snap("FastSAM unloaded")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="static_scene dir")
    ap.add_argument("--prompt", type=str, default="screwdriver")
    ap.add_argument("--stages", type=str, default="sam3,sam3d")
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--fastsam-weights", type=str, default="FastSAM-x.pt")
    ap.add_argument("--clip-model", type=str, default="ViT-B-32")
    ap.add_argument("--clip-pretrained", type=str, default="openai")
    args = ap.parse_args()

    rgb_dir = args.data / "rgb"
    if args.image is None:
        frames = sorted(rgb_dir.glob("*.png"))
        args.image = frames[-1]  # LAST static frame (matches phase0a)
    print(f"[measure] image = {args.image}", flush=True)
    print(f"[measure] GPU = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"[measure] total VRAM = {torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MiB", flush=True)

    scratch = args.data / "_vram_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    mask_path = scratch / "sam3_top_mask.png"
    mask_path = mask_path if mask_path.exists() else None

    if "sam3" in stages:
        mask_path = stage_sam3(args, scratch)
    if "fastsam" in stages:
        stage_fastsam(args, scratch)
    if "sam3d" in stages:
        stage_sam3d(args, scratch, mask_path)
    if "coresident" in stages:
        seg = stage_fastsam(args, scratch, keep_loaded=True)
        snap("FastSAM resident, loading SAM3D next")
        stage_sam3d(args, scratch, mask_path, keep_loaded=False)
        del seg
        _empty(); snap("coresident done, all unloaded")

    print_table()


if __name__ == "__main__":
    raise SystemExit(main())
