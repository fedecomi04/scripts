"""SAM3D trim probe: full (fp32 generators) vs trimmed, output + VRAM diff.

Trim = (a) cast the two big diffusion generators (ss_generator, slat_generator)
to fp16 — SAFE because the forward already runs under autocast(float16), so the
fp32 weights are cast to fp16 per-op anyway; storing them fp16 just removes the
wasted fp32 residency; (b) move never-invoked modules (slat_decoder_mesh,
ss_encoder, slat_decoder_gs_4) to CPU for the gaussian-only path.

Runs both in one process (sequentially, unload between) and reports gaussian
count + centroid + bbox + chamfer between the two clouds, plus resident/peak VRAM.

Usage (sam3 env):
  <sam3_env_python> scripts/sam3d_trim_probe.py --data <dataset>/static_scene
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

_THIS = Path(__file__).resolve()
_UTILS = _THIS.parents[1] / "dynamic_gs" / "utils"


def _load_module(name):
    spec = importlib.util.spec_from_file_location(f"_tp_{name}", _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_inference(_sd):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(_sd._write_runtime_config()))
    Inference = _sd._import_official_api()
    inf = Inference(cfg, compile=False)
    inf.hfer_2d = 0
    inf._pipeline.hfer_2d = 0
    inf._pipeline.ss_params = {"ss_faster_stride": 3, "ss_warmup": 2, "ss_order": 1, "ss_momentum_beta": 0.5}
    inf._pipeline.slat_params = {"slat_thresh": 0.5, "slat_warmup": 2, "slat_token_ratio": 0.15}
    inf._pipeline.mesh_params = {"mesh_spectral_threshold_low": 0.5, "mesh_spectral_threshold_high": 0.7}
    inf._pipeline.enable_mesh = False
    return inf


def _apply_trim(inf):
    p = inf._pipeline
    moved, halved = [], []
    for name in ["slat_decoder_mesh", "ss_encoder", "slat_decoder_gs_4"]:
        m = p.models.get(name) if hasattr(p.models, "get") else (p.models[name] if name in p.models else None)
        if m is not None:
            try:
                m.to("cpu"); moved.append(name)
            except Exception as e:
                print(f"  trim: could not move {name} to cpu: {e}")
    for name in ["ss_generator", "slat_generator"]:
        if name in p.models and p.models[name] is not None:
            p.models[name].half(); halved.append(name)
    # also fp16 the DINOv2 condition embedders (they run under autocast too)
    ce = getattr(p, "condition_embedders", None)
    if isinstance(ce, dict):
        for k, m in ce.items():
            if m is not None:
                try:
                    m.half(); halved.append(k)
                except Exception as e:
                    print(f"  trim: could not fp16 {k}: {e}")
    gc.collect(); torch.cuda.empty_cache()
    print(f"  trim: moved-to-cpu={moved}  fp16={halved}")


def _infer_xyz(inf, _sd, image, mask, pm):
    out = inf(image, mask, seed=42, pointmap=pm)
    torch.cuda.synchronize()
    xyz = out["gs"].get_xyz.detach().cpu().numpy()
    return xyz


def _chamfer(a, b, k=4000):
    # subsample both, mean nearest-neighbour both directions (cheap O(k^2))
    ra = a[np.random.default_rng(0).choice(len(a), min(k, len(a)), replace=False)]
    rb = b[np.random.default_rng(1).choice(len(b), min(k, len(b)), replace=False)]
    ta, tb = torch.tensor(ra), torch.tensor(rb)
    d = torch.cdist(ta, tb)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


def _stats(xyz):
    return dict(n=len(xyz), centroid=xyz.mean(0).round(4).tolist(),
                bbox=(xyz.max(0) - xyz.min(0)).round(4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    _sd = _load_module("sam3d")

    img_path = sorted((args.data / "rgb").glob("*.png"))[-1]
    mask_path = args.data / "_vram_scratch" / "sam3_top_mask.png"
    image = np.array(Image.open(img_path).convert("RGB"))
    m = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
    ri, rm = _sd._resize_image_and_mask(image, m, max_side=518)
    tj = json.load(open(args.data / "transforms.json"))
    intr = {"fx": tj["fl_x"], "fy": tj.get("fl_y", tj["fl_x"]), "cx": tj["cx"], "cy": tj["cy"]}
    depth_m = np.array(Image.open(args.data / "depth" / (img_path.stem + ".tiff"))).astype(np.float32) * 1e-3
    pm_full = _sd._build_pytorch3d_pointmap(depth_m, intr)
    th, tw = ri.shape[:2]
    pm = torch.nn.functional.interpolate(torch.from_numpy(pm_full).permute(2, 0, 1)[None], size=(th, tw), mode="nearest").squeeze(0).permute(1, 2, 0).contiguous()

    def run(trim: bool):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        inf = _build_inference(_sd)
        if trim:
            _apply_trim(inf)
        resident = torch.cuda.memory_allocated() / 2**20
        t1 = time.perf_counter()
        xyz = _infer_xyz(inf, _sd, ri, rm, pm)
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"[{'TRIM' if trim else 'FULL'}] load={t1-t0:.1f}s infer={time.perf_counter()-t1:.1f}s "
              f"resident={resident:.0f} peak={peak:.0f} MiB  stats={_stats(xyz)}")
        del inf; gc.collect(); torch.cuda.empty_cache()
        return xyz, resident, peak

    full_xyz, fr, fp = run(False)
    trim_xyz, tr, tp = run(True)
    ch = _chamfer(full_xyz, trim_xyz)
    print("\n==================== SAM3D TRIM RESULT ====================")
    print(f"resident: full {fr:.0f} -> trim {tr:.0f} MiB   ({fr-tr:.0f} MiB saved)")
    print(f"peak:     full {fp:.0f} -> trim {tp:.0f} MiB   ({fp-tp:.0f} MiB saved)")
    print(f"gs count: full {len(full_xyz)} -> trim {len(trim_xyz)}")
    print(f"chamfer(full,trim) = {ch*1000:.2f} mm  (object scale bbox ~{(full_xyz.max(0)-full_xyz.min(0)).max()*1000:.0f} mm)")
    print("===========================================================")


if __name__ == "__main__":
    raise SystemExit(main())
