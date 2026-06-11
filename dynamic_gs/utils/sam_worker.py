"""Persistent SAM3 + Fast-SAM3D worker — one long-lived subprocess in the
``sam3_dynamic_gs`` conda env, shared by both models.

Background. The shipped path spawns ``conda run -n sam3_dynamic_gs python ...``
once per SAM3 call and once per SAM3D call. Profile breakdown for SAM3 (see
``/tmp/profile_sam3.sh``):

    bash + conda activation .... 0.33 s
    torch import ............... 0.72 s
    sam3 imports ............... 0.84 s
    build_sam3_image_model() ... 6.20 s   ← weights load (DINO + heads)
    FIRST inference (cold CUDA)  0.41 s
    SECOND inference (warm) .... 0.12 s   ← steady-state floor
    THIRD inference (text only)  0.03 s   ← same image, just re-prompt

Every per-call subprocess pays the first ~9 s. A persistent worker pays it
once at startup. Subsequent inferences cost the 0.12 s warm-floor.

VRAM budget on a 16 GB GPU (MEASURED 2026-06-11 on RTX 5070 Ti, 15842 MiB total;
nvidia-smi per-pid resident / peak — supersedes old eyeballed comments):

    SAM3 ............ resident 3772  peak 4522 MiB
    SAM3D ........... resident 12042 peak 13006 MiB  (fp32 generators dominate)
    FastSAM+CLIP .... resident  854  peak 1930 MiB   (~4.4x lighter than SAM3)
    splatfacto step . peak ~1110 MiB (572k gs, 800x800; far below old 5-8 GB note)
    TSDF integrate .. ~3 GB during capture       (comment-derived; re-measure)
    TSDF.finalize() . ~12.3 GB peak              (comment-derived; re-measure)

SAM3 + SAM3D = 17.7 GB → does NOT fit. FastSAM + SAM3D = ~15 GB → DOES fit, which
is why FastSAM replaces SAM3 as the default segmentation backend: it lets SAM3D
load from the start and (trimmed) run parallel to splatfacto. The worker still
exposes explicit ``load_*`` / ``unload_*`` so callers sequence by VRAM budget:

    capture starts → worker boots → load_fastsam (parallel with TSDF integrate)
    user Enter → fastsam_infer (warm) → unload_fastsam
    load_sam3d → sam3d_infer → unload_sam3d → shutdown
    fusion_runner.stop_and_finalize() (TSDF spikes alone on GPU)

Every ``load_*`` / ``*_infer`` response carries ``gpu_resident_mb`` /
``gpu_peak_mb`` (torch allocator) so the orchestration budget can be re-checked
from real numbers at runtime.

Protocol. JSON-over-stdin/stdout, one request per line. Client writes one
JSON object; worker writes exactly one JSON response per request. Stderr is
forwarded to the parent's stderr (preserves traceback visibility).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


_CONDA_ROOT = Path.home() / "miniconda3"
_DEFAULT_ENV = os.environ.get("DYNAMIC_GS_SAM_WORKER_ENV", "sam3_dynamic_gs")
_THIS_FILE = Path(__file__).resolve()


def _load_sibling_module(name: str):
    """Import a sibling .py file by path WITHOUT going through the package init.

    ``dynamic_gs/utils/__init__.py`` pulls in nerfstudio (via rgbd_decode),
    which doesn't exist in the ``sam3_dynamic_gs`` env. The worker only
    needs a couple of helpers (filters, runtime config writer, etc.) from
    sibling modules — load them directly from disk.
    """
    import importlib.util as _ilu
    cache_key = f"_sam_worker_sib_{name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    path = _THIS_FILE.parent / f"{name}.py"
    spec = _ilu.spec_from_file_location(cache_key, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[cache_key] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# WORKER (runs inside the sam3_dynamic_gs conda env)
# ===========================================================================


class _SamWorker:
    """Long-lived state holder for both SAM3 and Fast-SAM3D.

    Lives inside the worker subprocess. Loaded models are kept on GPU between
    requests; ``unload_*`` moves to CPU + ``cuda.empty_cache()``.
    """

    def __init__(self) -> None:
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3d_inference = None
        self.sam3d_runtime_cfg = None
        self.fastsam_seg = None
        # Lazy import torch once at startup — re-importing inside handlers is
        # free but the first ``import torch`` costs 0.7 s.
        import torch
        self.torch = torch

    def _gpu_mem(self) -> Dict[str, float]:
        """Current torch allocator resident + peak (MiB). Permanent
        instrumentation so the orchestration budget is checkable from real
        numbers in every load/infer response."""
        if not self.torch.cuda.is_available():
            return {"gpu_resident_mb": 0.0, "gpu_peak_mb": 0.0}
        return {
            "gpu_resident_mb": round(self.torch.cuda.memory_allocated() / 2**20, 1),
            "gpu_peak_mb": round(self.torch.cuda.max_memory_allocated() / 2**20, 1),
        }

    def _reset_peak(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    # -------- SAM3 --------------------------------------------------------

    def load_sam3(self, *, confidence_threshold: float = 0.1) -> Dict[str, Any]:
        if self.sam3_processor is not None:
            return {"status": "ok", "already_loaded": True}
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        t0 = time.perf_counter()
        self._reset_peak()
        self.sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(self.sam3_model, confidence_threshold=confidence_threshold)
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "load_seconds": elapsed, **self._gpu_mem()}

    def unload_sam3(self) -> Dict[str, Any]:
        if self.sam3_processor is None:
            return {"status": "ok", "already_unloaded": True}
        self.sam3_processor = None
        self.sam3_model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"status": "ok"}

    def sam3_infer(self, *,
                   image_path: str,
                   text_prompt: str,
                   output_dir: str,
                   output_stem: str,
                   min_area_ratio: float = 0.002,
                   max_area_ratio: float = 0.25,
                   dedup_iou: float = 0.6,
                   max_objects: int = 8,
                   min_score: float = 0.2) -> Dict[str, Any]:
        if self.sam3_processor is None:
            raise RuntimeError("SAM3 not loaded; call load_sam3 first")
        # The filter/dedup/save logic is identical to
        # sam3_segmentation.run_sam3_segmentation; we keep it here because
        # importing that module would also import argparse/subprocess
        # machinery we don't need in the worker.
        import numpy as np
        from PIL import Image
        _seg = _load_sibling_module("sam3_segmentation")
        _compute_iou = _seg._compute_iou
        _touches_n_borders = _seg._touches_n_borders

        image = Image.open(image_path).convert("RGB")
        image_area = image.width * image.height

        t_infer = time.perf_counter()
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.sam3_processor.set_image(image)
            output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - t_infer

        masks = output["masks"]
        scores = output["scores"]
        boxes = output["boxes"]
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
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        candidates = []
        for i in range(masks.shape[0]):
            m = (masks[i] > 0.5).astype(np.uint8)
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

        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for obj_idx, c in enumerate(deduped):
            mask_filename = f"{output_stem}_obj_{obj_idx:02d}_mask.png"
            mask_path = out_dir / mask_filename
            Image.fromarray(c["mask"] * 255).save(mask_path)
            results.append({
                "mask_path": str(mask_path),
                "score": c["score"],
                "bbox": c["bbox"],
                "mask_area": c["mask_area"],
                "object_index": obj_idx,
            })

        summary_path = out_dir / f"{output_stem}_sam3_results.json"
        summary_path.write_text(json.dumps({
            "image_path": str(image_path),
            "text_prompt": text_prompt,
            "total_raw_masks": int(masks.shape[0]),
            "total_after_filtering": len(results),
            "filter_params": {
                "min_area_ratio": min_area_ratio,
                "max_area_ratio": max_area_ratio,
                "dedup_iou": dedup_iou,
                "max_objects": max_objects,
                "min_score": min_score,
            },
            "objects": results,
        }, indent=2) + "\n")

        return {
            "status": "ok",
            "objects": results,
            "infer_seconds": infer_seconds,
        }

    def sam3_infer_raw(self, *,
                       image_path: str,
                       text_prompt: str,
                       output_dir: str,
                       output_stem: str,
                       min_score: float = 0.0) -> Dict[str, Any]:
        """Run SAM3 and write all masks above min_score with NO area/border/dedup filters.

        Writes:
          <output_dir>/<output_stem>_raw_masks.npz with keys:
            - masks: (K, H, W) bool
            - scores: (K,) float32
            - boxes: (K, 4) float32  # xyxy
        Returns:
          {"status": "ok", "masks_path": str, "num_masks": int, "infer_seconds": float}
        """
        if self.sam3_processor is None:
            raise RuntimeError("SAM3 not loaded; call load_sam3 first")
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        t_infer = time.perf_counter()
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.sam3_processor.set_image(image)
            output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - t_infer

        masks = output["masks"]
        scores = output["scores"]
        boxes = output["boxes"]
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
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        # Binarize masks to bool — no area/border/dedup filtering, just optional
        # min_score gate (default 0.0 keeps everything).
        masks_bool = (masks > 0.5)
        keep_idx = []
        for i in range(masks_bool.shape[0]):
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < min_score:
                continue
            keep_idx.append(i)

        if keep_idx:
            kept_masks = masks_bool[keep_idx].astype(np.bool_)
            kept_scores = scores[keep_idx].astype(np.float32)
            if len(boxes) >= max(keep_idx) + 1:
                kept_boxes = boxes[keep_idx].astype(np.float32)
            else:
                kept_boxes = np.zeros((len(keep_idx), 4), dtype=np.float32)
        else:
            H, W = (masks_bool.shape[1], masks_bool.shape[2]) if masks_bool.ndim == 3 else (image.height, image.width)
            kept_masks = np.zeros((0, H, W), dtype=np.bool_)
            kept_scores = np.zeros((0,), dtype=np.float32)
            kept_boxes = np.zeros((0, 4), dtype=np.float32)

        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        masks_path = out_dir / f"{output_stem}_raw_masks.npz"
        np.savez(str(masks_path), masks=kept_masks, scores=kept_scores, boxes=kept_boxes)

        return {
            "status": "ok",
            "masks_path": str(masks_path),
            "num_masks": int(kept_masks.shape[0]),
            "infer_seconds": infer_seconds,
        }

    # -------- SAM3D -------------------------------------------------------

    def load_sam3d(self) -> Dict[str, Any]:
        if self.sam3d_inference is not None:
            return {"status": "ok", "already_loaded": True}
        _sd = _load_sibling_module("sam3d")
        _import_official_api = _sd._import_official_api
        _write_runtime_config = _sd._write_runtime_config
        from omegaconf import OmegaConf
        t0 = time.perf_counter()
        runtime_config_path = _write_runtime_config()
        Inference = _import_official_api()
        cfg = OmegaConf.load(str(runtime_config_path))
        inference = Inference(cfg, compile=False)
        # Same get_params() / hfer_2d patches the shipped sam3d.py applies.
        # See sam3d.py:660-678 for the rationale (Fast-SAM3D issue #9).
        inference.hfer_2d = 0
        inference._pipeline.hfer_2d = 0
        inference._pipeline.ss_params = {
            "ss_faster_stride": 3, "ss_warmup": 2, "ss_order": 1, "ss_momentum_beta": 0.5,
        }
        inference._pipeline.slat_params = {
            "slat_thresh": 0.5, "slat_warmup": 2, "slat_token_ratio": 0.15,
        }
        inference._pipeline.mesh_params = {
            "mesh_spectral_threshold_low": 0.5, "mesh_spectral_threshold_high": 0.7,
        }
        inference._pipeline.enable_mesh = False
        self.sam3d_inference = inference
        self.sam3d_runtime_cfg = cfg
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "load_seconds": elapsed, **self._gpu_mem()}

    def unload_sam3d(self) -> Dict[str, Any]:
        if self.sam3d_inference is None:
            return {"status": "ok", "already_unloaded": True}
        self.sam3d_inference = None
        self.sam3d_runtime_cfg = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"status": "ok"}

    def sam3d_infer(self, *,
                    render_image_path: str,
                    object_mask_paths: List[str],
                    output_stems: List[str],
                    output_dir: str,
                    image_dir: Optional[str] = None,
                    max_side: int = 518,
                    depth_path: Optional[str] = None,
                    intrinsics_path: Optional[str] = None) -> Dict[str, Any]:
        if self.sam3d_inference is None:
            raise RuntimeError("SAM3D not loaded; call load_sam3d first")
        # Reuse the shipped multi-object body, but inject our already-loaded
        # inference instance instead of building a new one. Simplest: import
        # the helpers and run the per-mask loop here directly. This avoids
        # forking sam3d.run_sam3d_multi_object's outer "build model" code.
        import numpy as np
        from PIL import Image
        _sd = _load_sibling_module("sam3d")
        _load_binary_mask = _sd._load_binary_mask
        _resize_image_and_mask = _sd._resize_image_and_mask
        _save_preview = _sd._save_preview
        _build_pytorch3d_pointmap = _sd._build_pytorch3d_pointmap
        get_sam3d_output_paths = _sd.get_sam3d_output_paths
        torch = self.torch

        render_image_path_p = Path(render_image_path)
        output_dir_p = Path(output_dir); output_dir_p.mkdir(parents=True, exist_ok=True)
        image_dir_p = Path(image_dir) if image_dir else output_dir_p
        image_dir_p.mkdir(parents=True, exist_ok=True)

        if len(object_mask_paths) != len(output_stems):
            raise ValueError(f"mask count ({len(object_mask_paths)}) != stem count ({len(output_stems)})")

        image_pil = Image.open(render_image_path_p).convert("RGB")
        image_rgb = np.array(image_pil)

        masks = []
        for mp in object_mask_paths:
            m = _load_binary_mask(Path(mp), image_pil.size)
            if image_rgb.shape[:2] != m.shape[:2]:
                raise ValueError(f"SAM3D image/mask shape mismatch: {image_rgb.shape} vs {m.shape}")
            masks.append(m)

        pointmap_full = None
        if depth_path and intrinsics_path:
            depth_m = np.array(Image.open(depth_path)).astype(np.float32)
            intrinsics = json.loads(Path(intrinsics_path).read_text())
            if depth_m.shape[:2] != image_rgb.shape[:2]:
                depth_pil = Image.fromarray(depth_m).resize(
                    (image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
                depth_m = np.array(depth_pil, dtype=np.float32)
            pointmap_full = _build_pytorch3d_pointmap(depth_m, intrinsics)

        inference = self.sam3d_inference

        all_results: List[Dict[str, Any]] = []
        t_total = time.perf_counter()
        for i, (mask, stem) in enumerate(zip(masks, output_stems)):
            paths = get_sam3d_output_paths(output_dir_p, stem, image_dir=image_dir_p)
            ply_path = paths["ply_path"]; pose_path = paths["pose_path"]
            preview_path = paths["preview_path"]; run_info_path = paths["run_info_path"]
            mesh_ply_path = paths["mesh_ply_path"]

            if int(mask.sum()) == 0:
                all_results.append({"status": "skipped", "reason": "empty mask"})
                continue

            # OOM retry ladder mirrors sam3d.run_sam3d_multi_object.
            candidate_sizes: List[int] = []
            for size in [max_side, 112, 96, 80, 64, 48]:
                size = min(int(size), int(max_side))
                if size not in candidate_sizes:
                    candidate_sizes.append(size)
            output = None
            used_shape = None
            attempted = []
            for cand in candidate_sizes:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cur_image, cur_mask = _resize_image_and_mask(image_rgb, mask, max_side=cand)
                attempted.append((cand, tuple(cur_image.shape)))
                _save_preview(cur_mask, cur_image, preview_path)
                resized_pointmap = None
                if pointmap_full is not None:
                    ph, pw = pointmap_full.shape[:2]
                    th, tw = cur_image.shape[:2]
                    if (ph, pw) == (th, tw):
                        resized_pointmap = torch.from_numpy(pointmap_full)
                    else:
                        pm_t = torch.from_numpy(pointmap_full).permute(2, 0, 1).unsqueeze(0)
                        pm_t = torch.nn.functional.interpolate(pm_t, size=(th, tw), mode="nearest")
                        resized_pointmap = pm_t.squeeze(0).permute(1, 2, 0).contiguous()
                try:
                    if resized_pointmap is not None:
                        output = inference(cur_image, cur_mask, seed=42, pointmap=resized_pointmap)
                    else:
                        output = inference(cur_image, cur_mask, seed=42)
                    used_shape = tuple(cur_image.shape)
                    break
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    output = None
                    continue

            if output is None or "gs" not in output:
                all_results.append({"status": "failed", "attempted_sizes": attempted})
                continue

            output["gs"].save_ply(str(ply_path))

            # Mesh export (best-effort).
            mesh_saved = False
            mesh_list = output.get("mesh")
            if mesh_list is not None and len(mesh_list) > 0:
                mr = mesh_list[0]
                if getattr(mr, "success", True):
                    try:
                        import trimesh
                        verts = mr.vertices.detach().cpu().numpy()
                        faces = mr.faces.detach().cpu().numpy()
                        if verts.shape[0] > 0 and faces.shape[0] > 0:
                            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                            tm.export(str(mesh_ply_path))
                            mesh_saved = True
                    except Exception as exc:
                        print(f"[sam_worker] mesh export failed for {stem}: {exc}", file=sys.stderr)

            pose_data: Dict[str, List[float]] = {}
            for key in ("translation", "rotation", "scale"):
                v = output.get(key)
                if v is not None:
                    pose_data[key] = torch.as_tensor(v).detach().cpu().reshape(-1).tolist()
            if "rotation" not in pose_data or len(pose_data["rotation"]) != 4:
                all_results.append({"status": "failed", "reason": "no valid rotation"})
                continue
            pose_path.write_text(json.dumps(pose_data, indent=2) + "\n")
            run_info_path.write_text(
                f"SAM3D worker run\n"
                f"render: {render_image_path}\n"
                f"mask: {object_mask_paths[i]}\n"
                f"image_shape: {tuple(image_rgb.shape)}\n"
                f"used_shape: {used_shape}\n"
                f"attempted: {attempted}\n"
                f"mesh_saved: {mesh_ply_path if mesh_saved else 'NOT SAVED'}\n"
            )
            all_results.append({
                "status": "ok",
                "ply_path": str(ply_path),
                "pose_path": str(pose_path),
                "preview_path": str(preview_path),
                "mesh_ply_path": str(mesh_ply_path) if mesh_saved else None,
                "used_shape": used_shape,
            })
            del output

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "ok",
            "results": all_results,
            "total_seconds": time.perf_counter() - t_total,
            **self._gpu_mem(),
        }

    # -------- FastSAM (lightweight SAM3 alternative) ----------------------

    def load_fastsam(self, *,
                     fastsam_weights: str = "FastSAM-x.pt",
                     clip_model: str = "ViT-B-32-quickgelu",
                     clip_pretrained: str = "openai") -> Dict[str, Any]:
        if self.fastsam_seg is not None:
            return {"status": "ok", "already_loaded": True, **self._gpu_mem()}
        _fs = _load_sibling_module("fastsam_segmentation")
        self._reset_peak()
        t0 = time.perf_counter()
        self.fastsam_seg = _fs.FastSamTextSegmenter(
            weights=fastsam_weights, clip_model=clip_model, clip_pretrained=clip_pretrained,
        )
        return {"status": "ok", "load_seconds": time.perf_counter() - t0, **self._gpu_mem()}

    def unload_fastsam(self) -> Dict[str, Any]:
        if self.fastsam_seg is None:
            return {"status": "ok", "already_unloaded": True}
        self.fastsam_seg = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"status": "ok", **self._gpu_mem()}

    def fastsam_infer(self, *,
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
                      imgsz: int = 1024) -> Dict[str, Any]:
        if self.fastsam_seg is None:
            raise RuntimeError("FastSAM not loaded; call load_fastsam first")
        self._reset_peak()
        t0 = time.perf_counter()
        objects = self.fastsam_seg.infer(
            image_path=image_path, text_prompt=text_prompt,
            output_dir=output_dir, output_stem=output_stem,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
            dedup_iou=dedup_iou, max_objects=max_objects, min_score=min_score,
            fastsam_conf=fastsam_conf, fastsam_iou=fastsam_iou, imgsz=imgsz,
        )
        return {"status": "ok", "objects": objects,
                "infer_seconds": time.perf_counter() - t0, **self._gpu_mem()}

    def fastsam_infer_raw(self, *,
                          image_path: str,
                          text_prompt: str,
                          output_dir: str,
                          output_stem: str,
                          min_score: float = 0.0,
                          fastsam_conf: float = 0.4,
                          fastsam_iou: float = 0.9,
                          imgsz: int = 1024) -> Dict[str, Any]:
        if self.fastsam_seg is None:
            raise RuntimeError("FastSAM not loaded; call load_fastsam first")
        self._reset_peak()
        t0 = time.perf_counter()
        res = self.fastsam_seg.infer_raw(
            image_path=image_path, text_prompt=text_prompt,
            output_dir=output_dir, output_stem=output_stem, min_score=min_score,
            fastsam_conf=fastsam_conf, fastsam_iou=fastsam_iou, imgsz=imgsz,
        )
        return {"status": "ok", "masks_path": res["masks_path"],
                "num_masks": res["num_masks"],
                "infer_seconds": time.perf_counter() - t0, **self._gpu_mem()}


def _worker_main() -> int:
    """Subprocess entrypoint. Read JSON lines from stdin, write JSON lines to stdout.

    SAM3D's pipeline (sam3d_objects.pipeline.*) writes loguru status lines to
    stdout that can JSON-parse as lists (e.g. ``cfg_interval=[0, 500]``),
    which would poison the JSON-line protocol on this stdout. We redirect
    the worker's stdout to a captured FD reserved for protocol use, then
    point sys.stdout at the original stderr's destination so any rogue
    print() inside SAM3D / SAM3 goes there instead.
    """
    # Reserve fd 1 for protocol writes; reroute python-level sys.stdout to stderr.
    _proto = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr   # any print() in nested code → goes to stderr now
    def _emit(obj: Dict[str, Any]) -> None:
        _proto.write(json.dumps(obj) + "\n")
        _proto.flush()

    w = _SamWorker()
    _emit({"status": "ready"})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _emit({"status": "error", "msg": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            _emit({"status": "ok", "msg": "bye"})
            return 0

        handler_map = {
            "load_sam3": w.load_sam3,
            "unload_sam3": w.unload_sam3,
            "sam3_infer": w.sam3_infer,
            "sam3_infer_raw": w.sam3_infer_raw,
            "load_sam3d": w.load_sam3d,
            "unload_sam3d": w.unload_sam3d,
            "sam3d_infer": w.sam3d_infer,
            "load_fastsam": w.load_fastsam,
            "unload_fastsam": w.unload_fastsam,
            "fastsam_infer": w.fastsam_infer,
            "fastsam_infer_raw": w.fastsam_infer_raw,
        }
        h = handler_map.get(cmd)
        if h is None:
            _emit({"status": "error", "msg": f"unknown cmd: {cmd}"})
            continue

        try:
            args = {k: v for k, v in req.items() if k != "cmd"}
            resp = h(**args) if args else h()
            _emit(resp)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"status": "error", "msg": f"{type(exc).__name__}: {exc}"})

    return 0


# ===========================================================================
# CLIENT (lives in the dynamic_gs env; spawns the worker)
# ===========================================================================


class SamWorkerClient:
    """Spawn the persistent worker in the ``sam3_dynamic_gs`` conda env.

    Usage::

        client = SamWorkerClient()              # spawn; ~1.3 s for env + imports
        client.load_sam3(background=True)        # ~6 s, can run during capture
        objs = client.sam3_infer(...)            # warm: ~0.12 s
        client.unload_sam3()
        client.load_sam3d()
        client.sam3d_infer(...)
        client.unload_sam3d()
        client.close()
    """

    def __init__(self,
                 conda_env: str = _DEFAULT_ENV,
                 startup_timeout_s: float = 30.0) -> None:
        env_prefix = _CONDA_ROOT / "envs" / conda_env
        env_python = env_prefix / "bin" / "python"
        if not env_python.exists():
            raise FileNotFoundError(
                f"SAM worker env python not found at {env_python}. "
                f"Expected conda env '{conda_env}' under {env_prefix.parent}."
            )

        # Same trick as anysplat_decode: invoke env python directly + prepend
        # env/lib to LD_LIBRARY_PATH. Bypasses ``conda run`` (no PATH dep,
        # saves ~0.5 s, lets us keep stdin/stdout pipes attached).
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure the worker can import dynamic_gs.* (it imports relative helpers).
        env["PYTHONPATH"] = (
            str(_THIS_FILE.parents[2]) + ":" + env.get("PYTHONPATH", "")
        ).rstrip(":")

        cmd = [str(env_python), "-u", str(_THIS_FILE), "--worker"]
        self._proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,   # forward to parent's stderr — we want tracebacks visible
            text=True, bufsize=1,
        )
        # Block until the worker announces ready.
        t0 = time.time()
        while True:
            if time.time() - t0 > startup_timeout_s:
                self._proc.kill()
                raise TimeoutError(f"SAM worker startup exceeded {startup_timeout_s}s")
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"SAM worker exited during startup (rc={self._proc.returncode}). "
                    f"Check stderr above for the cause."
                )
            line = self._proc.stdout.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            if msg.get("status") == "ready":
                self._spawn_seconds = time.time() - t0
                return

    @property
    def spawn_seconds(self) -> float:
        return getattr(self, "_spawn_seconds", 0.0)

    def _request(self, cmd: str, *, timeout_s: float = 600.0, **kwargs: Any) -> Dict[str, Any]:
        if self._proc.poll() is not None:
            raise RuntimeError(f"SAM worker is no longer running (rc={self._proc.returncode})")
        req = {"cmd": cmd, **kwargs}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"SAM worker '{cmd}' exceeded {timeout_s}s")
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"SAM worker died during '{cmd}' (rc={self._proc.returncode})")
                continue
            try:
                resp = json.loads(line.strip())
            except Exception:
                continue
            # SAM3D's pipeline writes loguru messages that occasionally JSON-parse
            # as lists or numbers (e.g. ``cfg_interval=[0, 500]``). Only accept
            # dicts that carry a ``status`` field — anything else is upstream
            # noise we should skip.
            if not isinstance(resp, dict) or "status" not in resp:
                continue
            if resp.get("status") == "error":
                raise RuntimeError(f"SAM worker '{cmd}' error: {resp.get('msg')}")
            return resp

    # -- SAM3 ---------------------------------------------------------------

    def load_sam3(self, confidence_threshold: float = 0.1, timeout_s: float = 60.0) -> float:
        """Block on SAM3 load. Returns model-load seconds (excludes IPC)."""
        resp = self._request("load_sam3", confidence_threshold=confidence_threshold, timeout_s=timeout_s)
        return float(resp.get("load_seconds", 0.0))

    def unload_sam3(self, timeout_s: float = 15.0) -> None:
        self._request("unload_sam3", timeout_s=timeout_s)

    def sam3_infer(self,
                   *,
                   image_path: Path,
                   text_prompt: str,
                   output_dir: Path,
                   output_stem: str,
                   min_area_ratio: float = 0.002,
                   max_area_ratio: float = 0.25,
                   dedup_iou: float = 0.6,
                   max_objects: int = 8,
                   min_score: float = 0.2,
                   timeout_s: float = 60.0) -> List[Dict[str, Any]]:
        resp = self._request(
            "sam3_infer", timeout_s=timeout_s,
            image_path=str(image_path), text_prompt=text_prompt,
            output_dir=str(output_dir), output_stem=output_stem,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
            dedup_iou=dedup_iou, max_objects=max_objects, min_score=min_score,
        )
        return resp.get("objects", [])

    def sam3_infer_raw(self,
                       *,
                       image_path: Path,
                       text_prompt: str,
                       output_dir: Path,
                       output_stem: str,
                       min_score: float = 0.0,
                       timeout_s: float = 60.0) -> dict:
        """Returns {"masks_path": Path, "num_masks": int, "infer_seconds": float}."""
        resp = self._request(
            "sam3_infer_raw", timeout_s=timeout_s,
            image_path=str(image_path), text_prompt=text_prompt,
            output_dir=str(output_dir), output_stem=output_stem,
            min_score=min_score,
        )
        return {
            "masks_path": Path(resp["masks_path"]),
            "num_masks": int(resp.get("num_masks", 0)),
            "infer_seconds": float(resp.get("infer_seconds", 0.0)),
        }

    # -- SAM3D --------------------------------------------------------------

    def load_sam3d(self, timeout_s: float = 60.0) -> float:
        resp = self._request("load_sam3d", timeout_s=timeout_s)
        return float(resp.get("load_seconds", 0.0))

    def unload_sam3d(self, timeout_s: float = 15.0) -> None:
        self._request("unload_sam3d", timeout_s=timeout_s)

    def sam3d_infer(self,
                    *,
                    render_image_path: Path,
                    object_mask_paths: List[Path],
                    output_stems: List[str],
                    output_dir: Path,
                    image_dir: Optional[Path] = None,
                    max_side: int = 518,
                    depth_path: Optional[Path] = None,
                    intrinsics_path: Optional[Path] = None,
                    timeout_s: float = 600.0) -> List[Dict[str, Any]]:
        resp = self._request(
            "sam3d_infer", timeout_s=timeout_s,
            render_image_path=str(render_image_path),
            object_mask_paths=[str(p) for p in object_mask_paths],
            output_stems=list(output_stems),
            output_dir=str(output_dir),
            image_dir=str(image_dir) if image_dir else None,
            max_side=max_side,
            depth_path=str(depth_path) if depth_path else None,
            intrinsics_path=str(intrinsics_path) if intrinsics_path else None,
        )
        return resp.get("results", [])

    # -- FastSAM ------------------------------------------------------------

    def load_fastsam(self,
                     fastsam_weights: str = "FastSAM-x.pt",
                     clip_model: str = "ViT-B-32-quickgelu",
                     clip_pretrained: str = "openai",
                     timeout_s: float = 120.0) -> float:
        """Block on FastSAM+CLIP load. Returns model-load seconds.

        ``timeout_s`` allows for first-run weight downloads (FastSAM-x ~140 MB +
        CLIP ViT-B-32). Once cached, load is ~few s."""
        resp = self._request("load_fastsam", timeout_s=timeout_s,
                             fastsam_weights=fastsam_weights,
                             clip_model=clip_model, clip_pretrained=clip_pretrained)
        return float(resp.get("load_seconds", 0.0))

    def unload_fastsam(self, timeout_s: float = 15.0) -> None:
        self._request("unload_fastsam", timeout_s=timeout_s)

    def fastsam_infer(self,
                      *,
                      image_path: Path,
                      text_prompt: str,
                      output_dir: Path,
                      output_stem: str,
                      min_area_ratio: float = 0.002,
                      max_area_ratio: float = 0.25,
                      dedup_iou: float = 0.6,
                      max_objects: int = 8,
                      min_score: float = 0.2,
                      fastsam_conf: float = 0.4,
                      fastsam_iou: float = 0.9,
                      imgsz: int = 1024,
                      timeout_s: float = 60.0) -> List[Dict[str, Any]]:
        resp = self._request(
            "fastsam_infer", timeout_s=timeout_s,
            image_path=str(image_path), text_prompt=text_prompt,
            output_dir=str(output_dir), output_stem=output_stem,
            min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
            dedup_iou=dedup_iou, max_objects=max_objects, min_score=min_score,
            fastsam_conf=fastsam_conf, fastsam_iou=fastsam_iou, imgsz=imgsz,
        )
        return resp.get("objects", [])

    def fastsam_infer_raw(self,
                          *,
                          image_path: Path,
                          text_prompt: str,
                          output_dir: Path,
                          output_stem: str,
                          min_score: float = 0.0,
                          fastsam_conf: float = 0.4,
                          fastsam_iou: float = 0.9,
                          imgsz: int = 1024,
                          timeout_s: float = 60.0) -> dict:
        """Returns {"masks_path": Path, "num_masks": int, "infer_seconds": float}."""
        resp = self._request(
            "fastsam_infer_raw", timeout_s=timeout_s,
            image_path=str(image_path), text_prompt=text_prompt,
            output_dir=str(output_dir), output_stem=output_stem, min_score=min_score,
            fastsam_conf=fastsam_conf, fastsam_iou=fastsam_iou, imgsz=imgsz,
        )
        return {
            "masks_path": Path(resp["masks_path"]),
            "num_masks": int(resp.get("num_masks", 0)),
            "infer_seconds": float(resp.get("infer_seconds", 0.0)),
        }

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "SamWorkerClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent SAM3 + SAM3D worker")
    p.add_argument("--worker", action="store_true", help="Run as the worker subprocess (do not invoke directly).")
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker_main()
    print("This script is the worker subprocess. Use SamWorkerClient() from the parent.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
