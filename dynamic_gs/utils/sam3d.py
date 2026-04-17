from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np
from PIL import Image
import torch

SAM3D_REPO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sam-3d-objects"
SAM3D_CONFIG_PATH = SAM3D_REPO_ROOT / "checkpoints" / "hf" / "pipeline.yaml"
SAM3D_RUNTIME_CONFIG_PATH = SAM3D_REPO_ROOT / "checkpoints" / "hf" / "pipeline_runtime_small.yaml"


def get_sam3d_output_paths(
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    return {
        "ply_path": output_dir / f"{output_stem}_raw_output.ply",
        "pose_path": output_dir / f"{output_stem}_pose.json",
        "preview_path": image_dir / f"{output_stem}_preview.png",
        "run_info_path": output_dir / f"{output_stem}_run_info.txt",
        "glb_path": output_dir / f"{output_stem}_mesh.glb",
    }


def resolve_sam3d_pose_path(raw_ply_path: Path, fallback_pose_path: Path | None = None) -> Path | None:
    raw_ply_path = Path(raw_ply_path)
    candidates: list[Path] = []
    if raw_ply_path.name.endswith("_raw_output.ply"):
        candidates.append(raw_ply_path.with_name(raw_ply_path.name[: -len("_raw_output.ply")] + "_pose.json"))
    if fallback_pose_path is not None:
        candidates.append(Path(fallback_pose_path))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def load_sam3d_pose(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    pose: Dict[str, np.ndarray] = {}
    for key in ("translation", "rotation", "scale"):
        value = payload.get(key)
        if value is None:
            continue
        pose[key] = np.asarray(value, dtype=np.float32).reshape(-1)
    return pose


def sam3d_pose_has_rotation(path: Path | None) -> bool:
    if path is None or not Path(path).exists():
        return False
    try:
        pose = load_sam3d_pose(Path(path))
    except Exception:
        return False
    rotation = pose.get("rotation")
    return rotation is not None and rotation.size == 4 and np.isfinite(rotation).all()


def _load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, resample=Image.NEAREST)
    return (np.array(mask_image) > 127).astype(np.uint8)


def _install_kaolin_stub() -> None:
    if "kaolin" in sys.modules:
        return

    kaolin = ModuleType("kaolin")
    visualize = ModuleType("kaolin.visualize")
    render = ModuleType("kaolin.render")
    camera = ModuleType("kaolin.render.camera")
    utils = ModuleType("kaolin.utils")
    testing = ModuleType("kaolin.utils.testing")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    visualize.IpyTurntableVisualizer = _Dummy
    camera.Camera = _Dummy
    camera.CameraExtrinsics = _Dummy
    camera.PinholeIntrinsics = _Dummy
    testing.check_tensor = lambda *args, **kwargs: True
    render.camera = camera
    utils.testing = testing
    kaolin.visualize = visualize
    kaolin.render = render
    kaolin.utils = utils

    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.visualize"] = visualize
    sys.modules["kaolin.render"] = render
    sys.modules["kaolin.render.camera"] = camera
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing


def _import_official_api():
    notebook_dir = SAM3D_REPO_ROOT / "notebook"
    _install_kaolin_stub()
    for path in (str(SAM3D_REPO_ROOT), str(notebook_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ["LIDRA_SKIP_INIT"] = "true"
    from inference import Inference  # type: ignore

    return Inference


def _write_runtime_config() -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(SAM3D_CONFIG_PATH)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.dtype = "float16"
    config.depth_model.device = "cpu"
    config.decode_formats = ["gaussian"]
    config.slat_decoder_mesh_config_path = None
    config.slat_decoder_mesh_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    OmegaConf.save(config, SAM3D_RUNTIME_CONFIG_PATH)
    return SAM3D_RUNTIME_CONFIG_PATH


def _save_preview(mask: np.ndarray, image_rgb: np.ndarray, preview_path: Path) -> None:
    overlay = image_rgb.copy().astype(np.float32)
    overlay[mask > 0] = 0.65 * overlay[mask > 0] + 0.35 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(preview_path)


def prepare_cropped_sam3d_inputs(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    padding: int = 32,
) -> Dict[str, Path]:
    """Crop the SAM3D inputs tightly around the object mask for lighter inference."""

    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_pil = Image.open(render_image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    mask = _load_binary_mask(object_mask_path, image_pil.size)
    if int(mask.sum()) == 0:
        raise ValueError("SAM3D crop input mask is empty.")

    ys, xs = np.nonzero(mask > 0)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    side = max(y1 - y0, x1 - x0) + 2 * int(padding)
    side = max(side, 32)

    crop_y0 = max(0, int(round(center_y - side / 2)))
    crop_x0 = max(0, int(round(center_x - side / 2)))
    crop_y1 = min(image_rgb.shape[0], crop_y0 + side)
    crop_x1 = min(image_rgb.shape[1], crop_x0 + side)
    crop_y0 = max(0, crop_y1 - side)
    crop_x0 = max(0, crop_x1 - side)

    cropped_image = image_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
    cropped_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

    cropped_render_path = image_dir / f"{output_stem}_crop_render.png"
    cropped_mask_path = image_dir / f"{output_stem}_crop_mask.png"
    Image.fromarray(cropped_image).save(cropped_render_path)
    Image.fromarray((cropped_mask > 0).astype(np.uint8) * 255).save(cropped_mask_path)

    return {
        "render_image_path": cropped_render_path,
        "object_mask_path": cropped_mask_path,
    }


def _resize_image_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    if max(height, width) <= max_side:
        return image_rgb, mask

    scale = max_side / float(max(height, width))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image_resized = np.array(Image.fromarray(image_rgb).resize(new_size, resample=Image.BILINEAR))
    mask_resized = np.array(
        Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(new_size, resample=Image.NEAREST)
    )
    return image_resized, (mask_resized > 127).astype(np.uint8)


def run_sam3d_single_object(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    max_side: int = 518,
) -> Dict[str, Path]:
    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_paths = get_sam3d_output_paths(output_dir, output_stem, image_dir=image_dir)
    ply_path = output_paths["ply_path"]
    glb_path = output_paths["glb_path"]
    pose_path = output_paths["pose_path"]
    preview_path = output_paths["preview_path"]
    run_info_path = output_paths["run_info_path"]

    if not render_image_path.exists():
        raise FileNotFoundError(render_image_path)
    if not object_mask_path.exists():
        raise FileNotFoundError(object_mask_path)
    if not SAM3D_REPO_ROOT.exists():
        raise FileNotFoundError(SAM3D_REPO_ROOT)
    if not SAM3D_CONFIG_PATH.exists():
        raise FileNotFoundError(SAM3D_CONFIG_PATH)

    image_pil = Image.open(render_image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    mask = _load_binary_mask(object_mask_path, image_pil.size)
    if image_rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"SAM3D image/mask shape mismatch: {image_rgb.shape} vs {mask.shape}")
    if int(mask.sum()) == 0:
        raise ValueError("SAM3D input mask is empty.")

    attempted_sizes = []
    candidate_sizes = []
    for size in [max_side, 112, 96, 80, 64, 48]:
        size = min(int(size), int(max_side))
        if size not in candidate_sizes:
            candidate_sizes.append(size)

    runtime_config_path = _write_runtime_config()
    Inference = _import_official_api()
    used_shape = None

    output = None
    for candidate_size in candidate_sizes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        resized_image_rgb, resized_mask = _resize_image_and_mask(image_rgb, mask, max_side=candidate_size)
        attempted_sizes.append((candidate_size, tuple(resized_image_rgb.shape)))
        _save_preview(resized_mask, resized_image_rgb, preview_path)

        inference = None
        try:
            inference = Inference(str(runtime_config_path), compile=False)
            output = inference(resized_image_rgb, resized_mask, seed=42)
            used_shape = tuple(resized_image_rgb.shape)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output = None
            continue
        finally:
            if inference is not None:
                del inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if output is None:
        raise RuntimeError(f"SAM3D failed with CUDA OOM for all candidate sizes: {attempted_sizes}")

    if "gs" not in output:
        raise RuntimeError("SAM3D did not return a gaussian output.")
    output["gs"].save_ply(str(ply_path))

    pose_data = {}
    for key in ("translation", "rotation", "scale"):
        value = output.get(key)
        if value is not None:
            pose_data[key] = torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
    if "rotation" not in pose_data or len(pose_data["rotation"]) != 4:
        raise RuntimeError("SAM3D did not return a valid object rotation pose.")
    if pose_data:
        pose_path.write_text(json.dumps(pose_data, indent=2) + "\n")

    run_info: List[str] = [
        "SAM 3D Objects dynamic-gs integration run",
        f"Rendered image: {render_image_path}",
        f"Object mask: {object_mask_path}",
        f"Original image shape: {tuple(image_rgb.shape)}",
        f"Used inference image shape: {used_shape}",
        f"Attempted sizes: {attempted_sizes}",
        f"Repo root: {SAM3D_REPO_ROOT}",
        f"Config path: {SAM3D_CONFIG_PATH}",
        f"Saved runtime config: {runtime_config_path}",
        f"Saved gaussian splat: {ply_path}",
        f"Saved pose sidecar: {pose_path}",
        f"Saved preview: {preview_path}",
    ]

    glb = output.get("glb")
    if glb is None:
        run_info.append("No GLB object was returned by SAM 3D Objects.")
    else:
        glb.export(str(glb_path))
        run_info.append(f"Saved mesh/glb: {glb_path}")

    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_info_path.write_text("\n".join(run_info) + "\n")
    return output_paths


def run_sam3d_single_object_subprocess(
    render_image_path: Path,
    object_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    image_dir: Path | None = None,
    max_side: int = 518,
) -> Dict[str, Path]:
    """Run the working SAM3D generation path in a fresh Python process.

    This keeps the heavy SAM3D CUDA state separate from the main `ns-train`
    process, which is much more memory-stable on the 8 GB GPU.
    """

    render_image_path = Path(render_image_path)
    object_mask_path = Path(object_mask_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-image",
        str(render_image_path),
        "--object-mask",
        str(object_mask_path),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        output_stem,
        "--image-dir",
        str(image_dir),
        "--max-side",
        str(max_side),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SAM3D subprocess failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    output_paths = get_sam3d_output_paths(output_dir, output_stem, image_dir=image_dir)
    resolved_pose_path = resolve_sam3d_pose_path(output_paths["ply_path"], output_paths["pose_path"])
    if not sam3d_pose_has_rotation(resolved_pose_path):
        raise RuntimeError(
            f"SAM3D subprocess produced `{output_paths['ply_path']}` but no valid rotation pose sidecar was found."
        )
    output_paths["pose_path"] = resolved_pose_path if resolved_pose_path is not None else output_paths["pose_path"]
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-image", type=Path, required=True)
    parser.add_argument("--object-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=str, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=518)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    try:
        run_sam3d_single_object(
            render_image_path=args.render_image,
            object_mask_path=args.object_mask,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            image_dir=args.image_dir,
            max_side=args.max_side,
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"SAM3D worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI path
    raise SystemExit(_main())
