#!/usr/bin/env python3
"""Standalone SAM 3D Objects single-object test.

What this does:
- loads one rendered RGB image and one binary object mask
- checks that they are aligned
- runs the official SAM 3D Objects single-object inference path
- saves the generated Gaussian splat / mesh outputs when available
- always writes a short run log next to the inputs

How to run:
- activate `radiance_ros`
- make sure `third_party/sam-3d-objects` exists
- make sure the official checkpoints are downloaded to
  `third_party/sam-3d-objects/checkpoints/hf`
- see `third_party/sam-3d-objects/LOCAL_COMPATIBILITY_NOTES.md` for the small
  compatibility patches this environment currently needs
- run `python test_sam3d_single_object.py`

Expected input files:
- arm_05460_render.png
- arm_05460_live_object_mask.png
from the `render_masks_esam` folder below.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np
from PIL import Image

IMAGE_PATH = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_masks_esam/"
    "arm_05460_render.png"
)
MASK_PATH = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45/dynamic_scene/render_masks_esam/"
    "arm_05460_live_object_mask.png"
)
OUTPUT_DIR = IMAGE_PATH.parent
REPO_ROOT = Path(__file__).resolve().parent / "third_party" / "sam-3d-objects"
CONFIG_PATH = REPO_ROOT / "checkpoints" / "hf" / "pipeline.yaml"
RUNTIME_CONFIG_PATH = REPO_ROOT / "checkpoints" / "hf" / "pipeline_runtime_small.yaml"

PLY_PATH = OUTPUT_DIR / "sam3d_object_gs.ply"
GLB_PATH = OUTPUT_DIR / "sam3d_object_mesh.glb"
PREVIEW_PATH = OUTPUT_DIR / "sam3d_preview.png"
RUN_INFO_PATH = OUTPUT_DIR / "sam3d_run_info.txt"


def write_run_info(lines: List[str]) -> None:
    RUN_INFO_PATH.write_text("\n".join(lines) + "\n")


def load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, resample=Image.NEAREST)
    mask = np.array(mask_image) > 127
    return mask.astype(np.uint8)


def save_preview(mask: np.ndarray, image_rgb: np.ndarray) -> None:
    overlay = image_rgb.copy().astype(np.float32)
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay[mask > 0] = 0.65 * overlay[mask > 0] + 0.35 * color
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(PREVIEW_PATH)


def install_kaolin_stub() -> None:
    """SAM 3D's notebook wrapper imports kaolin for visualization helpers.

    The single-object inference path used here does not rely on those classes,
    so a tiny stub is enough for this standalone test when kaolin is absent.
    """

    if "kaolin" in sys.modules:
        return

    kaolin = ModuleType("kaolin")
    visualize = ModuleType("kaolin.visualize")
    render = ModuleType("kaolin.render")
    camera = ModuleType("kaolin.render.camera")
    utils = ModuleType("kaolin.utils")
    testing = ModuleType("kaolin.utils.testing")

    class _Dummy:  # pragma: no cover - tiny compatibility shim
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


def import_official_api():
    notebook_dir = REPO_ROOT / "notebook"
    install_kaolin_stub()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(notebook_dir))
    import os

    os.environ["LIDRA_SKIP_INIT"] = "true"
    from inference import Inference, load_image  # type: ignore

    return Inference, load_image


def maybe_export_glb(output: dict, run_info: List[str]) -> None:
    glb = output.get("glb")
    if glb is None:
        run_info.append("No GLB object was returned by SAM 3D Objects.")
        return

    glb.export(str(GLB_PATH))
    run_info.append(f"Saved mesh/glb: {GLB_PATH}")

    if hasattr(glb, "save_image"):
        try:
            image_bytes = glb.save_image(resolution=(800, 800))
            if image_bytes:
                PREVIEW_PATH.write_bytes(image_bytes)
                run_info.append(f"Saved preview: {PREVIEW_PATH}")
        except Exception as exc:  # pragma: no cover - optional preview path
            run_info.append(f"Preview export skipped: {type(exc).__name__}: {exc}")


def write_runtime_config(run_info: List[str]) -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(CONFIG_PATH)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.dtype = "float16"
    config.depth_model.device = "cpu"
    config.decode_formats = ["gaussian"]
    config.slat_decoder_mesh_config_path = None
    config.slat_decoder_mesh_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    OmegaConf.save(config, RUNTIME_CONFIG_PATH)
    run_info.append(f"Saved runtime config: {RUNTIME_CONFIG_PATH}")
    return RUNTIME_CONFIG_PATH


def main() -> int:
    run_info: List[str] = [
        "SAM 3D Objects standalone test",
        f"Rendered image: {IMAGE_PATH}",
        f"Object mask: {MASK_PATH}",
        f"Repo root: {REPO_ROOT}",
        f"Config path: {CONFIG_PATH}",
    ]

    try:
        import torch

        run_info.append(f"torch: {torch.__version__}")
        run_info.append(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            run_info.append(f"gpu: {props.name}")
            run_info.append(f"vram_gb: {props.total_memory / (1024 ** 3):.2f}")
    except Exception as exc:  # pragma: no cover - environment guard
        run_info.append(f"torch import failed: {type(exc).__name__}: {exc}")

    if not IMAGE_PATH.exists():
        run_info.append("ERROR: rendered image does not exist.")
        write_run_info(run_info)
        return 1
    if not MASK_PATH.exists():
        run_info.append("ERROR: object mask does not exist.")
        write_run_info(run_info)
        return 1
    if not REPO_ROOT.exists():
        run_info.append("ERROR: sam-3d-objects repo is not cloned.")
        write_run_info(run_info)
        return 1

    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_rgb = np.array(image_pil)
    mask = load_binary_mask(MASK_PATH, image_pil.size)

    print(f"Rendered image shape: {image_rgb.shape}")
    print(f"Mask shape: {mask.shape}")
    run_info.append(f"render image shape: {image_rgb.shape}")
    run_info.append(f"mask shape: {mask.shape}")
    run_info.append(f"mask pixels > 0: {int(mask.sum())}")

    if image_rgb.shape[:2] != mask.shape[:2]:
        run_info.append("ERROR: image and mask shapes still do not match after resize.")
        write_run_info(run_info)
        return 1
    if int(mask.sum()) == 0:
        run_info.append("ERROR: binary object mask is empty.")
        write_run_info(run_info)
        return 1

    save_preview(mask, image_rgb)
    run_info.append(f"Saved input overlay preview: {PREVIEW_PATH}")

    blockers: List[str] = []
    try:
        Inference, load_image = import_official_api()
    except Exception as exc:
        Inference = None
        load_image = None
        blockers.append(
            f"Failed to import official API: {type(exc).__name__}: {exc}"
        )
        blockers.append(
            "This usually means an official SAM 3D Objects dependency is missing "
            "(in this environment, `kaolin` and `pytorch3d` are the main blockers)."
        )

    if not CONFIG_PATH.exists():
        blockers.append("Official checkpoint/config path is missing.")
        blockers.append(
            "Expected: third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml"
        )
        blockers.append(
            "The checkpoint bundle is gated on Hugging Face; run `hf auth login` and "
            "download `facebook/sam-3d-objects` into `third_party/sam-3d-objects/checkpoints/hf`."
        )

    if blockers:
        run_info.append("ERROR: SAM 3D Objects is not ready for inference in this environment.")
        run_info.extend(blockers)
        write_run_info(run_info)
        return 1

    assert Inference is not None
    assert load_image is not None

    try:
        runtime_config_path = write_runtime_config(run_info)
        image = load_image(str(IMAGE_PATH))
        inference = Inference(str(runtime_config_path), compile=False)
        output = inference(image, mask, seed=42)

        if "gs" in output:
            output["gs"].save_ply(str(PLY_PATH))
            run_info.append(f"Saved gaussian splat: {PLY_PATH}")
        else:
            run_info.append("No Gaussian splat was returned by SAM 3D Objects.")

        maybe_export_glb(output, run_info)
        write_run_info(run_info)
        return 0
    except Exception as exc:
        run_info.append(f"ERROR: inference failed: {type(exc).__name__}: {exc}")
        run_info.append("Traceback:")
        run_info.extend(traceback.format_exc().splitlines())
        write_run_info(run_info)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
