from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image

SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
SAM2_REPO_PATH = Path.home() / ".cache" / "sam2_repo"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_PATH = Path.home() / ".cache" / "sam2" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_OBJECT_ID = 1


def ensure_sam2_repo(repo_path: Path = SAM2_REPO_PATH) -> Path:
    repo_path = Path(repo_path)
    package_root = repo_path / "sam2"
    if package_root.exists():
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", SAM2_REPO_URL, str(repo_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch the official SAM2 source.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to clone SAM2 from {SAM2_REPO_URL}:\n{exc.stderr}") from exc

    if not package_root.exists():
        raise RuntimeError(f"SAM2 clone at {repo_path} is missing the python package.")
    return repo_path


def ensure_sam2_checkpoint(checkpoint_path: Path = SAM2_CHECKPOINT_PATH) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return checkpoint_path
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, checkpoint_path)
    return checkpoint_path


def _import_build_sam2_video_predictor(repo_path: Path = SAM2_REPO_PATH):
    repo_path = ensure_sam2_repo(repo_path)
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SAM2 requires the official source repo plus `hydra-core`, `iopath`, and `omegaconf` "
            "in the active environment."
        ) from exc
    return build_sam2_video_predictor


def build_sam2_tiny_video_predictor(device: torch.device) -> object:
    build_fn = _import_build_sam2_video_predictor()
    checkpoint_path = ensure_sam2_checkpoint()
    return build_fn(
        SAM2_CONFIG_NAME,
        str(checkpoint_path),
        device=str(device),
    )


def _save_jpg_frame(image: torch.Tensor, path: Path) -> None:
    image = image.detach().float().clamp(0.0, 1.0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("SAM2 propagation expects RGB images with shape [H, W, 3].")
    Image.fromarray(image.mul(255).byte().cpu().numpy()).save(path, format="JPEG", quality=95)


def _to_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return (mask.detach().float().cpu() > 0.5).to(torch.uint8)


def query_sam2_propagated_mask(
    predictor: object,
    previous_rendered_rgb: torch.Tensor,
    current_rendered_rgb: torch.Tensor,
    previous_mask: torch.Tensor,
) -> torch.Tensor:
    # SAM2 only produces object_score_logits in eval mode
    if hasattr(predictor, "eval"):
        predictor.eval()
    with tempfile.TemporaryDirectory(prefix="dynamic_gs_sam2_") as tmp_dir:
        frame_dir = Path(tmp_dir)
        _save_jpg_frame(previous_rendered_rgb, frame_dir / "0.jpg")
        _save_jpg_frame(current_rendered_rgb, frame_dir / "1.jpg")

        state = predictor.init_state(
            str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=SAM2_OBJECT_ID,
            mask=_to_binary_mask(previous_mask),
        )

        propagated = None
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=2,
        ):
            if frame_idx != 1:
                continue
            if not object_ids:
                continue
            propagated = mask_logits[0, 0] > 0
            break

    if propagated is None:
        height, width = current_rendered_rgb.shape[:2]
        return torch.zeros((height, width), dtype=torch.bool, device=current_rendered_rgb.device)

    return propagated.to(device=current_rendered_rgb.device, dtype=torch.bool)
