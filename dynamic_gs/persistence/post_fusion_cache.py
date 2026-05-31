"""Post-fusion model snapshot save/load (warm-cache between sessions).

The snapshot is written at the static→dynamic transition (after Phase 0b
fuses SAM3D objects into the scene). On a subsequent run, loading it skips
static training + Phase 0b entirely and jumps straight to dynamic.

Module is intentionally pure-model: it does NOT touch pipeline-level state
(``_sam3d_inserted``, ``_static_converged_step``, ``_step_offset``, ...).
The caller is responsible for setting those after a successful load.

Caveat: the snapshot is NOT config-tagged. If ``sh_degree``, background
color, or the camera-optimizer mode changes between save and load, the
loaded tensors won't match the rebuilt model and ``load_state_dict`` will
raise. Delete the .pt to recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class PostFusionLoadResult:
    success: bool
    num_points: int = 0
    error: Optional[str] = None


def save_post_fusion_state(model, cache_path: Path) -> bool:
    """Snapshot the post-fusion model so a future run can warm-start.

    Writes ``model.state_dict()`` + ``num_points`` to ``cache_path``.
    Returns True on success.
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_points": int(model.num_points),
            },
            cache_path,
        )
        CONSOLE.log(
            f"[post-fusion-cache] saved {cache_path.name} "
            f"(N={int(model.num_points)} Gaussians)"
        )
        return True
    except Exception as exc:
        CONSOLE.log(f"[post-fusion-cache] save failed: {exc}")
        return False


def load_post_fusion_state(model, cache_path: Path, device) -> PostFusionLoadResult:
    """Restore a post-fusion snapshot into ``model``.

    The cold-start model was built from the SfM seed PLY (small N). The
    snapshot has N_post Gaussians (post Phase-0b insertions), so each
    ``gauss_params`` Parameter has to be re-allocated at N_post before
    ``load_state_dict`` can copy values in.

    The means-grad hook (dynamic-phase gradient masking) is re-bound to
    the freshly-allocated means Parameter — the old hook was bound to the
    pre-resize tensor and would no longer fire.

    Returns ``PostFusionLoadResult(success=True, num_points=N)`` on success,
    or ``PostFusionLoadResult(success=False, error=...)`` on any failure.
    Callers should treat False as "fall back to standard static + fusion".
    """
    try:
        blob = torch.load(cache_path, map_location=device)
        state_dict = blob["model_state_dict"]
        target_n = int(blob["num_points"])
    except Exception as exc:
        msg = f"could not read {cache_path.name}: {exc}"
        CONSOLE.log(f"[post-fusion-cache] {msg}")
        return PostFusionLoadResult(success=False, error=msg)

    means_device = model.means.device
    try:
        # Reallocate each gauss_params Parameter at the saved size.
        for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
            sd_key = f"gauss_params.{name}"
            if sd_key not in state_dict:
                msg = f"missing {sd_key}; falling back to static+fusion"
                CONSOLE.log(f"[post-fusion-cache] {msg}")
                return PostFusionLoadResult(success=False, error=msg)
            old_param = model.gauss_params[name]
            new_tensor = state_dict[sd_key].to(device=means_device, dtype=old_param.dtype)
            model.gauss_params[name] = torch.nn.Parameter(
                new_tensor.clone(),
                requires_grad=old_param.requires_grad,
            )

        # The model's load_state_dict override sees object_flags.shape[0]
        # != target_n and rebuilds the persistent buffers at target_n
        # before copying. strict=False so any new/old keys (e.g. viewer
        # GUI handles) don't crash the load.
        model.load_state_dict(state_dict, strict=False)

        # Re-bind the means-grad hook to the freshly-allocated Parameter
        # (the previous hook was bound to a stale tensor).
        if hasattr(model, "_mask_means_grad"):
            model.gauss_params["means"].register_hook(model._mask_means_grad)
    except Exception as exc:
        msg = f"state_dict load failed: {exc}"
        CONSOLE.log(f"[post-fusion-cache] {msg}")
        return PostFusionLoadResult(success=False, error=msg)

    CONSOLE.log(
        f"[post-fusion-cache] loaded {cache_path.name} "
        f"(N={target_n} Gaussians); skipping static + Phase 0b."
    )
    return PostFusionLoadResult(success=True, num_points=target_n)
