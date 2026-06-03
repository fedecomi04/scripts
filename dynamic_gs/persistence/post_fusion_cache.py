"""Static-scene snapshot save/load (warm-cache between sessions).

The snapshot is written at the static→dynamic transition (after either
Phase 0b in ``static-gs`` or the seed-labeled-and-trained pass in
``static-gs-preseg``). On a subsequent run, loading it skips static
training entirely and jumps straight to dynamic.

Default filename is ``static_state.pt``; the legacy name
``post_fusion_state.pt`` (produced by older static-gs runs) is read as a
fallback by ``load_post_fusion_state`` so existing datasets still warm-start
cleanly. Function names keep the ``post_fusion`` prefix to preserve the
public API used by external dump/merge scripts.

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


_LEGACY_CACHE_NAME = "post_fusion_state.pt"


def save_post_fusion_state(model, cache_path: Path) -> bool:
    """Snapshot the trained static model so a future run can warm-start.

    Writes ``model.state_dict()`` + ``num_points`` to ``cache_path``. The
    default filename is ``static_state.pt`` (configured per-pipeline);
    the function itself writes wherever ``cache_path`` points. Returns
    True on success.
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
            f"[static-cache] saved {cache_path.name} "
            f"(N={int(model.num_points)} Gaussians)"
        )
        return True
    except Exception as exc:
        CONSOLE.log(f"[static-cache] save failed: {exc}")
        return False


def _resolve_cache_path(cache_path: Path) -> Path:
    """Backward-compat: if ``cache_path`` doesn't exist but a legacy
    ``post_fusion_state.pt`` sits next to it, return the legacy path.
    Otherwise return ``cache_path`` unchanged (the caller will see the
    original FileNotFound). Eases the rename for users with existing
    cached snapshots.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path
    legacy = cache_path.with_name(_LEGACY_CACHE_NAME)
    if legacy.exists() and legacy.name != cache_path.name:
        CONSOLE.log(
            f"[static-cache] {cache_path.name} missing; reading legacy "
            f"{legacy.name} (rename it to {cache_path.name} to silence)."
        )
        return legacy
    return cache_path


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
    cache_path = _resolve_cache_path(cache_path)
    try:
        blob = torch.load(cache_path, map_location=device)
        state_dict = blob["model_state_dict"]
        target_n = int(blob["num_points"])
    except Exception as exc:
        msg = f"could not read {cache_path.name}: {exc}"
        CONSOLE.log(f"[static-cache] {msg}")
        return PostFusionLoadResult(success=False, error=msg)

    means_device = model.means.device
    try:
        # Reallocate each gauss_params Parameter at the saved size.
        for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
            sd_key = f"gauss_params.{name}"
            if sd_key not in state_dict:
                msg = f"missing {sd_key}; falling back to static+fusion"
                CONSOLE.log(f"[static-cache] {msg}")
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
        CONSOLE.log(f"[static-cache] {msg}")
        return PostFusionLoadResult(success=False, error=msg)

    CONSOLE.log(
        f"[static-cache] loaded {cache_path.name} "
        f"(N={target_n} Gaussians); skipping static + Phase 0b."
    )
    return PostFusionLoadResult(success=True, num_points=target_n)
