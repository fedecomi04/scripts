"""Idempotent wrapper to ensure Design Invariant #3.

Invariant #3: ``<dataset>/static_scene/transforms.json`` must contain
ICP-refined poses (``pose_source == "icp_refined_from_urdf_v1"``) and
``transforms_urdf_backup.json`` must exist preserving the original URDF
poses.

This module wraps :func:`rewrite_transforms_with_icp` (see the script of
the same name in ``scripts/``) with:

1. An idempotency guard — re-running on an already-refined dataset is a
   cheap no-op.
2. A backup-collision guard — refuses to clobber an existing
   ``transforms_urdf_backup.json`` unless ``force=True``.
3. A CPU-mode default (``DGS_FUSION_DEVICE=cpu``) so this can run safely
   alongside SAM2/SAM3 GPU workloads. Pre-existing ``DGS_FUSION_DEVICE``
   settings are respected.
4. A stale-PLY warning so callers know if the seed cloud predates the
   refined transforms (the existing PLY is still usable for preseg vote
   transfer, but a refresh may be desirable).

Public API: :func:`refine_poses_and_refuse`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add scripts/ to sys.path so we can import rewrite_transforms_with_icp.
# This file lives at scripts/dynamic_gs/utils/icp_pose_refine.py; the
# script lives at scripts/rewrite_transforms_with_icp.py — go up two.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Delegate to the existing implementation (DRY).
from rewrite_transforms_with_icp import (  # noqa: E402
    rewrite_transforms_with_icp as _rewrite_transforms_with_icp,
)


_POSE_SOURCE_TAG = "icp_refined_from_urdf_v1"
_LOG = "[icp-pose-refine]"


def _log(msg: str) -> None:
    print(f"{_LOG} {msg}", flush=True)


def refine_poses_and_refuse(dataset_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Ensure Design Invariant #3 holds for ``dataset_dir``.

    If ``<dataset_dir>/static_scene/transforms.json`` already has
    ``pose_source == "icp_refined_from_urdf_v1"``, this is a no-op and
    returns immediately with ``{'skipped': True, ...}``.

    Otherwise:
      1. Back up ``transforms.json`` → ``transforms_urdf_backup.json``
         (atomic copy; aborts if the backup already exists and
         ``force=False``).
      2. Run :func:`rewrite_transforms_with_icp` in CPU fusion mode
         (``DGS_FUSION_DEVICE=cpu``) to re-fuse and capture refined
         per-frame poses.
      3. The delegated rewrite writes refined transforms with the
         ``pose_source`` flag (atomic via ``.tmp`` + ``replace``).
      4. Log a warning if the seed PLY mtime predates the refined
         transforms (PLY refresh may be desired but is not done here).

    Parameters
    ----------
    dataset_dir:
        Dataset root containing ``static_scene/``.
    force:
        If ``True``, overwrite an existing ``transforms_urdf_backup.json``
        with the current ``transforms.json`` before refining. Use with
        care — this discards any prior backup.

    Returns
    -------
    dict
        Always contains: ``refined`` (bool), ``skipped`` (bool),
        ``reason`` (str | None), ``backup_path`` (str),
        ``transforms_path`` (str).
        When refined: also ``frames`` (int), ``dt_mm_median`` (float),
        ``dR_deg_median`` (float).
    """
    dataset_dir = Path(dataset_dir).resolve()
    static_dir = dataset_dir / "static_scene"
    transforms_path = static_dir / "transforms.json"
    backup_path = static_dir / "transforms_urdf_backup.json"

    if not transforms_path.exists():
        raise FileNotFoundError(
            f"{_LOG} transforms.json not found: {transforms_path}"
        )

    # --- Idempotency guard ---------------------------------------------------
    try:
        meta = json.loads(transforms_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"{_LOG} failed to parse {transforms_path}: {exc}"
        ) from exc

    if meta.get("pose_source") == _POSE_SOURCE_TAG:
        _log(
            f"transforms.json already marked pose_source={_POSE_SOURCE_TAG!r} "
            f"→ skipping (idempotent no-op)"
        )
        # Sanity: warn if backup is somehow missing despite the flag.
        if not backup_path.exists():
            _log(
                f"WARNING: transforms marked refined but backup is missing "
                f"at {backup_path} — invariant #3 partially violated, "
                f"but cannot reconstruct URDF poses from refined ones"
            )
        return {
            "refined": False,
            "skipped": True,
            "reason": "already_refined",
            "backup_path": str(backup_path),
            "transforms_path": str(transforms_path),
        }

    # --- Backup-collision guard ---------------------------------------------
    if backup_path.exists() and not force:
        # The transforms.json is NOT marked refined (we'd have returned
        # above), yet a backup exists. This indicates a partially-completed
        # prior run, manual tampering, or some other inconsistent state.
        # Refuse to clobber; the caller can pass force=True to override.
        raise RuntimeError(
            f"{_LOG} backup already exists at {backup_path} but "
            f"transforms.json is not marked refined "
            f"(pose_source={meta.get('pose_source')!r}). "
            f"Refusing to clobber the existing backup. "
            f"Pass force=True to overwrite, or inspect the dataset manually."
        )

    # Atomic backup (copy2 preserves mtime so the stale-PLY check below
    # remains meaningful). If force=True and the backup already exists,
    # overwrite it with the current transforms.json.
    try:
        # shutil.copy2 will overwrite if the destination exists.
        shutil.copy2(transforms_path, backup_path)
        if force and backup_path.exists():
            _log(f"force=True — overwrote backup at {backup_path}")
        else:
            _log(f"backed up URDF transforms → {backup_path}")
    except Exception as exc:
        raise RuntimeError(
            f"{_LOG} failed to back up transforms.json to {backup_path}: {exc}"
        ) from exc

    # --- Run ICP refine -----------------------------------------------------
    # Force CPU fusion unless the caller has already set the env var.
    # GPU fusion may compete with SAM2/SAM3 for VRAM in the preseg pipeline;
    # CPU is slower (per CLAUDE.md: ~74 s for 71 frames) but always safe.
    prev_device = os.environ.get("DGS_FUSION_DEVICE")
    if prev_device is None:
        os.environ["DGS_FUSION_DEVICE"] = "cpu"
        _log("setting DGS_FUSION_DEVICE=cpu (no prior value)")
    else:
        _log(f"DGS_FUSION_DEVICE already set to {prev_device!r} — respecting caller")

    transforms_mtime_before = transforms_path.stat().st_mtime
    t0 = time.time()
    try:
        # The delegated function performs its own backup step internally
        # (it skips if backup_path already exists, which it does now —
        # we just created it). It then runs ICP and rewrites
        # transforms.json with pose_source=_POSE_SOURCE_TAG.
        result = _rewrite_transforms_with_icp(dataset_dir, dry_run=False)
    except Exception as exc:
        # Re-raise with context. Note: the delegated function may have
        # partially written transforms.json before failing — but it uses
        # atomic .tmp + replace, so transforms.json is either fully old
        # or fully new. The backup we made is the source of truth either
        # way.
        raise RuntimeError(
            f"{_LOG} ICP refine failed for {dataset_dir}: {exc}"
        ) from exc
    finally:
        # Restore env var to its prior state to avoid leaking into
        # downstream code in the same process.
        if prev_device is None:
            os.environ.pop("DGS_FUSION_DEVICE", None)
        else:
            os.environ["DGS_FUSION_DEVICE"] = prev_device

    elapsed = time.time() - t0
    _log(f"ICP refine complete in {elapsed:.1f}s")

    # --- Stale-PLY warning --------------------------------------------------
    ply_path = static_dir / "depth_camera_init_points.ply"
    transforms_mtime_after = transforms_path.stat().st_mtime
    if ply_path.exists():
        ply_mtime = ply_path.stat().st_mtime
        if ply_mtime < transforms_mtime_after:
            _log(
                f"WARNING: seed PLY {ply_path.name} (mtime "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ply_mtime))}) "
                f"predates refined transforms.json "
                f"(mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(transforms_mtime_after))}). "
                f"PLY is still usable for preseg vote transfer, but consider "
                f"re-fusing to align the seed cloud with refined poses."
            )
    else:
        _log(
            f"NOTE: no seed PLY at {ply_path} — skipping stale-PLY check"
        )

    # --- Build return dict --------------------------------------------------
    frames = int(result.get("frames", 0))
    dt_mm = result.get("dt_mm", [])
    dR_deg = result.get("dR_deg", [])
    dt_mm_median = float(np.median(dt_mm)) if len(dt_mm) else 0.0
    dR_deg_median = float(np.median(dR_deg)) if len(dR_deg) else 0.0

    return {
        "refined": True,
        "skipped": False,
        "reason": None,
        "backup_path": str(backup_path),
        "transforms_path": str(transforms_path),
        "frames": frames,
        "dt_mm_median": dt_mm_median,
        "dR_deg_median": dR_deg_median,
        "elapsed_seconds": float(result.get("elapsed_seconds", elapsed)),
    }


__all__ = ["refine_poses_and_refuse"]
