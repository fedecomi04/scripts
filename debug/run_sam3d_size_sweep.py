#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_gs.utils.sam3d import prepare_cropped_sam3d_inputs, run_sam3d_single_object_subprocess

DEFAULT_RENDER_DIR = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45_w_background/dynamic_scene/render_masks_esam"
)
DEFAULT_RENDER_IMAGE = DEFAULT_RENDER_DIR / "arm_05460_render.png"
DEFAULT_MASK_IMAGE = DEFAULT_RENDER_DIR / "arm_05460_render_object_mask.png"
DEFAULT_DEBUG_DIR = DEFAULT_RENDER_DIR.parent / "debug"
DEFAULT_SIZES = [48, 64, 80, 96, 112, 192]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3D with a sweep of max-side sizes.")
    parser.add_argument("--render-image", type=Path, default=DEFAULT_RENDER_IMAGE)
    parser.add_argument("--object-mask", type=Path, default=DEFAULT_MASK_IMAGE)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def _copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    args = _parse_args()
    render_image = args.render_image.resolve()
    object_mask = args.object_mask.resolve()
    debug_dir = args.debug_dir.resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not render_image.exists():
        raise FileNotFoundError(f"Render image not found: {render_image}")
    if not object_mask.exists():
        raise FileNotFoundError(f"Object mask not found: {object_mask}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.output_name or f"sam3d_size_sweep_{timestamp}"
    run_root = debug_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    crop_root = run_root / "cropped_inputs"
    cropped = prepare_cropped_sam3d_inputs(
        render_image_path=render_image,
        object_mask_path=object_mask,
        output_dir=crop_root,
        output_stem="arm_05460_sweep_base",
        image_dir=crop_root,
    )

    summary_lines = [
        "SAM3D size sweep",
        f"render_image: {render_image}",
        f"object_mask: {object_mask}",
        f"cropped_render_image: {cropped['render_image_path']}",
        f"cropped_object_mask: {cropped['object_mask_path']}",
        f"output_root: {run_root}",
        f"sizes: {list(args.sizes)}",
        "",
    ]

    for size in args.sizes:
        size_dir = run_root / f"size_{int(size):03d}"
        size_dir.mkdir(parents=True, exist_ok=True)
        output_stem = f"arm_05460_sam3d_size_{int(size):03d}"
        t0 = time.perf_counter()
        status = "success"
        error_text = ""
        outputs = {}

        try:
            outputs = run_sam3d_single_object_subprocess(
                render_image_path=cropped["render_image_path"],
                object_mask_path=cropped["object_mask_path"],
                output_dir=size_dir,
                output_stem=output_stem,
                image_dir=size_dir,
                max_side=int(size),
            )
        except Exception as exc:
            status = "error"
            error_text = f"{type(exc).__name__}: {exc}"
            (size_dir / "error.txt").write_text(error_text + "\n")

        elapsed = time.perf_counter() - t0

        run_info_path = size_dir / f"{output_stem}_run_info.txt"
        pose_path = outputs.get("pose_path")
        ply_path = outputs.get("ply_path")
        preview_path = outputs.get("preview_path")

        summary_lines.extend(
            [
                f"[size={int(size)}]",
                f"status: {status}",
                f"elapsed_seconds: {elapsed:.3f}",
                f"run_info_path: {run_info_path if run_info_path.exists() else 'missing'}",
                f"ply_path: {ply_path if ply_path and Path(ply_path).exists() else 'missing'}",
                f"pose_path: {pose_path if pose_path and Path(pose_path).exists() else 'missing'}",
                f"preview_path: {preview_path if preview_path and Path(preview_path).exists() else 'missing'}",
                f"error: {error_text if error_text else 'none'}",
                "",
            ]
        )

    summary_path = run_root / "sweep_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
