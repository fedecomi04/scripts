#!/usr/bin/env python3
"""Prepare raw recorded datasets for the static/dynamic pipeline layout.

This script converts raw recordings with a flat layout:

<raw_dataset>/
  rgb/
  depth/
  masks/
  transforms.json
  depth_camera_init_points.ply

into pipeline-ready datasets:

<output_dataset>/
  static_scene/
    rgb/
    depth/
    masks/
    transforms.json
    depth_camera_init_points.ply
  dynamic_scene/
    rgb/
    depth/
    masks/
    transforms.json
    debug/
    render_masks_esam/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets"
)
DEFAULT_RAW_ROOT = DEFAULT_DATASET_ROOT / "dynaarm_gs_depth_mask_01"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rewrite_frame_paths(frame: dict) -> dict:
    updated = dict(frame)
    updated["file_path"] = f"./rgb/{Path(frame['file_path']).name}"
    if "depth_file_path" in frame:
        updated["depth_file_path"] = f"./depth/{Path(frame['depth_file_path']).name}"
    if "mask_path" in frame:
        updated["mask_path"] = f"./masks/{Path(frame['mask_path']).name}"
    return updated


def _write_scene(
    source_root: Path,
    scene_root: Path,
    scene_meta: dict,
    frames: Iterable[dict],
    include_ply: bool,
) -> None:
    frame_list = list(frames)
    (scene_root / "rgb").mkdir(parents=True, exist_ok=True)
    (scene_root / "depth").mkdir(parents=True, exist_ok=True)
    (scene_root / "masks").mkdir(parents=True, exist_ok=True)

    for frame in frame_list:
        rgb_name = Path(frame["file_path"]).name
        _link_or_copy(source_root / "rgb" / rgb_name, scene_root / "rgb" / rgb_name)

        depth_name = Path(frame["depth_file_path"]).name
        _link_or_copy(source_root / "depth" / depth_name, scene_root / "depth" / depth_name)

        mask_path = frame.get("mask_path")
        if mask_path:
            mask_name = Path(mask_path).name
            _link_or_copy(source_root / "masks" / mask_name, scene_root / "masks" / mask_name)

    meta_out = dict(scene_meta)
    meta_out["frames"] = [_rewrite_frame_paths(frame) for frame in frame_list]

    if include_ply and scene_meta.get("ply_file_path"):
        ply_name = Path(scene_meta["ply_file_path"]).name
        _link_or_copy(source_root / ply_name, scene_root / ply_name)
        meta_out["ply_file_path"] = ply_name
    else:
        meta_out.pop("ply_file_path", None)

    (scene_root / "transforms.json").write_text(json.dumps(meta_out, indent=2) + "\n")


def _prepare_dataset(
    source_root: Path,
    output_root: Path,
    last_static_frame: str,
    overwrite: bool,
) -> tuple[int, int]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)

    meta = json.loads((source_root / "transforms.json").read_text())
    frames = meta.get("frames", [])
    target_stem = f"arm_{last_static_frame}"
    stems = [Path(frame["file_path"]).stem for frame in frames]

    if target_stem not in stems:
        raise ValueError(f"{target_stem} not found in {source_root / 'transforms.json'}")

    split_idx = stems.index(target_stem)
    static_frames = frames[: split_idx + 1]
    dynamic_frames = frames[split_idx + 1 :]

    static_meta = {k: v for k, v in meta.items() if k != "frames"}
    dynamic_meta = {k: v for k, v in meta.items() if k not in {"frames", "ply_file_path"}}

    _write_scene(
        source_root=source_root,
        scene_root=output_root / "static_scene",
        scene_meta=static_meta,
        frames=static_frames,
        include_ply=True,
    )
    _write_scene(
        source_root=source_root,
        scene_root=output_root / "dynamic_scene",
        scene_meta=dynamic_meta,
        frames=dynamic_frames,
        include_ply=False,
    )

    (output_root / "dynamic_scene" / "debug").mkdir(parents=True, exist_ok=True)
    (output_root / "dynamic_scene" / "render_masks_esam").mkdir(parents=True, exist_ok=True)

    return len(static_frames), len(dynamic_frames)


def _parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"Expected <dataset_name>:<last_static_frame>, got {text!r}"
        )
    name, frame = text.split(":", 1)
    frame = frame.strip()
    if not frame.isdigit():
        raise argparse.ArgumentTypeError(f"Frame must be numeric, got {frame!r}")
    return name.strip(), frame.zfill(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root folder containing raw recorded datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder where pipeline-ready datasets will be written.",
    )
    parser.add_argument(
        "--pair",
        dest="pairs",
        action="append",
        type=_parse_pair,
        required=True,
        help="Dataset split pair formatted as <dataset_name>:<last_static_frame>.",
    )
    parser.add_argument(
        "--prefix",
        default="dynamic_gs_test_",
        help="Output dataset name prefix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output datasets with the same names.",
    )
    args = parser.parse_args()

    for dataset_name, last_static_frame in args.pairs:
        source_root = args.raw_root / dataset_name
        output_root = args.output_root / f"{args.prefix}{dataset_name}"
        static_count, dynamic_count = _prepare_dataset(
            source_root=source_root,
            output_root=output_root,
            last_static_frame=last_static_frame,
            overwrite=args.overwrite,
        )
        print(
            f"{dataset_name} -> {output_root.name}: "
            f"static={static_count}, dynamic={dynamic_count}, split=arm_{last_static_frame}"
        )


if __name__ == "__main__":
    main()
