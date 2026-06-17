#!/usr/bin/env python3
"""Split a flat ZED dataset (rgb/depth/masks/transforms.json) into
static_scene/ + dynamic_scene/ at a given frame index, for the dynamic-gs pipeline.

static_scene = frames [0, split), dynamic_scene = frames [split, end).
Each subdir gets its own rgb/ depth/ masks/ + transforms.json with file_paths
renumbered from 000000 and re-pathed (./rgb/.. etc). Camera intrinsics copied verbatim.

Usage: split_flat_dataset.py <flat_src> <out_dir> <split_index>
"""
import json, shutil, sys
from pathlib import Path


def resolve(src: Path, rel: str) -> Path:
    return (src / rel.lstrip("./")).resolve()


def build_phase(src, out_phase, frames, intr_keys, meta):
    for sub in ("rgb", "depth", "masks"):
        (out_phase / sub).mkdir(parents=True, exist_ok=True)
    new_frames = []
    for i, f in enumerate(frames):
        stem = "frame_%06d" % i
        rgb_src = resolve(src, f["file_path"])
        dep_src = resolve(src, f["depth_file_path"])
        msk_src = resolve(src, f["mask_path"])
        rgb_ext = rgb_src.suffix
        dep_ext = dep_src.suffix
        msk_ext = msk_src.suffix
        shutil.copy2(rgb_src, out_phase / "rgb" / (stem + rgb_ext))
        shutil.copy2(dep_src, out_phase / "depth" / (stem + dep_ext))
        shutil.copy2(msk_src, out_phase / "masks" / (stem + msk_ext))
        new_frames.append({
            "file_path": "./rgb/%s%s" % (stem, rgb_ext),
            "depth_file_path": "./depth/%s%s" % (stem, dep_ext),
            "mask_path": "./masks/%s%s" % (stem, msk_ext),
            "transform_matrix": f["transform_matrix"],
        })
    out = {k: meta[k] for k in intr_keys if k in meta}
    out["frames"] = new_frames
    with open(out_phase / "transforms.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return len(new_frames)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    split = int(sys.argv[3])
    meta = json.load(open(src / "transforms.json"))
    frames = meta["frames"]
    intr_keys = ("fl_x", "fl_y", "cx", "cy", "w", "h", "camera_model", "k1", "k2", "p1", "p2")
    n_static = build_phase(src, out / "static_scene", frames[:split], intr_keys, meta)
    n_dyn = build_phase(src, out / "dynamic_scene", frames[split:], intr_keys, meta)
    print("static_scene: %d frames (0..%d)" % (n_static, split - 1))
    print("dynamic_scene: %d frames (%d..%d)" % (n_dyn, split, len(frames) - 1))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
