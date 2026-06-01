"""Open3D viewer for the TEASER vs CPD registration comparison.

Loads the outputs of ``run_teaser_registration_only.py`` plus the original
CPD-aligned source (saved by the live Phase-0b run as
``..._source_visible_work_iter_00.ply``) and shows them in one window:

  - Target (existing scene region)   : gray
  - Original CPD aligned (saved)     : red
  - Re-run CPD aligned (sanity check): orange (toggle 1)
  - TEASER aligned                   : green (toggle 2)

Press H in the Open3D window for the full key list.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def _load(path: Path, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if color is not None:
        pcd.paint_uniform_color(list(color))
    return pcd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--object-stem", default="static0_obj_00_sam3d")
    ap.add_argument("--no-paint", action="store_true",
                    help="Keep PLY colors instead of repainting per source")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Substrings; only show TEASER variants whose filename contains any of these. "
                         "Example: --variants v5 v7 v8")
    ap.add_argument("--no-target", action="store_true",
                    help="Hide the target cloud (useful when comparing only source alignments).")
    args = ap.parse_args()

    art_dir = args.dataset_root / "dynamic_scene" / "initialization_artifacts"
    cmp_dir = art_dir / "registration_compare"

    target = cmp_dir / f"{args.object_stem}_target_ref.ply"
    cpd_rerun = cmp_dir / f"{args.object_stem}_cpd_rerun_aligned.ply"
    cpd_orig = art_dir / f"{args.object_stem}_source_visible_work_iter_00.ply"

    teaser_variants = sorted(cmp_dir.glob(f"{args.object_stem}_teaser*_aligned.ply"))
    if args.variants:
        teaser_variants = [p for p in teaser_variants if any(v in p.name for v in args.variants)]
    if not target.exists() or not teaser_variants:
        print(f"FATAL: missing comparison outputs in {cmp_dir}.\n"
              f"Run scripts/run_teaser_registration_only.py {args.dataset_root} first.")
        return 1

    # Palette for TEASER variants — distinct hues. Cycles if >8 variants.
    teaser_palette = [
        (0.10, 0.75, 0.20),  # green
        (0.10, 0.55, 0.95),  # blue
        (0.95, 0.85, 0.10),  # yellow
        (0.85, 0.10, 0.85),  # magenta
        (0.10, 0.85, 0.85),  # cyan
        (0.95, 0.45, 0.10),  # orange
        (0.55, 0.35, 0.95),  # purple
        (0.45, 0.95, 0.45),  # light green
    ]
    paint = not args.no_paint
    geoms: list[o3d.geometry.Geometry] = []
    legend: list[tuple[str, tuple[float, float, float] | None, int]] = []

    def add(label: str, path: Path, color: tuple[float, float, float] | None):
        pcd = _load(path, color=color if paint else None)
        geoms.append(pcd)
        legend.append((label, color, len(pcd.points)))

    if not args.no_target:
        add("target", target, (0.55, 0.55, 0.55))
    if cpd_orig.exists():
        add("CPD (original, saved)", cpd_orig, (0.85, 0.10, 0.10))
    if cpd_rerun.exists():
        add("CPD (re-run)", cpd_rerun, (1.00, 0.55, 0.10))
    for i, vpath in enumerate(teaser_variants):
        tag = vpath.stem.replace(f"{args.object_stem}_teaser", "").replace("_aligned", "").lstrip("_") or "default"
        add(f"TEASER [{tag}]", vpath, teaser_palette[i % len(teaser_palette)])

    bbox = geoms[0].get_axis_aligned_bounding_box()
    frame_size = max(0.05, 0.25 * float(np.linalg.norm(bbox.get_extent())))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    print("Loaded clouds:")
    for label, color, n in legend:
        hex_color = "" if color is None else f"  ({int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x})"
        print(f"  {label:<40s} {n:>8,}{hex_color}")
    print()
    print("Press H in the Open3D window for keyboard shortcuts; close window to exit.")

    o3d.visualization.draw_geometries(
        geoms, window_name="SAM3D registration: CPD vs TEASER++",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
