"""Score every TEASER/CPD alignment against the target cloud.

For each aligned source PLY in ``registration_compare/`` (plus the original
CPD result in ``initialization_artifacts/``), reports:
    fitness@<thr> : fraction of source points with a target NN within <thr>
    inlier_rmse  : RMS of the inlier distances
    mean_nn      : mean source->target nearest-neighbor distance
    chamfer      : symmetric mean(NN src->tgt) + mean(NN tgt->src)

Two reference baselines are also shown:
    BASELINE_identity : the source_reg_ref.ply (no refinement at all)
    BASELINE_original_cpd : the original live-pipeline CPD result

Caveats:
- Target is a sparse single-view back-projection. A source that collapses
  onto a small visible patch can score artificially low on mean_nn but
  cover poorly. Watch chamfer + fitness together, not just mean_nn.
- These metrics measure "does the source land near target points." They do
  NOT measure pose correctness when the target is partial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def _eval(src_pts: np.ndarray, tgt_pts: np.ndarray, threshold: float) -> dict[str, float]:
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))
    tgt_tree = o3d.geometry.KDTreeFlann(tgt)
    src_tree = o3d.geometry.KDTreeFlann(src)

    n_src = len(src.points)
    n_tgt = len(tgt.points)
    src_to_tgt = np.empty(n_src, dtype=np.float64)
    for i, p in enumerate(src.points):
        _, _, d2 = tgt_tree.search_knn_vector_3d(p, 1)
        src_to_tgt[i] = np.sqrt(d2[0])
    tgt_to_src = np.empty(n_tgt, dtype=np.float64)
    for i, p in enumerate(tgt.points):
        _, _, d2 = src_tree.search_knn_vector_3d(p, 1)
        tgt_to_src[i] = np.sqrt(d2[0])

    inlier_mask = src_to_tgt < threshold
    return {
        "n_src": float(n_src),
        "n_tgt": float(n_tgt),
        "fitness": float(inlier_mask.mean()),
        "inlier_rmse": float(np.sqrt((src_to_tgt[inlier_mask] ** 2).mean())) if inlier_mask.any() else float("nan"),
        "mean_nn": float(src_to_tgt.mean()),
        "median_nn": float(np.median(src_to_tgt)),
        "chamfer": float(src_to_tgt.mean() + tgt_to_src.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--object-stem", default="static0_obj_00_sam3d")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="Inlier distance threshold for fitness/inlier_rmse (m).")
    args = ap.parse_args()

    art_dir = args.dataset_root / "dynamic_scene" / "initialization_artifacts"
    cmp_dir = art_dir / "registration_compare"
    target = cmp_dir / f"{args.object_stem}_target_ref.ply"
    if not target.exists():
        print(f"FATAL: missing target {target}")
        return 1
    tgt_pts = np.asarray(o3d.io.read_point_cloud(str(target)).points)

    rows: list[tuple[str, dict[str, float]]] = []

    identity_ply = art_dir / f"{args.object_stem}_source_reg_ref.ply"
    if identity_ply.exists():
        src_pts = np.asarray(o3d.io.read_point_cloud(str(identity_ply)).points)
        rows.append(("BASELINE_identity (no refine)", _eval(src_pts, tgt_pts, args.threshold)))

    cpd_orig = art_dir / f"{args.object_stem}_source_visible_work_iter_00.ply"
    if cpd_orig.exists():
        src_pts = np.asarray(o3d.io.read_point_cloud(str(cpd_orig)).points)
        rows.append(("BASELINE_original_cpd (live)", _eval(src_pts, tgt_pts, args.threshold)))

    for vpath in sorted(cmp_dir.glob(f"{args.object_stem}_teaser*_aligned.ply")):
        tag = vpath.stem.replace(f"{args.object_stem}_teaser_", "").replace("_aligned", "")
        src_pts = np.asarray(o3d.io.read_point_cloud(str(vpath)).points)
        rows.append((f"TEASER {tag}", _eval(src_pts, tgt_pts, args.threshold)))

    sort_keys = ["chamfer", "fitness", "inlier_rmse", "mean_nn"]
    print(f"\n  Target points: {len(tgt_pts):,}    Inlier threshold: {args.threshold*1000:.0f} mm\n")
    header = f"  {'variant':<40s}  {'n_src':>8s}  {'fitness':>9s}  {'inlier_rmse':>12s}  {'mean_nn':>9s}  {'chamfer':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in rows:
        print(f"  {name:<40s}  {m['n_src']:>8.0f}  {m['fitness']*100:>8.1f}%  {m['inlier_rmse']*1000:>9.2f} mm  {m['mean_nn']*1000:>7.2f} mm  {m['chamfer']*1000:>7.2f} mm")

    print()
    print("  Sorted by chamfer (lower = better symmetric coverage):")
    for name, m in sorted(rows, key=lambda x: x[1]["chamfer"]):
        print(f"    {m['chamfer']*1000:>7.2f} mm   {name}")
    print()
    print("  Sorted by fitness (higher = better inlier coverage):")
    for name, m in sorted(rows, key=lambda x: -x[1]["fitness"]):
        print(f"    {m['fitness']*100:>5.1f}%   {name}")

    out_json = cmp_dir / "scores.json"
    out_json.write_text(json.dumps({name: m for name, m in rows}, indent=2))
    print(f"\n  Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
