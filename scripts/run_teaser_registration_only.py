"""Run TEASER++ (and re-run CPD) on already-prepared SAM3D inputs from a dataset.

Reads the downsampled source/target PLYs that ``register_and_fuse_sam3d_object``
wrote during the original Phase-0b run, calls each backend's refinement
function directly, and writes aligned outputs to a NEW subdir
``initialization_artifacts/registration_compare/`` so the original CPD
artifacts are untouched.

Usage:
    python scripts/run_teaser_registration_only.py <dataset_root>

Then:
    python scripts/view_registration_comparison.py <dataset_root>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dynamic_gs.utils.sam3d_fusion import (  # noqa: E402
    _median_nn_distance,
    _run_icp_polish,
    _run_probreg_similarity_refinement,
    _run_teaser_reproject_refinement,
    _run_teaser_similarity_refinement,
    _transform_points,
    load_sam3d_gaussian_ply,
    save_point_cloud,
)


def _load_target_ref(target_ply: Path) -> tuple[np.ndarray, np.ndarray]:
    pts, colors = load_sam3d_gaussian_ply(target_ply)
    return pts.astype(np.float32), colors.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--object-stem", default="static0_obj_00_sam3d",
                    help="Output stem used by the original Phase-0b run (default: static0_obj_00_sam3d)")
    ap.add_argument("--noise-bound", type=float, default=0.02,
                    help="TEASER noise_bound in metric meters")
    ap.add_argument("--color-weight", type=float, default=0.3,
                    help="TEASER color weight in [0,1]")
    ap.add_argument("--max-correspondences", type=int, default=5000)
    ap.add_argument("--feature-radius-mult", type=float, default=5.0)
    ap.add_argument("--normal-radius-mult", type=float, default=2.0)
    ap.add_argument("--fpfh-max-nn", type=int, default=100,
                    help="Cap on FPFH neighbor count. Raise to make feature_radius_mult actually matter on dense clouds.")
    ap.add_argument("--normal-max-nn", type=int, default=30)
    ap.add_argument("--ratio-thresh", type=float, default=None,
                    help="Lowe ratio test on source->target descriptor matches. None disables. Typical: 0.85-0.95.")
    ap.add_argument("--multiscale-fpfh", action="store_true",
                    help="Use multi-scale FPFH (radii 5x, 10x, 20x voxel concatenated) instead of single-radius.")
    ap.add_argument("--normal-filter-deg", type=float, default=None,
                    help="Reject mutual-NN correspondences whose source/target normals disagree by > N degrees (uses abs(cos)).")
    ap.add_argument("--reproject-corr", action="store_true",
                    help="After the FPFH-TEASER pass, rebuild correspondences via Euclidean NN in 3D and re-solve with TEASER.")
    ap.add_argument("--reproject-max-corr-mult", type=float, default=3.0,
                    help="Euclidean NN max distance for --reproject-corr, in units of voxel_size.")
    ap.add_argument("--reproject-noise-bound", type=float, default=0.005,
                    help="Noise bound for the second TEASER pass in --reproject-corr (meters).")
    ap.add_argument("--post-icp", action="store_true",
                    help="Apply point-to-plane ICP polish after the (last) TEASER pass.")
    ap.add_argument("--icp-max-corr-mult", type=float, default=2.0,
                    help="ICP max_correspondence_distance in units of voxel_size.")
    ap.add_argument("--icp-iterations", type=int, default=50)
    ap.add_argument("--post-cpd", action="store_true",
                    help="Apply probreg CPD as a final polish stage, initialized from current transform.")
    ap.add_argument("--run-cpd", action="store_true",
                    help="Also re-run probreg CPD for comparison (slow; the original CPD ply is already on disk).")
    ap.add_argument("--out-tag", default="",
                    help="Optional suffix appended to TEASER output filenames so multiple variants can coexist.")
    args = ap.parse_args()
    tag = f"_{args.out_tag}" if args.out_tag else ""

    art_dir = args.dataset_root / "dynamic_scene" / "initialization_artifacts"
    if not art_dir.exists():
        print(f"FATAL: artifact dir not found: {art_dir}")
        return 1

    src_ref_path = art_dir / f"{args.object_stem}_source_reg_ref.ply"
    tgt_ref_path = art_dir / f"{args.object_stem}_target_reg_ref.ply"
    cpd_ref_path = art_dir / f"{args.object_stem}_source_visible_work_iter_00.ply"
    if not src_ref_path.exists() or not tgt_ref_path.exists():
        print(f"FATAL: missing inputs in {art_dir}: source_reg_ref + target_reg_ref")
        return 1

    out_dir = art_dir / "registration_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_pts, src_colors = load_sam3d_gaussian_ply(src_ref_path)
    tgt_pts, tgt_colors = _load_target_ref(tgt_ref_path)
    src_pts = src_pts.astype(np.float32)
    src_colors = src_colors.astype(np.float32)

    # voxel_size derived the same way register_and_fuse_sam3d_object would for
    # these already-downsampled inputs (mostly idempotent).
    src_spacing = float(_median_nn_distance(src_pts))
    tgt_spacing = float(_median_nn_distance(tgt_pts))
    voxel_size = max(3.0 * max(src_spacing, tgt_spacing), 1e-3)
    init = np.eye(4, dtype=np.float32)

    print(f"Source points: {len(src_pts):,}   spacing: {src_spacing:.5f} m")
    print(f"Target points: {len(tgt_pts):,}   spacing: {tgt_spacing:.5f} m")
    print(f"voxel_size used downstream: {voxel_size:.5f} m")
    print()

    if args.run_cpd:
        print("== Running probreg CPD refinement ==")
        t0 = time.time()
        cpd_T, cpd_n, cpd_meta = _run_probreg_similarity_refinement(
            src_pts, src_colors, tgt_pts, tgt_colors, init.copy(), voxel_size,
        )
        cpd_secs = time.time() - t0
        print(f"  done in {cpd_secs:.2f}s  | meta: {cpd_meta}")
        print()
    else:
        cpd_T = init.copy()
        cpd_n = 0
        cpd_meta = {"stop_reason": "not_run_use_existing_artifact"}
        cpd_secs = 0.0
        print("== Skipping CPD re-run (pass --run-cpd to enable). ==")
        print("    The original CPD result is still available at:")
        print(f"    {cpd_ref_path}")
        print()

    multi_scale_radii = [5.0, 10.0, 20.0] if args.multiscale_fpfh else None

    print("== Running TEASER++ refinement (FPFH stage) ==")
    t0 = time.time()
    teaser_T, teaser_n, teaser_meta = _run_teaser_similarity_refinement(
        src_pts, src_colors, tgt_pts, tgt_colors, init.copy(), voxel_size,
        noise_bound=args.noise_bound,
        max_correspondences=args.max_correspondences,
        normal_radius_mult=args.normal_radius_mult,
        feature_radius_mult=args.feature_radius_mult,
        color_weight=args.color_weight,
        fpfh_max_nn=args.fpfh_max_nn,
        normal_max_nn=args.normal_max_nn,
        ratio_thresh=args.ratio_thresh,
        multi_scale_radii=multi_scale_radii,
        normal_filter_deg=args.normal_filter_deg,
    )
    teaser_secs = time.time() - t0
    print(f"  done in {teaser_secs:.2f}s  | meta: {teaser_meta}")
    print()

    final_T = teaser_T
    reproject_meta = None
    reproject_secs = 0.0
    if args.reproject_corr:
        print("== Stage 2: Euclidean-NN reproject + TEASER ==")
        t0 = time.time()
        final_T, _, reproject_meta = _run_teaser_reproject_refinement(
            src_pts, tgt_pts, final_T, voxel_size,
            noise_bound=args.reproject_noise_bound,
            max_corr_dist_mult=args.reproject_max_corr_mult,
            max_correspondences=args.max_correspondences,
        )
        reproject_secs = time.time() - t0
        print(f"  done in {reproject_secs:.2f}s  | meta: {reproject_meta}")
        print()

    icp_meta = None
    icp_secs = 0.0
    if args.post_icp:
        print("== Stage 3: Point-to-plane ICP polish ==")
        t0 = time.time()
        final_T, icp_meta = _run_icp_polish(
            src_pts, tgt_pts, final_T, voxel_size,
            max_corr_dist_mult=args.icp_max_corr_mult,
            iterations=args.icp_iterations,
        )
        icp_secs = time.time() - t0
        print(f"  done in {icp_secs:.2f}s  | meta: {icp_meta}")
        print()

    cpd_polish_meta = None
    cpd_polish_secs = 0.0
    if args.post_cpd:
        print("== Stage 4: probreg CPD polish (slow, may take ~30-120 s) ==")
        t0 = time.time()
        final_T, _cpd_n, cpd_polish_meta = _run_probreg_similarity_refinement(
            src_pts, src_colors, tgt_pts, tgt_colors, final_T, voxel_size,
        )
        cpd_polish_secs = time.time() - t0
        print(f"  done in {cpd_polish_secs:.2f}s  | meta: {cpd_polish_meta}")
        print()

    teaser_aligned = _transform_points(src_pts, final_T)
    teaser_out = out_dir / f"{args.object_stem}_teaser{tag}_aligned.ply"
    target_copy = out_dir / f"{args.object_stem}_target_ref.ply"
    save_point_cloud(teaser_out, teaser_aligned, src_colors)
    save_point_cloud(target_copy, tgt_pts, tgt_colors)

    cpd_out = out_dir / f"{args.object_stem}_cpd_rerun_aligned.ply"
    if args.run_cpd:
        cpd_aligned = _transform_points(src_pts, cpd_T)
        save_point_cloud(cpd_out, cpd_aligned, src_colors)

    metadata = {
        "object_stem": args.object_stem,
        "source_reg_ref_ply": str(src_ref_path),
        "target_reg_ref_ply": str(tgt_ref_path),
        "original_cpd_visible_ply": str(cpd_ref_path) if cpd_ref_path.exists() else None,
        "source_points": int(len(src_pts)),
        "target_points": int(len(tgt_pts)),
        "voxel_size": float(voxel_size),
        "cpd": {
            "wall_secs": float(cpd_secs),
            "transform_4x4": cpd_T.tolist(),
            "correspondences": int(cpd_n),
            "meta": cpd_meta,
        },
        "teaser": {
            "wall_secs": float(teaser_secs),
            "transform_4x4": teaser_T.tolist(),
            "correspondences": int(teaser_n),
            "meta": teaser_meta,
            "params": {
                "noise_bound": args.noise_bound,
                "color_weight": args.color_weight,
                "max_correspondences": args.max_correspondences,
                "feature_radius_mult": args.feature_radius_mult,
                "normal_radius_mult": args.normal_radius_mult,
                "fpfh_max_nn": args.fpfh_max_nn,
                "normal_max_nn": args.normal_max_nn,
                "ratio_thresh": args.ratio_thresh,
                "multi_scale_fpfh": args.multiscale_fpfh,
                "normal_filter_deg": args.normal_filter_deg,
            },
        },
        "reproject": None if reproject_meta is None else {
            "wall_secs": float(reproject_secs), "meta": reproject_meta,
        },
        "icp": None if icp_meta is None else {
            "wall_secs": float(icp_secs), "meta": icp_meta,
        },
        "cpd_polish": None if cpd_polish_meta is None else {
            "wall_secs": float(cpd_polish_secs), "meta": cpd_polish_meta,
        },
        "final_transform_4x4": final_T.tolist(),
    }
    meta_out = out_dir / f"{args.object_stem}_metadata{tag}.json"
    meta_out.write_text(json.dumps(metadata, indent=2))

    print("== Done ==")
    print(f"  outputs: {out_dir}")
    if args.run_cpd:
        print(f"  CPD re-run aligned : {cpd_out.name}")
    print(f"  TEASER aligned     : {teaser_out.name}")
    print(f"  Target copy        : {target_copy.name}")
    print(f"  Metadata           : {meta_out.name}")
    print()
    print("To visualize:")
    print(f"  python {Path(__file__).parent}/view_registration_comparison.py {args.dataset_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
