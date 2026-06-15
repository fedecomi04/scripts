"""Visualize, in Nerfstudio's viewer, the object reconstruction for the last
static frame of a dataset:

  (A) GROUND-TRUTH object point cloud  — the object mask back-projected through
      the real sensor depth into the scene world frame (reuses the repo's
      ``backproject_mask_to_world``).
  (B) WORLD-TRACING object             — the WT object model's predicted cloud
      (``world_tracing_worker.py`` output), placed into the scene frame and
      ICP-refined onto (A). [only with ``--wt-pkl``]
  (+) scene backdrop                   — the trained static scene from
      ``static_state.pt`` for spatial context. [default on; ``--no-backdrop``]

Everything is rendered as gaussian splats (points enter as small isotropic
gaussians), not a raw point cloud, via the same Splatfacto+viewer scaffold as
``view_anysplat_nerfstudio.py``.

Zero-arg run shows GT (real colors) + backdrop for the screwdriver scene. Add
``--wt-pkl <out.pkl>`` once the WT weights are available to overlay (B).

Run from the ``dynamic_gs`` env (with the env activated so gsplat's JIT build
works). Open http://localhost:7007 after launch.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_SCRIPTS = Path(__file__).resolve().parent          # .../scripts/scripts
_REPO = _SCRIPTS.parent                              # .../scripts
sys.path.insert(0, str(_SCRIPTS))                    # for view_anysplat_nerfstudio
sys.path.insert(0, str(_REPO))                       # for dynamic_gs

from view_anysplat_nerfstudio import build_pipeline, render_check, write_one_frame_dataset  # noqa: E402

_DEFAULT_DS = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver recorded full")
_SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# scene camera + ground-truth cloud
# --------------------------------------------------------------------------- #
def build_scene_camera(transforms_path: Path, frame_stem: str, device: str):
    from nerfstudio.cameras.cameras import Cameras, CameraType

    tj = json.loads(transforms_path.read_text())
    fx, fy, cx, cy = float(tj["fl_x"]), float(tj["fl_y"]), float(tj["cx"]), float(tj["cy"])
    W, H = int(tj["w"]), int(tj["h"])
    frame = next(f for f in tj["frames"] if frame_stem in f["file_path"])
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)  # 4x4 OpenGL
    cam = Cameras(
        camera_to_worlds=torch.tensor(c2w[:3, :4], dtype=torch.float32)[None],
        fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    return cam, c2w, (fx, fy, cx, cy, W, H)


def ground_truth_cloud(ds: Path, cam, frame_stem: str, device: str):
    from dynamic_gs.fusion.phase0 import backproject_mask_to_world

    art = ds / "dynamic_scene" / "initialization_artifacts"
    dbg = ds / "dynamic_scene" / "initialization_debug"
    depth_m = cv2.imread(str(art / "static0_full_depth_meters.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    rgb = cv2.cvtColor(cv2.imread(str(ds / "static_scene" / "rgb" / f"{frame_stem}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(dbg / "static0_obj_00_mask.png"), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask > 127
    pts, cols = backproject_mask_to_world(mask_bool, torch.from_numpy(depth_m), torch.from_numpy(rgb), cam.to("cpu"))
    cam.to(device)
    return pts.astype(np.float32), cols.astype(np.float32)


# --------------------------------------------------------------------------- #
# World-Tracing cloud + placement (the NDP-gap proxy: ICP onto GT)
# --------------------------------------------------------------------------- #
def world_tracing_cloud(wt_pkl: Path):
    d = pickle.load(open(wt_pkl, "rb"))
    xyz, mask, rgb = d["xyz"], d["mask"], d["rgb"]           # (L,S,S,3),(L,S,S),(S,S,3)
    pts, cols = [], []
    for L in range(xyz.shape[0]):
        m = mask[L]
        if not m.any():
            continue
        pts.append(xyz[L][m])
        cols.append(rgb[m])  # back-layer color is the front pixel's — approximate, viz only
    return np.concatenate(pts).astype(np.float32), np.concatenate(cols).astype(np.float32)


def _rigid_init_to_gt(pts_cam_rdf: np.ndarray, c2w: np.ndarray, gt_pts: np.ndarray):
    """RDF camera-space -> OpenGL scene world, then bbox-scale + centroid onto GT.

    Mirrors the rigid init the SAM3D pipeline bakes into ``scaled_source`` before
    NDP. Returns (pts_init (N,3) float64, scale).
    """
    pts_gl = pts_cam_rdf * np.array([1.0, -1.0, -1.0], np.float32)  # OpenCV/RDF -> OpenGL cam axes
    R, t = c2w[:3, :3], c2w[:3, 3]
    pts0 = (pts_gl @ R.T + t).astype(np.float64)
    gt = gt_pts.astype(np.float64)
    diag = lambda p: float(np.linalg.norm(p.max(0) - p.min(0)) + 1e-9)
    s = diag(gt) / diag(pts0)
    return ((pts0 - pts0.mean(0)) * s + gt.mean(0)), s


def place_wt_into_scene(pts_cam_rdf: np.ndarray, c2w: np.ndarray, gt_pts: np.ndarray, method: str):
    """Place the WT object cloud into the scene world frame.

    method:
      'rigid' — bbox-scale + centroid onto GT only (no refinement).
      'icp'   — rigid init + point-to-point ICP onto GT.
      'ndp'   — rigid init + the EXACT pipeline NDP non-rigid registration
                (``deform_source_to_target`` with the default ``_NDP_CONFIG``:
                Sim3, m=9, iters=500, lr=0.01, samples=6000, w_reg=1.0) — i.e.
                the same registration SAM3D output goes through in Phase-0b.
    Returns (pts_world (N,3) float32, info dict).
    """
    pts_init, s = _rigid_init_to_gt(pts_cam_rdf, c2w, gt_pts)
    info = {"method": method, "init_scale": s}
    if method == "rigid" or gt_pts.shape[0] < 10:
        return pts_init.astype(np.float32), info

    if method == "ndp":
        from dynamic_gs.utils.ndp_register import deform_source_to_target, _NDP_CONFIG
        warped, meta = deform_source_to_target(pts_init.astype(np.float64), gt_pts.astype(np.float64))
        info["ndp_config"] = {k: _NDP_CONFIG[k] for k in ("motion_type", "m", "iters", "lr", "samples", "w_reg")}
        info.update({k: v for k, v in meta.items() if np.isscalar(v) or isinstance(v, (int, float, str))})
        return warped.astype(np.float32), info

    import open3d as o3d  # method == "icp"
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_init))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts.astype(np.float64)))
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    pts_ref = np.asarray(src.transform(reg.transformation).points).astype(np.float32)
    info.update({"fitness": float(reg.fitness), "inlier_rmse": float(reg.inlier_rmse)})
    return pts_ref, info


# --------------------------------------------------------------------------- #
# points/gaussians assembly
# --------------------------------------------------------------------------- #
def points_to_gauss(pts: np.ndarray, cols: np.ndarray, radius_m: float, opacity: float, tint):
    N = pts.shape[0]
    c = np.tile(np.asarray(tint, np.float32), (N, 1)) if tint is not None else cols.astype(np.float32)
    return {
        "means": pts.astype(np.float32),
        "scales": np.full((N, 3), np.log(max(radius_m, 1e-6)), np.float32),
        "quats": np.tile(np.array([1, 0, 0, 0], np.float32), (N, 1)),
        "features_dc": ((c - 0.5) / _SH_C0).astype(np.float32),
        "features_rest": np.zeros((N, 15, 3), np.float32),
        "opacities": np.full((N, 1), np.log(opacity / (1 - opacity)), np.float32),
    }


def load_backdrop(static_state: Path, hide_object_id: int | None):
    sd = torch.load(static_state, map_location="cpu")["model_state_dict"]
    g = {k.split("gauss_params.")[1]: sd[k].numpy() for k in sd if k.startswith("gauss_params.")}
    if hide_object_id is not None and "object_instance_ids" not in g:
        ids = sd.get("object_instance_ids")
        if ids is not None:
            keep = (ids.numpy().reshape(-1) != hide_object_id)
            g = {k: v[keep] for k, v in g.items()}
    return g


def concat(*dicts):
    dicts = [d for d in dicts if d is not None and d["means"].shape[0] > 0]
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0]}


def inject(model, g: dict, device: str):
    model.gauss_params = nn.ParameterDict({
        k: nn.Parameter(torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=device), requires_grad=False)
        for k, v in g.items()
    }).to(device)
    model.step = 30000
    model.crop_box = None
    return g["means"].shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=_DEFAULT_DS)
    ap.add_argument("--frame", default="arm_00026", help="last static frame stem")
    ap.add_argument("--wt-pkl", type=Path, default=None, help="World-Tracing worker output to overlay")
    ap.add_argument("--no-backdrop", action="store_true", help="hide the static-scene context")
    ap.add_argument("--hide-existing-object", type=int, default=None, metavar="ID",
                    help="drop this object_instance_id from the backdrop (e.g. 1 = the SAM3D screwdriver)")
    ap.add_argument("--no-icp", action="store_true", help="(deprecated) alias for --placement rigid")
    ap.add_argument("--placement", choices=["icp", "ndp", "rigid"], default="icp",
                    help="WT->scene registration: icp (default), ndp (the EXACT Phase-0b NDP "
                         "non-rigid registration SAM3D goes through), or rigid (init only)")
    ap.add_argument("--gt-radius", type=float, default=0.0012)
    ap.add_argument("--wt-radius", type=float, default=0.0012)
    ap.add_argument("--gt-tint", default=None, help="R,G,B in [0,1] to recolor GT (default: real colors)")
    ap.add_argument("--wt-tint", default="1,0,1", help="R,G,B in [0,1] for WT (default magenta; 'real' = keep colors)")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/object_recon_view"))
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parse_tint = lambda s: None if s in (None, "real") else np.array([float(x) for x in s.split(",")], np.float32)

    cam, c2w, (fx, fy, cx, cy, W, H) = build_scene_camera(args.dataset / "static_scene" / "transforms.json", args.frame, device)

    gt_pts, gt_cols = ground_truth_cloud(args.dataset, cam, args.frame, device)
    print(f"[obj-view] GT cloud: {gt_pts.shape[0]} pts")
    gt_g = points_to_gauss(gt_pts, gt_cols, args.gt_radius, 0.99, parse_tint(args.gt_tint))

    wt_g = None
    if args.wt_pkl:
        wt_cam, wt_cols = world_tracing_cloud(args.wt_pkl)
        method = "rigid" if args.no_icp else args.placement
        wt_pts, info = place_wt_into_scene(wt_cam, c2w, gt_pts, method)
        print(f"[obj-view] WT cloud: {wt_pts.shape[0]} pts; placement={info}")
        wt_g = points_to_gauss(wt_pts, wt_cols, args.wt_radius, 0.99, parse_tint(args.wt_tint))

    backdrop = None if args.no_backdrop else load_backdrop(args.dataset / "static_scene" / "static_state.pt", args.hide_existing_object)
    if backdrop is not None:
        print(f"[obj-view] backdrop: {backdrop['means'].shape[0]} gaussians")

    combined = concat(backdrop, gt_g, wt_g)

    data_dir = args.out_dir / "ns_data"
    write_one_frame_dataset(data_dir, args.dataset / "static_scene" / "rgb" / f"{args.frame}.png", c2w, fx, fy, cx, cy, W, H)
    config, pipeline = build_pipeline(data_dir, args.out_dir / "ns_out", "white", args.port, device)
    n = inject(pipeline.model, combined, device)
    print(f"[obj-view] injected {n} gaussians (backdrop + GT{' + WT' if wt_g else ''})")

    render_check(pipeline, args.out_dir / "verification_render.png", device)
    if args.no_viewer:
        print("[obj-view] --no-viewer; exiting after verification render.")
        return

    from nerfstudio.scripts.viewer.run_viewer import _start_viewer
    print(f"\n[obj-view] launching Nerfstudio viewer on http://localhost:{args.port}  (Ctrl-C to quit)\n", flush=True)
    _start_viewer(config, pipeline, step=30000)


if __name__ == "__main__":
    main()
