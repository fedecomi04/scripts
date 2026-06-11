"""Run AnySplat on a single image and view the RAW canonical gaussians as
splats (NOT a point cloud) in Nerfstudio's viewer.

Pipeline:
    1. Spawn the AnySplat worker (conda env ``anysplat_dynamic_gs``) on the one
       image → ``.npz`` with means/log_scales/quats/SH/opacity in AnySplat's
       canonical frame, plus the predicted camera (extrinsic + intrinsics).
    2. Stuff those gaussians straight into a vanilla Splatfacto model's
       ``gauss_params`` (no Umeyama, no scene reprojection — exactly what the
       model produced).
    3. Build a one-frame Nerfstudio dataset using AnySplat's predicted camera
       (OpenCV→OpenGL converted) so the viewer opens looking at the object,
       render one verification frame to disk, then launch the NS viewer.

The viewer rasterizes via gsplat (server-side), so you see real gaussian
splats, not points.

Run from the ``dynamic_gs`` env (has nerfstudio + gsplat). Example:

    /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python \\
        /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/scripts/view_anysplat_nerfstudio.py \\
        --image /home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/new_env/static_scene/rgb/arm_00023.png

Then open http://localhost:7007 in a browser.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]          # .../scripts
_WORKER_SCRIPT = _SCRIPTS_DIR / "anysplat_worker.py"
_ANYSPLAT_REPO = _SCRIPTS_DIR / "third_party" / "AnySplat"


# --------------------------------------------------------------------------- #
# 1. AnySplat inference (subprocess in the sibling conda env)
# --------------------------------------------------------------------------- #
def run_anysplat(image: Path, output_npz: Path, conda_env: str = "anysplat_dynamic_gs") -> Path:
    env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
    env_python = env_prefix / "bin" / "python"
    if not env_python.exists():
        raise FileNotFoundError(f"AnySplat env python not found: {env_python}")
    if not _WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"AnySplat worker not found: {_WORKER_SCRIPT}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(env_python), "-u", str(_WORKER_SCRIPT),
        "--image", str(image),
        "--output", str(output_npz),
        "--anysplat-repo", str(_ANYSPLAT_REPO),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[view-anysplat] running AnySplat on {image.name} (env {conda_env})...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if res.returncode != 0 or not output_npz.exists():
        raise RuntimeError(f"AnySplat worker failed (exit {res.returncode}):\n{res.stdout[-3000:]}")
    print(f"[view-anysplat]   done in {time.time()-t0:.1f}s → {output_npz}", flush=True)
    return output_npz


def load_npz(npz: Path) -> dict:
    # The worker writes via pickle.dump (flat dict of ndarrays), not np.savez.
    with open(npz, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# 2. One-frame Nerfstudio dataset from AnySplat's predicted camera
# --------------------------------------------------------------------------- #
def write_one_frame_dataset(
    data_dir: Path, image: Path, c2w_opengl: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int
) -> None:
    """Write transforms.json (+ symlink the image) for a single OpenGL camera."""
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    link = img_dir / image.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(image.resolve())

    meta = {
        "camera_model": "OPENCV",
        "fl_x": float(fx), "fl_y": float(fy), "cx": float(cx), "cy": float(cy),
        "w": int(w), "h": int(h),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [{
            "file_path": f"images/{image.name}",
            "transform_matrix": c2w_opengl.tolist(),
        }],
    }
    (data_dir / "transforms.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# 3. Build the Splatfacto pipeline + inject the AnySplat gaussians
# --------------------------------------------------------------------------- #
def build_pipeline(data_dir: Path, out_dir: Path, background: str, port: int, device: str):
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.configs.base_config import ViewerConfig

    config = copy.deepcopy(method_configs["splatfacto"])
    config.output_dir = out_dir
    config.experiment_name = "anysplat_view"
    config.method_name = "splatfacto"
    config.timestamp = "view"
    config.data = data_dir
    config.vis = "viewer"
    config.viewer = ViewerConfig(websocket_port=port, make_share_url=False)

    dp = config.pipeline.datamanager.dataparser
    dp.data = data_dir
    dp.orientation_method = "none"
    dp.center_method = "none"
    dp.auto_scale_poses = False
    dp.load_3D_points = False
    dp.eval_mode = "all"

    config.pipeline.datamanager.data = data_dir
    config.pipeline.datamanager.cache_images = "gpu"

    m = config.pipeline.model
    m.random_init = True          # avoid needing seed points; we overwrite gauss_params below
    m.num_random = 100
    m.background_color = background

    base_dir = config.get_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    pipeline = config.pipeline.setup(device=device, test_mode="test")
    pipeline.eval()
    return config, pipeline


def inject_gaussians(model, d: dict, opacity_min: float, device: str) -> int:
    import torch.nn as nn

    means = torch.tensor(np.ascontiguousarray(d["means_canonical"]), dtype=torch.float32, device=device)
    scales = torch.tensor(np.ascontiguousarray(d["log_scales"]), dtype=torch.float32, device=device)       # log-scale
    quats = torch.tensor(np.ascontiguousarray(d["quats_wxyz"]), dtype=torch.float32, device=device)
    opac = torch.tensor(np.ascontiguousarray(d["opacity_logits"]), dtype=torch.float32, device=device).reshape(-1, 1)
    fdc = torch.tensor(np.ascontiguousarray(d["features_dc"]), dtype=torch.float32, device=device)          # (N, 3)
    frest = torch.tensor(np.ascontiguousarray(d["features_rest"]), dtype=torch.float32, device=device)      # (N, 15, 3)

    if opacity_min > 0.0:
        keep = torch.sigmoid(opac.squeeze(1)) >= opacity_min
        means, scales, quats = means[keep], scales[keep], quats[keep]
        opac, fdc, frest = opac[keep], fdc[keep], frest[keep]
        print(f"[view-anysplat] opacity filter (>= {opacity_min}): kept {int(keep.sum())}/{keep.numel()}")

    model.gauss_params = nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=False),
        "scales": nn.Parameter(scales, requires_grad=False),
        "quats": nn.Parameter(quats, requires_grad=False),
        "features_dc": nn.Parameter(fdc, requires_grad=False),
        "features_rest": nn.Parameter(frest, requires_grad=False),
        "opacities": nn.Parameter(opac, requires_grad=False),
    }).to(device)
    model.step = 30000            # activate all SH bands at render time
    model.crop_box = None
    return means.shape[0]


def render_check(pipeline, out_png: Path, device: str) -> None:
    from PIL import Image

    cam = pipeline.datamanager.train_dataset.cameras[0:1].to(device)
    with torch.no_grad():
        out = pipeline.model.get_outputs_for_camera(cam)
    rgb = out["rgb"].detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out_png)
    print(f"[view-anysplat] verification render → {out_png}  (rgb range {rgb.min():.3f}..{rgb.max():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path,
                    default=Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/new_env/static_scene/rgb/arm_00023.png"))
    ap.add_argument("--npz", type=Path, default=None, help="Reuse an existing AnySplat .npz instead of re-running")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/anysplat_view"))
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--opacity-min", type=float, default=0.0, help="Drop splats with sigmoid(opacity) < this")
    ap.add_argument("--background", choices=["white", "black", "random"], default="white")
    ap.add_argument("--no-viewer", action="store_true", help="Render the verification frame and exit (skip launching the viewer)")
    args = ap.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- AnySplat ---
    npz = args.npz if args.npz else run_anysplat(args.image, args.out_dir / "anysplat.npz")
    d = load_npz(npz)
    print(f"[view-anysplat] loaded {d['means_canonical'].shape[0]} gaussians from {npz}")

    # --- predicted camera: AnySplat is OpenCV (y down, z fwd); NS is OpenGL → flip y,z ---
    c2w_cv = np.asarray(d["pred_extrinsic_c2w"][0], dtype=np.float64)
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
    from PIL import Image
    W, H = Image.open(args.image).size
    K = np.asarray(d["pred_intrinsic_norm"][0], dtype=np.float64)   # normalized
    fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H

    data_dir = args.out_dir / "ns_data"
    write_one_frame_dataset(data_dir, args.image, c2w_gl, fx, fy, cx, cy, W, H)

    # --- pipeline + inject ---
    config, pipeline = build_pipeline(data_dir, args.out_dir / "ns_out", args.background, args.port, device)
    n = inject_gaussians(pipeline.model, d, args.opacity_min, device)
    print(f"[view-anysplat] injected {n} gaussians into Splatfacto model")

    render_check(pipeline, args.out_dir / "verification_render.png", device)

    if args.no_viewer:
        print("[view-anysplat] --no-viewer set; exiting after verification render.", flush=True)
        return

    # --- launch the Nerfstudio viewer ---
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nerfstudio"))  # no-op if pip-installed
    from nerfstudio.scripts.viewer.run_viewer import _start_viewer
    print(f"\n[view-anysplat] launching Nerfstudio viewer on http://localhost:{args.port}  (Ctrl-C to quit)\n", flush=True)
    _start_viewer(config, pipeline, step=30000)


if __name__ == "__main__":
    main()
