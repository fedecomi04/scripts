"""Debug `dynamic_gs/utils/rgbd_decode.py` end-to-end on ONE frame.

No ns-train, no model load — pure disk inputs. Lets you see exactly what
`decode_component_to_gaussians` produces on the same frame Mode A targets.

Reads (from the dataset):
  - RGB:        dynamic_scene/rgb/<frame>.png
  - depth:      dynamic_scene/depth/<frame>.tiff   (uint16, scale 1e-3)
  - intrinsics: dynamic_scene/transforms.json
  - CDN mask:   dynamic_scene/change_detection_masks/<frame>_cdn_mask.png
                (pre-computed by the prior pipeline run; this script does
                NOT recompute it from scratch)

Writes (into outputs/rgbd_decode_debug/):
  - input_rgb_<frame>.png + input_cdn_<frame>.png
  - largest_component_<frame>.png      — the binary component the decoder ran on
  - decoded_pointcloud_<frame>.ply     — means + RGB (point cloud)
  - decoded_splats_<frame>.pt          — means + covariances + rgbs + opacities
                                         (load in view_splats_viser.py)
  - decode_summary_<frame>.txt         — counts, bounding boxes, opacity/scale
                                         distribution, diagnostics dict
  - sfm_reference.ply                  — first few thousand SfM init points
                                         so you can drop both into the same
                                         viewer to see if the decoded patch
                                         lands on the table surface

Run:
    ENV=~/miniconda3/envs/dynamic_gs PATH=$ENV/bin:$PATH \
    CUDA_HOME=$ENV LD_LIBRARY_PATH=$ENV/lib PYTHONNOUSERSITE=1 \
    python scripts/debug_rgbd_decode.py --frame arm_05616
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamic_gs.utils.rgbd_decode import decode_component_to_gaussians  # noqa: E402
from dynamic_gs.utils.active_mask import select_top_n_components_filtered  # noqa: E402

DATASET = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/"
    "datasets/dynamic_gs_test_2026-03-28_19-49-45_w_background"
)
OUT_DIR = ROOT / "outputs" / "rgbd_decode_debug"
DEPTH_UNIT_SCALE = 1e-3


class _MinimalCamera:
    """Adapter exposing the attrs `_camera_intrinsics` + `_camera_c2w` read.

    `decode_component_to_gaussians` only touches: fx, fy, cx, cy, width,
    height, camera_to_worlds. So a tiny stub works without booting
    nerfstudio's `Cameras`.
    """

    def __init__(self, fx, fy, cx, cy, w, h, c2w_4x4, device):
        self.fx = torch.tensor([float(fx)], device=device)
        self.fy = torch.tensor([float(fy)], device=device)
        self.cx = torch.tensor([float(cx)], device=device)
        self.cy = torch.tensor([float(cy)], device=device)
        self.width = torch.tensor([int(w)], device=device)
        self.height = torch.tensor([int(h)], device=device)
        # rgbd_decode reads c2w[:3, :3] / c2w[:3, 3], so pass a (3, 4) or (4, 4) tensor.
        self.camera_to_worlds = torch.tensor(c2w_4x4[:3], dtype=torch.float32, device=device)


def load_camera(transforms_json: Path, frame_basename: str):
    with transforms_json.open() as f:
        meta = json.load(f)
    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    w = int(meta["w"])
    h = int(meta["h"])
    for frame in meta["frames"]:
        if Path(frame["file_path"]).stem == frame_basename:
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            return c2w, fx, fy, cx, cy, w, h
    raise KeyError(f"frame {frame_basename!r} not in {transforms_json}")


def save_ply_xyz_rgb(xyz: np.ndarray, rgb01: np.ndarray, path: Path) -> None:
    n = xyz.shape[0]
    arr = np.zeros(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["red"] = (rgb01[:, 0] * 255.0).clip(0, 255).astype(np.uint8)
    arr["green"] = (rgb01[:, 1] * 255.0).clip(0, 255).astype(np.uint8)
    arr["blue"] = (rgb01[:, 2] * 255.0).clip(0, 255).astype(np.uint8)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def quats_scales_to_covariance(quats_wxyz: torch.Tensor, scales_lin: torch.Tensor) -> torch.Tensor:
    """Build (N, 3, 3) covariances = R * diag(scales^2) * R^T."""
    w, x, y, z = quats_wxyz.unbind(-1)
    # Standard wxyz → rotation matrix.
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)  # (N, 3, 3)
    S2 = (scales_lin ** 2)[:, :, None] * torch.eye(3, device=R.device, dtype=R.dtype)[None]
    cov = R @ S2 @ R.transpose(-1, -2)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="arm_05616",
                    help="dynamic frame basename (no .png)")
    ap.add_argument("--min-valid-fraction", type=float, default=0.95)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--area-ratio", type=float, default=0.3)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--opacity", type=float, default=0.99)
    ap.add_argument("--normal-smoothing-radius", type=int, default=3)
    ap.add_argument("--scale-multiplier", type=float, default=5.0,
                    help="multiply pixel-width scales (1.0 = old behavior, sub-pixel)")
    ap.add_argument("--sfm-sample", type=int, default=50000,
                    help="how many SfM init points to dump for reference")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dynamic = DATASET / "dynamic_scene"
    rgb_path = dynamic / "rgb" / f"{args.frame}.png"
    depth_path = dynamic / "depth" / f"{args.frame}.tiff"
    cdn_path = dynamic / "change_detection_masks" / f"{args.frame}_cdn_mask.png"
    transforms_json = dynamic / "transforms.json"
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    if not cdn_path.is_file():
        raise FileNotFoundError(cdn_path)

    # 1) RGB → (H, W, 3) float in [0, 1]
    rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    H, W = rgb_np.shape[:2]
    print(f"[debug] frame={args.frame} rgb shape={rgb_np.shape}")

    # 2) Depth → metres
    depth_u16 = np.asarray(Image.open(depth_path), dtype=np.uint16)
    depth_m = depth_u16.astype(np.float32) * DEPTH_UNIT_SCALE
    print(f"[debug] depth m: shape={depth_m.shape} valid_pct="
          f"{100.0 * (depth_m > 0).mean():.1f}% range=[{depth_m[depth_m > 0].min():.3f}, {depth_m.max():.3f}]")

    # 3) CDN mask
    cdn_np = np.asarray(Image.open(cdn_path).convert("L"), dtype=np.uint8)
    print(f"[debug] CDN total active px = {(cdn_np > 127).sum()} / {H*W}")
    Image.fromarray((rgb_np * 255).astype(np.uint8)).save(OUT_DIR / f"input_rgb_{args.frame}.png")
    Image.fromarray(cdn_np).save(OUT_DIR / f"input_cdn_{args.frame}.png")

    # 4) Camera
    c2w, fx, fy, cx, cy, w_, h_ = load_camera(transforms_json, args.frame)
    if (w_, h_) != (W, H):
        raise RuntimeError(f"transforms.json says (w,h)=({w_},{h_}) but RGB is ({W},{H})")
    camera = _MinimalCamera(fx, fy, cx, cy, W, H, c2w, device)
    print(f"[debug] camera: fx={fx:.2f} cx={cx:.2f} translation={c2w[:3, 3].tolist()}")

    # 5) Component selection on the CDN mask
    cdn_t = torch.from_numpy(cdn_np.astype(np.float32) / 255.0).to(device).unsqueeze(-1)  # (H,W,1)
    comps = select_top_n_components_filtered(
        cdn_t, n=args.top_n, area_ratio=args.area_ratio, min_area=args.min_area
    )
    print(f"[debug] components kept (Mode A policy n={args.top_n} ratio={args.area_ratio} min={args.min_area}): {len(comps)}")
    if not comps:
        print("[debug] no components survived the filter — Mode A wouldn't have inserted anything")
        return 1

    # 6) Decode each component; save the LARGEST one for the splats dump
    rgb_t = torch.from_numpy(rgb_np).to(device).float()
    depth_t = torch.from_numpy(depth_m).to(device).float()
    summary_lines = [
        f"frame: {args.frame}",
        f"image shape: {H}x{W}",
        f"camera fx,fy,cx,cy: {fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}",
        f"c2w translation: {c2w[:3, 3].tolist()}",
        f"depth scale factor: {DEPTH_UNIT_SCALE}",
        f"CDN active px (full): {int((cdn_np > 127).sum())}",
        f"components after Mode A filter (n={args.top_n}, ratio={args.area_ratio}, min={args.min_area}): {len(comps)}",
    ]
    largest_decoded = None
    for k, comp in enumerate(comps):
        comp_bool = (comp[..., 0] > 0.5) if comp.ndim == 3 else (comp > 0.5)
        area = int(comp_bool.sum().item())
        print(f"\n[debug] === component {k}: area={area} px ===")
        Image.fromarray((comp_bool.cpu().numpy().astype(np.uint8) * 255)).save(
            OUT_DIR / f"component_{k:02d}_{args.frame}.png"
        )
        decoded = decode_component_to_gaussians(
            camera, rgb_t, depth_t, comp,
            opacity=args.opacity,
            normal_smoothing_radius=args.normal_smoothing_radius,
            min_valid_fraction=args.min_valid_fraction,
            scale_multiplier=args.scale_multiplier,
        )
        if decoded is None:
            print(f"[debug] component {k}: returned None (empty)")
            summary_lines.append(f"  comp {k}: area={area} → empty")
            continue
        if decoded.get("skipped"):
            print(f"[debug] component {k}: SKIPPED, diag={decoded['diagnostics']}")
            summary_lines.append(f"  comp {k}: area={area} SKIPPED valid_frac={decoded['diagnostics']['valid_fraction']:.3f}")
            continue

        xyz = decoded["xyz"].detach().cpu().numpy()
        feats_dc = decoded["features_dc"].detach().cpu().numpy()
        opac_raw = decoded["opacities"].detach().cpu().numpy().squeeze(-1)
        scales_log = decoded["scales"].detach().cpu().numpy()
        scales_lin = np.exp(scales_log)
        # SH DC → RGB ([sigmoid is sigmoid(C0*x + 0.5) etc, but easier to invert RGB2SH]).
        # rgbd_decode used `RGB2SH(rgb)` which is `(rgb - 0.5) / C0`. So RGB = C0*sh + 0.5.
        C0 = 0.28209479177387814
        rgb01 = np.clip(C0 * feats_dc + 0.5, 0.0, 1.0)

        msg = (
            f"  comp {k}: area={area} → decoded={xyz.shape[0]} "
            f"valid_frac={decoded['diagnostics']['valid_fraction']:.3f} "
            f"depth_m=[{decoded['diagnostics']['depth_min_m']:.3f}, "
            f"{decoded['diagnostics']['depth_max_m']:.3f}] "
            f"xyz_bbox=x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}] "
            f"y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}] "
            f"z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}] "
            f"scales_lin=[{scales_lin.min():.5f}, {scales_lin.max():.5f}] mean={scales_lin.mean():.5f} "
            f"opacity_raw_logit=[{opac_raw.min():.2f}, {opac_raw.max():.2f}] "
            f"sigmoid={1/(1+np.exp(-opac_raw.mean())):.3f}"
        )
        print(msg)
        summary_lines.append(msg)

        if largest_decoded is None or xyz.shape[0] > largest_decoded["xyz"].shape[0]:
            largest_decoded = {
                "xyz": xyz, "rgb01": rgb01, "scales_lin": scales_lin,
                "opac_raw": opac_raw, "quats": decoded["quats"].detach().cpu().numpy(),
                "comp_idx": k,
            }

    if largest_decoded is None:
        print("[debug] no component produced Gaussians (all skipped or empty)")
        return 1

    # 7) Save largest component as PLY (point cloud) and .pt (splats)
    xyz = largest_decoded["xyz"]
    rgb01 = largest_decoded["rgb01"]
    scales_lin = largest_decoded["scales_lin"]
    opac_raw = largest_decoded["opac_raw"]
    quats = largest_decoded["quats"]

    save_ply_xyz_rgb(xyz, rgb01, OUT_DIR / f"decoded_pointcloud_{args.frame}.ply")

    quats_t = torch.from_numpy(quats).float()
    scales_t = torch.from_numpy(scales_lin).float()
    covs = quats_scales_to_covariance(quats_t, scales_t).numpy()
    opacities_01 = 1.0 / (1.0 + np.exp(-opac_raw))
    torch.save(
        {
            "means": xyz.astype(np.float32),
            "covariances": covs.astype(np.float32),
            "rgbs": rgb01.astype(np.float32),
            "opacities": opacities_01.astype(np.float32),
            "anchor_frame": args.frame,
            "selected_frames": [args.frame],
        },
        OUT_DIR / f"decoded_splats_{args.frame}.pt",
    )

    # 8) Also dump SfM init points as reference (so you can drop both into viser)
    sfm_path = DATASET / "static_scene" / "transforms.json"
    try:
        with sfm_path.open() as f:
            sfm_meta = json.load(f)
        ply_rel = sfm_meta.get("ply_file_path")
        if ply_rel:
            sfm_ply = (DATASET / "static_scene" / ply_rel).resolve()
            if sfm_ply.is_file():
                sfm_pd = PlyData.read(str(sfm_ply))["vertex"]
                pts = np.stack([sfm_pd["x"], sfm_pd["y"], sfm_pd["z"]], axis=-1).astype(np.float32)
                if "red" in sfm_pd.dtype.names:
                    sfm_rgb = np.stack([sfm_pd["red"], sfm_pd["green"], sfm_pd["blue"]], axis=-1).astype(np.float32) / 255.0
                else:
                    sfm_rgb = np.full_like(pts, 0.5)
                n = pts.shape[0]
                if n > args.sfm_sample:
                    idx = np.random.default_rng(0).choice(n, args.sfm_sample, replace=False)
                    pts, sfm_rgb = pts[idx], sfm_rgb[idx]
                save_ply_xyz_rgb(pts, sfm_rgb, OUT_DIR / "sfm_reference.ply")
                summary_lines.append(
                    f"sfm reference: {pts.shape[0]} pts (sampled from {n}); "
                    f"bbox x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
                    f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
                    f"z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]"
                )
    except Exception as exc:
        summary_lines.append(f"sfm reference dump failed: {exc}")

    summary_lines.append(f"largest-component dump: comp_idx={largest_decoded['comp_idx']}, N={xyz.shape[0]}")
    (OUT_DIR / f"decode_summary_{args.frame}.txt").write_text("\n".join(summary_lines) + "\n")
    print()
    print("[debug] outputs in:", OUT_DIR)
    for p in sorted(OUT_DIR.iterdir()):
        if args.frame in p.name or "sfm" in p.name or "summary" in p.name:
            print(" ", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
