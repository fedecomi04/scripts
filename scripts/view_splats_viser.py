"""Open a viser web viewer for a saved Gaussian-splat .pt file produced
by mvsplat_mvp.py / depthsplat_mvp.py.

Usage:
    python scripts/view_splats_viser.py outputs/depthsplat_mvp/depthsplat_gaussians.pt
    python scripts/view_splats_viser.py outputs/mvsplat_mvp/mvsplat_gaussians.pt

The script keeps a viser server running at http://localhost:8080 until
Ctrl-C. Optional flags:
    --port 8081
    --opacity-min 0.05   drop nearly-transparent splats before sending
    --max-points 0       0 = send all; otherwise random-subsample to N
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import viser


def load_blob(pt_path: Path) -> dict:
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    for k in ("means", "covariances", "rgbs", "opacities"):
        if k not in blob:
            raise KeyError(f"{pt_path}: missing '{k}'")
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", type=Path, help="path to the saved .pt blob")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--opacity-min", type=float, default=0.0,
                    help="drop splats with opacity < this (0 = keep all)")
    ap.add_argument("--max-points", type=int, default=0,
                    help="random-subsample to N points after opacity filter (0 = send all)")
    ap.add_argument("--highlight-inserted", action="store_true",
                    help="color all inserted-flag=1 splats bright magenta (requires .pt with inserted_flags)")
    ap.add_argument("--only-inserted", action="store_true",
                    help="drop every splat where inserted_flag != 1 (requires inserted_flags)")
    ap.add_argument("--force-opacity", type=float, default=0.0,
                    help="If > 0, override every splat opacity to this value (debug: rules out low-opacity invisibility).")
    ap.add_argument("--diag-marker", action="store_true",
                    help="Draw a large red sphere at the scene centroid (debug: verify camera placement vs splat visibility).")
    args = ap.parse_args()

    blob = load_blob(args.pt)
    means = np.asarray(blob["means"], dtype=np.float32)
    covs = np.asarray(blob["covariances"], dtype=np.float32)
    rgbs = np.asarray(blob["rgbs"], dtype=np.float32)
    # opacities may be saved as (N,) or (N, 1); normalize to (N,) for masking.
    opacities = np.asarray(blob["opacities"], dtype=np.float32).reshape(-1)
    anchor = blob.get("anchor_frame", "?")
    selected = blob.get("selected_frames", [])
    n_total = means.shape[0]
    print(f"[viser] loaded {n_total} splats from {args.pt}")
    print(f"[viser] anchor frame = {anchor}, selected = {selected}")
    print(f"[viser] opacity range = [{opacities.min():.3f}, {opacities.max():.3f}], mean = {opacities.mean():.3f}")

    inserted_flags = blob.get("inserted_flags")
    inserted_flags = np.asarray(inserted_flags) if inserted_flags is not None else None
    if inserted_flags is not None and inserted_flags.shape[0] != means.shape[0]:
        print(f"[viser] WARN inserted_flags length mismatch ({inserted_flags.shape[0]} vs {means.shape[0]}) — ignoring")
        inserted_flags = None
    # `object_instance_ids == 999` = Mode A / Mode B feedforward inserts only.
    # `inserted_flags` includes all Phase 0b SAM3D object inserts too.
    instance_ids = blob.get("object_instance_ids")
    instance_ids = np.asarray(instance_ids) if instance_ids is not None else None
    if instance_ids is not None and instance_ids.shape[0] != means.shape[0]:
        instance_ids = None
    if instance_ids is not None:
        # Override inserted_flags with the feedforward-only mask if available.
        ff_mask = (instance_ids == 999).astype(np.uint8)
        print(f"[viser] feedforward-only mask (instance_id==999): {int(ff_mask.sum())} splats "
              f"(vs {int((inserted_flags > 0).sum()) if inserted_flags is not None else 'n/a'} inserted_flag=1)")
        inserted_flags = ff_mask

    keep = opacities >= args.opacity_min
    means, covs, rgbs, opacities = means[keep], covs[keep], rgbs[keep], opacities[keep]
    if inserted_flags is not None:
        inserted_flags = inserted_flags[keep]
    print(f"[viser] after opacity≥{args.opacity_min}: {means.shape[0]} splats")
    if inserted_flags is not None:
        print(f"[viser] inserted_flag=1 count: {int((inserted_flags > 0).sum())}")

    if args.only_inserted:
        if inserted_flags is None:
            print("[viser] --only-inserted requested but no inserted_flags in blob; ignoring")
        else:
            keep = inserted_flags > 0
            means, covs, rgbs, opacities = means[keep], covs[keep], rgbs[keep], opacities[keep]
            inserted_flags = inserted_flags[keep]
            print(f"[viser] after --only-inserted: {means.shape[0]} splats")

    if args.highlight_inserted and inserted_flags is not None:
        mask = inserted_flags > 0
        rgbs = rgbs.copy()
        rgbs[mask] = np.array([1.0, 0.0, 1.0], dtype=rgbs.dtype)  # magenta
        print(f"[viser] highlighted {int(mask.sum())} inserted splats in magenta")

    if args.max_points > 0 and means.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(means.shape[0], args.max_points, replace=False)
        means, covs, rgbs, opacities = means[idx], covs[idx], rgbs[idx], opacities[idx]
        print(f"[viser] subsampled to {means.shape[0]} splats")

    # viser does a Cholesky decomposition on each covariance — drop any that
    # are not positive-definite (degenerate splats with a zero-eigenvalue axis,
    # common in DepthSplat's low-opacity tail). Also regularize a tiny epsilon
    # on the diagonal to avoid numerical edge cases.
    eig_min = np.linalg.eigvalsh(covs.astype(np.float64)).min(axis=-1)
    pd_keep = eig_min > 1e-9
    if (~pd_keep).any():
        print(f"[viser] dropping {int((~pd_keep).sum())} non-PD covariances")
        means, covs, rgbs, opacities = means[pd_keep], covs[pd_keep], rgbs[pd_keep], opacities[pd_keep]
    covs = covs + np.eye(3, dtype=covs.dtype)[None] * 1e-7
    if args.force_opacity > 0:
        opacities = np.full_like(opacities, args.force_opacity)
        print(f"[viser] forced every opacity → {args.force_opacity}")
    print(f"[viser] sending {means.shape[0]} splats to viewer")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.world_axes.visible = True

    # CHUNK uploads at ≤80k splats per message. Each viser splat handle is one
    # WebSocket message; at 64 B/splat (centers 12 + cov 36 + rgbs 12 + opac 4),
    # 80k ≈ 5 MB, well under Chrome's 16 MB frame ceiling. A single 389k message
    # silently fails → white screen. Pattern stolen from utils/viser_direct.py.
    MAX_PER_HANDLE = 80_000
    opac_2d = opacities.reshape(-1, 1)
    n_kept = means.shape[0]
    if n_kept <= MAX_PER_HANDLE:
        print(f"[viser] uploading {n_kept} splats (single message)", flush=True)
        server.scene._add_gaussian_splats(
            name="/splats", centers=means, covariances=covs,
            rgbs=rgbs, opacities=opac_2d,
        )
    else:
        n_chunks = (n_kept + MAX_PER_HANDLE - 1) // MAX_PER_HANDLE
        print(f"[viser] uploading {n_kept} splats in {n_chunks} chunks of ≤{MAX_PER_HANDLE}", flush=True)
        for i in range(n_chunks):
            start = i * MAX_PER_HANDLE
            end = min(start + MAX_PER_HANDLE, n_kept)
            server.scene._add_gaussian_splats(
                name=f"/splats/chunk_{i:02d}",
                centers=means[start:end], covariances=covs[start:end],
                rgbs=rgbs[start:end], opacities=opac_2d[start:end],
            )

    # Compute centroid + bbox now (before optional diag marker uses them).
    centroid_for_marker = means.mean(axis=0).astype(np.float32)
    bbox_diag_for_marker = float(np.linalg.norm(means.max(axis=0) - means.min(axis=0)))
    if args.diag_marker:
        radius = max(bbox_diag_for_marker * 0.05, 0.03)
        try:
            server.scene.add_icosphere(
                name="/diag/centroid",
                position=tuple(centroid_for_marker.tolist()),
                radius=radius,
                color=(255, 0, 0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            print(f"[viser] diag marker (red sphere r={radius:.3f} m) at centroid {tuple(centroid_for_marker.round(3))}")
        except Exception as exc:
            print(f"[viser] could not add diag marker: {exc}")

    # Aim each new client's camera at the scene centroid so the user isn't
    # staring into empty space when the scene is off-origin. Viser's auto-fit
    # often zooms to nothing on off-origin scenes (white screen). Scene is
    # Nerfstudio/Z-up: z is world-up, table sits on the xy plane.
    centroid = means.mean(axis=0).astype(np.float32)
    bbox_diag = float(np.linalg.norm(means.max(axis=0) - means.min(axis=0)))
    cam_dist = max(bbox_diag * 0.9, 0.5)
    # Side-front view: offset along +x, slightly above (up = +z).
    cam_pos = (centroid + np.array([cam_dist, 0.0, cam_dist * 0.4], dtype=np.float32))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    print(f"[viser] scene centroid {tuple(centroid.round(3))}, bbox_diag {bbox_diag:.2f} m, "
          f"placing camera at {tuple(cam_pos.round(3))} looking at centroid (up=+z)",
          flush=True)

    @server.on_client_connect
    def _aim_camera(client) -> None:
        try:
            client.camera.position = cam_pos
            client.camera.look_at = centroid
            client.camera.up_direction = up
            print(f"[viser] client connected — camera set", flush=True)
        except Exception as exc:
            print(f"[viser] could not set camera for client: {exc}", flush=True)

    print(f"[viser] serving at http://localhost:{args.port}/  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[viser] shutting down")


if __name__ == "__main__":
    sys.exit(main() or 0)
