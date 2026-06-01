"""Per-point curvature + color-gradient + depth analysis for a fused PLY.

Two modes:

  --mode rank   (default — Strategy original)
    Compute the percentile-rank score
        score = depth_weight * (0.5 * rank(curv) + 0.5 * rank(cgrad))
    Threshold = top-N% percentile of `score`. Relative to scene context.

  --mode absolute
    Use raw curvature + color-gradient thresholds:
        high_detail = (curvature > --curv-thresh) OR (color_grad > --cgrad-thresh)
    Absolute meaning, robust to scene scale. The depth weight is still
    saved as a side-product but is not applied to the mask directly —
    a downstream downsample step can use it as a tie-breaker.

Caching
    The heavy compute (kNN, curvature, color_grad, depth) is dumped to
    .npy sidecars on first run. Subsequent runs with different
    thresholds load those instantly (sub-second).

Outputs (next to the input PLY, stem = ply.stem):
    {stem}_curvature.npy     — float32 (N,)  raw curvature
    {stem}_cgrad.npy         — float32 (N,)  raw color stdev
    {stem}_depth.npy         — float32 (N,)  distance to camera0
    {stem}_feature_score.npy — float32 (N,)  rank-mode final score
    {stem}_score_curve.png   — full diagnostic plot (rank mode)
    {stem}_abs_<tag>.png     — diagnostic plot (absolute mode), tag
                               encodes the thresholds, e.g.
                               abs_curv0p02_cgrad0p05.png

Usage:
    # initial run (heavy compute, ~35 s on 10 M cloud)
    python scripts/analyze_feature_score.py

    # iterate on absolute thresholds (cheap, uses cache)
    python scripts/analyze_feature_score.py --mode absolute \\
        --curv-thresh 0.02 --cgrad-thresh 0.05
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def _find_reference_camera_position(ply_path: Path, which: str = "last") -> np.ndarray | None:
    """Resolve the reference camera world position from a sibling
    transforms.json.

    Args:
        which: "last" → last frame's c2w translation (the operator's
               final viewpoint, typically near the object of interest).
               "first" → first frame.
    Walks up from the PLY's dir until it finds a `static_scene/transforms.json`
    (or any `transforms.json`).
    Returns the 3-vector translation, or None if not found.
    """
    for parent in [ply_path.parent, *ply_path.parents]:
        for cand in [parent / "static_scene" / "transforms.json", parent / "transforms.json"]:
            if cand.exists():
                meta = json.loads(cand.read_text())
                frames = meta.get("frames", [])
                if not frames:
                    continue
                fr = frames[-1] if which == "last" else frames[0]
                T = np.asarray(fr["transform_matrix"], dtype=np.float64)
                return T[:3, 3]
    return None


def _compute_or_load_components(ply_path: Path, k: int = 20, force: bool = False):
    """Compute (curvature, color_grad, depth, points, colors, cam0) for
    the cloud at ``ply_path``, caching each array as a .npy next to the
    PLY. Subsequent calls with the same PLY return instantly.

    Returns: dict with keys
        pts, cols, N, curvature, color_grad, depth, cam0, d_near, d_far
    """
    pc = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() else None
    if cols is None or cols.size == 0:
        raise SystemExit("input PLY has no colors — cannot compute color gradient")
    N = pts.shape[0]

    curv_npy = ply_path.with_name(ply_path.stem + "_curvature.npy")
    cgrad_npy = ply_path.with_name(ply_path.stem + "_cgrad.npy")
    depth_npy = ply_path.with_name(ply_path.stem + "_depth.npy")

    # Use LAST camera pose as the depth reference. The operator finishes
    # their sweep facing the object of interest, so distances measured
    # from the last pose track "near my target" semantics. Switching
    # back to the first pose is a one-liner if needed.
    cam0 = _find_reference_camera_position(ply_path, which="last")

    # Curvature + color_grad are cloud-only quantities, cacheable independently.
    feat_cache_ok = (not force) and curv_npy.exists() and cgrad_npy.exists()
    if feat_cache_ok:
        curvature = np.load(curv_npy)
        color_grad = np.load(cgrad_npy)
        if curvature.size == N and color_grad.size == N:
            print(f"[cache] loaded {N:,}-pt curvature + cgrad from .npy")
        else:
            feat_cache_ok = False
            print(f"[cache] feature sidecar size mismatch — recomputing")

    if not feat_cache_ok:
        t = time.time()
        tree = cKDTree(pts)
        print(f"[compute] cKDTree build       {time.time()-t:.2f}s")

        t = time.time()
        _, knn_idx = tree.query(pts, k=k, workers=-1)
        print(f"[compute] kNN query (N={N:,}, k={k}) {time.time()-t:.2f}s")

        t = time.time()
        P = pts[knn_idx]                              # (N, K, 3)
        Pc = P - P.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", Pc, Pc) / k
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0, None)
        s = eigvals.sum(axis=1)
        curvature = np.where(s > 1e-12, eigvals[:, 0] / s, 0.0).astype(np.float32)
        print(f"[compute] curvature           {time.time()-t:.2f}s")

        t = time.time()
        C = cols[knn_idx]
        color_std_ch = C.std(axis=1)
        color_grad = np.linalg.norm(color_std_ch, axis=1).astype(np.float32)
        print(f"[compute] color gradient      {time.time()-t:.2f}s")

        np.save(curv_npy, curvature)
        np.save(cgrad_npy, color_grad)
        print(f"[cache] wrote {curv_npy.name}, {cgrad_npy.name}")

    # Depth depends on the chosen reference pose — cache it separately
    # so changing the reference doesn't invalidate the expensive feature
    # cache. Verify the saved depth matches the current reference by
    # spot-checking max distance (cheap, no recompute when unchanged).
    depth_cache_ok = (not force) and depth_npy.exists()
    if depth_cache_ok:
        depth_arr = np.load(depth_npy)
        if depth_arr.size != N:
            depth_cache_ok = False
        elif cam0 is not None:
            # Sample a few points to check the cached depth was computed
            # against the current cam0 (within rounding tolerance).
            idx = np.array([0, N // 2, N - 1])
            expected = np.linalg.norm(pts[idx] - cam0, axis=1).astype(np.float32)
            if not np.allclose(depth_arr[idx], expected, atol=1e-3):
                depth_cache_ok = False
                print(f"[cache] depth sidecar was computed against a different reference — recomputing")
        if depth_cache_ok:
            print(f"[cache] loaded {N:,}-pt depth from .npy")
    if not depth_cache_ok:
        if cam0 is None:
            depth_arr = np.zeros(N, dtype=np.float32)
            print("[compute] WARNING: no transforms.json found — depth = 0")
        else:
            depth_arr = np.linalg.norm(pts - cam0, axis=1).astype(np.float32)
            print(f"[compute] depth from cam0={cam0.round(3).tolist()}")
        np.save(depth_npy, depth_arr)
        print(f"[cache] wrote {depth_npy.name}")

    # Depth range (always recomputed; cheap).
    if cam0 is None or depth_arr.max() < 1e-6:
        d_near = d_far = 0.0
    else:
        d_near = float(np.percentile(depth_arr, 1))
        d_far = float(np.percentile(depth_arr, 99))

    return {
        "pts": pts, "cols": cols, "N": N,
        "curvature": curvature, "color_grad": color_grad,
        "depth": depth_arr, "cam0": cam0,
        "d_near": d_near, "d_far": d_far,
    }


def _depth_weight(depth_arr, d_near, d_far, alpha=0.5):
    """Linear floor=(1-alpha) depth weight."""
    if d_far - d_near < 1e-6:
        return np.ones(depth_arr.size, dtype=np.float32)
    fade = ((depth_arr - d_near) / (d_far - d_near)).clip(0.0, 1.0)
    return (1.0 - alpha * fade).astype(np.float32)


def _run_rank_mode(ply_path, comp, K):
    """Rank-based score + diagnostic plot (the existing strategy)."""
    N = comp["N"]
    pts = comp["pts"]
    curvature = comp["curvature"]
    color_grad = comp["color_grad"]
    depth_arr = comp["depth"]
    cam0_used = comp["cam0"]
    d_near = comp["d_near"]; d_far = comp["d_far"]

    curv_rank = (rankdata(curvature, method="average") - 1) / max(N - 1, 1)
    cgrad_rank = (rankdata(color_grad, method="average") - 1) / max(N - 1, 1)
    raw_score = (0.5 * curv_rank + 0.5 * cgrad_rank).astype(np.float32)

    DEPTH_ALPHA = 0.5
    depth_weight = _depth_weight(depth_arr, d_near, d_far, alpha=DEPTH_ALPHA)
    score = (depth_weight * raw_score).clip(0.0, 1.0).astype(np.float32)

    out_npy = ply_path.with_name(ply_path.stem + "_feature_score.npy")
    np.save(out_npy, score)
    print(f"[rank] wrote {out_npy}")

    # --- Plot.
    sorted_score = np.sort(score)
    p25 = np.percentile(sorted_score, 25); p50 = np.percentile(sorted_score, 50); p75 = np.percentile(sorted_score, 75)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    ax.plot(np.linspace(0, 100, N), sorted_score, lw=1.0, label="final (with depth penalty)")
    ax.plot(np.linspace(0, 100, N), np.sort(raw_score), lw=0.8, color="gray", alpha=0.6, label="raw (no depth penalty)")
    for pct, val in [(25, p25), (50, p50), (75, p75)]:
        ax.axvline(pct, color="gray", ls="--", alpha=0.4, lw=0.6)
        ax.axhline(val, color="gray", ls=":", alpha=0.4, lw=0.6)
        ax.annotate(f"p{pct}={val:.3f}", (pct, val), fontsize=8, color="dimgray")
    ax.set_xlabel("percentile of points"); ax.set_ylabel("feature score")
    ax.set_title("Sorted final vs raw score"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(score, bins=120, color="steelblue", alpha=0.85)
    ax.axvline(p75, color="red", ls="--", lw=1.0, label=f"75th pct = {p75:.3f}")
    ax.set_xlabel("final feature score"); ax.set_ylabel("# points")
    ax.set_title("Final score histogram"); ax.legend()

    ax = axes[0, 2]
    ax.hist(depth_weight, bins=120, color="orange", alpha=0.85)
    title = "Depth penalty weight"
    if cam0_used is not None:
        title += f"  (cam0; d_near={d_near:.2f}m, d_far={d_far:.2f}m)"
    ax.set_xlabel("depth_weight (1=near, far→1-α)"); ax.set_ylabel("# points"); ax.set_title(title)

    ax = axes[1, 0]
    ax.hist(curvature, bins=120, color="darkgreen", alpha=0.7)
    ax.set_xlabel("raw curvature  λ₀/Σλ"); ax.set_ylabel("# points")
    ax.set_title("Raw curvature distribution"); ax.set_yscale("log")

    ax = axes[1, 1]
    ax.hist(color_grad, bins=120, color="purple", alpha=0.7)
    ax.set_xlabel("raw color stdev  ‖σ(RGB)‖"); ax.set_ylabel("# points")
    ax.set_title("Raw color-gradient distribution"); ax.set_yscale("log")

    ax = axes[1, 2]
    if cam0_used is not None:
        sample = np.random.default_rng(0).choice(N, size=min(50000, N), replace=False)
        ax.scatter(depth_arr[sample], score[sample], s=1.0, alpha=0.15, color="steelblue")
        ax.axvline(d_near, color="red", ls=":", lw=0.8, label=f"d_near={d_near:.2f}")
        ax.axvline(d_far, color="red", ls=":", lw=0.8, label=f"d_far={d_far:.2f}")
        ax.set_xlabel("distance from first camera (m)"); ax.set_ylabel("final feature score")
        ax.set_title("Score vs depth (50k random points)"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no transforms.json — depth penalty disabled", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(f"{ply_path.name}  N={N:,}  k={K}  raw=0.5·curv+0.5·cgrad,  final=depth·raw", fontsize=11)
    fig.tight_layout()
    out_png = ply_path.with_name(ply_path.stem + "_score_curve.png")
    fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"[rank] wrote {out_png}")

    print()
    print("=== rank-mode feature score summary ===")
    print(f"  N = {N:,}")
    print(f"  raw curvature : min {curvature.min():.4f}  median {np.median(curvature):.4f}  max {curvature.max():.4f}")
    print(f"  raw color std : min {color_grad.min():.4f}  median {np.median(color_grad):.4f}  max {color_grad.max():.4f}")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"     p{q:>2d} = {np.percentile(score, q):.4f}")
    print(f"\n  → 75th percentile = {p75:.4f}")
    print(f"     above p75: {int((score >= p75).sum()):,}")
    print(f"     below p75: {int((score <  p75).sum()):,}")


def _run_absolute_mode(ply_path, comp, K, curv_thresh, cgrad_thresh, tag=None,
                       depth_near=1.0, depth_far=None):
    """Absolute-threshold mode + per-point keep weight for the downsample.

    Two independent quantities are computed and saved:

      1. **High-detail mask** (pure feature-based, scene-scale invariant)
         high = (curvature > curv_thresh) OR (color_grad > cgrad_thresh)
         No depth involved here — a sharp corner is a sharp corner
         regardless of where it sits in the scene.

      2. **Depth keep_weight** for the downsample step (saved separately)
         keep_weight = 1                                  if d ≤ depth_near
         keep_weight = 1 - (d - dn) / (df - dn)            if dn < d ≤ df
         keep_weight = 0                                  if d > df
         "Points within depth_near of the reference camera are NEVER
         downsampled (weight=1). Beyond that, the weight fades linearly
         to 0 at depth_far, meaning the downsample step is free to thin
         those points more aggressively."

    depth_far defaults to the 99th-percentile per-point depth so the
    linear fade spans the actual scene extent.

    Writes:
      - {stem}_abs_{tag}_mask.npy        (bool)
      - {stem}_abs_{tag}_keepweight.npy  (float32 [0,1])  — for downsample
      - {stem}_abs_{tag}.png             (diagnostic plot)
      - {stem}_abs_{tag}_red.ply         (recolored visualization)
    """
    N = comp["N"]
    pts = comp["pts"]; cols = comp["cols"]
    curvature = comp["curvature"]; color_grad = comp["color_grad"]
    depth_arr = comp["depth"]
    cam0_used = comp["cam0"]
    d_near_pct = comp["d_near"]; d_far_pct = comp["d_far"]

    # 1) Pure feature-based high-detail mask (NO depth involvement).
    high_curv = curvature > curv_thresh
    high_cgrad = color_grad > cgrad_thresh
    high = high_curv | high_cgrad

    # 2) keep_weight: BINARY — 1.0 inside `depth_near` (always kept by
    # the downsample step), 0.0 beyond. The previous linear-fade form
    # was dropped on user request — they want one hard "never downsample"
    # zone, no soft falloff.
    keep_weight = (depth_arr <= depth_near).astype(np.float32)
    n_always_keep = int(keep_weight.sum())
    pct_always_keep = 100.0 * n_always_keep / N
    n_fully_downsamplable = N - n_always_keep
    # depth_far is kept for the plot axis but no longer affects the mask.
    if depth_far is None:
        depth_far = d_far_pct if d_far_pct > depth_near else (depth_near + 1.0)
    n_high = int(high.sum())
    pct_high = 100.0 * n_high / N

    n_curv_only = int((high_curv & ~high_cgrad).sum())
    n_cgrad_only = int((high_cgrad & ~high_curv).sum())
    n_both = int((high_curv & high_cgrad).sum())

    if tag is None:
        # Encode thresholds as "curv0p020_cgrad0p050".
        tag = f"curv{str(curv_thresh).replace('.', 'p')}_cgrad{str(cgrad_thresh).replace('.', 'p')}"
    out_mask_npy = ply_path.with_name(ply_path.stem + f"_abs_{tag}_mask.npy")
    np.save(out_mask_npy, high)
    out_keep_npy = ply_path.with_name(ply_path.stem + f"_abs_{tag}_keepweight.npy")
    np.save(out_keep_npy, keep_weight)

    # Recolored PLY: detail = red, "always-keep" near zone tinted blue,
    # remainder = original RGB. Helps verify the depth gate visually.
    pc = o3d.io.read_point_cloud(str(ply_path))
    recolor = np.asarray(pc.colors).copy()
    in_near_zone = depth_arr <= depth_near
    # Tint the near zone slightly blue (mix with original color so it's
    # still recognizable). Only points NOT classified as detail.
    tint_mask = in_near_zone & ~high
    if tint_mask.any():
        recolor[tint_mask] = 0.7 * recolor[tint_mask] + 0.3 * np.array([0.2, 0.4, 1.0])
    # Detail in red on top.
    recolor[high] = [1.0, 0.0, 0.0]
    pc.colors = o3d.utility.Vector3dVector(recolor.clip(0.0, 1.0))
    out_red_ply = ply_path.with_name(ply_path.stem + f"_abs_{tag}_red.ply")
    o3d.io.write_point_cloud(str(out_red_ply), pc)

    # Plot.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0, 0]
    ax.hist(curvature, bins=120, color="darkgreen", alpha=0.7)
    ax.axvline(curv_thresh, color="red", ls="--", lw=1.0, label=f"thr = {curv_thresh}")
    ax.set_xlabel("curvature  λ₀/Σλ"); ax.set_ylabel("# points")
    ax.set_title(f"Curvature  (>thr: {high_curv.sum():,} / {pct_high:.1f}% combined)")
    ax.set_yscale("log"); ax.legend()

    ax = axes[0, 1]
    ax.hist(color_grad, bins=120, color="purple", alpha=0.7)
    ax.axvline(cgrad_thresh, color="red", ls="--", lw=1.0, label=f"thr = {cgrad_thresh}")
    ax.set_xlabel("color stdev  ‖σ(RGB)‖"); ax.set_ylabel("# points")
    ax.set_title(f"Color gradient  (>thr: {high_cgrad.sum():,})")
    ax.set_yscale("log"); ax.legend()

    ax = axes[0, 2]
    # 2D log-density of curvature vs color_grad with the thresholds overlaid.
    h, xed, yed = np.histogram2d(curvature, color_grad, bins=80,
                                 range=[[0, min(curvature.max(), curv_thresh*5+1e-6)],
                                        [0, min(color_grad.max(), cgrad_thresh*5+1e-6)]])
    ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
              extent=[xed[0], xed[-1], yed[0], yed[-1]], cmap="viridis")
    ax.axvline(curv_thresh, color="red", ls="--", lw=1.0)
    ax.axhline(cgrad_thresh, color="red", ls="--", lw=1.0)
    ax.set_xlabel("curvature"); ax.set_ylabel("color stdev")
    ax.set_title("Joint distribution (log1p density)")

    ax = axes[1, 0]
    labels = ["flat\n(both ≤)", "curv only", "cgrad only", "both >"]
    counts = [N - n_high, n_curv_only, n_cgrad_only, n_both]
    colors = ["lightgray", "limegreen", "magenta", "red"]
    ax.bar(labels, counts, color=colors, alpha=0.85)
    for i, c in enumerate(counts):
        ax.annotate(f"{c:,}\n({100*c/N:.1f}%)", (i, c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# points")
    ax.set_title(f"Classification (high-detail total: {n_high:,} = {pct_high:.1f}%)")
    ax.set_yscale("log")

    ax = axes[1, 1]
    if cam0_used is not None:
        sample = np.random.default_rng(0).choice(N, size=min(50000, N), replace=False)
        is_high_sample = high[sample]
        ax.scatter(depth_arr[sample][~is_high_sample], curvature[sample][~is_high_sample],
                   s=0.8, alpha=0.10, color="gray", label="flat")
        ax.scatter(depth_arr[sample][is_high_sample], curvature[sample][is_high_sample],
                   s=1.2, alpha=0.40, color="red", label="high (feature-only)")
        # Show the curv_thresh as a flat line — depth no longer warps it.
        ax.axhline(curv_thresh, color="red", ls="--", lw=0.8, label=f"curv_thresh={curv_thresh}")
        # Shade the always-keep zone (d ≤ depth_near).
        ax.axvspan(0, depth_near, alpha=0.10, color="blue", label="always-keep zone")
        ax.axvline(depth_near, color="blue", ls=":", lw=0.8)
        ax.axvline(depth_far, color="black", ls=":", lw=0.5, alpha=0.4)
        ax.set_ylim(0, max(curv_thresh * 4, curvature[sample].max() * 1.05))
        ax.set_xlabel("distance from camera (m)"); ax.set_ylabel("curvature")
        ax.set_title("Depth vs curvature  (depth does NOT affect mask)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no depth", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 2]
    if cam0_used is not None:
        sample = np.random.default_rng(0).choice(N, size=min(50000, N), replace=False)
        ax.scatter(depth_arr[sample], keep_weight[sample], s=0.6, alpha=0.20, color="orange")
        ax.axvline(depth_near, color="blue", ls=":", lw=1.0, label=f"depth_near={depth_near}m")
        ax.set_xlabel("distance from camera (m)")
        ax.set_ylabel("keep_weight (BINARY)")
        ax.set_title(
            f"keep_weight = 1 for d ≤ {depth_near}m  ({n_always_keep:,} pts = {pct_always_keep:.1f}%)\n"
            f"keep_weight = 0 for d > {depth_near}m  ({n_fully_downsamplable:,} pts)")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "no depth", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(
        f"{ply_path.name}  N={N:,}  ABSOLUTE: high = (curv>{curv_thresh}) OR (cgrad>{cgrad_thresh})  "
        f"|  binary keep_weight (1 if d ≤ {depth_near}m else 0)",
        fontsize=10)
    fig.tight_layout()
    out_png = ply_path.with_name(ply_path.stem + f"_abs_{tag}.png")
    fig.savefig(out_png, dpi=110); plt.close(fig)

    print()
    print(f"=== absolute-threshold summary  (curv>{curv_thresh}, cgrad>{cgrad_thresh}) ===")
    print(f"  N total            : {N:,}")
    print(f"  flat               : {N - n_high:,}  ({100*(N-n_high)/N:.1f}%)")
    print(f"  curv only          : {n_curv_only:,}  ({100*n_curv_only/N:.1f}%)")
    print(f"  cgrad only         : {n_cgrad_only:,}  ({100*n_cgrad_only/N:.1f}%)")
    print(f"  both > thr         : {n_both:,}  ({100*n_both/N:.1f}%)")
    print(f"  HIGH (any >)       : {n_high:,}  ({pct_high:.1f}%)")
    print(f"  always-keep zone   : {n_always_keep:,}  ({pct_always_keep:.1f}%)  d ≤ {depth_near}m")
    print(f"  far zone (kw=0)    : {n_fully_downsamplable:,}  d ≥ {depth_far:.2f}m")
    print(f"  wrote {out_mask_npy.name}")
    print(f"  wrote {out_keep_npy.name}")
    print(f"  wrote {out_red_ply.name}")
    print(f"  wrote {out_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", type=Path, nargs="?",
                    default=Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/validate_run_1/_voxel_sweep/voxel_015_x0p1mm.ply"))
    ap.add_argument("--k", type=int, default=20, help="kNN neighbours for cov + color stdev")
    ap.add_argument("--mode", choices=["rank", "absolute"], default="rank")
    ap.add_argument("--curv-thresh", type=float, default=0.02,
                    help="Absolute curvature threshold (used in --mode absolute).")
    ap.add_argument("--cgrad-thresh", type=float, default=0.05,
                    help="Absolute color-gradient threshold (used in --mode absolute).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Filename tag override for absolute-mode outputs.")
    ap.add_argument("--depth-near", type=float, default=1.0,
                    help="Depth (m) within which depth_weight=1 (no penalty). Default 1.0.")
    ap.add_argument("--depth-far", type=float, default=None,
                    help="Depth (m) at which depth_weight=0 (point masked out). "
                         "Default: scene 99th-percentile distance.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute curvature/cgrad/depth even if cache exists.")
    args = ap.parse_args()

    ply_path = args.ply_path.resolve()
    print(f"[analyze] PLY: {ply_path}")
    comp = _compute_or_load_components(ply_path, k=args.k, force=args.force)

    if args.mode == "rank":
        _run_rank_mode(ply_path, comp, args.k)
    else:
        _run_absolute_mode(ply_path, comp, args.k, args.curv_thresh, args.cgrad_thresh,
                           tag=args.tag,
                           depth_near=args.depth_near,
                           depth_far=args.depth_far)


if __name__ == "__main__":
    main()
