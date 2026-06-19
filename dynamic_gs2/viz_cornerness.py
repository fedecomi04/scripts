"""viz_cornerness.py — color a dumped FF insert by its kNN-PCA corner_score so you can SEE which
points the pipeline treats as corners (and pick corner_merge_threshold / corner_var_scale by eye).

Dump an insert first:  DGS_DUMP_INSERT=/tmp/dgs2_insert.npz DGS_FF_VOXEL_MERGE_M=0 DGS_FF_GROW_INPLANE=1.0 \
                       dynamic_gs2/view_dynamic.sh "<dataset>" transforms_313_trimmed.json --ff
Then:  <env>/bin/python -m dynamic_gs2.viz_cornerness /tmp/dgs2_insert.npz [var_scale] [knn_k]

Writes two PLYs next to the npz:
  *_score.ply  — continuous blue(flat)->red(corner) heatmap of corner_score
  *_split.ply  — hard split at each threshold isn't possible in one file; this one uses thr=0.15
Open in any PLY viewer (e.g. the SuperSplat / MeshLab / `view_static_ckpt`-style tools).
Also prints the corner-FRACTION at several thresholds so you can choose corner_merge_threshold.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from dynamic_gs2.dynamic_ff_backends import _corner_score


def _write_ply(path, xyz, rgb):
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(n):
            f.write(f"{xyz[i,0]:.5f} {xyz[i,1]:.5f} {xyz[i,2]:.5f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
    print(f"  wrote {path}  (N={n})")


def main():
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/dgs2_insert.npz")
    var_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    knn_k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    d = np.load(npz)
    means = torch.from_numpy(d["means"]).float()
    cs = _corner_score(means, knn_k, var_scale).cpu().numpy()
    print(f"corner_score on {npz.name}: N={cs.shape[0]} var_scale={var_scale} knn_k={knn_k}")
    print(f"  distribution p50/p90/p99/max = {np.quantile(cs,0.5):.3f}/{np.quantile(cs,0.9):.3f}/"
          f"{np.quantile(cs,0.99):.3f}/{cs.max():.3f}")
    print("  corner fraction by threshold:")
    for thr in (0.10, 0.15, 0.20, 0.30, 0.50):
        print(f"    thr={thr:.2f} -> {100*np.mean(cs>=thr):.1f}% flagged corner")
    xyz = d["means"]
    # continuous heatmap: blue (flat, cs=0) -> red (corner, cs=1)
    heat = np.stack([cs, np.zeros_like(cs), 1.0 - cs], axis=1)
    _write_ply(npz.with_name(npz.stem + "_score.ply"), xyz, heat)
    # hard split at 0.15: red=corner, gray=flat
    thr = 0.15
    split = np.where((cs >= thr)[:, None], np.array([[1.0, 0, 0]]), np.array([[0.6, 0.6, 0.6]]))
    _write_ply(npz.with_name(npz.stem + "_split15.ply"), xyz, split)
    print("Open the *_score.ply (blue=flat, red=corner) to choose corner_merge_threshold.")


if __name__ == "__main__":
    main()
