"""A/B: scale-clamp-from-0 (buggy) vs from-130 (fixed) on the SAME data, multiscale 1/4->1/2->full.
Measures the END-OF-TRAIN opacity purge survival (the 'holes' metric): how many gaussians keep
opacity >= 0.05 after training. Buggy clamp during the coarse warmup => mass under-opacity => big purge.
Run: python -m dynamic_gs2.verify._probe_clamp_timing <dataset_dir>"""
import sys, threading
import numpy as np, torch
from pathlib import Path
from dynamic_gs2 import config as C, static_train, static_fuse
from dynamic_gs2.gaussian_set import GaussianSet
from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.static_persist import seed_ply_path

DS = Path(sys.argv[1]); cfg = C.load_runtime_config(); dev = "cuda"
sx, sr = static_fuse.load_seed_ply(seed_ply_path(DS))
def train(clamp_start):
    lock = threading.RLock()
    sm = SceneModel(cfg, dev, seed_xyz=sx, seed_rgb=sr, phase="static",
                    num_downscales=2, resolution_schedule=50)
    sm.attach_render_lock(lock); gset = GaussianSet(sm, lock, freelist=False)
    cams, batches = static_fuse.load_static_cameras(DS / "static_scene", dev)
    dmax = float(cfg.depth.scene_depth_max_m)
    def dk(b):
        d = b.get("depth_image")
        if dmax <= 0 or d is None: return None
        d = d.to(dev).float(); d = d[..., None] if d.ndim == 2 else d
        return ((d > 0.05) & (d < dmax)).float()
    sm.set_mask_provider(dk)
    static_train.train_static(sm, gset, cams, batches, num_steps=500, means_lr=0.0,
                              mixed_precision=False, early_stop_loss=0.0,
                              scale_clamp_start_step=clamp_start)
    n0 = gset.num_points
    purged = static_fuse.purge_low_opacity(gset, cfg.budget.static_opacity_purge_threshold)
    return n0, n0 - purged, purged

for cs, label in [(0, "clamp-from-0 (BUGGY)"), (130, "clamp-from-130 (FIXED)")]:
    n0, survive, purged = train(cs)
    print(f"{label:28s}: {n0:,} -> purge kept {survive:,} ({survive/n0*100:.0f}%)  dropped {purged:,}")
