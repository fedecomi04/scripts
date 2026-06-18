"""visualize.py — render the new pipeline tracking a recorded episode into a side-by-side mp4.

Replays a recorded dataset through the new dynamic pipeline (warm-load -> tracker ->
write-pose) and, each tick, renders the scene at the frame camera. Writes a
[ live frame | new-pipeline rendered scene ] video so the tracking is visible (the
gaussian object moves as the tracker drives it).

Usage (from scripts/):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -m dynamic_gs2.visualize \
        "<dataset>" [--transforms transforms_313_trimmed.json] [--out out.mp4]
        [--max-frames N] [--fps 15] [--width 960]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as C
from . import static_persist as P
from .adapters_source import ReplaySource, ShmRing, camera_from_frame
from .dynamic_track import ReferenceObjectPose, XFeatTracker
from .pipeline import DynamicLoop, pick_d0_instance_id


def _render_bgr(scene_model, lock, camera) -> np.ndarray:
    with lock:
        rgb, _, _ = scene_model.render(camera)
    img = (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--transforms", default="transforms_313_trimmed.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=960, help="per-panel width")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = Path(args.data)
    out = args.out or str(data / "dynamic_gs2_viz.mp4")
    cfg = C.load_runtime_config()
    dev = args.device

    sm, gset, lock = P.build_loaded_scene(cfg, dev, data / "static_scene" / "static_state.pt", phase="dynamic")
    d0 = pick_d0_instance_id(gset)
    tracker = XFeatTracker(dev, cfg.tracker, cfg.pose_filter)
    ref = ReferenceObjectPose(d0)

    src = ReplaySource(data, mode="fast", transforms_name=args.transforms)
    src.attach("dgs2_viz_shm")
    ring = ShmRing("dgs2_viz_shm")
    intr = ring.intrinsics()
    loop = DynamicLoop(sm, gset, lock, tracker, ref, d0, cfg, dev)
    loop.tracker_intr = intr

    pw = args.width
    ph = int(round(intr.height * pw / intr.width))
    tmp = Path(tempfile.mkdtemp(prefix="dgs2_viz_"))
    n = 0
    print(f"[viz] rendering tracked scene -> {out} (d0={d0}, {gset.num_points} gaussians)")
    try:
        while True:
            fr = src.next_frame()
            if fr is None or (args.max_frames and n >= args.max_frames):
                break
            ring_fr = ring.peek_latest() or fr
            row = loop.step(ring_fr)                       # tracks + writes pose into the scene
            rendered = _render_bgr(sm, lock, camera_from_frame(ring_fr, intr, dev))
            live = cv2.resize(ring_fr.rgb_bgr, (pw, ph))
            rend = cv2.resize(rendered, (pw, ph))
            # overlay tick + tracking info on the rendered panel
            ok = row.get("tracking_ok"); tmm = (sum(x * x for x in row.get("t", [0, 0, 0]))) ** 0.5 * 1000
            cv2.putText(rend, f"t{row['tick']} ok={ok} |t|={tmm:.0f}mm inl={row.get('inliers',0)}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.putText(live, "LIVE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(rend, "NEW-PIPELINE RENDER", (8, ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imwrite(str(tmp / f"f{n:05d}.png"), np.hstack([live, rend]))
            n += 1
            if n % 50 == 0:
                print(f"[viz] {n} frames")
    finally:
        ring.close(); src.close()

    print(f"[viz] encoding {n} frames @ {args.fps}fps")
    import os
    ff_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}  # system ffmpeg needs system libs
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                   check=True, capture_output=True, env=ff_env)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    print(f"[viz] DONE -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
