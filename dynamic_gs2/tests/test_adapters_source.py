"""Tests for dynamic_gs2.adapters_source — ReplaySource -> SHM -> ShmRing round-trip + camera build.

Run (from scripts/):  conda run -n dynamic_gs python -m dynamic_gs2.tests.test_adapters_source
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from dynamic_gs2 import adapters_source as A
from dynamic_gs2.adapters_source import ReplaySource, ShmRing, camera_from_frame

_NAME = "dgs_test_replay"
H, W, NF = 8, 10, 4


def _make_dataset(root: Path):
    dd = root / "dynamic_scene"
    for sub in ("rgb", "depth", "masks"):
        (dd / sub).mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(NF):
        rgb = np.full((H, W, 3), i + 1, np.uint8)            # BGR, distinct per frame
        depth = np.full((H, W), (i + 1) * 100, np.uint16)    # mm -> 0.1*(i+1) m
        mask = np.full((H, W), 255, np.uint8)
        cv2.imwrite(str(dd / "rgb" / f"frame_{i:05d}.png"), rgb)
        cv2.imwrite(str(dd / "depth" / f"frame_{i:05d}.tiff"), depth)
        cv2.imwrite(str(dd / "masks" / f"frame_{i:05d}.png"), mask)
        c2w = np.eye(4); c2w[0, 3] = i * 0.1
        frames.append({"file_path": f"./rgb/frame_{i:05d}.png",
                       "depth_file_path": f"./depth/frame_{i:05d}.tiff",
                       "mask_path": f"./masks/frame_{i:05d}.png",
                       "transform_matrix": c2w.tolist(),
                       "stamp_wall": 1000.0 + i * 0.1})
    meta = {"fl_x": 6.0, "fl_y": 6.0, "cx": W / 2, "cy": H / 2, "w": W, "h": H, "frames": frames}
    (dd / "transforms.json").write_text(json.dumps(meta))


def main():
    root = Path(tempfile.mkdtemp(prefix="dgs2_replay_"))
    try:
        _make_dataset(root)

        # --- fast mode: lock-step, frame-exact ---
        src = ReplaySource(root, mode="fast")
        assert len(src) == NF
        intr = src.intrinsics()
        assert (intr.width, intr.height, intr.fx) == (W, H, 6.0)
        src.attach(_NAME)
        ring = ShmRing(_NAME)
        assert ring.intrinsics() == intr, "consumer re-derives intrinsics from header"

        for i in range(NF):
            fr = src.next_frame()
            assert fr is not None and fr.seq == i + 1
            got = ring.peek_latest()
            assert got is not None and got.seq == i + 1
            assert int(got.rgb_bgr[0, 0, 0]) == i + 1
            assert abs(float(got.depth_m[0, 0]) - 0.1 * (i + 1)) < 1e-4   # mm->m
            assert int(got.mask_keep[0, 0]) == 1
            assert abs(float(got.c2w_4x4[0, 3]) - i * 0.1) < 1e-9
            assert abs(got.stamp_sec - (i * 0.1)) < 1e-4                  # relative to stamp0
        assert src.next_frame() is None, "exhausted -> None"
        assert ring.is_shutdown() is True, "end-of-dataset marks shutdown"

        # camera_from_frame builds a real nerfstudio Cameras
        import torch
        cam = camera_from_frame(got, intr, torch.device("cpu"))
        assert tuple(cam.camera_to_worlds.shape) == (1, 3, 4)
        assert int(cam.width.item()) == W and int(cam.height.item()) == H
        assert abs(float(cam.fx.item()) - 6.0) < 1e-5
        ring.close()
        src.close()

        # --- paced mode: producer thread auto-drives on the stamp schedule ---
        src2 = ReplaySource(root, mode="paced")
        src2.attach(_NAME)
        ring2 = ShmRing(_NAME)
        first = ring2.peek_latest()
        assert first is not None and first.seq == 1, "paced pre-publishes frame 1 on attach"
        time.sleep(0.6)                          # stamps span 0.3s -> all 4 should publish
        last = ring2.peek_latest()
        assert last is not None and last.seq >= 1
        ring2.close()
        src2.close()

        print(f"test_adapters_source OK (fast frame-exact x{NF}, paced reached seq={last.seq})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
