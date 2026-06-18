"""Round-trip + edge tests for dynamic_gs2.frame (the SHM codec contract).

Run: python -m dynamic_gs2.tests.test_frame   (from scripts/)
"""
import sys
from multiprocessing import shared_memory

import numpy as np

from dynamic_gs2 import frame as F


def _make_frame(seq, h=12, w=16):
    rng = np.random.default_rng(seq)
    return F.Frame(
        seq=seq,
        stamp_sec=1000.0 + seq * 0.0333,
        rgb_bgr=rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        depth_m=rng.random((h, w), dtype=np.float32) * 3.0,
        mask_keep=(rng.random((h, w)) > 0.1).astype(np.uint8),
        c2w_4x4=rng.random((4, 4)),
    )


def main():
    h, w, ns = 12, 16, 4
    intr = F.Intrinsics(width=w, height=h, fx=520.5, fy=521.0, cx=8.0, cy=6.0)
    layout = F.compute_layout(h, w, ns)
    shm = shared_memory.SharedMemory(create=True, size=layout.total_size)
    buf = shm.buf
    try:
        F.pack_header(buf, intr, ns)

        # header round-trips, magic/version validated
        ri, rns, latest, ready, shutdown = F.read_header(buf)
        assert ri == intr, (ri, intr)
        assert rns == ns and latest == 0 and not ready and not shutdown
        assert F.read_latest(buf, layout) is None, "no frame published yet -> None"

        # write/read round-trip for several frames (exercises the ring + seqlock)
        for seq in range(1, 2 * ns + 3):
            fr = _make_frame(seq, h, w)
            F.write_frame(buf, layout, fr)
            got = F.read_latest(buf, layout)
            assert got is not None
            assert got.seq == seq
            assert abs(got.stamp_sec - fr.stamp_sec) < 1e-6
            assert np.array_equal(got.rgb_bgr, fr.rgb_bgr), "rgb torn"
            assert np.array_equal(got.depth_m, fr.depth_m), "depth torn"
            assert np.array_equal(got.mask_keep, fr.mask_keep), "mask torn"
            assert np.allclose(got.c2w_4x4, fr.c2w_4x4), "c2w torn"
            # returned arrays must OWN their data (survive the next write)
            snapshot = got.rgb_bgr.copy()
            F.write_frame(buf, layout, _make_frame(seq + 100, h, w))
            assert np.array_equal(got.rgb_bgr, snapshot), "read_latest must copy out, not view"

        # foreign magic is rejected
        bad = bytearray(layout.total_size)
        try:
            F.read_header(memoryview(bad))
            raise AssertionError("expected magic mismatch")
        except ValueError:
            pass

        print("test_frame OK  (layout total = %d bytes, slot = %d)" % (layout.total_size, layout.slot_size))
    finally:
        del buf
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    sys.exit(main())
