"""Tests for dynamic_gs2.shm_channel — producer/consumer lifecycle + seqlock round-trip.

Run: python -m dynamic_gs2.tests.test_shm_channel   (from scripts/)
"""
import sys
import threading

import numpy as np

from dynamic_gs2 import frame as F
from dynamic_gs2 import shm_channel as SC
from dynamic_gs2.frame import Frame, Intrinsics

_NAME = "dgs_test_shm_chan"


def _mk_frame(seq, h=12, w=16):
    rgb = np.full((h, w, 3), seq % 256, dtype=np.uint8)
    depth = np.full((h, w), 0.5 + seq, dtype=np.float32)
    mask = np.ones((h, w), dtype=np.uint8)
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = seq * 0.01
    return Frame(seq=seq, stamp_sec=100.0 + seq, rgb_bgr=rgb, depth_m=depth,
                 mask_keep=mask, c2w_4x4=c2w)


def main():
    intr = Intrinsics(width=16, height=12, fx=8.0, fy=8.0, cx=8.0, cy=6.0)

    prod = SC.ShmProducer(intr, name=_NAME, num_slots=4)
    try:
        cons = SC.ShmConsumer(name=_NAME)

        # nothing published yet
        assert cons.read_latest() is None, "no frame -> None"
        assert cons.is_shutdown() is False

        # consumer re-derives intrinsics from the header
        assert cons.intrinsics == intr
        assert cons.layout.total_size == prod.layout.total_size

        # round-trip several frames; latest-wins
        for s in range(1, 8):
            prod.write(_mk_frame(s))
            got = cons.read_latest()
            assert got is not None and got.seq == s
            assert got.stamp_sec == 100.0 + s
            assert int(got.rgb_bgr[0, 0, 0]) == s % 256
            assert abs(float(got.depth_m[0, 0]) - (0.5 + s)) < 1e-4
            assert abs(float(got.c2w_4x4[0, 3]) - s * 0.01) < 1e-9
            # returned arrays are OWNED (survive the next write)
            got.rgb_bgr[0, 0, 0] = 222
            prod.write(_mk_frame(s + 100))
            assert int(got.rgb_bgr[0, 0, 0]) == 222, "consumer copy must be detached"

        # shutdown flag propagates
        prod.mark_shutdown()
        assert cons.is_shutdown() is True

        # consumer.close is detach-only: the segment + a fresh attach still work
        cons.close()
        assert cons.read_latest() is None, "closed consumer -> None"
        cons.close()  # idempotent
        cons2 = SC.ShmConsumer(name=_NAME)   # would FileNotFoundError if consumer unlinked
        prod.write(_mk_frame(900))
        assert cons2.read_latest().seq == 900, "segment survived consumer.close"
        cons2.close()

        # concurrent read during close must never touch freed memory (no crash)
        cons3 = SC.ShmConsumer(name=_NAME)
        stop = threading.Event()

        def _spin():
            while not stop.is_set():
                cons3.read_latest()  # tolerates None once closed

        t = threading.Thread(target=_spin)
        t.start()
        cons3.close()
        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive(), "reader thread should exit cleanly after close"
    finally:
        prod.close(unlink=True)

    # producer.close(unlink=True) reclaimed the name -> attach now fails
    try:
        SC.ShmConsumer(name=_NAME)
        raise AssertionError("expected segment to be unlinked after producer close")
    except FileNotFoundError:
        pass

    # __init__ unlinks-stale-before-create: a leaked prior segment is reclaimed
    leaked = SC.shared_memory.SharedMemory(name=_NAME, create=True, size=4096)
    leaked.close()  # leak it (don't unlink) -> simulate SIGKILLed prior producer
    prod2 = SC.ShmProducer(intr, name=_NAME, num_slots=4)  # must not raise FileExistsError
    prod2.close(unlink=True)

    print("test_shm_channel OK")


if __name__ == "__main__":
    sys.exit(main())
