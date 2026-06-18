"""shm_channel.py — POSIX shared-memory lifecycle for the live RGB-D-mask-pose ring.

Thin lifecycle layer over the `frame.py` codec (D1: frame.py OWNS Layout/Frame/
Intrinsics/seqlock; this module never redefines them). One single-writer producer
owns the segment for its whole life; one-or-more consumers attach read-only and
NEVER unlink. The lock-free seqlock (slot.seq tag first, header.latest_seq last)
is the only cross-process synchronization — see frame.write_frame / frame.read_latest.

Stdlib + numpy + frame.py only — NO torch / nerfstudio / rospy, so the producer
side stays importable from the minimal py3.8 `dynamic_gs_ros` env.
"""
from __future__ import annotations

import atexit
import threading
from multiprocessing import resource_tracker, shared_memory
from typing import Optional

from . import frame as _f
from .frame import DEFAULT_NUM_SLOTS, DEFAULT_SHM_NAME, Frame, Intrinsics, Layout


def _unlink_stale(name: str) -> None:
    """Reclaim a leaked segment from a SIGKILLed prior producer (crash recovery)."""
    try:
        stale = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    try:
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass


class ShmProducer:
    """Creates + owns the segment. Sole writer. Unlinks on normal exit / atexit.

    The producer's __init__ unlinks-stale-before-create, so a crashed prior
    producer's leaked `/dev/shm/<name>` is reclaimed on the next launch (the real
    reclamation path); the atexit hook handles the normal-exit case.
    """

    def __init__(self, intr: Intrinsics, name: str = DEFAULT_SHM_NAME,
                 num_slots: int = DEFAULT_NUM_SLOTS):
        self.name = name
        self.layout: Layout = _f.compute_layout(intr.height, intr.width, num_slots)
        _unlink_stale(name)
        self._shm = shared_memory.SharedMemory(
            name=name, create=True, size=self.layout.total_size)
        self._closed = False
        buf = self._shm.buf
        _f.pack_header(buf, intr, num_slots)
        _f.set_ready(buf, True)             # latest_seq stays 0 until first write
        atexit.register(self._atexit_unlink)

    def write(self, frame: Frame) -> None:
        """Seqlock write into slot (seq % num_slots). Single-writer, non-blocking."""
        _f.write_frame(self._shm.buf, self.layout, frame)

    def mark_shutdown(self) -> None:
        """Signal consumers to stop. Does NOT clear slots, does NOT unlink."""
        if not self._closed:
            _f.set_shutdown(self._shm.buf, True)

    def close(self, unlink: bool = True) -> None:
        """Detach; unlink only on a normal producer exit (it owns the name)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
        finally:
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def _atexit_unlink(self) -> None:
        # Best-effort clean /dev/shm on interpreter exit; idempotent with close().
        try:
            self.close(unlink=True)
        except Exception:
            pass


class ShmConsumer:
    """Attaches read-only; NEVER unlinks. Close-safe against an in-flight read.

    Calls resource_tracker.unregister so the reader's atexit can never unlink a
    name it did not create (CPython bug #38119). Validates magic + layout version
    on attach (frame.read_header), failing loudly on a schema drift.
    """

    def __init__(self, name: str = DEFAULT_SHM_NAME):
        self.name = name
        self._shm = shared_memory.SharedMemory(name=name, create=False)
        # Reader must never reclaim a segment it only attached to.
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        intr, num_slots, _latest, _ready, _shutdown = _f.read_header(self._shm.buf)
        self.intrinsics: Intrinsics = intr
        self.layout: Layout = _f.compute_layout(intr.height, intr.width, num_slots)
        self._closed = False
        self._io_lock = threading.Lock()   # serializes read_latest vs close

    def read_latest(self) -> Optional[Frame]:
        """Lock-free seqlock read of the freshest frame (owned copy), or None.

        Returns None if closed, if nothing published yet, or under producer
        saturation (torn-read retry cap in frame.read_latest). Callers tolerate None.
        """
        with self._io_lock:
            if self._closed:
                return None
            return _f.read_latest(self._shm.buf, self.layout)

    def is_shutdown(self) -> bool:
        """Producer asked consumers to stop (header.shutdown). False if closed."""
        with self._io_lock:
            if self._closed:
                return False
            return _f.read_header(self._shm.buf)[4]

    def close(self) -> None:
        """Idempotent detach. Serialized against an in-flight read_latest."""
        with self._io_lock:
            if self._closed:
                return
            self._closed = True
            self._shm.close()
