"""frame.py — the single source of truth for the sensor-frame contract + SHM codec.

Per rewrite_spec/frame.md + adapters_source.md (D1: this module OWNS Frame,
Intrinsics, the SHM layout, the codec, and LAYOUT_VERSION; shm_channel imports
from here, never redefines).

One Frame = one synced sensor tuple in the producer's numpy/CPU domain. Geometry
is metric; depth is float32 metres; c2w is OpenGL camera->world (the scene's
native convention). Everything ROS/robot/camera-specific lives upstream (the
source adapter); everything gaussian-specific lives downstream. This module knows
neither.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# --- versioned contract -------------------------------------------------------
LAYOUT_VERSION = 1            # bump on ANY change to Frame fields / slot layout
SHM_MAGIC = b"DGS1"          # 4 bytes; mismatch => wrong/foreign segment
DEFAULT_SHM_NAME = "dgs_frame_v1"
DEFAULT_NUM_SLOTS = 4         # ring depth; drop-oldest, latest-wins (ARCH #3)
_READ_RETRIES = 5            # seqlock torn-read retries before giving up


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Frame:
    """One synced sensor tuple. Immutable so it survives the SHM boundary safely.

    depth_m is float32 metres (0 == invalid/no-return) — the uint16-mm disk
    representation is internal to the replay source, never seen here.
    """
    seq: int                 # monotone, source-assigned, starts at 1
    stamp_sec: float         # CAPTURE event-time (sensor/sim clock), NEVER now()
    rgb_bgr: np.ndarray      # (H,W,3) uint8, BGR (cv2 convention)
    depth_m: np.ndarray      # (H,W)   float32, metres
    mask_keep: np.ndarray    # (H,W)   uint8 {0,1}, 1 == keep (robot excluded)
    c2w_4x4: np.ndarray      # (4,4)   float64, OpenGL camera->world


# --- SHM layout ---------------------------------------------------------------
# Header (little-endian "<", no padding): magic, version, height, width,
# num_slots, fx, fy, cx, cy, latest_seq, ready, shutdown.
_HDR_FMT = "<4sIIIIddddqII"
_HDR_SIZE = struct.calcsize(_HDR_FMT)        # 68
_OFF_LATEST_SEQ = struct.calcsize("<4sIIIIdddd")   # 52
_OFF_READY = _OFF_LATEST_SEQ + 8                    # 60
_OFF_SHUTDOWN = _OFF_READY + 4                      # 64
# Per-slot meta (precedes the rgb/depth/mask payload): seq(q), stamp(d), c2w(16d).
_SLOT_META_FMT = "<qd16d"
_SLOT_META_SIZE = struct.calcsize(_SLOT_META_FMT)   # 144


@dataclass(frozen=True)
class Layout:
    """Byte offsets for a given (H, W, num_slots) — pure arithmetic, no buffer."""
    height: int
    width: int
    num_slots: int
    rgb_bytes: int
    depth_bytes: int
    mask_bytes: int
    slot_size: int
    total_size: int


def compute_layout(height: int, width: int, num_slots: int = DEFAULT_NUM_SLOTS) -> Layout:
    hw = height * width
    rgb_b, depth_b, mask_b = hw * 3, hw * 4, hw * 1
    slot = _SLOT_META_SIZE + rgb_b + depth_b + mask_b
    return Layout(height, width, num_slots, rgb_b, depth_b, mask_b,
                  slot, _HDR_SIZE + num_slots * slot)


def pack_header(buf, intr: Intrinsics, num_slots: int) -> None:
    """Write the header into buf[:_HDR_SIZE]. latest_seq=0, ready=0, shutdown=0."""
    struct.pack_into(_HDR_FMT, buf, 0, SHM_MAGIC, LAYOUT_VERSION,
                     intr.height, intr.width, num_slots,
                     intr.fx, intr.fy, intr.cx, intr.cy, 0, 0, 0)


def read_header(buf) -> Tuple[Intrinsics, int, int, bool, bool]:
    """(intrinsics, num_slots, latest_seq, ready, shutdown). Validates magic+version."""
    magic, ver, h, w, ns, fx, fy, cx, cy, latest, ready, shutdown = \
        struct.unpack_from(_HDR_FMT, buf, 0)
    if magic != SHM_MAGIC:
        raise ValueError("SHM magic mismatch: %r (foreign/stale segment)" % (magic,))
    if ver != LAYOUT_VERSION:
        raise ValueError("SHM layout version %d != expected %d (rebuild producer)"
                         % (ver, LAYOUT_VERSION))
    return Intrinsics(w, h, fx, fy, cx, cy), ns, latest, bool(ready), bool(shutdown)


def set_latest_seq(buf, seq: int) -> None:
    struct.pack_into("<q", buf, _OFF_LATEST_SEQ, seq)


def set_ready(buf, ready: bool) -> None:
    struct.pack_into("<I", buf, _OFF_READY, 1 if ready else 0)


def set_shutdown(buf, shutdown: bool) -> None:
    struct.pack_into("<I", buf, _OFF_SHUTDOWN, 1 if shutdown else 0)


def _slot_payload_views(buf, layout: Layout, slot_idx: int):
    """numpy views (rgb, depth, mask) backed by the slot's payload region of buf."""
    base = _HDR_SIZE + slot_idx * layout.slot_size + _SLOT_META_SIZE
    h, w = layout.height, layout.width
    rgb = np.ndarray((h, w, 3), dtype=np.uint8, buffer=buf, offset=base)
    depth = np.ndarray((h, w), dtype=np.float32, buffer=buf, offset=base + layout.rgb_bytes)
    mask = np.ndarray((h, w), dtype=np.uint8, buffer=buf,
                      offset=base + layout.rgb_bytes + layout.depth_bytes)
    return rgb, depth, mask


def write_frame(buf, layout: Layout, frame: Frame) -> None:
    """Producer-side seqlock write of `frame` into slot (seq % num_slots).

    Order: tag slot.seq -> copy meta+payload -> publish header.latest_seq (barrier).
    Single-writer only.
    """
    slot_idx = frame.seq % layout.num_slots
    base = _HDR_SIZE + slot_idx * layout.slot_size
    c2w = np.ascontiguousarray(frame.c2w_4x4, dtype=np.float64).reshape(16)
    struct.pack_into(_SLOT_META_FMT, buf, base, frame.seq, float(frame.stamp_sec), *c2w.tolist())
    rgb_v, depth_v, mask_v = _slot_payload_views(buf, layout, slot_idx)
    rgb_v[:] = frame.rgb_bgr
    depth_v[:] = frame.depth_m
    mask_v[:] = frame.mask_keep
    set_latest_seq(buf, frame.seq)          # publish last


def read_latest(buf, layout: Layout) -> Optional[Frame]:
    """Consumer-side lock-free seqlock read of the freshest slot, or None.

    Returns None when no frame has been published yet or every retry saw a torn
    read (the slot was overwritten mid-copy — only possible if the writer lapped
    the ring during the read). The returned Frame OWNS its arrays (copied out),
    so it survives the next write.
    """
    for _ in range(_READ_RETRIES):
        latest = struct.unpack_from("<q", buf, _OFF_LATEST_SEQ)[0]
        if latest <= 0:
            return None
        slot_idx = latest % layout.num_slots
        base = _HDR_SIZE + slot_idx * layout.slot_size
        s1 = struct.unpack_from("<q", buf, base)[0]
        if s1 != latest:
            continue                         # slot already being overwritten
        seq, stamp, *c2w_flat = struct.unpack_from(_SLOT_META_FMT, buf, base)
        rgb_v, depth_v, mask_v = _slot_payload_views(buf, layout, slot_idx)
        rgb = rgb_v.copy(); depth = depth_v.copy(); mask = mask_v.copy()
        s2 = struct.unpack_from("<q", buf, base)[0]
        if s2 != latest:
            continue                         # writer lapped the ring mid-copy -> retry
        return Frame(seq=seq, stamp_sec=stamp,
                     rgb_bgr=rgb, depth_m=depth, mask_keep=mask,
                     c2w_4x4=np.asarray(c2w_flat, dtype=np.float64).reshape(4, 4))
    return None
