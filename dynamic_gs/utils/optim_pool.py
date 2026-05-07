"""Per-frame optim pool used by the decoupled tracking/optimization
dynamic phase.

The dynamic phase splits into two independent loops sharing one Gaussian
scene:

* **Tracking loop** runs FoundationPose every incoming frame so the
  object pose stays current, then asks the keyframe filter whether the
  frame should be optimized. Accepted frames with a non-trivial change
  mask are pushed onto an :class:`OptimPool`.

* **Optim loop** picks pool entries in round-robin order and runs one
  training step per pick. Pool entries store the *capture-time* CDN and
  ground-truth tensors; the per-step *effective* loss mask is
  ``cdn ⊙ (1 − render_object_mask(camera, current_object_pose))`` so any
  pixel currently occluded by the (possibly far-moved) object is skipped.

This module is the small data structure backing that loop. Eviction
policy lives in the pipeline (epoch budget OR loss dropped to a relative
fraction of its first-iteration value).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class OptimFrame:
    """One frame queued for optimization, snapshotted at tracker-tick time."""

    frame_idx: int
    camera: Any  # nerfstudio Cameras (single-frame slice)
    cdn: torch.Tensor  # (H, W, 1) float, capture-time change mask
    epochs_used: int = 0
    initial_loss: Optional[float] = None
    last_loss: Optional[float] = None
    # Live-mode only: the (image, mask, depth_image) batch sourced from
    # rospy at capture time. Recorded mode leaves this None and the
    # pipeline pins the datamanager to ``frame_idx`` to fetch the same
    # tensors from disk-backed cache. Used by `_dynamic_get_train_loss_dict`
    # to build the loss directly without consulting the dataparser.
    live_batch: Optional[dict] = None


class OptimPool:
    """FIFO with a capacity cap and a round-robin cursor.

    ``push`` drops the oldest entry on overflow (mirrors live behavior:
    if optim falls behind tracking, the queue forgets the oldest backlog).
    ``pick_round_robin`` rotates through the entries in arrival order so
    every frame in the pool gets steady attention.
    Eviction is caller-driven via ``evict``.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._q: deque[OptimFrame] = deque()
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._q)

    @property
    def is_empty(self) -> bool:
        return len(self._q) == 0

    def push(self, frame: OptimFrame) -> Optional[OptimFrame]:
        evicted: Optional[OptimFrame] = None
        if len(self._q) >= self.capacity:
            evicted = self._q.popleft()
            if self._cursor > 0:
                self._cursor -= 1
        self._q.append(frame)
        return evicted

    def pick_round_robin(self) -> Optional[OptimFrame]:
        n = len(self._q)
        if n == 0:
            return None
        frame = self._q[self._cursor % n]
        self._cursor = (self._cursor + 1) % n
        return frame

    def evict(self, frame: OptimFrame) -> None:
        try:
            idx = list(self._q).index(frame)
        except ValueError:
            return
        del self._q[idx]
        if self._cursor > idx:
            self._cursor -= 1
        if len(self._q) == 0:
            self._cursor = 0
        elif self._cursor >= len(self._q):
            self._cursor = 0
