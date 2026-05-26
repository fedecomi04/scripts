"""Purged files kept for reference but not loaded by the runtime.

Contents:
  * Tracker backends superseded by ``xfeat_motion``:
    ``cotracker_motion.py``, ``tapir_motion.py``, ``tapnext_motion.py``,
    ``klt_motion.py``. Their shared dataclass + static helpers were
    factored into ``dynamic_gs.utils.tracker_common`` before the move.
  * ``live_ros_subscriber.py`` — superseded by ``live_shm_reader.py``
    (shared-memory subscriber + separate ROS publisher subprocess).

To revive any of these: re-add the dispatch / call sites that were
removed, fix imports to ``tracker_common``, and move back to
``dynamic_gs/utils/``.
"""
