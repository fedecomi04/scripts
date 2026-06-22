"""dynamic_track.py — XFeat rigid-object tracker (main thread, read-only).

Thin WRAP of the proven dynamic_gs.utils.xfeat_motion.XFeatMotionEstimator behind a
clean config-driven interface (seed/track + MotionEstimate). The heavy XFeat+LighterGlue
+RANSAC/Kabsch+multi-anchor logic is REUSED verbatim (battle-tested); this module only
provides the clean seam (config in, MotionEstimate out, no model access). The pipeline
renders the object mask under _model_lock and hands it in; the tracker holds NO lock and
never touches the GaussianSet (rewrite_spec/dynamic_track.md, Invariants #4/#8/#9).

ReferenceObjectPose turns the tracker's (R,t) + the D0 reference into the means/quats
subset the pipeline writes via GaussianSet.write_object_pose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class MotionEstimate:
    success: bool
    ready: bool
    rotation: np.ndarray            # (3,3) D0->current object rotation, world frame
    translation: np.ndarray         # (3,)
    inlier_count: int
    correspondence_count: int
    mean_residual: float
    median_residual: float
    timings: dict = field(default_factory=dict)


@dataclass
class TrackerInputs:
    rgb: torch.Tensor               # (H,W,3) float [0,1] on GPU
    depth: torch.Tensor             # (H,W) float32 metres, 0==invalid
    camera: object                  # nerfstudio Cameras (1 cam)
    keep_mask: Optional[torch.Tensor] = None     # gripper/robot keep mask (1==keep)
    object_mask: Optional[torch.Tensor] = None   # rendered tracked-object footprint
    stamp_sec: float = 0.0


class XFeatTracker:
    """Config-driven wrapper over the proven XFeatMotionEstimator. Main thread only."""

    def __init__(self, device, tracker_cfg, pose_filter_cfg):
        from dynamic_gs.utils.xfeat_motion import XFeatMotionEstimator
        # scale-aware anchor selection is read from env by the old estimator — translate cfg->env once.
        os.environ["DGS_XFEAT_SCALE_SELECT"] = "1" if tracker_cfg.scale_select else "0"
        self._est = XFeatMotionEstimator(
            device=device,
            top_k=tracker_cfg.top_k,
            min_track_points=tracker_cfg.min_track_points,
            ransac_iterations=tracker_cfg.ransac_iterations,
            ransac_inlier_threshold=tracker_cfg.ransac_inlier_threshold,
            anchor_rotation_gate_deg=tracker_cfg.rotation_gate_deg,
            anchor_scale_gate=tracker_cfg.scale_gate_ratio,
            lighterglue_depth_confidence=tracker_cfg.lighterglue_depth_confidence,
            pose_filter_enabled=pose_filter_cfg.enabled,
            pose_filter_accel_sigma=pose_filter_cfg.accel_sigma,
            pose_filter_alpha_sigma=pose_filter_cfg.alpha_sigma,
            pose_filter_meas_trans_sigma_m=pose_filter_cfg.meas_trans_mm / 1000.0,
            pose_filter_meas_rot_sigma_deg=pose_filter_cfg.meas_rot_deg,
            pose_filter_fixed_fps=pose_filter_cfg.fixed_fps,
            static_hold_window=tracker_cfg.static_hold_window,
            static_hold_trans_m=tracker_cfg.static_hold_trans_mm / 1000.0,
            static_hold_rot_deg=tracker_cfg.static_hold_rot_deg,
        )
        self.min_track_points = self._est.min_track_points

    @property
    def ready(self) -> bool:
        return bool(self._est.ready)

    def seed(self, inp: TrackerInputs) -> int:
        """Seed the D0 anchor from this frame's object-masked keypoints. Returns kept count."""
        mask = inp.object_mask if inp.object_mask is not None else inp.keep_mask
        return int(self._est.initialize(inp.rgb, inp.depth, inp.camera, mask))

    def track(self, inp: TrackerInputs) -> MotionEstimate:
        e = self._est.estimate_and_advance(
            current_rgb=inp.rgb, current_depth=inp.depth, current_camera=inp.camera,
            current_mask=inp.keep_mask, current_object_mask=inp.object_mask,
            current_stamp_sec=inp.stamp_sec,   # capture event-time -> KF dt = real gap between used frames
        )
        return MotionEstimate(
            success=bool(e.success), ready=bool(e.ready),
            rotation=np.asarray(e.rotation, dtype=np.float32),
            translation=np.asarray(e.translation, dtype=np.float32),
            inlier_count=int(e.inlier_count), correspondence_count=int(e.correspondence_count),
            mean_residual=float(e.mean_residual), median_residual=float(e.median_residual),
            timings=dict(e.timings or {}),
        )


# ----------------------------------------------------- reference-pose -> subset write
def _normalize_quats(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dim=-1)


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(3,3) rotation -> (4,) wxyz quaternion. Branch on the LARGEST of {trace, R00, R11, R22} so it is
    numerically stable for ALL rotation angles (the trace-only method gives a degenerate sqrt for angles
    >=120deg, trace <= 0). This is the single-matrix form of dynamic_ff_backends._rotmat_to_quat_wxyz —
    same argmax-branch algorithm + same epsilons (sqrt clamp_min 1e-12, normalize clamp_min 1e-9) — so the
    tracker's per-tick quaternion path and the FF batched path use one stable algorithm."""
    assert R.shape == (3, 3)
    r00, r11, r22 = R[0, 0], R[1, 1], R[2, 2]
    tr = r00 + r11 + r22
    cand = torch.stack([tr, r00, r11, r22])
    branch = int(cand.argmax())
    if branch == 0:                                  # trace branch
        s = torch.sqrt((tr + 1.0).clamp_min(1e-12)) * 2.0
        q = torch.stack([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif branch == 1:                                # R00 branch
        s = torch.sqrt((1.0 + r00 - r11 - r22).clamp_min(1e-12)) * 2.0
        q = torch.stack([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif branch == 2:                                # R11 branch
        s = torch.sqrt((1.0 - r00 + r11 - r22).clamp_min(1e-12)) * 2.0
        q = torch.stack([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:                                            # R22 branch
        s = torch.sqrt((1.0 - r00 - r11 + r22).clamp_min(1e-12)) * 2.0
        q = torch.stack([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / q.norm().clamp_min(1e-9)


class ReferenceObjectPose:
    """Snapshot the tracked object's D0 means/quats; apply (R,t) -> transformed subset.

    IDENTITY-KEYED (not position-keyed). The GaussianSet free-list REORDERS rows (swap-remove /
    insert), so row position is NOT stable across ticks. Each tracked row carries a stable `gauss_uid`;
    capture() stores the reference rest-pose KEYED BY uid, and apply() returns the transformed subset
    together with each row's uid. write_object_pose then resolves uid->live row under the lock, so the
    pose always lands on the right gaussian regardless of any reordering.
    """

    def __init__(self, d0_instance_id: int):
        self.d0_id = int(d0_instance_id)
        self._ref_means: Optional[torch.Tensor] = None
        self._ref_quats: Optional[torch.Tensor] = None
        self._ref_uid: Optional[torch.Tensor] = None      # uid of each ref row (same order as _ref_*)
        # _ref_uid is fixed after capture(), so its argsort is too -> precompute once instead of every tick.
        self._ref_order: Optional[torch.Tensor] = None        # argsort(_ref_uid)
        self._ref_uid_sorted: Optional[torch.Tensor] = None   # _ref_uid[_ref_order]

    def reference_centroid(self):
        """Mean of the captured D0 reference means -> the object's single-point hypothesis (a (3,) numpy
        array), or None if nothing is captured yet. Used by the pose log to record where the object is."""
        if self._ref_means is None:
            return None
        return self._ref_means.mean(0).detach().cpu().numpy()

    def capture(self, snapshot) -> int:
        ids = snapshot.buffers["object_instance_ids"][:, 0]
        mask = ids == self.d0_id
        ref_uid = snapshot.buffers["gauss_uid"][:, 0][mask].detach().clone()
        c = ref_uid.shape[0]                                   # mask count without an .item() CUDA sync
        if c == 0:
            self._ref_means = self._ref_quats = self._ref_uid = None
            self._ref_order = self._ref_uid_sorted = None
            return 0
        self._ref_means = snapshot.params["means"][mask].detach().clone()
        self._ref_quats = snapshot.params["quats"][mask].detach().clone()
        self._ref_uid = ref_uid
        self._ref_order = torch.argsort(self._ref_uid)        # invariant after capture -> hoist out of apply()
        self._ref_uid_sorted = self._ref_uid[self._ref_order]
        return c

    def apply(self, rotation, translation, snapshot):
        """-> (means_subset, quats_subset, subset_uid) or None if not capturable/mismatched.
        means_subset/quats_subset are the transformed rest-pose; subset_uid[i] is the stable gauss_uid of
        the gaussian row i belongs to. write_object_pose resolves uid->live row under its own lock, so a
        concurrent FF cull/insert/reorder between this snapshot and the write can't misplace the pose."""
        if self._ref_means is None:
            return None
        ids = snapshot.buffers["object_instance_ids"][:, 0]
        mask = ids == self.d0_id
        cur_uid = snapshot.buffers["gauss_uid"][:, 0][mask]      # uids in current mask order
        c = cur_uid.shape[0]                                     # tensor-shape count -> no .item() CUDA sync
        if c != self._ref_means.shape[0]:
            return None
        # reorder the reference rows so ref row r corresponds to the gaussian at current row r:
        # gather[r] = index into _ref_* whose uid == cur_uid[r]. argsort(_ref_uid) is invariant after
        # capture() so it is precomputed there; reuse it here. Same drift pattern as write_object_pose:
        # searchsorted().clamp(max=N-1) then equality-check (no host-side pos.max() sync).
        dev = self._ref_means.device
        ref_uid_sorted = self._ref_uid_sorted.to(cur_uid.device)
        pos = torch.searchsorted(ref_uid_sorted, cur_uid).clamp(max=ref_uid_sorted.shape[0] - 1)
        if not torch.equal(ref_uid_sorted[pos], cur_uid):
            return None                                         # uid set drifted (shouldn't happen)
        gather = self._ref_order.to(dev)[pos.to(dev)]           # ref-row index for each current row
        ref_means = self._ref_means[gather]
        ref_quats = self._ref_quats[gather]
        dt = self._ref_means.dtype
        R = torch.as_tensor(rotation, device=dev, dtype=dt).reshape(3, 3)
        t = torch.as_tensor(translation, device=dev, dtype=dt).reshape(3)
        if not torch.isfinite(R).all() or not torch.isfinite(t).all():
            return None
        means_sub = ref_means @ R.transpose(0, 1) + t[None, :]
        dq = rotation_matrix_to_quaternion(R).to(dev).expand(c, -1)
        quats_sub = _normalize_quats(_quat_mul(dq, ref_quats))
        return means_sub, quats_sub, cur_uid.detach().clone()      # uid-keyed, NOT a positional mask
