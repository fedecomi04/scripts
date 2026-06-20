"""pipeline.py — the dynamic-phase orchestrator (god-file).

Wires source -> warm-loaded scene -> tracker -> (optional) FF worker, owns the ONE
_model_lock and the per-tick DynamicLoop. Recorded + live collapse into one loop fed
by a FrameSource adapter (ReplaySource or Ros1Source) through the SHM ring.

This module also provides run_recorded_trace(): a headless A/B driver that replays a
recorded dataset (fast, frame-exact) through the new pipeline and writes a per-tick
new_trace.jsonl (rigid transform + FF counts) for comparison against the old pipeline's
old_trace.jsonl (rewrite_spec/VERIFICATION.md).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import static_persist
from . import timing as _timing
from .adapters_source import ReplaySource, ShmRing, camera_from_frame
from .dynamic_feedforward import FeedforwardDispatch, FeedforwardWorker
from .dynamic_track import ReferenceObjectPose, TrackerInputs, XFeatTracker


# --------------------------------------------------------------- D0 helpers
def pick_d0_instance_id(gset) -> int:
    """The tracked object = the most-common non-zero object_instance_id in the loaded scene."""
    ids = gset.snapshot().buffers["object_instance_ids"][:, 0]
    nz = ids[ids > 0]
    if nz.numel() == 0:
        return -1
    vals, counts = torch.unique(nz, return_counts=True)
    return int(vals[int(counts.argmax())].item())


def _rgb_gpu(frame, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame.rgb_bgr[..., ::-1])).float().to(device) / 255.0


def _depth_gpu(frame, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame.depth_m)).to(device)


# --------------------------------------------------------------- the loop
class DynamicLoop:
    """Per-tick: tracker tick (render object-mask under lock, track, write pose) + optional FF.
    Single-threaded driving for the recorded A/B; the FF worker is the only bg thread."""

    def __init__(self, scene_model, gset, lock, tracker, ref_pose, d0_id, cfg, device,
                 ff_worker: Optional[FeedforwardWorker] = None, on_render=None):
        self.sm = scene_model
        self.g = gset
        self.lock = lock
        self.tracker = tracker
        self.ref = ref_pose
        self.d0_id = d0_id
        self.cfg = cfg
        self.device = device
        self.ff = ff_worker
        self.on_render = on_render
        self._tick = 0
        self._seeded = False
        self._filter_depth = bool(getattr(getattr(cfg, "depth", None), "filter_enabled", True))

    def _object_mask(self, camera) -> torch.Tensor:
        # Build the instance mask and render it under ONE lock hold: a fresh snapshot taken inside
        # the lock keeps the mask length == the live count, so a concurrent FF insert can't make the
        # mask (count N) disagree with the rendered params (count N') -> torn read -> IndexError.
        with self.lock:
            inst = (self.g.snapshot().buffers["object_instance_ids"][:, 0] == self.d0_id)
            return self.sm.render_object_mask(camera, inst)

    def reset_for_loop(self) -> None:
        """Replay wrapped to frame 0: restore the dynamic state so each pass is identical.
        Object gaussians -> D0 rest pose, all FF inserts dropped, deferred-cull hides cleared, and
        the tracker re-seeded on the next frame (initialize() clears anchors + KF + cumulative pose).
        ref is re-captured at the restored rest pose, so it stays the original D0 reference."""
        snap = self.g.snapshot()
        out = self.ref.apply(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), snap)
        if out is not None:
            self.g.write_object_pose(*out)                    # tracked object back to its D0 rest pose
        ids = self.g.snapshot().buffers["object_instance_ids"][:, 0]
        ins = torch.nonzero(ids == int(self.cfg.feedforward.insert_id), as_tuple=False).flatten()
        if ins.numel() > 0:
            self.g.cull(ins)                                  # drop every FF insert (instance_id == insert_id)
        self.sm.set_hidden_indices(None)                      # un-hide the deferred-cull rows
        self._seeded = False                                  # next frame re-seeds the tracker from frame 0

    @staticmethod
    def _bbox_from_mask(objmask, pad, W, H):
        ys, xs = torch.where(objmask)
        if xs.numel() == 0:
            return None
        x0 = max(0, int(xs.min()) - pad); y0 = max(0, int(ys.min()) - pad)
        x1 = min(W, int(xs.max()) + pad + 1); y1 = min(H, int(ys.max()) + pad + 1)
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            return None
        return x0, y0, x1, y1

    def _crop(self, rgb, depth, keep, objmask, cam, bbox, intr):
        """Crop rgb/depth/keep/objmask to bbox + rebuild a Cameras with cx/cy shifted
        (fx/fy unchanged so depth backprojection stays metric). Matches old _crop_for_xfeat."""
        from nerfstudio.cameras.cameras import Cameras, CameraType
        x0, y0, x1, y1 = bbox
        c2w = cam.camera_to_worlds[0] if cam.camera_to_worlds.ndim == 3 else cam.camera_to_worlds
        cam_c = Cameras(camera_to_worlds=c2w.unsqueeze(0).cpu(),
                        fx=intr.fx, fy=intr.fy, cx=intr.cx - x0, cy=intr.cy - y0,
                        width=int(x1 - x0), height=int(y1 - y0),
                        camera_type=CameraType.PERSPECTIVE).to(self.device)
        return (rgb[y0:y1, x0:x1].contiguous(), depth[y0:y1, x0:x1].contiguous(),
                keep[y0:y1, x0:x1].contiguous(), objmask[y0:y1, x0:x1].contiguous(), cam_c)

    def step(self, frame) -> dict:
        """One tick on a Frame. Returns a trace row (tick, frame_seq, R, t, ok, inliers, ff_*)."""
        self._tick += 1
        # Stamp a tracker-tick marker so the FF report can show WHERE the tracker ticked inside an
        # FF cycle (it can tick several times during the long lock-free AnySplat decode) — i.e. it
        # proves the FF bg thread isn't blocking the tracker.
        _timing.get_ledger().event("tracker_tick", tick=self._tick)
        intr = self.tracker_intr
        cam = camera_from_frame(frame, intr, self.device)
        rgb = _rgb_gpu(frame, self.device)
        depth = _depth_gpu(frame, self.device)
        keep = torch.from_numpy(np.ascontiguousarray(frame.mask_keep)).to(self.device).float()
        # Match old recorded path: filter depth (median+bilateral) at the batch source so the
        # tracker's RANSAC-Kabsch 3D points are clean, and composite the gripper-masked region
        # with the scene background so the tracker can't lock onto gripper texture.
        if self._filter_depth:
            try:
                from dynamic_gs.utils.depth_filter import filter_depth_torch
                depth = filter_depth_torch(depth)
            except Exception:
                pass
        km = keep if keep.ndim == 2 else keep[..., 0]
        bg = self.sm._get_background_color().to(self.device).view(1, 1, -1)
        rgb = rgb * km[..., None] + bg * (1.0 - km[..., None])
        snap = self.g.snapshot()
        objmask = self._object_mask(cam)

        # Crop tracker inputs to the object bbox so XFeat's top_k keypoints land ON the
        # (small) object instead of being spread over the full 1200p frame (matches old
        # _object_crop_bbox/_crop_for_xfeat). World-frame R,t is crop-invariant.
        t_rgb, t_depth, t_keep, t_objmask, t_cam = rgb, depth, keep, objmask, cam
        if getattr(self.cfg.tracker, "crop_to_object_bbox", True):
            bbox = self._bbox_from_mask(objmask, int(self.cfg.tracker.crop_padding_px),
                                        intr.width, intr.height)
            if bbox is not None:
                t_rgb, t_depth, t_keep, t_objmask, t_cam = self._crop(
                    rgb, depth, keep, objmask, cam, bbox, intr)

        inp = TrackerInputs(rgb=t_rgb, depth=t_depth, camera=t_cam, keep_mask=t_keep,
                            object_mask=t_objmask, stamp_sec=frame.stamp_sec)

        row = {"tick": self._tick, "frame_seq": int(frame.seq), "ff_fired": False,
               "ff_inserted": 0, "ff_culled": 0}
        if not self._seeded:
            self.ref.capture(snap)
            kept = self.tracker.seed(inp)
            self._seeded = self.tracker.ready
            row.update(seed=int(kept), tracking_ok=False,
                       R=np.eye(3).tolist(), t=[0, 0, 0], inliers=0,
                       total_gauss=self.g.num_points)
            return row

        est = self.tracker.track(inp)
        row["tracking_ok"] = bool(est.success)
        row["inliers"] = int(est.inlier_count)
        row["R"] = np.asarray(est.rotation, float).tolist()
        row["t"] = np.asarray(est.translation, float).reshape(3).tolist()
        if est.success:
            out = self.ref.apply(est.rotation, est.translation, snap)
            if out is not None:
                ms, qs, uids = out                       # uids: stable gauss_uid per subset row (NOT a mask)
                self.g.write_object_pose(ms, qs, uids)   # resolves uid->live row under the lock (FF-race-safe)

        if self.ff is not None and self.ff.due(self._tick):
            d = FeedforwardDispatch(
                seq=int(frame.seq), camera=cam, rgb_bgr=frame.rgb_bgr.copy(),
                depth_m=depth, object_mask=objmask, gripper_keep=keep,
                scene_intr={"fl_x": self.tracker_intr.fx, "fl_y": self.tracker_intr.fy,
                            "cx": self.tracker_intr.cx, "cy": self.tracker_intr.cy,
                            "w": self.tracker_intr.width, "h": self.tracker_intr.height},
                d0_instance_id=self.d0_id)
            row["ff_fired"] = self.ff.dispatch(d)
        row["ff_inserted"] = int(getattr(self.ff, "last_inserted", 0)) if self.ff else 0
        row["ff_culled"] = int(getattr(self.ff, "last_culled", 0)) if self.ff else 0
        row["total_gauss"] = self.g.num_points
        if self.on_render is not None:
            self.on_render(cam)
        return row


# --------------------------------------------------------------- recorded A/B driver
def run_recorded_trace(data_dir, cfg, device, out_trace: str, *, ff_enabled: bool = False,
                       transforms_name: str = "transforms.json", max_frames: Optional[int] = None,
                       cache_name: str = static_persist.DEFAULT_CACHE_NAME,
                       shm_name: str = "dgs_dynamic_gs2_ab") -> dict:
    """Replay a recorded dataset (fast, frame-exact) through the new pipeline; write a
    per-tick trace. ff_enabled=False validates the tracker pose path (no AnySplat needed)."""
    data_dir = Path(data_dir)
    cache = static_persist.warm_cache_path(data_dir, cache_name)
    sm, gset, lock = static_persist.build_loaded_scene(cfg, device, cache, phase="dynamic")
    d0_id = pick_d0_instance_id(gset)
    ref = ReferenceObjectPose(d0_instance_id=d0_id)
    tracker = XFeatTracker(device, cfg.tracker, cfg.pose_filter)

    src = ReplaySource(data_dir, mode="fast", transforms_name=transforms_name)
    src.attach(shm_name)
    ring = ShmRing(shm_name)
    intr = ring.intrinsics()

    ff = None
    if ff_enabled:
        from .dynamic_ff_backends import make_cdn_fn, make_decode_fn, AnysplatHandle
        anysplat = AnysplatHandle(device)
        anysplat.prewarm()       # warm-start AnySplat at load time so the first decode isn't a cold-start
        ff = FeedforwardWorker(gset, lock, cfg.feedforward, cfg.budget,
                               cdn_fn=make_cdn_fn(sm, lock, cfg, intr, data_dir=data_dir),
                               decode_fn=make_decode_fn(anysplat, cfg, intr),
                               set_hidden_fn=sm.set_hidden_indices)   # Option A: defer cull

    loop = DynamicLoop(sm, gset, lock, tracker, ref, d0_id, cfg, device, ff_worker=ff)
    loop.tracker_intr = intr

    n = 0
    t0 = time.time()
    rows = []
    out = open(out_trace, "w")
    try:
        while True:
            fr = src.next_frame()
            if fr is None or (max_frames is not None and n >= max_frames):
                break
            ring_fr = ring.peek_latest()
            row = loop.step(ring_fr if ring_fr is not None else fr)
            out.write(json.dumps(row) + "\n")
            rows.append(row)
            n += 1
        if ff is not None:
            ff.close()
    finally:
        out.close()
        ring.close()
        src.close()
    dt = time.time() - t0
    ok = sum(1 for r in rows if r.get("tracking_ok"))
    summary = {"frames": n, "tracking_ok": ok, "final_gauss": gset.num_points,
               "ff_inserts_total": sum(r["ff_inserted"] for r in rows),
               "wall_s": round(dt, 1), "hz": round(n / dt, 2) if dt > 0 else 0,
               "d0_instance_id": d0_id}
    print(f"[pipeline] recorded trace -> {out_trace}: {summary}")
    return summary


# --------------------------------------------------------------- recorded VIEW (validate w/ viewer, no sim)
def run_view_recorded(data_dir, cfg, device, *, transforms_name: str = "transforms.json",
                      fps: float = 10.0, ff_enabled: bool = False, loop_forever: bool = True,
                      cache_name: str = static_persist.DEFAULT_CACHE_NAME) -> None:
    """Replay a recorded dataset through the pipeline at ~fps WITH the viser-direct viewer up,
    so the operator can orbit and watch the tracker drive the scene — visual validation with
    NO live sim. Open http://localhost:<viser.port>. Ctrl-C to stop."""
    from .dynamic_viz import ViserBridge
    data_dir = Path(data_dir)
    cache = static_persist.warm_cache_path(data_dir, cache_name)
    sm, gset, lock = static_persist.build_loaded_scene(cfg, device, cache, phase="dynamic")
    d0_id = pick_d0_instance_id(gset)
    ref = ReferenceObjectPose(d0_instance_id=d0_id)
    tracker = XFeatTracker(device, cfg.tracker, cfg.pose_filter)

    ff = None
    if ff_enabled:
        from .dynamic_ff_backends import make_cdn_fn, make_decode_fn, AnysplatHandle

    src = ReplaySource(data_dir, mode="fast", transforms_name=transforms_name)
    src.attach("dgs2_view_shm")
    ring = ShmRing("dgs2_view_shm")
    intr = ring.intrinsics()
    if ff_enabled:
        anysplat = AnysplatHandle(device)
        anysplat.prewarm()       # warm-start AnySplat at load time so the first decode isn't a cold-start
        ff = FeedforwardWorker(gset, lock, cfg.feedforward, cfg.budget,
                               cdn_fn=make_cdn_fn(sm, lock, cfg, intr, data_dir=data_dir),
                               decode_fn=make_decode_fn(anysplat, cfg, intr),
                               set_hidden_fn=sm.set_hidden_indices)   # Option A: defer cull
    loop = DynamicLoop(sm, gset, lock, tracker, ref, d0_id, cfg, device, ff_worker=ff)
    loop.tracker_intr = intr

    def render_fn(cam):
        with lock:
            rgb, _, _ = sm.render(cam)
        return rgb

    bridge = ViserBridge(cfg.viser, device=device)
    bridge.attach(render_fn)
    frames = src._frames
    if frames:
        bridge.set_initial_camera(np.asarray(frames[-1]["transform_matrix"], float))  # operator's final view
    print(f"[pipeline] VIEW: d0={d0_id}, {gset.num_points} gaussians. Open http://localhost:{cfg.viser.port} and orbit.")
    dt = 1.0 / max(fps, 0.1)
    try:
        while True:
            src._idx = 0
            src._seq = 0
            loop._seeded = False
            loop._tick = 0
            ref._ref_means = None
            last_n = gset.num_points
            while True:
                fr = src.next_frame()
                if fr is None:
                    break
                ring_fr = ring.peek_latest() or fr
                row = loop.step(ring_fr)
                n_now = int(row.get("total_gauss", last_n))
                if ff is not None and n_now != last_n:                 # log only REAL scene-count changes
                    print(f"[FF] tick {row['tick']}: scene {last_n} -> {n_now} "
                          f"(+{n_now-last_n})", flush=True)
                    last_n = n_now
                bridge.update_camera_feed(ring_fr.rgb_bgr)
                bridge.update_tracked_camera(ring_fr.c2w_4x4)
                bridge.request_render()
                time.sleep(dt)
            if not loop_forever:
                break
            print("[pipeline] VIEW: episode end — resetting scene + replaying (Ctrl-C to stop)")
            # reload the SAME gset from the warm cache (keeps render_fn's tensor binding valid
            # via sm.rebind inside reload); re-seed the tracker on the next frame.
            static_persist.load_warm_cache(gset, cache, cfg)
    except KeyboardInterrupt:
        print("[pipeline] VIEW: stopped")
    finally:
        if ff is not None:
            ff.close()
        bridge.close()
        ring.close()
        src.close()


# --------------------------------------------------------------- live driver (operator step 4)
def run_live(data_dir, cfg, device, *, source_kind: str = "live_bridge", ff_enabled: bool = False,
             max_seconds: Optional[float] = None, cache_name: str = static_persist.DEFAULT_CACHE_NAME,
             **source_opts) -> None:
    """Live dynamic phase: warm-load the scene, open a live source (default the bridge over
    the proven ROS publisher), and tick the tracker on the freshest SHM frame until shutdown.

    NOTE: requires a live Gazebo/ROS stack — validated by the OPERATOR (pipeline step 4).
    The recorded path (run_recorded_trace) is the unattended-validated one."""
    from .adapters_source import open_source
    data_dir = Path(data_dir)
    cache = static_persist.warm_cache_path(data_dir, cache_name)
    sm, gset, lock = static_persist.build_loaded_scene(cfg, device, cache, phase="dynamic")
    d0_id = pick_d0_instance_id(gset)
    ref = ReferenceObjectPose(d0_instance_id=d0_id)
    tracker = XFeatTracker(device, cfg.tracker, cfg.pose_filter)

    ff = None
    # Warm-start AnySplat BEFORE the source starts: its ~17s model load + first-inference warm-up must
    # finish before any frames flow, otherwise the first decode stalls ~16s mid-episode (and for paced
    # replay the episode wall-clock — started by open_source's producer thread — would advance through
    # the whole load). So spawn the worker now, then wait_ready() just before open_source: the load
    # overlaps the scene warm-load above and becomes honest startup latency, never a mid-run stall.
    anysplat = None
    if ff_enabled:
        from .dynamic_ff_backends import make_cdn_fn, make_decode_fn, AnysplatHandle
        anysplat = AnysplatHandle(device)
        anysplat.prewarm()
        anysplat.wait_ready()    # block until loaded; producer clock (next line) starts on a ready worker
    # The 'replay' source needs the dataset dir (it paces recorded frames into SHM like the live
    # publisher); pass our resolved data_dir through.
    if source_kind == "replay":
        source_opts.setdefault("data_dir", data_dir)
    src = open_source(source_kind, shm_name=cfg.shm_name, attach=True, **source_opts)
    ring = ShmRing(cfg.shm_name)
    if ff_enabled:
        ff = FeedforwardWorker(gset, lock, cfg.feedforward, cfg.budget,
                               cdn_fn=make_cdn_fn(sm, lock, cfg, ring.intrinsics(), data_dir=data_dir),
                               decode_fn=make_decode_fn(anysplat, cfg, ring.intrinsics()),
                               set_hidden_fn=sm.set_hidden_indices)   # Option A: defer cull

    loop = DynamicLoop(sm, gset, lock, tracker, ref, d0_id, cfg, device, ff_worker=ff)
    loop.tracker_intr = ring.intrinsics()

    from .dynamic_viz import ViserBridge
    def _render_fn(cam):
        with lock:
            rgb, _, _ = sm.render(cam)
        return rgb
    bridge = ViserBridge(cfg.viser, device=device)
    bridge.attach(_render_fn)

    print(f"[pipeline] LIVE: d0_instance_id={d0_id}, scene={gset.num_points} gaussians, source={source_kind}")
    last_seq, last_stamp, t0 = -1, None, time.time()
    try:
        while True:
            if max_seconds is not None and time.time() - t0 > max_seconds:
                break
            fr = ring.peek_latest()
            if fr is None or int(fr.seq) == last_seq:
                time.sleep(0.002)
                if ring.is_shutdown():
                    break
                continue
            last_seq = int(fr.seq)
            # Looping replay: a big backward jump in the capture stamp means the episode wrapped to
            # frame 0 -> reset the dynamic state so every pass starts from the same D0 scene/tracker.
            if last_stamp is not None and fr.stamp_sec < last_stamp - 0.5:
                loop.reset_for_loop()
            last_stamp = fr.stamp_sec
            loop.step(fr)
            bridge.update_camera_feed(fr.rgb_bgr)
            bridge.update_tracked_camera(fr.c2w_4x4)
            bridge.request_render()
    except KeyboardInterrupt:
        print("[pipeline] LIVE: interrupted by operator")
    finally:
        if ff is not None:
            ff.close()
        bridge.close()
        ring.close()
        src.close()
        static_persist.save_warm_cache(gset, data_dir, cfg, filename="post_dynamic_state.pt")
        print(f"[pipeline] LIVE: saved post_dynamic_state.pt ({gset.num_points} gaussians)")
        if ff_enabled:                                       # FF timing report (always-on; written once at end)
            report_path = Path(data_dir) / "timing_report_ff.txt"
            _timing.get_ledger().write(report_path)
            print(f"[pipeline] LIVE: wrote FF timing report -> {report_path}")


# --------------------------------------------------------------- CLI
def _main():
    import argparse
    from . import config as _C
    ap = argparse.ArgumentParser(description="dynamic_gs2 dynamic-phase runner")
    ap.add_argument("--mode", choices=["recorded", "live", "view"], required=True)
    ap.add_argument("--data", required=True, help="dataset dir (with static_scene/static_state.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ff", action=argparse.BooleanOptionalAction, default=True,
                    help="feedforward (needs AnySplat worker); ON by default, --no-ff disables")
    ap.add_argument("--source", default="live_bridge", help="live source kind (live_bridge|ros1)")
    ap.add_argument("--transforms", default="transforms.json", help="recorded/view: transforms json name")
    ap.add_argument("--out-trace", default=None, help="recorded: new_trace.jsonl path")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fps", type=float, default=10.0, help="view: replay rate")
    ap.add_argument("--once", action="store_true", help="view: play once instead of looping")
    ap.add_argument("--loop", action="store_true",
                    help="replay-as-live: replay the episode forever (Ctrl-C to stop); tracker snap-resets at each wrap")
    ap.add_argument("--replaced-cull", dest="replaced_cull", action="store_true", default=None,
                    help="FF: cull the thin slab of old geometry the insert overwrites (caps growth)")
    ap.add_argument("--no-replaced-cull", dest="replaced_cull", action="store_false",
                    help="FF: disable the replaced-surface cull (insert-only)")
    args = ap.parse_args()
    cfg = _C.load_runtime_config()
    if args.replaced_cull is not None:                       # CLI overrides the config/env default
        import dataclasses as _dc
        cfg = _dc.replace(cfg, feedforward=_dc.replace(cfg.feedforward,
                                                        cull_replaced_enabled=bool(args.replaced_cull)))
    if args.mode == "recorded":
        out = args.out_trace or str(Path(args.data) / "new_trace.jsonl")
        run_recorded_trace(args.data, cfg, args.device, out, ff_enabled=args.ff,
                           transforms_name=args.transforms, max_frames=args.max_frames)
    elif args.mode == "view":
        run_view_recorded(args.data, cfg, args.device, transforms_name=args.transforms,
                          fps=args.fps, ff_enabled=args.ff, loop_forever=not args.once)
    else:
        # 'replay' simulates live: a producer thread paces recorded frames into SHM on their capture
        # schedule and the tracker reads the freshest (dropping stale frames if it falls behind) —
        # an honest real-time test without a Gazebo/ROS stack.
        src_opts = {}
        if args.source == "replay":
            src_opts = dict(replay_mode="paced", transforms_name=args.transforms,
                            replay_fps=args.fps, loop=args.loop)
        run_live(args.data, cfg, args.device, source_kind=args.source,
                 ff_enabled=args.ff, max_seconds=args.max_seconds, **src_opts)


if __name__ == "__main__":
    _main()
