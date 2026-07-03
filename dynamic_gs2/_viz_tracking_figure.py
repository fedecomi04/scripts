"""_viz_tracking_figure.py — presentation figure/video of how the XFeat+LighterGlue
rigid-object tracker works, read STRAIGHT FROM THE PIPELINE (no re-implementation).

It constructs the *actual* dynamic_gs.utils.xfeat_motion.XFeatMotionEstimator that the
live pipeline uses, seeds the D0 anchor on dynamic frame 0, then runs
estimate_and_advance() on later frames. Every keypoint, every LighterGlue match and every
RANSAC/Kabsch inlier drawn here is what the tracker computed — this module only renders it.

Pipeline parity (dynamic_gs2/pipeline.py):
  * The live pipeline CROPS the tracker input to the object's bbox so XFeat's top_k
    keypoints land ON the small object, not spread over the full 800x800 frame. We
    replicate that crop; world-frame (R,t) is crop-invariant so the estimate is identical.
  * The pipeline restricts per-frame matches to the RENDERED object footprint. We reuse
    the pipeline's OWN dumped rendered object masks (dynamic_scene/_ff_debug/*_objmask.png)
    as `current_object_mask`, so the match filter + object/background split are exactly
    what the pipeline used on those frames.

Left panel  : the FIXED D0 anchor RGB with the object keypoints (green) that seed the tracker.
Right panel : a later frame; matched-inlier keypoints green, unmatched / RANSAC-rejected red;
              thin lines join matched pairs (green inlier / red outlier). Caption states the
              estimator + the real RANSAC inlier/correspondence count and median residual.

Outputs (under <data>/dynamic_scene/):
  tracking_figure_frameNNNN.png   — one still per sampled frame
  tracking_explainer.mp4          — the sampled frames as a slow explainer clip

Run in the `dynamic_gs` conda env (XFeat + nerfstudio + gsplat live there).
"""
from __future__ import annotations

import re
import sys
import json
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dynamic_gs.utils.xfeat_motion import XFeatMotionEstimator  # noqa: E402

DATA = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "recording_15fps_2026-06-11_115107"
)
DYN = DATA / "dynamic_scene"
FF_DEBUG = DYN / "_ff_debug"

SAMPLE_STRIDE = 6           # render ~1 frame every 6 (a rendered footprint exists every 6) ...
SAMPLE_COUNT = 30           # ... 30 panels total (3x the original 10), frames ~11..185
FIRST_SAMPLE = 10
CROP_PAD = 90               # px around the object bbox — tight enough that XFeat's top_k budget
                            # lands mostly ON the object, so the object-only seed is feature-rich

# TrackerConfig defaults, verbatim from dynamic_gs2/config.py.
TOP_K = 1024
RANSAC_ITERS = 68
RANSAC_INLIER_M = 0.004
MIN_TRACK_POINTS = 12
# LighterGlue's learned confidence gate. The default (0.1) is TOO conservative on this small,
# rotated object: on the frames that looked "lost" it returned <=4 matches even though the
# matches genuinely exist (raw MNN found ~108, and MNN->RANSAC recovers a valid 89-inlier / 1.9 mm
# object pose). Lowering the gate to 0.02 is what recovers the matches; RANSAC's 4 mm geometric
# gate then rejects any wrong ones, so the pose stays clean.
LIGHTERGLUE_MIN_CONF = 0.02

# drawing (BGR)
GREEN = (50, 220, 50); RED = (70, 70, 215)
LINE_GREEN = (60, 200, 60); LINE_RED = (80, 80, 200)
KP_RADIUS = 3               # green (matched) keypoints
BG_RADIUS = 2               # red (unmatched) keypoints — smaller so green pops
LINE_THICKNESS = 1          # "very thin"


def _load_transforms():
    meta = json.loads((DYN / "transforms.json").read_text())
    intr = dict(fx=float(meta["fl_x"]), fy=float(meta["fl_y"]),
                cx=float(meta["cx"]), cy=float(meta["cy"]),
                w=int(meta["w"]), h=int(meta["h"]))

    def _idx(f):
        n = re.findall(r"\d+", f["file_path"])
        return int(n[-1]) if n else 0

    return intr, sorted(meta["frames"], key=_idx)


def _rendered_objmask_index():
    """frame_number -> rendered object-footprint mask path, from the pipeline's dumps."""
    idx = {}
    for p in glob.glob(str(FF_DEBUG / "*_objmask.png")):
        m = re.search(r"frame_(\d+)_objmask", p)
        if m:
            idx[int(m.group(1))] = p
    return idx


def _read_full(frame_meta):
    rgb_bgr = cv2.imread(str(DYN / frame_meta["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
    dpath = frame_meta.get("depth_file_path") or \
        frame_meta["file_path"].replace("rgb", "depth").replace(".png", ".tiff")
    d = cv2.imread(str(DYN / dpath.lstrip("./")), cv2.IMREAD_UNCHANGED)
    depth_m = d.astype(np.float32) * 1e-3
    c2w = np.asarray(frame_meta["transform_matrix"], dtype=np.float32)[:3, :4]
    return rgb_bgr, depth_m, c2w


def _bbox_from_mask(mask_bool, pad, W, H):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad, W - 1), min(int(ys.max()) + pad, H - 1))


def _make_camera(c2w, intr, bbox, device):
    """Cameras for a CROP: shift cx,cy by the crop origin (pose unchanged -> crop-invariant R,t)."""
    from nerfstudio.cameras.cameras import Cameras, CameraType
    x0, y0, x1, y1 = bbox
    return Cameras(
        camera_to_worlds=torch.from_numpy(np.ascontiguousarray(c2w)).unsqueeze(0),
        fx=intr["fx"], fy=intr["fy"], cx=intr["cx"] - x0, cy=intr["cy"] - y0,
        width=int(x1 - x0 + 1), height=int(y1 - y0 + 1),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)


def _crop_tensors(rgb_bgr, depth_m, bbox, device):
    x0, y0, x1, y1 = bbox
    rgb_c = np.ascontiguousarray(rgb_bgr[y0:y1 + 1, x0:x1 + 1, ::-1])
    depth_c = np.ascontiguousarray(depth_m[y0:y1 + 1, x0:x1 + 1])
    return (torch.from_numpy(rgb_c).float().to(device) / 255.0,
            torch.from_numpy(depth_c).to(device))


def _extract_all_keypoints(est, rgb_t):
    kp, _d, _g, _s = est._extract(est._prepare_rgb_gpu(rgb_t))
    return kp


def _draw_points(img, pts, color, radius=KP_RADIUS):
    for x, y in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def _panel_label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (35, 35, 35), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def _compose(anchor_bgr, anchor_obj_kp, anchor_bg_kp,
             curr_bgr, curr_green_kp, curr_red_kp,
             match_anchor_xy, match_curr_xy, inlier_mask, caption, right_label, lost=False):
    H, W = anchor_bgr.shape[:2]
    left, right = anchor_bgr.copy(), curr_bgr.copy()
    if lost:                                   # red banner across the current panel
        cv2.rectangle(right, (0, 32), (W, 60), (30, 30, 160), -1)
        cv2.putText(right, "TRACKING LOST vs D0 anchor", (10, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    _draw_points(left, anchor_bg_kp, RED, radius=2)
    _draw_points(left, anchor_obj_kp, GREEN)
    _panel_label(left, "D0 anchor  (green = XFeat features on object -> tracker seed)")
    _draw_points(right, curr_red_kp, RED, radius=2)
    _draw_points(right, curr_green_kp, GREEN)
    _panel_label(right, right_label)

    gap = 24
    canvas = np.full((H + 92, W * 2 + gap, 3), 20, np.uint8)
    canvas[0:H, 0:W] = left
    canvas[0:H, W + gap:W * 2 + gap] = right
    off = W + gap
    for i in range(len(match_anchor_xy)):
        ax, ay = match_anchor_xy[i]; cx, cy = match_curr_xy[i]
        col = LINE_GREEN if (len(inlier_mask) and inlier_mask[i]) else LINE_RED
        cv2.line(canvas, (int(round(ax)), int(round(ay))),
                 (int(round(cx)) + off, int(round(cy))), col, LINE_THICKNESS, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, H + 2), (canvas.shape[1], H + 92), (245, 245, 245), -1)
    cv2.putText(canvas, caption, (14, H + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "green = matched RANSAC inlier   |   red = unmatched or RANSAC-rejected   "
                "|   line = LighterGlue correspondence",
                (14, H + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def _split_obj_bg(kp_full, objmask_full, W, H):
    xs = np.clip(np.round(kp_full[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(kp_full[:, 1]).astype(int), 0, H - 1)
    in_obj = objmask_full[ys, xs]
    return kp_full[in_obj], kp_full[~in_obj]


def _project_world_to_px(pt_world, c2w, intr):
    """OpenGL c2w[:3,:4] world point -> full-frame pixel (x, y) (inverse of backproject_to_world)."""
    R, t = c2w[:, :3], c2w[:, 3]
    pc = R.T @ (np.asarray(pt_world, np.float64) - t)
    z = -pc[2]
    if z <= 1e-6:
        return None
    return (intr["fx"] * (pc[0] / z) + intr["cx"], intr["fy"] * (-pc[1] / z) + intr["cy"])


def _object_bbox_for_frame(fi, objmask_idx, est, obj_centroid_world, c2w, intr, W, H):
    """Prefer the pipeline's dumped rendered footprint; else follow the object by projecting
    the D0 centroid through the tracker's running pose (both give an object-focused crop)."""
    if fi in objmask_idx:
        m = cv2.imread(objmask_idx[fi], cv2.IMREAD_GRAYSCALE) > 0
        bb = _bbox_from_mask(m, CROP_PAD, W, H)
        if bb is not None:
            return bb, m
    obj_now = est._cumulative_R @ obj_centroid_world + est._cumulative_t
    proj = _project_world_to_px(obj_now, c2w, intr) or (intr["cx"], intr["cy"])
    half = 140
    x0 = int(np.clip(proj[0] - half, 0, W - 1)); x1 = int(np.clip(proj[0] + half, 0, W - 1))
    y0 = int(np.clip(proj[1] - half, 0, H - 1)); y1 = int(np.clip(proj[1] + half, 0, H - 1))
    return (x0, y0, x1, y1), None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intr, frames = _load_transforms()
    W, H = intr["w"], intr["h"]

    objmask_idx = _rendered_objmask_index()
    avail = sorted(objmask_idx)
    # Sample ~1 frame every SAMPLE_STRIDE, SAMPLE_COUNT total, but ONLY frames that have a real
    # rendered object footprint (the pipeline dumps one every 6 frames). This guarantees each
    # panel gets an accurate object-region crop + match filter, so the object<->object matching
    # is clean; frames without a footprint fall back to a coarse crop and can drop the match.
    ideal = [FIRST_SAMPLE + i * SAMPLE_STRIDE for i in range(SAMPLE_COUNT)]
    picks = sorted({min(avail, key=lambda a: abs(a - k)) for k in ideal if k < len(frames)})
    picks_set = set(picks)
    print(f"[sample] render frames {picks}  (nearest rendered-footprint frames to {ideal}; "
          f"tracker runs through ALL 1..{picks[-1]} to keep the anchor-pool + pose state authentic)")

    est = XFeatMotionEstimator(
        device=device, top_k=TOP_K, min_track_points=MIN_TRACK_POINTS,
        ransac_iterations=RANSAC_ITERS, ransac_inlier_threshold=RANSAC_INLIER_M,
        lighterglue_min_conf=LIGHTERGLUE_MIN_CONF,
        pose_filter_enabled=False, static_hold_enabled=False,   # raw geometric estimate
    )

    # --- D0 anchor = the REAL dynamic frame 0 (photographic — XFeat/LighterGlue descriptors must
    # match the real dynamic frames; the SAM3D Gaussian RENDER has a render->real appearance gap
    # that collapses matching, measured <=13 matches + frequent loss). The seed crop already
    # focuses XFeat's top_k budget on the object. We seed WITH the object mask so the anchor holds
    # ONLY object descriptors — this is critical: an unmasked seed lets LighterGlue match the
    # abundant, stable TABLE/BACKGROUND features, and RANSAC's majority then locks onto the
    # background's zero motion (measured: 123 "inliers" but only 1 inside the object mask, 1 mm
    # pose = tracking the table, not the object). Masking the seed forces object<->object matches,
    # so the green correspondences are genuinely on the object and the pose reflects its motion.
    SAM3D_MASK = DYN / "initialization_debug" / "static0_obj_00_mask.png"
    seed_objmask = cv2.imread(str(SAM3D_MASK), cv2.IMREAD_GRAYSCALE) > 0
    # dilate the tight SAM3D mask so the seed keeps the object's full textured surface (edges/
    # handle), not just the interior — this is the object region used to BOTH seed and display.
    obj_region = cv2.dilate(seed_objmask.astype(np.uint8),
                            np.ones((15, 15), np.uint8), iterations=1).astype(bool)
    rgb0_bgr, depth0_m, c2w0 = _read_full(frames[0])

    d0_bbox = _bbox_from_mask(obj_region, CROP_PAD, W, H)
    dx0, dy0 = d0_bbox[0], d0_bbox[1]
    rgb0_t, depth0_t = _crop_tensors(rgb0_bgr, depth0_m, d0_bbox, device)
    cam0 = _make_camera(c2w0, intr, d0_bbox, device)
    obj_region_crop_t = torch.from_numpy(
        obj_region[d0_bbox[1]:d0_bbox[3] + 1, d0_bbox[0]:d0_bbox[2] + 1].astype(np.float32)).to(device)
    kept = est.initialize(rgb0_t, depth0_t, cam0, obj_region_crop_t)   # OBJECT-ONLY seed
    print(f"[seed] D0 anchor seeded with {kept} object-only depth-valid keypoints")

    obj_centroid_world = est._anchors[0].world_3d.mean(axis=0).astype(np.float64)
    # The estimator's actual anchor keypoints are object-only by construction -> full-frame coords.
    anchor_obj_kp = est._anchors[0].keypoints.astype(np.float32) + np.array([dx0, dy0], np.float32)
    anchor_bg_kp = np.empty((0, 2), np.float32)   # nothing off-object seeds the anchor

    out_frames = []
    for fi in range(1, picks[-1] + 1):
        rgb_bgr, depth_m, c2w = _read_full(frames[fi])
        bbox, objmask_full = _object_bbox_for_frame(
            fi, objmask_idx, est, obj_centroid_world, c2w, intr, W, H)
        cx0, cy0 = bbox[0], bbox[1]
        rgb_t, depth_t = _crop_tensors(rgb_bgr, depth_m, bbox, device)
        cam = _make_camera(c2w, intr, bbox, device)

        gripper = cv2.imread(str(DYN / frames[fi]["mask_path"].lstrip("./")), cv2.IMREAD_GRAYSCALE)
        keep_t = None
        if gripper is not None:
            keep_t = torch.from_numpy(
                (gripper[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] > 0).astype(np.float32)).to(device)
        objmask_crop_t = None
        if objmask_full is not None:
            objmask_crop_t = torch.from_numpy(
                objmask_full[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1].astype(np.float32)).to(device)

        est_out = est.estimate_and_advance(
            current_rgb=rgb_t, current_depth=depth_t, current_camera=cam,
            current_mask=keep_t, current_object_mask=objmask_crop_t,
            current_stamp_sec=fi / 15.0,
        )
        if fi not in picks_set:
            continue   # advance the tracker on every frame, but only render the sampled ones

        m_anchor = est_out.previous_points_xy if est_out.previous_points_xy is not None else np.empty((0, 2), np.float32)
        m_curr = est_out.current_points_xy if est_out.current_points_xy is not None else np.empty((0, 2), np.float32)
        inl = est_out.tracked_inlier_mask if est_out.tracked_inlier_mask is not None else np.zeros((len(m_anchor),), bool)
        m_anchor_full = m_anchor + np.array([dx0, dy0], np.float32) if len(m_anchor) else m_anchor
        m_curr_full = m_curr + np.array([cx0, cy0], np.float32) if len(m_curr) else m_curr

        curr_all_full = _extract_all_keypoints(est, rgb_t) + np.array([cx0, cy0], np.float32)
        inlier_curr_full = m_curr_full[inl] if len(m_curr_full) else m_curr_full
        green_set = set(map(tuple, np.round(inlier_curr_full, 1))) if len(inlier_curr_full) else set()
        is_green = np.array([tuple(np.round(p, 1)) in green_set for p in curr_all_full], bool) \
            if len(curr_all_full) else np.zeros((0,), bool)
        curr_green, curr_red = curr_all_full[is_green], curr_all_full[~is_green]

        R = np.asarray(est_out.rotation, np.float64)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        t_mm = np.linalg.norm(est_out.translation) * 1000.0
        medres_mm = est_out.median_residual * 1000.0 if np.isfinite(est_out.median_residual) else float("nan")

        # A frame with no correspondences = the object has rotated too far for the SINGLE D0
        # anchor: LighterGlue finds nothing to match. This is the motivation for the multi-anchor
        # pool (which the full pipeline uses) — call it out explicitly so the loss reads as the
        # intended teaching point, not a broken render.
        lost = est_out.correspondence_count == 0
        if lost:
            caption = (
                "TRACKING LOST -- object rotated beyond the single D0 anchor: LighterGlue finds "
                "0 correspondences. This is WHY the pipeline keeps a growing multi-anchor pool "
                "(re-anchor as the view changes) instead of one fixed reference."
            )
            right_label = f"frame {fi}  (t = {fi/15.0:.2f} s)   --  no match against the D0 anchor"
        else:
            caption = (
                f"XFeat+LighterGlue features -> RANSAC-Kabsch rigid pose:  "
                f"{est_out.inlier_count}/{est_out.correspondence_count} inlier correspondences  "
                f"(3-pt RANSAC, {RANSAC_ITERS} iters, {RANSAC_INLIER_M*1000:.0f} mm inlier gate)  ->  "
                f"median residual {medres_mm:.2f} mm,  pose {ang:.1f} deg / {t_mm:.0f} mm from D0"
            )
            right_label = f"frame {fi}  (t = {fi/15.0:.2f} s,  +{fi} frames after D0)"

        fig = _compose(rgb0_bgr, anchor_obj_kp, anchor_bg_kp, rgb_bgr, curr_green, curr_red,
                       m_anchor_full, m_curr_full, inl, caption, right_label, lost=lost)
        out_path = DYN / f"tracking_figure_frame{fi:04d}.png"
        cv2.imwrite(str(out_path), fig)
        out_frames.append(fig)
        print(f"[fig] {out_path.name}  inliers={est_out.inlier_count}/{est_out.correspondence_count}"
              f"  medres={medres_mm:.2f}mm  success={est_out.success}")

    if out_frames:
        h, w = out_frames[0].shape[:2]
        vpath = DYN / "tracking_explainer.mp4"
        fps = 10.0
        hold = 6                              # hold each panel 6 frames -> ~0.6 s at 10 fps (readable)
        vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in out_frames:
            for _ in range(hold):
                vw.write(f)
        vw.release()
        print(f"[video] {vpath}  ({len(out_frames)} panels, {fps:.0f} fps, {hold} frames/panel, {w}x{h})")
    print("[done]")


if __name__ == "__main__":
    main()
