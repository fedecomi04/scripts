#!/usr/bin/env python
"""Decode a ZED SVO2 recording into a dynamic_gs dataset folder.

Runs on the JETSON with the ZED venv python (pyzed needs the ZED SDK):
    /home/shengzhiwang/zed_env/bin/python zed_svo_to_dataset.py \
        --svo /path/scene.svo2 --out /path/out_dataset \
        --settings /home/shengzhiwang/zed_settings

Produces the exact layout the pipeline expects:
    out/
      rgb/   frame_000000.png      (BGR PNG, rectified LEFT image)
      depth/ frame_000000.tiff     (uint16 millimetres, 0 = invalid)
      masks/ frame_000000.png      (uint8, all 255 = keep everything; no robot in view)
      transforms.json              (fl_x/fl_y/cx/cy/w/h + per-frame OpenGL c2w)

Per-frame pose comes from the ZED's VIO (visual-inertial positional tracking).
COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP => the pose IS the OpenGL c2w the dataparser
uses. The depth/RGB are the rectified LEFT camera, so intrinsics are pinhole with
zero distortion (no camera_model/k1.. written, matching existing datasets).
"""
import argparse, json, os
import numpy as np
import pyzed.sl as sl

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


def pose_to_c2w(pose):
    """4x4 OpenGL camera-to-world from an sl.Pose (RIGHT_HANDED_Y_UP)."""
    T = sl.Transform()
    pose.pose_data(T)
    try:
        m = np.array(T.m, dtype=np.float64).reshape(4, 4)
        if m.shape == (4, 4):
            return m
    except Exception:
        pass
    R = np.array(pose.get_rotation_matrix(sl.Rotation()).r, dtype=np.float64).reshape(3, 3)
    t = np.array(pose.get_translation(sl.Translation()).get(), dtype=np.float64).reshape(3)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return c2w


def write_depth_tiff(path, dmm):
    if _HAVE_CV2:
        # LZW compression to match existing datasets; falls back if unsupported
        try:
            cv2.imwrite(path, dmm, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
            return
        except Exception:
            cv2.imwrite(path, dmm)
            return
    from PIL import Image
    Image.fromarray(dmm).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default=None, help="optional_settings_path (calib dir)")
    ap.add_argument("--depth-mode", default="NEURAL",
                    choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_PLUS"])
    ap.add_argument("--confidence", type=int, default=75,
                    help="SDK confidence_threshold 1..100; LOWER discards MORE depth (keeps only "
                         "high-confidence points -> fewer points, fewer ghosts). On NEURAL the "
                         "score saturates: 65/75/90 give ~the same ~69%% coverage on zed_scene. "
                         "Default 75 (tuned on zed_scene, clean + dense).")
    ap.add_argument("--texture-confidence", type=int, default=100,
                    help="SDK texture_confidence_threshold 1..100; LOWER rejects depth on "
                         "low-texture surfaces. Default 100 (reject nothing on texture). On NEURAL "
                         "keep at 100 (NEURAL INFERS textureless depth well; lowering it deletes "
                         "good inferred depth). Measured cliff on zed_scene @conf75: "
                         "tex100->68.9%% tex99->50.4%% tex95->18.1%%.")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth grabbed frame")
    ap.add_argument("--min-depth", type=float, default=0.3, help="metres (ZED X 2.2mm min ~0.3)")
    ap.add_argument("--max-depth", type=float, default=20.0, help="metres")
    ap.add_argument("--prefix", default="frame_")
    args = ap.parse_args()

    rgb_dir = os.path.join(args.out, "rgb")
    depth_dir = os.path.join(args.out, "depth")
    mask_dir = os.path.join(args.out, "masks")
    for d in (rgb_dir, depth_dir, mask_dir):
        os.makedirs(d, exist_ok=True)

    cam = sl.Camera()
    ip = sl.InitParameters()
    ip.set_from_svo_file(args.svo)
    if args.settings:
        # SDK concatenates path + "SN<serial>.conf", so a trailing slash is required
        ip.optional_settings_path = os.path.join(args.settings, "")
    ip.coordinate_units = sl.UNIT.METER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL c2w
    ip.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    ip.depth_minimum_distance = args.min_depth
    ip.depth_maximum_distance = args.max_depth
    ip.svo_real_time_mode = False
    st = cam.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("ZED open failed: %s (calib file present in --settings?)" % st)

    cam.enable_positional_tracking(sl.PositionalTrackingParameters())

    info = cam.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    W = int(info.camera_configuration.resolution.width)
    H = int(info.camera_configuration.resolution.height)
    n_total = cam.get_svo_number_of_frames()
    print("SVO: %d frames @ %dx%d | fx=%.2f fy=%.2f cx=%.2f cy=%.2f | depth=%s"
          % (n_total, W, H, calib.fx, calib.fy, calib.cx, calib.cy, args.depth_mode))

    img, depth, pose = sl.Mat(), sl.Mat(), sl.Pose()
    rt = sl.RuntimeParameters()
    if args.confidence is not None:
        rt.confidence_threshold = args.confidence
    if args.texture_confidence is not None:
        rt.texture_confidence_threshold = args.texture_confidence
    valid_pcts = []
    frames = []
    grabbed = -1
    kept = 0
    n_tracked = 0
    while True:
        g = cam.grab(rt)
        if g == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if g != sl.ERROR_CODE.SUCCESS:
            continue
        grabbed += 1
        if grabbed % args.every != 0:
            continue
        track_state = cam.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if track_state == sl.POSITIONAL_TRACKING_STATE.OK:
            n_tracked += 1
        c2w = pose_to_c2w(pose)

        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        bgr = img.get_data()[:, :, :3].copy()                       # BGR
        d = depth.get_data().astype(np.float32)                     # metres; NaN/Inf invalid
        dmm = np.clip(np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0),
                      0, 65535).astype(np.uint16)                   # uint16 mm, 0 = invalid
        valid_pcts.append(100.0 * float((dmm > 0).mean()))          # depth-coverage %

        name = "%s%06d" % (args.prefix, kept)
        if _HAVE_CV2:
            cv2.imwrite(os.path.join(rgb_dir, name + ".png"), bgr)
            cv2.imwrite(os.path.join(mask_dir, name + ".png"),
                        np.full((H, W), 255, np.uint8))
        else:
            from PIL import Image
            Image.fromarray(bgr[:, :, ::-1]).save(os.path.join(rgb_dir, name + ".png"))
            Image.fromarray(np.full((H, W), 255, np.uint8)).save(os.path.join(mask_dir, name + ".png"))
        write_depth_tiff(os.path.join(depth_dir, name + ".tiff"), dmm)

        frames.append({
            "file_path": "./rgb/%s.png" % name,
            "depth_file_path": "./depth/%s.tiff" % name,
            "mask_path": "./masks/%s.png" % name,
            "transform_matrix": c2w.tolist(),
        })
        kept += 1
    cam.close()

    meta = {
        "fl_x": float(calib.fx), "fl_y": float(calib.fy),
        "cx": float(calib.cx), "cy": float(calib.cy),
        "w": W, "h": H,
        "frames": frames,
    }
    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=2)
    vp = np.array(valid_pcts) if valid_pcts else np.array([0.0])
    print("wrote %d frames (%d VIO-OK) -> %s" % (kept, n_tracked, args.out))
    print("depth coverage: mean %.1f%% valid (min %.1f%%, max %.1f%%)  "
          "[lower confidence -> lower %% but fewer ghosts]"
          % (vp.mean(), vp.min(), vp.max()))


if __name__ == "__main__":
    main()
