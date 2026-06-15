#!/usr/bin/env python
"""Record a fixed-length SVO from the live ZED. Run on the Jetson with the ZED venv python:

    /home/shengzhiwang/zed_env/bin/python zed_record.py \
        --out /home/shengzhiwang/scene.svo2 --res HD1200 --fps 15 --seconds 20 \
        --settings /home/shengzhiwang/zed_settings

Records raw stereo (depth_mode=NONE -> light, full-rate); depth + VIO pose are
recomputed later on replay by zed_svo_to_dataset.py. Exact frame count, clean stop.
"""
import argparse, os, time
import pyzed.sl as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="HD1200",
                    choices=["HD1200", "HD1080", "SVGA", "VGA", "HD720"])
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--settings", default=None, help="calib dir (optional_settings_path)")
    ap.add_argument("--compression", default="H264", choices=["H264", "H265", "LOSSLESS"])
    ap.add_argument("--delay", type=float, default=5.0,
                    help="countdown seconds before recording starts (camera warms up)")
    ap.add_argument("--exposure", type=int, default=-1,
                    help="0-100 (percent of frame-time exposure); -1=auto. LOWER = less motion blur")
    ap.add_argument("--gain", type=int, default=-1,
                    help="0-100 sensor gain; -1=auto. Raise to brighten a low --exposure")
    args = ap.parse_args()

    cam = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = getattr(sl.RESOLUTION, args.res)
    ip.camera_fps = args.fps
    ip.depth_mode = sl.DEPTH_MODE.NONE          # raw stereo only; depth recomputed on replay
    ip.coordinate_units = sl.UNIT.METER
    if args.settings:
        ip.optional_settings_path = os.path.join(args.settings, "")
    st = cam.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("ZED open failed: %s" % st)

    info = cam.get_camera_information()
    W = info.camera_configuration.resolution.width
    H = info.camera_configuration.resolution.height
    print("camera: %dx%d, requested %d fps (reports %s) | %s compression"
          % (W, H, args.fps, info.camera_configuration.fps, args.compression))

    if args.exposure >= 0 or args.gain >= 0:
        cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)   # manual exposure/gain
        if args.exposure >= 0:
            cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, args.exposure)
        if args.gain >= 0:
            cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, args.gain)
        print("manual exposure=%s gain=%s (auto AEC/AGC OFF) -- lower exposure = less blur"
              % (args.exposure, args.gain))

    rt = sl.RuntimeParameters()
    if args.delay > 0:
        print("Get ready -- recording starts in %.0f s (camera warming up)..." % args.delay)
        t_end = time.time() + args.delay
        last = None
        while time.time() < t_end:
            cam.grab(rt)  # keep streaming so auto-exposure settles
            rem = int(t_end - time.time()) + 1
            if rem != last:
                print("  %d..." % rem)
                last = rem

    rp = sl.RecordingParameters(args.out, getattr(sl.SVO_COMPRESSION_MODE, args.compression))
    if cam.enable_recording(rp) != sl.ERROR_CODE.SUCCESS:
        cam.close()
        raise SystemExit("enable_recording failed (NVENC/codec?)")

    n = int(round(args.seconds * args.fps))
    print(">>> RECORDING %d frames (~%.0fs). GO!" % (n, args.seconds))
    t0 = time.time()
    done = 0
    for i in range(n):
        if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
            done += 1
    dt = time.time() - t0
    cam.disable_recording()
    cam.close()
    print("recorded %d/%d frames in %.1fs (%.1f fps actual) -> %s"
          % (done, n, dt, done / dt if dt > 0 else 0.0, args.out))


if __name__ == "__main__":
    main()
