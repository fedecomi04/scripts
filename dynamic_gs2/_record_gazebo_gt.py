#!/usr/bin/env python
"""Record one Gazebo model's true pose, at the camera frame timestamps, to a CSV.

Runs alongside a live pipeline run. The pipeline already writes the tracker's estimated object
pose to <data_dir>/object_track_poses.jsonl (one record per tick, carrying t_sec); this logs the
matching ground truth so the two can be compared by _eval_pose_rmse.py.

WHY IT IS DRIVEN BY THE CAMERA, NOT BY A TIMER
  The tracker ticks on camera frames and logs each pose against that frame's header stamp. A GT
  log sampled on its own schedule would never land on those instants, and interpolating across
  the gap invents error that is not tracking error -- at 10 Hz that fake term is millimetres, the
  same order as the quantity being measured.
  /gazebo/model_states is free-running and carries no header stamp, so it cannot be made to fire
  at a camera stamp. Instead this keeps a short buffer of model_states samples and writes ONE row
  per camera frame, interpolated to that frame's OWN header stamp -- taken from camera_info,
  which the camera plugin publishes with the same stamp as the image and which costs nothing to
  subscribe to (no image payload). Every row therefore carries a timestamp the tracker also uses,
  so the eval joins on equal values, and the residual interpolation term is bounded by the buffer
  spacing (~5 ms => micrometres) instead of by the output rate.

Columns: t, x, y, z, qx, qy, qz, qw -- also the format _eval_tracking_error.py's --gt reads.

Runs in the py3.8 `dynamic_gs_ros` env (see record_gt.sh). Ctrl-C to stop; the file is flushed as
it goes, so an interrupted run still leaves a usable log.
"""
from __future__ import print_function

import argparse
import bisect
import math
import os
import time

import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import CameraInfo

# Must match the camera the pipeline consumes (ros_mask.py owns the canonical topic names).
DEFAULT_CAMERA_INFO_TOPIC = "/dynaarm_arm/dynaarm_arm/camera1/camera_info"

# Models that are never the manipulated object, so auto-pick can skip them.
_IGNORE_EXACT = ("ground_plane", "sun")
_IGNORE_SUBSTR = ("dynaarm", "robot")

_BUFFER_MAX = 512          # samples kept for interpolation (~2.5 s at 200 Hz)


def pick_model(names):
    """The single plausible object model, or None if it is ambiguous."""
    cand = [n for n in names
            if n not in _IGNORE_EXACT
            and not any(s in n.lower() for s in _IGNORE_SUBSTR)]
    return cand[0] if len(cand) == 1 else None


def qlerp(q0, q1, a):
    """Normalised quaternion lerp along the shortest arc. Over the few-ms gaps this interpolates,
    it is indistinguishable from slerp far below the precision of the measurement."""
    d = sum(x * y for x, y in zip(q0, q1))
    if d < 0.0:                                   # shortest path
        q1 = tuple(-x for x in q1)
    q = tuple(q0[i] + a * (q1[i] - q0[i]) for i in range(4))
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return tuple(x / n for x in q)


class GtRecorder(object):
    def __init__(self, out_path, model, buffer_hz, camera_info_topic):
        self.out_path = out_path
        self.model = model
        self.min_dt = (1.0 / buffer_hz) if buffer_hz > 0 else 0.0
        self.buf_t, self.buf_p, self.buf_q = [], [], []
        self.n_written = 0
        self.n_no_cover = 0
        self.last_buf_t = -1e9
        self.last_report = 0.0
        self.warned_missing = False
        self.fh = open(out_path, "w")
        self.fh.write("# t,x,y,z,qx,qy,qz,qw  (gazebo model '%s', at camera frame stamps)\n"
                      % (model or "auto"))
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_states, queue_size=2)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.on_camera, queue_size=5)

    # ---- buffer the free-running truth ----
    def on_states(self, msg):
        now = rospy.Time.now().to_sec()
        if now <= 0.0 or now - self.last_buf_t < self.min_dt:
            return                                # /clock not up yet, or buffer already dense
        if self.model is None:
            self.model = pick_model(msg.name)
            if self.model is None:
                if not self.warned_missing:
                    rospy.logwarn("[gt] cannot auto-pick a model; pass --model. Available: %s",
                                  ", ".join(msg.name))
                    self.warned_missing = True
                return
            rospy.loginfo("[gt] auto-picked model '%s'", self.model)
        try:
            i = msg.name.index(self.model)
        except ValueError:
            if not self.warned_missing:
                rospy.logwarn("[gt] model '%s' not in /gazebo/model_states. Available: %s",
                              self.model, ", ".join(msg.name))
                self.warned_missing = True
            return
        self.last_buf_t = now
        p, q = msg.pose[i].position, msg.pose[i].orientation
        self.buf_t.append(now)
        self.buf_p.append((p.x, p.y, p.z))
        self.buf_q.append((q.x, q.y, q.z, q.w))
        if len(self.buf_t) > _BUFFER_MAX:
            del self.buf_t[:-_BUFFER_MAX], self.buf_p[:-_BUFFER_MAX], self.buf_q[:-_BUFFER_MAX]

    # ---- emit exactly one row per camera frame, at that frame's own stamp ----
    def on_camera(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0 or len(self.buf_t) < 2:
            return
        if t < self.buf_t[0] or t > self.buf_t[-1]:
            # The frame stamp is outside the buffered truth (startup, or a stalled topic). Writing
            # an extrapolated row would be exactly the invented error this design avoids, so skip.
            self.n_no_cover += 1
            if self.n_no_cover % 200 == 1:
                rospy.logwarn_throttle(5.0, "[gt] %d camera frames outside the truth buffer "
                                             "(skipped; is /gazebo/model_states alive?)",
                                       self.n_no_cover)
            return
        i = bisect.bisect_left(self.buf_t, t)
        if i == 0:
            p, q = self.buf_p[0], self.buf_q[0]
        else:
            t0, t1 = self.buf_t[i - 1], self.buf_t[i]
            a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            p0, p1 = self.buf_p[i - 1], self.buf_p[i]
            p = tuple(p0[k] + a * (p1[k] - p0[k]) for k in range(3))
            q = qlerp(self.buf_q[i - 1], self.buf_q[i], a)
        self.fh.write("%.6f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f\n"
                      % (t, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
        self.n_written += 1
        if self.n_written % 20 == 0:
            self.fh.flush()
        wall = time.time()
        if wall - self.last_report > 5.0:
            self.last_report = wall
            rospy.loginfo("[gt] %d rows at camera stamps, t=%.2f, pos=(%.3f %.3f %.3f)",
                          self.n_written, t, p[0], p[1], p[2])

    def close(self):
        try:
            self.fh.flush()
            self.fh.close()
        except Exception:
            pass
        rospy.loginfo("[gt] wrote %d rows -> %s (%d frames skipped for lack of buffered truth)",
                      self.n_written, self.out_path, self.n_no_cover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--model", default=None,
                    help="gazebo model name (default: auto-pick the one non-robot model)")
    ap.add_argument("--buffer-hz", type=float, default=200.0,
                    help="cap on buffered truth samples/sec (default 200; output rate is the "
                         "camera frame rate regardless)")
    ap.add_argument("--camera-info", default=DEFAULT_CAMERA_INFO_TOPIC,
                    help="camera_info topic whose header stamps drive the output rows")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    rospy.init_node("dgs_gazebo_gt_recorder", anonymous=True, disable_signals=False)
    rec = GtRecorder(args.out, args.model, args.buffer_hz, args.camera_info)
    rospy.on_shutdown(rec.close)
    rospy.loginfo("[gt] recording '%s' at the stamps of %s -> %s (Ctrl-C to stop)",
                  args.model or "auto", args.camera_info, args.out)
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
