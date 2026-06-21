"""dynamic_viz.py — viser-direct live viewer (server-side rasterize + push-image).

The ONE sanctioned live-viz surface (Invariant #9 — never the NS viewer). An event-driven
daemon render thread reads each connected client's camera, renders TWO views (the camera-pose
view + a bird's-eye view), composites them with the real camera feed into one 2x2 canvas, and
pushes the JPEG as the client background. The bridge holds NO model and NO lock — render_fn
carries both. Reuses the proven viser<->OpenGL camera-convention helpers from
dynamic_gs.utils.viser_direct. (rewrite_spec/dynamic_viz.md.)

Composite layout (canvas is 2*render_w wide, 2*render_h tall):
  bottom-left  = render 1   (source picked live: Cam / Top / Left / Right / Manual)
  bottom-right = the real camera feed
  top-centre   = render 2   (same source choice; shown only in "2 render" mode)
  top-left HUD = live tracker Hz + feedforward Hz + each render's current source
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    import viser
    from dynamic_gs.utils.viser_direct import _build_camera_from_viser, _FLIP_YZ, _rotmat_to_quat_wxyz_np
    _HAVE_VISER = True
except Exception:                                  # pragma: no cover
    _HAVE_VISER = False


def _axis_angle_to_R(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues: 3x3 rotation of `theta` radians about (unit-normalised) `axis`."""
    axis = np.asarray(axis, np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _secondary_view_c2w(c2w_3x4: np.ndarray, pivot_dist: float, direction: str, deg: float,
                        world_up) -> np.ndarray:
    """Orbit-view c2w from a main-view (nerfstudio OpenGL) c2w, looking at the target `pivot_dist`
    metres ahead. 'Top' pitches `deg` about the camera's own right axis (bird's-eye). 'Left'/'Right'
    are LEVEL side views: the camera orbits `deg` about `world_up` at the target's height and looks
    horizontally, so the horizon stays flat regardless of the capture camera's forward tilt."""
    R = np.asarray(c2w_3x4[:3, :3], np.float64)
    p = np.asarray(c2w_3x4[:3, 3], np.float64)
    right, fwd = R[:, 0], -R[:, 2]                  # OpenGL camera looks down its local -Z
    pivot = p + fwd * pivot_dist
    out = np.eye(4, dtype=np.float32)
    if direction == "Top":                          # bird's-eye: pitch about the camera's right axis
        Rr = _axis_angle_to_R(right, np.deg2rad(-deg))
        out[:3, :3] = (Rr @ R).astype(np.float32)
        out[:3, 3] = (pivot + Rr @ (p - pivot)).astype(np.float32)
        return out[:3, :4]
    # Level side view: orbit the azimuth about world-up, then lookAt the target with up = world_up.
    U = np.asarray(world_up, np.float64)
    U = U / (np.linalg.norm(U) + 1e-12)
    fwd_h = fwd - np.dot(fwd, U) * U                 # horizontal component of the view direction
    nrm = np.linalg.norm(fwd_h)
    fwd_h = fwd_h / nrm if nrm > 1e-6 else right     # degenerate (looking straight up/down): fall back
    theta = np.deg2rad(-deg if direction == "Left" else deg)   # Left -> the camera's own left
    new_fwd = _axis_angle_to_R(U, theta) @ fwd_h
    eye = pivot - new_fwd * pivot_dist               # new_fwd is horizontal -> eye at the target height
    back = -new_fwd                                  # OpenGL c2w columns = right, up, back
    x = np.cross(U, back); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(back, x)
    out[:3, 0], out[:3, 1], out[:3, 2] = x.astype(np.float32), y.astype(np.float32), back.astype(np.float32)
    out[:3, 3] = eye.astype(np.float32)
    return out[:3, :4]


def _cam_with_c2w(template, c2w_3x4_np: np.ndarray, device):
    """A nerfstudio Cameras with `template`'s intrinsics but a swapped camera_to_worlds."""
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    c2w = torch.from_numpy(np.ascontiguousarray(c2w_3x4_np, dtype=np.float32)).to(device).unsqueeze(0)
    return Cameras(camera_to_worlds=c2w, fx=template.fx, fy=template.fy, cx=template.cx,
                   cy=template.cy, width=template.width, height=template.height,
                   camera_type=CameraType.PERSPECTIVE)


class ViserBridge:
    """Spin up once per dynamic run. attach(render_fn) then request_render() per mutation."""

    # Each render panel picks its source live (button group): "Cam" = the tracked capture pose,
    # "Top/Left/Right" = an orbit view of the tracked pose (+ angle slider), "Manual" = the viser
    # orbit camera you drag. Orbits look at the target TOP_VIEW_PIVOT_M metres ahead.
    ORBIT_DIRS = ("Top", "Left", "Right")
    RENDER_SRCS = ("Cam",) + ORBIT_DIRS + ("Manual",)
    ORBIT_DEFAULT_DEG = {"Top": 45.0, "Left": 90.0, "Right": 90.0}
    TOP_VIEW_PIVOT_M = 0.5                # look-target distance ahead of the camera (≈ object dist)
    WORLD_UP = (0.0, 0.0, 1.0)           # scene gravity axis (derived from the capture poses):
                                         # Left/Right side views are levelled to this so the
                                         # horizon stays flat despite the capture camera's down-tilt.

    # How many views the render thread draws (live-selectable). Fewer renders = higher tracker Hz.
    VIEW_MODES = ("Camera", "1 render", "2 render")
    DEFAULT_VIEW_MODE = "1 render"

    def __init__(self, cfg, *, device="cuda", render_size=(1280, 800), jpeg_quality: int = 88):
        self.enabled = bool(getattr(cfg, "enabled", True)) and _HAVE_VISER
        self.port = int(getattr(cfg, "port", 8081))
        self.device = device
        self.render_w, self.render_h = render_size
        self.jpeg_quality = jpeg_quality
        self._render_fn: Optional[Callable] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._closing = False
        self._initial_c2w: Optional[np.ndarray] = None
        self._follow_c2w: Optional[np.ndarray] = None
        self._feed_rgb: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._applied = set()
        self._trk_dt_ema: Optional[float] = None    # EMA of the tracker tick interval -> HUD Hz
        self._trk_last_t: Optional[float] = None
        self._ff_dt_ema: Optional[float] = None      # EMA of the FF fire interval -> HUD Hz
        self._ff_last_t: Optional[float] = None
        self._ff_active = False                      # True once the first FF fire is seen
        if not self.enabled:
            if not _HAVE_VISER:
                print("[viser] viser unavailable — viewer disabled (headless).")
            return
        self._server = viser.ViserServer(port=self.port)
        self._build_gui()
        self._server.on_client_connect(self._on_connect)

    # ---- GUI ----
    def _build_gui(self):
        with self._server.gui.add_folder("Tracker view"):
            # How many views to render: "Camera" = feed only (ZERO renders — fastest, no render-lock
            # contention), "1 render" = render 1 + feed, "2 render" = render 1 + render 2 + feed.
            self._gui_view_mode = self._server.gui.add_button_group("Views", self.VIEW_MODES)
            self._gui_view_mode.value = self.DEFAULT_VIEW_MODE

            # Each render's source + (for orbit views) its angle. Render 1 = bottom-left,
            # render 2 = top-centre. Defaults: R1 tracks the camera, R2 is the bird's-eye.
            self._gui_r1_src = self._server.gui.add_button_group("Render 1", self.RENDER_SRCS)
            self._gui_r1_src.value = "Top"          # default: a single bird's-eye render
            self._gui_r1_deg = self._server.gui.add_slider("R1 angle", min=0, max=180, step=5,
                                                           initial_value=self.ORBIT_DEFAULT_DEG["Top"])
            self._gui_r2_src = self._server.gui.add_button_group("Render 2", self.RENDER_SRCS)
            self._gui_r2_src.value = "Top"
            self._gui_r2_deg = self._server.gui.add_slider("R2 angle", min=0, max=180, step=5,
                                                           initial_value=self.ORBIT_DEFAULT_DEG["Top"])

            @self._gui_view_mode.on_click
            def _(_):
                self._sync_gui_visibility()
                self._wake.set()

            @self._gui_r1_src.on_click
            def _(_):
                self._snap_angle(self._gui_r1_src, self._gui_r1_deg)
                self._sync_gui_visibility()
                self._wake.set()

            @self._gui_r2_src.on_click
            def _(_):
                self._snap_angle(self._gui_r2_src, self._gui_r2_deg)
                self._sync_gui_visibility()
                self._wake.set()

            @self._gui_r1_deg.on_update
            def _(_):
                self._wake.set()

            @self._gui_r2_deg.on_update
            def _(_):
                self._wake.set()

            self._sync_gui_visibility()

    def _snap_angle(self, src_handle, deg_handle) -> None:
        """When a render switches to an orbit direction, jump its slider to that direction's default."""
        if src_handle.value in self.ORBIT_DIRS:
            deg_handle.value = self.ORBIT_DEFAULT_DEG[src_handle.value]

    def _sync_gui_visibility(self) -> None:
        """Show only the controls that matter: render N's controls iff that render is active, and its
        angle slider iff its source is an orbit direction."""
        mode = self._gui_view_mode.value
        r1_on = mode in ("1 render", "2 render")
        r2_on = mode == "2 render"
        self._gui_r1_src.visible = r1_on
        self._gui_r1_deg.visible = r1_on and self._gui_r1_src.value in self.ORBIT_DIRS
        self._gui_r2_src.visible = r2_on
        self._gui_r2_deg.visible = r2_on and self._gui_r2_src.value in self.ORBIT_DIRS

    # ---- lifecycle ----
    def attach(self, render_fn: Callable) -> None:
        self._render_fn = render_fn
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._render_loop, name="viser-direct-render", daemon=True)
        self._thread.start()
        print(f"[viser] viewer at http://localhost:{self.port}  (orbit to inspect the tracked scene)")

    def set_initial_camera(self, c2w_4x4: np.ndarray) -> None:
        self._initial_c2w = np.asarray(c2w_4x4, float)

    def update_tracked_camera(self, c2w_4x4: np.ndarray) -> None:
        # Called once per tracker tick -> its call rate IS the live tracker Hz (HUD readout).
        t = time.perf_counter()
        with self._lock:
            self._follow_c2w = np.asarray(c2w_4x4, float)
            if self._trk_last_t is not None:
                self._trk_dt_ema = self._ema(self._trk_dt_ema, t - self._trk_last_t)
            self._trk_last_t = t

    def note_ff_tick(self) -> None:
        """Pipeline calls this each time a feedforward dispatch actually fires -> FF Hz (HUD)."""
        t = time.perf_counter()
        with self._lock:
            self._ff_active = True
            if self._ff_last_t is not None:
                self._ff_dt_ema = self._ema(self._ff_dt_ema, t - self._ff_last_t)
            self._ff_last_t = t

    @staticmethod
    def _ema(prev: Optional[float], sample: float, a: float = 0.2) -> float:
        return sample if prev is None else (1.0 - a) * prev + a * sample

    def _hud_lines(self) -> list:
        """Caller must hold self._lock (reads the EMA state)."""
        trk = f"{1.0 / self._trk_dt_ema:4.1f} Hz" if self._trk_dt_ema else "-- Hz"
        if not self._ff_active:
            ff = "off"
        else:
            ff = f"{1.0 / self._ff_dt_ema:4.1f} Hz" if self._ff_dt_ema else "-- Hz"
        return [f"track {trk}", f"FF    {ff}"]

    def update_camera_feed(self, rgb_or_bgr: np.ndarray, is_bgr: bool = True) -> None:
        if not self.enabled or not self._server.get_clients():
            return
        img = rgb_or_bgr[..., ::-1] if is_bgr else rgb_or_bgr
        with self._lock:
            self._feed_rgb = np.ascontiguousarray(img)

    @property
    def is_closing(self) -> bool:
        return self._closing

    def has_clients(self) -> bool:
        return self.enabled and bool(self._server.get_clients())

    def request_render(self, *_a, **_k) -> None:
        if self.enabled and not self._closing:
            self._wake.set()

    # ---- client connect + initial camera ----
    def _on_connect(self, client):
        try:
            ph = np.zeros((self.render_h, self.render_w, 3), np.uint8)
            ph[:] = (np.array([0.86, 0.92, 1.0]) * 255).astype(np.uint8)  # Gazebo sky bg placeholder
            client.scene.set_background_image(ph, format="jpeg")
        except Exception:
            pass
        self._apply_initial(client)
        self._wake.set()

    def _apply_initial(self, client):
        if self._initial_c2w is None or client.client_id in self._applied:
            return
        try:
            R_nerf = self._initial_c2w[:3, :3]
            R_viser = (R_nerf @ _FLIP_YZ).astype(np.float32)
            client.camera.wxyz = _rotmat_to_quat_wxyz_np(R_viser)
            client.camera.position = self._initial_c2w[:3, 3].astype(np.float32)
            self._applied.add(client.client_id)
        except Exception:
            pass

    # ---- render thread ----
    def _render_to_np(self, cam) -> np.ndarray:
        rgb = self._render_fn(cam)                          # locked render (pipeline owns lock)
        return (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    def _compose(self, mode: str, render_cam: Optional[np.ndarray], render_top: Optional[np.ndarray],
                 feed_rgb: Optional[np.ndarray], hud: list) -> np.ndarray:
        """Assemble the 2x2 canvas. "Camera": feed fills it. "1 render": [render | feed] across the
        bottom. "2 render": + the 2nd view top-centre. Canvas footprint is constant across modes."""
        import cv2
        W, H = self.render_w, self.render_h
        canvas = np.empty((2 * H, 2 * W, 3), np.uint8)
        canvas[:] = (np.array([0.86, 0.92, 1.0]) * 255).astype(np.uint8)   # Gazebo sky bg
        if mode == "Camera":
            if feed_rgb is not None:                                        # feed fills the whole canvas
                canvas[:] = cv2.resize(feed_rgb, (2 * W, 2 * H))
        else:
            if render_cam is not None:                                      # bottom-left: camera-pose render
                canvas[H:2 * H, 0:W] = render_cam
            if feed_rgb is not None:                                        # bottom-right: real camera feed
                canvas[H:2 * H, W:2 * W] = cv2.resize(feed_rgb, (W, H))
            if render_top is not None:                                      # top-centre: 2nd view (2-render only)
                x0 = W // 2
                canvas[0:H, x0:x0 + W] = render_top
        for i, line in enumerate(hud):                                      # HUD in the top-left
            cv2.putText(canvas, line, (16, 42 + i * 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (20, 20, 20), 2, cv2.LINE_AA)
        return canvas

    def _src_label(self, src: str, deg: float) -> str:
        return f"{src} {deg:.0f}deg" if src in self.ORBIT_DIRS else src

    def _render_for(self, src: str, deg: float, template_cam, tracked) -> np.ndarray:
        """Render one panel from its chosen source -> uint8 RGB. 'Manual' uses the viser orbit camera;
        'Cam'/'Top'/'Left'/'Right' are built off the tracked capture pose (orbit camera untouched)."""
        if src == "Manual":
            return self._render_to_np(template_cam)
        base = tracked if tracked is not None else template_cam.camera_to_worlds[0].detach().cpu().numpy()
        if src == "Cam":
            c2w = np.asarray(base, np.float32)[:3, :4]
        else:                                          # Top / Left / Right orbit of the tracked pose
            c2w = _secondary_view_c2w(base, self.TOP_VIEW_PIVOT_M, src, deg, self.WORLD_UP)
        return self._render_to_np(_cam_with_c2w(template_cam, c2w, self.device))

    def _render_loop(self):
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            self._wake.clear()
            if self._stop.is_set():
                break
            clients = self._server.get_clients()
            if not clients:
                continue
            with self._lock:
                feed = self._feed_rgb
                tracked = self._follow_c2w                      # latest tracked capture pose (or None)
                hud = self._hud_lines()
            mode = self._gui_view_mode.value                   # "Camera" | "1 render" | "2 render"
            r1_src, r1_deg = self._gui_r1_src.value, float(self._gui_r1_deg.value)
            r2_src, r2_deg = self._gui_r2_src.value, float(self._gui_r2_deg.value)
            hud = hud + [f"view: {mode}"]
            if mode != "Camera":
                hud = hud + [f"R1: {self._src_label(r1_src, r1_deg)}"]
            if mode == "2 render":
                hud = hud + [f"R2: {self._src_label(r2_src, r2_deg)}"]
            for cid, client in clients.items():
                try:
                    self._apply_initial(client)
                    # "Camera" -> push the feed only: NO render_fn call at all (compute saved).
                    if mode == "Camera":
                        client.scene.set_background_image(
                            self._compose(mode, None, None, feed, hud),
                            format="jpeg", jpeg_quality=self.jpeg_quality)
                        continue
                    template = _build_camera_from_viser(client.camera, self.render_w, self.render_h, self.device)
                    rgb1 = self._render_for(r1_src, r1_deg, template, tracked)
                    rgb2 = self._render_for(r2_src, r2_deg, template, tracked) if mode == "2 render" else None
                    canvas = self._compose(mode, rgb1, rgb2, feed, hud)
                    client.scene.set_background_image(canvas, format="jpeg", jpeg_quality=self.jpeg_quality)
                except Exception as exc:
                    print(f"[viser] render failed for client {cid}: {exc}", flush=True)

    def close(self):
        self._closing = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
