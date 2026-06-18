"""dynamic_viz.py — viser-direct live viewer (server-side rasterize + push-image).

The ONE sanctioned live-viz surface (Invariant #9 — never the NS viewer). An event-driven
daemon render thread reads each connected client's camera, calls the injected render_fn
(locked snapshot render owned by the pipeline) and pushes the JPEG as the client background.
The bridge holds NO model and NO lock — render_fn carries both. Reuses the proven
viser<->OpenGL camera-convention helpers from dynamic_gs.utils.viser_direct.
(rewrite_spec/dynamic_viz.md.)
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


class ViserBridge:
    """Spin up once per dynamic run. attach(render_fn) then request_render() per mutation."""

    def __init__(self, cfg, *, device="cuda", render_size=(960, 600), jpeg_quality: int = 88):
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
        self._follow = False
        self._feed_rgb: Optional[np.ndarray] = None
        self._feed_gui = None
        self._lock = threading.Lock()
        self._applied = set()
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
            self._gui_follow = self._server.gui.add_checkbox("Follow tracked frame", False)
            self._gui_feed = self._server.gui.add_checkbox("Show camera feed", True)

            @self._gui_follow.on_update
            def _(_):
                self._follow = bool(self._gui_follow.value)

    # ---- lifecycle ----
    def attach(self, render_fn: Callable, snapshot_fn: Optional[Callable] = None) -> None:
        self._render_fn = render_fn
        self._snapshot_fn = snapshot_fn
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._render_loop, name="viser-direct-render", daemon=True)
        self._thread.start()
        print(f"[viser] viewer at http://localhost:{self.port}  (orbit to inspect the tracked scene)")

    def set_initial_camera(self, c2w_4x4: np.ndarray) -> None:
        self._initial_c2w = np.asarray(c2w_4x4, float)

    def update_tracked_camera(self, c2w_4x4: np.ndarray) -> None:
        with self._lock:
            self._follow_c2w = np.asarray(c2w_4x4, float)

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
            ph[:] = (np.array([1.0, 0.92, 0.86]) * 255).astype(np.uint8)  # bg-ish placeholder
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
    def _render_loop(self):
        import cv2
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            self._wake.clear()
            if self._stop.is_set():
                break
            clients = self._server.get_clients()
            if not clients:
                continue
            # feed thumbnail (encode once, render thread)
            with self._lock:
                feed = self._feed_rgb
                follow = self._follow_c2w if self._follow else None
            for cid, client in clients.items():
                try:
                    self._apply_initial(client)
                    if follow is not None:
                        R_viser = (follow[:3, :3] @ _FLIP_YZ).astype(np.float32)
                        client.camera.wxyz = _rotmat_to_quat_wxyz_np(R_viser)
                        client.camera.position = follow[:3, 3].astype(np.float32)
                    cam = _build_camera_from_viser(client.camera, self.render_w, self.render_h, self.device)
                    rgb = self._render_fn(cam)                         # locked render (pipeline owns lock)
                    rgb_np = (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
                    client.scene.set_background_image(rgb_np, format="jpeg", jpeg_quality=self.jpeg_quality)
                    if feed is not None and self._gui_feed.value:
                        if self._feed_gui is None:
                            self._feed_gui = self._server.gui.add_image(feed, label="camera feed")
                        else:
                            self._feed_gui.image = feed
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
