"""Viser-direct visualization — server-side rasterize + push-image pattern.

History (kept for the reader who comes back to this file):
The prior implementation used ``viser.GaussianSplatHandle`` per-tick
``wxyz``/``position`` writes (and a separate handle per feedforward
insert). Object motion was visible but every property write triggered a
brief whole-scene flash because viser's GaussianSplatHandle is
officially flagged "Work-in-progress" across every released version
(0.2.7 through 1.0.29 as of 2026-05-31) — its WebGL splat renderer
remounts the canvas on each property write. A version bump to 1.0.29
made the browser tab crash entirely instead of flashing. See
``memory/project_viser_path_a_status.md`` for the full post-mortem.

This implementation follows the canonical ecosystem pattern (nerfview,
GaussianEditor, hwanhuh/2D-GS-Viser-Viewer, leggedrobotics/DiskChunGS):

  1. NO native splat handles in the browser. Empty scene.
  2. A background render thread polls each connected client's camera
     pose, calls ``model.get_outputs(camera)`` server-side, and pushes
     the resulting RGB image to that client via
     ``client.scene.set_background_image(...)``. One atomic full-frame
     replacement per push — no in-between scene-rebuild state, no flash.
  3. Browser keeps full 6DoF camera control. We just read
     ``client.camera.position/.wxyz/.fov`` per render tick.
  4. A ``model_lock`` (acquired by the pipeline around any tracker write
     and by the render thread around each ``get_outputs`` call) prevents
     mid-render races on ``model.means`` etc.

Trade-off vs the deleted Path A: the training GPU now also serves
viewer frames (~25 ms / get_outputs at 512×512). For N connected
clients with different camera poses we do N renders per tick.

API surface kept stable so legacy pipeline call sites stay no-op
compatible: ``setup_handles``, ``push_tracker_transform``,
``add_ff_insert_chunk``, ``refresh_static_handle``,
``maybe_flush_ff_handle``, ``flush_pending_ff`` are all retained as
thin stubs (most do nothing; ``setup_handles`` just records the initial
camera pose).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import torch

try:
    import viser  # type: ignore
    _HAS_VISER = True
except Exception:  # pragma: no cover
    viser = None  # type: ignore
    _HAS_VISER = False


# ---------------------------------------------------------------------------
# Quaternion -> rotation matrix (wxyz)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """(4,) wxyz numpy quaternion -> (3, 3) numpy rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return R


def _rotmat_to_quat_wxyz_np(R: np.ndarray) -> np.ndarray:
    """(3, 3) numpy rotation matrix -> (4,) wxyz numpy quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


# Viser cameras are OpenCV-convention at the local camera frame (Y-down,
# Z-forward), even though viser's world frame is +Y up. To convert from
# viser's R_world_camera to nerfstudio's c2w[:3,:3] (OpenGL: Y-up,
# Z-back at the camera), multiply by a 180-degree rotation about X.
# This is exactly what nerfstudio's own viewer does in
# ``nerfstudio.viewer.viewer.Viewer.get_camera_state``.
_FLIP_YZ = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Build a nerfstudio Cameras object from a viser ClientHandle camera state
# ---------------------------------------------------------------------------

def _build_camera_from_viser(client_camera, W: int, H: int, device) -> "Cameras":
    """Convert (position, wxyz, fov) from viser into a nerfstudio Cameras.

    Viser cameras are OpenGL convention: +X right, +Y up, -Z forward, with
    ``fov`` being the vertical field of view in radians. Nerfstudio
    ``Cameras`` uses the same OpenGL convention for ``camera_to_worlds``
    (c2w) when built directly, so the rotation matrix from the quaternion
    drops in without an extra coordinate flip.
    """
    from nerfstudio.cameras.cameras import Cameras, CameraType

    pos = np.asarray(client_camera.position, dtype=np.float32).reshape(3)
    wxyz = np.asarray(client_camera.wxyz, dtype=np.float32).reshape(4)
    fov_v = float(getattr(client_camera, "fov", np.deg2rad(60.0)))
    R_viser = _quat_wxyz_to_rotmat_np(wxyz)
    # viser camera (Y-down, Z-forward) -> nerfstudio (Y-up, Z-back)
    R_nerf = (R_viser @ _FLIP_YZ).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_nerf
    c2w[:3, 3] = pos
    c2w_3x4 = torch.from_numpy(c2w[:3, :4]).to(device).unsqueeze(0)  # (1, 3, 4)

    # fy = 0.5 * H / tan(fov_v / 2); fx = fy * (W/H) / aspect — but viser's
    # aspect is exactly W/H so the (W/H)/aspect factor cancels: fx == fy in
    # pixel units.
    fy = 0.5 * H / float(np.tan(fov_v / 2.0))
    fx = fy
    cx = 0.5 * W
    cy = 0.5 * H

    cameras = Cameras(
        camera_to_worlds=c2w_3x4,
        fx=torch.tensor([[fx]], device=device, dtype=torch.float32),
        fy=torch.tensor([[fy]], device=device, dtype=torch.float32),
        cx=torch.tensor([[cx]], device=device, dtype=torch.float32),
        cy=torch.tensor([[cy]], device=device, dtype=torch.float32),
        width=torch.tensor([[W]], device=device, dtype=torch.int32),
        height=torch.tensor([[H]], device=device, dtype=torch.int32),
        camera_type=CameraType.PERSPECTIVE,
    )
    return cameras


class ViserDirectScene:
    """Server-side rasterize + push-image viser viewer.

    Spin up once. Call :meth:`attach_model` once the pipeline's model is
    loaded (post-warm-cache). The background render thread starts as
    soon as a model is attached and at least one client is connected.

    Pipeline contract:
      * Acquire :attr:`model_lock` (a re-entrant lock) around any code
        path that mutates ``model.means / .quats / .features_dc / ...``
        (tracker rigid transform, FF inserts, FF deletes). The render
        thread acquires the same lock before each ``get_outputs`` call.
      * No other API call is required for per-tick motion / FF updates
        to appear in the browser. The render loop pulls live model
        state each iteration; mutations are visible on the next push.
    """

    def __init__(
        self,
        port: int = 8081,
        render_hz: float = 15.0,
        render_size: tuple[int, int] = (1920, 1080),  # (W, H)
        jpeg_quality: int = 92,
        # Legacy kwargs kept so old call sites don't trip — ignored.
        opacity_floor: float = 0.05,
        static_refresh_min_gap_s: float = -1.0,
        push_min_gap_s: float = 0.033,
        ff_coalesce_gap_s: float = 1.0,
    ):
        if not _HAS_VISER:
            raise RuntimeError(
                "viser is not installed in the dynamic_gs env. "
                "pip install viser, or disable enable_viser_direct."
            )
        self.server = viser.ViserServer(port=port)
        self.port = port
        self.render_hz = float(render_hz)
        self.render_size = (int(render_size[0]), int(render_size[1]))
        self.jpeg_quality = int(jpeg_quality)

        # Pipeline coordination
        self.model_lock = threading.RLock()
        self._model = None
        self._device = None
        self._initial_c2w: Optional[np.ndarray] = None
        self._initial_look_at: Optional[np.ndarray] = None
        self._initial_fov_y: Optional[float] = None  # vertical FOV in radians

        # Render state
        self._stop_event = threading.Event()
        # Event-driven render: tracker (or any mutation site) calls
        # request_render() to set this. The render thread waits on it
        # rather than polling at fixed Hz, so every tracker tick =>
        # exactly one render. Also set when a client connects so the
        # first frame goes out without waiting for a tick.
        self._render_requested = threading.Event()
        self._render_thread: Optional[threading.Thread] = None
        # Per-client display state. We keep the ImageHandle for each
        # client so we update in place instead of allocating new handles
        # (atomic single-image replacement = no flash).
        self._client_state: dict[int, dict] = {}
        self._client_state_lock = threading.Lock()
        # Track which client_ids have already been snapped to the initial
        # camera pose. Each client gets the snap AT MOST ONCE so we never
        # override a user-controlled move once they've started navigating.
        self._initial_camera_applied: set[int] = set()
        # Diagnostics
        self._render_count = 0
        self._render_error_count = 0
        self._last_diag_t = time.time()
        self._render_window_total_ms = 0.0

        # --- Live camera feed (side-panel thumbnail) ---
        # The pipeline pushes the current tracked frame's RGB here each tick via
        # update_camera_feed(); the render loop refreshes the GUI image in place.
        self._feed_rgb: Optional[np.ndarray] = None       # (H, W, 3) uint8 RGB
        self._feed_dirty: bool = False
        self._feed_lock = threading.Lock()
        self._feed_gui_image = None                        # server.gui image handle
        # --- Follow-tracked-frame pose ---
        # When the GUI toggle is on, the render loop snaps each connected
        # client's camera to the tracked frame's c2w (set via
        # update_tracked_camera()) before rendering — so the splat view matches
        # the camera the tracker is seeing.
        self._follow_c2w: Optional[np.ndarray] = None      # (3,4) or (4,4) c2w
        self._follow_lock = threading.Lock()
        self._follow_toggle = None                         # server.gui checkbox handle
        self._feed_toggle = None                           # server.gui checkbox handle
        # --- End-of-run "keep alive" shutdown control ---
        # When a recorded run finishes, the pipeline can keep this server up so
        # the operator inspects the final scene, then clicks the "Shutdown
        # viewer" button (or the pipeline times out) to release the block.
        self._shutdown_requested = threading.Event()
        self._shutdown_button = None                       # server.gui button handle
        self._gui_built: bool = False

        # World axes are useful for orientation when the scene is empty
        # before the first render lands.
        try:
            self.server.scene.world_axes.visible = True
        except Exception:
            pass

        # Camera-feed + follow-pose GUI (shared across all clients).
        self._build_gui()

        # Wire client connect/disconnect.
        @self.server.on_client_connect
        def _on_connect(client):
            self._on_client_connect(client)

        @self.server.on_client_disconnect
        def _on_disconnect(client):
            self._on_client_disconnect(client)

        print(
            f"[viser-direct] server up on port {port} (render_hz={self.render_hz}, "
            f"render_size={self.render_size[0]}x{self.render_size[1]}) — "
            f"open http://localhost:{port}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Camera-feed + follow-pose GUI
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        """Add the shared GUI controls once: a 'Camera feed' folder with the
        live-feed thumbnail + a show/hide checkbox, and a 'Follow tracked
        frame' checkbox. All server-level so every client sees them."""
        if self._gui_built:
            return
        try:
            with self.server.gui.add_folder("Tracker view"):
                self._feed_toggle = self.server.gui.add_checkbox(
                    "Show camera feed", initial_value=True,
                )
                self._follow_toggle = self.server.gui.add_checkbox(
                    "Follow tracked frame", initial_value=False,
                )
            self._gui_built = True
        except Exception as exc:
            print(f"[viser-direct] GUI build failed: {exc}", flush=True)

    def keep_alive_until_shutdown(self, banner: str = "Run finished") -> None:
        """End-of-run hook: add a 'Shutdown viewer' button so the operator can
        inspect the final scene, then click to release the block. Idempotent.

        Pair with :meth:`wait_for_shutdown`. The render thread keeps running so
        the browser stays interactive while blocked."""
        if self._stop_event.is_set():
            return
        try:
            if self._shutdown_button is None:
                with self.server.gui.add_folder(banner):
                    self._shutdown_button = self.server.gui.add_button(
                        "Shutdown viewer", color="red",
                    )

                    @self._shutdown_button.on_click
                    def _on_shutdown(_event) -> None:
                        self._shutdown_requested.set()
        except Exception as exc:
            print(f"[viser-direct] shutdown-button add failed: {exc}", flush=True)
            # If the button can't be added, don't trap the process forever.
            self._shutdown_requested.set()

    def wait_for_shutdown(self, timeout_s: Optional[float] = None) -> bool:
        """Block until the 'Shutdown viewer' button is clicked (or timeout).
        Returns True if the button fired, False on timeout. Returns immediately
        if the server is already closing."""
        if self._stop_event.is_set():
            return True
        return self._shutdown_requested.wait(timeout=timeout_s)

    def update_camera_feed(self, rgb: np.ndarray) -> None:
        """Pipeline hook: stash the current tracked frame's RGB (H,W,3 uint8)
        to be shown as the side-panel feed thumbnail on the next render. Cheap
        (just a reference swap under a lock); the JPEG encode happens in the
        render loop only when a client is connected."""
        if rgb is None:
            return
        with self._feed_lock:
            self._feed_rgb = rgb
            self._feed_dirty = True

    def update_tracked_camera(self, camera_to_world: np.ndarray) -> None:
        """Pipeline hook: stash the tracked frame's camera c2w (3x4 or 4x4).
        When the 'Follow tracked frame' toggle is on, the render loop snaps the
        viewer camera to this pose before rendering."""
        if camera_to_world is None:
            return
        with self._follow_lock:
            self._follow_c2w = np.asarray(camera_to_world, dtype=np.float32)

    def _refresh_feed_image(self) -> None:
        """Render-thread side: update the GUI feed thumbnail in place if a new
        frame arrived and the toggle is on. Adds the image handle lazily."""
        toggle_on = self._feed_toggle is None or bool(self._feed_toggle.value)
        with self._feed_lock:
            rgb = self._feed_rgb if self._feed_dirty else None
            self._feed_dirty = False
        if not toggle_on:
            if self._feed_gui_image is not None:
                try:
                    self._feed_gui_image.visible = False
                except Exception:
                    pass
            return
        if rgb is None:
            return
        try:
            if self._feed_gui_image is None:
                self._feed_gui_image = self.server.gui.add_image(
                    rgb, label="camera feed", format="jpeg", jpeg_quality=70,
                )
            else:
                self._feed_gui_image.image = rgb
                self._feed_gui_image.visible = True
        except Exception as exc:
            if self._render_error_count <= 3:
                print(f"[viser-direct] feed image update failed: {exc}", flush=True)

    def _apply_follow_pose(self, client) -> None:
        """Render-thread side: if 'Follow tracked frame' is on, snap this
        client's camera to the latest tracked c2w.

        MUST use the same nerfstudio(Y-up,Z-back) -> viser(Y-down,Z-forward)
        conversion as :meth:`_apply_initial_camera`: ``R_viser = R_nerf @
        _FLIP_YZ`` (the inverse of the ``@ _FLIP_YZ`` applied on the read path
        in ``_build_camera_from_viser``). Omitting the flip feeds a wrong-handed
        rotation to viser and the followed view is misaligned (only the initial
        frame, which goes through the correct path, looked right)."""
        if self._follow_toggle is None or not bool(self._follow_toggle.value):
            return
        with self._follow_lock:
            c2w = self._follow_c2w
        if c2w is None:
            return
        try:
            c2w = np.asarray(c2w, dtype=np.float32)
            R_nerf = c2w[:3, :3]
            pos = c2w[:3, 3]
            R_viser = (R_nerf @ _FLIP_YZ).astype(np.float32)
            client.camera.position = pos
            client.camera.wxyz = _rotmat_to_quat_wxyz_np(R_viser)
        except Exception as exc:
            if self._render_error_count <= 3:
                print(f"[viser-direct] follow-pose snap failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Pipeline-facing API
    # ------------------------------------------------------------------

    def attach_model(self, model, device=None) -> None:
        """Hand the live model to the render thread + start it.

        Called by the pipeline once the warm cache is loaded. Safe to
        call multiple times (subsequent calls just swap the reference).
        """
        self._model = model
        if device is not None:
            self._device = device
        else:
            try:
                self._device = next(model.parameters()).device
            except Exception:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._render_thread is None or not self._render_thread.is_alive():
            self._stop_event.clear()
            self._render_thread = threading.Thread(
                target=self._render_loop,
                name="viser-direct-render",
                daemon=True,
            )
            self._render_thread.start()

    def set_initial_camera(
        self,
        c2w_4x4: np.ndarray,
        look_at: Optional[np.ndarray] = None,
        fov_y_rad: Optional[float] = None,
    ) -> None:
        """Set the camera pose that newly-connected clients land on AND
        snap any already-connected clients that haven't been snapped yet.

        ``c2w_4x4`` is the dataset c2w (Nerfstudio / OpenGL convention,
        4×4). ``look_at`` (optional) is a world-space point the camera
        should orient toward; defaults to ``cam_pos + camera_forward``
        (handled inside :meth:`_apply_initial_camera`).

        Each client gets at most ONE initial-camera snap per session,
        tracked by ``_initial_camera_applied``. This way a user who
        connected pre-D0 (and is currently looking at an arbitrary
        default viser pose) gets snapped to the dataset's first frame
        the moment D0 establishes :attr:`_initial_c2w`; but a user who
        connects, snaps, then orbits the scene with the mouse will
        NEVER get their navigation undone by a later D0 / re-attach.
        """
        c2w = np.asarray(c2w_4x4, dtype=np.float32)
        if c2w.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float32)
            tmp[:3, :4] = c2w
            c2w = tmp
        self._initial_c2w = c2w
        if look_at is not None:
            self._initial_look_at = np.asarray(look_at, dtype=np.float32).reshape(3)
        if fov_y_rad is not None:
            self._initial_fov_y = float(fov_y_rad)
        # Push the freshly-set camera to any clients that connected
        # BEFORE this call. They've never been snapped, so do it now.
        with self._client_state_lock:
            pending = [
                (cid, st["client"])
                for cid, st in self._client_state.items()
                if cid not in self._initial_camera_applied
            ]
        for cid, client in pending:
            self._apply_initial_camera(client)
            self._initial_camera_applied.add(cid)
        if pending:
            self.request_render()

    # ------------------------------------------------------------------
    # Legacy no-op stubs (kept so old pipeline call sites don't break)
    # ------------------------------------------------------------------

    def setup_handles(
        self,
        model,
        tracked_instance_id: Optional[int] = None,
        initial_c2w: Optional[np.ndarray] = None,
    ) -> None:
        """Legacy entry point. We no longer build any splat handles; we
        just attach the model + remember the initial camera pose."""
        self.attach_model(model)
        if initial_c2w is not None:
            self.set_initial_camera(initial_c2w)

    def push_tracker_transform(self, R, t) -> None:
        """Legacy stub. The render thread reads ``model.means`` directly
        each tick; per-tick rigid transforms become visible on the next
        push without any explicit call."""
        return

    def add_ff_insert_chunk(self, model, inserted_ids) -> None:
        """Legacy stub. New FF gaussians appear automatically in the
        next render (since the pipeline appends them to model params
        under :attr:`model_lock`)."""
        return

    def maybe_flush_ff_handle(self, model, force: bool = False) -> None:
        """Legacy stub. No FF handle to flush — see :meth:`add_ff_insert_chunk`."""
        return

    def flush_pending_ff(self, model) -> None:
        """Legacy stub. No pending FF state to flush."""
        return

    def refresh_static_handle(self, model) -> None:
        """Legacy stub. There is no static handle to refresh."""
        return

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _on_client_connect(self, client) -> None:
        # Pre-allocate the per-client display state with a placeholder
        # image; the render loop will replace it as soon as the model
        # is attached.
        W, H = self.render_size
        placeholder = np.zeros((H, W, 3), dtype=np.uint8)
        try:
            client.scene.set_background_image(placeholder, format="jpeg")
        except Exception as exc:
            print(f"[viser-direct] set_background_image (placeholder) failed: {exc}", flush=True)
        with self._client_state_lock:
            self._client_state[int(client.client_id)] = {
                "client": client,
                "last_pos": None,
                "last_wxyz": None,
                "last_fov": None,
            }
        # Camera-move trigger: every drag / scroll / pan in the browser
        # fires this. Without it the render only repaints on tracker
        # ticks (potentially every 10s+ if FF is slow) -> input lag is
        # awful. With it the user gets ~60Hz responsive camera control
        # AND the per-tick model-mutation renders.
        try:
            @client.camera.on_update
            def _on_camera_update(_camera) -> None:
                self.request_render()
        except Exception as exc:
            print(f"[viser-direct] on_camera_update wiring failed: {exc}", flush=True)
        # If we already have an initial camera (because D0 fired before
        # the client connected), snap them now and mark them done so the
        # next set_initial_camera doesn't re-snap them. If _initial_c2w
        # is still None, set_initial_camera() will snap this client when
        # it later gets set (also marking them done).
        if self._initial_c2w is not None:
            self._apply_initial_camera(client)
            self._initial_camera_applied.add(int(client.client_id))
        # Trigger one render so the new client sees the live scene
        # immediately instead of waiting for the next tracker tick.
        self.request_render()
        print(f"[viser-direct] client {client.client_id} connected", flush=True)

    def _on_client_disconnect(self, client) -> None:
        with self._client_state_lock:
            self._client_state.pop(int(client.client_id), None)
        self._initial_camera_applied.discard(int(client.client_id))
        print(f"[viser-direct] client {client.client_id} disconnected", flush=True)

    def _apply_initial_camera(self, client) -> None:
        if self._initial_c2w is None:
            return
        try:
            c2w = self._initial_c2w
            cam_pos = c2w[:3, 3].astype(np.float32)
            R_nerf = c2w[:3, :3].astype(np.float32)
            # nerfstudio c2w (Y-up, Z-back at camera) -> viser (Y-down,
            # Z-forward). _FLIP_YZ is its own inverse, so we multiply on
            # the right with the same matrix to invert the conversion
            # done on the read path in ``_build_camera_from_viser``.
            R_viser = (R_nerf @ _FLIP_YZ).astype(np.float32)
            wxyz = _rotmat_to_quat_wxyz_np(R_viser)
            client.camera.position = cam_pos
            client.camera.wxyz = wxyz
        except Exception as exc:
            print(f"[viser-direct] initial camera set failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Render loop (background thread)
    # ------------------------------------------------------------------

    @property
    def is_closing(self) -> bool:
        """True once :meth:`close` has begun tearing the server down. Off-thread
        callers (e.g. the AnySplat FF bg thread) check this before pushing to
        viser so they don't ``submit`` onto an executor that's already shutting
        down at interpreter exit (``cannot schedule new futures after shutdown``)."""
        return self._stop_event.is_set()

    def request_render(self) -> None:
        """Wake the render thread for ONE pass. Called by the pipeline
        from every tracker tick and from every FF insertion site (and
        on client connect). Decouples render cadence from any fixed
        polling rate — there's exactly one render per tracker tick, so
        every per-tick mutation lands in the browser without delay.

        No-op once the server is closing — a late bg push after teardown
        would otherwise raise ``cannot schedule new futures after shutdown``."""
        if self._stop_event.is_set():
            return
        self._render_requested.set()

    def _render_loop(self) -> None:
        """Event-driven render loop. Blocks on ``_render_requested``;
        wakes once per request, renders for every connected client,
        sleeps. No polling, no hardcoded Hz cap.
        """
        while not self._stop_event.is_set():
            # Wait for a render request (or a short timeout so we can
            # exit promptly on stop).
            triggered = self._render_requested.wait(timeout=1.0)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue  # spurious wake / no request, keep waiting
            self._render_requested.clear()
            try:
                self._render_once()
            except Exception as exc:
                self._render_error_count += 1
                if self._render_error_count <= 3 or (self._render_error_count % 50) == 0:
                    print(f"[viser-direct] render error #{self._render_error_count}: {exc}", flush=True)
            # 1 Hz diagnostic
            now = time.time()
            if (now - self._last_diag_t) >= 1.0 and self._render_count > 0:
                n = max(self._render_count, 1)
                avg_ms = self._render_window_total_ms / n
                print(
                    f"[viser-direct/render] {n} frames in last {now - self._last_diag_t:.1f}s "
                    f"(avg {avg_ms:.1f} ms, errors={self._render_error_count})",
                    flush=True,
                )
                self._render_count = 0
                self._render_window_total_ms = 0.0
                self._last_diag_t = now

    def _render_once(self) -> None:
        if self._stop_event.is_set():
            return  # server tearing down — don't push onto a closing executor
        model = self._model
        if model is None:
            return
        # Snapshot the client list quickly to avoid holding the lock
        # during the (slow) render.
        with self._client_state_lock:
            clients = [(cid, st["client"]) for cid, st in self._client_state.items()]
        if not clients:
            return
        # Refresh the side-panel camera-feed thumbnail once per render (shared
        # GUI element, not per-client).
        self._refresh_feed_image()
        W, H = self.render_size
        for cid, client in clients:
            try:
                # If 'Follow tracked frame' is on, snap this client's camera to
                # the tracked frame's pose BEFORE reading it back to render —
                # so the splat view matches the tracker's camera.
                self._apply_follow_pose(client)
                camera = _build_camera_from_viser(client.camera, W, H, self._device)
            except Exception as exc:
                if self._render_error_count <= 3:
                    print(f"[viser-direct] camera build failed for client {cid}: {exc}", flush=True)
                continue
            # Take the model lock briefly around the forward render so a
            # mid-frame tracker write doesn't corrupt the rasterization.
            t_render = time.time()
            try:
                with self.model_lock, torch.no_grad():
                    outputs = model.get_outputs(camera)
                rgb_t = outputs.get("rgb")
                if rgb_t is None:
                    continue
                # rgb is (H, W, 3) float in [0, 1] (Splatfacto convention).
                rgb_np = (rgb_t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy()
            except Exception as exc:
                if self._render_error_count <= 3:
                    print(f"[viser-direct] get_outputs failed for client {cid}: {exc}", flush=True)
                continue
            render_ms = (time.time() - t_render) * 1000.0
            self._render_window_total_ms += render_ms
            try:
                client.scene.set_background_image(rgb_np, format="jpeg", jpeg_quality=self.jpeg_quality)
                self._render_count += 1
            except Exception as exc:
                if self._render_error_count <= 3:
                    print(f"[viser-direct] set_background_image failed for client {cid}: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._stop_event.set()
        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        try:
            self.server.stop()
        except Exception:
            pass
