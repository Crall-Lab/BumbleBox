from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Optional

from .image_controls import camera_readout
from .tuning import infrared_enabled



@dataclass(frozen=True)
class LiveMonitorOptions:
    # The OwlSight selects a cropped 16:9 sensor mode for 1280x960. This
    # validated 4:3 request selects its full-field binned sensor mode instead.
    rgb_width: int = 1920
    rgb_height: int = 1440
    display_fps: float = 10.0
    duration_seconds: float = 0.0
    thermal_override: Optional[bool] = None
    realsense_override: Optional[bool] = None
    # The monitor's own NoIR tuning switch, for this session: True or False
    # starts it on or off. ``None`` starts it where it was last left, or, if
    # it has never been used, where recordings are (camera.infrared).
    noir_override: Optional[bool] = None
    # The config file this monitor remembers its NoIR switch for. ``None``
    # remembers nothing.
    remember_key: Optional[str] = None
    # On a Pi with two cameras: True or False starts the monitor showing both
    # side by side, or only the selected one. ``None`` starts it as it was
    # last left (remembered for ``box_key``), or with one if never used.
    both_override: Optional[bool] = None
    # The first camera's config file, which stands for the whole box.
    box_key: Optional[str] = None
    # What to call the selected camera's feed when the box has two.
    rgb_title: Optional[str] = None

    def validate(self) -> None:
        if self.rgb_width <= 0 or self.rgb_height <= 0:
            raise ValueError("Live-monitor RGB dimensions must be positive.")
        if self.rgb_width % 2 or self.rgb_height % 2:
            raise ValueError("Live-monitor RGB dimensions must be even for YUV420 capture.")
        if self.display_fps <= 0:
            raise ValueError("Live-monitor display FPS must be greater than zero.")
        if self.duration_seconds < 0:
            raise ValueError("Live-monitor duration must be zero or greater.")


@dataclass(frozen=True)
class MonitorCamera:
    """The other RGB camera of a two-camera Pi, for the side-by-side view."""

    config: dict[str, Any]
    title: str = "Camera 2"
    # Whether it is camera 1. Camera 1 is always drawn on the left.
    first: bool = False


# The feed of the camera selected in the main window, and of the other one.
RGB_STREAM = "rgb"
OTHER_RGB_STREAM = "rgb2"
# Two cameras that ask for different tuning files cannot both have theirs:
# libcamera loads one for the whole program.
TUNING_CONFLICT = "the two cameras need different tuning files, and one window can load only one"


@dataclass(frozen=True)
class LiveFramePacket:
    stream: str
    image_bgr: Any
    sequence: int
    captured_monotonic_s: float
    # The camera's own brightness image, when the feed has one. The grayscale
    # view shows this as it is, rather than working brightness back out of the
    # colour picture.
    image_gray: Any = None
    # What the camera is using for this frame (sensor gain, white balance), in words.
    info: str = ""


class LatestFrameStore:
    """Bounded cross-thread frame exchange: exactly one frame per displayed stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, LiveFramePacket] = {}

    def put(self, packet: LiveFramePacket) -> None:
        with self._lock:
            self._frames[packet.stream] = packet

    def snapshot(self) -> dict[str, LiveFramePacket]:
        with self._lock:
            return dict(self._frames)


class _SmoothedRange:
    def __init__(self, *, alpha: float = 0.18) -> None:
        self.alpha = float(alpha)
        self.low: Optional[float] = None
        self.high: Optional[float] = None

    def update(self, low: float, high: float) -> tuple[float, float]:
        if high <= low:
            high = low + 1.0
        if self.low is None or self.high is None:
            self.low, self.high = float(low), float(high)
        else:
            self.low = (1.0 - self.alpha) * self.low + self.alpha * float(low)
            self.high = (1.0 - self.alpha) * self.high + self.alpha * float(high)
        if self.high <= self.low:
            self.high = self.low + 1.0
        return self.low, self.high


class ThermalColorizer:
    def __init__(self) -> None:
        self._range = _SmoothedRange()

    def colorize(self, raw16: Any) -> Any:
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"OpenCV and NumPy are required for thermal preview: {exc}") from exc

        frame = np.asarray(raw16)
        if frame.ndim != 2:
            raise ValueError(f"Thermal preview expects a 2D frame, got {frame.shape}.")
        low, high = np.percentile(frame, (2.0, 98.0))
        low, high = self._range.update(float(low), float(high))
        scaled = ((frame.astype(np.float32) - low) * (255.0 / (high - low))).clip(0, 255)
        return cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_INFERNO)


class DepthColorizer:
    def __init__(self) -> None:
        self._range = _SmoothedRange()

    def colorize(self, raw16: Any) -> Any:
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"OpenCV and NumPy are required for depth preview: {exc}") from exc

        depth = np.asarray(raw16)
        if depth.ndim != 2:
            raise ValueError(f"Depth preview expects a 2D frame, got {depth.shape}.")
        valid = depth[depth > 0]
        if not valid.size:
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        near, far = np.percentile(valid, (2.0, 98.0))
        near, far = self._range.update(float(near), float(far))
        # Near pixels receive warmer colors while missing depth remains black.
        scaled = ((far - depth.astype(np.float32)) * (255.0 / (far - near))).clip(0, 255)
        scaled[depth == 0] = 0
        colorized = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
        colorized[depth == 0] = 0
        return colorized


def resolve_local_monitor_sensors(
    config: dict[str, Any],
    *,
    thermal_override: Optional[bool] = None,
    realsense_override: Optional[bool] = None,
) -> set[str]:
    available = {"rgb", "thermal", "realsense"}
    distributed = config.get("distributed_capture", {})
    if isinstance(distributed, dict) and bool(distributed.get("enabled", False)):
        local_node = next(
            (
                node
                for node in distributed.get("nodes", [])
                if isinstance(node, dict) and bool(node.get("local", False))
            ),
            None,
        )
        available = (
            {str(item).strip().lower() for item in local_node.get("sensors", [])}
            if local_node is not None
            else set()
        )

    requested = {"rgb"}
    thermal_enabled = bool(config.get("thermal", {}).get("enabled", False))
    realsense_enabled = bool(config.get("realsense", {}).get("enabled", False))
    if thermal_override is not None:
        thermal_enabled = bool(thermal_override)
    if realsense_override is not None:
        realsense_enabled = bool(realsense_override)
    if thermal_enabled:
        requested.add("thermal")
    if realsense_enabled:
        requested.add("realsense")
    return requested & available


MonitorStatusCallback = Callable[[str, str, str], None]


class LiveMonitorBackend:
    """Own camera workers while exposing only their latest display frame."""

    def __init__(
        self,
        config: dict[str, Any],
        options: LiveMonitorOptions,
        *,
        status_callback: Optional[MonitorStatusCallback] = None,
        other_camera: Optional[MonitorCamera] = None,
    ) -> None:
        options.validate()
        self.config = deepcopy(config)
        self.options = options
        self.other_camera = other_camera
        self.status_callback = status_callback
        self.frame_store = LatestFrameStore()
        self.sensor_names = resolve_local_monitor_sensors(
            self.config,
            thermal_override=options.thermal_override,
            realsense_override=options.realsense_override,
        )
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        # NoIR tuning, for this monitor only. It starts where it was last
        # left, or where recordings are if it has never been used, and never
        # changes the saved config.
        self.recording_noir = infrared_enabled(self.config)
        remembered = remembered_noir(options.remember_key)
        if options.noir_override is not None:
            self.noir = bool(options.noir_override)
        elif remembered is not None:
            self.noir = remembered
        else:
            self.noir = self.recording_noir
        self._rgb_thread: Optional[threading.Thread] = None
        self._rgb_restart = threading.Event()
        self._sequences: dict[str, int] = {}
        # Both cameras side by side, on a Pi that has two. Starts as it was
        # last left; one camera if it has never been used.
        self.both = False
        if self.can_show_both:
            remembered_both = _remembered("both_cameras", options.box_key)
            if options.both_override is not None:
                self.both = bool(options.both_override)
            elif remembered_both is not None:
                self.both = remembered_both

    @property
    def can_show_both(self) -> bool:
        """Whether this Pi has a second RGB camera to show beside the selected one."""
        return self.other_camera is not None and RGB_STREAM in self.sensor_names

    def rgb_streams(self) -> list[str]:
        """The RGB feeds being shown now, camera 1 first."""
        if RGB_STREAM not in self.sensor_names:
            return []
        if not (self.both and self.other_camera is not None):
            return [RGB_STREAM]
        return [OTHER_RGB_STREAM, RGB_STREAM] if self.other_camera.first else [RGB_STREAM, OTHER_RGB_STREAM]

    def streams(self) -> list[str]:
        """Every feed being shown now, in display order."""
        return self.rgb_streams() + [stream for stream in monitor_streams(self.sensor_names) if stream != RGB_STREAM]

    def _rgb_configs(self) -> list[tuple[str, dict[str, Any]]]:
        configs = {RGB_STREAM: self.config}
        if self.other_camera is not None:
            configs[OTHER_RGB_STREAM] = self.other_camera.config
        return [(stream, configs[stream]) for stream in self.rgb_streams()]

    @property
    def tuning_file_is_named(self) -> bool:
        """A tuning file named by hand in the config is used whatever the NoIR switch says."""
        return self.config.get("camera", {}).get("tuning_file") not in (None, "")

    def _restart_rgb(self) -> None:
        if RGB_STREAM not in self.sensor_names or self.stop_event.is_set():
            return
        self._rgb_restart.set()
        if self._rgb_thread is not None and not self._rgb_thread.is_alive():
            # The cameras stopped with an error; try again with the new choice.
            self._launch_rgb()

    def _launch_rgb(self) -> None:
        streams = (RGB_STREAM, OTHER_RGB_STREAM) if self.other_camera is not None else (RGB_STREAM,)
        self._launch("rgb", streams, self._run_rgb)

    def set_both(self, enabled: bool) -> None:
        """Show both cameras side by side, or only the selected one.

        The cameras are closed and opened again: two cameras in one window
        have to be opened together so that they share one tuning choice.
        """
        enabled = bool(enabled) and self.can_show_both
        if enabled == self.both:
            return
        self.both = enabled
        _remember("both_cameras", self.options.box_key, enabled)
        self._restart_rgb()

    def set_noir(self, enabled: bool) -> None:
        """Switch the monitor's camera between its NoIR and its standard tuning.

        A camera reads its tuning file when it is opened, so the camera is
        closed and opened again. Recordings are not affected.
        """
        enabled = bool(enabled)
        if enabled == self.noir:
            return
        self.noir = enabled
        remember_noir(self.options.remember_key, enabled)
        self._restart_rgb()

    def _status(self, stream: str, state: str, message: str) -> None:
        if self.status_callback is not None:
            try:
                self.status_callback(stream, state, message)
            except Exception:
                pass

    def _launch(self, name: str, streams: tuple[str, ...], target: Callable[[], None]) -> None:
        def wrapped() -> None:
            for stream in streams:
                self._status(stream, "starting", "Starting camera...")
            try:
                target()
            except Exception as exc:
                for stream in streams:
                    self._status(stream, "error", str(exc))
            else:
                for stream in streams:
                    self._status(stream, "stopped", "Stream stopped.")

        thread = threading.Thread(target=wrapped, name=f"bbx-monitor-{name}", daemon=True)
        self.threads.append(thread)
        if name == "rgb":
            self._rgb_thread = thread
        thread.start()

    def start(self) -> None:
        if self.threads:
            return
        if "rgb" in self.sensor_names:
            self._launch_rgb()
        else:
            self._status("rgb", "disabled", "Assigned to another Pi or disabled.")
        if "thermal" in self.sensor_names:
            self._launch("thermal", ("thermal",), self._run_thermal)
        else:
            self._status("thermal", "disabled", "Not enabled on this Pi.")
        if "realsense" in self.sensor_names:
            self._launch(
                "realsense",
                ("realsense_color", "realsense_depth"),
                self._run_realsense,
            )
        else:
            self._status("realsense_color", "disabled", "Not enabled on this Pi.")
            self._status("realsense_depth", "disabled", "Not enabled on this Pi.")

    def stop(self, *, timeout: float = 1.5) -> None:
        self.stop_event.set()
        deadline = time.monotonic() + max(0.0, float(timeout))
        for thread in self.threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

    def _put(
        self, stream: str, frame: Any, sequence: int, captured: float, gray: Any = None, info: str = ""
    ) -> None:
        self.frame_store.put(
            LiveFramePacket(
                stream=stream,
                image_bgr=frame,
                sequence=int(sequence),
                captured_monotonic_s=float(captured),
                image_gray=gray,
                info=str(info or ""),
            )
        )

    def _run_rgb(self) -> None:
        """Keep the RGB camera, or both of them, open and feeding frames.

        Runs until the monitor closes. A switch that needs the cameras opened
        differently (NoIR tuning, one camera or both) closes them and goes
        round again.
        """
        if bool(self.config.get("runtime", {}).get("use_mock_camera", False)):
            self._run_mock_rgb()
            return

        from .camera_select import connected_camera_model
        from .run_engine import _PicameraCaptureSession
        from .tuning import resolve_camera_tuning_file

        while not self.stop_event.is_set():
            self._rgb_restart.clear()
            noir = self.noir
            cameras = self._rgb_configs()
            plans = []
            for stream, config in cameras:
                preview_config = deepcopy(config)
                preview_config.setdefault("camera", {})["width"] = int(self.options.rgb_width)
                preview_config["camera"]["height"] = int(self.options.rgb_height)
                preview_config["camera"]["fps_target"] = float(self.options.display_fps)
                # The tuning file is chosen from the recording's own camera
                # settings, not from the smaller size the monitor opens the
                # camera at, with this monitor's NoIR switch in place of
                # camera.infrared.
                tuning_config = deepcopy(config)
                tuning_config.setdefault("camera", {})["infrared"] = noir
                plans.append((stream, preview_config, tuning_config))

            # One camera chooses its own tuning file. Two share libcamera's
            # camera manager, which loads one tuning choice for both when it
            # starts. If both want the same file they get it; if not, each is
            # left on its own standard tuning, which always fits its sensor.
            session_options: dict[str, Any] = {}
            conflict = False
            if len(plans) > 1:
                wanted = {
                    resolve_camera_tuning_file(tuning_config, detected_sensor=connected_camera_model(config))
                    for (_, config), (_, _, tuning_config) in zip(cameras, plans)
                }
                conflict = len(wanted) > 1
                session_options["tuning_file"] = None if conflict else next(iter(wanted))

            with ExitStack() as stack:
                workers: list[threading.Thread] = []
                for stream, preview_config, tuning_config in plans:
                    try:
                        session = stack.enter_context(
                            _PicameraCaptureSession(preview_config, tuning_config=tuning_config, **session_options)
                        )
                        session.start()
                    except Exception as exc:
                        # One camera that will not open does not take the other with it.
                        self._status(stream, "error", str(exc))
                        continue
                    opened_with = getattr(session, "resolved_tuning_file", None)
                    if "tuning_file" in session_options:
                        # The first camera opened decides what the manager loaded.
                        session_options["tuning_file"] = opened_with
                    tuning = monitor_tuning_label(noir, opened_with, conflict=conflict)
                    self._status(
                        stream,
                        "live",
                        f"Live at {self.options.rgb_width} x {self.options.rgb_height}, {tuning}.",
                    )
                    monochrome = bool(preview_config["camera"].get("monochrome_output", False))
                    worker = threading.Thread(
                        target=self._feed_rgb,
                        args=(stream, session, monochrome, tuning),
                        name=f"bbx-monitor-{stream}-frames",
                        daemon=True,
                    )
                    workers.append(worker)
                    worker.start()
                while not self.stop_event.is_set() and not self._rgb_restart.is_set():
                    self.stop_event.wait(0.05)
                for worker in workers:
                    worker.join(timeout=2.0)
            if self._rgb_restart.is_set() and not self.stop_event.is_set():
                for stream in self.rgb_streams():
                    self._status(stream, "starting", "Opening the camera again...")

    def _feed_rgb(self, stream: str, session: Any, monochrome: bool, tuning: str) -> None:
        """Pass one open camera's frames to the display until told to stop."""
        from .run_engine import _frame_luma, _frame_to_bgr

        interval = 1.0 / float(self.options.display_fps)
        try:
            while not self.stop_event.is_set() and not self._rgb_restart.is_set():
                frame, metadata = capture_frame_and_metadata(session.picam2)
                captured = time.perf_counter()
                self._sequences[stream] = self._sequences.get(stream, 0) + 1
                # The tuning in use, then the gain and white balance the
                # camera is using: the numbers to copy when fixing them on
                # the Settings page.
                readout = camera_readout(metadata)
                self._put(
                    stream,
                    _frame_to_bgr(frame, monochrome_output=monochrome),
                    self._sequences[stream],
                    captured,
                    gray=_frame_luma(frame),
                    info=f"{tuning}  |  {readout}" if readout else tuning,
                )
                self.stop_event.wait(interval)
        except Exception as exc:
            self._status(stream, "error", str(exc))

    def _run_mock_rgb(self) -> None:
        from .run_engine import _frame_luma, _frame_to_bgr
        from .simulated_capture import simulated_rgb_yuv420_frame

        interval = 1.0 / float(self.options.display_fps)
        started = time.perf_counter()
        live: set[str] = set()
        while not self.stop_event.is_set():
            captured = time.perf_counter()
            streams = self.rgb_streams()
            for stream in streams:
                if stream not in live:
                    self._status(stream, "live", "Simulated RGB stream.")
                # The two simulated cameras are out of step, so they can be told apart.
                phase = (captured - started + (2.0 if stream == OTHER_RGB_STREAM else 0.0)) % 4.0
                frame = simulated_rgb_yuv420_frame(self.options.rgb_width, self.options.rgb_height, phase, 4.0)
                self._sequences[stream] = self._sequences.get(stream, 0) + 1
                self._put(stream, _frame_to_bgr(frame), self._sequences[stream], captured, gray=_frame_luma(frame))
            live = set(streams)
            self.stop_event.wait(interval)

    def _run_thermal(self) -> None:
        colorizer = ThermalColorizer()
        if bool(self.config.get("runtime", {}).get("use_mock_camera", False)):
            from .simulated_capture import simulated_thermal_frame

            section = self.config.get("thermal", {})
            width = int(section.get("width", 160))
            height = int(section.get("height", 120))
            source_fps = max(0.5, float(section.get("fps_target", 8.7)))
            interval = 1.0 / min(source_fps, float(self.options.display_fps))
            sequence = 0
            started = time.perf_counter()
            self._status("thermal", "live", "Simulated thermal stream.")
            while not self.stop_event.is_set():
                captured = time.perf_counter()
                phase = (captured - started) % 4.0
                raw = simulated_thermal_frame(width, height, phase, 4.0)
                sequence += 1
                self._put("thermal", colorizer.colorize(raw), sequence, captured)
                self.stop_event.wait(interval)
            return

        from .run_engine import (
            _ThermalCaptureSession,
            _decode_thermal_raw16_frame,
            _infer_thermal_raw16_layout,
        )

        sequence = 0
        with _ThermalCaptureSession(self.config) as session:
            self._status("thermal", "live", f"Live from {session.device_path}.")
            while not self.stop_event.is_set():
                ok, frame = session.capture.read()
                captured = time.perf_counter()
                if not ok or frame is None:
                    continue
                layout = _infer_thermal_raw16_layout(frame)
                if layout is None:
                    raise RuntimeError(
                        f"Thermal monitor received an unsupported frame layout: "
                        f"shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}"
                    )
                raw = _decode_thermal_raw16_frame(frame, layout)
                sequence += 1
                self._put("thermal", colorizer.colorize(raw), sequence, captured)

    def _run_realsense(self) -> None:
        depth_colorizer = DepthColorizer()
        if bool(self.config.get("runtime", {}).get("use_mock_camera", False)):
            from .simulated_capture import (
                simulated_realsense_color_frame,
                simulated_realsense_depth_frame,
            )

            section = self.config.get("realsense", {})
            width = int(section.get("color_width", 848))
            height = int(section.get("color_height", 480))
            depth_width = int(section.get("depth_width", width))
            depth_height = int(section.get("depth_height", height))
            interval = 1.0 / float(self.options.display_fps)
            sequence = 0
            started = time.perf_counter()
            self._status("realsense_color", "live", "Simulated RealSense color stream.")
            self._status("realsense_depth", "live", "Simulated RealSense depth stream.")
            while not self.stop_event.is_set():
                captured = time.perf_counter()
                phase = (captured - started) % 4.0
                color = simulated_realsense_color_frame(width, height, phase, 4.0)
                depth = simulated_realsense_depth_frame(depth_width, depth_height, phase, 4.0)
                sequence += 1
                self._put("realsense_color", color, sequence, captured)
                self._put(
                    "realsense_depth",
                    depth_colorizer.colorize(depth),
                    sequence,
                    captured,
                )
                self.stop_event.wait(interval)
            return

        from .realsense_camera import RealSenseRecordingSession

        next_display = 0.0
        display_interval = 1.0 / float(self.options.display_fps)
        sequence = 0
        with tempfile.TemporaryDirectory(prefix="bbx-live-realsense-") as directory:
            with RealSenseRecordingSession(
                self.config,
                session_dir=directory,
                session_name="live_monitor",
            ) as session:
                self._status(
                    "realsense_color",
                    "live",
                    f"Live from RealSense {session.device.serial}.",
                )
                self._status(
                    "realsense_depth",
                    "live",
                    f"Live from RealSense {session.device.serial}.",
                )
                while not self.stop_event.is_set():
                    frameset = session.pipeline.wait_for_frames(1000)
                    captured = time.perf_counter()
                    depth, color, _depth_frame, _color_frame = session._process_frameset(frameset)
                    if captured < next_display:
                        continue
                    next_display = captured + display_interval
                    sequence += 1
                    self._put("realsense_color", color, sequence, captured)
                    self._put(
                        "realsense_depth",
                        depth_colorizer.colorize(depth),
                        sequence,
                        captured,
                    )


def _monitor_lock_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    root = Path(runtime) if runtime else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    user_id = str(os.getuid()) if hasattr(os, "getuid") else str(Path.home())
    return root / f"bumblebox-v2-live-monitor-{user_id}.lock"


def live_monitor_is_running() -> bool:
    try:
        from PyQt5.QtCore import QLockFile
    except Exception:
        return False
    lock = QLockFile(str(_monitor_lock_path()))
    lock.setStaleLockTime(30_000)
    acquired = lock.tryLock(0)
    if acquired:
        lock.unlock()
        return False
    return True


MONITOR_STYLE = """
QWidget {
    color: #293936;
    font-family: "Avenir Next", "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget#MonitorRoot { background: #f5f6f1; }
QFrame#MonitorHeader {
    background: #e6f1eb;
    border: 1px solid #d3e3da;
    border-radius: 16px;
}
QLabel#MonitorTitle { color: #24483f; font-size: 25px; font-weight: 600; }
QLabel#MonitorSubtitle { color: #63756f; font-size: 13px; }
QFrame#VideoCard {
    background: #fffefb;
    border: 1px solid #dde2dc;
    border-radius: 14px;
}
QLabel#VideoTitle { color: #314b45; font-size: 15px; font-weight: 600; }
QLabel#VideoStatus { color: #6d7b77; font-size: 12px; }
QLabel#VideoSurface {
    background: #17211f;
    color: #b8c7c2;
    border-radius: 9px;
    padding: 4px;
}
QLabel#PrivacyNote {
    background: #fff0cf;
    color: #79551a;
    border: 1px solid #edd9a7;
    border-radius: 9px;
    padding: 8px 11px;
    font-weight: 600;
}
QPushButton {
    background: #e2ede7;
    color: #2f5f54;
    border: 1px solid #cedfd6;
    border-radius: 9px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #d6e7de; }
QPushButton#CloseButton { background: #39786a; color: white; border-color: #39786a; }
QPushButton#PanelButton { padding: 4px 10px; font-size: 12px; border-radius: 7px; }
QPushButton:checked { background: #39786a; color: white; border-color: #39786a; }
QLabel#MonitorEmpty { color: #63756f; font-size: 15px; }
QWidget#MonitorRoot[fullView="true"] { background: #101614; }
QFrame#VideoCard[fullView="true"] { background: #101614; border: none; border-radius: 0; }
QFrame#VideoCard[fullView="true"] QLabel#VideoTitle { color: #e3ece9; }
QFrame#VideoCard[fullView="true"] QLabel#VideoStatus { color: #9db0aa; }
QFrame#VideoCard[fullView="true"] QLabel#VideoSurface { border-radius: 0; padding: 0; }
"""


STREAM_TITLES = {
    "rgb": "RGB camera",
    "rgb2": "Second RGB camera",
    "thermal": "PureThermal / Lepton",
    "realsense_color": "RealSense Color",
    "realsense_depth": "RealSense Depth",
}


def monitor_state_path() -> Path:
    """Where the monitor keeps what it remembers on this computer.

    The same folder as the encode benchmark result. It is kept out of the
    config file on purpose: the monitor changes no settings.
    """
    base = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "BumbleBox" / "live_monitor.json"


def _read_monitor_state() -> dict[str, Any]:
    try:
        state = json.loads(monitor_state_path().read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _remembered(switch: str, key: Optional[str]) -> Optional[bool]:
    """Where a switch was last left for this config file; ``None`` if it was never used."""
    if not key:
        return None
    saved = _read_monitor_state().get(switch)
    value = saved.get(str(key)) if isinstance(saved, dict) else None
    return value if isinstance(value, bool) else None


def remembered_noir(key: Optional[str]) -> Optional[bool]:
    """Where the NoIR switch was last left for this config file; ``None`` if it was never used."""
    return _remembered("noir", key)


def remember_noir(key: Optional[str], enabled: bool) -> None:
    """Remember where the NoIR switch was left."""
    _remember("noir", key, enabled)


def _remember(switch: str, key: Optional[str], enabled: bool) -> None:
    """Remember where a switch was left. Failing to write is not an error: the monitor carries on."""
    if not key:
        return
    try:
        state = _read_monitor_state()
        saved = state.get(switch)
        if not isinstance(saved, dict):
            saved = {}
        saved[str(key)] = bool(enabled)
        state[switch] = saved
        path = monitor_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        scratch.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(scratch, path)
    except OSError:
        pass


def monitor_tuning_label(noir: bool, tuning_file: Any, conflict: bool = False) -> str:
    """The tuning the monitor's camera was opened with, in a few words."""
    if conflict and not tuning_file:
        return f"standard tuning ({TUNING_CONFLICT})"
    if tuning_file:
        name = Path(str(tuning_file)).name
        if name.endswith("_noir.json"):
            return f"NoIR tuning ({name})"
        return f"tuning {name}"
    if noir:
        return "standard tuning (NoIR tuning was not available for this camera)"
    return "standard tuning"


def capture_frame_and_metadata(picam2: Any) -> tuple[Any, Any]:
    """One frame and what the camera reports about it (``None`` where it reports nothing)."""
    capture_request = getattr(picam2, "capture_request", None)
    if not callable(capture_request):
        return picam2.capture_array(), None
    request = capture_request()
    try:
        frame = request.make_array("main")
        try:
            metadata = request.get_metadata()
        except Exception:
            metadata = None
    finally:
        request.release()
    return frame, metadata


PREVIEW_ONLY_NOTE = (
    "PREVIEW ONLY: no images or video are saved, and nothing here changes how recordings "
    "are made. Close this monitor before recording."
)

# Feeds that are real colour images. Thermal and depth are drawn with a false
# colour scale that carries the measurement, so the grayscale view leaves them.
COLOUR_STREAMS = {"rgb", "rgb2", "realsense_color"}


# The contrast boost is the one the tag detector applies before it looks for
# tags (tag_tracking_utils), so the boosted view is what the detector sees.
CONTRAST_BOOST_CLIP_LIMIT = 2.0
CONTRAST_BOOST_TILES = (8, 8)


def boost_contrast(gray: Any) -> Any:
    """Local contrast enhancement (CLAHE) of an 8-bit grayscale image."""
    import cv2
    import numpy as np

    array = np.ascontiguousarray(gray)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {array.shape}.")
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_BOOST_CLIP_LIMIT, tileGridSize=CONTRAST_BOOST_TILES)
    return clahe.apply(array)


def bgr_to_gray(frame: Any) -> Any:
    """Brightness of a BGR image as an 8-bit array.

    For feeds that only arrive in colour (the depth camera's colour image).
    The main camera supplies its own brightness image, which is used instead.
    """
    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected a BGR image, got shape {array.shape}.")
    blue, green, red = (array[:, :, channel].astype(np.uint16) for channel in range(3))
    # Rec. 601 weights scaled to 256 (0.114, 0.587, 0.299).
    return ((blue * 29 + green * 150 + red * 77) >> 8).astype(np.uint8)


def monitor_streams(sensor_names: set[str]) -> list[str]:
    """The feeds to show, in display order, for the sensors this setup uses.

    A depth camera contributes two feeds (colour and depth). A sensor that is
    off, or assigned to another Pi, contributes none, so a camera-only box
    gets a single feed rather than three empty panels.
    """
    streams: list[str] = []
    if "rgb" in sensor_names:
        streams.append("rgb")
    if "thermal" in sensor_names:
        streams.append("thermal")
    if "realsense" in sensor_names:
        streams.extend(["realsense_color", "realsense_depth"])
    return streams


def monitor_grid(count: int, pair_first: bool = False) -> list[tuple[int, int, int, int]]:
    """Where each of ``count`` feeds goes: (row, column, rows spanned, columns spanned).

    One feed fills the window; two sit side by side; with three the first is
    large on the left and the other two are stacked beside it; four make a
    2 x 2 grid.

    ``pair_first`` is for two cameras shown together: the first two feeds
    share the top row, each half the width, and the rest sit in a row below.
    """
    if count <= 0:
        return []
    if pair_first and count > 2:
        below = count - 2
        cells = [(0, 0, 1, below), (0, below, 1, below)]
        cells.extend((1, 2 * index, 1, 2) for index in range(below))
        return cells
    if count == 1:
        return [(0, 0, 1, 1)]
    if count == 2:
        return [(0, 0, 1, 1), (0, 1, 1, 1)]
    if count == 3:
        return [(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
    return [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)][:count]


def rgb_stream_title(config: dict[str, Any]) -> str:
    """Name the RGB feed after the camera profile when one is selected."""
    try:
        from .camera_profiles import CAMERA_PROFILES

        key = str(config.get("camera", {}).get("profile", "custom"))
        profile = CAMERA_PROFILES.get(key)
        if profile is not None and key != "custom":
            return f"{STREAM_TITLES['rgb']} ({profile.label})"
    except Exception:
        pass
    return STREAM_TITLES["rgb"]


def create_live_monitor_window(
    config: dict[str, Any], options: LiveMonitorOptions, other_camera: Optional[MonitorCamera] = None
) -> Any:
    """Build the monitor window. A ``QApplication`` must already exist.

    The monitor only looks: it saves no images and changes no settings.
    """
    options.validate()
    try:
        from PyQt5.QtCore import QObject, QSize, Qt, QTimer, pyqtSignal
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - desktop dependency
        raise RuntimeError(
            "The live monitor requires PyQt5. Rerun start_bumblebox.sh or install python3-pyqt5."
        ) from exc

    class StatusBridge(QObject):
        status = pyqtSignal(str, str, str)

    class ImageSurface(QLabel):
        double_clicked = pyqtSignal()

        def __init__(self, *, nearest: bool = False) -> None:
            super().__init__("Waiting for frames...")
            self.setObjectName("VideoSurface")
            self.setAlignment(Qt.AlignCenter)
            self.setMinimumSize(300, 210)
            self._source: Optional[QPixmap] = None
            self._nearest = bool(nearest)
            self._frame: Any = None
            self._gray: Any = None
            self._grayscale = False
            self._contrast_boost = False

        def sizeHint(self) -> Any:
            # A fixed request rather than the size of the picture last shown.
            # The layout decides how large the picture is drawn; a label that
            # asked for its picture's size would take the room that lets the
            # rows of the grid be given different shares.
            return QSize(300, 210)

        def minimumSizeHint(self) -> Any:
            return QSize(300, 210)

        def forget(self) -> None:
            """Drop the picture being shown, so an old frame is not left on screen."""
            self._source = None
            self._frame = None
            self._gray = None
            self.clear()

        def set_grayscale(self, enabled: bool) -> None:
            """Draw in grayscale or colour. Redraws the frame on screen, so it works while paused."""
            self._grayscale = bool(enabled)
            if self._frame is not None:
                self._render()

        def set_contrast_boost(self, enabled: bool) -> None:
            """Boost local contrast in the grayscale view. Display only."""
            self._contrast_boost = bool(enabled)
            if self._frame is not None:
                self._render()

        def set_bgr(self, frame: Any, gray: Any = None) -> None:
            """Show a frame. ``gray`` is the camera's own brightness image, if the feed has one."""
            import numpy as np

            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Monitor display expects BGR image data, got {array.shape}.")
            if gray is not None:
                gray = np.asarray(gray)
                if gray.shape != array.shape[:2]:
                    gray = None  # not the same picture; fall back to the colour one
            self._frame = array
            self._gray = gray
            self._render()

        def shown_gray(self) -> Any:
            """The grayscale image as it is drawn: the camera's brightness, boosted if asked."""
            gray = self._gray if self._gray is not None else bgr_to_gray(self._frame)
            return boost_contrast(gray) if self._contrast_boost else gray

        def _render(self) -> None:
            import numpy as np

            array = self._frame
            height, width = int(array.shape[0]), int(array.shape[1])
            if self._grayscale:
                gray = np.ascontiguousarray(self.shown_gray())
                image = QImage(gray.data, width, height, int(gray.strides[0]), QImage.Format_Grayscale8).copy()
            else:
                rgb = np.ascontiguousarray(array[:, :, ::-1])
                image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888).copy()
            self._source = QPixmap.fromImage(image)
            self._redraw()

        def _redraw(self) -> None:
            if self._source is None:
                return
            mode = Qt.FastTransformation if self._nearest else Qt.SmoothTransformation
            self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio, mode))

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._redraw()

        def mouseDoubleClickEvent(self, event: Any) -> None:
            self.double_clicked.emit()

    class VideoPanel(QFrame):
        full_view_requested = pyqtSignal()

        def __init__(self, title: str, *, nearest: bool = False) -> None:
            super().__init__()
            self.setObjectName("VideoCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(13, 12, 13, 13)
            layout.setSpacing(7)
            self._layout = layout
            title_row = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setObjectName("VideoTitle")
            self.full_view_button = QPushButton("Full Screen")
            self.full_view_button.setObjectName("PanelButton")
            self.full_view_button.setToolTip("Show only this feed, filling the screen. Double-clicking the image does the same.")
            self.full_view_button.clicked.connect(self.full_view_requested.emit)
            title_row.addWidget(title_label, 1)
            title_row.addWidget(self.full_view_button)
            self.status_label = QLabel("Starting...")
            self.status_label.setObjectName("VideoStatus")
            self.status_label.setWordWrap(True)
            self.surface = ImageSurface(nearest=nearest)
            self.surface.double_clicked.connect(self.full_view_requested.emit)
            layout.addLayout(title_row)
            layout.addWidget(self.status_label)
            layout.addWidget(self.surface, 1)
            self.last_sequence = 0
            self.last_captured: Optional[float] = None
            self.frame_times: deque[float] = deque(maxlen=30)
            self.state = "starting"
            self.message = "Starting..."
            self.dimensions: Optional[tuple[int, int]] = None
            self.info = ""

        def reset(self) -> None:
            """Back to how the panel is before its camera has delivered anything."""
            self.surface.forget()
            self.last_captured = None
            self.frame_times.clear()
            self.dimensions = None
            self.info = ""
            self.state = "starting"
            self.message = "Starting camera..."
            self.refresh_status()

        def set_full_view(self, enabled: bool) -> None:
            """Restyle for filling the screen: dark surround, no card border, a way back."""
            self.setProperty("fullView", "true" if enabled else "false")
            self.full_view_button.setText("Exit Full Screen (Esc)" if enabled else "Full Screen")
            self._layout.setContentsMargins(*((10, 8, 10, 8) if enabled else (13, 12, 13, 13)))
            for widget in (self, self.status_label, self.surface, *self.findChildren(QLabel)):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

        def set_status(self, state: str, message: str) -> None:
            self.state = str(state)
            self.message = str(message)
            if state in {"error", "disabled", "stopped"} and self.last_sequence == 0:
                self.surface.clear()
                self.surface.setText(message)
            self.refresh_status()

        def set_packet(self, packet: LiveFramePacket) -> None:
            if packet.sequence == self.last_sequence:
                return
            self.last_sequence = packet.sequence
            self.last_captured = packet.captured_monotonic_s
            self.frame_times.append(packet.captured_monotonic_s)
            frame = packet.image_bgr
            self.info = packet.info
            self.dimensions = (int(frame.shape[1]), int(frame.shape[0]))
            self.surface.set_bgr(frame, gray=packet.image_gray)
            self.state = "live"
            self.refresh_status()

        def refresh_status(self) -> None:
            if self.state != "live" or self.last_captured is None:
                self.status_label.setText(self.message)
                return
            now = time.perf_counter()
            age_ms = max(0.0, (now - self.last_captured) * 1000.0)
            fps = 0.0
            if len(self.frame_times) > 1:
                elapsed = self.frame_times[-1] - self.frame_times[0]
                fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            width, height = self.dimensions or (0, 0)
            text = f"{width} x {height}  |  {fps:.1f} displayed FPS  |  {age_ms:.0f} ms old"
            if self.info:
                text += f"  |  {self.info}"
            self.status_label.setText(text)

    class LiveMonitorWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BumbleBox Live Camera Monitor")
            self.setMinimumSize(920, 650)
            self.bridge = StatusBridge()
            self.bridge.status.connect(self._set_status)
            self.backend = LiveMonitorBackend(
                config,
                options,
                status_callback=lambda stream, state, message: self.bridge.status.emit(
                    stream, state, message
                ),
                other_camera=other_camera,
            )
            # Only the feeds this setup uses get a panel.
            self.streams = self.backend.streams()
            self.both = self.backend.both
            self.resize(1420 if len(self.streams) > 1 else 1100, 900)
            self.paused = False
            self.grayscale_view = False
            self.contrast_boost = False
            self.noir = self.backend.noir
            self.full_view: Optional[str] = None
            self._was_maximized = False
            self._build()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh)
            self.timer.start(max(30, int(round(1000.0 / options.display_fps))))
            QTimer.singleShot(0, self.backend.start)

        def _build(self) -> None:
            root = QWidget()
            root.setObjectName("MonitorRoot")
            self.setCentralWidget(root)
            self._root = root
            outer = QVBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 16)
            outer.setSpacing(12)
            self._outer = outer

            header = QFrame()
            header.setObjectName("MonitorHeader")
            self.header = header
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(18, 14, 16, 14)
            title_col = QVBoxLayout()
            title = QLabel("Live Camera Monitor")
            title.setObjectName("MonitorTitle")
            subtitle = QLabel()
            subtitle.setObjectName("MonitorSubtitle")
            subtitle.setWordWrap(True)
            self.subtitle = subtitle
            self._update_subtitle()
            title_col.addWidget(title)
            title_col.addWidget(subtitle)
            header_layout.addLayout(title_col, 1)
            has_colour_feed = any(stream in COLOUR_STREAMS for stream in self.streams)
            self.gray_view_button = QPushButton("Grayscale View")
            self.gray_view_button.setCheckable(True)
            self.gray_view_button.setToolTip(
                "Show the camera image in grayscale. This only changes what you see here; "
                "recordings are not affected (that setting is on the Run page of the main window). "
                "Shortcut: G."
            )
            self.gray_view_button.toggled.connect(self.set_grayscale_view)
            self.gray_view_button.setVisible(has_colour_feed)
            self.boost_button = QPushButton("Contrast Boost")
            self.boost_button.setCheckable(True)
            self.boost_button.setToolTip(
                "Bring out detail in dark and bright areas, the way the tag detector sees the image. "
                "Shown in grayscale. This only changes what you see here; recordings are not affected. "
                "Shortcut: C."
            )
            self.boost_button.toggled.connect(self.set_contrast_boost)
            self.boost_button.setVisible(has_colour_feed)
            self.noir_button = QPushButton("NoIR Tuning")
            self.noir_button.setCheckable(True)
            self.noir_button.setChecked(self.noir)
            if self.backend.tuning_file_is_named:
                self.noir_button.setEnabled(False)
                self.noir_button.setToolTip(
                    "This camera's tuning file is named in the config (camera.tuning_file), "
                    "so there is nothing to switch here."
                )
            else:
                self.noir_button.setToolTip(
                    "Open the camera with its NoIR tuning (for a camera with no infrared filter) or with "
                    "its standard tuning. The camera closes and opens again, which takes a few seconds. "
                    "With both cameras shown it applies to both. This only changes what you see here, and "
                    "the monitor opens with it where you last left it. Recordings follow Infrared light on "
                    "the Settings page of the main window. Shortcut: N."
                )
            self.noir_button.toggled.connect(self.set_noir)
            self.noir_button.setVisible("rgb" in self.streams)
            self.both_button = QPushButton("Both Cameras")
            self.both_button.setCheckable(True)
            self.both_button.setChecked(self.both)
            self.both_button.setToolTip(
                "Show the two cameras side by side, or only the one selected in the main window. "
                "The cameras close and open again, which takes a few seconds. The monitor opens with "
                "this where you last left it. Shortcut: B."
            )
            self.both_button.toggled.connect(self.set_both)
            self.both_button.setVisible(self.backend.can_show_both)
            self.pause_button = QPushButton("Pause Display")
            self.pause_button.clicked.connect(self._toggle_pause)
            close_button = QPushButton("Close Monitor")
            close_button.setObjectName("CloseButton")
            close_button.clicked.connect(self.close)
            header_layout.addWidget(self.both_button)
            header_layout.addWidget(self.noir_button)
            header_layout.addWidget(self.gray_view_button)
            header_layout.addWidget(self.boost_button)
            header_layout.addWidget(self.pause_button)
            header_layout.addWidget(close_button)
            outer.addWidget(header)

            note = QLabel(PREVIEW_ONLY_NOTE)
            note.setObjectName("PrivacyNote")
            note.setWordWrap(True)
            self.note = note
            self._label_toggles()
            outer.addWidget(note)

            grid = QGridLayout()
            grid.setSpacing(12)
            self.grid = grid
            titles = dict(STREAM_TITLES, rgb=options.rgb_title or rgb_stream_title(config))
            # The second camera has a panel from the start, out of sight until
            # Both Cameras is switched on.
            panel_streams = list(self.streams)
            if self.backend.can_show_both:
                titles[OTHER_RGB_STREAM] = other_camera.title
                if OTHER_RGB_STREAM not in panel_streams:
                    panel_streams.append(OTHER_RGB_STREAM)
            self.panels = {
                stream: VideoPanel(titles[stream], nearest=(stream == "thermal")) for stream in panel_streams
            }
            for stream, panel in self.panels.items():
                panel.full_view_requested.connect(lambda selected=stream: self.toggle_full_view(selected))
            self.empty_label = QLabel(
                "No cameras are assigned to this Pi in this setup, so there is nothing to preview."
            )
            self.empty_label.setObjectName("MonitorEmpty")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setVisible(not self.panels)
            outer.addLayout(grid, 1)
            outer.addWidget(self.empty_label, 1 if not self.panels else 0)
            self._arrange(self.streams)

        def _arrange(self, streams: list[str]) -> None:
            """Lay out the given feeds in the grid and hide the rest."""
            for panel in self.panels.values():
                self.grid.removeWidget(panel)
                panel.setVisible(False)
            for index in range(max(2, self.grid.rowCount())):
                self.grid.setRowStretch(index, 0)
            for index in range(max(2, self.grid.columnCount())):
                self.grid.setColumnStretch(index, 0)
            # Two cameras shown together share the top row and get most of the height.
            pair = len([stream for stream in streams if stream in (RGB_STREAM, OTHER_RGB_STREAM)]) == 2
            cells = monitor_grid(len(streams), pair_first=pair)
            for stream, (row, column, row_span, column_span) in zip(streams, cells):
                panel = self.panels[stream]
                self.grid.addWidget(panel, row, column, row_span, column_span)
                panel.setVisible(True)
                for used_row in range(row, row + row_span):
                    self.grid.setRowStretch(used_row, 2 if pair and used_row == 0 and len(streams) > 2 else 1)
                for used_column in range(column, column + column_span):
                    self.grid.setColumnStretch(used_column, 1)

        def _update_subtitle(self) -> None:
            feeds = len(self.streams)
            showing = "no camera feeds" if not feeds else "1 camera feed" if feeds == 1 else f"{feeds} camera feeds"
            self.subtitle.setText(
                f"Showing {showing} for this setup. RGB preview {options.rgb_width} x {options.rgb_height}; "
                f"display capped at {options.display_fps:g} FPS. Double-click a feed for full screen."
            )

        def _label_toggles(self) -> None:
            # The state is in the words as well as the colour.
            self.gray_view_button.setText(f"Grayscale View: {'On' if self.grayscale_view else 'Off'}")
            self.boost_button.setText(f"Contrast Boost: {'On' if self.contrast_boost else 'Off'}")
            self.noir_button.setText(f"NoIR Tuning: {'On' if self.noir else 'Off'}")
            self.both_button.setText(f"Both Cameras: {'On' if self.both else 'Off'}")
            # Say so when the picture here is not built the way a recording's is.
            text = PREVIEW_ONLY_NOTE
            if "rgb" in self.streams and not self.backend.tuning_file_is_named and self.noir != self.backend.recording_noir:
                text += (
                    f" NoIR tuning is {'on' if self.noir else 'off'} here and "
                    f"{'on' if self.backend.recording_noir else 'off'} for recordings "
                    "(Infrared light, Settings page)."
                )
            self.note.setText(text)

        def _apply_view(self) -> None:
            """Push the two view choices to the buttons and the colour feeds."""
            for button, state in (
                (self.gray_view_button, self.grayscale_view),
                (self.boost_button, self.contrast_boost),
            ):
                if button.isChecked() != state:
                    button.blockSignals(True)
                    button.setChecked(state)
                    button.blockSignals(False)
            for stream, panel in self.panels.items():
                if stream in COLOUR_STREAMS:
                    panel.surface.set_contrast_boost(self.contrast_boost)
                    panel.surface.set_grayscale(self.grayscale_view)
            self._label_toggles()

        def set_grayscale_view(self, enabled: bool) -> None:
            """Show the colour camera feeds in grayscale. Changes the display only."""
            self.grayscale_view = bool(enabled)
            if not self.grayscale_view:
                self.contrast_boost = False  # the boost is a grayscale view
            self._apply_view()

        def set_both(self, enabled: bool) -> None:
            """Show the two cameras side by side, or only the selected one."""
            self.backend.set_both(enabled)
            self.both = self.backend.both
            if self.both_button.isChecked() != self.both:
                self.both_button.blockSignals(True)
                self.both_button.setChecked(self.both)
                self.both_button.blockSignals(False)
            streams = self.backend.streams()
            if self.full_view is not None and self.full_view not in streams:
                self.exit_full_view()
            for stream in (RGB_STREAM, OTHER_RGB_STREAM):
                # The cameras are being opened again: no old frame is left showing.
                if stream in self.panels:
                    self.panels[stream].reset()
            self.streams = streams
            if self.full_view is None:
                self._arrange(self.streams)
            self._update_subtitle()
            self._label_toggles()

        def set_noir(self, enabled: bool) -> None:
            """Open the camera with its NoIR or its standard tuning. Changes this monitor only."""
            self.noir = bool(enabled)
            if self.noir_button.isChecked() != self.noir:
                self.noir_button.blockSignals(True)
                self.noir_button.setChecked(self.noir)
                self.noir_button.blockSignals(False)
            self.backend.set_noir(self.noir)
            self._label_toggles()

        def set_contrast_boost(self, enabled: bool) -> None:
            """Boost local contrast, in grayscale. Changes the display only."""
            self.contrast_boost = bool(enabled)
            if self.contrast_boost:
                self.grayscale_view = True
            self._apply_view()

        def toggle_full_view(self, stream: str) -> None:
            """Fill the screen with one feed, or go back to all of them."""
            if self.full_view == stream:
                self.exit_full_view()
            else:
                self.enter_full_view(stream)

        def enter_full_view(self, stream: str) -> None:
            if stream not in self.panels:
                return
            if self.full_view is None:
                self._was_maximized = self.isMaximized()
            else:
                self.panels[self.full_view].set_full_view(False)
            self.full_view = stream
            self.header.setVisible(False)
            self.note.setVisible(False)
            self._outer.setContentsMargins(0, 0, 0, 0)
            self._set_root_full_view(True)
            self.panels[stream].set_full_view(True)
            self._arrange([stream])
            self.showFullScreen()

        def exit_full_view(self) -> None:
            if self.full_view is None:
                return
            self.panels[self.full_view].set_full_view(False)
            self.full_view = None
            self.header.setVisible(True)
            self.note.setVisible(True)
            self._outer.setContentsMargins(18, 18, 18, 16)
            self._set_root_full_view(False)
            self._arrange(self.streams)
            if self._was_maximized:
                self.showMaximized()
            else:
                self.showNormal()

        def _set_root_full_view(self, enabled: bool) -> None:
            self._root.setProperty("fullView", "true" if enabled else "false")
            self._root.style().unpolish(self._root)
            self._root.style().polish(self._root)

        def keyPressEvent(self, event: Any) -> None:
            if event.key() == Qt.Key_Escape and self.full_view is not None:
                self.exit_full_view()
                return
            if event.key() == Qt.Key_F11 and self.streams:
                self.toggle_full_view(self.full_view or self.streams[0])
                return
            if event.key() == Qt.Key_Space:
                self._toggle_pause()
                return
            if event.key() == Qt.Key_G and not self.gray_view_button.isHidden():
                self.set_grayscale_view(not self.grayscale_view)
                return
            if event.key() == Qt.Key_C and not self.boost_button.isHidden():
                self.set_contrast_boost(not self.contrast_boost)
                return
            if event.key() == Qt.Key_N and not self.noir_button.isHidden() and self.noir_button.isEnabled():
                self.set_noir(not self.noir)
                return
            if event.key() == Qt.Key_B and self.backend.can_show_both:
                self.set_both(not self.both)
                return
            super().keyPressEvent(event)

        def _set_status(self, stream: str, state: str, message: str) -> None:
            panel = self.panels.get(stream)
            if panel is not None:
                panel.set_status(state, message)

        def _refresh(self) -> None:
            if not self.paused:
                packets = self.backend.frame_store.snapshot()
                for stream, packet in packets.items():
                    panel = self.panels.get(stream)
                    if panel is not None:
                        panel.set_packet(packet)
            for panel in self.panels.values():
                panel.refresh_status()

        def _toggle_pause(self) -> None:
            self.paused = not self.paused
            self.pause_button.setText("Resume Display" if self.paused else "Pause Display")

        def closeEvent(self, event: Any) -> None:
            self.timer.stop()
            self.backend.stop(timeout=1.0)
            super().closeEvent(event)

    return LiveMonitorWindow()


def launch_live_monitor(
    config: dict[str, Any], options: LiveMonitorOptions, other_camera: Optional[MonitorCamera] = None
) -> int:
    options.validate()
    from .qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    try:
        from PyQt5.QtCore import QLockFile, QTimer
        from PyQt5.QtGui import QFont, QFontDatabase
        from PyQt5.QtWidgets import QApplication, QMessageBox
    except ImportError as exc:  # pragma: no cover - desktop dependency
        raise RuntimeError(
            "The live monitor requires PyQt5. Rerun start_bumblebox.sh or install python3-pyqt5."
        ) from exc

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("BumbleBox Live Monitor")
    app.setStyleSheet(MONITOR_STYLE)
    available = set(QFontDatabase().families())
    family = next(
        (name for name in ("Avenir Next", "Nunito Sans", "Noto Sans", "DejaVu Sans") if name in available),
        app.font().family(),
    )
    app.setFont(QFont(family, 10))
    lock = QLockFile(str(_monitor_lock_path()))
    lock.setStaleLockTime(30_000)
    if not lock.tryLock(100):
        QMessageBox.information(
            None,
            "Live monitor is already open",
            "Use the existing BumbleBox Live Camera Monitor window.",
        )
        return 1
    app._bumblebox_live_monitor_lock = lock
    window = create_live_monitor_window(config, options, other_camera)
    window.show()
    if options.duration_seconds > 0:
        QTimer.singleShot(int(round(options.duration_seconds * 1000.0)), window.close)
    return int(app.exec_())
