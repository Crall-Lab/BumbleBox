from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Optional


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
class LiveFramePacket:
    stream: str
    image_bgr: Any
    sequence: int
    captured_monotonic_s: float


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
    ) -> None:
        options.validate()
        self.config = deepcopy(config)
        self.options = options
        self.status_callback = status_callback
        self.frame_store = LatestFrameStore()
        self.sensor_names = resolve_local_monitor_sensors(
            self.config,
            thermal_override=options.thermal_override,
            realsense_override=options.realsense_override,
        )
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

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
        thread.start()

    def start(self) -> None:
        if self.threads:
            return
        if "rgb" in self.sensor_names:
            self._launch("rgb", ("rgb",), self._run_rgb)
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

    def _put(self, stream: str, frame: Any, sequence: int, captured: float) -> None:
        self.frame_store.put(
            LiveFramePacket(
                stream=stream,
                image_bgr=frame,
                sequence=int(sequence),
                captured_monotonic_s=float(captured),
            )
        )

    def _run_rgb(self) -> None:
        if bool(self.config.get("runtime", {}).get("use_mock_camera", False)):
            self._run_mock_rgb()
            return

        from .run_engine import _PicameraCaptureSession, _frame_to_bgr

        preview_config = deepcopy(self.config)
        preview_config.setdefault("camera", {})["width"] = int(self.options.rgb_width)
        preview_config["camera"]["height"] = int(self.options.rgb_height)
        preview_config["camera"]["fps_target"] = float(self.options.display_fps)
        monochrome = bool(preview_config["camera"].get("monochrome_output", False))
        interval = 1.0 / float(self.options.display_fps)
        sequence = 0
        with _PicameraCaptureSession(preview_config) as session:
            session.start()
            self._status(
                "rgb",
                "live",
                f"Live at {self.options.rgb_width} x {self.options.rgb_height}.",
            )
            while not self.stop_event.is_set():
                frame = session.picam2.capture_array()
                captured = time.perf_counter()
                sequence += 1
                self._put(
                    "rgb",
                    _frame_to_bgr(frame, monochrome_output=monochrome),
                    sequence,
                    captured,
                )
                self.stop_event.wait(interval)

    def _run_mock_rgb(self) -> None:
        from .run_engine import _frame_to_bgr
        from .simulated_capture import simulated_rgb_yuv420_frame

        interval = 1.0 / float(self.options.display_fps)
        sequence = 0
        started = time.perf_counter()
        self._status("rgb", "live", "Simulated RGB stream.")
        while not self.stop_event.is_set():
            captured = time.perf_counter()
            phase = (captured - started) % 4.0
            frame = simulated_rgb_yuv420_frame(
                self.options.rgb_width,
                self.options.rgb_height,
                phase,
                4.0,
            )
            sequence += 1
            self._put("rgb", _frame_to_bgr(frame), sequence, captured)
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
"""


def launch_live_monitor(config: dict[str, Any], options: LiveMonitorOptions) -> int:
    options.validate()
    from .qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    try:
        from PyQt5.QtCore import QObject, QLockFile, Qt, QTimer, pyqtSignal
        from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
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
        def __init__(self, *, nearest: bool = False) -> None:
            super().__init__("Waiting for frames...")
            self.setObjectName("VideoSurface")
            self.setAlignment(Qt.AlignCenter)
            self.setMinimumSize(300, 210)
            self._source: Optional[QPixmap] = None
            self._nearest = bool(nearest)

        def set_bgr(self, frame: Any) -> None:
            import numpy as np

            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Monitor display expects BGR image data, got {array.shape}.")
            rgb = np.ascontiguousarray(array[:, :, ::-1])
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            image = QImage(
                rgb.data,
                width,
                height,
                int(rgb.strides[0]),
                QImage.Format_RGB888,
            ).copy()
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

    class VideoPanel(QFrame):
        def __init__(self, title: str, *, nearest: bool = False) -> None:
            super().__init__()
            self.setObjectName("VideoCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(13, 12, 13, 13)
            layout.setSpacing(7)
            title_label = QLabel(title)
            title_label.setObjectName("VideoTitle")
            self.status_label = QLabel("Starting...")
            self.status_label.setObjectName("VideoStatus")
            self.status_label.setWordWrap(True)
            self.surface = ImageSurface(nearest=nearest)
            layout.addWidget(title_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.surface, 1)
            self.last_sequence = 0
            self.last_captured: Optional[float] = None
            self.frame_times: deque[float] = deque(maxlen=30)
            self.state = "starting"
            self.message = "Starting..."
            self.dimensions: Optional[tuple[int, int]] = None

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
            self.dimensions = (int(frame.shape[1]), int(frame.shape[0]))
            self.surface.set_bgr(frame)
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
            self.status_label.setText(
                f"{width} x {height}  |  {fps:.1f} displayed FPS  |  {age_ms:.0f} ms old"
            )

    class LiveMonitorWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BumbleBox Live Camera Monitor")
            self.resize(1420, 900)
            self.setMinimumSize(920, 650)
            self.bridge = StatusBridge()
            self.bridge.status.connect(self._set_status)
            self.backend = LiveMonitorBackend(
                config,
                options,
                status_callback=lambda stream, state, message: self.bridge.status.emit(
                    stream, state, message
                ),
            )
            self.paused = False
            self._build()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh)
            self.timer.start(max(30, int(round(1000.0 / options.display_fps))))
            QTimer.singleShot(0, self.backend.start)

        def _build(self) -> None:
            root = QWidget()
            root.setObjectName("MonitorRoot")
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 16)
            outer.setSpacing(12)

            header = QFrame()
            header.setObjectName("MonitorHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(18, 14, 16, 14)
            title_col = QVBoxLayout()
            title = QLabel("Live Camera Monitor")
            title.setObjectName("MonitorTitle")
            subtitle = QLabel(
                f"OwlSight preview {options.rgb_width} x {options.rgb_height}; "
                f"full-frame fit; display capped at {options.display_fps:g} FPS"
            )
            subtitle.setObjectName("MonitorSubtitle")
            title_col.addWidget(title)
            title_col.addWidget(subtitle)
            header_layout.addLayout(title_col, 1)
            self.pause_button = QPushButton("Pause Display")
            self.pause_button.clicked.connect(self._toggle_pause)
            close_button = QPushButton("Close Monitor")
            close_button.setObjectName("CloseButton")
            close_button.clicked.connect(self.close)
            header_layout.addWidget(self.pause_button)
            header_layout.addWidget(close_button)
            outer.addWidget(header)

            note = QLabel(
                "PREVIEW ONLY: no images or video are saved. Close this monitor before recording."
            )
            note.setObjectName("PrivacyNote")
            outer.addWidget(note)

            grid = QGridLayout()
            grid.setSpacing(12)
            self.panels = {
                "rgb": VideoPanel("OwlSight / RGB"),
                "thermal": VideoPanel("PureThermal / Lepton", nearest=True),
                "realsense_color": VideoPanel("RealSense Color"),
                "realsense_depth": VideoPanel("RealSense Depth"),
            }
            grid.addWidget(self.panels["rgb"], 0, 0)
            grid.addWidget(self.panels["thermal"], 0, 1)
            grid.addWidget(self.panels["realsense_color"], 1, 0)
            grid.addWidget(self.panels["realsense_depth"], 1, 1)
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            outer.addLayout(grid, 1)

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
    window = LiveMonitorWindow()
    window.show()
    if options.duration_seconds > 0:
        QTimer.singleShot(int(round(options.duration_seconds * 1000.0)), window.close)
    return int(app.exec_())
