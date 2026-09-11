from __future__ import annotations

import math
from typing import Any


def simulated_frame_times(duration: float, fps: float) -> list[float]:
    """Return a deterministic capture schedule without waiting in real time."""
    safe_fps = max(float(fps), 1e-6)
    frame_count = max(1, int(round(max(0.0, float(duration)) * safe_fps)))
    return [float(index) / safe_fps for index in range(frame_count)]


def simulated_target_center(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> tuple[int, int]:
    """Place one shared moving target in normalized image coordinates."""
    width = max(1, int(width))
    height = max(1, int(height))
    safe_duration = max(float(duration), 1e-6)
    phase = min(1.0, max(0.0, float(time_s) / safe_duration))
    x_norm = 0.12 + (0.76 * phase)
    y_norm = 0.50 + (0.18 * math.sin(phase * math.tau))
    return (
        min(width - 1, max(0, int(round(x_norm * (width - 1))))),
        min(height - 1, max(0, int(round(y_norm * (height - 1))))),
    )


def _target_bounds(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> tuple[int, int, int, int]:
    center_x, center_y = simulated_target_center(width, height, time_s, duration)
    radius = max(1, min(int(width), int(height)) // 12)
    return (
        max(0, center_x - radius),
        min(int(width), center_x + radius + 1),
        max(0, center_y - radius),
        min(int(height), center_y + radius + 1),
    )


def simulated_rgb_yuv420_frame(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> Any:
    import numpy as np

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Simulated YUV420 dimensions must be positive even integers.")

    frame = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    luma = frame[:height]
    horizontal = np.linspace(35, 95, width, dtype=np.uint8)
    luma[:] = horizontal[None, :]
    x0, x1, y0, y1 = _target_bounds(width, height, time_s, duration)
    luma[y0:y1, x0:x1] = 235
    return frame


def simulated_thermal_frame(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> Any:
    import numpy as np

    width = int(width)
    height = int(height)
    horizontal = np.linspace(28900, 29200, width, dtype=np.uint16)
    frame = np.broadcast_to(horizontal, (height, width)).copy()
    x0, x1, y0, y1 = _target_bounds(width, height, time_s, duration)
    frame[y0:y1, x0:x1] = np.uint16(30500)
    return frame


def simulated_realsense_depth_frame(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> Any:
    import numpy as np

    width = int(width)
    height = int(height)
    horizontal = np.linspace(850, 1250, width, dtype=np.uint16)
    frame = np.broadcast_to(horizontal, (height, width)).copy()
    x0, x1, y0, y1 = _target_bounds(width, height, time_s, duration)
    frame[y0:y1, x0:x1] = np.uint16(525)
    return frame


def simulated_realsense_color_frame(
    width: int,
    height: int,
    time_s: float,
    duration: float,
) -> Any:
    import numpy as np

    width = int(width)
    height = int(height)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(25, 90, width, dtype=np.uint8)
    frame[:, :, 0] = gradient[None, :]
    frame[:, :, 1] = gradient[None, :] // 2
    frame[:, :, 2] = 20
    x0, x1, y0, y1 = _target_bounds(width, height, time_s, duration)
    frame[y0:y1, x0:x1] = (30, 220, 245)
    return frame
