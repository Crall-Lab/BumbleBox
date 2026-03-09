from __future__ import annotations

import os
from typing import Mapping


def _looks_like_cv2_qt_path(value: str) -> bool:
    normalized = str(value or "").replace("\\", "/").lower()
    return "/cv2/qt/" in normalized


def build_qt_safe_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)

    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = env.get(key, "")
        if value and _looks_like_cv2_qt_path(value):
            env.pop(key, None)

    if env.get("WAYLAND_DISPLAY") and not str(env.get("QT_QPA_PLATFORM", "")).strip():
        env["QT_QPA_PLATFORM"] = "wayland"

    return env


def sanitize_current_qt_env() -> None:
    safe_env = build_qt_safe_env()

    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_QPA_PLATFORM"):
        if key in safe_env:
            os.environ[key] = safe_env[key]
        else:
            os.environ.pop(key, None)
