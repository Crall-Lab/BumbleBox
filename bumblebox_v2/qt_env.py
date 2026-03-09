from __future__ import annotations

import os
from typing import Mapping


def _looks_like_cv2_qt_path(value: str) -> bool:
    normalized = str(value or "").replace("\\", "/").lower()
    return "/cv2/qt/" in normalized


def _is_wayland_session(env: Mapping[str, str]) -> bool:
    session_type = str(env.get("XDG_SESSION_TYPE", "")).strip().lower()
    if session_type == "wayland":
        return True
    return bool(str(env.get("WAYLAND_DISPLAY", "")).strip())


def build_qt_safe_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)

    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = env.get(key, "")
        if value and _looks_like_cv2_qt_path(value):
            env.pop(key, None)

    # On Pi desktop sessions under Wayland, forcing Qt to use Wayland avoids xcb/plugin mismatches
    # inherited from pip OpenCV or older shell state.
    if _is_wayland_session(env):
        env["QT_QPA_PLATFORM"] = "wayland"
    elif str(env.get("QT_QPA_PLATFORM", "")).strip().lower() == "wayland":
        env.pop("QT_QPA_PLATFORM", None)

    # Platform theme plugins can also be inherited from unrelated desktop state.
    if _looks_like_cv2_qt_path(env.get("QT_QPA_PLATFORMTHEME", "")):
        env.pop("QT_QPA_PLATFORMTHEME", None)

    return env


def sanitize_current_qt_env() -> None:
    safe_env = build_qt_safe_env()

    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_QPA_PLATFORM"):
        if key in safe_env:
            os.environ[key] = safe_env[key]
        else:
            os.environ.pop(key, None)
