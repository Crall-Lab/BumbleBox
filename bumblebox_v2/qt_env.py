from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
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


def build_camera_safe_env(
    base_env: Mapping[str, str] | None = None,
    *,
    include_qt: bool = False,
) -> dict[str, str]:
    env = build_qt_safe_env(base_env) if include_qt else dict(base_env if base_env is not None else os.environ)

    for key in list(env.keys()):
        upper = key.upper()
        if upper.startswith(("LIBCAMERA_", "PICAMERA2_", "IPA_")):
            env.pop(key, None)

    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(key, None)

    env["TMPDIR"] = "/tmp"
    return env


def _stage_macos_qt_plugins() -> Path | None:
    """Copy PyQt plugins outside a hidden venv so Qt can discover them on macOS."""
    if sys.platform != "darwin":
        return None
    try:
        from importlib.util import find_spec

        spec = find_spec("PyQt5")
        if spec is None or not spec.submodule_search_locations:
            return None
        package_root = Path(next(iter(spec.submodule_search_locations)))
        source_root = package_root / "Qt5" / "plugins"
        if not source_root.is_dir():
            return None

        destination_root = Path(tempfile.gettempdir()) / f"bumblebox-qt-plugins-{os.getuid()}"
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if destination.exists() and destination.stat().st_size == source.stat().st_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # copyfile intentionally avoids preserving the source's macOS hidden flag.
            shutil.copyfile(source, destination)
            destination.chmod(source.stat().st_mode)
        return destination_root
    except Exception:
        return None


def sanitize_current_qt_env() -> None:
    safe_env = build_qt_safe_env()

    staged_plugins = _stage_macos_qt_plugins()
    if staged_plugins is not None:
        safe_env["QT_PLUGIN_PATH"] = str(staged_plugins)
        safe_env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(staged_plugins / "platforms")

    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_QPA_PLATFORM"):
        if key in safe_env:
            os.environ[key] = safe_env[key]
        else:
            os.environ.pop(key, None)
