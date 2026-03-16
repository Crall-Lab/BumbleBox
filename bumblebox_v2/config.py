from __future__ import annotations

from copy import deepcopy
import getpass
import os
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency check
    yaml = None


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "default_config.yaml"
DEFAULT_USER_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"

VALID_PIPELINE_MODES = {
    "record_only",
    "track_only",
    "record_and_track",
    "mixed_schedule",
}

VALID_TRACKING_SOURCES = {"ram", "video"}
VALID_SCHED_BACKENDS = {"systemd", "cron"}
VALID_SCHED_SCOPES = {"system", "user"}
VALID_FLEET_ROLES = {"standalone", "queen", "worker"}
VALID_CAMERA_MODELS = {
    "auto",
    "hq",
    "hq_noir",
    "module3",
    "module3_wide",
    "module3_standard",
    "module3_noir",
}
VALID_PREVIEW_WINDOWS = {"QTGL", "QT", "DRM"}
VALID_CAMERA_CODECS = {"mp4", "mjpeg"}
VALID_THERMAL_PIXEL_FORMATS = {"auto", "y16", "gray8", "rgb"}
VALID_UI_THEME_MODES = {"dark", "light"}
SERVICE_USER_AUTO_SENTINELS = {"", "auto", "current", "default", "pi", "root"}


class ConfigError(ValueError):
    """Raised when the BumbleBox config is malformed."""


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for BumbleBox V2 config support. "
            "Install with: pip3 install pyyaml"
        )


def detect_current_user() -> str:
    for key in ("SUDO_USER", "USER", "LOGNAME"):
        value = str(os.environ.get(key, "")).strip()
        if value and value != "root":
            return value
    try:
        value = str(getpass.getuser()).strip()
        if value and value != "root":
            return value
    except Exception:
        pass
    return "pi"


def normalize_service_user_value(raw: Any) -> str:
    value = str(raw if raw is not None else "").strip()
    if value.lower() in SERVICE_USER_AUTO_SENTINELS:
        return detect_current_user()
    return value


def _normalize_service_user_in_config(config: Dict[str, Any]) -> None:
    scheduling = config.setdefault("scheduling", {})
    if isinstance(scheduling, dict):
        scheduling["service_user"] = normalize_service_user_value(scheduling.get("service_user"))


def _read_yaml(path: Path) -> Dict[str, Any]:
    _require_yaml()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config at {path} must be a YAML mapping/object.")
    return raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expect_keys(config: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Missing required top-level config keys: {', '.join(missing)}")


def validate_config(config: Dict[str, Any]) -> None:
    _expect_keys(
        config,
        [
            "system",
            "camera",
            "thermal",
            "pipeline",
            "capture",
            "tracking",
            "cleaning",
            "metrics",
            "calibration",
            "runtime",
            "scheduling",
            "fleet",
        ],
    )

    mode = config["pipeline"].get("mode")
    if mode not in VALID_PIPELINE_MODES:
        raise ConfigError(
            f"pipeline.mode must be one of {sorted(VALID_PIPELINE_MODES)}, got: {mode}"
        )

    tracking_source = config["pipeline"].get("tracking_source")
    if tracking_source not in VALID_TRACKING_SOURCES:
        raise ConfigError(
            "pipeline.tracking_source must be 'ram' or 'video', "
            f"got: {tracking_source}"
        )

    excluded_tag_ids = config.get("tracking", {}).get("excluded_tag_ids", [])
    if not isinstance(excluded_tag_ids, list):
        raise ConfigError("tracking.excluded_tag_ids must be a list of tag IDs")
    for idx, tag_id in enumerate(excluded_tag_ids):
        try:
            int(tag_id)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"tracking.excluded_tag_ids[{idx}] must be an integer-like tag ID"
            ) from exc

    backend = config["scheduling"].get("backend")
    if backend not in VALID_SCHED_BACKENDS:
        raise ConfigError(
            f"scheduling.backend must be one of {sorted(VALID_SCHED_BACKENDS)}, got: {backend}"
        )
    scope = config["scheduling"].get("scope", "system")
    if scope not in VALID_SCHED_SCOPES:
        raise ConfigError(
            f"scheduling.scope must be one of {sorted(VALID_SCHED_SCOPES)}, got: {scope}"
        )

    fleet = config.get("fleet", {})
    if not isinstance(fleet, dict):
        raise ConfigError("fleet must be a mapping/object")
    fleet_role = fleet.get("role", "standalone")
    if fleet_role not in VALID_FLEET_ROLES:
        raise ConfigError(f"fleet.role must be one of {sorted(VALID_FLEET_ROLES)}, got: {fleet_role}")
    queen_local_pipeline_enabled = fleet.get("queen_local_pipeline_enabled", True)
    if not isinstance(queen_local_pipeline_enabled, bool):
        raise ConfigError("fleet.queen_local_pipeline_enabled must be true or false")

    fleet_workers = fleet.get("workers", [])
    if not isinstance(fleet_workers, list):
        raise ConfigError("fleet.workers must be a list")
    for idx, worker in enumerate(fleet_workers):
        if not isinstance(worker, dict):
            raise ConfigError(f"fleet.workers[{idx}] must be a mapping/object")
        host = str(worker.get("host", "")).strip()
        if not host:
            raise ConfigError(f"fleet.workers[{idx}].host must be set")
        port = int(worker.get("port", 22))
        if port <= 0:
            raise ConfigError(f"fleet.workers[{idx}].port must be > 0")

    fleet_ssh = fleet.get("ssh", {})
    if fleet_ssh is not None and not isinstance(fleet_ssh, dict):
        raise ConfigError("fleet.ssh must be a mapping/object")
    if isinstance(fleet_ssh, dict):
        ssh_port = int(fleet_ssh.get("port", 22))
        if ssh_port <= 0:
            raise ConfigError("fleet.ssh.port must be > 0")
        timeout = float(fleet_ssh.get("connect_timeout_seconds", 5))
        if timeout <= 0:
            raise ConfigError("fleet.ssh.connect_timeout_seconds must be > 0")

    fleet_media = fleet.get("queen_media_schedule", {})
    if fleet_media is not None and not isinstance(fleet_media, dict):
        raise ConfigError("fleet.queen_media_schedule must be a mapping/object")
    if isinstance(fleet_media, dict):
        media_enabled = fleet_media.get("enabled", False)
        if not isinstance(media_enabled, bool):
            raise ConfigError("fleet.queen_media_schedule.enabled must be true or false")

        try:
            pull_interval = int(fleet_media.get("pull_interval_minutes", config["capture"].get("record_interval_minutes", 30)))
        except (TypeError, ValueError) as exc:
            raise ConfigError("fleet.queen_media_schedule.pull_interval_minutes must be an integer") from exc
        if pull_interval <= 0:
            raise ConfigError("fleet.queen_media_schedule.pull_interval_minutes must be > 0")

        try:
            track_interval = int(fleet_media.get("track_interval_minutes", 60))
        except (TypeError, ValueError) as exc:
            raise ConfigError("fleet.queen_media_schedule.track_interval_minutes must be an integer") from exc
        if track_interval <= 0:
            raise ConfigError("fleet.queen_media_schedule.track_interval_minutes must be > 0")

        output_root = fleet_media.get("output_root")
        if output_root is not None and not isinstance(output_root, str):
            raise ConfigError("fleet.queen_media_schedule.output_root must be null or a string path")

        try:
            max_total = int(fleet_media.get("max_videos_total", 200))
        except (TypeError, ValueError) as exc:
            raise ConfigError("fleet.queen_media_schedule.max_videos_total must be an integer") from exc
        if max_total <= 0:
            raise ConfigError("fleet.queen_media_schedule.max_videos_total must be > 0")

        try:
            max_per_worker = int(fleet_media.get("max_videos_per_worker", 1))
        except (TypeError, ValueError) as exc:
            raise ConfigError("fleet.queen_media_schedule.max_videos_per_worker must be an integer") from exc
        if max_per_worker <= 0:
            raise ConfigError("fleet.queen_media_schedule.max_videos_per_worker must be > 0")

        try:
            cooldown = int(fleet_media.get("cooldown_minutes", 60))
        except (TypeError, ValueError) as exc:
            raise ConfigError("fleet.queen_media_schedule.cooldown_minutes must be an integer") from exc
        if cooldown < 0:
            raise ConfigError("fleet.queen_media_schedule.cooldown_minutes must be >= 0")

        max_queen_load = fleet_media.get("max_queen_load_1m", 3.0)
        if max_queen_load is not None:
            try:
                max_queen_load_val = float(max_queen_load)
            except (TypeError, ValueError) as exc:
                raise ConfigError("fleet.queen_media_schedule.max_queen_load_1m must be numeric or null") from exc
            if max_queen_load_val <= 0:
                raise ConfigError("fleet.queen_media_schedule.max_queen_load_1m must be > 0 when set")

        min_queen_mem = fleet_media.get("min_queen_mem_gb", 0.8)
        if min_queen_mem is not None:
            try:
                min_queen_mem_val = float(min_queen_mem)
            except (TypeError, ValueError) as exc:
                raise ConfigError("fleet.queen_media_schedule.min_queen_mem_gb must be numeric or null") from exc
            if min_queen_mem_val < 0:
                raise ConfigError("fleet.queen_media_schedule.min_queen_mem_gb must be >= 0 when set")

        disable_visualization = fleet_media.get("disable_visualization", False)
        if not isinstance(disable_visualization, bool):
            raise ConfigError("fleet.queen_media_schedule.disable_visualization must be true or false")
        allow_when_active = fleet_media.get("allow_when_queen_bbox_active", False)
        if not isinstance(allow_when_active, bool):
            raise ConfigError("fleet.queen_media_schedule.allow_when_queen_bbox_active must be true or false")

    camera_model = config["camera"].get("model")
    if camera_model not in VALID_CAMERA_MODELS:
        raise ConfigError(
            f"camera.model must be one of {sorted(VALID_CAMERA_MODELS)}, got: {camera_model}"
        )
    camera_infrared = config["camera"].get("infrared", False)
    if not isinstance(camera_infrared, bool):
        raise ConfigError("camera.infrared must be true or false")
    camera_monochrome_output = config["camera"].get("monochrome_output", False)
    if not isinstance(camera_monochrome_output, bool):
        raise ConfigError("camera.monochrome_output must be true or false")
    preview_window = str(config["camera"].get("preview_window", "QT")).upper()
    if preview_window not in VALID_PREVIEW_WINDOWS:
        raise ConfigError(
            f"camera.preview_window must be one of {sorted(VALID_PREVIEW_WINDOWS)}, got: {preview_window}"
        )
    camera_codec = str(config["camera"].get("codec", "mp4")).strip().lower()
    if camera_codec not in VALID_CAMERA_CODECS:
        raise ConfigError(
            f"camera.codec must be one of {sorted(VALID_CAMERA_CODECS)}, got: {camera_codec}"
        )
    mp4_codec = str(config["camera"].get("mp4_codec", "libx264")).strip()
    if not mp4_codec:
        raise ConfigError("camera.mp4_codec must be a non-empty string")

    thermal = config.get("thermal", {})
    if not isinstance(thermal, dict):
        raise ConfigError("thermal must be a mapping/object")

    thermal_enabled = thermal.get("enabled", False)
    if not isinstance(thermal_enabled, bool):
        raise ConfigError("thermal.enabled must be true or false")

    thermal_device_path = thermal.get("device_path", "auto")
    if thermal_device_path is not None and not isinstance(thermal_device_path, str):
        raise ConfigError("thermal.device_path must be null or a string path")

    thermal_width = int(thermal.get("width", 160))
    if thermal_width <= 0:
        raise ConfigError("thermal.width must be > 0")

    thermal_height = int(thermal.get("height", 120))
    if thermal_height <= 0:
        raise ConfigError("thermal.height must be > 0")

    thermal_fps = float(thermal.get("fps_target", 8.7))
    if thermal_fps <= 0:
        raise ConfigError("thermal.fps_target must be > 0")

    thermal_pixel_format = str(thermal.get("pixel_format", "auto")).strip().lower()
    if thermal_pixel_format not in VALID_THERMAL_PIXEL_FORMATS:
        raise ConfigError(
            "thermal.pixel_format must be one of "
            f"{sorted(VALID_THERMAL_PIXEL_FORMATS)}, got: {thermal_pixel_format}"
        )

    thermal_expected_name = thermal.get("expected_name", "PureThermal")
    if thermal_expected_name is not None and not isinstance(thermal_expected_name, str):
        raise ConfigError("thermal.expected_name must be null or a string")

    ram_override = config["system"].get("ram_gb_override")
    if ram_override not in (None, "", 0):
        try:
            ram_gb = float(ram_override)
        except (TypeError, ValueError) as exc:
            raise ConfigError("system.ram_gb_override must be numeric when set") from exc
        if ram_gb <= 0:
            raise ConfigError("system.ram_gb_override must be > 0 when set")

    fps_target = config["camera"].get("fps_target", 0)
    if fps_target <= 0:
        raise ConfigError("camera.fps_target must be > 0")

    recording_seconds = config["capture"].get("recording_seconds", 0)
    if recording_seconds <= 0:
        raise ConfigError("capture.recording_seconds must be > 0")

    record_interval = config["capture"].get("record_interval_minutes", 0)
    track_interval = config["capture"].get("track_interval_minutes", 0)
    if record_interval <= 0:
        raise ConfigError("capture.record_interval_minutes must be > 0")
    if track_interval <= 0:
        raise ConfigError("capture.track_interval_minutes must be > 0")

    digital_zoom = config["camera"].get("digital_zoom")
    if digital_zoom is not None:
        if not isinstance(digital_zoom, (list, tuple)) or len(digital_zoom) != 4:
            raise ConfigError("camera.digital_zoom must be null or a 4-value list/tuple")

    contact_distance_cm = config["metrics"].get("contact_distance_cm", 0)
    if contact_distance_cm <= 0:
        raise ConfigError("metrics.contact_distance_cm must be > 0")

    pixels_per_cm = config["calibration"].get("pixels_per_cm", 0)
    if pixels_per_cm <= 0:
        raise ConfigError("calibration.pixels_per_cm must be > 0")

    warmup = float(config["runtime"].get("camera_warmup_seconds", 0))
    if warmup < 0:
        raise ConfigError("runtime.camera_warmup_seconds must be >= 0")
    render_tracking_video = config["runtime"].get("render_tracking_video", False)
    if not isinstance(render_tracking_video, bool):
        raise ConfigError("runtime.render_tracking_video must be true or false")

    ui_theme_mode = str(config["runtime"].get("ui_theme_mode", "dark")).strip().lower()
    if ui_theme_mode not in VALID_UI_THEME_MODES:
        raise ConfigError(
            f"runtime.ui_theme_mode must be one of {sorted(VALID_UI_THEME_MODES)}, got: {ui_theme_mode}"
        )


def load_defaults() -> Dict[str, Any]:
    defaults = _read_yaml(DEFAULT_CONFIG_PATH)
    try:
        # Keep default service user aligned with the account running setup on this machine.
        _normalize_service_user_in_config(defaults)
    except Exception:
        # Defaults should still load even if user detection fails.
        pass
    return defaults


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    defaults = load_defaults()

    if config_path.exists():
        user_config = _read_yaml(config_path)
        merged = _deep_merge(defaults, user_config)
    else:
        merged = defaults

    _normalize_service_user_in_config(merged)
    validate_config(merged)
    return merged


def save_config(config_path: str | Path, config: Dict[str, Any]) -> None:
    _require_yaml()
    _normalize_service_user_in_config(config)
    validate_config(config)

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def write_default_config(config_path: str | Path, force: bool = False) -> Path:
    config_path = Path(config_path)
    if config_path.exists() and not force:
        raise FileExistsError(f"Config already exists at {config_path}")

    config = load_defaults()
    save_config(config_path, config)
    return config_path
