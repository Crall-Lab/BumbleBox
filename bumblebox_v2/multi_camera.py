"""Two Raspberry Pi cameras on one Pi, recording on the same schedule.

Each camera has its own config file, so everything that already works for one
camera (the recorder, the timers, the checks, the GUI pages) works for each
of two. The first camera keeps the normal config (``config.yaml``); the
second lives next to it (``config.camera2.yaml``). What makes the pair safe
to run together:

* each config names a different camera, pinned to one physical connector
  when both cameras could be detected (see ``camera_select``);
* each has its own timer names, so both schedules can be installed;
* each camera has its own capture lock, so they record at the same moment;
* the second camera's sessions are tagged (``<host>_cam2_<date>_<time>``),
  so the two cameras never write files with the same name.

This module has no GUI code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .camera_select import (
    DetectedCamera,
    configured_camera_id,
    configured_camera_num,
    detect_cameras,
    resolve_camera_num,
    session_tag,
)
from .config import load_config, save_config

SECOND_CAMERA_FILE_PART = "camera2"
SECOND_CAMERA_TAG = "cam2"
SECOND_CAMERA_UNIT_SUFFIX = "-cam2"

# Settings that decide when a camera records. Kept the same on both cameras
# when the user wants them on one schedule.
SCHEDULE_KEYS = (
    ("pipeline", "mode"),
    ("capture", "recording_seconds"),
    ("capture", "record_interval_minutes"),
    ("capture", "track_interval_minutes"),
)

MEMORY_FAIL_RATIO = 0.80
MEMORY_WARN_RATIO = 0.60


@dataclass
class CameraEntry:
    index: int  # 1 for the first camera, 2 for the second
    path: Path
    config: Dict[str, Any]

    @property
    def label(self) -> str:
        return f"Camera {self.index}"

    @property
    def long_label(self) -> str:
        colony = str((self.config.get("system") or {}).get("colony_id") or "").strip()
        return f"{self.label} (colony {colony})" if colony else self.label

    @property
    def tag(self) -> str:
        return session_tag(self.config)

    @property
    def pinned(self) -> bool:
        return configured_camera_id(self.config) is not None


def second_camera_config_path(primary_path: str | Path) -> Path:
    """``config.yaml`` -> ``config.camera2.yaml``, in the same folder."""
    primary = Path(primary_path)
    return primary.with_name(f"{primary.stem}.{SECOND_CAMERA_FILE_PART}{primary.suffix}")


def has_second_camera(primary_path: str | Path) -> bool:
    return second_camera_config_path(primary_path).is_file()


def load_cameras(primary_path: str | Path, primary_config: Optional[Dict[str, Any]] = None) -> List[CameraEntry]:
    """The camera configs of this box: one entry, or two when a second camera is set up."""
    primary_path = Path(primary_path)
    entries = [
        CameraEntry(1, primary_path, primary_config if primary_config is not None else load_config(primary_path))
    ]
    second_path = second_camera_config_path(primary_path)
    if second_path.is_file():
        entries.append(CameraEntry(2, second_path, load_config(second_path)))
    return entries


def suggest_second_colony_id(primary_config: Dict[str, Any]) -> str:
    """'01' -> '02'; anything that is not a number gets '-2' added."""
    current = str((primary_config.get("system") or {}).get("colony_id") or "").strip()
    if current.isdigit():
        return str(int(current) + 1).zfill(len(current))
    return f"{current}-2" if current else "02"


@dataclass
class SecondCameraResult:
    path: Path
    pinned: bool
    notes: List[str] = field(default_factory=list)


def _pin(entries_config: Sequence[Dict[str, Any]], cameras: Sequence[DetectedCamera]) -> bool:
    """Give each config the libcamera Id of a different connected camera.

    The first config keeps the camera it already records from; the second
    gets the other one. Returns ``False``, changing nothing, unless two
    cameras with distinct Ids are connected.
    """
    usable = [camera for camera in cameras if camera.camera_id]
    if len(usable) < 2 or len({camera.camera_id for camera in usable}) < 2:
        return False
    first_config, second_config = entries_config[0], entries_config[1]
    try:
        first_num = resolve_camera_num(first_config, list(usable))
    except Exception:
        first_num = usable[0].num
    first = next((camera for camera in usable if camera.num == first_num), usable[0])
    second = next(camera for camera in usable if camera.camera_id != first.camera_id)
    first_config.setdefault("camera", {}).update({"camera_num": first.num, "camera_id": first.camera_id})
    second_config.setdefault("camera", {}).update({"camera_num": second.num, "camera_id": second.camera_id})
    return True


def create_second_camera(
    primary_path: str | Path,
    *,
    colony_id: Optional[str] = None,
    data_root: Optional[str] = None,
    cameras: Optional[Sequence[DetectedCamera]] = None,
) -> SecondCameraResult:
    """Create the second camera's config from the first camera's.

    The second camera starts with the first one's recording settings and
    schedule, with its schedule off. Thermal, depth and two-Pi capture are
    switched off in it: those devices exist once and stay with the first
    camera. ``cameras`` is what is connected (detected when omitted).
    """
    primary_path = Path(primary_path)
    second_path = second_camera_config_path(primary_path)
    if second_path.exists():
        raise FileExistsError(f"A second camera is already set up: {second_path}")

    primary = load_config(primary_path)
    second = deepcopy(primary)
    second_camera = second.setdefault("camera", {})
    second_camera["camera_num"] = 1 if configured_camera_num(primary) == 0 else 0
    second_camera["camera_id"] = None
    second_camera["session_tag"] = SECOND_CAMERA_TAG

    scheduling = second.setdefault("scheduling", {})
    prefix = str((primary.get("scheduling") or {}).get("unit_prefix", "bumblebox-v2")).strip() or "bumblebox-v2"
    scheduling["unit_prefix"] = f"{prefix}{SECOND_CAMERA_UNIT_SUFFIX}"
    scheduling["enabled"] = False

    for section in ("thermal", "realsense", "distributed_capture"):
        if isinstance(second.get(section), dict):
            second[section]["enabled"] = False
    if isinstance(second.get("fleet"), dict):
        second["fleet"]["role"] = "standalone"
    if isinstance(second.get("setup"), dict) and second["setup"].get("hardware_profile") not in (None, "custom"):
        second["setup"]["hardware_profile"] = "rgb_only"

    system = second.setdefault("system", {})
    system["colony_id"] = str(colony_id).strip() if colony_id and str(colony_id).strip() else suggest_second_colony_id(primary)
    if data_root and str(data_root).strip():
        system["data_root"] = str(data_root).strip()

    detected = list(cameras) if cameras is not None else detect_cameras()
    pinned = _pin([primary, second], detected)
    notes: List[str] = []
    if pinned:
        notes.append("Each config is pinned to one camera, so unplugging one cannot make the other record in its place.")
    else:
        notes.append(
            f"Only {len(detected)} camera(s) could be detected, so the cameras are matched by number and not "
            "pinned. Connect both and pin them (`bbx second-camera pin`, or Hardware in the GUI)."
        )

    save_config(second_path, second)
    if pinned:
        save_config(primary_path, primary)
    return SecondCameraResult(path=second_path, pinned=pinned, notes=notes)


def pin_cameras(primary_path: str | Path, *, cameras: Optional[Sequence[DetectedCamera]] = None) -> bool:
    """Pin both configs to the cameras connected now. ``False`` if two were not found."""
    entries = load_cameras(primary_path)
    if len(entries) < 2:
        raise FileNotFoundError("No second camera is set up.")
    detected = list(cameras) if cameras is not None else detect_cameras()
    if not _pin([entries[0].config, entries[1].config], detected):
        return False
    for entry in entries:
        save_config(entry.path, entry.config)
    return True


def remove_second_camera(primary_path: str | Path, *, now: Optional[datetime] = None) -> Path:
    """Set the second camera's config aside (renamed, not deleted). Its data is untouched.

    The caller must stop the second camera's schedule first; its timers point
    at this config file.
    """
    second_path = second_camera_config_path(primary_path)
    if not second_path.is_file():
        raise FileNotFoundError("No second camera is set up.")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = second_path.with_name(f"{second_path.stem}.removed-{stamp}{second_path.suffix}")
    second_path.rename(target)
    return target


def check_camera_set(entries: Sequence[CameraEntry]) -> List[str]:
    """Reasons two camera configs would get in each other's way."""
    if len(entries) < 2:
        return []
    first, second = entries[0], entries[1]
    problems: List[str] = []

    def prefix(entry: CameraEntry) -> str:
        return str((entry.config.get("scheduling") or {}).get("unit_prefix", "bumblebox-v2")).strip()

    if prefix(first) == prefix(second):
        problems.append(
            "Both cameras use the same timer names (scheduling.unit_prefix), so only one schedule can be installed."
        )
    first_id, second_id = configured_camera_id(first.config), configured_camera_id(second.config)
    same_pinned = first_id is not None and first_id == second_id
    same_number = first_id is None and second_id is None and configured_camera_num(first.config) == configured_camera_num(
        second.config
    )
    if same_pinned or same_number:
        problems.append("Both configs record from the same camera.")
    if first.tag == second.tag:
        problems.append(
            "Both cameras name their sessions the same way (camera.session_tag), so their files could be confused."
        )
    return problems


def schedule_values(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{section}.{key}": (config.get(section) or {}).get(key)
        for section, key in SCHEDULE_KEYS
    }


def schedules_match(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    return schedule_values(first) == schedule_values(second)


def copy_schedule(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """Make ``target`` record when ``source`` does. Returns whether anything changed."""
    changed = False
    for section, key in SCHEDULE_KEYS:
        value = (source.get(section) or {}).get(key)
        if (target.get(section) or {}).get(key) != value:
            target.setdefault(section, {})[key] = deepcopy(value)
            changed = True
    return changed


@dataclass
class CombinedLoad:
    level: str  # "ok", "warn" or "fail"
    message: str
    total_bytes: int = 0
    ram_bytes: int = 0


def _format_gib(value: float) -> str:
    return f"{value / 2**30:.1f} GiB"


def combined_memory(
    entries: Sequence[CameraEntry],
    *,
    schedule_check: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Optional[CombinedLoad]:
    """Memory both cameras need when they capture at the same moment.

    Timers are aligned to the clock, so two cameras with any intervals do
    capture together at least once a day; the sum is what the Pi must hold.
    Each camera's own check only knows about itself. ``None`` for one camera.
    """
    if len(entries) < 2:
        return None
    if schedule_check is None:
        from .schedule_check import run_schedule_check as schedule_check

    total = 0
    ram = 0
    for entry in entries:
        try:
            metrics = schedule_check(entry.config).metrics
        except Exception as exc:
            return CombinedLoad("warn", f"Memory for both cameras together could not be estimated: {exc}")
        total += int(metrics.get("estimated_frame_buffer_bytes", 0) or 0)
        ram = max(ram, int(metrics.get("total_ram_bytes", 0) or 0))
    if ram <= 0:
        return CombinedLoad("ok", f"Both cameras together hold about {_format_gib(total)} during a recording.", total, ram)
    ratio = total / ram
    text = (
        f"Both cameras capture at the same time and together hold about {_format_gib(total)} "
        f"of {_format_gib(ram)} memory ({ratio * 100:.0f}%)."
    )
    if ratio >= MEMORY_FAIL_RATIO:
        return CombinedLoad("fail", text + " That is too much; shorten the recordings or encode while recording.", total, ram)
    if ratio >= MEMORY_WARN_RATIO:
        return CombinedLoad("warn", text + " That leaves little room for tracking.", total, ram)
    return CombinedLoad("ok", text, total, ram)
