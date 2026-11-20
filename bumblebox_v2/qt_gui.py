from __future__ import annotations

import copy
from datetime import datetime
import os
import signal
from pathlib import Path
import sys
import tempfile
from typing import Any

from .qt_env import sanitize_current_qt_env

sanitize_current_qt_env()

try:
    from PyQt5.QtCore import (
        QEvent,
        QObject,
        QLockFile,
        QProcess,
        QRunnable,
        QSize,
        QStandardPaths,
        QThreadPool,
        Qt,
        QTime,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt5.QtGui import QDesktopServices, QFont, QFontDatabase, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QAbstractItemView,
        QHeaderView,
        QInputDialog,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QTimeEdit,
        QVBoxLayout,
        QWidget,
        QWizard,
        QWizardPage,
    )
except ImportError as exc:  # pragma: no cover - depends on desktop runtime
    raise RuntimeError(
        "The BumbleBox Qt GUI requires PyQt5. Rerun start_bumblebox.sh, or install "
        "python3-pyqt5 on Raspberry Pi / pyqt5 in the BumbleBox environment."
    ) from exc

from .box_status import (
    DRIVE_NOT_MOUNTED,
    LEVEL_FAIL,
    LEVEL_OFF,
    LEVEL_OK,
    LEVEL_WARN,
    RECORDING_RUN_MODES,
    SCHEDULE_OFF,
    SCHEDULE_ON,
    SCHEDULE_PARTIAL,
    BoxStatus,
    FailureExplanation,
    StorageEstimate,
    StartReview,
    collect_box_status,
    command_label,
    describe_moment,
    describe_relative,
    benchmark_matches_camera,
    benchmark_result_for,
    describe_days,
    describe_schedule_plan,
    estimate_storage,
    explain_failure,
    format_bytes,
    installed_units_are_current,
    keeps_up,
    load_benchmark,
    recording_signature,
    recordings_per_day,
    format_start_review,
    review_schedule_start,
    session_size_bytes,
    signature_from_snapshot,
)
from .camera_profiles import CAMERA_PROFILES, apply_camera_profile
from .camera_select import DetectedCamera, detect_cameras
from .config import DEFAULT_USER_CONFIG_PATH, load_config, load_defaults, save_config, validate_config
from .data_drive import capture_in_progress, save_drive_choice
from .hardware_profiles import HARDWARE_PROFILES, apply_hardware_profile
from . import image_controls
from .multimodal_calibration import (
    MANIFEST_NAME as CALIBRATION_MANIFEST_NAME,
    add_calibration_session,
    format_calibration_project_status,
    get_calibration_project_status,
    initialize_calibration_project,
)
from .multi_camera import (
    CameraEntry,
    check_camera_set,
    combined_memory,
    copy_schedule,
    create_second_camera,
    load_cameras,
    pin_cameras,
    remove_second_camera,
    schedule_values,
    schedules_match,
    suggest_second_colony_id,
)
from .qt_data_drive import DataDriveDialog
from .recording_options import recording_is_grayscale, set_recording_grayscale
from .schedule_check import run_schedule_check
from .status_history import list_recent_run_records, load_run_summary
from .stream_writer import FRAME_BUFFER_STREAM, frame_buffer_mode
from .time_units import (
    DEFAULT_START_TIME,
    DURATION_UNITS,
    INTERVAL_UNITS,
    MINUTES_PER_DAY,
    best_unit,
    describe_interval,
    parse_time_of_day,
    to_base,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BBX_SCRIPT = REPO_ROOT / "bbx.py"

NAV_TITLES = ("Overview", "Run", "Results", "Hardware", "Settings", "Advanced")
PAGE_INDEX_OVERVIEW = 0
PAGE_INDEX_RUN = 1
PAGE_INDEX_RESULTS = 2
PAGE_INDEX_HARDWARE = 3
PAGE_INDEX_SETTINGS = 4
PAGE_INDEX_ADVANCED = 5

STATUS_REFRESH_MS = 30_000
SETTINGS_PREVIEW_DELAY_MS = 250
DEFAULT_BENCHMARK_ENCODERS = ("mjpeg", "libx264:ultrafast", "libx264:veryfast", "libx264:fast")
MP4_PRESET_CHOICES = (
    ("auto", "Automatic (ultrafast while recording, fast afterwards)"),
    ("ultrafast", "Ultrafast (largest files, least CPU)"),
    ("superfast", "Superfast"),
    ("veryfast", "Very fast"),
    ("faster", "Faster"),
    ("fast", "Fast"),
    ("medium", "Medium"),
    ("slow", "Slow (smallest files, most CPU)"),
)
# Settings page choice -> capture.restart_on_failure
RESTART_CHOICES = {"auto": "auto", "always": True, "never": False}
FRAME_BUFFER_HINTS = {
    "ram": (
        "Every frame is kept in memory until the recording ends, then encoded. Gives the "
        "highest frame rate, but memory use grows with recording length."
    ),
    "stream": (
        "Frames are encoded to disk during the recording, so memory use stays small and "
        "recordings can be long. The encoder must keep up with the camera."
    ),
}
# Settings fixed by a named camera profile are shown but cannot be edited here.
PROFILE_CONTROLLED_SETTINGS = {
    "camera.fps_target": "fps_target",
    "camera.codec": "codec",
    "camera.monochrome_output": "monochrome_output",
}
# Cancel asks the command to stop (SIGINT, like Ctrl+C) so it can close files
# and write its summary, then insists if it does not stop.
CANCEL_INTERRUPT_GRACE_MS = 15_000
CANCEL_TERMINATE_GRACE_MS = 5_000
# Session folders measured for the storage-per-day estimate.
STORAGE_ESTIMATE_SCAN_RUNS = 25

# Result buttons that only make sense with a given piece of hardware.
RESULT_BUTTON_HARDWARE = {
    "thermal": "thermal",
    "depth": "realsense",
    "depth_color": "realsense",
    "depth_data": "realsense",
    "metadata": "realsense",
    "manifest": "distributed",
    "calibration": "multimodal",
    "sync": "multimodal",
}
_LEVEL_LABEL_NAMES = {
    LEVEL_OK: "StatusGood",
    LEVEL_WARN: "StatusWarn",
    LEVEL_FAIL: "StatusFail",
    LEVEL_OFF: "StatusNeutral",
}
_LEVEL_CARD_TONES = {
    LEVEL_OK: "mint",
    LEVEL_WARN: "warn",
    LEVEL_FAIL: "fail",
    LEVEL_OFF: "neutral",
}

PAGE_WELCOME = 0
PAGE_HARDWARE = 1
PAGE_RGB = 2
PAGE_THERMAL = 3
PAGE_REALSENSE = 4
PAGE_DISTRIBUTED = 5
PAGE_EXPERIMENT = 6
PAGE_SUMMARY = 7


APP_STYLE = """
QWidget {
    color: #293936;
    font-family: "Avenir Next", "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 14px;
}
QMainWindow, QWizard, QDialog, QWidget#AppRoot { background: #f7f6f1; }
QWidget#ContentShell {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #faf9f5,
        stop: 0.58 #f6f7f2,
        stop: 1 #eef5f1
    );
}
QLabel { background: transparent; }
QLabel#PageLabel {
    color: #668078;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
}
QLabel#Subtitle { color: #61706c; font-size: 15px; }
QLabel#SectionTitle { color: #243633; font-size: 27px; font-weight: 600; }
QLabel#Eyebrow {
    color: #657a74;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#StatusGood {
    background: #e4f2e9;
    color: #27694f;
    border-radius: 9px;
    padding: 9px 12px;
    font-weight: 600;
}
QLabel#StatusWarn {
    background: #fff0d4;
    color: #8b5d17;
    border-radius: 9px;
    padding: 9px 12px;
    font-weight: 600;
}
QLabel#StatusFail {
    background: #fbe3df;
    color: #8f2f24;
    border-radius: 9px;
    padding: 9px 12px;
    font-weight: 600;
}
QLabel#StatusNeutral {
    background: #ecefeb;
    color: #52625e;
    border-radius: 9px;
    padding: 9px 12px;
    font-weight: 600;
}
QLabel#Brand {
    color: #2f695d;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#BrandSubtitle, QLabel#PrivacyNote {
    color: #71827c;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#ProfileBadge {
    background: #fff0bd;
    color: #694f13;
    border: 1px solid #f0d987;
    border-radius: 13px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
}
QLabel#Hint { color: #6b7773; font-size: 13px; }
QFrame#Sidebar {
    background: #e7f0eb;
    border: none;
    border-right: 1px solid #d4e2da;
}
QFrame#Card {
    background: #fffefb;
    border: 1px solid #e3e4dc;
    border-radius: 15px;
}
QFrame#Card[tone="peach"] { background: #fff0e7; border-color: #f2d9ca; }
QFrame#Card[tone="sun"] { background: #fff6d9; border-color: #efdfaa; }
QFrame#Card[tone="sky"] { background: #eaf4f6; border-color: #d3e5e8; }
QFrame#Card[tone="mint"] { background: #e9f3ee; border-color: #d2e4da; }
QFrame#Card[tone="fail"] { background: #fbe3df; border-color: #e9b9b1; }
QFrame#Card[tone="warn"] { background: #fff0d4; border-color: #ecd49b; }
QFrame#Card[tone="neutral"] { background: #f1f2ee; border-color: #dfe2dc; }
QFrame#Card QLabel[valueLabel="true"] {
    color: #273a36;
    font-size: 16px;
    font-weight: 600;
}
QFrame#Card QLabel[detailLabel="true"] { color: #4f605b; font-size: 13px; }
QFrame#Card[tone="fail"] QLabel[valueLabel="true"] { color: #8f2f24; }
QGroupBox {
    background: rgba(255, 254, 251, 238);
    border: 1px solid #e0e3da;
    border-radius: 14px;
    color: #314440;
    font-weight: 600;
    margin-top: 15px;
    padding: 20px 16px 14px 16px;
}
QGroupBox[tone="peach"] { background: #fff5ee; border-color: #f0ddd2; }
QGroupBox[tone="sun"] { background: #fff9e8; border-color: #eee2bd; }
QGroupBox[tone="sky"] { background: #f0f8f9; border-color: #d8e8ea; }
QGroupBox[tone="mint"] { background: #f0f7f3; border-color: #d7e7de; }
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 7px;
    background: transparent;
}
QListWidget {
    background: transparent;
    color: #49605a;
    border: none;
    outline: none;
    padding: 22px 0;
}
QListWidget::item {
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 12px 14px;
    margin: 2px 0;
}
QListWidget::item:hover { background: #f2f7f4; color: #315c52; }
QListWidget::item:selected {
    background: #fffefb;
    color: #2c675b;
    border: 1px solid #d5e4dc;
    font-weight: 600;
}
QPushButton {
    background: #e4eee9;
    color: #2d5b51;
    border: 1px solid #d1e1d9;
    border-radius: 10px;
    padding: 9px 15px;
    font-weight: 600;
}
QPushButton:hover { background: #d8e8e0; border-color: #bcd3c8; }
QPushButton:pressed { background: #cde0d7; }
QPushButton:disabled {
    background: #edf0ec;
    color: #a4aaa6;
    border-color: #e2e5e1;
}
QPushButton#Primary {
    background: #39786a;
    color: #ffffff;
    border-color: #39786a;
}
QPushButton#Primary:hover { background: #306b5f; border-color: #306b5f; }
QPushButton#Accent {
    background: #e5ad42;
    color: #4a3811;
    border-color: #e5ad42;
}
QPushButton#Accent:hover { background: #dca037; border-color: #dca037; }
QPushButton#Quiet {
    background: #fffefb;
    color: #52625e;
    border-color: #dcded7;
}
QPushButton#Quiet:hover { background: #f5f5ef; border-color: #cdd4cf; }
QPushButton:checked { background: #39786a; color: #ffffff; border-color: #39786a; }
QPushButton#Segment {
    background: #fffefb;
    color: #52625e;
    border: 1px solid #cfdcd5;
    border-radius: 0;
    padding: 6px 14px;
}
QPushButton#Segment:hover { background: #f2f7f4; }
QPushButton#Segment:checked { background: #39786a; color: #ffffff; border-color: #39786a; }
QPushButton#Segment[position="first"] { border-top-left-radius: 10px; border-bottom-left-radius: 10px; }
QPushButton#Segment[position="last"] {
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
    border-left: none;
}
QPushButton#Primary:disabled, QPushButton#Accent:disabled, QPushButton#Quiet:disabled {
    background: #edf0ec;
    color: #a4aaa6;
    border-color: #e2e5e1;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTimeEdit {
    background: #ffffff;
    border: 1px solid #ccd3ce;
    border-radius: 9px;
    padding: 8px 10px;
    selection-background-color: #b9d9cf;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTimeEdit:focus {
    border: 2px solid #4d8b7d;
    padding: 7px 9px;
}
QSpinBox, QDoubleSpinBox, QTimeEdit { min-width: 96px; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QTimeEdit:disabled {
    background: #f1f3f0;
    color: #8d9692;
}
/* The styled step buttons have no arrow images and render as blank bars, so
   they are hidden; values are typed, or changed with the arrow keys. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QTimeEdit::up-button, QTimeEdit::down-button { width: 0; border: none; }
QComboBox::drop-down { border: none; width: 24px; }
QCheckBox { spacing: 9px; color: #40524e; }
QCheckBox:disabled { color: #9aa39f; }
QCheckBox::indicator { width: 17px; height: 17px; }
QToolTip {
    background: #263b37;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 8px;
}
QTableWidget {
    background: #fffefb;
    alternate-background-color: #f5f8f5;
    border: 1px solid #dde2dc;
    border-radius: 11px;
    gridline-color: #e9ece7;
    selection-background-color: #fff0bd;
    selection-color: #30433f;
}
QHeaderView::section {
    background: #dfece6;
    color: #36554d;
    border: none;
    border-right: 1px solid #cfded7;
    padding: 9px 7px;
    font-weight: 600;
}
QTextEdit {
    background: #213431;
    color: #dceae5;
    border: 1px solid #314944;
    border-radius: 10px;
    padding: 8px;
    selection-background-color: #4a7e72;
    font-size: 12px;
}
QScrollArea#PageScroll, QScrollArea#PageScroll > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: transparent;
    width: 11px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #c6d3cd;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #aebfb7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent;
    height: 11px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #c6d3cd;
    min-width: 28px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QFrame#WelcomeCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #e8f3ed,
        stop: 0.58 #f5f5e8,
        stop: 1 #fff0dc
    );
    border: 1px solid #d7e4dc;
    border-radius: 18px;
}
QLabel#WelcomeHeroTitle {
    color: #27453e;
    font-size: 25px;
    font-weight: 600;
}
QFrame#StepCard {
    background: rgba(255, 255, 255, 180);
    border: 1px solid rgba(255, 255, 255, 210);
    border-radius: 11px;
}
QLabel#StepNumber {
    background: #e5ad42;
    color: #493810;
    border-radius: 12px;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    font-weight: 700;
}
QFrame#SummaryCard {
    background: #edf6f1;
    border: 1px solid #d3e5dc;
    border-radius: 14px;
}
QWizardPage { background: #f7f6f1; }
QWizard QPushButton { min-width: 88px; }
QWizard QLabel { background: transparent; }
"""


def _encode_benchmark_path() -> Path:
    """Where the settings page keeps the last encode benchmark run on this computer."""
    base = QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation)
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "BumbleBox" / "encode_benchmark_latest.json"


def _same_drive(first: Any, second: Any) -> bool:
    """Whether two folders are on the same drive, judged by their nearest existing parents."""

    def device(path: Any) -> int:
        probe = Path(str(path)).expanduser()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return probe.stat().st_dev

    try:
        return device(first) == device(second)
    except OSError:
        return False


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    section, _, key = dotted_key.partition(".")
    config.setdefault(section, {})[key] = value


def _get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    section, _, key = dotted_key.partition(".")
    values = config.get(section)
    return values.get(key, default) if isinstance(values, dict) else default


def _load_config_or_defaults(path: Path) -> dict[str, Any]:
    return load_config(path) if path.exists() else load_defaults()


def _set_combo_data(combo: QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _page_heading(title: str, subtitle: str) -> tuple[QLabel, QLabel]:
    heading = QLabel(title)
    heading.setObjectName("SectionTitle")
    description = QLabel(subtitle)
    description.setObjectName("Subtitle")
    description.setWordWrap(True)
    return heading, description


class _WheelOnlyWhenFocused(QObject):
    """Stops the mouse wheel from changing a field the pointer merely passes over.

    On a scrolling page, a spin box or drop-down under the pointer would
    otherwise take the wheel and change its value instead of scrolling.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Wheel and isinstance(watched, QWidget) and not watched.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(watched, event)


UNIT_NAMES = {"s": ("second", "seconds"), "min": ("minute", "minutes"), "h": ("hour", "hours"), "day": ("day", "days")}


class _TrimmedDoubleSpinBox(QDoubleSpinBox):
    """Shows 2 rather than 2.00, and 1.5 rather than 1.50."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")


class _UnitValueEdit(QWidget):
    """A number with its unit beside it, such as "2 hours".

    The config keeps a recording length in seconds and intervals in minutes;
    ``value()`` and ``setValue()`` speak that base unit, while the person
    types whichever unit suits. Choosing another unit keeps the number as
    typed ("2", then "hours", means two hours).

    Any number can be typed in any unit, because the number is usually typed
    before the unit is chosen: "12" must be allowed while the unit still says
    days. A value outside the allowed range is reported by ``problem()``
    instead of being quietly changed.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        units: tuple[tuple[str, float], ...],
        *,
        minimum: float,
        maximum: float,
        whole: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._units = units
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._whole = bool(whole)  # the base value must be a whole number (whole minutes)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.number = _TrimmedDoubleSpinBox()
        self.number.setDecimals(2)
        self.unit = QComboBox()
        for name, _size in units:
            self.unit.addItem(UNIT_NAMES.get(name, (name, name))[1], name)
        layout.addWidget(self.number)
        layout.addWidget(self.unit)
        layout.addStretch(1)
        self._apply_range()
        self.number.valueChanged.connect(self._number_changed)
        self.unit.currentIndexChanged.connect(self._unit_changed)

    def _apply_range(self) -> None:
        self.number.setRange(0.01, 1_000_000.0)
        self._name_units()

    def _name_units(self) -> None:
        """Say "1 day" and "2 days"."""
        one = abs(self.number.value() - 1.0) < 1e-9
        for index in range(self.unit.count()):
            names = UNIT_NAMES.get(str(self.unit.itemData(index)))
            if names is not None:
                self.unit.setItemText(index, names[0] if one else names[1])

    def _number_changed(self) -> None:
        self._name_units()
        self.changed.emit()

    def _unit_changed(self) -> None:
        self._apply_range()
        self.changed.emit()

    def value(self) -> float | int:
        """The amount in the base unit (the first unit given)."""
        amount = to_base(self.number.value(), str(self.unit.currentData()), self._units)
        if self._whole or abs(amount - round(amount)) < 1e-9:
            return int(round(amount))
        return round(amount, 3)

    def clamped_value(self) -> float | int:
        """``value()`` brought inside the allowed range."""
        amount = min(self._maximum, max(self._minimum, self.value()))
        return int(amount) if float(amount).is_integer() else amount

    def problem(self) -> str | None:
        """Why the typed value cannot be used, in the largest sensible unit, or ``None``."""
        amount = to_base(self.number.value(), str(self.unit.currentData()), self._units)
        if amount > self._maximum + 1e-9:
            number, unit = best_unit(self._maximum, self._units)
            return f"at most {_TrimmedDoubleSpinBox.textFromValue(self.number, number)} {UNIT_NAMES[unit][0 if number == 1 else 1]}"
        if amount < self._minimum - 1e-9:
            number, unit = best_unit(self._minimum, self._units)
            return f"at least {_TrimmedDoubleSpinBox.textFromValue(self.number, number)} {UNIT_NAMES[unit][0 if number == 1 else 1]}"
        return None

    def setValue(self, amount: Any) -> None:
        """Show ``amount`` (in the base unit) in the largest unit that keeps it a whole number."""
        number, unit = best_unit(amount, self._units)
        blocked = self.blockSignals(True)
        self.number.blockSignals(True)
        self.unit.blockSignals(True)
        try:
            index = self.unit.findData(unit)
            self.unit.setCurrentIndex(index if index >= 0 else 0)
            self._apply_range()
            self.number.setValue(number)
            self._name_units()
        finally:
            self.number.blockSignals(False)
            self.unit.blockSignals(False)
            self.blockSignals(blocked)
        self.changed.emit()

    def fields(self) -> tuple[QWidget, QWidget]:
        return self.number, self.unit


def _time_edit(value: Any) -> QTimeEdit:
    """A 24-hour HH:MM field holding ``value`` (or midnight if it is not a time)."""
    edit = QTimeEdit()
    edit.setDisplayFormat("HH:mm")
    try:
        minutes = parse_time_of_day(value)
    except ValueError:
        minutes = 0
    edit.setTime(QTime(minutes // 60, minutes % 60))
    return edit


class _FitScrollArea(QScrollArea):
    """A scroll area that asks for the height of its content.

    The stock size hint is capped at a fixed number of text lines, which would
    make the content scroll even when the window has room for all of it.
    """

    def sizeHint(self) -> QSize:
        content = self.widget()
        if content is None:
            return super().sizeHint()
        hint = content.sizeHint()
        height = hint.height()
        width = self.viewport().width()
        if width > 0 and content.hasHeightForWidth():
            # Wrapped text needs more or fewer lines depending on the width.
            height = content.heightForWidth(width)
        return QSize(hint.width(), height + 2 * self.frameWidth())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # When the content grows or shrinks (a status text changed), ask the
        # page to lay this area out again at its new height.
        if watched is self.widget() and event.type() in (QEvent.LayoutRequest, QEvent.Resize):
            self.updateGeometry()
        return super().eventFilter(watched, event)


class _RunHistorySignals(QObject):
    # generation, the shown camera's runs, error, {camera index: runs} for the other camera
    finished = pyqtSignal(int, object, object, object)


def _is_finished_recording(payload: dict[str, Any]) -> bool:
    return bool(payload.get("success", False)) and str(payload.get("mode")) in RECORDING_RUN_MODES


class _RunHistoryWorker(QRunnable):
    """Reads recent run summaries off the GUI thread.

    ``targets`` is ``(camera index, data folder, session tag)`` for each camera
    of this box. Two cameras usually share a data folder, so each folder is
    scanned once and the runs are split by the tag in their summaries.
    """

    def __init__(
        self,
        generation: int,
        targets: list[tuple[int, str, str]],
        active_index: int = 1,
        limit: int = 60,
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.targets = [(int(index), str(root), str(tag)) for index, root, tag in targets]
        self.active_index = int(active_index)
        self.limit = int(limit)
        self.signals = _RunHistorySignals()

    def _emit_finished(
        self,
        payloads: list[dict[str, Any]],
        error: str | None,
        others: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        try:
            self.signals.finished.emit(self.generation, payloads, error, others or {})
        except RuntimeError:
            # The application window may close while a removable-drive scan is finishing.
            pass

    @staticmethod
    def _load_payload(record: Any) -> dict[str, Any] | None:
        try:
            payload = load_run_summary(record.summary_path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        payload = dict(payload)
        payload["_summary_path"] = str(record.summary_path.resolve())
        recorded_session_text = str(payload.get("session_dir") or "").strip()
        recorded_session_dir = Path(recorded_session_text).expanduser()
        payload["_session_dir_local"] = str(
            recorded_session_dir.resolve()
            if recorded_session_text and recorded_session_dir.exists()
            else record.summary_path.parent.resolve()
        )
        return payload

    def run(self) -> None:
        by_camera: dict[int, list[dict[str, Any]]] = {index: [] for index, _, _ in self.targets}
        measured = {index: 0 for index in by_camera}
        folders: dict[str, dict[str, int]] = {}
        for index, root, tag in self.targets:
            folders.setdefault(root, {})[tag] = index
        try:
            for root, tags in folders.items():
                for record in list_recent_run_records(root, limit=self.limit * len(tags)):
                    index = tags.get(record.camera_session_tag)
                    if index is None or len(by_camera[index]) >= self.limit:
                        continue
                    payload = self._load_payload(record)
                    if payload is None:
                        continue
                    if measured[index] < STORAGE_ESTIMATE_SCAN_RUNS and _is_finished_recording(payload):
                        measured[index] += 1
                        payload["_session_bytes"] = session_size_bytes(payload["_session_dir_local"])
                        payload["_recording_signature"] = signature_from_snapshot(payload.get("config_snapshot_path"))
                    by_camera[index].append(payload)
        except Exception as exc:
            self._emit_finished([], str(exc))
            return
        active = by_camera.pop(self.active_index, [])
        self._emit_finished(active, None, by_camera)


class _BoxStatusSignals(QObject):
    finished = pyqtSignal(int, object)


class _BoxStatusWorker(QRunnable):
    """Checks storage and the schedule off the GUI thread.

    A stalled drive or a slow answer from systemd must not freeze the window.
    """

    def __init__(self, generation: int, targets: list[tuple[int, dict[str, Any], Path]]) -> None:
        super().__init__()
        self.generation = int(generation)
        self.targets = targets  # (camera index, config, config path)
        self.signals = _BoxStatusSignals()

    def run(self) -> None:
        statuses: dict[int, Any] = {}
        for index, config, config_path in self.targets:
            try:
                statuses[index] = collect_box_status(config, config_path)
            except Exception:  # pragma: no cover - collect_box_status guards its parts
                continue
        try:
            self.signals.finished.emit(self.generation, statuses)
        except RuntimeError:
            # The window may close while a check is finishing.
            pass


class WelcomePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Welcome to BumbleBox")
        self.setSubTitle(
            "This wizard records the hardware you are actually using and shows only the setup steps that apply."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 20, 18, 14)
        layout.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("WelcomeCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 24, 24, 24)
        hero_layout.setSpacing(12)

        eyebrow = QLabel("QUICK, GUIDED SETUP")
        eyebrow.setObjectName("Eyebrow")
        hero_layout.addWidget(eyebrow)
        hero_title = QLabel("A clear start for every BumbleBox")
        hero_title.setObjectName("WelcomeHeroTitle")
        hero_title.setWordWrap(True)
        hero_layout.addWidget(hero_title)
        hero_copy = QLabel(
            "Choose only the devices in this setup. BumbleBox will keep the everyday workspace focused on what you use."
        )
        hero_copy.setObjectName("Subtitle")
        hero_copy.setWordWrap(True)
        hero_layout.addWidget(hero_copy)

        steps = QHBoxLayout()
        steps.setSpacing(10)
        for number, title in (
            ("1", "Choose hardware"),
            ("2", "Set capture basics"),
            ("3", "Review and save"),
        ):
            step = QFrame()
            step.setObjectName("StepCard")
            step_layout = QHBoxLayout(step)
            step_layout.setContentsMargins(10, 10, 10, 10)
            step_layout.setSpacing(8)
            number_label = QLabel(number)
            number_label.setObjectName("StepNumber")
            number_label.setAlignment(Qt.AlignCenter)
            title_label = QLabel(title)
            title_label.setWordWrap(True)
            step_layout.addWidget(number_label)
            step_layout.addWidget(title_label, 1)
            steps.addWidget(step, 1)
        hero_layout.addLayout(steps)
        layout.addWidget(hero)

        note = QLabel(
            "You can rerun this wizard at any time. It does not erase tracking parameters or previous data."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

    def nextId(self) -> int:
        return PAGE_HARDWARE


class HardwarePage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Choose a hardware profile")
        self.setSubTitle("Optional setup pages are skipped when their device is not part of this BumbleBox.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        self.profile = QComboBox()
        for key, profile in HARDWARE_PROFILES.items():
            self.profile.addItem(profile.label, key)
        _set_combo_data(self.profile, str(config.get("setup", {}).get("hardware_profile", "custom")))
        form.addRow("Hardware profile", self.profile)
        layout.addLayout(form)

        self.description = QLabel()
        self.description.setObjectName("Hint")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.thermal = QCheckBox("Use a PureThermal / Lepton camera")
        self.thermal.setChecked(bool(config.get("thermal", {}).get("enabled", False)))
        self.realsense = QCheckBox("Use a RealSense depth camera")
        self.realsense.setChecked(bool(config.get("realsense", {}).get("enabled", False)))
        self.second_pi = QCheckBox("Use a second Raspberry Pi for simultaneous capture")
        self.second_pi.setChecked(bool(config.get("distributed_capture", {}).get("enabled", False)))
        layout.addWidget(self.thermal)
        layout.addWidget(self.realsense)
        layout.addWidget(self.second_pi)
        two_cameras_hint = QLabel(
            "Two Raspberry Pi cameras on this one Pi? Finish this setup for the first camera, then add "
            "the second on the Hardware page."
        )
        two_cameras_hint.setObjectName("Hint")
        two_cameras_hint.setWordWrap(True)
        layout.addWidget(two_cameras_hint)
        layout.addStretch(1)
        self.profile.currentIndexChanged.connect(self._profile_changed)
        self._profile_changed()

    def _profile_changed(self) -> None:
        key = str(self.profile.currentData())
        profile = HARDWARE_PROFILES[key]
        self.description.setText(profile.description)
        custom = key == "custom"
        self.thermal.setEnabled(custom)
        self.realsense.setEnabled(custom)
        if not custom:
            self.thermal.setChecked(profile.thermal_enabled)
            self.realsense.setChecked(profile.realsense_enabled)

    def nextId(self) -> int:
        return PAGE_RGB


class RgbPage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure the primary camera")
        self.setSubTitle("Named profiles keep HQ and OwlSight comparisons repeatable.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        self.profile = QComboBox()
        for key, profile in CAMERA_PROFILES.items():
            self.profile.addItem(profile.label, key)
        _set_combo_data(self.profile, str(config.get("camera", {}).get("profile", "custom")))
        form.addRow("Camera profile", self.profile)
        layout.addLayout(form)
        self.description = QLabel()
        self.description.setObjectName("Hint")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        visible_note = QLabel(
            "OwlSight uses visible illumination because the stock OV64A40 camera is treated as IR-cut. "
            "The HQ NoIR reference profile keeps the established infrared workflow."
        )
        visible_note.setObjectName("Hint")
        visible_note.setWordWrap(True)
        layout.addWidget(visible_note)
        layout.addStretch(1)
        self.profile.currentIndexChanged.connect(self._changed)
        self._changed()

    def _changed(self) -> None:
        profile = CAMERA_PROFILES[str(self.profile.currentData())]
        self.description.setText(profile.description)

    def nextId(self) -> int:
        wizard = self.wizard()
        if wizard.uses_thermal():
            return PAGE_THERMAL
        if wizard.uses_realsense():
            return PAGE_REALSENSE
        return PAGE_DISTRIBUTED if wizard.uses_second_pi() else PAGE_EXPERIMENT


class ThermalPage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure thermal capture")
        self.setSubTitle("Auto discovery is recommended; a successful check can later pin the stable V4L by-id path.")
        form = QFormLayout(self)
        form.setContentsMargins(18, 18, 18, 14)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        thermal = config.get("thermal", {})
        self.device = QLineEdit(str(thermal.get("device_path", "auto") or "auto"))
        self.width = QSpinBox()
        self.width.setRange(1, 4096)
        self.width.setValue(int(thermal.get("width", 160)))
        self.height = QSpinBox()
        self.height.setRange(1, 4096)
        self.height.setValue(int(thermal.get("height", 120)))
        form.addRow("Device path", self.device)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)

    def nextId(self) -> int:
        if self.wizard().uses_realsense():
            return PAGE_REALSENSE
        return PAGE_DISTRIBUTED if self.wizard().uses_second_pi() else PAGE_EXPERIMENT


class RealSensePage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure RealSense depth")
        self.setSubTitle(
            "These are initial D405 test settings. The hardware check will confirm which exact profiles the connected camera accepts."
        )
        form = QFormLayout(self)
        form.setContentsMargins(18, 18, 18, 14)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        depth = config.get("realsense", {})
        self.serial = QLineEdit(str(depth.get("device_serial", "auto") or "auto"))
        self.depth_width = QSpinBox()
        self.depth_width.setRange(1, 4096)
        self.depth_width.setValue(int(depth.get("depth_width", 848)))
        self.depth_height = QSpinBox()
        self.depth_height.setRange(1, 4096)
        self.depth_height.setValue(int(depth.get("depth_height", 480)))
        self.color_width = QSpinBox()
        self.color_width.setRange(1, 4096)
        self.color_width.setValue(int(depth.get("color_width", 848)))
        self.color_height = QSpinBox()
        self.color_height.setRange(1, 4096)
        self.color_height.setValue(int(depth.get("color_height", 480)))
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(int(depth.get("fps", 30)))
        form.addRow("Device serial", self.serial)
        form.addRow("Depth width", self.depth_width)
        form.addRow("Depth height", self.depth_height)
        form.addRow("Color width", self.color_width)
        form.addRow("Color height", self.color_height)
        form.addRow("Frames per second", self.fps)

    def nextId(self) -> int:
        return PAGE_DISTRIBUTED if self.wizard().uses_second_pi() else PAGE_EXPERIMENT


class DistributedCapturePage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure the second capture Pi")
        self.setSubTitle(
            "Each camera records locally. Both Pis prepare first, then begin from one shared UTC deadline."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)
        distributed = config.get("distributed_capture", {})
        remote = next(
            (item for item in distributed.get("nodes", []) if not bool(item.get("local", False))),
            {},
        )
        fleet_ssh = config.get("fleet", {}).get("ssh", {})
        default_user = str(remote.get("user") or fleet_ssh.get("user") or "pi")

        connection = QFormLayout()
        connection.setHorizontalSpacing(18)
        connection.setVerticalSpacing(10)
        self.host = QLineEdit(str(remote.get("host") or "bumblebox-02.local"))
        self.user = QLineEdit(default_user)
        self.repo_path = QLineEdit(
            str(remote.get("repo_path") or f"/home/{default_user}/Desktop/BumbleBox")
        )
        self.config_path = QLineEdit(
            str(
                remote.get("config_path")
                or f"/home/{default_user}/Desktop/BumbleBox/bumblebox_v2/config.yaml"
            )
        )
        self.data_root = QLineEdit(
            str(remote.get("data_root") or f"/home/{default_user}/Desktop/BumbleBoxData")
        )
        connection.addRow("Second Pi hostname or IP", self.host)
        connection.addRow("SSH username", self.user)
        connection.addRow("BumbleBox repository", self.repo_path)
        connection.addRow("Worker config file", self.config_path)
        connection.addRow("Worker data folder", self.data_root)
        layout.addLayout(connection)

        assignments = QGroupBox("Camera assignments")
        assignment_form = QFormLayout(assignments)
        assignment_form.setHorizontalSpacing(18)
        current_owners = {}
        for item in distributed.get("nodes", []):
            owner = "primary" if bool(item.get("local", False)) else "worker"
            for sensor in item.get("sensors", []):
                current_owners[str(sensor).lower()] = owner

        has_saved_remote = bool(remote)

        def location_combo(sensor: str, default_owner: str = "primary") -> QComboBox:
            combo = QComboBox()
            combo.addItem("Primary Pi", "primary")
            combo.addItem("Second Pi", "worker")
            owner = current_owners.get(sensor, default_owner) if has_saved_remote else default_owner
            _set_combo_data(combo, owner)
            return combo

        self.rgb_location = location_combo("rgb")
        self.thermal_location = location_combo("thermal")
        self.realsense_location = location_combo("realsense", "worker")
        assignment_form.addRow("RGB / OwlSight", self.rgb_location)
        assignment_form.addRow("Thermal", self.thermal_location)
        assignment_form.addRow("RealSense", self.realsense_location)
        layout.addWidget(assignments)
        self.transfer_after_capture = QCheckBox(
            "Copy second-Pi outputs back to the primary Pi after each recording"
        )
        self.transfer_after_capture.setChecked(bool(distributed.get("transfer_after_capture", False)))
        layout.addWidget(self.transfer_after_capture)
        note = QLabel(
            "Recommended first layout: OwlSight RGB + thermal on the primary Pi, RealSense on the second Pi. "
            "You can change any assignment later without changing recording commands."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

    def initializePage(self) -> None:
        wizard = self.wizard()
        self.thermal_location.setEnabled(wizard.uses_thermal())
        self.realsense_location.setEnabled(wizard.uses_realsense())

    def nextId(self) -> int:
        return PAGE_EXPERIMENT

    def validatePage(self) -> bool:
        if not self.host.text().strip() or not self.user.text().strip():
            QMessageBox.warning(
                self,
                "Second Pi details required",
                "Enter the second Pi hostname/IP and SSH username.",
            )
            return False
        wizard = self.wizard()
        locations = [str(self.rgb_location.currentData())]
        if wizard.uses_thermal():
            locations.append(str(self.thermal_location.currentData()))
        if wizard.uses_realsense():
            locations.append(str(self.realsense_location.currentData()))
        if "worker" not in locations:
            QMessageBox.warning(
                self,
                "No camera assigned to second Pi",
                "Assign at least one enabled camera to the second Pi, or go back and disable two-Pi capture.",
            )
            return False
        return True


class ExperimentPage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Set experiment basics")
        self.setSubTitle("These are the settings needed for normal recording; specialized controls remain outside the wizard.")
        form = QFormLayout(self)
        form.setContentsMargins(18, 18, 18, 14)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        self.colony = QLineEdit(str(config.get("system", {}).get("colony_id", "01")))
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.data_root = QLineEdit(str(config.get("system", {}).get("data_root", "/mnt/bumblebox/data")))
        browse = QPushButton("Choose")
        browse.setObjectName("Quiet")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.data_root, 1)
        path_layout.addWidget(browse)
        self.mode = QComboBox()
        for value, label in (
            ("record_only", "Record only"),
            ("record_and_track", "Record and track"),
            ("track_only", "Track only"),
            ("mixed_schedule", "Mixed schedule"),
        ):
            self.mode.addItem(label, value)
        _set_combo_data(self.mode, str(config.get("pipeline", {}).get("mode", "record_and_track")))
        self.recording_seconds = _UnitValueEdit(DURATION_UNITS, minimum=1, maximum=86400)
        self.recording_seconds.setValue(config.get("capture", {}).get("recording_seconds", 20))
        self.interval_minutes = _UnitValueEdit(INTERVAL_UNITS, minimum=1, maximum=MINUTES_PER_DAY, whole=True)
        self.interval_minutes.setValue(config.get("capture", {}).get("record_interval_minutes", 30))
        self.start_time = _time_edit(config.get("scheduling", {}).get("start_time", DEFAULT_START_TIME))
        self.start_time.setToolTip(
            "Recordings are counted from this time of day. For one recording a day, this is when it starts."
        )
        form.addRow("Colony ID", self.colony)
        form.addRow("Data folder", path_row)
        form.addRow("Default run mode", self.mode)
        form.addRow("Record for", self.recording_seconds)
        form.addRow("Record every", self.interval_minutes)
        form.addRow("Starting at", self.start_time)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose BumbleBox data folder", self.data_root.text())
        if selected:
            self.data_root.setText(selected)

    def nextId(self) -> int:
        return PAGE_SUMMARY


class SummaryPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Review setup")
        self.setSubTitle("Finish writes these choices to the BumbleBox configuration.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        card = QFrame()
        card.setObjectName("SummaryCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 20)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        card_layout.addWidget(self.summary)
        layout.addWidget(card)
        layout.addStretch(1)

    def initializePage(self) -> None:
        wizard = self.wizard()
        hardware = HARDWARE_PROFILES[str(wizard.hardware_page.profile.currentData())].label
        camera = CAMERA_PROFILES[str(wizard.rgb_page.profile.currentData())].label
        optional = []
        if wizard.uses_thermal():
            optional.append("PureThermal / Lepton")
        if wizard.uses_realsense():
            optional.append("RealSense depth")
        capture_layout = "One Pi"
        if wizard.uses_second_pi():
            capture_layout = (
                "Two Pis; "
                f"RGB={wizard.distributed_page.rgb_location.currentText()}, "
                f"thermal={wizard.distributed_page.thermal_location.currentText() if wizard.uses_thermal() else 'off'}, "
                f"depth={wizard.distributed_page.realsense_location.currentText() if wizard.uses_realsense() else 'off'}"
            )
        self.summary.setText(
            f"<b>Hardware profile:</b> {hardware}<br>"
            f"<b>Primary camera:</b> {camera}<br>"
            f"<b>Optional devices:</b> {', '.join(optional) if optional else 'None'}<br>"
            f"<b>Capture layout:</b> {capture_layout}<br>"
            f"<b>Colony:</b> {wizard.experiment_page.colony.text().strip()}<br>"
            f"<b>Data folder:</b> {wizard.experiment_page.data_root.text().strip()}<br>"
            f"<b>Run mode:</b> {wizard.experiment_page.mode.currentData()}"
        )

    def nextId(self) -> int:
        return -1


class BumbleBoxSetupWizard(QWizard):
    def __init__(self, config_path: Path, config: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.config = config
        self.setWindowTitle("BumbleBox Setup Wizard")
        self.setObjectName("SetupWizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.resize(760, 600)

        self.welcome_page = WelcomePage()
        self.hardware_page = HardwarePage(config)
        self.rgb_page = RgbPage(config)
        self.thermal_page = ThermalPage(config)
        self.realsense_page = RealSensePage(config)
        self.distributed_page = DistributedCapturePage(config)
        self.experiment_page = ExperimentPage(config)
        self.summary_page = SummaryPage()
        self.setPage(PAGE_WELCOME, self.welcome_page)
        self.setPage(PAGE_HARDWARE, self.hardware_page)
        self.setPage(PAGE_RGB, self.rgb_page)
        self.setPage(PAGE_THERMAL, self.thermal_page)
        self.setPage(PAGE_REALSENSE, self.realsense_page)
        self.setPage(PAGE_DISTRIBUTED, self.distributed_page)
        self.setPage(PAGE_EXPERIMENT, self.experiment_page)
        self.setPage(PAGE_SUMMARY, self.summary_page)
        self.setStartId(PAGE_WELCOME)
        self.button(QWizard.FinishButton).setText("Save setup")
        self.button(QWizard.FinishButton).setObjectName("Primary")

    def uses_thermal(self) -> bool:
        return self.hardware_page.thermal.isChecked()

    def uses_realsense(self) -> bool:
        return self.hardware_page.realsense.isChecked()

    def uses_second_pi(self) -> bool:
        return self.hardware_page.second_pi.isChecked()

    def accept(self) -> None:
        profile_key = str(self.hardware_page.profile.currentData())
        updated = apply_hardware_profile(self.config, profile_key)
        if profile_key == "custom":
            updated.setdefault("thermal", {})["enabled"] = self.uses_thermal()
            updated.setdefault("realsense", {})["enabled"] = self.uses_realsense()

        camera_profile = str(self.rgb_page.profile.currentData())
        updated.setdefault("camera", {})["profile"] = camera_profile
        updated = apply_camera_profile(updated, camera_profile)

        thermal = updated.setdefault("thermal", {})
        thermal["enabled"] = self.uses_thermal()
        thermal["device_path"] = self.thermal_page.device.text().strip() or "auto"
        thermal["width"] = self.thermal_page.width.value()
        thermal["height"] = self.thermal_page.height.value()

        realsense = updated.setdefault("realsense", {})
        realsense["enabled"] = self.uses_realsense()
        realsense["device_serial"] = self.realsense_page.serial.text().strip() or "auto"
        realsense["depth_width"] = self.realsense_page.depth_width.value()
        realsense["depth_height"] = self.realsense_page.depth_height.value()
        realsense["color_width"] = self.realsense_page.color_width.value()
        realsense["color_height"] = self.realsense_page.color_height.value()
        realsense["fps"] = self.realsense_page.fps.value()

        distributed = updated.setdefault("distributed_capture", {})
        distributed["enabled"] = self.uses_second_pi()
        distributed["role"] = "controller" if self.uses_second_pi() else "standalone"
        distributed["controller_node"] = "primary"
        distributed["transfer_after_capture"] = self.distributed_page.transfer_after_capture.isChecked()
        primary_sensors = []
        worker_sensors = []
        locations = {"rgb": str(self.distributed_page.rgb_location.currentData())}
        if self.uses_thermal():
            locations["thermal"] = str(self.distributed_page.thermal_location.currentData())
        if self.uses_realsense():
            locations["realsense"] = str(self.distributed_page.realsense_location.currentData())
        for sensor, owner in locations.items():
            (worker_sensors if owner == "worker" else primary_sensors).append(sensor)
        if not self.uses_second_pi():
            primary_sensors = list(locations)
        worker_user = self.distributed_page.user.text().strip() or "pi"
        distributed["nodes"] = [
            {
                "name": "primary",
                "host": "localhost",
                "local": True,
                "enabled": True,
                "sensors": primary_sensors,
                "user": None,
                "port": 22,
                "repo_path": None,
                "config_path": None,
                "data_root": None,
            }
        ]
        had_remote_node = any(
            not bool(node.get("local", False))
            for node in self.config.get("distributed_capture", {}).get("nodes", [])
        )
        if self.uses_second_pi() or had_remote_node:
            distributed["nodes"].append(
                {
                    "name": "worker",
                    "host": self.distributed_page.host.text().strip(),
                    "local": False,
                    "enabled": self.uses_second_pi(),
                    "sensors": worker_sensors,
                    "user": worker_user,
                    "port": 22,
                    "repo_path": self.distributed_page.repo_path.text().strip() or None,
                    "config_path": self.distributed_page.config_path.text().strip() or None,
                    "data_root": self.distributed_page.data_root.text().strip() or None,
                }
            )

        updated.setdefault("system", {})["colony_id"] = self.experiment_page.colony.text().strip() or "01"
        updated["system"]["data_root"] = self.experiment_page.data_root.text().strip()
        updated.setdefault("pipeline", {})["mode"] = str(self.experiment_page.mode.currentData())
        updated.setdefault("capture", {})["recording_seconds"] = self.experiment_page.recording_seconds.clamped_value()
        updated["capture"]["record_interval_minutes"] = int(self.experiment_page.interval_minutes.clamped_value())
        updated.setdefault("scheduling", {})["start_time"] = self.experiment_page.start_time.time().toString("HH:mm")
        setup = updated.setdefault("setup", {})
        setup["completed"] = True
        setup["hardware_profile"] = profile_key
        setup["completed_at"] = datetime.now().astimezone().isoformat()

        try:
            save_config(self.config_path, updated)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save setup", str(exc))
            return
        self.config = updated
        super().accept()


class SecondCameraDialog(QDialog):
    """Asks the two things that differ for a second camera on the same Pi."""

    def __init__(self, parent: QWidget | None, *, detected: list[DetectedCamera], colony_id: str, data_root: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add a second camera")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        intro = QLabel(
            "Camera 2 will record alongside Camera 1 on this Pi. It starts with Camera 1's recording "
            "settings and its own schedule, which stays off until you start it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        if len(detected) >= 2:
            found = "Cameras connected now: " + "; ".join(camera.description for camera in detected) + "."
            found += " Each camera's settings will be pinned to one of them."
        else:
            found = (
                f"{len(detected)} camera(s) detected right now. You can add Camera 2 anyway and connect it "
                "later; press Pin Cameras on the Hardware page once both are connected."
            )
        self.detected_label = QLabel(found)
        self.detected_label.setObjectName("Hint")
        self.detected_label.setWordWrap(True)
        layout.addWidget(self.detected_label)

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        self.colony_id = QLineEdit(colony_id)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.data_root = QLineEdit(data_root)
        browse = QPushButton("Choose")
        browse.setObjectName("Quiet")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.data_root, 1)
        path_layout.addWidget(browse)
        form.addRow("Colony ID for Camera 2", self.colony_id)
        form.addRow("Data folder for Camera 2", path_row)
        layout.addLayout(form)
        folder_hint = QLabel(
            "The same folder as Camera 1 is fine: Camera 2's sessions are named ..._cam2_..., so the "
            "two cameras' files cannot be confused."
        )
        folder_hint.setObjectName("Hint")
        folder_hint.setWordWrap(True)
        layout.addWidget(folder_hint)

        buttons = QDialogButtonBox()
        add = buttons.addButton("Add Camera 2", QDialogButtonBox.AcceptRole)
        add.setObjectName("Primary")
        buttons.addButton("Cancel", QDialogButtonBox.RejectRole).setObjectName("Quiet")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose Camera 2's data folder", self.data_root.text())
        if selected:
            self.data_root.setText(selected)


class BumbleBoxQtGUI(QMainWindow):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        # A Pi can have two cameras, each with its own config file. The window
        # shows one at a time: config_path and config are the shown camera's.
        self.primary_config_path = config_path
        self.config_path = config_path
        self.config = _load_config_or_defaults(config_path)
        self.camera_index = 1
        self.cameras: list[CameraEntry] = [CameraEntry(1, config_path, self.config)]
        self._camera_statuses: dict[int, BoxStatus] = {}
        self._other_camera_payloads: dict[int, list[dict[str, Any]]] = {}
        self._command_on_success: Any = None
        self.command_queue: list[list[str]] = []
        self.command_success_message: tuple[str, str] | None = None
        self._run_history_generation = 0
        self._run_history_loading = False
        self._run_history_refresh_pending = False
        self._run_history_payloads: list[dict[str, Any]] = []
        self._selected_run_payload: dict[str, Any] | None = None
        self._run_history_workers: dict[int, _RunHistoryWorker] = {}
        self._run_history_error: str | None = None
        self._box_status: BoxStatus | None = None
        self._box_status_generation = 0
        self._box_status_loading = False
        self._box_status_refresh_pending = False
        self._box_status_workers: dict[int, _BoxStatusWorker] = {}
        self._last_scheduled_run_seen: datetime | None = None
        self._current_command: list[str] = []
        self._command_output: list[str] = []
        self._cancel_requested = False
        self._cancelled_steps = 0
        self._cancel_stage = 0
        self._storage_estimate: StorageEstimate | None = None
        self.encode_benchmark_path = _encode_benchmark_path()
        self._settings_loading = False
        self._settings_saved_values: dict[str, Any] | None = None
        self._settings_profile_override: str | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(self._escalate_cancel)
        self._settings_preview_timer = QTimer(self)
        self._settings_preview_timer.setSingleShot(True)
        self._settings_preview_timer.timeout.connect(self._refresh_settings_preview)
        self.setWindowTitle("BumbleBox")
        self.resize(1180, 780)
        self.setMinimumSize(940, 660)
        self._build_ui()
        self.refresh_from_config()
        QTimer.singleShot(100, self.refresh_results)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self.refresh_box_status)
        self._status_timer.start()
        if not bool(self.config.get("setup", {}).get("completed", False)):
            QTimer.singleShot(0, self.open_setup_wizard)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(224)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 30, 18, 20)
        side_layout.setSpacing(2)
        brand = QLabel("BUMBLEBOX")
        brand.setObjectName("Brand")
        side_layout.addWidget(brand)
        sub = QLabel("COLONY IMAGING")
        sub.setObjectName("BrandSubtitle")
        side_layout.addWidget(sub)
        self.navigation = QListWidget()
        self.navigation.setSpacing(3)
        for title in NAV_TITLES:
            self.navigation.addItem(QListWidgetItem(title))
        self.navigation.currentRowChanged.connect(self._navigate)
        side_layout.addWidget(self.navigation, 1)
        version = QLabel("LOCAL / PRIVATE / YOUR DATA")
        version.setObjectName("PrivacyNote")
        side_layout.addWidget(version)
        layout.addWidget(sidebar)

        content = QWidget()
        content.setObjectName("ContentShell")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(38, 28, 38, 30)
        content_layout.setSpacing(14)
        header = QHBoxLayout()
        self.page_title = QLabel("OVERVIEW")
        self.page_title.setObjectName("PageLabel")
        header.addWidget(self.page_title)
        header.addStretch(1)
        # Shown only on a Pi with two cameras: picks which camera every page shows.
        self.camera_switch = QWidget()
        switch_layout = QHBoxLayout(self.camera_switch)
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_layout.setSpacing(0)
        self.camera_button_group = QButtonGroup(self)
        self.camera_button_group.setExclusive(True)
        self.camera_buttons: dict[int, QPushButton] = {}
        for index, position in ((1, "first"), (2, "last")):
            button = QPushButton(f"Camera {index}")
            button.setObjectName("Segment")
            button.setProperty("position", position)
            button.setCheckable(True)
            button.setToolTip("Show this camera's status, recordings and settings.")
            button.clicked.connect(lambda _checked=False, selected=index: self._select_camera(selected))
            self.camera_button_group.addButton(button, index)
            self.camera_buttons[index] = button
            switch_layout.addWidget(button)
        self.camera_switch.setVisible(False)
        header.addWidget(self.camera_switch)
        header.addSpacing(10)
        self.profile_badge = QLabel()
        self.profile_badge.setObjectName("ProfileBadge")
        header.addWidget(self.profile_badge)
        content_layout.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_overview_page())
        self.pages.addWidget(self._build_run_page())
        self.pages.addWidget(self._build_results_page())
        self.pages.addWidget(self._build_hardware_page())
        self.pages.addWidget(self._build_settings_page())
        self.pages.addWidget(self._build_advanced_page())
        content_layout.addWidget(self.pages, 1)
        layout.addWidget(content, 1)
        self.navigation.setCurrentRow(0)

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Experiment readiness",
            "A concise view of this BumbleBox. Device-specific setup stays hidden unless the device is enabled.",
        )
        self.live_monitor_button = QPushButton("Open Live Camera Monitor")
        self.live_monitor_button.setObjectName("Primary")
        self.live_monitor_button.setToolTip(
            "See what the cameras see, live, without saving anything. Close the monitor before recording."
        )
        self.live_monitor_button.clicked.connect(self._open_live_monitor)
        heading_row = QHBoxLayout()
        heading_row.addWidget(heading, 1)
        heading_row.addWidget(self.live_monitor_button, 0, Qt.AlignVCenter)
        layout.addLayout(heading_row)
        layout.addWidget(subtitle)
        self.overview_subtitle = subtitle
        self.cameras_strip = QFrame()
        self.cameras_strip.setObjectName("Card")
        self.cameras_strip.setProperty("tone", "neutral")
        strip_layout = QVBoxLayout(self.cameras_strip)
        strip_layout.setContentsMargins(17, 13, 17, 13)
        strip_layout.setSpacing(8)
        strip_title = QLabel("BOTH CAMERAS")
        strip_title.setObjectName("Eyebrow")
        strip_layout.addWidget(strip_title)
        strip_row = QHBoxLayout()
        strip_row.setSpacing(10)
        self.camera_status_labels: dict[int, QLabel] = {}
        for index in (1, 2):
            label = QLabel("")
            label.setObjectName("StatusNeutral")
            label.setWordWrap(True)
            self.camera_status_labels[index] = label
            strip_row.addWidget(label, 1)
        strip_layout.addLayout(strip_row)
        self.cameras_strip.setVisible(False)
        layout.addWidget(self.cameras_strip)
        self.setup_status = QLabel()
        self.setup_status.setWordWrap(True)
        layout.addWidget(self.setup_status)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        self.camera_card = self._status_card("Primary camera", "", tone="peach")
        self.thermal_card = self._status_card("Thermal", "", tone="sun")
        self.realsense_card = self._status_card("RealSense", "", tone="sky")
        self.storage_card = self._status_card("Data storage", "", tone="mint")
        self.data_drive_button = QPushButton("Data Drive...")
        self.data_drive_button.setObjectName("Quiet")
        self.data_drive_button.setToolTip(
            "Mount a drive at the data folder, choose whether it is mounted again after a restart, "
            "or remove it safely."
        )
        self.data_drive_button.clicked.connect(self._open_data_drive)
        # Above the stretch that ends the card, so it sits under the status text.
        self.storage_card.layout().insertWidget(self.storage_card.layout().count() - 1, self.data_drive_button, 0, Qt.AlignLeft)
        for column, card in enumerate((self.camera_card, self.thermal_card, self.realsense_card, self.storage_card)):
            cards.addWidget(card, 0, column)
            cards.setColumnStretch(column, 1)
        cards.setVerticalSpacing(12)
        self.schedule_card = self._status_card("Automated recording", "Checking...", tone="neutral")
        self.latest_card = self._status_card("Latest recording", "Loading...", tone="neutral")
        cards.addWidget(self.schedule_card, 1, 0, 1, 2)
        cards.addWidget(self.latest_card, 1, 2, 1, 2)
        layout.addLayout(cards)

        actions = QHBoxLayout()
        setup = QPushButton("Run Setup Wizard")
        setup.setObjectName("Accent")
        setup.clicked.connect(self.open_setup_wizard)
        doctor = QPushButton("Run Health Check")
        doctor.clicked.connect(lambda: self.run_bbx_command(["doctor"]))
        storage = QPushButton("Check Storage")
        storage.setObjectName("Quiet")
        storage.clicked.connect(lambda: self.run_bbx_command(["storage", "status"]))
        actions.addWidget(setup)
        actions.addWidget(doctor)
        actions.addWidget(storage)
        actions.addStretch(1)
        layout.addLayout(actions)
        layout.addStretch(1)
        return page

    @staticmethod
    def _status_card(title: str, value: str, *, tone: str = "mint") -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setProperty("tone", tone)
        card.setMinimumHeight(126)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(17, 16, 17, 15)
        layout.setSpacing(9)
        eyebrow = QLabel(title.upper())
        eyebrow.setObjectName("Eyebrow")
        value_label = QLabel(value)
        value_label.setWordWrap(True)
        value_label.setProperty("valueLabel", True)
        detail_label = QLabel("")
        detail_label.setWordWrap(True)
        detail_label.setProperty("detailLabel", True)
        detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_label.setVisible(False)
        layout.addWidget(eyebrow)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        layout.addStretch(1)
        card.title_label = eyebrow
        card.value_label = value_label
        card.detail_label = detail_label
        return card

    @staticmethod
    def _repolish(*widgets: QWidget) -> None:
        for widget in widgets:
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _set_card(self, card: QFrame, headline: str, detail: str = "", tone: str | None = None) -> None:
        card.value_label.setText(headline)
        card.detail_label.setText(detail)
        card.detail_label.setVisible(bool(detail))
        if tone is not None and card.property("tone") != tone:
            card.setProperty("tone", tone)
            self._repolish(card, card.value_label, card.detail_label)

    def _set_status_label(self, label: QLabel, level: str, text: str) -> None:
        name = _LEVEL_LABEL_NAMES.get(level, "StatusNeutral")
        label.setText(text)
        if label.objectName() != name:
            label.setObjectName(name)
            self._repolish(label)
        label.setVisible(True)

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Capture and automation",
            "Record once now, or start and stop the recording schedule.",
        )
        self.record_gray_button = QPushButton("Record in Grayscale: Off")
        self.record_gray_button.setCheckable(True)
        self.record_gray_button.setToolTip(
            "Save this camera's recordings in grayscale (smaller files) or in colour. Saved straight "
            "away and used from the next recording. To look at the camera in grayscale without "
            "changing recordings, use Grayscale View in the Live Camera Monitor."
        )
        self.record_gray_button.clicked.connect(self._toggle_record_grayscale)
        heading_row = QHBoxLayout()
        heading_row.addWidget(heading, 1)
        heading_row.addWidget(self.record_gray_button, 0, Qt.AlignVCenter)
        layout.addLayout(heading_row)
        layout.addWidget(subtitle)
        # The controls scroll when the window is short, so the outcome of the
        # last command and its output always stay in view below them.
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 6, 0)
        controls_layout.setSpacing(12)
        row = QHBoxLayout()
        self.run_mode = QComboBox()
        for value in ("record_only", "record_and_track", "track_only"):
            self.run_mode.addItem(value.replace("_", " ").title(), value)
        row.addWidget(QLabel("Run mode"))
        row.addWidget(self.run_mode)
        run = QPushButton("Run Once Now")
        run.setObjectName("Accent")
        run.clicked.connect(self._run_once)
        row.addWidget(run)
        self.run_once_button = run
        self.run_both_button = QPushButton("Record Both Now")
        self.run_both_button.setToolTip(
            "Records from both cameras at the same moment, each with its own settings. "
            "Use it to check that this Pi can carry both."
        )
        self.run_both_button.clicked.connect(self._run_both_once)
        self.run_both_button.setVisible(False)
        row.addWidget(self.run_both_button)
        row.addStretch(1)
        controls_layout.addLayout(row)

        automation = QGroupBox("Automated recording")
        automation.setProperty("tone", "mint")
        automation_layout = QVBoxLayout(automation)
        self.schedule_status_label = QLabel("Checking the schedule...")
        self.schedule_status_label.setObjectName("StatusNeutral")
        self.schedule_status_label.setWordWrap(True)
        automation_layout.addWidget(self.schedule_status_label)
        self.schedule_detail_label = QLabel(
            "Recordings start on the clock and keep going when this window is closed."
        )
        self.schedule_detail_label.setWordWrap(True)
        self.schedule_detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        automation_layout.addWidget(self.schedule_detail_label)
        self.both_cameras_checkbox = QCheckBox("Start and stop both cameras together")
        self.both_cameras_checkbox.setChecked(True)
        self.both_cameras_checkbox.setVisible(False)
        automation_layout.addWidget(self.both_cameras_checkbox)
        buttons = QHBoxLayout()
        start = QPushButton("Start Automated Recording")
        start.setObjectName("Primary")
        start.clicked.connect(self._start_automation)
        stop = QPushButton("Stop Automated Recording")
        stop.setObjectName("Quiet")
        stop.clicked.connect(self._stop_automation)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        buttons.addStretch(1)
        automation_layout.addLayout(buttons)
        controls_layout.addWidget(automation)

        self.distributed_run_group = QGroupBox("Two-Pi capture readiness")
        self.distributed_run_group.setProperty("tone", "sky")
        distributed_layout = QHBoxLayout(self.distributed_run_group)
        self.distributed_run_status = QLabel()
        self.distributed_run_status.setWordWrap(True)
        distributed_check = QPushButton("Check Both Pis")
        distributed_check.setObjectName("Quiet")
        distributed_check.clicked.connect(
            lambda: self.run_bbx_command(["distributed-capture", "check"])
        )
        distributed_hardware_check = QPushButton("Check All Assigned Cameras")
        distributed_hardware_check.clicked.connect(
            lambda: self.run_bbx_command(
                ["distributed-capture", "check", "--probe-hardware"]
            )
        )
        distributed_layout.addWidget(self.distributed_run_status, 1)
        distributed_layout.addWidget(distributed_check)
        distributed_layout.addWidget(distributed_hardware_check)
        controls_layout.addWidget(self.distributed_run_group)

        latest = QGroupBox("Latest recording")
        latest.setProperty("tone", "sky")
        latest_layout = QHBoxLayout(latest)
        self.latest_run_status = QLabel("Loading recent recording status...")
        self.latest_run_status.setWordWrap(True)
        latest_layout.addWidget(self.latest_run_status, 1)
        latest_buttons = QHBoxLayout()
        self.latest_session_button = QPushButton("Open Session Folder")
        self.latest_session_button.setObjectName("Quiet")
        self.latest_session_button.setEnabled(False)
        self.latest_session_button.clicked.connect(
            lambda: self._open_run_artifact(self._latest_run_payload(), "_session_dir_local")
        )
        self.latest_depth_button = QPushButton("Open Latest Depth Preview")
        self.latest_depth_button.setObjectName("Quiet")
        self.latest_depth_button.setEnabled(False)
        self.latest_depth_button.setVisible(False)
        self.latest_depth_button.clicked.connect(
            lambda: self._open_run_artifact(
                self._latest_run_payload(), "realsense_depth_preview_video_path"
            )
        )
        self.latest_results_button = QPushButton("View All Results")
        self.latest_results_button.setObjectName("Primary")
        self.latest_results_button.clicked.connect(lambda: self.navigation.setCurrentRow(PAGE_INDEX_RESULTS))
        latest_buttons.addWidget(self.latest_session_button)
        latest_buttons.addWidget(self.latest_depth_button)
        latest_buttons.addWidget(self.latest_results_button)
        latest_layout.addLayout(latest_buttons)
        controls_layout.addWidget(latest)
        scroll = _FitScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(controls)
        controls.setAutoFillBackground(False)
        scroll.viewport().setAutoFillBackground(False)
        layout.addWidget(scroll)
        command_row = QHBoxLayout()
        command_row.setSpacing(10)
        self.command_status = QLabel("")
        self.command_status.setObjectName("StatusNeutral")
        self.command_status.setWordWrap(True)
        self.command_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command_status.setVisible(False)
        command_row.addWidget(self.command_status, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("Quiet")
        self.cancel_button.setToolTip("Stop the command that is running now.")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_command)
        command_row.addWidget(self.cancel_button, 0, Qt.AlignVCenter)
        layout.addLayout(command_row)
        layout.addWidget(self._build_console(), 1)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Recording results",
            "Recent recordings and their files. Select one to open its video or folder.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        toolbar = QHBoxLayout()
        self.results_status = QLabel("Recent runs have not been loaded.")
        self.results_status.setWordWrap(True)
        self.results_refresh_button = QPushButton("Refresh Results")
        self.results_refresh_button.setObjectName("Quiet")
        self.results_refresh_button.clicked.connect(self.refresh_results)
        toolbar.addWidget(self.results_status, 1)
        toolbar.addWidget(self.results_refresh_button)
        layout.addLayout(toolbar)

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["Started", "Status", "Mode", "RGB", "Thermal", "Depth", "RGB FPS"]
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        results_header = self.results_table.horizontalHeader()
        results_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        results_header.setSectionResizeMode(0, QHeaderView.Stretch)
        results_header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.results_table.itemSelectionChanged.connect(self._result_selection_changed)
        layout.addWidget(self.results_table, 1)

        selected = QGroupBox("Selected recording")
        selected.setProperty("tone", "sky")
        selected_layout = QVBoxLayout(selected)
        self.result_detail = QLabel("Select a recording to inspect its outputs.")
        self.result_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_detail.setWordWrap(True)
        selected_layout.addWidget(self.result_detail)
        artifact_buttons = QGridLayout()
        self.result_button_grid = artifact_buttons
        button_specs = (
            ("session", "Open Session Folder", "_session_dir_local"),
            ("rgb", "Open RGB Video", "video_path"),
            ("thermal", "Open RGB + Thermal", "thermal_side_by_side_video_path"),
            ("depth", "Open Depth Preview", "realsense_depth_preview_video_path"),
            ("depth_color", "Open RealSense Color", "realsense_color_video_path"),
            ("depth_data", "Open Depth Data Folder", "realsense_raw_depth_npy_path"),
            ("metadata", "Open RealSense Metadata", "realsense_metadata_json_path"),
            ("manifest", "Open Multi-Pi Manifest", "distributed_manifest_path"),
        )
        self.result_artifact_buttons: dict[str, tuple[QPushButton, str]] = {}
        for index, (key, label, payload_key) in enumerate(button_specs):
            button = QPushButton(label)
            button.setObjectName("Quiet" if key != "session" else "Accent")
            button.clicked.connect(
                lambda _checked=False, selected_key=payload_key: self._open_run_artifact(
                    self._selected_run_payload,
                    selected_key,
                    open_parent=selected_key == "realsense_raw_depth_npy_path",
                )
            )
            button.setEnabled(False)
            self.result_artifact_buttons[key] = (button, payload_key)
        self.result_add_calibration_button = QPushButton("Add to Calibration Project")
        self.result_add_calibration_button.setObjectName("Primary")
        self.result_add_calibration_button.setToolTip(
            "Create or select a multimodal calibration project in Advanced, then register this run."
        )
        self.result_add_calibration_button.clicked.connect(self._add_selected_run_to_calibration)
        self.result_add_calibration_button.setEnabled(False)
        self.result_sync_analysis_button = QPushButton("Analyze Shared Timing Cue")
        self.result_sync_analysis_button.setObjectName("Quiet")
        self.result_sync_analysis_button.setToolTip(
            "Estimate thermal and depth offset/jitter from a recording with a shared moving or occlusion cue."
        )
        self.result_sync_analysis_button.clicked.connect(self._analyze_selected_run_sync)
        self.result_sync_analysis_button.setEnabled(False)
        # Laid out by _layout_result_buttons(), which leaves out buttons for
        # hardware this box does not have.
        self.result_button_order: list[tuple[str, QPushButton]] = [
            *((key, button) for key, (button, _) in self.result_artifact_buttons.items()),
            ("calibration", self.result_add_calibration_button),
            ("sync", self.result_sync_analysis_button),
        ]
        self._visible_result_buttons: tuple[str, ...] = ()
        selected_layout.addLayout(artifact_buttons)
        layout.addWidget(selected)
        return page

    def _build_hardware_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Hardware checks",
            "Only devices selected in the setup profile are shown here.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        self.rgb_group = QGroupBox("Primary camera")
        self.rgb_group.setProperty("tone", "peach")
        rgb_buttons = QHBoxLayout(self.rgb_group)
        camera_check = QPushButton("Check Camera")
        camera_check.clicked.connect(lambda: self.run_bbx_command(["camera-check"]))
        camera_preview = QPushButton("Preview 15 Seconds")
        camera_preview.clicked.connect(lambda: self.run_bbx_command(["camera-preview", "--seconds", "15"]))
        rgb_buttons.addWidget(camera_check)
        rgb_buttons.addWidget(camera_preview)
        rgb_buttons.addStretch(1)
        layout.addWidget(self.rgb_group)

        self.thermal_group = QGroupBox("PureThermal / Lepton")
        self.thermal_group.setProperty("tone", "sun")
        thermal_buttons = QHBoxLayout(self.thermal_group)
        thermal_check = QPushButton("Check Thermal Camera")
        thermal_check.clicked.connect(lambda: self.run_bbx_command(["thermal-check"]))
        thermal_snapshot = QPushButton("Capture Thermal Snapshot")
        thermal_snapshot.clicked.connect(lambda: self.run_bbx_command(["thermal-snapshot"]))
        thermal_buttons.addWidget(thermal_check)
        thermal_buttons.addWidget(thermal_snapshot)
        thermal_buttons.addStretch(1)
        layout.addWidget(self.thermal_group)

        self.realsense_group = QGroupBox("RealSense depth")
        self.realsense_group.setProperty("tone", "sky")
        depth_buttons = QHBoxLayout(self.realsense_group)
        depth_check = QPushButton("Check and Pin RealSense")
        depth_check.clicked.connect(lambda: self.run_bbx_command(["realsense-check", "--apply"]))
        depth_snapshot = QPushButton("Capture Depth Snapshot")
        depth_snapshot.clicked.connect(lambda: self.run_bbx_command(["realsense-snapshot"]))
        depth_buttons.addWidget(depth_check)
        depth_buttons.addWidget(depth_snapshot)
        depth_buttons.addStretch(1)
        layout.addWidget(self.realsense_group)

        self.distributed_hardware_group = QGroupBox("Second Raspberry Pi")
        self.distributed_hardware_group.setProperty("tone", "mint")
        distributed_buttons = QHBoxLayout(self.distributed_hardware_group)
        self.distributed_hardware_label = QLabel()
        self.distributed_hardware_label.setWordWrap(True)
        distributed_check = QPushButton("Check Two-Pi Connection")
        distributed_check.clicked.connect(
            lambda: self.run_bbx_command(["distributed-capture", "check"])
        )
        distributed_buttons.addWidget(self.distributed_hardware_label, 1)
        distributed_buttons.addWidget(distributed_check)
        layout.addWidget(self.distributed_hardware_group)
        self.second_camera_group = QGroupBox("Second camera")
        self.second_camera_group.setProperty("tone", "sky")
        second_layout = QVBoxLayout(self.second_camera_group)
        self.second_camera_label = QLabel()
        self.second_camera_label.setWordWrap(True)
        self.second_camera_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        second_layout.addWidget(self.second_camera_label)
        second_buttons = QHBoxLayout()
        self.add_second_camera_button = QPushButton("Add Second Camera...")
        self.add_second_camera_button.setObjectName("Accent")
        self.add_second_camera_button.clicked.connect(self._add_second_camera)
        self.pin_cameras_button = QPushButton("Pin Cameras")
        self.pin_cameras_button.setToolTip(
            "Ties each camera's settings to one physical camera, so unplugging one cannot swap them."
        )
        self.pin_cameras_button.clicked.connect(self._pin_cameras)
        self.remove_second_camera_button = QPushButton("Remove Second Camera...")
        self.remove_second_camera_button.setObjectName("Quiet")
        self.remove_second_camera_button.clicked.connect(self._remove_second_camera)
        for button in (self.add_second_camera_button, self.pin_cameras_button, self.remove_second_camera_button):
            second_buttons.addWidget(button)
        second_buttons.addStretch(1)
        second_layout.addLayout(second_buttons)
        layout.addWidget(self.second_camera_group)
        change = QPushButton("Change Hardware Profile")
        change.setObjectName("Quiet")
        change.clicked.connect(self.open_setup_wizard)
        layout.addWidget(change, 0, Qt.AlignLeft)
        layout.addStretch(1)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Recording settings",
            "What each recording captures and how it is saved. Changes apply from the next recording.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(12)

        def form_box(title: str, tone: str) -> tuple[QGroupBox, QFormLayout]:
            box = QGroupBox(title)
            box.setProperty("tone", tone)
            form = QFormLayout(box)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(10)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            return box, form

        def hint(text: str = "") -> QLabel:
            label = QLabel(text)
            label.setObjectName("Hint")
            label.setWordWrap(True)
            # Without Expanding, a form keeps a wrapped label at its natural
            # width and can cut off its last line.
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            return label

        def row_label(text: str = "") -> QLabel:
            # One label width for every group, so the fields line up down the page.
            label = QLabel(text)
            label.setFixedWidth(176)
            return label

        schedule_box, schedule_form = form_box("Schedule", "mint")
        self.settings_mode = QComboBox()
        for value, label in (
            ("record_and_track", "Record, then track"),
            ("record_only", "Record only"),
            ("mixed_schedule", "Record and track on separate schedules"),
            ("track_only", "Track only (no video saved)"),
        ):
            self.settings_mode.addItem(label, value)
        # Lengths and intervals are typed in whichever unit suits; the config
        # still stores seconds and minutes.
        self.settings_recording_seconds = _UnitValueEdit(DURATION_UNITS, minimum=1, maximum=86400)
        self.settings_record_interval = _UnitValueEdit(
            INTERVAL_UNITS, minimum=1, maximum=MINUTES_PER_DAY, whole=True
        )
        self.settings_track_interval = _UnitValueEdit(
            INTERVAL_UNITS, minimum=1, maximum=MINUTES_PER_DAY, whole=True
        )
        self.settings_start_time = _time_edit(DEFAULT_START_TIME)
        self.settings_start_hint = hint()
        start_row = QHBoxLayout()
        start_row.setSpacing(12)
        start_row.addWidget(self.settings_start_time)
        start_row.addWidget(self.settings_start_hint, 1)
        self._settings_track_interval_label = row_label("Track every")
        schedule_form.addRow(row_label("Run mode"), self.settings_mode)
        schedule_form.addRow(row_label("Record for"), self.settings_recording_seconds)
        schedule_form.addRow(row_label("Record every"), self.settings_record_interval)
        schedule_form.addRow(row_label("Starting at"), start_row)
        schedule_form.addRow(self._settings_track_interval_label, self.settings_track_interval)
        content_layout.addWidget(schedule_box)

        camera_box, camera_form = form_box("Camera", "peach")
        self.settings_fps = QDoubleSpinBox()
        self.settings_fps.setRange(0.1, 120.0)
        self.settings_fps.setDecimals(2)
        self.settings_fps.setSingleStep(0.5)
        self.settings_fps.setSuffix(" fps")
        self.settings_resolution = QLabel()
        self.settings_resolution.setWordWrap(True)
        self.settings_resolution.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.settings_monochrome = QCheckBox("Save grayscale video (smaller files)")
        self.settings_profile_note = hint()
        self.settings_custom_camera_button = QPushButton("Use Custom Camera Settings")
        self.settings_custom_camera_button.setObjectName("Quiet")
        self.settings_custom_camera_button.setToolTip(
            "Keeps the profile's current values and lets you change them here. Applied when you save."
        )
        self.settings_custom_camera_button.clicked.connect(self._use_custom_camera_settings)
        self._settings_profile_row = QWidget()
        profile_row = QHBoxLayout(self._settings_profile_row)
        profile_row.setContentsMargins(0, 0, 0, 0)
        profile_row.addWidget(self.settings_profile_note, 1)
        profile_row.addWidget(self.settings_custom_camera_button, 0, Qt.AlignVCenter)
        self._settings_profile_row_label = row_label()
        camera_form.addRow(row_label("Frame rate"), self.settings_fps)
        camera_form.addRow(row_label("Resolution"), self.settings_resolution)
        camera_form.addRow(row_label(), self.settings_monochrome)
        camera_form.addRow(self._settings_profile_row_label, self._settings_profile_row)
        content_layout.addWidget(camera_box)

        picture_box, picture_form = form_box("Picture", "peach")

        def gain_spin(low: float, high: float, value: float) -> _TrimmedDoubleSpinBox:
            spin = _TrimmedDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setMaximumWidth(110)
            return spin

        def auto_or_fixed() -> QComboBox:
            combo = QComboBox()
            combo.addItem("Automatic", "auto")
            combo.addItem("Fixed", "fixed")
            return combo

        def field_row(*widgets: QWidget) -> QWidget:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            for widget in widgets:
                row_layout.addWidget(widget)
            row_layout.addStretch(1)
            return row

        self.settings_gain_mode = auto_or_fixed()
        self.settings_gain = gain_spin(*image_controls.GAIN_RANGE, 2.0)
        self.settings_gain.setToolTip("How much the sensor amplifies the picture. Higher is brighter and noisier.")
        self.settings_wb_mode = auto_or_fixed()
        self.settings_red_gain = gain_spin(*image_controls.COLOUR_GAIN_RANGE, 1.0)
        self.settings_blue_gain = gain_spin(*image_controls.COLOUR_GAIN_RANGE, 1.0)
        self._settings_red_label = QLabel("red")
        self._settings_blue_label = QLabel("blue")
        self.settings_contrast = gain_spin(0.5, 2.0, image_controls.DEFAULT_CONTRAST)
        self.settings_contrast.setSingleStep(0.05)
        self.settings_contrast.setToolTip(
            "1 is the camera's own picture. Higher makes darks darker and brights brighter, "
            "and is saved into the recording."
        )
        self.settings_noise_reduction = QComboBox()
        for label, value in (
            ("Off", "Off"),
            ("Camera's own setting (minimal)", "Auto"),
            ("Fast", "Fast"),
            ("High quality", "HighQuality"),
        ):
            if value == image_controls.DEFAULT_NOISE_REDUCTION:
                label += " (default)"
            self.settings_noise_reduction.addItem(label, value)
        self.settings_noise_reduction.setToolTip(
            "High quality removes the most noise. It can lower the frame rate the camera reaches: "
            "if recordings come out slower than the frame rate set above, choose Fast."
        )
        self.settings_picture_hint = hint(
            "These are applied inside the camera, so they change recordings, the live monitor and what "
            "the tag detector sees. To choose fixed values: leave gain and white balance on Automatic, "
            "open the Live Camera Monitor, and copy the numbers shown under the picture once they "
            "settle. Save here before opening the monitor to see a change."
        )
        picture_form.addRow(row_label("Sensor gain"), field_row(self.settings_gain_mode, self.settings_gain))
        picture_form.addRow(
            row_label("White balance"),
            field_row(
                self.settings_wb_mode,
                self._settings_red_label,
                self.settings_red_gain,
                self._settings_blue_label,
                self.settings_blue_gain,
            ),
        )
        picture_form.addRow(row_label("Contrast"), field_row(self.settings_contrast))
        picture_form.addRow(row_label("Noise reduction"), self.settings_noise_reduction)
        picture_form.addRow(row_label(), self.settings_picture_hint)
        content_layout.addWidget(picture_box)

        video_box, video_form = form_box("Video", "sky")
        self.settings_frame_buffer = QComboBox()
        self.settings_frame_buffer.addItem("Hold frames in memory, encode afterwards", "ram")
        self.settings_frame_buffer.addItem("Encode while recording", "stream")
        self.settings_frame_buffer_hint = hint()
        self.settings_codec = QComboBox()
        self.settings_codec.addItem("MP4 (H.264)", "mp4")
        self.settings_codec.addItem("MJPEG (AVI, much larger files)", "mjpeg")
        self.settings_preset = QComboBox()
        for value, label in MP4_PRESET_CHOICES:
            self.settings_preset.addItem(label, value)
        self._settings_preset_label = row_label("MP4 encoder speed")
        self.settings_benchmark_label = hint()
        self.settings_benchmark_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.settings_benchmark_button = QPushButton("Run Encode Benchmark")
        self.settings_benchmark_button.setObjectName("Quiet")
        self.settings_benchmark_button.setToolTip(
            "Times each encoder at this resolution on this computer, using your latest recording "
            "when there is one. Takes a minute or two; output appears on the Run page."
        )
        self.settings_benchmark_button.clicked.connect(self._run_encode_benchmark)
        benchmark_row = QVBoxLayout()
        benchmark_row.setSpacing(8)
        benchmark_row.addWidget(self.settings_benchmark_label)
        benchmark_row.addWidget(self.settings_benchmark_button, 0, Qt.AlignLeft)
        self.settings_chunk_minutes = _TrimmedDoubleSpinBox()
        self.settings_chunk_minutes.setRange(0.0, 1440.0)
        self.settings_chunk_minutes.setDecimals(1)
        self.settings_chunk_minutes.setSuffix(" min")
        self.settings_chunk_minutes.setSpecialValueText("Never (one file)")
        self.settings_chunk_hint = hint(
            "Long recordings are saved as a row of files with no gap between them, so a failure "
            "costs at most the file being written. A shorter recording stays one file."
        )
        self._settings_chunk_label = row_label("New video file every")
        self.settings_restart = QComboBox()
        # Text keys, not True/False: Qt treats the text "auto" and the value
        # True as equal when it searches the list.
        self.settings_restart.addItem("Start again for the time left (recordings of 2 min or more)", "auto")
        self.settings_restart.addItem("Always start again for the time left", "always")
        self.settings_restart.addItem("Do not start again", "never")
        self.settings_restart.setToolTip(
            "If the recorder crashes, or stops making progress, it is stopped and a new recording "
            "is started for the rest of the session."
        )
        video_form.addRow(row_label("Frame handling"), self.settings_frame_buffer)
        video_form.addRow(row_label(), self.settings_frame_buffer_hint)
        video_form.addRow(self._settings_chunk_label, self.settings_chunk_minutes)
        video_form.addRow(row_label(), self.settings_chunk_hint)
        video_form.addRow(row_label("If a recording fails"), self.settings_restart)
        video_form.addRow(row_label("Format"), self.settings_codec)
        video_form.addRow(self._settings_preset_label, self.settings_preset)
        video_form.addRow(row_label("Encode benchmark"), benchmark_row)
        content_layout.addWidget(video_box)

        storage_box, storage_form = form_box("Storage", "sun")
        self.settings_min_free = QDoubleSpinBox()
        self.settings_min_free.setRange(0.0, 10000.0)
        self.settings_min_free.setDecimals(1)
        self.settings_min_free.setSuffix(" GB free")
        storage_form.addRow(row_label("Refuse to record below"), self.settings_min_free)
        content_layout.addWidget(storage_box)

        effects = QGroupBox("What these settings mean")
        effects_layout = QVBoxLayout(effects)
        self.settings_check_status = QLabel("")
        self.settings_check_status.setObjectName("StatusNeutral")
        self.settings_check_status.setWordWrap(True)
        self.settings_issues_label = QLabel("")
        self.settings_issues_label.setWordWrap(True)
        self.settings_issues_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.settings_effects_label = QLabel("")
        self.settings_effects_label.setWordWrap(True)
        self.settings_effects_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        effects_layout.addWidget(self.settings_check_status)
        effects_layout.addWidget(self.settings_issues_label)
        effects_layout.addWidget(self.settings_effects_label)
        content_layout.addWidget(effects)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.viewport().setAutoFillBackground(False)
        layout.addWidget(scroll, 1)

        bar = QHBoxLayout()
        self.settings_summary_label = QLabel("")
        self.settings_summary_label.setWordWrap(True)
        self.settings_discard_button = QPushButton("Discard Changes")
        self.settings_discard_button.setObjectName("Quiet")
        self.settings_discard_button.clicked.connect(self._discard_settings)
        self.settings_save_button = QPushButton("Save Settings")
        self.settings_save_button.setObjectName("Primary")
        self.settings_save_button.clicked.connect(lambda: self._save_settings(offer_apply=True))
        bar.addWidget(self.settings_summary_label, 1)
        bar.addWidget(self.settings_discard_button)
        bar.addWidget(self.settings_save_button)
        layout.addLayout(bar)

        self._wheel_guard = _WheelOnlyWhenFocused(self)
        unit_fields = (self.settings_recording_seconds, self.settings_record_interval, self.settings_track_interval)
        for unit_field in unit_fields:
            unit_field.changed.connect(self._settings_changed)
        self.settings_start_time.timeChanged.connect(lambda *_: self._settings_changed())
        picture_spins = (self.settings_gain, self.settings_red_gain, self.settings_blue_gain, self.settings_contrast)
        for spin in (self.settings_fps, self.settings_min_free, self.settings_chunk_minutes, *picture_spins):
            spin.valueChanged.connect(lambda *_: self._settings_changed())
        guarded = [
            self.settings_fps,
            self.settings_min_free,
            self.settings_start_time,
            self.settings_chunk_minutes,
            *picture_spins,
        ]
        for unit_field in unit_fields:
            guarded.extend(unit_field.fields())
        for field in guarded:
            field.setFocusPolicy(Qt.StrongFocus)
            field.installEventFilter(self._wheel_guard)
        for combo in (
            self.settings_mode,
            self.settings_frame_buffer,
            self.settings_codec,
            self.settings_preset,
            self.settings_restart,
            self.settings_noise_reduction,
        ):
            combo.currentIndexChanged.connect(lambda *_: self._settings_changed())
            # Shrink with the window instead of forcing the page wider than it is.
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(18)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.setMaximumWidth(460)
            combo.setFocusPolicy(Qt.StrongFocus)
            combo.installEventFilter(self._wheel_guard)
        self.settings_monochrome.toggled.connect(lambda *_: self._settings_changed())
        for combo in (self.settings_gain_mode, self.settings_wb_mode):
            combo.currentIndexChanged.connect(lambda *_: self._settings_changed())
            combo.setFocusPolicy(Qt.StrongFocus)
            combo.installEventFilter(self._wheel_guard)
        return page

    # -- Settings page ---------------------------------------------------------

    def _settings_profile_key(self) -> str:
        if self._settings_profile_override:
            return self._settings_profile_override
        return str(_get_nested(self.config, "camera.profile", "custom") or "custom")

    def _settings_profile_fixed(self) -> set[str]:
        """Settings keys that the camera profile overwrites whenever the config is saved."""
        profile = CAMERA_PROFILES.get(self._settings_profile_key())
        if profile is None:
            return set()
        values = set(profile.camera_values)
        return {key for key, camera_key in PROFILE_CONTROLLED_SETTINGS.items() if camera_key in values}

    def _settings_values(self) -> dict[str, Any]:
        return {
            "pipeline.mode": self.settings_mode.currentData(),
            "capture.recording_seconds": self.settings_recording_seconds.value(),
            "capture.record_interval_minutes": int(self.settings_record_interval.value()),
            "capture.track_interval_minutes": int(self.settings_track_interval.value()),
            "scheduling.start_time": self.settings_start_time.time().toString("HH:mm"),
            "camera.fps_target": round(float(self.settings_fps.value()), 2),
            "camera.monochrome_output": bool(self.settings_monochrome.isChecked()),
            "camera.analogue_gain": (
                round(float(self.settings_gain.value()), 2)
                if self.settings_gain_mode.currentData() == "fixed"
                else image_controls.AUTO
            ),
            "camera.colour_gains": (
                [round(float(self.settings_red_gain.value()), 2), round(float(self.settings_blue_gain.value()), 2)]
                if self.settings_wb_mode.currentData() == "fixed"
                else image_controls.AUTO
            ),
            "camera.contrast": round(float(self.settings_contrast.value()), 2),
            "camera.noise_reduction": self.settings_noise_reduction.currentData(),
            "camera.codec": self.settings_codec.currentData(),
            "camera.mp4_preset": self.settings_preset.currentData(),
            "capture.frame_buffer": self.settings_frame_buffer.currentData(),
            "capture.chunk_minutes": round(float(self.settings_chunk_minutes.value()), 1),
            "capture.restart_on_failure": RESTART_CHOICES.get(str(self.settings_restart.currentData()), "auto"),
            "system.min_free_space_gb": round(float(self.settings_min_free.value()), 1),
            "camera.profile": self._settings_profile_key(),
        }

    def _settings_field_problems(self) -> list[str]:
        """Values typed on the Settings page that the recorder cannot use."""
        fields = [("Record for", self.settings_recording_seconds), ("Record every", self.settings_record_interval)]
        if not self.settings_track_interval.isHidden():
            fields.append(("Track every", self.settings_track_interval))
        return [f"{label}: {field.problem()}." for label, field in fields if field.problem()]

    def _settings_dirty(self) -> bool:
        return self._settings_saved_values is not None and self._settings_values() != self._settings_saved_values

    def _apply_settings_to(self, config: dict[str, Any]) -> dict[str, Any]:
        """Write the values changed on the Settings page into ``config``.

        Only fields the user actually changed are written. The config file can
        be changed elsewhere while this window is open (the live monitor's
        Record in Grayscale button, or editing the file by hand); writing
        every field would put the form's older values back over those.
        """
        fixed = self._settings_profile_fixed()
        shown_when_loaded = self._settings_saved_values or {}
        for key, value in self._settings_values().items():
            if key in fixed:
                continue
            if key in shown_when_loaded and shown_when_loaded[key] == value:
                continue
            _set_nested(config, key, value)
        return config

    def _settings_candidate(self) -> dict[str, Any]:
        """The config as it would be saved, with the camera profile applied."""
        return apply_camera_profile(self._apply_settings_to(copy.deepcopy(self.config)))

    def _load_settings_form(self) -> None:
        config = self.config
        self._settings_loading = True
        try:
            self._settings_profile_override = None
            _set_combo_data(self.settings_mode, str(_get_nested(config, "pipeline.mode", "record_and_track")))
            self.settings_recording_seconds.setValue(_get_nested(config, "capture.recording_seconds", 20) or 20)
            self.settings_record_interval.setValue(int(_get_nested(config, "capture.record_interval_minutes", 30) or 30))
            self.settings_track_interval.setValue(int(_get_nested(config, "capture.track_interval_minutes", 5) or 5))
            try:
                start = parse_time_of_day(_get_nested(config, "scheduling.start_time", DEFAULT_START_TIME))
            except ValueError:
                start = 0
            self.settings_start_time.setTime(QTime(start // 60, start % 60))
            self.settings_fps.setValue(float(_get_nested(config, "camera.fps_target", 5.0) or 5.0))
            self.settings_monochrome.setChecked(bool(_get_nested(config, "camera.monochrome_output", False)))
            self._load_picture_settings(config)
            _set_combo_data(self.settings_codec, str(_get_nested(config, "camera.codec", "mp4")).lower())
            _set_combo_data(self.settings_preset, str(_get_nested(config, "camera.mp4_preset", "auto")).lower())
            _set_combo_data(self.settings_frame_buffer, frame_buffer_mode(config))
            try:
                self.settings_chunk_minutes.setValue(float(_get_nested(config, "capture.chunk_minutes", 10)))
            except (TypeError, ValueError):
                self.settings_chunk_minutes.setValue(10.0)
            restart = _get_nested(config, "capture.restart_on_failure", "auto")
            restart_key = ("always" if restart else "never") if isinstance(restart, bool) else "auto"
            self.settings_restart.setCurrentIndex(max(0, self.settings_restart.findData(restart_key)))
            self.settings_min_free.setValue(float(_get_nested(config, "system.min_free_space_gb", 1.0) or 0.0))
        finally:
            self._settings_loading = False
        self._settings_saved_values = self._settings_values()
        self._settings_changed()

    def _load_picture_settings(self, config: dict[str, Any]) -> None:
        """Show the camera's picture settings. A value the recorder could not use shows as the default."""

        def read(getter: Any, fallback: Any) -> Any:
            try:
                return getter(config)
            except image_controls.ImageSettingError:
                return fallback

        gain = read(image_controls.analogue_gain, None)
        _set_combo_data(self.settings_gain_mode, "auto" if gain is None else "fixed")
        if gain is not None:
            self.settings_gain.setValue(gain)
        gains = read(image_controls.colour_gains, None)
        _set_combo_data(self.settings_wb_mode, "auto" if gains is None else "fixed")
        if gains is not None:
            self.settings_red_gain.setValue(gains[0])
            self.settings_blue_gain.setValue(gains[1])
        self.settings_contrast.setValue(read(image_controls.contrast, image_controls.DEFAULT_CONTRAST))
        _set_combo_data(self.settings_noise_reduction, read(image_controls.noise_reduction, "Auto"))

    def _settings_changed(self) -> None:
        if self._settings_loading:
            return
        fixed_gain = self.settings_gain_mode.currentData() == "fixed"
        self.settings_gain.setEnabled(fixed_gain)
        fixed_balance = self.settings_wb_mode.currentData() == "fixed"
        for widget in (
            self.settings_red_gain,
            self.settings_blue_gain,
            self._settings_red_label,
            self._settings_blue_label,
        ):
            widget.setEnabled(fixed_balance)
        mode = self.settings_mode.currentData()
        tracks_separately = mode in {"mixed_schedule", "track_only"}
        self._settings_track_interval_label.setVisible(tracks_separately)
        self.settings_track_interval.setVisible(tracks_separately)
        start_text = self.settings_start_time.time().toString("HH:mm")
        interval = int(
            self.settings_track_interval.value() if mode == "track_only" else self.settings_record_interval.value()
        )
        if interval >= MINUTES_PER_DAY:
            self.settings_start_hint.setText("It starts at this time every day.")
        else:
            self.settings_start_hint.setText(
                f"Falls on {start_text} and {describe_interval(interval)} after it, around the clock."
            )
        is_mp4 = self.settings_codec.currentData() == "mp4"
        self._settings_preset_label.setEnabled(is_mp4)
        self.settings_preset.setEnabled(is_mp4)
        self.settings_frame_buffer_hint.setText(FRAME_BUFFER_HINTS.get(self.settings_frame_buffer.currentData(), ""))
        # Splitting into files only exists when encoding while recording.
        splits = self.settings_frame_buffer.currentData() == "stream"
        for widget in (self._settings_chunk_label, self.settings_chunk_minutes, self.settings_chunk_hint):
            widget.setVisible(splits)

        fixed = self._settings_profile_fixed()
        self.settings_fps.setEnabled("camera.fps_target" not in fixed)
        self.settings_codec.setEnabled("camera.codec" not in fixed)
        self.settings_monochrome.setEnabled("camera.monochrome_output" not in fixed)
        profile_label = CAMERA_PROFILES.get(self._settings_profile_key(), CAMERA_PROFILES["custom"]).label
        if fixed:
            self.settings_profile_note.setText(
                f"Frame rate, video format and grayscale are set by the camera profile “{profile_label}”."
            )
        elif self._settings_profile_override:
            self.settings_profile_note.setText("Custom camera settings will be used from when you save.")
        else:
            self.settings_profile_note.setText("")
        self.settings_profile_note.setVisible(bool(self.settings_profile_note.text()))
        self.settings_custom_camera_button.setVisible(bool(fixed))
        self._settings_profile_row.setVisible(bool(self.settings_profile_note.text()))
        self._settings_profile_row_label.setVisible(bool(self.settings_profile_note.text()))

        width = _get_nested(self.config, "camera.width", "?")
        height = _get_nested(self.config, "camera.height", "?")
        where = "from the camera profile" if fixed else "change it in the config file"
        self.settings_resolution.setText(f"{width} × {height}  ({where})")

        dirty = self._settings_dirty()
        self.settings_save_button.setEnabled(dirty)
        self.settings_discard_button.setEnabled(dirty)
        self._schedule_settings_preview()

    def _schedule_settings_preview(self) -> None:
        self._settings_preview_timer.start(SETTINGS_PREVIEW_DELAY_MS)

    def _use_custom_camera_settings(self) -> None:
        self._settings_profile_override = "custom"
        self._settings_changed()

    def _discard_settings(self) -> None:
        self._load_settings_form()

    def _load_benchmark(self) -> dict[str, Any] | None:
        return load_benchmark(self.encode_benchmark_path)

    def _refresh_settings_preview(self) -> None:
        candidate = self._settings_candidate()
        problems: list[str] = self._settings_field_problems()
        warnings: list[str] = []
        try:
            validate_config(candidate)
        except Exception as exc:
            problems.append(f"Not valid: {exc}")
        metrics: dict[str, Any] = {}
        try:
            report = run_schedule_check(candidate)
        except Exception as exc:
            problems.append(f"The schedule could not be checked: {exc}")
        else:
            metrics = dict(report.metrics)
            for item in report.items:
                if item.status == "FAIL":
                    problems.append(f"{item.name}: {item.message}")
                elif item.status == "WARN":
                    warnings.append(f"{item.name}: {item.message}")

        benchmark = self._load_benchmark()
        free_gb = self._box_status.storage.free_gb if self._box_status is not None else None
        estimate = estimate_storage(
            candidate,
            self._run_history_payloads,
            free_gb=free_gb,
            benchmark=benchmark,
            sharing_drive=self._sharing_drive(),
        )
        streaming = frame_buffer_mode(candidate) == FRAME_BUFFER_STREAM

        lines: list[str] = [describe_schedule_plan(candidate)]
        per_day = recordings_per_day(candidate)
        lines.append(
            f"Recordings per day: {per_day}" if per_day else "Recordings per day: none (track-only mode saves no video)."
        )
        buffer_bytes = metrics.get("estimated_frame_buffer_bytes")
        total_ram = metrics.get("total_ram_bytes")
        if per_day and isinstance(buffer_bytes, (int, float)) and buffer_bytes > 0:
            why = "frames waiting to be encoded" if streaming else "every frame is held until the recording ends"
            of_total = f" of {format_bytes(total_ram)}" if isinstance(total_ram, (int, float)) and total_ram > 0 else ""
            lines.append(f"Memory per recording: {format_bytes(buffer_bytes)}{of_total} on this computer ({why}).")
        lines.append(f"Storage: {estimate.headline}.")
        if estimate.detail:
            lines.append(estimate.detail)
        try:
            lines.append(f"Picture: {image_controls.describe_settings(candidate)}.")
        except image_controls.ImageSettingError:
            pass  # reported with the other config problems
        if estimate.level == LEVEL_WARN and estimate.days_left is not None:
            warnings.append(
                f"Storage: at this rate the drive is full in {describe_days(estimate.days_left)}, "
                "after which recording is refused."
            )

        other = self._other_camera()
        if other is not None:
            load = combined_memory([CameraEntry(self.camera_index, self.config_path, candidate), other])
            if load is not None:
                if load.level == "fail":
                    problems.append(f"Memory: {load.message}")
                elif load.level == "warn":
                    warnings.append(f"Memory: {load.message}")
                else:
                    lines.append(load.message)

        verdict = self._update_benchmark_label(candidate, benchmark)
        if streaming and verdict == "no":
            problems.append(
                "Encoder: the selected encoder was slower than the camera in the encode benchmark, "
                "so a recording that encodes while recording would fall behind."
            )
        elif streaming and verdict == "marginal":
            warnings.append(
                "Encoder: the selected encoder only just kept up in the encode benchmark. "
                "The camera also uses the CPU during a real recording, so it may fall behind."
            )

        counts = []
        if problems:
            counts.append(f"{len(problems)} problem" + ("s" if len(problems) > 1 else ""))
        if warnings:
            counts.append(f"{len(warnings)} warning" + ("s" if len(warnings) > 1 else ""))
        level = LEVEL_FAIL if problems else LEVEL_WARN if warnings else LEVEL_OK
        summary = ", ".join(counts) if counts else "No problems found"
        self._set_status_label(
            self.settings_check_status,
            level,
            summary + (" with these settings." if counts else " with these settings."),
        )
        issues = [f"Problem: {text}" for text in problems] + [f"Warning: {text}" for text in warnings]
        self.settings_issues_label.setText("\n".join(issues))
        self.settings_issues_label.setVisible(bool(issues))
        self.settings_effects_label.setText("\n".join(lines))

        bar = []
        if self._settings_dirty():
            bar.append("Unsaved changes")
        bar.append(summary)
        if estimate.bytes_per_day:
            bar.append(estimate.headline)
        self.settings_summary_label.setText(" · ".join(bar) + ".")

    def _update_benchmark_label(self, candidate: dict[str, Any], benchmark: dict[str, Any] | None) -> str | None:
        """Describe the last encode benchmark for the selected encoder; return its verdict."""
        if benchmark is None:
            self.settings_benchmark_label.setText(
                "Not run on this computer yet. It measures which encoders keep up with the camera."
            )
            return None
        try:
            measured = datetime.fromtimestamp(self.encode_benchmark_path.stat().st_mtime)
            now = datetime.now()
            when = (
                f" today at {measured:%H:%M}"
                if measured.date() == now.date()
                else f" on {describe_moment(measured, now)}"
            )
        except OSError:
            when = ""
        if not benchmark_matches_camera(candidate, benchmark):
            colour = "grayscale" if benchmark.get("monochrome_output") else "colour"
            self.settings_benchmark_label.setText(
                f"The last run{when} measured {benchmark.get('width')} × {benchmark.get('height')} {colour} "
                "video. Run it again for the current settings."
            )
            return None
        fps = float(_get_nested(candidate, "camera.fps_target", 0) or 0)
        speeds = []
        for result in benchmark.get("results") or []:
            if isinstance(result, dict) and not result.get("error"):
                name = "MJPEG" if result.get("codec") == "mjpeg" else str(result.get("mp4_preset") or result.get("label"))
                speeds.append(f"{name} {float(result.get('frames_per_second') or 0):.1f}")
        everything = f"All measured, in fps: {', '.join(speeds)}." if speeds else ""
        result = benchmark_result_for(candidate, benchmark)
        if result is None:
            self.settings_benchmark_label.setText(
                f"The last run{when} did not measure the selected encoder. {everything}".strip()
            )
            return None
        verdict = keeps_up(result.get("frames_per_second"), fps)
        phrase = {
            "yes": "keeps up with",
            "marginal": "only just keeps up with",
            "no": "is slower than",
        }.get(verdict, "was measured against")
        streaming = frame_buffer_mode(candidate) == FRAME_BUFFER_STREAM
        meaning = (
            ""
            if streaming
            else " With frames held in memory this only sets how long encoding takes after each recording."
        )
        self.settings_benchmark_label.setText(
            f"Selected encoder: {float(result.get('frames_per_second') or 0):.1f} fps, measured{when}, "
            f"so it {phrase} the camera at {fps:g} fps.{meaning} {everything}"
        )
        return verdict

    def _benchmark_source_video(self, candidate: dict[str, Any]) -> Path | None:
        """The newest saved recording at the camera's resolution, to benchmark real footage."""
        _, _, _, width, height = recording_signature(candidate)
        for payload in self._run_history_payloads:
            signature = payload.get("_recording_signature")
            if not signature or tuple(signature)[3:5] != (width, height):
                continue
            video = self._run_artifact_path(payload, "video_path")
            if video is not None and video.is_file():
                return video
        return None

    def _run_encode_benchmark(self) -> None:
        candidate = self._settings_candidate()
        camera = candidate.get("camera", {})
        encoders = list(DEFAULT_BENCHMARK_ENCODERS)
        preset = str(camera.get("mp4_preset", "auto"))
        if camera.get("codec") == "mp4" and preset != "auto":
            extra = f"{camera.get('mp4_codec', 'libx264')}:{preset}"
            if extra not in encoders:
                encoders.append(extra)
        arguments = [
            "encode-benchmark",
            "--fps",
            f"{float(camera.get('fps_target', 5.0)):g}",
            "--monochrome" if bool(camera.get("monochrome_output", False)) else "--color",
            "--encoders",
            ",".join(encoders),
            "--json-out",
            str(self.encode_benchmark_path),
        ]
        source = self._benchmark_source_video(candidate)
        if source is not None:
            arguments += ["--source-video", str(source)]
        self.run_bbx_command(arguments)

    def _save_settings(self, *, offer_apply: bool) -> bool:
        field_problems = self._settings_field_problems()
        if field_problems:
            QMessageBox.critical(
                self, "Settings not saved", "These values cannot be used:\n" + "\n".join(field_problems)
            )
            return False
        try:
            config = load_config(self.config_path) if self.config_path.exists() else copy.deepcopy(self.config)
            schedule_before = schedule_values(config)
            self._apply_settings_to(config)
            save_config(self.config_path, config)
        except Exception as exc:
            QMessageBox.critical(self, "Settings not saved", f"The recording settings were not saved:\n{exc}")
            return False
        self.config = load_config(self.config_path)
        if schedule_values(self.config) != schedule_before:
            self._offer_same_schedule_to_other_camera()
        self._load_settings_form()
        self.refresh_from_config()
        if offer_apply:
            self._offer_to_apply_schedule()
        return True

    def _offer_same_schedule_to_other_camera(self) -> None:
        """After a schedule change on one camera, offer to keep the two cameras in step."""
        other = self._other_camera()
        if other is None:
            return
        try:
            other_config = load_config(other.path)
        except Exception:
            return
        if schedules_match(self.config, other_config) or not self._ask_copy_schedule(other):
            return
        copy_schedule(self.config, other_config)
        try:
            save_config(other.path, other_config)
        except Exception as exc:
            QMessageBox.critical(self, "Schedule not copied", f"{other.label}'s settings were not changed:\n{exc}")

    def _ask_copy_schedule(self, other: CameraEntry) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(f"Use this schedule for {other.label} too?")
        box.setText(
            f"You changed when {self._active_camera().label} records. {other.label} still has its own schedule."
        )
        box.setInformativeText(
            "With the same run mode, recording length and interval, both cameras record at the same moments."
        )
        both = box.addButton("Both Cameras", QMessageBox.AcceptRole)
        only = box.addButton(f"{self._active_camera().label} Only", QMessageBox.RejectRole)
        box.setDefaultButton(both)
        box.setEscapeButton(only)
        box.exec_()
        return box.clickedButton() is both

    def _offer_to_apply_schedule(self) -> None:
        """Offer to restart any running schedule whose timers no longer match its settings."""
        stale: list[CameraEntry] = []
        for entry in self.cameras:
            status = self._camera_statuses.get(entry.index)
            if entry.index == self.camera_index and self._box_status is not None:
                status = self._box_status
            if status is None or status.schedule.state not in {SCHEDULE_ON, SCHEDULE_PARTIAL}:
                continue
            if installed_units_are_current(entry.config, entry.path) is False:
                stale.append(entry)
        if stale and self._ask_apply_schedule():
            self._start_automation(stale)

    def _ask_apply_schedule(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Apply to the running schedule?")
        box.setText("The recording schedule is still using the previous run mode, interval or recording length.")
        box.setInformativeText(
            "Other changes, such as the frame rate or video format, are used by the next recording without this."
        )
        apply_now = box.addButton("Apply Now", QMessageBox.AcceptRole)
        later = box.addButton("Later", QMessageBox.RejectRole)
        box.setDefaultButton(apply_now)
        box.setEscapeButton(later)
        box.exec_()
        return box.clickedButton() is apply_now

    def _resolve_unsaved_settings(self, action: str) -> bool:
        """Before an action that reads the saved config, settle unsaved settings."""
        if not self._settings_dirty():
            return True
        choice = self._ask_unsaved_settings(action)
        if choice == "save":
            return self._save_settings(offer_apply=False)
        return choice == "ignore"

    def _ask_unsaved_settings(self, action: str) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Save the recording settings first?")
        box.setText(f"The Settings page has changes that are not saved. If you {action} now, it uses the saved settings.")
        save = box.addButton("Save and Continue", QMessageBox.AcceptRole)
        ignore = box.addButton("Continue Without Saving", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save)
        box.setEscapeButton(cancel)
        box.exec_()
        clicked = box.clickedButton()
        return "save" if clicked is save else "ignore" if clicked is ignore else "cancel"

    def _ask_unsaved_settings_on_close(self) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Save the recording settings?")
        box.setText("The Settings page has changes that are not saved.")
        save = box.addButton("Save", QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save)
        box.setEscapeButton(cancel)
        box.exec_()
        clicked = box.clickedButton()
        return "save" if clicked is save else "discard" if clicked is discard else "cancel"

    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Advanced tools",
            "Specialized tracking, calibration, fleet, and storage editors remain available while their Qt pages are migrated.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        legacy = QPushButton("Open Legacy Advanced Tools")
        legacy.setObjectName("Quiet")
        legacy.clicked.connect(lambda: self.run_bbx_command(["gui", "--legacy"], detached=True))
        config = QPushButton("Open Config File")
        config.setObjectName("Quiet")
        config.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config_path))))
        docs = QPushButton("Open Project Documentation")
        docs.setObjectName("Quiet")
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(REPO_ROOT / "docs"))))
        workspace = QGroupBox("Workspace")
        workspace.setProperty("tone", "mint")
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.addWidget(legacy)
        workspace_layout.addWidget(config)
        workspace_layout.addWidget(docs)
        workspace_layout.addStretch(1)
        layout.addWidget(workspace)

        calibration = QGroupBox("RGB + thermal + depth calibration")
        calibration.setProperty("tone", "peach")
        calibration_layout = QVBoxLayout(calibration)
        calibration_note = QLabel(
            "Create a versioned project for shared-cue timing captures, multiple nest-depth layers, "
            "spatial transforms, and held-out validation. Large recordings stay in their session folders."
        )
        calibration_note.setWordWrap(True)
        calibration_layout.addWidget(calibration_note)
        self.calibration_project_label = QLabel()
        self.calibration_project_label.setWordWrap(True)
        self.calibration_project_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        calibration_layout.addWidget(self.calibration_project_label)
        calibration_buttons = QGridLayout()
        create_calibration = QPushButton("Create Project")
        create_calibration.setObjectName("Accent")
        create_calibration.clicked.connect(self._create_calibration_project)
        choose_calibration = QPushButton("Use Existing Project")
        choose_calibration.setObjectName("Quiet")
        choose_calibration.clicked.connect(self._choose_calibration_project)
        self.open_calibration_button = QPushButton("Open Project")
        self.open_calibration_button.setObjectName("Quiet")
        self.open_calibration_button.clicked.connect(self._open_calibration_project)
        self.calibration_status_button = QPushButton("Check Readiness")
        self.calibration_status_button.clicked.connect(self._show_calibration_project_status)
        calibration_guide = QPushButton("Open Workflow Guide")
        calibration_guide.setObjectName("Quiet")
        calibration_guide.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(REPO_ROOT / "docs" / "Multimodal_Calibration.md"))
            )
        )
        for index, button in enumerate((
            create_calibration,
            choose_calibration,
            self.open_calibration_button,
            self.calibration_status_button,
            calibration_guide,
        )):
            calibration_buttons.addWidget(button, index // 3, index % 3)
        calibration_layout.addLayout(calibration_buttons)
        layout.addWidget(calibration)
        migration = QLabel(
            "Migration rule: working capture and analysis services are reused; only the operator interface is being replaced. "
            "This keeps the Qt transition testable and prevents a GUI rewrite from changing scientific outputs."
        )
        migration.setObjectName("Hint")
        migration.setWordWrap(True)
        layout.addWidget(migration)
        layout.addStretch(1)
        return page

    def _build_console(self) -> QGroupBox:
        box = QGroupBox("Command output")
        box.setProperty("tone", "mint")
        layout = QVBoxLayout(box)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        # The controls above get the height they ask for first; the console
        # takes whatever is left, never less than a few lines.
        self.console.setMinimumHeight(96)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.console.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.console)
        return box

    def _latest_run_payload(self) -> dict[str, Any] | None:
        if not self._run_history_payloads:
            return None
        return self._run_history_payloads[0]

    @staticmethod
    def _run_artifact_path(payload: dict[str, Any] | None, key: str) -> Path | None:
        if payload is None:
            return None
        raw_path = str(payload.get(key) or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        return path.resolve() if path.exists() else None

    def _open_run_artifact(
        self,
        payload: dict[str, Any] | None,
        key: str,
        *,
        open_parent: bool = False,
    ) -> None:
        path = self._run_artifact_path(payload, key)
        if path is None:
            QMessageBox.warning(
                self,
                "Output not available",
                "That output was not recorded or the referenced file is no longer available.",
            )
            return
        target = path.parent if open_parent and path.is_file() else path
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            QMessageBox.warning(self, "Could not open output", str(target))

    @staticmethod
    def _format_run_timestamp(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "Unknown"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S") if parsed.tzinfo else parsed.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _format_optional_fps(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "n/a"

    def _format_run_detail(self, payload: dict[str, Any]) -> str:
        warning_count = len(payload.get("warnings", []) or [])
        error_count = len(payload.get("errors", []) or [])
        thermal_status = "disabled"
        if bool(payload.get("thermal_enabled", False)):
            thermal_status = (
                f"{int(payload.get('thermal_frames_captured', 0) or 0)} frames at "
                f"{self._format_optional_fps(payload.get('thermal_actual_fps'))} fps"
            )
        depth_status = "disabled"
        if bool(payload.get("realsense_enabled", False)):
            depth_status = (
                f"{int(payload.get('realsense_frames_captured', 0) or 0)} frames at "
                f"{self._format_optional_fps(payload.get('realsense_actual_fps'))} fps; "
                f"scale {payload.get('realsense_depth_scale_meters') or 'n/a'} m/unit"
            )
        distributed = payload.get("distributed_capture")
        distributed_status = "disabled"
        if isinstance(distributed, dict) and bool(distributed.get("enabled", False)):
            nodes = distributed.get("nodes", {})
            successful = sum(
                1 for item in nodes.values() if isinstance(item, dict) and bool(item.get("success"))
            ) if isinstance(nodes, dict) else 0
            distributed_status = (
                f"plan {distributed.get('plan_id') or 'unknown'}; "
                f"{successful}/{len(nodes) if isinstance(nodes, dict) else 0} nodes successful"
            )
        # Lines for optional hardware appear only when this box has it or the
        # recording used it.
        hardware = self._configured_hardware() | self._payload_hardware(payload)
        lines = [
            f"Session: {payload.get('session_name') or 'unknown'}",
            f"Started: {self._format_run_timestamp(payload.get('started_at'))}",
            f"Result: {'Success' if bool(payload.get('success', False)) else 'Failed'} | "
            f"Mode: {payload.get('mode') or 'unknown'}",
            f"RGB: {int(payload.get('frames_captured', 0) or 0)} frames at "
            f"{self._format_optional_fps(payload.get('actual_fps'))} fps",
        ]
        parts = payload.get("video_parts") or []
        if len(parts) > 1:
            lines.append(f"Video: {len(parts)} files, one after the other (open the session folder to see them all)")
        if "thermal" in hardware:
            lines.append(f"Thermal: {thermal_status}")
        if "realsense" in hardware:
            lines.append(f"RealSense: {depth_status}")
        if "distributed" in hardware:
            lines.append(f"Distributed capture: {distributed_status}")
        lines.append(f"Warnings: {warning_count} | Errors: {error_count}")
        return "\n".join(lines)

    def _set_result_artifact_buttons(self, payload: dict[str, Any] | None) -> None:
        for button, payload_key in self.result_artifact_buttons.values():
            button.setEnabled(self._run_artifact_path(payload, payload_key) is not None)
        summary_available = self._run_artifact_path(payload, "_summary_path") is not None
        self.result_add_calibration_button.setEnabled(
            summary_available and self._configured_calibration_project() is not None
        )
        self.result_sync_analysis_button.setEnabled(summary_available)
        self._layout_result_buttons(self._configured_hardware() | self._payload_hardware(payload))

    def _configured_hardware(self) -> set[str]:
        """Optional hardware this box is set up to use."""
        hardware: set[str] = set()
        if bool(self.config.get("thermal", {}).get("enabled", False)):
            hardware.add("thermal")
        if bool(self.config.get("realsense", {}).get("enabled", False)):
            hardware.add("realsense")
        if bool(self.config.get("distributed_capture", {}).get("enabled", False)):
            hardware.add("distributed")
        if hardware & {"thermal", "realsense"}:
            hardware.add("multimodal")
        return hardware

    @staticmethod
    def _payload_hardware(payload: dict[str, Any] | None) -> set[str]:
        """Optional hardware a recording used, so older recordings keep their buttons."""
        if not payload:
            return set()
        hardware: set[str] = set()
        if bool(payload.get("thermal_enabled", False)):
            hardware.add("thermal")
        if bool(payload.get("realsense_enabled", False)):
            hardware.add("realsense")
        distributed = payload.get("distributed_capture")
        if isinstance(distributed, dict) and bool(distributed.get("enabled", False)):
            hardware.add("distributed")
        if hardware & {"thermal", "realsense"}:
            hardware.add("multimodal")
        return hardware

    def _layout_result_buttons(self, hardware: set[str]) -> None:
        visible = tuple(
            key
            for key, _ in self.result_button_order
            if RESULT_BUTTON_HARDWARE.get(key) is None or RESULT_BUTTON_HARDWARE[key] in hardware
        )
        if visible == self._visible_result_buttons:
            return
        self._visible_result_buttons = visible
        for _, button in self.result_button_order:
            self.result_button_grid.removeWidget(button)
            button.setVisible(False)
        columns = 3
        for index, key in enumerate(visible):
            button = dict(self.result_button_order)[key]
            self.result_button_grid.addWidget(button, index // columns, index % columns)
            button.setVisible(True)

    def _update_results_hardware_columns(self) -> None:
        hardware = self._configured_hardware()
        for payload in self._run_history_payloads:
            hardware |= self._payload_hardware(payload)
        self.results_table.setColumnHidden(4, "thermal" not in hardware)
        self.results_table.setColumnHidden(5, "realsense" not in hardware)

    def _analyze_selected_run_sync(self) -> None:
        summary = self._run_artifact_path(self._selected_run_payload, "_summary_path")
        if summary is None:
            QMessageBox.warning(self, "Run summary unavailable", "Select a locally available recording first.")
            return
        self.run_bbx_command(
            ["distributed-capture", "analyze-sync", "--summary", str(summary)]
        )

    def _configured_calibration_project(self) -> Path | None:
        raw_path = str(
            self.config.get("calibration", {}).get("multimodal_project_path") or ""
        ).strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if not (path / CALIBRATION_MANIFEST_NAME).is_file():
            return None
        return path.resolve()

    def _save_calibration_project_path(self, project_dir: Path) -> None:
        self.config.setdefault("calibration", {})["multimodal_project_path"] = str(
            project_dir.resolve()
        )
        save_config(self.config_path, self.config)
        self.refresh_from_config()
        self._set_result_artifact_buttons(self._selected_run_payload)

    def _create_calibration_project(self) -> None:
        parent = QFileDialog.getExistingDirectory(
            self,
            "Choose where to create the calibration project",
            str(Path(self.config.get("system", {}).get("data_root", Path.home())).expanduser()),
        )
        if not parent:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Calibration project name",
            "Project folder and display name:",
            text="BumbleBoxCalibration",
        )
        if not accepted:
            return
        safe_name = str(name).strip().replace("/", "-").replace("\\", "-")
        if not safe_name:
            QMessageBox.warning(self, "Project name required", "Enter a project name.")
            return
        project_dir = Path(parent).expanduser().resolve() / safe_name
        try:
            result = initialize_calibration_project(
                self.config,
                project_dir,
                project_name=str(name).strip(),
            )
            self._save_calibration_project_path(Path(result.project_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Could not create calibration project", str(exc))
            return
        QMessageBox.information(
            self,
            "Calibration project created",
            f"Project created at:\n{result.project_dir}\n\nRegister suitable runs from the Results page.",
        )

    def _choose_calibration_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose an existing calibration project",
            str(Path.home()),
        )
        if not directory:
            return
        project_dir = Path(directory).expanduser().resolve()
        try:
            get_calibration_project_status(project_dir)
            self._save_calibration_project_path(project_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid calibration project", str(exc))

    def _open_calibration_project(self) -> None:
        project_dir = self._configured_calibration_project()
        if project_dir is None:
            QMessageBox.warning(self, "No calibration project", "Create or select a project first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(project_dir)))

    def _show_calibration_project_status(self) -> None:
        project_dir = self._configured_calibration_project()
        if project_dir is None:
            QMessageBox.warning(self, "No calibration project", "Create or select a project first.")
            return
        try:
            report = format_calibration_project_status(
                get_calibration_project_status(project_dir)
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not read calibration project", str(exc))
            return
        QMessageBox.information(self, "Calibration readiness", report)

    def _add_selected_run_to_calibration(self) -> None:
        project_dir = self._configured_calibration_project()
        summary_path = self._run_artifact_path(self._selected_run_payload, "_summary_path")
        if project_dir is None or summary_path is None:
            QMessageBox.warning(
                self,
                "Calibration input unavailable",
                "Select a run and configure a calibration project first.",
            )
            return
        depth_layer, accepted = QInputDialog.getText(
            self,
            "Calibration depth layer",
            "Physical depth label (for example floor, mid, or upper):",
        )
        if not accepted:
            return
        if not str(depth_layer).strip():
            QMessageBox.warning(
                self,
                "Depth layer required",
                "A depth-layer label is required so multi-depth coverage can be verified.",
            )
            return
        try:
            result = add_calibration_session(
                project_dir,
                summary_path,
                depth_layer=str(depth_layer).strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not add calibration capture", str(exc))
            return
        state = "registered" if result.added else "was already registered"
        QMessageBox.information(
            self,
            "Calibration capture",
            f"{result.capture_id} {state}.\nAvailable streams: "
            f"{', '.join(result.available_streams) or 'none'}\nMissing streams: "
            f"{', '.join(result.missing_streams) or 'none'}",
        )

    def _result_selection_changed(self) -> None:
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            self._selected_run_payload = None
            self.result_detail.setText("Select a recording to inspect its outputs.")
            self._set_result_artifact_buttons(None)
            return
        row = selected_rows[0].row()
        item = self.results_table.item(row, 0)
        payload_index = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(payload_index, int) or not (0 <= payload_index < len(self._run_history_payloads)):
            self._selected_run_payload = None
            self._set_result_artifact_buttons(None)
            return
        payload = self._run_history_payloads[payload_index]
        self._selected_run_payload = payload
        self.result_detail.setText(self._format_run_detail(payload))
        self._set_result_artifact_buttons(payload)

    def refresh_results(self) -> None:
        if self._run_history_loading:
            self._run_history_refresh_pending = True
            return
        self._run_history_generation += 1
        generation = self._run_history_generation
        self._run_history_loading = True
        self.results_refresh_button.setEnabled(False)
        self.results_status.setText("Scanning recent recording summaries...")
        self.latest_run_status.setText("Loading recent recording status...")
        targets = [
            (entry.index, str(entry.config.get("system", {}).get("data_root", "")).strip(), entry.tag)
            for entry in self.cameras
        ]
        worker = _RunHistoryWorker(generation, targets, self.camera_index, limit=60)
        worker.signals.finished.connect(self._run_history_loaded)
        self._run_history_workers[generation] = worker
        self.thread_pool.start(worker)

    def _run_history_loaded(
        self,
        generation: int,
        payloads: object,
        error: object,
        others: object = None,
    ) -> None:
        self._run_history_workers.pop(generation, None)
        if generation != self._run_history_generation:
            return
        self._run_history_loading = False
        self._other_camera_payloads = others if isinstance(others, dict) else {}
        self.results_refresh_button.setEnabled(True)
        self._selected_run_payload = None
        self.results_table.setRowCount(0)

        self._run_history_error = str(error) if error else None
        if error:
            self._run_history_payloads = []
            self._update_latest_card()
            self.results_status.setText(f"Could not scan recording summaries: {error}")
            self.latest_session_button.setEnabled(False)
            self._set_latest_depth_available(False)
            self._set_result_artifact_buttons(None)
            if self._run_history_refresh_pending:
                self._run_history_refresh_pending = False
                QTimer.singleShot(0, self.refresh_results)
            return

        normalized = [dict(item) for item in payloads if isinstance(item, dict)] if isinstance(payloads, list) else []
        self._run_history_payloads = normalized
        self._update_latest_card()
        self._update_results_hardware_columns()
        self._update_storage_estimate()
        data_root = str(self.config.get("system", {}).get("data_root", "Not set"))
        self.results_status.setText(f"Showing {len(normalized)} recent recording(s) from {data_root}")

        for row, payload in enumerate(normalized):
            rgb_frames = int(payload.get("frames_captured", 0) or 0)
            thermal_frames = int(payload.get("thermal_frames_captured", 0) or 0)
            depth_frames = int(payload.get("realsense_frames_captured", 0) or 0)
            values = (
                self._format_run_timestamp(payload.get("started_at")),
                "Success" if bool(payload.get("success", False)) else "Failed",
                str(payload.get("mode") or "unknown"),
                str(rgb_frames),
                str(thermal_frames) if bool(payload.get("thermal_enabled", False)) else "Off",
                str(depth_frames) if bool(payload.get("realsense_enabled", False)) else "Off",
                self._format_optional_fps(payload.get("actual_fps")),
            )
            self.results_table.insertRow(row)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, row)
                self.results_table.setItem(row, column, item)

        latest = self._latest_run_payload()
        if latest is None:
            self.latest_session_button.setEnabled(False)
            self._set_latest_depth_available(False)
            self.result_detail.setText("No recording results are available.")
            self._set_result_artifact_buttons(None)
            if self._run_history_refresh_pending:
                self._run_history_refresh_pending = False
                QTimer.singleShot(0, self.refresh_results)
            return

        self.latest_session_button.setEnabled(
            self._run_artifact_path(latest, "_session_dir_local") is not None
        )
        self._set_latest_depth_available(
            self._run_artifact_path(latest, "realsense_depth_preview_video_path") is not None
        )
        self.results_table.selectRow(0)
        if self._run_history_refresh_pending:
            self._run_history_refresh_pending = False
            QTimer.singleShot(0, self.refresh_results)

    def _set_latest_depth_available(self, available: bool) -> None:
        # Shown only when there is a depth preview to open, so a box without a
        # depth camera does not carry a permanently disabled button.
        self.latest_depth_button.setEnabled(available)
        self.latest_depth_button.setVisible(available)

    def _navigate(self, index: int) -> None:
        if index < 0:
            return
        self.pages.setCurrentIndex(index)
        self._update_page_title()
        if index == PAGE_INDEX_RESULTS and not self._run_history_payloads:
            self.refresh_results()

    def open_setup_wizard(self) -> None:
        wizard = BumbleBoxSetupWizard(self.config_path, self.config, self)
        if wizard.exec_() == QWizard.Accepted:
            self.config = load_config(self.config_path)
            self.refresh_from_config()
            QMessageBox.information(
                self,
                "Setup saved",
                "The hardware profile and experiment basics were saved. Run the visible hardware checks before recording.",
            )

    def refresh_from_config(self) -> None:
        # Whatever is on disk now is what this window has seen.
        self._config_stamp_seen = self._config_file_stamp()
        self._reload_cameras()
        setup = self.config.get("setup", {})
        profile_key = str(setup.get("hardware_profile", "custom"))
        profile = HARDWARE_PROFILES.get(profile_key, HARDWARE_PROFILES["custom"])
        self.profile_badge.setText(profile.label)
        completed = bool(setup.get("completed", False))
        self.setup_status.setObjectName("StatusGood" if completed else "StatusWarn")
        self.setup_status.setText(
            "Setup profile saved. Run hardware checks before production capture."
            if completed
            else "Setup is incomplete. Run the wizard before recording."
        )
        self.setup_status.style().unpolish(self.setup_status)
        self.setup_status.style().polish(self.setup_status)

        camera_profile = str(self.config.get("camera", {}).get("profile", "custom"))
        camera_label = CAMERA_PROFILES.get(camera_profile, CAMERA_PROFILES["custom"]).label
        thermal_enabled = bool(self.config.get("thermal", {}).get("enabled", False))
        depth_enabled = bool(self.config.get("realsense", {}).get("enabled", False))
        distributed = self.config.get("distributed_capture", {})
        distributed_enabled = bool(distributed.get("enabled", False))
        local_sensors = {"rgb", "thermal", "realsense"}
        if distributed_enabled:
            local_node = next(
                (node for node in distributed.get("nodes", []) if bool(node.get("local", False))),
                {},
            )
            local_sensors = {str(item) for item in local_node.get("sensors", [])}
        self.camera_card.value_label.setText(
            camera_label + (" on second Pi" if distributed_enabled and "rgb" not in local_sensors else "")
        )
        self.thermal_card.value_label.setText(
            "Enabled on this Pi" if thermal_enabled and "thermal" in local_sensors
            else "Enabled on second Pi" if thermal_enabled
            else "Not used"
        )
        self.realsense_card.value_label.setText(
            "Enabled on this Pi" if depth_enabled and "realsense" in local_sensors
            else "Enabled on second Pi" if depth_enabled
            else "Not used"
        )
        if self._box_status is None:
            self._set_card(
                self.storage_card,
                "Checking...",
                str(self.config.get("system", {}).get("data_root", "Not set")),
                "neutral",
            )
        self.rgb_group.setVisible(not distributed_enabled or "rgb" in local_sensors)
        self.thermal_group.setVisible(thermal_enabled and (not distributed_enabled or "thermal" in local_sensors))
        self.realsense_group.setVisible(depth_enabled and (not distributed_enabled or "realsense" in local_sensors))
        node_descriptions = []
        for node in distributed.get("nodes", []):
            if not bool(node.get("enabled", True)):
                continue
            location = "this Pi" if bool(node.get("local", False)) else str(node.get("host") or "second Pi")
            node_descriptions.append(
                f"{location}: {', '.join(str(item) for item in node.get('sensors', [])) or 'no cameras'}"
            )
        distributed_text = "; ".join(node_descriptions)
        self.distributed_run_group.setVisible(distributed_enabled)
        self.distributed_hardware_group.setVisible(distributed_enabled)
        self.distributed_run_status.setText(
            "Assignments: " + distributed_text + ". Run a check before production capture."
        )
        self.distributed_hardware_label.setText(distributed_text)
        _set_combo_data(self.run_mode, str(self.config.get("pipeline", {}).get("mode", "record_and_track")))
        calibration_path = self._configured_calibration_project()
        configured_path = str(
            self.config.get("calibration", {}).get("multimodal_project_path") or ""
        ).strip()
        if calibration_path is not None:
            self.calibration_project_label.setText(f"Current project: {calibration_path}")
        elif configured_path:
            self.calibration_project_label.setText(
                f"Configured project is unavailable: {configured_path}"
            )
        else:
            self.calibration_project_label.setText("No multimodal calibration project selected.")
        self.open_calibration_button.setEnabled(calibration_path is not None)
        self.calibration_status_button.setEnabled(calibration_path is not None)
        self._set_result_artifact_buttons(self._selected_run_payload)
        self._update_results_hardware_columns()
        if not self._settings_dirty():
            self._load_settings_form()
        self._show_record_grayscale(recording_is_grayscale(self.config))
        self._update_camera_widgets()
        self._update_storage_estimate()
        self.refresh_box_status()

    # -- Two cameras on one Pi --------------------------------------------------

    def _reload_cameras(self) -> None:
        """Re-read which cameras this box has (one config file each)."""
        try:
            cameras = load_cameras(
                self.primary_config_path,
                self.config if self.camera_index == 1 else _load_config_or_defaults(self.primary_config_path),
            )
        except Exception:
            # A second config that cannot be read must not take the window down
            # with it; carry on with the first camera alone.
            primary = self.config if self.camera_index == 1 else _load_config_or_defaults(self.primary_config_path)
            cameras = [CameraEntry(1, self.primary_config_path, primary)]
        if self.camera_index > len(cameras):
            self.camera_index = 1
            self.config_path = self.primary_config_path
            self.config = cameras[0].config
            self._reset_camera_view()
        cameras[self.camera_index - 1].config = self.config
        self.cameras = cameras

    def _active_camera(self) -> CameraEntry:
        return self.cameras[self.camera_index - 1]

    def _other_camera(self) -> CameraEntry | None:
        return next((entry for entry in self.cameras if entry.index != self.camera_index), None)

    def _reset_camera_view(self) -> None:
        """Forget what was shown for the previous camera."""
        self._box_status = self._camera_statuses.get(self.camera_index)
        # A scan still running was started for the other camera; its result
        # must not land in this camera's view.
        self._run_history_generation += 1
        self._run_history_loading = False
        self._run_history_refresh_pending = False
        self._run_history_payloads = []
        self._other_camera_payloads = {}
        self._run_history_error = None
        self._selected_run_payload = None
        self._last_scheduled_run_seen = None
        self._storage_estimate = None
        self._settings_saved_values = None
        self.results_table.setRowCount(0)

    def _update_page_title(self) -> None:
        item = self.navigation.currentItem()
        title = item.text().upper() if item else "BUMBLEBOX"
        if len(self.cameras) == 2:
            title = f"{title}  ·  {self._active_camera().label.upper()}"
        self.page_title.setText(title)

    def _update_camera_widgets(self) -> None:
        two = len(self.cameras) == 2
        active = self._active_camera()
        self.camera_switch.setVisible(two)
        for index, button in self.camera_buttons.items():
            if index <= len(self.cameras):
                button.setText(self.cameras[index - 1].long_label)
            button.setChecked(index == self.camera_index)
        self.cameras_strip.setVisible(two)
        self.run_both_button.setVisible(two)
        self.both_cameras_checkbox.setVisible(two)
        self.run_once_button.setText(f"Record {active.label} Now" if two else "Run Once Now")
        self.overview_subtitle.setText(
            f"Showing {active.long_label}. Use the camera buttons at the top right to see the other one."
            if two
            else "A concise view of this BumbleBox. Device-specific setup stays hidden unless the device is enabled."
        )
        self.rgb_group.setTitle(active.long_label if two else "Primary camera")
        self.camera_card.title_label.setText((active.label if two else "Primary camera").upper())

        if two:
            first, second = self.cameras
            pinned = first.pinned and second.pinned
            lines = [
                f"{second.long_label} records alongside {first.long_label}. Each has its own settings, "
                "schedule and status; pick one with the camera buttons at the top right.",
                (
                    "Each camera's settings are pinned to one physical camera."
                    if pinned
                    else "The cameras are matched by number, not pinned. If one is unplugged, the other could "
                    "record in its place. Connect both and press Pin Cameras."
                ),
            ]
            lines += [f"Problem: {problem}" for problem in check_camera_set(self.cameras)]
            self.second_camera_label.setText("\n".join(lines))
        else:
            self.second_camera_label.setText(
                "This Pi can record from two Raspberry Pi cameras at the same time. The second camera gets "
                "its own settings, schedule and status, and can record into the same data folder."
            )
        self.add_second_camera_button.setVisible(not two)
        self.pin_cameras_button.setVisible(two and not all(entry.pinned for entry in self.cameras))
        self.remove_second_camera_button.setVisible(two)
        self._render_camera_strip()
        self._update_page_title()

    def _render_camera_strip(self) -> None:
        """One line per camera on Overview: schedule state, and storage if it is refusing."""
        for entry in self.cameras:
            label = self.camera_status_labels.get(entry.index)
            if label is None:
                continue
            status = self._camera_statuses.get(entry.index)
            shown = " (shown below)" if entry.index == self.camera_index else ""
            if status is None:
                self._set_status_label(label, LEVEL_OFF, f"{entry.long_label}{shown}: checking...")
                continue
            text = f"{entry.long_label}{shown}: {status.schedule.headline}"
            level = status.schedule.level
            if status.storage.level == LEVEL_FAIL:
                text += f"  ·  {status.storage.headline}"
                level = LEVEL_FAIL
            self._set_status_label(label, level, text)

    def _select_camera(self, index: int) -> None:
        if index == self.camera_index or index > len(self.cameras):
            self._update_camera_widgets()
            return
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self,
                "Command still running",
                "Wait for the current command to finish, or cancel it on the Run page, before switching cameras.",
            )
            self._update_camera_widgets()
            return
        if self._settings_dirty():
            choice = self._ask_unsaved_settings_on_close()
            if choice == "cancel" or (choice == "save" and not self._save_settings(offer_apply=False)):
                self._update_camera_widgets()
                return
        entry = self.cameras[index - 1]
        self.camera_index = index
        self.config_path = entry.path
        self.config = _load_config_or_defaults(entry.path)
        self._reset_camera_view()
        self.refresh_from_config()
        if self._box_status is not None:
            self._apply_box_status(self._box_status)
        self.refresh_results()

    def _add_second_camera(self) -> None:
        if len(self.cameras) == 2:
            return
        detected = detect_cameras()
        details = self._ask_second_camera_details(detected)
        if details is None:
            return
        colony_id, data_root = details
        first_root = str(self.cameras[0].config.get("system", {}).get("data_root", "")).strip()
        try:
            result = create_second_camera(
                self.primary_config_path,
                colony_id=colony_id,
                data_root=data_root if data_root and data_root != first_root else None,
                cameras=detected,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Second camera not added", str(exc))
            return
        if self.camera_index == 1 and self.config_path.exists():
            self.config = load_config(self.config_path)  # pinning may have changed it
        self.refresh_from_config()
        self.refresh_results()
        QMessageBox.information(
            self,
            "Second camera added",
            "Camera 2 starts with Camera 1's recording settings, and its schedule is off.\n\n"
            + "\n".join(result.notes)
            + "\n\nNext: pick Camera 2 at the top right, check its camera on Hardware, then start both "
            "schedules on Run.",
        )

    def _ask_second_camera_details(self, detected: list[DetectedCamera]) -> tuple[str, str] | None:
        """Ask for the second camera's colony ID and data folder; ``None`` if cancelled."""
        primary = self.cameras[0].config
        dialog = SecondCameraDialog(
            self,
            detected=detected,
            colony_id=suggest_second_colony_id(primary),
            data_root=str(primary.get("system", {}).get("data_root", "")),
        )
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.colony_id.text().strip(), dialog.data_root.text().strip()

    def _pin_cameras(self) -> None:
        try:
            pinned = pin_cameras(self.primary_config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cameras not pinned", str(exc))
            return
        if not pinned:
            QMessageBox.warning(
                self,
                "Cameras not pinned",
                "Both cameras must be connected and detected to pin them. Check the ribbon cables "
                "(Check Camera on this page), then try again.",
            )
            return
        self.config = load_config(self.config_path)
        self.refresh_from_config()
        QMessageBox.information(
            self, "Cameras pinned", "Each camera's settings are now tied to one physical camera."
        )

    def _remove_second_camera(self) -> None:
        if len(self.cameras) < 2:
            return
        second = self.cameras[1]
        if not self._confirm_remove_second_camera(second):
            return
        status = self._camera_statuses.get(2)
        scheduled = status is not None and status.schedule.state in {SCHEDULE_ON, SCHEDULE_PARTIAL}
        if scheduled:
            # Its timers point at its config file, so stop them before the file goes.
            self.run_bbx_commands(
                [["systemd-disable", "--config", str(second.path)]],
                on_success=self._finish_remove_second_camera,
            )
        else:
            self._finish_remove_second_camera()

    def _confirm_remove_second_camera(self, second: CameraEntry) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Remove the second camera?")
        box.setText(f"{second.long_label} will stop recording and leave this window.")
        box.setInformativeText(
            "Its recordings are not touched, and its settings are kept in a renamed file so it can be "
            "restored. Camera 1 is not affected."
        )
        remove = box.addButton("Remove Camera 2", QMessageBox.DestructiveRole)
        keep = box.addButton("Keep It", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec_()
        return box.clickedButton() is remove

    def _finish_remove_second_camera(self) -> None:
        try:
            kept_as = remove_second_camera(self.primary_config_path)
        except OSError as exc:
            QMessageBox.critical(self, "Second camera not removed", str(exc))
            return
        self._camera_statuses.pop(2, None)
        self.camera_index = 1
        self.config_path = self.primary_config_path
        self.config = _load_config_or_defaults(self.primary_config_path)
        self._reset_camera_view()
        self.refresh_from_config()
        self.refresh_results()
        QMessageBox.information(
            self,
            "Second camera removed",
            f"Its settings were kept as {kept_as.name}. Its recordings were not touched.",
        )

    def _config_file_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.config_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def changeEvent(self, event: Any) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._reload_if_config_changed()

    def _reload_if_config_changed(self) -> bool:
        """Pick up a config file that was changed outside this window.

        Called when the window comes back to the front, for example after the
        live monitor saved a recording setting. Unsaved edits on the Settings
        page are kept; only the fields the user changed are written on save.
        """
        stamp = self._config_file_stamp()
        if stamp is None or stamp == getattr(self, "_config_stamp_seen", None):
            return False
        self._config_stamp_seen = stamp
        if self.process.state() != QProcess.NotRunning:
            return False  # a running command reloads the config itself when it ends
        try:
            changed = load_config(self.config_path)
        except Exception:
            return False
        self._adopt_config(changed)
        return True

    def _adopt_config(self, config: dict[str, Any]) -> None:
        """Take a config that was changed outside the Settings page.

        Whether the Settings page has unsaved edits has to be judged before the
        swap: afterwards the form would be compared with the new config and
        look edited when it is only out of date.
        """
        untouched = not self._settings_dirty()
        self.config = config
        if untouched:
            self._load_settings_form()
        self.refresh_from_config()

    def closeEvent(self, event: Any) -> None:
        if self._settings_dirty():
            choice = self._ask_unsaved_settings_on_close()
            if choice == "cancel" or (choice == "save" and not self._save_settings(offer_apply=False)):
                event.ignore()
                return
        self._status_timer.stop()
        super().closeEvent(event)

    # -- Box status: storage, schedule, latest recording ---------------------

    def refresh_box_status(self) -> None:
        if self._box_status_loading:
            self._box_status_refresh_pending = True
            return
        self._box_status_generation += 1
        self._box_status_loading = True
        worker = _BoxStatusWorker(
            self._box_status_generation,
            [(entry.index, copy.deepcopy(entry.config), entry.path) for entry in self.cameras],
        )
        worker.signals.finished.connect(self._box_status_loaded)
        self._box_status_workers[worker.generation] = worker
        self.thread_pool.start(worker)

    def _box_status_loaded(self, generation: int, status: object) -> None:
        self._box_status_workers.pop(generation, None)
        if generation != self._box_status_generation:
            return
        self._box_status_loading = False
        statuses = {self.camera_index: status} if isinstance(status, BoxStatus) else status
        if isinstance(statuses, dict):
            self._camera_statuses = {
                int(index): value for index, value in statuses.items() if isinstance(value, BoxStatus)
            }
            shown = self._camera_statuses.get(self.camera_index)
            if shown is not None:
                self._apply_box_status(shown)
            self._render_camera_strip()
        if self._box_status_refresh_pending:
            self._box_status_refresh_pending = False
            QTimer.singleShot(0, self.refresh_box_status)

    def _apply_box_status(self, status: BoxStatus) -> None:
        self._box_status = status
        self._camera_statuses[self.camera_index] = status
        self._render_camera_strip()
        schedule = status.schedule
        self._update_storage_estimate()
        self._set_card(
            self.schedule_card, schedule.headline, schedule.detail, _LEVEL_CARD_TONES.get(schedule.level, "neutral")
        )
        self._set_status_label(self.schedule_status_label, schedule.level, f"Automated recording: {schedule.headline}")
        self.schedule_detail_label.setText(schedule.detail)
        self._update_latest_card()

        # A scheduled run that finished while this window was open produced a
        # new recording; rescan the data folder only then, not on every tick.
        finished = [
            timer.last_started
            for timer in schedule.timers
            if timer.last_started is not None and not timer.running
        ]
        newest = max(finished) if finished else None
        if newest is not None and newest != self._last_scheduled_run_seen:
            first_look = self._last_scheduled_run_seen is None
            self._last_scheduled_run_seen = newest
            if not first_look:
                self.refresh_results()

    def _update_storage_estimate(self) -> None:
        """Recompute storage per day from recent recordings and the free space."""
        free_gb = self._box_status.storage.free_gb if self._box_status is not None else None
        self._storage_estimate = estimate_storage(
            self.config,
            self._run_history_payloads,
            free_gb=free_gb,
            benchmark=self._load_benchmark(),
            sharing_drive=self._sharing_drive(),
        )
        self._render_storage_card()
        self._schedule_settings_preview()

    def _sharing_drive(self) -> tuple[str, float | None] | None:
        """The other camera's label and storage use per day, if it fills the same drive."""
        other = self._other_camera()
        if other is None:
            return None
        if not _same_drive(
            self.config.get("system", {}).get("data_root", ""), other.config.get("system", {}).get("data_root", "")
        ):
            return None
        other_estimate = estimate_storage(
            other.config, self._other_camera_payloads.get(other.index, []), benchmark=self._load_benchmark()
        )
        return other.label, other_estimate.bytes_per_day

    def _render_storage_card(self) -> None:
        if self._box_status is None:
            return
        storage = self._box_status.storage
        level = storage.level
        detail = storage.detail
        estimate = self._storage_estimate
        # When recording is refused the estimate would only distract from why.
        if estimate is not None and level != LEVEL_FAIL and estimate.level != LEVEL_OFF:
            detail = f"{detail}\n{estimate.headline}" if detail else estimate.headline
            if estimate.level == LEVEL_WARN and level == LEVEL_OK:
                level = LEVEL_WARN
        self._set_card(self.storage_card, storage.headline, detail, _LEVEL_CARD_TONES.get(level, "neutral"))
        missing = storage.headline == DRIVE_NOT_MOUNTED
        name = "Accent" if missing else "Quiet"
        self.data_drive_button.setText("Mount Data Drive..." if missing else "Data Drive...")
        if self.data_drive_button.objectName() != name:
            self.data_drive_button.setObjectName(name)
            self._repolish(self.data_drive_button)

    # -- Data drive ---------------------------------------------------------------

    def _drive_busy_reason(self) -> str | None:
        """Why the data drive must not be changed right now, if a recording is using it."""
        if self.process.state() != QProcess.NotRunning:
            return "A BumbleBox command is running. Wait for it to finish, or cancel it on the Run page."
        for status in self._camera_statuses.values():
            if any(timer.running for timer in status.schedule.timers):
                return "A scheduled recording is running. Wait for it to finish before changing the data drive."
        try:
            if capture_in_progress(entry.config for entry in self.cameras):
                return "A recording is in progress. Wait for it to finish before changing the data drive."
        except Exception:
            pass
        return None

    def _open_data_drive(self) -> None:
        dialog = DataDriveDialog(
            self,
            config_path=self.config_path,
            config=self.config,
            on_choice=self._save_drive_choice,
            camera_label=self._active_camera().label if len(self.cameras) > 1 else "",
            busy_reason=self._drive_busy_reason,
        )
        dialog.exec_()
        self.refresh_box_status()

    def _save_drive_choice(self, data_folder: str | None, require_drive: bool) -> dict[str, Any]:
        """Record what the Data Drive window did, for this camera and one sharing its folder."""
        old_folder = str(self.config.get("system", {}).get("data_root", "") or "")
        paths = [self.config_path]
        other = self._other_camera()
        if other is not None and old_folder:
            other_folder = str(other.config.get("system", {}).get("data_root", "") or "")
            if other_folder and os.path.abspath(other_folder) == os.path.abspath(old_folder):
                paths.append(other.path)
        save_drive_choice(paths, require_drive=require_drive, data_folder=data_folder)
        self._adopt_config(load_config(self.config_path))
        if data_folder and os.path.abspath(data_folder) != os.path.abspath(old_folder or data_folder):
            self.refresh_results()
        return self.config

    @staticmethod
    def _parse_run_start(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed

    def _update_latest_card(self) -> None:
        if self._run_history_loading and not self._run_history_payloads:
            return
        headline, detail, tone = self._latest_recording_summary()
        self._set_card(self.latest_card, headline, detail, tone)
        self.latest_run_status.setText(f"{headline}\n{detail}" if detail else headline)

    def _latest_recording_summary(self) -> tuple[str, str, str]:
        """Headline, detail and card tone for the newest recording in the data folder."""
        if self._run_history_error:
            return (
                "Unavailable",
                f"The data folder could not be read: {self._run_history_error}",
                "warn",
            )
        latest = self._latest_run_payload()
        if latest is None:
            return "No recordings yet", "", "neutral"
        now = datetime.now()
        started = self._parse_run_start(latest.get("started_at"))
        when = (
            f"{describe_moment(started, now)} ({describe_relative(started, now)})"
            if started is not None
            else "Unknown time"
        )
        if bool(latest.get("success", False)):
            frames = int(latest.get("frames_captured", 0) or 0)
            try:
                fps = f"{float(latest.get('actual_fps')):.1f} fps"
            except (TypeError, ValueError):
                fps = "unknown fps"
            mode = str(latest.get("mode") or "unknown mode").replace("_", " ")
            return f"{when} · Success", f"{frames} frames at {fps}, {mode}", "mint"
        errors = [str(item).strip() for item in (latest.get("errors") or []) if str(item).strip()]
        return (
            f"{when} · FAILED",
            errors[0] if errors else "No reason was recorded; see Results for this session.",
            "fail",
        )

    def _run_once(self) -> None:
        if not self._confirm_live_monitor_closed("start a recording"):
            return
        if not self._resolve_unsaved_settings("start a recording"):
            return
        self.run_bbx_command(["run-once", "--mode", str(self.run_mode.currentData())])

    def _show_record_grayscale(self, enabled: bool) -> None:
        self.record_gray_button.setChecked(bool(enabled))
        self.record_gray_button.setText(f"Record in Grayscale: {'On' if enabled else 'Off'}")

    def _toggle_record_grayscale(self, checked: bool) -> None:
        """Save whether the shown camera's recordings are grayscale, straight away."""
        try:
            change = set_recording_grayscale(self.config_path, bool(checked))
        except Exception as exc:
            self._show_record_grayscale(recording_is_grayscale(self.config))
            QMessageBox.critical(self, "Recording setting not changed", str(exc))
            return
        self._adopt_config(load_config(self.config_path))
        if change.left_profile:
            # Leaving a named camera profile is a change the user did not ask for by name.
            QMessageBox.information(self, "Camera profile changed", change.message)

    def _run_both_once(self) -> None:
        if len(self.cameras) < 2:
            return
        if not self._confirm_live_monitor_closed("start a recording"):
            return
        if not self._resolve_unsaved_settings("start a recording"):
            return
        self.run_bbx_command(
            [
                "run-once",
                "--all-cameras",
                "--mode",
                str(self.run_mode.currentData()),
                "--config",
                str(self.primary_config_path),
            ]
        )

    def _open_live_monitor(self) -> None:
        from .live_monitor import live_monitor_is_running

        if live_monitor_is_running():
            QMessageBox.information(
                self,
                "Live monitor is already open",
                "Use the existing BumbleBox Live Camera Monitor window.",
            )
            return
        self.run_bbx_command(["live-monitor"], detached=True)

    def _confirm_live_monitor_closed(self, action: str) -> bool:
        from .live_monitor import live_monitor_is_running

        if not live_monitor_is_running():
            return True
        QMessageBox.warning(
            self,
            "Close the live monitor first",
            f"Close the BumbleBox Live Camera Monitor before you {action}. "
            "The monitor currently owns the camera devices.",
        )
        return False

    def _automation_targets(self) -> list[CameraEntry]:
        """The cameras Start and Stop act on: both when the box is ticked, else the shown one."""
        if len(self.cameras) == 2 and self.both_cameras_checkbox.isChecked():
            return list(self.cameras)
        return [self._active_camera()]

    def _schedule_known_off(self, index: int) -> bool:
        status = self._camera_statuses.get(index)
        return status is not None and status.schedule.state == SCHEDULE_OFF

    def _start_automation(self, targets: list[CameraEntry] | None = None) -> None:
        if not self._confirm_live_monitor_closed("start automated recording"):
            return
        if not self._resolve_unsaved_settings("start automated recording"):
            return
        targets = list(targets) if targets else self._automation_targets()
        two = len(self.cameras) == 2
        review = StartReview()
        prepared: list[tuple[CameraEntry, dict[str, Any]]] = []
        for entry in targets:
            config = load_config(entry.path)
            scheduling = config.setdefault("scheduling", {})
            scheduling["enabled"] = True
            scheduling["backend"] = "systemd"
            scheduling["scope"] = "user"
            prepared.append((entry, config))
            one = review_schedule_start(config)
            prefix = f"{entry.label}: " if two else ""
            review.problems += [prefix + problem for problem in one.problems]
            review.suggestions += [text for text in one.suggestions if text not in review.suggestions]
        if two:
            # Judge the pair as it will run: each camera's own check knows
            # nothing about the other recording at the same moment.
            starting = {entry.index: config for entry, config in prepared}
            pair = [CameraEntry(entry.index, entry.path, starting.get(entry.index, entry.config)) for entry in self.cameras]
            review.problems += check_camera_set(pair)
            load = combined_memory(pair)
            if load is not None and load.level == "fail":
                review.problems.append(load.message)
        if not review.ok and not self._confirm_start_despite_problems(review):
            return
        commands: list[list[str]] = []
        for entry, config in prepared:
            save_config(entry.path, config)
            if entry.index == self.camera_index:
                self.config = config
            which = ["--config", str(entry.path)] if two else []
            commands += [["systemd-write", *which], ["systemd-install", *which]]
        both = len(prepared) == 2
        self.run_bbx_commands(
            commands,
            success_message=(
                "Automated recording started for both cameras" if both else "Automated recording started",
                "The clock-aligned recording schedule is active and will continue after logout.",
            ),
        )

    def _confirm_start_despite_problems(self, review: StartReview) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Check before starting")
        box.setText(
            "Automated recording has not been started. The current settings are unlikely to work:"
        )
        box.setInformativeText(
            format_start_review(review)
            + "\n\nThe memory and timing figures are estimates. Start anyway?"
        )
        start_anyway = box.addButton("Start Anyway", QMessageBox.DestructiveRole)
        cancel = box.addButton("Don't Start", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec_()
        return box.clickedButton() is start_anyway

    def _stop_automation(self) -> None:
        targets = self._automation_targets()
        if len(targets) == 2:
            # Leave out a camera whose schedule is already off; stopping it would only report an error.
            targets = [entry for entry in targets if not self._schedule_known_off(entry.index)] or [
                self._active_camera()
            ]
        if not self._confirm_stop_automation(targets):
            return
        two = len(self.cameras) == 2
        commands: list[list[str]] = []
        for entry in targets:
            config = load_config(entry.path)
            config.setdefault("scheduling", {})["enabled"] = False
            save_config(entry.path, config)
            if entry.index == self.camera_index:
                self.config = config
            commands.append(["systemd-disable", *(["--config", str(entry.path)] if two else [])])
        both = len(targets) == 2
        message = (
            "Automated recording stopped for both cameras" if both else "Automated recording stopped",
            "The BumbleBox recording timer is disabled.",
        )
        if len(commands) == 1:
            self.run_bbx_command(commands[0], success_message=message)
        else:
            self.run_bbx_commands(commands, success_message=message)

    def _confirm_stop_automation(self, targets: list[CameraEntry] | None = None) -> bool:
        targets = list(targets) if targets else [self._active_camera()]
        schedule = self._box_status.schedule if self._box_status is not None else None
        if len(targets) == 1 and targets[0].index != self.camera_index:
            other_status = self._camera_statuses.get(targets[0].index)
            schedule = other_status.schedule if other_status is not None else None
        if len(targets) == 1 and schedule is not None and schedule.state == SCHEDULE_OFF:
            # Nothing is scheduled, so stopping changes nothing worth asking about.
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        if len(targets) == 2:
            box.setWindowTitle("Stop automated recording on both cameras?")
            box.setText("No more recordings will start on either camera until you press Start Automated Recording.")
        elif len(self.cameras) == 2:
            box.setWindowTitle(f"Stop automated recording on {targets[0].label}?")
            box.setText(
                f"No more recordings will start on {targets[0].label} until you press Start Automated "
                "Recording. The other camera is not affected."
            )
        else:
            box.setWindowTitle("Stop automated recording?")
            box.setText("No more recordings will start until you press Start Automated Recording.")
        details = ["A recording that is already in progress finishes normally."]
        if schedule is not None and schedule.next_run is not None:
            now = datetime.now()
            details.append(
                f"The next one is due {describe_moment(schedule.next_run, now)} "
                f"({describe_relative(schedule.next_run, now)})."
            )
        box.setInformativeText(" ".join(details))
        stop = box.addButton("Stop Recording Schedule", QMessageBox.DestructiveRole)
        keep = box.addButton("Keep Recording", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec_()
        return box.clickedButton() is stop

    def run_bbx_command(
        self,
        arguments: list[str],
        *,
        detached: bool = False,
        success_message: tuple[str, str] | None = None,
        on_success: Any = None,
    ) -> None:
        if detached:
            detached_arguments = list(arguments)
            if "--config" not in detached_arguments:
                detached_arguments.extend(["--config", str(self.config_path)])
            QProcess.startDetached(
                sys.executable,
                [str(BBX_SCRIPT), *detached_arguments],
                str(REPO_ROOT),
            )
            return
        self.run_bbx_commands([arguments], success_message=success_message, on_success=on_success)

    def run_bbx_commands(
        self,
        commands: list[list[str]],
        *,
        success_message: tuple[str, str] | None = None,
        on_success: Any = None,
    ) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self,
                "Command already running",
                "Wait for the current BumbleBox command to finish, or press Cancel on the Run page.",
            )
            return
        self.command_queue = [list(command) for command in commands]
        self.command_success_message = success_message
        self._command_on_success = on_success
        self._cancel_requested = False
        self.console.clear()
        self.navigation.setCurrentRow(PAGE_INDEX_RUN)
        self._start_next_command()

    def _start_next_command(self) -> None:
        if not self.command_queue:
            return
        arguments = self.command_queue.pop(0)
        self._current_command = list(arguments)
        self._command_output = []
        self._set_status_label(
            self.command_status,
            LEVEL_OFF,
            f"{command_label(arguments)}: running since {datetime.now():%H:%M:%S}. Output appears below.",
        )
        if "--config" not in arguments:
            arguments.extend(["--config", str(self.config_path)])
        command_text = " ".join([str(BBX_SCRIPT), *arguments])
        self.console.append(f"$ {command_text}\n")
        self.cancel_button.setText("Cancel")
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.start(sys.executable, [str(BBX_SCRIPT), *arguments])

    def _cancel_command(self) -> None:
        if self.process.state() == QProcess.NotRunning or self._cancel_requested:
            return
        if self._current_command[:1] == ["run-once"] and not self._confirm_cancel_recording():
            return
        if self.process.state() == QProcess.NotRunning:
            return  # it finished while the question was open
        self._cancel_requested = True
        self._cancelled_steps = len(self.command_queue)
        self.command_queue.clear()
        self.command_success_message = None
        self._command_on_success = None
        self.cancel_button.setText("Cancelling...")
        self.cancel_button.setEnabled(False)
        self._set_status_label(
            self.command_status, LEVEL_OFF, f"{command_label(self._current_command)}: cancelling..."
        )
        self.console.append("\n[cancel requested]")
        self._cancel_stage = 0
        self._interrupt_process()
        self._cancel_timer.start(CANCEL_INTERRUPT_GRACE_MS)

    def _confirm_cancel_recording(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Cancel this recording?")
        box.setText("The run stops now and is listed as failed in Results.")
        box.setInformativeText(
            "Video already written to disk is kept in its session folder. "
            "Frames still held in memory are lost."
        )
        stop = box.addButton("Stop Recording", QMessageBox.DestructiveRole)
        keep = box.addButton("Keep Recording", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec_()
        return box.clickedButton() is stop

    def _interrupt_process(self) -> None:
        """Ask the running command to stop the way Ctrl+C does."""
        pid = int(self.process.processId() or 0)
        if pid > 0 and hasattr(signal, "SIGINT") and os.name == "posix":
            try:
                os.kill(pid, signal.SIGINT)
                return
            except OSError:
                pass
        self.process.terminate()

    def _escalate_cancel(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        if self._cancel_stage == 0:
            self._cancel_stage = 1
            self.console.append("\n[the command did not stop; terminating it]")
            self.process.terminate()
            self._cancel_timer.start(CANCEL_TERMINATE_GRACE_MS)
        else:
            self.console.append("\n[the command still did not stop; killing it]")
            self.process.kill()

    def _command_ended(self) -> None:
        self._cancel_timer.stop()
        self.cancel_button.setVisible(False)

    def _read_process_output(self) -> None:
        if not self.process.isOpen():
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._command_output.append(data)
            self.console.moveCursor(QTextCursor.End)
            self.console.insertPlainText(data)
            self.console.ensureCursorVisible()

    def _process_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._read_process_output()
        self._command_ended()
        self.console.append(f"\n[finished with exit code {exit_code}]")
        self.refresh_results()
        if self._cancel_requested:
            self._cancel_requested = False
            self.command_queue.clear()
            self.command_success_message = None
            label = command_label(self._current_command)
            if exit_code == 0 and status == QProcess.NormalExit:
                # It completed before the interrupt reached it.
                skipped = " The steps after it were not run." if self._cancelled_steps else ""
                self._set_status_label(
                    self.command_status, LEVEL_OK, f"{label} finished before it could be cancelled.{skipped}"
                )
                self._cancelled_steps = 0
                if self.config_path.exists():
                    self.config = load_config(self.config_path)
                    self.refresh_from_config()
                return
            self._cancelled_steps = 0
            self.refresh_box_status()
            note = (
                " It is listed as failed in Results; anything written before the cancel is in its session folder."
                if self._current_command[:1] == ["run-once"]
                else ""
            )
            self._set_status_label(
                self.command_status, LEVEL_WARN, f"{command_label(self._current_command)} cancelled.{note}"
            )
            return
        crashed = status == QProcess.CrashExit
        if exit_code != 0 or crashed:
            self.command_queue.clear()
            self.command_success_message = None
            self._command_on_success = None
            self.refresh_box_status()
            label = command_label(self._current_command)
            if crashed:
                explanation = FailureExplanation(
                    f"{label} stopped unexpectedly",
                    "The program crashed or was killed before it finished. "
                    "Anything it printed is in the command output.",
                )
            else:
                explanation = explain_failure(
                    "".join(self._command_output),
                    exit_code=exit_code,
                    command=self._current_command,
                )
            self._report_command_failure(explanation)
            return
        if self.command_queue:
            self._start_next_command()
            return
        success_title = self.command_success_message[0] if self.command_success_message else None
        self._set_status_label(
            self.command_status,
            LEVEL_OK,
            success_title or f"{command_label(self._current_command)} finished.",
        )
        on_success, self._command_on_success = self._command_on_success, None
        if on_success is not None:
            on_success()
        if self.config_path.exists():
            self.config = load_config(self.config_path)
            self.refresh_from_config()
        else:
            self.refresh_box_status()
        if self.command_success_message is not None:
            title, message = self.command_success_message
            self.command_success_message = None
            QMessageBox.information(self, title, message)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        # A crash also emits finished(); a failure to start does not, and
        # would otherwise leave the window showing nothing at all.
        if error != QProcess.FailedToStart:
            return
        self._command_ended()
        self._cancel_requested = False
        self.command_queue.clear()
        self.command_success_message = None
        self._report_command_failure(
            FailureExplanation(
                f"{command_label(self._current_command)} could not be started",
                f"BumbleBox could not launch its own command: {self.process.errorString()}",
            )
        )

    def _report_command_failure(self, explanation: FailureExplanation) -> None:
        self._set_status_label(self.command_status, LEVEL_FAIL, f"{explanation.title}\n{explanation.reason}")
        self._show_failure_dialog(explanation)

    def _show_failure_dialog(self, explanation: FailureExplanation) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(explanation.title)
        box.setText(f"{explanation.reason}\n\nThe full output is under Command output on the Run page.")
        # A missing drive is the one failure this window can fix on the spot.
        mount = None
        if explanation.title == DRIVE_NOT_MOUNTED:
            mount = box.addButton("Mount Data Drive...", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec_()
        if mount is not None and box.clickedButton() is mount:
            self._open_data_drive()


def _single_instance_lock() -> QLockFile:
    runtime_dir = QStandardPaths.writableLocation(QStandardPaths.RuntimeLocation)
    lock_root = Path(runtime_dir) if runtime_dir else Path(tempfile.gettempdir())
    lock_root.mkdir(parents=True, exist_ok=True)
    user_id = str(os.getuid()) if hasattr(os, "getuid") else str(Path.home())
    lock = QLockFile(str(lock_root / f"bumblebox-v2-gui-{user_id}.lock"))
    lock.setStaleLockTime(30_000)
    return lock


def _preferred_ui_font() -> str:
    available = set(QFontDatabase().families())
    for family in ("Avenir Next", "Nunito Sans", "Noto Sans", "DejaVu Sans"):
        if family in available:
            return family
    return QApplication.font().family()


def launch(*, config_path: str | Path = DEFAULT_USER_CONFIG_PATH) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BumbleBox")
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont(_preferred_ui_font(), 11))
    lock = _single_instance_lock()
    if not lock.tryLock(100):
        QMessageBox.information(
            None,
            "BumbleBox is already open",
            "Only one BumbleBox GUI can run at a time. Use the existing window.",
        )
        return 1
    app._bumblebox_instance_lock = lock
    window = BumbleBoxQtGUI(Path(config_path).expanduser().resolve())
    window.show()
    return int(app.exec_())
