from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .qt_env import sanitize_current_qt_env

sanitize_current_qt_env()

try:
    from PyQt5.QtCore import (
        QObject,
        QLockFile,
        QProcess,
        QRunnable,
        QStandardPaths,
        QThreadPool,
        Qt,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt5.QtGui import QDesktopServices, QFont, QFontDatabase, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
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
        QSpinBox,
        QStackedWidget,
        QAbstractItemView,
        QHeaderView,
        QInputDialog,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
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

from .camera_profiles import CAMERA_PROFILES, apply_camera_profile
from .config import DEFAULT_USER_CONFIG_PATH, load_config, load_defaults, save_config
from .hardware_profiles import HARDWARE_PROFILES, apply_hardware_profile
from .multimodal_calibration import (
    MANIFEST_NAME as CALIBRATION_MANIFEST_NAME,
    add_calibration_session,
    format_calibration_project_status,
    get_calibration_project_status,
    initialize_calibration_project,
)
from .status_history import list_recent_run_records, load_run_summary


REPO_ROOT = Path(__file__).resolve().parents[1]
BBX_SCRIPT = REPO_ROOT / "bbx.py"

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
QMainWindow, QWizard, QWidget#AppRoot { background: #f7f6f1; }
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
QFrame#Card QLabel[valueLabel="true"] {
    color: #273a36;
    font-size: 16px;
    font-weight: 600;
}
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
QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #ccd3ce;
    border-radius: 9px;
    padding: 8px 10px;
    selection-background-color: #b9d9cf;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 2px solid #4d8b7d;
    padding: 7px 9px;
}
QComboBox::drop-down { border: none; width: 24px; }
QCheckBox { spacing: 9px; color: #40524e; }
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


class _RunHistorySignals(QObject):
    finished = pyqtSignal(int, object, object)


class _RunHistoryWorker(QRunnable):
    def __init__(self, generation: int, data_root: str, limit: int = 60) -> None:
        super().__init__()
        self.generation = int(generation)
        self.data_root = str(data_root)
        self.limit = int(limit)
        self.signals = _RunHistorySignals()

    def _emit_finished(self, payloads: list[dict[str, Any]], error: str | None) -> None:
        try:
            self.signals.finished.emit(self.generation, payloads, error)
        except RuntimeError:
            # The application window may close while a removable-drive scan is finishing.
            pass

    def run(self) -> None:
        payloads: list[dict[str, Any]] = []
        try:
            for record in list_recent_run_records(self.data_root, limit=self.limit):
                try:
                    payload = load_run_summary(record.summary_path)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                payload = dict(payload)
                payload["_summary_path"] = str(record.summary_path.resolve())
                recorded_session_text = str(payload.get("session_dir") or "").strip()
                recorded_session_dir = Path(recorded_session_text).expanduser()
                payload["_session_dir_local"] = str(
                    recorded_session_dir.resolve()
                    if recorded_session_text and recorded_session_dir.exists()
                    else record.summary_path.parent.resolve()
                )
                payloads.append(payload)
        except Exception as exc:
            self._emit_finished([], str(exc))
            return
        self._emit_finished(payloads, None)


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
        self.recording_seconds = QSpinBox()
        self.recording_seconds.setRange(1, 86400)
        self.recording_seconds.setValue(int(config.get("capture", {}).get("recording_seconds", 20)))
        self.interval_minutes = QSpinBox()
        self.interval_minutes.setRange(1, 1440)
        self.interval_minutes.setValue(int(config.get("capture", {}).get("record_interval_minutes", 30)))
        form.addRow("Colony ID", self.colony)
        form.addRow("Data folder", path_row)
        form.addRow("Default run mode", self.mode)
        form.addRow("Recording duration (seconds)", self.recording_seconds)
        form.addRow("Recording interval (minutes)", self.interval_minutes)

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
        updated.setdefault("capture", {})["recording_seconds"] = self.experiment_page.recording_seconds.value()
        updated["capture"]["record_interval_minutes"] = self.experiment_page.interval_minutes.value()
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


class BumbleBoxQtGUI(QMainWindow):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.config_path = config_path
        self.config = _load_config_or_defaults(config_path)
        self.command_queue: list[list[str]] = []
        self.command_success_message: tuple[str, str] | None = None
        self._run_history_generation = 0
        self._run_history_loading = False
        self._run_history_refresh_pending = False
        self._run_history_payloads: list[dict[str, Any]] = []
        self._selected_run_payload: dict[str, Any] | None = None
        self._run_history_workers: dict[int, _RunHistoryWorker] = {}
        self.thread_pool = QThreadPool.globalInstance()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.setWindowTitle("BumbleBox")
        self.resize(1180, 780)
        self.setMinimumSize(940, 660)
        self._build_ui()
        self.refresh_from_config()
        QTimer.singleShot(100, self.refresh_results)
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
        for title in ("Overview", "Run", "Results", "Hardware", "Advanced"):
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
        self.profile_badge = QLabel()
        self.profile_badge.setObjectName("ProfileBadge")
        header.addWidget(self.profile_badge)
        content_layout.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_overview_page())
        self.pages.addWidget(self._build_run_page())
        self.pages.addWidget(self._build_results_page())
        self.pages.addWidget(self._build_hardware_page())
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
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        self.setup_status = QLabel()
        self.setup_status.setWordWrap(True)
        layout.addWidget(self.setup_status)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        self.camera_card = self._status_card("Primary camera", "", tone="peach")
        self.thermal_card = self._status_card("Thermal", "", tone="sun")
        self.realsense_card = self._status_card("RealSense", "", tone="sky")
        self.storage_card = self._status_card("Data storage", "", tone="mint")
        for column, card in enumerate((self.camera_card, self.thermal_card, self.realsense_card, self.storage_card)):
            cards.addWidget(card, 0, column)
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
        layout.addWidget(eyebrow)
        layout.addWidget(value_label)
        layout.addStretch(1)
        card.value_label = value_label
        return card

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Capture and automation",
            "Run one recording now or control the clock-aligned systemd recording schedule.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)
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
        self.live_monitor_button = QPushButton("Open Live Camera Monitor")
        self.live_monitor_button.setObjectName("Primary")
        self.live_monitor_button.setToolTip(
            "Preview enabled cameras together without saving data. Close the monitor before recording."
        )
        self.live_monitor_button.clicked.connect(self._open_live_monitor)
        row.addWidget(self.live_monitor_button)
        row.addStretch(1)
        layout.addLayout(row)

        automation = QGroupBox("Automated recording")
        automation.setProperty("tone", "mint")
        automation_layout = QVBoxLayout(automation)
        note = QLabel(
            "Start writes and installs user-scope timers. Recordings continue after logout when user lingering is enabled."
        )
        note.setWordWrap(True)
        automation_layout.addWidget(note)
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
        layout.addWidget(automation)

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
        layout.addWidget(self.distributed_run_group)

        latest = QGroupBox("Latest recording")
        latest.setProperty("tone", "sky")
        latest_layout = QVBoxLayout(latest)
        self.latest_run_status = QLabel("Loading recent recording status...")
        self.latest_run_status.setWordWrap(True)
        latest_layout.addWidget(self.latest_run_status)
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
        self.latest_depth_button.clicked.connect(
            lambda: self._open_run_artifact(
                self._latest_run_payload(), "realsense_depth_preview_video_path"
            )
        )
        self.latest_results_button = QPushButton("View All Results")
        self.latest_results_button.setObjectName("Primary")
        self.latest_results_button.clicked.connect(lambda: self.navigation.setCurrentRow(2))
        latest_buttons.addWidget(self.latest_session_button)
        latest_buttons.addWidget(self.latest_depth_button)
        latest_buttons.addWidget(self.latest_results_button)
        latest_buttons.addStretch(1)
        latest_layout.addLayout(latest_buttons)
        layout.addWidget(latest)
        layout.addWidget(self._build_console(), 1)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        heading, subtitle = _page_heading(
            "Recording results",
            "Inspect recent multimodal runs and open RGB, thermal, and RealSense outputs without searching through folders.",
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
            artifact_buttons.addWidget(button, index // 3, index % 3)
            self.result_artifact_buttons[key] = (button, payload_key)
        self.result_add_calibration_button = QPushButton("Add to Calibration Project")
        self.result_add_calibration_button.setObjectName("Primary")
        self.result_add_calibration_button.setToolTip(
            "Create or select a multimodal calibration project in Advanced, then register this run."
        )
        self.result_add_calibration_button.clicked.connect(self._add_selected_run_to_calibration)
        self.result_add_calibration_button.setEnabled(False)
        artifact_buttons.addWidget(self.result_add_calibration_button, 2, 2)
        self.result_sync_analysis_button = QPushButton("Analyze Shared Timing Cue")
        self.result_sync_analysis_button.setObjectName("Quiet")
        self.result_sync_analysis_button.setToolTip(
            "Estimate thermal and depth offset/jitter from a recording with a shared moving or occlusion cue."
        )
        self.result_sync_analysis_button.clicked.connect(self._analyze_selected_run_sync)
        self.result_sync_analysis_button.setEnabled(False)
        artifact_buttons.addWidget(self.result_sync_analysis_button, 3, 0)
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
        change = QPushButton("Change Hardware Profile")
        change.setObjectName("Quiet")
        change.clicked.connect(self.open_setup_wizard)
        layout.addWidget(change, 0, Qt.AlignLeft)
        layout.addStretch(1)
        return page

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
        self.console.setMinimumHeight(210)
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
        return "\n".join(
            (
                f"Session: {payload.get('session_name') or 'unknown'}",
                f"Started: {self._format_run_timestamp(payload.get('started_at'))}",
                f"Result: {'Success' if bool(payload.get('success', False)) else 'Failed'} | "
                f"Mode: {payload.get('mode') or 'unknown'}",
                f"RGB: {int(payload.get('frames_captured', 0) or 0)} frames at "
                f"{self._format_optional_fps(payload.get('actual_fps'))} fps",
                f"Thermal: {thermal_status}",
                f"RealSense: {depth_status}",
                f"Distributed capture: {distributed_status}",
                f"Warnings: {warning_count} | Errors: {error_count}",
            )
        )

    def _set_result_artifact_buttons(self, payload: dict[str, Any] | None) -> None:
        for button, payload_key in self.result_artifact_buttons.values():
            button.setEnabled(self._run_artifact_path(payload, payload_key) is not None)
        summary_available = self._run_artifact_path(payload, "_summary_path") is not None
        self.result_add_calibration_button.setEnabled(
            summary_available and self._configured_calibration_project() is not None
        )
        self.result_sync_analysis_button.setEnabled(summary_available)

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
        data_root = str(self.config.get("system", {}).get("data_root", "")).strip()
        worker = _RunHistoryWorker(generation, data_root, limit=60)
        worker.signals.finished.connect(self._run_history_loaded)
        self._run_history_workers[generation] = worker
        self.thread_pool.start(worker)

    def _run_history_loaded(
        self,
        generation: int,
        payloads: object,
        error: object,
    ) -> None:
        self._run_history_workers.pop(generation, None)
        if generation != self._run_history_generation:
            return
        self._run_history_loading = False
        self.results_refresh_button.setEnabled(True)
        self._selected_run_payload = None
        self.results_table.setRowCount(0)

        if error:
            self._run_history_payloads = []
            self.results_status.setText(f"Could not scan recording summaries: {error}")
            self.latest_run_status.setText("Recent recording status is unavailable.")
            self.latest_session_button.setEnabled(False)
            self.latest_depth_button.setEnabled(False)
            self._set_result_artifact_buttons(None)
            if self._run_history_refresh_pending:
                self._run_history_refresh_pending = False
                QTimer.singleShot(0, self.refresh_results)
            return

        normalized = [dict(item) for item in payloads if isinstance(item, dict)] if isinstance(payloads, list) else []
        self._run_history_payloads = normalized
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
            self.latest_run_status.setText("No BumbleBox recording summaries were found.")
            self.latest_session_button.setEnabled(False)
            self.latest_depth_button.setEnabled(False)
            self.result_detail.setText("No recording results are available.")
            self._set_result_artifact_buttons(None)
            if self._run_history_refresh_pending:
                self._run_history_refresh_pending = False
                QTimer.singleShot(0, self.refresh_results)
            return

        self.latest_run_status.setText(self._format_run_detail(latest))
        self.latest_session_button.setEnabled(
            self._run_artifact_path(latest, "_session_dir_local") is not None
        )
        self.latest_depth_button.setEnabled(
            self._run_artifact_path(latest, "realsense_depth_preview_video_path") is not None
        )
        self.results_table.selectRow(0)
        if self._run_history_refresh_pending:
            self._run_history_refresh_pending = False
            QTimer.singleShot(0, self.refresh_results)

    def _navigate(self, index: int) -> None:
        if index < 0:
            return
        self.pages.setCurrentIndex(index)
        item = self.navigation.item(index)
        self.page_title.setText(item.text().upper() if item else "BUMBLEBOX")
        if index == 2 and not self._run_history_payloads:
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
        self.storage_card.value_label.setText(str(self.config.get("system", {}).get("data_root", "Not set")))
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

    def _run_once(self) -> None:
        if not self._confirm_live_monitor_closed("start a recording"):
            return
        self.run_bbx_command(["run-once", "--mode", str(self.run_mode.currentData())])

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

    def _start_automation(self) -> None:
        if not self._confirm_live_monitor_closed("start automated recording"):
            return
        config = load_config(self.config_path)
        scheduling = config.setdefault("scheduling", {})
        scheduling["enabled"] = True
        scheduling["backend"] = "systemd"
        scheduling["scope"] = "user"
        save_config(self.config_path, config)
        self.config = config
        self.run_bbx_commands(
            [["systemd-write"], ["systemd-install"]],
            success_message=(
                "Automated recording started",
                "The clock-aligned recording schedule is active and will continue after logout.",
            ),
        )

    def _stop_automation(self) -> None:
        config = load_config(self.config_path)
        config.setdefault("scheduling", {})["enabled"] = False
        save_config(self.config_path, config)
        self.config = config
        self.run_bbx_command(
            ["systemd-disable"],
            success_message=("Automated recording stopped", "The BumbleBox recording timer is disabled."),
        )

    def run_bbx_command(
        self,
        arguments: list[str],
        *,
        detached: bool = False,
        success_message: tuple[str, str] | None = None,
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
        self.run_bbx_commands([arguments], success_message=success_message)

    def run_bbx_commands(
        self,
        commands: list[list[str]],
        *,
        success_message: tuple[str, str] | None = None,
    ) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Command already running", "Wait for the current BumbleBox command to finish.")
            return
        self.command_queue = [list(command) for command in commands]
        self.command_success_message = success_message
        self.console.clear()
        self.navigation.setCurrentRow(1)
        self._start_next_command()

    def _start_next_command(self) -> None:
        if not self.command_queue:
            return
        arguments = self.command_queue.pop(0)
        if "--config" not in arguments:
            arguments.extend(["--config", str(self.config_path)])
        command_text = " ".join([str(BBX_SCRIPT), *arguments])
        self.console.append(f"$ {command_text}\n")
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.start(sys.executable, [str(BBX_SCRIPT), *arguments])

    def _read_process_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.console.moveCursor(QTextCursor.End)
            self.console.insertPlainText(data)
            self.console.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.console.append(f"\n[finished with exit code {exit_code}]")
        self.refresh_results()
        if exit_code != 0:
            self.command_queue.clear()
            self.command_success_message = None
            return
        if self.command_queue:
            self._start_next_command()
            return
        if self.config_path.exists():
            self.config = load_config(self.config_path)
            self.refresh_from_config()
        if self.command_success_message is not None:
            title, message = self.command_success_message
            self.command_success_message = None
            QMessageBox.information(self, title, message)


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
