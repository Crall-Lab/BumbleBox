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
    from PyQt5.QtCore import QLockFile, QProcess, QStandardPaths, Qt, QTimer, QUrl
    from PyQt5.QtGui import QDesktopServices, QFont, QTextCursor
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


REPO_ROOT = Path(__file__).resolve().parents[1]
BBX_SCRIPT = REPO_ROOT / "bbx.py"

PAGE_WELCOME = 0
PAGE_HARDWARE = 1
PAGE_RGB = 2
PAGE_THERMAL = 3
PAGE_REALSENSE = 4
PAGE_EXPERIMENT = 5
PAGE_SUMMARY = 6


APP_STYLE = """
QWidget {
    background: #f4f0e6;
    color: #172326;
    font-family: "Avenir Next", "DejaVu Sans", sans-serif;
    font-size: 14px;
}
QMainWindow, QWizard { background: #f4f0e6; }
QLabel { background: transparent; }
QLabel#Title { font-size: 31px; font-weight: 700; color: #172326; }
QLabel#Subtitle { font-size: 15px; color: #526064; }
QLabel#SectionTitle { font-size: 23px; font-weight: 650; color: #172326; }
QLabel#Eyebrow { color: #92721f; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#StatusGood { color: #25664a; font-weight: 700; }
QLabel#StatusWarn { color: #9b5f16; font-weight: 700; }
QFrame#Sidebar { background: #172326; border: none; }
QFrame#Card, QGroupBox {
    background: #fffdf7;
    border: 1px solid #d7d0c0;
    border-radius: 12px;
}
QGroupBox {
    margin-top: 14px;
    padding: 18px 14px 12px 14px;
    font-weight: 700;
}
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }
QListWidget {
    background: #172326;
    color: #dce4df;
    border: none;
    outline: none;
    padding: 16px 10px;
}
QListWidget::item { padding: 13px 14px; margin: 3px 0; border-radius: 8px; }
QListWidget::item:selected { background: #d8a928; color: #172326; font-weight: 700; }
QPushButton {
    background: #172326;
    color: #ffffff;
    border: none;
    border-radius: 7px;
    padding: 9px 14px;
    font-weight: 650;
}
QPushButton:hover { background: #294044; }
QPushButton:disabled { background: #a8aca8; color: #ececec; }
QPushButton#Accent { background: #d8a928; color: #172326; }
QPushButton#Accent:hover { background: #e4bc4e; }
QPushButton#Quiet { background: #e6e0d3; color: #263438; }
QLineEdit, QComboBox, QSpinBox, QTextEdit {
    background: #ffffff;
    border: 1px solid #bbb5a8;
    border-radius: 6px;
    padding: 7px;
    selection-background-color: #d8a928;
}
QTextEdit { background: #132126; color: #dcebe8; font-family: "SFMono-Regular", "DejaVu Sans Mono", monospace; }
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


class WelcomePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Welcome to BumbleBox")
        self.setSubTitle(
            "This wizard records the hardware you are actually using and shows only the setup steps that apply."
        )
        layout = QVBoxLayout(self)
        note = QLabel(
            "You can rerun this wizard at any time. It does not erase tracking parameters or previous data."
        )
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
        form = QFormLayout()
        self.profile = QComboBox()
        for key, profile in HARDWARE_PROFILES.items():
            self.profile.addItem(profile.label, key)
        _set_combo_data(self.profile, str(config.get("setup", {}).get("hardware_profile", "custom")))
        form.addRow("Hardware profile", self.profile)
        layout.addLayout(form)

        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.thermal = QCheckBox("Use a PureThermal / Lepton camera")
        self.thermal.setChecked(bool(config.get("thermal", {}).get("enabled", False)))
        self.realsense = QCheckBox("Use a RealSense depth camera")
        self.realsense.setChecked(bool(config.get("realsense", {}).get("enabled", False)))
        layout.addWidget(self.thermal)
        layout.addWidget(self.realsense)
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
        form = QFormLayout()
        self.profile = QComboBox()
        for key, profile in CAMERA_PROFILES.items():
            self.profile.addItem(profile.label, key)
        _set_combo_data(self.profile, str(config.get("camera", {}).get("profile", "custom")))
        form.addRow("Camera profile", self.profile)
        layout.addLayout(form)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        visible_note = QLabel(
            "OwlSight uses visible illumination because the stock OV64A40 camera is treated as IR-cut. "
            "The HQ NoIR reference profile keeps the established infrared workflow."
        )
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
        return PAGE_EXPERIMENT


class ThermalPage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure thermal capture")
        self.setSubTitle("Auto discovery is recommended; a successful check can later pin the stable V4L by-id path.")
        form = QFormLayout(self)
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
        return PAGE_REALSENSE if self.wizard().uses_realsense() else PAGE_EXPERIMENT


class RealSensePage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Configure RealSense depth")
        self.setSubTitle(
            "These are initial D405 test settings. The hardware check will confirm which exact profiles the connected camera accepts."
        )
        form = QFormLayout(self)
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
        return PAGE_EXPERIMENT


class ExperimentPage(QWizardPage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.setTitle("Set experiment basics")
        self.setSubTitle("These are the settings needed for normal recording; specialized controls remain outside the wizard.")
        form = QFormLayout(self)
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
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        layout.addWidget(self.summary)
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
        self.summary.setText(
            f"<b>Hardware profile:</b> {hardware}<br>"
            f"<b>Primary camera:</b> {camera}<br>"
            f"<b>Optional devices:</b> {', '.join(optional) if optional else 'None'}<br>"
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
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.resize(720, 560)

        self.welcome_page = WelcomePage()
        self.hardware_page = HardwarePage(config)
        self.rgb_page = RgbPage(config)
        self.thermal_page = ThermalPage(config)
        self.realsense_page = RealSensePage(config)
        self.experiment_page = ExperimentPage(config)
        self.summary_page = SummaryPage()
        self.setPage(PAGE_WELCOME, self.welcome_page)
        self.setPage(PAGE_HARDWARE, self.hardware_page)
        self.setPage(PAGE_RGB, self.rgb_page)
        self.setPage(PAGE_THERMAL, self.thermal_page)
        self.setPage(PAGE_REALSENSE, self.realsense_page)
        self.setPage(PAGE_EXPERIMENT, self.experiment_page)
        self.setPage(PAGE_SUMMARY, self.summary_page)
        self.setStartId(PAGE_WELCOME)
        self.button(QWizard.FinishButton).setText("Save setup")

    def uses_thermal(self) -> bool:
        return self.hardware_page.thermal.isChecked()

    def uses_realsense(self) -> bool:
        return self.hardware_page.realsense.isChecked()

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
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.setWindowTitle("BumbleBox")
        self.resize(1180, 780)
        self.setMinimumSize(940, 660)
        self._build_ui()
        self.refresh_from_config()
        if not bool(self.config.get("setup", {}).get("completed", False)):
            QTimer.singleShot(0, self.open_setup_wizard)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 28, 18, 18)
        brand = QLabel("BUMBLEBOX")
        brand.setStyleSheet("color:#d8a928;font-size:20px;font-weight:800;letter-spacing:2px;")
        side_layout.addWidget(brand)
        sub = QLabel("COLONY IMAGING")
        sub.setStyleSheet("color:#8fa3a1;font-size:10px;font-weight:700;letter-spacing:1px;")
        side_layout.addWidget(sub)
        self.navigation = QListWidget()
        for title in ("Overview", "Run", "Hardware", "Advanced"):
            self.navigation.addItem(QListWidgetItem(title))
        self.navigation.currentRowChanged.connect(self._navigate)
        side_layout.addWidget(self.navigation, 1)
        version = QLabel("Qt migration · phase 1")
        version.setStyleSheet("color:#8fa3a1;font-size:11px;")
        side_layout.addWidget(version)
        layout.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(34, 26, 34, 26)
        header = QHBoxLayout()
        self.page_title = QLabel("Overview")
        self.page_title.setObjectName("Title")
        header.addWidget(self.page_title)
        header.addStretch(1)
        self.profile_badge = QLabel()
        self.profile_badge.setStyleSheet(
            "background:#eadcae;color:#57420d;border-radius:12px;padding:6px 12px;font-weight:700;"
        )
        header.addWidget(self.profile_badge)
        content_layout.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_overview_page())
        self.pages.addWidget(self._build_run_page())
        self.pages.addWidget(self._build_hardware_page())
        self.pages.addWidget(self._build_advanced_page())
        content_layout.addWidget(self.pages, 1)
        layout.addWidget(content, 1)
        self.navigation.setCurrentRow(0)

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
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
        self.camera_card = self._status_card("Primary camera", "")
        self.thermal_card = self._status_card("Thermal", "")
        self.realsense_card = self._status_card("RealSense", "")
        self.storage_card = self._status_card("Data storage", "")
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
    def _status_card(title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumHeight(120)
        layout = QVBoxLayout(card)
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
        layout.setContentsMargins(0, 10, 0, 0)
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
        row.addStretch(1)
        layout.addLayout(row)

        automation = QGroupBox("Automated recording")
        automation_layout = QVBoxLayout(automation)
        note = QLabel(
            "Start writes and installs user-scope timers. Recordings continue after logout when user lingering is enabled."
        )
        note.setWordWrap(True)
        automation_layout.addWidget(note)
        buttons = QHBoxLayout()
        start = QPushButton("Start Automated Recording")
        start.clicked.connect(self._start_automation)
        stop = QPushButton("Stop Automated Recording")
        stop.setObjectName("Quiet")
        stop.clicked.connect(self._stop_automation)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        buttons.addStretch(1)
        automation_layout.addLayout(buttons)
        layout.addWidget(automation)
        layout.addWidget(self._build_console(), 1)
        return page

    def _build_hardware_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        heading, subtitle = _page_heading(
            "Hardware checks",
            "Only devices selected in the setup profile are shown here.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        self.rgb_group = QGroupBox("Primary camera")
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
        depth_buttons = QHBoxLayout(self.realsense_group)
        depth_check = QPushButton("Check and Pin RealSense")
        depth_check.clicked.connect(lambda: self.run_bbx_command(["realsense-check", "--apply"]))
        depth_snapshot = QPushButton("Capture Depth Snapshot")
        depth_snapshot.clicked.connect(lambda: self.run_bbx_command(["realsense-snapshot"]))
        depth_buttons.addWidget(depth_check)
        depth_buttons.addWidget(depth_snapshot)
        depth_buttons.addStretch(1)
        layout.addWidget(self.realsense_group)
        change = QPushButton("Change Hardware Profile")
        change.setObjectName("Quiet")
        change.clicked.connect(self.open_setup_wizard)
        layout.addWidget(change, 0, Qt.AlignLeft)
        layout.addStretch(1)
        return page

    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        heading, subtitle = _page_heading(
            "Advanced tools",
            "Specialized tracking, calibration, fleet, and storage editors remain available while their Qt pages are migrated.",
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        legacy = QPushButton("Open Legacy Advanced Tools")
        legacy.clicked.connect(lambda: self.run_bbx_command(["gui", "--legacy"], detached=True))
        config = QPushButton("Open Config File")
        config.setObjectName("Quiet")
        config.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config_path))))
        docs = QPushButton("Open Project Documentation")
        docs.setObjectName("Quiet")
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(REPO_ROOT / "docs"))))
        layout.addWidget(legacy, 0, Qt.AlignLeft)
        layout.addWidget(config, 0, Qt.AlignLeft)
        layout.addWidget(docs, 0, Qt.AlignLeft)
        migration = QLabel(
            "Migration rule: working capture and analysis services are reused; only the operator interface is being replaced. "
            "This keeps the Qt transition testable and prevents a GUI rewrite from changing scientific outputs."
        )
        migration.setWordWrap(True)
        layout.addWidget(migration)
        layout.addStretch(1)
        return page

    def _build_console(self) -> QGroupBox:
        box = QGroupBox("Command output")
        layout = QVBoxLayout(box)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(210)
        layout.addWidget(self.console)
        return box

    def _navigate(self, index: int) -> None:
        if index < 0:
            return
        self.pages.setCurrentIndex(index)
        item = self.navigation.item(index)
        self.page_title.setText(item.text() if item else "BumbleBox")

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
        self.camera_card.value_label.setText(CAMERA_PROFILES.get(camera_profile, CAMERA_PROFILES["custom"]).label)
        thermal_enabled = bool(self.config.get("thermal", {}).get("enabled", False))
        depth_enabled = bool(self.config.get("realsense", {}).get("enabled", False))
        self.thermal_card.value_label.setText("Enabled" if thermal_enabled else "Not used")
        self.realsense_card.value_label.setText("Enabled" if depth_enabled else "Not used")
        self.storage_card.value_label.setText(str(self.config.get("system", {}).get("data_root", "Not set")))
        self.thermal_group.setVisible(thermal_enabled)
        self.realsense_group.setVisible(depth_enabled)
        _set_combo_data(self.run_mode, str(self.config.get("pipeline", {}).get("mode", "record_and_track")))

    def _run_once(self) -> None:
        self.run_bbx_command(["run-once", "--mode", str(self.run_mode.currentData())])

    def _start_automation(self) -> None:
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
            QProcess.startDetached(sys.executable, [str(BBX_SCRIPT), *arguments])
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


def launch(*, config_path: str | Path = DEFAULT_USER_CONFIG_PATH) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BumbleBox")
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont("Avenir Next", 11))
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
