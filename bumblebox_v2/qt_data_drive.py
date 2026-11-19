"""The Data Drive window: mount a drive at the data folder, and keep it after a restart.

Everything it decides comes from ``data_drive`` (no Qt there, so it is tested
without a display). The window itself changes nothing on the system: it runs
``bbx storage ...`` as root and then looks again at what is mounted.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import QProcess, QTimer, Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_drive import (
    ACTION_FORGET,
    ACTION_KEEP,
    ACTION_MOUNT_KEEP,
    ACTION_MOUNT_ONCE,
    ACTION_UNMOUNT,
    HIDDEN_ITEMS_LIMIT,
    LEVEL_FAIL,
    LEVEL_OK,
    LEVEL_WARN,
    REPO_ROOT,
    DataDriveState,
    DriveChoice,
    Elevation,
    explain_elevation_failure,
    find_elevation,
    items_hidden_by_mount,
    mount_folder_problem,
    own_disk_folder,
    read_data_drive_state,
    storage_command,
    summarize,
    terminal_command,
)

_CARD_TONES = {LEVEL_OK: "mint", LEVEL_WARN: "warn", LEVEL_FAIL: "fail"}
_STATUS_NAMES = {LEVEL_OK: "StatusGood", LEVEL_WARN: "StatusWarn", LEVEL_FAIL: "StatusFail"}
ACTION_TIMEOUT_MS = 120_000
DRIVE_POLL_MS = 3000

# Stands in for an exit code when the command could not be started at all.
PROCESS_DID_NOT_START = -1000

_WORKING_TEXT = {
    ACTION_MOUNT_KEEP: "Mounting the drive...",
    ACTION_MOUNT_ONCE: "Mounting the drive...",
    ACTION_KEEP: "Setting the drive to be mounted at startup...",
    ACTION_FORGET: "Changing what is mounted at startup...",
    ACTION_UNMOUNT: "Unmounting the drive...",
}


def _repolish(*widgets: QWidget) -> None:
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _eyebrow(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("Eyebrow")
    return label


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


def _last_meaningful_line(output: str) -> str:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if any(word in line.lower() for word in ("failed", "error", "elevated privileges")):
            return line
    return lines[-1] if lines else ""


class DataDriveDialog(QDialog):
    """Mount a drive at the data folder, choose whether it returns after a restart, or remove it."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        config_path: Path,
        config: Dict[str, Any],
        on_choice: Callable[[Optional[str], bool], Dict[str, Any]],
        camera_label: str = "",
        busy_reason: Optional[Callable[[], Optional[str]]] = None,
        read_state: Callable[[Dict[str, Any]], DataDriveState] = read_data_drive_state,
        elevation_finder: Callable[[], Elevation] = find_elevation,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Data drive ({camera_label})" if camera_label else "Data drive")
        self.setMinimumWidth(640)
        self.config_path = Path(config_path)
        self._config = copy.deepcopy(config)
        self._on_choice = on_choice
        self._busy_reason = busy_reason
        self._read_state = read_state
        self._elevation_finder = elevation_finder
        self._elevation: Optional[Elevation] = None
        self._state: Optional[DataDriveState] = None
        self._choices: List[DriveChoice] = []
        self._pending: Optional[Dict[str, Any]] = None
        self._output: List[str] = []
        self._working = False
        self._folder_problem: Optional[str] = None
        # Folders a drive must never be mounted over.
        self._protected = (REPO_ROOT, self.config_path.parent)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._timed_out)
        self._poll = QTimer(self)
        self._poll.setInterval(DRIVE_POLL_MS)
        self._poll.timeout.connect(self._poll_drives)

        self._build()
        self.refresh()
        self._poll.start()

    # -- Layout ----------------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        self.summary_card = QFrame()
        self.summary_card.setObjectName("Card")
        self.summary_card.setProperty("tone", "neutral")
        card_layout = QVBoxLayout(self.summary_card)
        card_layout.setContentsMargins(17, 14, 17, 14)
        card_layout.setSpacing(6)
        self.summary_headline = QLabel("")
        self.summary_headline.setProperty("valueLabel", True)
        self.summary_headline.setWordWrap(True)
        self.summary_detail = QLabel("")
        self.summary_detail.setProperty("detailLabel", True)
        self.summary_detail.setWordWrap(True)
        self.summary_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(self.summary_headline)
        card_layout.addWidget(self.summary_detail)
        layout.addWidget(self.summary_card)

        # Two alternative bodies; only one is visible, so the window is as tall as that one needs.
        self.no_drive_page = self._build_no_drive_page()
        self.mounted_page = self._build_mounted_page()
        layout.addWidget(self.no_drive_page)
        layout.addWidget(self.mounted_page)

        self.result_label = QLabel("")
        self.result_label.setObjectName("StatusNeutral")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_label.setVisible(False)
        layout.addWidget(self.result_label)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(110)
        self.details.setVisible(False)
        layout.addWidget(self.details)

        buttons = QHBoxLayout()
        self.own_disk_button = QPushButton("Record to the System Disk Instead...")
        self.own_disk_button.setObjectName("Quiet")
        self.own_disk_button.setToolTip(
            "For testing without a drive: recordings are saved on the Pi's own SD card or SSD."
        )
        self.own_disk_button.clicked.connect(self._use_own_disk)
        self.details_button = QPushButton("Show Details")
        self.details_button.setObjectName("Quiet")
        self.details_button.setCheckable(True)
        self.details_button.setVisible(False)
        self.details_button.toggled.connect(self._toggle_details)
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("Quiet")
        self.close_button.clicked.connect(self.close)
        self.mount_button = QPushButton("Mount Drive")
        self.mount_button.setObjectName("Primary")
        self.mount_button.setDefault(True)
        self.mount_button.clicked.connect(self._mount_selected)
        buttons.addWidget(self.own_disk_button)
        buttons.addWidget(self.details_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.mount_button)
        layout.addLayout(buttons)

    def _build_no_drive_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        heading = QHBoxLayout()
        heading.addWidget(_eyebrow("1 · Drive"), 1)
        self.look_again_button = QPushButton("Look Again")
        self.look_again_button.setObjectName("Quiet")
        self.look_again_button.setToolTip("Connected drives are also checked every few seconds.")
        self.look_again_button.clicked.connect(self.refresh)
        heading.addWidget(self.look_again_button)
        layout.addLayout(heading)

        self.drive_list = QWidget()
        self.drive_list_layout = QVBoxLayout(self.drive_list)
        self.drive_list_layout.setContentsMargins(0, 0, 0, 0)
        self.drive_list_layout.setSpacing(6)
        self.drive_group = QButtonGroup(self)
        self.drive_group.buttonToggled.connect(lambda *_: self._selection_changed())
        layout.addWidget(self.drive_list)
        self.drive_note = _hint()
        layout.addWidget(self.drive_note)

        layout.addSpacing(6)
        layout.addWidget(_eyebrow("2 · Folder"))
        self.folder_edit = QLineEdit()
        self.folder_edit.textChanged.connect(lambda *_: self._selection_changed())
        layout.addWidget(self.folder_edit)
        layout.addWidget(
            _hint("The drive appears as this folder, and recordings are saved in it. /mnt/bumblebox/data is the usual choice.")
        )
        self.folder_note = QLabel("")
        self.folder_note.setObjectName("StatusWarn")
        self.folder_note.setWordWrap(True)
        self.folder_note.setVisible(False)
        layout.addWidget(self.folder_note)

        layout.addSpacing(6)
        layout.addWidget(_eyebrow("3 · After a restart"))
        self.keep_radio = QRadioButton("Mount this drive here again automatically (recommended)")
        self.once_radio = QRadioButton("Do not mount it again")
        self.keep_radio.setChecked(True)
        self.restart_group = QButtonGroup(self)
        self.restart_group.addButton(self.keep_radio)
        self.restart_group.addButton(self.once_radio)
        self.restart_group.buttonToggled.connect(lambda *_: self._selection_changed())
        layout.addWidget(self.keep_radio)
        layout.addWidget(self.once_radio)
        self.restart_note = _hint()
        layout.addWidget(self.restart_note)
        return page

    def _build_mounted_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(_eyebrow("After a restart"))
        startup_row = QHBoxLayout()
        self.startup_label = QLabel("")
        self.startup_label.setWordWrap(True)
        self.startup_button = QPushButton("")
        self.startup_button.clicked.connect(self._toggle_startup)
        startup_row.addWidget(self.startup_label, 1)
        startup_row.addWidget(self.startup_button, 0, Qt.AlignTop)
        layout.addLayout(startup_row)

        layout.addSpacing(8)
        layout.addWidget(_eyebrow("Removing the drive"))
        remove_row = QHBoxLayout()
        remove_label = QLabel(
            "Unmount the drive before unplugging it, so that no recording is damaged. "
            "Recording is refused until a drive is mounted here again."
        )
        remove_label.setWordWrap(True)
        self.remove_button = QPushButton("Safely Remove Drive")
        self.remove_button.setObjectName("Quiet")
        self.remove_button.clicked.connect(self._remove_drive)
        remove_row.addWidget(remove_label, 1)
        remove_row.addWidget(self.remove_button, 0, Qt.AlignTop)
        layout.addLayout(remove_row)
        return page

    # -- Showing what is there ---------------------------------------------------

    def refresh(self) -> None:
        """Look at the drives and the data folder again."""
        self._show_state(self._read_state(self._config))

    @staticmethod
    def _signature(state: DataDriveState) -> tuple:
        return (
            state.mounted,
            state.mount_folder,
            state.writable,
            state.startup_entry,
            tuple((choice.path, choice.device.mountpoint, choice.device.uuid) for choice in state.choices),
        )

    def _poll_drives(self) -> None:
        """Notice a drive that was plugged in or pulled out while the window is open."""
        if self._working or not self.isVisible():
            return
        state = self._read_state(self._config)
        if self._state is None or self._signature(state) != self._signature(self._state):
            self._show_state(state)

    def _show_state(self, state: DataDriveState) -> None:
        self._state = state
        summary = summarize(state)
        self.summary_headline.setText(summary.headline)
        self.summary_detail.setText(summary.detail)
        self.summary_detail.setVisible(bool(summary.detail))
        tone = _CARD_TONES.get(summary.level, "neutral")
        if self.summary_card.property("tone") != tone:
            self.summary_card.setProperty("tone", tone)
            _repolish(self.summary_card, self.summary_headline, self.summary_detail)

        self.no_drive_page.setVisible(not state.mounted)
        self.mounted_page.setVisible(state.mounted)
        if state.mounted:
            self._show_mounted(state)
        else:
            self._show_no_drive(state)
        self.own_disk_button.setVisible(not state.mounted)
        self.mount_button.setVisible(not state.mounted)
        self._update_enabled()
        self._fit()

    def _fit(self) -> None:
        """Make the window as tall as its text needs at its current width.

        Wrapped labels need more height the narrower they are, which the
        ordinary size hint does not account for; without this they are clipped.
        """
        layout = self.layout()
        layout.activate()
        width = max(self.minimumWidth(), self.width())
        height = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.sizeHint().height()
        self.setMinimumHeight(0)
        self.resize(width, max(height, layout.minimumSize().height()))

    def _show_mounted(self, state: DataDriveState) -> None:
        if state.kept_at_startup and state.startup_is_this_drive:
            self.startup_label.setText("This drive is mounted here again automatically after a restart.")
            self.startup_button.setText("Stop Mounting at Startup")
            self.startup_button.setObjectName("Quiet")
            self.startup_button.setProperty("action", ACTION_FORGET)
        elif state.kept_at_startup:
            self.startup_label.setText(
                "Startup is set for a different drive at this folder. After a restart, this drive "
                "would not be mounted here."
            )
            self.startup_button.setText("Use This Drive at Startup")
            self.startup_button.setObjectName("Accent")
            self.startup_button.setProperty("action", ACTION_KEEP)
        else:
            self.startup_label.setText(
                "This drive is not mounted again after a restart. Recording will be refused until "
                "it is mounted here again."
            )
            self.startup_button.setText("Mount at Every Startup")
            self.startup_button.setObjectName("Accent")
            self.startup_button.setProperty("action", ACTION_KEEP)
        _repolish(self.startup_button)

    def _selected_choice(self) -> Optional[DriveChoice]:
        button = self.drive_group.checkedButton()
        if button is None:
            return None
        path = button.property("devicePath")
        for choice in self._choices:
            if choice.path == path:
                return choice
        return None

    def _show_no_drive(self, state: DataDriveState) -> None:
        selected = self._selected_choice()
        wanted = selected.path if selected is not None else None
        if wanted is None and state.preferred_choice is not None:
            wanted = state.preferred_choice.path

        for button in list(self.drive_group.buttons()):
            self.drive_group.removeButton(button)
            button.setParent(None)
            button.deleteLater()
        self._choices = list(state.choices)
        for choice in self._choices:
            button = QRadioButton(f"{choice.title}\n{choice.detail}")
            button.setProperty("devicePath", choice.path)
            self.drive_group.addButton(button)
            self.drive_list_layout.addWidget(button)
            if choice.path == wanted:
                button.setChecked(True)

        if not self.folder_edit.text().strip():
            self.folder_edit.setText(state.data_folder)
        self._selection_changed()

    def _folder(self) -> str:
        return self.folder_edit.text().strip()

    def _selection_changed(self) -> None:
        """Re-check the folder and the notes after a drive, folder or restart choice changed."""
        folder = self._folder()
        problem = mount_folder_problem(folder, protected=self._protected)
        hidden = 0 if problem else items_hidden_by_mount(folder)
        if problem:
            self.folder_note.setText(problem)
        elif hidden:
            count = f"more than {HIDDEN_ITEMS_LIMIT:,}" if hidden >= HIDDEN_ITEMS_LIMIT else f"{hidden:,}"
            self.folder_note.setText(
                f"{folder} already holds {count} item(s) on the system disk. They are not deleted, "
                "but they cannot be seen while a drive is mounted here."
            )
        self.folder_note.setVisible(bool(problem or hidden))
        self._folder_problem = problem

        if self.once_radio.isChecked():
            self.restart_note.setText(
                "After a restart no drive is mounted here, and recording is refused until one is "
                "mounted again from this window."
            )
        else:
            self.restart_note.setText(
                "If the drive is missing at startup, the Pi still starts and recording is refused."
            )
        self.drive_note.setText(self._drive_note())
        self.drive_note.setVisible(bool(self.drive_note.text()))
        self._update_enabled()
        self._fit()

    def _drive_note(self) -> str:
        """The line under the list of drives: what to do, or what will happen to the chosen one."""
        state = self._state
        if not self._choices:
            return (
                "No usable drive is connected. Plug in the drive; it appears here within a few "
                "seconds. A new drive must be formatted first (exFAT or ext4)."
            )
        choice = self._selected_choice()
        if choice is None:
            if state is not None and state.startup_uuid and not state.startup_drive_connected:
                return (
                    "Startup is set for a drive that is not connected now. Connect that drive, or "
                    "choose another one to use here instead."
                )
            return "Choose the drive that should hold the recordings."
        notes: List[str] = []
        if state is not None and state.startup_uuid:
            notes.append(
                "This is the drive startup is set for."
                if choice.device.uuid == state.startup_uuid
                else "Startup is set for a different drive; this one replaces it."
            )
        mounted_at = choice.device.mountpoint.strip()
        if mounted_at:
            notes.append(
                f"It is now at {mounted_at} and will be moved to the folder below. "
                "Close any file-manager windows showing it first."
            )
        return " ".join(notes)

    def _update_enabled(self) -> None:
        idle = not self._working
        mounted = bool(self._state and self._state.mounted)
        can_mount = (
            idle
            and not mounted
            and self._selected_choice() is not None
            and not self._folder_problem
        )
        self.mount_button.setEnabled(can_mount)
        for widget in (
            self.look_again_button,
            self.folder_edit,
            self.keep_radio,
            self.once_radio,
            self.drive_list,
            self.own_disk_button,
            self.startup_button,
            self.remove_button,
            self.close_button,
        ):
            widget.setEnabled(idle)

    def _toggle_details(self, shown: bool) -> None:
        self.details.setVisible(shown)
        self.details_button.setText("Hide Details" if shown else "Show Details")
        self._fit()

    def _show_result(self, level: str, text: str) -> None:
        name = _STATUS_NAMES.get(level, "StatusNeutral")
        self.result_label.setText(text)
        if self.result_label.objectName() != name:
            self.result_label.setObjectName(name)
            _repolish(self.result_label)
        self.result_label.setVisible(bool(text))
        self._fit()

    # -- Actions -----------------------------------------------------------------

    def _confirm(self, title: str, text: str, accept_label: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        accept = box.addButton(accept_label, QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        return box.clickedButton() is accept

    def _mount_selected(self) -> None:
        choice = self._selected_choice()
        folder = self._folder()
        if choice is None or mount_folder_problem(folder, protected=self._protected):
            return
        action = ACTION_MOUNT_ONCE if self.once_radio.isChecked() else ACTION_MOUNT_KEEP
        self._start(action, folder, choice.path)

    def _toggle_startup(self) -> None:
        state = self._state
        if state is None or not state.mounted:
            return
        action = str(self.startup_button.property("action") or "")
        if action == ACTION_FORGET and not self._confirm(
            "Stop mounting at startup",
            f"After a restart, no drive will be mounted at {state.mount_folder}, and recording "
            "will be refused until one is mounted here again.\n\nThe drive stays mounted for now.",
            "Stop Mounting at Startup",
        ):
            return
        device_path = state.drive.path if state.drive is not None else None
        self._start(action, state.mount_folder, device_path)

    def _remove_drive(self) -> None:
        state = self._state
        if state is None or not state.mounted:
            return
        if not self._confirm(
            "Safely remove the drive",
            f"Unmount the drive at {state.mount_folder}?\n\nRecording is refused until a drive is "
            "mounted here again. If automated recording is on, scheduled recordings will fail "
            "until then.",
            "Unmount Drive",
        ):
            return
        self._start(ACTION_UNMOUNT, state.mount_folder, None)

    def _use_own_disk(self) -> None:
        reason = self._busy_reason() if self._busy_reason is not None else None
        if reason:
            self._show_result(LEVEL_WARN, reason)
            return
        folder = own_disk_folder(self._config)
        if not self._confirm(
            "Record to the system disk",
            "Recordings will be saved on the system disk (the Pi's own SD card or SSD), in:\n\n"
            f"{folder}\n\n"
            "That disk is small. Use this for testing or short recordings; for an experiment, "
            "mount a data drive instead.",
            "Use the System Disk",
        ):
            return
        try:
            os.makedirs(folder, exist_ok=True)
            self._config = copy.deepcopy(self._on_choice(folder, False))
        except Exception as exc:
            self._show_result(LEVEL_FAIL, f"The data folder could not be changed: {exc}")
            return
        self.folder_edit.clear()
        self.refresh()
        self._show_result(LEVEL_WARN, f"Recordings will be saved on the system disk, in {folder}.")

    def _find_elevation(self) -> Elevation:
        if self._elevation is None:
            self._elevation = self._elevation_finder()
        return self._elevation

    def _start(self, action: str, folder: str, device_path: Optional[str]) -> None:
        if self._working:
            return
        reason = self._busy_reason() if self._busy_reason is not None else None
        if reason:
            self._show_result(LEVEL_WARN, reason)
            return
        elevation = self._find_elevation()
        self._pending = {"action": action, "folder": folder, "device": device_path, "elevation": elevation}
        self._output = []
        self.details.clear()
        if not elevation.available:
            self._show_terminal_command(
                "This desktop has no way to ask for administrator rights, so the drive was not changed."
            )
            self._pending = None
            return
        command = storage_command(
            action, elevation, config_path=self.config_path, mount_folder=folder, device_path=device_path
        )
        self._set_working(True)
        waiting = " Enter the administrator password when asked." if elevation.asks_for_password else ""
        self._show_result("", _WORKING_TEXT.get(action, "Working...") + waiting)
        self._launch(command)

    def _launch(self, command: List[str]) -> None:
        self._timeout.start(ACTION_TIMEOUT_MS)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.start(command[0], command[1:])

    def _set_working(self, working: bool) -> None:
        self._working = working
        self._update_enabled()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._output.append(text)

    def _process_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._timeout.stop()
        self._read_output()
        if status != QProcess.NormalExit and exit_code == 0:
            exit_code = 1
        self._action_finished(exit_code, "".join(self._output))

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.FailedToStart:
            return  # a crash or kill also ends in finished()
        self._timeout.stop()
        self._action_finished(PROCESS_DID_NOT_START, self.process.errorString())

    def _timed_out(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self._output.append("\nNo answer after two minutes; the command was stopped.\n")
        self.process.kill()

    def _show_terminal_command(self, reason: str) -> None:
        pending = self._pending or {}
        command = terminal_command(
            pending.get("action", ACTION_MOUNT_KEEP),
            config_path=self.config_path,
            mount_folder=pending.get("folder") or "",
            device_path=pending.get("device"),
        )
        self._show_result(
            LEVEL_FAIL,
            f"{reason}\n\nTo do it by hand, run this in a terminal on the Pi, then press Look Again:\n{command}",
        )

    def _action_finished(self, exit_code: int, output: str) -> None:
        """Handle the end of a storage command: look again, save the choice, say what happened."""
        pending, self._pending = self._pending, None
        self._set_working(False)
        self.details.setPlainText(output.strip())
        self.details_button.setVisible(bool(output.strip()))
        if pending is None:
            return
        action, folder, elevation = pending["action"], pending["folder"], pending["elevation"]

        if exit_code != 0:
            self._pending = pending  # for the terminal command, if it is shown
            not_allowed = explain_elevation_failure(elevation, exit_code)
            if exit_code == PROCESS_DID_NOT_START:
                self._show_terminal_command(f"The command could not be started ({output.strip() or 'unknown error'}).")
            elif not_allowed:
                self._show_terminal_command(not_allowed)
            else:
                self._show_result(
                    LEVEL_FAIL,
                    _last_meaningful_line(output) or f"The command failed (exit code {exit_code}).",
                )
            self._pending = None
            # The command may have got part of the way; show what is true now.
            self._show_state_keeping_result(self._read_state(self._config))
            return

        mounting = action in {ACTION_MOUNT_KEEP, ACTION_MOUNT_ONCE}
        candidate = copy.deepcopy(self._config)
        if mounting:
            candidate.setdefault("system", {})["data_root"] = folder
        state = self._read_state(candidate)
        if mounting and not state.mounted:
            self._show_result(
                LEVEL_FAIL,
                f"The command finished, but no drive is mounted at {folder}. Press Show Details for its output.",
            )
            self._show_state_keeping_result(self._read_state(self._config))
            return
        try:
            # Any of these means the box is set up to record to a drive, so a
            # run without one mounted must be refused.
            self._config = copy.deepcopy(self._on_choice(folder if mounting else None, True))
        except Exception as exc:
            self._show_result(LEVEL_FAIL, f"The drive was changed, but the settings could not be saved: {exc}")
            self._show_state_keeping_result(state)
            return
        self.folder_edit.clear()
        state = self._read_state(self._config)
        level, message = self._success_message(action, state)
        self._show_state_keeping_result(state)
        self._show_result(level, message)

    def _show_state_keeping_result(self, state: DataDriveState) -> None:
        shown, name = self.result_label.text(), self.result_label.objectName()
        self._show_state(state)
        self.result_label.setText(shown)
        self.result_label.setObjectName(name)
        self.result_label.setVisible(bool(shown))
        self._fit()

    @staticmethod
    def _success_message(action: str, state: DataDriveState) -> tuple[str, str]:
        if action in {ACTION_MOUNT_KEEP, ACTION_MOUNT_ONCE}:
            if not state.writable:
                return LEVEL_FAIL, (
                    f"The drive is mounted at {state.mount_folder}, but it cannot be written to. "
                    "It may be read-only or damaged."
                )
            after = (
                "It will be mounted here again after every restart."
                if action == ACTION_MOUNT_KEEP
                else "It will not be mounted again after a restart."
            )
            return LEVEL_OK, f"The drive is mounted at {state.mount_folder}. {after}"
        if action == ACTION_KEEP:
            return LEVEL_OK, "This drive will be mounted here again after every restart."
        if action == ACTION_FORGET:
            return LEVEL_OK, "No drive will be mounted here after a restart. The drive stays mounted for now."
        return LEVEL_OK, "The drive was unmounted and can be unplugged."

    # -- Closing -----------------------------------------------------------------

    def reject(self) -> None:
        if self._working:
            return  # do not walk away from a half-finished mount
        super().reject()

    def closeEvent(self, event: Any) -> None:
        if self._working:
            event.ignore()
            return
        super().closeEvent(event)

    def done(self, result: int) -> None:
        self._poll.stop()
        super().done(result)

