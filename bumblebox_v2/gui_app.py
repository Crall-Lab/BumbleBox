from __future__ import annotations

import csv
from copy import deepcopy
import json
import os
import queue
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .calibration import (
    apply_scale_to_config,
    calibrate_from_aruco_image,
    calibrate_from_points,
    capture_calibration_image,
    extract_points_from_labelme_json,
    format_calibration,
    parse_point,
)
from .camera_setup import (
    format_tracking_test_result,
    run_camera_tracking_test,
)
from .camera_profiles import (
    apply_camera_profile,
    camera_model_choices,
    camera_profile_choices,
    get_camera_model_info,
    get_camera_profile,
)
from .config import DEFAULT_USER_CONFIG_PATH, load_config, load_defaults, save_config, validate_config, write_default_config
from .doctor import format_report as format_doctor_report
from .doctor import run_doctor
from .fps_report import build_fps_report, format_report as format_fps_report
from .fps_sweep import (
    format_fps_sweep_report,
    fps_range,
    parse_fps_values,
    run_fps_sweep,
)
from .qt_env import build_camera_safe_env, build_qt_safe_env
from .fleet import (
    apply_queen_media_schedule_defaults,
    enroll_worker_config,
    format_fleet_discovery_report,
    format_fleet_enroll_result,
    format_fleet_init_result,
    format_queen_latest_status_report,
    format_fleet_status_report,
    format_queen_track_report,
    initialize_queen_config,
    recommended_media_max_videos_total,
    run_fleet_discovery,
    run_queen_pull_latest_videos,
    run_queen_latest_status,
    run_queen_pull_track_latest,
    run_queen_track_latest_videos,
    run_fleet_status,
    sync_media_capacity_to_workers,
)
from .gui_launcher import format_gui_shortcut_result, install_gui_shortcut
from .nest_labeling import (
    build_labelme_command,
    build_nest_labeling_command,
    check_nest_labeling_environment,
    default_calibration_labelmerc_path,
    default_script_path,
    format_nest_labeling_environment,
    launch_labelme,
    launch_nest_labeling,
)
from .roadmap import build_roadmap
from .runtime_alerts import build_runtime_alerts, format_runtime_alerts
from .run_bundle import export_run_bundle, format_bundle_export_result
from .run_engine import format_run_summary, record_live_test_clip, run_once
from .schedule_check import format_schedule_check_report, run_schedule_check
from .storage_manager import (
    build_storage_mount_sudo_command,
    build_storage_setup_sudo_command,
    discover_storage_devices,
    format_storage_mount_result,
    format_storage_setup_result,
    format_storage_status_report,
    get_storage_status,
    mount_storage_device_now,
    setup_storage_auto_mount,
)
from .status_history import list_recent_run_records, load_run_summary
from .systemd_units import (
    build_systemd_action_sudo_command,
    format_systemd_action_result,
    format_systemd_result,
    run_systemd_action,
    write_systemd_units,
)
from .thermal_camera import (
    apply_detected_thermal_config,
    format_thermal_check_result,
    run_thermal_check,
)
from .thermal_registration import (
    annotate_registration_overlay_with_session_tracking,
    apply_thermal_registration_to_config,
    format_thermal_registration,
    register_rgb_thermal_pair,
)

OCEAN_SLATE_PALETTE = {
    "app_bg": "#2B3A42",
    "panel_bg": "#334854",
    "tab_btn_bg": "#1F2A33",
    "tab_btn_active": "#2B3C49",
    "tab_selected": "#3F5B6F",
    "text_light": "#EAF2F7",
    "text_muted": "#C0D3E0",
    "entry_bg": "#22323D",
    "entry_fg": "#EAF2F7",
    "output_bg": "#111A22",
    "output_text": "#DCE8F2",
}

LAVENDER_LIGHT_PALETTE = {
    "app_bg": "#EDE2FA",
    "panel_bg": "#DFD1F4",
    "tab_btn_bg": "#C4B0E6",
    "tab_btn_active": "#B49ADB",
    "tab_selected": "#9D82CE",
    "text_light": "#1A1428",
    "text_muted": "#4F4563",
    "entry_bg": "#F7F1FF",
    "entry_fg": "#111111",
    "output_bg": "#FDFBFF",
    "output_text": "#111111",
}

OPTIMIZE_SWEEP_FIELDS = (
    (
        "minMarkerPerimeterRate",
        "float",
        "minMarkerPerimeterRate",
        "Minimum marker perimeter ratio. Raise to filter tiny false positives.",
    ),
    (
        "maxMarkerPerimeterRate",
        "float",
        "maxMarkerPerimeterRate",
        "Maximum marker perimeter ratio. Lower to filter unusually large false positives.",
    ),
    (
        "adaptiveThreshWinSizeMin",
        "int",
        "adaptiveThreshWinSizeMin",
        "Lower bound of adaptive threshold window size.",
    ),
    (
        "adaptiveThreshWinSizeMax",
        "int",
        "adaptiveThreshWinSizeMax",
        "Upper bound of adaptive threshold window size.",
    ),
    (
        "adaptiveThreshWinSizeStep",
        "int",
        "adaptiveThreshWinSizeStep",
        "Step size between min and max adaptive threshold windows.",
    ),
    (
        "polygonalApproxAccuracyRate",
        "float",
        "polygonalApproxAccuracyRate",
        "Polygon approximation tolerance. Lower values preserve more contour detail.",
    ),
    (
        "adaptiveThreshConstant",
        "int",
        "adaptiveThreshConstant",
        "Constant subtracted during adaptive thresholding. Useful for brightness-sensitive tuning.",
    ),
)

ARUCO_4X4_DICTIONARY_OPTIONS = ["4X4_50", "4X4_100", "4X4_250", "4X4_1000"]


class BumbleBoxV2GUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BumbleBox V2")
        self.geometry("980x680")
        self.minsize(920, 620)

        self.config_path_var = tk.StringVar(value=str(DEFAULT_USER_CONFIG_PATH))
        self.theme_mode_var = tk.StringVar(value=self._read_theme_mode_from_config())
        self.ui_mode_var = tk.StringVar(value="basic")
        self.config_fields: dict[str, tuple[tk.Variable, type]] = {}
        self._tab_lookup: dict[str, ttk.Frame] = {}
        self._advanced_widgets: list[tuple[tk.Widget, str, dict[str, object]]] = []
        self._advanced_widget_ids: set[str] = set()
        self._run_history_paths: dict[str, str] = {}
        self._run_history_thread: threading.Thread | None = None
        self._run_history_error: str | None = None
        self._run_history_records = None
        self._tracking_index_thread: threading.Thread | None = None
        self._tracking_index_output_text: str | None = None
        self._tracking_index_error: str | None = None
        self._nest_label_pid: int | None = None
        self._calibration_label_pid: int | None = None
        self._optimize_thread: threading.Thread | None = None
        self._optimize_error: str | None = None
        self._optimize_warning: str | None = None
        self._optimize_extra_report: str | None = None
        self._optimize_result = None
        self._optimize_refinement_result = None
        self._optimize_applied_config: str | None = None
        self._optimize_index_result = None
        self._optimize_index_error: str | None = None
        self._optimize_review_dialog: tk.Toplevel | None = None
        self._optimize_stop_event = threading.Event()
        self._optimize_progress_q: queue.Queue[
            tuple[int, int, list[dict[str, object]], dict[str, object]]
        ] = queue.Queue()
        self._optimize_started_monotonic: float | None = None
        self._optimize_last_progress_monotonic: float | None = None
        self._optimize_latest_progress: tuple[int, int, list[dict[str, object]], dict[str, object]] | None = None
        self._video_range_rows: list[dict[str, object]] = []
        self._video_range_selected_index: int | None = None
        self._video_range_tag_bounds: dict[str, dict[str, object]] = {}
        self._fps_sweep_thread: threading.Thread | None = None
        self._fps_sweep_error: str | None = None
        self._fps_sweep_report = None
        self._fps_sweep_output_text: str | None = None
        self._fps_sweep_report_has_errors = False
        self._fps_sweep_progress_q: queue.Queue[tuple[int, int, float]] = queue.Queue()
        self._fps_live_thread: threading.Thread | None = None
        self._fps_live_error: str | None = None
        self._fps_live_report: dict | None = None
        self._fps_live_capture = None
        self._fps_camera_reset_thread: threading.Thread | None = None
        self._fps_camera_reset_error: str | None = None
        self._fps_camera_reset_result = None
        self._fps_camera_reset_source: str | None = None
        self._thermal_check_result = None
        self._thermal_registration_summary_path: str | None = None
        self._config_form_canvas: tk.Canvas | None = None
        self._config_form_window: int | None = None
        self._theme_knob: tk.Scale | None = None
        self._suppress_theme_knob_callback = False
        self._palette = self._palette_for_theme_mode(self.theme_mode_var.get())
        self._results_sections: dict[tk.Text, dict[str, object]] = {}
        self._tab_titles: dict[str, str] = {}
        self._workflow_tabs: dict[str, list[str]] = {}
        self._workflow_label_var = tk.StringVar(value="")
        self._session_started_iso = datetime.now().isoformat(timespec="seconds")

        self._apply_ocean_slate_theme()
        self._setup_responsive_typography()
        self._build_header()
        self.main_content = ttk.Frame(self)
        self.main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._build_intro_page()
        self._build_notebook()
        self._show_intro()

    @staticmethod
    def _needs_scrollable_dialog(message_text: str) -> bool:
        if len(message_text) >= 280:
            return True
        if message_text.count("\n") >= 6:
            return True
        return False

    def _show_scrollable_dialog(self, title: str, message: str, level: str = "info") -> None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.resizable(True, True)
        dialog.minsize(560, 320)
        dialog.geometry("860x520")

        outer = ttk.Frame(dialog, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        level_text = {"error": "Error details", "warning": "Warning details", "info": "Details"}.get(level, "Details")
        ttk.Label(outer, text=level_text).pack(anchor="w")
        ttk.Label(outer, text="Scroll to view the full message.").pack(anchor="w", pady=(0, 6))

        text_frame = ttk.Frame(outer)
        text_frame.pack(fill=tk.BOTH, expand=True)
        text_widget = tk.Text(text_frame, wrap=tk.NONE)
        text_widget.grid(row=0, column=0, sticky="nsew")
        self._style_output_text(text_widget)
        y_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        text_widget.insert("1.0", message)
        text_widget.configure(state=tk.DISABLED)

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=(8, 0))

        def copy_details() -> None:
            try:
                self.clipboard_clear()
                self.clipboard_append(message)
            except Exception:
                pass

        ttk.Button(button_row, text="Copy Details", command=copy_details).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)

    def _show_message(self, level: str, title: str, message: str) -> None:
        text = str(message)
        if self._needs_scrollable_dialog(text):
            self._show_scrollable_dialog(title=title, message=text, level=level)
            return
        if level == "error":
            messagebox.showerror(title, text, parent=self)
        elif level == "warning":
            messagebox.showwarning(title, text, parent=self)
        else:
            messagebox.showinfo(title, text, parent=self)

    def _show_error(self, title: str, message: str) -> None:
        self._show_message("error", title, message)

    def _show_warning(self, title: str, message: str) -> None:
        self._show_message("warning", title, message)

    def _show_info(self, title: str, message: str) -> None:
        self._show_message("info", title, message)

    @staticmethod
    def _current_login_user() -> str:
        return (
            os.environ.get("SUDO_USER")
            or os.environ.get("USER")
            or os.environ.get("LOGNAME")
            or ""
        ).strip()

    def _user_linger_enabled(self) -> bool | None:
        user = self._current_login_user()
        if not user:
            return None
        if shutil.which("loginctl") is None:
            return None
        try:
            proc = subprocess.run(
                ["loginctl", "show-user", user, "-p", "Linger", "--value"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        value = (proc.stdout or "").strip().lower()
        if value == "yes":
            return True
        if value == "no":
            return False
        return None

    def _show_automation_started_popup(self, config: dict) -> None:
        scheduling = config.get("scheduling", {}) if isinstance(config.get("scheduling"), dict) else {}
        scope = str(scheduling.get("scope", "user")).strip().lower()
        if scope == "system":
            self._show_info(
                "Automated Recording Started",
                (
                    "Automated recording has started.\n\n"
                    "This system is using system-level timers, so scheduled recordings will continue "
                    "even if you log out."
                ),
            )
            return

        linger_enabled = self._user_linger_enabled()
        if linger_enabled is True:
            self._show_info(
                "Automated Recording Started",
                (
                    "Automated recording has started.\n\n"
                    "This BumbleBox is configured to keep scheduled recordings running even if you log out."
                ),
            )
            return

        self._show_info(
            "Automated Recording Started",
            (
                "Automated recording has started.\n\n"
                "Scheduled recordings will run while this BumbleBox user is logged in. "
                "If you need them to continue after logout, rerun the setup script to enable background scheduling."
            ),
        )

    def _apply_ocean_slate_theme(self) -> None:
        colors = self._palette
        style = ttk.Style(self)
        available = set(style.theme_names())
        if "clam" in available:
            style.theme_use("clam")

        self.configure(bg=colors["app_bg"])

        style.configure("TFrame", background=colors["panel_bg"])
        style.configure("TLabelframe", background=colors["panel_bg"], borderwidth=1)
        style.configure("TLabelframe.Label", background=colors["panel_bg"], foreground=colors["text_light"])
        style.configure("TLabel", background=colors["panel_bg"], foreground=colors["text_light"])
        style.configure(
            "TButton",
            background=colors["tab_btn_bg"],
            foreground=colors["text_light"],
            borderwidth=1,
            padding=(8, 4),
        )
        style.map(
            "TButton",
            background=[
                ("pressed", colors["tab_selected"]),
                ("active", colors["tab_btn_active"]),
            ],
            foreground=[
                ("disabled", colors["text_muted"]),
                ("!disabled", colors["text_light"]),
            ],
        )
        intro_primary_kwargs = {
            "background": colors["tab_btn_active"],
            "foreground": "#FFFFFF",
            "borderwidth": 1,
            "padding": (12, 7),
        }
        if hasattr(self, "_intro_button_font"):
            intro_primary_kwargs["font"] = self._intro_button_font
        style.configure("IntroPrimary.TButton", **intro_primary_kwargs)
        style.map(
            "IntroPrimary.TButton",
            background=[
                ("pressed", colors["tab_btn_bg"]),
                ("active", colors["tab_selected"]),
            ],
            foreground=[("disabled", "#E6E6E6"), ("!disabled", "#FFFFFF")],
        )
        style.configure("TCheckbutton", background=colors["panel_bg"], foreground=colors["text_light"])
        style.configure("TRadiobutton", background=colors["panel_bg"], foreground=colors["text_light"])
        style.map(
            "TCheckbutton",
            background=[("active", colors["panel_bg"])],
            foreground=[("disabled", colors["text_muted"]), ("!disabled", colors["text_light"])],
        )
        style.map(
            "TRadiobutton",
            background=[("active", colors["panel_bg"])],
            foreground=[("disabled", colors["text_muted"]), ("!disabled", colors["text_light"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=colors["entry_bg"],
            foreground=colors["entry_fg"],
        )
        style.map(
            "TEntry",
            fieldbackground=[("readonly", colors["entry_bg"])],
            foreground=[("readonly", colors["entry_fg"])],
        )
        style.configure(
            "TCombobox",
            fieldbackground=colors["entry_bg"],
            foreground=colors["entry_fg"],
            background=colors["tab_btn_bg"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["entry_bg"])],
            foreground=[("readonly", colors["entry_fg"])],
            selectbackground=[("readonly", colors["tab_selected"])],
            selectforeground=[("readonly", colors["text_light"])],
        )
        style.configure("TNotebook", background=colors["app_bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=colors["tab_btn_bg"],
            foreground=colors["text_light"],
            padding=(10, 5),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors["tab_selected"]), ("active", colors["tab_btn_active"])],
            foreground=[("selected", colors["text_light"]), ("!selected", colors["text_light"])],
        )
        style.configure(
            "Treeview",
            background=colors["output_bg"],
            foreground=colors["output_text"],
            fieldbackground=colors["output_bg"],
            rowheight=24,
        )
        style.map(
            "Treeview",
            background=[("selected", colors["tab_selected"])],
            foreground=[("selected", colors["text_light"])],
        )
        style.configure(
            "Treeview.Heading",
            background=colors["tab_btn_bg"],
            foreground=colors["text_light"],
        )
        style.configure(
            "RoadmapCard.TFrame",
            background=colors["entry_bg"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "RoadmapCardText.TLabel",
            background=colors["entry_bg"],
            foreground=colors["text_light"],
        )
        style.configure(
            "RoadmapCardMutedText.TLabel",
            background=colors["entry_bg"],
            foreground=colors["text_muted"],
        )
        style.configure(
            "RoadmapStateDone.TLabel",
            background=colors["entry_bg"],
            foreground="#63D47C",
        )
        style.configure(
            "RoadmapStateTodo.TLabel",
            background=colors["entry_bg"],
            foreground="#EACB63",
        )
        style.configure(
            "RoadmapStateOptional.TLabel",
            background=colors["entry_bg"],
            foreground=colors["text_muted"],
        )
        style.configure(
            "RoadmapStateDisabled.TLabel",
            background=colors["entry_bg"],
            foreground=colors["text_muted"],
        )
        style.configure(
            "RoadmapStateOptionalDone.TLabel",
            background=colors["entry_bg"],
            foreground="#63D47C",
        )
        style.configure(
            "RoadmapStateOptionalTodo.TLabel",
            background=colors["entry_bg"],
            foreground="#EACB63",
        )

    def _palette_for_theme_mode(self, mode: str) -> dict[str, str]:
        key = str(mode or "").strip().lower()
        if key == "light":
            return dict(LAVENDER_LIGHT_PALETTE)
        return dict(OCEAN_SLATE_PALETTE)

    def _read_theme_mode_from_config(self) -> str:
        try:
            config_path = Path(self.config_path_var.get()).expanduser()
            if config_path.exists():
                config = load_config(config_path)
            else:
                config = load_defaults()
            raw = str(config.get("runtime", {}).get("ui_theme_mode", "dark")).strip().lower()
            return "light" if raw == "light" else "dark"
        except Exception:
            return "dark"

    def _persist_theme_mode_to_config(self) -> None:
        try:
            config_path = Path(self.config_path_var.get()).expanduser()
            if config_path.exists():
                config = load_config(config_path)
            else:
                config = load_defaults()
            config.setdefault("runtime", {})
            config["runtime"]["ui_theme_mode"] = (
                "light" if str(self.theme_mode_var.get()).strip().lower() == "light" else "dark"
            )
            save_config(config_path, config)
        except Exception:
            # Best-effort persistence; GUI should still function if config write fails.
            pass

    def _set_theme_mode(
        self,
        mode: str,
        *,
        sync_knob: bool = True,
        persist_preference: bool = True,
    ) -> None:
        normalized = "light" if str(mode or "").strip().lower() == "light" else "dark"
        previous = str(self.theme_mode_var.get() or "dark").strip().lower()
        if normalized == previous and self._palette == self._palette_for_theme_mode(normalized):
            if persist_preference:
                self._persist_theme_mode_to_config()
            if sync_knob:
                self._sync_theme_knob_position()
            return

        self.theme_mode_var.set(normalized)
        self._palette = self._palette_for_theme_mode(normalized)
        self._apply_ocean_slate_theme()

        if self._config_form_canvas is not None:
            try:
                self._config_form_canvas.configure(bg=self._palette["panel_bg"])
            except Exception:
                pass

        for widget in list(self._results_sections.keys()):
            try:
                self._style_output_text(widget)
            except Exception:
                continue

        self._refresh_intro_theme_widgets()

        if persist_preference:
            self._persist_theme_mode_to_config()

        if sync_knob:
            self._sync_theme_knob_position()

    def _on_theme_knob_changed(self, value: str) -> None:
        if bool(getattr(self, "_suppress_theme_knob_callback", False)):
            return
        try:
            numeric = float(value)
        except Exception:
            return
        desired = "light" if numeric >= 0.5 else "dark"
        self._set_theme_mode(desired, sync_knob=False, persist_preference=True)

    def _on_theme_knob_released(self, _event=None) -> None:
        self._sync_theme_knob_position()

    def _sync_theme_knob_position(self) -> None:
        knob = getattr(self, "_theme_knob", None)
        if knob is None:
            return
        try:
            if not knob.winfo_exists():
                return
        except Exception:
            return
        target = 1.0 if str(self.theme_mode_var.get()).strip().lower() == "light" else 0.0
        self._suppress_theme_knob_callback = True
        try:
            knob.set(target)
        finally:
            self._suppress_theme_knob_callback = False

    def _refresh_intro_theme_widgets(self) -> None:
        colors = self._palette
        hero = getattr(self, "_intro_hero_frame", None)
        if hero is not None:
            try:
                hero.configure(bg=colors["entry_bg"], highlightbackground=colors["tab_selected"])
            except Exception:
                pass

        for frame in getattr(self, "_intro_card_frames", []):
            try:
                frame.configure(bg=colors["entry_bg"])
            except Exception:
                continue

        for frame in getattr(self, "_intro_card_header_frames", []):
            try:
                frame.configure(bg=colors["entry_bg"])
            except Exception:
                continue

        for label in getattr(self, "_intro_desc_labels", []):
            try:
                label.configure(bg=colors["entry_bg"], fg=colors["text_muted"])
            except Exception:
                continue

        self._refresh_intro_card_tiles()
        self._refresh_intro_title_pills()

        footer = getattr(self, "_intro_theme_footer", None)
        if footer is not None:
            try:
                footer.configure(bg=colors["panel_bg"])
            except Exception:
                pass

        panel = getattr(self, "_intro_theme_panel", None)
        if panel is not None:
            try:
                panel.configure(bg=colors["entry_bg"], highlightbackground=colors["tab_btn_active"])
            except Exception:
                pass

        title_label = getattr(self, "_intro_theme_title_label", None)
        if title_label is not None:
            try:
                title_label.configure(bg=colors["entry_bg"], fg=colors["text_light"])
            except Exception:
                pass

        mode = str(self.theme_mode_var.get() or "dark").strip().lower()
        dark_label = getattr(self, "_intro_theme_dark_label", None)
        if dark_label is not None:
            try:
                dark_label.configure(
                    bg=colors["entry_bg"],
                    fg=(colors["text_light"] if mode == "dark" else colors["text_muted"]),
                )
            except Exception:
                pass
        light_label = getattr(self, "_intro_theme_light_label", None)
        if light_label is not None:
            try:
                light_label.configure(
                    bg=colors["entry_bg"],
                    fg=(colors["text_light"] if mode == "light" else colors["text_muted"]),
                )
            except Exception:
                pass

        knob = getattr(self, "_theme_knob", None)
        if knob is not None:
            try:
                knob.configure(
                    bg=colors["entry_bg"],
                    fg=colors["text_light"],
                    activebackground=colors["tab_selected"],
                    troughcolor=colors["tab_btn_bg"],
                    highlightbackground=colors["entry_bg"],
                    highlightcolor=colors["entry_bg"],
                )
            except Exception:
                pass

    def _rounded_rect_points(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> list[float]:
        r = max(0.0, min(float(radius), (x2 - x1) / 2.0, (y2 - y1) / 2.0))
        return [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]

    def _create_intro_title_pill(
        self,
        parent: tk.Widget,
        *,
        text: str,
        font: tkfont.Font,
        fill: str,
        fg: str,
        bg: str,
        pad_x: int,
        pad_y: int,
        radius: int,
        role: str,
    ) -> tk.Canvas:
        canvas = tk.Canvas(parent, bd=0, highlightthickness=0, relief=tk.FLAT, bg=bg)
        # Initialize with a tiny valid polygon; coordinates are replaced during layout.
        rect_id = canvas.create_polygon(
            1, 1, 2, 1, 2, 2, 1, 2,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=fill,
        )
        text_id = canvas.create_text(0, 0, text=text, fill=fg, font=font)
        meta: dict[str, object] = {
            "canvas": canvas,
            "rect_id": rect_id,
            "text_id": text_id,
            "text": text,
            "font": font,
            "fill": fill,
            "fg": fg,
            "bg": bg,
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
            "radius": int(radius),
            "role": role,
        }
        if not hasattr(self, "_intro_title_pills"):
            self._intro_title_pills: list[dict[str, object]] = []
        self._intro_title_pills.append(meta)
        self._layout_intro_title_pill(meta)
        return canvas

    def _create_intro_card_tile(
        self,
        parent: tk.Widget,
        *,
        radius: int = 14,
        pad_x: int = 14,
        pad_y: int = 12,
    ) -> tuple[tk.Canvas, tk.Frame]:
        colors = self._palette
        canvas = tk.Canvas(parent, bd=0, highlightthickness=0, relief=tk.FLAT, bg=colors["panel_bg"])
        rect_id = canvas.create_polygon(
            1, 1, 2, 1, 2, 2, 1, 2,
            smooth=True,
            splinesteps=24,
            fill=colors["entry_bg"],
            outline=colors["tab_btn_active"],
        )
        frame = tk.Frame(canvas, bg=colors["entry_bg"], padx=pad_x, pady=pad_y)
        window_id = canvas.create_window((pad_x, pad_y), window=frame, anchor="nw", width=1)

        meta: dict[str, object] = {
            "canvas": canvas,
            "rect_id": rect_id,
            "window_id": window_id,
            "frame": frame,
            "radius": int(radius),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
        }
        if not hasattr(self, "_intro_card_tiles"):
            self._intro_card_tiles: list[dict[str, object]] = []
        self._intro_card_tiles.append(meta)

        canvas.bind("<Configure>", lambda _event, item=meta: self._layout_intro_card_tile(item), add="+")
        self.after_idle(lambda item=meta: self._layout_intro_card_tile(item))
        return canvas, frame

    def _layout_intro_card_tile(self, meta: dict[str, object]) -> None:
        canvas = meta.get("canvas")
        frame = meta.get("frame")
        if not isinstance(canvas, tk.Canvas) or not isinstance(frame, tk.Frame):
            return
        try:
            if not canvas.winfo_exists():
                return
        except Exception:
            return

        pad_x = max(0, int(meta.get("pad_x", 14)))
        pad_y = max(0, int(meta.get("pad_y", 12)))
        radius = max(0, int(meta.get("radius", 14)))
        width = max(80, int(canvas.winfo_width()))

        window_id = meta.get("window_id")
        if isinstance(window_id, int):
            content_width = max(1, width - (pad_x * 2))
            try:
                canvas.coords(window_id, pad_x, pad_y)
                canvas.itemconfigure(window_id, width=content_width)
            except Exception:
                return

        try:
            frame.update_idletasks()
        except Exception:
            pass
        min_height = max(120, int(frame.winfo_reqheight()) + (pad_y * 2))
        current_height = max(1, int(canvas.winfo_height()))
        target_height = max(current_height, min_height)
        if abs(target_height - current_height) > 1:
            try:
                canvas.configure(height=target_height)
            except Exception:
                return
        height = max(target_height, max(1, int(canvas.winfo_height())))

        points = self._rounded_rect_points(1, 1, max(2, width - 1), max(2, height - 1), radius)
        try:
            canvas.coords(meta["rect_id"], *points)
            canvas.itemconfigure(
                meta["rect_id"],
                fill=self._palette["entry_bg"],
                outline=self._palette["tab_btn_active"],
            )
            canvas.configure(bg=self._palette["panel_bg"])
            frame.configure(bg=self._palette["entry_bg"])
        except Exception:
            return

    def _refresh_intro_card_tiles(self) -> None:
        for meta in getattr(self, "_intro_card_tiles", []):
            self._layout_intro_card_tile(meta)

    def _layout_intro_title_pill(self, meta: dict[str, object]) -> None:
        canvas = meta["canvas"]
        if not isinstance(canvas, tk.Canvas):
            return
        font_obj = meta["font"]
        if not isinstance(font_obj, tkfont.Font):
            return
        text = str(meta.get("text", ""))
        pad_x = max(0, int(meta.get("pad_x", 10)))
        pad_y = max(0, int(meta.get("pad_y", 6)))
        width = max(12, int(font_obj.measure(text)) + (pad_x * 2))
        height = max(12, int(font_obj.metrics("linespace")) + (pad_y * 2))
        try:
            canvas.configure(width=width, height=height, bg=str(meta.get("bg", "#000000")))
        except Exception:
            return
        points = self._rounded_rect_points(1, 1, width - 1, height - 1, int(meta.get("radius", 10)))
        canvas.coords(meta["rect_id"], *points)
        canvas.coords(meta["text_id"], width / 2.0, height / 2.0)
        canvas.itemconfigure(
            meta["rect_id"],
            fill=str(meta.get("fill", "#333333")),
            outline=str(meta.get("fill", "#333333")),
        )
        canvas.itemconfigure(
            meta["text_id"],
            text=text,
            fill=str(meta.get("fg", "#FFFFFF")),
            font=font_obj,
        )

    def _refresh_intro_title_pills(self) -> None:
        colors = self._palette
        pills = getattr(self, "_intro_title_pills", [])
        for meta in pills:
            role = str(meta.get("role", "card"))
            meta["fill"] = colors["tab_selected"]
            meta["fg"] = "#FFFFFF"
            meta["bg"] = colors["entry_bg"]
            if role == "hero":
                meta["font"] = self._hero_title_font
                meta["radius"] = 14
                meta["pad_x"] = 24
                meta["pad_y"] = 10
            else:
                meta["font"] = self._intro_card_title_font
                meta["radius"] = 10
                meta["pad_x"] = 16
                meta["pad_y"] = 7
            self._layout_intro_title_pill(meta)

    def _setup_responsive_typography(self) -> None:
        self._base_window_width = 980
        self._base_window_height = 680
        self._font_base_sizes: dict[str, int] = {}

        for name in ["TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkCaptionFont", "TkFixedFont"]:
            try:
                font_obj = tkfont.nametofont(name)
                size = abs(int(font_obj.cget("size")))
                if size > 0:
                    self._font_base_sizes[name] = size
            except Exception:
                continue

        default_font = tkfont.nametofont("TkDefaultFont")
        family = str(default_font.cget("family"))
        default_size = self._font_base_sizes.get("TkDefaultFont", 10)
        hero_base = max(14, default_size + 4)
        card_base = max(11, default_size + 1)
        self._hero_title_font = tkfont.Font(
            self,
            family=family,
            size=max(18, int(round(hero_base * 1.5))),
            weight="bold",
        )
        self._header_title_font = tkfont.Font(self, family=family, size=max(13, default_size + 3), weight="bold")
        self._intro_card_title_font = tkfont.Font(
            self,
            family=family,
            size=max(16, int(round(card_base * 1.5))),
            weight="bold",
        )
        self._intro_button_font = tkfont.Font(self, family=family, size=max(11, default_size + 1), weight="bold")
        self._hero_title_base_size = abs(int(self._hero_title_font.cget("size")))
        self._header_title_base_size = abs(int(self._header_title_font.cget("size")))
        self._intro_card_title_base_size = abs(int(self._intro_card_title_font.cget("size")))
        self._intro_button_base_size = abs(int(self._intro_button_font.cget("size")))
        ttk.Style(self).configure("IntroPrimary.TButton", font=self._intro_button_font)
        self._font_scale = 1.0
        self.bind("<Configure>", self._on_root_resize, add="+")

    def _on_root_resize(self, event) -> None:
        if event.widget is not self:
            return
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        scale = min(width / self._base_window_width, height / self._base_window_height)
        scale = max(1.0, min(1.55, scale))

        if abs(scale - self._font_scale) < 0.03:
            return
        self._font_scale = scale

        for name, base_size in self._font_base_sizes.items():
            try:
                tkfont.nametofont(name).configure(size=max(8, int(round(base_size * scale))))
            except Exception:
                continue
        self._hero_title_font.configure(size=max(13, int(round(self._hero_title_base_size * scale))))
        self._header_title_font.configure(size=max(12, int(round(self._header_title_base_size * scale))))
        self._intro_card_title_font.configure(size=max(10, int(round(self._intro_card_title_base_size * scale))))
        self._intro_button_font.configure(size=max(9, int(round(self._intro_button_base_size * scale))))
        self._refresh_intro_card_tiles()
        self._refresh_intro_title_pills()
        self._refresh_intro_wraplength()
        self._refresh_roadmap_label_wraplength()

    def _style_output_text(self, widget: tk.Text) -> None:
        colors = self._palette
        widget.configure(
            bg=colors["output_bg"],
            fg=colors["output_text"],
            insertbackground=colors["output_text"],
            selectbackground=colors["tab_selected"],
            selectforeground=colors["text_light"],
            highlightthickness=1,
            highlightbackground=colors["panel_bg"],
            highlightcolor=colors["tab_selected"],
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )

    def _create_results_section(
        self,
        parent: tk.Widget,
        *,
        title: str = "Results",
        text_height: int = 9,
        default_visible: bool = False,
        auto_hide_when_empty: bool = True,
        show_status: bool = False,
        auto_height: bool = False,
        min_text_lines: int = 3,
        max_text_lines: int = 14,
        fill: str = tk.BOTH,
        expand: bool = True,
        pady: tuple[int, int] = (10, 0),
        wrap: str = tk.WORD,
        horizontal_scroll: bool = False,
    ) -> tk.Text:
        container = ttk.Frame(parent)
        container.pack(fill=(tk.BOTH if expand else tk.X), expand=expand, pady=pady)

        header = ttk.Frame(container)
        header.pack(fill=tk.X)
        status_var = tk.StringVar(value="No output") if show_status else None
        ttk.Label(header, text=title).pack(side=tk.LEFT)
        if status_var is not None:
            ttk.Label(header, textvariable=status_var).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.Frame(container)
        text_widget = tk.Text(body, wrap=wrap, height=text_height)
        text_widget.pack(side=tk.TOP, fill=fill, expand=expand)
        self._style_output_text(text_widget)
        if horizontal_scroll:
            x_scroll = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=text_widget.xview)
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            text_widget.configure(xscrollcommand=x_scroll.set)

        toggle_btn = ttk.Button(header, text="Show Results")
        toggle_btn.pack(side=tk.RIGHT)

        self._results_sections[text_widget] = {
            "container": container,
            "body": body,
            "button": toggle_btn,
            "status_var": status_var,
            "fill": fill,
            "expand": expand,
            "auto_hide_when_empty": auto_hide_when_empty,
            "auto_height": auto_height,
            "min_text_lines": max(1, int(min_text_lines)),
            "max_text_lines": max(1, int(max_text_lines)),
        }
        toggle_btn.configure(command=lambda widget=text_widget: self._toggle_results_section(widget))
        text_widget.bind("<<Modified>>", self._on_results_text_modified, add="+")
        text_widget.edit_modified(False)

        self._set_results_section_visible(text_widget, default_visible)
        return text_widget

    def _replace_text_preserving_scroll(
        self,
        text_widget: tk.Text,
        content: str,
        *,
        follow_bottom_threshold: float = 0.02,
    ) -> None:
        try:
            first, last = text_widget.yview()
        except Exception:
            first, last = 0.0, 1.0
        try:
            had_content = bool(text_widget.get("1.0", tk.END).strip())
        except Exception:
            had_content = False
        was_near_bottom = had_content and (1.0 - float(last)) <= follow_bottom_threshold

        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)

        def restore_view() -> None:
            try:
                if was_near_bottom:
                    text_widget.yview_moveto(1.0)
                else:
                    text_widget.yview_moveto(max(0.0, min(1.0, float(first))))
            except Exception:
                pass

        self.after_idle(restore_view)

    def _set_results_section_visible(self, text_widget: tk.Text, visible: bool) -> None:
        meta = self._results_sections.get(text_widget)
        if not meta:
            return

        body = meta["body"]
        button = meta["button"]
        fill = meta["fill"]
        expand = bool(meta["expand"])

        if visible:
            if body.winfo_manager() != "pack":
                body.pack(fill=fill, expand=expand, pady=(6, 0))
            button.configure(text="Hide Results")
        else:
            if body.winfo_manager() == "pack":
                body.pack_forget()
            button.configure(text="Show Results")

    def _toggle_results_section(self, text_widget: tk.Text) -> None:
        meta = self._results_sections.get(text_widget)
        if not meta:
            return
        body = meta["body"]
        currently_visible = body.winfo_manager() == "pack"
        self._set_results_section_visible(text_widget, not currently_visible)

    def _on_results_text_modified(self, event) -> None:
        text_widget = event.widget
        if not isinstance(text_widget, tk.Text):
            return
        if not text_widget.edit_modified():
            return

        meta = self._results_sections.get(text_widget)
        if meta is None:
            text_widget.edit_modified(False)
            return

        content = text_widget.get("1.0", tk.END).strip()
        status_var = meta["status_var"]
        if isinstance(status_var, tk.StringVar):
            status_var.set("Output available" if content else "No output")

        self._auto_size_results_text(text_widget, content)

        if content:
            self._set_results_section_visible(text_widget, True)
        elif bool(meta.get("auto_hide_when_empty", True)):
            self._set_results_section_visible(text_widget, False)

        text_widget.edit_modified(False)

    def _auto_size_results_text(self, text_widget: tk.Text, content: str) -> None:
        meta = self._results_sections.get(text_widget)
        if not meta or not bool(meta.get("auto_height", False)):
            return

        min_lines = max(1, int(meta.get("min_text_lines", 3)))
        max_lines = max(min_lines, int(meta.get("max_text_lines", 14)))
        line_count = max(1, len((content or "").splitlines()))
        target_lines = min(max_lines, max(min_lines, line_count + 1))
        try:
            text_widget.configure(height=target_lines)
        except Exception:
            return

    def _build_intro_page(self) -> None:
        colors = self._palette
        self.intro_frame = ttk.Frame(self.main_content, padding=12)

        self._intro_card_frames: list[tk.Frame] = []
        self._intro_card_header_frames: list[tk.Frame] = []
        self._intro_card_tiles: list[dict[str, object]] = []
        self._intro_title_pills: list[dict[str, object]] = []
        self._intro_desc_labels: list[tk.Label] = []

        hero = tk.Frame(
            self.intro_frame,
            bg=colors["entry_bg"],
            highlightthickness=1,
            highlightbackground=colors["tab_selected"],
            padx=18,
            pady=14,
        )
        self._intro_hero_frame = hero
        hero.pack(fill=tk.X, pady=(0, 14))
        self._intro_hero_title = self._create_intro_title_pill(
            hero,
            text="BumbleBox Control Center",
            font=self._hero_title_font,
            fill=colors["tab_selected"],
            fg="#FFFFFF",
            bg=colors["entry_bg"],
            pad_x=24,
            pad_y=10,
            radius=14,
            role="hero",
        )
        self._intro_hero_title.pack(anchor="center", pady=(2, 2))

        self._intro_cards_frame = ttk.Frame(self.intro_frame)
        self._intro_cards_frame.pack(fill=tk.BOTH, expand=True)

        workflows = [
            (
                "setup",
                "BumbleBox Setup",
                "Bring up hardware and configuration: roadmap, diagnostics, storage setup, camera setup, tracking optimization, FPS checks, and calibration.",
            ),
            (
                "schedule_run",
                "Schedule and Run",
                "Validate timing and memory assumptions, then manage scheduled execution and immediate runs.",
            ),
            (
                "nest_labeling",
                "Nest Labeling",
                "Check labeling environment readiness and launch the nest-labeling workflow.",
            ),
            (
                "fleet",
                "Fleet Setup",
                "Configure and monitor queen/worker BumbleBoxes, worker discovery, and media synchronization.",
            ),
        ]

        for idx, (key, label, desc) in enumerate(workflows):
            row = idx // 2
            col = idx % 2
            card_canvas, card = self._create_intro_card_tile(
                self._intro_cards_frame,
                radius=14,
                pad_x=14,
                pad_y=12,
            )
            self._intro_card_frames.append(card)
            card_canvas.grid(row=row, column=col, sticky="nsew", padx=7, pady=7)

            header = tk.Frame(card, bg=colors["entry_bg"])
            self._intro_card_header_frames.append(header)
            header.pack(fill=tk.X)
            title_label = self._create_intro_title_pill(
                header,
                text=label,
                font=self._intro_card_title_font,
                fill=colors["tab_selected"],
                fg="#FFFFFF",
                bg=colors["entry_bg"],
                pad_x=16,
                pad_y=7,
                radius=10,
                role="card",
            )
            title_label.pack(anchor="center", pady=(6, 0))

            desc_label = tk.Label(
                card,
                text=desc,
                bg=colors["entry_bg"],
                fg=colors["text_muted"],
                justify=tk.CENTER,
                anchor="center",
            )
            desc_label.pack(fill=tk.X, anchor="center", pady=(10, 12))
            self._intro_desc_labels.append(desc_label)

            ttk.Button(
                card,
                text="Open",
                style="IntroPrimary.TButton",
                command=lambda workflow_key=key: self._open_workflow(workflow_key),
            ).pack(anchor="center")

        self._intro_cards_frame.columnconfigure(0, weight=1)
        self._intro_cards_frame.columnconfigure(1, weight=1)
        self._intro_cards_frame.rowconfigure(0, weight=1)
        self._intro_cards_frame.rowconfigure(1, weight=1)

        self._intro_theme_footer = tk.Frame(self.intro_frame, bg=colors["panel_bg"])
        self._intro_theme_footer.pack(fill=tk.X, pady=(10, 0))

        self._intro_theme_panel = tk.Frame(
            self._intro_theme_footer,
            bg=colors["entry_bg"],
            highlightthickness=1,
            highlightbackground=colors["tab_btn_active"],
            padx=12,
            pady=8,
        )
        self._intro_theme_panel.pack(side=tk.RIGHT)

        self._intro_theme_title_label = tk.Label(
            self._intro_theme_panel,
            text="Appearance",
            bg=colors["entry_bg"],
            fg=colors["text_light"],
            font=self._header_title_font,
        )
        self._intro_theme_title_label.pack(side=tk.LEFT, padx=(0, 8))
        self._intro_theme_dark_label = tk.Label(
            self._intro_theme_panel,
            text="Dark",
            bg=colors["entry_bg"],
            fg=colors["text_light"],
        )
        self._intro_theme_dark_label.pack(side=tk.LEFT, padx=(0, 6))
        self._theme_knob = tk.Scale(
            self._intro_theme_panel,
            from_=0.0,
            to=1.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            length=50,
            showvalue=False,
            sliderlength=14,
            width=8,
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT,
            bg=colors["entry_bg"],
            fg=colors["text_light"],
            activebackground=colors["tab_selected"],
            troughcolor=colors["tab_btn_bg"],
            command=self._on_theme_knob_changed,
        )
        self._theme_knob.pack(side=tk.LEFT)
        self._theme_knob.bind("<ButtonRelease-1>", self._on_theme_knob_released)
        self._intro_theme_light_label = tk.Label(
            self._intro_theme_panel,
            text="Light",
            bg=colors["entry_bg"],
            fg=colors["text_light"],
        )
        self._intro_theme_light_label.pack(side=tk.LEFT, padx=(6, 0))

        self._sync_theme_knob_position()
        self._refresh_intro_card_tiles()
        self._refresh_intro_wraplength()

    def _refresh_intro_wraplength(self) -> None:
        cards_frame = getattr(self, "_intro_cards_frame", None)
        if cards_frame is None:
            return

        frame_width = cards_frame.winfo_width()
        if frame_width <= 0:
            frame_width = 900
        card_width = max(320, int((frame_width - 28) / 2))
        text_wrap = max(240, card_width - 44)

        labels = getattr(self, "_intro_desc_labels", [])
        for label in labels:
            try:
                label.configure(wraplength=text_wrap)
            except Exception:
                continue

    def _show_intro(self) -> None:
        self._workflow_label_var.set("")
        if hasattr(self, "home_button"):
            self.home_button.pack_forget()
        if hasattr(self, "notebook") and self.notebook.winfo_manager() == "pack":
            self.notebook.pack_forget()
        if self.intro_frame.winfo_manager() != "pack":
            self.intro_frame.pack(fill=tk.BOTH, expand=True)
        self._sync_theme_knob_position()
        self._refresh_intro_wraplength()

    def _open_workflow(self, workflow_key: str) -> None:
        tab_keys = self._workflow_tabs.get(workflow_key, [])
        if not tab_keys:
            return

        if self.intro_frame.winfo_manager() == "pack":
            self.intro_frame.pack_forget()
        if self.notebook.winfo_manager() != "pack":
            self.notebook.pack(fill=tk.BOTH, expand=True)
        if hasattr(self, "home_button"):
            if self.home_button.winfo_manager() != "pack":
                self.home_button.pack(side=tk.RIGHT)

        self._workflow_label_var.set(
            {
                "setup": "BumbleBox Setup",
                "schedule_run": "Schedule and Run",
                "nest_labeling": "Nest Labeling",
                "fleet": "Fleet Setup",
            }.get(workflow_key, workflow_key.title())
        )

        self._set_visible_tabs(tab_keys, select_key=tab_keys[0] if tab_keys else None)

    def _set_visible_tabs(self, tab_keys: list[str], select_key: str | None = None) -> None:
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

        for key in tab_keys:
            tab = self._tab_lookup.get(key)
            if tab is None:
                continue
            self.notebook.add(tab, text=self._tab_titles.get(key, key))

        if select_key is not None:
            tab = self._tab_lookup.get(select_key)
            if tab is not None:
                self.notebook.select(tab)

    def _build_header(self) -> None:
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.X)

        title_row = ttk.Frame(frame)
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(title_row, text="BumbleBox V2", font=self._header_title_font).pack(side=tk.LEFT)
        ttk.Label(title_row, textvariable=self._workflow_label_var).pack(side=tk.LEFT, padx=(14, 4))
        self.home_button = ttk.Button(title_row, text="Back to Home", command=self._show_intro)
        self.home_button.pack(side=tk.RIGHT)

        mode_row = ttk.Frame(frame)
        mode_row.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(mode_row, text="View mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_row,
            text="Basic",
            value="basic",
            variable=self.ui_mode_var,
            command=self._set_ui_mode,
        ).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Radiobutton(
            mode_row,
            text="Advanced",
            value="advanced",
            variable=self.ui_mode_var,
            command=self._set_ui_mode,
        ).pack(side=tk.LEFT, padx=(2, 8))
        self._make_help_button(
            mode_row,
            title="Basic vs Advanced",
            details=(
                "Basic mode hides less-common tuning controls so setup is simpler for first-time use.\n\n"
                "Advanced mode shows all controls, including deeper optimization/sweep options. "
                "Use Advanced when you need manual tuning.\n\n"
                "Basic hides rarely used tuning controls; core setup, scheduling, recording, and tracking "
                "workflows remain available."
            ),
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(frame, text="Basic hides rarely used controls.").grid(row=2, column=0, sticky="w", pady=(2, 0))
        frame.columnconfigure(0, weight=1)

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.main_content)

        self.doctor_tab = ttk.Frame(self.notebook, padding=12)
        self.storage_tab = ttk.Frame(self.notebook, padding=12)
        self.camera_setup_tab = ttk.Frame(self.notebook, padding=12)
        self.roadmap_tab = ttk.Frame(self.notebook, padding=12)
        self.fps_tab = ttk.Frame(self.notebook, padding=12)
        self.calibration_tab = ttk.Frame(self.notebook, padding=12)
        self.schedule_check_tab = ttk.Frame(self.notebook, padding=12)
        self.optimize_tracking_tab = ttk.Frame(self.notebook, padding=12)
        self.config_tab = ttk.Frame(self.notebook, padding=12)
        self.nest_label_tab = ttk.Frame(self.notebook, padding=12)
        self.fleet_tab = ttk.Frame(self.notebook, padding=12)
        self.run_tab = ttk.Frame(self.notebook, padding=12)

        self._build_doctor_tab()
        self._build_storage_tab()
        self._build_camera_setup_tab()
        self._build_roadmap_tab()
        self._build_config_tab()
        self._build_fps_tab()
        self._build_calibration_tab()
        self._build_schedule_check_tab()
        self._build_optimize_tracking_tab()
        self._build_nest_label_tab()
        self._build_fleet_tab()
        self._build_run_tab()

        self._tab_lookup = {
            "doctor": self.doctor_tab,
            "storage": self.storage_tab,
            "camera_setup": self.camera_setup_tab,
            "roadmap": self.roadmap_tab,
            "config": self.config_tab,
            "fps": self.fps_tab,
            "calibration": self.calibration_tab,
            "schedule_check": self.schedule_check_tab,
            "tracking_optimization": self.optimize_tracking_tab,
            "nest_labeling": self.nest_label_tab,
            "fleet": self.fleet_tab,
            "run": self.run_tab,
        }
        self._tab_titles = {
            "roadmap": "Setup Roadmap",
            "doctor": "Doctor",
            "storage": "Storage",
            "config": "Config Editor",
            "camera_setup": "Camera Setup",
            "tracking_optimization": "Tracking Optimization",
            "fps": "FPS Report",
            "calibration": "Calibration",
            "schedule_check": "Schedule Check",
            "run": "Schedule and Run",
            "nest_labeling": "Nest Labeling",
            "fleet": "Fleet Setup",
        }
        self._workflow_tabs = {
            "setup": ["roadmap", "doctor", "storage", "config", "camera_setup", "tracking_optimization", "fps", "calibration"],
            "schedule_run": ["schedule_check", "run"],
            "nest_labeling": ["nest_labeling"],
            "fleet": ["fleet"],
        }
        self._set_ui_mode()

    def _register_advanced_widget(self, widget: tk.Widget) -> None:
        widget_id = str(widget)
        if widget_id in self._advanced_widget_ids:
            return

        manager = widget.winfo_manager()
        if manager not in {"grid", "pack"}:
            return

        layout_info: dict[str, object] = {}
        if manager == "pack":
            layout_info = dict(widget.pack_info())
            if not layout_info.get("before"):
                parent = widget.master
                if parent is not None and hasattr(parent, "pack_slaves"):
                    siblings = list(parent.pack_slaves())
                    if widget in siblings:
                        index = siblings.index(widget)
                        if index + 1 < len(siblings):
                            layout_info["before"] = str(siblings[index + 1])

        self._advanced_widgets.append((widget, manager, layout_info))
        self._advanced_widget_ids.add(widget_id)

    def _toggle_advanced_widgets(self, show_advanced: bool) -> None:
        for widget, manager, layout_info in self._advanced_widgets:
            if show_advanced:
                if manager == "grid":
                    if widget.winfo_manager() != "grid":
                        widget.grid()
                elif manager == "pack":
                    if widget.winfo_manager() != "pack":
                        pack_kwargs: dict[str, object] = {}
                        for key, value in layout_info.items():
                            if key == "in":
                                if value:
                                    pack_kwargs["in_"] = value
                                continue
                            if key in {"after", "before"} and not value:
                                continue
                            pack_kwargs[key] = value
                        widget.pack(**pack_kwargs)
            else:
                if manager == "grid":
                    if widget.winfo_manager() == "grid":
                        widget.grid_remove()
                elif manager == "pack":
                    if widget.winfo_manager() == "pack":
                        widget.pack_forget()

    def _set_ui_mode(self) -> None:
        mode = self.ui_mode_var.get().strip().lower()
        self._toggle_advanced_widgets(show_advanced=(mode == "advanced"))
        self._refresh_config_field_visibility()

    def _go_to_tab(self, tab_key: str) -> None:
        tab = self._tab_lookup.get(tab_key)
        if tab is not None and str(tab) in self.notebook.tabs():
            self.notebook.select(tab)

    def _build_doctor_tab(self) -> None:
        controls = ttk.Frame(self.doctor_tab)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Run Doctor", command=self._run_doctor).pack(side=tk.LEFT)
        self._make_help_button(
            controls,
            title="Run Doctor",
            details=(
                "Runs environment checks for Python dependencies, camera stack availability, "
                "data-root write access, and key config sanity checks."
            ),
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(
            controls,
            text="Fix Venv Package Visibility",
            command=self._doctor_fix_venv_package_visibility,
        ).pack(side=tk.LEFT, padx=8)
        self._make_help_button(
            controls,
            title="Fix Venv Package Visibility",
            details=(
                "Attempts to make apt-installed Python camera packages (for example picamera2) "
                "visible to this BumbleBox virtual environment."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        self.doctor_output = self._create_results_section(
            self.doctor_tab,
            title="The Doctor's Results",
            text_height=12,
            default_visible=False,
            auto_hide_when_empty=True,
            show_status=False,
            auto_height=True,
            min_text_lines=6,
            max_text_lines=18,
        )

    def _build_storage_tab(self) -> None:
        top = ttk.Frame(self.storage_tab)
        top.pack(fill=tk.BOTH, expand=False)

        self.storage_mount_point_var = tk.StringVar(value=str(self._recommended_data_root()))
        self.storage_device_var = tk.StringVar(value="Auto (recommended)")
        self._storage_device_display_to_path: dict[str, str] = {}
        default_storage_path = str(self._recommended_data_root())

        settings = ttk.LabelFrame(top, text="Storage Configuration", padding=10)
        settings.pack(fill=tk.X)

        self._grid_help_label(
            settings,
            row=0,
            column=0,
            text="Step 1: Data folder path",
            help_title="Step 1: Data Folder Path",
            help_details=(
                "Choose which folder you want to mount your storage device to. "
                "Your data will be saved there, even though it is being written to the mounted storage device. "
                f"Default for this machine: {default_storage_path}"
            ),
        )
        ttk.Entry(settings, textvariable=self.storage_mount_point_var, width=48).grid(
            row=0, column=1, sticky="ew", padx=10, pady=4
        )
        ttk.Button(
            settings,
            text="Save Data Folder Path",
            command=self._save_storage_mount_point_to_config,
        ).grid(row=0, column=2, sticky="w", padx=(0, 6), pady=4)

        self._grid_help_label(
            settings,
            row=1,
            column=0,
            text="Step 2: Storage device",
            help_title="Step 2: Storage Device",
            help_details=(
                "Choose Auto to let BumbleBox select a detected partition, or pick a specific "
                "/dev/... device when you want explicit control. Then use the auto-mount button to set up "
                "the selected storage at the folder you chose in Step 1."
            ),
        )
        self.storage_device_combo = ttk.Combobox(
            settings,
            textvariable=self.storage_device_var,
            state="readonly",
            width=60,
        )
        self.storage_device_combo.grid(row=1, column=1, sticky="ew", padx=10, pady=4)
        device_actions = ttk.Frame(settings)
        device_actions.grid(row=1, column=2, sticky="w", pady=4)
        ttk.Button(
            device_actions,
            text="Refresh Devices",
            command=self._refresh_storage_device_choices,
        ).pack(side=tk.LEFT)

        actions = ttk.Frame(settings)
        actions.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Button(
            actions,
            text="Refresh Storage Status",
            command=self._refresh_storage_status,
        ).pack(side=tk.LEFT)
        self._make_help_button(
            actions,
            title="Refresh Storage Status",
            details="Re-checks mounted devices and reports where BumbleBox is currently writing data.",
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(
            actions,
            text="Set Up Auto-Mount Storage Device (stable long-term and across reboots)",
            command=self._setup_storage_auto_mount,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._make_help_button(
            actions,
            title="Set Up Auto-Mount Storage Device",
            details=(
                "Recommended Step 3. Creates or updates a stable UUID-based mount so the selected storage is mounted "
                "automatically at boot to your configured data folder. This is the recommended option for real "
                "experiments and long-term use."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        advanced_mount = ttk.LabelFrame(settings, text="Advanced Temporary Mount", padding=6)
        advanced_mount.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Label(
            advanced_mount,
            text=(
                "Optional advanced path: mount the selected storage device for this session only, "
                "without creating a persistent /etc/fstab entry."
            ),
            wraplength=860,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")
        mount_actions = ttk.Frame(advanced_mount)
        mount_actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(
            mount_actions,
            text="Mount Storage Device Once (temporary)",
            command=self._mount_storage_device,
        ).pack(side=tk.LEFT)
        self._make_help_button(
            mount_actions,
            title="Mount Storage Device Once (temporary)",
            details=(
                "Mounts the selected storage device at the chosen data folder for this session only. "
                "This is mainly for short tests or troubleshooting. For normal experiments, use the stable "
                "auto-mount option above instead."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))
        advanced_mount.columnconfigure(0, weight=1)
        self._register_advanced_widget(advanced_mount)

        info = (
            "Workflow:\n"
            "1. Choose the mount point.\n"
            "2. Choose the storage device.\n"
            "3. Click Set Up Auto-Mount Storage Device for the recommended stable setup used during experiments.\n"
            "Advanced only: use Mount Storage Device Once (temporary) for a one-session mount without persistent setup."
        )
        ttk.Label(settings, text=info, wraplength=860, justify=tk.LEFT).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        settings.columnconfigure(1, weight=1)

        self.storage_output = self._create_results_section(
            self.storage_tab,
            title="Storage Results",
            text_height=4,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=3,
            max_text_lines=9,
        )
        self._load_storage_mount_point_from_config()
        self._refresh_storage_device_choices()
        self._refresh_storage_status()

    def _build_camera_setup_tab(self) -> None:
        top = ttk.Frame(self.camera_setup_tab)
        top.pack(fill=tk.X)

        self.camera_preview_seconds_var = tk.StringVar(value="20")
        self.camera_preview_width_var = tk.StringVar(value="")
        self.camera_preview_height_var = tk.StringVar(value="")

        self.camera_test_seconds_var = tk.StringVar(value="20")
        self.camera_test_display_width_var = tk.StringVar(value="1280")
        self.camera_test_dictionary_var = tk.StringVar(value=self._current_config_tag_dictionary())
        self.camera_test_box_preset_var = tk.StringVar(value="auto")
        self.camera_test_show_rejected_var = tk.BooleanVar(value=False)
        self.camera_test_no_clahe_var = tk.BooleanVar(value=False)

        preview = ttk.LabelFrame(top, text="Step A: Camera Preview", padding=8)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(
            preview,
            text=(
                "Use this to verify camera focus, exposure, and framing before recording. "
                "Preview opens in a separate window."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._grid_help_label(
            preview,
            row=1,
            column=0,
            text="Duration (seconds)",
            help_title="Preview Duration",
            help_details="How long camera preview runs before closing automatically.",
        )
        ttk.Entry(preview, textvariable=self.camera_preview_seconds_var, width=8).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            preview,
            row=2,
            column=0,
            text="Preview backend",
            help_title="Preview Backend",
            help_details="Preview is fixed to QT for stability in the GUI workflow.",
        )
        ttk.Label(preview, text="QT (fixed)").grid(row=2, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            preview,
            row=3,
            column=0,
            text="Width override (optional)",
            help_title="Preview Width Override",
            help_details="Optional temporary width override for preview/testing only.",
        )
        ttk.Entry(preview, textvariable=self.camera_preview_width_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            preview,
            row=4,
            column=0,
            text="Height override (optional)",
            help_title="Preview Height Override",
            help_details="Optional temporary height override for preview/testing only.",
        )
        ttk.Entry(preview, textvariable=self.camera_preview_height_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=3)
        ttk.Button(preview, text="Run Camera Preview", command=self._run_camera_preview_setup).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        tracking = ttk.LabelFrame(top, text="Step B: Live Tag Tracking Test", padding=8)
        tracking.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            tracking,
            text=(
                "Shows live detections and summarizes detection rate. "
                "Press ESC in the OpenCV window to stop early."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._grid_help_label(
            tracking,
            row=1,
            column=0,
            text="Duration (seconds)",
            help_title="Tracking Test Duration",
            help_details="How long to run live tag detection before summarizing results.",
        )
        ttk.Entry(tracking, textvariable=self.camera_test_seconds_var, width=8).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            tracking,
            row=2,
            column=0,
            text="Display width",
            help_title="Display Width",
            help_details="Resizes preview display for readability; does not change sensor capture resolution.",
        )
        ttk.Entry(tracking, textvariable=self.camera_test_display_width_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            tracking,
            row=3,
            column=0,
            text="Tag dictionary",
            help_title="ArUco Dictionary Override",
            help_details=(
                "Dictionary used for this live tracking test. It starts from tracking.tag_dictionary in config; "
                "change it here for testing, or save the config editor field to make it the normal run setting."
            ),
        )
        ttk.Combobox(
            tracking,
            textvariable=self.camera_test_dictionary_var,
            values=ARUCO_4X4_DICTIONARY_OPTIONS,
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=3)
        self._grid_help_label(
            tracking,
            row=4,
            column=0,
            text="Box preset",
            help_title="Box Preset",
            help_details="Applies preset ArUco tuning profiles matched to common enclosure/camera setups.",
        )
        ttk.Combobox(
            tracking,
            textvariable=self.camera_test_box_preset_var,
            values=["auto", "custom", "koppert", "none"],
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(
            tracking,
            text="Show rejected marker candidates",
            variable=self.camera_test_show_rejected_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._make_help_button(
            tracking,
            title="Show Rejected Marker Candidates",
            details="Draws candidate quads that failed marker decoding to help diagnose threshold/perimeter settings.",
        ).grid(row=5, column=2, sticky="w", padx=(6, 0))
        ttk.Checkbutton(
            tracking,
            text="Disable CLAHE pre-processing",
            variable=self.camera_test_no_clahe_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._make_help_button(
            tracking,
            title="Disable CLAHE",
            details=(
                "CLAHE can improve contrast for small tags in uneven lighting. "
                "Disable this only when it appears to harm detection quality."
            ),
        ).grid(row=6, column=2, sticky="w", padx=(6, 0))
        actions = ttk.Frame(tracking)
        actions.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(actions, text="Run Live Tracking Test", command=self._run_camera_tracking_test_setup).pack(side=tk.LEFT)
        ttk.Button(actions, text="Run Full Setup Check (A then B)", command=self._run_full_camera_setup_check).pack(
            side=tk.LEFT, padx=8
        )

        thermal = ttk.LabelFrame(top, text="Step C: Thermal Camera Check", padding=8)
        thermal.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(
            thermal,
            text=(
                "Checks the attached PureThermal/Lepton board separately from the HQ camera, verifies a stable "
                "device path, and confirms raw Y16 access for radiometric-style capture."
            ),
            wraplength=860,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(
            thermal,
            text=(
                "If the check fully passes, 'Apply Detected Thermal Settings' writes the recommended stable path "
                "and Y16 settings into your active config."
            ),
            wraplength=860,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        thermal_actions = ttk.Frame(thermal)
        thermal_actions.grid(row=2, column=0, columnspan=3, sticky="w")
        self.thermal_check_button = ttk.Button(
            thermal_actions,
            text="Run Thermal Check",
            command=self._run_thermal_check_setup,
        )
        self.thermal_check_button.pack(side=tk.LEFT)
        self.thermal_apply_button = ttk.Button(
            thermal_actions,
            text="Apply Detected Thermal Settings",
            command=self._apply_detected_thermal_settings_setup,
            state=tk.DISABLED,
        )
        self.thermal_apply_button.pack(side=tk.LEFT, padx=8)
        self._make_help_button(
            thermal_actions,
            title="Thermal Camera Check",
            details=(
                "A fully passing thermal check must identify a stable /dev/v4l/by-id path and successfully read an "
                "explicit Y16 frame with a usable raw 16-bit layout. Only then should the thermal settings be applied."
            ),
        ).pack(side=tk.LEFT)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.camera_setup_output = self._create_results_section(
            self.camera_setup_tab,
            title="Camera Setup Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=14,
        )

    def _build_roadmap_tab(self) -> None:
        controls = ttk.Frame(self.roadmap_tab)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Refresh Roadmap", command=self._refresh_roadmap).pack(side=tk.LEFT)
        self.roadmap_summary_var = tk.StringVar(value="No roadmap loaded yet.")
        ttk.Label(controls, textvariable=self.roadmap_summary_var).pack(side=tk.LEFT, padx=(10, 0))

        legend = ttk.Frame(self.roadmap_tab)
        legend.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(legend, text="Click ? for details.").pack(side=tk.LEFT)

        paging = ttk.Frame(self.roadmap_tab)
        paging.pack(fill=tk.X, pady=(0, 6))
        self.roadmap_prev_btn = ttk.Button(
            paging,
            text="Previous Set",
            command=lambda: self._change_roadmap_page(-1),
        )
        self.roadmap_prev_btn.pack(side=tk.LEFT)
        self.roadmap_next_btn = ttk.Button(
            paging,
            text="Next Set",
            command=lambda: self._change_roadmap_page(1),
        )
        self.roadmap_next_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.roadmap_page_var = tk.StringVar(value="Set 1/1")
        ttk.Label(paging, textvariable=self.roadmap_page_var).pack(side=tk.LEFT, padx=(10, 0))

        self.roadmap_steps_frame = ttk.Frame(self.roadmap_tab)
        self.roadmap_steps_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self._roadmap_items: list[tuple[str, str]] = []
        self._roadmap_page = 0
        self._roadmap_page_count = 1
        self._roadmap_step_labels: list[ttk.Label] = []
        self._refresh_roadmap(select_tab=False)

    def _change_roadmap_page(self, delta: int) -> None:
        if not getattr(self, "_roadmap_items", None):
            return
        page_count = max(1, int(getattr(self, "_roadmap_page_count", 1)))
        if page_count <= 1:
            return
        current = int(getattr(self, "_roadmap_page", 0))
        new_page = min(page_count - 1, max(0, current + delta))
        if new_page == current:
            return
        self._roadmap_page = new_page
        self._render_roadmap_steps(self._roadmap_items)

    def _roadmap_wraplength(self) -> int:
        panel = getattr(self, "roadmap_steps_frame", None)
        if panel is None:
            return 760
        width = int(panel.winfo_width()) if panel.winfo_width() > 0 else 860
        return max(380, width - 220)

    def _refresh_roadmap_label_wraplength(self) -> None:
        labels = getattr(self, "_roadmap_step_labels", [])
        if not labels:
            return
        wrap = self._roadmap_wraplength()
        for label in labels:
            try:
                label.configure(wraplength=wrap)
            except Exception:
                continue

    def _roadmap_help_details(self, state: str, text: str) -> str:
        lower = text.lower()
        where = "The tab named in the step description."
        steps = [
            "Open the matching workflow from Home.",
            "Open the tab named in this step.",
            "Run the action in this step.",
        ]
        done_check = "You should see a success/result message in that tab."

        if "config file" in lower:
            where = "BumbleBox Setup -> Config Editor"
            steps = [
                "Open Config Editor.",
                "If the button says 'Config Missing! Create Config', click it.",
                "Click 'Load From File' then 'Validate'.",
                "Click 'Save Config' if needed.",
            ]
            done_check = "Config is present and validation reports success."
        elif "check camera" in lower or "camera setup" in lower:
            where = "BumbleBox Setup -> Camera Setup"
            steps = [
                "Click 'Run Camera Preview'.",
                "Check focus/framing in the preview window.",
                "Click 'Run Live Tracking Test'.",
            ]
            done_check = "Both preview and tracking test complete without errors."
        elif "calibration" in lower or "px/cm" in lower:
            where = "BumbleBox Setup -> Calibration"
            steps = [
                "Use 'Manual Scale (Recommended)'.",
                "Enter two points on the same plane and real distance in cm.",
                "Use a larger baseline (about 5 to 15 cm) for stability.",
                "Run calibration.",
            ]
            done_check = "Calibration results show updated px/cm and no error dialog."
        elif "pipeline mode" in lower:
            where = "BumbleBox Setup -> Config Editor"
            steps = [
                "Find 'Pipeline mode'.",
                "Pick the mode that matches your experiment.",
                "Save and validate config.",
            ]
            done_check = "Chosen mode is saved and config validates successfully."
        elif "deferred tracking" in lower:
            where = "BumbleBox Setup -> Config Editor"
            steps = [
                "Find 'Deferred tracking'.",
                "Turn it ON if recording uptime is your priority.",
                "Save and validate config.",
            ]
            done_check = "Deferred tracking setting is saved as intended."
        elif "mp4" in lower or "mjpeg" in lower:
            where = "BumbleBox Setup -> Config Editor"
            steps = [
                "Find 'Codec' in camera settings.",
                "Choose MP4 when MP4 framerate reporting is required.",
                "Find 'FPS report each recording' in runtime settings and keep it enabled.",
                "Save and validate config.",
            ]
            done_check = "Codec and FPS reporting settings are saved."
        elif "fps" in lower:
            where = "BumbleBox Setup -> FPS Report"
            steps = [
                "Run a single FPS report on a recording.",
                "Run FPS sweep to compare target vs real FPS.",
                "Review drift and limits before long runs.",
            ]
            done_check = "FPS results are visible and acceptable for your plan."
        elif "aruco" in lower or "tracking optimization" in lower:
            where = "BumbleBox Setup -> Tracking Optimization"
            steps = [
                "Set input path and profile.",
                "Set execution target for Pi-safe or desktop mode.",
                "Run optimization and review top candidates.",
                "Apply best parameters to config if needed.",
            ]
            done_check = "Optimization results are saved and best params are available."
        elif "schedule check" in lower or "timing margins" in lower:
            where = "Schedule and Run -> Schedule Check"
            steps = [
                "Set benchmark input (optional but recommended).",
                "Set sample frames and optional assumed RAM.",
                "Click 'Run Schedule Check'.",
            ]
            done_check = "Report shows no critical failures for your schedule."
        elif "nest label" in lower:
            where = "Nest Labeling workflow"
            steps = [
                "Set image folder.",
                "Click 'Check Environment'.",
                "Fix any missing dependency warnings.",
                "Click 'Launch Nest Labeling'.",
            ]
            done_check = "Labeling tool launches and can open your images."
        elif "fleet" in lower or "queen" in lower or "worker" in lower:
            where = "Fleet Setup workflow"
            steps = [
                "Set role and identity settings.",
                "Enroll worker boxes.",
                "Run fleet status and latest-status checks.",
            ]
            done_check = "Workers appear healthy/reachable in fleet status."
        elif "systemd" in lower or "scheduled runs" in lower:
            where = "Schedule and Run -> Schedule and Run"
            steps = [
                "Generate systemd units.",
                "Use 'Systemd Install' and 'Systemd Enable'.",
                "Use 'Systemd Status' to verify.",
            ]
            done_check = "Systemd status reports active/expected timers."
        elif "export-bundle" in lower or "bundle" in lower:
            where = "Schedule and Run -> Schedule and Run"
            steps = [
                "Refresh run history and select a run.",
                "Set export options.",
                "Click 'Export Selected Run Bundle'.",
            ]
            done_check = "Bundle export reports output path with no errors."

        status_line = {
            "DONE": "Status: completed based on current config/environment checks.",
            "TODO": "Status: not completed yet.",
            "OPTIONAL_DONE": "Status: completed optional step.",
            "OPTIONAL_TODO": "Status: optional step not completed yet.",
            "DISABLED": "Status: currently disabled for this codec/config choice.",
            "OPTIONAL": "Status: optional step.",
            "INFO": "Status: optional informational step.",
        }.get(state.upper(), f"Status: {state}")

        numbered_steps = "\n".join(f"{idx}. {item}" for idx, item in enumerate(steps, start=1))
        return (
            f"{status_line}\n\n"
            f"Step detail:\n{text}\n\n"
            f"Where to go:\n{where}\n\n"
            f"What to do:\n{numbered_steps}\n\n"
            f"How to tell it worked:\n{done_check}"
        )

    def _open_roadmap_help_dialog(self, title: str, details: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.configure(bg=self._palette["panel_bg"])
        dialog.geometry("760x520")
        dialog.minsize(560, 380)

        container = ttk.Frame(dialog, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text=title, font=self._header_title_font).pack(anchor=tk.W)
        ttk.Label(
            container,
            text="Step guidance",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 8))

        body = ttk.Frame(container)
        body.pack(fill=tk.BOTH, expand=True)
        help_text = tk.Text(body, wrap=tk.WORD)
        self._style_output_text(help_text)
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        help_scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=help_text.yview)
        help_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        help_text.configure(yscrollcommand=help_scroll.set)
        help_text.insert("1.0", details)
        help_text.configure(state=tk.DISABLED)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        try:
            x = self.winfo_rootx() + max(0, (self.winfo_width() - dialog.winfo_width()) // 2)
            y = self.winfo_rooty() + max(0, (self.winfo_height() - dialog.winfo_height()) // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        dialog.grab_set()
        dialog.focus_set()

    def _show_roadmap_step_help(self, state: str, text: str) -> None:
        details = self._roadmap_help_details(state, text)
        self._open_roadmap_help_dialog("Roadmap Task", details)

    def _split_help_summary_details(self, details: str) -> tuple[str, str]:
        text = str(details or "").strip()
        if not text:
            return ("No additional information is available for this item.", "")

        flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
        summary = flat
        sentence_end = -1
        for idx, ch in enumerate(flat):
            if ch in ".!?":
                if idx >= 24:
                    sentence_end = idx
                    break
        if sentence_end >= 0:
            summary = flat[: sentence_end + 1].strip()
        elif len(flat) > 170:
            summary = flat[:167].rstrip() + "..."

        if len(flat) <= len(summary) + 4:
            return summary, ""
        return summary, text

    def _show_help_dialog(self, title: str, details: str) -> None:
        summary, full_details = self._split_help_summary_details(details)
        has_more = bool(full_details)

        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.configure(bg=self._palette["panel_bg"])
        dialog.geometry("620x250")
        dialog.minsize(520, 220)

        container = ttk.Frame(dialog, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        ttk.Label(container, text=title, font=self._header_title_font).pack(anchor=tk.W)
        ttk.Label(container, text=summary, justify=tk.LEFT, wraplength=580).pack(
            anchor=tk.W, fill=tk.X, pady=(6, 10)
        )

        details_frame = ttk.Frame(container)
        details_text = tk.Text(details_frame, wrap=tk.WORD, height=8)
        self._style_output_text(details_text)
        details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=details_text.yview)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        details_text.configure(yscrollcommand=details_scroll.set)
        details_text.insert("1.0", full_details if full_details else summary)
        details_text.configure(state=tk.DISABLED)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X)

        shown = {"value": False}

        def _toggle_more() -> None:
            if not has_more:
                return
            if shown["value"]:
                details_frame.pack_forget()
                toggle_btn.configure(text="Show More")
                dialog.geometry("620x250")
                shown["value"] = False
            else:
                details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
                toggle_btn.configure(text="Hide Details")
                dialog.geometry("720x500")
                shown["value"] = True

        if has_more:
            toggle_btn = ttk.Button(footer, text="Show More", command=_toggle_more)
            toggle_btn.pack(side=tk.LEFT)

        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        try:
            x = self.winfo_rootx() + max(0, (self.winfo_width() - dialog.winfo_width()) // 2)
            y = self.winfo_rooty() + max(0, (self.winfo_height() - dialog.winfo_height()) // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        dialog.grab_set()
        dialog.focus_set()

    def _make_help_button(self, parent: tk.Widget, *, title: str, details: str) -> ttk.Button:
        return ttk.Button(
            parent,
            text="?",
            width=1,
            command=lambda t=title, d=details: self._show_help_dialog(t, d),
        )

    def _pack_help_label(
        self,
        parent: tk.Widget,
        *,
        text: str,
        help_title: str,
        help_details: str,
        padx: tuple[int, int] = (0, 0),
    ) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=padx)
        ttk.Label(frame, text=text).pack(side=tk.LEFT)
        self._make_help_button(frame, title=help_title, details=help_details).pack(side=tk.LEFT, padx=(4, 0))
        return frame

    def _grid_help_label(
        self,
        parent: tk.Widget,
        *,
        row: int,
        column: int,
        text: str,
        help_title: str,
        help_details: str,
        sticky: str = "w",
        padx: tuple[int, int] = (0, 0),
        pady: int = 0,
        columnspan: int = 1,
    ) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady, columnspan=columnspan)
        ttk.Label(frame, text=text).pack(side=tk.LEFT)
        help_column = column + max(2, int(columnspan) + 1)
        self._make_help_button(
            parent,
            title=help_title,
            details=help_details,
        ).grid(row=row, column=help_column, sticky="e", padx=(8, 0), pady=pady)
        return frame

    def _render_roadmap_steps(self, items: list[tuple[str, str]]) -> None:
        self._roadmap_items = list(items)
        for child in self.roadmap_steps_frame.winfo_children():
            child.destroy()
        self._roadmap_step_labels = []

        indexed_items: list[tuple[str, str]] = []
        for state, text in items:
            normalized_state = state.upper().strip()
            indexed_items.append((normalized_state, text))

        if len(indexed_items) <= 1:
            pages = [indexed_items]
        else:
            split = (len(indexed_items) + 1) // 2
            pages = [indexed_items[:split], indexed_items[split:]]

        self._roadmap_page_count = max(1, len(pages))
        self._roadmap_page = min(
            self._roadmap_page_count - 1,
            max(0, int(getattr(self, "_roadmap_page", 0))),
        )
        current_page_items = pages[self._roadmap_page] if pages else []

        def _is_done_state(state: str) -> bool:
            return state in {"DONE", "OPTIONAL_DONE"}

        page_done = [item for item in current_page_items if _is_done_state(item[0])]
        page_not_done = [item for item in current_page_items if not _is_done_state(item[0])]
        display_items = page_done + page_not_done

        for normalized_state, text in display_items:
            row = ttk.Frame(self.roadmap_steps_frame, style="RoadmapCard.TFrame", padding=(8, 8))
            row.pack(fill=tk.X, pady=(0, 7))
            row.columnconfigure(1, weight=1)

            state_block = ttk.Frame(row, style="RoadmapCard.TFrame")
            state_block.grid(row=0, column=0, sticky="nw", padx=(0, 10))

            is_optional = normalized_state.startswith("OPTIONAL") or normalized_state == "INFO"
            is_done = _is_done_state(normalized_state)
            is_disabled = normalized_state == "DISABLED"
            if is_done:
                state_label = "☑ DONE"
                state_style = "RoadmapStateDone.TLabel" if not is_optional else "RoadmapStateOptionalDone.TLabel"
            elif is_disabled:
                state_label = "—"
                state_style = "RoadmapStateDisabled.TLabel"
            else:
                state_label = "☐ TO DO"
                state_style = "RoadmapStateTodo.TLabel" if not is_optional else "RoadmapStateOptionalTodo.TLabel"
            ttk.Label(state_block, text=state_label, style=state_style).pack(anchor=tk.W)
            if is_optional:
                ttk.Label(state_block, text="optional", style="RoadmapStateOptional.TLabel").pack(anchor=tk.W)

            text_label = ttk.Label(
                row,
                text=text,
                style=("RoadmapCardMutedText.TLabel" if is_disabled else "RoadmapCardText.TLabel"),
                justify=tk.LEFT,
                wraplength=self._roadmap_wraplength(),
            )
            text_label.grid(row=0, column=1, sticky="ew")
            self._roadmap_step_labels.append(text_label)
            ttk.Button(
                row,
                text="?",
                width=1,
                command=lambda s=normalized_state, t=text: self._show_roadmap_step_help(s, t),
            ).grid(row=0, column=2, sticky="ne", padx=(8, 0))

        total = len(items)
        done_count = sum(1 for state, _text in indexed_items if _is_done_state(state))
        optional_count = sum(1 for state, _text in indexed_items if state.startswith("OPTIONAL") or state == "INFO")
        actionable_remaining = sum(1 for state, _text in indexed_items if state == "TODO")
        disabled_count = sum(1 for state, _text in indexed_items if state == "DISABLED")
        self.roadmap_summary_var.set(
            f"Completed: {done_count}/{total}   Remaining actionable: {actionable_remaining}   Optional: {optional_count}   Disabled: {disabled_count}"
        )
        current_page = self._roadmap_page + 1
        self.roadmap_page_var.set(f"Set {current_page}/{self._roadmap_page_count}")
        self.roadmap_prev_btn.configure(
            state=(tk.NORMAL if self._roadmap_page_count > 1 and self._roadmap_page > 0 else tk.DISABLED)
        )
        self.roadmap_next_btn.configure(
            state=(
                tk.NORMAL
                if self._roadmap_page_count > 1 and self._roadmap_page < self._roadmap_page_count - 1
                else tk.DISABLED
            )
        )

    def _build_config_tab(self) -> None:
        container = ttk.Frame(self.config_tab)
        container.pack(fill=tk.BOTH, expand=True)

        path_row = ttk.Frame(container)
        path_row.pack(fill=tk.X, pady=(0, 4))

        source_frame = ttk.Frame(path_row)
        source_frame.pack(fill=tk.X, expand=False)
        ttk.Label(source_frame, text="Config path").grid(row=0, column=0, sticky="w")
        self.config_path_display = ttk.Entry(source_frame, textvariable=self.config_path_var, width=58, state="readonly")
        self.config_path_display.grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        self.config_action_button = ttk.Button(source_frame, text="", command=self._handle_config_path_action)
        self.config_action_button.grid(row=0, column=2, sticky="w")
        source_frame.columnconfigure(1, weight=1)

        action_row = ttk.Frame(container)
        action_row.pack(fill=tk.X, pady=(0, 8))

        top_buttons = ttk.Frame(action_row)
        top_buttons.pack(anchor="e")
        ttk.Button(top_buttons, text="Load From File", command=self._load_config_into_editor).pack(side=tk.LEFT)
        self._make_help_button(
            top_buttons,
            title="Load From File",
            details="Loads values from the current config file path into editable UI fields.",
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(top_buttons, text="Validate", command=self._validate_editor_config).pack(side=tk.LEFT, padx=6)
        self._make_help_button(
            top_buttons,
            title="Validate",
            details="Checks current editor values against BumbleBox config schema without writing to disk.",
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(top_buttons, text="Save Config", command=self._save_editor_config).pack(side=tk.LEFT)
        self._make_help_button(
            top_buttons,
            title="Save Config",
            details="Validates and writes current editor values to the selected config path.",
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(
            container,
            text=(
                "Path is read-only here. Use the button to switch config files or create a missing default file."
            ),
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 6))

        self.config_page_var = tk.StringVar(value="")
        config_page_nav = ttk.Frame(container)
        config_page_nav.pack(fill=tk.X, pady=(0, 6))
        self.config_prev_btn = ttk.Button(config_page_nav, text="Previous", command=self._config_prev_page)
        self.config_prev_btn.pack(side=tk.LEFT)
        ttk.Label(config_page_nav, textvariable=self.config_page_var).pack(side=tk.LEFT, padx=10)
        self.config_next_btn = ttk.Button(config_page_nav, text="Next", command=self._config_next_page)
        self.config_next_btn.pack(side=tk.LEFT)
        self._make_help_button(
            config_page_nav,
            title="Config Groups",
            details=(
                "Config fields are split into pages by topic to reduce clutter:\n"
                "1) System Basics\n"
                "2) Camera Configuration\n"
                "3) Recording, Tracking, and Scheduling"
            ),
        ).pack(side=tk.LEFT, padx=(8, 0))

        form_canvas = tk.Canvas(
            container,
            highlightthickness=0,
            bg=self._palette["panel_bg"],
            bd=0,
        )
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=form_canvas.yview)
        self.config_form_frame = ttk.Frame(form_canvas)
        self.config_form_frame.bind(
            "<Configure>",
            lambda _event: form_canvas.configure(scrollregion=form_canvas.bbox("all")),
        )
        self._config_form_canvas = form_canvas
        self._config_form_window = form_canvas.create_window((0, 0), window=self.config_form_frame, anchor="nw")
        form_canvas.bind("<Configure>", self._on_config_canvas_resize)
        form_canvas.configure(yscrollcommand=scrollbar.set)
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._render_config_fields()

        self.config_output = self._create_results_section(
            self.config_tab,
            title="Config Messages",
            text_height=7,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=3,
            max_text_lines=10,
            fill=tk.X,
            expand=False,
        )
        self._load_config_into_editor()
        self._refresh_config_path_controls()

    def _on_config_canvas_resize(self, event) -> None:
        canvas = getattr(self, "_config_form_canvas", None)
        window_id = getattr(self, "_config_form_window", None)
        if canvas is None or window_id is None:
            return
        canvas.itemconfigure(window_id, width=max(1, int(event.width)))

    def _config_group_specs(self):
        camera_models = camera_model_choices()
        camera_profiles = camera_profile_choices()
        return [
            (
                "basic",
                "System Basics",
                [
                    ("Colony ID", "system.colony_id", str, None, None),
                    ("Data root", "system.data_root", str, None, None),
                    ("Use local tracking index", "local_index.enabled", bool, None, None),
                    ("Local tracking index", "local_index.path", str, None, None),
                    ("Pi model", "system.pi_model", str, ["auto", "pi4", "pi5"], None),
                    ("Fleet role", "fleet.role", str, ["standalone", "queen", "worker"], None),
                    ("Unit prefix", "scheduling.unit_prefix", str, None, None),
                    ("Service user", "scheduling.service_user", str, None, None),
                    ("UI theme mode", "runtime.ui_theme_mode", str, ["dark", "light"], None),
                    ("Use mock camera", "runtime.use_mock_camera", bool, None, None),
                ],
            ),
            (
                "camera",
                "Camera Configuration",
                [
                    ("Camera profile", "camera.profile", str, camera_profiles, None),
                    ("Camera model", "camera.model", str, camera_models, None),
                    ("Codec", "camera.codec", str, ["mp4", "mjpeg"], None),
                    ("MP4 encoder", "camera.mp4_codec", str, self._mp4_codec_display_options(), None),
                    ("Width (px)", "camera.width", int, None, None),
                    ("Height (px)", "camera.height", int, None, None),
                    ("FPS target", "camera.fps_target", float, None, None),
                    ("Shutter (us)", "camera.shutter_us", int, None, None),
                    (
                        "Autofocus mode",
                        "camera.autofocus_mode",
                        str,
                        ["default", "manual", "auto", "continuous"],
                        None,
                    ),
                    ("Manual lens position", "camera.lens_position", str, None, None),
                    ("Lock focus after warmup", "camera.focus_lock_after_warmup", bool, None, None),
                    (
                        "Autofocus range",
                        "camera.autofocus_range",
                        str,
                        ["normal", "macro", "full"],
                        None,
                    ),
                    (
                        "Autofocus speed",
                        "camera.autofocus_speed",
                        str,
                        ["normal", "fast"],
                        None,
                    ),
                    (
                        "Fast focus before capture",
                        "camera.autofocus_preflight_enabled",
                        bool,
                        None,
                        None,
                    ),
                    (
                        "Focus stream width",
                        "camera.autofocus_preflight_width",
                        int,
                        None,
                        None,
                    ),
                    (
                        "Focus stream height",
                        "camera.autofocus_preflight_height",
                        int,
                        None,
                        None,
                    ),
                    (
                        "Focus timeout (s)",
                        "camera.autofocus_preflight_timeout_seconds",
                        float,
                        None,
                        None,
                    ),
                    (
                        "Stable focus frames",
                        "camera.autofocus_preflight_stable_frames",
                        int,
                        None,
                        None,
                    ),
                    ("IR lighting", "camera.infrared", bool, None, None),
                    ("Monochrome output", "camera.monochrome_output", bool, None, None),
                    ("Use thermal camera", "thermal.enabled", bool, None, None),
                    ("Preview window", "camera.preview_window", str, ["QT"], None),
                    ("Tuning file", "camera.tuning_file", str, None, None),
                ],
            ),
            (
                "pipeline",
                "Recording, Tracking, and Scheduling",
                [
                    ("Pipeline mode", "pipeline.mode", str, ["record_only", "track_only", "record_and_track", "mixed_schedule"], None),
                    ("Tracking source", "pipeline.tracking_source", str, ["ram", "video"], None),
                    ("Tag dictionary", "tracking.tag_dictionary", str, ARUCO_4X4_DICTIONARY_OPTIONS, None),
                    ("Deferred tracking", "pipeline.defer_tracking_until_after_recording", bool, None, None),
                    ("Parallel tracking", "pipeline.parallel_tracking", bool, None, None),
                    ("Behavior metrics", "pipeline.calculate_behavior_metrics", bool, None, None),
                    ("Recording seconds", "capture.recording_seconds", int, None, None),
                    ("Record interval (min)", "capture.record_interval_minutes", int, None, None),
                    ("Track interval (min)", "capture.track_interval_minutes", int, None, None),
                    ("Scheduler backend", "scheduling.backend", str, ["systemd", "cron"], None),
                    ("Scheduler scope", "scheduling.scope", str, ["system", "user"], None),
                    ("Save frame timestamps", "runtime.save_frame_timestamps", bool, None, None),
                    ("FPS report each recording", "runtime.fps_report_on_each_recording", bool, None, None),
                    ("Render tracked video", "runtime.render_tracking_video", bool, None, None),
                    ("Queen local pipeline", "fleet.queen_local_pipeline_enabled", bool, None, {"queen"}),
                    ("Queen media enabled", "fleet.queen_media_schedule.enabled", bool, None, {"queen"}),
                    ("Queen pull interval (min)", "fleet.queen_media_schedule.pull_interval_minutes", int, None, {"queen"}),
                    ("Queen track interval (min)", "fleet.queen_media_schedule.track_interval_minutes", int, None, {"queen"}),
                    ("Queen media max videos", "fleet.queen_media_schedule.max_videos_total", int, None, {"queen"}),
                    ("Queen media cooldown (min)", "fleet.queen_media_schedule.cooldown_minutes", int, None, {"queen"}),
                ],
            ),
        ]

    @staticmethod
    def _mp4_codec_display_options() -> list[str]:
        return [
            "H.264 (Recommended)",
            "MPEG-4 (Faster/Larger)",
            "H.265 (Smaller/Slower)",
        ]

    @staticmethod
    def _mp4_codec_display_to_value() -> dict[str, str]:
        return {
            "H.264 (Recommended)": "libx264",
            "MPEG-4 (Faster/Larger)": "mpeg4",
            "H.265 (Smaller/Slower)": "libx265",
        }

    @classmethod
    def _mp4_codec_value_to_display(cls) -> dict[str, str]:
        display_to_value = cls._mp4_codec_display_to_value()
        mapping = {value: display for display, value in display_to_value.items()}
        mapping.update(
            {
                "mp4v": "MPEG-4 (Faster/Larger)",
                "h264": "H.264 (Recommended)",
                "x264": "H.264 (Recommended)",
                "avc": "H.264 (Recommended)",
                "h265": "H.265 (Smaller/Slower)",
                "hevc": "H.265 (Smaller/Slower)",
                "x265": "H.265 (Smaller/Slower)",
            }
        )
        return mapping

    @staticmethod
    def _normalize_display_dictionary(value: str | None) -> str:
        text = str(value or "").strip().upper()
        if text.startswith("DICT_"):
            text = text[5:]
        if text in ARUCO_4X4_DICTIONARY_OPTIONS:
            return text
        return "4X4_50"

    def _current_config_tag_dictionary(self) -> str:
        editor_field = getattr(self, "config_fields", {}).get("tracking.tag_dictionary")
        if editor_field is not None:
            try:
                return self._normalize_display_dictionary(editor_field[0].get())
            except Exception:
                pass
        try:
            config, _ = self._load_config_or_defaults()
            tracking = config.get("tracking", {})
            if isinstance(tracking, dict):
                return self._normalize_display_dictionary(tracking.get("tag_dictionary"))
        except Exception:
            pass
        return "4X4_50"

    def _tracking_allowed_tag_ids_from_config(self, config: dict) -> set[int]:
        tracking = config.get("tracking", {}) if isinstance(config.get("tracking", {}), dict) else {}
        out: set[int] = set()
        raw_ids = tracking.get("allowed_tag_ids", [])
        if isinstance(raw_ids, list):
            for raw_id in raw_ids:
                try:
                    out.add(int(raw_id))
                except (TypeError, ValueError):
                    continue
        raw_path = str(tracking.get("allowed_tag_ids_path") or "").strip()
        if raw_path:
            from .posthoc_tracking import load_tag_ids

            out.update(load_tag_ids(raw_path))
        return out

    def _sync_dictionary_selectors_from_config(self) -> None:
        dictionary = self._current_config_tag_dictionary()
        for attr_name in ("opt_dictionary_var", "camera_test_dictionary_var"):
            variable = getattr(self, attr_name, None)
            if variable is not None:
                variable.set(dictionary)

    def _config_widget_width(self, key: str, value_type: type, choices: list[str] | None) -> int:
        if choices:
            longest = max((len(str(choice)) for choice in choices), default=12)
            return max(10, min(24, longest + 3))

        if key in {"system.data_root", "local_index.path", "camera.tuning_file"}:
            return 24
        if key in {"system.colony_id"}:
            return 10
        if key in {"scheduling.unit_prefix", "scheduling.service_user", "camera.model", "camera.profile"}:
            return 18
        if value_type in {int, float}:
            return 10
        return 20

    def _config_field_help_details(self, key: str) -> str:
        details = {
            "system.colony_id": "Short identifier used in output filenames and metadata.",
            "system.data_root": "Root folder where recordings and outputs are written.",
            "local_index.enabled": (
                "Keep a lightweight local copy of run summaries, tracking CSVs, and optimization results "
                "on this computer."
            ),
            "local_index.path": (
                "Local folder for the lightweight tracking index. Default: LocalTrackingIndex inside the BumbleBox repo. "
                "Generated index artifacts are ignored by Git, and large videos are intentionally not copied here."
            ),
            "system.pi_model": "Hardware target hint. Use auto unless you need to force Pi4/Pi5 assumptions.",
            "camera.profile": (
                "Optional camera preset. Use custom for manual fields, hq_reference for the known-good HQ test "
                "envelope, or owlsight_reference to test OwlSight at the same 4056x3040 @ 7 fps envelope."
            ),
            "camera.model": "Camera hardware model hint. Affects defaults and camera-specific assumptions.",
            "fleet.role": "Choose standalone, queen, or worker behavior mode.",
            "scheduling.unit_prefix": "Prefix for generated timer/service unit names in systemd.",
            "scheduling.service_user": "Linux account that runs scheduled jobs in system-scope systemd mode.",
            "runtime.ui_theme_mode": "Default GUI theme mode when app starts.",
            "runtime.use_mock_camera": "Use synthetic camera frames for testing without camera hardware.",
            "camera.codec": "Recording codec. MP4 is compact and convenient; MJPEG is larger but simple per-frame encoding.",
            "camera.mp4_codec": (
                "MP4 encoder used after RAM capture. "
                "H.264 is the recommended default; MPEG-4 is faster but makes larger files; "
                "H.265 makes smaller files but is slower to encode."
            ),
            "camera.width": "Capture width in pixels. Higher values increase detail and resource usage.",
            "camera.height": "Capture height in pixels. Higher values increase detail and resource usage.",
            "camera.fps_target": "Requested capture framerate. Real framerate can differ; verify with FPS Report.",
            "camera.shutter_us": "Exposure time in microseconds. Longer exposure can brighten image but increase motion blur.",
            "camera.autofocus_mode": (
                "Focus behavior: default leaves libcamera unchanged; continuous keeps refocusing; "
                "auto performs one focus scan after startup; manual uses the optional lens position."
            ),
            "camera.lens_position": (
                "Optional manual focus position in diopters. Leave blank unless autofocus mode is manual. "
                "Zero represents infinity; larger values focus closer."
            ),
            "camera.focus_lock_after_warmup": (
                "With auto or continuous focus, read the achieved lens position after camera warmup and "
                "hold that position during capture."
            ),
            "camera.autofocus_range": (
                "Normal covers typical distances, macro emphasizes close subjects, and full searches "
                "the entire supported focus range."
            ),
            "camera.autofocus_speed": (
                "Normal prioritizes smooth focus movement; fast requests quicker lens movement."
            ),
            "camera.autofocus_preflight_enabled": (
                "Focus using a smaller, faster stream before full-resolution capture, retain the best "
                "observed lens position, then hold it during recording. Enabled by the OwlSight profile."
            ),
            "camera.autofocus_preflight_width": (
                "Width of the temporary low-resolution stream used only for autofocus."
            ),
            "camera.autofocus_preflight_height": (
                "Height of the temporary low-resolution stream used only for autofocus."
            ),
            "camera.autofocus_preflight_timeout_seconds": (
                "Maximum time allowed for the initial autofocus scan. If libcamera reports failure, "
                "BumbleBox can still retain the lens position with the highest observed focus score."
            ),
            "camera.autofocus_preflight_stable_frames": (
                "Number of consecutive focused frames required before accepting libcamera's focused state."
            ),
            "camera.infrared": (
                "Use IR/NoIR sensor tuning when no manual tuning file is set. "
                "When enabled, BumbleBox auto-resolves the camera's noir tuning file "
                "(for example imx477_noir.json or imx708_noir.json) for preview, recording, and calibration. "
                "Do not enable this for stock OwlSight cameras, which have an IR-cut filter."
            ),
            "camera.monochrome_output": (
                "Force grayscale-looking RGB output for preview, recording, and calibration captures. "
                "This is useful for NoIR cameras under IR lighting when you want black-and-white output "
                "instead of the normal magenta/purple color cast."
            ),
            "thermal.enabled": (
                "Enable or disable the thermal camera for recording runs. "
                "Use Camera Setup Step C to detect and apply the correct PureThermal device path and Y16 settings."
            ),
            "camera.preview_window": "Preview backend used by camera preview (fixed to QT for stable GUI behavior).",
            "camera.tuning_file": (
                "Optional manual libcamera tuning JSON override. "
                "Leave this blank to let BumbleBox auto-select standard or IR/NoIR tuning from camera model + IR lighting."
            ),
            "pipeline.mode": "Main run mode: record only, track only, record+track, or mixed schedule lanes.",
            "pipeline.tracking_source": "Track from in-memory frames (ram) or saved video files (video).",
            "tracking.tag_dictionary": (
                "ArUco dictionary used for tag detection in recording, tracking, camera tests, and optimization. "
                "This must match the dictionary used to generate or print the tags."
            ),
            "pipeline.defer_tracking_until_after_recording": "When enabled, tracking runs after recording to reduce runtime contention.",
            "pipeline.parallel_tracking": "Allow concurrent tracking work. Faster on strong hardware, heavier on limited Pi resources.",
            "pipeline.calculate_behavior_metrics": "Enable legacy in-box behavior metrics during tracking pipeline.",
            "capture.recording_seconds": "Length of each recording chunk.",
            "capture.record_interval_minutes": "Time between recording starts for scheduled capture.",
            "capture.track_interval_minutes": "Time between scheduled tracking jobs.",
            "scheduling.backend": "Scheduler backend implementation (systemd recommended on Pi OS).",
            "scheduling.scope": "System scope runs regardless of user login; user scope runs per-user session.",
            "runtime.save_frame_timestamps": "Write frame timestamp sidecars for real-FPS and timing analysis.",
            "runtime.fps_report_on_each_recording": "Automatically emit FPS summary after each recording.",
            "runtime.render_tracking_video": (
                "After tracking finishes, render a tagged overlay video from the tracking CSV. "
                "When thermal recording is also enabled, BumbleBox will also render a tracked RGB+thermal side-by-side comparison."
            ),
            "fleet.queen_local_pipeline_enabled": "When false, queen acts as interface/orchestrator without running local bbox pipeline.",
            "fleet.queen_media_schedule.enabled": "Enable queen media pull/track schedule from workers.",
            "fleet.queen_media_schedule.pull_interval_minutes": "How often queen pulls latest worker videos.",
            "fleet.queen_media_schedule.track_interval_minutes": "How often queen runs tracking/visualization on pulled media.",
            "fleet.queen_media_schedule.max_videos_total": "Upper bound on retained pulled/tracked videos for queen workload control.",
            "fleet.queen_media_schedule.cooldown_minutes": "Minimum delay before reprocessing same worker media set.",
        }
        return details.get(
            key,
            (
                f"Config parameter: {key}\n\n"
                "Use default unless you have a specific experiment or hardware reason to change it."
            ),
        )

    def _set_entry_width(self, widget: tk.Widget, width: int) -> None:
        try:
            widget.configure(width=max(8, int(width)))
        except Exception:
            return

    def _bind_expandable_entry(self, widget: tk.Widget, *, compact_width: int, expanded_width: int) -> None:
        self._set_entry_width(widget, compact_width)
        widget.bind(
            "<FocusIn>",
            lambda _event, w=widget, expanded=expanded_width: self._set_entry_width(w, expanded),
            add="+",
        )
        widget.bind(
            "<FocusOut>",
            lambda _event, w=widget, compact=compact_width: self._set_entry_width(w, compact),
            add="+",
        )

    def _render_config_fields(self) -> None:
        group_defs = self._config_group_specs()
        self._config_field_rows: dict[str, ttk.Frame] = {}
        self._config_field_roles: dict[str, set[str] | None] = {}
        self._config_field_widgets: dict[str, tk.Widget] = {}
        self._config_group_frames: dict[str, ttk.LabelFrame] = {}
        self._config_group_bodies: dict[str, ttk.Frame] = {}
        self._config_group_order: list[str] = []
        self._config_group_titles: dict[str, str] = {}
        self._config_page_index = 0

        for group_id, group_title, _specs in group_defs:
            group_frame = ttk.LabelFrame(self.config_form_frame, text=group_title, padding=8)
            group_frame.grid(row=0, column=0, sticky="nw")
            group_frame.columnconfigure(0, weight=1)
            group_frame.grid_anchor("nw")
            group_body = ttk.Frame(group_frame)
            group_body.grid(row=0, column=0, sticky="ew")
            group_body.columnconfigure(0, weight=1)
            self._config_group_frames[group_id] = group_frame
            self._config_group_bodies[group_id] = group_body
            self._config_group_order.append(group_id)
            self._config_group_titles[group_id] = group_title

        for group_id, _group_title, specs in group_defs:
            group_frame = self._config_group_frames[group_id]
            group_body = self._config_group_bodies[group_id]
            row_index = 0
            for label, key, value_type, choices, roles in specs:
                row_frame = ttk.Frame(group_body)
                row_frame.grid(row=row_index, column=0, sticky="ew", pady=2)
                row_frame.columnconfigure(3, weight=1)
                ttk.Label(row_frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
                widget_width = self._config_widget_width(key, value_type, choices)

                if value_type is bool:
                    variable = tk.BooleanVar(value=False)
                    widget = ttk.Checkbutton(row_frame, variable=variable)
                    widget.grid(row=0, column=1, sticky="w", pady=1)
                elif choices:
                    variable = tk.StringVar(value=str(choices[0]))
                    widget = ttk.Combobox(
                        row_frame,
                        textvariable=variable,
                        values=choices,
                        state="readonly",
                        width=widget_width,
                    )
                    widget.grid(row=0, column=1, sticky="w", pady=1)
                else:
                    variable = tk.StringVar(value="")
                    widget = ttk.Entry(row_frame, textvariable=variable, width=widget_width)
                    widget.grid(row=0, column=1, sticky="w", pady=1)
                    if key in {"system.data_root", "camera.tuning_file"}:
                        self._bind_expandable_entry(
                            widget,
                            compact_width=widget_width,
                            expanded_width=56,
                        )

                if key == "camera.tuning_file":
                    ttk.Button(row_frame, text="Browse", command=self._browse_tuning_file).grid(
                        row=0, column=2, sticky="w", padx=(6, 0)
                    )
                if key == "tracking.tag_dictionary":
                    widget.bind(
                        "<<ComboboxSelected>>",
                        lambda _event: self._sync_dictionary_selectors_from_config(),
                        add="+",
                    )

                self._make_help_button(
                    row_frame,
                    title=label,
                    details=self._config_field_help_details(key),
                ).grid(row=0, column=4, sticky="e", padx=(8, 0), pady=1)

                self.config_fields[key] = (variable, value_type)
                self._config_field_widgets[key] = widget
                self._config_field_rows[key] = row_frame
                self._config_field_roles[key] = set(roles) if roles else None
                row_index += 1

            if group_id == "basic":
                ttk.Label(
                    group_body,
                    text="Unit prefix controls generated timer/service names (for example: bumblebox-v2-record.timer).",
                    justify=tk.LEFT,
                    wraplength=620,
                ).grid(row=row_index, column=0, sticky="w", pady=(8, 0))
                row_index += 1

            if group_id == "camera":
                self._camera_max_status_var = tk.StringVar(
                    value="Tip: reads connected camera modes first, then falls back to camera model defaults."
                )
                actions = ttk.Frame(group_body)
                actions.grid(row=row_index, column=0, sticky="w", pady=(8, 0))
                self._camera_actions_row = actions
                ttk.Button(
                    actions,
                    text="Apply Camera Profile",
                    command=self._apply_camera_profile_to_editor,
                ).pack(side=tk.LEFT)
                ttk.Button(
                    actions,
                    text="Use Max Resolution",
                    command=self._apply_camera_max_resolution,
                ).pack(side=tk.LEFT, padx=(8, 0))
                ttk.Label(actions, textvariable=self._camera_max_status_var).pack(side=tk.LEFT, padx=(8, 0))
                row_index += 1

            if group_id == "pipeline":
                self._pipeline_tracking_hint_var = tk.StringVar(value="")
                ttk.Label(
                    group_body,
                    textvariable=self._pipeline_tracking_hint_var,
                    justify=tk.LEFT,
                    wraplength=620,
                ).grid(row=row_index, column=0, sticky="w", pady=(8, 0))
                row_index += 1

        self.config_form_frame.columnconfigure(0, weight=1)
        self.config_form_frame.grid_anchor("nw")

        role_binding = getattr(self, "_config_role_trace_bound", False)
        role_field = self.config_fields.get("fleet.role")
        if (not role_binding) and role_field is not None:
            role_var = role_field[0]
            try:
                role_var.trace_add("write", lambda *_args: self._refresh_config_field_visibility())
                self._config_role_trace_bound = True
            except Exception:
                self._config_role_trace_bound = False

        pipeline_binding = getattr(self, "_pipeline_trace_bound", False)
        mode_field = self.config_fields.get("pipeline.mode")
        source_field = self.config_fields.get("pipeline.tracking_source")
        if (not pipeline_binding) and mode_field is not None and source_field is not None:
            try:
                mode_field[0].trace_add("write", lambda *_args: self._update_pipeline_tracking_hint())
                source_field[0].trace_add("write", lambda *_args: self._update_pipeline_tracking_hint())
                self._pipeline_trace_bound = True
            except Exception:
                self._pipeline_trace_bound = False

        codec_binding = getattr(self, "_config_codec_trace_bound", False)
        codec_field = self.config_fields.get("camera.codec")
        if (not codec_binding) and codec_field is not None:
            try:
                codec_field[0].trace_add("write", lambda *_args: self._refresh_config_field_visibility())
                self._config_codec_trace_bound = True
            except Exception:
                self._config_codec_trace_bound = False

        thermal_binding = getattr(self, "_config_thermal_trace_bound", False)
        thermal_field = self.config_fields.get("thermal.enabled")
        if (not thermal_binding) and thermal_field is not None:
            try:
                thermal_field[0].trace_add("write", lambda *_args: self._refresh_thermal_controls_state())
                self._config_thermal_trace_bound = True
            except Exception:
                self._config_thermal_trace_bound = False

        self._set_config_page(0)
        self._update_pipeline_tracking_hint()
        self._refresh_config_field_visibility()
        self._refresh_thermal_controls_state()

    def _set_config_page(self, page_index: int) -> None:
        order = getattr(self, "_config_group_order", [])
        if not order:
            return
        bounded = max(0, min(len(order) - 1, int(page_index)))
        self._config_page_index = bounded
        frames = getattr(self, "_config_group_frames", {})
        for idx, group_id in enumerate(order):
            frame = frames.get(group_id)
            if frame is None:
                continue
            if idx == bounded:
                frame.grid()
            else:
                frame.grid_remove()
        self._update_config_page_controls()

    def _config_prev_page(self) -> None:
        self._set_config_page(getattr(self, "_config_page_index", 0) - 1)

    def _config_next_page(self) -> None:
        self._set_config_page(getattr(self, "_config_page_index", 0) + 1)

    def _update_config_page_controls(self) -> None:
        order = getattr(self, "_config_group_order", [])
        if not order:
            return
        idx = max(0, min(len(order) - 1, int(getattr(self, "_config_page_index", 0))))
        group_id = order[idx]
        title = getattr(self, "_config_group_titles", {}).get(group_id, group_id)
        if hasattr(self, "config_page_var"):
            self.config_page_var.set(f"Group {idx + 1}/{len(order)}: {title}")
        if hasattr(self, "config_prev_btn"):
            self.config_prev_btn.configure(state=(tk.NORMAL if idx > 0 else tk.DISABLED))
        if hasattr(self, "config_next_btn"):
            self.config_next_btn.configure(state=(tk.NORMAL if idx < len(order) - 1 else tk.DISABLED))

    def _browse_tuning_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select camera tuning file",
            initialdir=str(Path(self.config_path_var.get()).expanduser().parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        field = self.config_fields.get("camera.tuning_file")
        if field is None:
            return
        field[0].set(selected)

    def _apply_camera_profile_to_editor(self) -> None:
        profile_field = self.config_fields.get("camera.profile")
        if profile_field is None:
            return

        profile_name = str(profile_field[0].get()).strip() or "custom"
        profile = get_camera_profile(profile_name)
        if profile is None:
            self._show_error("Camera profile", f"Unknown camera profile: {profile_name}")
            return

        if profile.key == "custom":
            message = "Camera profile is custom; no preset fields were changed."
            if hasattr(self, "_camera_max_status_var"):
                self._camera_max_status_var.set(message)
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, message)
            return

        changed: list[str] = []
        for key, value in profile.camera_values.items():
            field = self.config_fields.get(f"camera.{key}")
            if field is None:
                continue
            variable, value_type = field
            if value_type is bool:
                variable.set(bool(value))
            elif value is None:
                variable.set("")
            else:
                variable.set(str(value))
            changed.append(f"camera.{key} = {value!r}")

        profile_field[0].set(profile.key)
        status = f"Applied camera profile: {profile.key} ({profile.description})"
        if hasattr(self, "_camera_max_status_var"):
            self._camera_max_status_var.set(status)
        self.config_output.delete("1.0", tk.END)
        self.config_output.insert(tk.END, status)
        if changed:
            self.config_output.insert(tk.END, "\n\nUpdated fields:\n- " + "\n- ".join(changed))
        model_info = get_camera_model_info(profile.camera_values.get("model"))
        if model_info is not None and not model_info.supports_infrared:
            self.config_output.insert(
                tk.END,
                "\n\nNote: this camera profile is marked as not IR-capable; use visible illumination.",
            )

    def _infer_camera_model_max_resolution(self, model: str) -> tuple[int, int] | None:
        model_info = get_camera_model_info(model)
        return model_info.max_resolution if model_info else None

    def _coerce_size_tuple(self, value) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                width = int(value[0])
                height = int(value[1])
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                return None
        return None

    def _probe_connected_camera_max_resolution(self) -> tuple[int, int] | None:
        try:
            from picamera2 import Picamera2
        except Exception:
            return None

        picam2 = None
        sizes: list[tuple[int, int]] = []
        try:
            picam2 = Picamera2()
            sensor_modes = getattr(picam2, "sensor_modes", None)
            if isinstance(sensor_modes, (list, tuple)):
                for mode in sensor_modes:
                    size = None
                    if isinstance(mode, dict):
                        size = mode.get("size") or mode.get("resolution") or mode.get("output_size")
                    else:
                        size = mode
                    maybe_size = self._coerce_size_tuple(size)
                    if maybe_size is not None:
                        sizes.append(maybe_size)

            properties = getattr(picam2, "camera_properties", None)
            if isinstance(properties, dict):
                for key in ("PixelArraySize", "PixelArray", "SensorResolution"):
                    maybe_size = self._coerce_size_tuple(properties.get(key))
                    if maybe_size is not None:
                        sizes.append(maybe_size)
        except Exception:
            return None
        finally:
            if picam2 is not None:
                try:
                    picam2.close()
                except Exception:
                    pass

        if not sizes:
            return None
        return max(sizes, key=lambda item: item[0] * item[1])

    def _apply_camera_max_resolution(self) -> None:
        width_field = self.config_fields.get("camera.width")
        height_field = self.config_fields.get("camera.height")
        model_field = self.config_fields.get("camera.model")
        if width_field is None or height_field is None:
            return

        detected = self._probe_connected_camera_max_resolution()
        source = "connected camera"
        if detected is None:
            model_name = str(model_field[0].get()).strip().lower() if model_field else "auto"
            detected = self._infer_camera_model_max_resolution(model_name)
            source = f"camera model preset ({model_name})"

        if detected is None:
            message = "Could not detect max resolution. Connect a camera or select a specific camera model."
            if hasattr(self, "_camera_max_status_var"):
                self._camera_max_status_var.set(message)
            self._show_error("Camera resolution detection failed", message)
            return

        width, height = detected
        width_field[0].set(str(width))
        height_field[0].set(str(height))
        if hasattr(self, "_camera_max_status_var"):
            self._camera_max_status_var.set(f"Set width/height to {width}x{height} from {source}.")
        self.config_output.delete("1.0", tk.END)
        self.config_output.insert(tk.END, f"Applied max resolution: {width}x{height}\nSource: {source}")

    def _update_pipeline_tracking_hint(self) -> None:
        hint_var = getattr(self, "_pipeline_tracking_hint_var", None)
        if hint_var is None:
            return
        mode_field = self.config_fields.get("pipeline.mode")
        source_field = self.config_fields.get("pipeline.tracking_source")
        mode = str(mode_field[0].get()).strip().lower() if mode_field else ""
        source = str(source_field[0].get()).strip().lower() if source_field else ""

        if mode == "record_and_track" and source == "ram":
            hint = (
                "record_and_track + tracking_source=ram keeps frame data in memory during recording. "
                "This is fast for short runs, but RAM-heavy for long recordings or high FPS. "
                "Choose tracking_source=video for lower RAM pressure and longer reliable recording windows."
            )
        elif mode == "record_and_track" and source == "video":
            hint = (
                "record_and_track + tracking_source=video tracks from saved video. "
                "This is usually safer for long-duration recording because RAM pressure is lower."
            )
        elif mode == "track_only":
            hint = "track_only ignores recording cadence and focuses on scheduled tracking jobs."
        elif mode == "record_only":
            hint = "record_only captures video without tag-tracking jobs."
        else:
            hint = (
                "mixed_schedule can run separate record and track lanes. "
                "Use Schedule Check and Schedule and Run tabs to validate timing."
            )
        hint_var.set(hint)

    def _refresh_config_field_visibility(self) -> None:
        rows = getattr(self, "_config_field_rows", {})
        if not rows:
            return
        roles_cfg = getattr(self, "_config_field_roles", {})
        fleet_role = "standalone"
        role_field = self.config_fields.get("fleet.role")
        if role_field is not None:
            try:
                fleet_role = str(role_field[0].get()).strip().lower() or "standalone"
            except Exception:
                fleet_role = "standalone"
        codec_value = "mp4"
        codec_field = self.config_fields.get("camera.codec")
        if codec_field is not None:
            try:
                codec_value = str(codec_field[0].get()).strip().lower() or "mp4"
            except Exception:
                codec_value = "mp4"
        ui_mode = str(self.ui_mode_var.get()).strip().lower()

        for key, row in rows.items():
            allowed_roles = roles_cfg.get(key)
            should_show = allowed_roles is None or fleet_role in allowed_roles
            if key == "camera.mp4_codec":
                should_show = should_show and codec_value == "mp4" and ui_mode == "advanced"
            if key in {
                "camera.preview_window",
                "camera.tuning_file",
                "camera.lens_position",
                "camera.focus_lock_after_warmup",
                "camera.autofocus_range",
                "camera.autofocus_speed",
                "camera.autofocus_preflight_width",
                "camera.autofocus_preflight_height",
                "camera.autofocus_preflight_timeout_seconds",
                "camera.autofocus_preflight_stable_frames",
                "scheduling.backend",
                "scheduling.scope",
                "scheduling.unit_prefix",
                "scheduling.service_user",
            }:
                should_show = should_show and ui_mode == "advanced"
            if should_show:
                row.grid()
            else:
                row.grid_remove()

        camera_actions_row = getattr(self, "_camera_actions_row", None)
        if camera_actions_row is not None:
            if ui_mode == "advanced":
                camera_actions_row.grid()
            else:
                camera_actions_row.grid_remove()
        self._update_pipeline_tracking_hint()

    def _thermal_enabled_from_editor(self) -> bool:
        field = self.config_fields.get("thermal.enabled")
        if field is None:
            return False
        try:
            return bool(field[0].get())
        except Exception:
            return False

    def _refresh_thermal_controls_state(self) -> None:
        thermal_enabled = self._thermal_enabled_from_editor()
        thermal_check_button = getattr(self, "thermal_check_button", None)
        thermal_apply_button = getattr(self, "thermal_apply_button", None)

        if thermal_check_button is not None:
            thermal_check_button.configure(state=tk.NORMAL if thermal_enabled else tk.DISABLED)

        if thermal_apply_button is not None:
            apply_enabled = thermal_enabled and self._thermal_check_can_apply()
            thermal_apply_button.configure(state=tk.NORMAL if apply_enabled else tk.DISABLED)

    def _build_fps_tab(self) -> None:
        top = ttk.Frame(self.fps_tab)
        top.pack(fill=tk.X)

        self.video_path_var = tk.StringVar()
        self.timestamps_path_var = tk.StringVar()
        self.recording_seconds_var = tk.StringVar()
        self.fps_live_seconds_var = tk.StringVar(value="10.0")
        self.fps_live_status_var = tk.StringVar(value="Idle")
        self.fps_live_reset_status_var = tk.StringVar(value="")
        self.fps_sweep_values_var = tk.StringVar(value="")
        self.fps_sweep_start_var = tk.StringVar(value="2.0")
        self.fps_sweep_stop_var = tk.StringVar(value="20.0")
        self.fps_sweep_step_var = tk.StringVar(value="2.0")
        self.fps_sweep_probe_seconds_var = tk.StringVar(value="20.0")
        self.fps_sweep_assume_ram_var = tk.StringVar(value="")
        self.fps_sweep_mock_var = tk.BooleanVar(value=False)
        self.fps_sweep_session_only_var = tk.BooleanVar(value=True)
        self.fps_sweep_status_var = tk.StringVar(value="Idle")
        self.fps_sweep_reset_status_var = tk.StringVar(value="")

        report_frame = ttk.LabelFrame(top, text="Single Video FPS Report", padding=8)
        report_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._grid_help_label(
            report_frame,
            row=0,
            column=0,
            text="Video path",
            help_title="Video Path",
            help_details="Path to recorded video file to analyze real framerate and timing drift.",
        )
        ttk.Entry(report_frame, textvariable=self.video_path_var, width=58).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )

        self._grid_help_label(
            report_frame,
            row=1,
            column=0,
            text="Timestamps path (optional)",
            help_title="Timestamp Sidecar",
            help_details="Optional frame timestamp file saved during recording for more accurate FPS diagnostics.",
        )
        ttk.Entry(report_frame, textvariable=self.timestamps_path_var, width=58).grid(
            row=1, column=1, sticky="ew", padx=8, pady=4
        )

        self._grid_help_label(
            report_frame,
            row=2,
            column=0,
            text="Expected seconds (optional)",
            help_title="Expected Duration",
            help_details="If provided, compares actual video duration against expected recording duration.",
        )
        ttk.Entry(report_frame, textvariable=self.recording_seconds_var, width=20).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Button(report_frame, text="Run FPS Report", command=self._run_fps_report).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        live_frame = ttk.LabelFrame(report_frame, text="Live FPS Test Capture", padding=6)
        live_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(
            live_frame,
            text=(
                "Record a short test clip now using the current config, then run the standard FPS report on it. "
                "Useful when you do not already have a video."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._grid_help_label(
            live_frame,
            row=1,
            column=0,
            text="Live test seconds",
            help_title="Live Test Seconds",
            help_details=(
                "Length of the temporary recording created for FPS analysis. "
                "Uses current codec, resolution, shutter, and target FPS from Config Editor."
            ),
        )
        ttk.Entry(live_frame, textvariable=self.fps_live_seconds_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8, pady=3
        )
        self.fps_live_run_btn = ttk.Button(
            live_frame,
            text="Capture Live Test + Run Report",
            command=self._start_live_fps_report,
        )
        self.fps_live_run_btn.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(live_frame, textvariable=self.fps_live_status_var).grid(
            row=2, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 0)
        )
        reset_controls = ttk.Frame(live_frame)
        reset_controls.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.fps_camera_reset_btn = ttk.Button(
            reset_controls,
            text="Reset Camera",
            command=lambda: self._start_fps_camera_reset("live"),
        )
        self.fps_camera_reset_btn.pack(side=tk.LEFT)
        self._make_help_button(
            reset_controls,
            title="Reset Camera",
            details=(
                "Runs a conservative open/close probe to release and reacquire the Pi camera stack. "
                "Use this if the camera is reported missing or busy after an FPS test."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(reset_controls, textvariable=self.fps_live_reset_status_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        live_frame.columnconfigure(2, weight=1)
        report_frame.columnconfigure(1, weight=1)

        sweep_frame = ttk.LabelFrame(top, text="FPS Sweep Capacity Test", padding=8)
        sweep_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            sweep_frame,
            text=(
                "Runs increasing FPS probes, compares target vs real FPS, and estimates max recording duration "
                "using the current workflow. RAM-backed workflows try to measure real memory growth on this machine "
                "while frames are held in memory, then fall back to a heuristic only when direct RAM samples are unavailable. "
                "MJPEG video-backed workflows use measured file growth and free storage under the current data root. "
                "If tracking ran this app session, it also estimates tracking time for each recording duration."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._grid_help_label(
            sweep_frame,
            row=1,
            column=0,
            text="Start",
            help_title="Sweep Start FPS",
            help_details="Starting FPS value for capacity sweep.",
        )
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_start_var, width=8).grid(
            row=1, column=1, sticky="w", padx=8, pady=3
        )
        self._grid_help_label(
            sweep_frame,
            row=2,
            column=0,
            text="Stop",
            help_title="Sweep Stop FPS",
            help_details="Maximum FPS value tested in sweep.",
        )
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_stop_var, width=8).grid(
            row=2, column=1, sticky="w", padx=8, pady=3
        )
        self._grid_help_label(
            sweep_frame,
            row=3,
            column=0,
            text="Step",
            help_title="Sweep Step",
            help_details="Increment between tested FPS values.",
        )
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_step_var, width=8).grid(
            row=3, column=1, sticky="w", padx=8, pady=3
        )
        self._grid_help_label(
            sweep_frame,
            row=4,
            column=0,
            text="Probe seconds",
            help_title="Probe Duration",
            help_details="Recording length used for each FPS probe point in the sweep.",
        )
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_probe_seconds_var, width=8).grid(
            row=4, column=1, sticky="w", padx=8, pady=3
        )

        advanced = ttk.LabelFrame(sweep_frame, text="Advanced Sweep Options", padding=6)
        advanced.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self._grid_help_label(
            advanced,
            row=0,
            column=0,
            text="FPS list (optional csv)",
            help_title="Custom FPS List",
            help_details="Comma-separated explicit FPS values. If set, this overrides start/stop/step generation.",
        )
        ttk.Entry(advanced, textvariable=self.fps_sweep_values_var, width=22).grid(
            row=0, column=1, sticky="w", padx=8, pady=3
        )
        self._grid_help_label(
            advanced,
            row=1,
            column=0,
            text="Assume RAM GiB (RAM-backed only)",
            help_title="Assumed RAM",
            help_details=(
                "Simulate capacity on a target machine when you are not measuring on the actual Pi. "
                "When this is set, BumbleBox uses the heuristic RAM model instead of empirical on-machine RAM profiling. "
                "Ignored for disk-backed MJPEG sweep runs."
            ),
        )
        ttk.Entry(advanced, textvariable=self.fps_sweep_assume_ram_var, width=8).grid(
            row=1, column=1, sticky="w", padx=8, pady=3
        )
        options = ttk.Frame(advanced)
        options.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(options, text="Use mock camera", variable=self.fps_sweep_mock_var).pack(side=tk.LEFT)
        self._make_help_button(
            options,
            title="Use Mock Camera",
            details="Uses simulated frames to estimate scheduling behavior when camera hardware is unavailable.",
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Checkbutton(
            options,
            text="Use tracking from current app session only",
            variable=self.fps_sweep_session_only_var,
        ).pack(side=tk.LEFT, padx=10)
        self._make_help_button(
            options,
            title="Session-only Tracking Estimate",
            details=(
                "When enabled, tracking-time estimates use measurements from tracking runs in this GUI session only."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))
        advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(advanced)

        controls = ttk.Frame(sweep_frame)
        controls.grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.fps_sweep_run_btn = ttk.Button(controls, text="Run FPS Sweep", command=self._start_fps_sweep)
        self.fps_sweep_run_btn.pack(side=tk.LEFT)
        self.fps_sweep_reset_btn = ttk.Button(
            controls,
            text="Reset Camera",
            command=lambda: self._start_fps_camera_reset("sweep"),
        )
        self.fps_sweep_reset_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._make_help_button(
            controls,
            title="Reset Camera",
            details=(
                "Runs the same camera recovery action used in the live FPS section. "
                "Use this if the sweep cannot reacquire the camera after a previous test."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(controls, textvariable=self.fps_sweep_status_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(controls, textvariable=self.fps_sweep_reset_status_var).pack(side=tk.LEFT, padx=(4, 0))
        sweep_frame.columnconfigure(1, weight=1)
        sweep_frame.columnconfigure(3, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.fps_output = self._create_results_section(
            self.fps_tab,
            title="FPS Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=16,
        )

    def _build_calibration_tab(self) -> None:
        container = ttk.Frame(self.calibration_tab)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            container,
            highlightthickness=0,
            bg=self._palette["panel_bg"],
            bd=0,
        )
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scroll_body = ttk.Frame(canvas)
        scroll_body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        scroll_window = canvas.create_window((0, 0), window=scroll_body, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event, item=scroll_window, widget=canvas: widget.itemconfigure(item, width=max(1, int(event.width))),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        top = ttk.Frame(scroll_body)
        top.pack(fill=tk.X)

        self.manual_point_a = tk.StringVar(value="0,0")
        self.manual_point_b = tk.StringVar(value="500,0")
        self.manual_distance_cm = tk.StringVar(value="10.0")
        self.calibration_labelme_image_path_var = tk.StringVar(value="")
        self.calibration_labelme_json_path_var = tk.StringVar(value="")
        self.thermal_registration_rgb_image_var = tk.StringVar(value="")
        self.thermal_registration_thermal_image_var = tk.StringVar(value="")

        self.aruco_image = tk.StringVar()
        self.aruco_marker_size_mm = tk.StringVar(value="5.0")
        self.aruco_dictionary = tk.StringVar(value="4X4_50")
        self.aruco_marker_id = tk.StringVar(value="")

        manual = ttk.LabelFrame(top, text="Manual Scale (Recommended)", padding=8)
        manual.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(
            manual,
            text=(
                "Use two points on the same imaging plane. "
                "Prefer a larger known distance (for example 5-15 cm) to reduce noise."
            ),
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._grid_help_label(
            manual,
            row=1,
            column=0,
            text="Point A (x,y)",
            help_title="Point A",
            help_details="First pixel coordinate on the same plane as Point B.",
        )
        ttk.Entry(manual, textvariable=self.manual_point_a).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            manual,
            row=2,
            column=0,
            text="Point B (x,y)",
            help_title="Point B",
            help_details="Second pixel coordinate used with Point A to estimate pixel-to-cm scale.",
        )
        ttk.Entry(manual, textvariable=self.manual_point_b).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            manual,
            row=3,
            column=0,
            text="Real distance (cm)",
            help_title="Real Distance",
            help_details="Measured real-world distance between A and B in centimeters.",
        )
        ttk.Entry(manual, textvariable=self.manual_distance_cm).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(manual, text="Calibrate from Points", command=self._calibrate_manual).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        picker = ttk.LabelFrame(manual, text="Camera Point Picker (LabelMe)", padding=8)
        picker.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(
            picker,
            text=(
                "Capture a fresh image from the camera and open LabelMe directly. "
                "Draw one line across the known-distance endpoints, save, then load points."
            ),
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(
            picker,
            text="Capture Image + Open LabelMe",
            command=self._capture_and_open_calibration_labelme,
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        self._grid_help_label(
            picker,
            row=2,
            column=0,
            text="Captured image",
            help_title="Captured Calibration Image",
            help_details="Latest calibration image captured from the Pi camera and opened in LabelMe.",
        )
        ttk.Entry(
            picker,
            textvariable=self.calibration_labelme_image_path_var,
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            picker,
            row=3,
            column=0,
            text="Expected LabelMe JSON path",
            help_title="LabelMe JSON",
            help_details=(
                "JSON saved by LabelMe containing your point annotations. "
                "This is the expected save path and is not created until you save in LabelMe."
            ),
        )
        ttk.Entry(
            picker,
            textvariable=self.calibration_labelme_json_path_var,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(
            picker,
            text="Load LabelMe Points -> Point A/B",
            command=self._load_calibration_points_from_labelme,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(
            picker,
            text="Load + Calibrate from LabelMe Points",
            command=self._load_and_calibrate_from_labelme,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        picker.columnconfigure(1, weight=1)
        manual.columnconfigure(1, weight=1)

        aruco = ttk.LabelFrame(top, text="ArUco Marker (Optional Advanced)", padding=8)
        aruco.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            aruco,
            text=(
                "Only use this mode if you are calibrating from a known-size printed ArUco marker. "
                "Marker size is required so px/cm can be computed."
            ),
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._grid_help_label(
            aruco,
            row=1,
            column=0,
            text="Image path",
            help_title="Calibration Image",
            help_details="Path to an image containing a clearly visible known-size ArUco marker.",
        )
        ttk.Entry(aruco, textvariable=self.aruco_image).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            aruco,
            row=2,
            column=0,
            text="Marker size (mm)",
            help_title="Marker Size",
            help_details="Physical printed marker edge size in millimeters.",
        )
        ttk.Entry(aruco, textvariable=self.aruco_marker_size_mm).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            aruco,
            row=3,
            column=0,
            text="Dictionary",
            help_title="ArUco Dictionary",
            help_details="Dictionary used by your printed marker (for example 4X4_50 or 4X4_100).",
        )
        ttk.Entry(aruco, textvariable=self.aruco_dictionary).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        self._grid_help_label(
            aruco,
            row=4,
            column=0,
            text="Marker ID (optional)",
            help_title="Marker ID",
            help_details="Optional specific marker ID to target when multiple markers are visible.",
        )
        ttk.Entry(aruco, textvariable=self.aruco_marker_id).grid(row=4, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(aruco, text="Calibrate from ArUco", command=self._calibrate_aruco).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        aruco.columnconfigure(1, weight=1)
        self._register_advanced_widget(aruco)

        thermal_reg = ttk.LabelFrame(top, text="Thermal Registration (Approximate 2D)", padding=8)
        thermal_reg.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(
            thermal_reg,
            text=(
                "Pick corresponding points side by side on one RGB frame and one thermal frame to build an approximate 2D warp. "
                "You will place one pair at a time and can drag points to fine-tune them before finishing. "
                "This helps align the two cameras for visualization, but it does not fully correct the 3D nest geometry."
            ),
            wraplength=860,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._grid_help_label(
            thermal_reg,
            row=1,
            column=0,
            text="RGB image",
            help_title="RGB Registration Image",
            help_details="RGB frame used as the reference target for thermal registration.",
        )
        ttk.Entry(thermal_reg, textvariable=self.thermal_registration_rgb_image_var, width=72).grid(
            row=1, column=1, sticky="ew", padx=8, pady=3
        )
        ttk.Button(
            thermal_reg,
            text="Browse RGB",
            command=self._browse_thermal_registration_rgb_image,
        ).grid(row=1, column=2, sticky="w", padx=(0, 6), pady=3)

        self._grid_help_label(
            thermal_reg,
            row=2,
            column=0,
            text="Thermal image",
            help_title="Thermal Registration Image",
            help_details="Thermal frame that shows the same scene as the RGB image.",
        )
        ttk.Entry(thermal_reg, textvariable=self.thermal_registration_thermal_image_var, width=72).grid(
            row=2, column=1, sticky="ew", padx=8, pady=3
        )
        ttk.Button(
            thermal_reg,
            text="Browse Thermal",
            command=self._browse_thermal_registration_thermal_image,
        ).grid(row=2, column=2, sticky="w", padx=(0, 6), pady=3)

        reg_actions = ttk.Frame(thermal_reg)
        reg_actions.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(
            reg_actions,
            text="Use Latest Session Midframes",
            command=self._load_latest_thermal_registration_frames,
        ).pack(side=tk.LEFT)
        ttk.Button(
            reg_actions,
            text="Pick Points + Register + Save",
            command=self._register_thermal_from_images,
        ).pack(side=tk.LEFT, padx=8)
        self._make_help_button(
            reg_actions,
            title="Thermal Registration",
            details=(
                "RGB and thermal frames open side by side in one window. Click the next point on the highlighted image, "
                "then click the matching point on the other image. The thermal pane is upscaled for easier point placement, "
                "but BumbleBox converts those clicks back to native thermal coordinates before fitting the warp. "
                "Existing points can be dragged to adjust them, mouse wheel zoom changes the display scale, "
                "and right-clicking near a point deletes that pair. At least 4 point pairs are required. "
                "BumbleBox then saves a homography, warped thermal preview, overlay preview, and registration JSON."
            ),
        ).pack(side=tk.LEFT)
        thermal_reg.columnconfigure(1, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.calibration_output = self._create_results_section(
            scroll_body,
            title="Calibration Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=14,
        )

    def _build_schedule_check_tab(self) -> None:
        top = ttk.Frame(self.schedule_check_tab)
        top.pack(fill=tk.X)

        self.schedule_benchmark_input_var = tk.StringVar(value="")
        self.schedule_benchmark_frames_var = tk.StringVar(value="80")
        self.schedule_assume_ram_gb_var = tk.StringVar(value="")

        ttk.Label(top, text="Benchmark input (optional)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.schedule_benchmark_input_var, width=90).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )

        ttk.Label(top, text="Benchmark sample frames").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.schedule_benchmark_frames_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Label(top, text="Assume RAM (GiB, optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.schedule_assume_ram_gb_var, width=10).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Button(top, text="Run Schedule Check", command=self._run_schedule_check).grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )

        info = (
            "Use this before long experiments to estimate whether record/track timing "
            "and memory budget fit your hardware. Add benchmark input for better estimates, "
            "or set an assumed RAM value to simulate a target Pi."
        )
        ttk.Label(top, text=info, wraplength=800, justify=tk.LEFT).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        top.columnconfigure(1, weight=1)

        self.schedule_check_output = self._create_results_section(
            self.schedule_check_tab,
            title="Schedule Check Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=16,
        )

    def _run_schedule_check(self) -> None:
        try:
            config, _ = self._load_effective_action_config()
            benchmark_input = self.schedule_benchmark_input_var.get().strip() or None
            benchmark_frames = int(self.schedule_benchmark_frames_var.get().strip())
            if benchmark_frames <= 0:
                raise ValueError("Benchmark frame count must be >= 1")
            assume_ram_text = self.schedule_assume_ram_gb_var.get().strip()
            assume_ram_gb = float(assume_ram_text) if assume_ram_text else None
            if assume_ram_gb is not None and assume_ram_gb <= 0:
                raise ValueError("Assumed RAM must be > 0")

            report = run_schedule_check(
                config=config,
                benchmark_input=benchmark_input,
                benchmark_frames=benchmark_frames,
                assume_ram_gb=assume_ram_gb,
            )
            self.schedule_check_output.delete("1.0", tk.END)
            self.schedule_check_output.insert(
                tk.END,
                "Using Config Editor snapshot for schedule check.\n\n"
                + format_schedule_check_report(report),
            )
            self.notebook.select(self.schedule_check_tab)
            if report.has_failures:
                self._show_warning(
                    "Schedule check",
                    "Schedule check found failures. Review suggestions in the output panel.",
                )
        except Exception as exc:
            self._show_error("Schedule check failed", str(exc))

    def _build_optimize_tracking_tab(self) -> None:
        controls_container = ttk.Frame(self.optimize_tracking_tab)
        controls_container.pack(fill=tk.BOTH, expand=True)

        controls_canvas = tk.Canvas(
            controls_container,
            highlightthickness=0,
            bg=self._palette["panel_bg"],
            bd=0,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient=tk.VERTICAL,
            command=controls_canvas.yview,
        )
        top = ttk.Frame(controls_canvas)
        top.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_window = controls_canvas.create_window((0, 0), window=top, anchor="nw")
        controls_canvas.bind(
            "<Configure>",
            lambda event, item=controls_window, widget=controls_canvas: widget.itemconfigure(
                item,
                width=max(1, int(event.width)),
            ),
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def scroll_optimize_controls(event: tk.Event) -> str:
            delta = getattr(event, "delta", 0)
            if delta:
                steps = -1 if delta > 0 else 1
            else:
                steps = -1 if getattr(event, "num", 0) == 4 else 1
            controls_canvas.yview_scroll(steps * 3, "units")
            return "break"

        def bind_optimize_control_scroll(widget: tk.Widget) -> None:
            widget.bind("<MouseWheel>", scroll_optimize_controls, add="+")
            widget.bind("<Button-4>", scroll_optimize_controls, add="+")
            widget.bind("<Button-5>", scroll_optimize_controls, add="+")
            for child in widget.winfo_children():
                bind_optimize_control_scroll(child)

        self.opt_input_path_var = tk.StringVar(value="")
        self.opt_output_dir_var = tk.StringVar(value="")
        self.opt_profile_var = tk.StringVar(value="quick")
        self.opt_dictionary_var = tk.StringVar(value=self._current_config_tag_dictionary())
        self.opt_tag_size_mm_var = tk.StringVar(value="2.5")
        self.opt_sample_frames_var = tk.StringVar(value="40")
        self.opt_execution_target_var = tk.StringVar(value="pi_safe")
        self.opt_workers_var = tk.StringVar(value="")
        self.opt_expected_tags_var = tk.StringVar(value="")
        self.opt_early_stop_patience_var = tk.StringVar(value="0")
        self.opt_early_stop_min_improvement_var = tk.StringVar(value="0")
        self.opt_preview_var = tk.BooleanVar(value=False)
        self.opt_preview_frames_var = tk.StringVar(value="240")
        self.opt_top_k_var = tk.StringVar(value="5")
        self.opt_refinement_rounds_var = tk.StringVar(value="2")
        self.opt_refinement_seed_count_var = tk.StringVar(value="5")
        self.opt_refinement_seed_source_var = tk.StringVar(value="mean_detection")
        self.opt_refinement_validation_multiplier_var = tk.StringVar(value="2")
        self.video_range_manifest_var = tk.StringVar(value="")
        self.video_range_source_root_var = tk.StringVar(value="")
        self.video_range_bounds_file_var = tk.StringVar(value="")
        self.video_range_selected_var = tk.StringVar(value="No video range manifest loaded.")
        self.video_range_status_var = tk.StringVar(value="Load a video range CSV to measure per-video tag bounds.")
        self._opt_sweep_profile_vars: dict[str, dict[str, tk.StringVar]] = {}
        self._opt_profile_notebook: ttk.Notebook | None = None
        self._opt_profile_tabs: dict[str, ttk.Frame] = {}
        self._opt_profile_syncing = False
        self._opt_review_perimeter_bounds: dict[str, object] | None = None
        self.opt_status_var = tk.StringVar(value="Idle")
        self._optimize_top_k = 5

        self._grid_help_label(
            top,
            row=0,
            column=0,
            text="Input path (video or image folder)",
            help_title="Optimization Input",
            help_details=(
                "Path to a representative video or image folder used to test ArUco parameter sweeps. "
                "Image folders are searched recursively, so you can point this at a day/colony folder containing PNG frames. "
                "Generated optimizer output folders are skipped to avoid re-optimizing annotated review images."
            ),
        )
        input_frame = ttk.Frame(top)
        input_frame.grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Entry(input_frame, textvariable=self.opt_input_path_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True)
        input_frame.columnconfigure(0, weight=1)

        self._grid_help_label(
            top,
            row=1,
            column=0,
            text="Profile",
            help_title="Optimization Profile",
            help_details=(
                "Quick tests fewer combinations; Daily keeps measured size bounds fixed and sweeps "
                "lighting-sensitive parameters; Deep explores more combinations and takes longer."
            ),
        )
        profile_combo = ttk.Combobox(
            top,
            textvariable=self.opt_profile_var,
            values=["quick", "balanced", "deep", "daily"],
            state="readonly",
            width=20,
        )
        profile_combo.grid(row=1, column=1, sticky="w", padx=8, pady=4)
        profile_combo.bind("<<ComboboxSelected>>", self._on_optimize_profile_selected)

        self._grid_help_label(
            top,
            row=2,
            column=0,
            text="Tag dictionary",
            help_title="Tag Dictionary",
            help_details=(
                "ArUco dictionary used for optimization. It starts from tracking.tag_dictionary in config. "
                "Use Save To Config if this should become the dictionary used by normal recording/tracking runs."
            ),
        )
        dictionary_frame = ttk.Frame(top)
        dictionary_frame.grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Combobox(
            dictionary_frame,
            textvariable=self.opt_dictionary_var,
            values=ARUCO_4X4_DICTIONARY_OPTIONS,
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)
        ttk.Button(
            dictionary_frame,
            text="Save To Config",
            command=self._save_optimizer_dictionary_to_config,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self._grid_help_label(
            top,
            row=3,
            column=0,
            text="Tag size (mm)",
            help_title="Tag Size",
            help_details="Physical marker size in millimeters; used to shape parameter heuristics.",
        )
        tag_size_frame = ttk.Frame(top)
        tag_size_frame.grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Entry(tag_size_frame, textvariable=self.opt_tag_size_mm_var, width=10).pack(side=tk.LEFT)
        ttk.Button(
            tag_size_frame,
            text="Measure Smallest & Largest Tags",
            command=self._open_tag_perimeter_measurement_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._make_help_button(
            tag_size_frame,
            title="Measure Smallest & Largest Tags",
            details=(
                "Open representative frames and measure the smallest and largest real tags you want BumbleBox "
                "to detect. BumbleBox uses those pixel perimeters to choose safer minMarkerPerimeterRate and "
                "maxMarkerPerimeterRate sweep values for the first optimization pass."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._grid_help_label(
            top,
            row=4,
            column=0,
            text="Sample frames",
            help_title="Sample Frames",
            help_details="Number of frames sampled for scoring. More frames improve robustness but increase runtime.",
        )
        ttk.Entry(top, textvariable=self.opt_sample_frames_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=4)

        self._grid_help_label(
            top,
            row=5,
            column=0,
            text="Expected tags/frame (optional)",
            help_title="Expected Tags",
            help_details=(
                "Expected real tag count visible in a typical frame. For example, use about 150 if roughly "
                "150 tags are visible. This helps the optimizer prefer candidates that detect more true tags "
                "without letting false positives dominate."
            ),
        )
        ttk.Entry(top, textvariable=self.opt_expected_tags_var, width=10).grid(row=5, column=1, sticky="w", padx=8, pady=4)

        self._grid_help_label(
            top,
            row=6,
            column=0,
            text="Execution target",
            help_title="Execution Target",
            help_details=(
                "Pi-safe limits worker usage for reliability on Raspberry Pi. "
                "Desktop mode can use more cores for faster sweeps."
            ),
        )
        execution_frame = ttk.Frame(top)
        execution_frame.grid(row=6, column=1, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(
            execution_frame,
            text="Pi-safe (recommended on Pi)",
            variable=self.opt_execution_target_var,
            value="pi_safe",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            execution_frame,
            text="Desktop (more cores)",
            variable=self.opt_execution_target_var,
            value="desktop",
        ).pack(side=tk.LEFT, padx=12)

        review_note = ttk.Frame(top)
        review_note.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            review_note,
            text="After optimization, BumbleBox can open a top-5 review window before applying any parameters.",
        ).pack(side=tk.LEFT)
        self._make_help_button(
            review_note,
            title="Optimization Review",
            details=(
                "When optimization finishes, BumbleBox can open a review window for the top five candidates. "
                "You can browse representative frames, compare candidate parameters, and apply the one you choose."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        video_range_frame = self._build_video_range_calibration_panel(top)
        video_range_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        advanced = ttk.LabelFrame(top, text="Advanced Optimization Options", padding=6)
        advanced.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._grid_help_label(
            advanced,
            row=0,
            column=0,
            text="Output root (optional)",
            help_title="Output Root",
            help_details="Optional output directory for optimizer reports and preview artifacts.",
        )
        ttk.Entry(advanced, textvariable=self.opt_output_dir_var, width=70).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self._grid_help_label(
            advanced,
            row=1,
            column=0,
            text="Workers (optional override)",
            help_title="Workers",
            help_details="Manual worker count override. Leave blank to let optimizer choose based on target mode.",
        )
        ttk.Entry(advanced, textvariable=self.opt_workers_var, width=10).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        self._grid_help_label(
            advanced,
            row=2,
            column=0,
            text="Early stop patience",
            help_title="Early Stop Patience",
            help_details=(
                "Set to 0 to test every parameter combination. This is the default now because the GUI has "
                "an End Early button. Use a positive number only if you intentionally want automatic early stopping."
            ),
        )
        ttk.Entry(advanced, textvariable=self.opt_early_stop_patience_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        self._grid_help_label(
            advanced,
            row=3,
            column=0,
            text="Early stop min improvement",
            help_title="Minimum Improvement",
            help_details=(
                "Minimum score gain considered meaningful when early stopping is enabled. This is ignored when "
                "early stop patience is 0."
            ),
        )
        ttk.Entry(advanced, textvariable=self.opt_early_stop_min_improvement_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=4)
        self._grid_help_label(
            advanced,
            row=4,
            column=0,
            text="Top results to show",
            help_title="Top Results",
            help_details="Number of best parameter candidates shown in optimizer output.",
        )
        ttk.Entry(advanced, textvariable=self.opt_top_k_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=4)
        preview_options = ttk.Frame(advanced)
        preview_options.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            preview_options,
            text="Write preview video",
            variable=self.opt_preview_var,
        ).pack(side=tk.LEFT, padx=12)
        self._make_help_button(
            preview_options,
            title="Write Preview Video",
            details="Writes an annotated short output clip using selected candidate parameters for visual inspection.",
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(preview_options, text="Preview frames").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(preview_options, textvariable=self.opt_preview_frames_var, width=8).pack(side=tk.LEFT)
        self._make_help_button(
            preview_options,
            title="Preview Frames",
            details="Maximum number of frames rendered in preview output.",
        ).pack(side=tk.LEFT, padx=(4, 0))

        refinement_options = ttk.Frame(advanced)
        refinement_options.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(refinement_options, text="Iterative refinement rounds").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(refinement_options, textvariable=self.opt_refinement_rounds_var, width=5).pack(side=tk.LEFT)
        ttk.Label(refinement_options, text="seed candidates").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(refinement_options, textvariable=self.opt_refinement_seed_count_var, width=5).pack(side=tk.LEFT)
        ttk.Label(refinement_options, text="seed source").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(
            refinement_options,
            textvariable=self.opt_refinement_seed_source_var,
            values=["mean_detection", "score"],
            state="readonly",
            width=15,
        ).pack(side=tk.LEFT)
        ttk.Label(refinement_options, text="validation frame multiplier").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(refinement_options, textvariable=self.opt_refinement_validation_multiplier_var, width=5).pack(side=tk.LEFT)
        self._make_help_button(
            refinement_options,
            title="Iterative Refinement",
            details=(
                "After a broad optimization run, this can refine around the current top candidates. "
                "Use mean_detection to refine around the highest-detection candidates, or score to refine around "
                "the objective-function winners. "
                "Round 1 uses wider nearby values, later rounds use narrower values, and the final "
                "validation pass re-scores finalists on more sampled frames."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        sweep_frame = ttk.LabelFrame(
            advanced,
            text="Profile Sweep Values (editable, comma-separated)",
            padding=6,
        )
        sweep_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(
            sweep_frame,
            text=(
                "Each tab shows the current sweep values for that profile. "
                "Edit any field, or revert the tab to defaults for the current tag size."
            ),
        ).grid(row=0, column=0, sticky="w")
        self._opt_profile_notebook = ttk.Notebook(sweep_frame)
        self._opt_profile_notebook.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._opt_profile_notebook.bind("<<NotebookTabChanged>>", self._on_optimize_profile_tab_changed)
        for profile in ("quick", "balanced", "deep", "daily"):
            tab = self._build_optimize_sweep_profile_tab(self._opt_profile_notebook, profile)
            self._opt_profile_tabs[profile] = tab
            self._opt_profile_notebook.add(tab, text=profile)
            self._reset_optimize_sweep_profile_defaults(profile)
        self._select_optimize_profile_tab(self.opt_profile_var.get().strip().lower() or "quick")
        sweep_frame.columnconfigure(0, weight=1)

        advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(advanced)

        controls = ttk.Frame(top)
        controls.grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.optimize_run_btn = ttk.Button(
            controls,
            text="Run optimize-tracking",
            command=self._start_optimize_tracking,
        )
        self.optimize_run_btn.pack(side=tk.LEFT)
        self.optimize_refine_btn = ttk.Button(
            controls,
            text="Run Iterative Refinement From Selected Top 5",
            command=self._start_optimize_refinement,
            state=tk.DISABLED,
        )
        self.optimize_refine_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.optimize_stop_btn = ttk.Button(
            controls,
            text="End Early",
            command=self._stop_optimize_tracking,
            state=tk.DISABLED,
        )
        self.optimize_stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(controls, textvariable=self.opt_status_var).pack(side=tk.LEFT, padx=10)

        top.columnconfigure(1, weight=1)
        bind_optimize_control_scroll(controls_canvas)
        bind_optimize_control_scroll(top)

        self.optimize_output = self._create_results_section(
            self.optimize_tracking_tab,
            title="Optimization Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=16,
            expand=False,
            wrap=tk.NONE,
            horizontal_scroll=True,
        )

    def _build_video_range_calibration_panel(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Video Range Tag Bounds", padding=6)

        self._grid_help_label(
            frame,
            row=0,
            column=0,
            text="Video range CSV",
            help_title="Video Range CSV",
            help_details=(
                "CSV with video_id and frame_ranges columns. This is used to step through the requested "
                "videos and measure tag-size bounds on the exact frames that will be tracked."
            ),
        )
        manifest_row = ttk.Frame(frame)
        manifest_row.grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Entry(manifest_row, textvariable=self.video_range_manifest_var, width=70).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )
        ttk.Button(manifest_row, text="Browse", command=self._browse_video_range_manifest).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        self._grid_help_label(
            frame,
            row=1,
            column=0,
            text="Project/source root",
            help_title="Project Source Root",
            help_details=(
                "Folder that contains the videos and, when available, the extracted frames/<video_id> folders. "
                "For the current project this is the test_august_june25 project folder."
            ),
        )
        source_row = ttk.Frame(frame)
        source_row.grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Entry(source_row, textvariable=self.video_range_source_root_var, width=70).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )
        ttk.Button(source_row, text="Browse", command=self._browse_video_range_source_root).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        self._grid_help_label(
            frame,
            row=2,
            column=0,
            text="Bounds JSON",
            help_title="Per-Video Bounds JSON",
            help_details=(
                "Where the per-video smallest/largest tag measurements are saved. The future video-range "
                "tracking workflow can read this file to use different perimeter bounds for different videos."
            ),
        )
        bounds_row = ttk.Frame(frame)
        bounds_row.grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Entry(bounds_row, textvariable=self.video_range_bounds_file_var, width=70).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )
        ttk.Button(bounds_row, text="Browse", command=self._browse_video_range_bounds_file).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        controls = ttk.Frame(frame)
        controls.grid(row=3, column=1, sticky="w", padx=8, pady=(6, 2))
        ttk.Button(controls, text="Load CSV", command=self._load_video_range_manifest).pack(side=tk.LEFT)
        ttk.Button(controls, text="Prev Video", command=lambda: self._shift_video_range_selection(-1)).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        ttk.Button(controls, text="Next Video", command=lambda: self._shift_video_range_selection(1)).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )
        ttk.Button(
            controls,
            text="Measure Selected Video",
            command=self._open_video_range_tag_measurement,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._make_help_button(
            controls,
            title="Measure Selected Video",
            details=(
                "Opens the corner-marking tool on only the selected video's requested frame range. "
                "After you apply the measurement, BumbleBox saves the smallest/largest tag bounds for that video."
            ),
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(frame, textvariable=self.video_range_selected_var, justify=tk.LEFT).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Label(frame, textvariable=self.video_range_status_var, justify=tk.LEFT).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(2, 0),
        )

        frame.columnconfigure(1, weight=1)
        return frame

    def _save_optimizer_dictionary_to_config(self) -> None:
        try:
            dictionary = self._normalize_display_dictionary(self.opt_dictionary_var.get())
            config, config_path = self._load_config_or_defaults()
            config.setdefault("tracking", {})["tag_dictionary"] = dictionary
            validate_config(config)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="optimizer_tag_dictionary",
            )
            self._load_config_into_editor()
            self.opt_dictionary_var.set(dictionary)
        except Exception as exc:
            self._show_error("Save dictionary failed", str(exc))
            return

        history_note = self._format_config_history_note(snapshot_path, history_warning)
        message = f"Saved tracking.tag_dictionary = {dictionary} to:\n{config_path}"
        if history_note:
            message += f"\n\n{history_note}"
        self.optimize_output.delete("1.0", tk.END)
        self.optimize_output.insert(tk.END, message)
        self._show_info("Tag dictionary saved", message)

    def _parse_csv_numeric_values(self, raw: str, *, label: str, value_type: str) -> list[float | int]:
        text = str(raw or "").strip()
        if not text:
            return []
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens:
            raise ValueError(f"{label} is empty.")

        out: list[float | int] = []
        for token in tokens:
            if value_type == "float":
                try:
                    out.append(float(token))
                except Exception as exc:
                    raise ValueError(f"{label} has invalid float value: {token}") from exc
                continue

            if value_type == "int":
                try:
                    as_float = float(token)
                except Exception as exc:
                    raise ValueError(f"{label} has invalid integer value: {token}") from exc
                if not as_float.is_integer():
                    raise ValueError(f"{label} requires whole numbers, got: {token}")
                out.append(int(as_float))
                continue

            raise ValueError(f"Unsupported parse type: {value_type}")

        unique = []
        for value in out:
            if value not in unique:
                unique.append(value)
        return unique

    def _format_optimize_sweep_values(self, values: list[float | int]) -> str:
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:g}")
            else:
                formatted.append(str(value))
        return ", ".join(formatted)

    def _current_opt_tag_size_for_defaults(self) -> float:
        from .tracking_optimizer import DEFAULT_TAG_SIZE_MM

        raw = self.opt_tag_size_mm_var.get().strip()
        if not raw:
            return DEFAULT_TAG_SIZE_MM
        try:
            parsed = float(raw)
        except Exception:
            return DEFAULT_TAG_SIZE_MM
        if parsed <= 0:
            return DEFAULT_TAG_SIZE_MM
        return parsed

    def _resolve_optimize_profile_sweep_defaults(self, profile: str) -> dict[str, list[float | int]]:
        from .tracking_optimizer import resolve_profile_parameter_space

        return resolve_profile_parameter_space(
            profile=profile,
            tag_size_mm=self._current_opt_tag_size_for_defaults(),
            frame_width=4056,
            frame_height=3040,
        )

    def _build_optimize_sweep_profile_tab(self, notebook: ttk.Notebook, profile: str) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=6)
        profile_vars: dict[str, tk.StringVar] = {}
        self._opt_sweep_profile_vars[profile] = profile_vars

        for row, (key, _value_type, help_title, help_details) in enumerate(OPTIMIZE_SWEEP_FIELDS):
            profile_vars[key] = tk.StringVar(value="")
            self._grid_help_label(
                tab,
                row=row,
                column=0,
                text=key,
                help_title=help_title,
                help_details=help_details,
            )
            ttk.Entry(
                tab,
                textvariable=profile_vars[key],
                width=34,
            ).grid(row=row, column=1, sticky="ew", padx=8, pady=2)

        controls = ttk.Frame(tab)
        controls.grid(row=len(OPTIMIZE_SWEEP_FIELDS), column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(
            controls,
            text="Revert To Profile Defaults",
            command=lambda selected_profile=profile: self._reset_optimize_sweep_profile_defaults(selected_profile),
        ).pack(side=tk.LEFT)
        ttk.Label(
            controls,
            text="Blank fields fall back to defaults for the current tag size.",
        ).pack(side=tk.LEFT, padx=(10, 0))

        tab.columnconfigure(1, weight=1)
        return tab

    def _reset_optimize_sweep_profile_defaults(self, profile: str) -> None:
        defaults = self._resolve_optimize_profile_sweep_defaults(profile)
        profile_vars = self._opt_sweep_profile_vars.get(profile, {})
        for key, _value_type, _help_title, _help_details in OPTIMIZE_SWEEP_FIELDS:
            if key not in profile_vars:
                continue
            profile_vars[key].set(self._format_optimize_sweep_values(list(defaults[key])))

    def _select_optimize_profile_tab(self, profile: str) -> None:
        if self._opt_profile_notebook is None:
            return
        tab = self._opt_profile_tabs.get(profile)
        if tab is None:
            return
        self._opt_profile_syncing = True
        try:
            self.opt_profile_var.set(profile)
            self._opt_profile_notebook.select(tab)
        finally:
            self._opt_profile_syncing = False

    def _on_optimize_profile_selected(self, _event: object | None = None) -> None:
        if self._opt_profile_syncing:
            return
        self._select_optimize_profile_tab(self.opt_profile_var.get().strip().lower())

    def _on_optimize_profile_tab_changed(self, _event: object | None = None) -> None:
        if self._opt_profile_syncing or self._opt_profile_notebook is None:
            return
        tab_id = self._opt_profile_notebook.select()
        if not tab_id:
            return
        profile = str(self._opt_profile_notebook.tab(tab_id, "text")).strip().lower()
        if not profile:
            return
        self._opt_profile_syncing = True
        try:
            self.opt_profile_var.set(profile)
        finally:
            self._opt_profile_syncing = False

    def _build_optimizer_measurement_frame_refs(self, input_path: str, *, max_frames: int = 25) -> list[dict[str, object]]:
        import cv2

        from .tracking_optimizer import (
            IMAGE_EXTENSIONS,
            VIDEO_EXTENSIONS,
            find_supported_image_paths,
            _resolve_user_path,
            _sample_indices,
        )

        resolved = _resolve_user_path(input_path)
        if resolved.is_dir():
            image_paths = find_supported_image_paths(resolved)
            if not image_paths:
                raise RuntimeError(f"No supported image files found recursively in: {resolved}")
            indices = _sample_indices(len(image_paths), min(max_frames, len(image_paths)))
            return [
                {
                    "kind": "image",
                    "path": image_paths[index],
                    "label": f"{image_paths[index].name} ({index + 1}/{len(image_paths)})",
                }
                for index in indices
            ]

        if not resolved.is_file():
            raise FileNotFoundError(f"Input path does not exist: {resolved}")

        suffix = resolved.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return [{"kind": "image", "path": resolved, "label": resolved.name}]

        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input extension for measurement: {resolved.suffix}")

        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {resolved}")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if total_frames > 0:
            indices = _sample_indices(total_frames, min(max_frames, total_frames))
        else:
            indices = [0]
        return [
            {
                "kind": "video",
                "path": resolved,
                "frame_index": index,
                "label": f"{resolved.name} frame {index}",
            }
            for index in indices
        ]

    def _load_optimizer_measurement_frame_ref(self, frame_ref: dict[str, object]) -> tuple[object, str]:
        import cv2

        path = Path(frame_ref["path"])
        kind = str(frame_ref["kind"])
        label = str(frame_ref.get("label") or path)
        if kind == "image":
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {path}")
            return frame, label

        if kind == "video":
            frame_index = int(frame_ref.get("frame_index") or 0)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame {frame_index} from video: {path}")
            finally:
                cap.release()
            return frame, label

        raise ValueError(f"Unsupported frame reference kind: {kind}")

    def _browse_video_range_manifest(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video range CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_range_manifest_var.set(path)
        if not self.video_range_bounds_file_var.get().strip():
            self.video_range_bounds_file_var.set(self._default_video_range_bounds_file())

    def _browse_video_range_source_root(self) -> None:
        path = filedialog.askdirectory(title="Select project/source root")
        if not path:
            return
        self.video_range_source_root_var.set(path)
        if not self.video_range_bounds_file_var.get().strip():
            self.video_range_bounds_file_var.set(self._default_video_range_bounds_file())

    def _browse_video_range_bounds_file(self) -> None:
        initial = self.video_range_bounds_file_var.get().strip() or self._default_video_range_bounds_file()
        path = filedialog.asksaveasfilename(
            title="Save per-video tag bounds JSON",
            initialfile=Path(initial).name if initial else "video_range_tag_bounds.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.video_range_bounds_file_var.set(path)

    def _default_video_range_bounds_file(self) -> str:
        manifest_text = self.video_range_manifest_var.get().strip()
        if manifest_text:
            manifest_path = Path(manifest_text).expanduser()
            return str(manifest_path.with_name(f"{manifest_path.stem}_tag_bounds.json"))
        source_text = self.video_range_source_root_var.get().strip()
        if source_text:
            return str(Path(source_text).expanduser() / "video_range_tag_bounds.json")
        return "video_range_tag_bounds.json"

    def _parse_video_frame_ranges(self, row: dict[str, str]) -> list[tuple[int, int]]:
        text = str(row.get("frame_ranges") or "").strip()
        if not text:
            start_text = str(row.get("first_start_frame") or "").strip()
            end_text = str(row.get("last_end_frame") or "").strip()
            if start_text and end_text:
                text = f"{start_text}-{end_text}"
        if not text:
            raise ValueError("missing frame_ranges")

        ranges: list[tuple[int, int]] = []
        for token in re.split(r"[;,]+", text):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(float(start_text.strip()))
                end = int(float(end_text.strip()))
            else:
                start = end = int(float(token))
            if start < 0 or end < 0:
                raise ValueError(f"negative frame index in {text!r}")
            if end < start:
                start, end = end, start
            ranges.append((start, end))
        if not ranges:
            raise ValueError("no usable frame ranges")
        return ranges

    def _video_range_indices(self, ranges: list[tuple[int, int]], *, max_frames: int | None = None) -> list[int]:
        indices: list[int] = []
        seen: set[int] = set()
        for start, end in ranges:
            for frame_index in range(int(start), int(end) + 1):
                if frame_index in seen:
                    continue
                seen.add(frame_index)
                indices.append(frame_index)
        indices.sort()
        if max_frames is not None and len(indices) > max_frames:
            from .tracking_optimizer import _sample_indices

            sample_positions = _sample_indices(len(indices), max_frames)
            indices = [indices[position] for position in sample_positions]
        return indices

    def _resolve_video_range_video_path(self, source_root: Path, video_id: str) -> Path | None:
        from .tracking_optimizer import VIDEO_EXTENSIONS

        search_roots = [
            source_root / "input_data" / "val",
            source_root / "input_data",
            source_root,
        ]
        candidates: list[Path] = []
        seen: set[Path] = set()
        for root in search_roots:
            if not root.exists():
                continue
            direct = [
                root / f"{video_id}{extension}"
                for extension in sorted(VIDEO_EXTENSIONS | {".mjpe"})
            ]
            for path in direct:
                if path.exists() and path.is_file() and path.resolve() not in seen:
                    seen.add(path.resolve())
                    candidates.append(path)
            for path in root.rglob(f"{video_id}.*"):
                if (
                    path.is_file()
                    and path.stem == video_id
                    and path.suffix.lower() in (VIDEO_EXTENSIONS | {".mjpe"})
                    and path.resolve() not in seen
                ):
                    seen.add(path.resolve())
                    candidates.append(path)
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (0 if "input_data/val" in str(item) else 1, str(item)))[0]

    def _load_video_range_bounds_file(self) -> None:
        path_text = self.video_range_bounds_file_var.get().strip()
        self._video_range_tag_bounds = {}
        if not path_text:
            return
        path = Path(path_text).expanduser()
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        raw_entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        if isinstance(raw_entries, dict):
            self._video_range_tag_bounds = {
                str(video_id): dict(value)
                for video_id, value in raw_entries.items()
                if isinstance(value, dict)
            }

    def _write_video_range_bounds_file(self) -> Path:
        path_text = self.video_range_bounds_file_var.get().strip() or self._default_video_range_bounds_file()
        self.video_range_bounds_file_var.set(path_text)
        path = Path(path_text).expanduser()
        payload = {
            "schema_version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "manifest_path": self.video_range_manifest_var.get().strip(),
            "source_root": self.video_range_source_root_var.get().strip(),
            "entries": self._video_range_tag_bounds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def _load_video_range_manifest(self) -> None:
        manifest_text = self.video_range_manifest_var.get().strip()
        source_text = self.video_range_source_root_var.get().strip()
        if not manifest_text:
            self._show_error("Missing CSV", "Choose the video range CSV first.")
            return
        if not source_text:
            self._show_error("Missing source root", "Choose the project/source root that contains the videos.")
            return

        manifest_path = Path(manifest_text).expanduser().resolve()
        source_root = Path(source_text).expanduser().resolve()
        if not manifest_path.exists():
            self._show_error("CSV not found", str(manifest_path))
            return
        if not source_root.exists():
            self._show_error("Source root not found", str(source_root))
            return
        if not self.video_range_bounds_file_var.get().strip():
            self.video_range_bounds_file_var.set(self._default_video_range_bounds_file())

        rows: list[dict[str, object]] = []
        errors: list[str] = []
        skipped_aug_2019 = 0
        skipped_2024 = 0
        with manifest_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader, start=2):
                video_id = str(row.get("video_id") or "").strip()
                if not video_id:
                    errors.append(f"row {row_index}: missing video_id")
                    continue
                if re.match(r"^\d{2}-Aug-2019_", video_id):
                    skipped_aug_2019 += 1
                    continue
                if "2024" in video_id:
                    skipped_2024 += 1
                    continue
                try:
                    ranges = self._parse_video_frame_ranges(row)
                except Exception as exc:
                    errors.append(f"{video_id}: {exc}")
                    continue
                video_path = self._resolve_video_range_video_path(source_root, video_id)
                frames_dir = source_root / "frames" / video_id
                rows.append(
                    {
                        "video_id": video_id,
                        "row": dict(row),
                        "frame_ranges": str(row.get("frame_ranges") or ""),
                        "ranges": ranges,
                        "video_path": str(video_path) if video_path else "",
                        "frames_dir": str(frames_dir) if frames_dir.exists() else "",
                    }
                )

        self._video_range_rows = rows
        self._video_range_selected_index = 0 if rows else None
        try:
            self._load_video_range_bounds_file()
        except Exception as exc:
            errors.append(f"could not load bounds JSON: {exc}")
        self._update_video_range_selection_display()

        missing_videos = sum(1 for row in rows if not str(row.get("video_path") or ""))
        message = f"Loaded {len(rows)} video range row(s)."
        if skipped_aug_2019:
            message += f" Skipped Aug-2019 no-tag row(s): {skipped_aug_2019}."
        if skipped_2024:
            message += f" Skipped 2024 no-tag row(s): {skipped_2024}."
        if missing_videos:
            message += f" Missing videos: {missing_videos}."
        if errors:
            message += " Issues: " + "; ".join(errors[:4])
            if len(errors) > 4:
                message += f"; plus {len(errors) - 4} more"
        self.video_range_status_var.set(message)

    def _selected_video_range_row(self) -> dict[str, object] | None:
        if self._video_range_selected_index is None:
            return None
        if not (0 <= self._video_range_selected_index < len(self._video_range_rows)):
            return None
        return self._video_range_rows[self._video_range_selected_index]

    def _shift_video_range_selection(self, delta: int) -> None:
        if not self._video_range_rows:
            self.video_range_status_var.set("Load a video range CSV first.")
            return
        current = self._video_range_selected_index or 0
        self._video_range_selected_index = max(0, min(len(self._video_range_rows) - 1, current + int(delta)))
        self._update_video_range_selection_display()

    def _update_video_range_selection_display(self) -> None:
        row = self._selected_video_range_row()
        if row is None:
            self.video_range_selected_var.set("No video range manifest loaded.")
            return
        video_id = str(row.get("video_id") or "")
        video_path = str(row.get("video_path") or "not found")
        frames_dir = str(row.get("frames_dir") or "none")
        ranges = row.get("ranges") or []
        range_text = ", ".join(f"{start}-{end}" for start, end in ranges) if isinstance(ranges, list) else ""
        measured = self._video_range_tag_bounds.get(video_id)
        measured_text = "not measured"
        if isinstance(measured, dict):
            min_values = measured.get("suggested_min_marker_perimeter_rate") or []
            max_values = measured.get("suggested_max_marker_perimeter_rate") or []
            measured_text = (
                "measured | "
                f"min {self._format_optimize_sweep_values(list(min_values)) if isinstance(min_values, list) else min_values} | "
                f"max {self._format_optimize_sweep_values(list(max_values)) if isinstance(max_values, list) else max_values}"
            )
        self.video_range_selected_var.set(
            (
                f"Selected {int(self._video_range_selected_index or 0) + 1}/{len(self._video_range_rows)}: {video_id}\n"
                f"Frame range(s): {range_text or row.get('frame_ranges') or 'none'} | Bounds: {measured_text}\n"
                f"Video: {video_path}\n"
                f"Extracted frames: {frames_dir}"
            )
        )

    def _frame_index_from_extracted_frame_path(self, path: Path) -> int | None:
        match = re.search(r"(\d+)$", path.stem)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _build_video_range_measurement_frame_refs(
        self,
        row: dict[str, object],
        *,
        max_frames: int = 25,
    ) -> list[dict[str, object]]:
        from .tracking_optimizer import IMAGE_EXTENSIONS

        ranges = row.get("ranges")
        if not isinstance(ranges, list):
            raise ValueError("Selected row has no parsed frame ranges.")
        frame_indices = self._video_range_indices(ranges, max_frames=max_frames)
        if not frame_indices:
            raise ValueError("Selected row has no frame indices.")

        frames_dir_text = str(row.get("frames_dir") or "").strip()
        if frames_dir_text:
            frames_dir = Path(frames_dir_text).expanduser()
            if frames_dir.exists():
                image_by_index: dict[int, Path] = {}
                for image_path in sorted(frames_dir.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    frame_index = self._frame_index_from_extracted_frame_path(image_path)
                    if frame_index is not None:
                        image_by_index[frame_index] = image_path
                refs = [
                    {
                        "kind": "image",
                        "path": image_by_index[frame_index],
                        "label": f"{row.get('video_id')} frame {frame_index} ({image_by_index[frame_index].name})",
                    }
                    for frame_index in frame_indices
                    if frame_index in image_by_index
                ]
                if refs:
                    return refs

        video_path_text = str(row.get("video_path") or "").strip()
        if not video_path_text:
            raise FileNotFoundError("No source video was resolved for the selected row.")
        video_path = Path(video_path_text).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Source video does not exist: {video_path}")
        return [
            {
                "kind": "video",
                "path": video_path,
                "frame_index": frame_index,
                "label": f"{row.get('video_id')} frame {frame_index}",
            }
            for frame_index in frame_indices
        ]

    def _open_video_range_tag_measurement(self) -> None:
        row = self._selected_video_range_row()
        if row is None:
            self._show_error("No selected video", "Load a video range CSV first.")
            return
        try:
            frame_refs = self._build_video_range_measurement_frame_refs(row)
        except Exception as exc:
            self._show_error("Measurement frames failed", str(exc))
            return

        def apply_video_measurement(
            measurement_payload: dict[str, dict[str, object]],
            min_values: list[float],
            max_values: list[float],
        ) -> None:
            video_id = str(row.get("video_id") or "")
            small_rate = float(measurement_payload["smallest"]["perimeter_rate"])
            large_rate = float(measurement_payload["largest"]["perimeter_rate"])
            if large_rate < small_rate:
                small_rate, large_rate = large_rate, small_rate
            self._video_range_tag_bounds[video_id] = {
                "video_id": video_id,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                "frame_ranges": row.get("frame_ranges"),
                "ranges": row.get("ranges"),
                "video_path": row.get("video_path"),
                "frames_dir": row.get("frames_dir"),
                "smallest": measurement_payload["smallest"],
                "largest": measurement_payload["largest"],
                "review_perimeter_bounds": [small_rate, large_rate],
                "suggested_min_marker_perimeter_rate": list(min_values),
                "suggested_max_marker_perimeter_rate": list(max_values),
            }
            try:
                bounds_path = self._write_video_range_bounds_file()
            except Exception as exc:
                self._show_error("Save bounds failed", str(exc))
                return
            self._update_video_range_selection_display()
            self.video_range_status_var.set(f"Saved tag bounds for {video_id}: {bounds_path}")
            self.optimize_output.delete("1.0", tk.END)
            self.optimize_output.insert(
                tk.END,
                (
                    f"Saved per-video tag bounds for {video_id}\n"
                    f"Bounds JSON: {bounds_path}\n"
                    f"Measured perimeter-rate bounds: {small_rate:.6f}-{large_rate:.6f}\n"
                    f"Suggested minMarkerPerimeterRate: {self._format_optimize_sweep_values(min_values)}\n"
                    f"Suggested maxMarkerPerimeterRate: {self._format_optimize_sweep_values(max_values)}\n"
                ),
            )

        self._open_tag_perimeter_measurement_dialog(
            frame_refs=frame_refs,
            apply_callback=apply_video_measurement,
            dialog_title=f"Measure Tag Bounds: {row.get('video_id')}",
            apply_button_text="Save Bounds For Selected Video",
            intro_note=(
                "Measure the smallest and largest real tags in this video's requested frame range. "
                "These bounds will be saved for this video only, so the range-tracking workflow can tune "
                "perimeter thresholds independently across videos."
            ),
        )

    def open_video_range_bounds_workflow(
        self,
        *,
        manifest_path: str | None = None,
        source_root: str | None = None,
        bounds_file: str | None = None,
    ) -> None:
        self._open_workflow("setup")
        try:
            self.notebook.select(self.optimize_tracking_tab)
        except Exception:
            pass

        if manifest_path:
            self.video_range_manifest_var.set(str(manifest_path))
        if source_root:
            self.video_range_source_root_var.set(str(source_root))
        if bounds_file:
            self.video_range_bounds_file_var.set(str(bounds_file))
        elif manifest_path and not self.video_range_bounds_file_var.get().strip():
            self.video_range_bounds_file_var.set(self._default_video_range_bounds_file())

        if manifest_path and source_root:
            self._load_video_range_manifest()
            self.optimize_output.delete("1.0", tk.END)
            self.optimize_output.insert(
                tk.END,
                (
                    "Video range tag-bounds workflow opened from the command line.\n\n"
                    "Use Prev Video / Next Video and Measure Selected Video to mark the smallest and "
                    "largest real tags. Save bounds for each video you want calibrated, then close the GUI "
                    "and rerun the optimize-video-ranges command without --open-gui."
                ),
            )
        else:
            self.video_range_status_var.set(
                "Set the video range CSV and project/source root, then click Load CSV."
            )

    def _measurement_payload_from_role_states(
        self,
        role_states: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for role in ("smallest", "largest"):
            state = role_states[role]
            payload[role] = {
                "perimeter_px": float(state["perimeter_px"]),
                "perimeter_rate": float(state["perimeter_rate"]),
                "source": str(state.get("source") or ""),
                "frame_size": str(state.get("frame_size") or ""),
                "points": [
                    [float(point[0]), float(point[1])]
                    for point in list(state.get("points") or [])
                ],
            }
        return payload

    def _open_tag_perimeter_measurement_dialog(
        self,
        *,
        frame_refs: list[dict[str, object]] | None = None,
        apply_callback: object | None = None,
        dialog_title: str = "Measure Smallest & Largest Tags",
        apply_button_text: str = "Apply Both To Active Profile",
        intro_note: str | None = None,
    ) -> None:
        import base64
        import math

        import cv2

        from .tracking_optimizer import suggest_marker_perimeter_rate_sweeps_from_measurements

        if frame_refs is None:
            input_path = self.opt_input_path_var.get().strip()
            if not input_path:
                self._show_error("Missing input", "Set an input video or image folder path first.")
                return

            try:
                frame_refs = self._build_optimizer_measurement_frame_refs(input_path)
            except Exception as exc:
                self._show_error("Measurement frames failed", str(exc))
                return
        if not frame_refs:
            self._show_error("Measurement frames failed", "No representative frames were found.")
            return

        max_display_width = 1040
        max_display_height = 680

        dialog = tk.Toplevel(self)
        dialog.title(dialog_title)
        dialog.geometry("1180x920")
        dialog.minsize(760, 560)
        dialog.transient(self)

        outer = ttk.Frame(dialog, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        note = intro_note or (
            "Use the frame controls to find good examples. Measure the SMALLEST real tag you want BumbleBox "
            "to detect and the LARGEST real tag you expect to accept. Avoid measuring artifacts. Drag any "
            "placed point to adjust it. Scroll over the image to zoom at the cursor. Right-drag, middle-drag, "
            "or enable Pan mode to move around while zoomed in."
        )
        ttk.Label(outer, text=note, wraplength=1080, justify=tk.LEFT).pack(anchor="w")

        frame_controls = ttk.Frame(outer)
        frame_controls.pack(fill=tk.X, pady=(8, 6))
        frame_info_var = tk.StringVar(value="")
        zoom_var = tk.StringVar(value="Zoom: 100%")
        active_role_var = tk.StringVar(value="smallest")
        pan_mode_var = tk.BooleanVar(value=False)
        ttk.Button(frame_controls, text="Prev Frame", command=lambda: shift_frame(-1)).pack(side=tk.LEFT)
        ttk.Button(frame_controls, text="Next Frame", command=lambda: shift_frame(1)).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(frame_controls, text="Zoom Out", command=lambda: adjust_zoom(1 / 1.35)).pack(side=tk.LEFT)
        ttk.Button(frame_controls, text="Zoom In", command=lambda: adjust_zoom(1.35)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(frame_controls, text="Fit", command=lambda: set_zoom(1.0)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(frame_controls, text="100%", command=lambda: zoom_to_actual()).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(frame_controls, textvariable=zoom_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            frame_controls,
            text="Pan mode",
            variable=pan_mode_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            frame_controls,
            text="Measure smallest tag",
            variable=active_role_var,
            value="smallest",
            command=lambda: redraw_points(),
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            frame_controls,
            text="Measure largest tag",
            variable=active_role_var,
            value="largest",
            command=lambda: redraw_points(),
        ).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Label(frame_controls, textvariable=frame_info_var, justify=tk.LEFT).pack(side=tk.LEFT)

        canvas_frame = ttk.Frame(outer)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        canvas = tk.Canvas(
            canvas_frame,
            width=800,
            height=500,
            highlightthickness=1,
            highlightbackground="#667680",
            bg="#111111",
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        result_var = tk.StringVar(value="Measure both the smallest and largest real tags.")
        role_states: dict[str, dict[str, object]] = {
            "smallest": {
                "label": "Smallest tag",
                "points": [],
                "frame_position": None,
                "perimeter_px": None,
                "perimeter_rate": None,
                "source": "",
                "frame_size": "",
            },
            "largest": {
                "label": "Largest tag",
                "points": [],
                "frame_position": None,
                "perimeter_px": None,
                "perimeter_rate": None,
                "source": "",
                "frame_size": "",
            },
        }
        current_frame: dict[str, object] = {
            "position": 0,
            "fit_scale": 1.0,
            "zoom": 1.0,
            "scale": 1.0,
            "width": 1,
            "height": 1,
            "display_width": 1,
            "display_height": 1,
            "source": "",
            "frame_bgr": None,
        }
        drag_state: dict[str, int | None] = {"index": None}
        pan_state: dict[str, bool] = {"active": False}

        result_label = ttk.Label(outer, textvariable=result_var, justify=tk.LEFT, wraplength=1080)
        result_label.pack(anchor="w", pady=(8, 0))

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(10, 0))

        apply_button = ttk.Button(footer, text=apply_button_text, state=tk.DISABLED)

        def current_state() -> dict[str, object]:
            return role_states[active_role_var.get()]

        def current_points() -> list[tuple[float, float]]:
            state = current_state()
            if state.get("frame_position") != current_frame["position"]:
                return []
            return state["points"]  # type: ignore[return-value]

        def set_current_points(points: list[tuple[float, float]]) -> None:
            state = current_state()
            state["points"] = points
            state["frame_position"] = current_frame["position"]
            state["source"] = current_frame["source"]
            state["frame_size"] = f"{current_frame['width']}x{current_frame['height']}"

        def clamp_point(x: float, y: float) -> tuple[float, float]:
            display_width = int(current_frame["display_width"])
            display_height = int(current_frame["display_height"])
            return (
                max(0.0, min(float(display_width - 1), float(x))),
                max(0.0, min(float(display_height - 1), float(y))),
            )

        def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def nearest_point_index(x: float, y: float, *, max_distance: float = 18.0) -> int | None:
            points = current_points()
            if not points:
                return None
            scale = float(current_frame["scale"])
            display_points = [(point[0] * scale, point[1] * scale) for point in points]
            distances = [point_distance((x, y), point) for point in display_points]
            index = min(range(len(distances)), key=distances.__getitem__)
            return index if distances[index] <= max_distance else None

        def update_active_measurement() -> None:
            state = current_state()
            points = current_points()
            state["perimeter_px"] = None
            state["perimeter_rate"] = None
            if len(points) < 4:
                update_summary()
                return

            perimeter_px = 0.0
            for idx in range(4):
                perimeter_px += point_distance(points[idx], points[(idx + 1) % 4])
            perimeter_rate = perimeter_px / float(max(int(current_frame["width"]), int(current_frame["height"])))
            state["perimeter_px"] = perimeter_px
            state["perimeter_rate"] = perimeter_rate
            update_summary()

        def update_summary() -> None:
            lines = []
            for role in ("smallest", "largest"):
                state = role_states[role]
                label = str(state["label"])
                rate = state.get("perimeter_rate")
                if rate is None:
                    points = current_points() if role == active_role_var.get() else []
                    lines.append(f"{label}: not measured yet ({len(points)}/4 points on active frame)")
                    continue
                lines.append(
                    f"{label}: perimeter={float(state['perimeter_px']):.1f}px, "
                    f"rate={float(rate):.6f}, source={state['source']}"
                )

            small_rate = role_states["smallest"].get("perimeter_rate")
            large_rate = role_states["largest"].get("perimeter_rate")
            if small_rate is not None and large_rate is not None:
                min_values, max_values = suggest_marker_perimeter_rate_sweeps_from_measurements(
                    float(small_rate),
                    float(large_rate),
                )
                apply_button.configure(state=tk.NORMAL)
                lines.extend(
                    [
                        "",
                        f"Suggested minMarkerPerimeterRate: {self._format_optimize_sweep_values(min_values)}",
                        f"Suggested maxMarkerPerimeterRate: {self._format_optimize_sweep_values(max_values)}",
                    ]
                )
            else:
                apply_button.configure(state=tk.DISABLED)
            result_var.set("\n".join(lines))

        def render_current_frame() -> None:
            frame_bgr = current_frame.get("frame_bgr")
            if frame_bgr is None:
                return
            frame_width = int(current_frame["width"])
            frame_height = int(current_frame["height"])
            scale = float(current_frame["scale"])
            display_width = max(1, int(round(frame_width * scale)))
            display_height = max(1, int(round(frame_height * scale)))
            display_bgr = cv2.resize(
                frame_bgr,
                (display_width, display_height),
                interpolation=cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA,
            )
            ok, png_buffer = cv2.imencode(".png", display_bgr)
            if not ok:
                self._show_error("Measurement frame failed", "Could not encode frame preview.")
                return
            image_data = base64.b64encode(png_buffer.tobytes()).decode("ascii")
            photo = tk.PhotoImage(data=image_data)
            canvas.delete("frame")
            canvas.create_image(0, 0, image=photo, anchor=tk.NW, tags=("frame",))
            canvas.image = photo
            current_frame["display_width"] = display_width
            current_frame["display_height"] = display_height
            canvas.configure(scrollregion=(0, 0, display_width, display_height))
            zoom_var.set(
                f"Zoom: {float(current_frame['zoom']) * 100:.0f}% fit | "
                f"{float(current_frame['scale']) * 100:.0f}% actual"
            )

        def redraw_points() -> None:
            canvas.delete("measurement")
            points = current_points()
            scale = float(current_frame["scale"])
            color = "#FFEB3B" if active_role_var.get() == "smallest" else "#40C4FF"
            if len(points) >= 2:
                line_points: list[float] = []
                for x, y in points:
                    line_points.extend([x * scale, y * scale])
                if len(points) == 4:
                    line_points.extend([points[0][0] * scale, points[0][1] * scale])
                canvas.create_line(
                    *line_points,
                    fill="#00E676",
                    width=3,
                    tags=("measurement",),
                )
            for idx, (x, y) in enumerate(points, start=1):
                display_x = x * scale
                display_y = y * scale
                radius = 6
                canvas.create_oval(
                    display_x - radius,
                    display_y - radius,
                    display_x + radius,
                    display_y + radius,
                    fill=color,
                    outline="#111111",
                    width=2,
                    tags=("measurement",),
                )
                canvas.create_text(
                    display_x + 14,
                    display_y - 14,
                    text=str(idx),
                    fill="#FFFFFF",
                    font=("TkDefaultFont", 14, "bold"),
                    tags=("measurement",),
                )
            update_summary()

        def redraw_image_and_points() -> None:
            render_current_frame()
            redraw_points()

        def current_view_focus() -> tuple[float, float, float, float] | None:
            scale = float(current_frame["scale"])
            if scale <= 0:
                return None
            view_x = max(1, int(canvas.winfo_width())) / 2.0
            view_y = max(1, int(canvas.winfo_height())) / 2.0
            return (
                canvas.canvasx(view_x) / scale,
                canvas.canvasy(view_y) / scale,
                view_x,
                view_y,
            )

        def set_zoom(zoom: float, *, focus: tuple[float, float, float, float] | None = None) -> None:
            current_frame["zoom"] = max(1.0, min(8.0, float(zoom)))
            current_frame["scale"] = float(current_frame["fit_scale"]) * float(current_frame["zoom"])
            redraw_image_and_points()
            if focus is not None:
                display_width = max(1, int(current_frame["display_width"]))
                display_height = max(1, int(current_frame["display_height"]))
                target_x = max(0.0, min(display_width, focus[0] * float(current_frame["scale"])))
                target_y = max(0.0, min(display_height, focus[1] * float(current_frame["scale"])))
                canvas.xview_moveto(max(0.0, min(1.0, (target_x - focus[2]) / display_width)))
                canvas.yview_moveto(max(0.0, min(1.0, (target_y - focus[3]) / display_height)))

        def adjust_zoom(factor: float, *, focus_event: tk.Event | None = None) -> None:
            focus = None
            if focus_event is not None:
                canvas_x = canvas.canvasx(focus_event.x)
                canvas_y = canvas.canvasy(focus_event.y)
                scale = float(current_frame["scale"])
                if scale > 0:
                    focus = (canvas_x / scale, canvas_y / scale, float(focus_event.x), float(focus_event.y))
            else:
                focus = current_view_focus()
            set_zoom(float(current_frame["zoom"]) * factor, focus=focus)

        def zoom_to_actual() -> None:
            fit_scale = float(current_frame["fit_scale"])
            if fit_scale <= 0:
                return
            set_zoom(1.0 / fit_scale, focus=current_view_focus())

        def load_frame(position: int) -> None:
            position = max(0, min(len(frame_refs) - 1, int(position)))
            try:
                frame_bgr, source_label = self._load_optimizer_measurement_frame_ref(frame_refs[position])
            except Exception as exc:
                self._show_error("Measurement frame failed", str(exc))
                return

            frame_height, frame_width = frame_bgr.shape[:2]
            fit_scale = min(
                max_display_width / float(frame_width),
                max_display_height / float(frame_height),
                1.0,
            )
            current_frame.update(
                {
                    "position": position,
                    "fit_scale": fit_scale,
                    "zoom": 1.0,
                    "scale": fit_scale,
                    "width": frame_width,
                    "height": frame_height,
                    "display_width": max(1, int(round(frame_width * fit_scale))),
                    "display_height": max(1, int(round(frame_height * fit_scale))),
                    "source": source_label,
                    "frame_bgr": frame_bgr,
                }
            )
            frame_info_var.set(
                f"Frame {position + 1}/{len(frame_refs)} | {source_label} | image {frame_width}x{frame_height}"
            )
            redraw_image_and_points()

        def shift_frame(delta: int) -> None:
            load_frame(int(current_frame["position"]) + delta)

        def update_canvas_cursor(*_args: object) -> None:
            canvas.configure(cursor="fleur" if pan_mode_var.get() else "")

        pan_mode_var.trace_add("write", update_canvas_cursor)

        def begin_pan(event: tk.Event) -> str:
            pan_state["active"] = True
            drag_state["index"] = None
            canvas.scan_mark(event.x, event.y)
            canvas.configure(cursor="fleur")
            return "break"

        def drag_pan(event: tk.Event) -> str:
            if pan_state.get("active"):
                canvas.scan_dragto(event.x, event.y, gain=1)
            return "break"

        def end_pan(_event: tk.Event) -> str:
            pan_state["active"] = False
            update_canvas_cursor()
            return "break"

        def on_press(event: tk.Event) -> str | None:
            if pan_mode_var.get():
                return begin_pan(event)
            x, y = clamp_point(canvas.canvasx(event.x), canvas.canvasy(event.y))
            index = nearest_point_index(x, y)
            points = current_points()
            if index is not None:
                drag_state["index"] = index
                return None
            if len(points) >= 4:
                return None
            scale = float(current_frame["scale"])
            if current_state().get("frame_position") != current_frame["position"]:
                points = []
            points.append((x / scale, y / scale))
            set_current_points(points)
            drag_state["index"] = len(points) - 1
            redraw_points()
            return None

        def on_drag(event: tk.Event) -> str | None:
            if pan_state.get("active"):
                return drag_pan(event)
            index = drag_state.get("index")
            if index is None:
                return None
            points = current_points()
            if not (0 <= int(index) < len(points)):
                return None
            x, y = clamp_point(canvas.canvasx(event.x), canvas.canvasy(event.y))
            scale = float(current_frame["scale"])
            points[int(index)] = (x / scale, y / scale)
            set_current_points(points)
            redraw_points()
            return None

        def on_release(_event: tk.Event) -> str | None:
            if pan_state.get("active"):
                return end_pan(_event)
            drag_state["index"] = None
            update_active_measurement()
            return None

        def reset_points() -> None:
            state = current_state()
            state["points"] = []
            state["frame_position"] = current_frame["position"]
            state["perimeter_px"] = None
            state["perimeter_rate"] = None
            state["source"] = ""
            state["frame_size"] = ""
            drag_state["index"] = None
            redraw_points()

        def apply_measurement() -> None:
            small_rate = role_states["smallest"].get("perimeter_rate")
            large_rate = role_states["largest"].get("perimeter_rate")
            if small_rate is None or large_rate is None:
                self._show_error("Missing measurement", "Measure both the smallest and largest real tags first.")
                return
            min_values, max_values = suggest_marker_perimeter_rate_sweeps_from_measurements(
                float(small_rate),
                float(large_rate),
            )
            if apply_callback is not None:
                if not callable(apply_callback):
                    self._show_error("Apply failed", "Measurement callback is not callable.")
                    return
                measurement_payload = self._measurement_payload_from_role_states(role_states)
                apply_callback(measurement_payload, list(min_values), list(max_values))
                return

            profile = self.opt_profile_var.get().strip().lower() or "quick"
            profile_vars = self._opt_sweep_profile_vars.get(profile)
            if (
                not profile_vars
                or "minMarkerPerimeterRate" not in profile_vars
                or "maxMarkerPerimeterRate" not in profile_vars
            ):
                self._show_error("Apply failed", f"Could not find profile sweep fields for: {profile}")
                return
            min_text = self._format_optimize_sweep_values(min_values)
            max_text = self._format_optimize_sweep_values(max_values)
            profile_vars["minMarkerPerimeterRate"].set(min_text)
            profile_vars["maxMarkerPerimeterRate"].set(max_text)
            self._opt_review_perimeter_bounds = {
                "profile": profile,
                "bounds": (float(small_rate), float(large_rate)),
            }
            self._select_optimize_profile_tab(profile)
            self.optimize_output.delete("1.0", tk.END)
            self.optimize_output.insert(
                tk.END,
                (
                    "Applied measured tag-size range to optimizer sweep.\n"
                    f"Profile: {profile}\n"
                    f"Smallest tag perimeter rate: {float(small_rate):.6f}\n"
                    f"Largest tag perimeter rate: {float(large_rate):.6f}\n"
                    f"minMarkerPerimeterRate values: {min_text}\n"
                    f"maxMarkerPerimeterRate values: {max_text}\n\n"
                    "Next step: run optimize-tracking, then optionally run iterative refinement from the top five."
                ),
            )
            self._show_info(
                "Measurement applied",
                f"Updated {profile} marker perimeter sweeps:\nmin: {min_text}\nmax: {max_text}",
            )

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<ButtonPress-2>", begin_pan)
        canvas.bind("<B2-Motion>", drag_pan)
        canvas.bind("<ButtonRelease-2>", end_pan)
        canvas.bind("<ButtonPress-3>", begin_pan)
        canvas.bind("<B3-Motion>", drag_pan)
        canvas.bind("<ButtonRelease-3>", end_pan)
        canvas.bind("<MouseWheel>", lambda event: adjust_zoom(1.25 if event.delta > 0 else 1 / 1.25, focus_event=event))
        canvas.bind("<Button-4>", lambda event: adjust_zoom(1.25, focus_event=event))
        canvas.bind("<Button-5>", lambda event: adjust_zoom(1 / 1.25, focus_event=event))
        dialog.bind("<plus>", lambda _event: adjust_zoom(1.35))
        dialog.bind("<equal>", lambda _event: adjust_zoom(1.35))
        dialog.bind("<minus>", lambda _event: adjust_zoom(1 / 1.35))
        dialog.bind("<KeyPress-0>", lambda _event: zoom_to_actual())
        dialog.bind("<KeyPress-f>", lambda _event: set_zoom(1.0))

        ttk.Button(footer, text="Reset Active Points", command=reset_points).pack(side=tk.LEFT)
        apply_button.configure(command=apply_measurement)
        apply_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)
        load_frame(0)

    def _current_opt_review_perimeter_bounds(self, profile: str) -> tuple[float, float] | None:
        entry = getattr(self, "_opt_review_perimeter_bounds", None)
        if not isinstance(entry, dict):
            return None
        if str(entry.get("profile", "")).strip().lower() != str(profile).strip().lower():
            return None
        bounds = entry.get("bounds")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            return None
        try:
            low = float(bounds[0])
            high = float(bounds[1])
        except Exception:
            return None
        if low <= 0 or high <= low:
            return None
        return low, high

    def _start_optimize_tracking(self) -> None:
        if self._optimize_thread and self._optimize_thread.is_alive():
            self._show_info("Optimization running", "Tracking optimization is already running.")
            return

        input_path = self.opt_input_path_var.get().strip()
        if not input_path:
            self._show_error("Missing input", "Set an input video or image folder path.")
            return

        try:
            tag_size_mm = float(self.opt_tag_size_mm_var.get().strip())
            if tag_size_mm <= 0:
                raise ValueError("tag size must be > 0")

            sample_frames = int(self.opt_sample_frames_var.get().strip())
            if sample_frames <= 0:
                raise ValueError("sample frames must be >= 1")

            workers_text = self.opt_workers_var.get().strip()
            workers = int(workers_text) if workers_text else None
            if workers is not None and workers <= 0:
                raise ValueError("workers must be >= 1")

            expected_text = self.opt_expected_tags_var.get().strip()
            expected_tags = float(expected_text) if expected_text else None
            if expected_tags is not None and expected_tags <= 0:
                raise ValueError("expected tags must be > 0")

            early_stop_patience = int(self.opt_early_stop_patience_var.get().strip())
            if early_stop_patience < 0:
                raise ValueError("early stop patience must be >= 0")

            early_stop_min_improvement = float(self.opt_early_stop_min_improvement_var.get().strip())
            if early_stop_min_improvement < 0:
                raise ValueError("early stop min improvement must be >= 0")

            preview_frames = int(self.opt_preview_frames_var.get().strip())
            if preview_frames <= 0:
                raise ValueError("preview frames must be >= 1")

            top_k = int(self.opt_top_k_var.get().strip())
            if top_k <= 0:
                raise ValueError("top results must be >= 1")

            profile = self.opt_profile_var.get().strip().lower()
            if profile not in self._opt_sweep_profile_vars:
                raise ValueError(f"Unknown optimization profile: {profile}")

            profile_defaults = self._resolve_optimize_profile_sweep_defaults(profile)
            profile_vars = self._opt_sweep_profile_vars[profile]
            sweep_overrides = {}
            for key, value_type, _help_title, _help_details in OPTIMIZE_SWEEP_FIELDS:
                default_text = self._format_optimize_sweep_values(list(profile_defaults[key]))
                default_values = self._parse_csv_numeric_values(
                    default_text,
                    label=f"{key} sweep",
                    value_type=value_type,
                )
                raw_text = profile_vars[key].get().strip()
                current_values = self._parse_csv_numeric_values(
                    raw_text or default_text,
                    label=f"{key} sweep",
                    value_type=value_type,
                )
                if current_values != default_values:
                    sweep_overrides[key] = current_values
            config_for_index, _config_path_for_index = self._load_config_or_defaults()
            valid_tag_ids = self._tracking_allowed_tag_ids_from_config(config_for_index)
        except Exception as exc:
            self._show_error("Invalid settings", str(exc))
            return

        while not self._optimize_progress_q.empty():
            try:
                self._optimize_progress_q.get_nowait()
            except queue.Empty:
                break

        self._optimize_error = None
        self._optimize_warning = None
        self._optimize_extra_report = None
        self._optimize_result = None
        self._optimize_refinement_result = None
        self._optimize_applied_config = None
        self._optimize_index_result = None
        self._optimize_index_error = None
        self._optimize_top_k = top_k
        self._optimize_stop_event.clear()
        self._optimize_started_monotonic = time.monotonic()
        self._optimize_last_progress_monotonic = None
        self._optimize_latest_progress = None

        optimize_kwargs = {
            "input_path": input_path,
            "profile": profile,
            "sample_frames": sample_frames,
            "dictionary_name": self._normalize_display_dictionary(self.opt_dictionary_var.get()),
            "tag_size_mm": tag_size_mm,
            "sweep_overrides": sweep_overrides or None,
            "execution_target": self.opt_execution_target_var.get().strip(),
            "workers": workers,
            "expected_tags": expected_tags,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_improvement": early_stop_min_improvement,
            "output_dir": self.opt_output_dir_var.get().strip() or None,
            "write_preview": bool(self.opt_preview_var.get()),
            "preview_frames": preview_frames,
            "top_k": max(top_k, 10),
            "review_perimeter_bounds": self._current_opt_review_perimeter_bounds(profile),
            "valid_tag_ids": valid_tag_ids or None,
        }
        self.optimize_output.delete("1.0", tk.END)
        self.optimize_output.insert(
            tk.END,
            "Running optimize-tracking...\n"
            "Evaluated parameter combinations: 0/?\n"
            "Elapsed: 0.0s\n"
            "The live top five and timing details will appear here after the first candidate finishes.\n",
        )
        self.opt_status_var.set("Running... evaluating parameter combinations")
        self.optimize_run_btn.config(state=tk.DISABLED)
        self.optimize_refine_btn.config(state=tk.DISABLED)
        self.optimize_stop_btn.config(state=tk.NORMAL)

        self._optimize_thread = threading.Thread(
            target=self._run_optimize_tracking_worker,
            args=(optimize_kwargs, config_for_index),
            daemon=True,
        )
        self._optimize_thread.start()
        self.after(200, self._poll_optimize_tracking)

    def _stop_optimize_tracking(self) -> None:
        if not self._optimize_thread or not self._optimize_thread.is_alive():
            return
        self._optimize_stop_event.set()
        self.optimize_stop_btn.config(state=tk.DISABLED)
        self.opt_status_var.set("Stopping after current parameter combination finishes...")

    def _run_optimize_tracking_worker(self, optimize_kwargs: dict, config_for_index: dict) -> None:
        from .tracking_optimizer import optimize_tracking
        from .tracking_index import sync_optimization_result

        def progress_callback(
            done: int,
            total: int,
            top_candidates: list[dict[str, object]],
            latest_candidate: dict[str, object],
        ) -> None:
            self._optimize_progress_q.put((done, total, top_candidates, latest_candidate))

        try:
            result = optimize_tracking(
                progress_callback=progress_callback,
                stop_requested=self._optimize_stop_event.is_set,
                **optimize_kwargs,
            )
            self._optimize_result = result
            try:
                self._optimize_index_result = sync_optimization_result(config_for_index, result)
            except Exception as exc:
                self._optimize_index_error = str(exc)
        except Exception as exc:
            self._optimize_error = str(exc)
            return

    def _start_optimize_refinement(self) -> None:
        if self._optimize_thread and self._optimize_thread.is_alive():
            self._show_info("Optimization running", "Tracking optimization is already running.")
            return
        if self._optimize_result is None:
            self._show_error(
                "No optimization result",
                "Run optimize-tracking first, then refine from its top candidates.",
            )
            return

        try:
            input_path = self.opt_input_path_var.get().strip()
            if not input_path:
                input_path = str(getattr(self._optimize_result, "input_path", "") or "")
            if not input_path:
                raise ValueError("Set an input video or image folder path.")

            tag_size_mm = float(self.opt_tag_size_mm_var.get().strip())
            if tag_size_mm <= 0:
                raise ValueError("tag size must be > 0")

            sample_frames = int(self.opt_sample_frames_var.get().strip())
            if sample_frames <= 0:
                raise ValueError("sample frames must be >= 1")

            rounds = int(self.opt_refinement_rounds_var.get().strip())
            if rounds <= 0:
                raise ValueError("iterative refinement rounds must be >= 1")

            seed_count = int(self.opt_refinement_seed_count_var.get().strip())
            if seed_count <= 0:
                raise ValueError("seed candidates must be >= 1")

            validation_multiplier = float(self.opt_refinement_validation_multiplier_var.get().strip())
            if validation_multiplier <= 0:
                raise ValueError("validation frame multiplier must be > 0")
            validation_sample_frames = max(1, int(round(sample_frames * validation_multiplier)))

            workers_text = self.opt_workers_var.get().strip()
            workers = int(workers_text) if workers_text else None
            if workers is not None and workers <= 0:
                raise ValueError("workers must be >= 1")

            expected_text = self.opt_expected_tags_var.get().strip()
            expected_tags = float(expected_text) if expected_text else None
            if expected_tags is not None and expected_tags <= 0:
                raise ValueError("expected tags must be > 0")

            preview_frames = int(self.opt_preview_frames_var.get().strip())
            if preview_frames <= 0:
                raise ValueError("preview frames must be >= 1")

            top_k = int(self.opt_top_k_var.get().strip())
            if top_k <= 0:
                raise ValueError("top results must be >= 1")

            profile = self.opt_profile_var.get().strip().lower()
            seed_source = self.opt_refinement_seed_source_var.get().strip().lower() or "mean_detection"
            if seed_source not in {"mean_detection", "score"}:
                raise ValueError("refinement seed source must be mean_detection or score")
            if seed_source == "mean_detection":
                source_candidates = list(getattr(self._optimize_result, "top_detection_candidates", []) or [])
                source_label = "top mean-detection candidates"
            else:
                source_candidates = list(getattr(self._optimize_result, "top_candidates", []) or [])
                source_label = "top score candidates"
            seed_params = [
                dict(candidate.params)
                for candidate in source_candidates[:seed_count]
            ]
            if not seed_params:
                raise ValueError(f"The previous optimization result has no {source_label} to refine.")
            config_for_index, _config_path_for_index = self._load_config_or_defaults()
            valid_tag_ids = self._tracking_allowed_tag_ids_from_config(config_for_index)
        except Exception as exc:
            self._show_error("Invalid refinement settings", str(exc))
            return

        while not self._optimize_progress_q.empty():
            try:
                self._optimize_progress_q.get_nowait()
            except queue.Empty:
                break

        self._optimize_error = None
        self._optimize_warning = None
        self._optimize_extra_report = None
        self._optimize_refinement_result = None
        self._optimize_applied_config = None
        self._optimize_index_result = None
        self._optimize_index_error = None
        self._optimize_top_k = top_k
        self._optimize_stop_event.clear()
        self._optimize_started_monotonic = time.monotonic()
        self._optimize_last_progress_monotonic = None
        self._optimize_latest_progress = None

        refine_kwargs = {
            "input_path": input_path,
            "seed_params": seed_params,
            "rounds": rounds,
            "seed_candidate_count": seed_count,
            "sample_frames": sample_frames,
            "validation_sample_frames": validation_sample_frames,
            "profile": profile,
            "dictionary_name": self._normalize_display_dictionary(self.opt_dictionary_var.get()),
            "tag_size_mm": tag_size_mm,
            "execution_target": self.opt_execution_target_var.get().strip(),
            "workers": workers,
            "expected_tags": expected_tags,
            "output_dir": self.opt_output_dir_var.get().strip() or None,
            "write_preview": bool(self.opt_preview_var.get()),
            "preview_frames": preview_frames,
            "top_k": max(top_k, seed_count, 10),
            "review_perimeter_bounds": self._current_opt_review_perimeter_bounds(profile),
            "valid_tag_ids": valid_tag_ids or None,
        }

        self.optimize_output.delete("1.0", tk.END)
        self.optimize_output.insert(
            tk.END,
            "Running iterative tracking refinement...\n"
            f"Seed candidates: {len(seed_params)}\n"
            f"Seed source: {source_label}\n"
            f"Refinement rounds: {rounds}\n"
            f"Validation sample frames: {validation_sample_frames}\n"
            "Live round timing and top-five details will appear after the first candidate finishes.\n",
        )
        self.opt_status_var.set(f"Running iterative refinement from {source_label}...")
        self.optimize_run_btn.config(state=tk.DISABLED)
        self.optimize_refine_btn.config(state=tk.DISABLED)
        self.optimize_stop_btn.config(state=tk.NORMAL)

        self._optimize_thread = threading.Thread(
            target=self._run_optimize_refinement_worker,
            args=(refine_kwargs, config_for_index),
            daemon=True,
        )
        self._optimize_thread.start()
        self.after(200, self._poll_optimize_tracking)

    def _run_optimize_refinement_worker(self, refine_kwargs: dict, config_for_index: dict) -> None:
        from .tracking_optimizer import (
            format_iterative_refinement_report,
            optimize_tracking_iterative_refinement,
        )
        from .tracking_index import sync_optimization_result

        def progress_callback(
            done: int,
            total: int,
            top_candidates: list[dict[str, object]],
            latest_candidate: dict[str, object],
        ) -> None:
            self._optimize_progress_q.put((done, total, top_candidates, latest_candidate))

        try:
            refinement_result = optimize_tracking_iterative_refinement(
                progress_callback=progress_callback,
                stop_requested=self._optimize_stop_event.is_set,
                **refine_kwargs,
            )
            self._optimize_refinement_result = refinement_result
            self._optimize_result = refinement_result.final_result
            self._optimize_extra_report = format_iterative_refinement_report(refinement_result)
            try:
                self._optimize_index_result = sync_optimization_result(config_for_index, refinement_result.final_result)
            except Exception as exc:
                self._optimize_index_error = str(exc)
        except Exception as exc:
            self._optimize_error = str(exc)
            return

    def _prompt_optimize_review(self) -> None:
        if self._optimize_result is None:
            return
        review_manifest = getattr(self._optimize_result, "review_manifest_json_path", None)
        if not review_manifest:
            return
        review_path = Path(review_manifest)
        if not review_path.exists():
            return
        confirmed = messagebox.askyesno(
            "Review optimization candidates",
            "Do you want to review the top five score candidates and the top five mean-detection candidates?",
            parent=self,
        )
        if confirmed:
            self._open_optimize_review_dialog(review_path)

    def _apply_selected_optimization_candidate(self, params: dict[str, float | int]) -> None:
        from .tracking_optimizer import apply_best_params_to_config
        from .tracking_index import format_local_index_result, sync_optimization_result

        index_note = ""
        try:
            config_path_obj = Path(self.config_path_var.get().strip() or str(DEFAULT_USER_CONFIG_PATH))
            if config_path_obj.exists():
                config = load_config(config_path_obj)
            else:
                config = load_defaults()

            updated = apply_best_params_to_config(config, params)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path_obj,
                updated,
                reason="optimize_tracking_apply_selected_candidate",
            )
            self._optimize_applied_config = str(config_path_obj)
            self._load_config_into_editor()
            if self._optimize_result is not None:
                try:
                    index_result = sync_optimization_result(
                        updated,
                        self._optimize_result,
                        selected_params=dict(params),
                        selected_label="selected_from_review",
                    )
                    index_note = "\n\n" + format_local_index_result(index_result)
                except Exception as exc:
                    index_note = f"\n\nLocal tracking index update failed: {exc}"
        except Exception as exc:
            self._show_error("Optimization apply failed", str(exc))
            return

        message = f"Applied selected optimization parameters to:\n{config_path_obj}"
        history_note = self._format_config_history_note(snapshot_path, history_warning)
        if history_note:
            message += f"\n\n{history_note}"
        if index_note:
            message += index_note
        self._show_info("Optimization parameters applied", message)

        self.optimize_output.insert(
            tk.END,
            f"\n\nApplied selected optimization parameters to config: {config_path_obj}{index_note}",
        )

    def _open_optimize_review_dialog(self, manifest_path: Path) -> None:
        try:
            with manifest_path.open() as f:
                manifest = json.load(f)
        except Exception as exc:
            self._show_error("Review load failed", str(exc))
            return

        candidates = manifest.get("candidates") or []
        if not candidates:
            self._show_error("Review unavailable", "No review candidates were generated for this optimization run.")
            return

        if self._optimize_review_dialog is not None and self._optimize_review_dialog.winfo_exists():
            self._optimize_review_dialog.destroy()

        dialog = tk.Toplevel(self)
        self._optimize_review_dialog = dialog
        dialog.title("Optimization Candidate Review")
        dialog.geometry("1440x980")
        dialog.minsize(1080, 760)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_optimize_review_dialog(dialog))

        outer = ttk.Frame(dialog, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)

        candidate_options = []
        candidate_lookup: dict[str, dict] = {}
        for index, candidate in enumerate(candidates, start=1):
            review_label = str(candidate.get("review_label") or "Review candidate")
            label = (
                f"{review_label} | "
                f"#{candidate['rank']} "
                f"score={candidate['score']:.3f} "
                f"detected={candidate['mean_detected']:.2f} "
                f"fps={candidate['eval_fps']:.1f}"
            )
            if label in candidate_lookup:
                label = f"{label} ({index})"
            candidate_options.append(label)
            candidate_lookup[label] = candidate

        candidate_var = tk.StringVar(value=candidate_options[0])
        frame_scale_var = tk.IntVar(value=1)
        summary_var = tk.StringVar(value="")
        frame_info_var = tk.StringVar(value="")

        ttk.Label(header, text="Candidate").grid(row=0, column=0, sticky="w")
        candidate_combo = ttk.Combobox(
            header,
            textvariable=candidate_var,
            values=candidate_options,
            state="readonly",
            width=56,
        )
        candidate_combo.grid(row=0, column=1, sticky="w", padx=8)

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        sidebar = ttk.Frame(body)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(sidebar, text="Candidate summary").pack(anchor="w")
        ttk.Label(
            sidebar,
            textvariable=summary_var,
            justify=tk.LEFT,
            wraplength=360,
        ).pack(anchor="w", pady=(4, 8))

        ttk.Label(sidebar, text="Parameters").pack(anchor="w")
        params_text = tk.Text(sidebar, height=16, width=42, wrap=tk.WORD)
        params_text.pack(fill=tk.BOTH, expand=False)
        params_text.configure(state=tk.DISABLED)

        viewer = ttk.Frame(body)
        viewer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        controls = ttk.Frame(viewer)
        controls.pack(fill=tk.X, pady=(0, 8))

        image_label = ttk.Label(viewer)
        image_label.pack(fill=tk.BOTH, expand=True)
        ttk.Label(viewer, textvariable=frame_info_var, justify=tk.LEFT).pack(anchor="w", pady=(6, 0))

        def _selected_candidate() -> dict:
            return candidate_lookup[candidate_var.get()]

        state: dict[str, object] = {"photo": None}

        def _set_params_text(candidate: dict) -> None:
            params_text.configure(state=tk.NORMAL)
            params_text.delete("1.0", tk.END)
            params_text.insert(tk.END, json.dumps(candidate["params"], indent=2, sort_keys=True))
            params_text.configure(state=tk.DISABLED)

        def _update_frame(*_args: object) -> None:
            candidate = _selected_candidate()
            frames = candidate.get("frames") or []
            if not frames:
                image_label.configure(image="", text="No review frames available.")
                frame_info_var.set("No review frames available.")
                return

            frame_idx = max(0, min(len(frames) - 1, int(frame_scale_var.get()) - 1))
            if frame_scale_var.get() != frame_idx + 1:
                frame_scale_var.set(frame_idx + 1)
            frame_entry = frames[frame_idx]

            photo = tk.PhotoImage(file=frame_entry["path"])
            image_label.configure(image=photo, text="")
            image_label.image = photo
            state["photo"] = photo
            frame_info_var.set(
                "Frame "
                f"{frame_idx + 1}/{len(frames)} | "
                f"sample index {frame_entry['sample_position']} | "
                f"source index {frame_entry['source_index']} | "
                f"detected {frame_entry['detected_count']} | "
                f"rejected {frame_entry['rejected_count']} | "
                f"small flags {frame_entry.get('below_min_perimeter_count', 0)} | "
                f"large flags {frame_entry.get('above_max_perimeter_count', 0)}"
            )

        def _update_candidate(*_args: object) -> None:
            candidate = _selected_candidate()
            avg_frame_ms = float(candidate.get("average_frame_ms", 0.0))
            frames = candidate.get("frames") or []
            summary_var.set(
                f"Review role: {candidate.get('review_label', 'Review candidate')}\n"
                f"Rank: {candidate['rank']}\n"
                f"Score: {candidate['score']:.4f}\n"
                f"Mean detections/frame: {candidate['mean_detected']:.3f}\n"
                f"Mean rejected/frame: {candidate['mean_rejected']:.3f}\n"
                f"Stability: {candidate['stability']:.3f}\n"
                f"Unique IDs: {candidate['unique_ids']}\n"
                f"Review frames: {len(frames)}\n"
                f"Small perimeter flags: {candidate.get('below_min_perimeter_count', 0)}\n"
                f"Large perimeter flags: {candidate.get('above_max_perimeter_count', 0)}\n"
                f"Average frame time: {avg_frame_ms:.2f} ms\n"
                f"Eval FPS: {candidate['eval_fps']:.2f}"
            )
            _set_params_text(candidate)
            frame_slider.configure(to=max(1, len(frames)))
            frame_scale_var.set(1)
            _update_frame()

        def _jump_frame(frame_number: int) -> None:
            candidate = _selected_candidate()
            frame_count = max(1, len(candidate.get("frames") or []))
            next_value = max(1, min(frame_count, frame_number))
            frame_scale_var.set(next_value)
            _update_frame()

        def _shift_frame(delta: int) -> None:
            candidate = _selected_candidate()
            frame_count = max(1, len(candidate.get("frames") or []))
            next_value = max(1, min(frame_count, int(frame_scale_var.get()) + delta))
            frame_scale_var.set(next_value)
            _update_frame()

        ttk.Label(controls, text="Review frame").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="First", command=lambda: _jump_frame(1)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Prev", command=lambda: _shift_frame(-1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Next", command=lambda: _shift_frame(1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Last", command=lambda: _jump_frame(10**9)).pack(side=tk.LEFT, padx=(6, 0))
        frame_slider = tk.Scale(
            controls,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            variable=frame_scale_var,
            showvalue=True,
            command=_update_frame,
            length=420,
        )
        frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            footer,
            text="Apply These Optimization Parameters",
            command=lambda: self._apply_selected_optimization_candidate(
                dict(_selected_candidate()["params"])
            ),
        ).pack(side=tk.LEFT)
        ttk.Button(
            footer,
            text="Close",
            command=lambda: self._close_optimize_review_dialog(dialog),
        ).pack(side=tk.RIGHT)

        candidate_combo.bind("<<ComboboxSelected>>", _update_candidate)
        _update_candidate()

    def _close_optimize_review_dialog(self, dialog: tk.Toplevel) -> None:
        if self._optimize_review_dialog is dialog:
            self._optimize_review_dialog = None
        dialog.destroy()

    def _format_optimize_live_progress(
        self,
        done: int,
        total: int,
        top_candidates: list[dict[str, object]],
        latest_candidate: dict[str, object] | None = None,
    ) -> str:
        from .tracking_optimizer import format_candidate_results_table

        now = time.monotonic()
        elapsed_s = (
            now - self._optimize_started_monotonic
            if self._optimize_started_monotonic is not None
            else 0.0
        )
        since_last_s = (
            now - self._optimize_last_progress_monotonic
            if self._optimize_last_progress_monotonic is not None
            else elapsed_s
        )
        last_runtime_s = (
            float(latest_candidate.get("runtime_seconds", 0.0) or 0.0)
            if latest_candidate
            else 0.0
        )
        avg_wall_s = elapsed_s / done if done > 0 else 0.0
        remaining = max(0, total - done)
        eta_s = avg_wall_s * remaining if done > 0 else 0.0
        stage = str(latest_candidate.get("stage", "")).strip() if latest_candidate else ""
        highest_detection = (
            latest_candidate.get("highest_detection_candidate")
            if isinstance(latest_candidate, dict)
            else None
        )
        top_detection_candidates = (
            latest_candidate.get("top_detection_candidates")
            if isinstance(latest_candidate, dict)
            else None
        )
        if not isinstance(top_detection_candidates, list):
            top_detection_candidates = [highest_detection] if isinstance(highest_detection, dict) else []

        lines = [
            "Running optimize-tracking..." if not stage else f"Running optimize-tracking... {stage}",
            f"Evaluated parameter combinations: {done}/{total}",
            (
                f"Elapsed: {self._format_seconds(elapsed_s)} | "
                f"Since last completed: {self._format_seconds(since_last_s)}"
            ),
            (
                f"Last completed combination runtime: "
                f"{self._format_seconds(last_runtime_s) if last_runtime_s > 0 else 'n/a'} | "
                f"Average wall time/combination: "
                f"{self._format_seconds(avg_wall_s) if avg_wall_s > 0 else 'n/a'} | "
                f"ETA: {self._format_seconds(eta_s) if eta_s > 0 else 'n/a'}"
            ),
            "",
            "Current top score candidates:",
        ]
        if not top_candidates:
            lines.append("No candidates have finished yet.")
            return "\n".join(lines)

        lines.append(format_candidate_results_table(top_candidates, ranking="score", max_rows=5))

        lines.extend(
            [
                "",
                "Table notes: detect/reject/std are average counts per sampled frame; "
                "test_s is wall time for that parameter set.",
                "Parameter columns: minPerim/maxPerim are marker perimeter-rate bounds, "
                "poly=polygonalApproxAccuracyRate, const=adaptiveThreshConstant.",
            ]
        )
        if top_detection_candidates:
            lines.extend(
                [
                    "",
                    "Top mean-detection candidates so far:",
                    format_candidate_results_table(
                        top_detection_candidates,
                        ranking="detection",
                        max_rows=5,
                    ),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remainder = seconds - (minutes * 60)
        if minutes < 60:
            return f"{minutes}m {remainder:04.1f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes:02d}m {remainder:04.1f}s"

    def _poll_optimize_tracking(self) -> None:
        from .tracking_optimizer import format_optimization_report
        from .tracking_index import format_local_index_result

        latest_progress = None
        while True:
            try:
                latest_progress = self._optimize_progress_q.get_nowait()
            except queue.Empty:
                break

        if latest_progress:
            done, total, top_candidates, latest_candidate = latest_progress
            self._optimize_latest_progress = latest_progress
            self._optimize_last_progress_monotonic = time.monotonic()
            if self._optimize_stop_event.is_set():
                self.opt_status_var.set(
                    f"Stopping... evaluated {done}/{total} parameter combinations"
                )
            else:
                self.opt_status_var.set(f"Running... evaluated {done}/{total} parameter combinations")
            self._replace_text_preserving_scroll(
                self.optimize_output,
                self._format_optimize_live_progress(done, total, top_candidates, latest_candidate),
            )
        elif self._optimize_thread and self._optimize_thread.is_alive() and self._optimize_latest_progress:
            done, total, top_candidates, latest_candidate = self._optimize_latest_progress
            self._replace_text_preserving_scroll(
                self.optimize_output,
                self._format_optimize_live_progress(done, total, top_candidates, latest_candidate),
            )
        elif self._optimize_thread and self._optimize_thread.is_alive() and self._optimize_started_monotonic:
            elapsed_s = time.monotonic() - self._optimize_started_monotonic
            self._replace_text_preserving_scroll(
                self.optimize_output,
                "Running optimize-tracking...\n"
                "Evaluated parameter combinations: 0/?\n"
                f"Elapsed: {self._format_seconds(elapsed_s)}\n"
                "Waiting for the first parameter combination to finish.\n",
            )

        if self._optimize_thread and self._optimize_thread.is_alive():
            self.after(1000, self._poll_optimize_tracking)
            return

        self.optimize_run_btn.config(state=tk.NORMAL)
        self.optimize_refine_btn.config(state=(tk.NORMAL if self._optimize_result is not None else tk.DISABLED))
        self.optimize_stop_btn.config(state=tk.DISABLED)
        self._optimize_started_monotonic = None
        self._optimize_last_progress_monotonic = None
        self._optimize_latest_progress = None

        if self._optimize_error:
            self.opt_status_var.set("Failed")
            self.optimize_output.insert(tk.END, f"\nError: {self._optimize_error}\n")
            self._show_error("optimize-tracking failed", self._optimize_error)
            return

        if self._optimize_result is None:
            self.opt_status_var.set("No result")
            self.optimize_output.insert(tk.END, "\nOptimization ended without a result.\n")
            return

        stopped_by_user = bool(getattr(self._optimize_result, "stopped_by_user", False))
        self.opt_status_var.set("Ended early" if stopped_by_user else "Completed")
        final_text_parts = []
        if self._optimize_extra_report:
            final_text_parts.append(self._optimize_extra_report)
            final_text_parts.append("Final Optimization Report\n-------------------------")
        final_text_parts.append(format_optimization_report(self._optimize_result, top_k=self._optimize_top_k))
        if self._optimize_index_result is not None:
            final_text_parts.append("Local Tracking Index\n--------------------\n" + format_local_index_result(self._optimize_index_result))
        if self._optimize_index_error:
            final_text_parts.append(f"Local tracking index update failed: {self._optimize_index_error}")
        if self._optimize_warning:
            self.opt_status_var.set("Completed with warning")
            final_text_parts.append(f"Warning: {self._optimize_warning}")
        self._replace_text_preserving_scroll(self.optimize_output, "\n\n".join(final_text_parts))
        self.notebook.select(self.optimize_tracking_tab)
        self._prompt_optimize_review()

    def _build_nest_label_tab(self) -> None:
        top = ttk.Frame(self.nest_label_tab)
        top.pack(fill=tk.X)

        self.nest_folder_var = tk.StringVar()
        self.nest_script_var = tk.StringVar(value=str(default_script_path()))
        self.nest_labelmerc_var = tk.StringVar(value="")
        self.nest_python_var = tk.StringVar(value="")

        ttk.Label(top, text="Image folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_folder_var, width=90).grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="Label script").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_script_var, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="labelmerc (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_labelmerc_var, width=90).grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="Label Python (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_python_var, width=90).grid(row=3, column=1, sticky="ew", padx=8, pady=4)

        buttons = ttk.Frame(top)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Check Environment", command=self._run_nest_label_check).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Launch Nest Labeling", command=self._launch_nest_labeling).pack(side=tk.LEFT, padx=8)

        top.columnconfigure(1, weight=1)

        self.nest_label_output = self._create_results_section(
            self.nest_label_tab,
            title="Nest Labeling Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=14,
        )

    def _run_nest_label_check(self) -> None:
        try:
            folder = self.nest_folder_var.get().strip() or None
            script = self.nest_script_var.get().strip() or None
            labelmerc = self.nest_labelmerc_var.get().strip() or None
            python_exe = self.nest_python_var.get().strip() or None
            env = check_nest_labeling_environment(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
                python_executable=python_exe,
            )
            self.nest_label_output.delete("1.0", tk.END)
            self.nest_label_output.insert(tk.END, format_nest_labeling_environment(env))
        except Exception as exc:
            self._show_error("Nest labeling check failed", str(exc))

    def _launch_nest_labeling(self) -> None:
        folder = self.nest_folder_var.get().strip()
        if not folder:
            self._show_error("Missing folder", "Set the image folder first.")
            return

        try:
            script = self.nest_script_var.get().strip() or None
            labelmerc = self.nest_labelmerc_var.get().strip() or None
            python_exe = self.nest_python_var.get().strip() or None
            process = launch_nest_labeling(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
                python_executable=python_exe,
            )
            self._nest_label_pid = process.pid
            command = build_nest_labeling_command(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
                python_executable=python_exe,
            )
            self.nest_label_output.delete("1.0", tk.END)
            self.nest_label_output.insert(
                tk.END,
                (
                    f"Launched nest labeling (pid {process.pid}).\n"
                    f"Command: {' '.join(shlex.quote(part) for part in command)}\n\n"
                    "If LabelMe does not open, run 'Check Environment' and install missing dependencies."
                ),
            )
            self.notebook.select(self.nest_label_tab)
        except Exception as exc:
            self._show_error("Launch failed", str(exc))

    def _build_fleet_tab(self) -> None:
        top = ttk.Frame(self.fleet_tab)
        top.pack(fill=tk.X)

        self.fleet_queen_host_var = tk.StringVar(value="")
        self.fleet_ssh_user_var = tk.StringVar(value="pi")
        self.fleet_identity_var = tk.StringVar(value="~/.ssh/bbx_fleet_ed25519")
        self.fleet_queen_mode_var = tk.StringVar(value="interface_only")

        self.fleet_worker_host_var = tk.StringVar(value="")
        self.fleet_worker_name_var = tk.StringVar(value="")
        self.fleet_worker_user_var = tk.StringVar(value="pi")
        self.fleet_worker_port_var = tk.StringVar(value="22")
        self.fleet_worker_data_root_var = tk.StringVar(value="/mnt/bumblebox/data")
        self.fleet_worker_unit_prefix_var = tk.StringVar(value="bumblebox-v2")
        self.fleet_worker_enabled_var = tk.BooleanVar(value=True)
        self.fleet_install_key_var = tk.BooleanVar(value=False)

        self.fleet_filter_var = tk.StringVar(value="")
        self.fleet_include_disabled_var = tk.BooleanVar(value=False)
        self.fleet_probe_reachability_var = tk.BooleanVar(value=True)
        self.fleet_discovery_ping_var = tk.BooleanVar(value=True)
        self.fleet_media_schedule_enabled_var = tk.BooleanVar(value=True)
        self.fleet_media_pull_interval_var = tk.StringVar(value="30")
        self.fleet_media_track_interval_var = tk.StringVar(value="60")
        self.fleet_pull_output_root_var = tk.StringVar(value="")
        self.fleet_pull_max_total_var = tk.StringVar(value="200")
        self.fleet_pull_max_per_worker_var = tk.StringVar(value="1")
        self.fleet_pull_cooldown_var = tk.StringVar(value="60")
        self.fleet_pull_max_load_var = tk.StringVar(value="3.0")
        self.fleet_pull_min_mem_var = tk.StringVar(value="0.8")
        self.fleet_pull_allow_active_var = tk.BooleanVar(value=False)
        self.fleet_pull_no_visual_var = tk.BooleanVar(value=False)
        self.fleet_pull_dry_run_var = tk.BooleanVar(value=False)

        queen_frame = ttk.LabelFrame(top, text="Queen Setup", padding=8)
        queen_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(queen_frame, text="Queen host/IP (optional)").grid(row=0, column=0, sticky="w")
        ttk.Entry(queen_frame, textvariable=self.fleet_queen_host_var, width=28).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(queen_frame, text="Default SSH user").grid(row=1, column=0, sticky="w")
        ttk.Entry(queen_frame, textvariable=self.fleet_ssh_user_var, width=12).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        queen_advanced = ttk.Frame(queen_frame)
        queen_advanced.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(queen_advanced, text="Identity file").grid(row=0, column=0, sticky="w")
        ttk.Entry(queen_advanced, textvariable=self.fleet_identity_var, width=36).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        queen_advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(queen_advanced)
        ttk.Label(
            queen_frame,
            text="Queen mode: choose interface-only or active BumbleBox behavior.",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ttk.Radiobutton(
            queen_frame,
            text="--queen-interface-only (controller only; no local recording/tracking)",
            value="interface_only",
            variable=self.fleet_queen_mode_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            queen_frame,
            text="--queen-bbox-active (queen also records/tracks locally)",
            value="bbox_active",
            variable=self.fleet_queen_mode_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(queen_frame, text="Init Queen + Save Config", command=self._fleet_init_queen).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        queen_frame.columnconfigure(1, weight=1)

        worker_frame = ttk.LabelFrame(top, text="Enroll Worker", padding=8)
        worker_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(worker_frame, text="Worker host/IP").grid(row=0, column=0, sticky="w")
        ttk.Entry(worker_frame, textvariable=self.fleet_worker_host_var, width=26).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(worker_frame, text="Name (optional)").grid(row=1, column=0, sticky="w")
        ttk.Entry(worker_frame, textvariable=self.fleet_worker_name_var, width=20).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(worker_frame, text="SSH user").grid(row=2, column=0, sticky="w")
        ttk.Entry(worker_frame, textvariable=self.fleet_worker_user_var, width=12).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(worker_frame, text="Worker enabled", variable=self.fleet_worker_enabled_var).grid(row=3, column=0, sticky="w", pady=(4, 0))

        worker_advanced = ttk.LabelFrame(worker_frame, text="Advanced Worker Options", padding=6)
        worker_advanced.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(worker_advanced, text="SSH port").grid(row=0, column=0, sticky="w")
        ttk.Entry(worker_advanced, textvariable=self.fleet_worker_port_var, width=8).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(worker_advanced, text="Worker data_root").grid(row=1, column=0, sticky="w")
        ttk.Entry(worker_advanced, textvariable=self.fleet_worker_data_root_var, width=36).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(worker_advanced, text="Unit prefix").grid(row=2, column=0, sticky="w")
        ttk.Entry(worker_advanced, textvariable=self.fleet_worker_unit_prefix_var, width=18).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(worker_advanced, text="Attempt key install now", variable=self.fleet_install_key_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        worker_advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(worker_advanced)

        ttk.Button(worker_frame, text="Enroll Worker + Save Config", command=self._fleet_enroll_worker).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        worker_frame.columnconfigure(1, weight=1)

        status_frame = ttk.LabelFrame(top, text="Fleet Status", padding=8)
        status_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        ttk.Label(status_frame, text="Filter worker (optional)").grid(row=0, column=0, sticky="w")
        ttk.Entry(status_frame, textvariable=self.fleet_filter_var, width=26).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(status_frame, text="Include disabled workers", variable=self.fleet_include_disabled_var).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        ttk.Button(status_frame, text="Run Fleet Status", command=self._fleet_run_status).grid(
            row=0, column=3, sticky="w", padx=(12, 0)
        )
        pull_frame = ttk.LabelFrame(status_frame, text="Queen Latest Sync + Tracking", padding=6)
        pull_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(
            pull_frame,
            text=(
                "Use Pull Latest frequently to keep each worker's latest_video current. "
                "Use Track Latest hourly to update each worker's latest_tracked video."
            ),
            wraplength=760,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Checkbutton(
            pull_frame,
            text="Enable queen media schedule (systemd-write will generate pull+track timers)",
            variable=self.fleet_media_schedule_enabled_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))
        ttk.Label(pull_frame, text="Pull interval (minutes)").grid(row=2, column=0, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_media_pull_interval_var, width=8).grid(
            row=2, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Track interval (minutes)").grid(row=2, column=2, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_media_track_interval_var, width=8).grid(
            row=2, column=3, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Output root (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_output_root_var, width=42).grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Max videos total").grid(row=4, column=0, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_max_total_var, width=8).grid(
            row=4, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Max per worker").grid(row=4, column=2, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_max_per_worker_var, width=8).grid(
            row=4, column=3, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Track cooldown (minutes)").grid(row=5, column=0, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_cooldown_var, width=8).grid(
            row=5, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Max queen load (1m)").grid(row=5, column=2, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_max_load_var, width=8).grid(
            row=5, column=3, sticky="w", padx=8, pady=3
        )
        ttk.Label(pull_frame, text="Min queen mem (GB)").grid(row=6, column=0, sticky="w")
        ttk.Entry(pull_frame, textvariable=self.fleet_pull_min_mem_var, width=8).grid(
            row=6, column=1, sticky="w", padx=8, pady=3
        )
        options = ttk.Frame(pull_frame)
        options.grid(row=7, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            options,
            text="Allow while queen BumbleBox pipeline is active",
            variable=self.fleet_pull_allow_active_var,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            options,
            text="No tracked-video rendering",
            variable=self.fleet_pull_no_visual_var,
        ).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(
            options,
            text="Dry run",
            variable=self.fleet_pull_dry_run_var,
        ).pack(side=tk.LEFT, padx=10)
        actions = ttk.Frame(pull_frame)
        actions.grid(row=8, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(actions, text="Save Schedule To Config", command=self._fleet_save_media_schedule).pack(side=tk.LEFT)
        ttk.Button(actions, text="Run Pull Latest", command=self._fleet_queen_pull_latest).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Run Track Latest", command=self._fleet_queen_track_latest).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Run Pull + Track", command=self._fleet_queen_pull_track).pack(side=tk.LEFT, padx=8)
        pull_frame.columnconfigure(1, weight=1)
        pull_frame.columnconfigure(3, weight=1)
        self._register_advanced_widget(pull_frame)

        latest_controls = ttk.Frame(status_frame)
        latest_controls.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            latest_controls,
            text="Probe worker reachability in latest-status refresh",
            variable=self.fleet_probe_reachability_var,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            latest_controls,
            text="Use ping in LAN discovery",
            variable=self.fleet_discovery_ping_var,
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            latest_controls,
            text="Refresh Latest/Online Matrix",
            command=self._fleet_refresh_latest_status,
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            latest_controls,
            text="Discover LAN Hosts",
            command=self._fleet_discover_lan,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            latest_controls,
            text="Auto-Set Max Videos From Workers",
            command=self._fleet_set_media_capacity_from_workers,
        ).pack(side=tk.LEFT, padx=6)

        latest_frame = ttk.LabelFrame(status_frame, text="Latest Pulled vs Latest Tracked (Per Worker)", padding=6)
        latest_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        latest_columns = ("worker", "host", "online", "latest_video", "latest_tracked", "lag_min", "state")
        self.fleet_latest_tree = ttk.Treeview(latest_frame, columns=latest_columns, show="headings", height=6)
        for col, label, width in [
            ("worker", "Worker", 120),
            ("host", "Host", 140),
            ("online", "Online", 70),
            ("latest_video", "Latest Video Pulled", 170),
            ("latest_tracked", "Latest Tracked", 170),
            ("lag_min", "Lag (min)", 90),
            ("state", "State", 170),
        ]:
            self.fleet_latest_tree.heading(col, text=label)
            self.fleet_latest_tree.column(col, width=width, stretch=(col in {"host", "state"}))
        self.fleet_latest_tree.pack(fill=tk.X, expand=False)

        self.fleet_latest_detail = self._create_results_section(
            latest_frame,
            title="Latest Matrix Details",
            text_height=7,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=3,
            max_text_lines=10,
            fill=tk.BOTH,
            expand=False,
            pady=(6, 0),
        )
        status_frame.columnconfigure(1, weight=1)
        status_frame.rowconfigure(3, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.fleet_output = self._create_results_section(
            self.fleet_tab,
            title="Fleet Results",
            text_height=11,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=4,
            max_text_lines=16,
        )

    def _fleet_init_queen(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            queen_pipeline_enabled = self.fleet_queen_mode_var.get() == "bbox_active"
            updated, result = initialize_queen_config(
                config=config,
                queen_host=self.fleet_queen_host_var.get().strip() or None,
                identity_file=self.fleet_identity_var.get().strip() or None,
                ssh_user=self.fleet_ssh_user_var.get().strip() or None,
                skip_keygen=False,
            )
            updated.setdefault("fleet", {})
            updated["fleet"]["queen_local_pipeline_enabled"] = queen_pipeline_enabled
            media = apply_queen_media_schedule_defaults(
                updated,
                enable=(not queen_pipeline_enabled),
            )
            self.fleet_media_schedule_enabled_var.set(bool(media.get("enabled", False)))
            self.fleet_media_pull_interval_var.set(str(media.get("pull_interval_minutes", 30)))
            self.fleet_media_track_interval_var.set(str(media.get("track_interval_minutes", 60)))
            self.fleet_pull_max_total_var.set(str(media.get("max_videos_total", 200)))
            self.fleet_pull_cooldown_var.set(str(media.get("cooldown_minutes", 60)))
            self.fleet_pull_max_load_var.set(str(media.get("max_queen_load_1m", 3.0)))
            self.fleet_pull_min_mem_var.set(str(media.get("min_queen_mem_gb", 0.8)))
            self.fleet_pull_allow_active_var.set(bool(media.get("allow_when_queen_bbox_active", False)))
            self.fleet_pull_no_visual_var.set(bool(media.get("disable_visualization", False)))
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="fleet_init_queen",
            )

            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_fleet_init_result(result, show_public_key=False))
            self.fleet_output.insert(
                tk.END,
                "\n\nSaved config. Queen mode: "
                + (
                    "--queen-bbox-active (local recording/tracking enabled)"
                    if queen_pipeline_enabled
                    else "--queen-interface-only (controller only)"
                ),
            )
            self.fleet_output.insert(
                tk.END,
                (
                    "\nQueen media schedule: "
                    f"{'enabled' if bool(media.get('enabled')) else 'disabled'} "
                    f"(pull every {media.get('pull_interval_minutes')} min, "
                    f"track every {media.get('track_interval_minutes')} min)."
                ),
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.fleet_output.insert(tk.END, f"\n{history_note}")
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            self._show_error("Fleet init failed", str(exc))

    def _fleet_enroll_worker(self) -> None:
        host = self.fleet_worker_host_var.get().strip()
        if not host:
            self._show_error("Missing host", "Set worker host/IP first.")
            return
        try:
            port = int(self.fleet_worker_port_var.get().strip())
            if port <= 0:
                raise ValueError("Port must be > 0")
        except Exception as exc:
            self._show_error("Invalid port", str(exc))
            return

        try:
            config, config_path = self._load_config_or_defaults()
            updated, result = enroll_worker_config(
                config=config,
                host=host,
                name=self.fleet_worker_name_var.get().strip() or None,
                user=self.fleet_worker_user_var.get().strip() or None,
                port=port,
                data_root=self.fleet_worker_data_root_var.get().strip() or "/mnt/bumblebox/data",
                unit_prefix=self.fleet_worker_unit_prefix_var.get().strip() or "bumblebox-v2",
                enabled=bool(self.fleet_worker_enabled_var.get()),
                identity_file=self.fleet_identity_var.get().strip() or None,
                install_key=bool(self.fleet_install_key_var.get()),
            )
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="fleet_enroll_worker",
            )
            self.fleet_pull_max_total_var.set(str(result.queen_media_max_videos_total))
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_fleet_enroll_result(result))
            self.fleet_output.insert(tk.END, f"\n\nSaved config: {config_path}")
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.fleet_output.insert(tk.END, f"\n{history_note}")
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            self._show_error("Fleet enroll failed", str(exc))

    def _fleet_run_status(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            report = run_fleet_status(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                identity_file=self.fleet_identity_var.get().strip() or None,
            )
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_fleet_status_report(report))
            self.notebook.select(self.fleet_tab)
            if report.fail_count > 0:
                self._show_warning("Fleet status", "Fleet status found one or more FAIL workers.")
        except Exception as exc:
            self._show_error("Fleet status failed", str(exc))

    def _fleet_set_media_capacity_from_workers(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            target = sync_media_capacity_to_workers(config, include_disabled=False, minimum=1)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="fleet_sync_media_capacity",
            )
            self.fleet_pull_max_total_var.set(str(target))
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(
                tk.END,
                (
                    "Updated queen media capacity from enabled worker count.\n"
                    f"- max_videos_total: {target}\n"
                    f"- config: {config_path}"
                ),
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.fleet_output.insert(tk.END, f"\n{history_note}")
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            self._show_error("Capacity update failed", str(exc))

    def _fleet_refresh_latest_status(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            report = run_queen_latest_status(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                output_root=self.fleet_pull_output_root_var.get().strip() or None,
                identity_file=self.fleet_identity_var.get().strip() or None,
                probe_reachability=bool(self.fleet_probe_reachability_var.get()),
            )
            self._populate_fleet_latest_tree(report)
            self.fleet_latest_detail.delete("1.0", tk.END)
            self.fleet_latest_detail.insert(tk.END, format_queen_latest_status_report(report))
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_queen_latest_status_report(report))
            self.notebook.select(self.fleet_tab)
            if report.offline_count > 0:
                self._show_warning(
                    "Worker offline warning",
                    f"{report.offline_count} configured worker(s) appear offline.",
                )
        except Exception as exc:
            self._show_error("Latest status failed", str(exc))

    def _populate_fleet_latest_tree(self, report) -> None:
        for iid in self.fleet_latest_tree.get_children():
            self.fleet_latest_tree.delete(iid)

        for index, item in enumerate(report.items):
            lag_text = "n/a" if item.track_lag_minutes is None else f"{item.track_lag_minutes:.1f}"
            self.fleet_latest_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    item.worker_name,
                    item.worker_host,
                    ("yes" if item.online else ("no" if item.online is False else "n/a")),
                    item.latest_video_pulled_at or "n/a",
                    item.latest_tracked_at or "n/a",
                    lag_text,
                    item.status,
                ),
            )

    def _fleet_discover_lan(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            report = run_fleet_discovery(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                ping_probe=bool(self.fleet_discovery_ping_var.get()),
            )
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_fleet_discovery_report(report))
            self.notebook.select(self.fleet_tab)
            if report.configured_workers_offline > 0:
                self._show_warning(
                    "Worker offline warning",
                    f"{report.configured_workers_offline} configured worker(s) are not reachable.",
                )
        except Exception as exc:
            self._show_error("Fleet discovery failed", str(exc))

    def _parse_fleet_media_settings(self) -> dict[str, object]:
        try:
            pull_interval = int(self.fleet_media_pull_interval_var.get().strip())
            track_interval = int(self.fleet_media_track_interval_var.get().strip())
            max_total = int(self.fleet_pull_max_total_var.get().strip())
            max_per_worker = int(self.fleet_pull_max_per_worker_var.get().strip())
            cooldown_minutes = int(self.fleet_pull_cooldown_var.get().strip())
            max_load_1m = float(self.fleet_pull_max_load_var.get().strip())
            min_mem_gb = float(self.fleet_pull_min_mem_var.get().strip())
            if pull_interval <= 0 or track_interval <= 0:
                raise ValueError("Pull/track intervals must be >= 1 minute.")
            if max_total <= 0 or max_per_worker <= 0:
                raise ValueError("Max video counts must be >= 1.")
            if cooldown_minutes < 0:
                raise ValueError("Cooldown must be >= 0.")
        except Exception as exc:
            self._show_error("Invalid fleet media settings", str(exc))
            return {}

        return {
            "enabled": bool(self.fleet_media_schedule_enabled_var.get()),
            "pull_interval_minutes": pull_interval,
            "track_interval_minutes": track_interval,
            "output_root": self.fleet_pull_output_root_var.get().strip() or None,
            "max_videos_total": max_total,
            "max_videos_per_worker": max_per_worker,
            "cooldown_minutes": cooldown_minutes,
            "max_queen_load_1m": max_load_1m,
            "min_queen_mem_gb": min_mem_gb,
            "allow_when_queen_bbox_active": bool(self.fleet_pull_allow_active_var.get()),
            "disable_visualization": bool(self.fleet_pull_no_visual_var.get()),
            "dry_run": bool(self.fleet_pull_dry_run_var.get()),
        }

    def _fleet_save_media_schedule(self) -> None:
        settings = self._parse_fleet_media_settings()
        if not settings:
            return

        try:
            config, config_path = self._load_config_or_defaults()
            media = apply_queen_media_schedule_defaults(
                config,
                enable=bool(settings["enabled"]),
            )
            media["pull_interval_minutes"] = int(settings["pull_interval_minutes"])
            media["track_interval_minutes"] = int(settings["track_interval_minutes"])
            media["output_root"] = settings["output_root"]
            media["max_videos_total"] = int(settings["max_videos_total"])
            media["max_videos_per_worker"] = int(settings["max_videos_per_worker"])
            media["cooldown_minutes"] = int(settings["cooldown_minutes"])
            media["max_queen_load_1m"] = float(settings["max_queen_load_1m"])
            media["min_queen_mem_gb"] = float(settings["min_queen_mem_gb"])
            media["allow_when_queen_bbox_active"] = bool(settings["allow_when_queen_bbox_active"])
            media["disable_visualization"] = bool(settings["disable_visualization"])
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="fleet_save_media_schedule",
            )

            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(
                tk.END,
                "Saved fleet.queen_media_schedule to config.\n"
                f"- enabled: {media['enabled']}\n"
                f"- pull_interval_minutes: {media['pull_interval_minutes']}\n"
                f"- track_interval_minutes: {media['track_interval_minutes']}\n"
                f"- max_videos_total: {media['max_videos_total']}\n"
                f"- cooldown_minutes: {media['cooldown_minutes']}\n"
                f"- output_root: {media.get('output_root') or '(default)'}\n\n"
                "Next: run systemd-write then systemd-install to apply timers.",
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.fleet_output.insert(tk.END, f"\n\n{history_note}")
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            self._show_error("Save schedule failed", str(exc))

    def _fleet_queen_pull_latest(self) -> None:
        settings = self._parse_fleet_media_settings()
        if not settings:
            return

        try:
            config, _ = self._load_config_or_defaults()
            report = run_queen_pull_latest_videos(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                identity_file=self.fleet_identity_var.get().strip() or None,
                output_root=str(settings["output_root"]) if settings["output_root"] else None,
                max_videos_total=int(settings["max_videos_total"]),
                dry_run=bool(settings["dry_run"]),
            )
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_queen_track_report(report))
            self.notebook.select(self.fleet_tab)
            if report.videos_failed > 0:
                self._show_warning(
                    "Queen pull latest",
                    "Queen pull latest completed with failures. Review output for details.",
                )
        except Exception as exc:
            self._show_error("Queen pull latest failed", str(exc))

    def _fleet_queen_track_latest(self) -> None:
        settings = self._parse_fleet_media_settings()
        if not settings:
            return

        try:
            config, _ = self._load_config_or_defaults()
            report = run_queen_track_latest_videos(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                output_root=str(settings["output_root"]) if settings["output_root"] else None,
                max_videos_total=int(settings["max_videos_total"]),
                cooldown_minutes=int(settings["cooldown_minutes"]),
                with_visualization=not bool(settings["disable_visualization"]),
                dry_run=bool(settings["dry_run"]),
                allow_when_queen_bbox_active=bool(settings["allow_when_queen_bbox_active"]),
                max_queen_load_1m=float(settings["max_queen_load_1m"]),
                min_queen_mem_available_gb=float(settings["min_queen_mem_gb"]),
            )
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_queen_track_report(report))
            self.notebook.select(self.fleet_tab)
            if report.videos_failed > 0:
                self._show_warning(
                    "Queen track latest",
                    "Queen track latest completed with failures. Review output for details.",
                )
        except Exception as exc:
            self._show_error("Queen track latest failed", str(exc))

    def _fleet_queen_pull_track(self) -> None:
        settings = self._parse_fleet_media_settings()
        if not settings:
            return

        try:
            config, _ = self._load_config_or_defaults()
            report = run_queen_pull_track_latest(
                config=config,
                include_disabled=bool(self.fleet_include_disabled_var.get()),
                worker_filter=self.fleet_filter_var.get().strip() or None,
                identity_file=self.fleet_identity_var.get().strip() or None,
                output_root=str(settings["output_root"]) if settings["output_root"] else None,
                max_videos_total=int(settings["max_videos_total"]),
                max_videos_per_worker=int(settings["max_videos_per_worker"]),
                cooldown_minutes=int(settings["cooldown_minutes"]),
                with_visualization=not bool(settings["disable_visualization"]),
                dry_run=bool(settings["dry_run"]),
                allow_when_queen_bbox_active=bool(settings["allow_when_queen_bbox_active"]),
                max_queen_load_1m=float(settings["max_queen_load_1m"]),
                min_queen_mem_available_gb=float(settings["min_queen_mem_gb"]),
            )
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_queen_track_report(report))
            self.notebook.select(self.fleet_tab)
            if report.videos_failed > 0:
                self._show_warning(
                    "Queen pull-track",
                    "Queen pull-track completed with failures. Review output for details.",
                )
        except Exception as exc:
            self._show_error("Queen pull-track failed", str(exc))

    def _build_run_tab(self) -> None:
        top = ttk.Frame(self.run_tab)
        top.pack(fill=tk.X)

        self.run_mode_var = tk.StringVar(value="")
        self.run_mock_var = tk.BooleanVar(value=False)
        self.systemd_output_dir_var = tk.StringVar(value=str(Path.cwd() / "systemd"))

        ttk.Label(top, text="Run mode override").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Combobox(
            top,
            textvariable=self.run_mode_var,
            values=["", "record_only", "track_only", "record_and_track"],
            state="readonly",
            width=24,
        )
        mode_box.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        mode_box.current(0)

        ttk.Checkbutton(top, text="Use mock camera", variable=self.run_mock_var).grid(
            row=0, column=2, sticky="w", padx=12
        )

        ttk.Button(top, text="Run Once Now", command=self._run_once_now).grid(row=0, column=3, sticky="w", padx=8)

        automation_box = ttk.LabelFrame(top, text="Automated Recording", padding=6)
        automation_box.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(
            automation_box,
            text=(
                "Repeated recording uses scheduled systemd timers. "
                "Start saves the current Config Editor settings, writes the timer files, "
                "installs them into systemd, and starts them now. Stop disables the timers and stops future "
                "scheduled runs. In Basic mode BumbleBox uses user-scope timers so automation can start without sudo."
            ),
            wraplength=900,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")
        automation_actions = ttk.Frame(automation_box)
        automation_actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(
            automation_actions,
            text="Start Automated Recording",
            command=self._start_automated_recording,
        ).pack(side=tk.LEFT)
        ttk.Button(
            automation_actions,
            text="Stop Automated Recording",
            command=self._stop_automated_recording,
        ).pack(side=tk.LEFT, padx=(8, 0))
        automation_box.columnconfigure(0, weight=1)

        index_box = ttk.LabelFrame(top, text="Local Tracking Index", padding=6)
        index_box.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(
            index_box,
            text=(
                "BumbleBox keeps a lightweight local copy of run summaries, tracking CSVs, and optimization "
                "results in repo-local LocalTrackingIndex by default. Generated artifacts are ignored by Git, "
                "and large videos are intentionally not copied here."
            ),
            wraplength=900,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")
        index_actions = ttk.Frame(index_box)
        index_actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(
            index_actions,
            text="Open Tracking Index",
            command=self._open_local_tracking_index,
        ).pack(side=tk.LEFT)
        self.tracking_index_sync_btn = ttk.Button(
            index_actions,
            text="Sync Recent Runs To Index",
            command=self._sync_recent_runs_to_tracking_index,
        )
        self.tracking_index_sync_btn.pack(side=tk.LEFT, padx=(8, 0))
        index_box.columnconfigure(0, weight=1)

        systemd_advanced = ttk.LabelFrame(top, text="Advanced Timer Controls", padding=6)
        systemd_advanced.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(systemd_advanced, text="Timer/unit file output dir").grid(row=0, column=0, sticky="w")
        ttk.Entry(systemd_advanced, textvariable=self.systemd_output_dir_var, width=70).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=8, pady=4
        )
        ttk.Button(systemd_advanced, text="Write Timer Files", command=self._generate_systemd_units).grid(
            row=0, column=3, sticky="w", padx=8
        )
        ttk.Label(
            systemd_advanced,
            text=(
                "Manual flow: write timer files if you want to inspect them, then install them "
                "into systemd. The install step already enables and starts the timers."
            ),
            wraplength=900,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 4))

        actions = ttk.Frame(systemd_advanced)
        actions.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 2))
        ttk.Button(actions, text="Install + Start Timers", command=lambda: self._systemd_action("install")).pack(side=tk.LEFT)
        ttk.Button(actions, text="Enable + Start Installed Timers", command=lambda: self._systemd_action("enable")).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Disable + Stop Timers", command=lambda: self._systemd_action("disable")).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Show Timer Status", command=lambda: self._systemd_action("status")).pack(side=tk.LEFT)
        ttk.Button(actions, text="Install GUI Desktop Icon", command=self._install_gui_shortcut).pack(side=tk.LEFT, padx=12)
        systemd_advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(systemd_advanced)

        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)

        self.run_output = self._create_results_section(
            self.run_tab,
            title="Run & Schedule Results",
            text_height=9,
            default_visible=False,
            auto_hide_when_empty=True,
            auto_height=True,
            min_text_lines=3,
            max_text_lines=12,
            fill=tk.X,
            expand=False,
        )

        alerts_frame = ttk.LabelFrame(self.run_tab, text="Runtime Alerts", padding=8)
        alerts_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
        ttk.Button(alerts_frame, text="Refresh Runtime Alerts", command=self._refresh_runtime_alerts).pack(anchor=tk.W)
        self.runtime_alert_output = self._create_results_section(
            alerts_frame,
            title="Runtime Alerts Output",
            text_height=5,
            default_visible=True,
            auto_hide_when_empty=False,
            auto_height=True,
            min_text_lines=3,
            max_text_lines=8,
            fill=tk.X,
            expand=False,
            pady=(6, 0),
        )

        history_frame = ttk.LabelFrame(self.run_tab, text="Recent Runs", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.run_history_refresh_btn = ttk.Button(
            history_frame,
            text="Refresh Run History",
            command=self._refresh_run_history,
        )
        self.run_history_refresh_btn.pack(anchor=tk.W)

        self.bundle_output_dir_var = tk.StringVar(value=str(Path.cwd() / "bundles"))
        self.bundle_skip_video_var = tk.BooleanVar(value=False)
        self.bundle_core_only_var = tk.BooleanVar(value=False)
        self.bundle_zip_var = tk.BooleanVar(value=True)

        export_controls = ttk.LabelFrame(history_frame, text="Advanced Export Options", padding=6)
        export_controls.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(export_controls, text="Bundle output dir").pack(side=tk.LEFT)
        ttk.Entry(export_controls, textvariable=self.bundle_output_dir_var, width=38).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Checkbutton(export_controls, text="Skip video", variable=self.bundle_skip_video_var).pack(side=tk.LEFT)
        ttk.Checkbutton(export_controls, text="Core only", variable=self.bundle_core_only_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(export_controls, text="Zip", variable=self.bundle_zip_var).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            export_controls,
            text="Export Selected Run Bundle",
            command=self._export_selected_run_bundle,
        ).pack(side=tk.LEFT, padx=8)

        columns = ("started", "mode", "fps", "success", "warnings", "errors", "session")
        self.run_history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=7)
        for col, label, width in [
            ("started", "Started", 170),
            ("mode", "Mode", 120),
            ("fps", "FPS", 80),
            ("success", "OK", 60),
            ("warnings", "Warn", 60),
            ("errors", "Err", 50),
            ("session", "Session", 260),
        ]:
            self.run_history_tree.heading(col, text=label)
            self.run_history_tree.column(col, width=width, stretch=(col == "session"))
        self.run_history_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 8))
        self.run_history_tree.bind("<<TreeviewSelect>>", self._on_run_history_select)

        self.run_history_detail_frame = ttk.LabelFrame(history_frame, text="Selected Run Details", padding=6)
        self.run_history_detail = tk.Text(self.run_history_detail_frame, wrap=tk.WORD, height=6)
        self.run_history_detail.pack(fill=tk.BOTH, expand=False)
        self._style_output_text(self.run_history_detail)
        self._set_run_history_detail_text("")
        self._register_advanced_widget(export_controls)
        self.after(100, self._refresh_run_history)

    def _open_path_in_file_manager(self, path: Path) -> None:
        if sys.platform == "darwin":
            command = ["open", str(path)]
        elif os.name == "nt":
            command = ["explorer", str(path)]
        else:
            command = ["xdg-open", str(path)]
        subprocess.Popen(command)

    def _open_local_tracking_index(self) -> None:
        from .tracking_index import ensure_local_index_root

        try:
            config, _config_path = self._load_config_or_defaults()
            root = ensure_local_index_root(config)
            self._open_path_in_file_manager(root)
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, f"Opened local tracking index:\n{root}")
        except Exception as exc:
            self._show_error("Open tracking index failed", str(exc))

    def _sync_recent_runs_to_tracking_index(self) -> None:
        if self._tracking_index_thread and self._tracking_index_thread.is_alive():
            self._show_info("Tracking index sync", "A tracking index sync is already running.")
            return

        try:
            config, _config_path = self._load_config_or_defaults()
            data_root = str(config.get("system", {}).get("data_root", "")).strip()
            if not data_root:
                raise ValueError("system.data_root is empty in config.")
        except Exception as exc:
            self._show_error("Tracking index sync failed", str(exc))
            return

        self._tracking_index_output_text = None
        self._tracking_index_error = None
        self.tracking_index_sync_btn.config(state=tk.DISABLED)
        self.run_output.delete("1.0", tk.END)
        self.run_output.insert(tk.END, "Syncing recent runs to local tracking index...\n")

        def _worker() -> None:
            from .tracking_index import summarize_index_results, sync_run_summary_file

            try:
                records = list_recent_run_records(data_root, limit=80)
                results = []
                for record in records:
                    payload = load_run_summary(record.summary_path)
                    results.append(
                        sync_run_summary_file(
                            config,
                            record.summary_path,
                            summary_payload=payload,
                        )
                    )
                self._tracking_index_output_text = summarize_index_results(results)
            except Exception as exc:
                self._tracking_index_error = str(exc)

        self._tracking_index_thread = threading.Thread(target=_worker, daemon=True)
        self._tracking_index_thread.start()
        self.after(150, self._poll_tracking_index_sync)

    def _poll_tracking_index_sync(self) -> None:
        thread = self._tracking_index_thread
        if thread and thread.is_alive():
            self.after(150, self._poll_tracking_index_sync)
            return

        self._tracking_index_thread = None
        self.tracking_index_sync_btn.config(state=tk.NORMAL)
        if self._tracking_index_error:
            error = self._tracking_index_error
            self._tracking_index_error = None
            self._show_error("Tracking index sync failed", error)
            self.run_output.insert(tk.END, f"\nError: {error}\n")
            return

        text = self._tracking_index_output_text or "Tracking index sync finished."
        self._tracking_index_output_text = None
        self.run_output.delete("1.0", tk.END)
        self.run_output.insert(tk.END, text)

    def _load_config_or_defaults(self):
        config_path = Path(self.config_path_var.get()).expanduser()
        if config_path.exists():
            return load_config(config_path), config_path
        return load_defaults(), config_path

    @staticmethod
    def _get_nested(config: dict, dotted_key: str):
        value = config
        for part in dotted_key.split("."):
            value = value[part]
        return value

    @staticmethod
    def _set_nested(config: dict, dotted_key: str, value):
        parts = dotted_key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _refresh_config_path_controls(self) -> None:
        button = getattr(self, "config_action_button", None)
        if button is None:
            return
        path = Path(self.config_path_var.get()).expanduser()
        if path.exists():
            button.configure(text="Change Config")
        else:
            button.configure(text="Config Missing! Create Config")

    def _handle_config_path_action(self) -> None:
        current_path = Path(self.config_path_var.get()).expanduser()
        if not current_path.exists():
            self._create_config()
            return

        selected = filedialog.askopenfilename(
            title="Select BumbleBox Config",
            initialdir=str(current_path.parent),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not selected:
            return

        selected_path = Path(selected).expanduser()
        self.config_path_var.set(str(selected_path))
        self._refresh_config_path_controls()
        self._set_theme_mode(
            self._read_theme_mode_from_config(),
            sync_knob=True,
            persist_preference=False,
        )
        self._load_config_into_editor()
        self.config_output.delete("1.0", tk.END)
        self.config_output.insert(tk.END, f"Switched to config:\n{selected_path}")

    def _load_config_into_editor(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            load_note: str | None = None
            self._set_theme_mode(
                str(config.get("runtime", {}).get("ui_theme_mode", "dark")),
                sync_knob=True,
                persist_preference=False,
            )
            for key, (variable, value_type) in self.config_fields.items():
                raw = self._get_nested(config, key)
                if key == "camera.preview_window":
                    raw = "QT"
                elif key == "camera.mp4_codec":
                    raw_text = str(raw).strip().lower()
                    if raw_text in {"mpeg4", "mp4v"}:
                        raw = "H.264 (Recommended)"
                        load_note = (
                            "Legacy MPEG-4 MP4 default detected. "
                            "The editor now selects H.264 (Recommended). Save config to make that change permanent."
                        )
                    else:
                        raw = self._mp4_codec_value_to_display().get(
                            raw_text,
                            "H.264 (Recommended)",
                        )
                elif key == "tracking.tag_dictionary":
                    raw = self._normalize_display_dictionary(raw)
                if value_type is bool:
                    variable.set(bool(raw))
                else:
                    variable.set("" if raw is None else str(raw))
            self._refresh_config_field_visibility()
            self._refresh_thermal_controls_state()
            self._sync_dictionary_selectors_from_config()
            self._refresh_config_path_controls()
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(
                tk.END,
                "Loaded configuration into editor."
                + (f"\n{load_note}" if load_note else ""),
            )
        except Exception as exc:
            self._show_error("Load config failed", str(exc))

    def _build_editor_config(self):
        config, _ = self._load_config_or_defaults()
        for key, (variable, value_type) in self.config_fields.items():
            if value_type is bool:
                parsed = bool(variable.get())
            elif value_type is int:
                parsed = int(str(variable.get()).strip())
            elif value_type is float:
                parsed = float(str(variable.get()).strip())
            else:
                text = str(variable.get()).strip()
                if key in {"camera.tuning_file", "camera.lens_position"} and text == "":
                    parsed = None
                elif key == "camera.lens_position":
                    parsed = float(text)
                elif key == "camera.codec":
                    parsed = text.lower() or "mp4"
                elif key == "camera.mp4_codec":
                    parsed = self._mp4_codec_display_to_value().get(text, "libx264")
                elif key == "camera.preview_window":
                    parsed = "QT"
                elif key == "tracking.tag_dictionary":
                    parsed = self._normalize_display_dictionary(text)
                else:
                    parsed = text
            self._set_nested(config, key, parsed)
        return apply_camera_profile(config)

    def _load_effective_action_config(self) -> tuple[dict, Path]:
        _, config_path = self._load_config_or_defaults()
        if not self.config_fields:
            return self._load_config_or_defaults()
        config = self._build_editor_config()
        validate_config(config)
        return config, config_path

    @staticmethod
    def _history_reason_slug(reason: str) -> str:
        cleaned = []
        for ch in str(reason).strip().lower():
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        slug = "".join(cleaned).strip("_")
        return slug or "config_update"

    def _write_config_history_snapshot(
        self,
        *,
        config_path: Path,
        config: dict,
        reason: str,
    ) -> tuple[Path | None, str | None]:
        try:
            data_root_text = str(config.get("system", {}).get("data_root", "")).strip()
            if not data_root_text:
                raise ValueError("system.data_root is empty")

            recorded_at = datetime.now()
            history_dir = Path(data_root_text).expanduser() / "config_history"
            history_dir.mkdir(parents=True, exist_ok=True)

            snapshot_path = history_dir / (
                f"{recorded_at.strftime('%Y-%m-%d_%H-%M-%S_%f')}_"
                f"{self._history_reason_slug(reason)}.json"
            )
            payload = {
                "recorded_at": recorded_at.isoformat(timespec="seconds"),
                "reason": reason,
                "active_config_path": str(Path(config_path).expanduser().resolve()),
                "config": config,
            }
            snapshot_path.write_text(json.dumps(payload, indent=2))
            return snapshot_path, None
        except Exception as exc:
            return None, str(exc)

    def _save_config_with_history(
        self,
        config_path: Path,
        config: dict,
        *,
        reason: str,
    ) -> tuple[Path | None, str | None]:
        save_config(config_path, config)
        return self._write_config_history_snapshot(
            config_path=config_path,
            config=config,
            reason=reason,
        )

    @staticmethod
    def _format_config_history_note(snapshot_path: Path | None, warning: str | None) -> str:
        if snapshot_path is not None:
            return f"Config history snapshot: {snapshot_path}"
        if warning:
            return f"Config history warning: {warning}"
        return ""

    def _validate_editor_config(self) -> None:
        try:
            config = self._build_editor_config()
            validate_config(config)
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, "Config is valid.")
        except Exception as exc:
            self._show_error("Validation failed", str(exc))

    def _save_editor_config(self) -> None:
        try:
            config = self._build_editor_config()
            validate_config(config)
            _, config_path = self._load_config_or_defaults()
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="gui_save_config",
            )
            self._set_theme_mode(
                str(config.get("runtime", {}).get("ui_theme_mode", "dark")),
                sync_knob=True,
                persist_preference=False,
            )
            self._refresh_config_path_controls()
            self.config_output.delete("1.0", tk.END)
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            self.config_output.insert(
                tk.END,
                "Saved config to "
                f"{config_path}"
                + (f"\n{history_note}" if history_note else ""),
            )
        except Exception as exc:
            self._show_error("Save failed", str(exc))

    def _create_config(self) -> None:
        try:
            path = write_default_config(self.config_path_var.get(), force=False)
            self._persist_theme_mode_to_config()
            self._load_config_into_editor()
            self._refresh_config_path_controls()
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(
                tk.END,
                (
                    "Created a new default config file.\n"
                    f"Path: {path}\n"
                    "The file has been loaded into the editor."
                ),
            )
            self._show_info("Config created", f"Created default config at:\n{path}")
        except FileExistsError:
            self._refresh_config_path_controls()
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(
                tk.END,
                "Config file already exists at the selected path; no overwrite was performed.",
            )
            self._show_info("Config exists", "Config already exists. Keeping current file.")
        except Exception as exc:
            self._show_error("Error", str(exc))

    def _selected_storage_device_path(self) -> str | None:
        selected = self.storage_device_var.get().strip()
        return self._storage_device_display_to_path.get(selected)

    def _refresh_storage_device_choices(self, preferred_path: str | None = None) -> None:
        auto_label = "Auto (recommended)"
        choices = [auto_label]
        mapping: dict[str, str] = {}
        devices = discover_storage_devices()

        for device in devices:
            if device.dev_type != "part":
                continue
            if not device.path.startswith("/dev/"):
                continue
            if not device.has_filesystem:
                continue
            if device.mountpoint in {"/", "/boot", "/boot/firmware"}:
                continue
            descriptor = (
                f"{device.path} ({device.size or '?'} {device.fstype or 'unknown fs'}"
                f", {device.display_name})"
            )
            if device.mountpoint:
                if str(device.mountpoint).startswith("/media/"):
                    descriptor += f" currently desktop-mounted at {device.mountpoint}"
                else:
                    descriptor += f" currently mounted at {device.mountpoint}"
            choices.append(descriptor)
            mapping[descriptor] = device.path

        self._storage_device_display_to_path = mapping
        self.storage_device_combo.configure(values=choices)

        current = self.storage_device_var.get().strip()
        if current in choices:
            return
        self.storage_device_var.set(auto_label)

    @staticmethod
    def _recommended_data_root() -> Path:
        return (Path.home() / "BumbleBoxData").expanduser()

    @staticmethod
    def _desktop_data_shortcut_path() -> Path:
        return (Path.home() / "Desktop" / "BumbleBoxData").expanduser()

    def _should_offer_desktop_data_shortcut(self, mount_point: str) -> bool:
        try:
            mount_path = Path(mount_point).expanduser().resolve()
        except Exception:
            return False
        desktop_dir = self._desktop_data_shortcut_path().parent
        try:
            desktop_dir_resolved = desktop_dir.resolve()
        except Exception:
            desktop_dir_resolved = desktop_dir
        return mount_path.parent != desktop_dir_resolved

    def _ensure_desktop_data_shortcut(self, mount_point: str) -> tuple[Path, str]:
        mount_path = Path(mount_point).expanduser().resolve()
        desktop_shortcut = self._desktop_data_shortcut_path()
        desktop_shortcut.parent.mkdir(parents=True, exist_ok=True)

        if desktop_shortcut.exists() or desktop_shortcut.is_symlink():
            try:
                existing_resolved = desktop_shortcut.resolve()
            except Exception:
                existing_resolved = None
            if existing_resolved == mount_path:
                return desktop_shortcut, "Desktop shortcut already points to the selected data folder."
            if desktop_shortcut.is_symlink():
                desktop_shortcut.unlink()
            else:
                alternate = desktop_shortcut.parent / "BumbleBoxData Shortcut"
                if alternate.exists() or alternate.is_symlink():
                    try:
                        alternate_resolved = alternate.resolve()
                    except Exception:
                        alternate_resolved = None
                    if alternate_resolved == mount_path:
                        return alternate, "Desktop shortcut already points to the selected data folder."
                    if alternate.is_symlink():
                        alternate.unlink()
                    else:
                        raise FileExistsError(
                            f"Desktop already contains {desktop_shortcut} and {alternate}. Remove or rename one of them to create a shortcut."
                        )
                alternate.symlink_to(mount_path, target_is_directory=True)
                return alternate, (
                    f"Desktop already contains a real folder at {desktop_shortcut}, so BumbleBox created a shortcut named "
                    f"{alternate.name} instead."
                )

        desktop_shortcut.symlink_to(mount_path, target_is_directory=True)
        return desktop_shortcut, "Created a Desktop shortcut to the selected data folder."

    def _offer_desktop_data_shortcut(self, mount_point: str) -> None:
        if not self._should_offer_desktop_data_shortcut(mount_point):
            return
        create_shortcut = messagebox.askyesno(
            "Desktop Data Shortcut",
            (
                f"BumbleBox is now using:\n{mount_point}\n\n"
                "Do you want a Desktop shortcut to this data folder too?"
            ),
        )
        if not create_shortcut:
            return
        try:
            shortcut_path, note = self._ensure_desktop_data_shortcut(mount_point)
        except Exception as exc:
            self._show_error("Desktop shortcut failed", str(exc))
            return
        messagebox.showinfo(
            "Desktop Data Shortcut",
            f"{note}\n\nShortcut path:\n{shortcut_path}",
        )

    def _load_storage_mount_point_from_config(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            mount_point = str(config.get("system", {}).get("data_root", "")).strip()
            if mount_point and mount_point != "/mnt/bumblebox/data":
                self.storage_mount_point_var.set(mount_point)
            else:
                self.storage_mount_point_var.set(str(self._recommended_data_root()))
        except Exception:
            pass

    def _save_storage_mount_point_to_config(self) -> None:
        mount_point = self.storage_mount_point_var.get().strip()
        if not mount_point:
            self._show_error("Invalid mount point", "Mount point cannot be empty.")
            return
        try:
            config, config_path = self._load_config_or_defaults()
            config.setdefault("system", {})
            config["system"]["data_root"] = mount_point
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="storage_save_mount_point",
            )
            self.storage_output.delete("1.0", tk.END)
            self.storage_output.insert(
                tk.END,
                f"Saved data folder path to config:\n- system.data_root: {mount_point}\n- config: {config_path}",
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.storage_output.insert(tk.END, f"\n{history_note}")
            self._refresh_storage_device_choices()
            self._refresh_storage_status()
        except Exception as exc:
            self._show_error("Save data folder path failed", str(exc))

    def _refresh_storage_status(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            mount_point = self.storage_mount_point_var.get().strip() or None
            report = get_storage_status(config=config, mount_point=mount_point)
            self._refresh_storage_device_choices(preferred_path=report.recommended_device_path)
            self.storage_output.delete("1.0", tk.END)
            self.storage_output.insert(tk.END, format_storage_status_report(report))
            selected_device = self._selected_storage_device_path()
            if selected_device:
                self.storage_output.insert(tk.END, f"\nSelected setup device: {selected_device}")
            else:
                self.storage_output.insert(tk.END, "\nSelected setup device: Auto (recommended)")
        except Exception as exc:
            self.storage_output.delete("1.0", tk.END)
            self.storage_output.insert(tk.END, f"[FAIL] Storage status failed: {exc}")

    def _run_pkexec_storage_setup(self, config_path: Path, mount_point: str, device_path: str | None) -> bool:
        if os.name != "posix":
            return False
        if shutil.which("pkexec") is None:
            return False

        repo_root = Path(__file__).resolve().parents[1]
        bbx_path = repo_root / "bbx.py"
        command = [
            "pkexec",
            sys.executable,
            str(bbx_path),
            "storage",
            "setup",
            "--config",
            str(config_path),
            "--mount-point",
            mount_point,
        ]
        if device_path:
            command.extend(["--device", device_path])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_camera_safe_env(include_qt=True),
        )
        self.storage_output.delete("1.0", tk.END)
        if proc.returncode == 0:
            text = (proc.stdout or "").strip() or "Storage setup completed with pkexec."
            self.storage_output.insert(tk.END, text)
            return True

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        self.storage_output.insert(
            tk.END,
            (
                "pkexec storage setup failed.\n"
                f"stdout:\n{out or '(empty)'}\n\n"
                f"stderr:\n{err or '(empty)'}"
            ),
        )
        return False

    def _run_pkexec_storage_mount(self, config_path: Path, mount_point: str, device_path: str | None) -> bool:
        if os.name != "posix":
            return False
        if shutil.which("pkexec") is None:
            return False

        repo_root = Path(__file__).resolve().parents[1]
        bbx_path = repo_root / "bbx.py"
        command = [
            "pkexec",
            sys.executable,
            str(bbx_path),
            "storage",
            "mount",
            "--config",
            str(config_path),
            "--mount-point",
            mount_point,
        ]
        if device_path:
            command.extend(["--device", device_path])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_camera_safe_env(include_qt=True),
        )
        self.storage_output.delete("1.0", tk.END)
        if proc.returncode == 0:
            text = (proc.stdout or "").strip() or "Storage mount completed with pkexec."
            self.storage_output.insert(tk.END, text)
            return True

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        self.storage_output.insert(
            tk.END,
            (
                "pkexec storage mount failed.\n"
                f"stdout:\n{out or '(empty)'}\n\n"
                f"stderr:\n{err or '(empty)'}"
            ),
        )
        return False

    def _persist_storage_mount_point_after_privileged_action(
        self,
        *,
        config_path: Path,
        config: dict,
        mount_point: str,
        reason: str,
    ) -> tuple[Path | None, str | None]:
        config.setdefault("system", {})
        config["system"]["data_root"] = mount_point
        return self._save_config_with_history(
            config_path,
            config,
            reason=reason,
        )

    def _mount_storage_device(self) -> None:
        mount_point = self.storage_mount_point_var.get().strip()
        if not mount_point:
            self._show_error("Invalid mount point", "Mount point cannot be empty.")
            return

        try:
            config, config_path = self._load_config_or_defaults()
            device_path = self._selected_storage_device_path()
            status = get_storage_status(config=config, mount_point=mount_point)
            if status.mounted and status.writable and (
                device_path is None or status.device_path == device_path
            ):
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_status_report(status))
                if str(config.get("system", {}).get("data_root", "")).strip() != mount_point:
                    config.setdefault("system", {})
                    config["system"]["data_root"] = mount_point
                    snapshot_path, history_warning = self._save_config_with_history(
                        config_path,
                        config,
                        reason="storage_mount_device",
                    )
                    self.storage_output.insert(tk.END, f"\n\nUpdated config: {config_path}")
                    history_note = self._format_config_history_note(snapshot_path, history_warning)
                    if history_note:
                        self.storage_output.insert(tk.END, f"\n{history_note}")
                return

            try:
                result = mount_storage_device_now(mount_point=mount_point, device_path=device_path)
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_mount_result(result))
                config.setdefault("system", {})
                config["system"]["data_root"] = mount_point
                snapshot_path, history_warning = self._save_config_with_history(
                    config_path,
                    config,
                    reason="storage_mount_device",
                )
                self.storage_output.insert(tk.END, f"\n\nUpdated config: {config_path}")
                history_note = self._format_config_history_note(snapshot_path, history_warning)
                if history_note:
                    self.storage_output.insert(tk.END, f"\n{history_note}")
                self._offer_desktop_data_shortcut(mount_point)
            except PermissionError:
                if self._run_pkexec_storage_mount(
                    config_path=config_path,
                    mount_point=mount_point,
                    device_path=device_path,
                ):
                    snapshot_path, history_warning = self._persist_storage_mount_point_after_privileged_action(
                        config_path=config_path,
                        config=config,
                        mount_point=mount_point,
                        reason="storage_mount_device",
                    )
                    self._refresh_storage_status()
                    history_note = self._format_config_history_note(snapshot_path, history_warning)
                    if history_note:
                        self.storage_output.insert(tk.END, f"\n{history_note}")
                    self._offer_desktop_data_shortcut(mount_point)
                    return

                sudo_cmd = build_storage_mount_sudo_command(
                    config_path=str(config_path),
                    mount_point=mount_point,
                    device_path=device_path,
                    apply_config=False,
                )
                self.storage_output.insert(
                    tk.END,
                    (
                        "\n\nStorage mount needs root privileges.\n"
                        "Run this in terminal on the Pi:\n"
                        f"{sudo_cmd}\n\n"
                        "Then, back in BumbleBox, click `Save Data Folder Path` if needed."
                    ),
                )
                return

        except Exception as exc:
            self._show_error("Storage mount failed", str(exc))

    def _setup_storage_auto_mount(self) -> None:
        mount_point = self.storage_mount_point_var.get().strip()
        if not mount_point:
            self._show_error("Invalid mount point", "Mount point cannot be empty.")
            return

        try:
            config, config_path = self._load_config_or_defaults()
            device_path = self._selected_storage_device_path()
            status = get_storage_status(config=config, mount_point=mount_point)
            if status.mounted and status.writable and (
                device_path is None or status.device_path == device_path
            ):
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_status_report(status))
                return

            try:
                result = setup_storage_auto_mount(mount_point=mount_point, device_path=device_path)
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_setup_result(result))
                config.setdefault("system", {})
                config["system"]["data_root"] = mount_point
                snapshot_path, history_warning = self._save_config_with_history(
                    config_path,
                    config,
                    reason="storage_setup_auto_mount",
                )
                self.storage_output.insert(tk.END, f"\n\nUpdated config: {config_path}")
                history_note = self._format_config_history_note(snapshot_path, history_warning)
                if history_note:
                    self.storage_output.insert(tk.END, f"\n{history_note}")
                self._offer_desktop_data_shortcut(mount_point)
            except PermissionError:
                if self._run_pkexec_storage_setup(
                    config_path=config_path,
                    mount_point=mount_point,
                    device_path=device_path,
                ):
                    snapshot_path, history_warning = self._persist_storage_mount_point_after_privileged_action(
                        config_path=config_path,
                        config=config,
                        mount_point=mount_point,
                        reason="storage_setup_auto_mount",
                    )
                    self._refresh_storage_status()
                    history_note = self._format_config_history_note(snapshot_path, history_warning)
                    if history_note:
                        self.storage_output.insert(tk.END, f"\n{history_note}")
                    self._offer_desktop_data_shortcut(mount_point)
                    return

                sudo_cmd = build_storage_setup_sudo_command(
                    config_path=str(config_path),
                    mount_point=mount_point,
                    device_path=device_path,
                    apply_config=False,
                )
                self.storage_output.insert(
                    tk.END,
                    (
                        "\n\nStorage setup needs root privileges.\n"
                        "Run this in terminal on the Pi:\n"
                        f"{sudo_cmd}\n\n"
                        "Then, back in BumbleBox, click `Save Data Folder Path` if needed."
                    ),
                )
        except Exception as exc:
            self._show_error("Storage setup failed", str(exc))

    def _run_doctor(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            results = run_doctor(config)
            self._render_doctor_report(results)
            self._refresh_storage_status()
        except Exception as exc:
            self._show_error("Doctor failed", str(exc))

    def _render_doctor_report(self, results) -> None:
        text = format_doctor_report(results)
        self.doctor_output.delete("1.0", tk.END)
        self.doctor_output.tag_configure("doctor_fail", foreground="#FF6B6B")
        self.doctor_output.tag_configure("doctor_warn", foreground="#EACB63")
        self.doctor_output.tag_configure("doctor_pass", foreground="#63D47C")

        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("[FAIL]"):
                self.doctor_output.insert(tk.END, line, ("doctor_fail",))
            elif stripped.startswith("[WARN]"):
                self.doctor_output.insert(tk.END, line, ("doctor_warn",))
            elif stripped.startswith("[PASS]"):
                self.doctor_output.insert(tk.END, line, ("doctor_pass",))
            else:
                self.doctor_output.insert(tk.END, line)

    def _doctor_fix_venv_package_visibility(self) -> None:
        try:
            in_venv = (
                hasattr(sys, "base_prefix")
                and str(sys.prefix) != str(getattr(sys, "base_prefix", sys.prefix))
            )
            if not in_venv:
                message = (
                    "Current Python is not running inside a virtual environment. "
                    "No venv package-visibility fix is needed."
                )
                self.doctor_output.delete("1.0", tk.END)
                self.doctor_output.insert(tk.END, message)
                return

            major = int(sys.version_info.major)
            minor = int(sys.version_info.minor)
            venv_site = Path(sys.prefix) / "lib" / f"python{major}.{minor}" / "site-packages"
            venv_site.mkdir(parents=True, exist_ok=True)

            candidate_paths = {
                "/usr/lib/python3/dist-packages",
                "/usr/local/lib/python3/dist-packages",
                f"/usr/lib/python{major}.{minor}/dist-packages",
                f"/usr/local/lib/python{major}.{minor}/dist-packages",
            }

            probe = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-c",
                    (
                        "import json,site,sys;"
                        "print(json.dumps({'site': site.getsitepackages(), 'path': sys.path}))"
                    ),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                try:
                    payload = json.loads((probe.stdout or "").strip() or "{}")
                except Exception:
                    payload = {}
                for key in ("site", "path"):
                    for entry in payload.get(key, []) or []:
                        entry_text = str(entry).strip()
                        if "dist-packages" in entry_text:
                            candidate_paths.add(entry_text)

            existing = sorted(
                {
                    str(Path(path).resolve())
                    for path in candidate_paths
                    if str(path).strip() and Path(path).exists()
                }
            )
            if not existing:
                raise RuntimeError(
                    "No system dist-packages directories were found to link into this venv."
                )

            pth_path = venv_site / "bumblebox_system_packages.pth"
            pth_path.write_text("\n".join(existing) + "\n")

            for path in existing:
                if path not in sys.path:
                    sys.path.append(path)

            config, _ = self._load_config_or_defaults()
            results = run_doctor(config)
            self._render_doctor_report(results)
            self.doctor_output.insert(
                tk.END,
                (
                    "\n\nApplied venv system-package visibility fix.\n"
                    f"Created: {pth_path}\n"
                    f"Linked {len(existing)} system path(s)."
                ),
            )
            self._show_info(
                "Venv package visibility fixed",
                "Linked system Python package paths into this venv. Re-run Doctor to verify dependencies.",
            )
        except Exception as exc:
            self._show_error("Fix failed", str(exc))

    def _run_camera_preview_setup(self) -> None:
        try:
            _, config_path = self._load_config_or_defaults()
            seconds = float(self.camera_preview_seconds_var.get().strip())
            if seconds <= 0:
                raise ValueError("Preview duration must be > 0")

            width_text = self.camera_preview_width_var.get().strip()
            height_text = self.camera_preview_height_var.get().strip()
            width = int(width_text) if width_text else None
            height = int(height_text) if height_text else None
            if width is not None and width <= 0:
                raise ValueError("Width override must be > 0")
            if height is not None and height <= 0:
                raise ValueError("Height override must be > 0")

            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(
                tk.END,
                "Starting camera preview. A preview window should open now.\n",
            )
            self.update_idletasks()
            preview_text = self._run_camera_preview_subprocess(
                config_path=config_path,
                preview_seconds=seconds,
                window="QT",
                width=width,
                height=height,
            )
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(tk.END, preview_text)
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            self._show_error("Camera preview failed", str(exc))

    def _run_camera_tracking_test_setup(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            seconds = float(self.camera_test_seconds_var.get().strip())
            if seconds <= 0:
                raise ValueError("Tracking test duration must be > 0")
            display_width = int(self.camera_test_display_width_var.get().strip())
            if display_width <= 0:
                raise ValueError("Display width must be > 0")

            dictionary = self._normalize_display_dictionary(self.camera_test_dictionary_var.get())
            box_preset = self.camera_test_box_preset_var.get().strip().lower()
            if box_preset == "auto":
                box_preset = None

            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(
                tk.END,
                "Starting live tracking test. Press ESC in the OpenCV window to stop early.\n",
            )
            self.update_idletasks()
            result = run_camera_tracking_test(
                config=config,
                test_seconds=seconds,
                display_width=display_width,
                dictionary_name=dictionary,
                box_preset=box_preset,
                show_rejected=bool(self.camera_test_show_rejected_var.get()),
                use_clahe=not bool(self.camera_test_no_clahe_var.get()),
            )
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(tk.END, format_tracking_test_result(result))
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            self._show_error("Live tracking test failed", str(exc))

    def _run_full_camera_setup_check(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            preview_seconds = float(self.camera_preview_seconds_var.get().strip())
            test_seconds = float(self.camera_test_seconds_var.get().strip())
            display_width = int(self.camera_test_display_width_var.get().strip())
            if preview_seconds <= 0 or test_seconds <= 0:
                raise ValueError("Both preview and test durations must be > 0")
            if display_width <= 0:
                raise ValueError("Display width must be > 0")

            width_text = self.camera_preview_width_var.get().strip()
            height_text = self.camera_preview_height_var.get().strip()
            width = int(width_text) if width_text else None
            height = int(height_text) if height_text else None
            if width is not None and width <= 0:
                raise ValueError("Width override must be > 0")
            if height is not None and height <= 0:
                raise ValueError("Height override must be > 0")

            dictionary = self._normalize_display_dictionary(self.camera_test_dictionary_var.get())
            box_preset = self.camera_test_box_preset_var.get().strip().lower()
            if box_preset == "auto":
                box_preset = None

            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(
                tk.END,
                (
                    "Running full camera setup check.\n"
                    "Step A: Preview starts now.\n"
                ),
            )
            self.update_idletasks()
            preview_text = self._run_camera_preview_subprocess(
                config_path=config_path,
                preview_seconds=preview_seconds,
                window="QT",
                width=width,
                height=height,
            )

            self.camera_setup_output.insert(
                tk.END,
                "Step B: Live tracking test starts now. Press ESC to stop early.\n",
            )
            self.update_idletasks()
            tracking_result = run_camera_tracking_test(
                config=config,
                test_seconds=test_seconds,
                display_width=display_width,
                dictionary_name=dictionary,
                box_preset=box_preset,
                show_rejected=bool(self.camera_test_show_rejected_var.get()),
                use_clahe=not bool(self.camera_test_no_clahe_var.get()),
            )

            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(tk.END, preview_text)
            self.camera_setup_output.insert(tk.END, "\n\n")
            self.camera_setup_output.insert(tk.END, format_tracking_test_result(tracking_result))
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            self._show_error("Full camera setup check failed", str(exc))

    def _run_thermal_check_setup(self) -> None:
        if not self._thermal_enabled_from_editor():
            self._refresh_thermal_controls_state()
            self._show_error(
                "Thermal check disabled",
                "Enable 'Use thermal camera' in Config Editor before running thermal camera actions.",
            )
            return
        try:
            config, _ = self._load_effective_action_config()
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(
                tk.END,
                "Running thermal camera check against the current Config Editor snapshot.\n",
            )
            self.update_idletasks()
            result = run_thermal_check(config=config)
            self._thermal_check_result = result
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(tk.END, format_thermal_check_result(result))
            if self._thermal_check_can_apply():
                self._refresh_thermal_controls_state()
                self.camera_setup_output.insert(
                    tk.END,
                    "\n\nThermal apply is available. Click 'Apply Detected Thermal Settings' to write the stable "
                    "path and Y16 settings into config.",
                )
            else:
                self._refresh_thermal_controls_state()
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            self._refresh_thermal_controls_state()
            self._show_error("Thermal check failed", str(exc))

    def _thermal_check_can_apply(self) -> bool:
        if self._thermal_check_result is None:
            return False
        try:
            apply_detected_thermal_config(load_defaults(), self._thermal_check_result)
            return True
        except Exception:
            return False

    def _apply_detected_thermal_settings_setup(self) -> None:
        if not self._thermal_enabled_from_editor():
            self._refresh_thermal_controls_state()
            self._show_error(
                "Thermal apply disabled",
                "Enable 'Use thermal camera' in Config Editor before applying thermal settings.",
            )
            return
        if self._thermal_check_result is None:
            self._show_error("Apply thermal settings failed", "Run Thermal Check first.")
            return
        try:
            config, config_path = self._load_effective_action_config()
            updated = apply_detected_thermal_config(config, self._thermal_check_result)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="thermal_check_apply",
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(
                tk.END,
                (
                    "Applied detected thermal settings to config.\n"
                    f"Config path: {config_path}\n"
                    f"thermal.device_path: {updated.get('thermal', {}).get('device_path')}\n"
                    f"thermal.pixel_format: {updated.get('thermal', {}).get('pixel_format')}\n"
                    f"thermal.enabled: {updated.get('thermal', {}).get('enabled')}"
                    + (f"\n{history_note}" if history_note else "")
                ),
            )
            self._refresh_thermal_controls_state()
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            self._show_error("Apply thermal settings failed", str(exc))

    def _run_camera_preview_subprocess(
        self,
        *,
        config_path: Path,
        preview_seconds: float,
        window: str,
        width: int | None,
        height: int | None,
    ) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        bbx_path = repo_root / "bbx.py"
        requested_window = "QT"

        def looks_like_backend_failure(stdout_text: str, stderr_text: str) -> bool:
            combined = f"{stdout_text}\n{stderr_text}".lower()
            failure_markers = [
                "commit failed",
                "could not connect to display",
                "no qt platform plugin could be initialized",
                "failed to initialize egl",
                "qt.qpa.xcb",
                "qxcbconnection",
                "could not load the qt platform plugin",
            ]
            return any(marker in combined for marker in failure_markers)

        command = [
            sys.executable,
            str(bbx_path),
            "camera-preview",
            "--config",
            str(config_path),
            "--seconds",
            f"{float(preview_seconds):.3f}",
            "--window",
            requested_window,
        ]
        if width is not None:
            command.extend(["--width", str(int(width))])
        if height is not None:
            command.extend(["--height", str(int(height))])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_camera_safe_env(),
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        backend_failed = looks_like_backend_failure(stdout, stderr)
        if proc.returncode == 0 and not backend_failed:
            return stdout or "Camera preview completed."

        details = [
            f"Camera preview failed with backend {requested_window}.",
            f"Exit code: {proc.returncode}",
            f"Detected backend failure markers: {'yes' if backend_failed else 'no'}",
            f"Command: {' '.join(shlex.quote(part) for part in command)}",
        ]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(details))

    def _run_camera_reset_subprocess(self, *, config_path: Path) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        bbx_path = repo_root / "bbx.py"
        command = [
            sys.executable,
            str(bbx_path),
            "camera-reset",
            "--config",
            str(config_path),
        ]

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_camera_safe_env(),
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode == 0:
            return stdout or "Camera reset completed."

        details = [
            "Camera reset failed.",
            f"Exit code: {proc.returncode}",
            f"Command: {' '.join(shlex.quote(part) for part in command)}",
        ]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(details))

    def _refresh_roadmap(self, *, select_tab: bool = True) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            items = build_roadmap(config, config_path)
            self._render_roadmap_steps(items)
            if select_tab and str(self.roadmap_tab) in self.notebook.tabs():
                self.notebook.select(self.roadmap_tab)
        except Exception as exc:
            self._show_error("Roadmap failed", str(exc))

    def _run_fps_report(self) -> None:
        try:
            recording_seconds = self.recording_seconds_var.get().strip()
            seconds = float(recording_seconds) if recording_seconds else None
            timestamps = self.timestamps_path_var.get().strip() or None
            report = build_fps_report(
                self.video_path_var.get().strip(),
                timestamps_path=timestamps,
                recording_seconds=seconds,
            )
            self.fps_output.delete("1.0", tk.END)
            self.fps_output.insert(tk.END, format_fps_report(report))
        except Exception as exc:
            self._show_error("FPS report failed", str(exc))

    def _start_live_fps_report(self) -> None:
        if self._fps_live_thread and self._fps_live_thread.is_alive():
            self._show_info("Live FPS test running", "A live FPS capture is already running.")
            return
        if self._fps_camera_reset_thread and self._fps_camera_reset_thread.is_alive():
            self._show_info("Camera reset running", "Wait for the camera reset to finish before starting a live FPS test.")
            return

        try:
            config, _ = self._load_effective_action_config()
            live_seconds_text = self.fps_live_seconds_var.get().strip()
            live_seconds = (
                float(live_seconds_text)
                if live_seconds_text
                else float(config.get("capture", {}).get("recording_seconds", 10.0))
            )
            if live_seconds <= 0:
                raise ValueError("Live test seconds must be > 0")
        except Exception as exc:
            self._show_error("Invalid live FPS settings", str(exc))
            return

        self._fps_live_error = None
        self._fps_live_report = None
        self._fps_live_capture = None
        self.fps_live_status_var.set("Recording test clip...")
        self.fps_live_run_btn.config(state=tk.DISABLED)
        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(
            tk.END,
            (
                "Recording live FPS test clip...\n"
                f"Requested seconds: {live_seconds:.2f}\n"
                f"Using Config Editor snapshot => codec {str(config.get('camera', {}).get('codec', 'mp4')).strip().lower()}\n"
                f"Using current camera config from BumbleBox => {float(config.get('camera', {}).get('fps_target', 0.0)):.1f} FPS\n"
            ),
        )

        self._fps_live_thread = threading.Thread(
            target=self._run_live_fps_report_worker,
            args=(config, live_seconds),
            daemon=True,
        )
        self._fps_live_thread.start()
        self.after(200, self._poll_live_fps_report)

    def _set_fps_reset_status(self, source: str | None, text: str) -> None:
        if source == "sweep":
            self.fps_sweep_reset_status_var.set(text)
        elif source == "live":
            self.fps_live_reset_status_var.set(text)

    def _start_fps_camera_reset(self, source: str) -> None:
        if self._fps_camera_reset_thread and self._fps_camera_reset_thread.is_alive():
            self._show_info("Camera reset running", "A camera reset is already running.")
            return
        if self._fps_live_thread and self._fps_live_thread.is_alive():
            self._show_info("Live FPS test running", "Wait for the live FPS test to finish before resetting the camera.")
            return
        if self._fps_sweep_thread and self._fps_sweep_thread.is_alive():
            self._show_info("FPS sweep running", "Wait for the FPS sweep to finish before resetting the camera.")
            return

        try:
            _config, config_path = self._load_config_or_defaults()
        except Exception as exc:
            self._show_error("Camera reset failed", str(exc))
            return

        self._fps_camera_reset_error = None
        self._fps_camera_reset_result = None
        self._fps_camera_reset_source = source
        self._set_fps_reset_status(source, "Resetting...")
        self.fps_camera_reset_btn.config(state=tk.DISABLED)
        if hasattr(self, "fps_sweep_reset_btn"):
            self.fps_sweep_reset_btn.config(state=tk.DISABLED)
        self.fps_output.insert(tk.END, "\nResetting camera...\n")

        self._fps_camera_reset_thread = threading.Thread(
            target=self._run_fps_camera_reset_worker,
            args=(config_path,),
            daemon=True,
        )
        self._fps_camera_reset_thread.start()
        self.after(200, self._poll_fps_camera_reset)

    def _run_fps_camera_reset_worker(self, config_path: Path) -> None:
        try:
            self._fps_camera_reset_result = self._run_camera_reset_subprocess(config_path=config_path)
        except Exception as exc:
            self._fps_camera_reset_error = str(exc)

    def _poll_fps_camera_reset(self) -> None:
        if self._fps_camera_reset_thread and self._fps_camera_reset_thread.is_alive():
            self.after(200, self._poll_fps_camera_reset)
            return

        self.fps_camera_reset_btn.config(state=tk.NORMAL)
        if hasattr(self, "fps_sweep_reset_btn"):
            self.fps_sweep_reset_btn.config(state=tk.NORMAL)
        source = self._fps_camera_reset_source
        if self._fps_camera_reset_error:
            self._set_fps_reset_status(source, "Failed")
            self.fps_output.insert(tk.END, f"Camera reset failed: {self._fps_camera_reset_error}\n")
            self._show_error("Camera reset failed", self._fps_camera_reset_error)
            self._fps_camera_reset_source = None
            return

        result_text = self._fps_camera_reset_result
        if result_text is None:
            self._set_fps_reset_status(source, "No result")
            self.fps_output.insert(tk.END, "Camera reset ended without a result.\n")
            self._fps_camera_reset_source = None
            return

        self._set_fps_reset_status(source, "Completed")
        self.fps_output.insert(tk.END, "\n" + result_text.strip() + "\n")
        self._fps_camera_reset_source = None

    def _run_live_fps_report_worker(self, config: dict, live_seconds: float) -> None:
        try:
            capture_result = record_live_test_clip(config, recording_seconds=live_seconds)
            report = build_fps_report(
                capture_result.video_path,
                timestamps_path=capture_result.timestamp_path,
                recording_seconds=live_seconds,
            )
            self._fps_live_capture = capture_result
            self._fps_live_report = report
        except Exception as exc:
            self._fps_live_error = str(exc)

    def _poll_live_fps_report(self) -> None:
        if self._fps_live_thread and self._fps_live_thread.is_alive():
            self.after(200, self._poll_live_fps_report)
            return

        self.fps_live_run_btn.config(state=tk.NORMAL)
        if self._fps_live_error:
            self.fps_live_status_var.set("Failed")
            self.fps_output.insert(tk.END, f"\nError: {self._fps_live_error}\n")
            self._show_error("Live FPS test failed", self._fps_live_error)
            return

        capture_result = self._fps_live_capture
        report = self._fps_live_report
        if capture_result is None or report is None:
            self.fps_live_status_var.set("No result")
            self.fps_output.insert(tk.END, "\nLive FPS test ended without a report.\n")
            return

        self.fps_live_status_var.set("Completed")
        self.video_path_var.set(str(capture_result.video_path))
        self.timestamps_path_var.set(str(capture_result.timestamp_path or ""))
        self.recording_seconds_var.set(f"{float(capture_result.requested_recording_seconds):.3f}")

        lines = [
            "Live FPS test capture complete.",
            f"Configured FPS target: {float(capture_result.configured_fps_target):.1f}",
            f"Requested codec: {capture_result.requested_video_codec}",
            f"Saved test clip codec: {capture_result.video_codec}",
            f"Video path: {capture_result.video_path}",
            f"Video size (bytes): {capture_result.video_size_bytes}",
            f"Midpoint PNG: {capture_result.recording_preview_png_path or '(none)'}",
            f"Timestamps path: {capture_result.timestamp_path or '(none)'}",
            f"Session dir: {capture_result.session_dir}",
            f"Captured frame count (timestamps): {capture_result.frames_captured}",
            f"Capture-estimated FPS: {capture_result.actual_fps}",
            "",
            format_fps_report(report),
        ]
        if capture_result.requested_mp4_codec:
            lines.insert(4, f"Requested MP4 encoder: {capture_result.requested_mp4_codec}")
        if capture_result.video_write_warning:
            lines.insert(8, f"Warning: {capture_result.video_write_warning}")
        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(tk.END, "\n".join(lines))

    def _start_fps_sweep(self) -> None:
        if self._fps_sweep_thread and self._fps_sweep_thread.is_alive():
            self._show_info("FPS sweep running", "An FPS sweep is already running.")
            return
        if self._fps_camera_reset_thread and self._fps_camera_reset_thread.is_alive():
            self._show_info("Camera reset running", "Wait for the camera reset to finish before starting an FPS sweep.")
            return

        try:
            config, config_path = self._load_effective_action_config()
            fps_values_text = self.fps_sweep_values_var.get().strip()
            if fps_values_text:
                fps_values = parse_fps_values(fps_values_text)
            else:
                fps_values = fps_range(
                    float(self.fps_sweep_start_var.get().strip()),
                    float(self.fps_sweep_stop_var.get().strip()),
                    float(self.fps_sweep_step_var.get().strip()),
                )

            probe_seconds = float(self.fps_sweep_probe_seconds_var.get().strip())
            if probe_seconds <= 0:
                raise ValueError("Probe seconds must be > 0")

            assume_ram_text = self.fps_sweep_assume_ram_var.get().strip()
            assume_ram_gb = float(assume_ram_text) if assume_ram_text else None
            if assume_ram_gb is not None and assume_ram_gb <= 0:
                raise ValueError("Assumed RAM must be > 0")
        except Exception as exc:
            self._show_error("Invalid FPS sweep settings", str(exc))
            return

        while not self._fps_sweep_progress_q.empty():
            try:
                self._fps_sweep_progress_q.get_nowait()
            except queue.Empty:
                break

        self._fps_sweep_error = None
        self._fps_sweep_report = None
        self._fps_sweep_output_text = None
        self._fps_sweep_report_has_errors = False
        session_start = self._session_started_iso if bool(self.fps_sweep_session_only_var.get()) else None

        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(
            tk.END,
            (
                "Running FPS sweep...\n"
                f"Using Config Editor snapshot => codec {str(config.get('camera', {}).get('codec', 'mp4')).strip().lower()}, "
                f"mode {str(config.get('pipeline', {}).get('mode', 'record_and_track')).strip().lower()}, "
                f"tracking_source {str(config.get('pipeline', {}).get('tracking_source', 'ram')).strip().lower()}\n"
            ),
        )
        if (
            str(config.get("camera", {}).get("codec", "mp4")).strip().lower() == "mjpeg"
            and str(config.get("pipeline", {}).get("mode", "record_and_track")).strip().lower() == "record_and_track"
            and str(config.get("pipeline", {}).get("tracking_source", "ram")).strip().lower() == "ram"
        ):
            self.fps_output.insert(
                tk.END,
                (
                    "MJPEG override: this sweep will still use the RAM-backed model because "
                    "record_and_track + tracking_source=ram keeps frames in memory during recording.\n"
                ),
            )
        self.fps_sweep_status_var.set(
            "Running..." if bool(self.fps_sweep_mock_var.get()) else "Running in subprocess..."
        )
        self.fps_sweep_run_btn.config(state=tk.DISABLED)

        self._fps_sweep_thread = threading.Thread(
            target=self._run_fps_sweep_worker,
            args=(
                config,
                config_path,
                fps_values,
                probe_seconds,
                assume_ram_gb,
                bool(self.fps_sweep_mock_var.get()),
                session_start,
            ),
            daemon=True,
        )
        self._fps_sweep_thread.start()
        self.after(200, self._poll_fps_sweep)

    def _run_fps_sweep_worker(
        self,
        config: dict,
        config_path: Path,
        fps_values: list[float],
        probe_seconds: float,
        assume_ram_gb: float | None,
        use_mock_camera: bool,
        session_start: str | None,
    ) -> None:
        def progress_callback(done: int, total: int, target_fps: float) -> None:
            self._fps_sweep_progress_q.put((done, total, target_fps))

        try:
            if use_mock_camera:
                report = run_fps_sweep(
                    config=config,
                    fps_values=fps_values,
                    probe_seconds=probe_seconds,
                    assume_ram_gb=assume_ram_gb,
                    use_mock_camera=True,
                    session_start_iso=session_start,
                    progress_callback=progress_callback,
                )
                self._fps_sweep_report = report
                return

            output_text, has_errors = self._run_fps_sweep_subprocess(
                config=config,
                config_path=config_path,
                fps_values=fps_values,
                probe_seconds=probe_seconds,
                assume_ram_gb=assume_ram_gb,
                session_start=session_start,
            )
            self._fps_sweep_output_text = output_text
            self._fps_sweep_report_has_errors = has_errors
        except Exception as exc:
            self._fps_sweep_error = str(exc)

    def _poll_fps_sweep(self) -> None:
        latest_progress = None
        while True:
            try:
                latest_progress = self._fps_sweep_progress_q.get_nowait()
            except queue.Empty:
                break

        if latest_progress:
            done, total, target_fps = latest_progress
            self.fps_sweep_status_var.set(f"Running... {done}/{total} (target {target_fps:.2f})")

        if self._fps_sweep_thread and self._fps_sweep_thread.is_alive():
            self.after(200, self._poll_fps_sweep)
            return

        self.fps_sweep_run_btn.config(state=tk.NORMAL)
        if self._fps_sweep_error:
            self.fps_sweep_status_var.set("Failed")
            self.fps_output.insert(tk.END, f"\nError: {self._fps_sweep_error}\n")
            self._show_error("FPS sweep failed", self._fps_sweep_error)
            return

        if self._fps_sweep_output_text is not None:
            self.fps_sweep_status_var.set(
                "Completed with warnings" if self._fps_sweep_report_has_errors else "Completed"
            )
            self.fps_output.delete("1.0", tk.END)
            self.fps_output.insert(tk.END, self._fps_sweep_output_text)
            if self._fps_sweep_report_has_errors:
                self._show_warning("FPS sweep", "One or more FPS probes failed. Review the output for details.")
            return

        if self._fps_sweep_report is None:
            self.fps_sweep_status_var.set("No result")
            self.fps_output.insert(tk.END, "\nFPS sweep ended without a report.\n")
            return

        self.fps_sweep_status_var.set("Completed")
        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(tk.END, format_fps_sweep_report(self._fps_sweep_report))
        if self._fps_sweep_report.has_errors:
            self._show_warning("FPS sweep", "One or more FPS probes failed. Review the output for details.")

    def _run_fps_sweep_subprocess(
        self,
        *,
        config: dict,
        config_path: Path,
        fps_values: list[float],
        probe_seconds: float,
        assume_ram_gb: float | None,
        session_start: str | None,
    ) -> tuple[str, bool]:
        repo_root = Path(__file__).resolve().parents[1]
        bbx_path = repo_root / "bbx.py"
        with tempfile.TemporaryDirectory(prefix="bbx_fps_sweep_") as temp_dir:
            temp_config_path = Path(temp_dir) / config_path.name
            save_config(temp_config_path, config)
            command = [
                sys.executable,
                str(bbx_path),
                "fps-sweep",
                "--config",
                str(temp_config_path),
                "--fps-values",
                ",".join(f"{float(value):g}" for value in fps_values),
                "--probe-seconds",
                f"{float(probe_seconds):.3f}",
            ]
            if assume_ram_gb is not None:
                command.extend(["--assume-ram-gb", f"{float(assume_ram_gb):.3f}"])
            if session_start:
                command.extend(["--session-start", session_start])

            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=build_camera_safe_env(),
            )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        has_report = "FPS Sweep Report" in stdout

        if proc.returncode == 0 and has_report:
            return stdout, False

        if has_report:
            details = stdout
            if stderr:
                details = f"{details}\n\nstderr:\n{stderr}"
            return details, True

        detail_lines = [
            "FPS sweep failed.",
            f"Exit code: {proc.returncode}",
            f"Command: {' '.join(shlex.quote(part) for part in command)}",
        ]
        if stdout:
            detail_lines.append(f"stdout:\n{stdout}")
        if stderr:
            detail_lines.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(detail_lines))

    def _capture_and_open_calibration_labelme(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            data_root = str(config.get("system", {}).get("data_root", "")).strip()
            if not data_root:
                raise ValueError("system.data_root is empty in config.")
            capture_dir = Path(data_root).expanduser().resolve() / "calibration"

            image_path = capture_calibration_image(
                config,
                output_dir=capture_dir,
                filename_prefix="calibration_capture",
            )
            json_path = image_path.with_suffix(".json")
            self.calibration_labelme_image_path_var.set(str(image_path))
            self.calibration_labelme_json_path_var.set(str(json_path))

            python_override = None
            nest_python_var = getattr(self, "nest_python_var", None)
            if nest_python_var is not None:
                text = str(nest_python_var.get()).strip()
                if text:
                    python_override = text

            process = launch_labelme(
                image_path=str(image_path),
                python_executable=python_override,
                labelmerc_override=str(default_calibration_labelmerc_path()),
            )
            self._calibration_label_pid = process.pid
            command = build_labelme_command(
                image_path=str(image_path),
                python_executable=python_override,
                labelmerc_override=str(default_calibration_labelmerc_path()),
            )
            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                (
                    f"Captured calibration image: {image_path}\n"
                    f"Expected LabelMe JSON after Save: {json_path}\n"
                    f"Launched LabelMe (pid {process.pid}).\n"
                    f"Command: {' '.join(shlex.quote(part) for part in command)}\n\n"
                    "No JSON exists yet until you save in LabelMe.\n"
                    "In LabelMe, draw one line from Point A to Point B on the known-distance endpoints and save.\n"
                    "The first click becomes Point A and the second click becomes Point B.\n"
                    "Then click 'Load LabelMe Points -> Point A/B'."
                ),
            )
        except Exception as exc:
            self._show_error("LabelMe calibration launch failed", str(exc))

    def _load_calibration_points_from_labelme(self) -> None:
        try:
            json_text, note = self._populate_manual_points_from_labelme_json()
            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                (
                    f"Loaded points from: {json_text}\n"
                    f"Point A: {self.manual_point_a.get()}\n"
                    f"Point B: {self.manual_point_b.get()}\n"
                    f"{note}\n\n"
                    "Set the real distance in cm if needed, then click 'Calibrate from Points'."
                ),
            )
        except Exception as exc:
            self._show_error("Load LabelMe points failed", str(exc))

    def _populate_manual_points_from_labelme_json(self) -> tuple[str, str]:
        try:
            json_text = self.calibration_labelme_json_path_var.get().strip()
            if not json_text:
                image_text = self.calibration_labelme_image_path_var.get().strip()
                if image_text:
                    json_text = str(Path(image_text).with_suffix(".json"))
                    self.calibration_labelme_json_path_var.set(json_text)

            if not json_text:
                raise ValueError(
                    "No LabelMe JSON path is set. Capture/open an image first, label it, save, then retry."
                )

            json_path = Path(json_text).expanduser().resolve()
            if not json_path.exists():
                raise FileNotFoundError(
                    f"LabelMe JSON not found yet: {json_path}\n"
                    "After placing points in LabelMe, save the file, then click Load again."
                )

            point_a, point_b, note = extract_points_from_labelme_json(json_text)
            self.manual_point_a.set(f"{point_a[0]:.3f},{point_a[1]:.3f}")
            self.manual_point_b.set(f"{point_b[0]:.3f},{point_b[1]:.3f}")
            return json_text, note
        except Exception:
            raise

    def _load_and_calibrate_from_labelme(self) -> None:
        try:
            json_text, note = self._populate_manual_points_from_labelme_json()
            config, config_path = self._load_config_or_defaults()
            result = calibrate_from_points(
                parse_point(self.manual_point_a.get().strip()),
                parse_point(self.manual_point_b.get().strip()),
                float(self.manual_distance_cm.get().strip()),
            )
            confirmed = messagebox.askyesno(
                "Confirm calibration save",
                (
                    "Load + Calibrate will update and save calibration values in your active config file.\n\n"
                    f"Config file: {config_path}\n"
                    f"Point A: {self.manual_point_a.get()}\n"
                    f"Point B: {self.manual_point_b.get()}\n"
                    f"Real distance (cm): {self.manual_distance_cm.get().strip()}\n"
                    f"Computed pixels/cm: {result.pixels_per_cm:.6f}\n\n"
                    "Save these values now?"
                ),
            )
            if not confirmed:
                self.calibration_output.delete("1.0", tk.END)
                self.calibration_output.insert(
                    tk.END,
                    (
                        "Calibration save cancelled by user.\n"
                        f"Loaded points from: {json_text}\n"
                        f"Point A: {self.manual_point_a.get()}\n"
                        f"Point B: {self.manual_point_b.get()}\n"
                        "No config changes were written."
                    ),
                )
                return
            updated = apply_scale_to_config(config, result)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="calibration_load_and_apply",
            )
            calibration_text = format_calibration(
                result,
                pixel_contact_distance=updated.get("metrics", {}).get("pixel_contact_distance"),
            )
            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                (
                    f"Loaded points from: {json_text}\n"
                    f"Point A: {self.manual_point_a.get()}\n"
                    f"Point B: {self.manual_point_b.get()}\n"
                    f"{note}\n\n"
                    f"{calibration_text}"
                ),
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.calibration_output.insert(tk.END, f"\n\n{history_note}")
        except Exception as exc:
            self._show_error("Load + calibrate failed", str(exc))

    def _calibrate_manual(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            result = calibrate_from_points(
                parse_point(self.manual_point_a.get().strip()),
                parse_point(self.manual_point_b.get().strip()),
                float(self.manual_distance_cm.get().strip()),
            )
            updated = apply_scale_to_config(config, result)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="calibration_manual",
            )

            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                format_calibration(
                    result,
                    pixel_contact_distance=updated.get("metrics", {}).get("pixel_contact_distance"),
                ),
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.calibration_output.insert(tk.END, f"\n\n{history_note}")
        except Exception as exc:
            self._show_error("Manual calibration failed", str(exc))

    def _calibrate_aruco(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            marker_id_text = self.aruco_marker_id.get().strip()
            marker_id = int(marker_id_text) if marker_id_text else None

            result = calibrate_from_aruco_image(
                image_path=self.aruco_image.get().strip(),
                marker_size_mm=float(self.aruco_marker_size_mm.get().strip()),
                dictionary_name=self.aruco_dictionary.get().strip(),
                marker_id=marker_id,
            )
            updated = apply_scale_to_config(config, result)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="calibration_aruco",
            )

            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                format_calibration(
                    result,
                    pixel_contact_distance=updated.get("metrics", {}).get("pixel_contact_distance"),
                ),
            )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.calibration_output.insert(tk.END, f"\n\n{history_note}")
        except Exception as exc:
            self._show_error("ArUco calibration failed", str(exc))

    def _browse_thermal_registration_rgb_image(self) -> None:
        initial_dir = str(Path(self.config_path_var.get()).expanduser().parent)
        selected = filedialog.askopenfilename(
            title="Select RGB registration image",
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self._thermal_registration_summary_path = None
        self.thermal_registration_rgb_image_var.set(selected)

    def _browse_thermal_registration_thermal_image(self) -> None:
        initial_dir = str(Path(self.config_path_var.get()).expanduser().parent)
        selected = filedialog.askopenfilename(
            title="Select thermal registration image",
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self._thermal_registration_summary_path = None
        self.thermal_registration_thermal_image_var.set(selected)

    def _load_latest_thermal_registration_frames(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            data_root_text = str(config.get("system", {}).get("data_root", "")).strip()
            if not data_root_text:
                raise ValueError("system.data_root is empty in config.")
            data_root = Path(data_root_text).expanduser()

            records = list_recent_run_records(data_root, limit=60)
            chosen_payload = None
            chosen_summary_path = None
            for record in records:
                payload = load_run_summary(record.summary_path)
                rgb_path = str(payload.get("recording_preview_png_path") or "").strip()
                thermal_path = str(payload.get("thermal_preview_png_path") or "").strip()
                if rgb_path and thermal_path:
                    chosen_payload = payload
                    chosen_summary_path = record.summary_path
                    break

            if chosen_payload is None or chosen_summary_path is None:
                raise RuntimeError(
                    "No recent run summary contained both an RGB midpoint PNG and a thermal midpoint PNG. "
                    "Run a thermal-enabled recording first, then try again."
                )

            rgb_path = Path(str(chosen_payload.get("recording_preview_png_path", "")).strip()).expanduser().resolve()
            thermal_path = Path(str(chosen_payload.get("thermal_preview_png_path", "")).strip()).expanduser().resolve()
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB midpoint PNG not found: {rgb_path}")
            if not thermal_path.exists():
                raise FileNotFoundError(f"Thermal midpoint PNG not found: {thermal_path}")

            self.thermal_registration_rgb_image_var.set(str(rgb_path))
            self.thermal_registration_thermal_image_var.set(str(thermal_path))
            self._thermal_registration_summary_path = str(chosen_summary_path)

            session_name = str(chosen_payload.get("session_name", "")).strip() or chosen_summary_path.stem
            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                (
                    "Loaded latest session midframes for thermal registration.\n"
                    f"Session: {session_name}\n"
                    f"Run summary: {chosen_summary_path}\n"
                    f"RGB image: {rgb_path}\n"
                    f"Thermal image: {thermal_path}\n\n"
                    "Next step: click 'Pick Points + Register + Save' and mark the same landmarks in the same order."
                ),
            )
        except Exception as exc:
            self._show_error("Load thermal registration frames failed", str(exc))

    def _register_thermal_from_images(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            rgb_path_text = self.thermal_registration_rgb_image_var.get().strip()
            thermal_path_text = self.thermal_registration_thermal_image_var.get().strip()
            if not rgb_path_text:
                raise ValueError("Choose an RGB registration image first.")
            if not thermal_path_text:
                raise ValueError("Choose a thermal registration image first.")

            rgb_path = Path(rgb_path_text).expanduser().resolve()
            thermal_path = Path(thermal_path_text).expanduser().resolve()
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB registration image not found: {rgb_path}")
            if not thermal_path.exists():
                raise FileNotFoundError(f"Thermal registration image not found: {thermal_path}")

            data_root_text = str(config.get("system", {}).get("data_root", "")).strip()
            if not data_root_text:
                raise ValueError("system.data_root is empty in config.")
            data_root = Path(data_root_text).expanduser()
            output_dir = (
                data_root.resolve()
                / "calibration"
                / "thermal_registration"
                / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )

            registration = register_rgb_thermal_pair(
                rgb_image_path=rgb_path,
                thermal_image_path=thermal_path,
                output_dir=output_dir,
                min_points=4,
            )
            tracking_overlay_note = None
            summary_path_text = str(getattr(self, "_thermal_registration_summary_path", "") or "").strip()
            if summary_path_text:
                try:
                    run_summary = load_run_summary(summary_path_text)
                    summary_rgb = str(run_summary.get("recording_preview_png_path") or "").strip()
                    summary_thermal = str(run_summary.get("thermal_preview_png_path") or "").strip()
                    if summary_rgb and summary_thermal:
                        summary_rgb_path = Path(summary_rgb).expanduser().resolve()
                        summary_thermal_path = Path(summary_thermal).expanduser().resolve()
                        if summary_rgb_path == rgb_path and summary_thermal_path == thermal_path:
                            registration, tracking_overlay_note = annotate_registration_overlay_with_session_tracking(
                                registration=registration,
                                run_summary=run_summary,
                            )
                except Exception as exc:
                    tracking_overlay_note = f"Tracked temperature overlay skipped: {exc}"
            updated = apply_thermal_registration_to_config(config, registration)
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                updated,
                reason="thermal_registration_manual",
            )

            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(tk.END, format_thermal_registration(registration))
            if tracking_overlay_note:
                self.calibration_output.insert(tk.END, f"\n\nTracking overlay note: {tracking_overlay_note}")
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if history_note:
                self.calibration_output.insert(tk.END, f"\n\n{history_note}")
            self.notebook.select(self.calibration_tab)
        except Exception as exc:
            self._show_error("Thermal registration failed", str(exc))

    def _run_once_now(self) -> None:
        try:
            config, config_path = self._load_effective_action_config()
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="gui_run_once_apply",
            )
            run_config = deepcopy(config)
            run_config.setdefault("runtime", {})
            if self.run_mock_var.get():
                run_config["runtime"]["use_mock_camera"] = True

            mode_override = self.run_mode_var.get().strip() or None
            summary = run_once(config=run_config, mode_override=mode_override)
            self.run_output.delete("1.0", tk.END)
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            lines = [f"Applied Config Editor snapshot to {config_path}"]
            if history_note:
                lines.append(history_note)
            if self.run_mock_var.get():
                lines.append("Run-once override: mock camera enabled for this run only.")
            lines.extend(["", format_run_summary(summary)])
            self.run_output.insert(tk.END, "\n".join(lines))
            self._refresh_run_history()
            self.notebook.select(self.run_tab)
        except Exception as exc:
            self._show_error("Run failed", str(exc))

    def _generate_systemd_units(self) -> None:
        try:
            config, config_path = self._load_effective_action_config()
            snapshot_path, history_warning = self._save_config_with_history(
                config_path,
                config,
                reason="gui_systemd_generate_units",
            )
            result = write_systemd_units(
                config=config,
                config_path=config_path,
                output_dir=self.systemd_output_dir_var.get().strip(),
            )
            self.run_output.delete("1.0", tk.END)
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            self.run_output.insert(
                tk.END,
                (
                    f"Applied Config Editor snapshot to {config_path}\n"
                    + (f"{history_note}\n\n" if history_note else "\n")
                    + format_systemd_result(result)
                ),
            )
            self.notebook.select(self.run_tab)
        except Exception as exc:
            self._show_error("Systemd generation failed", str(exc))

    def _build_systemd_permission_guidance(
        self,
        *,
        config: dict[str, object],
        config_path: str | Path,
        output_dir: str,
    ) -> str:
        config_path = str(config_path)
        scope = str((config.get("scheduling") or {}).get("scope", "system")).strip().lower()
        if scope != "system":
            return ""
        install_cmd = build_systemd_action_sudo_command(
            action="install",
            config_path=config_path,
            output_dir=output_dir,
        )
        status_cmd = build_systemd_action_sudo_command(
            action="status",
            config_path=config_path,
            output_dir=output_dir,
        )
        return (
            "Scheduler scope is set to 'system', which writes timer files into /etc/systemd/system and "
            "requires root privileges.\n\n"
            "Options:\n"
            "1. Keep system scope and run this from a terminal:\n"
            f"   {install_cmd}\n"
            f"   {status_cmd}\n\n"
            "2. Or change Config Editor -> Scheduler scope to 'user' if per-user timers are acceptable. "
            "User-scope timers can be installed from the GUI without sudo. If they should continue after logout, "
            "run: loginctl enable-linger $(whoami)"
        )

    def _coerce_basic_automation_config(self, config: dict) -> tuple[dict, list[str]]:
        notes: list[str] = []
        ui_mode = str(self.ui_mode_var.get()).strip().lower()
        if ui_mode != "basic":
            return config, notes
        scheduling = config.setdefault("scheduling", {})
        backend_before = str(scheduling.get("backend", "systemd")).strip().lower()
        scope_before = str(scheduling.get("scope", "user")).strip().lower()
        if backend_before != "systemd":
            scheduling["backend"] = "systemd"
            notes.append("Basic-mode automation forced scheduler backend to systemd.")
        if scope_before != "user":
            scheduling["scope"] = "user"
            notes.append("Basic-mode automation forced scheduler scope to user so the GUI can start timers without sudo.")
        return config, notes

    def _start_automated_recording(self) -> None:
        try:
            config, config_path = self._load_effective_action_config()
            config, automation_notes = self._coerce_basic_automation_config(config)
            scheduling = config.setdefault("scheduling", {})
            scheduling["enabled"] = True
            output_dir = self.systemd_output_dir_var.get().strip()

            write_result = write_systemd_units(
                config=config,
                config_path=config_path,
                output_dir=output_dir,
            )
            install_result = run_systemd_action(
                config=config,
                action="install",
                output_dir=output_dir,
                config_path=config_path,
            )

            self.run_output.delete("1.0", tk.END)
            snapshot_path = None
            history_warning = None
            if install_result.success:
                snapshot_path, history_warning = self._save_config_with_history(
                    config_path,
                    config,
                    reason="gui_systemd_start_automation",
                )
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            lines = [
            ]
            if install_result.success:
                lines.append(f"Applied Config Editor snapshot to {config_path}")
                if history_note:
                    lines.append(history_note)
            else:
                lines.append("Config was not updated because timer installation did not complete successfully.")
            if automation_notes:
                lines.extend(automation_notes)
            lines.extend(
                [
                    "",
                    "Automated recording startup sequence",
                    "1. Wrote timer/service files for this config.",
                    "2. Installed those files into systemd.",
                    "3. The install step enabled the timers and started them immediately.",
                    "",
                    "Step 1: Write timer files",
                    format_systemd_result(write_result),
                    "",
                    "Step 2: Install timers into systemd and start them",
                    format_systemd_action_result(install_result),
                ]
            )
            self.run_output.insert(tk.END, "\n".join(lines))
            self.notebook.select(self.run_tab)
            if not install_result.success:
                guidance = self._build_systemd_permission_guidance(
                    config=config,
                    config_path=config_path,
                    output_dir=output_dir,
                )
                if guidance and "Permission denied while copying to" in (install_result.note or ""):
                    self.run_output.insert(tk.END, "\n\nPermission guidance\n" + guidance)
                    self._show_error("Automated recording start failed", guidance)
                else:
                    self._show_error(
                        "Automated recording start failed",
                        "One or more timer setup steps failed. See Run & Schedule Results for details.",
                    )
            else:
                self._show_automation_started_popup(config)
        except Exception as exc:
            self._show_error("Automated recording start failed", str(exc))

    def _stop_automated_recording(self) -> None:
        try:
            self._systemd_action("disable")
        except Exception as exc:
            self._show_error("Automated recording stop failed", str(exc))

    def _systemd_action(self, action: str) -> None:
        try:
            if action in {"install", "enable"}:
                config, config_path = self._load_effective_action_config()
                config.setdefault("scheduling", {})
                config["scheduling"]["enabled"] = True
            else:
                config, config_path = self._load_config_or_defaults()
                if action == "disable":
                    config.setdefault("scheduling", {})
                    config["scheduling"]["enabled"] = False
            snapshot_path = None
            history_warning = None
            result = run_systemd_action(
                config=config,
                action=action,
                output_dir=self.systemd_output_dir_var.get().strip(),
                config_path=config_path,
            )
            if result.success and action in {"install", "enable", "disable"}:
                snapshot_path, history_warning = self._save_config_with_history(
                    config_path,
                    config,
                    reason=f"gui_systemd_{action}",
                )
            self.run_output.delete("1.0", tk.END)
            history_note = self._format_config_history_note(snapshot_path, history_warning)
            if snapshot_path is not None or history_warning is not None:
                self.run_output.insert(
                    tk.END,
                    (
                        f"Applied Config Editor snapshot to {config_path}\n"
                        + (f"{history_note}\n\n" if history_note else "\n")
                    ),
                )
            self.run_output.insert(tk.END, format_systemd_action_result(result))
            guidance = self._build_systemd_permission_guidance(
                config=config,
                config_path=config_path,
                output_dir=self.systemd_output_dir_var.get().strip(),
            )
            if action == "install" and not result.success and guidance and "Permission denied while copying to" in (result.note or ""):
                self.run_output.insert(tk.END, "\n\nPermission guidance\n" + guidance)
                self._show_error("Systemd install failed", guidance)
            self.notebook.select(self.run_tab)
        except Exception as exc:
            self._show_error(f"Systemd {action} failed", str(exc))

    def _install_gui_shortcut(self) -> None:
        try:
            result = install_gui_shortcut()
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, format_gui_shortcut_result(result))
            self.notebook.select(self.run_tab)
            self._show_info(
                "GUI Desktop Icon",
                f"Desktop launcher created:\n{result.desktop_entry_path}",
            )
        except Exception as exc:
            self._show_error("GUI shortcut install failed", str(exc))

    def _refresh_runtime_alerts(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            alerts = build_runtime_alerts(config)
            text = format_runtime_alerts(alerts)
            self.runtime_alert_output.delete("1.0", tk.END)
            self.runtime_alert_output.insert(tk.END, text)
        except Exception as exc:
            self.runtime_alert_output.delete("1.0", tk.END)
            self.runtime_alert_output.insert(tk.END, f"[FAIL] Runtime alerts failed: {exc}")

    def _apply_run_history_records(self, records) -> None:
        self._run_history_paths.clear()

        for item in self.run_history_tree.get_children():
            self.run_history_tree.delete(item)

        for index, record in enumerate(records):
            iid = str(index)
            self._run_history_paths[iid] = str(record.summary_path)
            self.run_history_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    record.started_at,
                    record.mode,
                    f"{record.actual_fps:.3f}",
                    "yes" if record.success else "no",
                    record.warning_count,
                    record.error_count,
                    record.session_name,
                ),
            )
        self._set_run_history_detail_text("")

    def _poll_run_history_refresh(self) -> None:
        thread = self._run_history_thread
        if thread is not None and thread.is_alive():
            self.after(100, self._poll_run_history_refresh)
            return

        button = getattr(self, "run_history_refresh_btn", None)
        if button is not None:
            button.configure(state=tk.NORMAL)

        error = self._run_history_error
        records = self._run_history_records or []
        self._run_history_thread = None
        self._run_history_error = None
        self._run_history_records = None

        if error:
            self._set_run_history_detail_text("")
            self._show_error("Run history failed", error)
            return

        self._apply_run_history_records(records)

    def _refresh_run_history(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            data_root = config["system"]["data_root"]
            self._refresh_runtime_alerts()

            existing = self._run_history_thread
            if existing is not None and existing.is_alive():
                return

            button = getattr(self, "run_history_refresh_btn", None)
            if button is not None:
                button.configure(state=tk.DISABLED)

            self._run_history_error = None
            self._run_history_records = None

            def _worker() -> None:
                try:
                    self._run_history_records = list_recent_run_records(data_root, limit=40)
                except Exception as exc:
                    self._run_history_error = str(exc)

            self._run_history_thread = threading.Thread(target=_worker, daemon=True)
            self._run_history_thread.start()
            self.after(100, self._poll_run_history_refresh)
        except Exception as exc:
            self._show_error("Run history failed", str(exc))

    def _set_run_history_detail_text(self, text: str) -> None:
        self.run_history_detail.delete("1.0", tk.END)
        if text.strip():
            self.run_history_detail.insert(tk.END, text)
            if self.run_history_detail_frame.winfo_manager() != "pack":
                self.run_history_detail_frame.pack(fill=tk.X, expand=False)
        else:
            if self.run_history_detail_frame.winfo_manager() == "pack":
                self.run_history_detail_frame.pack_forget()

    def _on_run_history_select(self, _event=None) -> None:
        selected = self.run_history_tree.selection()
        if not selected:
            self._set_run_history_detail_text("")
            return
        iid = selected[0]
        path = self._run_history_paths.get(iid)
        if not path:
            self._set_run_history_detail_text("")
            return
        try:
            payload = load_run_summary(path)
            self._set_run_history_detail_text(json.dumps(payload, indent=2))
        except Exception as exc:
            self._show_error("Load summary failed", str(exc))

    def _export_selected_run_bundle(self) -> None:
        selected = self.run_history_tree.selection()
        if not selected:
            self._show_error("No run selected", "Select a run from Recent Runs first.")
            return

        iid = selected[0]
        summary_path = self._run_history_paths.get(iid)
        if not summary_path:
            self._show_error("Missing run path", "Could not resolve selected run summary path.")
            return

        try:
            config_path = Path(self.config_path_var.get().strip()).expanduser().resolve()
            config_for_bundle = str(config_path) if config_path.exists() and config_path.is_file() else None
            result = export_run_bundle(
                summary_path=summary_path,
                output_dir=self.bundle_output_dir_var.get().strip() or str(Path.cwd() / "bundles"),
                config_path=config_for_bundle,
                include_video=not bool(self.bundle_skip_video_var.get()),
                include_all_session_files=not bool(self.bundle_core_only_var.get()),
                zip_bundle=bool(self.bundle_zip_var.get()),
            )
            text = format_bundle_export_result(result)
            self._set_run_history_detail_text(text)
            self._show_info("Export complete", text)
        except Exception as exc:
            self._show_error("Bundle export failed", str(exc))


def launch(
    *,
    open_video_range_bounds: bool = False,
    video_range_manifest: str | None = None,
    video_range_source_root: str | None = None,
    video_range_bounds_file: str | None = None,
) -> None:
    app = BumbleBoxV2GUI()
    if open_video_range_bounds:
        app.after_idle(
            lambda: app.open_video_range_bounds_workflow(
                manifest_path=video_range_manifest,
                source_root=video_range_source_root,
                bounds_file=video_range_bounds_file,
            )
        )
    app.mainloop()
