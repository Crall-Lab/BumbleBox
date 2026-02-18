from __future__ import annotations

import json
import os
import queue
import shutil
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .calibration import (
    apply_scale_to_config,
    calibrate_from_aruco_image,
    calibrate_from_points,
    format_calibration,
    parse_point,
)
from .camera_setup import (
    format_camera_preview_result,
    format_tracking_test_result,
    run_camera_preview,
    run_camera_tracking_test,
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
    build_nest_labeling_command,
    check_nest_labeling_environment,
    default_script_path,
    format_nest_labeling_environment,
    launch_nest_labeling,
)
from .roadmap import render_roadmap
from .runtime_alerts import build_runtime_alerts, format_runtime_alerts
from .run_bundle import export_run_bundle, format_bundle_export_result
from .run_engine import format_run_summary, run_once
from .schedule_check import format_schedule_check_report, run_schedule_check
from .storage_manager import (
    build_storage_setup_sudo_command,
    format_storage_setup_result,
    format_storage_status_report,
    get_storage_status,
    setup_storage_auto_mount,
)
from .status_history import list_recent_run_records, load_run_summary
from .systemd_units import (
    format_systemd_action_result,
    format_systemd_result,
    run_systemd_action,
    write_systemd_units,
)


class BumbleBoxV2GUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BumbleBox V2")
        self.geometry("980x680")
        self.minsize(920, 620)

        self.config_path_var = tk.StringVar(value=str(DEFAULT_USER_CONFIG_PATH))
        self.ui_mode_var = tk.StringVar(value="basic")
        self.config_fields: dict[str, tuple[tk.Variable, type]] = {}
        self._tab_lookup: dict[str, ttk.Frame] = {}
        self._advanced_widgets: list[tuple[tk.Widget, str, dict[str, object]]] = []
        self._advanced_widget_ids: set[str] = set()
        self._run_history_paths: dict[str, str] = {}
        self._nest_label_pid: int | None = None
        self._optimize_thread: threading.Thread | None = None
        self._optimize_error: str | None = None
        self._optimize_warning: str | None = None
        self._optimize_result = None
        self._optimize_applied_config: str | None = None
        self._optimize_progress_q: queue.Queue[tuple[int, int]] = queue.Queue()
        self._fps_sweep_thread: threading.Thread | None = None
        self._fps_sweep_error: str | None = None
        self._fps_sweep_report = None
        self._fps_sweep_progress_q: queue.Queue[tuple[int, int, float]] = queue.Queue()
        self._session_started_iso = datetime.now().isoformat(timespec="seconds")

        self._build_header()
        self._build_notebook()

    def _build_header(self) -> None:
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Config path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.config_path_var, width=62).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Create Default Config", command=self._create_config).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Button(frame, text="Open Current Roadmap", command=self._refresh_roadmap).grid(row=0, column=3, sticky="w", padx=4)

        mode_row = ttk.Frame(frame)
        mode_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
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
        ttk.Label(
            mode_row,
            text="Basic hides rarely used tuning controls.",
        ).pack(side=tk.LEFT)

        setup_row = ttk.Frame(frame)
        setup_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(setup_row, text="Setup order:").pack(side=tk.LEFT)
        setup_steps = [
            ("1. Doctor", "doctor"),
            ("2. Camera Setup", "camera_setup"),
            ("3. Calibration", "calibration"),
            ("4. FPS Report", "fps"),
            ("5. Schedule Check", "schedule_check"),
            ("6. Run & Schedule", "run"),
        ]
        for label, key in setup_steps:
            ttk.Button(
                setup_row,
                text=label,
                command=lambda tab_key=key: self._go_to_tab(tab_key),
            ).pack(side=tk.LEFT, padx=3)

        frame.columnconfigure(1, weight=1)

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.doctor_tab = ttk.Frame(self.notebook, padding=12)
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

        self.notebook.add(self.doctor_tab, text="Doctor")
        self.notebook.add(self.camera_setup_tab, text="Camera Setup")
        self.notebook.add(self.roadmap_tab, text="Roadmap")
        self.notebook.add(self.config_tab, text="Config Editor")
        self.notebook.add(self.fps_tab, text="FPS Report")
        self.notebook.add(self.calibration_tab, text="Calibration")
        self.notebook.add(self.schedule_check_tab, text="Schedule Check")
        self.notebook.add(self.optimize_tracking_tab, text="Optimize Tracking")
        self.notebook.add(self.nest_label_tab, text="Nest Labeling")
        self.notebook.add(self.fleet_tab, text="Fleet")
        self.notebook.add(self.run_tab, text="Run & Schedule")

        self._build_doctor_tab()
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
            "camera_setup": self.camera_setup_tab,
            "roadmap": self.roadmap_tab,
            "config": self.config_tab,
            "fps": self.fps_tab,
            "calibration": self.calibration_tab,
            "schedule_check": self.schedule_check_tab,
            "optimize_tracking": self.optimize_tracking_tab,
            "nest_labeling": self.nest_label_tab,
            "fleet": self.fleet_tab,
            "run": self.run_tab,
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

    def _go_to_tab(self, tab_key: str) -> None:
        tab = self._tab_lookup.get(tab_key)
        if tab is not None:
            self.notebook.select(tab)

    def _build_doctor_tab(self) -> None:
        controls = ttk.Frame(self.doctor_tab)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Run Doctor", command=self._run_doctor).pack(side=tk.LEFT)

        storage_frame = ttk.LabelFrame(self.doctor_tab, text="Storage Setup", padding=8)
        storage_frame.pack(fill=tk.X, pady=(10, 0))

        self.storage_mount_point_var = tk.StringVar(value="/mnt/bumblebox/data")
        ttk.Label(storage_frame, text="Mount point").grid(row=0, column=0, sticky="w")
        ttk.Entry(storage_frame, textvariable=self.storage_mount_point_var, width=42).grid(
            row=0, column=1, sticky="ew", padx=8, pady=3
        )
        row_buttons = ttk.Frame(storage_frame)
        row_buttons.grid(row=0, column=2, sticky="w")
        ttk.Button(
            row_buttons,
            text="Save Mount Point To Config",
            command=self._save_storage_mount_point_to_config,
        ).pack(side=tk.LEFT)
        ttk.Button(
            row_buttons,
            text="Refresh Storage Status",
            command=self._refresh_storage_status,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            row_buttons,
            text="Setup Storage Auto-Mount",
            command=self._setup_storage_auto_mount,
        ).pack(side=tk.LEFT)

        info = (
            "If storage is already mounted at this path, BumbleBox shows where data is being written. "
            "If not mounted, Setup Storage Auto-Mount configures UUID-based /etc/fstab boot mounting."
        )
        ttk.Label(storage_frame, text=info, wraplength=860, justify=tk.LEFT).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 4)
        )
        storage_frame.columnconfigure(1, weight=1)

        self.storage_output = tk.Text(storage_frame, wrap=tk.WORD, height=7)
        self.storage_output.grid(row=2, column=0, columnspan=3, sticky="nsew")
        storage_frame.rowconfigure(2, weight=1)

        self.doctor_output = tk.Text(self.doctor_tab, wrap=tk.WORD)
        self.doctor_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._load_storage_mount_point_from_config()
        self._refresh_storage_status()

    def _build_camera_setup_tab(self) -> None:
        top = ttk.Frame(self.camera_setup_tab)
        top.pack(fill=tk.X)

        self.camera_preview_seconds_var = tk.StringVar(value="20")
        self.camera_preview_window_var = tk.StringVar(value="QTGL")
        self.camera_preview_width_var = tk.StringVar(value="")
        self.camera_preview_height_var = tk.StringVar(value="")

        self.camera_test_seconds_var = tk.StringVar(value="20")
        self.camera_test_display_width_var = tk.StringVar(value="1280")
        self.camera_test_dictionary_var = tk.StringVar(value="")
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
        ttk.Label(preview, text="Duration (seconds)").grid(row=1, column=0, sticky="w")
        ttk.Entry(preview, textvariable=self.camera_preview_seconds_var, width=8).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(preview, text="Preview window").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            preview,
            textvariable=self.camera_preview_window_var,
            values=["QTGL", "QT", "DRM"],
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(preview, text="Width override (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(preview, textvariable=self.camera_preview_width_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(preview, text="Height override (optional)").grid(row=4, column=0, sticky="w")
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
        ttk.Label(tracking, text="Duration (seconds)").grid(row=1, column=0, sticky="w")
        ttk.Entry(tracking, textvariable=self.camera_test_seconds_var, width=8).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(tracking, text="Display width").grid(row=2, column=0, sticky="w")
        ttk.Entry(tracking, textvariable=self.camera_test_display_width_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(tracking, text="Dictionary override (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(tracking, textvariable=self.camera_test_dictionary_var, width=14).grid(row=3, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(tracking, text="Box preset").grid(row=4, column=0, sticky="w")
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
        ttk.Checkbutton(
            tracking,
            text="Disable CLAHE pre-processing",
            variable=self.camera_test_no_clahe_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        actions = ttk.Frame(tracking)
        actions.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(actions, text="Run Live Tracking Test", command=self._run_camera_tracking_test_setup).pack(side=tk.LEFT)
        ttk.Button(actions, text="Run Full Setup Check (A then B)", command=self._run_full_camera_setup_check).pack(
            side=tk.LEFT, padx=8
        )

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.camera_setup_output = tk.Text(self.camera_setup_tab, wrap=tk.WORD)
        self.camera_setup_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_roadmap_tab(self) -> None:
        ttk.Button(self.roadmap_tab, text="Refresh Roadmap", command=self._refresh_roadmap).pack(anchor=tk.W)
        self.roadmap_output = tk.Text(self.roadmap_tab, wrap=tk.WORD)
        self.roadmap_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_config_tab(self) -> None:
        container = ttk.Frame(self.config_tab)
        container.pack(fill=tk.BOTH, expand=True)

        top_buttons = ttk.Frame(container)
        top_buttons.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(top_buttons, text="Load From File", command=self._load_config_into_editor).pack(side=tk.LEFT)
        ttk.Button(top_buttons, text="Validate", command=self._validate_editor_config).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_buttons, text="Save Config", command=self._save_editor_config).pack(side=tk.LEFT, padx=6)

        form_canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=form_canvas.yview)
        self.config_form_frame = ttk.Frame(form_canvas)
        self.config_form_frame.bind(
            "<Configure>",
            lambda _event: form_canvas.configure(scrollregion=form_canvas.bbox("all")),
        )
        form_canvas.create_window((0, 0), window=self.config_form_frame, anchor="nw")
        form_canvas.configure(yscrollcommand=scrollbar.set)
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._render_config_fields()

        self.config_output = tk.Text(self.config_tab, wrap=tk.WORD, height=10)
        self.config_output.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self._load_config_into_editor()

    def _render_config_fields(self) -> None:
        specs = [
            ("Colony ID", "system.colony_id", str, None),
            ("Data root", "system.data_root", str, None),
            ("Pi model", "system.pi_model", str, ["auto", "pi4", "pi5"]),
            ("Camera model", "camera.model", str, ["auto", "hq", "hq_noir", "module3", "module3_wide", "module3_standard", "module3_noir"]),
            ("Width (px)", "camera.width", int, None),
            ("Height (px)", "camera.height", int, None),
            ("FPS target", "camera.fps_target", float, None),
            ("Shutter (us)", "camera.shutter_us", int, None),
            ("Preview window", "camera.preview_window", str, ["QTGL", "QT", "DRM"]),
            ("Tuning file", "camera.tuning_file", str, None),
            ("Pipeline mode", "pipeline.mode", str, ["record_only", "track_only", "record_and_track", "mixed_schedule"]),
            ("Tracking source", "pipeline.tracking_source", str, ["ram", "video"]),
            ("Deferred tracking", "pipeline.defer_tracking_until_after_recording", bool, None),
            ("Parallel tracking", "pipeline.parallel_tracking", bool, None),
            ("Behavior metrics", "pipeline.calculate_behavior_metrics", bool, None),
            ("Recording seconds", "capture.recording_seconds", int, None),
            ("Record interval (min)", "capture.record_interval_minutes", int, None),
            ("Track interval (min)", "capture.track_interval_minutes", int, None),
            ("Scheduler backend", "scheduling.backend", str, ["systemd", "cron"]),
            ("Scheduler scope", "scheduling.scope", str, ["system", "user"]),
            ("Scheduling enabled", "scheduling.enabled", bool, None),
            ("Unit prefix", "scheduling.unit_prefix", str, None),
            ("Service user", "scheduling.service_user", str, None),
            ("Fleet role", "fleet.role", str, ["standalone", "queen", "worker"]),
            ("Queen local pipeline", "fleet.queen_local_pipeline_enabled", bool, None),
            ("Queen media enabled", "fleet.queen_media_schedule.enabled", bool, None),
            ("Queen pull interval (min)", "fleet.queen_media_schedule.pull_interval_minutes", int, None),
            ("Queen track interval (min)", "fleet.queen_media_schedule.track_interval_minutes", int, None),
            ("Queen media max videos", "fleet.queen_media_schedule.max_videos_total", int, None),
            ("Queen media cooldown (min)", "fleet.queen_media_schedule.cooldown_minutes", int, None),
            ("Use mock camera", "runtime.use_mock_camera", bool, None),
            ("Save frame timestamps", "runtime.save_frame_timestamps", bool, None),
            ("FPS report each recording", "runtime.fps_report_on_each_recording", bool, None),
        ]

        for row, (label, key, value_type, choices) in enumerate(specs):
            ttk.Label(self.config_form_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)

            if value_type is bool:
                variable = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(self.config_form_frame, variable=variable)
                widget.grid(row=row, column=1, sticky="w", pady=3)
            elif choices:
                variable = tk.StringVar(value=str(choices[0]))
                widget = ttk.Combobox(
                    self.config_form_frame,
                    textvariable=variable,
                    values=choices,
                    state="readonly",
                    width=28,
                )
                widget.grid(row=row, column=1, sticky="ew", pady=3)
            else:
                variable = tk.StringVar(value="")
                widget = ttk.Entry(self.config_form_frame, textvariable=variable, width=34)
                widget.grid(row=row, column=1, sticky="ew", pady=3)

            self.config_fields[key] = (variable, value_type)

        self.config_form_frame.columnconfigure(1, weight=1)

    def _build_fps_tab(self) -> None:
        top = ttk.Frame(self.fps_tab)
        top.pack(fill=tk.X)

        self.video_path_var = tk.StringVar()
        self.timestamps_path_var = tk.StringVar()
        self.recording_seconds_var = tk.StringVar()
        self.fps_sweep_values_var = tk.StringVar(value="")
        self.fps_sweep_start_var = tk.StringVar(value="2.0")
        self.fps_sweep_stop_var = tk.StringVar(value="20.0")
        self.fps_sweep_step_var = tk.StringVar(value="2.0")
        self.fps_sweep_probe_seconds_var = tk.StringVar(value="20.0")
        self.fps_sweep_assume_ram_var = tk.StringVar(value="")
        self.fps_sweep_mock_var = tk.BooleanVar(value=False)
        self.fps_sweep_session_only_var = tk.BooleanVar(value=True)
        self.fps_sweep_status_var = tk.StringVar(value="Idle")

        report_frame = ttk.LabelFrame(top, text="Single Video FPS Report", padding=8)
        report_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(report_frame, text="Video path").grid(row=0, column=0, sticky="w")
        ttk.Entry(report_frame, textvariable=self.video_path_var, width=58).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )

        ttk.Label(report_frame, text="Timestamps path (optional)").grid(row=1, column=0, sticky="w")
        ttk.Entry(report_frame, textvariable=self.timestamps_path_var, width=58).grid(
            row=1, column=1, sticky="ew", padx=8, pady=4
        )

        ttk.Label(report_frame, text="Expected seconds (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(report_frame, textvariable=self.recording_seconds_var, width=20).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Button(report_frame, text="Run FPS Report", command=self._run_fps_report).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        report_frame.columnconfigure(1, weight=1)

        sweep_frame = ttk.LabelFrame(top, text="FPS Sweep Capacity Test", padding=8)
        sweep_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            sweep_frame,
            text=(
                "Runs increasing FPS probes, compares target vs real FPS, and estimates max recording duration "
                "at safe/warn/high-risk RAM budgets. If tracking ran this app session, it also estimates "
                "tracking time for each recording duration."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(sweep_frame, text="Start").grid(row=1, column=0, sticky="w")
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_start_var, width=8).grid(
            row=1, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(sweep_frame, text="Stop").grid(row=1, column=2, sticky="w")
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_stop_var, width=8).grid(
            row=1, column=3, sticky="w", padx=8, pady=3
        )
        ttk.Label(sweep_frame, text="Step").grid(row=2, column=0, sticky="w")
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_step_var, width=8).grid(
            row=2, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(sweep_frame, text="Probe seconds").grid(row=2, column=2, sticky="w")
        ttk.Entry(sweep_frame, textvariable=self.fps_sweep_probe_seconds_var, width=8).grid(
            row=2, column=3, sticky="w", padx=8, pady=3
        )

        advanced = ttk.LabelFrame(sweep_frame, text="Advanced Sweep Options", padding=6)
        advanced.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(advanced, text="FPS list (optional csv)").grid(row=0, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.fps_sweep_values_var, width=22).grid(
            row=0, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(advanced, text="Assume RAM GiB (optional)").grid(row=1, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.fps_sweep_assume_ram_var, width=8).grid(
            row=1, column=1, sticky="w", padx=8, pady=3
        )
        options = ttk.Frame(advanced)
        options.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(options, text="Use mock camera", variable=self.fps_sweep_mock_var).pack(side=tk.LEFT)
        ttk.Checkbutton(
            options,
            text="Use tracking from current app session only",
            variable=self.fps_sweep_session_only_var,
        ).pack(side=tk.LEFT, padx=10)
        advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(advanced)

        controls = ttk.Frame(sweep_frame)
        controls.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.fps_sweep_run_btn = ttk.Button(controls, text="Run FPS Sweep", command=self._start_fps_sweep)
        self.fps_sweep_run_btn.pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.fps_sweep_status_var).pack(side=tk.LEFT, padx=10)
        sweep_frame.columnconfigure(1, weight=1)
        sweep_frame.columnconfigure(3, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.fps_output = tk.Text(self.fps_tab, wrap=tk.WORD)
        self.fps_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_calibration_tab(self) -> None:
        top = ttk.Frame(self.calibration_tab)
        top.pack(fill=tk.X)

        self.manual_point_a = tk.StringVar(value="0,0")
        self.manual_point_b = tk.StringVar(value="500,0")
        self.manual_distance_cm = tk.StringVar(value="10.0")

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
        ttk.Label(manual, text="Point A (x,y)").grid(row=1, column=0, sticky="w")
        ttk.Entry(manual, textvariable=self.manual_point_a).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(manual, text="Point B (x,y)").grid(row=2, column=0, sticky="w")
        ttk.Entry(manual, textvariable=self.manual_point_b).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(manual, text="Real distance (cm)").grid(row=3, column=0, sticky="w")
        ttk.Entry(manual, textvariable=self.manual_distance_cm).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(manual, text="Calibrate from Points", command=self._calibrate_manual).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
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
        ttk.Label(aruco, text="Image path").grid(row=1, column=0, sticky="w")
        ttk.Entry(aruco, textvariable=self.aruco_image).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(aruco, text="Marker size (mm)").grid(row=2, column=0, sticky="w")
        ttk.Entry(aruco, textvariable=self.aruco_marker_size_mm).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(aruco, text="Dictionary").grid(row=3, column=0, sticky="w")
        ttk.Entry(aruco, textvariable=self.aruco_dictionary).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(aruco, text="Marker ID (optional)").grid(row=4, column=0, sticky="w")
        ttk.Entry(aruco, textvariable=self.aruco_marker_id).grid(row=4, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(aruco, text="Calibrate from ArUco", command=self._calibrate_aruco).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        aruco.columnconfigure(1, weight=1)
        self._register_advanced_widget(aruco)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.calibration_output = tk.Text(self.calibration_tab, wrap=tk.WORD)
        self.calibration_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

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

        self.schedule_check_output = tk.Text(self.schedule_check_tab, wrap=tk.WORD)
        self.schedule_check_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _run_schedule_check(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
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
            self.schedule_check_output.insert(tk.END, format_schedule_check_report(report))
            self.notebook.select(self.schedule_check_tab)
            if report.has_failures:
                messagebox.showwarning(
                    "Schedule check",
                    "Schedule check found failures. Review suggestions in the output panel.",
                )
        except Exception as exc:
            messagebox.showerror("Schedule check failed", str(exc))

    def _build_optimize_tracking_tab(self) -> None:
        top = ttk.Frame(self.optimize_tracking_tab)
        top.pack(fill=tk.X)

        self.opt_input_path_var = tk.StringVar(value="")
        self.opt_output_dir_var = tk.StringVar(value="")
        self.opt_profile_var = tk.StringVar(value="quick")
        self.opt_dictionary_var = tk.StringVar(value="4X4_50")
        self.opt_tag_size_mm_var = tk.StringVar(value="2.5")
        self.opt_sample_frames_var = tk.StringVar(value="80")
        self.opt_execution_target_var = tk.StringVar(value="pi_safe")
        self.opt_workers_var = tk.StringVar(value="")
        self.opt_expected_tags_var = tk.StringVar(value="")
        self.opt_early_stop_patience_var = tk.StringVar(value="40")
        self.opt_early_stop_min_improvement_var = tk.StringVar(value="0.002")
        self.opt_preview_var = tk.BooleanVar(value=False)
        self.opt_preview_frames_var = tk.StringVar(value="240")
        self.opt_apply_best_var = tk.BooleanVar(value=True)
        self.opt_top_k_var = tk.StringVar(value="5")
        self.opt_status_var = tk.StringVar(value="Idle")
        self._optimize_top_k = 5

        ttk.Label(top, text="Input path (video or image folder)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.opt_input_path_var, width=90).grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="Profile").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.opt_profile_var,
            values=["quick", "balanced", "deep"],
            state="readonly",
            width=20,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(top, text="Tag size (mm)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.opt_tag_size_mm_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(top, text="Sample frames").grid(row=3, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.opt_sample_frames_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(top, text="Execution target").grid(row=4, column=0, sticky="w")
        execution_frame = ttk.Frame(top)
        execution_frame.grid(row=4, column=1, sticky="w", padx=8, pady=4)
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

        options = ttk.Frame(top)
        options.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            options,
            text="Apply best params to current config",
            variable=self.opt_apply_best_var,
        ).pack(side=tk.LEFT)

        advanced = ttk.LabelFrame(top, text="Advanced Optimization Options", padding=6)
        advanced.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(advanced, text="Output root (optional)").grid(row=0, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_output_dir_var, width=70).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(advanced, text="Dictionary override").grid(row=1, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_dictionary_var, width=22).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(advanced, text="Workers (optional override)").grid(row=2, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_workers_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(advanced, text="Expected tags/frame (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_expected_tags_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(advanced, text="Early stop patience").grid(row=4, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_early_stop_patience_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(advanced, text="Early stop min improvement").grid(row=5, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_early_stop_min_improvement_var, width=10).grid(row=5, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(advanced, text="Top results to show").grid(row=6, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.opt_top_k_var, width=10).grid(row=6, column=1, sticky="w", padx=8, pady=4)
        preview_options = ttk.Frame(advanced)
        preview_options.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            preview_options,
            text="Write preview video",
            variable=self.opt_preview_var,
        ).pack(side=tk.LEFT, padx=12)
        ttk.Label(preview_options, text="Preview frames").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(preview_options, textvariable=self.opt_preview_frames_var, width=8).pack(side=tk.LEFT)
        advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(advanced)

        controls = ttk.Frame(top)
        controls.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.optimize_run_btn = ttk.Button(
            controls,
            text="Run optimize-tracking",
            command=self._start_optimize_tracking,
        )
        self.optimize_run_btn.pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.opt_status_var).pack(side=tk.LEFT, padx=10)

        top.columnconfigure(1, weight=1)

        self.optimize_output = tk.Text(self.optimize_tracking_tab, wrap=tk.WORD)
        self.optimize_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _start_optimize_tracking(self) -> None:
        if self._optimize_thread and self._optimize_thread.is_alive():
            messagebox.showinfo("Optimization running", "Tracking optimization is already running.")
            return

        input_path = self.opt_input_path_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Set an input video or image folder path.")
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
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        while not self._optimize_progress_q.empty():
            try:
                self._optimize_progress_q.get_nowait()
            except queue.Empty:
                break

        self._optimize_error = None
        self._optimize_warning = None
        self._optimize_result = None
        self._optimize_applied_config = None
        self._optimize_top_k = top_k

        optimize_kwargs = {
            "input_path": input_path,
            "profile": self.opt_profile_var.get().strip(),
            "sample_frames": sample_frames,
            "dictionary_name": self.opt_dictionary_var.get().strip() or "4X4_50",
            "tag_size_mm": tag_size_mm,
            "execution_target": self.opt_execution_target_var.get().strip(),
            "workers": workers,
            "expected_tags": expected_tags,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_improvement": early_stop_min_improvement,
            "output_dir": self.opt_output_dir_var.get().strip() or None,
            "write_preview": bool(self.opt_preview_var.get()),
            "preview_frames": preview_frames,
            "top_k": max(top_k, 10),
        }
        apply_best = bool(self.opt_apply_best_var.get())
        config_path = self.config_path_var.get().strip() or str(DEFAULT_USER_CONFIG_PATH)

        self.optimize_output.delete("1.0", tk.END)
        self.optimize_output.insert(tk.END, "Running optimize-tracking...\n")
        self.opt_status_var.set("Running...")
        self.optimize_run_btn.config(state=tk.DISABLED)

        self._optimize_thread = threading.Thread(
            target=self._run_optimize_tracking_worker,
            args=(optimize_kwargs, apply_best, config_path),
            daemon=True,
        )
        self._optimize_thread.start()
        self.after(200, self._poll_optimize_tracking)

    def _run_optimize_tracking_worker(self, optimize_kwargs: dict, apply_best: bool, config_path: str) -> None:
        from .tracking_optimizer import apply_best_params_to_config, optimize_tracking

        def progress_callback(done: int, total: int) -> None:
            self._optimize_progress_q.put((done, total))

        try:
            result = optimize_tracking(
                progress_callback=progress_callback,
                **optimize_kwargs,
            )
            self._optimize_result = result
        except Exception as exc:
            self._optimize_error = str(exc)
            return

        if apply_best:
            try:
                config_path_obj = Path(config_path)
                if config_path_obj.exists():
                    config = load_config(config_path_obj)
                else:
                    config = load_defaults()
                updated = apply_best_params_to_config(config, result.best_params)
                save_config(config_path_obj, updated)
                self._optimize_applied_config = str(config_path_obj)
            except Exception as exc:
                self._optimize_warning = f"Optimization finished, but config update failed: {exc}"

    def _poll_optimize_tracking(self) -> None:
        from .tracking_optimizer import format_optimization_report

        latest_progress = None
        while True:
            try:
                latest_progress = self._optimize_progress_q.get_nowait()
            except queue.Empty:
                break

        if latest_progress:
            done, total = latest_progress
            self.opt_status_var.set(f"Running... {done}/{total}")

        if self._optimize_thread and self._optimize_thread.is_alive():
            self.after(200, self._poll_optimize_tracking)
            return

        self.optimize_run_btn.config(state=tk.NORMAL)

        if self._optimize_error:
            self.opt_status_var.set("Failed")
            self.optimize_output.insert(tk.END, f"\nError: {self._optimize_error}\n")
            messagebox.showerror("optimize-tracking failed", self._optimize_error)
            return

        if self._optimize_result is None:
            self.opt_status_var.set("No result")
            self.optimize_output.insert(tk.END, "\nOptimization ended without a result.\n")
            return

        self.opt_status_var.set("Completed")
        self.optimize_output.delete("1.0", tk.END)
        self.optimize_output.insert(
            tk.END,
            format_optimization_report(self._optimize_result, top_k=self._optimize_top_k),
        )
        if self._optimize_applied_config:
            self.optimize_output.insert(
                tk.END,
                f"\n\nBest parameters were applied to config: {self._optimize_applied_config}",
            )
        if self._optimize_warning:
            self.opt_status_var.set("Completed with warning")
            self.optimize_output.insert(tk.END, f"\n\nWarning: {self._optimize_warning}")
        self.notebook.select(self.optimize_tracking_tab)

    def _build_nest_label_tab(self) -> None:
        top = ttk.Frame(self.nest_label_tab)
        top.pack(fill=tk.X)

        self.nest_folder_var = tk.StringVar()
        self.nest_script_var = tk.StringVar(value=str(default_script_path()))
        self.nest_labelmerc_var = tk.StringVar(value="")

        ttk.Label(top, text="Image folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_folder_var, width=90).grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="Label script").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_script_var, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(top, text="labelmerc (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.nest_labelmerc_var, width=90).grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        buttons = ttk.Frame(top)
        buttons.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Check Environment", command=self._run_nest_label_check).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Launch Nest Labeling", command=self._launch_nest_labeling).pack(side=tk.LEFT, padx=8)

        top.columnconfigure(1, weight=1)

        self.nest_label_output = tk.Text(self.nest_label_tab, wrap=tk.WORD)
        self.nest_label_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _run_nest_label_check(self) -> None:
        try:
            folder = self.nest_folder_var.get().strip() or None
            script = self.nest_script_var.get().strip() or None
            labelmerc = self.nest_labelmerc_var.get().strip() or None
            env = check_nest_labeling_environment(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
            )
            self.nest_label_output.delete("1.0", tk.END)
            self.nest_label_output.insert(tk.END, format_nest_labeling_environment(env))
        except Exception as exc:
            messagebox.showerror("Nest labeling check failed", str(exc))

    def _launch_nest_labeling(self) -> None:
        folder = self.nest_folder_var.get().strip()
        if not folder:
            messagebox.showerror("Missing folder", "Set the image folder first.")
            return

        try:
            script = self.nest_script_var.get().strip() or None
            labelmerc = self.nest_labelmerc_var.get().strip() or None
            process = launch_nest_labeling(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
            )
            self._nest_label_pid = process.pid
            command = build_nest_labeling_command(
                image_folder=folder,
                script_path=script,
                labelmerc_override=labelmerc,
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
            messagebox.showerror("Launch failed", str(exc))

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

        self.fleet_latest_detail = tk.Text(latest_frame, wrap=tk.WORD, height=7)
        self.fleet_latest_detail.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        status_frame.columnconfigure(1, weight=1)
        status_frame.rowconfigure(3, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self.fleet_output = tk.Text(self.fleet_tab, wrap=tk.WORD)
        self.fleet_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

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
            save_config(config_path, updated)

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
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            messagebox.showerror("Fleet init failed", str(exc))

    def _fleet_enroll_worker(self) -> None:
        host = self.fleet_worker_host_var.get().strip()
        if not host:
            messagebox.showerror("Missing host", "Set worker host/IP first.")
            return
        try:
            port = int(self.fleet_worker_port_var.get().strip())
            if port <= 0:
                raise ValueError("Port must be > 0")
        except Exception as exc:
            messagebox.showerror("Invalid port", str(exc))
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
            save_config(config_path, updated)
            self.fleet_pull_max_total_var.set(str(result.queen_media_max_videos_total))
            self.fleet_output.delete("1.0", tk.END)
            self.fleet_output.insert(tk.END, format_fleet_enroll_result(result))
            self.fleet_output.insert(tk.END, f"\n\nSaved config: {config_path}")
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            messagebox.showerror("Fleet enroll failed", str(exc))

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
                messagebox.showwarning("Fleet status", "Fleet status found one or more FAIL workers.")
        except Exception as exc:
            messagebox.showerror("Fleet status failed", str(exc))

    def _fleet_set_media_capacity_from_workers(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            target = sync_media_capacity_to_workers(config, include_disabled=False, minimum=1)
            save_config(config_path, config)
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
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            messagebox.showerror("Capacity update failed", str(exc))

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
                messagebox.showwarning(
                    "Worker offline warning",
                    f"{report.offline_count} configured worker(s) appear offline.",
                )
        except Exception as exc:
            messagebox.showerror("Latest status failed", str(exc))

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
                messagebox.showwarning(
                    "Worker offline warning",
                    f"{report.configured_workers_offline} configured worker(s) are not reachable.",
                )
        except Exception as exc:
            messagebox.showerror("Fleet discovery failed", str(exc))

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
            messagebox.showerror("Invalid fleet media settings", str(exc))
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
            save_config(config_path, config)

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
            self.notebook.select(self.fleet_tab)
        except Exception as exc:
            messagebox.showerror("Save schedule failed", str(exc))

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
                messagebox.showwarning(
                    "Queen pull latest",
                    "Queen pull latest completed with failures. Review output for details.",
                )
        except Exception as exc:
            messagebox.showerror("Queen pull latest failed", str(exc))

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
                messagebox.showwarning(
                    "Queen track latest",
                    "Queen track latest completed with failures. Review output for details.",
                )
        except Exception as exc:
            messagebox.showerror("Queen track latest failed", str(exc))

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
                messagebox.showwarning(
                    "Queen pull-track",
                    "Queen pull-track completed with failures. Review output for details.",
                )
        except Exception as exc:
            messagebox.showerror("Queen pull-track failed", str(exc))

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

        systemd_advanced = ttk.LabelFrame(top, text="Advanced Service Controls", padding=6)
        systemd_advanced.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(systemd_advanced, text="Systemd output dir").grid(row=0, column=0, sticky="w")
        ttk.Entry(systemd_advanced, textvariable=self.systemd_output_dir_var, width=70).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=8, pady=4
        )
        ttk.Button(systemd_advanced, text="Generate systemd Units", command=self._generate_systemd_units).grid(
            row=0, column=3, sticky="w", padx=8
        )

        actions = ttk.Frame(systemd_advanced)
        actions.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 2))
        ttk.Button(actions, text="Systemd Install", command=lambda: self._systemd_action("install")).pack(side=tk.LEFT)
        ttk.Button(actions, text="Systemd Enable", command=lambda: self._systemd_action("enable")).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Systemd Disable", command=lambda: self._systemd_action("disable")).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Systemd Status", command=lambda: self._systemd_action("status")).pack(side=tk.LEFT)
        ttk.Button(actions, text="Install GUI Desktop Icon", command=self._install_gui_shortcut).pack(side=tk.LEFT, padx=12)
        systemd_advanced.columnconfigure(1, weight=1)
        self._register_advanced_widget(systemd_advanced)

        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)

        self.run_output = tk.Text(self.run_tab, wrap=tk.WORD)
        self.run_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        alerts_frame = ttk.LabelFrame(self.run_tab, text="Runtime Alerts", padding=8)
        alerts_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
        ttk.Button(alerts_frame, text="Refresh Runtime Alerts", command=self._refresh_runtime_alerts).pack(anchor=tk.W)
        self.runtime_alert_output = tk.Text(alerts_frame, wrap=tk.WORD, height=5)
        self.runtime_alert_output.pack(fill=tk.X, expand=False, pady=(6, 0))

        history_frame = ttk.LabelFrame(self.run_tab, text="Recent Runs", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Button(history_frame, text="Refresh Run History", command=self._refresh_run_history).pack(anchor=tk.W)

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
        self.run_history_tree.pack(fill=tk.X, pady=(8, 8))
        self.run_history_tree.bind("<<TreeviewSelect>>", self._on_run_history_select)

        self.run_history_detail = tk.Text(history_frame, wrap=tk.WORD, height=8)
        self.run_history_detail.pack(fill=tk.BOTH, expand=True)
        self._register_advanced_widget(export_controls)
        self._refresh_runtime_alerts()
        self._refresh_run_history()

    def _load_config_or_defaults(self):
        config_path = Path(self.config_path_var.get())
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

    def _load_config_into_editor(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            for key, (variable, value_type) in self.config_fields.items():
                raw = self._get_nested(config, key)
                if value_type is bool:
                    variable.set(bool(raw))
                else:
                    variable.set("" if raw is None else str(raw))
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, "Loaded configuration into editor.")
        except Exception as exc:
            messagebox.showerror("Load config failed", str(exc))

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
                if key == "camera.tuning_file" and text == "":
                    parsed = None
                else:
                    parsed = text
            self._set_nested(config, key, parsed)
        return config

    def _validate_editor_config(self) -> None:
        try:
            config = self._build_editor_config()
            validate_config(config)
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, "Config is valid.")
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))

    def _save_editor_config(self) -> None:
        try:
            config = self._build_editor_config()
            validate_config(config)
            _, config_path = self._load_config_or_defaults()
            save_config(config_path, config)
            self.config_output.delete("1.0", tk.END)
            self.config_output.insert(tk.END, f"Saved config to {config_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _create_config(self) -> None:
        try:
            path = write_default_config(self.config_path_var.get(), force=False)
            messagebox.showinfo("Config created", f"Created default config at:\n{path}")
        except FileExistsError:
            messagebox.showinfo("Config exists", "Config already exists. Keeping current file.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _load_storage_mount_point_from_config(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            mount_point = str(config.get("system", {}).get("data_root", "")).strip()
            if mount_point:
                self.storage_mount_point_var.set(mount_point)
        except Exception:
            pass

    def _save_storage_mount_point_to_config(self) -> None:
        mount_point = self.storage_mount_point_var.get().strip()
        if not mount_point:
            messagebox.showerror("Invalid mount point", "Mount point cannot be empty.")
            return
        try:
            config, config_path = self._load_config_or_defaults()
            config.setdefault("system", {})
            config["system"]["data_root"] = mount_point
            save_config(config_path, config)
            self.storage_output.delete("1.0", tk.END)
            self.storage_output.insert(
                tk.END,
                f"Saved mount point to config:\n- system.data_root: {mount_point}\n- config: {config_path}",
            )
            self._refresh_storage_status()
        except Exception as exc:
            messagebox.showerror("Save mount point failed", str(exc))

    def _refresh_storage_status(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            mount_point = self.storage_mount_point_var.get().strip() or None
            report = get_storage_status(config=config, mount_point=mount_point)
            self.storage_output.delete("1.0", tk.END)
            self.storage_output.insert(tk.END, format_storage_status_report(report))
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
            "--apply-config",
        ]
        if device_path:
            command.extend(["--device", device_path])

        proc = subprocess.run(command, capture_output=True, text=True, check=False)
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

    def _setup_storage_auto_mount(self) -> None:
        mount_point = self.storage_mount_point_var.get().strip()
        if not mount_point:
            messagebox.showerror("Invalid mount point", "Mount point cannot be empty.")
            return

        try:
            config, config_path = self._load_config_or_defaults()
            status = get_storage_status(config=config, mount_point=mount_point)
            if status.mounted and status.writable:
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_status_report(status))
                return

            try:
                result = setup_storage_auto_mount(mount_point=mount_point)
                self.storage_output.delete("1.0", tk.END)
                self.storage_output.insert(tk.END, format_storage_setup_result(result))
                config.setdefault("system", {})
                config["system"]["data_root"] = mount_point
                save_config(config_path, config)
                self.storage_output.insert(tk.END, f"\n\nUpdated config: {config_path}")
            except PermissionError:
                if self._run_pkexec_storage_setup(config_path=config_path, mount_point=mount_point, device_path=None):
                    self._refresh_storage_status()
                    return

                sudo_cmd = build_storage_setup_sudo_command(
                    config_path=str(config_path),
                    mount_point=mount_point,
                    apply_config=True,
                )
                self.storage_output.insert(
                    tk.END,
                    (
                        "\n\nStorage setup needs root privileges.\n"
                        "Run this in terminal on the Pi:\n"
                        f"{sudo_cmd}"
                    ),
                )
        except Exception as exc:
            messagebox.showerror("Storage setup failed", str(exc))

    def _run_doctor(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            results = run_doctor(config)
            self.doctor_output.delete("1.0", tk.END)
            self.doctor_output.insert(tk.END, format_doctor_report(results))
            self._refresh_storage_status()
        except Exception as exc:
            messagebox.showerror("Doctor failed", str(exc))

    def _run_camera_preview_setup(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
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
            result = run_camera_preview(
                config=config,
                preview_seconds=seconds,
                window=self.camera_preview_window_var.get().strip(),
                width=width,
                height=height,
            )
            self.camera_setup_output.delete("1.0", tk.END)
            self.camera_setup_output.insert(tk.END, format_camera_preview_result(result))
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            messagebox.showerror("Camera preview failed", str(exc))

    def _run_camera_tracking_test_setup(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            seconds = float(self.camera_test_seconds_var.get().strip())
            if seconds <= 0:
                raise ValueError("Tracking test duration must be > 0")
            display_width = int(self.camera_test_display_width_var.get().strip())
            if display_width <= 0:
                raise ValueError("Display width must be > 0")

            dictionary = self.camera_test_dictionary_var.get().strip() or None
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
            messagebox.showerror("Live tracking test failed", str(exc))

    def _run_full_camera_setup_check(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
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

            dictionary = self.camera_test_dictionary_var.get().strip() or None
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
            preview_result = run_camera_preview(
                config=config,
                preview_seconds=preview_seconds,
                window=self.camera_preview_window_var.get().strip(),
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
            self.camera_setup_output.insert(tk.END, format_camera_preview_result(preview_result))
            self.camera_setup_output.insert(tk.END, "\n\n")
            self.camera_setup_output.insert(tk.END, format_tracking_test_result(tracking_result))
            self.notebook.select(self.camera_setup_tab)
        except Exception as exc:
            messagebox.showerror("Full camera setup check failed", str(exc))

    def _refresh_roadmap(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            self.roadmap_output.delete("1.0", tk.END)
            self.roadmap_output.insert(tk.END, render_roadmap(config, config_path))
            self.notebook.select(self.roadmap_tab)
        except Exception as exc:
            messagebox.showerror("Roadmap failed", str(exc))

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
            messagebox.showerror("FPS report failed", str(exc))

    def _start_fps_sweep(self) -> None:
        if self._fps_sweep_thread and self._fps_sweep_thread.is_alive():
            messagebox.showinfo("FPS sweep running", "An FPS sweep is already running.")
            return

        try:
            config, _ = self._load_config_or_defaults()
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
            messagebox.showerror("Invalid FPS sweep settings", str(exc))
            return

        while not self._fps_sweep_progress_q.empty():
            try:
                self._fps_sweep_progress_q.get_nowait()
            except queue.Empty:
                break

        self._fps_sweep_error = None
        self._fps_sweep_report = None
        session_start = self._session_started_iso if bool(self.fps_sweep_session_only_var.get()) else None

        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(tk.END, "Running FPS sweep...\n")
        self.fps_sweep_status_var.set("Running...")
        self.fps_sweep_run_btn.config(state=tk.DISABLED)

        self._fps_sweep_thread = threading.Thread(
            target=self._run_fps_sweep_worker,
            args=(
                config,
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
        fps_values: list[float],
        probe_seconds: float,
        assume_ram_gb: float | None,
        use_mock_camera: bool,
        session_start: str | None,
    ) -> None:
        def progress_callback(done: int, total: int, target_fps: float) -> None:
            self._fps_sweep_progress_q.put((done, total, target_fps))

        try:
            report = run_fps_sweep(
                config=config,
                fps_values=fps_values,
                probe_seconds=probe_seconds,
                assume_ram_gb=assume_ram_gb,
                use_mock_camera=use_mock_camera if use_mock_camera else None,
                session_start_iso=session_start,
                progress_callback=progress_callback,
            )
            self._fps_sweep_report = report
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
            messagebox.showerror("FPS sweep failed", self._fps_sweep_error)
            return

        if self._fps_sweep_report is None:
            self.fps_sweep_status_var.set("No result")
            self.fps_output.insert(tk.END, "\nFPS sweep ended without a report.\n")
            return

        self.fps_sweep_status_var.set("Completed")
        self.fps_output.delete("1.0", tk.END)
        self.fps_output.insert(tk.END, format_fps_sweep_report(self._fps_sweep_report))
        if self._fps_sweep_report.has_errors:
            messagebox.showwarning("FPS sweep", "One or more FPS probes failed. Review the output for details.")

    def _calibrate_manual(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            result = calibrate_from_points(
                parse_point(self.manual_point_a.get().strip()),
                parse_point(self.manual_point_b.get().strip()),
                float(self.manual_distance_cm.get().strip()),
            )
            updated = apply_scale_to_config(config, result)
            save_config(config_path, updated)

            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                format_calibration(
                    result,
                    pixel_contact_distance=updated.get("metrics", {}).get("pixel_contact_distance"),
                ),
            )
        except Exception as exc:
            messagebox.showerror("Manual calibration failed", str(exc))

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
            save_config(config_path, updated)

            self.calibration_output.delete("1.0", tk.END)
            self.calibration_output.insert(
                tk.END,
                format_calibration(
                    result,
                    pixel_contact_distance=updated.get("metrics", {}).get("pixel_contact_distance"),
                ),
            )
        except Exception as exc:
            messagebox.showerror("ArUco calibration failed", str(exc))

    def _run_once_now(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            config.setdefault("runtime", {})
            if self.run_mock_var.get():
                config["runtime"]["use_mock_camera"] = True

            mode_override = self.run_mode_var.get().strip() or None
            summary = run_once(config=config, mode_override=mode_override)
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, format_run_summary(summary))
            self._refresh_run_history()
            self.notebook.select(self.run_tab)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))

    def _generate_systemd_units(self) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            result = write_systemd_units(
                config=config,
                config_path=config_path,
                output_dir=self.systemd_output_dir_var.get().strip(),
            )
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, format_systemd_result(result))
            self.notebook.select(self.run_tab)
        except Exception as exc:
            messagebox.showerror("Systemd generation failed", str(exc))

    def _systemd_action(self, action: str) -> None:
        try:
            config, config_path = self._load_config_or_defaults()
            result = run_systemd_action(
                config=config,
                action=action,
                output_dir=self.systemd_output_dir_var.get().strip(),
                config_path=config_path,
            )
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, format_systemd_action_result(result))
            self.notebook.select(self.run_tab)
        except Exception as exc:
            messagebox.showerror(f"Systemd {action} failed", str(exc))

    def _install_gui_shortcut(self) -> None:
        try:
            result = install_gui_shortcut()
            self.run_output.delete("1.0", tk.END)
            self.run_output.insert(tk.END, format_gui_shortcut_result(result))
            self.notebook.select(self.run_tab)
            messagebox.showinfo(
                "GUI Desktop Icon",
                f"Desktop launcher created:\n{result.desktop_entry_path}",
            )
        except Exception as exc:
            messagebox.showerror("GUI shortcut install failed", str(exc))

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

    def _refresh_run_history(self) -> None:
        try:
            config, _ = self._load_config_or_defaults()
            data_root = config["system"]["data_root"]
            records = list_recent_run_records(data_root, limit=40)
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
            self._refresh_runtime_alerts()
        except Exception as exc:
            messagebox.showerror("Run history failed", str(exc))

    def _on_run_history_select(self, _event=None) -> None:
        selected = self.run_history_tree.selection()
        if not selected:
            return
        iid = selected[0]
        path = self._run_history_paths.get(iid)
        if not path:
            return
        try:
            payload = load_run_summary(path)
            self.run_history_detail.delete("1.0", tk.END)
            self.run_history_detail.insert(tk.END, json.dumps(payload, indent=2))
        except Exception as exc:
            messagebox.showerror("Load summary failed", str(exc))

    def _export_selected_run_bundle(self) -> None:
        selected = self.run_history_tree.selection()
        if not selected:
            messagebox.showerror("No run selected", "Select a run from Recent Runs first.")
            return

        iid = selected[0]
        summary_path = self._run_history_paths.get(iid)
        if not summary_path:
            messagebox.showerror("Missing run path", "Could not resolve selected run summary path.")
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
            self.run_history_detail.delete("1.0", tk.END)
            self.run_history_detail.insert(tk.END, text)
            messagebox.showinfo("Export complete", text)
        except Exception as exc:
            messagebox.showerror("Bundle export failed", str(exc))


def launch() -> None:
    app = BumbleBoxV2GUI()
    app.mainloop()
