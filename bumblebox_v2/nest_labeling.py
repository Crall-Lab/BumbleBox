from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_SCRIPT_NAME = "LabelNests_GUI.1.16.py"


@dataclass(frozen=True)
class NestLabelingEnvironment:
    script_path: Path
    image_folder: Optional[Path]
    labelmerc_path: Optional[Path]
    script_exists: bool
    image_folder_exists: bool
    labelme_module_available: bool
    labelme_cli_path: Optional[str]
    pyqt5_available: bool

    @property
    def ready(self) -> bool:
        labelme_ok = self.labelme_module_available or bool(self.labelme_cli_path)
        return self.script_exists and self.image_folder_exists and self.pyqt5_available and labelme_ok


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_script_path() -> Path:
    return repo_root() / DEFAULT_SCRIPT_NAME


def resolve_labelmerc_path(labelmerc_override: Optional[str] = None, image_folder: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if labelmerc_override:
        candidates.append(Path(labelmerc_override).expanduser())

    env_path = os.environ.get("BUMBLEBOX_LABELMERC")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    if image_folder:
        candidates.append(Path(image_folder).expanduser() / "labelmerc")

    candidates.append(repo_root() / "labelmerc")
    candidates.append(Path("~/.labelmerc").expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def check_nest_labeling_environment(
    image_folder: Optional[str],
    script_path: Optional[str] = None,
    labelmerc_override: Optional[str] = None,
) -> NestLabelingEnvironment:
    script = Path(script_path).expanduser().resolve() if script_path else default_script_path().resolve()
    folder_path = Path(image_folder).expanduser().resolve() if image_folder else None
    labelmerc_path = resolve_labelmerc_path(labelmerc_override=labelmerc_override, image_folder=image_folder)

    return NestLabelingEnvironment(
        script_path=script,
        image_folder=folder_path,
        labelmerc_path=labelmerc_path,
        script_exists=script.exists(),
        image_folder_exists=bool(folder_path and folder_path.exists() and folder_path.is_dir()),
        labelme_module_available=importlib.util.find_spec("labelme") is not None,
        labelme_cli_path=shutil.which("labelme"),
        pyqt5_available=importlib.util.find_spec("PyQt5") is not None,
    )


def format_nest_labeling_environment(env: NestLabelingEnvironment) -> str:
    lines = [
        "Nest Labeling Environment",
        "-------------------------",
        f"Script path: {env.script_path}",
        f"Script found: {'yes' if env.script_exists else 'no'}",
        f"Image folder: {env.image_folder if env.image_folder else '(not set)'}",
        f"Image folder exists: {'yes' if env.image_folder_exists else 'no'}",
        f"PyQt5 available in current Python: {'yes' if env.pyqt5_available else 'no'}",
        f"labelme module available in current Python: {'yes' if env.labelme_module_available else 'no'}",
        f"labelme CLI on PATH: {env.labelme_cli_path or 'not found'}",
        f"Resolved labelmerc: {env.labelmerc_path or '(none, LabelMe defaults will be used)'}",
        f"Ready to launch: {'yes' if env.ready else 'no'}",
    ]
    if not env.ready:
        lines.append("")
        lines.append("Common fixes:")
        lines.append("1) Install dependencies: sudo apt install python3-pyqt5 labelme")
        lines.append("2) Or in venv: python3 -m pip install pyqt5 labelme")
        lines.append(f"3) Confirm script exists at: {env.script_path}")
    return "\n".join(lines)


def build_nest_labeling_command(
    image_folder: str,
    script_path: Optional[str] = None,
    labelmerc_override: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> list[str]:
    folder = Path(image_folder).expanduser().resolve()
    script = Path(script_path).expanduser().resolve() if script_path else default_script_path().resolve()
    labelmerc_path = resolve_labelmerc_path(labelmerc_override=labelmerc_override, image_folder=str(folder))

    command = [python_executable or sys.executable, str(script), str(folder)]
    if labelmerc_path:
        command.extend(["--labelmerc", str(labelmerc_path)])
    return command


def launch_nest_labeling(
    image_folder: str,
    script_path: Optional[str] = None,
    labelmerc_override: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> subprocess.Popen:
    env = check_nest_labeling_environment(
        image_folder=image_folder,
        script_path=script_path,
        labelmerc_override=labelmerc_override,
    )
    if not env.script_exists:
        raise FileNotFoundError(f"Nest labeling script not found: {env.script_path}")
    if not env.image_folder_exists:
        raise FileNotFoundError(f"Image folder not found or not a directory: {env.image_folder}")
    if not env.pyqt5_available:
        raise RuntimeError("PyQt5 is not available in current Python environment.")
    if not env.labelme_module_available and not env.labelme_cli_path:
        raise RuntimeError("LabelMe not found in current Python environment or PATH.")

    command = build_nest_labeling_command(
        image_folder=image_folder,
        script_path=script_path,
        labelmerc_override=labelmerc_override,
        python_executable=python_executable,
    )
    return subprocess.Popen(
        command,
        cwd=str(repo_root()),
        start_new_session=True,
    )
