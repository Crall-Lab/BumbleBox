from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .qt_env import build_qt_safe_env


DEFAULT_SCRIPT_NAME = "LabelNests_GUI.1.16.py"
DEFAULT_LABEL_ENV_RELATIVE = Path(".venvs") / "bbx-label"
LABEL_PYTHON_ENV_VAR = "BUMBLEBOX_NEST_PYTHON"


@dataclass(frozen=True)
class NestLabelingEnvironment:
    script_path: Path
    image_folder: Optional[Path]
    labelmerc_path: Optional[Path]
    python_executable: Path
    python_source: str
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


def default_label_python_path() -> Path:
    return repo_root() / DEFAULT_LABEL_ENV_RELATIVE / "bin" / "python"


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _python_can_import(python_path: Path, module_name: str) -> bool:
    if not _is_executable_file(python_path):
        return False
    probe = subprocess.run(
        [str(python_path), "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def _labelme_cli_for_python(python_path: Path) -> Optional[str]:
    if not _is_executable_file(python_path):
        return None
    bin_dir = python_path.parent
    candidate = bin_dir / "labelme"
    if _is_executable_file(candidate):
        return str(candidate)
    fallback = shutil.which("labelme")
    return fallback


def resolve_nest_label_python(python_executable: Optional[str] = None) -> tuple[Path, str]:
    if python_executable:
        candidate = Path(python_executable).expanduser().resolve()
        return candidate, "explicit argument"

    env_value = os.environ.get(LABEL_PYTHON_ENV_VAR, "").strip()
    candidates: list[tuple[Path, str]] = []
    if env_value:
        candidates.append((Path(env_value).expanduser().resolve(), f"env:{LABEL_PYTHON_ENV_VAR}"))
    candidates.extend(
        [
            (default_label_python_path().resolve(), "repo .venvs/bbx-label"),
            ((Path.home() / ".venvs" / "bbx-label" / "bin" / "python").resolve(), "~/.venvs/bbx-label"),
            ((repo_root() / ".venv" / "bin" / "python").resolve(), "repo .venv"),
            (Path(sys.executable).resolve(), "current interpreter"),
        ]
    )

    first_existing: Optional[tuple[Path, str]] = None
    seen: set[Path] = set()
    for path, source in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not _is_executable_file(path):
            continue
        if first_existing is None:
            first_existing = (path, source)
        has_pyqt5 = _python_can_import(path, "PyQt5")
        has_labelme_module = _python_can_import(path, "labelme")
        has_labelme_cli = bool(_labelme_cli_for_python(path))
        if has_pyqt5 and (has_labelme_module or has_labelme_cli):
            return path, source

    if first_existing is not None:
        return first_existing
    return Path(sys.executable).resolve(), "current interpreter"


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
    python_executable: Optional[str] = None,
) -> NestLabelingEnvironment:
    script = Path(script_path).expanduser().resolve() if script_path else default_script_path().resolve()
    folder_path = Path(image_folder).expanduser().resolve() if image_folder else None
    labelmerc_path = resolve_labelmerc_path(labelmerc_override=labelmerc_override, image_folder=image_folder)
    python_path, python_source = resolve_nest_label_python(python_executable)
    pyqt5_available = _python_can_import(python_path, "PyQt5")
    labelme_module_available = _python_can_import(python_path, "labelme")
    labelme_cli_path = _labelme_cli_for_python(python_path)

    return NestLabelingEnvironment(
        script_path=script,
        image_folder=folder_path,
        labelmerc_path=labelmerc_path,
        python_executable=python_path,
        python_source=python_source,
        script_exists=script.exists(),
        image_folder_exists=bool(folder_path and folder_path.exists() and folder_path.is_dir()),
        labelme_module_available=labelme_module_available,
        labelme_cli_path=labelme_cli_path,
        pyqt5_available=pyqt5_available,
    )


def format_nest_labeling_environment(env: NestLabelingEnvironment) -> str:
    lines = [
        "Nest Labeling Environment",
        "-------------------------",
        f"Script path: {env.script_path}",
        f"Script found: {'yes' if env.script_exists else 'no'}",
        f"Image folder: {env.image_folder if env.image_folder else '(not set)'}",
        f"Image folder exists: {'yes' if env.image_folder_exists else 'no'}",
        f"Selected label Python: {env.python_executable} ({env.python_source})",
        f"PyQt5 available in selected Python: {'yes' if env.pyqt5_available else 'no'}",
        f"labelme module available in selected Python: {'yes' if env.labelme_module_available else 'no'}",
        f"labelme CLI path: {env.labelme_cli_path or 'not found'}",
        f"Resolved labelmerc: {env.labelmerc_path or '(none, LabelMe defaults will be used)'}",
        f"Ready to launch: {'yes' if env.ready else 'no'}",
    ]
    if not env.ready:
        lines.append("")
        lines.append("Common fixes:")
        lines.append("1) Run setup script to create/update dedicated label env: bash scripts/setup_venv.sh")
        lines.append("2) On Pi, ensure apt Qt is present: sudo apt install python3-pyqt5")
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
    resolved_python, _ = resolve_nest_label_python(python_executable)

    command = [str(resolved_python), str(script), str(folder)]
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
        python_executable=python_executable,
    )
    if not env.script_exists:
        raise FileNotFoundError(f"Nest labeling script not found: {env.script_path}")
    if not env.image_folder_exists:
        raise FileNotFoundError(f"Image folder not found or not a directory: {env.image_folder}")
    if not env.pyqt5_available:
        raise RuntimeError(
            f"PyQt5 is not available in selected label Python: {env.python_executable} "
            "(run setup_venv.sh to prepare dedicated label env)."
        )
    if not env.labelme_module_available and not env.labelme_cli_path:
        raise RuntimeError(
            f"LabelMe not found in selected label Python or PATH: {env.python_executable} "
            "(run setup_venv.sh to prepare dedicated label env)."
        )

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
        env=build_qt_safe_env(),
    )


def build_labelme_command(
    image_path: str,
    python_executable: Optional[str] = None,
) -> list[str]:
    target = Path(image_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Image path not found: {target}")

    resolved_python, _ = resolve_nest_label_python(python_executable)
    if not _is_executable_file(resolved_python):
        raise RuntimeError(f"Selected Python is not executable: {resolved_python}")

    labelme_cli_path = _labelme_cli_for_python(resolved_python)
    if labelme_cli_path:
        return [labelme_cli_path, str(target)]

    if _python_can_import(resolved_python, "labelme"):
        return [str(resolved_python), "-m", "labelme", str(target)]

    raise RuntimeError(
        f"LabelMe not found in selected labeling environment: {resolved_python}. "
        "Run setup_venv.sh to prepare the dedicated label environment."
    )


def launch_labelme(
    image_path: str,
    python_executable: Optional[str] = None,
) -> subprocess.Popen:
    command = build_labelme_command(
        image_path=image_path,
        python_executable=python_executable,
    )
    process = subprocess.Popen(
        command,
        cwd=str(repo_root()),
        start_new_session=True,
        env=build_qt_safe_env(),
    )
    # Detect immediate startup failures (for example missing Qt plugin) and raise a clear error.
    time.sleep(0.8)
    return_code = process.poll()
    if return_code is not None:
        probe = subprocess.run(
            command,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=False,
            env=build_qt_safe_env(),
        )
        stdout = (probe.stdout or "").strip()
        stderr = (probe.stderr or "").strip()
        message_parts = [
            f"LabelMe exited immediately (code {return_code}).",
            f"Command: {' '.join(command)}",
        ]
        if stdout:
            message_parts.append(f"stdout:\n{stdout}")
        if stderr:
            message_parts.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(message_parts))
    return process
