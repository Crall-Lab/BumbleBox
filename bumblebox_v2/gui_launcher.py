from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GuiShortcutInstallResult:
    name: str
    repo_root: Path
    desktop_entry_path: Path
    applications_entry_path: Path
    launcher_script_path: Path
    icon_path: Path
    trusted_mark_set: bool
    dry_run: bool


def _expand(path: Optional[str | Path], default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser()


def _resolve_desktop_dir(home: Path) -> Path:
    env_dir = os.environ.get("XDG_DESKTOP_DIR")
    if env_dir:
        return Path(os.path.expandvars(env_dir)).expanduser()

    config_path = home / ".config" / "user-dirs.dirs"
    if config_path.exists():
        text = config_path.read_text()
        match = re.search(r'^XDG_DESKTOP_DIR="([^"]+)"', text, flags=re.MULTILINE)
        if match:
            raw = match.group(1).replace("$HOME", str(home))
            return Path(raw).expanduser()

    return home / "Desktop"


def _sanitize_desktop_filename(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in {" ", "-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    base = "".join(out).strip() or "BumbleBox GUI"
    if not base.endswith(".desktop"):
        base = f"{base}.desktop"
    return base


def _launcher_script_text(repo_root: Path) -> str:
    repo_q = shlex.quote(str(repo_root))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"REPO_ROOT={repo_q}",
            'if [ -x "$REPO_ROOT/.venvs/bbx-runtime/bin/python" ]; then',
            '  exec "$REPO_ROOT/.venvs/bbx-runtime/bin/python" "$REPO_ROOT/bbx.py" gui',
            "fi",
            'if [ -x "$REPO_ROOT/.venv/bin/python" ]; then',
            '  exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/bbx.py" gui',
            "fi",
            "if command -v python3 >/dev/null 2>&1; then",
            '  exec python3 "$REPO_ROOT/bbx.py" gui',
            "fi",
            'echo "python3 not found. Install python3 or run $REPO_ROOT/start_bumblebox.sh." >&2',
            "exit 1",
            "",
        ]
    )


def _desktop_entry_text(name: str, comment: str, exec_path: Path, icon_path: Path, working_dir: Path) -> str:
    return "\n".join(
        [
            "[Desktop Entry]",
            "Version=1.0",
            "Type=Application",
            f"Name={name}",
            f"Comment={comment}",
            f"Exec={exec_path}",
            f"Icon={icon_path}",
            "Terminal=false",
            "StartupNotify=true",
            "Categories=Science;Utility;",
            f"Path={working_dir}",
            "",
        ]
    )


def _default_icon_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#e9f5db"/>
<stop offset="100%" stop-color="#a6d96a"/>
</linearGradient>
</defs>
<rect x="8" y="8" width="240" height="240" rx="44" fill="url(#bg)" stroke="#3a5a40" stroke-width="10"/>
<circle cx="128" cy="136" r="54" fill="#f1c232" stroke="#2f2f2f" stroke-width="10"/>
<rect x="86" y="124" width="84" height="24" fill="#2f2f2f"/>
<circle cx="104" cy="136" r="6" fill="#ffffff"/>
<circle cx="152" cy="136" r="6" fill="#ffffff"/>
<line x1="128" y1="82" x2="128" y2="44" stroke="#2f2f2f" stroke-width="8" stroke-linecap="round"/>
<circle cx="128" cy="36" r="9" fill="#2f2f2f"/>
</svg>
"""


def _mark_trusted(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["gio", "set", str(path), "metadata::trusted", "true"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def install_gui_shortcut(
    *,
    name: str = "BumbleBox GUI",
    comment: str = "Launch BumbleBox V2 GUI",
    repo_root: Optional[str | Path] = None,
    desktop_dir: Optional[str | Path] = None,
    applications_dir: Optional[str | Path] = None,
    bin_dir: Optional[str | Path] = None,
    icon_path: Optional[str | Path] = None,
    dry_run: bool = False,
) -> GuiShortcutInstallResult:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[1]
    home = Path.home()
    desktop = _expand(desktop_dir, _resolve_desktop_dir(home)).resolve()
    apps = _expand(applications_dir, home / ".local" / "share" / "applications").resolve()
    local_bin = _expand(bin_dir, home / ".local" / "bin").resolve()
    icon = _expand(icon_path, home / ".local" / "share" / "icons" / "bumblebox-gui.svg").resolve()

    filename = _sanitize_desktop_filename(name)
    desktop_entry = desktop / filename
    apps_entry = apps / filename
    script_name = "bumblebox-gui"
    script_path = local_bin / script_name

    script_text = _launcher_script_text(repo)
    entry_text = _desktop_entry_text(name, comment, script_path, icon, repo)
    trusted = False

    if not dry_run:
        desktop.mkdir(parents=True, exist_ok=True)
        apps.mkdir(parents=True, exist_ok=True)
        local_bin.mkdir(parents=True, exist_ok=True)
        icon.parent.mkdir(parents=True, exist_ok=True)

        if icon_path is None:
            if not icon.exists():
                icon.write_text(_default_icon_svg())
        elif not icon.exists():
            raise FileNotFoundError(
                f"Custom icon path does not exist: {icon}. "
                "Either provide an existing icon path or omit --icon-path to use the default icon."
            )

        script_path.write_text(script_text)
        script_path.chmod(0o755)

        desktop_entry.write_text(entry_text)
        desktop_entry.chmod(0o755)

        apps_entry.write_text(entry_text)
        apps_entry.chmod(0o644)

        trusted = _mark_trusted(desktop_entry)

    return GuiShortcutInstallResult(
        name=name,
        repo_root=repo,
        desktop_entry_path=desktop_entry,
        applications_entry_path=apps_entry,
        launcher_script_path=script_path,
        icon_path=icon,
        trusted_mark_set=trusted,
        dry_run=dry_run,
    )


def format_gui_shortcut_result(result: GuiShortcutInstallResult) -> str:
    lines = [
        "GUI Desktop Shortcut",
        "--------------------",
        f"Name: {result.name}",
        f"Repo root: {result.repo_root}",
        f"Desktop launcher: {result.desktop_entry_path}",
        f"Applications launcher: {result.applications_entry_path}",
        f"Launcher script: {result.launcher_script_path}",
        f"Icon: {result.icon_path}",
        f"GIO trusted metadata set: {result.trusted_mark_set}",
    ]
    if result.dry_run:
        lines.append("Dry run: no files were written.")
    return "\n".join(lines)
