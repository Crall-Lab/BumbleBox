# GUI Desktop Launcher

You can create a clickable Desktop icon for BumbleBox GUI on Raspberry Pi (or other Debian desktops).

## One-command install

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py gui-install-shortcut
```

This creates:

- Desktop launcher: `~/Desktop/BumbleBox GUI.desktop`
- App-menu launcher: `~/.local/share/applications/BumbleBox GUI.desktop`
- Launcher script: `~/.local/bin/bumblebox-gui`
- Icon: `~/.local/share/icons/bumblebox-gui.svg`

The launcher script prefers:

1. `repo/.venv/bin/python` (if present)
2. `python3` on PATH

## Custom paths or dry run

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py gui-install-shortcut --dry-run
python3 /Users/aec/Desktop/BumbleBox/bbx.py gui-install-shortcut --desktop-dir /tmp --applications-dir /tmp --bin-dir /tmp
```

## GUI path

The launcher now opens the primary PyQt GUI. Its first launch opens a conditional setup wizard; hardware pages are skipped for devices not selected in the hardware profile.

During the staged migration, specialized tools that have not yet moved to Qt remain available through `Advanced -> Open Legacy Advanced Tools` or:

```bash
./bbx gui --legacy
```

Only one primary PyQt GUI instance is allowed per user. The legacy tools window is a separate migration bridge and should be closed when it is no longer needed.
