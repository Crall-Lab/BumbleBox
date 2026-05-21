# Nest Labeling on Debian/Raspberry Pi

This repo includes a PyQt-based nest labeling interface:

- `/Users/aec/Desktop/BumbleBox/LabelNests_GUI.1.16.py`

It is now integrated into BumbleBox V2:

- CLI: `bbx nest-label check|launch`
- GUI: `Nest Labeling` tab
- Auto interpreter selection: BumbleBox first tries dedicated label env at `/Users/aec/Desktop/BumbleBox/.venvs/bbx-label`

## Recommended install path (Raspberry Pi OS / Debian)

For best Qt compatibility on Pi, use Debian packages first:

```bash
sudo apt update
sudo apt install python3-pyqt5 labelme
```

If `labelme` is unavailable in your apt repositories, install `python3-pyqt5` only and run LabelMe on a desktop machine (recommended for stability/performance).
Or keep labeling on Pi with the dedicated label env created by `scripts/setup_venv.sh`.

Then check readiness:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py nest-label check --folder /path/to/composite_images
```

If the check says `Ready to launch: yes`, launch:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py nest-label launch --folder /path/to/composite_images
```

## Optional venv path

If you need a project-specific environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install pyqt5 labelme
```

This path is less predictable on Debian than distro packages for Qt plugins, so prefer apt unless you need isolation.
With Python 3.13 on Pi, pip installs for `pyqt5`/`labelme` may fail due to missing wheels and source-build tool requirements (`qmake`).

## LabelMe config file (`labelmerc`)

The nest-labeling launcher checks for LabelMe config in this order:

1. `--labelmerc /path/to/labelmerc` (CLI/script argument)
2. `BUMBLEBOX_LABELMERC` environment variable
3. `labelmerc` inside the selected image folder
4. `labelmerc` in BumbleBox repo root
5. `~/.labelmerc`

If none exists, LabelMe still launches using defaults.

## Interpreter selection

By default, BumbleBox nest-label commands and GUI auto-select a labeling interpreter in this order:

1. `BUMBLEBOX_NEST_PYTHON` (if set)
2. `/Users/aec/Desktop/BumbleBox/.venvs/bbx-label/bin/python`
3. `~/.venvs/bbx-label/bin/python`
4. `/Users/aec/Desktop/BumbleBox/.venv/bin/python`
5. current Python interpreter

## Common issues

1. `PyQt5 is not available`
- Fix: `sudo apt install python3-pyqt5`

2. `LabelMe not found`
- Fix: `sudo apt install labelme` or `python3 -m pip install labelme`

3. Qt platform plugin error (`xcb`/display errors)
- Usually from mixed apt/pip Qt stacks. Reinstall using one method consistently (apt recommended on Pi OS).
