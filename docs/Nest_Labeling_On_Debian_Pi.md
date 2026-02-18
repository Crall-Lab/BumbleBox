# Nest Labeling on Debian/Raspberry Pi

This repo includes a PyQt-based nest labeling interface:

- `/Users/aec/Desktop/BumbleBox/LabelNests_GUI.1.16.py`

It is now integrated into BumbleBox V2:

- CLI: `bbx nest-label check|launch`
- GUI: `Nest Labeling` tab

## Recommended install path (Raspberry Pi OS / Debian)

For best Qt compatibility on Pi, use Debian packages first:

```bash
sudo apt update
sudo apt install python3-pyqt5 labelme
```

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

## LabelMe config file (`labelmerc`)

The nest-labeling launcher checks for LabelMe config in this order:

1. `--labelmerc /path/to/labelmerc` (CLI/script argument)
2. `BUMBLEBOX_LABELMERC` environment variable
3. `labelmerc` inside the selected image folder
4. `labelmerc` in BumbleBox repo root
5. `~/.labelmerc`

If none exists, LabelMe still launches using defaults.

## Common issues

1. `PyQt5 is not available`
- Fix: `sudo apt install python3-pyqt5`

2. `LabelMe not found`
- Fix: `sudo apt install labelme` or `python3 -m pip install labelme`

3. Qt platform plugin error (`xcb`/display errors)
- Usually from mixed apt/pip Qt stacks. Reinstall using one method consistently (apt recommended on Pi OS).
