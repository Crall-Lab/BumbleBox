#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
BumbleBox one-command environment setup.

Usage:
  bash scripts/setup_venv.sh [options]

Options:
  --python <bin>            Python interpreter to use (default: python3)
  --venv-dir <path>         Virtual environment path (default: <repo>/.venv)
  --skip-nest-label         Skip installing PyQt5 + labelme
  --skip-picamera2          Skip installing picamera2
  --skip-pip-upgrade        Skip pip/setuptools/wheel upgrade
  --no-smoke-check          Skip import smoke checks
  -h, --help                Show this help text
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python3"
VENV_DIR="${REPO_ROOT}/.venv"
INSTALL_NEST_LABEL=1
INSTALL_PICAMERA2=1
SKIP_PIP_UPGRADE=0
RUN_SMOKE_CHECK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --venv-dir" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --skip-nest-label)
      INSTALL_NEST_LABEL=0
      shift
      ;;
    --skip-picamera2)
      INSTALL_PICAMERA2=0
      shift
      ;;
    --skip-pip-upgrade)
      SKIP_PIP_UPGRADE=1
      shift
      ;;
    --no-smoke-check)
      RUN_SMOKE_CHECK=0
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found on PATH: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[BumbleBox] Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[BumbleBox] Reusing existing virtual environment: $VENV_DIR"
fi

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Virtual environment python not found or not executable: $VENV_PY" >&2
  exit 1
fi

if [[ "$SKIP_PIP_UPGRADE" -eq 0 ]]; then
  echo "[BumbleBox] Upgrading pip toolchain in venv"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
fi

CORE_PACKAGES=(
  pyyaml
  numpy
  pandas
  opencv-contrib-python
)

echo "[BumbleBox] Installing core Python packages"
"$VENV_PY" -m pip install "${CORE_PACKAGES[@]}"

if [[ "$INSTALL_NEST_LABEL" -eq 1 ]]; then
  echo "[BumbleBox] Note: on Debian/Pi, apt packages can be more stable for Qt:"
  echo "           sudo apt install python3-pyqt5 labelme"
  echo "[BumbleBox] Installing nest-label packages (PyQt5, labelme)"
  "$VENV_PY" -m pip install pyqt5 labelme
fi

if [[ "$INSTALL_PICAMERA2" -eq 1 ]]; then
  echo "[BumbleBox] Note: on Raspberry Pi OS, apt is usually preferred for picamera2:"
  echo "           sudo apt install python3-picamera2"
  echo "[BumbleBox] Installing picamera2 package"
  "$VENV_PY" -m pip install picamera2
fi

if [[ "$RUN_SMOKE_CHECK" -eq 1 ]]; then
  echo "[BumbleBox] Running smoke checks"
  "$VENV_PY" - <<'PY'
import importlib
import sys

required = ["yaml", "numpy", "pandas", "cv2"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    print("Missing imports after install:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)

import cv2
has_aruco = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
if not has_aruco:
    print("Warning: cv2.aruco/ArucoDetector unavailable (check opencv-contrib-python install).")

print("Smoke checks passed.")
PY
fi

cat <<EOF

[BumbleBox] Setup complete.
Use BumbleBox with:
  ${VENV_PY} ${REPO_ROOT}/bbx.py doctor
  ${VENV_PY} ${REPO_ROOT}/bbx.py gui

Activate manually when needed:
  source ${VENV_DIR}/bin/activate
EOF
