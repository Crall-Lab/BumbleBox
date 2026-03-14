#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
BumbleBox one-command environment setup.

Usage:
  bash scripts/setup_venv.sh [options]

Options:
  --python <bin>            Python interpreter to use for main env (default: python3)
  --venv-dir <path>         Main BumbleBox venv path (default: <repo>/.venv)
  --system-site-packages    Create main env with system site packages visible
  --no-system-site-packages Keep main env isolated from system site packages
  --skip-nest-label         Skip pip install of PyQt5 + labelme in main env
  --skip-picamera2          Skip pip install of picamera2 in main env
  --force-pip-nest-label    Force pip install for PyQt5 + labelme in main env (Pi defaults to skip)
  --force-pip-picamera2     Force pip install for picamera2 in main env (Pi defaults to skip)
  --skip-label-env          Do not create/update dedicated labeling env
  --label-venv-dir <path>   Dedicated labeling env path (default: <repo>/.venvs/bbx-label)
  --label-python <bin>      Python interpreter for dedicated labeling env (default: --python value)
  --skip-system-deps        Do not auto-install apt dependencies (Pi)
  --skip-pip-upgrade        Skip pip/setuptools/wheel upgrade
  --skip-gui-shortcut       Do not auto-install Desktop GUI launcher/icon at end of setup
  --no-smoke-check          Skip import smoke checks
  -h, --help                Show this help text
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python3"
VENV_DIR="${REPO_ROOT}/.venv"
USE_SYSTEM_SITE_PACKAGES=0
USER_SET_SYSTEM_SITE_PACKAGES=0
INSTALL_NEST_LABEL=1
INSTALL_PICAMERA2=1
FORCE_PIP_NEST_LABEL=0
FORCE_PIP_PICAMERA2=0
SETUP_LABEL_ENV=1
LABEL_VENV_DIR="${REPO_ROOT}/.venvs/bbx-label"
LABEL_PYTHON_BIN=""
AUTO_SYSTEM_DEPS=1
SKIP_PIP_UPGRADE=0
INSTALL_GUI_SHORTCUT=1
RUN_SMOKE_CHECK=1
PI_MODEL=""
SKIPPED_PIP_NEST_LABEL_ON_PI=0
SKIPPED_PIP_PICAMERA2_ON_PI=0

detect_pi_model() {
  local model=""
  if [[ -r /proc/device-tree/model ]]; then
    model="$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || true)"
  fi
  if [[ -z "$model" && -r /sys/firmware/devicetree/base/model ]]; then
    model="$(tr -d '\000' < /sys/firmware/devicetree/base/model 2>/dev/null || true)"
  fi
  if [[ "$model" == *"Raspberry Pi"* ]]; then
    echo "$model"
    return 0
  fi
  return 1
}

venv_args() {
  local include_system="${1:-0}"
  local args=()
  if [[ "$include_system" -eq 1 ]]; then
    args+=(--system-site-packages)
  fi
  printf "%s\n" "${args[@]}"
}

create_venv() {
  local python_bin="$1"
  local target_dir="$2"
  local include_system="${3:-0}"
  local args=()
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && args+=("$arg")
  done < <(venv_args "$include_system")

  if "$python_bin" -m venv "${args[@]}" "$target_dir"; then
    return 0
  fi

  echo "[BumbleBox] Standard venv creation failed for $target_dir; retrying with --copies"
  "$python_bin" -m venv --copies "${args[@]}" "$target_dir"
}

probe_venv_python() {
  local py_bin="$1"
  local probe_out
  probe_out="$("$py_bin" -c 'print("BBX_PY_OK")' 2>&1 < /dev/null || true)"
  [[ "$probe_out" == "BBX_PY_OK" ]]
}

ensure_venv() {
  local python_bin="$1"
  local target_dir="$2"
  local include_system="${3:-0}"
  mkdir -p "$(dirname "$target_dir")"

  if [[ ! -d "$target_dir" ]]; then
    echo "[BumbleBox] Creating virtual environment: $target_dir"
    create_venv "$python_bin" "$target_dir" "$include_system"
  else
    echo "[BumbleBox] Reusing existing virtual environment: $target_dir"
  fi

  local py_bin="${target_dir}/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "[BumbleBox] Virtual environment python not found or not executable: $py_bin"
    echo "[BumbleBox] Recreating venv."
    rm -rf "$target_dir"
    create_venv "$python_bin" "$target_dir" "$include_system"
    py_bin="${target_dir}/bin/python"
  fi

  if [[ "$include_system" -eq 1 ]] && [[ -f "$target_dir/pyvenv.cfg" ]]; then
    if ! grep -Eq '^include-system-site-packages *= *true' "$target_dir/pyvenv.cfg"; then
      echo "[BumbleBox] Existing venv at $target_dir does not include system site-packages; recreating."
      rm -rf "$target_dir"
      create_venv "$python_bin" "$target_dir" "$include_system"
      py_bin="${target_dir}/bin/python"
    fi
  fi

  if ! probe_venv_python "$py_bin"; then
    echo "[BumbleBox] Existing venv python probe failed for $target_dir; recreating."
    rm -rf "$target_dir"
    create_venv "$python_bin" "$target_dir" "$include_system"
    py_bin="${target_dir}/bin/python"
  fi

  if ! probe_venv_python "$py_bin"; then
    echo "[BumbleBox] Failed to validate venv python executable: $py_bin" >&2
    echo "[BumbleBox] Try recreating manually with: $python_bin -m venv --copies $target_dir" >&2
    exit 1
  fi
}

is_package_installed() {
  local pkg="$1"
  if ! command -v dpkg-query >/dev/null 2>&1; then
    return 1
  fi
  dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"
}

maybe_install_apt_packages() {
  local reason="$1"
  shift
  local desired=("$@")
  local missing=()

  if [[ "$AUTO_SYSTEM_DEPS" -eq 0 ]]; then
    return 1
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi

  for pkg in "${desired[@]}"; do
    if ! is_package_installed "$pkg"; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  echo "[BumbleBox] Installing system packages for ${reason}: ${missing[*]}"
  if command -v sudo >/dev/null 2>&1; then
    if ! sudo apt-get update || ! sudo apt-get install -y "${missing[@]}"; then
      echo "[BumbleBox] WARNING: Failed to install system packages: ${missing[*]}"
      return 1
    fi
    return 0
  fi

  if ! apt-get update || ! apt-get install -y "${missing[@]}"; then
    echo "[BumbleBox] WARNING: Failed to install system packages: ${missing[*]}"
    return 1
  fi
  return 0
}

install_label_env_packages() {
  local label_py="$1"
  local label_ready=0

  if [[ "$SKIP_PIP_UPGRADE" -eq 0 ]]; then
    "$label_py" -m pip install --upgrade pip setuptools wheel
  fi

  if ! "$label_py" -c "import PyQt5" >/dev/null 2>&1; then
    echo "[BumbleBox] PyQt5 not importable in label env; attempting pip install pyqt5"
    if ! "$label_py" -m pip install pyqt5; then
      echo "[BumbleBox] WARNING: Could not install pyqt5 into label env."
      echo "           On Pi, install with apt: sudo apt install python3-pyqt5"
    fi
  fi

  echo "[BumbleBox] Installing LabelMe in dedicated label env"
  if "$label_py" -m pip install labelme; then
    label_ready=1
  else
    echo "[BumbleBox] Standard labelme install failed; trying Pi-safe fallback."
    if "$label_py" -m pip install --no-deps labelme; then
      "$label_py" -m pip install \
        imgviz \
        loguru \
        matplotlib \
        natsort \
        numpy \
        osam \
        pillow \
        pyyaml \
        qtpy \
        scikit-image \
        onnxruntime \
        gdown || true
      if "$label_py" -c "import labelme" >/dev/null 2>&1; then
        label_ready=1
      fi
    fi
  fi

  if [[ "$label_ready" -eq 0 ]]; then
    echo "[BumbleBox] WARNING: LabelMe install in dedicated label env may be incomplete."
  fi
}

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
    --system-site-packages)
      USE_SYSTEM_SITE_PACKAGES=1
      USER_SET_SYSTEM_SITE_PACKAGES=1
      shift
      ;;
    --no-system-site-packages)
      USE_SYSTEM_SITE_PACKAGES=0
      USER_SET_SYSTEM_SITE_PACKAGES=1
      shift
      ;;
    --skip-nest-label)
      INSTALL_NEST_LABEL=0
      shift
      ;;
    --skip-picamera2)
      INSTALL_PICAMERA2=0
      shift
      ;;
    --force-pip-nest-label)
      FORCE_PIP_NEST_LABEL=1
      shift
      ;;
    --force-pip-picamera2)
      FORCE_PIP_PICAMERA2=1
      shift
      ;;
    --skip-label-env)
      SETUP_LABEL_ENV=0
      shift
      ;;
    --label-venv-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --label-venv-dir" >&2; exit 1; }
      LABEL_VENV_DIR="$2"
      shift 2
      ;;
    --label-python)
      [[ $# -ge 2 ]] || { echo "Missing value for --label-python" >&2; exit 1; }
      LABEL_PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-system-deps)
      AUTO_SYSTEM_DEPS=0
      shift
      ;;
    --skip-pip-upgrade)
      SKIP_PIP_UPGRADE=1
      shift
      ;;
    --skip-gui-shortcut)
      INSTALL_GUI_SHORTCUT=0
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

if [[ -z "$LABEL_PYTHON_BIN" ]]; then
  LABEL_PYTHON_BIN="$PYTHON_BIN"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found on PATH: $PYTHON_BIN" >&2
  exit 1
fi
if ! command -v "$LABEL_PYTHON_BIN" >/dev/null 2>&1; then
  echo "Label Python not found on PATH: $LABEL_PYTHON_BIN" >&2
  exit 1
fi

if PI_MODEL="$(detect_pi_model)"; then
  echo "[BumbleBox] Detected Raspberry Pi hardware: ${PI_MODEL}"
  if [[ "$USER_SET_SYSTEM_SITE_PACKAGES" -eq 0 ]]; then
    USE_SYSTEM_SITE_PACKAGES=1
    echo "[BumbleBox] Pi mode: enabling --system-site-packages for main env (recommended for apt picamera2)."
  fi
  if [[ "$INSTALL_NEST_LABEL" -eq 1 && "$FORCE_PIP_NEST_LABEL" -eq 0 ]]; then
    INSTALL_NEST_LABEL=0
    SKIPPED_PIP_NEST_LABEL_ON_PI=1
  fi
  if [[ "$INSTALL_PICAMERA2" -eq 1 && "$FORCE_PIP_PICAMERA2" -eq 0 ]]; then
    INSTALL_PICAMERA2=0
    SKIPPED_PIP_PICAMERA2_ON_PI=1
  fi

  maybe_install_apt_packages "Pi camera + thermal stack" python3-picamera2 libcamera-apps ffmpeg v4l-utils || true
  if [[ "$SETUP_LABEL_ENV" -eq 1 ]]; then
    maybe_install_apt_packages "Pi Qt stack for nest labeling" python3-pyqt5 || true
  fi
fi

ensure_venv "$PYTHON_BIN" "$VENV_DIR" "$USE_SYSTEM_SITE_PACKAGES"
VENV_PY="${VENV_DIR}/bin/python"

if [[ "$SKIP_PIP_UPGRADE" -eq 0 ]]; then
  echo "[BumbleBox] Upgrading pip toolchain in main env"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
fi

CORE_PACKAGES=(
  pyyaml
  numpy
  pandas
  opencv-contrib-python
)

echo "[BumbleBox] Installing core Python packages in main env"
"$VENV_PY" -m pip install "${CORE_PACKAGES[@]}"

if [[ "$INSTALL_NEST_LABEL" -eq 1 ]]; then
  echo "[BumbleBox] Installing nest-label packages in main env (PyQt5, labelme)"
  if ! "$VENV_PY" -m pip install pyqt5 labelme; then
    echo "[BumbleBox] WARNING: main-env nest-label package install failed."
    echo "           Dedicated label env setup (below) is recommended."
  fi
fi

if [[ "$INSTALL_PICAMERA2" -eq 1 ]]; then
  echo "[BumbleBox] Installing picamera2 package in main env"
  if ! "$VENV_PY" -m pip install picamera2; then
    echo "[BumbleBox] WARNING: main-env picamera2 install failed."
    echo "           On Pi, apt install is preferred: sudo apt install python3-picamera2"
  fi
fi

if [[ "$SKIPPED_PIP_PICAMERA2_ON_PI" -eq 1 ]]; then
  echo "[BumbleBox] Pi mode: skipped pip picamera2 install in main env."
fi
if [[ "$SKIPPED_PIP_NEST_LABEL_ON_PI" -eq 1 ]]; then
  echo "[BumbleBox] Pi mode: skipped pip nest-label install in main env."
fi

LABEL_VENV_PY=""
if [[ "$SETUP_LABEL_ENV" -eq 1 ]]; then
  ensure_venv "$LABEL_PYTHON_BIN" "$LABEL_VENV_DIR" 1
  LABEL_VENV_PY="${LABEL_VENV_DIR}/bin/python"
  echo "[BumbleBox] Preparing dedicated label env at: ${LABEL_VENV_DIR}"
  install_label_env_packages "$LABEL_VENV_PY"
fi

if [[ "$RUN_SMOKE_CHECK" -eq 1 ]]; then
  echo "[BumbleBox] Running smoke checks for main env"
  "$VENV_PY" - <<'PY'
import importlib
import shutil
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

if shutil.which("ffmpeg"):
    print("ffmpeg available on PATH.")
else:
    print("Warning: ffmpeg not found on PATH. MP4 recording requires ffmpeg.")

if shutil.which("v4l2-ctl"):
    print("v4l2-ctl available on PATH.")
else:
    print("Warning: v4l2-ctl not found on PATH. Thermal camera diagnostics work better with v4l-utils installed.")

print("Main env smoke checks passed.")
PY

  if [[ -n "$LABEL_VENV_PY" ]]; then
    echo "[BumbleBox] Running smoke checks for dedicated label env"
    "$LABEL_VENV_PY" - <<'PY'
import importlib

have_pyqt = importlib.util.find_spec("PyQt5") is not None
have_labelme = importlib.util.find_spec("labelme") is not None
print(f"Label env PyQt5 available: {have_pyqt}")
print(f"Label env labelme available: {have_labelme}")
PY
  fi
fi

if [[ "$INSTALL_GUI_SHORTCUT" -eq 1 ]]; then
  echo "[BumbleBox] Installing GUI desktop icon/launcher"
  if "$VENV_PY" "${REPO_ROOT}/bbx.py" gui-install-shortcut >/dev/null 2>&1; then
    echo "[BumbleBox] Desktop GUI launcher/icon installed."
  else
    echo "[BumbleBox] Note: Desktop GUI launcher/icon could not be installed (headless session or desktop path unavailable)."
  fi
fi

cat <<EOF

[BumbleBox] Setup complete.
Main BumbleBox env:
  ${VENV_PY}
Use BumbleBox with:
  ${VENV_PY} ${REPO_ROOT}/bbx.py doctor
  ${VENV_PY} ${REPO_ROOT}/bbx.py gui

EOF

if [[ -n "$LABEL_VENV_PY" ]]; then
  cat <<EOF
Dedicated nest-label env:
  ${LABEL_VENV_PY}
Auto-detected by BumbleBox nest-label commands and GUI.
Manual override (optional):
  export BUMBLEBOX_NEST_PYTHON=${LABEL_VENV_PY}

EOF
fi

cat <<EOF
Activate main env manually when needed:
  source ${VENV_DIR}/bin/activate
EOF
