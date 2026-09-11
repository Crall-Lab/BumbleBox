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
  --skip-nest-label         Skip labelme in main env (the BumbleBox PyQt GUI remains installed)
  --skip-picamera2          Skip pip install of picamera2 in main env
  --force-pip-nest-label    Force pip install for labelme in main env (Pi defaults to skip)
  --force-pip-picamera2     Force pip install for picamera2 in main env (Pi defaults to skip)
  --install-realsense       Install optional pyrealsense2 support (source fallback on ARM64 Pi)
  --skip-label-env          Do not create/update dedicated labeling env
  --label-venv-dir <path>   Dedicated labeling env path (default: <repo>/.venvs/bbx-label)
  --label-python <bin>      Python interpreter for dedicated labeling env (default: --python value)
  --skip-system-deps        Do not auto-install apt dependencies (Pi)
  --skip-enable-linger     Do not auto-enable lingering for the BumbleBox user
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
INSTALL_REALSENSE=0
FORCE_PIP_NEST_LABEL=0
FORCE_PIP_PICAMERA2=0
SETUP_LABEL_ENV=1
LABEL_VENV_DIR="${REPO_ROOT}/.venvs/bbx-label"
LABEL_PYTHON_BIN=""
AUTO_SYSTEM_DEPS=1
AUTO_ENABLE_LINGER=1
SKIP_PIP_UPGRADE=0
INSTALL_GUI_SHORTCUT=1
RUN_SMOKE_CHECK=1
PI_MODEL=""
SKIPPED_PIP_NEST_LABEL_ON_PI=0
SKIPPED_PIP_PICAMERA2_ON_PI=0
LINGER_RESULT=""
REALSENSE_VERSION="${BUMBLEBOX_REALSENSE_VERSION:-2.58.1}"

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

run_privileged() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  echo "[BumbleBox] ERROR: This RealSense installation step requires root access." >&2
  return 1
}

install_realsense_from_source() {
  local architecture
  architecture="$(uname -m)"
  if [[ -z "$PI_MODEL" || "$architecture" != "aarch64" ]]; then
    return 1
  fi

  echo "[BumbleBox] No ARM64 pyrealsense2 wheel is available; building Librealsense ${REALSENSE_VERSION}."
  maybe_install_apt_packages \
    "Librealsense source build" \
    cmake build-essential git libssl-dev libusb-1.0-0-dev pkg-config libudev-dev python3-dev || return 1

  local cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/bumblebox"
  local source_dir="${cache_root}/librealsense-v${REALSENSE_VERSION}"
  local build_dir="${source_dir}/build-bbx"
  mkdir -p "$cache_root"

  if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
    if [[ -e "$source_dir" ]]; then
      echo "[BumbleBox] ERROR: RealSense source cache is incomplete: ${source_dir}" >&2
      echo "           Move that path aside and rerun setup." >&2
      return 1
    fi
    git clone \
      --branch "v${REALSENSE_VERSION}" \
      --depth 1 \
      https://github.com/realsenseai/librealsense.git \
      "$source_dir" || return 1
  else
    echo "[BumbleBox] Reusing cached Librealsense source: ${source_dir}"
  fi

  cmake \
    -S "$source_dir" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$VENV_PY" \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_ROSBAG2=OFF || return 1

  local build_jobs=1
  if command -v nproc >/dev/null 2>&1; then
    build_jobs="$(nproc)"
    if [[ "$build_jobs" -gt 1 ]]; then
      build_jobs=$((build_jobs - 1))
    fi
  fi
  cmake --build "$build_dir" --parallel "$build_jobs" || return 1
  run_privileged cmake --install "$build_dir" || return 1

  local udev_rule="${source_dir}/config/99-realsense-libusb.rules"
  if [[ -f "$udev_rule" ]]; then
    run_privileged install -m 0644 "$udev_rule" /etc/udev/rules.d/99-realsense-libusb.rules || return 1
    run_privileged udevadm control --reload-rules || return 1
    run_privileged udevadm trigger || return 1
  fi
  run_privileged ldconfig || return 1

  # Source installs use /usr/local, which isolated venvs do not always include.
  local python_minor
  local venv_site
  python_minor="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  venv_site="$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')"
  printf "/usr/local/lib/python%s/dist-packages\n" "$python_minor" \
    > "${venv_site}/bumblebox_realsense.pth"

  "$VENV_PY" -c "import pyrealsense2" >/dev/null 2>&1
}

linger_target_user() {
  if [[ ${EUID:-$(id -u)} -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    printf "%s\n" "${SUDO_USER}"
    return 0
  fi
  id -un
}

ensure_user_linger() {
  local target_user
  target_user="$(linger_target_user)"
  if [[ -z "$target_user" ]]; then
    LINGER_RESULT="[BumbleBox] Note: Could not determine which user should get linger enabled."
    return 0
  fi

  if ! command -v loginctl >/dev/null 2>&1; then
    LINGER_RESULT="[BumbleBox] Note: loginctl is not available on this system. User timers will only run while ${target_user} is logged in."
    return 0
  fi

  local current_state=""
  current_state="$(loginctl show-user "$target_user" -p Linger --value 2>/dev/null || true)"
  if [[ "$current_state" == "yes" ]]; then
    LINGER_RESULT="[BumbleBox] User lingering already enabled for ${target_user}."
    return 0
  fi

  echo "[BumbleBox] Enabling user lingering for ${target_user} so user-scope scheduled recordings can continue after logout"
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    if loginctl enable-linger "$target_user"; then
      LINGER_RESULT="[BumbleBox] Enabled user lingering for ${target_user}."
      return 0
    fi
  elif command -v sudo >/dev/null 2>&1; then
    if sudo loginctl enable-linger "$target_user"; then
      LINGER_RESULT="[BumbleBox] Enabled user lingering for ${target_user}."
      return 0
    fi
  else
    LINGER_RESULT="[BumbleBox] WARNING: Could not enable user lingering for ${target_user} automatically because sudo is unavailable."
    return 0
  fi

  LINGER_RESULT="[BumbleBox] WARNING: Failed to enable user lingering for ${target_user}. User timers will still work while logged in."
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
    --install-realsense)
      INSTALL_REALSENSE=1
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
    --skip-enable-linger)
      AUTO_ENABLE_LINGER=0
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

  maybe_install_apt_packages \
    "Pi camera, GUI, and distributed-capture stack" \
    python3-picamera2 python3-pyqt5 libcamera-apps ffmpeg v4l-utils \
    chrony rsync openssh-client || true
  if [[ "$INSTALL_REALSENSE" -eq 1 ]]; then
    maybe_install_apt_packages "RealSense USB support" libusb-1.0-0 udev || true
  fi
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

if ! "$VENV_PY" -c "import PyQt5" >/dev/null 2>&1; then
  echo "[BumbleBox] Installing PyQt5 for the primary GUI"
  if ! "$VENV_PY" -m pip install pyqt5; then
    echo "[BumbleBox] WARNING: PyQt5 installation failed."
    echo "           On Raspberry Pi, install python3-pyqt5 and use --system-site-packages."
  fi
fi

if [[ "$INSTALL_REALSENSE" -eq 1 ]]; then
  echo "[BumbleBox] Installing optional RealSense Python support"
  if "$VENV_PY" -c "import pyrealsense2" >/dev/null 2>&1; then
    echo "[BumbleBox] Existing RealSense Python support is importable; skipping installation."
  elif ! "$VENV_PY" -m pip install pyrealsense2; then
    if ! install_realsense_from_source; then
      echo "[BumbleBox] ERROR: pyrealsense2 installation failed for this Python/architecture." >&2
      echo "           Follow the official Librealsense source-build instructions, then retry." >&2
      exit 1
    fi
  fi
  if ! "$VENV_PY" -c "import pyrealsense2" >/dev/null 2>&1; then
    echo "[BumbleBox] ERROR: RealSense installation completed, but pyrealsense2 is not importable." >&2
    exit 1
  fi
  echo "[BumbleBox] RealSense Python support is ready."
fi

if [[ "$INSTALL_NEST_LABEL" -eq 1 ]]; then
  echo "[BumbleBox] Installing nest-label packages in main env (labelme)"
  if ! "$VENV_PY" -m pip install labelme; then
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

required = ["yaml", "numpy", "pandas", "cv2", "PyQt5"]
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

for command, purpose in (("ssh", "second-Pi control"), ("rsync", "resumable artifact transfer")):
    if shutil.which(command):
        print(f"{command} available on PATH.")
    else:
        print(f"Warning: {command} not found on PATH. It is required for {purpose}.")

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

if [[ "$AUTO_ENABLE_LINGER" -eq 1 ]]; then
  ensure_user_linger
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

if [[ -n "$LINGER_RESULT" ]]; then
  cat <<EOF

${LINGER_RESULT}
EOF
fi

cat <<EOF
Activate main env manually when needed:
  source ${VENV_DIR}/bin/activate
EOF
