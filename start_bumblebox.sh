#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${REPO_ROOT}/scripts/setup_venv.sh"
RUNTIME_VENV_DIR="${REPO_ROOT}/.venvs/bbx-runtime"
LEGACY_VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
  echo "[BumbleBox] Setup script not found: ${SETUP_SCRIPT}" >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    bash "$SETUP_SCRIPT" --help
    cat <<EOF

Quick use:
  bash ${REPO_ROOT}/start_bumblebox.sh

This wrapper configures runtime env:
  ${RUNTIME_VENV_DIR}

Then installs GUI launcher and prints next steps.
EOF
    exit 0
  fi
done

mkdir -p "${REPO_ROOT}/.venvs"

if [[ ! -e "$RUNTIME_VENV_DIR" && -d "$LEGACY_VENV_DIR" ]]; then
  echo "[BumbleBox] Reusing existing legacy env via descriptive path: ${RUNTIME_VENV_DIR}"
  ln -s "$LEGACY_VENV_DIR" "$RUNTIME_VENV_DIR"
fi

echo "[BumbleBox] Running setup..."
bash "$SETUP_SCRIPT" --venv-dir "$RUNTIME_VENV_DIR" "$@"

RUNTIME_PY="${RUNTIME_VENV_DIR}/bin/python"
if [[ ! -x "$RUNTIME_PY" && -x "${LEGACY_VENV_DIR}/bin/python" ]]; then
  RUNTIME_PY="${LEGACY_VENV_DIR}/bin/python"
fi
if [[ ! -x "$RUNTIME_PY" ]]; then
  echo "[BumbleBox] Could not find runtime Python after setup." >&2
  echo "[BumbleBox] Expected one of:" >&2
  echo "  ${RUNTIME_VENV_DIR}/bin/python" >&2
  echo "  ${LEGACY_VENV_DIR}/bin/python" >&2
  exit 1
fi

if [[ ! -e "$LEGACY_VENV_DIR" ]]; then
  ln -s ".venvs/bbx-runtime" "$LEGACY_VENV_DIR"
  echo "[BumbleBox] Added compatibility link: ${LEGACY_VENV_DIR} -> .venvs/bbx-runtime"
fi

echo "[BumbleBox] Installing GUI desktop icon/launcher..."
if ! "$RUNTIME_PY" "${REPO_ROOT}/bbx.py" gui-install-shortcut >/dev/null 2>&1; then
  echo "[BumbleBox] WARNING: Desktop shortcut install did not complete."
  echo "           You can retry with: ${REPO_ROOT}/bbx gui-install-shortcut"
fi

cat <<EOF

==================================================
Welcome to BumbleBox
==================================================
Setup complete.

Recommended runtime env:
  ${RUNTIME_VENV_DIR}

You do NOT need to run "source .../activate" each time.

Start BumbleBox GUI:
  1) Click the Desktop icon: BumbleBox GUI
  2) Or run: ${REPO_ROOT}/bbx-gui

Run from command line:
  ${REPO_ROOT}/bbx doctor
  ${REPO_ROOT}/bbx gui
  ${REPO_ROOT}/bbx roadmap

First-time bring-up checklist:
  ${REPO_ROOT}/bbx init
  ${REPO_ROOT}/bbx doctor
  ${REPO_ROOT}/bbx camera-preview --seconds 20
  ${REPO_ROOT}/bbx camera-test-tracking --seconds 20
==================================================
EOF
