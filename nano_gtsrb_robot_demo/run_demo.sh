#!/usr/bin/env bash
set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: this demo must run as root."
    echo
    echo "Run:"
    echo "  sudo su"
    echo "  cd /home/xilinx/nano_gtsrb_demo"
    echo "  ./run_demo.sh"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load board / PYNQ / XRT environment.
if [ -r /etc/environment ]; then
    . /etc/environment
fi

for f in /etc/profile.d/*.sh; do
    [ -r "$f" ] && . "$f"
done

PYTHON_BIN="/usr/local/share/pynq-venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: PYNQ Python not found:"
    echo "  $PYTHON_BIN"
    exit 1
fi

# Verify FPGA visibility.
"$PYTHON_BIN" - <<'PY'
from pynq.pl_server.device import Device

if len(Device.devices) == 0:
    raise RuntimeError(
        "PYNQ loaded but FPGA device is not visible"
    )
PY

cd "$PROJECT_DIR"

echo "Using Python: $PYTHON_BIN"
echo "XILINX_XRT: ${XILINX_XRT:-<not set>}"

exec "$PYTHON_BIN" \
    -X faulthandler \
    "$PROJECT_DIR/demo.py" \
    "$@"
