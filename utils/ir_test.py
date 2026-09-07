#!/usr/bin/env python3
"""
IR sensor test script for robot_control_0.
Reads IR_STATUS at ~5 Hz and logs: timestamp, raw hex, left bit, right bit.
ENABLE is never touched. Run as root (sudo).
Usage: sudo python3 ir_test.py [N_samples]
"""
import sys
import time
import signal

sys.path.insert(0, "/home/xilinx/overlays/nano_gtsrb_robot/robot_control")

from pynq import Overlay, MMIO
from robot_control import RobotControl

OVERLAY_BIT = "/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.bit"
RC_BASE = 0xA0010000
INTERVAL = 0.20   # 5 Hz

ol = Overlay(OVERLAY_BIT, download=False)
mmio = MMIO(RC_BASE, 0x1000)
rc = RobotControl(mmio)

# Safety: assert ENABLE=0 before any reads
assert not rc.is_enabled(), "ENABLE debe ser 0 antes de empezar"

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = continuo
label = sys.argv[2] if len(sys.argv) > 2 else "MANUAL"

running = True
def _stop(s, f):
    global running
    running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

print(f"[IR TEST] label={label}  ENABLE={rc.is_enabled()}  ctrl=0x{mmio.read(0x08):08X}")
print(f"{'timestamp':>14}  {'IR_HEX':>8}  {'left(b0)':>8}  {'right(b1)':>9}  label")
print("-" * 60)

count = 0
t_start = time.perf_counter()
while running:
    ir_raw = mmio.read(0x0C)
    left  = bool(ir_raw & 0x1)
    right = bool(ir_raw & 0x2)
    ts = time.perf_counter() - t_start
    print(f"{ts:>14.3f}  0x{ir_raw:06X}  {str(left):>8}  {str(right):>9}  {label}")
    sys.stdout.flush()
    count += 1
    if n_samples and count >= n_samples:
        break
    time.sleep(INTERVAL)

print(f"\n[IR TEST] {count} muestras. ENABLE={rc.is_enabled()} LEFT=0x{mmio.read(0x00)&0xFF:02X} RIGHT=0x{mmio.read(0x04)&0xFF:02X}")
