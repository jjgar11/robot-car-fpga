#!/usr/bin/env python3
"""
Line following — Ultra96-V2 / robot_control_0
Logica: izq detecta negro -> gira izquierda, der detecta negro -> gira derecha
"""
import sys, time, signal

sys.path.insert(0, "/home/xilinx/overlays/nano_gtsrb_robot/robot_control")
from pynq import Overlay, MMIO
from robot_control import RobotControl

# ── Parametros ───────────────────────────────────────────────
BASE_DUTY     = 20    # duty recto
TURN_OUTER    = 30    # rueda exterior en curva (minimo fisico)
TURN_INNER    = 0     # rueda interior en correcciones (0 = coast)
CORNER_INNER  = 20    # rueda interior en esquinas (frena sin rotar, giro mas suave)
LOOP_HZ       = 20    # 50 ms/iter — sale antes de esquinas si la linea es fina
KICK_EVERY    = 8     # 8 x 50 ms = 400 ms < 1 s timeout
# ────────────────────────────────────────────────────────────

OVERLAY_BIT = "/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.bit"
RC_BASE     = 0xA0010000
INTERVAL    = 1.0 / LOOP_HZ

ol   = Overlay(OVERLAY_BIT, download=False)
mmio = MMIO(RC_BASE, 0x1000)
rc   = RobotControl(mmio)

rc.coast_both()
rc.enable(False)
assert not rc.is_enabled()
assert not rc.read_fault().estop_active

running = True
def _sig(s, f):
    global running
    running = False
signal.signal(signal.SIGINT,  _sig)
signal.signal(signal.SIGTERM, _sig)

def stop_safe(msg=""):
    rc.coast_both()
    rc.enable(False)
    f = rc.read_fault()
    print("\n[STOP] %s" % msg)
    print("  ENABLE=%s  L=0x%02X  R=0x%02X  wdg=%s" % (
          rc.is_enabled(), mmio.read(0x00)&0xFF, mmio.read(0x04)&0xFF,
          f.watchdog_expired))

rc.kick_watchdog()
rc.clear_fault_bits(watchdog=True)
rc.enable(True)
assert rc.is_enabled()

print("[LINE FOLLOW] BASE=%d%%  TURN=%d%%  HZ=%d" % (BASE_DUTY, TURN_OUTER, LOOP_HZ))
print("%6s  %4s  %4s  %4s  accion" % ("t(s)", "IR", "Lcmd", "Rcmd"))
print("-" * 38)

t_start    = time.perf_counter()
iter_count = 0

while running:
    t0 = time.perf_counter()

    if iter_count % KICK_EVERY == 0:
        rc.kick_watchdog()

    ir      = mmio.read(0x0C)
    left_s  = bool(ir & 0x1)
    right_s = bool(ir & 0x2)

    if not left_s and not right_s:        # 0x00 — recto
        action = "RECTO  "
        rc.set_left( True, BASE_DUTY)
        rc.set_right(True, BASE_DUTY)

    elif left_s and not right_s:          # 0x01 — izq negro -> GIRA IZQUIERDA
        action = "GIRA_IZQ"
        if TURN_INNER > 0:
            rc.set_left(True, TURN_INNER)
        else:
            rc.coast_left()
        rc.set_right(True, TURN_OUTER)

    elif not left_s and right_s:          # 0x02 — der negro -> GIRA DERECHA
        action = "GIRA_DER"
        rc.set_left(True, TURN_OUTER)
        if TURN_INNER > 0:
            rc.set_right(True, TURN_INNER)
        else:
            rc.coast_right()

    else:                                 # 0x03 — esquina CCW -> gira izquierda (suave)
        action = "CORNER_L"
        if CORNER_INNER > 0:
            rc.set_left(True, CORNER_INNER)
        else:
            rc.coast_left()
        rc.set_right(True, TURN_OUTER)

    ts  = time.perf_counter() - t_start
    lc  = mmio.read(0x00) & 0xFF
    rc_ = mmio.read(0x04) & 0xFF
    print("%6.2f  0x%02X  0x%02X  0x%02X  %s" % (ts, ir, lc, rc_, action))
    sys.stdout.flush()

    fault = rc.read_fault()
    if fault.watchdog_expired:
        stop_safe("watchdog expirado")
        sys.exit(1)

    iter_count += 1
    elapsed = time.perf_counter() - t0
    time.sleep(max(0.0, INTERVAL - elapsed))

stop_safe("Ctrl+C")
