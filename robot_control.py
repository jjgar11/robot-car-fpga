"""
Python wrapper for the robot_control_axi peripheral (see
models/nano/robot_control/README.md for the full register map and hardware
design). Accepts an injected MMIO-like object so it can be unit-tested
without any FPGA/overlay/DMA access -- see
models/nano/robot_control/software/test_robot_control.py.

This module deliberately does NOT:
  - import pynq / Overlay anything,
  - open /dev/mem,
  - touch any DMA object.
It only calls `.read(offset)` / `.write(offset, value)` on whatever object
is handed to its constructor. A real deployment passes a
`pynq.MMIO`-derived object (or `overlay.robot_control_0` if a PYNQ
DefaultIP driver is later written for this peripheral); tests here pass a
small in-memory fake.
"""
from dataclasses import dataclass


# ---- register offsets (must match rtl/robot_control_axi.v exactly) ----
REG_LEFT_CMD  = 0x00
REG_RIGHT_CMD = 0x04
REG_CONTROL   = 0x08
REG_IR_STATUS = 0x0C
REG_WDG_KICK  = 0x10
REG_WDG_CFG   = 0x14
REG_ESTOP     = 0x18
REG_FAULT     = 0x1C
REG_VERSION   = 0x20

CONTROL_ENABLE_BIT     = 0
CONTROL_STOP_POLICY_BIT = 1
CONTROL_ESTOP_CLEAR_BIT = 2

WDG_ENABLE_BIT = 0
WDG_POLICY_BIT = 1

FAULT_WDG_BIT          = 0
FAULT_ESTOP_ACTIVE_BIT = 1
FAULT_LEFT_INVALID_BIT  = 2
FAULT_RIGHT_INVALID_BIT = 3

CMD_COAST = 0x00
CMD_BRAKE = 0xFF


class InvalidDutyError(ValueError):
    pass


def encode_motor_cmd(direction_forward: bool, duty: int) -> int:
    """Encode (direction, duty 0-100) into the byte format the hardware
    expects: bit7=direction, bits6:0=duty. Raises InvalidDutyError for
    duty outside [0,100] -- the caller decides whether to catch this or
    let it propagate; the hardware itself would saturate+fault on an
    out-of-range value, but the software wrapper rejects it up front so a
    bug is caught at the call site instead of silently saturating.
    """
    if not (0 <= duty <= 100):
        raise InvalidDutyError(f"duty must be 0-100, got {duty}")
    value = duty & 0x7F
    if direction_forward:
        value |= 0x80
    return value


@dataclass
class FaultStatus:
    watchdog_expired: bool
    estop_active: bool
    left_cmd_invalid: bool
    right_cmd_invalid: bool

    @property
    def any(self) -> bool:
        return self.watchdog_expired or self.estop_active or \
            self.left_cmd_invalid or self.right_cmd_invalid


class RobotControl:
    def __init__(self, mmio):
        """`mmio` must provide .read(offset:int)->int and
        .write(offset:int, value:int)->None, e.g. pynq.MMIO or a test
        double. No overlay is loaded and no DMA is touched here."""
        self._mmio = mmio

    # ---- motor commands ----
    def set_left(self, direction_forward: bool, duty: int):
        self._mmio.write(REG_LEFT_CMD, encode_motor_cmd(direction_forward, duty))

    def set_right(self, direction_forward: bool, duty: int):
        self._mmio.write(REG_RIGHT_CMD, encode_motor_cmd(direction_forward, duty))

    def coast_left(self):
        self._mmio.write(REG_LEFT_CMD, CMD_COAST)

    def coast_right(self):
        self._mmio.write(REG_RIGHT_CMD, CMD_COAST)

    def coast_both(self):
        self.coast_left()
        self.coast_right()

    def brake_left(self):
        self._mmio.write(REG_LEFT_CMD, CMD_BRAKE)

    def brake_right(self):
        self._mmio.write(REG_RIGHT_CMD, CMD_BRAKE)

    def brake_both(self):
        self.brake_left()
        self.brake_right()

    def get_left_cmd(self) -> int:
        return self._mmio.read(REG_LEFT_CMD) & 0xFF

    def get_right_cmd(self) -> int:
        return self._mmio.read(REG_RIGHT_CMD) & 0xFF

    # ---- control ----
    def _read_control(self) -> int:
        return self._mmio.read(REG_CONTROL)

    def enable(self, on: bool):
        val = self._read_control()
        if on:
            val |= (1 << CONTROL_ENABLE_BIT)
        else:
            val &= ~(1 << CONTROL_ENABLE_BIT)
        # never let a stray ESTOP_CLEAR_REQ bit leak through from a
        # read-modify-write on a register that also has a self-clearing bit
        val &= ~(1 << CONTROL_ESTOP_CLEAR_BIT)
        self._mmio.write(REG_CONTROL, val)

    def is_enabled(self) -> bool:
        return bool(self._read_control() & (1 << CONTROL_ENABLE_BIT))

    def set_stop_policy(self, brake_on_stop: bool):
        """Policy applied when ENABLE=0 or ESTOP is active (NOT the
        watchdog-expiry policy, which is separate -- see
        configure_watchdog)."""
        val = self._read_control()
        if brake_on_stop:
            val |= (1 << CONTROL_STOP_POLICY_BIT)
        else:
            val &= ~(1 << CONTROL_STOP_POLICY_BIT)
        val &= ~(1 << CONTROL_ESTOP_CLEAR_BIT)
        self._mmio.write(REG_CONTROL, val)

    # ---- ESTOP ----
    def estop(self):
        """Sticky, write-1-to-set. Overrides LEFT_CMD/RIGHT_CMD immediately
        and stays active until clear_estop() is called explicitly."""
        self._mmio.write(REG_ESTOP, 1)

    def is_estop_active(self) -> bool:
        return bool(self._mmio.read(REG_ESTOP) & 0x1)

    def clear_estop(self):
        """Explicit clear sequence: CONTROL.bit2 (ESTOP_CLEAR_REQ), a
        DIFFERENT register/bit than the one that raised it (writing 0 to
        REG_ESTOP itself has no effect by design). Preserves the current
        ENABLE/stop-policy bits (read-modify-write)."""
        val = self._read_control()
        val |= (1 << CONTROL_ESTOP_CLEAR_BIT)
        self._mmio.write(REG_CONTROL, val)

    # ---- watchdog ----
    def kick_watchdog(self):
        self._mmio.write(REG_WDG_KICK, 1)

    def configure_watchdog(self, enable: bool, brake_on_expiry: bool):
        val = 0
        if enable:
            val |= (1 << WDG_ENABLE_BIT)
        if brake_on_expiry:
            val |= (1 << WDG_POLICY_BIT)
        self._mmio.write(REG_WDG_CFG, val)

    # ---- sensors / status ----
    def read_ir(self):
        return self._mmio.read(REG_IR_STATUS) & 0x3

    def read_fault(self) -> FaultStatus:
        val = self._mmio.read(REG_FAULT)
        return FaultStatus(
            watchdog_expired=bool(val & (1 << FAULT_WDG_BIT)),
            estop_active=bool(val & (1 << FAULT_ESTOP_ACTIVE_BIT)),
            left_cmd_invalid=bool(val & (1 << FAULT_LEFT_INVALID_BIT)),
            right_cmd_invalid=bool(val & (1 << FAULT_RIGHT_INVALID_BIT)),
        )

    def clear_fault_bits(self, watchdog=False, left_invalid=False, right_invalid=False):
        """Write-1-to-clear the requested sticky FAULT_STATUS bits. Note
        (see README): a bit re-latches on the next cycle if its underlying
        condition (e.g. an out-of-range LEFT_CMD still sitting in the
        register) is still active -- fix the condition first."""
        val = 0
        if watchdog:
            val |= (1 << FAULT_WDG_BIT)
        if left_invalid:
            val |= (1 << FAULT_LEFT_INVALID_BIT)
        if right_invalid:
            val |= (1 << FAULT_RIGHT_INVALID_BIT)
        self._mmio.write(REG_FAULT, val)

    def read_version(self):
        raw = self._mmio.read(REG_VERSION)
        major = (raw >> 16) & 0xFFFF
        minor = (raw >> 8) & 0xFF
        patch = raw & 0xFF
        return (major, minor, patch)
