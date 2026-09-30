from dataclasses import dataclass


@dataclass
class DryRunFault:
    watchdog_expired: bool = False
    estop_active: bool = False


class DryRunRobotControl:
    """
    Software-only implementation of the RobotControl interface.

    No PYNQ.
    No MMIO.
    No FPGA accesses.
    """

    def __init__(self):
        self._enabled = False
        self._ir = 0b00

        self._left = 0
        self._right = 0

        self._fault = DryRunFault()

    def set_left(self, forward, duty):
        self._left = duty

    def set_right(self, forward, duty):
        self._right = duty

    def coast_left(self):
        self._left = 0

    def coast_right(self):
        self._right = 0

    def coast_both(self):
        self._left = 0
        self._right = 0

    def enable(self, on):
        self._enabled = bool(on)

    def is_enabled(self):
        return self._enabled

    def set_stop_policy(self, brake):
        pass

    def clear_estop(self):
        self._fault.estop_active = False

    def kick_watchdog(self):
        pass

    def configure_watchdog(
        self,
        enable,
        brake_on_expiry,
    ):
        pass

    def read_ir(self):
        return self._ir

    def set_ir(self, value):
        self._ir = value & 0b11

    def read_fault(self):
        return self._fault

    def clear_fault_bits(self, **kwargs):
        self._fault.watchdog_expired = False

    def read_version(self):
        return (1, 0, 0)
