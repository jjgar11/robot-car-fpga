"""Movement process: line-following + stop-sign state machine.

Designed to run as an independent child process spawned by robot_main.py.
Hardware access goes through RobotControl (MMIO wrapper) only — never /dev/mem.
"""

import signal
import sys
import time
from collections import deque
from queue import Empty


# ---------------------------------------------------------------------------
# Register offsets inside robot_control_0 (0xA0010000)
# ---------------------------------------------------------------------------
REG_CONTROL   = 0x08   # bit0 = ENABLE, bit1 = COAST_LEFT, bit2 = COAST_RIGHT
REG_IR        = 0x0C   # bit0 = left IR (1=black), bit1 = right IR (1=black)
REG_LEFT_CMD  = 0x00   # bit7=DIR (1=forward), bits[6:0]=duty %
REG_RIGHT_CMD = 0x04   # same layout
REG_FAULT     = 0x1C   # bit0 = watchdog_expired, bit1 = estop
REG_VERSION   = 0x20   # expected 0x00010000
REG_WDG_KICK  = 0x10   # write anything to kick watchdog


# ---------------------------------------------------------------------------
# Dry-run MMIO stub (no real hardware access)
# ---------------------------------------------------------------------------
class DryRunMMIO:
    """Simulates MMIO registers; prints writes to stdout."""

    def __init__(self):
        self.regs = {
            REG_CONTROL:   0x00,
            REG_IR:        0x00,
            REG_WDG_KICK:  0x00,
            REG_LEFT_CMD:  0x00,
            REG_RIGHT_CMD: 0x00,
            REG_FAULT:     0x00,
            REG_VERSION:   0x00010000,
        }

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int) -> None:
        self.regs[offset] = value
        print(f"[DRY-RUN] MMIO write 0x{offset:03X} = 0x{value:08X}", flush=True)


# ---------------------------------------------------------------------------
# MovementController
# ---------------------------------------------------------------------------
from motion_config import (
    DEFAULT_PROFILE, PROFILES,
    STOP_HOLD_S, STOP_COOLDOWN_S,
    MOVEMENT_HZ, WDG_KICK_EVERY,
    IR_FILTER_SAMPLES, MAX_EVENT_AGE_S,
)
from sign_config import classify_sign


class MovementController:
    """Encapsulates control logic; injectable rc and mmio enable unit testing."""

    def __init__(self, rc, mmio, dry_run: bool = False):
        self._rc       = rc
        self._mmio     = mmio
        self._dry_run  = dry_run

        # State machine
        self._state             = 'LINE_FOLLOW'
        self._profile           = DEFAULT_PROFILE
        self._stop_deadline     = 0.0
        self._stop_cooldown_until = 0.0
        self._iter_count        = 0

        # IR majority-vote buffer
        self._ir_history: deque = deque(maxlen=IR_FILTER_SAMPLES)

    # ------------------------------------------------------------------
    def step(self, event=None, mono_now: float = None) -> dict:
        """Execute one control iteration (call at MOVEMENT_HZ).

        Parameters
        ----------
        event:    DetectionEvent or None
        mono_now: override time.monotonic() for deterministic unit tests
        """
        if mono_now is None:
            mono_now = time.monotonic()

        # 1. Kick watchdog periodically
        if self._iter_count % WDG_KICK_EVERY == 0:
            self._mmio.write(REG_WDG_KICK, 1)

        # 2. Read IR with majority vote
        raw_ir = self._mmio.read(REG_IR) & 0x03
        self._ir_history.append(raw_ir)
        ir = self._majority_ir()

        # 3. Process incoming sign event
        if event is not None and event.ready:
            age = mono_now - event.timestamp
            if age <= MAX_EVENT_AGE_S:
                self._handle_event(event, mono_now)

        # 4. State machine
        action = 'NONE'
        left_s = right_s = 0

        if self._state == 'STOP_HOLD':
            self._coast_both()
            action = 'COAST'
            if mono_now >= self._stop_deadline:
                # Resume line-following; start cooldown
                self._stop_cooldown_until = mono_now + STOP_COOLDOWN_S
                self._state = 'LINE_FOLLOW'
                action = 'RESUME'
        else:  # LINE_FOLLOW
            left_s, right_s, action = self._apply_line_follow(ir)

        self._iter_count += 1
        return {
            'state':        self._state,
            'ir':           ir,
            'left_s':       left_s,
            'right_s':      right_s,
            'action':       action,
            'profile_name': self._profile.name,
            'iter':         self._iter_count,
        }

    # ------------------------------------------------------------------
    def _majority_ir(self) -> int:
        """Return the majority-voted IR value (left bit and right bit independently)."""
        if not self._ir_history:
            return 0
        n = len(self._ir_history)
        left_votes  = sum(1 for v in self._ir_history if v & 0x01)
        right_votes = sum(1 for v in self._ir_history if v & 0x02)
        left_black  = 1 if left_votes  > n // 2 else 0
        right_black = 1 if right_votes > n // 2 else 0
        return (right_black << 1) | left_black

    # ------------------------------------------------------------------
    def _handle_event(self, event, mono_now: float) -> None:
        category = classify_sign(event.class_id)

        if category == 'stop':
            # Ignore during cooldown or if already stopping
            if mono_now < self._stop_cooldown_until:
                return
            if self._state == 'STOP_HOLD':
                return  # don't reset deadline mid-stop
            self._state         = 'STOP_HOLD'
            self._stop_deadline = mono_now + STOP_HOLD_S
        elif category in ('speed_slow', 'speed_fast'):
            # Map category to profile key
            profile_key = category[len('speed_'):]   # 'slow', 'normal', 'fast'
            if self._state != 'STOP_HOLD':
                self._profile = PROFILES[profile_key]
            # A non-stop sign clears the cooldown so the next stop can act sooner
            if category != 'stop' and mono_now < self._stop_cooldown_until:
                self._stop_cooldown_until = 0.0
        elif category == 'other':
            # Non-stop sign: clear cooldown early
            if mono_now < self._stop_cooldown_until:
                self._stop_cooldown_until = 0.0

    # ------------------------------------------------------------------
    def _apply_line_follow(self, ir: int):
        """Apply line-follow corrections based on filtered IR value.

        Returns (left_speed, right_speed, action_label).
        """
        p = self._profile
        if ir == 0x00:   # both white → go straight
            self._set_left(p.base_duty)
            self._set_right(p.base_duty)
            return p.base_duty, p.base_duty, 'STRAIGHT'
        elif ir == 0x01: # left black → turn left
            self._coast_left()
            self._set_right(p.turn_outer)
            return 0, p.turn_outer, 'TURN_LEFT'
        elif ir == 0x02: # right black → turn right
            self._set_left(p.turn_outer)
            self._coast_right()
            return p.turn_outer, 0, 'TURN_RIGHT'
        else:            # 0x03 both black → corner CCW
            self._set_left(p.corner_inner)
            self._set_right(p.turn_outer)
            return p.corner_inner, p.turn_outer, 'CORNER'

    # ------------------------------------------------------------------
    # Low-level motor helpers
    # ------------------------------------------------------------------
    def _set_left(self, duty: int) -> None:
        self._mmio.write(REG_LEFT_CMD, (duty & 0x7F) | 0x80)

    def _set_right(self, duty: int) -> None:
        self._mmio.write(REG_RIGHT_CMD, (duty & 0x7F) | 0x80)

    def _coast_left(self) -> None:
        self._mmio.write(REG_LEFT_CMD, 0x00)

    def _coast_right(self) -> None:
        self._mmio.write(REG_RIGHT_CMD, 0x00)

    def _coast_both(self) -> None:
        self._coast_left()
        self._coast_right()

    # ------------------------------------------------------------------
    def enable(self, on: bool) -> None:
        self._rc.enable(on)

    def safe_stop(self) -> None:
        self._coast_both()
        self._rc.enable(False)

    def check_fault(self):
        return self._rc.read_fault()


# ---------------------------------------------------------------------------
# movement_process — entry point for the child process
# ---------------------------------------------------------------------------
def movement_process(
    inf_to_mov_q,
    shutdown_event,
    ready_event,
    startup_q,
    enable_event,
    dry_run: bool = False,
):
    """Child-process entry point.  Called after fork() by robot_main."""
    # 1. Ignore SIGINT in child (supervisor handles it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # 2. Insert module search paths
    sys.path.insert(0, '/home/xilinx/overlays/nano_gtsrb_robot/robot_control')
    sys.path.insert(0, '/home/xilinx/nano_gtsrb_robot_demo')

    rc   = None
    ctrl = None

    try:
        # 3. Import hardware libraries (after fork so parent never loads them)
        if dry_run:
            mmio = DryRunMMIO()
            # Minimal stub so we don't need pynq at all in dry-run
            class _DryRunRC:
                def __init__(self, mmio):
                    self._mmio = mmio
                def enable(self, on):
                    self._mmio.write(REG_CONTROL, 0x01 if on else 0x00)
                def is_enabled(self):
                    return bool(self._mmio.read(REG_CONTROL) & 0x01)
                def kick_watchdog(self):
                    self._mmio.write(REG_WDG_KICK, 1)
                def clear_fault_bits(self, **_kw):
                    self._mmio.write(REG_FAULT, 0)
                def read_fault(self):
                    raw = self._mmio.read(REG_FAULT)
                    class _F:
                        def __init__(self, r):
                            self.watchdog_expired = bool(r & 0x01)
                            self.estop_active     = bool(r & 0x02)
                    return _F(raw)
                def read_version(self):
                    return self._mmio.read(REG_VERSION)
            rc = _DryRunRC(mmio)
        else:
            from pynq import MMIO
            from robot_control import RobotControl
            from motion_config import OVERLAY_BIT, RC_BASE, RC_SIZE
            mmio = MMIO(RC_BASE, RC_SIZE)
            rc   = RobotControl(mmio)

        # 6. Verify hardware version
        version = mmio.read(REG_VERSION)
        if version != 0x00010000:
            startup_q.put({'ok': False, 'error': f'Bad VERSION: 0x{version:08X}'})
            return

        # 7. Assert ENABLE=0 at startup
        assert not rc.is_enabled(), "Motor enable should be 0 at startup"

        # 8. Report to supervisor
        startup_q.put({'version': version, 'enable': rc.is_enabled(), 'ok': True})

        # 9. Signal that hardware is initialised
        ready_event.set()

        # 10. Wait for supervisor to tell us it's OK to run motors
        enable_event.wait()
        if shutdown_event.is_set():
            return

        # 11. Build controller
        ctrl = MovementController(rc, mmio, dry_run)

        # 12. Prime watchdog and enable
        rc.kick_watchdog()
        rc.clear_fault_bits(watchdog=True)
        rc.enable(True)

        # 13. Main control loop at MOVEMENT_HZ
        period = 1.0 / MOVEMENT_HZ
        next_tick = time.monotonic()

        while not shutdown_event.is_set():
            try:
                event = inf_to_mov_q.get_nowait()
            except Empty:
                event = None

            ctrl.step(event=event)

            fault = ctrl.check_fault()
            if fault.watchdog_expired or fault.estop_active:
                print(f"[movement] FAULT detected: wdg={fault.watchdog_expired} "
                      f"estop={fault.estop_active}", flush=True)
                break

            # Maintain loop rate
            next_tick += period
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # recover if we fell behind

    except Exception as exc:
        print(f"[movement] Exception: {exc}", flush=True)
        import traceback; traceback.print_exc()
        try:
            startup_q.put({'ok': False, 'error': str(exc)})
        except Exception:
            pass
    finally:
        if ctrl is not None:
            ctrl.safe_stop()
        elif rc is not None:
            rc.enable(False)
        print("[movement] process exiting", flush=True)
