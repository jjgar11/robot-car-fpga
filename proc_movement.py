"""Movement process: line-following + stop-sign state machine.

Designed to run as an independent child process spawned by robot_main.py.
Hardware access goes through RobotControl (MMIO wrapper) only — never /dev/mem.
"""

import signal
import sys
import time
from collections import deque
from queue import Empty

from motion_config import (
    DEFAULT_PROFILE, PROFILES,
    STOP_HOLD_S, STOP_COOLDOWN_S,
    MOVEMENT_HZ, WDG_KICK_EVERY,
    IR_FILTER_SAMPLES, MAX_EVENT_AGE_S,
)
from sign_config import classify_sign


class MovementController:
    """Encapsulates control logic; injectable rc and mmio enable unit testing."""

    def __init__(self, rc):
        self._rc       = rc

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
            self._rc.kick_watchdog()

        # 2. Read IR with majority vote
        raw_ir = self._rc.read_ir()
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
            self._rc.coast_both()
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
        if ir == 0b00:   # both white → go straight
            self._rc.set_left(p.base_duty)
            self._rc.set_right(p.base_duty)
            return p.base_duty, p.base_duty, 'STRAIGHT'
        elif ir == 0b01: # left black → turn left
            self._rc.coast_left()
            self._rc.set_right(p.turn_outer)
            return 0, p.turn_outer, 'TURN_LEFT'
        elif ir == 0b10: # right black → turn right
            self._rc.set_left(p.turn_outer)
            self._rc.coast_right()
            return p.turn_outer, 0, 'TURN_RIGHT'
        else:            # 0x03 both black → corner CCW
            self._rc.brake_both()
            return 0, 0, 'LOST'

    # ------------------------------------------------------------------
    def enable(self, on: bool) -> None:
        self._rc.enable(on)

    def safe_stop(self) -> None:
        self._rc.coast_both()
        self._rc.enable(False)

    def check_fault(self):
        return self._rc.read_fault()


# ---------------------------------------------------------------------------
# movement_process — entry point for the child process
# ---------------------------------------------------------------------------
def movement_process(inf_to_mov_q, shutdown_event, ready_event, startup_q, enable_event, config):
    """Child-process entry point.  Called after fork() by robot_main."""
    # 1. Ignore SIGINT in child (supervisor handles it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from pynq import MMIO
        from robot_control import RobotControl
        from motion_config import RC_BASE, RC_SIZE

        rc   = RobotControl(MMIO(RC_BASE, RC_SIZE))

        # 6. Verify hardware version
        version = rc.read_version()
        if version != (1, 0, 0):
            startup_q.put({'ok': False, 'error': f'Bad VERSION: 0x{version:08X}'})
            ready_event.set()
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
        ctrl = MovementController(rc)

        # 12. Prime watchdog and enable
        rc.enable(False)
        rc.set_stop_policy(False)
        rc.configure_watchdog(
            enable=True,
            brake_on_expiry=False,
        )

        rc.kick_watchdog()
        rc.clear_fault_bits(watchdog=True)
        rc.clear_estop()
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
