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
    EXPECTED_VERSION,
)
from sign_config import classify_sign


class MovementController:
    """
    Movement state machine and line-following controller.

    Hardware access is provided through an injected RobotControl-compatible
    backend, allowing real hardware and dry-run operation.
    """

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
    def initialize(self):
        """Prepare movement hardware but DO NOT enable the motors or watchdog."""

        version = self._rc.read_version()

        if version != EXPECTED_VERSION:
            raise RuntimeError(f"Bad VERSION: {version}")

        # Always enter a known safe startup state.
        self._rc.enable(False)
        self._rc.coast_both()

        if self._rc.is_enabled():
            raise RuntimeError("Motor controller could not be disabled")

        self._rc.set_stop_policy(False)

        self._rc.configure_watchdog(enable=False, brake_on_expiry=False)

        self._rc.clear_fault_bits(watchdog=True)
        self._rc.clear_estop()

        return {"version": version, "enable": False}

    # ------------------------------------------------------------------
    def arm(self):
        """Enable watchdog and motion after supervisor authorization."""

        # Motors must still be off at this point
        if self._rc.is_enabled():
            raise RuntimeError("Motor controller unexpectedly enabled before arm")

        self._rc.configure_watchdog(enable=True, brake_on_expiry=False)
        self._rc.kick_watchdog()
        self._rc.enable(True)

        if not self._rc.is_enabled():
            raise RuntimeError("Motor controller failed to enable")

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

        match self._state:

            case 'STOP_HOLD':
                self._rc.coast_both()
                action = 'COAST'
                if mono_now >= self._stop_deadline:
                    # Resume line-following; start cooldown
                    self._stop_cooldown_until = mono_now + STOP_COOLDOWN_S
                    self._state = 'LINE_FOLLOW'
                    action = 'RESUME'

            case 'LINE_FOLLOW':
                left_s, right_s, action = self._apply_line_follow(ir)

            case _:
                raise RuntimeError(f"Unknown movement state: {self._state}")

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

        if self._state == 'LINE_FOLLOW':

            if category == 'stop' and mono_now >= self._stop_cooldown_until:
                    self._state = 'STOP_HOLD'
                    self._stop_deadline = mono_now + STOP_HOLD_S

            elif category in ('speed_slow', 'speed_fast'):
                profile_key = category[len('speed_'):]   # 'slow', 'normal', 'fast', etc
                self._profile = PROFILES[profile_key]

        # A non-stop sign clears the cooldown so the next stop can act sooner
        if category != 'stop' and mono_now < self._stop_cooldown_until:
            self._stop_cooldown_until = 0.0

    # ------------------------------------------------------------------
    def _apply_line_follow(self, ir: int):
        """Apply line-follow corrections based on filtered IR value.

        Returns (left_speed, right_speed, action_label).
        """
        p = self._profile
        if ir == 0b00:   # both white → go straight
            self._rc.set_left(True, p.base_duty)
            self._rc.set_right(True, p.base_duty)
            return p.base_duty, p.base_duty, 'STRAIGHT'
        elif ir == 0b01: # left black → turn left
            self._rc.coast_left()
            self._rc.set_right(True, p.turn_outer)
            return 0, p.turn_outer, 'TURN_LEFT'
        elif ir == 0b10: # right black → turn right
            self._rc.set_left(True, p.turn_outer)
            self._rc.coast_right()
            return p.turn_outer, 0, 'TURN_RIGHT'
        else:            # 0b11 both black → corner CCW
            self._rc.set_left(True, p.corner_inner)
            self._rc.set_right(True, p.turn_outer)
            return p.corner_inner, p.turn_outer, 'CORNER'

    # ------------------------------------------------------------------

    def safe_stop(self) -> None:
        self._rc.coast_both()
        self._rc.enable(False)

    def check_fault(self):
        return self._rc.read_fault()

def create_robot_control(dry_run: bool):
    if dry_run:
        from robot_control_dry_run import DryRunRobotControl

        print("[movement] DRY-RUN mode", flush=True)

        return DryRunRobotControl()

    # Import hardware only in real mode
    from pynq import MMIO
    from robot_control import RobotControl
    from motion_config import (RC_BASE, RC_SIZE)

    return RobotControl(MMIO(RC_BASE, RC_SIZE))

def run_control_loop(ctrl, inf_to_mov_q, shutdown_event):

    period = 1.0 / MOVEMENT_HZ
    next_tick = time.monotonic()

    while not shutdown_event.is_set():
        try:
            event = inf_to_mov_q.get_nowait()
        except Empty:
            event = None

        ctrl.step(event=event)

        fault = ctrl.check_fault()

        if (fault.watchdog_expired or fault.estop_active):
            print("[movement] FAULT detected: "
                f"wdg={fault.watchdog_expired} "
                f"estop={fault.estop_active}", flush=True)
            break

        next_tick += period
        sleep_for = next_tick - time.monotonic()

        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

# ---------------------------------------------------------------------------
# movement_process — entry point for the child process
# ---------------------------------------------------------------------------
def movement_process(inf_to_mov_q, shutdown_event, ready_event, startup_q, enable_event, dry_run = False):
    """Child-process entry point.  Called after fork() by robot_main."""

    # 1. Ignore SIGINT in child (supervisor handles it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    ctrl = None
    startup_complete = False

    try:
        # --------------------------------
        # Backend

        rc   = create_robot_control(dry_run)

        ctrl = MovementController(rc)

        # --------------------------------
        # Safe initialization

        startup_info = ctrl.initialize()

        startup_q.put({**startup_info, 'ok': True})
        ready_event.set()
        startup_complete = True

        # --------------------------------
        # Supervisor authorization

        enable_event.wait()
        if shutdown_event.is_set():
            return

        # --------------------------------
        # Arm movement

        ctrl.arm()

        # --------------------------------
        # Control loop

        run_control_loop(ctrl, inf_to_mov_q, shutdown_event)

    except Exception as exc:
        print(f"[movement] Exception: {exc}", flush=True)

        import traceback;
        traceback.print_exc()

        if not startup_complete:
            try:
                startup_q.put({'ok': False, 'error': str(exc)})
                ready_event.set()
            except Exception:
                pass

    finally:
        if ctrl is not None:
            ctrl.safe_stop()

        print("[movement] process exiting", flush=True)
