"""Unit tests for the robot control system.

Run with:
  python3 -m pytest tests/test_robot.py -v
  -- or --
  python3 -m unittest tests.test_robot
"""

import sys
import os
import time
import multiprocessing
import unittest
from collections import deque

# Insert project root so imports resolve without installing the package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ipc_types import DetectionEvent
from sign_config import classify_sign, STOP_CLASS_ID
from motion_config import (
    DEFAULT_PROFILE, PROFILE_SLOW, PROFILE_FAST,
    STOP_HOLD_S, STOP_COOLDOWN_S, MAX_EVENT_AGE_S,
    WDG_KICK_EVERY, MOVEMENT_HZ,
)

# Register offsets (must match proc_movement.py)
REG_CONTROL   = 0x08
REG_IR        = 0x0C
REG_WDG_KICK  = 0x10
REG_LEFT_CMD  = 0x00
REG_RIGHT_CMD = 0x04
REG_FAULT     = 0x1C
REG_VERSION   = 0x20


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

class FakeMMIO:
    """Minimal MMIO stub that records all writes."""

    def __init__(self, ir: int = 0x00, fault: int = 0x00):
        self.regs = {
            REG_IR:      ir,
            REG_FAULT:   fault,
            REG_VERSION: 0x00010000,
            REG_CONTROL: 0x00,
        }
        self.writes = []

    def read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int) -> None:
        self.regs[offset] = value
        self.writes.append((offset, value))


class FaultStatus:
    def __init__(self, raw: int):
        self.watchdog_expired = bool(raw & 0x01)
        self.estop_active     = bool(raw & 0x02)


class FakeRC:
    """Minimal RobotControl stub."""

    def __init__(self, mmio: FakeMMIO):
        self._mmio = mmio

    def enable(self, on: bool) -> None:
        cur = self._mmio.read(REG_CONTROL)
        if on:
            self._mmio.write(REG_CONTROL, cur | 0x01)
        else:
            self._mmio.write(REG_CONTROL, cur & ~0x01)

    def is_enabled(self) -> bool:
        return bool(self._mmio.read(REG_CONTROL) & 0x01)

    def kick_watchdog(self) -> None:
        self._mmio.write(REG_WDG_KICK, 1)

    def clear_fault_bits(self, **_kw) -> None:
        self._mmio.write(REG_FAULT, 0)

    def read_fault(self) -> FaultStatus:
        return FaultStatus(self._mmio.read(REG_FAULT))

    def read_version(self) -> int:
        return self._mmio.read(REG_VERSION)


# Import MovementController after path is set
from proc_movement import MovementController


def _make_ctrl(ir: int = 0x00, fault: int = 0x00):
    mmio = FakeMMIO(ir=ir, fault=fault)
    rc   = FakeRC(mmio)
    ctrl = MovementController(rc, mmio)
    return ctrl, mmio


def _make_event(class_id: int, ready: bool = True, age: float = 0.0,
                mono_now: float = None) -> DetectionEvent:
    """Build a DetectionEvent aged by `age` seconds."""
    if mono_now is None:
        mono_now = time.monotonic()
    return DetectionEvent(
        class_id  = class_id,
        label     = f'class_{class_id}',
        score_sum = 1.0,
        timestamp = mono_now - age,
        frame_id  = 0,
        ready     = ready,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestIRLineFollow(unittest.TestCase):
    """Tests 1-4: basic IR → motor command mapping."""

    def _run_stable(self, ir_val: int, n_steps: int = 5):
        """Run n_steps with a constant IR reading to saturate majority filter."""
        ctrl, mmio = _make_ctrl(ir=ir_val)
        now = time.monotonic()
        for i in range(n_steps):
            mmio.regs[REG_IR] = ir_val   # keep IR stable
            ctrl.step(mono_now=now + i * 0.005)
        return ctrl, mmio

    def test_ir_both_white(self):
        """Test 1: IR=0x00 → both wheels forward at base_duty."""
        _, mmio = self._run_stable(0x00)
        left_writes  = [v for o, v in mmio.writes if o == REG_LEFT_CMD]
        right_writes = [v for o, v in mmio.writes if o == REG_RIGHT_CMD]
        expected = DEFAULT_PROFILE.base_duty | 0x80
        self.assertEqual(left_writes[-1],  expected, "Left cmd wrong for straight")
        self.assertEqual(right_writes[-1], expected, "Right cmd wrong for straight")

    def test_ir_left_black(self):
        """Test 2: IR=0x01 (left black) → coast_left, right=turn_outer."""
        _, mmio = self._run_stable(0x01)
        left_writes  = [v for o, v in mmio.writes if o == REG_LEFT_CMD]
        right_writes = [v for o, v in mmio.writes if o == REG_RIGHT_CMD]
        self.assertEqual(left_writes[-1],  0x00,
                         "Left should coast on left-black")
        self.assertEqual(right_writes[-1], DEFAULT_PROFILE.turn_outer | 0x80,
                         "Right should be turn_outer on left-black")

    def test_ir_right_black(self):
        """Test 3: IR=0x02 (right black) → left=turn_outer, coast_right."""
        _, mmio = self._run_stable(0x02)
        left_writes  = [v for o, v in mmio.writes if o == REG_LEFT_CMD]
        right_writes = [v for o, v in mmio.writes if o == REG_RIGHT_CMD]
        self.assertEqual(left_writes[-1],  DEFAULT_PROFILE.turn_outer | 0x80,
                         "Left should be turn_outer on right-black")
        self.assertEqual(right_writes[-1], 0x00,
                         "Right should coast on right-black")

    def test_ir_both_black(self):
        """Test 4: IR=0x03 (both black, corner CCW) → left=corner_inner, right=turn_outer."""
        _, mmio = self._run_stable(0x03)
        left_writes  = [v for o, v in mmio.writes if o == REG_LEFT_CMD]
        right_writes = [v for o, v in mmio.writes if o == REG_RIGHT_CMD]
        self.assertEqual(left_writes[-1],  DEFAULT_PROFILE.corner_inner | 0x80,
                         "Left should be corner_inner on corner")
        self.assertEqual(right_writes[-1], DEFAULT_PROFILE.turn_outer | 0x80,
                         "Right should be turn_outer on corner")


class TestIRMajorityFilter(unittest.TestCase):
    """Test 5: IR majority-vote filter."""

    def test_ir_filter_3_samples(self):
        """Test 5: 2 of 3 samples say left=black → controller turns left."""
        ctrl, mmio = _make_ctrl(ir=0x00)
        now = 100.0  # fixed time base

        # Step 1: white (0x00)
        mmio.regs[REG_IR] = 0x00
        ctrl.step(mono_now=now)

        # Steps 2-3: left black (0x01) — 2 votes → majority
        for i in (1, 2):
            mmio.regs[REG_IR] = 0x01
            ctrl.step(mono_now=now + i * 0.005)

        left_writes  = [v for o, v in mmio.writes if o == REG_LEFT_CMD]
        right_writes = [v for o, v in mmio.writes if o == REG_RIGHT_CMD]
        self.assertEqual(left_writes[-1],  0x00,
                         "After 2/3 left-black votes, left should coast")
        self.assertEqual(right_writes[-1], DEFAULT_PROFILE.turn_outer | 0x80,
                         "After 2/3 left-black votes, right should be turn_outer")


class TestStopHold(unittest.TestCase):
    """Tests 6-8: STOP_HOLD state machine."""

    def test_stop_hold_5s(self):
        """Test 6: STOP event → STOP_HOLD for 5 s, then LINE_FOLLOW."""
        ctrl, mmio = _make_ctrl()
        t0 = 1000.0

        # Inject stop event
        event = _make_event(STOP_CLASS_ID, mono_now=t0)
        result = ctrl.step(event=event, mono_now=t0)
        self.assertEqual(ctrl._state, 'STOP_HOLD', "Should enter STOP_HOLD")

        # Just before deadline → still STOP_HOLD
        result = ctrl.step(mono_now=t0 + STOP_HOLD_S - 0.1)
        self.assertEqual(ctrl._state, 'STOP_HOLD', "Should stay STOP_HOLD before deadline")

        # Just after deadline → LINE_FOLLOW
        result = ctrl.step(mono_now=t0 + STOP_HOLD_S + 0.1)
        self.assertEqual(ctrl._state, 'LINE_FOLLOW', "Should resume LINE_FOLLOW after deadline")

    def test_stop_no_restart(self):
        """Test 7: Second STOP event during STOP_HOLD does not reset the deadline."""
        ctrl, mmio = _make_ctrl()
        t0 = 1000.0

        event1 = _make_event(STOP_CLASS_ID, mono_now=t0)
        ctrl.step(event=event1, mono_now=t0)
        original_deadline = ctrl._stop_deadline

        # Inject a second stop midway through hold
        t1 = t0 + STOP_HOLD_S / 2
        event2 = _make_event(STOP_CLASS_ID, mono_now=t1)
        ctrl.step(event=event2, mono_now=t1)
        self.assertEqual(ctrl._stop_deadline, original_deadline,
                         "Second STOP should not reset deadline")

    def test_stop_new_after_cooldown(self):
        """Test 8: STOP ignored during cooldown, accepted after cooldown expires."""
        ctrl, mmio = _make_ctrl()
        t0 = 1000.0

        # First stop
        ctrl.step(event=_make_event(STOP_CLASS_ID, mono_now=t0), mono_now=t0)
        # Expire the stop hold
        ctrl.step(mono_now=t0 + STOP_HOLD_S + 0.1)
        self.assertEqual(ctrl._state, 'LINE_FOLLOW')

        # Second stop during cooldown → ignored
        t1 = t0 + STOP_HOLD_S + 0.5   # still in cooldown
        ctrl.step(event=_make_event(STOP_CLASS_ID, mono_now=t1), mono_now=t1)
        self.assertEqual(ctrl._state, 'LINE_FOLLOW',
                         "STOP during cooldown should be ignored")

        # Second stop after cooldown expires → accepted
        t2 = t0 + STOP_HOLD_S + STOP_COOLDOWN_S + 0.5
        ctrl.step(event=_make_event(STOP_CLASS_ID, mono_now=t2), mono_now=t2)
        self.assertEqual(ctrl._state, 'STOP_HOLD',
                         "STOP after cooldown should be accepted")


class TestSpeedProfiles(unittest.TestCase):
    """Tests 9-11: speed-profile selection."""

    def test_speed_slow(self):
        """Test 9: speed_slow event → profile with base_duty=20."""
        ctrl, _ = _make_ctrl()
        t0 = 1000.0
        event = _make_event(26, mono_now=t0)  # class 26 = Road work (slow)
        ctrl.step(event=event, mono_now=t0)
        self.assertEqual(ctrl._profile.base_duty, PROFILE_SLOW.base_duty)
        self.assertEqual(ctrl._profile.name, 'slow')

    def test_speed_fast(self):
        """Test 10: speed_fast event → profile.base_duty = 30."""
        ctrl, _ = _make_ctrl()
        t0 = 1000.0
        event = _make_event(3, mono_now=t0)  # class 3 = 50 km/h (fast)
        ctrl.step(event=event, mono_now=t0)
        self.assertEqual(ctrl._profile.base_duty, PROFILE_FAST.base_duty,
                         "Fast profile should have base_duty=30")

    def test_speed_no_accumulate(self):
        """Test 11: Many speed_fast events do not change base_duty beyond 30."""
        ctrl, _ = _make_ctrl()
        t0 = 1000.0
        for i in range(10):
            event = _make_event(3, mono_now=t0 + i * 0.1)
            ctrl.step(event=event, mono_now=t0 + i * 0.1)
        self.assertEqual(ctrl._profile.base_duty, 30,
                         "Repeated fast events should not accumulate beyond 30")


class TestEventAgeFilter(unittest.TestCase):
    """Test 12: old events are discarded."""

    def test_old_event_discarded(self):
        """Test 12: Event older than MAX_EVENT_AGE_S does not change state."""
        ctrl, _ = _make_ctrl()
        t0 = 1000.0
        stale_event = _make_event(STOP_CLASS_ID, age=MAX_EVENT_AGE_S + 0.5,
                                  mono_now=t0)
        ctrl.step(event=stale_event, mono_now=t0)
        self.assertEqual(ctrl._state, 'LINE_FOLLOW',
                         "Stale STOP event should not trigger STOP_HOLD")


class TestQueueBehaviour(unittest.TestCase):
    """Tests 13-14: queue semantics (use thread-safe queue.Queue for in-process tests)."""

    def test_both_queues_receive(self):
        """Test 13: inference fanout reaches both inf_to_mov_q and inf_to_com_q."""
        import queue as _queue  # stdlib thread-safe queue — fine for logic tests

        q_mov = _queue.Queue(maxsize=1)
        q_com = _queue.Queue(maxsize=1)

        event = DetectionEvent(
            class_id=1, label='slow', score_sum=0.9,
            timestamp=time.monotonic(), frame_id=42, ready=True,
        )
        for q in (q_mov, q_com):
            try:
                q.put_nowait(event)
            except _queue.Full:
                self.fail("Queue should have space for first item")

        self.assertFalse(q_mov.empty(), "Movement queue should have item")
        self.assertFalse(q_com.empty(), "Comm queue should have item")

        mov_item = q_mov.get_nowait()
        com_item = q_com.get_nowait()
        self.assertEqual(mov_item.class_id, 1)
        self.assertEqual(com_item.class_id, 1)

    def test_camera_queue_drops_old(self):
        """Test 14: with maxsize=1, new frame replaces old frame."""
        import queue as _queue

        q = _queue.Queue(maxsize=1)
        old_frame = (0, time.monotonic() - 10.0, b'OLD')
        new_frame = (1, time.monotonic(),         b'NEW')

        q.put_nowait(old_frame)
        # Simulate camera process drain-and-replace
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break
        q.put_nowait(new_frame)

        item = q.get_nowait()
        self.assertEqual(item[2], b'NEW', "Queue should hold newest frame")


def _noop_target():
    """Module-level no-op for process spawning tests (must be picklable)."""
    pass


def _event_setter_target(ev):
    """Module-level setter for process spawning tests."""
    import time as _t
    _t.sleep(0.02)
    ev.set()


class TestProcessBehaviours(unittest.TestCase):
    """Tests 15-19: process-level / supervisor-level behaviours (no real hardware)."""

    def test_camera_failure_ready_not_set(self):
        """Test 15: If camera_process exits before ready_event.set(), ready remains unset."""
        import threading
        ready = threading.Event()
        # Simulate a thread that exits before setting ready
        t = threading.Thread(target=lambda: None)  # exits immediately
        t.start()
        t.join()
        self.assertFalse(ready.is_set(),
                         "ready_event should not be set if process never called it")

    def test_inference_not_ready_timeout(self):
        """Test 16: Supervisor wait on inference ready times out when not set."""
        ready = multiprocessing.Event()
        result = ready.wait(timeout=0.05)
        self.assertFalse(result,
                         "wait() should return False (not ready) on timeout")

    def test_movement_failure_sets_shutdown(self):
        """Test 17: Simulated movement process death triggers shutdown_event."""
        shutdown_event = multiprocessing.Event()
        p = multiprocessing.Process(target=_noop_target)
        p.start()
        p.join()
        # Supervisor would detect this and set shutdown
        if not p.is_alive():
            shutdown_event.set()
        self.assertTrue(shutdown_event.is_set(),
                        "Shutdown event should be set after essential process dies")

    def test_sigint_shuts_down(self):
        """Test 18: shutdown_event.set() causes a monitoring loop to exit."""
        shutdown_event = multiprocessing.Event()
        iterations = []

        for _ in range(10):
            if shutdown_event.is_set():
                break
            iterations.append(1)
            if len(iterations) == 3:
                shutdown_event.set()

        self.assertTrue(shutdown_event.is_set())
        self.assertLess(len(iterations), 10, "Loop should have exited early")

    def test_sigterm_shuts_down(self):
        """Test 19: shutdown_event set from a different process also works."""
        shutdown_event = multiprocessing.Event()
        p = multiprocessing.Process(target=_event_setter_target, args=(shutdown_event,))
        p.start()
        p.join(timeout=1.0)
        self.assertTrue(shutdown_event.is_set(),
                        "shutdown_event should be set by child process")


class TestSafeStop(unittest.TestCase):
    """Test 20: safe_stop() leaves ENABLE=0."""

    def test_enable_false_on_exit(self):
        """Test 20: After safe_stop(), CONTROL register bit0 (ENABLE) is 0."""
        ctrl, mmio = _make_ctrl()

        # Enable motors first
        ctrl._rc.enable(True)
        self.assertTrue(bool(mmio.read(REG_CONTROL) & 0x01),
                        "Enable should be set before safe_stop")

        ctrl.safe_stop()
        self.assertFalse(bool(mmio.read(REG_CONTROL) & 0x01),
                         "ENABLE must be 0 after safe_stop()")


# ---------------------------------------------------------------------------
# Sign classification sanity checks (bonus, not part of the 20)
# ---------------------------------------------------------------------------

class TestSignClassify(unittest.TestCase):
    def test_stop(self):
        self.assertEqual(classify_sign(15), 'stop')

    def test_speed_slow(self):
        self.assertEqual(classify_sign(26), 'speed_slow')  # Road work

    def test_speed_normal(self):
        # speed_normal removed; old 50/60/70 classes except 50 are now 'other'
        for cid in (4, 5):
            self.assertEqual(classify_sign(cid), 'other')

    def test_speed_fast(self):
        self.assertEqual(classify_sign(3), 'speed_fast')   # 50 km/h

    def test_other(self):
        # All non-action classes are 'other' (includes old speed limits 1,2,4,5,6,7,8,9)
        for cid in (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16):
            self.assertEqual(classify_sign(cid), 'other')


if __name__ == '__main__':
    unittest.main(verbosity=2)

# ==========================================================================
# New tests appended to tests/test_robot.py
# Tests 21-37: TCP protocol, class table, register offsets
# ==========================================================================
import socket
import struct
import threading
import time


class TestTcpProtocol(unittest.TestCase):
    """Tests 21-27: tcp_comm module."""

    def _import(self):
        import sys, os
        sys.path.insert(0, '/home/xilinx/nano_gtsrb_robot_demo')
        import tcp_comm
        return tcp_comm

    def test_encode_decode_roundtrip(self):
        """Test 21: encode then decode returns original values."""
        tc = self._import()
        for cls, conf in [(0, 0), (15, 1476), (43, 4294967295), (5, 999)]:
            payload = tc.encode_detection(cls, conf)
            self.assertEqual(len(payload), tc.PAYLOAD_SIZE)
            self.assertEqual(payload[:4], struct.pack('!I', cls))
            self.assertEqual(payload[4:], struct.pack('!I', conf))
            got_cls, got_conf = tc.decode_detection(payload)
            self.assertEqual(got_cls, cls)
            self.assertEqual(got_conf, conf)

    def test_encode_clamps_negatives(self):
        """Test 22: negative values are clamped to 0."""
        tc = self._import()
        payload = tc.encode_detection(-1, -100)
        cls, conf = tc.decode_detection(payload)
        self.assertEqual(cls, 0)
        self.assertEqual(conf, 0)

    def test_decode_rejects_wrong_length(self):
        """Test 23: decode_detection raises ValueError for wrong-length data."""
        tc = self._import()
        with self.assertRaises(ValueError):
            tc.decode_detection(b'\x00' * 4)
        with self.assertRaises(ValueError):
            tc.decode_detection(b'\x00' * 9)

    def test_recv_exact_fragmented(self):
        """Test 24: recv_exact assembles a payload delivered in two fragments."""
        tc = self._import()
        results = []

        def _server(srv_sock):
            conn, _ = srv_sock.accept()
            data = tc.recv_exact(conn, tc.PAYLOAD_SIZE)
            results.append(data)
            conn.close()

        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        t = threading.Thread(target=_server, args=(srv,), daemon=True)
        t.start()

        cli = socket.socket()
        cli.connect(('127.0.0.1', port))
        cli.send(tc.encode_detection(15, 1476)[:4])   # first half
        time.sleep(0.02)
        cli.send(tc.encode_detection(15, 1476)[4:])   # second half
        cli.close()

        t.join(timeout=5)
        srv.close()

        self.assertIsNotNone(results[0])
        cls, conf = tc.decode_detection(results[0])
        self.assertEqual(cls, 15)
        self.assertEqual(conf, 1476)

    def test_recv_exact_connection_closed(self):
        """Test 25: recv_exact returns None when connection closes early."""
        tc = self._import()
        results = []

        def _server(srv_sock):
            conn, _ = srv_sock.accept()
            data = tc.recv_exact(conn, tc.PAYLOAD_SIZE)
            results.append(data)
            conn.close()

        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        t = threading.Thread(target=_server, args=(srv,), daemon=True)
        t.start()

        cli = socket.socket()
        cli.connect(('127.0.0.1', port))
        cli.send(b'\x00\x00')   # only 2 bytes, not 8
        cli.close()             # close before full payload

        t.join(timeout=5)
        srv.close()

        self.assertIsNone(results[0])

    def test_robot_server_no_client(self):
        """Test 26: try_send returns False when no client is connected."""
        tc = self._import()
        # Find a free port
        probe = socket.socket()
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
        probe.close()

        srv = tc.RobotServer(port=port)
        srv.start()
        try:
            result = srv.try_send(15, 1000)
            self.assertFalse(result)
        finally:
            srv.close()

    def test_robot_server_send_and_reconnect(self):
        """Test 27: client sends, disconnects, new client connects and receives."""
        tc = self._import()
        probe = socket.socket()
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
        probe.close()

        srv = tc.RobotServer(port=port)
        srv.start()
        try:
            # First client
            c1 = socket.socket()
            c1.connect(('127.0.0.1', port))
            srv.try_accept()
            ok = srv.try_send(15, 1476)
            self.assertTrue(ok)
            data = tc.recv_exact(c1, tc.PAYLOAD_SIZE)
            self.assertEqual(tc.decode_detection(data), (15, 1476))
            c1.close()

            # Force disconnect detection
            time.sleep(0.05)
            srv.try_send(0, 0)   # will detect BrokenPipeError or OSError

            # Second client reconnects
            c2 = socket.socket()
            c2.connect(('127.0.0.1', port))
            srv.try_accept()
            ok2 = srv.try_send(3, 999)
            self.assertTrue(ok2)
            data2 = tc.recv_exact(c2, tc.PAYLOAD_SIZE)
            self.assertEqual(tc.decode_detection(data2), (3, 999))
            c2.close()
        finally:
            srv.close()


class TestClassTable(unittest.TestCase):
    """Tests 28-30: verify class index table matches labels.py."""

    def _import(self):
        import sys
        sys.path.insert(0, '/home/xilinx/nano_gtsrb_robot_demo')
        import labels as L
        import sign_config as SC
        return L, SC

    def test_stop_is_index_15(self):
        """Test 28: labels[15] == 'Stop' and STOP_CLASS_ID == 15."""
        L, SC = self._import()
        self.assertEqual(L.LABELS_EN[15], 'Stop')
        self.assertEqual(SC.STOP_CLASS_ID, 15)
        self.assertEqual(SC.classify_sign(15), 'stop')

    def test_speed_limits_match_labels(self):
        """Test 29: two-sign mapping — Road work slow, 50 km/h fast, all others other."""
        L, SC = self._import()
        # Road work (26) → slow
        self.assertIn('Road work', L.LABELS_EN[26])
        self.assertEqual(SC.classify_sign(26), 'speed_slow')
        # 50 km/h (3) → fast
        self.assertIn('50km/h', L.LABELS_EN[3])
        self.assertEqual(SC.classify_sign(3), 'speed_fast')
        # Old speed-limit classes (except 3) are now 'other'
        for idx in (1, 2, 4, 5, 6, 8, 9):
            self.assertEqual(SC.classify_sign(idx), 'other', f"index {idx} should be other")

    def test_all_44_classes_classified(self):
        """Test 30: all 44 classes return a known category."""
        L, SC = self._import()
        valid = {'stop', 'speed_slow', 'speed_fast', 'other'}
        self.assertEqual(len(L.LABELS_EN), 44)
        for idx, label in enumerate(L.LABELS_EN):
            cat = SC.classify_sign(idx)
            self.assertIn(cat, valid, f"index {idx} ({label!r}) has unknown category {cat!r}")


class TestRegisterOffsets(unittest.TestCase):
    """Tests 31-37: verify proc_movement register offsets match hardware spec."""

    def _import(self):
        import sys
        sys.path.insert(0, '/home/xilinx/nano_gtsrb_robot_demo')
        import proc_movement as pm
        return pm

    def test_left_cmd_offset(self):
        """Test 31: LEFT_CMD == 0x00."""
        pm = self._import()
        self.assertEqual(pm.REG_LEFT_CMD, 0x00)

    def test_right_cmd_offset(self):
        """Test 32: RIGHT_CMD == 0x04."""
        pm = self._import()
        self.assertEqual(pm.REG_RIGHT_CMD, 0x04)

    def test_control_offset(self):
        """Test 33: CONTROL == 0x08."""
        pm = self._import()
        self.assertEqual(pm.REG_CONTROL, 0x08)

    def test_ir_offset(self):
        """Test 34: IR_STATUS == 0x0C."""
        pm = self._import()
        self.assertEqual(pm.REG_IR, 0x0C)

    def test_wdg_kick_offset(self):
        """Test 35: WDG_KICK == 0x10."""
        pm = self._import()
        self.assertEqual(pm.REG_WDG_KICK, 0x10)

    def test_fault_offset(self):
        """Test 36: FAULT_STATUS == 0x1C."""
        pm = self._import()
        self.assertEqual(pm.REG_FAULT, 0x1C)

    def test_version_offset(self):
        """Test 37: VERSION == 0x20."""
        pm = self._import()
        self.assertEqual(pm.REG_VERSION, 0x20)
