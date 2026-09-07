#!/usr/bin/env python3
"""robot_main.py — 4-process robot control supervisor.

Usage examples
--------------
# Smoke-test with fake motors:
  python3 robot_main.py --dry-run

# Camera + inference + logging only (no motors):
  python3 robot_main.py --camera-inference-only

# Full system (all 4 processes):
  python3 robot_main.py --full --confirm
"""

import argparse
import multiprocessing
import os
import signal
import sys
import time

# ---------------------------------------------------------------------------
# Paths to validate before starting
# ---------------------------------------------------------------------------
DEMO_DIR       = '/home/xilinx/nano_gtsrb_robot_demo'
OVERLAY_BIT    = '/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.bit'
OVERLAY_HWH    = '/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.hwh'
DRIVER_DIR     = '/home/xilinx/overlays/nano_gtsrb_robot/robot_control'
ROBOT_CTRL_PY  = os.path.join(DRIVER_DIR, 'robot_control.py')

# Timeout constants (seconds)
TIMEOUT_INFERENCE = 90
TIMEOUT_MOVEMENT  = 15
TIMEOUT_CAMERA    = 15
TIMEOUT_COMM      = 5


# ---------------------------------------------------------------------------
# Config dataclass compatible with existing code
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class SupervisorConfig:
    camera_device: int = 0
    frame_width:   int = 320
    frame_height:  int = 240
    target_fps:    int = 20
    score_window:  int = 10
    log_path:      str = None
    tcp_port:      int = 50007


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Robot control supervisor')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',               action='store_true',
                      help='Fake MMIO — no real motors')
    mode.add_argument('--camera-inference-only', action='store_true',
                      help='Camera + inference + comm only (no movement)')
    mode.add_argument('--movement-only',         action='store_true',
                      help='Movement + IR only (no camera/inference)')
    mode.add_argument('--full',                  action='store_true',
                      help='All 4 processes (requires --confirm)')
    p.add_argument('--confirm', action='store_true',
                   help='Required with --full to enable motors')
    p.add_argument('--fps',    type=int, default=20,
                   metavar='N', help='Camera target FPS (default 20)')
    p.add_argument('--window', type=int, default=10,
                   metavar='N', help='Temporal filter window (default 10)')
    p.add_argument('--port',   type=int, default=50007,
                   metavar='N', help='TCP port for comm process (default 50007)')
    p.add_argument('--log',    type=str, default=None,
                   metavar='PATH', help='Comm log path')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------
def validate_files(skip_hardware: bool = False):
    """Raise SystemExit if required files are missing."""
    errors = []
    if not skip_hardware:
        for path in (OVERLAY_BIT, OVERLAY_HWH, DRIVER_DIR, ROBOT_CTRL_PY):
            if not os.path.exists(path):
                errors.append(f"  Missing: {path}")
    if errors:
        print("[supervisor] Pre-flight check FAILED:")
        for e in errors:
            print(e)
        sys.exit(1)
    print("[supervisor] Pre-flight check OK", flush=True)


# ---------------------------------------------------------------------------
# Process launchers
# ---------------------------------------------------------------------------
def start_inference(cam_to_inf_q, inf_to_mov_q, inf_to_com_q,
                    shutdown_event, ready_event, config):
    sys.path.insert(0, DEMO_DIR)
    from proc_inference import inference_process
    p = multiprocessing.Process(
        target=inference_process,
        args=(cam_to_inf_q, inf_to_mov_q, inf_to_com_q,
              shutdown_event, ready_event, config),
        name='inference',
        daemon=False,
    )
    p.start()
    return p


def start_movement(inf_to_mov_q, shutdown_event, ready_event,
                   startup_q, enable_event, dry_run):
    sys.path.insert(0, DEMO_DIR)
    from proc_movement import movement_process
    p = multiprocessing.Process(
        target=movement_process,
        args=(inf_to_mov_q, shutdown_event, ready_event,
              startup_q, enable_event, dry_run),
        name='movement',
        daemon=False,
    )
    p.start()
    return p


def start_camera(cam_to_inf_q, shutdown_event, ready_event, config):
    sys.path.insert(0, DEMO_DIR)
    from proc_camera import camera_process
    p = multiprocessing.Process(
        target=camera_process,
        args=(cam_to_inf_q, shutdown_event, ready_event, config),
        name='camera',
        daemon=False,
    )
    p.start()
    return p


def start_comm(inf_to_com_q, shutdown_event, ready_event, log_path, tcp_port=50007):
    sys.path.insert(0, DEMO_DIR)
    from proc_comm import comm_process
    p = multiprocessing.Process(
        target=comm_process,
        args=(inf_to_com_q, shutdown_event, ready_event),
        kwargs={'log_path': log_path, 'tcp_port': tcp_port},
        name='comm',
        daemon=False,
    )
    p.start()
    return p


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def shutdown_all(processes, shutdown_event, timeout_each=5.0):
    shutdown_event.set()
    for p in processes:
        if p is not None and p.is_alive():
            p.join(timeout=timeout_each)
            if p.is_alive():
                print(f"[supervisor] Terminating {p.name} (did not exit cleanly)", flush=True)
                p.terminate()
                p.join(timeout=2.0)
            if p.is_alive():
                print(f"[supervisor] Killing {p.name} (SIGTERM not effective)", flush=True)
                p.kill()
                p.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.full and not args.confirm:
        print("ERROR: --full requires --confirm (this will enable motors!)")
        sys.exit(1)

    # Determine which subsystems to run
    run_inference = args.full or args.camera_inference_only
    run_camera    = args.full or args.camera_inference_only
    run_movement  = args.full or args.movement_only or args.dry_run
    run_comm      = True  # always log

    dry_run = args.dry_run

    # Pre-flight: skip hardware checks in dry-run mode
    validate_files(skip_hardware=dry_run)

    # Build config
    config = SupervisorConfig(
        target_fps   = args.fps,
        score_window = args.window,
        log_path     = args.log,
        tcp_port     = args.port,
    )

    # IPC primitives
    cam_to_inf_q = multiprocessing.Queue(maxsize=1)
    inf_to_mov_q = multiprocessing.Queue(maxsize=1)
    inf_to_com_q = multiprocessing.Queue(maxsize=1)
    startup_q    = multiprocessing.Queue()  # movement → supervisor startup info

    shutdown_event  = multiprocessing.Event()
    inf_ready_event = multiprocessing.Event()
    mov_ready_event = multiprocessing.Event()
    cam_ready_event = multiprocessing.Event()
    com_ready_event = multiprocessing.Event()
    enable_event    = multiprocessing.Event()  # supervisor → movement: OK to enable

    processes = []

    # ----------------------------------------------------------------
    # Signal handling (supervisor only)
    # ----------------------------------------------------------------
    def _sig_handler(signum, frame):
        print(f"\n[supervisor] Signal {signum} received — shutting down …", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ----------------------------------------------------------------
    # Start processes in dependency order
    # ----------------------------------------------------------------
    proc_inf = proc_mov = proc_cam = proc_com = None

    try:
        # 1. Inference first (programs FPGA — longest startup)
        if run_inference:
            print("[supervisor] Starting inference process …", flush=True)
            proc_inf = start_inference(
                cam_to_inf_q, inf_to_mov_q, inf_to_com_q,
                shutdown_event, inf_ready_event, config,
            )
            processes.append(proc_inf)
            if not inf_ready_event.wait(timeout=TIMEOUT_INFERENCE):
                print("[supervisor] ERROR: inference process did not become ready in time",
                      flush=True)
                shutdown_event.set()
                shutdown_all(processes, shutdown_event)
                sys.exit(1)
            print("[supervisor] Inference ready.", flush=True)

        # 2. Movement
        if run_movement:
            print("[supervisor] Starting movement process …", flush=True)
            proc_mov = start_movement(
                inf_to_mov_q, shutdown_event,
                mov_ready_event, startup_q, enable_event, dry_run,
            )
            processes.append(proc_mov)
            if not mov_ready_event.wait(timeout=TIMEOUT_MOVEMENT):
                print("[supervisor] ERROR: movement process did not become ready in time",
                      flush=True)
                shutdown_event.set()
                shutdown_all(processes, shutdown_event)
                sys.exit(1)
            # Read startup diagnostics
            try:
                info = startup_q.get(timeout=2.0)
                if info.get('ok'):
                    print(f"[supervisor] Movement ready — "
                          f"VERSION=0x{info.get('version', 0):08X} "
                          f"ENABLE={info.get('enable')}", flush=True)
                else:
                    print(f"[supervisor] Movement startup error: {info.get('error')}", flush=True)
                    shutdown_event.set()
                    shutdown_all(processes, shutdown_event)
                    sys.exit(1)
            except Exception as exc:
                print(f"[supervisor] Could not read startup_q: {exc}", flush=True)

        # 3. Camera
        if run_camera:
            print("[supervisor] Starting camera process …", flush=True)
            proc_cam = start_camera(
                cam_to_inf_q, shutdown_event, cam_ready_event, config,
            )
            processes.append(proc_cam)
            if not cam_ready_event.wait(timeout=TIMEOUT_CAMERA):
                print("[supervisor] ERROR: camera process did not become ready in time",
                      flush=True)
                shutdown_event.set()
                shutdown_all(processes, shutdown_event)
                sys.exit(1)
            print("[supervisor] Camera ready.", flush=True)

        # 4. Comm
        if run_comm:
            print("[supervisor] Starting comm process …", flush=True)
            proc_com = start_comm(
                inf_to_com_q, shutdown_event, com_ready_event,
                log_path=config.log_path,
                tcp_port=config.tcp_port,
            )
            processes.append(proc_com)
            if not com_ready_event.wait(timeout=TIMEOUT_COMM):
                print("[supervisor] ERROR: comm process did not become ready in time",
                      flush=True)
                shutdown_event.set()
                shutdown_all(processes, shutdown_event)
                sys.exit(1)
            print("[supervisor] Comm ready.", flush=True)

        # 5. Enable motors (movement process waits for this)
        if run_movement:
            if dry_run:
                print("[supervisor] DRY-RUN: setting enable_event (no real motors)", flush=True)
            elif args.camera_inference_only:
                pass  # enable_event never set → motors never enabled
            else:
                print("[supervisor] Enabling motors …", flush=True)
            enable_event.set()

        print("[supervisor] All processes running. Press Ctrl-C to stop.", flush=True)

        # ----------------------------------------------------------------
        # Monitor loop — check liveness every 500 ms
        # ----------------------------------------------------------------
        essential = [p for p in (proc_inf, proc_mov, proc_cam) if p is not None]

        while not shutdown_event.is_set():
            time.sleep(0.5)
            for p in essential:
                if p is not None and not p.is_alive():
                    print(f"[supervisor] ESTOP — essential process '{p.name}' died "
                          f"(exit code {p.exitcode})", flush=True)
                    shutdown_event.set()
                    break

    except KeyboardInterrupt:
        # SIGINT already handled; just fall through
        pass
    finally:
        print("[supervisor] Shutting down all processes …", flush=True)
        shutdown_all(processes, shutdown_event, timeout_each=8.0)
        print("[supervisor] Clean exit.", flush=True)


if __name__ == '__main__':
    main()
