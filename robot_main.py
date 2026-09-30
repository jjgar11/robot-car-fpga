#!/usr/bin/env python3
"""
robot_main.py — 4-process robot control supervisor.
"""

import multiprocessing
import os
import signal
import sys
import time

from config import DemoConfig, DetectionEvent
from proc_inference import inference_process
from proc_movement import movement_process
from proc_camera import camera_process
from proc_comm import comm_process

# Timeout constants (seconds)
TIMEOUT_INFERENCE = 90
TIMEOUT_MOVEMENT  = 15
TIMEOUT_CAMERA    = 15
TIMEOUT_COMM      = 5


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------
def validate_files(config):
    """Raise SystemExit if required files are missing."""
    errors = []
    required = (
        config.bitfile,
        config.hwhfile,
        config.driver_dir,
    )
    for path in required:
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
def start_inference(cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, ready_event, config, detection_event):
    p = multiprocessing.Process(
        target=inference_process,
        args=(cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, ready_event, config, detection_event),
        name='inference',
        daemon=False,
    )
    p.start()
    return p


def start_movement(inf_to_mov_q, shutdown_event, ready_event, startup_q, enable_event, config):
    p = multiprocessing.Process(
        target=movement_process,
        args=(inf_to_mov_q, shutdown_event, ready_event, startup_q, enable_event, config),
        name='movement',
        daemon=False,
    )
    p.start()
    return p


def start_camera(cam_to_inf_q, shutdown_event, ready_event, config):
    p = multiprocessing.Process(
        target=camera_process,
        args=(cam_to_inf_q, shutdown_event, ready_event, config),
        name='camera',
        daemon=False,
    )
    p.start()
    return p


def start_comm(inf_to_com_q, shutdown_event, ready_event, config):
    p = multiprocessing.Process(
        target=comm_process,
        args=(inf_to_com_q, shutdown_event, ready_event, config),
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

    config = DemoConfig()
    detection_event = DetectionEvent

    # Pre-flight: skip hardware checks in dry-run mode
    validate_files(config)

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
    proc_inference = proc_movement = proc_camera = proc_comm = None

    try:
        # 1. Inference first (programs FPGA — longest startup)
        print("[supervisor] Starting inference process …", flush=True)
        proc_inference = start_inference(
            cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, inf_ready_event, config, detection_event,
        )
        processes.append(proc_inference)
        if not inf_ready_event.wait(timeout=TIMEOUT_INFERENCE):
            print("[supervisor] ERROR: inference process did not become ready in time", flush=True)
            shutdown_event.set()
            shutdown_all(processes, shutdown_event)
            sys.exit(1)
        print("[supervisor] Inference ready.", flush=True)

        # 2. Movement
        print("[supervisor] Starting movement process …", flush=True)
        proc_movement = start_movement(
            inf_to_mov_q, shutdown_event, mov_ready_event, startup_q, enable_event, config
        )
        processes.append(proc_movement)
        if not mov_ready_event.wait(timeout=TIMEOUT_MOVEMENT):
            print("[supervisor] ERROR: movement process did not become ready in time", flush=True)
            shutdown_event.set()
            shutdown_all(processes, shutdown_event)
            sys.exit(1)
        # Read startup diagnostics
        try:
            info = startup_q.get(timeout=2.0)

            if not info.get("ok"):
                raise RuntimeError(info.get("error", "Movement initialization failed"))

            print(
                f"[supervisor] Movement ready — "
                f"VERSION={info.get('version', 0)} "
                f"ENABLE={info.get('enable')}", flush=True,
            )

        except Exception as exc:
            raise RuntimeError(f"Could not initialize movement: {exc}") from exc

        # 3. Camera
        print("[supervisor] Starting camera process …", flush=True)
        proc_camera = start_camera(
            cam_to_inf_q, shutdown_event, cam_ready_event, config,
        )
        processes.append(proc_camera)
        if not cam_ready_event.wait(timeout=TIMEOUT_CAMERA):
            print("[supervisor] ERROR: camera process did not become ready in time", flush=True)
            shutdown_event.set()
            shutdown_all(processes, shutdown_event)
            sys.exit(1)
        print("[supervisor] Camera ready.", flush=True)

        # 4. Comm
        print("[supervisor] Starting comm process …", flush=True)
        proc_comm = start_comm(
            inf_to_com_q, shutdown_event, com_ready_event, config,
        )
        processes.append(proc_comm)
        if not com_ready_event.wait(timeout=TIMEOUT_COMM):
            print("[supervisor] ERROR: comm process did not become ready in time", flush=True)
            shutdown_event.set()
            shutdown_all(processes, shutdown_event)
            sys.exit(1)
        print("[supervisor] Comm ready.", flush=True)

        # 5. Enable motors (movement process waits for this)
        print("[supervisor] Enabling motors …", flush=True)
        enable_event.set()

        print("[supervisor] All processes running. Press Ctrl-C to stop.", flush=True)

        # ----------------------------------------------------------------
        # Monitor loop — check liveness every 500 ms
        # ----------------------------------------------------------------
        essential = [p for p in processes if p is not None]

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
