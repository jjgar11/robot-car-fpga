"""Communication / logging process.

Receives DetectionEvents from the inference process:
  - Writes structured JSON-lines records to disk (always).
  - Encodes and sends (class_id, confidence) to a laptop via TCP (best-effort).

TCP server is non-blocking: if no client is connected, detections are only
logged to disk.  If the client disconnects, the server waits for a new one
transparently.  Neither inference nor movement is blocked.
"""
from __future__ import annotations

import json
import signal
import sys
import time

DEFAULT_LOG_PATH = '/tmp/robot_detections.jsonl'
DEFAULT_TCP_PORT = 50007


def comm_process(
    inf_to_com_q,
    shutdown_event,
    ready_event,
    log_path: str = None,
    tcp_port: int = DEFAULT_TCP_PORT,
):
    """Child-process entry point for the comms/logging process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    sys.path.insert(0, '/home/xilinx/nano_gtsrb_robot_demo')
    from tcp_comm import RobotServer

    server = RobotServer(port=tcp_port)

    try:
        server.start()

        with open(log_path, 'a', buffering=1) as log:
            ready_event.set()

            while not shutdown_event.is_set():
                # 1. Try to get a detection (short timeout to stay responsive)
                event = None
                try:
                    event = inf_to_com_q.get(timeout=0.1)
                except Exception:
                    pass

                # 2. Try to accept a new TCP client (non-blocking, 0.1 s window)
                if not server.has_client:
                    server.try_accept()

                if event is None:
                    continue

                # 3. Human-readable console summary
                print(
                    f"[comm] frame={event.frame_id:06d} "
                    f"class={event.class_id:02d} "
                    f"label='{event.label}' "
                    f"score={event.score_sum:.1f} "
                    f"ready={event.ready}",
                    flush=True,
                )

                # 4. JSON Lines record
                log.write(json.dumps({
                    'ts':       event.timestamp,
                    'frame_id': event.frame_id,
                    'class_id': event.class_id,
                    'label':    event.label,
                    'score':    event.score_sum,
                    'ready':    event.ready,
                }) + '\n')
                log.flush()

                # 5. TCP send — only when temporal filter is stable
                if event.ready:
                    confidence = max(0, int(event.score_sum))
                    server.try_send(event.class_id, confidence)

    except Exception as exc:
        print(f"[comm] Exception: {exc}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        server.close()
        print("[comm] process exiting", flush=True)
