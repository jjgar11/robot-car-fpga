"""Communication / logging process.

Logs DetectionEvents, prints runtime diagnostics and sends JSON telemetry to a
laptop over TCP without blocking inference or movement.
"""
from __future__ import annotations
import json
import signal
import time
from queue import Empty


def comm_process(inf_to_com_q, shutdown_event, ready_event, config):
    """Child-process entry point for the comms/logging process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from tcp_comm import RobotServer
    server = RobotServer(port=config.tcp_port)
    try:
        server.start()

        with open(config.log_path, 'a', buffering=1) as log:
            ready_event.set()
            last_label, last_print = None, 0.0

            while not shutdown_event.is_set():
                event = None

                try:
                    event = inf_to_com_q.get(timeout=0.1)
                except Empty: 
                    pass

                if not server.has_client: 
                    server.try_accept()

                if event is None: 
                    continue

                payload = event.to_dict(include_scores=False)

                # One compact line per second, plus immediately when the stable sign changes
                label = event.primary_detection["label"] if event.detections else None
                now = time.monotonic()
                if label != last_label or now - last_print >= 1.0:
                    sign = f"{label} (id={event.primary_detection['class_id']})" if label else "SEARCHING"
                    print(f"[comm] frame={event.frame_id:06d} | {sign} | total={event.detect_ms:.1f} ms | fpga={event.fpga_total_ms:.1f} ms", flush=True)
                    last_label, last_print = label, now

                log.write(json.dumps(payload, separators=(",", ":")) + "\n")
                log.flush()

                if server.has_client:
                    server.try_send(payload)

    except Exception as exc:
        print(f"[comm] Exception: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        server.close()
        print("[comm] process exiting", flush=True)
