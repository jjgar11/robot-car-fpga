"""Communication / logging process.

Logs DetectionEvents, prints runtime diagnostics and sends JSON telemetry to a
laptop over TCP without blocking inference or movement.
"""
from __future__ import annotations
import json
import signal
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
                status = "STABLE" if event.ready else "SEARCHING"

                print(
                    f"[comm] frame={event.frame_id:06d} | "
                    f"{status} | "
                    f"raw={event.raw_detection_count} "
                    f"frame_sign={event.grouped_detection_count} "
                    f"stable={event.stable_detection_count} | "
                    f"classifier_calls={event.classifier_calls}",
                    flush=True,
                )

                print(
                    f"[comm] timing | "
                    f"queue={event.queue_delay_ms:.1f} ms | "
                    f"detect={event.detect_ms:.1f} ms | "
                    f"filter={event.filter_ms:.2f} ms | "
                    f"FPGA_total={event.fpga_total_ms:.1f} ms | "
                    f"FPGA_avg={event.fpga_avg_ms:.2f} ms | "
                    f"non_FPGA={event.non_fpga_ms:.1f} ms | "
                    f"end_to_end={event.end_to_end_ms:.1f} ms",
                    flush=True,
                )

                if event.detections:
                    for index, detection in enumerate(event.detections, start=1):
                        cx, cy, size = detection["bbox"]
                        primary = " PRIMARY" if index == 1 else ""

                        print(
                            f"[comm]   #{index}{primary} | "
                            f"{detection['label']} | "
                            f"id={detection['class_id']} | "
                            f"crop_center=({cx},{cy}) size={size} | "
                            f"hits={detection['hits']}/{detection['window_size']} | "
                            f"score_sum={detection['score_sum']:.1f} | "
                            f"score_mean={detection['score_mean']:.1f} | "
                            f"accepted_crops={detection['raw_count']}",
                            flush=True,
                        )
                else:
                    print("[comm]   no stable traffic signs", flush=True)

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
