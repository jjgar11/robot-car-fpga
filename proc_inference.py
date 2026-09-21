"""Inference process.

Loads the FINN overlay, runs INT4-packed DMA inference, applies a temporal
filter, and fans the resulting DetectionEvent out to both downstream queues.
"""

import signal
import sys
import time
from queue import Empty, Full
from motion_config import MAX_EVENT_AGE_S

def inference_process(cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, ready_event, config):
    """Child-process entry point for inference."""
    # Ignore SIGINT — supervisor handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from inference_pipe import InferencePipe
        from temporal_filter import TemporalScoreFilter
        from ipc_types import DetectionEvent

        # Load overlay and build inference pipeline
        pipe     = InferencePipe(config, clear_pynq_state=True)
        temporal = TemporalScoreFilter(config.score_window)

        ready_event.set()

        while not shutdown_event.is_set():
            # Fetch frame (block briefly so we don't spin)
            try:
                frame_id, ts, frame = cam_to_inf_q.get(timeout=0.5)
            except Empty:
                continue

            # Discard frames that are too old to be useful
            age = time.monotonic() - ts
            if age > MAX_EVENT_AGE_S:
                continue

            # Check shutdown before starting a (potentially blocking) DMA transfer
            if shutdown_event.is_set():
                break
            # Run inference
            result = pipe.process_frame(frame)

            # Apply temporal filter
            stable = temporal.update(result['scores'])

            event = DetectionEvent(
                class_id  = stable.get('class_id') or 0,
                label     = stable.get('label', ''),
                score_sum = stable.get('score_sum') or 0.0,
                timestamp = time.monotonic(),
                frame_id  = frame_id,
                ready     = stable.get('ready', False),
                fpga_ms   = result["fpga_ms"],
            )

            # Fan out to both consumers (drop silently if queue is full)
            for q in (inf_to_mov_q, inf_to_com_q):
                try:
                    q.put_nowait(event)
                except Full:
                    try:
                        q.get_nowait()
                    except Empty:
                        pass

                    try:
                        q.put_nowait(event)
                    except Full:
                        pass

    except Exception as exc:
        print(f"[inference] Exception: {exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        # Cancel feeder threads so the process exits even if downstream readers
        # have already gone and the pipes are full.
        inf_to_mov_q.cancel_join_thread()
        inf_to_com_q.cancel_join_thread()
        print("[inference] process exiting", flush=True)
