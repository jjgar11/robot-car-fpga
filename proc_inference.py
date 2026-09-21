"""Inference process.

Loads the FINN overlay, runs INT4-packed DMA inference, applies a temporal
filter, and fans the resulting DetectionEvent out to both downstream queues.
"""

import signal
import time
from queue import Empty, Full
from motion_config import MAX_EVENT_AGE_S
from inference_pipe import InferencePipe
from collections import deque
import numpy as np
from labels import LABELS_EN

def inference_process(cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, ready_event, config):
    """Child-process entry point for inference."""
    # Ignore SIGINT — supervisor handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
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

            event = config.DetectionEvent(
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


class TemporalScoreFilter:
    """
    Sliding temporal window.

    Decision = argmax(sum(scores of last N processed frames))
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def reset(self):
        self.history.clear()

    @property
    def ready(self):
        return (len(self.history) == self.window_size)

    def update(self, scores):
        self.history.append(np.asarray(scores).copy())

        if not self.ready:
            return {
                "ready": False,
                "window_fill": len(self.history),
                "window_size": (self.window_size),
                "class_id": None,
                "label": "Warming up",
                "score_sum": None,
            }

        accumulated_scores = np.sum(np.stack(self.history), axis=0)

        class_id = int(np.argmax(accumulated_scores))

        return {
            "ready": True,
            "window_fill": len(self.history),
            "window_size": (self.window_size),
            "class_id": class_id,
            "label": (LABELS_EN[class_id]),
            "score_sum": float(accumulated_scores[class_id]),
        }
