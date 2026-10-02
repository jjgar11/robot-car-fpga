"""Inference process.

Loads the fixed-crop FINN pipeline, performs frame score consensus
and temporal label voting, then publishes DetectionEvent
objects to movement and communications.
"""

import signal
import time
from collections import deque
from queue import Empty, Full


from inference_pipe import InferencePipe
from labels import LABELS_EN
from motion_config import MAX_EVENT_AGE_S


def inference_process(cam_to_inf_q, inf_to_mov_q, inf_to_com_q, shutdown_event, ready_event, config, detection_event):
    """Child-process entry point for inference."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        pipe = InferencePipe(config, clear_pynq_state=True)

        temporal = TemporalDetectionFilter(
            window_size = config.score_window,
            min_hits = config.min_hits,
        )

        ready_event.set()
        last_frame_id = -1

        while not shutdown_event.is_set():
            item = cam_to_inf_q.get(last_frame_id, timeout=0.5)
            if item is None:
                continue
            frame_id, frame_timestamp, frame = item
            last_frame_id = frame_id

            inference_start = time.monotonic()
            age = inference_start - frame_timestamp

            if age > MAX_EVENT_AGE_S:
                continue

            if shutdown_event.is_set():
                break

            queue_delay_ms = age * 1000.0

            result = pipe.process_frame(frame)

            stable_detections = temporal.update(result["frame_detection"])

            event_time = time.monotonic()

            event = detection_event(
                detections=stable_detections,
                timestamp=event_time,
                frame_timestamp=frame_timestamp,
                frame_id=frame_id,
                ready=bool(stable_detections),
                detect_ms=(event_time - inference_start) * 1000.0,  # frame processing incl. temporal filter
                fpga_total_ms=result["fpga_ms"],
                classifier_calls=result["classifier_calls"],
                queue_delay_ms=queue_delay_ms,
                end_to_end_ms=(event_time - frame_timestamp) * 1000.0,
                raw_detection_count=len(result["detections"]),
                grouped_detection_count=int(result["frame_class_id"] != 0),
            )

            # Fan out the same immutable-by-convention event to both consumers.
            # If a queue is full, discard its stale event and keep the newest.
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
        import traceback
        traceback.print_exc()

    finally:
        inf_to_mov_q.cancel_join_thread()
        inf_to_com_q.cancel_join_thread()
        print("[inference] process exiting", flush=True)


class TemporalDetectionFilter:
    """Vote only on frame labels; keep latest metadata for the winning class."""
    def __init__(self, window_size=5, min_hits=2):
        if not 1 <= min_hits <= window_size: raise ValueError("Require 1 <= min_hits <= window_size")
        self.window_size, self.min_hits = window_size, min_hits
        self.history, self.latest = deque(maxlen=window_size), {}

    def reset(self):
        self.history.clear()
        self.latest.clear()

    def update(self, detections):

        class_id = int(detections["class_id"]) if detections else 0
        self.history.append(class_id)

        if class_id:
            self.latest[class_id] = detections.copy()
        self.latest = {c: d for c, d in self.latest.items() if c in self.history}

        counts = {c: self.history.count(c) for c in self.latest}
        valid = [c for c, hits in counts.items() if hits >= self.min_hits]

        if not valid:
            return []
        reverse = list(reversed(self.history))
        winner = max(valid, key=lambda c: (counts[c], -reverse.index(c)))
        result = self.latest[winner].copy()

        score_sum = float(result["scores"][winner])
        result.update(label=LABELS_EN[winner], score_sum=score_sum, score_mean=score_sum / result["raw_count"], area=result["bbox"][2] ** 2, hits=counts[winner], window_size=self.window_size)
        return [result]
