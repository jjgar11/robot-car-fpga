"""Inference process.

Loads the detector + FINN classifier pipeline, performs same-frame spatial
grouping and temporal score accumulation, then publishes DetectionEvent
objects to movement and communications.
"""

import math
import signal
import time
from collections import deque
from queue import Empty, Full

import numpy as np

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
            stable_ratio = config.stable_ratio,
            max_misses = config.max_misses,
            iou_threshold = config.iou_threshold,
        )

        ready_event.set()
        last_frame_id = None

        while not shutdown_event.is_set():
            try:
                frame_id, frame_timestamp, frame = cam_to_inf_q.get(timeout=0.5)
            except Empty:
                continue

            if frame_id == last_frame_id:
                continue
            last_frame_id = frame_id

            inference_start = time.monotonic()
            age = inference_start - frame_timestamp

            if age > MAX_EVENT_AGE_S:
                continue

            if shutdown_event.is_set():
                break

            queue_delay_ms = age * 1000.0

            result = pipe.process_frame(frame)

            filter_start = time.perf_counter()
            stable_detections = temporal.update(result["detections"])
            filter_ms = (time.perf_counter() - filter_start) * 1000.0

            event_time = time.monotonic()

            classifier_calls = result["classifier_calls"]
            fpga_total_ms = result["fpga_ms"]
            fpga_avg_ms = (fpga_total_ms / classifier_calls if classifier_calls else 0.0)

            event = detection_event(
                detections=stable_detections,
                timestamp=event_time,
                frame_timestamp=frame_timestamp,
                frame_id=frame_id,
                ready=bool(stable_detections),
                detect_ms=result["detect_ms"],
                filter_ms=filter_ms,
                fpga_total_ms=fpga_total_ms,
                fpga_avg_ms=fpga_avg_ms,
                classifier_calls=classifier_calls,
                queue_delay_ms=queue_delay_ms,
                end_to_end_ms=(event_time - frame_timestamp) * 1000.0,
                raw_detection_count=len(result["detections"]),
                grouped_detection_count=temporal.last_grouped_count,
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


def bbox_iou(a, b):
    """IoU for square boxes represented as (center_x, center_y, size)."""

    acx, acy, asize = map(float, a)
    bcx, bcy, bsize = map(float, b)

    ahalf = asize / 2.0
    bhalf = bsize / 2.0

    ax1, ay1 = acx - ahalf, acy - ahalf
    ax2, ay2 = acx + ahalf, acy + ahalf
    bx1, by1 = bcx - bhalf, bcy - bhalf
    bx2, by2 = bcx + bhalf, bcy + bhalf

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih
    union = asize * asize + bsize * bsize - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


class TemporalDetectionFilter:
    """Track spatial detections and accumulate their 44-class scores over time."""

    def __init__(self, window_size=5, stable_ratio=0.60, max_misses=3, iou_threshold=0.30):
        self.window_size = int(window_size)
        self.stable_ratio = float(stable_ratio)
        self.min_hits = math.ceil(self.window_size * self.stable_ratio)
        self.max_misses = int(max_misses)
        self.iou_threshold = float(iou_threshold)

        self.tracks = {}
        self.next_track_id = 0
        self.tick = 0
        self.last_grouped_count = 0

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 0
        self.tick = 0
        self.last_grouped_count = 0

    def _group_same_frame(self, detections):
        """Merge overlapping raw boxes belonging to the same physical sign."""

        groups = []

        # Larger boxes are considered first so the representative bbox of a
        # group naturally follows the most prominent candidate.
        ordered = sorted(detections, key=lambda d: d["bbox"][2], reverse=True)

        for detection in ordered:
            bbox = detection["bbox"]
            scores = np.asarray(detection["scores"], dtype=np.float32)

            best_group = None
            best_iou = 0.0

            for group in groups:
                iou = bbox_iou(bbox, group["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_group = group

            if (best_group is not None and best_iou >= self.iou_threshold):
                best_group["scores"].append(scores.copy())
                best_group["raw_count"] += 1

                if bbox[2] > best_group["bbox"][2]:
                    best_group["bbox"] = bbox
            else:
                groups.append({
                    "bbox": bbox,
                    "scores": [scores.copy()],
                    "raw_count": 1,
                })

        frame_detections = []

        for group in groups:
            stacked = np.stack(group["scores"], axis=0)

            # Average multiple raw boxes within the same frame so a frame with
            # more overlapping candidate boxes does not get extra temporal
            # weight merely because it produced more crops.
            frame_scores = np.mean(stacked, axis=0)

            frame_detections.append({
                "bbox": group["bbox"],
                "scores": frame_scores.copy(),
                "raw_count": group["raw_count"],
            })

        self.last_grouped_count = len(frame_detections)
        return frame_detections

    def _new_track(self, detection):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "bbox": detection["bbox"],
            "history": deque([(self.tick, detection["scores"].copy())]),
            "misses": 0,
            "last_seen_tick": self.tick,
            "raw_count": detection["raw_count"],
        }

        return track_id

    def _prune_history(self, track):
        """Keep only observations whose frame tick is inside the window."""

        oldest_tick = self.tick - self.window_size + 1

        while (track["history"] and track["history"][0][0] < oldest_tick):
            track["history"].popleft()

    def _match_tracks(self, frame_detections):
        """Greedy one-to-one association using descending IoU."""

        existing_track_ids = list(self.tracks.keys())
        candidates = []

        for detection_index, detection in enumerate(frame_detections):
            for track_id in existing_track_ids:
                track = self.tracks[track_id]
                iou = bbox_iou(detection["bbox"], track["bbox"])

                if iou >= self.iou_threshold:
                    candidates.append((iou, detection_index, track_id))

        candidates.sort(key=lambda item: item[0], reverse=True)

        used_detections = set()
        matched_tracks = set()

        for _, detection_index, track_id in candidates:
            if (detection_index in used_detections or track_id in matched_tracks):
                continue

            detection = frame_detections[detection_index]
            track = self.tracks[track_id]

            track["bbox"] = detection["bbox"]
            track["history"].append((self.tick, detection["scores"].copy()))
            track["misses"] = 0
            track["last_seen_tick"] = self.tick
            track["raw_count"] = detection["raw_count"]

            used_detections.add(detection_index)
            matched_tracks.add(track_id)

        # Any current observation that did not match an old track starts a
        # new independent temporal track.
        for detection_index, detection in enumerate(frame_detections):
            if detection_index in used_detections:
                continue

            track_id = self._new_track(detection)
            matched_tracks.add(track_id)

        # Existing tracks that were not observed this frame are aged.
        for track_id in existing_track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]["misses"] += 1

        # Remove stale tracks after the configured tolerance.
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["misses"] > self.max_misses:
                del self.tracks[track_id]

        return matched_tracks

    def update(self, detections):
        """Return stable current detections sorted largest -> smallest."""

        self.tick += 1

        frame_detections = self._group_same_frame(detections)
        seen_tracks = self._match_tracks(frame_detections)

        for track in self.tracks.values():
            self._prune_history(track)

        stable_detections = []

        # Only report tracks actually visible in the current frame.
        for track_id in seen_tracks:
            if track_id not in self.tracks:
                continue

            track = self.tracks[track_id]
            history = track["history"]
            hits = len(history)

            if hits < self.min_hits:
                continue

            accumulated_scores = np.sum(
                np.stack([scores for _, scores in history]),
                axis=0,
            )

            class_id = int(np.argmax(accumulated_scores))

            score_sum = float(accumulated_scores[class_id])
            _, _, size = track["bbox"]

            stable_detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "label": LABELS_EN[class_id],
                "score_sum": score_sum,
                "score_mean": score_sum / hits,
                "scores": accumulated_scores.copy(),
                "bbox": track["bbox"],
                "area": int(size * size),
                "hits": hits,
                "window_size": self.window_size,
                "stable_ratio": self.stable_ratio,
                "raw_count": track["raw_count"],
            })

        stable_detections.sort(key=lambda detection: detection["area"], reverse=True)

        return stable_detections
