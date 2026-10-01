from dataclasses import dataclass, asdict
import threading
import time

import cv2
import numpy as np

from camera_pipe import CameraPipe
from inference_pipe import InferencePipe
from proc_inference import TemporalScoreFilter
from integration_pipes import (
    MovementPipe,
    CommunicationPipe,
)


@dataclass
class PipelineSnapshot:
    frame_id: int

    # Stable temporal prediction
    label: str
    class_id: int | None
    score_sum: float | None

    # Raw frame prediction
    raw_label: str
    raw_class_id: int
    raw_score: float

    # Metrics
    fps: float
    camera_fps: float
    capture_age_ms: float
    preprocess_ms: float
    fpga_ms: float
    loop_ms: float

    # Temporal filter state
    window_ready: bool
    window_fill: int

    def to_dict(self):
        return asdict(self)


class PipeManager:
    """
    Main orchestrator.

    Current four-pipe architecture:

        CameraPipe
            ↓
        preprocessing
            ↓
        InferencePipe
            ↓
        TemporalScoreFilter
            ↓
        ├─ MovementPipe
        ├─ CommunicationPipe
        └─ optional UI callback

    Camera capture runs in its own thread.
    FINN/PYNQ access remains owned by the manager inference flow.
    """

    def __init__(
        self,
        config,
        camera=None,
        inference=None,
        movement=None,
        communication=None,
        clear_pynq_state=True,
    ):
        self.config = config

        self.camera = (
            camera
            if camera is not None
            else CameraPipe(config)
        )

        self.inference = (
            inference
            if inference is not None
            else InferencePipe(
                config,
                clear_pynq_state=(
                    clear_pynq_state
                ),
            )
        )

        self.movement = (
            movement
            if movement is not None
            else MovementPipe()
        )

        self.communication = (
            communication
            if communication is not None
            else CommunicationPipe()
        )

        self.temporal = (
            TemporalScoreFilter(
                config.score_window
            )
        )

        self.running = False
        self.stop_event = (
            threading.Event()
        )

    def _preprocess(self, frame_bgr):
        t0 = time.perf_counter()

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        h, w = frame_rgb.shape[:2]

        side = int(
            min(h, w)
            * self.config.crop_fraction
        )

        side = max(
            side,
            self.config.input_size,
        )

        x0 = (w - side) // 2

        max_shift = (
            h - side
        ) // 2

        y0 = (
            (h - side) // 2
            + int(
                max_shift
                * self.config.tilt_fraction
            )
        )

        y0 = max(
            0,
            min(
                y0,
                h - side,
            ),
        )

        crop = frame_rgb[
            y0:y0 + side,
            x0:x0 + side,
        ]

        frame64 = cv2.resize(
            crop,
            (
                self.config.input_size,
                self.config.input_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        x_float = (
            frame64.astype(np.float32)
            / 255.0
        )

        x_int4 = np.clip(
            np.floor(
                x_float * 8.0
                + 0.5
            ),
            -8,
            7,
        ).astype(np.float32)

        preprocess_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        return (
            x_int4,
            preprocess_ms,
        )

    def start(self):
        if self.running:
            return

        self.stop_event.clear()
        self.temporal.reset()

        self.movement.start()
        self.communication.start()
        self.camera.start()

        self.running = True

    def stop(self):
        if not self.running:
            return

        self.stop_event.set()
        self.running = False

        self.camera.stop()
        self.movement.stop()
        self.communication.stop()

    def run(
        self,
        on_update=None,
        max_frames=None,
    ):
        self.start()

        last_frame_id = -1

        fps_start = time.perf_counter()
        fps_frames = 0
        measured_fps = 0.0

        processed_frames = 0

        if self.config.target_fps > 0:
            target_period = (
                1.0
                / self.config.target_fps
            )
        else:
            target_period = 0.0

        try:
            while (
                not self.stop_event.is_set()
            ):
                loop_start = (
                    time.perf_counter()
                )

                (
                    frame_id,
                    capture_ts,
                    frame_bgr,
                ) = (
                    self.camera.get_latest_frame(
                        after_id=last_frame_id,
                        timeout_s=1.0,
                    )
                )

                last_frame_id = frame_id

                capture_age_ms = (
                    time.perf_counter()
                    - capture_ts
                ) * 1000.0

                # Real producer-side camera FPS
                camera_fps = (
                    self.camera.capture_fps
                )

                frame64, preprocess_ms = (
                    self._preprocess(
                        frame_bgr
                    )
                )

                result = (
                    self.inference.process_frame(
                        frame64
                    )
                )

                result[
                    "preprocess_ms"
                ] = preprocess_ms

                stable = (
                    self.temporal.update(
                        result["scores"]
                    )
                )

                fps_frames += 1
                processed_frames += 1

                now = time.perf_counter()
                fps_elapsed = (
                    now - fps_start
                )

                if fps_elapsed >= 1.0:
                    measured_fps = (
                        fps_frames
                        / fps_elapsed
                    )

                    fps_frames = 0
                    fps_start = now

                loop_ms = (
                    time.perf_counter()
                    - loop_start
                ) * 1000.0

                snapshot = (
                    PipelineSnapshot(
                        frame_id=frame_id,

                        label=stable[
                            "label"
                        ],
                        class_id=stable[
                            "class_id"
                        ],
                        score_sum=stable[
                            "score_sum"
                        ],

                        raw_label=result[
                            "raw_label"
                        ],
                        raw_class_id=result[
                            "raw_class_id"
                        ],
                        raw_score=result[
                            "raw_score"
                        ],

                        fps=measured_fps,
                        camera_fps=(
                            camera_fps
                        ),
                        capture_age_ms=(
                            capture_age_ms
                        ),
                        preprocess_ms=result[
                            "preprocess_ms"
                        ],
                        fpga_ms=result[
                            "fpga_ms"
                        ],
                        loop_ms=loop_ms,

                        window_ready=stable[
                            "ready"
                        ],
                        window_fill=stable[
                            "window_fill"
                        ],
                    )
                )

                self.movement.on_prediction(
                    snapshot
                )

                self.communication.publish(
                    snapshot
                )

                if on_update is not None:
                    on_update(
                        snapshot,
                        frame_bgr,
                    )

                work_time = (
                    time.perf_counter()
                    - loop_start
                )

                remaining = (
                    target_period
                    - work_time
                )

                if remaining > 0:
                    time.sleep(
                        remaining
                    )

                if (
                    max_frames is not None
                    and processed_frames
                    >= max_frames
                ):
                    break

        except KeyboardInterrupt:
            print("\nStopped")

        finally:
            self.stop()