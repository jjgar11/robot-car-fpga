from collections import deque
import threading
import time

import cv2


class CameraPipe:
    """
    Captura continua de cámara en un thread separado.

    - Conserva solamente los frames más recientes.
    - No acumula una cola de frames antiguos.
    - Mide los FPS reales de captura.
    """

    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=2)
        self.condition = threading.Condition()
        self.running = False
        self.thread = None
        self.cap = None
        self.frame_id = 0
        self.invalid_frames = 0

        # FPS reales del productor de cámara
        self.capture_fps = 0.0
        self._fps_start = None
        self._fps_frames = 0

    # =========================================================
    # START
    # =========================================================

    def start(self):
        if self.running:
            return

        self.frames.clear()
        self.frame_id = 0
        self.invalid_frames = 0
        self.capture_fps = 0.0
        self._fps_frames = 0

        self.cap = cv2.VideoCapture(
            self.config.camera_device,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir "
                f"{self.config.camera_device}"
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *self.config.camera_fourcc
            ),
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.camera_width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.camera_height,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.config.camera_fps,
        )

        # NO usar CAP_PROP_BUFFERSIZE = 1.
        # Ya comprobamos que perjudicaba el rendimiento.

        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        fourcc = int(
            self.cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        fourcc_str = "".join(
            chr(
                (fourcc >> (8 * i))
                & 0xFF
            )
            for i in range(4)
        )

        print(
            f"Camera: "
            f"{width}x{height} "
            f"@ {fps:.1f} FPS "
            f"[{fourcc_str}]"
        )

        # Warmup
        for _ in range(5):
            ret, frame = self.cap.read()

            if ret and frame is not None:
                timestamp = time.perf_counter()

                self.frames.append(
                    (
                        self.frame_id,
                        timestamp,
                        frame,
                    )
                )

                self.frame_id += 1

        if not self.frames:
            self.cap.release()
            self.cap = None

            raise RuntimeError(
                "Camera opened but produced no frames"
            )

        self._fps_start = time.perf_counter()
        self._fps_frames = 0
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="camera-pipe",
            daemon=True,
        )

        self.thread.start()

    # =========================================================
    # CAMERA THREAD
    # =========================================================

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.invalid_frames += 1
                time.sleep(0.005)
                continue

            timestamp = time.perf_counter()

            # -------------------------------------------------
            # Medición real de FPS de cámara
            # -------------------------------------------------

            self._fps_frames += 1

            elapsed = (
                timestamp
                - self._fps_start
            )

            if elapsed >= 1.0:
                self.capture_fps = (
                    self._fps_frames
                    / elapsed
                )

                self._fps_frames = 0
                self._fps_start = timestamp

            # -------------------------------------------------
            # Frame más reciente
            # -------------------------------------------------

            item = (
                self.frame_id,
                timestamp,
                frame,
            )

            self.frame_id += 1

            with self.condition:
                self.frames.append(
                    item
                )

                self.condition.notify_all()

    # =========================================================
    # CONSUMER
    # =========================================================

    def get_latest_frame(
        self,
        after_id=-1,
        timeout_s=1.0,
    ):
        deadline = (
            time.perf_counter()
            + timeout_s
        )

        with self.condition:
            while self.running:
                if self.frames:
                    frame = self.frames[-1]

                    if frame[0] > after_id:
                        return frame

                remaining = (
                    deadline
                    - time.perf_counter()
                )

                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting "
                        "for camera frame"
                    )

                self.condition.wait(
                    timeout=min(
                        remaining,
                        0.1,
                    )
                )

        raise RuntimeError(
            "Camera pipe stopped"
        )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):
        if not self.running:
            return

        self.running = False

        with self.condition:
            self.condition.notify_all()

        if self.thread is not None:
            self.thread.join(
                timeout=2.0
            )

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.thread = None