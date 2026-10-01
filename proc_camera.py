"""Camera capture process.

Runs as an independent child process. Captures frames from a V4L2 camera
and pushes them to a maxsize=1 queue, always keeping only the freshest frame.
"""

import signal
import time
import cv2
import numpy as np
from queue import Empty, Full

def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.transpose(frame_rgb, (2, 0, 1))
    frame = np.asarray(frame_rgb)[np.newaxis, ...]
    return frame

def camera_process(cam_to_inf_q, shutdown_event, ready_event, config):
    """Child-process entry point for camera capture."""
    # Ignore SIGINT — supervisor handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    cap = None
    try:
        if isinstance(config.camera_device, str):
            cap = cv2.VideoCapture(config.camera_device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(config.camera_device), cv2.CAP_V4L2)

        # Configure capture parameters
        fourcc = cv2.VideoWriter_fourcc(*config.camera_fourcc)
        cap.set(cv2.CAP_PROP_FOURCC,       fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        cap.set(cv2.CAP_PROP_FPS,          config.camera_fps)

        if not cap.isOpened():
            print("[camera] Failed to open camera device", flush=True)
            return

        # Warmup: discard initial frames (exposure settling)
        for _ in range(5):
            cap.read()

        ready_event.set()

        frame_id = 0
        while not shutdown_event.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = preprocess(frame_bgr)
            item = (frame_id, time.monotonic(), frame)
            frame_id += 1

            # Drop stale frame from queue, then push new one (non-blocking)
            try:
                cam_to_inf_q.put_nowait(item)
            except Full:
                try:
                    cam_to_inf_q.get_nowait()
                except Empty:
                    pass
                try:
                    cam_to_inf_q.put_nowait(item)
                except Full:
                    pass

    except Exception as exc:
        print(f"[camera] Exception: {exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        # Cancel the queue feeder thread so the process exits immediately even
        # if the inference reader is already gone and the pipe is full.
        cam_to_inf_q.cancel_join_thread()
        if cap is not None:
            cap.release()
        print("[camera] process exiting", flush=True)
