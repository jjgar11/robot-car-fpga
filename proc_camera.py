"""Camera capture process.

Runs as an independent child process. Captures frames from a V4L2 camera
and publishes them to a SharedFrame, always keeping only the freshest frame.
"""

import multiprocessing
import signal
import time
import cv2
import numpy as np

def preprocess(frame_bgr, out):
    """BGR uint8 -> RGB float32 0-1, written HWC-contiguous into out."""
    np.divide(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), np.float32(255.0), out=out, dtype=np.float32)

class SharedFrame:
    """Latest-frame triple buffer in shared memory: the producer writes a slot that is
    neither published nor being read; publish/get swap slot ownership under one lock."""
    def __init__(self, height, width):
        self.shape, self.size = (height, width, 3), height * width * 3
        self.buf = multiprocessing.RawArray('f', 3 * self.size)
        self.meta = multiprocessing.RawArray('d', (-1.0, 0.0, 0.0, -1.0))  # frame_id, timestamp, published slot, reading slot
        self.cond = multiprocessing.Condition()

    def slot(self, index):
        return np.frombuffer(self.buf, np.float32, self.size, index * self.size * 4).reshape(self.shape)

    def free_slot(self):
        with self.cond:
            busy = (self.meta[2], self.meta[3])
        return next(i for i in range(3) if i not in busy)

    def publish(self, index, frame_id, timestamp):
        with self.cond:
            self.meta[:3] = (frame_id, timestamp, index)
            self.cond.notify_all()

    def get(self, last_frame_id, timeout):
        """Newest frame newer than last_frame_id as (frame_id, timestamp, (1,3,H,W) view), or None.
        The view stays valid until the next get()."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.meta[0] > last_frame_id, timeout):
                return None
            frame_id, timestamp, index = self.meta[:3]
            self.meta[3] = index
        return int(frame_id), timestamp, np.transpose(self.slot(int(index)), (2, 0, 1))[np.newaxis]

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

            # Timestamp at read return, so camera preprocessing counts as queue delay
            timestamp = time.monotonic()
            index = cam_to_inf_q.free_slot()
            preprocess(frame_bgr, cam_to_inf_q.slot(index))
            cam_to_inf_q.publish(index, frame_id, timestamp)
            frame_id += 1

    except Exception as exc:
        print(f"[camera] Exception: {exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        print("[camera] process exiting", flush=True)
