"""Camera capture process.

Runs as an independent child process. Captures frames from a V4L2 camera
and pushes them to a maxsize=1 queue, always keeping only the freshest frame.
"""

import signal
import time


def camera_process(cam_to_inf_q, shutdown_event, ready_event, config):
    """Child-process entry point for camera capture."""
    # Ignore SIGINT — supervisor handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import cv2  # import after fork

    cap = None
    try:
        # Support both DemoConfig field names (camera_*) and SupervisorConfig names (frame_*)
        camera_device = getattr(config, 'camera_device', '/dev/video0')
        frame_width   = getattr(config, 'camera_width',  getattr(config, 'frame_width',  320))
        frame_height  = getattr(config, 'camera_height', getattr(config, 'frame_height', 240))
        target_fps    = getattr(config, 'camera_fps',    getattr(config, 'target_fps',   20.0))
        fourcc_str    = getattr(config, 'camera_fourcc', 'MJPG')

        if isinstance(camera_device, str):
            cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(camera_device), cv2.CAP_V4L2)

        # Configure capture parameters
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC,       fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS,          target_fps)

        if not cap.isOpened():
            print("[camera] Failed to open camera device", flush=True)
            return

        # Warmup: discard initial frames (exposure settling)
        for _ in range(5):
            cap.read()

        ready_event.set()

        frame_id = 0
        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            item = (frame_id, time.monotonic(), frame)
            frame_id += 1

            # Drop stale frame from queue, then push new one (non-blocking)
            while not cam_to_inf_q.empty():
                try:
                    cam_to_inf_q.get_nowait()
                except Exception:
                    break
            try:
                cam_to_inf_q.put_nowait(item)
            except Exception:
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
