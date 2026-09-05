"""Inference process.

Loads the FINN overlay, runs INT4-packed DMA inference, applies a temporal
filter, and fans the resulting DetectionEvent out to both downstream queues.
"""

import signal
import sys
import time
from queue import Empty, Full

MAX_EVENT_AGE_S = 2.0   # discard frames older than this before inference

DEMO_DIR = '/home/xilinx/nano_gtsrb_robot_demo'


def inference_process(
    cam_to_inf_q,
    inf_to_mov_q,
    inf_to_com_q,
    shutdown_event,
    ready_event,
    config,
):
    """Child-process entry point for FINN inference."""
    # Ignore SIGINT — supervisor handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Insert paths so local modules resolve after fork
    sys.path.insert(0, DEMO_DIR)

    try:
        from inference_pipe import InferencePipe
        from temporal_filter import TemporalScoreFilter
        from ipc_types import DetectionEvent

        # Build an InferencePipe-compatible config.
        # InferencePipe expects the existing DemoConfig dataclass from config.py.
        # If the passed-in config is already compatible (has base_dir), use it
        # directly; otherwise build a DemoConfig using its defaults and then
        # overwrite the fields that matter for inference/camera from our config.
        if not hasattr(config, 'base_dir'):
            from config import DemoConfig
            import dataclasses
            # Collect only the fields DemoConfig actually declares
            dc_fields = {f.name for f in dataclasses.fields(DemoConfig)}
            # Map SupervisorConfig field names → DemoConfig field names
            field_map = {
                'score_window': 'score_window',
                'target_fps':   'target_fps',
            }
            overrides = {}
            for src, dst in field_map.items():
                if dst in dc_fields and hasattr(config, src):
                    overrides[dst] = getattr(config, src)
            inf_config = DemoConfig(**overrides)
        else:
            inf_config = config

        # Load overlay and build inference pipeline
        pipe     = InferencePipe(inf_config, clear_pynq_state=True)
        temporal = TemporalScoreFilter(inf_config.score_window)

        ready_event.set()

        while not shutdown_event.is_set():
            # Fetch frame (block briefly so we don't spin)
            try:
                frame_id, ts, frame = cam_to_inf_q.get(timeout=0.5)
            except (Empty, Exception):
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
