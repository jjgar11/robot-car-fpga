import sys
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from labels import LABELS_EN

class InferencePipe:
    """
    FINN inference for one real frame at a time.

    Runtime batch = 1.
    """

    BATCH_SIZE = 1

    def __init__(self, config, clear_pynq_state=True):
        self.config = config
        self.clear_pynq_state = clear_pynq_state
        self.accel = None
        self._pool = ThreadPoolExecutor(4)  # cv2.resize releases the GIL

        self._load_overlay()

    def _load_overlay(self):
        sys.path.insert(0, str(self.config.driver_dir))

        from driver import FINNExampleOverlay, io_shape_dict
        from pynq.pl_server import global_state

        if self.clear_pynq_state:
            global_state.clear_global_state()

        self.accel = FINNExampleOverlay(
            bitfile_name=str(self.config.bitfile),
            platform="zynq-iodma",
            io_shape_dict=io_shape_dict,
            batch_size=self.BATCH_SIZE,
            fclk_mhz=self.config.fclk_mhz,
            download=True,
            runtime_weight_dir=self.config.runtime_weight_dir,
        )

        print("[inference] FINN overlay loaded", flush=True)
        print(f"[inference] bitfile={self.accel.bitfile_name}", flush=True)
        print(f"[inference] runtime_batch={self.BATCH_SIZE}", flush=True)

    @staticmethod
    def quantize_pack_int4(crops):
        """(N,64,64,3) float crops -> floor(x*8+0.5) clamped to INT4, packed as N FINN (1,64,64,1,2) buffers."""
        x = np.clip(np.floor(crops * 8.0 + 0.5), -8, 7)
        u = (x.astype(np.int8).view(np.uint8) & 0xF)[:, None, :, :, None, :]

        out = np.empty(u.shape[:-1] + (2,), dtype=np.uint8)

        out[..., 0] = (u[..., 0] | (u[..., 1] << 4))

        out[..., 1] = u[..., 2]

        return out

    def label_func(self, packed):
        """Classify one packed INT4 crop and preserve all 44 FINN scores."""

        np.copyto(self.accel.ibuf_packed_device[0], packed)

        self.accel.ibuf_packed_device[0].flush()

        idma = self.accel.idma[0]
        odma = self.accel.odma[0]

        if not (idma.read(0x00) & 0x4):
            raise RuntimeError("IDMA not idle")

        if not (odma.read(0x00) & 0x4):
            raise RuntimeError("ODMA not idle")

        fpga_start = time.perf_counter()

        self.accel.execute_on_buffers(asynch=True, batch_size=1)

        while True:
            status = odma.read(0x00)

            if status & 0x2:
                fpga_end = time.perf_counter()
                break

            if (time.perf_counter() - fpga_start > self.config.fpga_timeout_s):
                raise RuntimeError("FPGA timeout")

            time.sleep(0.0001)

        fpga_ms = (fpga_end - fpga_start) * 1000.0

        self._frame_fpga_ms += fpga_ms
        self._frame_classifier_calls += 1

        self.accel.copy_output_data_from_device(self.accel.obuf_packed[0], ind=0)

        packed_output = np.asarray(self.accel.obuf_packed[0])
        if packed_output.dtype != np.uint8 or packed_output.shape != (1, 44, 2):
            raise ValueError(f"Unexpected packed output: {packed_output.shape}, {packed_output.dtype}")
        obuf_folded = np.ascontiguousarray(packed_output).view("<i2").reshape(1, 44, 1).astype(np.float32)
        y = self.accel.unfold_output(obuf_folded, ind=0).copy()

        scores = y[0]

        class_id = int(np.argmax(scores))

        return {
            "class_id": class_id,
            "scores": scores.copy()
        }

    def process_frame(self, x):
        """Classify fixed crops and sum scores from crops whose label is not 0."""

        self._frame_fpga_ms = 0.0
        self._frame_classifier_calls = 0

        expected = (1, 3, self.config.camera_height, self.config.camera_width)

        if x.shape != expected:
            raise ValueError(
                f"Unexpected input shape: {x.shape}; expected {expected}"
            )

        height, width = x.shape[-2:]
        image = np.ascontiguousarray(np.transpose(x[0], (1, 2, 0)))
        detections, boxes = [], []

        for fraction in self.config.crop_sizes:
            size = max(1, min(min(height, width), round(min(height, width) * fraction)))
    
            for px, py in self.config.crop_positions:
                left = max(0, min(width - size, round(px * width - size / 2)))
                top = max(0, min(height - size, round(py * height - size / 2)))
                boxes.append((left, top, size))

        resize = lambda b: cv2.resize(image[b[1]:b[1] + b[2], b[0]:b[0] + b[2]], (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_AREA)
        packed = self.quantize_pack_int4(np.stack(list(self._pool.map(resize, boxes))))

        for (left, top, size), buf in zip(boxes, packed):
            detection = self.label_func(buf)

            if detection["class_id"] == 0: 
                continue

            detection["bbox"] = (left + size / 2, top + size / 2, size)
            detections.append(detection)

        frame_detection = None

        if detections:
            scores = np.sum(np.stack([d["scores"] for d in detections]), axis=0, dtype=np.float64)
            class_id = int(np.argmax(scores))

            if class_id != 0:
                representative = max(detections, key=lambda d: d["scores"][class_id])
                frame_detection = {
                    "class_id": class_id, 
                    "scores": scores, 
                    "bbox": representative["bbox"], 
                    "raw_count": len(detections)
                }

        return {
            "detections": detections,
            "frame_detection": frame_detection,
            "frame_class_id": frame_detection["class_id"] if frame_detection else 0,
            "fpga_ms": self._frame_fpga_ms,
            "classifier_calls": self._frame_classifier_calls,
        }
