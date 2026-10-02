import sys
import time
import numpy as np
import cv2

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
        self._output_verified = False

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
    def pack_input_int4_fast(x_folded):
        x = np.asarray(x_folded)

        if x.shape[-1] != 3:
            raise ValueError(f"Unexpected folded shape: {x.shape}")

        if (np.any(x < -8) or np.any(x > 7)):
            raise ValueError(
                f"INT4 out of range: "
                f"min={x.min()}, "
                f"max={x.max()}"
            )

        u = (x.astype(np.int16) & 0xF).astype(np.uint8)

        out = np.empty(x.shape[:-1] + (2,), dtype=np.uint8)

        out[..., 0] = (u[..., 0] | (u[..., 1] << 4))

        out[..., 1] = u[..., 2]

        return out

    def label_func(self, frame64):
        """Classify one detector crop and preserve all 44 FINN scores."""

        # RGB HWC -> normalized BHWC
        x_float = frame64.astype(np.float32)[None, ...]

        # Quantize INT4
        x_batch = np.clip(np.floor(x_float * 8.0 + 0.5), -8, 7).astype(np.float32)

        folded = self.accel.fold_input(x_batch, ind=0)

        packed = (self.pack_input_int4_fast(folded))

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
        if not self._output_verified:
            reference = self.accel.unpack_output(self.accel.obuf_packed[0], ind=0)
            if not np.array_equal(obuf_folded, reference): raise RuntimeError("Fast INT16 output differs from FINN driver")
            self._output_verified = True
            print("[inference] Fast INT16 output verified against FINN driver", flush=True)

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

        t0 = time.perf_counter()
        
        expected = (1, 3, self.config.camera_height, self.config.camera_width)

        if x.shape != expected:
            raise ValueError(
                f"Unexpected input shape: {x.shape}; expected {expected}"
            )

        height, width = x.shape[-2:]
        image = np.transpose(x[0], (1, 2, 0))
        detections, crop_labels = [], []

        for fraction in self.config.crop_sizes:
            size = max(1, min(min(height, width), round(min(height, width) * fraction)))
    
            for px, py in self.config.crop_positions:

                left = max(0, min(width - size, round(px * width - size / 2)))
                top = max(0, min(height - size, round(py * height - size / 2)))
                crop = image[top:top + size, left:left + size]

                crop64 = cv2.resize(crop, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_AREA)
                detection = self.label_func(crop64)
                crop_labels.append(detection["class_id"])

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

        detect_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": detections,
            "frame_detection": frame_detection,
            "frame_class_id": frame_detection["class_id"] if frame_detection else 0,
            "crop_labels": crop_labels,
            "fpga_ms": self._frame_fpga_ms,
            "classifier_calls": self._frame_classifier_calls,
            "detect_ms": detect_ms,
        }
