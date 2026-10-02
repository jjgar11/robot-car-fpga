import sys
import time
import numpy as np
import torch

from labels import LABELS_EN
from Detect import Locator, detect

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
        self.locator = Locator('locator.pth') 
        torch.set_num_threads(3)  # 4 threads oversubscribe the 4 cores shared with the camera process
        self._output_verified = False
        self._input_verified = False

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

    def label_func(self, frame64):
        """Classify one detector crop and preserve all 44 FINN scores."""

        # Torch BCHW -> NumPy BHWC
        x_float = np.transpose(frame64.detach().cpu().numpy(), (0, 2, 3, 1))

        packed = self.quantize_pack_int4(x_float)[0]
        if not self._input_verified:
            x_batch = np.clip(np.floor(x_float.astype(np.float32) * 8.0 + 0.5), -8, 7).astype(np.float32)
            if not np.array_equal(packed, self.accel.pack_input(self.accel.fold_input(x_batch, ind=0), ind=0)): raise RuntimeError("Fast INT4 input differs from FINN driver")
            self._input_verified = True
            print("[inference] Fast INT4 input verified against FINN driver", flush=True)

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
        """Run David's detector and FINN classification on one camera frame."""

        self._frame_fpga_ms = 0.0
        self._frame_classifier_calls = 0

        t0 = time.perf_counter()
        
        expected = (1, 3, self.config.camera_height, self.config.camera_width)

        if x.shape != expected:
            raise ValueError(
                f"Unexpected input shape: {x.shape}; expected {expected}"
            )

        x = torch.from_numpy(x).float()
        detections = detect(x, self.locator, self.label_func)
        for detection in detections:
            cx, cy, size = detection["bbox"]

            detection["bbox"] = (
                int(cx.item()) if torch.is_tensor(cx) else int(cx),
                int(cy.item()) if torch.is_tensor(cy) else int(cy),
                int(size.item()) if torch.is_tensor(size) else int(size),
            )

        detect_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": detections,
            "fpga_ms": self._frame_fpga_ms,
            "classifier_calls": self._frame_classifier_calls,
            "detect_ms": detect_ms,
        }
