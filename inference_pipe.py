import sys
import time
import numpy as np

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
            runtime_weight_dir=(
                self.config.runtime_weight_dir
            ),
        )

        print("FINN overlay loaded")
        print("bitfile:", self.accel.bitfile_name)
        print("runtime batch:", self.BATCH_SIZE)

    @staticmethod
    def pack_input_int4_fast(x_folded):
        x = np.asarray(x_folded)

        if x.shape[-1] != 3:
            raise ValueError(f"Unexpected folded shape: "
                f"{x.shape}")

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

    def infer_preprocessed(self, x_frame):
        x_batch = np.asarray(x_frame)[np.newaxis, ...]

        expected = (1, self.config.input_size, self.config.input_size, 3)

        if x_batch.shape != expected:
            raise ValueError(
                f"Unexpected input shape: "
                f"{x_batch.shape}; "
                f"expected {expected}"
            )

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

        fpga_ms = (fpga_end - fpga_start) * 1_000.0

        self.accel.copy_output_data_from_device(self.accel.obuf_packed[0], ind=0)

        obuf_folded = (self.accel.unpack_output(self.accel.obuf_packed[0], ind=0))

        y = self.accel.unfold_output(obuf_folded, ind=0).copy()

        scores = y[0]

        class_id = int(np.argmax(scores))

        return {
            "raw_class_id": class_id,
            "raw_label": LABELS_EN[class_id],
            "raw_score": float(scores[class_id]),
            "scores": scores.copy(),
            "fpga_ms": fpga_ms,
        }

    def process_frame(self, x):
        result = (self.infer_preprocessed(x))

        return result
