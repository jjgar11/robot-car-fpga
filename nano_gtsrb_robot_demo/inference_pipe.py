from pathlib import Path
import sys
import time

import cv2
import numpy as np

from labels import LABELS_EN


class InferencePipe:
    """
    FINN inference for one real frame at a time.

    Runtime batch = 1.
    """

    BATCH_SIZE = 1

    def __init__(
        self,
        config,
        clear_pynq_state=True,
    ):
        self.config = config
        self.clear_pynq_state = clear_pynq_state
        self.accel = None

        self._load_overlay()

    def _load_overlay(self):
        base = Path(self.config.base_dir)

        bitfile = (
            base / self.config.bitfile_name
        )
        hwhfile = (
            base / self.config.hwhfile_name
        )
        driver_dir = (
            base / self.config.driver_dir_name
        )

        if not bitfile.is_file():
            raise FileNotFoundError(bitfile)

        if not hwhfile.is_file():
            raise FileNotFoundError(hwhfile)

        if not driver_dir.is_dir():
            raise FileNotFoundError(driver_dir)

        driver_path = str(driver_dir)

        if driver_path not in sys.path:
            sys.path.insert(0, driver_path)

        from driver import (
            FINNExampleOverlay,
            io_shape_dict,
        )
        from pynq.pl_server import global_state

        if self.clear_pynq_state:
            global_state.clear_global_state()

        self.accel = FINNExampleOverlay(
            bitfile_name=str(bitfile),
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

    def preprocess(self, frame_bgr):
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

        crop_shift_percent = self.config.tilt_fraction
        max_shift = (h - side) // 2

        y0 = (h - side) // 2 + int(max_shift * crop_shift_percent)
        y0 = max(0, min(y0, h - side))

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
                x_float * 8.0 + 0.5
            ),
            -8,
            7,
        ).astype(np.float32)

        preprocess_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        return (
            x_int4,
            frame64,
            preprocess_ms,
        )

    @staticmethod
    def pack_input_int4_fast(
        x_folded
    ):
        x = np.asarray(x_folded)

        if x.shape[-1] != 3:
            raise ValueError(
                f"Unexpected folded shape: "
                f"{x.shape}"
            )

        if (
            np.any(x < -8)
            or np.any(x > 7)
        ):
            raise ValueError(
                f"INT4 out of range: "
                f"min={x.min()}, "
                f"max={x.max()}"
            )

        u = (
            x.astype(np.int16) & 0xF
        ).astype(np.uint8)

        out = np.empty(
            x.shape[:-1] + (2,),
            dtype=np.uint8,
        )

        out[..., 0] = (
            u[..., 0]
            | (u[..., 1] << 4)
        )

        out[..., 1] = u[..., 2]

        return out

    def infer_preprocessed(
        self,
        x_frame,
    ):
        x_batch = np.asarray(
            x_frame
        )[np.newaxis, ...]

        expected = (
            1,
            self.config.input_size,
            self.config.input_size,
            3,
        )

        if x_batch.shape != expected:
            raise ValueError(
                f"Unexpected input shape: "
                f"{x_batch.shape}; "
                f"expected {expected}"
            )

        folded = self.accel.fold_input(
            x_batch,
            ind=0,
        )

        packed = (
            self.pack_input_int4_fast(
                folded
            )
        )

        np.copyto(
            self.accel.ibuf_packed_device[0],
            packed,
        )

        self.accel.ibuf_packed_device[
            0
        ].flush()

        idma = self.accel.idma[0]
        odma = self.accel.odma[0]

        if not (
            idma.read(0x00) & 0x4
        ):
            raise RuntimeError(
                "IDMA not idle"
            )

        if not (
            odma.read(0x00) & 0x4
        ):
            raise RuntimeError(
                "ODMA not idle"
            )

        t0 = time.perf_counter()

        self.accel.execute_on_buffers(
            asynch=True,
            batch_size=1,
        )

        while True:
            status = odma.read(0x00)

            if status & 0x2:
                break

            if (
                time.perf_counter() - t0
                > self.config.fpga_timeout_s
            ):
                raise RuntimeError(
                    "FPGA timeout"
                )

            time.sleep(0.0001)

        t1 = time.perf_counter()

        self.accel.copy_output_data_from_device(
            self.accel.obuf_packed[0],
            ind=0,
        )

        obuf_folded = (
            self.accel.unpack_output(
                self.accel.obuf_packed[0],
                ind=0,
            )
        )

        y = self.accel.unfold_output(
            obuf_folded,
            ind=0,
        ).copy()

        scores = y[0]

        class_id = int(
            np.argmax(scores)
        )

        return {
            "raw_class_id": class_id,
            "raw_label": (
                LABELS_EN[class_id]
            ),
            "raw_score": float(
                scores[class_id]
            ),
            "scores": scores.copy(),
            "fpga_ms": (
                t1 - t0
            ) * 1000.0,
        }

    def process_frame(
        self,
        frame_bgr,
    ):
        (
            x,
            image64,
            preprocess_ms,
        ) = self.preprocess(
            frame_bgr
        )

        result = (
            self.infer_preprocessed(
                x
            )
        )

        result[
            "preprocess_ms"
        ] = preprocess_ms

        result[
            "image64"
        ] = image64

        return result
