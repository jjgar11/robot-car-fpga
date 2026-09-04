#!/usr/bin/env python3

import argparse

from config import DemoConfig
from pipe_manager import PipeManager


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Nano GTSRB FINN camera demo"
        )
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target inference FPS",
    )

    parser.add_argument(
        "--crop",
        type=float,
        default=0.30,
        help="Central crop fraction",
    )

    parser.add_argument(
        "--tilt",
        type=float,
        default=0.0,
        help="Vertical tilt fraction",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Temporal score window",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print one status line every N frames",
    )

    parser.add_argument(
        "--no-clear-pynq-state",
        action="store_true",
        help=(
            "Do not clear PYNQ global state "
            "before overlay load"
        ),
    )

    args = parser.parse_args()

    config = DemoConfig(
        target_fps=args.fps,
        crop_fraction=args.crop,
        tilt_fraction=args.tilt,
        score_window=args.window,
    )

    manager = PipeManager(
        config,
        clear_pynq_state=(
            not args.no_clear_pynq_state
        ),
    )

    counter = {"n": 0}

    def console_update(
        snapshot,
        frame_bgr,
    ):
        counter["n"] += 1

        if (
            counter["n"]
            % args.print_every
            != 0
        ):
            return

        if snapshot.window_ready:
            pred = (
                f"{snapshot.label} | "
                f"ID {snapshot.class_id} | "
                f"sum "
                f"{snapshot.score_sum:.1f}"
            )
        else:
            pred = (
                f"Warming up "
                f"{snapshot.window_fill}/"
                f"{config.score_window}"
            )

        print(
            "\r"
            f"{pred} | "
            f"FPS {snapshot.fps:.1f} | "
            f"camFPS "
            f"{snapshot.camera_fps:.1f} | "
            f"age "
            f"{snapshot.capture_age_ms:.1f} ms | "
            f"prep "
            f"{snapshot.preprocess_ms:.1f} ms | "
            f"FPGA "
            f"{snapshot.fpga_ms:.1f} ms"
            "                    ",
            end="",
            flush=False,
        )

    manager.run(
        on_update=console_update
    )


if __name__ == "__main__":
    main()
