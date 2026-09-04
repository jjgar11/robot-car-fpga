import cv2


def annotate_frame(
    frame_bgr,
    snapshot,
    crop_fraction,
    tilt_fraction,
    input_size=64,
):
    """
    Show the FULL camera frame.

    A rectangle marks the exact central region later cropped and resized
    to 64x64 for FINN.
    """

    frame = frame_bgr.copy()

    h, w = frame.shape[:2]

    side = int(
        min(h, w)
        * crop_fraction
    )

    side = max(
        side,
        input_size,
    )

    x0 = (w - side) // 2

    crop_shift_percent = tilt_fraction
    max_shift = (h - side) // 2

    y0 = (h - side) // 2 + int(max_shift * crop_shift_percent)
    y0 = max(0, min(y0, h - side))

    cv2.rectangle(
        frame,
        (x0, y0),
        (
            x0 + side,
            y0 + side,
        ),
        (0, 255, 0),
        3,
    )

    if snapshot.window_ready:
        line1 = (
            f"{snapshot.label} | "
            f"ID {snapshot.class_id} | "
            f"sum {snapshot.score_sum:.1f}"
        )
    else:
        line1 = (
            f"Warming up "
            f"{snapshot.window_fill}"
        )

    line2 = (
        f"FPGA {snapshot.fpga_ms:.1f} ms | "
        f"FPS {snapshot.fps:.1f}"
    )

    cv2.putText(
        frame,
        line1,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        line2,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    return frame
