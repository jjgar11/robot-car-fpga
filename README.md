# Nano GTSRB FINN Demo — Ultra96-V2 robot

Camera + FINN inference + line-following pipeline running as four
independent OS processes on the Ultra96-V2.

## Current baseline

- Camera: `/dev/video0`, 640x480 @ 30 FPS MJPG
- FINN runtime batch: 1, input: 64x64x3 NHWC INT4
- Central crop: configurable (default 0.30), vertical tilt: configurable (default 0.0)
- Temporal decision: argmax over sum of latest 10 score vectors
- Observed: ~8–10 inference FPS, ~33 ms FPGA per frame
- Line-follow loop: 200 Hz, IR majority-vote filter (3 samples)
- Watchdog: kicked every 400 ms (< 1 s hardware timeout)

## Deployment paths

FINN overlay:
```
/home/xilinx/overlays/nano_gtsrb_robot/
```

Application:
```
/home/xilinx/nano_gtsrb_robot_demo/
```

## Running the full system (4-process robot)

Login as root on the board, then:

```bash
source /etc/profile.d/xrt_setup.sh && \
cd /home/xilinx/nano_gtsrb_robot_demo && \
/usr/local/share/pynq-venv/bin/python3 robot_main.py --full --confirm
```

Stop with `Ctrl+C`. The supervisor coasts motors and sets ENABLE=0 before exit.

### Other run modes

```bash
# Camera + inference + comm only — no motors
/usr/local/share/pynq-venv/bin/python3 robot_main.py --camera-inference-only

# Line-following only — no camera or inference
/usr/local/share/pynq-venv/bin/python3 robot_main.py --movement-only

# Full simulation — fake MMIO, no real motors
/usr/local/share/pynq-venv/bin/python3 robot_main.py --dry-run
```

The `source` step is required; without it PYNQ raises "No Devices Found".

## Sign behaviour

| Sign | Class | Action |
|---|---|---|
| Stop | 15 | Coast both sides for 5 s, then resume |
| Speed limit 50 km/h | 3 | Switch to fast profile (35% PWM) |
| Road work | 26 | Switch to slow profile (25% PWM) |
| Anything else | — | Ignored |

Default profile on startup: **slow (25% PWM)**.

## Jupyter notebook

Open `/home/xilinx/jupyter_notebooks/hls/gtsrb/nano_gtsrb_demo.ipynb`.

The notebook imports from `/home/xilinx/nano_gtsrb_robot_demo/`. After
modifying any `.py` file, **restart the kernel** before re-running; Python
does not reload already-imported modules.

The preview shows the full 640x480 frame with a green rectangle marking
the exact crop region sent to FINN. `tilt_fraction` shifts the rectangle
vertically (0.0 = centre, 1.0 = bottom).

## TCP communication

The comm process listens on port 50007 (configurable with `--port`).
Payload: 8 bytes big-endian — `uint32 class_id`, `uint32 confidence`.
`confidence = int(score_sum)` where `score_sum` is the sum of raw FINN
logits for the winning class over the 10-frame window.

Detections are also written to `/tmp/robot_detections.jsonl`.
