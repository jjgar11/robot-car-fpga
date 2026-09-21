from dataclasses import dataclass, field
from pathlib import Path
import time


@dataclass
class DemoConfig:
    # Camera
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    camera_fourcc: str = "MJPG"

    # Model / preprocessing
    crop_fraction: float = 0.35
    tilt_fraction: float = 0.0
    input_size: int = 64
    score_window: int = 5
    target_fps: float = 20.0

    # FINN deployment
    base_dir: Path = Path("/home/xilinx/overlays/nano_gtsrb_robot")
    bitfile_name: str = "nano_gtsrb_robot.bit"
    hwhfile_name: str = "nano_gtsrb_robot.hwh"
    driver_dir_name: str = "driver"
    robot_control_dir_name: str = "robot_control"
    runtime_weight_dir: str = "/tmp/no_runtime_weights_nano"

    # FPGA runtime
    fclk_mhz: float = 100.0
    fpga_timeout_s: float = 1.0

    log_path: str = "/tmp/robot_detections.jsonl"
    tcp_port: int = 50007

    @property
    def bitfile(self) -> Path:
        return self.base_dir / self.bitfile_name

    @property
    def hwhfile(self) -> Path:
        return self.base_dir / self.hwhfile_name

    @property
    def driver_dir(self) -> Path:
        return self.base_dir / self.driver_dir_name


"""IPC message types shared between robot processes."""
@dataclass
class DetectionEvent:
    """Carries a sign-detection result from inference to downstream processes."""
    class_id: int
    label: str
    score_sum: float
    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    ready: bool = False  # True when temporal filter has a stable result
    fpga_ms: float = 0.0  # FPGA inference duration, for logging/telemetry only
