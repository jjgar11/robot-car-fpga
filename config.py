from dataclasses import dataclass


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

    # Application
    target_fps: float = 20.0

    # FINN deployment — nuevo overlay combinado
    base_dir: str = "/home/xilinx/overlays/nano_gtsrb_robot"
    bitfile_name: str = "nano_gtsrb_robot.bit"
    hwhfile_name: str = "nano_gtsrb_robot.hwh"
    driver_dir_name: str = "driver"
    runtime_weight_dir: str = "/tmp/no_runtime_weights_nano"

    # FPGA runtime
    fclk_mhz: float = 100.0
    fpga_timeout_s: float = 1.0
