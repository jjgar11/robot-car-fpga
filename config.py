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
    target_fps: float = 20.0

    # Temporal Detection Filter
    score_window: int = 4
    stable_ratio = 0.20
    max_misses = 4
    iou_threshold = 0.25

    # FINN deployment
    base_dir: Path = Path("/home/xilinx/overlays/deploy_optimized")
    # base_dir: Path = Path("/home/xilinx/overlays/nano_gtsrb_robot")
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
    detections: list[dict]
    timestamp: float
    frame_timestamp: float
    frame_id: int
    ready: bool
    detect_ms: float
    filter_ms: float
    fpga_total_ms: float
    fpga_avg_ms: float
    classifier_calls: int
    queue_delay_ms: float
    end_to_end_ms: float
    raw_detection_count: int
    grouped_detection_count: int

    @property
    def primary_detection(self):
        return self.detections[0] if self.detections else None

    @property
    def stable_detection_count(self):
        return len(self.detections)

    @property
    def non_fpga_ms(self):
        return max(0.0, self.detect_ms - self.fpga_total_ms)

    def to_dict(self, include_scores=False):
        detections = []
        for detection in self.detections:
            item = detection.copy()
            if "bbox" in item: item["bbox"] = [int(v) for v in item["bbox"]]
            if "scores" in item:
                if include_scores:
                    scores = item["scores"]
                    item["scores"] = scores.tolist() if hasattr(scores, "tolist") else list(scores)
                else: item.pop("scores")
            detections.append(item)
        return {
            "type": "detection_event",
            "version": 1,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "frame_timestamp": self.frame_timestamp,
            "ready": self.ready,
            "detections": detections,
            "counts": {
                "raw_detections": self.raw_detection_count,
                "grouped_detections": self.grouped_detection_count,
                "stable_detections": self.stable_detection_count,
                "classifier_calls": self.classifier_calls
            },
            "timing": {
                "queue_delay_ms": self.queue_delay_ms,
                "detect_ms": self.detect_ms,
                "filter_ms": self.filter_ms,
                "fpga_total_ms": self.fpga_total_ms,
                "fpga_avg_ms": self.fpga_avg_ms,
                "non_fpga_ms": self.non_fpga_ms,
                "end_to_end_ms": self.end_to_end_ms
            },
        }
