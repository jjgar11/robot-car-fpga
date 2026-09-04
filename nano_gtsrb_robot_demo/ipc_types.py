"""IPC message types shared between robot processes."""
from dataclasses import dataclass, field
import time


@dataclass
class DetectionEvent:
    """Carries a sign-detection result from inference to downstream processes."""
    class_id: int
    label: str
    score_sum: float
    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    ready: bool = False  # True when temporal filter has a stable result
