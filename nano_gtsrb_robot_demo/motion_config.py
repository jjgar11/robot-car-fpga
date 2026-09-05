"""Motion profiles and constants for the robot movement process."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionProfile:
    name: str
    base_duty: int    # straight-line speed duty cycle (%)
    turn_outer: int   # outer wheel duty during turn (minimum for rotation: 30%)
    corner_inner: int # inner wheel duty at corner (resistance, no rotation)


# Two profiles only.
# Minimum duty for physical wheel rotation is 25%.
# Below 25% PWM produces motor sound/resistance but no rotation.
PROFILE_SLOW = MotionProfile('slow', base_duty=25, turn_outer=30, corner_inner=5)
PROFILE_FAST = MotionProfile('fast', base_duty=35, turn_outer=30, corner_inner=5)

PROFILES: dict = {
    'slow': PROFILE_SLOW,
    'fast': PROFILE_FAST,
}

# Robot starts slow; a 50 km/h sign switches to fast; Road work switches back to slow.
DEFAULT_PROFILE = PROFILE_SLOW

# Stop behaviour
STOP_HOLD_S     = 5.0   # seconds to coast when a Stop sign is seen
STOP_COOLDOWN_S = 3.0   # seconds to ignore subsequent Stop signs after resuming

# Control loop timing
MOVEMENT_HZ    = 200    # iterations per second → 5 ms per iteration
WDG_KICK_EVERY = 80     # kick every 80 × 5 ms = 400 ms (< 1 s watchdog)

# IR sensor filtering
IR_FILTER_SAMPLES = 3   # majority-vote window

# Maximum acceptable age of a DetectionEvent
MAX_EVENT_AGE_S = 2.0

# Robot-control peripheral address
RC_BASE = 0xA0010000
RC_SIZE = 0x1000

# Overlay paths
OVERLAY_BIT = '/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.bit'
OVERLAY_HWH = '/home/xilinx/overlays/nano_gtsrb_robot/nano_gtsrb_robot.hwh'
