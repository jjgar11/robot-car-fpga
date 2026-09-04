# Class indices from labels.py (44 GTSRB classes, 0-43)
STOP_CLASS_ID      = 15   # "Stop"           → STOP_HOLD 5 s
ROAD_WORK_CLASS_ID = 26   # "Road work"      → perfil lento (20% PWM)
SPEED_50_CLASS_ID  = 3    # "Speed limit (50km/h)" → perfil rápido (30% PWM)


def classify_sign(class_id: int) -> str:
    """Return action category for a detected class.

    Returns
    -------
    'stop'       → enter STOP_HOLD
    'speed_slow' → switch to slow profile (Road work sign)
    'speed_fast' → switch to fast profile (50 km/h sign)
    'other'      → ignore
    """
    if class_id == STOP_CLASS_ID:
        return 'stop'
    if class_id == ROAD_WORK_CLASS_ID:
        return 'speed_slow'
    if class_id == SPEED_50_CLASS_ID:
        return 'speed_fast'
    return 'other'
