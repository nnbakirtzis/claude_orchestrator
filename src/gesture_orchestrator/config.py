"""Thresholds, cooldowns, and constants for gesture detection."""

from dataclasses import dataclass, field


@dataclass
class GestureConfig:
    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # MediaPipe model path (None = auto-detect hand_landmarker.task)
    model_path: str | None = None

    # MediaPipe Hands
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5

    # Hand-raised thresholds
    wrist_y_threshold: float = 0.35          # wrist must be above this (normalized, 0=top)
    wrist_y_enter_threshold: float = 0.32    # stricter threshold for entering raised state
    wrist_y_exit_threshold: float = 0.40     # relaxed threshold for leaving raised state
    min_extended_fingers: int = 4            # at least 4 fingers extended
    finger_extension_margin: float = 0.02    # fingertip_y must be < mcp_y - margin

    # Angle-based finger detection
    use_angle_detection: bool = True
    finger_extension_angle_threshold: float = 160.0  # degrees, straight ~180, curled ~90

    # Smoothing
    landmark_smoothing_alpha: float = 0.4
    gesture_confidence_alpha: float = 0.3
    gesture_confidence_threshold: float = 0.6

    # Clap thresholds
    clap_distance_threshold: float = 0.08    # palm centers must be this close
    clap_approach_velocity: float = 0.25     # palms must have been this far apart recently

    # Sustain: gesture must be held for this many consecutive frames
    sustain_frames: int = 15

    # Cooldowns (seconds)
    raise_cooldown: float = 3.0
    clap_cooldown: float = 5.0

    # Dispatch
    project_dir: str = "."

    # Interactive terminal
    interactive_terminal: bool = True

    # Voice input
    voice_enabled: bool = True
    voice_timeout: float = 10.0
    voice_silence_timeout: float = 2.0
    voice_model_path: str | None = None
    voice_sample_rate: int = 16000
    voice_energy_threshold: float = 300.0

    # Overlay
    show_overlay: bool = True
    debug: bool = False


# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17

# PIP joint indices (for angle-based detection)
THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

# Finger definitions: (tip_index, mcp_index)
FINGERS = [
    (INDEX_TIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_MCP),
    (RING_TIP, RING_MCP),
    (PINKY_TIP, PINKY_MCP),
]

# Finger joints for angle detection: (tip, pip, mcp)
FINGER_JOINTS = [
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
]

# Thumb joints for angle detection: (tip, ip, mcp)
THUMB_JOINTS = (THUMB_TIP, THUMB_IP, THUMB_MCP)

# Thumb uses x-axis comparison instead of y-axis
THUMB = (THUMB_TIP, THUMB_MCP)

# Palm center approximation landmarks
PALM_CENTER_LANDMARKS = [0, 5, 9, 13, 17]
