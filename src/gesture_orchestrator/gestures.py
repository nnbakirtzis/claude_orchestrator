"""Heuristic gesture classification from hand landmarks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto

from .config import (
    FINGER_JOINTS,
    FINGERS,
    PALM_CENTER_LANDMARKS,
    THUMB,
    THUMB_JOINTS,
    WRIST,
    GestureConfig,
)
from .detector import HandData


class GestureType(Enum):
    NONE = auto()
    PLANNER_ACTIVATE = auto()   # Left hand raised
    CODER_ACTIVATE = auto()     # Right hand raised
    SYNC_EXECUTE = auto()       # Clap


@dataclass
class GestureState:
    """Tracks sustained gesture detection and cooldowns."""
    current_candidate: GestureType = GestureType.NONE
    sustain_count: int = 0
    last_triggered: dict[GestureType, float] = field(default_factory=dict)
    # For clap velocity detection
    prev_palm_distance: float = 1.0
    # Hysteresis: tracks whether each hand is currently in "raised" state
    hand_raised_state: dict[str, bool] = field(default_factory=dict)


def _palm_center(hand: HandData) -> tuple[float, float]:
    """Compute approximate palm center from key landmarks."""
    xs = [hand.landmarks[i][0] for i in PALM_CENTER_LANDMARKS]
    ys = [hand.landmarks[i][1] for i in PALM_CENTER_LANDMARKS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _angle_between_joints(
    tip: tuple[float, float, float],
    pip: tuple[float, float, float],
    mcp: tuple[float, float, float],
) -> float:
    """Compute angle at PIP/IP joint in degrees using dot product.

    Straight finger ~ 180 degrees, curled ~ 90 degrees.
    """
    # Vector from pip to tip
    v1 = (tip[0] - pip[0], tip[1] - pip[1], tip[2] - pip[2])
    # Vector from pip to mcp
    v2 = (mcp[0] - pip[0], mcp[1] - pip[1], mcp[2] - pip[2])

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    if mag1 < 1e-9 or mag2 < 1e-9:
        return 180.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _count_extended_fingers(hand: HandData, config: GestureConfig) -> int:
    """Count how many fingers are extended.

    Uses angle-based detection when config.use_angle_detection is True,
    with y-position fallback otherwise.
    """
    count = 0

    if config.use_angle_detection:
        # Four fingers via angle detection
        for tip_idx, pip_idx, mcp_idx in FINGER_JOINTS:
            angle = _angle_between_joints(
                hand.landmarks[tip_idx],
                hand.landmarks[pip_idx],
                hand.landmarks[mcp_idx],
            )
            if angle >= config.finger_extension_angle_threshold:
                count += 1

        # Thumb via angle detection
        t_tip, t_ip, t_mcp = THUMB_JOINTS
        thumb_angle = _angle_between_joints(
            hand.landmarks[t_tip],
            hand.landmarks[t_ip],
            hand.landmarks[t_mcp],
        )
        if thumb_angle >= config.finger_extension_angle_threshold:
            count += 1
    else:
        # Y-position fallback (original method)
        for tip_idx, mcp_idx in FINGERS:
            tip_y = hand.landmarks[tip_idx][1]
            mcp_y = hand.landmarks[mcp_idx][1]
            if tip_y < mcp_y - config.finger_extension_margin:
                count += 1

        # Thumb: use x-axis (tip further from palm center than mcp)
        thumb_tip_x = hand.landmarks[THUMB[0]][0]
        thumb_mcp_x = hand.landmarks[THUMB[1]][0]
        if hand.handedness == "Right":
            if thumb_tip_x > thumb_mcp_x + config.finger_extension_margin:
                count += 1
        else:
            if thumb_tip_x < thumb_mcp_x - config.finger_extension_margin:
                count += 1

    return count


def _is_hand_raised(
    hand: HandData, config: GestureConfig, state: GestureState | None = None,
) -> bool:
    """Check if hand is raised, with optional hysteresis.

    When state is provided, uses wrist_y_enter_threshold to enter raised state
    and wrist_y_exit_threshold to leave it (hysteresis).
    When state is None, uses the simple wrist_y_threshold.
    """
    wrist_y = hand.landmarks[WRIST][1]

    if state is not None:
        currently_raised = state.hand_raised_state.get(hand.handedness, False)
        if currently_raised:
            # Use relaxed threshold to exit
            if wrist_y >= config.wrist_y_exit_threshold:
                state.hand_raised_state[hand.handedness] = False
                return False
        else:
            # Use stricter threshold to enter
            if wrist_y >= config.wrist_y_enter_threshold:
                return False
            # Passed strict threshold, mark as raised
            state.hand_raised_state[hand.handedness] = True
    else:
        # Simple threshold (backward compat for tests without state)
        if wrist_y >= config.wrist_y_threshold:
            return False

    extended = _count_extended_fingers(hand, config)
    return extended >= config.min_extended_fingers


def _detect_clap(hands: list[HandData], state: GestureState, config: GestureConfig) -> bool:
    """Detect clap: two hands, palms close, with approach velocity."""
    if len(hands) < 2:
        state.prev_palm_distance = 1.0
        return False

    c1 = _palm_center(hands[0])
    c2 = _palm_center(hands[1])
    distance = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    had_velocity = state.prev_palm_distance >= config.clap_approach_velocity
    is_close = distance < config.clap_distance_threshold

    state.prev_palm_distance = distance

    return is_close and had_velocity


def classify_gesture(
    hands: list[HandData],
    state: GestureState,
    config: GestureConfig,
) -> GestureType | None:
    """
    Classify current frame's gesture. Returns a GestureType when a gesture
    has been sustained long enough and cooldown has elapsed, or None.
    """
    now = time.monotonic()

    # Determine raw gesture for this frame
    raw = GestureType.NONE

    if _detect_clap(hands, state, config):
        raw = GestureType.SYNC_EXECUTE
    else:
        for hand in hands:
            if _is_hand_raised(hand, config, state):
                if hand.handedness == "Left":
                    raw = GestureType.PLANNER_ACTIVATE
                else:
                    raw = GestureType.CODER_ACTIVATE
                break  # First raised hand wins

    # Sustain logic
    if raw == state.current_candidate and raw != GestureType.NONE:
        state.sustain_count += 1
    else:
        state.current_candidate = raw
        state.sustain_count = 1 if raw != GestureType.NONE else 0

    # Check if sustained long enough
    if state.sustain_count < config.sustain_frames:
        return None

    # Check cooldown
    gesture = state.current_candidate
    cooldown = (
        config.clap_cooldown if gesture == GestureType.SYNC_EXECUTE
        else config.raise_cooldown
    )
    last = state.last_triggered.get(gesture, 0.0)
    if now - last < cooldown:
        return None

    # Trigger!
    state.last_triggered[gesture] = now
    state.sustain_count = 0  # Reset to prevent immediate re-trigger
    return gesture
