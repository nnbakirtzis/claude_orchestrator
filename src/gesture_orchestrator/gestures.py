"""Heuristic gesture classification from hand landmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from .config import (
    FINGERS,
    PALM_CENTER_LANDMARKS,
    THUMB,
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


def _palm_center(hand: HandData) -> tuple[float, float]:
    """Compute approximate palm center from key landmarks."""
    xs = [hand.landmarks[i][0] for i in PALM_CENTER_LANDMARKS]
    ys = [hand.landmarks[i][1] for i in PALM_CENTER_LANDMARKS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _count_extended_fingers(hand: HandData, config: GestureConfig) -> int:
    """Count how many fingers are extended (fingertip above MCP joint)."""
    count = 0
    for tip_idx, mcp_idx in FINGERS:
        tip_y = hand.landmarks[tip_idx][1]
        mcp_y = hand.landmarks[mcp_idx][1]
        if tip_y < mcp_y - config.finger_extension_margin:
            count += 1

    # Thumb: use x-axis (tip further from palm center than mcp)
    thumb_tip_x = hand.landmarks[THUMB[0]][0]
    thumb_mcp_x = hand.landmarks[THUMB[1]][0]
    if hand.handedness == "Right":
        # User's right hand: thumb extends right = higher x in mirrored frame
        if thumb_tip_x > thumb_mcp_x + config.finger_extension_margin:
            count += 1
    else:
        # User's left hand: thumb extends left = lower x in mirrored frame
        if thumb_tip_x < thumb_mcp_x - config.finger_extension_margin:
            count += 1

    return count


def _is_hand_raised(hand: HandData, config: GestureConfig) -> bool:
    """Check if hand is raised: wrist in upper frame + fingers extended."""
    wrist_y = hand.landmarks[WRIST][1]
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
            if _is_hand_raised(hand, config):
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
