"""Tests for gesture classification with fabricated landmarks."""

import time

import pytest

from gesture_orchestrator.config import GestureConfig
from gesture_orchestrator.detector import HandData
from gesture_orchestrator.gestures import (
    GestureState,
    GestureType,
    classify_gesture,
    _count_extended_fingers,
    _is_hand_raised,
)


def _make_landmarks(wrist_y=0.5, fingers_extended=True):
    """Create a 21-landmark hand with configurable wrist and finger positions."""
    # Start with all landmarks at neutral position
    landmarks = [(0.5, 0.5, 0.0)] * 21

    # Set wrist position
    landmarks[0] = (0.5, wrist_y, 0.0)

    if fingers_extended:
        # Fingers extended: tips above MCPs (lower y = higher in frame)
        # Index
        landmarks[5] = (0.4, 0.45, 0.0)   # MCP
        landmarks[8] = (0.4, 0.20, 0.0)    # Tip (above MCP)
        # Middle
        landmarks[9] = (0.45, 0.45, 0.0)
        landmarks[12] = (0.45, 0.20, 0.0)
        # Ring
        landmarks[13] = (0.5, 0.45, 0.0)
        landmarks[16] = (0.5, 0.20, 0.0)
        # Pinky
        landmarks[17] = (0.55, 0.45, 0.0)
        landmarks[20] = (0.55, 0.20, 0.0)
        # Thumb (extended to left for left hand)
        landmarks[2] = (0.35, 0.40, 0.0)   # MCP
        landmarks[4] = (0.25, 0.35, 0.0)   # Tip (further left)
    else:
        # Fingers curled: tips below MCPs
        landmarks[5] = (0.4, 0.40, 0.0)
        landmarks[8] = (0.4, 0.50, 0.0)
        landmarks[9] = (0.45, 0.40, 0.0)
        landmarks[12] = (0.45, 0.50, 0.0)
        landmarks[13] = (0.5, 0.40, 0.0)
        landmarks[16] = (0.5, 0.50, 0.0)
        landmarks[17] = (0.55, 0.40, 0.0)
        landmarks[20] = (0.55, 0.50, 0.0)
        landmarks[2] = (0.35, 0.40, 0.0)
        landmarks[4] = (0.36, 0.40, 0.0)

    return landmarks


def _make_hand(handedness="Left", wrist_y=0.5, fingers_extended=True):
    return HandData(
        landmarks=_make_landmarks(wrist_y, fingers_extended),
        handedness=handedness,
        score=0.95,
    )


@pytest.fixture
def config():
    return GestureConfig(sustain_frames=1)  # No sustain for unit tests


@pytest.fixture
def config_sustained():
    return GestureConfig(sustain_frames=3)


class TestFingerCounting:
    def test_all_extended(self, config):
        hand = _make_hand(handedness="Left", fingers_extended=True)
        assert _count_extended_fingers(hand, config) == 5

    def test_none_extended(self, config):
        hand = _make_hand(handedness="Left", fingers_extended=False)
        assert _count_extended_fingers(hand, config) == 0


class TestHandRaised:
    def test_raised_left(self, config):
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=True)
        assert _is_hand_raised(hand, config) is True

    def test_not_raised_low_wrist(self, config):
        hand = _make_hand(handedness="Left", wrist_y=0.6, fingers_extended=True)
        assert _is_hand_raised(hand, config) is False

    def test_not_raised_fingers_curled(self, config):
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=False)
        assert _is_hand_raised(hand, config) is False


class TestGestureClassification:
    def test_left_hand_raised_triggers_planner(self, config):
        state = GestureState()
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=True)
        result = classify_gesture([hand], state, config)
        assert result == GestureType.PLANNER_ACTIVATE

    def test_right_hand_raised_triggers_coder(self, config):
        state = GestureState()
        hand = _make_hand(handedness="Right", wrist_y=0.2, fingers_extended=True)
        # Right hand thumb extends right (lower x for user perspective)
        hand.landmarks[2] = (0.65, 0.40, 0.0)  # MCP
        hand.landmarks[4] = (0.75, 0.35, 0.0)  # Tip further right... wait
        # For right hand, thumb tip x < mcp x
        hand.landmarks[2] = (0.65, 0.40, 0.0)
        hand.landmarks[4] = (0.55, 0.35, 0.0)
        result = classify_gesture([hand], state, config)
        assert result == GestureType.CODER_ACTIVATE

    def test_no_hands_returns_none(self, config):
        state = GestureState()
        result = classify_gesture([], state, config)
        assert result is None

    def test_cooldown_prevents_retrigger(self, config):
        state = GestureState()
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=True)

        # First trigger
        result1 = classify_gesture([hand], state, config)
        assert result1 == GestureType.PLANNER_ACTIVATE

        # Immediate re-trigger should be blocked by cooldown
        state.sustain_count = 0
        state.current_candidate = GestureType.NONE
        result2 = classify_gesture([hand], state, config)
        assert result2 is None

    def test_sustain_requirement(self, config_sustained):
        state = GestureState()
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=True)

        # First 2 frames: not enough sustain
        assert classify_gesture([hand], state, config_sustained) is None
        assert classify_gesture([hand], state, config_sustained) is None

        # Third frame: sustained enough
        result = classify_gesture([hand], state, config_sustained)
        assert result == GestureType.PLANNER_ACTIVATE

    def test_clap_detection(self, config):
        config_clap = GestureConfig(sustain_frames=1)
        state = GestureState()

        left = _make_hand(handedness="Left", wrist_y=0.4, fingers_extended=True)
        right = _make_hand(handedness="Right", wrist_y=0.4, fingers_extended=True)

        # Set palms far apart first (for velocity)
        for i in [0, 5, 9, 13, 17]:
            left.landmarks[i] = (0.2, left.landmarks[i][1], 0.0)
            right.landmarks[i] = (0.8, right.landmarks[i][1], 0.0)

        # Frame 1: far apart - sets prev_palm_distance
        classify_gesture([left, right], state, config_clap)

        # Now move palms together
        for i in [0, 5, 9, 13, 17]:
            left.landmarks[i] = (0.50, left.landmarks[i][1], 0.0)
            right.landmarks[i] = (0.52, right.landmarks[i][1], 0.0)

        result = classify_gesture([left, right], state, config_clap)
        assert result == GestureType.SYNC_EXECUTE
