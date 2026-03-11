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
    _angle_between_joints,
)


def _make_landmarks(wrist_y=0.5, fingers_extended=True):
    """Create a 21-landmark hand with configurable wrist and finger positions.

    Sets PIP joints between tip and MCP so angle-based detection works:
    - Extended: tip-pip-mcp form a straight line (~180 deg)
    - Curled: tip is near mcp, pip is offset (~90 deg)
    """
    # Start with all landmarks at neutral position
    landmarks = [(0.5, 0.5, 0.0)] * 21

    # Set wrist position
    landmarks[0] = (0.5, wrist_y, 0.0)

    if fingers_extended:
        # Fingers extended: tips above MCPs, PIP in between (straight line ~180 deg)
        # Index: MCP(5), PIP(6), TIP(8)
        landmarks[5] = (0.4, 0.45, 0.0)   # MCP
        landmarks[6] = (0.4, 0.325, 0.0)  # PIP (midpoint)
        landmarks[8] = (0.4, 0.20, 0.0)   # Tip
        # Middle: MCP(9), PIP(10), TIP(12)
        landmarks[9] = (0.45, 0.45, 0.0)
        landmarks[10] = (0.45, 0.325, 0.0)
        landmarks[12] = (0.45, 0.20, 0.0)
        # Ring: MCP(13), PIP(14), TIP(16)
        landmarks[13] = (0.5, 0.45, 0.0)
        landmarks[14] = (0.5, 0.325, 0.0)
        landmarks[16] = (0.5, 0.20, 0.0)
        # Pinky: MCP(17), PIP(18), TIP(20)
        landmarks[17] = (0.55, 0.45, 0.0)
        landmarks[18] = (0.55, 0.325, 0.0)
        landmarks[20] = (0.55, 0.20, 0.0)
        # Thumb: MCP(2), IP(3), TIP(4) - extended straight to the left
        landmarks[2] = (0.35, 0.40, 0.0)   # MCP
        landmarks[3] = (0.30, 0.375, 0.0)  # IP (midpoint)
        landmarks[4] = (0.25, 0.35, 0.0)   # Tip
    else:
        # Fingers curled: tip near mcp level, pip offset (sharp angle ~90 deg)
        # Index
        landmarks[5] = (0.4, 0.40, 0.0)    # MCP
        landmarks[6] = (0.4, 0.35, 0.0)    # PIP (higher)
        landmarks[8] = (0.4, 0.42, 0.0)    # Tip (curled back down near MCP)
        # Middle
        landmarks[9] = (0.45, 0.40, 0.0)
        landmarks[10] = (0.45, 0.35, 0.0)
        landmarks[12] = (0.45, 0.42, 0.0)
        # Ring
        landmarks[13] = (0.5, 0.40, 0.0)
        landmarks[14] = (0.5, 0.35, 0.0)
        landmarks[16] = (0.5, 0.42, 0.0)
        # Pinky
        landmarks[17] = (0.55, 0.40, 0.0)
        landmarks[18] = (0.55, 0.35, 0.0)
        landmarks[20] = (0.55, 0.42, 0.0)
        # Thumb curled
        landmarks[2] = (0.35, 0.40, 0.0)   # MCP
        landmarks[3] = (0.33, 0.38, 0.0)   # IP
        landmarks[4] = (0.36, 0.40, 0.0)   # Tip (curled back)

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


class TestAngleDetection:
    def test_straight_finger_angle(self):
        # Points in a straight line (vertically)
        tip = (0.5, 0.1, 0.0)
        pip = (0.5, 0.3, 0.0)
        mcp = (0.5, 0.5, 0.0)
        angle = _angle_between_joints(tip, pip, mcp)
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_curled_finger_angle(self):
        # 90-degree angle
        tip = (0.6, 0.3, 0.0)
        pip = (0.5, 0.3, 0.0)
        mcp = (0.5, 0.5, 0.0)
        angle = _angle_between_joints(tip, pip, mcp)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_zero_length_returns_180(self):
        p = (0.5, 0.5, 0.0)
        assert _angle_between_joints(p, p, p) == 180.0


class TestFingerCounting:
    def test_all_extended(self, config):
        hand = _make_hand(handedness="Left", fingers_extended=True)
        assert _count_extended_fingers(hand, config) == 5

    def test_none_extended(self, config):
        hand = _make_hand(handedness="Left", fingers_extended=False)
        assert _count_extended_fingers(hand, config) == 0

    def test_angle_detection_fallback(self, config):
        """When angle detection is off, uses y-position method."""
        config_no_angle = GestureConfig(sustain_frames=1, use_angle_detection=False)
        hand = _make_hand(handedness="Left", fingers_extended=True)
        assert _count_extended_fingers(hand, config_no_angle) == 5

    def test_angle_detection_fallback_curled(self, config):
        config_no_angle = GestureConfig(sustain_frames=1, use_angle_detection=False)
        hand = _make_hand(handedness="Left", fingers_extended=False)
        assert _count_extended_fingers(hand, config_no_angle) == 0


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

    def test_hysteresis_enter_exit(self):
        """Test hysteresis: stricter enter threshold, relaxed exit threshold."""
        config = GestureConfig(
            sustain_frames=1,
            wrist_y_enter_threshold=0.32,
            wrist_y_exit_threshold=0.40,
        )
        state = GestureState()

        # Below enter threshold → raised
        hand = _make_hand(handedness="Left", wrist_y=0.25, fingers_extended=True)
        assert _is_hand_raised(hand, config, state) is True
        assert state.hand_raised_state["Left"] is True

        # Between enter and exit thresholds → still raised (hysteresis)
        hand = _make_hand(handedness="Left", wrist_y=0.35, fingers_extended=True)
        assert _is_hand_raised(hand, config, state) is True

        # Above exit threshold → no longer raised
        hand = _make_hand(handedness="Left", wrist_y=0.45, fingers_extended=True)
        assert _is_hand_raised(hand, config, state) is False
        assert state.hand_raised_state["Left"] is False


class TestGestureClassification:
    def test_left_hand_raised_triggers_planner(self, config):
        state = GestureState()
        hand = _make_hand(handedness="Left", wrist_y=0.2, fingers_extended=True)
        result = classify_gesture([hand], state, config)
        assert result == GestureType.PLANNER_ACTIVATE

    def test_right_hand_raised_triggers_coder(self, config):
        state = GestureState()
        hand = _make_hand(handedness="Right", wrist_y=0.2, fingers_extended=True)
        # Set right hand thumb landmarks for angle detection
        # Thumb extended to the right: MCP(2), IP(3), TIP(4) in a line
        hand.landmarks[2] = (0.65, 0.40, 0.0)   # MCP
        hand.landmarks[3] = (0.70, 0.375, 0.0)   # IP
        hand.landmarks[4] = (0.75, 0.35, 0.0)    # Tip
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
