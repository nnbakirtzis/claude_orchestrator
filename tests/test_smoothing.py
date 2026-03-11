"""Tests for EMA smoothing, landmark smoothing, and gesture confidence tracking."""

import pytest

from gesture_orchestrator.smoothing import (
    ExponentialMovingAverage,
    GestureConfidenceTracker,
    LandmarkSmoother,
)


class TestExponentialMovingAverage:
    def test_first_value_passes_through(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        assert ema.update(10.0) == 10.0

    def test_converges_toward_constant(self):
        ema = ExponentialMovingAverage(alpha=0.3)
        ema.update(0.0)
        for _ in range(50):
            val = ema.update(100.0)
        assert val == pytest.approx(100.0, abs=0.1)

    def test_smooths_jitter(self):
        ema = ExponentialMovingAverage(alpha=0.2)
        ema.update(50.0)
        # Alternate between 40 and 60 (jittery around 50)
        values = []
        for i in range(20):
            values.append(ema.update(60.0 if i % 2 == 0 else 40.0))
        # Smoothed values should stay near 50
        assert all(40 < v < 60 for v in values)
        assert abs(values[-1] - 50.0) < 5.0

    def test_reset(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(100.0)
        ema.reset()
        assert ema.value is None
        assert ema.update(50.0) == 50.0


class TestLandmarkSmoother:
    def test_first_frame_passes_through(self):
        smoother = LandmarkSmoother(alpha=0.5)
        landmarks = [(float(i), float(i), 0.0) for i in range(21)]
        result = smoother.smooth(landmarks)
        assert result == landmarks

    def test_jitter_reduction(self):
        smoother = LandmarkSmoother(alpha=0.3)
        base = [(0.5, 0.5, 0.0)] * 21
        smoother.smooth(base)

        # Add jitter
        jittered = [(0.55, 0.45, 0.0)] * 21
        result = smoother.smooth(jittered)

        # Should be between base and jittered (smoothed)
        for x, y, z in result:
            assert 0.5 < x < 0.55
            assert 0.45 < y < 0.5

    def test_reset(self):
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.smooth([(1.0, 1.0, 1.0)] * 21)
        smoother.reset()
        # After reset, next value should pass through
        result = smoother.smooth([(0.0, 0.0, 0.0)] * 21)
        assert result[0] == (0.0, 0.0, 0.0)


class TestGestureConfidenceTracker:
    def test_sustained_detection_builds_confidence(self):
        tracker = GestureConfidenceTracker(alpha=0.3)
        for _ in range(20):
            tracker.update("planner", True)
        assert tracker.confidence("planner") > 0.9

    def test_no_detection_drops_confidence(self):
        tracker = GestureConfidenceTracker(alpha=0.3)
        for _ in range(20):
            tracker.update("planner", True)
        for _ in range(20):
            tracker.update("planner", False)
        assert tracker.confidence("planner") < 0.1

    def test_unknown_gesture_returns_zero(self):
        tracker = GestureConfidenceTracker(alpha=0.3)
        assert tracker.confidence("unknown") == 0.0

    def test_reset_clears_all(self):
        tracker = GestureConfidenceTracker(alpha=0.3)
        tracker.update("planner", True)
        tracker.reset()
        assert tracker.confidence("planner") == 0.0
