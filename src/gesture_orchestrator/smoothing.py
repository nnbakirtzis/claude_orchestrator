"""Exponential moving average smoothing for landmarks and gesture confidence."""

from __future__ import annotations


class ExponentialMovingAverage:
    """Single-value EMA filter."""

    def __init__(self, alpha: float = 0.4):
        self._alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self._alpha * raw + (1 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


class LandmarkSmoother:
    """Smooth all 21 hand landmarks (x, y, z each) with independent EMAs."""

    NUM_LANDMARKS = 21
    COORDS = 3  # x, y, z

    def __init__(self, alpha: float = 0.4):
        self._filters: list[list[ExponentialMovingAverage]] = [
            [ExponentialMovingAverage(alpha) for _ in range(self.COORDS)]
            for _ in range(self.NUM_LANDMARKS)
        ]

    def smooth(
        self, landmarks: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        result = []
        for i, (x, y, z) in enumerate(landmarks):
            sx = self._filters[i][0].update(x)
            sy = self._filters[i][1].update(y)
            sz = self._filters[i][2].update(z)
            result.append((sx, sy, sz))
        return result

    def reset(self) -> None:
        for row in self._filters:
            for f in row:
                f.reset()


class GestureConfidenceTracker:
    """Smooth binary gesture signals (1.0 detected / 0.0 not) into a confidence score."""

    def __init__(self, alpha: float = 0.3):
        self._trackers: dict[str, ExponentialMovingAverage] = {}
        self._alpha = alpha

    def update(self, gesture_key: str, detected: bool) -> float:
        if gesture_key not in self._trackers:
            self._trackers[gesture_key] = ExponentialMovingAverage(self._alpha)
        return self._trackers[gesture_key].update(1.0 if detected else 0.0)

    def confidence(self, gesture_key: str) -> float:
        tracker = self._trackers.get(gesture_key)
        if tracker is None or tracker.value is None:
            return 0.0
        return tracker.value

    def reset(self) -> None:
        self._trackers.clear()
