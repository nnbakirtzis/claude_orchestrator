"""MediaPipe Hand Landmarker wrapper returning structured landmark data."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .config import GestureConfig

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Default model path relative to project root
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "hand_landmarker.task"
)


@dataclass
class HandData:
    """Processed hand detection result."""
    landmarks: list[tuple[float, float, float]]  # (x, y, z) normalized
    handedness: str  # "Left" or "Right" from USER's perspective (inverted)
    score: float


class HandDetector:
    def __init__(self, config: GestureConfig, model_path: str | None = None):
        self._config = config

        resolved = os.path.abspath(model_path or _DEFAULT_MODEL)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {resolved}. "
                "Download it with:\n"
                "  wget https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_time_ms = int(time.monotonic() * 1000)
        logger.info("HandLandmarker initialized with model: %s", resolved)

    def detect(self, frame: np.ndarray) -> list[HandData]:
        """Detect hands in a BGR frame. Returns list of HandData."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.monotonic() * 1000) - self._start_time_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        hands: list[HandData] = []
        for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
            # Invert handedness: MediaPipe labels from camera perspective (mirrored)
            mp_label = handedness_list[0].category_name  # "Left" or "Right"
            user_label = "Right" if mp_label == "Left" else "Left"
            score = handedness_list[0].score

            landmarks = [(lm.x, lm.y, lm.z) for lm in lm_list]

            hands.append(HandData(
                landmarks=landmarks,
                handedness=user_label,
                score=score,
            ))

        return hands

    def close(self) -> None:
        self._landmarker.close()
        logger.info("HandLandmarker closed")
