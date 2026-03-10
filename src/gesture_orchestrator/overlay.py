"""Visual feedback overlay on webcam feed."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import HandData
from .gestures import GestureState, GestureType
from .config import PALM_CENTER_LANDMARKS


# Colors (BGR)
COLOR_LEFT = (255, 100, 0)    # Blue-ish for planner
COLOR_RIGHT = (0, 200, 0)     # Green for coder
COLOR_CLAP = (0, 100, 255)    # Orange for sync
COLOR_WHITE = (255, 255, 255)
COLOR_BUSY = (0, 0, 255)      # Red when dispatching

GESTURE_LABELS = {
    GestureType.PLANNER_ACTIVATE: "PLANNER (Left Hand)",
    GestureType.CODER_ACTIVATE: "CODER (Right Hand)",
    GestureType.SYNC_EXECUTE: "SYNC (Clap!)",
}


def draw_overlay(
    frame: np.ndarray,
    hands: list[HandData],
    state: GestureState,
    triggered: GestureType | None,
    busy: bool,
) -> np.ndarray:
    """Draw landmarks, labels, and status on frame. Modifies frame in-place."""
    h, w = frame.shape[:2]

    # Draw hand landmarks
    for hand in hands:
        color = COLOR_LEFT if hand.handedness == "Left" else COLOR_RIGHT
        # Draw connections between landmarks
        for i, (x, y, _z) in enumerate(hand.landmarks):
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 3, color, -1)

        # Label handedness near wrist
        wx, wy = hand.landmarks[0][0], hand.landmarks[0][1]
        cv2.putText(
            frame,
            hand.handedness,
            (int(wx * w) - 20, int(wy * h) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Show current candidate and sustain progress
    candidate = state.current_candidate
    if candidate != GestureType.NONE:
        label = GESTURE_LABELS.get(candidate, "")
        progress = f"{state.sustain_count}/15"
        cv2.putText(frame, f"{label} [{progress}]", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    # Show triggered gesture
    if triggered is not None:
        label = GESTURE_LABELS.get(triggered, "TRIGGERED")
        color = COLOR_CLAP if triggered == GestureType.SYNC_EXECUTE else COLOR_WHITE
        cv2.putText(frame, f">>> {label} <<<", (10, 65),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Busy indicator
    if busy:
        cv2.putText(frame, "AGENT RUNNING...", (10, h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BUSY, 2)

    # Instructions
    cv2.putText(frame, "Q/ESC: Quit", (w - 150, h - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    return frame
