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
    sustain_frames: int = 15,
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
        progress = f"{state.sustain_count}/{sustain_frames}"
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


def draw_listening_screen(
    frame: np.ndarray,
    partial_text: str,
    elapsed: float,
    timeout: float,
) -> np.ndarray:
    """Draw a 'listening for voice' screen. Modifies frame in-place."""
    h, w = frame.shape[:2]

    # Dim the frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # "LISTENING..." header
    cv2.putText(frame, "LISTENING...", (w // 2 - 120, h // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

    # Partial transcription
    if partial_text:
        # Wrap text if too long
        max_chars = w // 12
        lines = [partial_text[i:i + max_chars] for i in range(0, len(partial_text), max_chars)]
        for idx, line in enumerate(lines[:3]):  # Max 3 lines
            y_pos = h // 2 + idx * 30
            cv2.putText(frame, line, (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    # Timer bar
    if timeout > 0:
        progress = min(elapsed / timeout, 1.0)
        bar_width = w - 60
        bar_x = 30
        bar_y = h - 50
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10),
                       (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                       (bar_x + int(bar_width * progress), bar_y + 10),
                       (0, 200, 255), -1)

    return frame
