"""OpenCV webcam wrapper."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        logger.info("Camera opened: device=%d, %dx%d", self._device_index, self._width, self._height)

    def read(self) -> np.ndarray | None:
        """Read a single frame. Returns BGR ndarray or None on failure."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()
