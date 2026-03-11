"""Voice input via Vosk speech recognition."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

from .config import GestureConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "vosk-model-small-en-us-0.15"
DEFAULT_MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_MODEL_NAME}.zip"
MODELS_DIR = Path.home() / ".gesture_orchestrator" / "models"


class VoiceCapture:
    """Record audio and transcribe via Vosk."""

    def __init__(self, config: GestureConfig):
        self._config = config
        self._model = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if microphone and Vosk model are accessible."""
        if self._available is not None:
            return self._available

        self._available = False
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except ImportError:
            logger.debug("vosk or sounddevice not installed")
            return False

        # Check for microphone
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            has_input = any(
                d.get("max_input_channels", 0) > 0
                for d in (devices if isinstance(devices, list) else [devices])
            )
            if not has_input:
                logger.debug("No input devices found")
                return False
        except Exception:
            logger.debug("Could not query audio devices")
            return False

        # Check/load model
        model_path = self._resolve_model_path()
        if model_path is None:
            return False

        try:
            import vosk
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(model_path))
            self._available = True
        except Exception:
            logger.exception("Failed to load Vosk model from %s", model_path)
            return False

        return True

    def listen(
        self,
        timeout: float | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> str | None:
        """Record audio and return transcription, or None on failure/timeout.

        Args:
            timeout: Max recording duration in seconds (default: config.voice_timeout)
            on_partial: Callback for intermediate transcription results
        """
        if not self._available or self._model is None:
            return None

        import sounddevice as sd
        import vosk

        timeout = timeout or self._config.voice_timeout
        sample_rate = self._config.voice_sample_rate

        # Audio beep to indicate listening started
        self._beep_start()

        rec = vosk.KaldiRecognizer(self._model, sample_rate)
        rec.SetWords(True)

        speech_started = False
        silence_start: float | None = None
        start_time = time.monotonic()
        result_text: str | None = None

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=4000,
            ) as stream:
                while True:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        logger.debug("Voice recording timed out at %.1fs", elapsed)
                        break

                    data, overflowed = stream.read(4000)
                    if overflowed:
                        logger.debug("Audio buffer overflow")

                    audio_bytes = bytes(data)

                    # RMS energy for silence detection
                    rms = self._compute_rms(data)

                    if rms > self._config.voice_energy_threshold:
                        speech_started = True
                        silence_start = None
                    elif speech_started:
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif time.monotonic() - silence_start >= self._config.voice_silence_timeout:
                            logger.debug("Silence detected, stopping recording")
                            break

                    if rec.AcceptWaveform(audio_bytes):
                        res = json.loads(rec.Result())
                        text = res.get("text", "").strip()
                        if text:
                            result_text = text
                    else:
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        if partial_text and on_partial:
                            on_partial(partial_text)

                # Get final result
                final = json.loads(rec.FinalResult())
                final_text = final.get("text", "").strip()
                if final_text:
                    result_text = final_text

        except Exception:
            logger.exception("Error during voice recording")
            return None

        # Beep to indicate end
        self._beep_end()

        return result_text if result_text else None

    def _resolve_model_path(self) -> Path | None:
        """Find or download the Vosk model."""
        # User-specified path
        if self._config.voice_model_path:
            p = Path(self._config.voice_model_path)
            if p.is_dir():
                return p
            logger.error("Voice model path does not exist: %s", p)
            return None

        # Check default location
        model_dir = MODELS_DIR / DEFAULT_MODEL_NAME
        if model_dir.is_dir():
            return model_dir

        # Auto-download
        logger.info("Downloading Vosk model '%s'...", DEFAULT_MODEL_NAME)
        return self._download_model()

    def _download_model(self) -> Path | None:
        """Download the default Vosk model."""
        try:
            import urllib.request

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            zip_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.zip"

            urllib.request.urlretrieve(DEFAULT_MODEL_URL, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(MODELS_DIR)

            zip_path.unlink()

            model_dir = MODELS_DIR / DEFAULT_MODEL_NAME
            if model_dir.is_dir():
                logger.info("Model downloaded to %s", model_dir)
                return model_dir

            logger.error("Model extraction failed")
            return None
        except Exception:
            logger.exception("Failed to download Vosk model")
            return None

    @staticmethod
    def _compute_rms(data) -> float:
        """Compute RMS energy of int16 audio data."""
        import numpy as np
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    @staticmethod
    def _beep_start() -> None:
        """Beep to indicate listening started."""
        if sys.platform == "win32":
            try:
                import winsound
                winsound.Beep(1000, 200)
            except Exception:
                pass

    @staticmethod
    def _beep_end() -> None:
        """Beep to indicate listening ended."""
        if sys.platform == "win32":
            try:
                import winsound
                winsound.Beep(800, 150)
            except Exception:
                pass
