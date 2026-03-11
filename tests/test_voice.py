"""Tests for voice capture with mocked sounddevice and vosk."""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from gesture_orchestrator.config import GestureConfig
from gesture_orchestrator.voice import VoiceCapture


@pytest.fixture
def config():
    return GestureConfig(
        voice_enabled=True,
        voice_timeout=5.0,
        voice_silence_timeout=1.0,
        voice_sample_rate=16000,
        voice_energy_threshold=300.0,
    )


class TestVoiceCaptureAvailability:
    def test_not_available_without_sounddevice(self, config):
        vc = VoiceCapture(config)
        with patch.dict("sys.modules", {"sounddevice": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # Force re-check
                vc._available = None
                assert vc.is_available() is False

    def test_not_available_without_input_devices(self, config):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 0}]
        mock_vosk = MagicMock()

        vc = VoiceCapture(config)
        with patch.dict("sys.modules", {"sounddevice": mock_sd, "vosk": mock_vosk}):
            with patch.object(vc, "_resolve_model_path", return_value=None):
                vc._available = None
                result = vc.is_available()
                # No input devices → not available
                assert result is False

    def test_available_with_mic_and_model(self, config):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        mock_vosk = MagicMock()
        mock_model = MagicMock()
        mock_vosk.Model.return_value = mock_model

        vc = VoiceCapture(config)

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "vosk": mock_vosk}):
            with patch.object(vc, "_resolve_model_path", return_value="/tmp/model"):
                vc._available = None
                result = vc.is_available()
                assert result is True
                assert vc._model is mock_model


class TestVoiceCaptureListening:
    def test_listen_returns_none_when_not_available(self, config):
        vc = VoiceCapture(config)
        vc._available = False
        assert vc.listen() is None

    def test_listen_transcribes_audio(self, config):
        config.voice_silence_timeout = 0.0
        vc = VoiceCapture(config)
        vc._available = True

        mock_recognizer = MagicMock()
        call_count = [0]

        def accept_waveform(data):
            call_count[0] += 1
            return call_count[0] >= 2

        mock_recognizer.AcceptWaveform = accept_waveform
        mock_recognizer.PartialResult.return_value = json.dumps({"partial": "hello"})
        mock_recognizer.Result.return_value = json.dumps({"text": "hello world"})
        mock_recognizer.FinalResult.return_value = json.dumps({"text": "hello world"})

        loud_audio = np.full(4000, 500, dtype=np.int16)
        silent_audio = np.zeros(4000, dtype=np.int16)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [
            (loud_audio, False),
            (loud_audio, False),
            (silent_audio, False),
            (silent_audio, False),
        ]
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vosk = MagicMock()
        mock_vosk.KaldiRecognizer.return_value = mock_recognizer

        vc._model = MagicMock()

        # Patch the modules so the lazy imports inside listen() get our mocks
        import sys as _sys
        old_sd = _sys.modules.get("sounddevice")
        old_vosk = _sys.modules.get("vosk")
        _sys.modules["sounddevice"] = mock_sd
        _sys.modules["vosk"] = mock_vosk
        try:
            with patch.object(vc, "_beep_start"), patch.object(vc, "_beep_end"):
                result = vc.listen(timeout=5.0)
        finally:
            if old_sd is not None:
                _sys.modules["sounddevice"] = old_sd
            else:
                _sys.modules.pop("sounddevice", None)
            if old_vosk is not None:
                _sys.modules["vosk"] = old_vosk
            else:
                _sys.modules.pop("vosk", None)

        assert result == "hello world"

    def test_listen_calls_on_partial(self, config):
        vc = VoiceCapture(config)
        vc._available = True

        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = json.dumps({"partial": "testing"})
        mock_recognizer.FinalResult.return_value = json.dumps({"text": ""})

        mock_vosk = MagicMock()
        mock_vosk.KaldiRecognizer.return_value = mock_recognizer

        mock_sd = MagicMock()
        mock_stream = MagicMock()
        audio = np.full(4000, 500, dtype=np.int16)
        mock_stream.read.side_effect = [(audio, False)] * 3
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_sd.InputStream.return_value = mock_stream

        vc._model = MagicMock()

        partial_results = []

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "vosk": mock_vosk}):
            with patch.object(vc, "_beep_start"), patch.object(vc, "_beep_end"):
                vc.listen(timeout=0.1, on_partial=lambda t: partial_results.append(t))

        assert len(partial_results) > 0
        assert "testing" in partial_results

    def test_listen_returns_none_on_silence(self, config):
        vc = VoiceCapture(config)
        vc._available = True

        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = json.dumps({"partial": ""})
        mock_recognizer.FinalResult.return_value = json.dumps({"text": ""})

        mock_vosk = MagicMock()
        mock_vosk.KaldiRecognizer.return_value = mock_recognizer

        mock_sd = MagicMock()
        mock_stream = MagicMock()
        silent = np.zeros(4000, dtype=np.int16)
        mock_stream.read.side_effect = [(silent, False)] * 5
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_sd.InputStream.return_value = mock_stream

        vc._model = MagicMock()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "vosk": mock_vosk}):
            with patch.object(vc, "_beep_start"), patch.object(vc, "_beep_end"):
                result = vc.listen(timeout=0.1)

        assert result is None


class TestRMSComputation:
    def test_rms_of_silence(self):
        data = np.zeros(1000, dtype=np.int16)
        assert VoiceCapture._compute_rms(data) == 0.0

    def test_rms_of_loud_signal(self):
        data = np.full(1000, 1000, dtype=np.int16)
        assert VoiceCapture._compute_rms(data) == pytest.approx(1000.0, abs=1.0)

    def test_rms_of_empty(self):
        data = np.array([], dtype=np.int16)
        assert VoiceCapture._compute_rms(data) == 0.0
