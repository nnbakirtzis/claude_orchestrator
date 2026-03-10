"""Tests for dispatcher with mocked subprocess."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gesture_orchestrator.dispatcher import Dispatcher, AGENTS
from gesture_orchestrator.gestures import GestureType


class TestDispatcher:
    def test_dispatch_planner(self):
        dispatcher = Dispatcher(project_dir="/tmp/test")

        with patch("gesture_orchestrator.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Plan created", stderr=""
            )

            result = dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
            assert result is True

            dispatcher.wait(timeout=5.0)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "claude" in cmd[0]
            assert "--agent" in cmd
            assert "planner" in cmd
            assert call_args[1]["cwd"] == "/tmp/test"

    def test_dispatch_coder(self):
        dispatcher = Dispatcher(project_dir="/tmp/test")

        with patch("gesture_orchestrator.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Code written", stderr=""
            )

            result = dispatcher.dispatch(GestureType.CODER_ACTIVATE)
            assert result is True

            dispatcher.wait(timeout=5.0)

            cmd = mock_run.call_args[0][0]
            assert "coder" in cmd

    def test_dispatch_sync(self):
        dispatcher = Dispatcher(project_dir="/tmp/test")

        with patch("gesture_orchestrator.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Synced", stderr=""
            )

            result = dispatcher.dispatch(GestureType.SYNC_EXECUTE)
            assert result is True
            dispatcher.wait(timeout=5.0)

    def test_dispatch_none_rejected(self):
        dispatcher = Dispatcher()
        assert dispatcher.dispatch(GestureType.NONE) is False

    def test_busy_flag(self):
        dispatcher = Dispatcher()

        import threading
        barrier = threading.Barrier(2, timeout=5)

        def slow_run(*args, **kwargs):
            barrier.wait()  # Wait until test checks busy flag
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch("gesture_orchestrator.dispatcher.subprocess.run", side_effect=slow_run):
            dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)

            # Small sleep to let thread start
            import time
            time.sleep(0.1)

            assert dispatcher.busy is True

            barrier.wait()  # Release the mock
            dispatcher.wait(timeout=5.0)

    def test_handles_missing_claude(self):
        dispatcher = Dispatcher()

        with patch(
            "gesture_orchestrator.dispatcher.subprocess.run",
            side_effect=FileNotFoundError("claude not found"),
        ):
            result = dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
            assert result is True  # Dispatched to thread
            dispatcher.wait(timeout=5.0)
            # Should not crash, just log error
            assert not dispatcher.busy
