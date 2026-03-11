"""Tests for dispatcher with mocked subprocess."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gesture_orchestrator.config import GestureConfig
from gesture_orchestrator.dispatcher import Dispatcher, AGENTS
from gesture_orchestrator.gestures import GestureType


class TestDispatcher:
    """Tests for background (non-interactive) dispatch mode."""

    def _make_dispatcher(self, project_dir="/tmp/test"):
        config = GestureConfig(project_dir=project_dir, interactive_terminal=False)
        return Dispatcher(project_dir=project_dir, config=config)

    def test_dispatch_planner(self):
        dispatcher = self._make_dispatcher()

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
        dispatcher = self._make_dispatcher()

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
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Synced", stderr=""
            )

            result = dispatcher.dispatch(GestureType.SYNC_EXECUTE)
            assert result is True
            dispatcher.wait(timeout=5.0)

    def test_dispatch_none_rejected(self):
        dispatcher = self._make_dispatcher()
        assert dispatcher.dispatch(GestureType.NONE) is False

    def test_busy_flag(self):
        dispatcher = self._make_dispatcher()

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
        dispatcher = self._make_dispatcher()

        with patch(
            "gesture_orchestrator.dispatcher.subprocess.run",
            side_effect=FileNotFoundError("claude not found"),
        ):
            result = dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
            assert result is True  # Dispatched to thread
            dispatcher.wait(timeout=5.0)
            # Should not crash, just log error
            assert not dispatcher.busy

    def test_dispatch_with_custom_prompt(self):
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Done", stderr=""
            )

            result = dispatcher.dispatch(
                GestureType.PLANNER_ACTIVATE, prompt="Build a REST API"
            )
            assert result is True
            dispatcher.wait(timeout=5.0)

            cmd = mock_run.call_args[0][0]
            assert "Build a REST API" in cmd


class TestInteractiveDispatcher:
    """Tests for interactive terminal dispatch mode."""

    def _make_dispatcher(self, project_dir="/tmp/test"):
        config = GestureConfig(project_dir=project_dir, interactive_terminal=True)
        return Dispatcher(project_dir=project_dir, config=config)

    def test_dispatch_opens_terminal(self):
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.shutil.which", return_value="C:\\wt.exe"):
            with patch("gesture_orchestrator.dispatcher.subprocess.Popen") as mock_popen:
                result = dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
                assert result is True
                dispatcher.wait(timeout=5.0)

                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                assert "C:\\wt.exe" in call_args
                assert "-d" in call_args

    def test_dispatch_falls_back_to_cmd(self):
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.shutil.which", return_value=None):
            with patch("gesture_orchestrator.dispatcher.subprocess.Popen") as mock_popen:
                result = dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
                assert result is True
                dispatcher.wait(timeout=5.0)

                # Should have used shell=True (cmd fallback)
                mock_popen.assert_called_once()
                assert mock_popen.call_args[1].get("shell") is True

    def test_dispatch_with_custom_prompt(self):
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.shutil.which", return_value="C:\\wt.exe"):
            with patch("gesture_orchestrator.dispatcher.subprocess.Popen") as mock_popen:
                result = dispatcher.dispatch(
                    GestureType.CODER_ACTIVATE, prompt="Fix the login bug"
                )
                assert result is True
                dispatcher.wait(timeout=5.0)

                call_args = mock_popen.call_args[0][0]
                # The prompt should appear in the command
                cmd_str = " ".join(str(a) for a in call_args)
                assert "Fix the login bug" in cmd_str

    def test_interactive_resets_busy_after_launch(self):
        dispatcher = self._make_dispatcher()

        with patch("gesture_orchestrator.dispatcher.shutil.which", return_value="C:\\wt.exe"):
            with patch("gesture_orchestrator.dispatcher.subprocess.Popen"):
                dispatcher.dispatch(GestureType.PLANNER_ACTIVATE)
                dispatcher.wait(timeout=5.0)
                assert dispatcher.busy is False
