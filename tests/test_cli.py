"""Tests for CLI startup flow."""

import os
from unittest.mock import patch

from gesture_orchestrator.cli import _resolve_project_dir


class TestResolveProjectDir:
    """Tests for the interactive project directory prompt."""

    def test_returns_default_on_empty_input(self):
        with patch("builtins.input", return_value=""):
            result = _resolve_project_dir(".")
            assert os.path.isabs(result)
            assert result == os.path.abspath(".")

    def test_returns_absolute_path(self, tmp_path):
        with patch("builtins.input", return_value=str(tmp_path)):
            result = _resolve_project_dir(".")
            assert os.path.isabs(result)
            assert result == str(tmp_path)

    def test_rejects_invalid_then_accepts_valid(self, tmp_path):
        inputs = iter(["/nonexistent/path/abc123", str(tmp_path)])
        with patch("builtins.input", side_effect=inputs):
            result = _resolve_project_dir(".")
            assert result == str(tmp_path)

    def test_expands_tilde(self):
        home = os.path.expanduser("~")
        with patch("builtins.input", return_value="~"):
            result = _resolve_project_dir(".")
            assert result == home

    def test_uses_cli_value_as_default(self, tmp_path):
        with patch("builtins.input", return_value=""):
            result = _resolve_project_dir(str(tmp_path))
            assert result == str(tmp_path)
