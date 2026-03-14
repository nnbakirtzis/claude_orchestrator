"""Gesture → Claude CLI subprocess dispatch."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading

from .config import GestureConfig
from .gestures import GestureType

logger = logging.getLogger(__name__)

AGENTS = {
    "planner": {
        "description": "Plans implementation strategy",
        "prompt": "You are a planning agent. Analyze the task and create a step-by-step implementation plan.",
    },
    "coder": {
        "description": "Implements code changes",
        "prompt": "You are a coding agent. Implement changes following existing patterns.",
    },
}

# Default prompts for each gesture
DEFAULT_PROMPTS = {
    GestureType.PLANNER_ACTIVATE: "Review the current project state and create an implementation plan. Write it to tasks/todo.md.",
    GestureType.CODER_ACTIVATE: "Read tasks/todo.md and implement the next unchecked item. Mark it complete when done.",
    GestureType.SYNC_EXECUTE: "Read tasks/todo.md, review recent changes, and implement the next step. Coordinate planning and coding.",
}


class Dispatcher:
    def __init__(self, project_dir: str = ".", config: GestureConfig | None = None):
        self._project_dir = project_dir
        self._config = config or GestureConfig(project_dir=project_dir)
        self._lock = threading.Lock()
        self._busy = False
        self._current_thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def dispatch(self, gesture: GestureType, prompt: str | None = None) -> bool:
        """
        Dispatch a gesture to the appropriate Claude agent.
        Returns True if dispatched, False if busy or invalid gesture.

        If prompt is provided, it overrides the default prompt for the gesture.
        """
        if gesture == GestureType.NONE:
            return False

        with self._lock:
            if self._busy:
                logger.warning("Dispatch skipped: already running a command")
                return False
            self._busy = True

        logger.debug(
            "Dispatching gesture=%s interactive=%s project_dir=%s",
            gesture.name,
            self._config.interactive_terminal,
            self._project_dir,
        )

        if self._config.interactive_terminal:
            thread = threading.Thread(
                target=self._run_interactive,
                args=(gesture, prompt),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=self._run_background,
                args=(gesture, prompt),
                daemon=True,
            )
        thread.start()
        self._current_thread = thread
        return True

    def _build_prompt(self, gesture: GestureType, prompt: str | None) -> str:
        """Resolve the prompt to use for a gesture."""
        if prompt:
            return prompt
        return DEFAULT_PROMPTS.get(gesture, "Help me with this project.")

    def _run_interactive(self, gesture: GestureType, prompt: str | None = None) -> None:
        """Open an interactive Claude Code terminal window."""
        try:
            agent_name, resolved_prompt = self._resolve(gesture, prompt)
            logger.info("Opening interactive terminal: %s → agent=%s", gesture.name, agent_name)
            logger.debug("Prompt: %s", resolved_prompt)

            launched = self._try_windows_terminal(resolved_prompt)
            if not launched:
                launched = self._try_cmd(resolved_prompt)

            if launched:
                logger.info("Interactive terminal opened for %s", agent_name)
            else:
                logger.warning("Could not open interactive terminal, falling back to background")
                self._run_background(gesture, prompt)
                return
        except Exception:
            logger.exception("Unexpected error launching interactive terminal")
        finally:
            # Reset busy shortly after launch (user interacts with the terminal)
            with self._lock:
                self._busy = False

    def _try_windows_terminal(self, prompt: str) -> bool:
        """Try launching via Windows Terminal (wt)."""
        wt = shutil.which("wt")
        if not wt:
            logger.debug("Windows Terminal (wt) not found on PATH")
            return False

        try:
            escaped_prompt = prompt.replace('"', '\\"')
            cmd = [
                wt, "-d", self._project_dir,
                "cmd", "/k",
                f'claude "{escaped_prompt}"',
            ]
            logger.debug("Launching Windows Terminal: %s", cmd)
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return True
        except Exception:
            logger.exception("Failed to launch Windows Terminal")
            return False

    def _try_cmd(self, prompt: str) -> bool:
        """Fallback: open via cmd.exe start command."""
        try:
            escaped_prompt = prompt.replace('"', '\\"')
            cmd = f'start cmd /k "cd /d {self._project_dir} && claude \\"{escaped_prompt}\\""'
            logger.debug("Launching cmd fallback: %s", cmd)
            subprocess.Popen(cmd, shell=True)
            return True
        except Exception:
            logger.exception("Failed to launch cmd terminal")
            return False

    def _run_background(self, gesture: GestureType, prompt: str | None = None) -> None:
        """Execute the Claude CLI command in a background thread (original behavior)."""
        try:
            agent_name, resolved_prompt = self._resolve(gesture, prompt)
            agents_json = json.dumps(AGENTS)

            cmd = [
                "claude",
                "--print",
                "--agents", agents_json,
                "--agent", agent_name,
                resolved_prompt,
            ]

            logger.info("Dispatching %s → agent=%s", gesture.name, agent_name)
            logger.debug("Command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                cwd=self._project_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info("Agent %s completed successfully", agent_name)
                if result.stdout:
                    logger.info("Output:\n%s", result.stdout[:500])
            else:
                logger.error("Agent %s failed (rc=%d): %s", agent_name, result.returncode, result.stderr[:500])

        except subprocess.TimeoutExpired:
            logger.error("Agent command timed out after 300s")
        except FileNotFoundError:
            logger.error("'claude' CLI not found. Ensure Claude Code is installed and on PATH.")
        except Exception:
            logger.exception("Unexpected error in dispatch")
        finally:
            with self._lock:
                self._busy = False

    def _resolve(self, gesture: GestureType, prompt: str | None = None) -> tuple[str, str]:
        """Map gesture to (agent_name, prompt)."""
        resolved_prompt = self._build_prompt(gesture, prompt)
        if gesture == GestureType.PLANNER_ACTIVATE:
            return "planner", resolved_prompt
        elif gesture == GestureType.CODER_ACTIVATE:
            return "coder", resolved_prompt
        elif gesture == GestureType.SYNC_EXECUTE:
            return "coder", resolved_prompt
        else:
            raise ValueError(f"Unknown gesture: {gesture}")

    def wait(self, timeout: float = 10.0) -> None:
        """Wait for the current dispatch to finish."""
        if self._current_thread is not None and self._current_thread.is_alive():
            self._current_thread.join(timeout=timeout)
