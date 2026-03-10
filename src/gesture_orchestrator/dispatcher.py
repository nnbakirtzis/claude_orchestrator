"""Gesture → Claude CLI subprocess dispatch."""

from __future__ import annotations

import json
import logging
import subprocess
import threading

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
    def __init__(self, project_dir: str = "."):
        self._project_dir = project_dir
        self._lock = threading.Lock()
        self._busy = False
        self._current_thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    def dispatch(self, gesture: GestureType) -> bool:
        """
        Dispatch a gesture to the appropriate Claude agent.
        Returns True if dispatched, False if busy or invalid gesture.
        """
        if gesture == GestureType.NONE:
            return False

        if not self._lock.acquire(blocking=False):
            logger.warning("Dispatch skipped: already running a command")
            return False

        self._busy = True
        self._lock.release()

        thread = threading.Thread(
            target=self._run,
            args=(gesture,),
            daemon=True,
        )
        thread.start()
        self._current_thread = thread
        return True

    def _run(self, gesture: GestureType) -> None:
        """Execute the Claude CLI command in a background thread."""
        try:
            agent_name, prompt = self._resolve(gesture)
            agents_json = json.dumps(AGENTS)

            cmd = [
                "claude",
                "--print",
                "--agents", agents_json,
                "--agent", agent_name,
                prompt,
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
            self._busy = False

    def _resolve(self, gesture: GestureType) -> tuple[str, str]:
        """Map gesture to (agent_name, prompt)."""
        if gesture == GestureType.PLANNER_ACTIVATE:
            return "planner", DEFAULT_PROMPTS[gesture]
        elif gesture == GestureType.CODER_ACTIVATE:
            return "coder", DEFAULT_PROMPTS[gesture]
        elif gesture == GestureType.SYNC_EXECUTE:
            return "coder", DEFAULT_PROMPTS[gesture]
        else:
            raise ValueError(f"Unknown gesture: {gesture}")

    def wait(self, timeout: float = 10.0) -> None:
        """Wait for the current dispatch to finish."""
        if self._current_thread is not None and self._current_thread.is_alive():
            self._current_thread.join(timeout=timeout)
