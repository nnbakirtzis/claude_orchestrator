"""Main loop, argparse, and shutdown."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

import cv2

from .camera import Camera
from .config import GestureConfig
from .detector import HandDetector
from .dispatcher import Dispatcher
from .gestures import GestureState, GestureType, classify_gesture
from .overlay import draw_overlay
from .smoothing import LandmarkSmoother

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gesture-orchestrator",
        description="Gesture-powered Claude Code agent orchestrator",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0,
        help="Cooldown between gesture triggers in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--project-dir", type=str, default=".",
        help="Project directory for Claude agent commands (default: cwd)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable webcam overlay window",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to hand_landmarker.task model file (default: auto-detect)",
    )
    parser.add_argument(
        "--no-terminal", action="store_true",
        help="Run Claude agents in background instead of opening a terminal window",
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice input for custom prompts",
    )
    parser.add_argument(
        "--voice-model", type=str, default=None,
        help="Path to Vosk speech recognition model directory",
    )
    parser.add_argument(
        "--voice-timeout", type=float, default=10.0,
        help="Maximum voice recording duration in seconds (default: 10.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = GestureConfig(
        camera_index=args.device,
        raise_cooldown=args.cooldown,
        show_overlay=not args.no_overlay,
        debug=args.debug,
        project_dir=args.project_dir,
        model_path=args.model,
        interactive_terminal=not args.no_terminal,
        voice_enabled=not args.no_voice,
        voice_model_path=args.voice_model,
        voice_timeout=args.voice_timeout,
    )

    # Initialize voice capture if enabled
    voice_capture = None
    if config.voice_enabled:
        try:
            from .voice import VoiceCapture
            voice_capture = VoiceCapture(config)
            if voice_capture.is_available():
                logger.info("Voice input enabled")
            else:
                logger.info("Voice input not available (missing mic or model)")
                voice_capture = None
        except ImportError:
            logger.info("Voice dependencies not installed (vosk/sounddevice), voice disabled")

    # Graceful shutdown
    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", sig)
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    camera = Camera(config.camera_index, config.frame_width, config.frame_height)
    detector = HandDetector(config, model_path=config.model_path)
    dispatcher = Dispatcher(config.project_dir, config=config)
    state = GestureState()

    # Per-hand landmark smoothers
    smoothers: dict[str, LandmarkSmoother] = {}

    try:
        camera.open()
        logger.info("Gesture Orchestrator started. Press Q or ESC to quit.")

        while not shutdown:
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame, retrying...")
                continue

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect hands
            hands = detector.detect(frame)

            # Apply landmark smoothing per hand
            for hand in hands:
                if hand.handedness not in smoothers:
                    smoothers[hand.handedness] = LandmarkSmoother(
                        alpha=config.landmark_smoothing_alpha
                    )
                hand.landmarks = smoothers[hand.handedness].smooth(hand.landmarks)

            # Classify gesture
            triggered = classify_gesture(hands, state, config)

            # Dispatch if triggered
            if triggered is not None:
                logger.info("Gesture triggered: %s", triggered.name)

                prompt = None
                if voice_capture is not None:
                    from .overlay import draw_listening_screen
                    logger.info("Listening for voice command...")

                    partial_text = ""

                    def on_partial(text: str) -> None:
                        nonlocal partial_text
                        partial_text = text
                        if config.show_overlay:
                            listening_frame = frame.copy()
                            draw_listening_screen(
                                listening_frame, partial_text,
                                0.0, config.voice_timeout,
                            )
                            cv2.imshow("Gesture Orchestrator", listening_frame)
                            cv2.waitKey(1)

                    prompt = voice_capture.listen(
                        timeout=config.voice_timeout,
                        on_partial=on_partial,
                    )
                    if prompt:
                        logger.info("Voice command: %s", prompt)
                    else:
                        logger.info("No voice input, using default prompt")

                dispatcher.dispatch(triggered, prompt=prompt)

            # Overlay
            if config.show_overlay:
                draw_overlay(frame, hands, state, triggered, dispatcher.busy)
                cv2.imshow("Gesture Orchestrator", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    logger.info("Quit requested via keyboard")
                    break
            else:
                # Without overlay, still need a small delay
                cv2.waitKey(1)

    except RuntimeError as e:
        logger.error("Camera error: %s", e)
        return 1
    finally:
        logger.info("Cleaning up...")
        camera.release()
        detector.close()
        dispatcher.wait(timeout=10.0)
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
