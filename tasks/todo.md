# Gesture-Powered Agent Orchestrator - Implementation

## Phase 1: Project Scaffold + Camera
- [x] Create `pyproject.toml` with deps and entry point
- [x] `config.py` - all constants (thresholds, cooldowns, camera index)
- [x] `camera.py` - OpenCV webcam wrapper
- [x] `cli.py` + `__main__.py` - main loop with argparse and shutdown

## Phase 2: Hand Detection
- [x] `detector.py` - MediaPipe Hands wrapper with inverted handedness
- [x] `overlay.py` - draw landmarks + handedness labels on frame

## Phase 3: Gesture Classification
- [x] `gestures.py` - heuristic rules (hand raised, clap)
- [x] `tests/test_gestures.py` - unit tests with fabricated landmarks

## Phase 4: Claude Code Dispatch
- [x] `dispatcher.py` - gesture → Claude CLI subprocess mapping
- [x] `tests/test_dispatcher.py` - mocked subprocess tests

## Phase 5: Polish
- [x] Graceful shutdown (SIGINT, release camera, close MediaPipe)
- [x] CLI args: --device, --cooldown, --project-dir, --debug, --no-overlay
- [x] Logging with Python logging module

## Verification
- [x] 17/17 tests passing
- [ ] Manual: webcam overlay appears (requires camera hardware)
- [ ] Manual: gesture triggers work end-to-end
