# Gesture-Powered Claude Code Agent Orchestrator

A CLI tool that uses your webcam + MediaPipe to detect hand gestures and trigger Claude Code agent commands.

## Setup

```bash
# Create and activate venv (one-time)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (one-time)
pip install pytest pytest-mock opencv-python mediapipe "numpy>=1.24,<2.0"
```

## Running the App

```bash
# Activate venv
source venv/bin/activate

# Start the gesture orchestrator
PYTHONPATH=src python -m gesture_orchestrator
```

**Gestures:**
- 🤚 Raise **left hand** (fingers extended, wrist high) → Triggers **Planner** agent
- 🤚 Raise **right hand** (fingers extended, wrist high) → Triggers **Coder** agent
- 👏 **Clap** (both palms together with velocity) → Triggers **Sync** (coordinate planning + coding)

**Controls:**
- Press `Q` or `ESC` to quit
- Sustain gesture for ~0.5 seconds to trigger
- 3-second cooldown between raises, 5-second for clap

**Options:**
```bash
PYTHONPATH=src python -m gesture_orchestrator \
  --device 0 \                    # Camera index (default: 0)
  --cooldown 3.0 \                # Cooldown in seconds (default: 3)
  --project-dir . \               # Project directory (default: cwd)
  --debug \                        # Enable debug logging
  --no-overlay                     # Disable webcam window
```

## Running Tests

```bash
# Activate venv
source venv/bin/activate

# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test file
PYTHONPATH=src pytest tests/test_gestures.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=gesture_orchestrator --cov-report=term-missing
```

**Test Status:** 17/17 passing
- 11 gesture classification tests (finger counting, hand raise detection, clap, sustain, cooldown)
- 6 dispatcher tests (Claude CLI invocation, busy flag, error handling)

## Architecture

```
gesture_orchestrator/
├── camera.py        # OpenCV webcam wrapper
├── detector.py      # MediaPipe Hands (21 landmarks per hand)
├── gestures.py      # Heuristic gesture classification
├── dispatcher.py    # Gesture → Claude CLI subprocess
├── overlay.py       # Visual feedback on camera feed
├── cli.py           # Main event loop, argparse, shutdown
└── config.py        # Thresholds, constants, landmark indices
```

## Key Features

- **Handedness inversion**: MediaPipe's camera-perspective labels are flipped so your left hand = Planner
- **Non-blocking dispatch**: Claude agent commands run in a daemon thread; doesn't block gesture detection
- **Sustain requirement**: Gesture must be held for ~15 frames (~0.5s) to prevent false triggers
- **Velocity detection for clap**: Palms must approach from distance (>0.25) to be recognized
- **Graceful shutdown**: SIGINT cleanup releases camera, closes MediaPipe, waits for running commands

## Troubleshooting

**Camera not opening?**
```bash
# Check available cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try a different device index
PYTHONPATH=src python -m gesture_orchestrator --device 1
```

**Claude CLI not found?**
Ensure `claude` is installed and on your PATH. The dispatcher runs `claude --print --agent <name> <prompt>`.

**Gestures not triggering?**
- Enable debug logging: `--debug`
- Check the overlay window for landmark visualization
- Ensure sustain time (0.5s) and cooldown (3-5s) have elapsed
