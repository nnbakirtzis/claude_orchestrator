"""Allow running as `python -m gesture_orchestrator`."""

import sys

from .cli import main

sys.exit(main())
