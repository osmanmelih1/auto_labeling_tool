"""Entry point for the Sophtrun Auto Labeling Tool desktop application.

Launching the GUI through this module guarantees two things that the pipeline
modules rely on:

1. The project root is on ``sys.path``, so ``src.*`` imports resolve.
2. The current working directory is the project root, so every relative
   ``data/...`` path used by the decoupled step modules resolves correctly no
   matter where the user started the application from.

Usage:
    uv run main.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Start the PyQt6 desktop GUI and block until the window is closed.

    Returns:
        int: The Qt application exit code, suitable for ``sys.exit``.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # The step modules address the data hierarchy with relative paths by design.
    os.chdir(PROJECT_ROOT)

    from PyQt6.QtWidgets import QApplication

    from src.gui.app import AutoLabelingApp

    print(f"[*] Project root: {PROJECT_ROOT}")
    print("[*] Starting Auto Labeling Tool GUI...")

    app = QApplication(sys.argv)
    window = AutoLabelingApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
