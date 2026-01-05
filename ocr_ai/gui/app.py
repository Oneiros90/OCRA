"""Application entry point for the Qt-based GUI."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .exception_handling import install_global_exception_hook
from .main_window import MainWindow


def run() -> None:
    """Launch the Qt event loop and main window."""
    app = QApplication(sys.argv)
    install_global_exception_hook()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main() -> None:
    """Alias for run(), kept for backwards compatibility."""
    run()
