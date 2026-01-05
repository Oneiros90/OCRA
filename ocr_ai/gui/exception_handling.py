"""Global exception hook utilities for the GUI."""
from __future__ import annotations

import sys
import traceback
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

_GLOBAL_EXCEPTION_HOOK_INSTALLED = False


def install_global_exception_hook() -> None:
    """Ensure every uncaught exception prints to stderr and shows a popup."""
    global _GLOBAL_EXCEPTION_HOOK_INSTALLED
    if _GLOBAL_EXCEPTION_HOOK_INSTALLED:
        return
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        try:
            previous_hook(exc_type, exc_value, exc_traceback)
        except Exception:
            print(tb, file=sys.stderr, flush=True)

        app = QApplication.instance()

        if app is None:
            print(tb, file=sys.stderr, flush=True)
            return

        def _show_message() -> None:
            QMessageBox.critical(app.activeWindow() or None, "Errore inatteso", tb)

        QTimer.singleShot(0, _show_message)

    sys.excepthook = _hook
    _GLOBAL_EXCEPTION_HOOK_INSTALLED = True
