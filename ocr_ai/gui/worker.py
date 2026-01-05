"""Generic QObject-based worker for running background tasks."""
from __future__ import annotations

import inspect
import sys
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot


class Worker(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        print(f"[GUI] Worker started: {self._fn.__name__}", flush=True)
        try:
            kwargs = dict(self._kwargs)
            fn_params = inspect.signature(self._fn).parameters
            if "progress_callback" in fn_params and "progress_callback" not in kwargs:
                kwargs["progress_callback"] = self.progress.emit
            outcome = self._fn(*self._args, **kwargs)
        except Exception as exc:  # pragma: no cover - GUI feedback only
            tb = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
            print(tb, file=sys.stderr, flush=True)
            self.error.emit(tb)
        else:
            self.result.emit(outcome)
        finally:
            print(f"[GUI] Worker finished: {self._fn.__name__}", flush=True)
            self.finished.emit()
