#!/usr/bin/env python3
"""Desktop GUI for the OCR + translation pipeline."""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Callable, List, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PySide6.QtCore import QObject, QPointF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QCloseEvent,
    QTextDocument,
    QTextOption,
)
from PySide6.QtWidgets import (
    QScrollArea,
    QSlider,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ocr_ai.constants import DEFAULT_MAX_CHARS, DEFAULT_OCR_LANGS, DEFAULT_TARGET_LANG, MODEL_PRESETS
from ocr_ai.ocr_engine import OcrRegion, build_readers, run_ocr_detailed
from ocr_ai.runtime import normalize_lang_code, resolve_model_name, should_use_gpu
from ocr_ai.text_utils import chunk_text
from ocr_ai.translation import Translator


@dataclass
class OcrPayload:
    regions: List[OcrRegion]
    combined_text: str


@dataclass
class TranslationPayload:
    overlay_texts: List[str]
    combined_text: str


COMMON_LANG_CHOICES: list[tuple[str, str]] = [
    ("Italiano", "it"),
    ("Inglese", "en"),
    ("Spagnolo", "es"),
    ("Francese", "fr"),
    ("Tedesco", "de"),
    ("Portoghese", "pt"),
    ("Russo", "ru"),
    ("Ucraino", "uk"),
    ("Polacco", "pl"),
    ("Rumeno", "ro"),
]


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


class Worker(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(str)

    def __init__(self, fn, *args, **kwargs):
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


class ImageCanvas(QLabel):
    zoomChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 400)
        self.setStyleSheet("background-color: #0f1115; border: 1px solid #1f232b;")
        self._base_pixmap: QPixmap | None = None
        self._annotated_pixmap: QPixmap | None = None
        self._font_size = 18
        self._font_family = "Noto Sans"
        self._zoom = 1.0
        self._min_zoom = 0.25
        self._max_zoom = 4.0

    def set_image(self, pixmap: QPixmap) -> None:
        self._base_pixmap = pixmap
        self._annotated_pixmap = pixmap
        self._zoom = 1.0
        self._update_scaled()
        self.zoomChanged.emit(self._zoom)

    def show_regions(self, regions: Sequence[OcrRegion], overlay_texts: Sequence[str] | None = None) -> None:
        if not self._base_pixmap:
            return
        pixmap = self._base_pixmap.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for idx, region in enumerate(regions):
            if overlay_texts is not None and idx < len(overlay_texts):
                text = overlay_texts[idx]
            else:
                text = region.text
            polygon = QPolygonF([QPointF(x, y) for x, y in region.bbox])
            pen = QPen(QColor(30, 136, 229))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(QColor(30, 136, 229, 90))
            painter.drawPolygon(polygon)
            painter.setPen(Qt.white)
            fitted_font = self._font_for_rect(polygon.boundingRect(), text)
            painter.setFont(fitted_font)
            painter.drawText(
                polygon.boundingRect(),
                Qt.AlignCenter | Qt.TextWordWrap,
                text,
            )
        painter.end()
        self._annotated_pixmap = pixmap
        self._update_scaled()

    def set_zoom(self, zoom: float) -> None:
        if not self._base_pixmap:
            return
        clamped = max(self._min_zoom, min(self._max_zoom, zoom))
        if abs(clamped - self._zoom) < 1e-3:
            return
        self._zoom = clamped
        self._update_scaled()
        self.zoomChanged.emit(self._zoom)

    def zoom_factor(self) -> float:
        return self._zoom

    def fit_to_viewport(self, viewport_size: QSize) -> None:
        pixmap = self._annotated_pixmap or self._base_pixmap
        if not pixmap or viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return
        w_ratio = viewport_size.width() / pixmap.width()
        h_ratio = viewport_size.height() / pixmap.height()
        target = min(1.0, w_ratio, h_ratio)
        self._zoom = max(self._min_zoom, min(self._max_zoom, target))
        self._update_scaled()
        self.zoomChanged.emit(self._zoom)

    def has_image(self) -> bool:
        return self._base_pixmap is not None

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 1.1 if delta > 0 else 0.9
            self.set_zoom(self._zoom * step)
            event.accept()
            return
        super().wheelEvent(event)

    def _update_scaled(self) -> None:
        pixmap = self._annotated_pixmap or self._base_pixmap
        if not pixmap:
            self.clear()
            self.setMinimumSize(500, 400)
            return
        target_w = max(1, int(pixmap.width() * self._zoom))
        target_h = max(1, int(pixmap.height() * self._zoom))
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())

    def clear_overlays(self) -> None:
        if self._base_pixmap:
            self._annotated_pixmap = self._base_pixmap
            self._update_scaled()
        else:
            self._annotated_pixmap = None
            self.clear()

    def _font_for_rect(self, rect, text: str) -> QFont:
        text = text.strip() or "?"
        min_size = 8
        max_dim = max(12, min(rect.width(), rect.height(), 400))
        max_size = int(max(self._font_size * 3, max_dim))
        option = QTextOption()
        option.setAlignment(Qt.AlignCenter)
        option.setWrapMode(QTextOption.WordWrap)
        doc = QTextDocument()
        doc.setDefaultTextOption(option)
        doc.setPlainText(text)
        doc.setDocumentMargin(0)
        doc.setTextWidth(rect.width())

        def fits(size: int) -> bool:
            font = QFont(self._font_family, size)
            doc.setDefaultFont(font)
            layout_size = doc.size()
            return layout_size.width() <= rect.width() * 1.02 and layout_size.height() <= rect.height() * 0.98

        best = min_size
        low, high = min_size, max_size
        while low <= high:
            mid = (low + high) // 2
            if fits(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return QFont(self._font_family, best)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCR AI Translator")
        self.resize(1400, 800)
        self.image_canvas = ImageCanvas()
        self.image_canvas.zoomChanged.connect(self._handle_canvas_zoom_change)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setWidget(self.image_canvas)
        self.status_label = QLabel("Pronto")
        self.statusBar().addPermanentWidget(self.status_label)
        self.current_image: Path | None = None
        self.current_regions: List[OcrRegion] = []
        self._threads: List[QThread] = []
        self._workers: List[Worker] = []
        self._overlay_texts: List[str] | None = None
        self.overlays_visible = True
        self._build_ui()
        self.image_canvas.clear()

    def _build_ui(self) -> None:
        splitter = QSplitter()
        splitter.addWidget(self.image_scroll)
        splitter.addWidget(self._build_controls())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self.open_button = QPushButton("Apri immagine…")
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)

        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._handle_zoom_slider_change)
        self.zoom_value_label = QLabel("100%")
        self.zoom_reset_btn = QPushButton("Adatta")
        self.zoom_reset_btn.clicked.connect(self._reset_zoom_to_fit)
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_value_label)
        zoom_row.addWidget(self.zoom_reset_btn)
        layout.addLayout(zoom_row)

        self.toggle_overlays_btn = QPushButton("Nascondi riquadri")
        self.toggle_overlays_btn.setEnabled(False)
        self.toggle_overlays_btn.clicked.connect(self._toggle_overlays)
        layout.addWidget(self.toggle_overlays_btn)

        layout.addWidget(self._build_ocr_group())
        layout.addWidget(self._build_translation_group())

        self.ocr_text = QPlainTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setPlaceholderText("Testo riconosciuto")

        self.translation_text = QPlainTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setPlaceholderText("Traduzione")

        layout.addWidget(QLabel("Testo OCR"))
        layout.addWidget(self.ocr_text, 1)
        layout.addWidget(QLabel("Traduzione"))
        layout.addWidget(self.translation_text, 1)
        layout.addStretch(1)
        return container

    def _build_ocr_group(self) -> QGroupBox:
        group = QGroupBox("Riconoscimento")
        form = QFormLayout()

        self.doc_lang_combo = QComboBox()
        primary_lang = next((chunk.strip() for chunk in DEFAULT_OCR_LANGS.split(",") if chunk.strip()), None)
        matched_index = -1
        for idx, (label, code) in enumerate(COMMON_LANG_CHOICES):
            self.doc_lang_combo.addItem(f"{label} ({code})", code)
            if matched_index == -1 and primary_lang and code == primary_lang:
                matched_index = idx
        if matched_index >= 0:
            self.doc_lang_combo.setCurrentIndex(matched_index)
        else:
            self.doc_lang_combo.setCurrentIndex(0)
        form.addRow("Lingua documento", self.doc_lang_combo)

        self.allow_gpu = QCheckBox("Consenti uso GPU se disponibile")
        form.addRow("Hardware", self.allow_gpu)

        self.paragraph_mode = QCheckBox("Raggruppa testo in paragrafi")
        self.paragraph_mode.setChecked(True)
        form.addRow("Layout OCR", self.paragraph_mode)

        self.run_ocr_btn = QPushButton("Esegui OCR")
        self.run_ocr_btn.setEnabled(False)
        self.run_ocr_btn.clicked.connect(self.run_ocr)
        form.addRow(self.run_ocr_btn)

        group.setLayout(form)
        return group

    def _build_translation_group(self) -> QGroupBox:
        group = QGroupBox("Traduzione")
        form = QFormLayout()

        self.model_choice = QComboBox()
        self.model_choice.addItems(sorted(MODEL_PRESETS.keys()))
        form.addRow("Preset modello", self.model_choice)

        self.custom_model = QLineEdit()
        self.custom_model.setPlaceholderText("ID personalizzato Hugging Face")
        form.addRow("Modello custom", self.custom_model)

        self.max_chars = QSpinBox()
        self.max_chars.setRange(100, 2000)
        self.max_chars.setValue(DEFAULT_MAX_CHARS)
        form.addRow("Max caratteri chunk", self.max_chars)

        self.target_lang = QLineEdit(DEFAULT_TARGET_LANG)
        form.addRow("Lingua destinazione", self.target_lang)

        self.translate_btn = QPushButton("Traduci testo")
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self.run_translation)
        form.addRow(self.translate_btn)

        group.setLayout(form)
        return group

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona immagine", str(Path.home()))
        if not file_path:
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Errore", "Immagine non valida o non supportata")
            return
        self.current_image = Path(file_path)
        self.image_canvas.set_image(pixmap)
        self.run_ocr_btn.setEnabled(True)
        QTimer.singleShot(0, self._reset_zoom_to_fit)
        self._reset_ocr_outputs()

    def run_ocr(self) -> None:
        if not self.current_image:
            QMessageBox.information(self, "Immagine mancante", "Carica prima un'immagine")
            return
        document_lang = self._selected_document_language()
        if not document_lang:
            QMessageBox.warning(self, "Lingua documento", "Scegli o inserisci la lingua del documento")
            return
        print("[GUI] run_ocr invoked", flush=True)
        self._prepare_for_new_ocr_run()
        self._start_task(
            perform_ocr_task,
            self.current_image,
            [document_lang],
            not self.allow_gpu.isChecked(),
            self.paragraph_mode.isChecked(),
            on_success=self._handle_ocr_success,
            busy_message="OCR in corso…",
        )

    def run_translation(self) -> None:
        if not self.current_regions:
            QMessageBox.information(self, "OCR mancante", "Esegui prima l'OCR")
            return
        source = self._selected_document_language()
        target = self.target_lang.text().strip()
        if not source or not target:
            QMessageBox.warning(self, "Lingue", "Specifica lingua documento e destinazione")
            return
        print("[GUI] run_translation invoked", flush=True)
        self._start_task(
            perform_translation_task,
            self.current_regions,
            source,
            target,
            self.model_choice.currentText(),
            self.custom_model.text().strip() or None,
            self.max_chars.value(),
            on_success=self._handle_translation_success,
            busy_message="Traduzione in corso…",
        )

    def _handle_ocr_success(self, payload: OcrPayload) -> None:
        self.current_regions = payload.regions
        self._overlay_texts = None
        self.ocr_text.setPlainText(payload.combined_text)
        if self.current_regions:
            self._render_overlays()
            self.translate_btn.setEnabled(self.current_image is not None)
        else:
            QMessageBox.information(self, "Nessun testo", "Non è stato trovato testo nell'immagine")

    def _handle_translation_success(self, payload: TranslationPayload) -> None:
        self.translation_text.setPlainText(payload.combined_text)
        if self.current_regions:
            self._overlay_texts = payload.overlay_texts
            self._render_overlays()

    def _reset_ocr_outputs(self) -> None:
        self.current_regions = []
        self._overlay_texts = None
        self.overlays_visible = True
        self.ocr_text.clear()
        self.translation_text.clear()
        self.translate_btn.setEnabled(False)
        self.image_canvas.clear_overlays()
        if hasattr(self, "toggle_overlays_btn"):
            self.toggle_overlays_btn.setEnabled(False)
            self.toggle_overlays_btn.setText("Nascondi riquadri")

    def _prepare_for_new_ocr_run(self) -> None:
        self.translation_text.clear()
        self.translate_btn.setEnabled(False)
        self.image_canvas.clear_overlays()
        self._overlay_texts = None
        if hasattr(self, "toggle_overlays_btn"):
            self.toggle_overlays_btn.setEnabled(False)

    def _start_task(self, fn, *args, on_success, busy_message: str) -> None:
        print(f"[GUI] Scheduling task: {fn.__name__}", flush=True)
        self._set_busy(True, busy_message)
        thread = QThread()
        worker = Worker(fn, *args)
        worker.moveToThread(thread)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.result.connect(on_success)
        worker.error.connect(self._handle_task_error)
        worker.progress.connect(self.status_label.setText)
        worker.finished.connect(lambda: self._set_busy(False, "Pronto"))
        worker.finished.connect(lambda: self._release_worker(worker))
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        thread.started.connect(worker.run)
        thread.start()
        self._threads.append(thread)
        self._workers.append(worker)
        print(f"[GUI] Active threads: {len(self._threads)}", flush=True)
        print(f"[GUI] Thread launched for {fn.__name__}", flush=True)

    def _selected_document_language(self) -> str:
        data = self.doc_lang_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        fallback = self.doc_lang_combo.itemData(0)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return COMMON_LANG_CHOICES[0][1]

    def _set_busy(self, busy: bool, message: str) -> None:
        self.centralWidget().setDisabled(busy)
        self.status_label.setText(message)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _render_overlays(self) -> None:
        if not self.current_regions:
            self.image_canvas.clear_overlays()
            self.toggle_overlays_btn.setEnabled(False)
            self.toggle_overlays_btn.setText("Nascondi riquadri")
            return
        self.toggle_overlays_btn.setEnabled(True)
        if not self.overlays_visible:
            self.toggle_overlays_btn.setText("Mostra riquadri")
            self.image_canvas.clear_overlays()
            return
        payload = self._overlay_texts if self._overlay_texts else None
        self.image_canvas.show_regions(self.current_regions, payload)
        self.toggle_overlays_btn.setText("Nascondi riquadri")

    def _toggle_overlays(self) -> None:
        if not self.current_regions:
            return
        self.overlays_visible = not self.overlays_visible
        self._render_overlays()

    def _handle_zoom_slider_change(self, value: int) -> None:
        self.zoom_value_label.setText(f"{value}%")
        self.image_canvas.set_zoom(value / 100.0)

    def _handle_canvas_zoom_change(self, factor: float) -> None:
        if not hasattr(self, "zoom_slider"):
            return
        percent = int(round(factor * 100))
        percent = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), percent))
        if self.zoom_slider.value() != percent:
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(percent)
            self.zoom_slider.blockSignals(False)
        self.zoom_value_label.setText(f"{percent}%")

    def _reset_zoom_to_fit(self) -> None:
        if not self.image_canvas.has_image():
            return
        viewport = self.image_scroll.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            QTimer.singleShot(0, self._reset_zoom_to_fit)
            return
        self.image_canvas.fit_to_viewport(viewport)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if any(thread.isRunning() for thread in self._threads):
            print("[GUI] Gracefully stopping worker threads before exit", flush=True)
            self.status_label.setText("Interruzione in corso…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            for thread in self._threads:
                if not thread.isRunning():
                    continue
                thread.requestInterruption()
                thread.quit()
                if not thread.wait(5000):
                    print("[GUI] Thread timeout, forcing terminate", flush=True)
                    thread.terminate()
                    thread.wait()
            QApplication.restoreOverrideCursor()
            self.status_label.setText("Pronto")
        self._threads.clear()
        super().closeEvent(event)

    def _cleanup_thread(self, thread: QThread) -> None:
        try:
            self._threads.remove(thread)
        except ValueError:
            return

    def _release_worker(self, worker: Worker) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            return

    def _handle_task_error(self, details: str) -> None:
        # Make sure the UI gets re-enabled even if the worker crashes before
        # emitting the "finished" signal.
        self._set_busy(False, "Pronto")
        QMessageBox.critical(self, "Errore", details)


def perform_ocr_task(
    image_path: Path,
    lang_codes: Sequence[str],
    force_cpu: bool,
    paragraph_mode: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> OcrPayload:
    progress = progress_callback or (lambda _msg: None)
    print(f"[GUI] Starting OCR task for {image_path}", flush=True)
    progress("Caricamento modelli EasyOCR…")
    use_gpu = should_use_gpu(force_cpu)
    print(f"[GUI] OCR will use GPU: {use_gpu}; languages: {lang_codes}", flush=True)
    readers = build_readers(lang_codes, use_gpu)
    print(f"[GUI] OCR readers ready ({len(readers)} bundle/s)", flush=True)
    progress("Esecuzione OCR…")
    regions = run_ocr_detailed(readers, image_path, paragraph=paragraph_mode)
    combined = "\n".join(region.text for region in regions)
    progress("OCR completato")
    print("[GUI] OCR task completed", flush=True)
    return OcrPayload(regions=regions, combined_text=combined)


def perform_translation_task(
    regions: Sequence[OcrRegion],
    source_lang: str,
    target_lang: str,
    model_choice: str,
    custom_model: str | None,
    max_chars: int,
    progress_callback: Callable[[str], None] | None = None,
) -> TranslationPayload:
    if not regions:
        return TranslationPayload([], "")
    progress = progress_callback or (lambda _msg: None)
    print("[GUI] Starting translation task", flush=True)
    progress("Caricamento modello di traduzione…")
    normalized_source = normalize_lang_code(source_lang)
    normalized_target = normalize_lang_code(target_lang)
    model_name = resolve_model_name(model_choice, custom_model)
    translator = Translator(model_name, target_lang=normalized_target)
    overlay_texts: List[str] = []
    combined_segments: List[str] = []
    total = len(regions)
    for idx, region in enumerate(regions, start=1):
        progress(f"Traduzione chunk {idx}/{total}…")
        chunks = chunk_text(region.text, max_chars)
        payload = chunks or [region.text]
        translated_segments = translator.translate(payload, source_lang=normalized_source)
        translated_text = " ".join(translated_segments).strip()
        if not translated_text:
            translated_text = region.text
        overlay_texts.append(translated_text)
        combined_segments.append(translated_text)
    combined_text = "\n".join(combined_segments)
    progress("Traduzione completata")
    print("[GUI] Translation task completed", flush=True)
    return TranslationPayload(overlay_texts=overlay_texts, combined_text=combined_text)


def main() -> None:
    app = QApplication(sys.argv)
    install_global_exception_hook()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
