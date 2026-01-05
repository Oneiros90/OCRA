"""Main Qt window that orchestrates OCR and translation flows."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QColor, QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
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
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ocr_ai.constants import DEFAULT_TARGET_LANG
from ocr_ai.ocr_engine import OcrRegion

from .tasks import OcrPayload, TranslationPayload, perform_ocr_task, perform_translation_task
from .worker import Worker
from .widgets.image_canvas import ImageCanvas

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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
        self._ocr_ready = False
        self._is_busy = False
        self._translation_overlay_enabled = True
        self._font_color = QColor(Qt.white)
        self._font_color.setAlpha(255)
        self._fill_color = QColor(30, 136, 229)
        self._fill_color.setAlpha(90)
        self._build_ui()
        self.image_canvas.set_overlay_style(
            text_color=self._font_color,
            fill_color=self._fill_color,
        )
        self.image_canvas.clear()
        self._update_panel_states()

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

        self.open_button = QPushButton("Apri immagine...")
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)

        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Zoom")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._handle_zoom_slider_change)
        self.zoom_value_label = QLabel("100%")
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_value_label)
        layout.addLayout(zoom_row)

        self.ocr_group = self._build_ocr_group()
        layout.addWidget(self.ocr_group)

        self.translation_group = self._build_translation_group()
        layout.addWidget(self.translation_group)

        self.export_image_btn = QPushButton("Esporta immagine annotata...")
        self.export_image_btn.clicked.connect(self._export_canvas_snapshot)
        layout.addWidget(self.export_image_btn)

        layout.addStretch(1)
        return container

    def _build_ocr_group(self) -> QGroupBox:
        group = QGroupBox("Riconoscimento OCR")
        layout = QVBoxLayout()
        form = QFormLayout()

        self.doc_lang_combo = QComboBox()
        self._populate_language_combo(self.doc_lang_combo, default_code="en")
        form.addRow("Lingua documento", self.doc_lang_combo)

        self.allow_gpu = QCheckBox("Consenti uso GPU se disponibile")
        form.addRow("Hardware", self.allow_gpu)

        self.paragraph_mode = QCheckBox("Raggruppa testo in paragrafi")
        self.paragraph_mode.setChecked(True)
        form.addRow("Layout OCR", self.paragraph_mode)

        self.run_ocr_btn = QPushButton("Esegui OCR")
        self.run_ocr_btn.clicked.connect(self.run_ocr)
        form.addRow(self.run_ocr_btn)

        self.toggle_overlays_btn = QPushButton("Nascondi riconoscimento")
        self.toggle_overlays_btn.setEnabled(False)
        self.toggle_overlays_btn.clicked.connect(self._toggle_overlays)
        form.addRow(self.toggle_overlays_btn)

        self.font_color_btn = QPushButton("Colore testo")
        self.font_color_btn.clicked.connect(self._choose_font_color)
        form.addRow("Colore testo", self.font_color_btn)
        self._update_color_button(self.font_color_btn, self._font_color)

        self.fill_color_btn = QPushButton("Colore riempimento")
        self.fill_color_btn.clicked.connect(self._choose_fill_color)
        form.addRow("Riempimento riquadro", self.fill_color_btn)
        self._update_color_button(self.fill_color_btn, self._fill_color)

        layout.addLayout(form)

        self.ocr_text = QPlainTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setPlaceholderText("Testo riconosciuto")
        layout.addWidget(QLabel("Testo OCR"))
        layout.addWidget(self.ocr_text, 1)

        group.setLayout(layout)
        return group

    def _build_translation_group(self) -> QGroupBox:
        group = QGroupBox("Traduzione")
        layout = QVBoxLayout()
        form = QFormLayout()

        self.target_lang_combo = QComboBox()
        self._populate_language_combo(self.target_lang_combo, default_code=DEFAULT_TARGET_LANG)
        form.addRow("Lingua destinazione", self.target_lang_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        form.addRow("OpenAI API key", self.api_key_input)

        self.attach_image_checkbox = QCheckBox("Allega immagine per il contesto")
        form.addRow("Contesto immagine", self.attach_image_checkbox)

        self.translate_btn = QPushButton("Traduci testo")
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self.run_translation)
        form.addRow(self.translate_btn)

        self.toggle_translation_btn = QPushButton("Mostra testo originale")
        self.toggle_translation_btn.setEnabled(False)
        self.toggle_translation_btn.clicked.connect(self._toggle_translation_mode)
        form.addRow(self.toggle_translation_btn)

        layout.addLayout(form)

        self.translation_text = QPlainTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setPlaceholderText("Traduzione")
        layout.addWidget(QLabel("Testo tradotto"))
        layout.addWidget(self.translation_text, 1)

        group.setLayout(layout)
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
            busy_message="OCR in corso...",
        )

    def run_translation(self) -> None:
        if not self.current_regions:
            QMessageBox.information(self, "OCR mancante", "Esegui prima l'OCR")
            return
        source = self._selected_document_language()
        target = self._selected_target_language()
        if not source or not target:
            QMessageBox.warning(self, "Lingue", "Specifica lingua documento e destinazione")
            return
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API key mancante", "Inserisci una OpenAI API key valida")
            return
        include_image = self.attach_image_checkbox.isChecked()
        image_for_prompt = self.current_image if include_image else None
        print("[GUI] run_translation invoked", flush=True)
        self._start_task(
            perform_translation_task,
            self.current_regions,
            source,
            target,
            api_key,
            include_image,
            image_for_prompt,
            on_success=self._handle_translation_success,
            busy_message="Traduzione in corso...",
        )

    def _handle_ocr_success(self, payload: OcrPayload) -> None:
        self.current_regions = payload.regions
        self._overlay_texts = None
        self.ocr_text.setPlainText(payload.combined_text)
        if self.current_regions:
            self._ocr_ready = True
            self.overlays_visible = True
            self._render_overlays()
        else:
            self._ocr_ready = False
            QMessageBox.information(self, "Nessun testo", "Non è stato trovato testo nell'immagine")
        self._update_panel_states()

    def _handle_translation_success(self, payload: TranslationPayload) -> None:
        self.translation_text.setPlainText(payload.combined_text)
        if self.current_regions:
            self._overlay_texts = payload.overlay_texts
            self._translation_overlay_enabled = True
            self._render_overlays()
            if hasattr(self, "toggle_translation_btn"):
                self.toggle_translation_btn.setEnabled(True)
                self.toggle_translation_btn.setText("Mostra testo originale")
        self._update_panel_states()

    def _reset_ocr_outputs(self) -> None:
        self.current_regions = []
        self._overlay_texts = None
        self._translation_overlay_enabled = True
        self.overlays_visible = True
        self._ocr_ready = False
        self.ocr_text.clear()
        self.translation_text.clear()
        self.translate_btn.setEnabled(False)
        self.image_canvas.clear_overlays()
        if hasattr(self, "toggle_overlays_btn"):
            self.toggle_overlays_btn.setEnabled(False)
            self.toggle_overlays_btn.setText("Nascondi riconoscimento")
        if hasattr(self, "toggle_translation_btn"):
            self.toggle_translation_btn.setEnabled(False)
            self.toggle_translation_btn.setText("Mostra testo originale")
        self._update_panel_states()

    def _prepare_for_new_ocr_run(self) -> None:
        self.translation_text.clear()
        self.translate_btn.setEnabled(False)
        self.image_canvas.clear_overlays()
        self._overlay_texts = None
        self._translation_overlay_enabled = True
        self._ocr_ready = False
        self.overlays_visible = True
        if hasattr(self, "toggle_overlays_btn"):
            self.toggle_overlays_btn.setEnabled(False)
            self.toggle_overlays_btn.setText("Nascondi riconoscimento")
        if hasattr(self, "toggle_translation_btn"):
            self.toggle_translation_btn.setEnabled(False)
            self.toggle_translation_btn.setText("Mostra testo originale")
        self._update_panel_states()

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

    def _selected_target_language(self) -> str:
        data = self.target_lang_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        fallback = self.target_lang_combo.itemData(0)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return COMMON_LANG_CHOICES[0][1]

    def _populate_language_combo(self, combo: QComboBox, default_code: str) -> None:
        combo.clear()
        matched_index = -1
        for idx, (label, code) in enumerate(COMMON_LANG_CHOICES):
            combo.addItem(label, code)
            if matched_index == -1 and code == default_code:
                matched_index = idx
        combo.setCurrentIndex(matched_index if matched_index >= 0 else 0)

    def _set_busy(self, busy: bool, message: str) -> None:
        self._is_busy = busy
        self.status_label.setText(message)
        self._update_panel_states()

    def _update_panel_states(self) -> None:
        has_image = self.image_canvas.has_image()
        has_ready_ocr = bool(self.current_regions) and self._ocr_ready
        has_translation = bool(self._overlay_texts)

        self.open_button.setEnabled(not self._is_busy)
        if hasattr(self, "zoom_slider"):
            self.zoom_slider.setEnabled(has_image and not self._is_busy)

        ocr_enabled = has_image and not self._is_busy
        translation_enabled = has_ready_ocr and not self._is_busy

        if hasattr(self, "ocr_group"):
            self.ocr_group.setEnabled(ocr_enabled)
        if hasattr(self, "translation_group"):
            self.translation_group.setEnabled(translation_enabled)

        if hasattr(self, "run_ocr_btn"):
            self.run_ocr_btn.setEnabled(has_image and not self._is_busy)
        if hasattr(self, "translate_btn"):
            self.translate_btn.setEnabled(translation_enabled)

        btn = getattr(self, "toggle_overlays_btn", None)
        if btn is not None:
            btn.setEnabled(has_ready_ocr and not self._is_busy)

        if hasattr(self, "toggle_translation_btn"):
            self.toggle_translation_btn.setEnabled(
                has_translation and has_ready_ocr and not self._is_busy
            )

        if hasattr(self, "font_color_btn"):
            self.font_color_btn.setEnabled(has_image and not self._is_busy)
        if hasattr(self, "fill_color_btn"):
            self.fill_color_btn.setEnabled(has_image and not self._is_busy)
        if hasattr(self, "export_image_btn"):
            self.export_image_btn.setEnabled(has_image)

    def _render_overlays(self) -> None:
        btn = getattr(self, "toggle_overlays_btn", None)
        if btn is None:
            return
        if not self._ocr_ready or not self.current_regions:
            self.image_canvas.clear_overlays()
            btn.setEnabled(False)
            btn.setText("Nascondi riconoscimento")
            return
        btn.setEnabled(True)
        if not self.overlays_visible:
            btn.setText("Mostra riconoscimento")
            self.image_canvas.clear_overlays()
            return
        payload = None
        if self._overlay_texts and self._translation_overlay_enabled:
            payload = self._overlay_texts
        self.image_canvas.show_regions(self.current_regions, payload)
        btn.setText("Nascondi riconoscimento")

    def _toggle_overlays(self) -> None:
        if not self._ocr_ready or not self.current_regions:
            return
        self.overlays_visible = not self.overlays_visible
        self._render_overlays()

    def _toggle_translation_mode(self) -> None:
        if not self._overlay_texts:
            return
        self._translation_overlay_enabled = not self._translation_overlay_enabled
        if self._translation_overlay_enabled:
            self.toggle_translation_btn.setText("Mostra testo originale")
        else:
            self.toggle_translation_btn.setText("Mostra traduzione")
        self._render_overlays()

    def _choose_font_color(self) -> None:
        color = QColorDialog.getColor(
            self._font_color,
            self,
            "Scegli colore del testo",
            QColorDialog.ShowAlphaChannel,
        )
        if not color.isValid():
            return
        self._font_color = color
        self._update_color_button(self.font_color_btn, color)
        self.image_canvas.set_overlay_style(text_color=color)
        self._render_overlays()

    def _choose_fill_color(self) -> None:
        color = QColorDialog.getColor(
            self._fill_color,
            self,
            "Scegli riempimento del riquadro",
            QColorDialog.ShowAlphaChannel,
        )
        if not color.isValid():
            return
        self._fill_color = color
        self.image_canvas.set_overlay_style(fill_color=color)
        self._update_color_button(self.fill_color_btn, color)
        self._render_overlays()

    def _update_color_button(self, button: QPushButton, color: QColor) -> None:
        contrast = self._contrast_color(color)
        background = self._rgba_css(color)
        button.setStyleSheet(
            f"background-color: {background}; color: {contrast}; border: 1px solid #444;"
        )

    def _contrast_color(self, color: QColor) -> str:
        if color.lightness() > 128:
            return "#000000"
        return "#ffffff"

    def _rgba_css(self, color: QColor) -> str:
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()/255:.2f})"

    def _export_canvas_snapshot(self) -> None:
        if not self.image_canvas.has_image():
            QMessageBox.information(self, "Nessuna immagine", "Carica prima un'immagine da esportare")
            return
        default_dir = self.current_image.parent if self.current_image else Path.home()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta immagine annotata",
            str(default_dir / "annotated.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not file_path:
            return
        success = self.image_canvas.export_view(Path(file_path))
        if success:
            QMessageBox.information(self, "Esportazione completata", f"File salvato in {file_path}")
        else:
            QMessageBox.warning(self, "Errore", "Impossibile salvare l'immagine annotata")

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
            self.status_label.setText("Interruzione in corso...")
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
        self._set_busy(False, "Pronto")
        QMessageBox.critical(self, "Errore", details)
