"""Main Qt window that orchestrates OCR and translation flows."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QColor, QCloseEvent
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
from ocr_ai.document_loader import (
    DOCUMENT_EXTENSIONS,
    DocumentLoadError,
    DocumentLoadResult,
    DocumentPage,
    load_document,
)
from ocr_ai.ocr_engine import OcrRegion

from .i18n.localizer import tr
from .tasks import OcrPayload, TranslationPayload, perform_ocr_task, perform_translation_task
from .worker import Worker
from .widgets.image_canvas import ImageCanvas

COMMON_LANG_CHOICES: list[tuple[str, str]] = [
    ("languages.italian", "it"),
    ("languages.english", "en"),
    ("languages.spanish", "es"),
    ("languages.french", "fr"),
    ("languages.german", "de"),
    ("languages.portuguese", "pt"),
    ("languages.russian", "ru"),
    ("languages.ukrainian", "uk"),
    ("languages.polish", "pl"),
    ("languages.romanian", "ro"),
]

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("window.title"))
        self.resize(1400, 800)
        self.image_canvas = ImageCanvas()
        self.image_canvas.zoomChanged.connect(self._handle_canvas_zoom_change)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setWidget(self.image_canvas)
        self.status_label = QLabel(tr("status.ready"))
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
        self._font_color = QColor(Qt.GlobalColor.white)
        self._font_color.setAlpha(255)
        self._fill_color = QColor(30, 136, 229)
        self._fill_color.setAlpha(90)
        self._document_result: DocumentLoadResult | None = None
        self._document_pages: list[DocumentPage] = []
        self._current_page_index = 0
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

        self.open_button = QPushButton(tr("buttons.open_image"))
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)

        page_row = QHBoxLayout()
        self.page_selector_label = QLabel(tr("labels.page_selector"))
        self.page_selector = QComboBox()
        self.page_selector.currentIndexChanged.connect(self._handle_page_selection_change)
        self.page_selector_label.setVisible(False)
        self.page_selector.setVisible(False)
        page_row.addWidget(self.page_selector_label)
        page_row.addWidget(self.page_selector, 1)
        layout.addLayout(page_row)

        zoom_row = QHBoxLayout()
        zoom_label = QLabel(tr("labels.zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
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

        self.export_image_btn = QPushButton(tr("buttons.export_image"))
        self.export_image_btn.clicked.connect(self._export_canvas_snapshot)
        layout.addWidget(self.export_image_btn)

        layout.addStretch(1)
        return container

    def _build_ocr_group(self) -> QGroupBox:
        group = QGroupBox(tr("groups.ocr"))
        layout = QVBoxLayout()
        form = QFormLayout()

        self.doc_lang_combo = QComboBox()
        self._populate_language_combo(self.doc_lang_combo, default_code="en")
        form.addRow(tr("labels.doc_language"), self.doc_lang_combo)

        self.allow_gpu = QCheckBox(tr("checkbox.allow_gpu"))
        form.addRow(tr("labels.hardware"), self.allow_gpu)

        self.paragraph_mode = QCheckBox(tr("checkbox.paragraph"))
        self.paragraph_mode.setChecked(True)
        form.addRow(tr("labels.ocr_layout"), self.paragraph_mode)

        self.run_ocr_btn = QPushButton(tr("buttons.run_ocr"))
        self.run_ocr_btn.clicked.connect(self.run_ocr)
        form.addRow(self.run_ocr_btn)

        self.toggle_overlays_btn = QPushButton(tr("buttons.hide_overlays"))
        self.toggle_overlays_btn.setEnabled(False)
        self.toggle_overlays_btn.clicked.connect(self._toggle_overlays)
        form.addRow(self.toggle_overlays_btn)

        self.font_color_btn = QPushButton()
        self.font_color_btn.clicked.connect(self._choose_font_color)
        form.addRow(tr("labels.text_color"), self.font_color_btn)
        self._update_color_button(self.font_color_btn, self._font_color)

        self.fill_color_btn = QPushButton()
        self.fill_color_btn.clicked.connect(self._choose_fill_color)
        form.addRow(tr("labels.box_color"), self.fill_color_btn)
        self._update_color_button(self.fill_color_btn, self._fill_color)

        layout.addLayout(form)

        self.ocr_text = QPlainTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setPlaceholderText(tr("placeholders.ocr_text"))
        layout.addWidget(QLabel(tr("labels.ocr_text")))
        layout.addWidget(self.ocr_text, 1)

        group.setLayout(layout)
        return group

    def _build_translation_group(self) -> QGroupBox:
        group = QGroupBox(tr("groups.translation"))
        layout = QVBoxLayout()
        form = QFormLayout()

        self.target_lang_combo = QComboBox()
        self._populate_language_combo(self.target_lang_combo, default_code=DEFAULT_TARGET_LANG)
        form.addRow(tr("labels.target_language"), self.target_lang_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText(tr("placeholders.api_key"))
        form.addRow(tr("labels.api_key"), self.api_key_input)

        self.attach_image_checkbox = QCheckBox(tr("labels.attach_image"))
        form.addRow(tr("labels.image_context"), self.attach_image_checkbox)

        self.translate_btn = QPushButton(tr("buttons.translate"))
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self.run_translation)
        form.addRow(self.translate_btn)

        self.toggle_translation_btn = QPushButton(tr("buttons.show_original"))
        self.toggle_translation_btn.setEnabled(False)
        self.toggle_translation_btn.clicked.connect(self._toggle_translation_mode)
        form.addRow(self.toggle_translation_btn)

        layout.addLayout(form)

        self.translation_text = QPlainTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setPlaceholderText(tr("placeholders.translation_text"))
        layout.addWidget(QLabel(tr("labels.translation_text")))
        layout.addWidget(self.translation_text, 1)

        group.setLayout(layout)
        return group

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialogs.open_image.title"),
            str(Path.home()),
            self._file_dialog_filter(),
        )
        if not file_path:
            return
        try:
            result = load_document(file_path)
        except DocumentLoadError as exc:
            QMessageBox.warning(
                self,
                tr("dialogs.document_error.title"),
                tr("dialogs.document_error.body", reason=str(exc)),
            )
            return
        self._apply_document_result(result)

    def _apply_document_result(self, result: DocumentLoadResult) -> None:
        self._dispose_document_result()
        self._document_result = result
        self._document_pages = result.pages
        if not self._document_pages:
            self._clear_document_state()
            QMessageBox.warning(
                self,
                tr("dialogs.document_error.title"),
                tr("dialogs.document_error.body", reason=tr("dialogs.document_error.empty")),
            )
            return
        self._current_page_index = 0
        self._refresh_page_selector()
        self._show_page(self._current_page_index)
        self._reset_ocr_outputs()

    def _refresh_page_selector(self) -> None:
        selector = getattr(self, "page_selector", None)
        label = getattr(self, "page_selector_label", None)
        if selector is None or label is None:
            return
        has_multiple = len(self._document_pages) > 1
        selector.blockSignals(True)
        selector.clear()
        label.setVisible(has_multiple)
        selector.setVisible(has_multiple)
        if has_multiple:
            for page in self._document_pages:
                selector.addItem(page.label)
            selector.setCurrentIndex(self._current_page_index)
        selector.blockSignals(False)

    def _handle_page_selection_change(self, index: int) -> None:
        if not self._document_pages or self._is_busy:
            return
        if index < 0 or index >= len(self._document_pages):
            return
        if index == self._current_page_index:
            return
        self._current_page_index = index
        self._show_page(index)
        self._reset_ocr_outputs()

    def _show_page(self, index: int) -> None:
        page = self._document_pages[index]
        self.current_image = page.image_path
        self.image_canvas.set_image(page.pixmap)
        QTimer.singleShot(0, self._reset_zoom_to_fit)

    def _file_dialog_filter(self) -> str:
        label = tr("dialogs.open_image.filter")
        patterns = " ".join(f"*{ext}" for ext in DOCUMENT_EXTENSIONS)
        return f"{label} ({patterns})"

    def _clear_document_state(self) -> None:
        self._document_pages = []
        self._current_page_index = 0
        selector = getattr(self, "page_selector", None)
        label = getattr(self, "page_selector_label", None)
        if selector is not None:
            selector.blockSignals(True)
            selector.clear()
            selector.blockSignals(False)
            selector.setVisible(False)
            selector.setEnabled(False)
        if label is not None:
            label.setVisible(False)
            label.setEnabled(False)
        self._dispose_document_result()

    def _dispose_document_result(self) -> None:
        if self._document_result is not None:
            self._document_result.cleanup()
            self._document_result = None

    def run_ocr(self) -> None:
        if not self.current_image:
            QMessageBox.information(
                self,
                tr("dialogs.no_image.title"),
                tr("dialogs.no_image.body"),
            )
            return
        document_lang = self._selected_document_language()
        if not document_lang:
            QMessageBox.warning(
                self,
                tr("dialogs.missing_doc_lang.title"),
                tr("dialogs.missing_doc_lang.body"),
            )
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
            busy_message=tr("busy.ocr"),
        )

    def run_translation(self) -> None:
        if not self.current_regions:
            QMessageBox.information(
                self,
                tr("dialogs.no_ocr.title"),
                tr("dialogs.no_ocr.body"),
            )
            return
        source = self._selected_document_language()
        target = self._selected_target_language()
        if not source or not target:
            QMessageBox.warning(
                self,
                tr("dialogs.mismatched_languages.title"),
                tr("dialogs.mismatched_languages.body"),
            )
            return
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(
                self,
                tr("dialogs.missing_api_key.title"),
                tr("dialogs.missing_api_key.body"),
            )
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
            busy_message=tr("busy.translation"),
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
            QMessageBox.information(
                self,
                tr("dialogs.no_text_found.title"),
                tr("dialogs.no_text_found.body"),
            )
        self._update_panel_states()

    def _handle_translation_success(self, payload: TranslationPayload) -> None:
        self.translation_text.setPlainText(payload.combined_text)
        if self.current_regions:
            self._overlay_texts = payload.overlay_texts
            self._translation_overlay_enabled = True
            self._render_overlays()
            if hasattr(self, "toggle_translation_btn"):
                self.toggle_translation_btn.setEnabled(True)
                self.toggle_translation_btn.setText(tr("buttons.show_original"))
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
            self.toggle_overlays_btn.setText(tr("buttons.hide_overlays"))
        if hasattr(self, "toggle_translation_btn"):
            self.toggle_translation_btn.setEnabled(False)
            self.toggle_translation_btn.setText(tr("buttons.show_original"))
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
            self.toggle_overlays_btn.setText(tr("buttons.hide_overlays"))
        if hasattr(self, "toggle_translation_btn"):
            self.toggle_translation_btn.setEnabled(False)
            self.toggle_translation_btn.setText(tr("buttons.show_original"))
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
        worker.finished.connect(lambda: self._set_busy(False, tr("status.ready")))
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
        for idx, (label_key, code) in enumerate(COMMON_LANG_CHOICES):
            combo.addItem(tr(label_key), code)
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
        if hasattr(self, "page_selector"):
            has_multiple = len(self._document_pages) > 1
            enable_selector = has_multiple and has_image and not self._is_busy
            self.page_selector.setEnabled(enable_selector)
            if hasattr(self, "page_selector_label"):
                self.page_selector_label.setEnabled(enable_selector)

    def _render_overlays(self) -> None:
        btn = getattr(self, "toggle_overlays_btn", None)
        if btn is None:
            return
        if not self._ocr_ready or not self.current_regions:
            self.image_canvas.clear_overlays()
            btn.setEnabled(False)
            btn.setText(tr("buttons.hide_overlays"))
            return
        btn.setEnabled(True)
        if not self.overlays_visible:
            btn.setText(tr("buttons.show_overlays"))
            self.image_canvas.clear_overlays()
            return
        payload = None
        if self._overlay_texts and self._translation_overlay_enabled:
            payload = self._overlay_texts
        self.image_canvas.show_regions(self.current_regions, payload)
        btn.setText(tr("buttons.hide_overlays"))

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
            self.toggle_translation_btn.setText(tr("buttons.show_original"))
        else:
            self.toggle_translation_btn.setText(tr("buttons.show_translation"))
        self._render_overlays()

    def _choose_font_color(self) -> None:
        color = QColorDialog.getColor(
            self._font_color,
            self,
            tr("dialogs.font_color.title"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
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
            tr("dialogs.fill_color.title"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
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
        button.setFixedSize(48, 16)
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
            QMessageBox.information(
                self,
                tr("dialogs.no_image.title"),
                tr("dialogs.no_image.body"),
            )
            return
        default_dir = self.current_image.parent if self.current_image else Path.home()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("dialogs.save_image.title"),
            str(default_dir / "annotated.png"),
            tr("dialogs.save_image.filter"),
        )
        if not file_path:
            return
        success = self.image_canvas.export_view(Path(file_path))
        if success:
            QMessageBox.information(
                self,
                tr("dialogs.export.success.title"),
                tr("dialogs.export.success.body", path=file_path),
            )
        else:
            QMessageBox.warning(
                self,
                tr("dialogs.export.error.title"),
                tr("dialogs.export.error.body"),
            )

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
            self.status_label.setText(tr("status.interrupting"))
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
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
            self.status_label.setText(tr("status.ready"))
        self._threads.clear()
        self._dispose_document_result()
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
        self._set_busy(False, tr("status.ready"))
        QMessageBox.critical(self, tr("dialogs.error.title"), details)
