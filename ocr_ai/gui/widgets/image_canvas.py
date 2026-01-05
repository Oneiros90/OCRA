"""Canvas widget that renders OCR overlays and supports zooming."""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF, QTextDocument, QTextOption
from PySide6.QtWidgets import QLabel

from ocr_ai.ocr_engine import OcrRegion


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
