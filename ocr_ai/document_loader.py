"""Document loading helpers for single images, PDFs, and DjVu files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import fitz  # PyMuPDF
from PySide6.QtGui import QImage, QPixmap

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
DOCUMENT_EXTENSIONS: tuple[str, ...] = IMAGE_EXTENSIONS + (".pdf", ".djvu")
PDF_RENDER_SCALE = 2.0
DJVU_RENDER_SCALE = 100  # percent


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be converted into pixmaps."""


@dataclass(slots=True)
class DocumentPage:
    pixmap: QPixmap
    image_path: Path
    label: str


@dataclass
class DocumentLoadResult:
    pages: list[DocumentPage]
    temp_dir: TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


def load_document(path: Path | str, pdf_scale: float = PDF_RENDER_SCALE) -> DocumentLoadResult:
    source = Path(path)
    if not source.exists():
        raise DocumentLoadError(f"File not found: {source}")
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _load_image(source)
    if suffix == ".pdf":
        return _load_pdf(source, pdf_scale)
    if suffix == ".djvu":
        return _load_djvu(source)
    pixmap = QPixmap(str(source))
    if pixmap.isNull():
        raise DocumentLoadError("Unsupported file type or unreadable image")
    return DocumentLoadResult(pages=[DocumentPage(pixmap=pixmap, image_path=source, label="Page 1")])


def _load_image(path: Path) -> DocumentLoadResult:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        raise DocumentLoadError("Invalid or corrupted image file")
    return DocumentLoadResult(pages=[DocumentPage(pixmap=pixmap, image_path=path, label="Page 1")])


def _load_pdf(path: Path, scale: float) -> DocumentLoadResult:
    try:
        document = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Unable to open PDF: {exc}") from exc
    temp_dir = TemporaryDirectory(prefix="ocra-pdf-")
    pages: list[DocumentPage] = []
    try:
        if document.page_count == 0:
            raise DocumentLoadError("The PDF does not contain any pages")
        matrix = fitz.Matrix(scale, scale)
        for idx in range(document.page_count):
            page = document.load_page(idx)
            pix = page.get_pixmap(matrix=matrix)
            pixmap = _fitz_pixmap_to_qpixmap(pix)
            image_path = Path(temp_dir.name) / f"page_{idx + 1}.png"
            if not pixmap.save(str(image_path), "PNG"):
                raise DocumentLoadError("Failed to store a rendered PDF page")
            pages.append(DocumentPage(pixmap=pixmap, image_path=image_path, label=f"Page {idx + 1}"))
    except Exception:
        temp_dir.cleanup()
        raise
    finally:
        document.close()
    return DocumentLoadResult(pages=pages, temp_dir=temp_dir)


def _load_djvu(path: Path) -> DocumentLoadResult:
    ddjvu = _require_binary("ddjvu")
    djvused = _require_binary("djvused")
    temp_dir = TemporaryDirectory(prefix="ocra-djvu-")
    pages: list[DocumentPage] = []
    try:
        safe_source = _stage_djvu_source(path, temp_dir)
        page_count = _djvu_page_count(djvused, safe_source)
        if page_count <= 0:
            raise DocumentLoadError("The DjVu file does not contain any pages")
        for page_index in range(1, page_count + 1):
            tiff_path = Path(temp_dir.name) / f"page_{page_index}.tiff"
            command = [
                ddjvu,
                "-format=tiff",
                f"-page={page_index}",
                f"-scale={DJVU_RENDER_SCALE}%",
                str(safe_source),
                str(tiff_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:  # noqa: PERF203
                details = _decode_stderr(exc.stderr)
                raise DocumentLoadError(f"ddjvu failed on page {page_index}: {details}") from exc
            pixmap = QPixmap(str(tiff_path))
            if pixmap.isNull():
                raise DocumentLoadError(f"Rendered DjVu page {page_index} is invalid")
            png_path = Path(temp_dir.name) / f"page_{page_index}.png"
            if not pixmap.save(str(png_path), "PNG"):
                raise DocumentLoadError("Failed to convert DjVu page to PNG")
            pages.append(DocumentPage(pixmap=pixmap, image_path=png_path, label=f"Page {page_index}"))
    except Exception:
        temp_dir.cleanup()
        raise
    return DocumentLoadResult(pages=pages, temp_dir=temp_dir)


def _djvu_page_count(binary: str, path: Path) -> int:
    try:
        result = subprocess.run(
            [binary, str(path), "-e", "n"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        details = _decode_stderr(exc.stderr)
        raise DocumentLoadError(f"Unable to inspect DjVu page count: {details}") from exc
    content = result.stdout.strip()
    if not content.isdigit():
        raise DocumentLoadError("Unexpected DjVu page count response")
    return int(content)


def _require_binary(name: str) -> str:
    executable = _platform_executable(name)
    bundled = _bundled_binary_path(executable)
    if bundled is not None:
        return str(bundled)
    binary = shutil.which(executable)
    if not binary:
        raise DocumentLoadError(
            f"Missing '{name}' executable. Install DjVuLibre tools or keep the bundled copy in place."
        )
    return binary


def _fitz_pixmap_to_qpixmap(pix: fitz.Pixmap) -> QPixmap:
    fmt = QImage.Format.Format_RGBA8888 if pix.alpha else QImage.Format.Format_RGB888
    image = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
    return QPixmap.fromImage(image.copy())


def _bundled_binary_path(executable: str) -> Path | None:
    vendor_dir = _vendor_djvulibre_dir()
    if vendor_dir is None:
        return None
    candidate = vendor_dir / executable
    if candidate.exists():
        return candidate
    return None


def _vendor_djvulibre_dir() -> Path | None:
    package_root = Path(__file__).resolve().parent
    search_roots = [package_root / "vendor" / "djvulibre"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_path = Path(meipass)
        search_roots.append(meipass_path / "ocr_ai" / "vendor" / "djvulibre")
        search_roots.append(meipass_path / "vendor" / "djvulibre")
    for root in search_roots:
        if root.exists():
            return root
    return None


def _platform_executable(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _decode_stderr(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, str):
        return stream.strip()
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="ignore").strip()
    return str(stream).strip()


def _stage_djvu_source(source: Path, temp_dir: TemporaryDirectory) -> Path:
    safe_name = "document.djvu"
    destination = Path(temp_dir.name) / safe_name
    try:
        shutil.copy2(source, destination)
    except OSError as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Unable to copy DjVu file: {exc}") from exc
    return destination
