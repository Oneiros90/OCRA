"""Background tasks and payload definitions for the GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

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
