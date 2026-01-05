"""Background tasks and payload definitions for the GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from ocr_ai.llm_translation import LlmTranslator
from ocr_ai.ocr_engine import OcrRegion, build_readers, run_ocr_detailed
from ocr_ai.runtime import normalize_lang_code, should_use_gpu


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
    api_key: str,
    attach_image: bool,
    image_path: Path | None,
    progress_callback: Callable[[str], None] | None = None,
) -> TranslationPayload:
    if not regions:
        return TranslationPayload([], "")
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key is required for translation")
    key = api_key.strip()
    progress = progress_callback or (lambda _msg: None)
    print("[GUI] Starting translation task", flush=True)
    progress("Preparing translation context...")
    normalized_source = normalize_lang_code(source_lang)
    normalized_target = normalize_lang_code(target_lang)
    translator = LlmTranslator(
        api_key=key,
        target_lang=normalized_target,
        include_image=attach_image and image_path is not None,
    )
    results = translator.translate_regions(
        regions=regions,
        source_lang=normalized_source,
        image_path=image_path if attach_image else None,
        progress_callback=progress,
    )
    overlay_map = {item.box_id: item.translation for item in results}
    notes_map = {item.box_id: item.notes for item in results if item.notes}
    overlay_texts: List[str] = []
    combined_segments: List[str] = []
    for idx, region in enumerate(regions):
        translated = overlay_map.get(idx, region.text)
        overlay_texts.append(translated)
        note = notes_map.get(idx)
        combined_segments.append(f"{translated} # {note}" if note else translated)
    combined_text = "\n".join(combined_segments)
    progress("Translation completed")
    print("[GUI] Translation task completed", flush=True)
    return TranslationPayload(overlay_texts=overlay_texts, combined_text=combined_text)
