from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import easyocr

from .constants import CYRILLIC_ONLY_LANGS

LangGroup = Tuple[str, ...]
ReaderBundle = Tuple[LangGroup, easyocr.Reader]


@dataclass(frozen=True)
class OcrRegion:
    bbox: Tuple[Tuple[float, float], ...]
    text: str
    confidence: float
    source_langs: LangGroup | None = None


def build_readers(lang_codes: Sequence[str], use_gpu: bool) -> List[ReaderBundle]:
    roman_langs = [code for code in lang_codes if code not in CYRILLIC_ONLY_LANGS]
    cyrillic_langs = [code for code in lang_codes if code in CYRILLIC_ONLY_LANGS]

    readers: List[ReaderBundle] = []
    if roman_langs:
        readers.append((tuple(roman_langs), easyocr.Reader(roman_langs, gpu=use_gpu)))
    if cyrillic_langs:
        cyrillic_with_en = sorted(set(cyrillic_langs + ["en"]))
        readers.append((tuple(cyrillic_with_en), easyocr.Reader(cyrillic_with_en, gpu=use_gpu)))
    return readers


def run_ocr(readers: Sequence[ReaderBundle], image_path: Path) -> List[str]:
    aggregated: List[str] = []
    seen: set[str] = set()
    regions = run_ocr_detailed(readers, image_path)
    for region in regions:
        normalized = region.text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        aggregated.append(normalized)
    return aggregated


def run_ocr_detailed(
    readers: Sequence[ReaderBundle],
    image_path: Path,
    *,
    paragraph: bool = True,
) -> List[OcrRegion]:
    regions: List[OcrRegion] = []
    seen_polygons: set[Tuple[float, ...]] = set()
    for idx, (lang_group, reader) in enumerate(readers, start=1):
        lang_display = ", ".join(lang_group)
        print(f"Running OCR pass {idx} with languages: {lang_display}...")
        results = reader.readtext(
            str(image_path),
            detail=1,
            paragraph=paragraph,
        )
        for raw in results:
            if not isinstance(raw, (list, tuple)):
                continue
            if len(raw) == 3:
                bbox, text, confidence = raw
            elif len(raw) == 2:
                bbox, text = raw
                confidence = 1.0
            else:
                continue
            normalized = str(text).strip()
            if not normalized:
                continue
            try:
                polygon = tuple((float(x), float(y)) for x, y in bbox)
            except Exception:
                continue
            key = tuple(round(coord, 1) for point in polygon for coord in point)
            if key in seen_polygons:
                continue
            seen_polygons.add(key)
            regions.append(
                OcrRegion(
                    bbox=polygon,
                    text=normalized,
                    confidence=float(confidence),
                    source_langs=lang_group,
                )
            )
    return regions
