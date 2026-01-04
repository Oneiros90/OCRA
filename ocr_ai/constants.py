from __future__ import annotations

from typing import Dict, Set

DEFAULT_OCR_LANGS = "ru,en,it,fr,es,de,pt"
DEFAULT_MODEL_CHOICE = "m2m100"
DEFAULT_MAX_CHARS = 600
DEFAULT_TARGET_LANG = "it"
MAX_GENERATION_LENGTH = 1024

LANG_CODE_MAP: Dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "pt-br": "pt",
}

CYRILLIC_ONLY_LANGS: Set[str] = {
    "ru",
    "rs_cyrillic",
    "be",
    "bg",
    "uk",
    "mn",
}

MODEL_PRESETS: Dict[str, str] = {
    "m2m100": "facebook/m2m100_418M",
    "marian_ru_it": "Helsinki-NLP/opus-mt-ru-it",
    "mbart50": "facebook/mbart-large-50-many-to-many-mmt",
}

MBART50_LANG_MAP: Dict[str, str] = {
    "en": "en_XX",
    "fr": "fr_XX",
    "es": "es_XX",
    "de": "de_DE",
    "it": "it_IT",
    "pt": "pt_PT",
    "ru": "ru_RU",
    "uk": "uk_UA",
    "bg": "bg_BG",
    "cs": "cs_CZ",
    "pl": "pl_PL",
    "ro": "ro_RO",
    "hu": "hu_HU",
}
