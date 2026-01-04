from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MODEL_CHOICE,
    DEFAULT_OCR_LANGS,
    MODEL_PRESETS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from an image and translate it to Italian using an AI model.",
    )
    parser.add_argument("image", type=Path, help="Path to the input image file")
    parser.add_argument(
        "--ocr-langs",
        default=DEFAULT_OCR_LANGS,
        help=f"Comma-separated EasyOCR language codes to load (default: {DEFAULT_OCR_LANGS})",
    )
    parser.add_argument(
        "--model-choice",
        choices=sorted(MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL_CHOICE,
        help="Preset Hugging Face model to use (default: m2m100)",
    )
    parser.add_argument(
        "--model",
        help="Custom Hugging Face model id (overrides --model-choice)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum characters per translation chunk (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--source-lang",
        required=True,
        help="Language code of the detected OCR text (e.g., ru, en, fr)",
    )
    parser.set_defaults(force_cpu=True)
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        dest="force_cpu",
        help="Force CPU execution even if CUDA is available (default: enabled)",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_false",
        dest="force_cpu",
        help="Allow GPU acceleration when available",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the translated text",
    )
    return parser.parse_args()
