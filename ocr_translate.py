#!/usr/bin/env python3
"""CLI entrypoint for OCR + AI translation to Italian."""
from __future__ import annotations

import sys

from ocr_ai.cli import parse_args
from ocr_ai.ocr_engine import build_readers, run_ocr
from ocr_ai.runtime import normalize_lang_code, resolve_model_name, should_use_gpu
from ocr_ai.text_utils import chunk_text, save_output
from ocr_ai.translation import Translator


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    langs = [lang.strip() for lang in args.ocr_langs.split(",") if lang.strip()]
    if not langs:
        raise SystemExit("Please provide at least one OCR language code")

    print(f"Loading EasyOCR with languages: {', '.join(langs)}")
    readers = build_readers(langs, should_use_gpu(args.force_cpu))
    if not readers:
        raise SystemExit("Unable to initialize EasyOCR readers with the provided languages")

    extracted_lines = run_ocr(readers, args.image)
    extracted_text = "\n".join(extracted_lines).strip()

    if not extracted_text:
        raise SystemExit("No text detected in the provided image")

    print("Detected text:\n" + extracted_text)

    source_lang = normalize_lang_code(args.source_lang)
    print(f"Using source language for translation: {source_lang}")

    chunks = chunk_text(extracted_text, max_chars=args.max_chars)
    if not chunks:
        chunks = [extracted_text]

    model_name = resolve_model_name(args.model_choice, args.model)
    print(f"Using translation model: {model_name}")

    translator = Translator(model_name)
    italian_segments = translator.translate(chunks, source_lang=source_lang)
    italian_text = "\n".join(italian_segments).strip()

    print("\nTraduzione in italiano:\n" + italian_text)

    if args.save:
        save_output(args.save, italian_text)
        print(f"Traduzione salvata in {args.save}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit("Interrotto dall'utente")
