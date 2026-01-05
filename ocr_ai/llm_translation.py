"""LLM-backed translation helpers for OCR regions."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence

import requests

from .gui.i18n.localizer import tr
from .ocr_engine import OcrRegion

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a diligent professional translator. Always respond with valid JSON and"
    " keep translations faithful to the source text while improving clarity."
)


@dataclass
class TranslationResult:
    box_id: int
    translation: str
    notes: str


class LlmTranslationError(RuntimeError):
    """Raised when the OpenAI roundtrip fails or returns malformed data."""


class LlmTranslator:
    def __init__(
        self,
        api_key: str,
        *,
        target_lang: str,
        include_image: bool,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(tr("errors.translation.api_key_missing"))
        self.api_key = api_key.strip()
        self.target_lang = target_lang
        self.include_image = include_image
        self.model = model

    def translate_regions(
        self,
        regions: Sequence[OcrRegion],
        source_lang: str,
        image_path: Path | None,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> List[TranslationResult]:
        if not regions:
            return []
        progress = progress_callback or (lambda _msg: None)
        progress(tr("progress.translation.prompt"))
        combined_text = build_combined_text(regions)
        regions_json = build_regions_json(regions)
        prompt = build_prompt(source_lang, self.target_lang, combined_text, regions_json)
        messages = build_messages(prompt, self.include_image, image_path)
        progress(tr("progress.translation.request"))
        response_text = self._call_openai(messages)
        progress(tr("progress.translation.parse"))
        return parse_translation_response(response_text)

    def _call_openai(self, messages: List[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise LlmTranslationError(tr("errors.translation.network")) from exc
        if response.status_code >= 400:
            raise LlmTranslationError(
                tr(
                    "errors.translation.http_failure",
                    status=response.status_code,
                    body=response.text[:400],
                )
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LlmTranslationError(tr("errors.translation.json_invalid")) from exc
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmTranslationError(tr("errors.translation.response_missing_text")) from exc


def build_combined_text(regions: Sequence[OcrRegion]) -> str:
    return "\n".join(region.text for region in regions if region.text).strip()


def build_regions_json(regions: Sequence[OcrRegion]) -> str:
    payload: List[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        payload.append(
            {
                "box_id": idx,
                "text": region.text,
                "bbox": [{"x": round(x, 2), "y": round(y, 2)} for x, y in region.bbox],
                "confidence": round(region.confidence, 4),
                "source_langs": list(region.source_langs or []),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_prompt(
    source_lang: str,
    target_lang: str,
    combined_text: str,
    regions_json: str,
) -> str:
    instructions = (
        "Translate every OCR box into the requested target language. Keep proper nouns and"
        " layout-specific text (buttons, UI labels) natural."
    )
    format_rules = (
        "Return ONLY a JSON array. Each element must contain:"
        " box_id (int), translation (string), notes (string; use an empty string if not needed)."
    )
    context_rules = (
        "Use the full OCR transcript plus per-box metadata to resolve ambiguous fragments."
        " If you are unsure, provide your best guess and mention the uncertainty in notes."
    )
    return (
        f"Task instructions for you (in English):\n"
        f"- {instructions}\n"
        f"- {format_rules}\n"
        f"- {context_rules}\n"
        f"\nSource language hint: {source_lang}\n"
        f"Target language: {target_lang}\n"
        "Respond with the JSON array directly, no prose outside the JSON."
        f"\n\nFull OCR text:\n```text\n{combined_text or '[empty]'}\n```"
        f"\n\nOCR regions JSON:\n```json\n{regions_json}\n```"
    )


def build_messages(prompt: str, include_image: bool, image_path: Path | None) -> List[dict[str, Any]]:
    user_content: List[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if include_image and image_path:
        user_content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def encode_image_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def parse_translation_response(content: str) -> List[TranslationResult]:
    json_block = extract_json_block(content)
    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise LlmTranslationError(tr("errors.translation.response_parse_failure")) from exc
    if not isinstance(payload, list):
        raise LlmTranslationError(tr("errors.translation.payload_not_array"))
    results: List[TranslationResult] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise LlmTranslationError(tr("errors.translation.entry_not_object"))
        try:
            box_id = int(entry["box_id"])
            translation = str(entry["translation"]).strip()
            notes = str(entry.get("notes", "")).strip()
        except (KeyError, ValueError, TypeError) as exc:
            raise LlmTranslationError(
                tr("errors.translation.entry_missing_fields")
            ) from exc
        results.append(TranslationResult(box_id=box_id, translation=translation, notes=notes))
    return results


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise LlmTranslationError(tr("errors.translation.empty_response"))
    # Attempt direct parse first
    if _looks_like_json(stripped):
        return stripped
    # Try fenced code blocks
    for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL):
        candidate = block.strip()
        if candidate and _looks_like_json(candidate):
            return candidate
    # Fallback: locate first JSON array
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        if _looks_like_json(candidate):
            return candidate
    raise LlmTranslationError(tr("errors.translation.no_json_array"))


def _looks_like_json(candidate: str) -> bool:
    try:
        json.loads(candidate)
        return True
    except json.JSONDecodeError:
        return False


__all__ = [
    "LlmTranslator",
    "LlmTranslationError",
    "TranslationResult",
    "build_prompt",
    "build_regions_json",
    "build_combined_text",
    "parse_translation_response",
    "extract_json_block",
]
