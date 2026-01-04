from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .constants import (
    DEFAULT_TARGET_LANG,
    MAX_GENERATION_LENGTH,
    MBART50_LANG_MAP,
)


@dataclass
class Translator:
    model_name: str
    target_lang: str = DEFAULT_TARGET_LANG
    max_length: int = MAX_GENERATION_LENGTH

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        tokenizer_name = self.tokenizer.__class__.__name__.lower()
        self._is_mbart = "mbart" in tokenizer_name

    def translate(self, chunks: Iterable[str], source_lang: str) -> List[str]:
        translations: List[str] = []
        for chunk in chunks:
            src_lang_code = self._normalize_for_tokenizer(source_lang)
            tgt_lang_code = self._normalize_for_tokenizer(self.target_lang)
            if hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = src_lang_code

            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            forced_bos_id = self._get_forced_bos_token_id(tgt_lang_code)
            generation_kwargs = {**encoded, "max_length": self.max_length}
            if forced_bos_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_id

            generated_tokens = self.model.generate(**generation_kwargs)
            translation = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            translations.extend(translation)
        return translations

    def _normalize_for_tokenizer(self, lang: str) -> str:
        lang_lower = lang.lower()
        if self._is_mbart:
            return MBART50_LANG_MAP.get(lang_lower, lang_lower)
        return lang_lower

    def _get_forced_bos_token_id(self, lang_code: str) -> int | None:
        if hasattr(self.tokenizer, "get_lang_id"):
            try:
                return self.tokenizer.get_lang_id(lang_code)
            except KeyError:
                return None
        if hasattr(self.tokenizer, "lang_code_to_id"):
            return self.tokenizer.lang_code_to_id.get(lang_code)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        return None
