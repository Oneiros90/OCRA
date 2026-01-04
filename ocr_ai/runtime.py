from __future__ import annotations

from .constants import LANG_CODE_MAP, MODEL_PRESETS

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required but could not be imported: {}".format(exc))


def should_use_gpu(force_cpu: bool) -> bool:
    if force_cpu:
        return False
    return torch.cuda.is_available()


def normalize_lang_code(lang: str) -> str:
    normalized = lang.strip().lower()
    return LANG_CODE_MAP.get(normalized, normalized)


def resolve_model_name(model_choice: str, override: str | None) -> str:
    if override:
        return override
    return MODEL_PRESETS[model_choice]
