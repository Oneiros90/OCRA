from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

# QLocale is intentionally avoided to prevent forcing Qt initialization before the GUI boots.
QLocale = None


class Localizer:
    """Simple JSON-based localization helper."""

    def __init__(self, translations_dir: Path, fallback_locale: str = "en") -> None:
        self._dir = translations_dir
        self._fallback = fallback_locale
        self._cache: Dict[str, Dict[str, str]] = {}
        self._active_locale = fallback_locale
        self._active_map: Dict[str, str] = {}
        self.set_locale(self._detect_locale())

    def _detect_locale(self) -> str:
        env_locale = self._locale_from_env()
        if env_locale:
            return env_locale
        apple_locale = self._locale_from_defaults()
        if apple_locale:
            return apple_locale
        return self._fallback

    def _locale_from_env(self) -> str | None:
        system_locale = (
            os.environ.get("LC_ALL")
            or os.environ.get("LC_MESSAGES")
            or os.environ.get("LANG")
        )
        if system_locale:
            raw = system_locale.strip().upper()
            if raw in {"C", "C.UTF-8", "POSIX"}:
                return None
            normalized = self._normalize_locale(system_locale)
            if normalized and normalized not in {"c", "posix"}:
                return normalized
        return None

    def _locale_from_defaults(self) -> str | None:
        try:
            result = subprocess.run(
                ["/usr/bin/defaults", "read", "-g", "AppleLocale"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return self._normalize_locale(result.stdout.strip())

    def _normalize_locale(self, value: str | None) -> str | None:
        if not value:
            return None
        token = value.split("_")[0].split("-")[0].lower().strip()
        return token or None

    def _load_locale(self, locale_code: str) -> Dict[str, str]:
        if locale_code not in self._cache:
            file_path = self._dir / f"{locale_code}.json"
            if file_path.exists():
                self._cache[locale_code] = json.loads(file_path.read_text(encoding="utf-8"))
            else:
                self._cache[locale_code] = {}
        return self._cache[locale_code]

    def set_locale(self, locale_code: str | None) -> None:
        normalized = (locale_code or self._fallback).split("-")[0].lower()
        data = self._load_locale(normalized)
        if not data:
            normalized = self._fallback
            data = self._load_locale(self._fallback)
        self._active_locale = normalized
        self._active_map = data

    def translate(self, key: str, **kwargs: Any) -> str:
        template = self._active_map.get(key)
        if template is None and self._active_locale != self._fallback:
            template = self._load_locale(self._fallback).get(key)
        if template is None:
            return key
        if kwargs:
            return template.format(**kwargs)
        return template


_singleton: Localizer | None = None


def _get_localizer() -> Localizer:
    global _singleton
    if _singleton is None:
        base = Path(__file__).resolve().parent / "translations"
        _singleton = Localizer(base)
    return _singleton


def tr(key: str, **kwargs: Any) -> str:
    return _get_localizer().translate(key, **kwargs)
