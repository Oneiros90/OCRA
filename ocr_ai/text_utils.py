from __future__ import annotations

from pathlib import Path
from typing import List


def chunk_text(text: str, max_chars: int) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for line in lines:
        next_len = current_len + len(line) + 1
        if next_len > max_chars and buffer:
            chunks.append(" ".join(buffer))
            buffer = [line]
            current_len = len(line)
        else:
            buffer.append(line)
            current_len = next_len
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def save_output(path: Path, italian_text: str) -> None:
    path.write_text(italian_text, encoding="utf-8")
