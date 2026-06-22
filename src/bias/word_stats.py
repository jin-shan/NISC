from __future__ import annotations

import re
from collections import Counter

from src.utils.io import read_text_lines


def load_stopwords(path: str) -> set[str]:
    return set(read_text_lines(path))


def _fallback_cut(text: str) -> list[str]:
    pieces: list[str] = []
    latin_buffer: list[str] = []
    for char in text:
        if re.match(r"[A-Za-z0-9_]", char):
            latin_buffer.append(char)
            continue
        if latin_buffer:
            pieces.append("".join(latin_buffer))
            latin_buffer = []
        if re.match(r"[\u4e00-\u9fff]", char):
            pieces.append(char)
    if latin_buffer:
        pieces.append("".join(latin_buffer))
    return pieces


def segment_text(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        return [token.strip() for token in jieba.cut(text) if token.strip()]
    except Exception:
        return _fallback_cut(text)


def normalize_tokens(tokens: list[str], stopwords: set[str]) -> list[str]:
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in stopwords:
            continue
        if re.fullmatch(r"\W+", token):
            continue
        cleaned.append(token)
    return cleaned


def document_frequency(texts: list[str], stopwords: set[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        tokens = normalize_tokens(segment_text(text), stopwords)
        for token in set(tokens):
            counts[token] += 1
    return counts
