from __future__ import annotations

from typing import Callable

from src.utils.io import read_text_lines


def load_blacklist(path: str) -> list[str]:
    return read_text_lines(path)


def deduplicate_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def filter_by_length(texts: list[str], min_length: int, max_length: int) -> list[str]:
    return [text for text in texts if min_length <= len(text) <= max_length]


def filter_by_blacklist(texts: list[str], blacklist: list[str]) -> list[str]:
    if not blacklist:
        return texts
    return [text for text in texts if not any(token in text for token in blacklist)]


def filter_by_predictor(
    texts: list[str],
    predictor: Callable[[list[str]], list[dict]],
    toxic_threshold: float,
) -> list[str]:
    if not texts:
        return []
    predictions = predictor(texts)
    return [text for text, row in zip(texts, predictions) if float(row["prob_toxic"]) < toxic_threshold]


def clean_generated_texts(
    texts: list[str],
    min_length: int,
    max_length: int,
    blacklist: list[str] | None = None,
    predictor: Callable[[list[str]], list[dict]] | None = None,
    toxic_threshold: float = 0.5,
) -> list[str]:
    cleaned = deduplicate_texts(texts)
    cleaned = filter_by_length(cleaned, min_length, max_length)
    cleaned = filter_by_blacklist(cleaned, blacklist or [])
    if predictor is not None:
        cleaned = filter_by_predictor(cleaned, predictor, toxic_threshold)
    return cleaned
