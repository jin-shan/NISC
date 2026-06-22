from __future__ import annotations

from statistics import mean
from typing import Any, Callable


def mask_word(text: str, word: str, replacement: str) -> str:
    return text.replace(word, replacement)


def validate_bias_sources(
    candidate_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    predictor: Callable[[list[str]], list[dict[str, Any]]],
    replacement: str,
    tau_mask: float,
    tau_flip: float,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for row in candidate_rows:
        word = row["word"]
        target_rows = [sample for sample in false_positive_rows if word in sample["text"]]
        if not target_rows:
            continue
        masked_texts = [mask_word(sample["text"], word, replacement) for sample in target_rows]
        masked_outputs = predictor(masked_texts)
        score_drops = [float(original["prob_toxic"]) - float(masked["prob_toxic"]) for original, masked in zip(target_rows, masked_outputs)]
        flip_ratio = mean(1.0 if int(masked["pred_label"]) == 0 else 0.0 for masked in masked_outputs)
        mean_drop = mean(score_drops)
        if mean_drop > tau_mask or flip_ratio > tau_flip:
            accepted.append(
                {
                    **row,
                    "mean_score_drop": float(mean_drop),
                    "flip_ratio": float(flip_ratio),
                }
            )
    return accepted
