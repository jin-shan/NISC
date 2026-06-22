from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.bias.word_stats import document_frequency


def rank_candidate_words(
    validation_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    stopwords: set[str],
    fmin: int,
    top_k: int,
    eps: float,
) -> list[dict[str, Any]]:
    fp_texts = [row["text"] for row in false_positive_rows]
    toxic_texts = [row["text"] for row in validation_rows if int(row["pred_label"]) == 1]
    non_toxic_texts = [row["text"] for row in validation_rows if int(row["pred_label"]) == 0]
    fp_counts = document_frequency(fp_texts, stopwords)
    toxic_counts = document_frequency(toxic_texts, stopwords)
    non_toxic_counts = document_frequency(non_toxic_texts, stopwords)
    toxic_total = max(len(toxic_texts), 1)
    non_toxic_total = max(len(non_toxic_texts), 1)
    ranked: list[dict[str, Any]] = []
    for word, count in fp_counts.items():
        if count < fmin:
            continue
        p_toxic = toxic_counts[word] / toxic_total
        p_non_toxic = non_toxic_counts[word] / non_toxic_total
        score = math.log((p_toxic + eps) / (p_non_toxic + eps))
        ranked.append(
            {
                "word": word,
                "fp_doc_freq": int(count),
                "toxic_doc_freq": int(toxic_counts[word]),
                "non_toxic_doc_freq": int(non_toxic_counts[word]),
                "bias_pmi": float(score),
            }
        )
    ranked.sort(key=lambda row: row["bias_pmi"], reverse=True)
    return ranked[:top_k]
