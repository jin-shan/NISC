from __future__ import annotations

from typing import Any


def select_high_confidence_false_positives(rows: list[dict[str, Any]], tau_conf: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row["label"]) != 0:
            continue
        if int(row["pred_label"]) != 1:
            continue
        score = float(row["prob_toxic"]) - float(row["prob_non_toxic"])
        if score > tau_conf:
            selected.append({**row, "score_margin": score})
    return selected
