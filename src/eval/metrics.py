from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_binary_metrics(labels: list[int], preds: list[int]) -> dict[str, Any]:
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, zero_division=0))
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {"acc": acc, "f1": f1, "fpr": fpr, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def probs_to_preds(prob_toxic: np.ndarray, threshold: float) -> list[int]:
    return (prob_toxic >= threshold).astype(int).tolist()
