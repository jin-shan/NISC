from __future__ import annotations


def compute_cps(f1: float, fpr: float, alpha: float) -> float:
    return float(alpha * f1 + (1.0 - alpha) * (1.0 - fpr))
