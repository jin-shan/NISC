from __future__ import annotations

from typing import Any

import torch


def aggregate_bias_scores(word_differences: dict[str, dict[int, torch.Tensor]]) -> list[dict[str, Any]]:
    if not word_differences:
        return []
    layer_scores: dict[int, torch.Tensor] = {}
    word_count = len(word_differences)
    for layer_map in word_differences.values():
        for layer_idx, tensor in layer_map.items():
            if layer_idx not in layer_scores:
                layer_scores[layer_idx] = tensor.clone()
            else:
                layer_scores[layer_idx] += tensor
    ranking: list[dict[str, Any]] = []
    for layer_idx, tensor in layer_scores.items():
        averaged = tensor / max(word_count, 1)
        for neuron_idx, score in enumerate(averaged.tolist()):
            ranking.append({"layer_idx": int(layer_idx), "neuron_idx": int(neuron_idx), "score": float(score)})
    ranking.sort(key=lambda row: row["score"], reverse=True)
    return ranking
