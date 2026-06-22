from __future__ import annotations

from typing import Any, Callable

from src.pruning.cps import compute_cps


def apply_ranked_pruning(model, ranking: list[dict[str, Any]], prune_count: int) -> None:
    model.reset_masks()
    grouped: dict[int, list[int]] = {}
    for row in ranking[:prune_count]:
        grouped.setdefault(int(row["layer_idx"]), []).append(int(row["neuron_idx"]))
    for layer_idx, neuron_indices in grouped.items():
        model.prune_neurons(layer_idx, neuron_indices)


def iterative_prune(
    model,
    ranking: list[dict[str, Any]],
    evaluator: Callable[[Any], dict[str, Any]],
    step_size: int,
    max_rounds: int,
    alpha: float,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    best_cps = float("-inf")
    best_round = 0
    best_mask_state = model.export_mask_state()
    total_ranked = len(ranking)
    if total_ranked == 0:
        metrics = evaluator(model)
        cps = compute_cps(metrics["f1"], metrics["fpr"], alpha)
        return {"history": [{"round": 0, "pruned": 0, "cps": cps, **metrics}], "best_round": 0, "best_cps": cps}
    for round_idx in range(1, max_rounds + 1):
        prune_count = min(round_idx * step_size, total_ranked)
        apply_ranked_pruning(model, ranking, prune_count)
        metrics = evaluator(model)
        cps = compute_cps(metrics["f1"], metrics["fpr"], alpha)
        record = {"round": round_idx, "pruned": prune_count, "cps": cps, **metrics}
        history.append(record)
        if cps > best_cps:
            best_cps = cps
            best_round = round_idx
            best_mask_state = model.export_mask_state()
    model.load_mask_state(best_mask_state)
    return {"history": history, "best_round": best_round, "best_cps": best_cps}
