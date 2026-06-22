from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TextOnlyDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def _collate_texts(tokenizer, max_length: int):
    def collate(rows: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(rows, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    return collate


def collect_toxic_contributions(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    toxic_label_id: int = 1,
) -> dict[int, torch.Tensor]:
    if not texts:
        return {}
    dataset = TextOnlyDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_texts(tokenizer, max_length))
    totals: dict[int, torch.Tensor] = {}
    sample_count = 0
    model.to(device)
    model.eval()
    layers = model.encoder_layers()
    for batch in dataloader:
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                output.retain_grad()
                captured[layer_idx] = output
                return output

            return hook

        for layer_idx, layer in enumerate(layers):
            handles.append(layer.intermediate.register_forward_hook(make_hook(layer_idx)))
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.logits[:, toxic_label_id].sum().backward()
        for handle in handles:
            handle.remove()
        for layer_idx, activation in captured.items():
            contribution = (activation.grad * activation).sum(dim=(0, 1)).detach().cpu()
            if layer_idx not in totals:
                totals[layer_idx] = contribution
            else:
                totals[layer_idx] += contribution
        sample_count += int(batch["input_ids"].shape[0])
    return {layer_idx: tensor / max(sample_count, 1) for layer_idx, tensor in totals.items()}


def differential_contributions(
    model,
    tokenizer,
    present_texts: list[str],
    masked_texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    toxic_label_id: int = 1,
) -> dict[int, torch.Tensor]:
    present = collect_toxic_contributions(model, tokenizer, present_texts, batch_size, max_length, device, toxic_label_id)
    masked = collect_toxic_contributions(model, tokenizer, masked_texts, batch_size, max_length, device, toxic_label_id)
    layer_ids = set(present) | set(masked)
    result: dict[int, torch.Tensor] = {}
    for layer_idx in layer_ids:
        present_tensor = present.get(layer_idx)
        masked_tensor = masked.get(layer_idx)
        if present_tensor is None and masked_tensor is None:
            continue
        if present_tensor is None:
            present_tensor = torch.zeros_like(masked_tensor)
        if masked_tensor is None:
            masked_tensor = torch.zeros_like(present_tensor)
        result[layer_idx] = present_tensor - masked_tensor
    return result
