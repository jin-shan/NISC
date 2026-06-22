from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.utils.io import ensure_dir, resolve_path


class MaskedSequenceClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = 2) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self._mask_buffer_names: list[str] = []
        self._hook_handles: list[Any] = []
        self._install_masks()
        self._maybe_load_mask_state(model_name_or_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, num_labels: int = 2) -> "MaskedSequenceClassifier":
        return cls(model_name_or_path=model_name_or_path, num_labels=num_labels)

    def _encoder_layers(self) -> list[nn.Module]:
        for attr_name in ("bert", "roberta", "deberta", "ernie"):
            if hasattr(self.model, attr_name):
                encoder = getattr(self.model, attr_name).encoder
                if hasattr(encoder, "layer"):
                    return list(encoder.layer)
        raise ValueError("unsupported transformer backbone for masked pruning")

    def encoder_layers(self) -> list[nn.Module]:
        return self._encoder_layers()

    def _make_mask_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            mask = getattr(self, self._mask_buffer_names[layer_idx]).to(device=output.device, dtype=output.dtype)
            shape = [1] * output.ndim
            shape[-1] = mask.shape[0]
            return output * mask.view(*shape)

        return hook

    def _install_masks(self) -> None:
        self.clear_hooks()
        self._mask_buffer_names = []
        for layer_idx, layer in enumerate(self._encoder_layers()):
            intermediate_size = int(layer.intermediate.dense.out_features)
            buffer_name = f"mask_{layer_idx}"
            self.register_buffer(buffer_name, torch.ones(intermediate_size))
            self._mask_buffer_names.append(buffer_name)
            handle = layer.intermediate.register_forward_hook(self._make_mask_hook(layer_idx))
            self._hook_handles.append(handle)

    def clear_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def forward(self, **kwargs: Any) -> Any:
        return self.model(**kwargs)

    def reset_masks(self) -> None:
        for name in self._mask_buffer_names:
            getattr(self, name).fill_(1.0)

    def prune_neurons(self, layer_idx: int, neuron_indices: list[int]) -> None:
        mask = getattr(self, self._mask_buffer_names[layer_idx])
        if not neuron_indices:
            return
        index_tensor = torch.tensor(neuron_indices, dtype=torch.long, device=mask.device)
        mask.index_fill_(0, index_tensor, 0.0)

    def export_mask_state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu().clone() for name in self._mask_buffer_names}

    def load_mask_state(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if hasattr(self, name):
                getattr(self, name).copy_(tensor.to(device=getattr(self, name).device))

    def active_neuron_count(self) -> int:
        return int(sum(int(getattr(self, name).sum().item()) for name in self._mask_buffer_names))

    def total_neuron_count(self) -> int:
        return int(sum(getattr(self, name).numel() for name in self._mask_buffer_names))

    def save_pretrained(self, output_dir: str | Path, save_tokenizer=None) -> None:
        target_dir = ensure_dir(output_dir)
        self.model.save_pretrained(str(target_dir))
        torch.save(self.export_mask_state(), target_dir / "mask_state.pt")
        if save_tokenizer is not None:
            save_tokenizer.save_pretrained(str(target_dir))

    def _maybe_load_mask_state(self, model_name_or_path: str) -> None:
        candidate = resolve_path(model_name_or_path)
        mask_file = candidate / "mask_state.pt"
        if candidate.exists() and mask_file.exists():
            state = torch.load(mask_file, map_location="cpu")
            self.load_mask_state(state)
