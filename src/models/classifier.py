from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.models.masking import MaskedSequenceClassifier


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def build_model(config: dict) -> MaskedSequenceClassifier:
    model_name = config["model"]["backbone_name"]
    num_labels = int(config["model"].get("num_labels", 2))
    return MaskedSequenceClassifier.from_pretrained(model_name_or_path=model_name, num_labels=num_labels)


def save_model_bundle(model: MaskedSequenceClassifier, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path) -> None:
    model.save_pretrained(output_dir, save_tokenizer=tokenizer)
