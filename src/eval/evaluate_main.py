from __future__ import annotations

from typing import Any

from src.data.loaders import build_dataloader, load_examples
from src.models.classifier import load_tokenizer
from src.models.masking import MaskedSequenceClassifier
from src.training.trainer_utils import evaluate_model, prediction_rows, select_device
from src.utils.io import load_config, write_json, write_jsonl


def run_main_evaluation(config_path: str, checkpoint_dir: str, split: str = "test") -> dict[str, Any]:
    config = load_config(config_path)
    split_key = f"{split}_path"
    examples = load_examples(
        path=config["data"][split_key],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        id_column=config["data"]["id_column"],
    )
    tokenizer = load_tokenizer(checkpoint_dir)
    dataloader = build_dataloader(
        examples,
        tokenizer,
        int(config["data"]["max_length"]),
        int(config["training"]["per_device_batch_size"]),
        False,
        int(config["data"].get("num_workers", 0)),
    )
    model = MaskedSequenceClassifier.from_pretrained(checkpoint_dir, num_labels=int(config["model"].get("num_labels", 2)))
    device = select_device(config["experiment"].get("device", "auto"))
    metrics, output = evaluate_model(model, dataloader, device, float(config["training"].get("decision_threshold", 0.5)))
    write_json(f"outputs/{config['experiment']['name']}/{split}_metrics.json", metrics)
    write_jsonl(f"outputs/{config['experiment']['name']}/{split}_predictions.jsonl", prediction_rows(output))
    return metrics
