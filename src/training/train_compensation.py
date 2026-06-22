from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.data.loaders import TextExample, build_dataloader, examples_from_rows, load_examples
from src.models.classifier import load_tokenizer, save_model_bundle
from src.models.masking import MaskedSequenceClassifier
from src.training.trainer_utils import evaluate_model, prediction_rows, select_device, train_model
from src.utils.io import ensure_dir, load_config, read_jsonl, write_json, write_jsonl, write_yaml
from src.utils.seed import set_seed


def mix_examples(train_examples: list[TextExample], compensation_examples: list[TextExample], seed: int) -> list[TextExample]:
    rng = random.Random(seed)
    if not compensation_examples:
        return train_examples
    sample_count = min(len(train_examples), len(compensation_examples))
    mixed_train = rng.sample(train_examples, sample_count)
    mixed_comp = rng.sample(compensation_examples, sample_count)
    return mixed_train + mixed_comp


def run_compensation_training(config_path: str, checkpoint_dir: str, compensation_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    set_seed(int(config["experiment"].get("seed", 42)))
    tokenizer = load_tokenizer(checkpoint_dir)
    train_examples = load_examples(
        path=config["data"]["train_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        id_column=config["data"]["id_column"],
    )
    dev_examples = load_examples(
        path=config["data"]["dev_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        id_column=config["data"]["id_column"],
    )
    test_examples = load_examples(
        path=config["data"]["test_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        id_column=config["data"]["id_column"],
    )
    compensation_rows = read_jsonl(compensation_path)
    compensation_examples = examples_from_rows(compensation_rows)
    mixed_train = mix_examples(train_examples, compensation_examples, seed=int(config["experiment"].get("seed", 42)))
    batch_size = int(config["training"]["per_device_batch_size"])
    max_length = int(config["data"]["max_length"])
    num_workers = int(config["data"].get("num_workers", 0))
    train_loader = build_dataloader(mixed_train, tokenizer, max_length, batch_size, True, num_workers)
    dev_loader = build_dataloader(dev_examples, tokenizer, max_length, batch_size, False, num_workers)
    test_loader = build_dataloader(test_examples, tokenizer, max_length, batch_size, False, num_workers)
    model = MaskedSequenceClassifier.from_pretrained(
        model_name_or_path=checkpoint_dir,
        num_labels=int(config["model"].get("num_labels", 2)),
    )
    device = select_device(config["experiment"].get("device", "auto"))
    output_dir = ensure_dir(Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "compensation")
    train_config = dict(config)
    train_config["training"] = dict(config["training"])
    train_config["training"]["epochs"] = int(config["compensation"].get("epochs", 2))
    train_config["training"]["learning_rate"] = float(config["compensation"].get("learning_rate", 5e-5))
    train_config["training"]["early_stopping_patience"] = int(config["compensation"].get("early_stopping_patience", 2))
    training_summary = train_model(model=model, train_dataloader=train_loader, dev_dataloader=dev_loader, config=train_config, device=device)
    decision_threshold = float(config["training"].get("decision_threshold", 0.5))
    dev_metrics, dev_output = evaluate_model(model, dev_loader, device, decision_threshold)
    test_metrics, test_output = evaluate_model(model, test_loader, device, decision_threshold)
    save_model_bundle(model=model, tokenizer=tokenizer, output_dir=output_dir / "checkpoint")
    write_yaml(output_dir / "config.yaml", config)
    write_json(output_dir / "train_summary.json", training_summary)
    write_json(output_dir / "dev_metrics.json", dev_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_jsonl(output_dir / "dev_predictions.jsonl", prediction_rows(dev_output))
    write_jsonl(output_dir / "test_predictions.jsonl", prediction_rows(test_output))
    return {
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoint"),
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
