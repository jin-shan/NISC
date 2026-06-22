from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from src.bias.mask_validation import mask_word
from src.data.loaders import build_dataloader, load_examples
from src.models.classifier import load_tokenizer
from src.models.masking import MaskedSequenceClassifier
from src.pruning.attribution import differential_contributions
from src.pruning.iterative_pruning import iterative_prune
from src.pruning.neuron_ranking import aggregate_bias_scores
from src.training.trainer_utils import evaluate_model, select_device
from src.utils.io import ensure_dir, load_config, read_jsonl, resolve_path, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--predictions")
    parser.add_argument("--bias-sources")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    checkpoint_dir = args.checkpoint_dir or f"outputs/{experiment_name}/base/checkpoint"
    prediction_path = args.predictions or f"outputs/{experiment_name}/bias/false_positives.jsonl"
    bias_source_path = args.bias_sources or f"outputs/{experiment_name}/bias/bias_sources.jsonl"
    output_dir = ensure_dir(f"outputs/{experiment_name}/pruning")
    model_source_path = resolve_path(checkpoint_dir)
    model_source = str(model_source_path) if model_source_path.exists() else config["model"]["backbone_name"]
    model = MaskedSequenceClassifier.from_pretrained(model_source, num_labels=int(config["model"].get("num_labels", 2)))
    tokenizer = load_tokenizer(model_source)
    device = select_device(config["experiment"].get("device", "auto"))
    false_positive_rows = read_jsonl(prediction_path)
    bias_source_rows = read_jsonl(bias_source_path)
    replacement = tokenizer.mask_token or "[MASK]"
    word_differences = {}
    for row in bias_source_rows:
        word = row["word"]
        present_texts = [sample["text"] for sample in false_positive_rows if word in sample["text"]]
        masked_texts = [mask_word(text, word, replacement) for text in present_texts]
        word_differences[word] = differential_contributions(
            model=model,
            tokenizer=tokenizer,
            present_texts=present_texts,
            masked_texts=masked_texts,
            batch_size=int(config["training"]["per_device_batch_size"]),
            max_length=int(config["data"]["max_length"]),
            device=device,
            toxic_label_id=1,
        )
    ranking = aggregate_bias_scores(word_differences)
    dev_examples = load_examples(
        path=config["data"]["dev_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        id_column=config["data"]["id_column"],
    )
    dev_loader = build_dataloader(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_length=int(config["data"]["max_length"]),
        batch_size=int(config["training"]["per_device_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    def evaluator(current_model):
        metrics, _ = evaluate_model(
            current_model,
            dev_loader,
            device=device,
            decision_threshold=float(config["training"].get("decision_threshold", 0.5)),
        )
        return metrics

    report = iterative_prune(
        model=model,
        ranking=ranking,
        evaluator=evaluator,
        step_size=int(config["pruning"]["step_size"]),
        max_rounds=int(config["pruning"]["max_rounds"]),
        alpha=float(config["pruning"]["alpha"]),
    )
    model.save_pretrained(output_dir / "checkpoint", save_tokenizer=tokenizer)
    write_jsonl(output_dir / "neuron_ranking.jsonl", ranking)
    write_json(output_dir / "pruning_report.json", report)


if __name__ == "__main__":
    main()
