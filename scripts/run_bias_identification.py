from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from src.bias.bias_pmi import rank_candidate_words
from src.bias.false_positive_mining import select_high_confidence_false_positives
from src.bias.mask_validation import validate_bias_sources
from src.bias.word_stats import load_stopwords
from src.models.classifier import load_tokenizer
from src.models.masking import MaskedSequenceClassifier
from src.training.trainer_utils import build_text_predictor, select_device
from src.utils.io import ensure_dir, load_config, read_jsonl, resolve_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--skip-mask-validation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    prediction_path = args.predictions or f"outputs/{experiment_name}/base/dev_predictions.jsonl"
    checkpoint_dir = args.checkpoint_dir or f"outputs/{experiment_name}/base/checkpoint"
    output_dir = ensure_dir(f"outputs/{experiment_name}/bias")
    prediction_rows = read_jsonl(prediction_path)
    false_positive_rows = select_high_confidence_false_positives(
        prediction_rows,
        tau_conf=float(config["bias"]["tau_conf"]),
    )
    stopwords = load_stopwords(config["bias"]["stopwords_path"])
    candidate_rows = rank_candidate_words(
        validation_rows=prediction_rows,
        false_positive_rows=false_positive_rows,
        stopwords=stopwords,
        fmin=int(config["bias"]["fmin"]),
        top_k=int(config["bias"]["top_k"]),
        eps=float(config["bias"]["eps"]),
    )
    write_jsonl(output_dir / "false_positives.jsonl", false_positive_rows)
    write_jsonl(output_dir / "candidates.jsonl", candidate_rows)
    if args.skip_mask_validation:
        write_jsonl(output_dir / "bias_sources.jsonl", candidate_rows)
        return
    source_path = resolve_path(checkpoint_dir)
    model_source = str(source_path) if source_path.exists() else config["model"]["backbone_name"]
    tokenizer = load_tokenizer(model_source)
    model = MaskedSequenceClassifier.from_pretrained(model_source, num_labels=int(config["model"].get("num_labels", 2)))
    device = select_device(config["experiment"].get("device", "auto"))
    predictor = build_text_predictor(
        model=model,
        tokenizer=tokenizer,
        max_length=int(config["data"]["max_length"]),
        device=device,
        decision_threshold=float(config["training"].get("decision_threshold", 0.5)),
        batch_size=int(config["training"]["per_device_batch_size"]),
    )
    replacement = tokenizer.mask_token or "[MASK]"
    validated = validate_bias_sources(
        candidate_rows=candidate_rows,
        false_positive_rows=false_positive_rows,
        predictor=predictor,
        replacement=replacement,
        tau_mask=float(config["bias"]["tau_mask"]),
        tau_flip=float(config["bias"]["tau_flip"]),
    )
    write_jsonl(output_dir / "bias_sources.jsonl", validated)


if __name__ == "__main__":
    main()
