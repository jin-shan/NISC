from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.compensation.filtering import load_blacklist
from src.compensation.generate import build_compensation_set
from src.compensation.llm_client import OpenAICompatibleClient
from src.models.classifier import load_tokenizer
from src.models.masking import MaskedSequenceClassifier
from src.training.trainer_utils import build_text_predictor, select_device
from src.utils.io import ensure_dir, load_config, read_jsonl, resolve_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bias-sources")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    bias_source_path = args.bias_sources or f"outputs/{experiment_name}/bias/bias_sources.jsonl"
    checkpoint_dir = args.checkpoint_dir or f"outputs/{experiment_name}/pruning/checkpoint"
    output_dir = ensure_dir(f"outputs/{experiment_name}/compensation_data")
    rows = read_jsonl(bias_source_path)
    words = [row["word"] for row in rows]
    blacklist = load_blacklist(config["compensation"]["toxic_blacklist_path"])
    predictor = None
    checkpoint_path = resolve_path(checkpoint_dir)
    if checkpoint_path.exists():
        tokenizer = load_tokenizer(str(checkpoint_path))
        model = MaskedSequenceClassifier.from_pretrained(str(checkpoint_path), num_labels=int(config["model"].get("num_labels", 2)))
        device = select_device(config["experiment"].get("device", "auto"))
        predictor = build_text_predictor(
            model=model,
            tokenizer=tokenizer,
            max_length=int(config["data"]["max_length"]),
            device=device,
            decision_threshold=float(config["training"].get("decision_threshold", 0.5)),
            batch_size=int(config["training"]["per_device_batch_size"]),
        )
    client = OpenAICompatibleClient(
        model_name=config["compensation"]["model_name"],
        api_key=args.api_key,
        base_url=args.base_url,
    )
    compensation_rows = build_compensation_set(
        words=words,
        client=client,
        scenario_count=int(config["compensation"]["scenario_count"]),
        samples_per_source=int(config["compensation"]["samples_per_source"]),
        min_length=int(config["compensation"]["sample_length_min"]),
        max_length=int(config["compensation"]["sample_length_max"]),
        blacklist=blacklist,
        predictor=predictor,
        toxic_threshold=float(config["training"].get("decision_threshold", 0.5)),
    )
    write_jsonl(output_dir / "compensation_clean.jsonl", compensation_rows)


if __name__ == "__main__":
    main()
