from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.bias.bias_pmi import rank_candidate_words
from src.bias.false_positive_mining import select_high_confidence_false_positives
from src.bias.mask_validation import validate_bias_sources
from src.bias.word_stats import load_stopwords
from src.compensation.filtering import load_blacklist
from src.compensation.generate import build_compensation_set
from src.compensation.llm_client import OpenAICompatibleClient
from src.data.loaders import build_dataloader, load_examples
from src.models.classifier import load_tokenizer
from src.models.masking import MaskedSequenceClassifier
from src.pruning.attribution import differential_contributions
from src.pruning.iterative_pruning import iterative_prune
from src.pruning.neuron_ranking import aggregate_bias_scores
from src.training.train_base import run_base_training
from src.training.train_compensation import run_compensation_training
from src.training.trainer_utils import build_text_predictor, evaluate_model, select_device
from src.utils.io import ensure_dir, load_config, read_jsonl, resolve_path, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    base_result = run_base_training(args.config)
    base_prediction_rows = read_jsonl(f"{base_result['output_dir']}/dev_predictions.jsonl")
    false_positive_rows = select_high_confidence_false_positives(base_prediction_rows, float(config["bias"]["tau_conf"]))
    stopwords = load_stopwords(config["bias"]["stopwords_path"])
    candidate_rows = rank_candidate_words(
        validation_rows=base_prediction_rows,
        false_positive_rows=false_positive_rows,
        stopwords=stopwords,
        fmin=int(config["bias"]["fmin"]),
        top_k=int(config["bias"]["top_k"]),
        eps=float(config["bias"]["eps"]),
    )
    checkpoint_dir = base_result["checkpoint_dir"]
    tokenizer = load_tokenizer(checkpoint_dir)
    model = MaskedSequenceClassifier.from_pretrained(checkpoint_dir, num_labels=int(config["model"].get("num_labels", 2)))
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
    bias_sources = validate_bias_sources(
        candidate_rows=candidate_rows,
        false_positive_rows=false_positive_rows,
        predictor=predictor,
        replacement=replacement,
        tau_mask=float(config["bias"]["tau_mask"]),
        tau_flip=float(config["bias"]["tau_flip"]),
    )
    bias_dir = ensure_dir(f"outputs/{experiment_name}/bias")
    write_jsonl(bias_dir / "false_positives.jsonl", false_positive_rows)
    write_jsonl(bias_dir / "candidates.jsonl", candidate_rows)
    write_jsonl(bias_dir / "bias_sources.jsonl", bias_sources)
    word_differences = {}
    for row in bias_sources:
        word = row["word"]
        present_texts = [sample["text"] for sample in false_positive_rows if word in sample["text"]]
        masked_texts = [text.replace(word, replacement) for text in present_texts]
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

    pruning_report = iterative_prune(
        model=model,
        ranking=ranking,
        evaluator=evaluator,
        step_size=int(config["pruning"]["step_size"]),
        max_rounds=int(config["pruning"]["max_rounds"]),
        alpha=float(config["pruning"]["alpha"]),
    )
    pruning_dir = ensure_dir(f"outputs/{experiment_name}/pruning")
    model.save_pretrained(pruning_dir / "checkpoint", save_tokenizer=tokenizer)
    write_jsonl(pruning_dir / "neuron_ranking.jsonl", ranking)
    write_json(pruning_dir / "pruning_report.json", pruning_report)
    client = OpenAICompatibleClient(
        model_name=config["compensation"]["model_name"],
        api_key=args.api_key,
        base_url=args.base_url,
    )
    blacklist = load_blacklist(config["compensation"]["toxic_blacklist_path"])
    compensation_rows = build_compensation_set(
        words=[row["word"] for row in bias_sources],
        client=client,
        scenario_count=int(config["compensation"]["scenario_count"]),
        samples_per_source=int(config["compensation"]["samples_per_source"]),
        min_length=int(config["compensation"]["sample_length_min"]),
        max_length=int(config["compensation"]["sample_length_max"]),
        blacklist=blacklist,
        predictor=predictor,
        toxic_threshold=float(config["training"].get("decision_threshold", 0.5)),
    )
    compensation_dir = ensure_dir(f"outputs/{experiment_name}/compensation_data")
    compensation_path = compensation_dir / "compensation_clean.jsonl"
    write_jsonl(compensation_path, compensation_rows)
    run_compensation_training(args.config, checkpoint_dir=str(pruning_dir / "checkpoint"), compensation_path=str(compensation_path))


if __name__ == "__main__":
    main()
