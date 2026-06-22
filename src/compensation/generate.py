from __future__ import annotations

from typing import Any, Callable

from src.compensation.filtering import clean_generated_texts
from src.compensation.prompts import parse_list_output, sample_prompt, scenario_prompt


def split_evenly(total: int, bucket_count: int) -> list[int]:
    if bucket_count <= 0:
        return []
    base = total // bucket_count
    remainder = total % bucket_count
    return [base + (1 if index < remainder else 0) for index in range(bucket_count)]


def generate_scenarios(client, word: str, scenario_count: int) -> list[str]:
    prompt = scenario_prompt(word=word, scenario_count=scenario_count)
    output = client.complete(prompt)
    return parse_list_output(output)[:scenario_count]


def generate_texts_for_word(
    client,
    word: str,
    scenario_count: int,
    samples_per_source: int,
    min_length: int,
    max_length: int,
    blacklist: list[str] | None = None,
    predictor: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    toxic_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    scenarios = generate_scenarios(client=client, word=word, scenario_count=scenario_count)
    counts = split_evenly(samples_per_source, len(scenarios))
    rows: list[dict[str, Any]] = []
    for scenario, count in zip(scenarios, counts):
        if count <= 0:
            continue
        prompt = sample_prompt(word=word, scenario=scenario, sample_count=count)
        output = client.complete(prompt)
        texts = parse_list_output(output)
        cleaned = clean_generated_texts(
            texts=texts,
            min_length=min_length,
            max_length=max_length,
            blacklist=blacklist,
            predictor=predictor,
            toxic_threshold=toxic_threshold,
        )
        for text in cleaned:
            rows.append({"id": f"{word}_{len(rows)}", "text": text, "label": 0, "source": word, "scenario": scenario})
    return rows


def build_compensation_set(
    words: list[str],
    client,
    scenario_count: int,
    samples_per_source: int,
    min_length: int,
    max_length: int,
    blacklist: list[str] | None = None,
    predictor: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    toxic_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for word in words:
        rows.extend(
            generate_texts_for_word(
                client=client,
                word=word,
                scenario_count=scenario_count,
                samples_per_source=samples_per_source,
                min_length=min_length,
                max_length=max_length,
                blacklist=blacklist,
                predictor=predictor,
                toxic_threshold=toxic_threshold,
            )
        )
    return rows
