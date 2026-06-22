from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.utils.io import resolve_path


@dataclass
class TextExample:
    example_id: str
    text: str
    label: int


class TextExampleDataset(Dataset):
    def __init__(self, examples: list[TextExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {"id": example.example_id, "text": example.text, "label": example.label}


class TextBatchCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [row["text"] for row in rows]
        labels = [int(row["label"]) for row in rows]
        ids = [str(row["id"]) for row in rows]
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["ids"] = ids
        batch["texts"] = texts
        return batch


def load_examples(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    id_column: str = "id",
) -> list[TextExample]:
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"data file not found: {file_path}")
    if file_path.suffix == ".jsonl":
        frame = pd.read_json(file_path, lines=True)
    elif file_path.suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif file_path.suffix == ".tsv":
        frame = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"unsupported data file format: {file_path.suffix}")
    required = {text_column, label_column}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if id_column not in frame.columns:
        frame[id_column] = [str(index) for index in range(len(frame))]
    examples: list[TextExample] = []
    for row in frame[[id_column, text_column, label_column]].itertuples(index=False):
        example_id, text, label = row
        examples.append(TextExample(example_id=str(example_id), text=str(text), label=int(label)))
    return examples


def build_dataloader(
    examples: list[TextExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TextExampleDataset(examples)
    collator = TextBatchCollator(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def examples_from_rows(rows: list[dict[str, Any]], label_key: str = "label") -> list[TextExample]:
    examples: list[TextExample] = []
    for index, row in enumerate(rows):
        example_id = str(row.get("id", index))
        examples.append(TextExample(example_id=example_id, text=str(row["text"]), label=int(row[label_key])))
    return examples
