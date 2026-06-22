from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.eval.metrics import compute_binary_metrics, probs_to_preds


@dataclass
class PredictionOutput:
    ids: list[str]
    texts: list[str]
    labels: list[int]
    preds: list[int]
    probs: list[list[float]]
    logits: list[list[float]]


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def split_batch(batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model_inputs: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value
        else:
            meta[key] = value
    return model_inputs, meta


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_optimizer(model: torch.nn.Module, config: dict) -> AdamW:
    training_cfg = config["training"]
    return AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float):
    warmup_steps = math.floor(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def prediction_rows(output: PredictionOutput) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(output.ids)):
        prob_non_toxic, prob_toxic = output.probs[index]
        logit_non_toxic, logit_toxic = output.logits[index]
        rows.append(
            {
                "id": output.ids[index],
                "text": output.texts[index],
                "label": int(output.labels[index]),
                "pred_label": int(output.preds[index]),
                "prob_non_toxic": float(prob_non_toxic),
                "prob_toxic": float(prob_toxic),
                "logit_non_toxic": float(logit_non_toxic),
                "logit_toxic": float(logit_toxic),
            }
        )
    return rows


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    decision_threshold: float,
) -> tuple[dict[str, Any], PredictionOutput]:
    model.eval()
    ids: list[str] = []
    texts: list[str] = []
    labels: list[int] = []
    logits_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    with torch.no_grad():
        for raw_batch in dataloader:
            model_inputs, meta = split_batch(raw_batch)
            model_inputs = move_to_device(model_inputs, device)
            labels.extend(model_inputs["labels"].detach().cpu().tolist())
            ids.extend(list(meta["ids"]))
            texts.extend(list(meta["texts"]))
            outputs = model(**model_inputs)
            logits = outputs.logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            logits_all.extend(logits.tolist())
            probs_all.extend(probs.tolist())
    prob_toxic = np.array([row[1] for row in probs_all], dtype=np.float32)
    preds = probs_to_preds(prob_toxic, threshold=decision_threshold)
    metrics = compute_binary_metrics(labels=labels, preds=preds)
    return metrics, PredictionOutput(
        ids=ids,
        texts=texts,
        labels=labels,
        preds=preds,
        probs=probs_all,
        logits=logits_all,
    )


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    dev_dataloader,
    config: dict,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    optimizer = build_optimizer(model, config)
    train_cfg = config["training"]
    accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    total_steps = max(1, math.ceil(len(train_dataloader) / accum_steps) * int(train_cfg["epochs"]))
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
    )
    use_fp16 = bool(train_cfg.get("fp16", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    patience = int(train_cfg.get("early_stopping_patience", 2))
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, raw_batch in enumerate(train_dataloader, start=1):
            model_inputs, _meta = split_batch(raw_batch)
            model_inputs = move_to_device(model_inputs, device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**model_inputs)
                loss = outputs.loss / accum_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * accum_steps
            if step % accum_steps == 0 or step == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        dev_metrics, _ = evaluate_model(
            model=model,
            dataloader=dev_dataloader,
            device=device,
            decision_threshold=float(train_cfg.get("decision_threshold", 0.5)),
        )
        epoch_record = {"epoch": epoch, "loss": running_loss / max(1, len(train_dataloader)), **dev_metrics}
        history.append(epoch_record)
        if dev_metrics["f1"] > best_f1:
            best_f1 = float(dev_metrics["f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_f1": best_f1, "history": history}


def build_text_predictor(
    model: torch.nn.Module,
    tokenizer,
    max_length: int,
    device: torch.device,
    decision_threshold: float,
    batch_size: int = 16,
) -> Callable[[list[str]], list[dict[str, Any]]]:
    model.to(device)
    model.eval()

    def predict(texts: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = move_to_device(encoded, device)
            with torch.no_grad():
                outputs = model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().tolist()
            for text, pair in zip(chunk, probs):
                pred_label = int(pair[1] >= decision_threshold)
                rows.append(
                    {
                        "text": text,
                        "pred_label": pred_label,
                        "prob_non_toxic": float(pair[0]),
                        "prob_toxic": float(pair[1]),
                    }
                )
        return rows

    return predict
