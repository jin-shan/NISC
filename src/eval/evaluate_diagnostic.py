from __future__ import annotations

from typing import Any

from src.eval.evaluate_main import run_main_evaluation


def run_diagnostic_evaluation(config_path: str, checkpoint_dir: str) -> dict[str, Any]:
    return run_main_evaluation(config_path=config_path, checkpoint_dir=checkpoint_dir, split="test")
