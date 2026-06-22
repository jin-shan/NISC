from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.training.train_compensation import run_compensation_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--compensation-path", required=True)
    args = parser.parse_args()
    run_compensation_training(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        compensation_path=args.compensation_path,
    )


if __name__ == "__main__":
    main()
