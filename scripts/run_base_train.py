from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.training.train_base import run_base_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_base_training(args.config)


if __name__ == "__main__":
    main()
