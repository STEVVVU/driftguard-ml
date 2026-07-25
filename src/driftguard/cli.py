"""Command-line interface for DriftGuard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import download_electricity
from .experiment import ExperimentConfig, run_experiment
from .real_experiment import ElectricityExperimentConfig, run_electricity_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the delayed-label drift experiment")
    run.add_argument("--samples", type=int, default=12_000)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--warmup", type=int, default=2_000)
    run.add_argument("--label-delay", type=int, default=40)
    run.add_argument("--drift-at", type=int)
    run.add_argument("--output", default="artifacts")
    electricity = subparsers.add_parser(
        "electricity", help="run a chronological benchmark on OpenML Electricity"
    )
    electricity.add_argument("--data", default="data/electricity.arff")
    electricity.add_argument("--warmup", type=int, default=5_000)
    electricity.add_argument("--label-delay", type=int, default=48)
    electricity.add_argument("--output", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_experiment(
            ExperimentConfig(
                samples=args.samples,
                seed=args.seed,
                warmup=args.warmup,
                label_delay=args.label_delay,
                drift_at=args.drift_at,
            ),
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "electricity":
        dataset = Path(args.data)
        download_electricity(dataset)
        result = run_electricity_experiment(
            dataset,
            ElectricityExperimentConfig(
                warmup=args.warmup,
                label_delay=args.label_delay,
            ),
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
