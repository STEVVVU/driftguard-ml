"""Command-line interface for DriftGuard."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .datasets import download_electricity
from .experiment import ExperimentConfig, run_experiment
from .real_experiment import DETECTORS, ElectricityExperimentConfig, run_electricity_experiment
from .report import DriftReportConfig, build_drift_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log progress to stderr (results still go to stdout as JSON)",
    )
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
    electricity.add_argument(
        "--detector",
        choices=DETECTORS,
        default="two_window",
        help="drift detector driving the adaptive model's reset-and-replay",
    )
    electricity.add_argument("--output", default="artifacts")
    report = subparsers.add_parser(
        "report", help="render an Evidently feature-drift report for the warmup split"
    )
    report.add_argument("--data", default="data/electricity.arff")
    report.add_argument("--warmup", type=int, default=5_000)
    report.add_argument("--output", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Logging is opt-in so the default stdout stays a single parseable JSON
    # document that a caller can pipe straight into jq.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
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
                detector=args.detector,
            ),
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "report":
        dataset = Path(args.data)
        download_electricity(dataset)
        result = build_drift_report(
            dataset,
            DriftReportConfig(warmup=args.warmup),
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
