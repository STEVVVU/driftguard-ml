#!/usr/bin/env python
"""Fail if the committed artifacts disagree with the figures quoted in the README.

The README quotes specific measured numbers -- 45,312 events, log loss -26.09%,
ROC-AUC 0.7725 -> 0.8501, seven drift alerts at named event indices, and so on.
Those are credibility-critical: a stale figure is worse than no figure at all.

CI already asserts that a fresh run reproduces `artifacts/*.json` byte-for-byte
(`git diff --exit-code`). That catches a *code* change that moves a metric, but it
cannot catch the other direction: regenerating the artifacts and forgetting to
update the README leaves both files self-consistent and the prose wrong. This
script closes that gap by restating each documented figure as an assertion.

If a change legitimately moves one of these, update this file and the README in
the same commit -- that is the point, not an inconvenience.

Usage:
    python scripts/check_documented_metrics.py [artifacts_dir]
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (json path, expected value, absolute tolerance)
#
# Tolerances are set to the precision the README actually prints, not tighter.
# The README quotes log loss to 4 decimals, so 5e-5 is the right gate: it fails on
# a real drift in the metric but not on a digit the prose never showed. Counts and
# event indices are exact -- they are integers, so any change is a real change.
Claim = tuple[str, float, float]

ELECTRICITY_CLAIMS: tuple[Claim, ...] = (
    # "Dataset: OpenML Electricity, all 45,312 rows in source chronological order."
    ("dataset.rows", 45_312, 0),
    # "Metrics cover the 40,312 post-warmup observations."
    ("frozen.observations", 40_312, 0),
    ("adaptive.observations", 40_312, 0),
    # The real-data result table.
    ("frozen.log_loss", 0.6409, 5e-5),
    ("adaptive.log_loss", 0.4737, 5e-5),
    ("frozen.accuracy_at_threshold", 0.7403, 5e-5),
    ("adaptive.accuracy_at_threshold", 0.7773, 5e-5),
    ("frozen.roc_auc", 0.7725, 5e-5),
    ("adaptive.roc_auc", 0.8501, 5e-5),
    ("frozen.average_precision", 0.7673, 5e-5),
    ("adaptive.average_precision", 0.8227, 5e-5),
    ("frozen.brier_score", 0.2010, 5e-5),
    ("adaptive.brier_score", 0.1545, 5e-5),
    ("frozen.ece", 0.1380, 5e-5),
    ("adaptive.ece", 0.0216, 5e-5),
    # "lowers log loss by 26.09%" -- also quoted as "26.1%" in Resume highlights.
    ("relative_log_loss_improvement", 0.2609, 5e-5),
    # "The audit processed all 45,312 labels with a minimum observed delay of 48
    #  and zero early-label violations."
    ("leakage_audit.labels_processed", 45_312, 0),
    ("leakage_audit.minimum_observed_delay", 48, 0),
    ("leakage_audit.violations", 0, 0),
    # "trains both models on the first 5,000 real observations" / 48-event delay.
    ("config.warmup", 5_000, 0),
    ("config.label_delay", 48, 0),
    ("config.detector", "two_window", 0),
)

# "records seven drift alerts (at events 7,366 / 11,607 / 15,762 / 19,481 /
#  23,044 / 28,979 / 31,691)"
EXPECTED_ALERTS: tuple[int, ...] = (7_366, 11_607, 15_762, 19_481, 23_044, 28_979, 31_691)

SYNTHETIC_CLAIMS: tuple[Claim, ...] = (
    # "the adaptive path lowers log loss by 6.97% (0.18313 -> 0.17037)"
    ("frozen.log_loss", 0.18313, 5e-6),
    ("adaptive.log_loss", 0.17037, 5e-6),
    ("relative_log_loss_improvement", 0.0697, 5e-5),
    # "improves top-5%-review precision from 0.150 to 0.246"
    ("frozen.precision_at_top_5pct", 0.150, 5e-4),
    ("adaptive.precision_at_top_5pct", 0.246, 5e-4),
    # "At event 6,600 it changes both covariates and the conditional fraud
    #  mechanism ... detects the shift at event 6,950"
    ("config.drift_at", 6_600, 0),
    ("leakage_audit.violations", 0, 0),
)

EXPECTED_SYNTHETIC_DETECTIONS: tuple[int, ...] = (6_950,)

# "5 of 13 features drift at p < 0.05: nswprice, nswdemand, vicprice, vicdemand,
#  transfer" and "The weekday one-hots and period do not (p ~ 1.0)".
REPORT_CLAIMS: tuple[Claim, ...] = (
    ("drifted_feature_count", 5, 0),
    ("drift_method", "ks", 0),
    ("drift_threshold", 0.05, 0),
)
# "compares the 5,000-row training window against the 40,312-row serving window".
# The Evidently snapshot does not record the split sizes, so these are asserted
# against the experiment artifact's warmup / observation counts instead of being
# restated here -- a self-comparison would pass vacuously.
EXPECTED_REFERENCE_ROWS = 5_000
EXPECTED_CURRENT_ROWS = 40_312
EXPECTED_DRIFTED_FEATURES = ["nswdemand", "nswprice", "transfer", "vicdemand", "vicprice"]
# "vicprice, vicdemand and transfer are constant across the first 5,000 rows"
EXPECTED_CONSTANT_FEATURES = ["transfer", "vicdemand", "vicprice"]
EXPECTED_FEATURE_COUNT = 13
# "The weekday one-hots and period do not (p ~ 1.0)" -- asserted as a floor rather
# than as exact 1.0 because day_1 legitimately scores 0.988.
STABLE_FEATURE_P_FLOOR = 0.95
STABLE_FEATURES = ("period", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7")

# The README's per-detector comparison table. Every data row is a claim about a
# configuration that was actually run, so every row must name a detector some
# committed artifact was produced with -- an unbacked row is exactly the defect
# this block exists to catch, and the river rows that used to sit here had no
# producer at all. Columns are matched by header text so a reordered or renamed
# column is reported rather than silently mis-read.
DETECTOR_TABLE_README = REPO_ROOT / "README.md"
DETECTOR_TABLE_HEADINGS = (
    "Detector",
    "Alerts",
    "Adaptive log loss",
    "Improvement vs frozen",
    "ROC-AUC",
)
# (column heading, json path, absolute tolerance) -- tolerances again match the
# precision the README prints, and the alert count is exact.
DETECTOR_COLUMNS: tuple[tuple[str, str, float], ...] = (
    ("Alerts", "drift_alerts_count", 0),
    ("Adaptive log loss", "adaptive.log_loss", 5e-5),
    ("Improvement vs frozen", "relative_log_loss_improvement", 5e-5),
    ("ROC-AUC", "adaptive.roc_auc", 5e-5),
)


def _dig(payload: object, dotted: str) -> object:
    """Resolve a dotted path like 'adaptive.log_loss' through nested dicts."""

    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _check(payload: object, claims: tuple[Claim, ...], label: str) -> list[str]:
    failures: list[str] = []
    for dotted, expected, tolerance in claims:
        try:
            actual = _dig(payload, dotted)
        except KeyError:
            failures.append(f"{label}: {dotted} is missing from the artifact")
            continue
        if isinstance(expected, str):
            if actual != expected:
                failures.append(f"{label}: {dotted} expected {expected!r}, got {actual!r}")
        elif not isinstance(actual, int | float) or not math.isclose(
            float(actual), float(expected), abs_tol=float(tolerance) or 0.0
        ):
            failures.append(f"{label}: {dotted} expected {expected} +/-{tolerance}, got {actual}")
    return failures


def _split_row(line: str) -> list[str]:
    """Split one markdown table line into stripped cells."""

    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_cell(text: str) -> float | None:
    """Read a table cell as a number, undoing bold markers and percent signs.

    Returns None when the cell is not a plain number, which is reported as a
    mismatch rather than silently skipped.
    """

    cleaned = text.replace("*", "").replace(",", "").strip()
    scale = 1.0
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
        scale = 0.01
    try:
        return float(cleaned) * scale
    except ValueError:
        return None


def _detector_artifacts(artifacts: Path) -> dict[str, tuple[str, dict]]:
    """Map detector name -> (artifact filename, payload) for every committed artifact.

    Only artifacts that record which detector produced them can back a row of the
    README's comparison table; anything else is not evidence for a detector claim.
    """

    index: dict[str, tuple[str, dict]] = {}
    for path in sorted(artifacts.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        config = payload.get("config")
        detector = config.get("detector") if isinstance(config, dict) else None
        if isinstance(detector, str):
            index.setdefault(detector, (path.name, payload))
    return index


def _check_detector_table(artifacts: Path) -> tuple[list[str], int]:
    """Assert every row of the README's detector comparison table has a producer.

    Each data row claims a measured configuration, so it must (a) name a detector
    that some committed artifact was generated with and (b) restate that
    artifact's numbers. A row with no backing artifact is a number with no
    producer, which is the failure this guard exists to make loud.

    Returns the failures plus the number of assertions actually performed, so a
    table that quietly loses rows shrinks the reported count instead of passing
    vacuously.
    """

    failures: list[str] = []
    if not DETECTOR_TABLE_README.exists():
        return [f"detector table: {DETECTOR_TABLE_README} is missing"], 0

    lines = DETECTOR_TABLE_README.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if _split_row(line) == list(DETECTOR_TABLE_HEADINGS)),
        None,
    )
    if header_index is None:
        # Not "nothing to check": the table is documented as the thing this guard
        # covers, so losing or renaming it must fail rather than pass silently.
        return [
            "detector table: no table with headings "
            f"{list(DETECTOR_TABLE_HEADINGS)} found in {DETECTOR_TABLE_README.name}"
        ], 0

    columns = {heading: position for position, heading in enumerate(DETECTOR_TABLE_HEADINGS)}
    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:  # skip the |---| alignment row
        if not line.strip().startswith("|"):
            break
        rows.append(_split_row(line))
    if not rows:
        return ["detector table: the comparison table has no data rows"], 0

    index = _detector_artifacts(artifacts)
    checked = 0
    for row in rows:
        if len(row) != len(DETECTOR_TABLE_HEADINGS):
            failures.append(f"detector table: row {row!r} does not have {len(columns)} cells")
            continue
        name_match = re.search(r"`([^`]+)`", row[columns["Detector"]])
        if name_match is None:
            failures.append(f"detector table: row {row!r} does not name a detector in backticks")
            continue
        detector = name_match.group(1)
        checked += 1
        backing = index.get(detector)
        if backing is None:
            failures.append(
                f"detector table: row `{detector}` is not backed by any committed artifact "
                f"(artifacts recording a detector: {sorted(index) or 'none'})"
            )
            continue
        artifact_name, payload = backing
        for heading, dotted, tolerance in DETECTOR_COLUMNS:
            checked += 1
            claimed = _parse_cell(row[columns[heading]])
            if claimed is None:
                failures.append(
                    f"detector table: `{detector}` {heading!r} cell "
                    f"{row[columns[heading]]!r} is not a number"
                )
                continue
            if dotted == "drift_alerts_count":
                actual: object = len(payload.get("drift_alerts", ()))
            else:
                try:
                    actual = _dig(payload, dotted)
                except KeyError:
                    failures.append(f"detector table: {dotted} is missing from {artifact_name}")
                    continue
            if not isinstance(actual, int | float) or not math.isclose(
                float(actual), claimed, abs_tol=float(tolerance) or 0.0
            ):
                failures.append(
                    f"detector table: `{detector}` {heading} reads {claimed} but "
                    f"{artifact_name} {dotted} is {actual}"
                )
    return failures, checked


def main(argv: list[str]) -> int:
    artifacts = Path(argv[0] if argv else "artifacts")
    failures: list[str] = []

    electricity_path = artifacts / "electricity_metrics.json"
    synthetic_path = artifacts / "metrics.json"
    report_path = artifacts / "electricity_drift_report.json"
    for path in (electricity_path, synthetic_path, report_path):
        if not path.exists():
            print(f"missing artifact: {path}", file=sys.stderr)
            return 2

    electricity = json.loads(electricity_path.read_text(encoding="utf-8"))
    failures += _check(electricity, ELECTRICITY_CLAIMS, "electricity")
    alerts = tuple(electricity.get("drift_alerts", ()))
    if alerts != EXPECTED_ALERTS:
        failures.append(f"electricity: drift_alerts expected {EXPECTED_ALERTS}, got {alerts}")

    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    failures += _check(synthetic, SYNTHETIC_CLAIMS, "synthetic")
    detections = tuple(synthetic.get("drift_detections", ()))
    if detections != EXPECTED_SYNTHETIC_DETECTIONS:
        failures.append(
            f"synthetic: drift_detections expected {EXPECTED_SYNTHETIC_DETECTIONS}, "
            f"got {detections}"
        )

    # The report summary is regenerated by `driftguard report` into the same
    # artifacts directory; the committed Evidently snapshot holds the raw metrics.
    # Recompute the digest from the snapshot so this script needs only the three
    # checked-in files and no extra state.
    snapshot = json.loads(report_path.read_text(encoding="utf-8"))
    summary = _summarize_snapshot(snapshot)
    failures += _check(summary, REPORT_CLAIMS, "report")
    if summary.get("drifted_features") != EXPECTED_DRIFTED_FEATURES:
        failures.append(
            f"report: drifted features expected {EXPECTED_DRIFTED_FEATURES}, "
            f"got {summary.get('drifted_features')}"
        )
    # Cross-check the documented split against the experiment artifact rather than
    # against a literal copied from the same place: the README's claim is that the
    # report splits the stream *the same way the experiment does*, so warmup and
    # post-warmup counts are the authoritative source for those two figures.
    if electricity["config"]["warmup"] != EXPECTED_REFERENCE_ROWS:
        failures.append(
            f"report: documented reference window {EXPECTED_REFERENCE_ROWS} does not match "
            f"the experiment warmup {electricity['config']['warmup']}"
        )
    if electricity["adaptive"]["observations"] != EXPECTED_CURRENT_ROWS:
        failures.append(
            f"report: documented serving window {EXPECTED_CURRENT_ROWS} does not match "
            f"the experiment's {electricity['adaptive']['observations']} scored observations"
        )
    if electricity["dataset"]["rows"] != EXPECTED_REFERENCE_ROWS + EXPECTED_CURRENT_ROWS:
        failures.append(
            f"report: {EXPECTED_REFERENCE_ROWS} + {EXPECTED_CURRENT_ROWS} does not account "
            f"for all {electricity['dataset']['rows']} rows"
        )

    p_values = summary.get("per_feature_drift_p_value", {})
    if len(p_values) != EXPECTED_FEATURE_COUNT:
        failures.append(f"report: expected {EXPECTED_FEATURE_COUNT} features, got {len(p_values)}")
    # "vicprice, vicdemand and transfer are constant across the first 5,000 rows."
    # Re-derived from the verified dataset rather than trusted, but only when the
    # data is present -- the unit-test job has no dataset and should not fail here.
    constant = _constant_reference_columns(EXPECTED_REFERENCE_ROWS)
    if constant is not None and constant != EXPECTED_CONSTANT_FEATURES:
        failures.append(
            f"report: columns constant in the reference window expected "
            f"{EXPECTED_CONSTANT_FEATURES}, got {constant}"
        )
    detector_failures, detector_checked = _check_detector_table(artifacts)
    failures += detector_failures

    for feature in STABLE_FEATURES:
        score = p_values.get(feature)
        if score is None or score < STABLE_FEATURE_P_FLOOR:
            failures.append(
                f"report: {feature} should stay stable (p >= {STABLE_FEATURE_P_FLOOR}), got {score}"
            )

    if failures:
        print("Documented metrics do not match the artifacts:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nIf a change legitimately moved one of these, update the README and "
            "this script together.",
            file=sys.stderr,
        )
        return 1

    checked = (
        len(ELECTRICITY_CLAIMS)
        + len(SYNTHETIC_CLAIMS)
        + len(REPORT_CLAIMS)
        + 7
        + len(STABLE_FEATURES)
        + detector_checked
    )
    print(f"All {checked} documented metrics match the committed artifacts.")
    return 0


def _constant_reference_columns(warmup: int) -> list[str] | None:
    """Names of feature columns with zero variance in the first ``warmup`` rows.

    Returns None when the dataset or its dependencies are unavailable, so this
    guard degrades to "not checked" rather than to a false failure in a job that
    deliberately does not download the data.
    """

    try:
        from driftguard.datasets import electricity_feature_columns, read_electricity
    except ImportError:
        return None
    dataset = Path("data/electricity.arff")
    if not dataset.exists():
        return None
    reference = read_electricity(dataset).iloc[:warmup][electricity_feature_columns()]
    # 1e-12 matches report.py's cutoff: these columns hold repeated decimal
    # literals, so their sample std lands at float noise rather than exact zero.
    return sorted(
        str(column) for column, spread in reference.std(ddof=1).items() if float(spread) < 1e-12
    )


def _summarize_snapshot(snapshot: object) -> dict[str, object]:
    """Reduce an Evidently snapshot to the headline numbers the README quotes.

    Mirrors ``driftguard.report._summarize`` deliberately rather than importing it:
    this script is a guard, and it should keep working even if that helper is
    refactored, so a README mismatch is reported as a mismatch rather than as an
    ImportError.
    """

    scores: dict[str, float] = {}
    summary: dict[str, object] = {}
    metrics = snapshot.get("metrics", []) if isinstance(snapshot, dict) else []
    for metric in metrics:
        config = metric.get("config", {})
        metric_type = str(config.get("type", ""))
        value = metric.get("value")
        if metric_type.endswith("DriftedColumnsCount") and isinstance(value, dict):
            summary["drifted_feature_count"] = value.get("count")
            summary["drifted_feature_share"] = value.get("share")
        elif metric_type.endswith("ValueDrift") and isinstance(value, int | float):
            column = config.get("column")
            if isinstance(column, str):
                scores[column] = float(value)
                summary.setdefault("drift_method", config.get("method"))
                summary.setdefault("drift_threshold", config.get("threshold"))
    summary["per_feature_drift_p_value"] = scores
    threshold = summary.get("drift_threshold")
    if isinstance(threshold, int | float):
        summary["drifted_features"] = sorted(
            column for column, score in scores.items() if score < threshold
        )
    return summary


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
