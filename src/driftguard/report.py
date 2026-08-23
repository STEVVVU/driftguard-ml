"""Evidently drift report over the Electricity stream's warmup/production split.

The event loop in :mod:`driftguard.real_experiment` detects drift *online* from a
single scalar (the reference model's loss). This module answers the complementary
question a reviewer asks after an alert fires -- "which input features actually
moved, and by how much?" -- by running ``evidently``'s ``DataDriftPreset`` over
the same data, split the same way the experiment splits it.

The split is chronological and matches the experiment exactly: rows before
``warmup`` are the reference distribution (what the frozen model was trained on)
and rows from ``warmup`` onward are the current distribution (what it was then
asked to score). Nothing is shuffled, so this is a genuine train-versus-serve
comparison rather than a random A/B of the same distribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from .datasets import electricity_feature_columns, read_electricity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftReportConfig:
    warmup: int = 5_000
    # The statistical test is pinned rather than left to Evidently's default.
    # Left to itself the preset picks its test from the sample size -- K-S p-value
    # under ~1,000 rows, normed Wasserstein distance above -- and those two report
    # on opposite scales (drift is p BELOW the threshold, but distance ABOVE it).
    # Pinning "ks" keeps the number and its comparison direction identical whether
    # the report runs on the full 45k stream or on a small test fixture.
    drift_method: str = "ks"
    drift_threshold: float = 0.05
    # Written next to the metrics JSON so the artifacts directory is one bundle.
    html_name: str = "electricity_drift_report.html"
    json_name: str = "electricity_drift_report.json"
    parquet_name: str = "electricity_reference_current.parquet"

    def validate(self) -> None:
        if self.warmup < 100:
            raise ValueError("warmup must be at least 100")
        if not 0.0 < self.drift_threshold < 1.0:
            raise ValueError("drift_threshold must be a p-value in (0, 1)")


# Shared default instance. Safe as a default argument because the dataclass is
# frozen, so no caller can mutate the value seen by the next one.
DEFAULT_REPORT_CONFIG = DriftReportConfig()


def build_drift_report(
    dataset: str | Path,
    config: DriftReportConfig = DEFAULT_REPORT_CONFIG,
    *,
    output_dir: str | Path = "artifacts",
) -> dict[str, object]:
    """Render an Evidently data-drift report for the warmup-vs-production split.

    Returns a small provenance/summary dict and writes three artifacts: the HTML
    report a human reads, the machine-readable JSON snapshot, and a parquet copy of
    the two frames so the comparison is reproducible without re-parsing the ARFF.
    """

    config.validate()
    frame = read_electricity(dataset)
    if len(frame) <= config.warmup:
        raise ValueError(f"dataset has {len(frame)} rows, not enough for warmup {config.warmup}")

    columns = electricity_feature_columns()
    # .copy() so the slices own their buffers; evidently holds on to these frames
    # and a view over the parent would keep the whole parsed frame alive.
    reference = frame.iloc[: config.warmup][columns].copy()
    current = frame.iloc[config.warmup :][columns].copy()
    logger.info(
        "drift report split: %d reference rows, %d current rows, %d features",
        len(reference),
        len(current),
        len(columns),
    )

    # The day_* one-hots are declared numerical rather than categorical on
    # purpose: as 0/1 indicators their mean *is* the category frequency, so the
    # numeric test picks up a shift in weekday mix without extra encoding, and one
    # statistical test then covers every column uniformly.
    definition = DataDefinition(numerical_columns=columns)
    report = Report(
        [DataDriftPreset(num_method=config.drift_method, num_threshold=config.drift_threshold)]
    )
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    html_path = destination / config.html_name
    json_path = destination / config.json_name
    parquet_path = destination / config.parquet_name
    snapshot.save_html(str(html_path))
    # newline="\n" for the same cross-platform reason as the metrics artifacts.
    json_path.write_text(snapshot.json(), encoding="utf-8", newline="\n")

    # One tidy frame with a `split` label rather than two files, so a reader can
    # reproduce the comparison with a single read_parquet + groupby.
    combined = pd.concat(
        [reference.assign(split="reference"), current.assign(split="current")],
        ignore_index=True,
    )
    combined.to_parquet(parquet_path, index=False)
    logger.info("wrote drift report artifacts to %s", destination)

    summary = _summarize(json_path)
    # On this dataset the Victorian columns (vicprice, vicdemand, transfer) are
    # *constant* across the first 5,000 rows -- the Electricity stream carries no
    # Victorian market data in its early period. A constant-to-variable column is
    # about as drifted as a feature can get, so K-S correctly returns a p-value of
    # ~0 for them; the caveat is that the *magnitude* carries no information there,
    # so those columns are named in the result rather than read as a ranking.
    # The cutoff is 1e-12 rather than exact zero because these columns hold
    # repeated decimal literals, so their sample std lands at float noise (~1e-16).
    degenerate = {
        str(column): float(spread)
        for column, spread in reference.std(ddof=1).items()
        if float(spread) < 1e-12
    }
    if degenerate:
        logger.warning(
            "reference window has %d constant column(s) %s; they are trivially "
            "drifted, so their p-values do not rank against the rest",
            len(degenerate),
            sorted(degenerate),
        )
    return {
        "warmup": config.warmup,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": columns,
        "constant_in_reference_window": sorted(degenerate),
        "artifacts": {
            "html": str(html_path),
            "json": str(json_path),
            "parquet": str(parquet_path),
        },
        **summary,
    }


def _summarize(json_path: Path) -> dict[str, object]:
    """Pull the headline drift numbers out of the Evidently snapshot JSON.

    Evidently's snapshot schema is version-dependent, so this reads defensively:
    it scans the flat metric list and keys off each entry's ``config.type``,
    reporting whatever it recognizes rather than assuming a fixed nesting. A
    schema change should cost a missing summary key, not a crashed CLI -- the
    authoritative artifacts are the HTML and the raw JSON, and this is a
    convenience digest over them.
    """

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    scores: dict[str, float] = {}
    summary: dict[str, object] = {}
    for metric in payload.get("metrics", []):
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
                # Every ValueDrift entry carries the same method/threshold for a
                # homogeneous column set, so recording it once is enough.
                summary.setdefault("drift_method", config.get("method"))
                summary.setdefault("drift_threshold", config.get("threshold"))
    if scores:
        summary["per_feature_drift_p_value"] = scores
        threshold = summary.get("drift_threshold")
        if isinstance(threshold, int | float):
            # K-S reports a p-value, so drift is a score strictly BELOW the
            # threshold -- the opposite direction to a distance-based test.
            summary["drifted_features"] = sorted(
                column for column, score in scores.items() if score < threshold
            )
    return summary
