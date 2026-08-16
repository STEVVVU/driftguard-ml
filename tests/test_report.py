import json

import pandas as pd
import pytest

from driftguard.report import DriftReportConfig, build_drift_report

_HEADER = (
    "@relation electricity",
    "@attribute date numeric",
    "@attribute day {1,2,3,4,5,6,7}",
    "@attribute period numeric",
    "@attribute nswprice numeric",
    "@attribute nswdemand numeric",
    "@attribute vicprice numeric",
    "@attribute vicdemand numeric",
    "@attribute transfer numeric",
    "@attribute class {UP,DOWN}",
    "@data",
)


def _write_shifting_arff(path, rows: int = 600, warmup: int = 200) -> None:
    """Write an ARFF whose nswprice distribution shifts after `warmup` rows.

    nswdemand is left constant on purpose so the test can also assert the
    zero-variance-reference reporting path.
    """

    lines = list(_HEADER)
    for index in range(rows):
        # Deterministic sawtooth, shifted by +0.5 after warmup.
        base = (index % 10) / 20.0
        nswprice = base + (0.5 if index >= warmup else 0.0)
        label = "UP" if index % 2 else "DOWN"
        lines.append(
            f"{index / rows},{index % 7 + 1},{(index % 48) / 48},{nswprice},"
            f"0.4,0.1,0.2,0.3,{label}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_report_writes_html_json_and_parquet_artifacts(tmp_path) -> None:
    dataset = tmp_path / "electricity.arff"
    _write_shifting_arff(dataset)
    output = tmp_path / "artifacts"

    result = build_drift_report(
        dataset, DriftReportConfig(warmup=200), output_dir=output
    )

    assert result["reference_rows"] == 200
    assert result["current_rows"] == 400
    for kind in ("html", "json", "parquet"):
        artifact = output / str(result["artifacts"][kind]).rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        assert artifact.exists() and artifact.stat().st_size > 0, kind

    # The raw snapshot must stay valid JSON: it is the machine-readable artifact
    # the summary digest is derived from, and CI asserts on it.
    snapshot = json.loads((output / "electricity_drift_report.json").read_text(encoding="utf-8"))
    assert snapshot["metrics"]


def test_report_detects_the_injected_feature_shift(tmp_path) -> None:
    dataset = tmp_path / "electricity.arff"
    _write_shifting_arff(dataset)

    result = build_drift_report(
        dataset, DriftReportConfig(warmup=200), output_dir=tmp_path / "artifacts"
    )

    scores = result["per_feature_drift_p_value"]
    assert isinstance(scores, dict)
    threshold = result["drift_threshold"]
    assert isinstance(threshold, int | float)
    # The report pins the K-S test, so these are p-values: a drifted column scores
    # BELOW the threshold. nswprice is the only column given a real distribution
    # shift, so it must be significant while an untouched column (period) must not.
    assert scores["nswprice"] < threshold
    assert scores["period"] >= threshold
    assert "nswprice" in result["drifted_features"]
    assert "period" not in result["drifted_features"]


def test_report_pins_the_same_test_regardless_of_sample_size(tmp_path) -> None:
    """The statistical test must not change with row count.

    Left to its default, Evidently's DataDriftPreset picks K-S p-value under
    ~1,000 rows and normed Wasserstein distance above it. Those report on opposite
    scales -- drift is a p-value BELOW the threshold but a distance ABOVE it -- so
    an unpinned method makes the reported number uninterpretable without also
    knowing the sample size. DriftReportConfig pins it; this locks that in.
    """

    methods = set()
    for rows in (600, 4_000):
        dataset = tmp_path / f"electricity_{rows}.arff"
        _write_shifting_arff(dataset, rows=rows, warmup=200)
        result = build_drift_report(
            dataset, DriftReportConfig(warmup=200), output_dir=tmp_path / f"artifacts_{rows}"
        )
        methods.add(result["drift_method"])
        assert result["drift_threshold"] == 0.05

    assert methods == {"ks"}


def test_report_names_columns_that_are_constant_in_the_reference_window(tmp_path) -> None:
    """Zero-variance reference columns must be flagged, not silently reported.

    Evidently's normed Wasserstein distance divides by the reference spread, so a
    constant reference column yields a score on a different scale entirely. The
    report names those columns so the number is not read as comparable.
    """

    dataset = tmp_path / "electricity.arff"
    _write_shifting_arff(dataset)

    result = build_drift_report(
        dataset, DriftReportConfig(warmup=200), output_dir=tmp_path / "artifacts"
    )

    constant = result["constant_in_reference_window"]
    assert isinstance(constant, list)
    # nswdemand/vicprice/vicdemand/transfer are written as fixed literals, and the
    # day one-hots cycle within every 7 rows so they are *not* constant.
    assert "nswdemand" in constant
    assert "nswprice" not in constant


def test_report_parquet_round_trips_both_splits(tmp_path) -> None:
    dataset = tmp_path / "electricity.arff"
    _write_shifting_arff(dataset)
    output = tmp_path / "artifacts"

    build_drift_report(dataset, DriftReportConfig(warmup=200), output_dir=output)

    frame = pd.read_parquet(output / "electricity_reference_current.parquet")
    assert frame["split"].value_counts().to_dict() == {"current": 400, "reference": 200}


def test_report_rejects_a_dataset_shorter_than_the_warmup(tmp_path) -> None:
    dataset = tmp_path / "electricity.arff"
    _write_shifting_arff(dataset, rows=300, warmup=100)

    with pytest.raises(ValueError, match="not enough for warmup"):
        build_drift_report(
            dataset, DriftReportConfig(warmup=500), output_dir=tmp_path / "artifacts"
        )
