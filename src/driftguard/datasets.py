"""Verified dataset download and chronological stream adapters.

The download path stays on the standard library (``urllib`` + ``hashlib``) on
purpose: the checksum gate is the trust boundary for this project, and it should
not depend on a third-party HTTP client. Parsing happens afterwards, with
``pandas.read_csv`` reading the *already verified* local file -- pandas is never
pointed at a URL, so nothing unverified reaches the parser.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ELECTRICITY_URL = "https://openml.org/data/v1/download/2419/electricity.arff"
ELECTRICITY_MD5 = "8ca97867d960ae029ae3a9ac2c923d34"
ELECTRICITY_FEATURES = (
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
)
# Full ARFF column order. `date` is read so the column count can be validated but
# is dropped from the feature set: it is a monotone index, and a linear model
# would extrapolate off the end of it rather than learn anything transferable.
ELECTRICITY_COLUMNS = ("date", "day", *ELECTRICITY_FEATURES, "class")
_ELECTRICITY_DAYS = ("1", "2", "3", "4", "5", "6", "7")


@dataclass(frozen=True)
class LabeledObservation:
    index: int
    features: dict[str, float]
    label: int


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_electricity(
    destination: str | Path,
    *,
    url: str = ELECTRICITY_URL,
    expected_md5: str = ELECTRICITY_MD5,
) -> Path:
    """Download atomically and reject content that differs from OpenML metadata."""

    destination = Path(destination)
    if destination.exists() and file_md5(destination) == expected_md5:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DriftGuard/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual_md5 = file_md5(partial)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"dataset checksum mismatch: expected {expected_md5}, received {actual_md5}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _data_section_offset(path: Path) -> int:
    """Return the number of leading lines to skip to reach the ARFF data rows.

    ARFF headers are variable-length, so the ``@data`` marker is located by a
    single cheap scan and its position handed to ``read_csv`` as ``skiprows``.
    """

    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if raw_line.strip().lower() == "@data":
                return line_number
    raise ValueError("ARFF file does not contain an @data section")


def read_electricity(path: str | Path) -> pd.DataFrame:
    """Parse a verified Electricity ARFF file into a row-ordered DataFrame.

    Returns the raw typed columns in source order with the ``day`` one-hot
    columns appended and ``label`` as 0/1. Row order is the evaluation order and
    is never sorted or shuffled here.
    """

    path = Path(path)
    skiprows = _data_section_offset(path)
    frame = pd.read_csv(
        path,
        skiprows=skiprows,
        header=None,
        names=list(ELECTRICITY_COLUMNS),
        # `day` and `class` are ARFF nominals; reading them as strings keeps the
        # validation below comparing like with like (a numeric `day` would make
        # "1" vs 1 checks depend on pandas' inference).
        dtype={"day": "string", "class": "string"},
        comment="%",
        skip_blank_lines=True,
        # The header scan already resolved the layout, so a short or long row is a
        # corrupt file rather than something to be tolerated.
        on_bad_lines="error",
    )

    if frame.empty:
        raise ValueError("ARFF file contains no observations")

    day = frame["day"].str.strip()
    invalid_days = ~day.isin(_ELECTRICITY_DAYS)
    if bool(invalid_days.any()):
        row = int(invalid_days.idxmax())
        raise ValueError(f"row {row} has invalid day {frame.loc[row, 'day']!r}")

    label = frame["class"].str.strip()
    invalid_labels = ~label.isin(("UP", "DOWN"))
    if bool(invalid_labels.any()):
        row = int(invalid_labels.idxmax())
        raise ValueError(f"row {row} has invalid class {frame.loc[row, 'class']!r}")

    numeric = frame[list(ELECTRICITY_FEATURES)].astype(np.float64)
    # One-hot the weekday as float so downstream feature dicts stay homogeneous.
    one_hot = pd.DataFrame(
        {f"day_{value}": (day == value).to_numpy(dtype=np.float64) for value in _ELECTRICITY_DAYS},
        index=frame.index,
    )
    parsed = pd.concat([numeric, one_hot], axis=1)
    parsed["label"] = (label == "UP").to_numpy(dtype=np.int64)
    return parsed


def electricity_feature_columns() -> list[str]:
    """Return the model feature column names in a fixed order."""

    return [*ELECTRICITY_FEATURES, *(f"day_{value}" for value in _ELECTRICITY_DAYS)]


def electricity_stream(path: str | Path) -> Iterator[LabeledObservation]:
    """Yield Electricity rows in source order without shuffling or lookahead."""

    frame = read_electricity(path)
    columns = electricity_feature_columns()
    # itertuples over a positionally-sliced frame: the event loop consumes one
    # row at a time, and .iloc-per-row or .iterrows would each be far slower.
    values = frame[columns].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    for index, (row, label) in enumerate(zip(values, labels, strict=True)):
        yield LabeledObservation(
            index=index,
            features=dict(zip(columns, row.tolist(), strict=True)),
            label=int(label),
        )
