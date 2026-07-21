"""Verified dataset download and chronological stream adapters."""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ELECTRICITY_URL = "https://openml.org/data/v1/download/2419/electricity.arff"
ELECTRICITY_MD5 = "8ca97867d960ae029ae3a9ac2c923d34"
ELECTRICITY_FEATURES = (
    "date",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
)


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
    request = urllib.request.Request(url, headers={"User-Agent": "DriftGuard/0.1"})
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


def electricity_stream(path: str | Path) -> Iterator[LabeledObservation]:
    """Yield Electricity rows in source order without shuffling or lookahead."""

    in_data = False
    index = 0
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            row = next(csv.reader([line]))
            if len(row) != 9:
                raise ValueError(f"row {index} has {len(row)} columns; expected 9")
            date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, label = row
            if day not in {"1", "2", "3", "4", "5", "6", "7"}:
                raise ValueError(f"row {index} has invalid day {day!r}")
            if label not in {"UP", "DOWN"}:
                raise ValueError(f"row {index} has invalid class {label!r}")

            numeric = (date, period, nswprice, nswdemand, vicprice, vicdemand, transfer)
            features = {
                name: float(value)
                for name, value in zip(ELECTRICITY_FEATURES, numeric, strict=True)
            }
            features.update({f"day_{value}": float(day == value) for value in "1234567"})
            yield LabeledObservation(index=index, features=features, label=int(label == "UP"))
            index += 1

    if not in_data:
        raise ValueError("ARFF file does not contain an @data section")
    if index == 0:
        raise ValueError("ARFF file contains no observations")

