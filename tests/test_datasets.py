import hashlib

import pytest

from driftguard.datasets import download_electricity, electricity_stream

ARFF = """\
@relation electricity
@attribute date numeric
@attribute day {1,2,3,4,5,6,7}
@attribute period numeric
@attribute nswprice numeric
@attribute nswdemand numeric
@attribute vicprice numeric
@attribute vicdemand numeric
@attribute transfer numeric
@attribute class {UP,DOWN}
@data
0,2,0.0,0.1,0.2,0.3,0.4,0.5,UP
0.1,7,0.2,0.3,0.4,0.5,0.6,0.7,DOWN
"""


def test_electricity_stream_preserves_order_and_encodes_day(tmp_path) -> None:
    source = tmp_path / "electricity.arff"
    source.write_text(ARFF, encoding="utf-8")

    rows = list(electricity_stream(source))

    assert [row.index for row in rows] == [0, 1]
    assert [row.label for row in rows] == [1, 0]
    assert rows[0].features["day_2"] == 1.0
    assert rows[0].features["day_7"] == 0.0
    assert rows[1].features["transfer"] == 0.7
    assert "date" not in rows[0].features


def test_download_rejects_checksum_mismatch_without_replacing_destination(tmp_path) -> None:
    source = tmp_path / "source.arff"
    source.write_text(ARFF, encoding="utf-8")
    destination = tmp_path / "cache" / "electricity.arff"

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_electricity(
            destination,
            url=source.as_uri(),
            expected_md5="0" * 32,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".arff.part").exists()


def test_download_accepts_verified_content_and_reuses_cache(tmp_path) -> None:
    source = tmp_path / "source.arff"
    payload = ARFF.encode()
    source.write_bytes(payload)
    destination = tmp_path / "cache" / "electricity.arff"
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    assert download_electricity(
        destination, url=source.as_uri(), expected_md5=checksum
    ) == destination
    source.unlink()
    assert download_electricity(
        destination, url=source.as_uri(), expected_md5=checksum
    ) == destination


def test_parser_rejects_invalid_class(tmp_path) -> None:
    source = tmp_path / "bad.arff"
    source.write_text(ARFF.replace(",UP", ",UNKNOWN"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid class"):
        list(electricity_stream(source))
