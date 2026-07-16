import math

from driftguard.model import FTRLProximal, OnlineStandardizer


def test_standardizer_does_not_include_current_row() -> None:
    scaler = OnlineStandardizer()
    assert scaler.transform_then_update({"x": 10.0}) == {"x": 0.0}
    assert scaler.transform_then_update({"x": 20.0}) == {"x": 0.0}
    transformed = scaler.transform_then_update({"x": 30.0})
    assert math.isclose(transformed["x"], 2.1213203436, rel_tol=1e-9)


def test_ftrl_learns_linearly_separable_examples() -> None:
    model = FTRLProximal(alpha=0.2, l2=0.1)
    for _ in range(80):
        model.update({"x": -2.0}, 0)
        model.update({"x": 2.0}, 1)
    assert model.predict_proba({"x": -2.0}) < 0.1
    assert model.predict_proba({"x": 2.0}) > 0.9
