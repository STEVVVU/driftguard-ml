import math

from driftguard.metrics import BinaryMetrics


def test_perfect_ranking_metrics() -> None:
    metrics = BinaryMetrics()
    for label, probability in [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]:
        metrics.add(label, probability)
    summary = metrics.summary(threshold=0.5)
    assert summary["roc_auc"] == 1.0
    assert summary["average_precision"] == 1.0
    assert summary["accuracy_at_threshold"] == 1.0
    assert math.isclose(summary["brier_score"], 0.025, abs_tol=1e-12)
