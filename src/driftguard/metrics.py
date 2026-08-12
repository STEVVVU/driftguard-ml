"""Binary classification metrics backed by scikit-learn and NumPy.

The scoring rules, ranking metrics, and threshold metrics all come from
``sklearn.metrics`` so the numbers this project reports are the same numbers a
reviewer would reproduce in a notebook. Only the two metrics scikit-learn does
not ship -- expected calibration error and top-fraction precision/recall -- are
computed here, vectorized with NumPy.

Observations are buffered in Python lists because the event loop appends one row
at a time (an ``np.append`` per event would be quadratic); the buffer is
converted to arrays once, lazily, when a summary is requested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Probabilities are clamped away from the open interval's endpoints on ingest so
# that a confidently wrong prediction contributes a large-but-finite log loss
# instead of an infinity that would poison every downstream aggregate.
_PROBABILITY_FLOOR = 1e-12
_PROBABILITY_CEILING = 1.0 - _PROBABILITY_FLOOR


@dataclass
class BinaryMetrics:
    labels: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    _cache: tuple[NDArray[np.int_], NDArray[np.float64]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def add(self, label: int, probability: float) -> None:
        self.labels.append(int(label))
        self.probabilities.append(
            min(max(float(probability), _PROBABILITY_FLOOR), _PROBABILITY_CEILING)
        )
        self._cache = None

    def _arrays(self) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Return the buffered observations as arrays, memoized between metrics.

        A single summary touches the same two arrays a dozen times, so building
        them once keeps the conversion cost off the per-metric path.
        """

        if self._cache is None:
            self._cache = (
                np.asarray(self.labels, dtype=np.int_),
                np.asarray(self.probabilities, dtype=np.float64),
            )
        return self._cache

    def summary(self, *, threshold: float = 0.1, calibration_bins: int = 10) -> dict[str, float | int]:
        if not self.labels:
            raise ValueError("no observations recorded")
        labels, _ = self._arrays()
        return {
            "observations": int(labels.size),
            "positive_rate": float(labels.mean()),
            "log_loss": self.log_loss(),
            "brier_score": self.brier_score(),
            "roc_auc": self.roc_auc(),
            "average_precision": self.average_precision(),
            "ece": self.expected_calibration_error(calibration_bins),
            "decision_threshold": threshold,
            "precision_at_threshold": self.precision(threshold),
            "recall_at_threshold": self.recall(threshold),
            "accuracy_at_threshold": self.accuracy(threshold),
            "precision_at_top_5pct": self.precision_at_fraction(0.05),
            "recall_at_top_5pct": self.recall_at_fraction(0.05),
        }

    def log_loss(self) -> float:
        labels, probabilities = self._arrays()
        # labels=[0, 1] is required: a post-warmup window can legitimately be
        # single-class, and without it sklearn would infer a one-class problem
        # and raise instead of scoring against the intended binary support.
        return float(log_loss(labels, probabilities, labels=[0, 1]))

    def brier_score(self) -> float:
        labels, probabilities = self._arrays()
        # scale_by_half=True is the classic binary mean-squared-error definition
        # (mean((p - y)^2)). Passing False would return the multiclass
        # sum-over-both-columns form, which is exactly twice as large; the
        # default of "auto" already picks True for binary input, but pinning it
        # keeps the reported number stable if that heuristic ever changes.
        return float(
            brier_score_loss(labels, probabilities, labels=[0, 1], scale_by_half=True)
        )

    def roc_auc(self) -> float:
        labels, probabilities = self._arrays()
        positives = int(labels.sum())
        # roc_auc_score raises on a single-class input; NaN is the honest answer
        # for "this window contained no ranking information".
        if positives == 0 or positives == labels.size:
            return float("nan")
        return float(roc_auc_score(labels, probabilities))

    def average_precision(self) -> float:
        labels, probabilities = self._arrays()
        if labels.sum() == 0:
            return float("nan")
        return float(average_precision_score(labels, probabilities))

    def expected_calibration_error(self, bins: int = 10) -> float:
        """Equal-width reliability-diagram gap, weighted by bin occupancy.

        scikit-learn exposes ``calibration_curve`` but not the aggregated ECE
        scalar, so the binning is done here. Bins are half-open ``[lo, hi)``
        with the top bin closed so that ``p == 1.0`` is not silently dropped.
        """

        if bins <= 0:
            raise ValueError("bins must be positive")
        labels, probabilities = self._arrays()
        total = labels.size
        # Interior edges are built as arange(1, bins) / bins rather than with
        # np.linspace: linspace computes 0.3/0.6/0.7 as 0.30000000000000004 etc.,
        # which shifts a probability sitting exactly on a boundary into the wrong
        # bin. Dividing the integer index reproduces the same double a Python
        # `bin_index / bins` produces, so bin membership is exact.
        edges = np.arange(1, bins, dtype=np.float64) / bins
        # digitize with right=False yields half-open [lo, hi) bins; clipping folds
        # p == 1.0 into the top bin instead of an out-of-range bucket.
        assignments = np.clip(np.digitize(probabilities, edges, right=False), 0, bins - 1)
        error = 0.0
        for bin_index in range(bins):
            members = assignments == bin_index
            count = int(members.sum())
            if count == 0:
                continue
            accuracy = float(labels[members].mean())
            confidence = float(probabilities[members].mean())
            error += count / total * abs(accuracy - confidence)
        return error

    def _predictions(self, threshold: float) -> NDArray[np.int_]:
        _, probabilities = self._arrays()
        return (probabilities >= threshold).astype(np.int_)

    def precision(self, threshold: float) -> float:
        labels, _ = self._arrays()
        # zero_division=0 preserves the prior contract: no predicted positives
        # scores as 0.0 precision rather than raising or returning NaN.
        return float(
            precision_score(labels, self._predictions(threshold), zero_division=0.0)
        )

    def recall(self, threshold: float) -> float:
        labels, _ = self._arrays()
        return float(recall_score(labels, self._predictions(threshold), zero_division=0.0))

    def accuracy(self, threshold: float) -> float:
        labels, _ = self._arrays()
        return float(accuracy_score(labels, self._predictions(threshold)))

    def _top_fraction_labels(self, fraction: float) -> NDArray[np.int_]:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        labels, probabilities = self._arrays()
        count = max(1, math.ceil(labels.size * fraction))
        # Descending stable sort so equal scores keep arrival order, matching the
        # "review the top k alerts a queue would surface" reading of the metric.
        order = np.argsort(-probabilities, kind="stable")
        return labels[order[:count]]

    def precision_at_fraction(self, fraction: float) -> float:
        selected = self._top_fraction_labels(fraction)
        return float(selected.mean())

    def recall_at_fraction(self, fraction: float) -> float:
        labels, _ = self._arrays()
        positives = int(labels.sum())
        return float(self._top_fraction_labels(fraction).sum() / positives) if positives else 0.0
