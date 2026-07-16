"""Dependency-free binary classification metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BinaryMetrics:
    labels: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    def add(self, label: int, probability: float) -> None:
        self.labels.append(int(label))
        self.probabilities.append(min(max(float(probability), 1e-12), 1.0 - 1e-12))

    def summary(self, *, threshold: float = 0.1, calibration_bins: int = 10) -> dict[str, float | int]:
        if not self.labels:
            raise ValueError("no observations recorded")
        return {
            "observations": len(self.labels),
            "positive_rate": sum(self.labels) / len(self.labels),
            "log_loss": self.log_loss(),
            "brier_score": self.brier_score(),
            "roc_auc": self.roc_auc(),
            "average_precision": self.average_precision(),
            "ece": self.expected_calibration_error(calibration_bins),
            "decision_threshold": threshold,
            "precision_at_threshold": self.precision(threshold),
            "recall_at_threshold": self.recall(threshold),
            "precision_at_top_5pct": self.precision_at_fraction(0.05),
            "recall_at_top_5pct": self.recall_at_fraction(0.05),
        }

    def log_loss(self) -> float:
        losses = [
            -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
            for label, probability in zip(self.labels, self.probabilities, strict=True)
        ]
        return sum(losses) / len(losses)

    def brier_score(self) -> float:
        errors = [
            (probability - label) ** 2
            for label, probability in zip(self.labels, self.probabilities, strict=True)
        ]
        return sum(errors) / len(errors)

    def roc_auc(self) -> float:
        positives = sum(self.labels)
        negatives = len(self.labels) - positives
        if positives == 0 or negatives == 0:
            return float("nan")

        ranked = sorted(zip(self.probabilities, self.labels, strict=True))
        rank_sum = 0.0
        index = 0
        while index < len(ranked):
            end = index + 1
            while end < len(ranked) and ranked[end][0] == ranked[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2.0
            rank_sum += average_rank * sum(label for _, label in ranked[index:end])
            index = end
        return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    def average_precision(self) -> float:
        positives = sum(self.labels)
        if positives == 0:
            return float("nan")
        true_positives = 0
        precision_sum = 0.0
        ranked = sorted(
            zip(self.probabilities, self.labels, strict=True), reverse=True, key=lambda item: item[0]
        )
        for rank, (_, label) in enumerate(ranked, start=1):
            if label:
                true_positives += 1
                precision_sum += true_positives / rank
        return precision_sum / positives

    def expected_calibration_error(self, bins: int = 10) -> float:
        if bins <= 0:
            raise ValueError("bins must be positive")
        total = len(self.labels)
        error = 0.0
        for bin_index in range(bins):
            lower = bin_index / bins
            upper = (bin_index + 1) / bins
            members = [
                (label, probability)
                for label, probability in zip(self.labels, self.probabilities, strict=True)
                if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
            ]
            if not members:
                continue
            accuracy = sum(label for label, _ in members) / len(members)
            confidence = sum(probability for _, probability in members) / len(members)
            error += len(members) / total * abs(accuracy - confidence)
        return error

    def precision(self, threshold: float) -> float:
        predictions = [probability >= threshold for probability in self.probabilities]
        predicted_positives = sum(predictions)
        if predicted_positives == 0:
            return 0.0
        true_positives = sum(
            label and prediction
            for label, prediction in zip(self.labels, predictions, strict=True)
        )
        return true_positives / predicted_positives

    def recall(self, threshold: float) -> float:
        positives = sum(self.labels)
        if positives == 0:
            return 0.0
        true_positives = sum(
            label and probability >= threshold
            for label, probability in zip(self.labels, self.probabilities, strict=True)
        )
        return true_positives / positives

    def _top_fraction_labels(self, fraction: float) -> list[int]:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        count = max(1, math.ceil(len(self.labels) * fraction))
        ranked = sorted(
            zip(self.probabilities, self.labels, strict=True), reverse=True, key=lambda item: item[0]
        )
        return [label for _, label in ranked[:count]]

    def precision_at_fraction(self, fraction: float) -> float:
        selected = self._top_fraction_labels(fraction)
        return sum(selected) / len(selected)

    def recall_at_fraction(self, fraction: float) -> float:
        positives = sum(self.labels)
        return sum(self._top_fraction_labels(fraction)) / positives if positives else 0.0
