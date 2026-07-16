"""Online preprocessing and FTRL-Proximal logistic regression."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class OnlineStandardizer:
    """Standardize features using only observations seen before the current row."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.means: dict[str, float] = defaultdict(float)
        self.m2: dict[str, float] = defaultdict(float)

    def transform(self, features: Mapping[str, float]) -> dict[str, float]:
        transformed: dict[str, float] = {}
        for name, value in features.items():
            count = self.counts[name]
            if count < 2:
                transformed[name] = 0.0
                continue
            variance = self.m2[name] / (count - 1)
            transformed[name] = (value - self.means[name]) / max(math.sqrt(variance), 1e-8)
        return transformed

    def update(self, features: Mapping[str, float]) -> None:
        for name, value in features.items():
            self.counts[name] += 1
            count = self.counts[name]
            delta = value - self.means[name]
            self.means[name] += delta / count
            self.m2[name] += delta * (value - self.means[name])

    def transform_then_update(self, features: Mapping[str, float]) -> dict[str, float]:
        transformed = self.transform(features)
        self.update(features)
        return transformed


class FTRLProximal:
    """Sparse logistic regression trained with Follow-The-Regularized-Leader."""

    def __init__(
        self,
        *,
        alpha: float = 0.08,
        beta: float = 1.0,
        l1: float = 0.0,
        l2: float = 1.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z: dict[str, float] = defaultdict(float)
        self.n: dict[str, float] = defaultdict(float)

    def _weight(self, name: str) -> float:
        z = self.z[name]
        l1 = 0.0 if name == "__bias__" else self.l1
        if abs(z) <= l1:
            return 0.0
        sign = -1.0 if z < 0 else 1.0
        denominator = (self.beta + math.sqrt(self.n[name])) / self.alpha + self.l2
        return -(z - sign * l1) / denominator

    def predict_proba(self, features: Mapping[str, float]) -> float:
        score = self._weight("__bias__")
        score += sum(self._weight(name) * value for name, value in features.items())
        return min(max(_sigmoid(score), 1e-7), 1.0 - 1e-7)

    def update(self, features: Mapping[str, float], label: int) -> float:
        probability = self.predict_proba(features)
        gradient_scale = probability - label
        values = {"__bias__": 1.0, **features}
        for name, value in values.items():
            gradient = gradient_scale * value
            old_weight = self._weight(name)
            sigma = (math.sqrt(self.n[name] + gradient * gradient) - math.sqrt(self.n[name]))
            sigma /= self.alpha
            self.z[name] += gradient - sigma * old_weight
            self.n[name] += gradient * gradient
        return probability

    def reset(self) -> None:
        self.z.clear()
        self.n.clear()

