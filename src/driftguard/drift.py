"""Streaming drift detectors."""

from __future__ import annotations

import math
from collections import deque


class PageHinkley:
    """Detect a sustained increase in a streaming loss signal."""

    def __init__(self, *, delta: float = 0.002, threshold: float = 8.0, min_samples: int = 100):
        if threshold <= 0 or min_samples <= 0:
            raise ValueError("threshold and min_samples must be positive")
        self.delta = delta
        self.threshold = threshold
        self.min_samples = min_samples
        self.reset()

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        return self.count >= self.min_samples and self.cumulative - self.minimum > self.threshold

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0


class TwoWindowDrift:
    """Detect a material, statistically significant increase between loss windows."""

    def __init__(
        self,
        *,
        reference_size: int = 500,
        recent_size: int = 300,
        min_effect: float = 0.025,
        z_threshold: float = 3.5,
        clip_at: float = 3.0,
    ) -> None:
        if min(reference_size, recent_size) < 2:
            raise ValueError("window sizes must be at least two")
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.min_effect = min_effect
        self.z_threshold = z_threshold
        self.clip_at = clip_at
        self.reset()

    @staticmethod
    def _mean_variance(values: deque[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, variance

    def update(self, value: float) -> bool:
        value = min(max(value, 0.0), self.clip_at)
        if len(self.reference) < self.reference_size:
            self.reference.append(value)
            return False
        self.recent.append(value)
        if len(self.recent) < self.recent_size:
            return False

        reference_mean, reference_variance = self._mean_variance(self.reference)
        recent_mean, recent_variance = self._mean_variance(self.recent)
        effect = recent_mean - reference_mean
        standard_error = math.sqrt(
            reference_variance / len(self.reference) + recent_variance / len(self.recent)
        )
        z_score = effect / max(standard_error, 1e-12)
        return effect >= self.min_effect and z_score >= self.z_threshold

    def reset(self) -> None:
        self.reference: deque[float] = deque(maxlen=self.reference_size)
        self.recent: deque[float] = deque(maxlen=self.recent_size)
