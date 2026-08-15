"""Streaming drift detectors.

Two families live here behind one interface (:class:`DriftDetector`):

* :class:`PageHinkley` and :class:`TwoWindowDrift` -- the in-repo detectors.
  ``PageHinkley`` is a *validated reference implementation*: ``tests/test_drift.py``
  asserts it emits the byte-identical alert sequence to ``river.drift.PageHinkley``
  across randomized configurations, so it is kept for readability rather than
  maintained as an independent guess at the algorithm.
* :class:`RiverDetector` -- an adapter that puts any ``river.base.DriftDetector``
  (``ADWIN``, ``PageHinkley``, ``KSWIN``, ...) behind the same interface, so the
  experiment loops can run a library detector in the production path unchanged.

``TwoWindowDrift`` remains the default production detector because it encodes an
operational requirement the library detectors do not: it alerts only on a *loss
increase* that is both statistically significant and larger than a configured
minimum effect size, and it clips outliers first. ADWIN, by contrast, is
two-sided (it also fires when loss *improves*) and has no effect-size floor. See
README "Drift detection" for the measured comparison.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Protocol

from river.base import DriftDetector as RiverDriftDetector


class DriftDetector(Protocol):
    """A streaming detector: feed it one value, learn whether to alert."""

    def update(self, value: float) -> bool:
        """Consume one observation and return True if drift is signalled."""

    def reset(self) -> None:
        """Discard accumulated state and start a fresh detection window."""


class PageHinkley:
    """Detect a sustained increase in a streaming loss signal.

    One-sided Page-Hinkley test: track the cumulative deviation of each value
    from the running mean, less a tolerance ``delta``, and alert once that
    cumulative sum has risen ``threshold`` above its own running minimum.

    Note that this class *latches* -- once tripped, ``update`` keeps returning
    True until :meth:`reset` is called, which lets a caller decide when to
    re-arm. ``river.drift.PageHinkley`` instead auto-resets on detection; with a
    ``reset()`` after each alert the two emit identical alert sequences.
    """

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
        # Deliberately left as pure-Python accumulation over a deque rather than
        # np.asarray(values): the windows are re-scored on every event once full,
        # and copying a deque into a fresh array each time costs more than the
        # sum it replaces at these window sizes (hundreds, not millions).
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


class RiverDetector:
    """Adapt a ``river`` drift detector to the :class:`DriftDetector` interface.

    river's detectors expose ``update(value)`` returning ``self`` and a separate
    ``drift_detected`` property; this normalizes both into one boolean return so
    the experiment loops are detector-agnostic.
    """

    def __init__(self, detector: RiverDriftDetector) -> None:
        self._prototype = detector
        self._detector = detector.clone()

    @property
    def detector(self) -> RiverDriftDetector:
        """The wrapped river detector, for inspecting its own attributes."""

        return self._detector

    def update(self, value: float) -> bool:
        self._detector.update(value)
        return bool(self._detector.drift_detected)

    def reset(self) -> None:
        # river detectors auto-reset internally on detection and expose no public
        # reset, so re-arming means cloning the unfitted prototype. clone() is
        # river's own idiom for "same hyperparameters, no state".
        self._detector = self._prototype.clone()
