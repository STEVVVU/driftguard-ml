"""Online preprocessing and FTRL-Proximal logistic regression.

Both estimators keep their ``Mapping[str, float]`` public API -- the event loop
stores per-event feature dicts and a mapping keeps sparse rows expressible -- but
the numeric state is held in NumPy arrays and every per-row computation is a
vectorized array expression rather than a Python loop over dict items. Feature
names are interned into stable dense column indices by :class:`FeatureSpace` the
first time they are seen.

Note on float ordering: NumPy sums with pairwise accumulation while the previous
pure-Python implementation summed left to right in dict-insertion order. The two
therefore disagree in the last few bits (~1e-15 relative on the inner products
here). Runs remain deterministic for a given NumPy build, and the tests assert
with tolerances rather than bit equality. See README "Determinism".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

# Guards a division by a near-zero standard deviation in the standardizer. Small
# enough not to distort a genuinely low-variance feature, large enough that a
# constant column maps to 0.0 instead of +-inf.
_MIN_STANDARD_DEVIATION = 1e-8

# Keeps predicted probabilities strictly inside (0, 1) so a confidently wrong
# prediction yields a large-but-finite log loss.
_PROBABILITY_MARGIN = 1e-7

BIAS = "__bias__"


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class FeatureSpace:
    """Intern feature names into stable, insertion-ordered column indices.

    A column index, once assigned, never changes, so array-backed parameter
    vectors can be indexed positionally and simply grow as new names appear.
    """

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._index: dict[str, int] = {}
        # Rows in a stream arrive with the same key order every time, so the
        # name-tuple -> index-array translation is memoized. Without this the
        # per-row dict lookups would dominate the vectorized arithmetic.
        self._gather_cache: dict[tuple[str, ...], NDArray[np.intp]] = {}
        for name in names:
            self.index_of(name)

    def __len__(self) -> int:
        return len(self._index)

    def index_of(self, name: str) -> int:
        """Return the column for ``name``, assigning the next free one if new."""

        index = self._index.get(name)
        if index is None:
            index = len(self._index)
            self._index[name] = index
        return index

    def columns(self, names: tuple[str, ...]) -> NDArray[np.intp]:
        """Return the column indices for ``names``, assigning any unseen ones."""

        cached = self._gather_cache.get(names)
        if cached is None:
            cached = np.fromiter(
                (self.index_of(name) for name in names), dtype=np.intp, count=len(names)
            )
            self._gather_cache[names] = cached
        return cached

    def resolve(
        self, features: Mapping[str, float]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Split a feature mapping into (column indices, values) arrays."""

        names = tuple(features)
        values = np.fromiter(
            (features[name] for name in names), dtype=np.float64, count=len(names)
        )
        return self.columns(names), values


def _grown(array: NDArray[np.float64], required: int) -> NDArray[np.float64]:
    """Return ``array`` with capacity for ``required`` columns, zero-filled.

    Capacity doubles so that a stream discovering N feature names does O(log N)
    reallocations. Columns past the highest name seen stay zero, which is exactly
    the state an unseen feature should have, so logical length is never tracked
    separately from capacity.
    """

    if required <= array.size:
        return array
    grown = np.zeros(max(required, 2 * array.size, 8), dtype=np.float64)
    grown[: array.size] = array
    return grown


class OnlineStandardizer:
    """Standardize features using only observations seen before the current row."""

    def __init__(self) -> None:
        self.space = FeatureSpace()
        self._counts: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self._means: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._m2: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    def _reserve(self, columns: NDArray[np.intp]) -> None:
        required = int(columns.max()) + 1 if columns.size else 0
        if required > self._counts.size:
            counts = np.zeros(max(required, 2 * self._counts.size, 8), dtype=np.int64)
            counts[: self._counts.size] = self._counts
            self._counts = counts
        self._means = _grown(self._means, required)
        self._m2 = _grown(self._m2, required)

    def transform(self, features: Mapping[str, float]) -> dict[str, float]:
        columns, values = self.space.resolve(features)
        self._reserve(columns)
        counts = self._counts[columns]
        # A feature with fewer than two prior observations has no usable variance
        # estimate yet, so it is emitted as 0.0 (its own running mean) rather than
        # guessed at. np.errstate suppresses the divide warning for those columns;
        # np.where discards their value regardless.
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = self._m2[columns] / np.maximum(counts - 1, 1)
            scale = np.maximum(np.sqrt(variance), _MIN_STANDARD_DEVIATION)
            standardized = np.where(counts < 2, 0.0, (values - self._means[columns]) / scale)
        return dict(zip(features, standardized.tolist(), strict=True))

    def update(self, features: Mapping[str, float]) -> None:
        columns, values = self.space.resolve(features)
        self._reserve(columns)
        # Welford's online moments, applied to this row's columns at once. Written
        # with explicit temporaries because the m2 term needs the mean from both
        # before and after the increment.
        self._counts[columns] += 1
        counts = self._counts[columns].astype(np.float64)
        previous_means = self._means[columns]
        delta = values - previous_means
        updated_means = previous_means + delta / counts
        self._means[columns] = updated_means
        self._m2[columns] += delta * (values - updated_means)

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
        self.space = FeatureSpace([BIAS])
        # Column 0 is the intercept, reserved by seeding the space with BIAS.
        self._bias_column = np.array([self.space.index_of(BIAS)], dtype=np.intp)
        self.z: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.n: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    def _reserve(self, columns: NDArray[np.intp]) -> None:
        required = int(columns.max()) + 1 if columns.size else 0
        self.z = _grown(self.z, required)
        self.n = _grown(self.n, required)

    def _weights(self, columns: NDArray[np.intp]) -> NDArray[np.float64]:
        """Reconstruct the proximal weights for ``columns`` from (z, n).

        FTRL-Proximal stores no weights; each is derived on demand as
        ``-(z - sign(z) * l1) / ((beta + sqrt(n)) / alpha + l2)``, which is zero
        wherever ``|z| <= l1``. The intercept is exempt from L1 so a sparse model
        can still shift its base rate.
        """

        z = self.z[columns]
        penalties = np.where(columns == self._bias_column[0], 0.0, self.l1)
        denominator = (self.beta + np.sqrt(self.n[columns])) / self.alpha + self.l2
        weights = -(z - np.sign(z) * penalties) / denominator
        return np.where(np.abs(z) <= penalties, 0.0, weights)

    def _row(
        self, features: Mapping[str, float]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return the (columns, values) of a row with the intercept prepended."""

        columns, values = self.space.resolve(features)
        columns = np.concatenate((self._bias_column, columns))
        values = np.concatenate((np.ones(1, dtype=np.float64), values))
        self._reserve(columns)
        return columns, values

    def predict_proba(self, features: Mapping[str, float]) -> float:
        columns, values = self._row(features)
        score = float(self._weights(columns) @ values)
        return min(max(_sigmoid(score), _PROBABILITY_MARGIN), 1.0 - _PROBABILITY_MARGIN)

    def update(self, features: Mapping[str, float], label: int) -> float:
        columns, values = self._row(features)
        weights = self._weights(columns)
        probability = min(
            max(_sigmoid(float(weights @ values)), _PROBABILITY_MARGIN),
            1.0 - _PROBABILITY_MARGIN,
        )
        gradients = (probability - label) * values
        squared = gradients * gradients
        accumulated = self.n[columns]
        # Per-coordinate learning-rate change. Computed from the pre-update
        # accumulator, so z and n must be written in this order: z consumes the
        # old n, then n absorbs the new gradient.
        sigma = (np.sqrt(accumulated + squared) - np.sqrt(accumulated)) / self.alpha
        self.z[columns] += gradients - sigma * weights
        self.n[columns] += squared
        return probability

    def reset(self) -> None:
        self.z.fill(0.0)
        self.n.fill(0.0)
