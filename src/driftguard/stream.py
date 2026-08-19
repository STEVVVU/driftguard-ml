"""A deterministic transaction stream with a documented concept shift."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Transaction:
    index: int
    features: dict[str, float]
    label: int
    regime: str


def transaction_stream(
    samples: int, *, seed: int = 7, drift_at: int | None = None
) -> Iterator[Transaction]:
    """Yield synthetic transactions whose fraud mechanism changes once.

    The post-drift regime changes both covariates and the conditional target
    distribution. That makes adaptation measurable without downloading data or
    hiding the ground-truth data-generating process.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    drift_at = drift_at if drift_at is not None else int(samples * 0.55)
    if not 0 < drift_at < samples:
        raise ValueError("drift_at must be inside the stream")

    rng = random.Random(seed)
    for index in range(samples):
        post_drift = index >= drift_at
        amount_log = rng.gauss(3.0 + 0.25 * post_drift, 0.9)
        velocity = min(rng.expovariate(1 / (1.8 + 0.8 * post_drift)), 10.0)
        foreign = float(rng.random() < (0.12 + 0.08 * post_drift))
        device_age = min(rng.expovariate(1 / (50.0 if post_drift else 80.0)), 365.0)
        trusted_payee = float(rng.random() < (0.40 - 0.10 * post_drift))
        hour = rng.randrange(24)
        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

        if post_drift:
            score = (
                -5.1
                + 0.30 * amount_log
                + 0.48 * velocity
                + 0.35 * foreign
                - 0.024 * device_age
                - 1.00 * trusted_payee
                + 1.20 * hour_cos
            )
        else:
            score = (
                -5.0
                + 0.62 * amount_log
                + 0.18 * velocity
                + 1.35 * foreign
                - 0.012 * device_age
                - 1.15 * trusted_payee
                - 0.85 * hour_sin
            )
        score += rng.gauss(0.0, 0.35)
        probability = 1.0 / (1.0 + math.exp(-score))
        label = int(rng.random() < probability)

        yield Transaction(
            index=index,
            features={
                "amount_log": amount_log,
                "velocity_1h": velocity,
                "foreign": foreign,
                "device_age_days": device_age,
                "trusted_payee": trusted_payee,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
            },
            label=label,
            regime="shifted" if post_drift else "initial",
        )
