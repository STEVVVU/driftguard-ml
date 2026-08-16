"""Prequential experiment runner with delayed labels and drift response."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .drift import TwoWindowDrift
from .metrics import BinaryMetrics
from .model import FTRLProximal, OnlineStandardizer
from .stream import transaction_stream


@dataclass(frozen=True)
class ExperimentConfig:
    samples: int = 12_000
    seed: int = 7
    warmup: int = 2_000
    label_delay: int = 40
    drift_at: int | None = None
    replay_size: int = 600
    detector_threshold: float = 3.5

    def validate(self) -> None:
        drift_at = self.drift_at if self.drift_at is not None else int(self.samples * 0.55)
        if self.samples <= 500:
            raise ValueError("samples must be greater than 500")
        if not 100 <= self.warmup < drift_at:
            raise ValueError("warmup must be at least 100 and finish before drift")
        if self.label_delay < 0:
            raise ValueError("label_delay cannot be negative")
        if self.replay_size <= 0:
            raise ValueError("replay_size must be positive")


@dataclass(frozen=True)
class PendingLabel:
    due_at: int
    event_index: int
    features: dict[str, float]
    label: int
    frozen_probability: float
    adaptive_probability: float
    regime: str


def _log_loss(label: int, probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(label * math.log(probability) + (1 - label) * math.log(1 - probability))


# Shared default instance. Safe as a default argument because the dataclass is
# frozen, so no caller can mutate the value the next one sees.
DEFAULT_CONFIG = ExperimentConfig()


def run_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG, *, output_dir: str | Path | None = None
) -> dict[str, object]:
    """Run an online experiment in event time.

    A row is predicted when it arrives, but neither model can train on it until
    its label becomes available. The frozen model stops learning after warmup;
    the adaptive model keeps learning and replays recent rows after a detected
    shift.
    """

    config.validate()
    drift_at = config.drift_at if config.drift_at is not None else int(config.samples * 0.55)
    scaler = OnlineStandardizer()
    frozen = FTRLProximal()
    adaptive = FTRLProximal()
    detector = TwoWindowDrift(z_threshold=config.detector_threshold)
    pending: deque[PendingLabel] = deque()
    replay: deque[tuple[dict[str, float], int]] = deque(maxlen=config.replay_size)
    frozen_metrics = BinaryMetrics()
    adaptive_metrics = BinaryMetrics()
    detections: list[int] = []
    label_audit: list[dict[str, int]] = []

    def reveal(current_index: int) -> None:
        while pending and pending[0].due_at <= current_index:
            item = pending.popleft()
            if item.event_index >= config.warmup:
                frozen_metrics.add(item.label, item.frozen_probability)
                adaptive_metrics.add(item.label, item.adaptive_probability)
            if item.event_index < config.warmup:
                frozen.update(item.features, item.label)
            adaptive.update(item.features, item.label)
            replay.append((item.features, item.label))
            label_audit.append(
                {
                    "event_index": item.event_index,
                    "revealed_at": current_index,
                }
            )

            if item.event_index >= config.warmup:
                # Monitor a fixed reference model. An adaptive model can mask the
                # shift it is learning through, while reference loss provides a
                # stable signal for operational alerting.
                loss = _log_loss(item.label, item.frozen_probability)
                if detector.update(loss):
                    detections.append(current_index)
                    adaptive.reset()
                    for features, label in replay:
                        adaptive.update(features, label)
                    detector.reset()

    for transaction in transaction_stream(config.samples, seed=config.seed, drift_at=drift_at):
        reveal(transaction.index)
        features = scaler.transform_then_update(transaction.features)
        pending.append(
            PendingLabel(
                due_at=transaction.index + config.label_delay,
                event_index=transaction.index,
                features=features,
                label=transaction.label,
                frozen_probability=frozen.predict_proba(features),
                adaptive_probability=adaptive.predict_proba(features),
                regime=transaction.regime,
            )
        )

    reveal(config.samples + config.label_delay)
    result: dict[str, object] = {
        "config": {**asdict(config), "drift_at": drift_at},
        "frozen": frozen_metrics.summary(),
        "adaptive": adaptive_metrics.summary(),
        "drift_detections": detections,
        "leakage_audit": {
            "labels_processed": len(label_audit),
            "minimum_observed_delay": min(
                row["revealed_at"] - row["event_index"] for row in label_audit
            ),
            "violations": sum(
                row["revealed_at"] - row["event_index"] < config.label_delay
                for row in label_audit
            ),
        },
    }
    frozen_loss = float(result["frozen"]["log_loss"])  # type: ignore[index]
    adaptive_loss = float(result["adaptive"]["log_loss"])  # type: ignore[index]
    result["relative_log_loss_improvement"] = (frozen_loss - adaptive_loss) / frozen_loss

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
