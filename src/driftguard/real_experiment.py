"""Chronological evaluation on OpenML's Electricity concept-drift stream."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from river import drift as river_drift

from .datasets import ELECTRICITY_MD5, ELECTRICITY_URL, electricity_stream
from .drift import DriftDetector, RiverDetector, TwoWindowDrift
from .metrics import BinaryMetrics
from .model import FTRLProximal, OnlineStandardizer

# Detectors selectable from the CLI. "two_window" is the default because it is
# the only one with a minimum-effect-size gate; see drift.py's module docstring.
DETECTORS = ("two_window", "river_adwin", "river_page_hinkley")


@dataclass(frozen=True)
class ElectricityExperimentConfig:
    warmup: int = 5_000
    label_delay: int = 48
    replay_size: int = 2_000
    detector: str = "two_window"
    detector_reference: int = 1_500
    detector_recent: int = 750
    detector_min_effect: float = 0.02
    detector_z: float = 3.5

    def validate(self) -> None:
        if self.warmup < 100:
            raise ValueError("warmup must be at least 100")
        if self.label_delay < 0:
            raise ValueError("label_delay cannot be negative")
        if min(self.replay_size, self.detector_reference, self.detector_recent) <= 0:
            raise ValueError("replay and detector window sizes must be positive")
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {self.detector!r}")

    def build_detector(self) -> DriftDetector:
        """Instantiate the configured detector.

        The river alternatives are wired up so the library detectors can drive the
        production path unchanged, which is what makes the "our detector agrees
        with the standard implementations" claim checkable end to end and not just
        in a unit test.
        """

        if self.detector == "river_adwin":
            return RiverDetector(river_drift.ADWIN(delta=0.002))
        if self.detector == "river_page_hinkley":
            # mode="up"/alpha=1.0 mirror the in-repo one-sided, un-faded test;
            # the threshold is scaled to the loss magnitude of this stream.
            return RiverDetector(
                river_drift.PageHinkley(
                    delta=0.005,
                    threshold=20.0,
                    min_instances=self.detector_recent,
                    mode="up",
                    alpha=1.0,
                )
            )
        return TwoWindowDrift(
            reference_size=self.detector_reference,
            recent_size=self.detector_recent,
            min_effect=self.detector_min_effect,
            z_threshold=self.detector_z,
        )


@dataclass(frozen=True)
class PendingObservation:
    due_at: int
    event_index: int
    frozen_features: dict[str, float]
    adaptive_features: dict[str, float]
    label: int
    frozen_probability: float
    adaptive_probability: float


def _log_loss(label: int, probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(label * math.log(probability) + (1 - label) * math.log(1 - probability))


# Shared default instance. Safe as a default argument because the dataclass is
# frozen, so no caller can mutate the value the next one sees.
DEFAULT_CONFIG = ElectricityExperimentConfig()


def run_electricity_experiment(
    dataset: str | Path,
    config: ElectricityExperimentConfig = DEFAULT_CONFIG,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Compare a deployment-frozen model with delayed-label online adaptation."""

    config.validate()
    frozen_scaler = OnlineStandardizer()
    adaptive_scaler = OnlineStandardizer()
    frozen = FTRLProximal()
    adaptive = FTRLProximal()
    detector = config.build_detector()
    pending: deque[PendingObservation] = deque()
    replay: deque[tuple[dict[str, float], int]] = deque(maxlen=config.replay_size)
    frozen_metrics = BinaryMetrics()
    adaptive_metrics = BinaryMetrics()
    drift_alerts: list[int] = []
    labels_processed = 0
    minimum_delay: int | None = None
    timing_violations = 0

    def reveal(current_index: int) -> None:
        nonlocal labels_processed, minimum_delay, timing_violations
        while pending and pending[0].due_at <= current_index:
            item = pending.popleft()
            observed_delay = current_index - item.event_index
            minimum_delay = (
                observed_delay if minimum_delay is None else min(minimum_delay, observed_delay)
            )
            timing_violations += int(observed_delay < config.label_delay)
            labels_processed += 1

            if item.event_index < config.warmup:
                frozen.update(item.frozen_features, item.label)
            else:
                frozen_metrics.add(item.label, item.frozen_probability)
                adaptive_metrics.add(item.label, item.adaptive_probability)

            adaptive.update(item.adaptive_features, item.label)
            replay.append((item.adaptive_features, item.label))
            if item.event_index >= config.warmup and detector.update(
                _log_loss(item.label, item.frozen_probability)
            ):
                drift_alerts.append(current_index)
                adaptive.reset()
                for features, label in replay:
                    adaptive.update(features, label)
                detector.reset()

    rows = 0
    for observation in electricity_stream(dataset):
        reveal(observation.index)
        frozen_features = frozen_scaler.transform(observation.features)
        if observation.index < config.warmup:
            frozen_scaler.update(observation.features)
        adaptive_features = adaptive_scaler.transform_then_update(observation.features)
        pending.append(
            PendingObservation(
                due_at=observation.index + config.label_delay,
                event_index=observation.index,
                frozen_features=frozen_features,
                adaptive_features=adaptive_features,
                label=observation.label,
                frozen_probability=frozen.predict_proba(frozen_features),
                adaptive_probability=adaptive.predict_proba(adaptive_features),
            )
        )
        rows += 1

    if rows <= config.warmup:
        raise ValueError(f"dataset has {rows} rows, not enough for warmup {config.warmup}")
    reveal(rows + config.label_delay)

    frozen_summary = frozen_metrics.summary(threshold=0.5)
    adaptive_summary = adaptive_metrics.summary(threshold=0.5)
    frozen_loss = float(frozen_summary["log_loss"])
    adaptive_loss = float(adaptive_summary["log_loss"])
    result: dict[str, object] = {
        "dataset": {
            "name": "OpenML Electricity",
            "url": ELECTRICITY_URL,
            "md5": ELECTRICITY_MD5,
            "rows": rows,
            "evaluation_order": "source chronological order",
        },
        "config": asdict(config),
        "frozen": frozen_summary,
        "adaptive": adaptive_summary,
        "relative_log_loss_improvement": (frozen_loss - adaptive_loss) / frozen_loss,
        "drift_alerts": drift_alerts,
        "leakage_audit": {
            "labels_processed": labels_processed,
            "minimum_observed_delay": minimum_delay,
            "violations": timing_violations,
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "electricity_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
