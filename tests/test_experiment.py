from driftguard.experiment import ExperimentConfig, run_experiment


def test_labels_are_never_used_before_they_arrive() -> None:
    result = run_experiment(
        ExperimentConfig(
            samples=1_200,
            warmup=200,
            label_delay=17,
            drift_at=650,
            detector_threshold=8.0,
        )
    )
    audit = result["leakage_audit"]
    assert audit["minimum_observed_delay"] >= 17
    assert audit["violations"] == 0


def test_adaptation_beats_frozen_pre_drift_model() -> None:
    result = run_experiment(
        ExperimentConfig(samples=4_000, warmup=700, label_delay=20, drift_at=2_000)
    )
    assert result["adaptive"]["log_loss"] < result["frozen"]["log_loss"]

