from driftguard.real_experiment import (
    ElectricityExperimentConfig,
    run_electricity_experiment,
)


def _write_regime_reversal(path, rows: int = 800, warmup: int = 200) -> None:
    lines = [
        "@relation electricity",
        "@attribute date numeric",
        "@attribute day {1,2,3,4,5,6,7}",
        "@attribute period numeric",
        "@attribute nswprice numeric",
        "@attribute nswdemand numeric",
        "@attribute vicprice numeric",
        "@attribute vicdemand numeric",
        "@attribute transfer numeric",
        "@attribute class {UP,DOWN}",
        "@data",
    ]
    for index in range(rows):
        signal = float(index % 2)
        label = signal if index < warmup else 1.0 - signal
        lines.append(
            f"{index / rows},{index % 7 + 1},0,{signal},0.2,0.3,0.4,0.5,"
            f"{'UP' if label else 'DOWN'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_real_experiment_respects_delay_and_adapts_after_reversal(tmp_path) -> None:
    dataset = tmp_path / "electricity.arff"
    _write_regime_reversal(dataset)
    result = run_electricity_experiment(
        dataset,
        ElectricityExperimentConfig(
            warmup=200,
            label_delay=9,
            replay_size=100,
            detector_reference=100,
            detector_recent=50,
            detector_min_effect=0.05,
            detector_z=2.0,
        ),
    )

    assert result["dataset"]["rows"] == 800
    assert result["adaptive"]["log_loss"] < result["frozen"]["log_loss"]
    assert result["leakage_audit"]["minimum_observed_delay"] >= 9
    assert result["leakage_audit"]["violations"] == 0

