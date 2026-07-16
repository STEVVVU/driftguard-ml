from driftguard.drift import PageHinkley, TwoWindowDrift


def test_page_hinkley_detects_sustained_loss_increase() -> None:
    detector = PageHinkley(delta=0.001, threshold=2.0, min_samples=30)
    stable = [detector.update(0.2) for _ in range(100)]
    shifted = [detector.update(0.8) for _ in range(100)]
    assert not any(stable)
    assert any(shifted)


def test_two_window_detector_ignores_spikes_but_finds_shift() -> None:
    detector = TwoWindowDrift(reference_size=50, recent_size=30, min_effect=0.1, z_threshold=3.0)
    stable = [0.2] * 50 + [0.2] * 29 + [2.5]
    assert not any(detector.update(value) for value in stable)
    assert any(detector.update(0.8) for _ in range(60))
