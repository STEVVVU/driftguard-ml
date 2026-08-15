import random

from river import drift as river_drift

from driftguard.drift import PageHinkley, RiverDetector, TwoWindowDrift


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


def _alert_indices_ours(detector: PageHinkley, values: list[float]) -> list[int]:
    """Alert indices from the in-repo detector, re-armed after each alert.

    The in-repo detector latches until reset while river's auto-resets on
    detection, so this reset is what makes the two sequences comparable at all --
    without it ours would report every subsequent event as an alert.
    """

    alerts = []
    for index, value in enumerate(values):
        if detector.update(value):
            alerts.append(index)
            detector.reset()
    return alerts


def _alert_indices_river(detector: RiverDetector, values: list[float]) -> list[int]:
    return [index for index, value in enumerate(values) if detector.update(value)]


def test_page_hinkley_matches_river_reference_across_configurations() -> None:
    """Pin the in-repo Page-Hinkley to river's as a validated reference impl.

    Exact sequence equality is asserted, not just "both eventually fired": the
    recurrences are algebraically the same, so any divergence beyond float noise
    would be a real behavioural difference worth failing on.
    """

    configurations = [
        (0.001, 2.0, 10),
        (0.002, 8.0, 30),
        (0.005, 20.0, 30),
        (0.01, 50.0, 100),
    ]
    for seed, (delta, threshold, min_samples) in enumerate(configurations):
        rng = random.Random(seed)
        values = [rng.gauss(0.2, 0.08) for _ in range(600)]
        values += [rng.gauss(0.9, 0.08) for _ in range(600)]

        ours = _alert_indices_ours(
            PageHinkley(delta=delta, threshold=threshold, min_samples=min_samples), values
        )
        theirs = _alert_indices_river(
            RiverDetector(
                # mode="up" and alpha=1.0 select river's one-sided, un-faded
                # variant, which is the test the in-repo class implements. river
                # defaults to mode="both" with alpha=0.9999 (exponential fading).
                river_drift.PageHinkley(
                    delta=delta,
                    threshold=threshold,
                    min_instances=min_samples,
                    mode="up",
                    alpha=1.0,
                )
            ),
            values,
        )

        assert ours == theirs, f"config {(delta, threshold, min_samples)} diverged"
        # Post-shift detection is asserted, but pre-shift silence deliberately is
        # not: the loosest configuration here (delta=0.001, threshold=2.0) does
        # false-positive on the noise prefix. That is a real property of a
        # too-sensitive threshold, and river reproduces it at the same index --
        # which is the agreement this test exists to pin down.
        assert any(index >= 600 for index in ours), "no alert after the injected shift"


def test_river_adwin_and_in_repo_detectors_agree_a_shift_occurred() -> None:
    """ADWIN, Page-Hinkley and the two-window test should all flag one shift.

    Alert *indices* are not compared here: ADWIN adapts its own window and has no
    effect-size floor, so it reacts on a different schedule by design. What is
    asserted is the operational claim -- all three flag the injected shift, and
    none of them flags the stable prefix.
    """

    rng = random.Random(7)
    stable = [rng.gauss(0.25, 0.05) for _ in range(1_000)]
    shifted = [rng.gauss(0.90, 0.05) for _ in range(1_000)]

    detectors: dict[str, PageHinkley | TwoWindowDrift | RiverDetector] = {
        "adwin": RiverDetector(river_drift.ADWIN(delta=0.002)),
        "page_hinkley": PageHinkley(delta=0.005, threshold=20.0, min_samples=100),
        "two_window": TwoWindowDrift(
            reference_size=400, recent_size=200, min_effect=0.05, z_threshold=3.5
        ),
    }
    for name, detector in detectors.items():
        assert not any(detector.update(value) for value in stable), f"{name} alerted pre-shift"
        assert any(detector.update(value) for value in shifted), f"{name} missed the shift"


def test_river_detector_reset_rearms_a_fired_detector() -> None:
    detector = RiverDetector(river_drift.ADWIN(delta=0.002))
    rng = random.Random(1)
    for _ in range(500):
        detector.update(rng.gauss(0.2, 0.05))
    assert any(detector.update(rng.gauss(2.0, 0.05)) for _ in range(500))

    detector.reset()
    # A re-armed detector has no history, so a constant stream looks stable again.
    assert not any(detector.update(0.5) for _ in range(200))
