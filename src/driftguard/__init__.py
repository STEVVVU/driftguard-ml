"""DriftGuard: leakage-safe online learning under delayed labels."""

from importlib.metadata import PackageNotFoundError, version

from .experiment import ExperimentConfig, run_experiment

__all__ = ["ExperimentConfig", "__version__", "run_experiment"]

try:
    # Read from installed distribution metadata rather than restating the literal.
    # A hardcoded string here had already drifted: pyproject said 0.2.0 while this
    # module still said 0.1.0, so the two disagreed depending on where you looked.
    __version__ = version("driftguard-ml")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"
