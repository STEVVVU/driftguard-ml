# DriftGuard ML

A delayed-feedback online-learning system built around production failure
modes: concept drift, temporal leakage, stale models, calibration, and labels
that arrive after predictions are served.

This is not a notebook. It is a deterministic experiment package with a CLI,
an auditable event-time loop, checksum-verified real data, unit tests, saved
metrics, and CI.

## What it demonstrates

- **Real chronological evaluation:** 45,312 observations from OpenML's
  Electricity stream are evaluated in source order, never randomly split.
- **Leakage-safe prequential evaluation:** predict first, then learn when the
  label becomes observable 48 events later.
- **Online learning:** sparse FTRL-Proximal logistic regression with a
  past-only streaming standardizer.
- **Drift monitoring:** a clipped two-window significance test watches a fixed
  reference model so adaptation cannot hide degradation.
- **Drift response:** the adaptive model resets and replays a bounded recent
  label buffer after an alert.
- **Evaluation depth:** log loss, Brier score, calibration error, ROC-AUC,
  average precision, accuracy, and a label-timing audit.
- **Reproducibility:** OpenML content is checked against its published MD5;
  a seeded synthetic stream provides a known drift-onset control.

## Resume highlights

- Built an online ML system that keeps learning after deployment while handling
  delayed labels, changing data, and stale-model risk.
- Improved log loss by **26.1%**, ROC-AUC from **0.773 to 0.850**, and accuracy
  from **74.03% to 77.73%** on 40,312 chronological holdout observations.
- Prevented future information from leaking into training by enforcing a
  48-event label delay across all **45,312** records, with **zero violations**.
- Implemented FTRL-Proximal learning, past-only feature scaling, drift alerts,
  bounded replay, checksum-verified ingestion, automated tests, and CI.

## Architecture

```text
chronological event stream
       |
       v
past-only standardizer ---> predict frozen + adaptive models
       |                                  |
       |                            pending-label queue
       |                                  |
       |                   label becomes observable at t + delay
       |                                  |
       +--------------------------> metrics + leakage audit
                                          |
                              adaptive update + replay buffer
                                          |
                         fixed-reference loss drift detector
```

The primary benchmark trains both models on the first 5,000 real observations.
The frozen model then represents a stale deployed artifact while the adaptive
model continues learning from labels after their 48-event delay.

## Real-data result

Dataset: OpenML Electricity, all 45,312 rows in source chronological order.
Metrics cover the 40,312 post-warmup observations.

| Metric | Frozen model | Adaptive model |
| --- | ---: | ---: |
| Log loss | 0.6409 | **0.4737** |
| Accuracy | 74.03% | **77.73%** |
| ROC-AUC | 0.7725 | **0.8501** |
| Average precision | 0.7673 | **0.8228** |
| Calibration error | 0.1380 | **0.0216** |

The adaptive path lowers log loss by **26.1%** and records seven drift alerts.
The audit processed all 45,312 labels with a minimum observed delay of 48 and
**zero early-label violations**. The timestamp column controls ordering but is
excluded from model features to avoid extrapolation artifacts.

The checked-in
[`artifacts/electricity_metrics.json`](artifacts/electricity_metrics.json)
contains the complete result and data provenance.

## Controlled drift result

The seeded fraud-risk generator remains as a stress test with known ground
truth. At event 6,600 it changes both covariates and the conditional fraud
mechanism. The adaptive path lowers log loss by **6.97%**, detects the shift at
event 6,950, and improves top-5%-review precision from 0.150 to 0.246. See
[`artifacts/metrics.json`](artifacts/metrics.json).

## Run it

No runtime dependencies are required.

```powershell
$env:PYTHONPATH = "src"
python -m driftguard electricity --output artifacts
python -m driftguard run --samples 12000 --output artifacts
```

The first command downloads the public dataset to `data/`, verifies OpenML's
published MD5, and runs the chronological benchmark. No runtime dependencies
are required.

For development:

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Repository map

```text
src/driftguard/
  datasets.py    verified OpenML download and chronological adapter
  model.py       online standardization and FTRL-Proximal
  stream.py      documented concept-shift generator
  drift.py       Page-Hinkley and robust two-window detectors
  metrics.py     imbalanced classification and calibration metrics
  experiment.py delayed-label event loop and leakage audit
  real_experiment.py real-data frozen-versus-adaptive evaluation
tests/           model, detector, metric, and temporal-contract tests
```

## Dataset provenance

The Electricity dataset is OpenML dataset 151, marked public and distributed
as 45,312 half-hour observations from the Australian electricity market. The
adapter uses OpenML's official download URL and published checksum. Source
ordering is preserved. The synthetic benchmark complements it by making exact
drift onset inspectable; it is not presented as real-world fraud evidence.

## Research context

The production-project framing follows the end-to-end standard illustrated by
[Made With ML](https://github.com/GokuMohandas/Made-With-ML). FTRL-Proximal is
used because it is a strong, interpretable online baseline for sparse risk and
advertising systems. Drift evaluation is intentionally performed in event time
rather than by randomly splitting rows.
