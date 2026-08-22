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
  past-only streaming standardizer, both vectorized with NumPy.
- **Drift monitoring:** a clipped two-window significance test watches a fixed
  reference model so adaptation cannot hide degradation, cross-validated
  against `river`'s reference detectors.
- **Drift response:** the adaptive model resets and replays a bounded recent
  label buffer after an alert.
- **Evaluation depth:** log loss, Brier score, calibration error, ROC-AUC,
  average precision, accuracy, and a label-timing audit — scored with
  `scikit-learn` so every number is reproducible in a notebook.
- **Feature-level drift reporting:** an `evidently` report showing *which*
  inputs moved between the training window and the serving window.
- **Reproducibility:** OpenML content is checked against its published MD5;
  a seeded synthetic stream provides a known drift-onset control.

## Resume highlights

- Built an online ML system that keeps learning after deployment while handling
  delayed labels, changing data, and stale-model risk.
- Improved log loss by **26.1%**, ROC-AUC from **0.773 to 0.850**, and accuracy
  from **74.03% to 77.73%** on 40,312 chronological holdout observations.
- Prevented future information from leaking into training by enforcing a
  48-event label delay across all **45,312** records, with **zero violations**.
- Validated hand-written drift detectors against `river`'s reference
  implementations, reaching the same adaptation quality with **7 alerts instead
  of 130** — a 94% cut in operator pages for equal model quality.
- Implemented FTRL-Proximal learning, past-only feature scaling, drift alerts,
  bounded replay, checksum-verified ingestion, `evidently` drift reporting,
  automated tests, type checking, and CI.

## Dependencies

| Package | Floor | Used for |
| --- | --- | --- |
| `numpy` | `>=2.1` | vectorized standardizer / FTRL math, metric buffers |
| `pandas` | `>=2.2` | ARFF ingestion via `read_csv`, report frames |
| `scikit-learn` | `>=1.5` | log loss, Brier, ROC-AUC, average precision, threshold metrics |
| `river` | `>=0.22` | reference streaming drift detectors (`ADWIN`, `PageHinkley`) |
| `evidently` | `>=0.7` | feature-level drift report artifact |
| `pyarrow` | `>=17` | parquet round-trip of the report's reference/current split |

Dev extra: `pytest>=8`, `ruff>=0.5`, `mypy>=1.11`, `pandas-stubs>=2.2`.
Verified against numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, river 0.25.0,
evidently 0.7.21, pyarrow 25.0.0 on CPython 3.11.9.

### Why `river` and `evidently`

They answer two different questions, and this project needs both.

`river` is the standard library for *online* ML, and its `drift` module is the
reference implementation of the detectors this project is about. Rather than ask
a reviewer to take a hand-written Page-Hinkley on trust, the in-repo detector is
pinned to river's in a test (see below).

`evidently` answers the question the online detector cannot. The event loop
detects drift from a single scalar — the reference model's loss — which tells you
*that* something broke but not *what*. `evidently`'s `DataDriftPreset` runs a
per-feature statistical comparison of the training window against the serving
window, which is the artifact you actually attach to an incident.

## Drift detection: reference implementations, not reinventions

**Both paths are kept, and the hand-written one is validated against the
library.** The choice was between deleting the in-repo detectors and keeping them
as unverified duplicates; this is the third option.

`tests/test_drift.py::test_page_hinkley_matches_river_reference_across_configurations`
asserts the in-repo `PageHinkley` emits the **byte-identical alert index
sequence** as `river.drift.PageHinkley(mode="up", alpha=1.0)` across four
`(delta, threshold, min_samples)` configurations. A wider 200-configuration
randomized sweep (seeds 0–199, `delta` 5e-4–0.02, `threshold` 1–100,
`min_samples` 5–300, 600 pre-shift + 600 post-shift observations each) matched
exactly as well — zero mismatched alert sequences. The single
behavioural difference is re-arming: the in-repo detector *latches* until
`reset()` is called (so a caller controls when to re-arm), while river's
auto-resets on detection — so the comparison resets after each alert.

`TwoWindowDrift` remains the **production default**, because it encodes an
operational requirement neither library detector has: it alerts only on a
one-sided loss *increase* that clears both a significance threshold and a
**minimum effect size**, after clipping outliers. ADWIN is two-sided (it also
fires when loss *improves*) and has no effect-size floor. That difference is
measurable — all three drive the same loop via `--detector`:

| Detector | Alerts | Adaptive log loss | Improvement vs frozen | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| `two_window` (default) | 7 | **0.4737** | **26.09%** | **0.8501** |
| `river_page_hinkley` | 41 | 0.4773 | 25.54% | 0.8472 |
| `river_adwin` | 130 | 0.4789 | 25.29% | 0.8462 |

All three adapt successfully and land within 0.8 points of each other, which is
the useful result: the outcome is robust to the detector choice. `two_window` is
kept as the default because it reaches a marginally better model with **7 alerts
instead of 130** — in an operational setting the alert count *is* the cost, since
each one pages someone and triggers a reset-and-replay.

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
| Average precision | 0.7673 | **0.8227** |
| Brier score | 0.2010 | **0.1545** |
| Calibration error | 0.1380 | **0.0216** |

The adaptive path lowers log loss by **26.09%** and records seven drift alerts
(at events 7,366 / 11,607 / 15,762 / 19,481 / 23,044 / 28,979 / 31,691). The
audit processed all 45,312 labels with a minimum observed delay of 48 and
**zero early-label violations**. The timestamp column controls ordering but is
excluded from model features to avoid extrapolation artifacts.

The checked-in
[`artifacts/electricity_metrics.json`](artifacts/electricity_metrics.json)
contains the complete result and data provenance.

## Feature-level drift report

`driftguard report` compares the 5,000-row training window against the
40,312-row serving window with a Kolmogorov-Smirnov test per feature, and writes
[`artifacts/electricity_drift_report.json`](artifacts/electricity_drift_report.json)
plus a rendered HTML report and a parquet copy of both splits.

**5 of 13 features drift** at p < 0.05: `nswprice`, `nswdemand`, `vicprice`,
`vicdemand`, `transfer`. The weekday one-hots and `period` do not (p ≈ 1.0),
which is the expected result — the calendar structure is stable while the market
signal is not. That is consistent with the online detector: the loss degradation
it alerts on is driven by the price/demand columns, not by a change in when
events arrive.

Two honest caveats, both surfaced in the report output:

- `vicprice`, `vicdemand` and `transfer` are **constant** across the first 5,000
  rows — the Electricity stream carries no Victorian market data in that early
  period. They are therefore trivially drifted, and the *magnitude* of their
  p-values carries no ranking information. The report names them in
  `constant_in_reference_window` and logs a warning.
- The statistical test is **pinned** to K-S rather than left to Evidently's
  default. Left to itself the preset selects its test from the sample size (K-S
  p-value below ~1,000 rows, normed Wasserstein distance above) and those report
  on *opposite* scales — drift is a p-value below the threshold but a distance
  above it. Pinning it keeps the number comparable between a full run and a test
  fixture.

## Controlled drift result

The seeded fraud-risk generator remains as a stress test with known ground
truth. At event 6,600 it changes both covariates and the conditional fraud
mechanism. The adaptive path lowers log loss by **6.97%** (0.18313 → 0.17037),
detects the shift at event 6,950, and improves top-5%-review precision from
0.150 to 0.246. See [`artifacts/metrics.json`](artifacts/metrics.json).

## Run it

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

.venv\Scripts\python -m driftguard electricity --output artifacts
.venv\Scripts\python -m driftguard report --output artifacts
.venv\Scripts\python -m driftguard run --samples 12000 --output artifacts
```

The first command downloads the public dataset to `data/`, verifies OpenML's
published MD5, and runs the chronological benchmark. `--detector` selects between
`two_window` (default), `river_adwin` and `river_page_hinkley`. Add `--verbose`
for progress logging; without it stdout stays a single parseable JSON document.

For development:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\mypy
```

## Determinism and numerical tolerance

Seeded runs remain reproducible: the synthetic stream is driven by a seeded
`random.Random`, and the real benchmark has no stochastic component at all.

Migrating the numeric core to NumPy and scikit-learn did move some digits, and
the reasons are worth being precise about:

- **NumPy accumulation order.** NumPy sums with pairwise accumulation while the
  previous implementation summed left-to-right in dict-insertion order. Predicted
  probabilities agree with the old implementation to **≤1e-15**: the worst
  per-event disagreement was 5.0e-16 on the reference replay (4,000 sparse events
  across three hyperparameter settings, including a mid-run reset) and ranged
  **1.1e-16 – 6.1e-16** across twelve further randomized replays (`random.Random`
  seeds 1000–1011; 8/40/200 features, feature scale 0.5–10, `alpha` 0.05–0.2,
  `beta` 0.5–2, `l1` 0–1, `l2` 0.01–2, reset at event 2,000). This is a range
  rather than a single number on purpose — the bound depends on the workload's
  feature count and scale, so a point value quoted from one replay would not
  survive another. The standardizer output is bit-identical. Log loss and Brier
  agree to **≤1e-14** (worst observed 5.2e-15 and 3.2e-15 over eight 20,000-event
  replays, seeds 3000–3007); ROC-AUC to 1.1e-16; precision, recall and accuracy
  are exactly equal.
- **Average precision genuinely changed**, from 0.8227688 to 0.8227457
  (2.8e-5 relative). This is a correctness fix, not noise: `sklearn` evaluates
  one point per *distinct* score threshold, whereas the removed rank-walk
  credited individual ranks inside a group of tied scores, which biased the
  metric upward. sklearn's definition is the standard one.
- **ARFF parsing is exactly identical.** `pandas.read_csv` reproduces the
  removed stdlib `csv` reader on all 45,312 real rows — same order, zero label
  mismatches, maximum feature difference 0.0.
- **Expected calibration error** needed `np.arange(1, bins) / bins` for its bin
  edges rather than `np.linspace`, which renders 0.3 / 0.6 / 0.7 inexactly and
  misfiled probabilities sitting exactly on those boundaries. With exact edges it
  matches the previous value to **≤1e-14** (1.1e-16 – 4.2e-15 over ten
  30,000-observation replays, seeds 2000–2009, half the probabilities placed
  exactly on a bin edge). With `np.linspace` edges the same replays disagree by up
  to 2.0e-3 — twelve orders of magnitude larger, which is what makes this a real
  bug and not a rounding preference.

Every metric in this README and in `artifacts/` was regenerated after the
migration; none is carried over from a prior run. Deleting `artifacts/` and
re-running all three CLI subcommands reproduces the two committed metrics files
byte-for-byte, on both Python versions in the CI matrix (3.11 and 3.13 agree to
the last digit). `scripts/check_documented_metrics.py` asserts that the figures
quoted in this file still match those artifacts, and CI runs it.

## Repository map

```text
src/driftguard/
  datasets.py    verified OpenML download and pandas ARFF adapter
  model.py       NumPy-vectorized online standardization and FTRL-Proximal
  stream.py      documented concept-shift generator
  drift.py       in-repo detectors + river adapter behind one Protocol
  metrics.py     scikit-learn scoring plus calibration and top-k metrics
  experiment.py  delayed-label event loop and leakage audit
  real_experiment.py real-data frozen-versus-adaptive evaluation
  report.py      Evidently feature-drift report
tests/           model, detector, metric, report, and temporal-contract tests
```

## Dataset provenance

The Electricity dataset is OpenML dataset 151, marked public and distributed
as 45,312 half-hour observations from the Australian electricity market. The
adapter uses OpenML's official download URL and published checksum. Source
ordering is preserved. The synthetic benchmark complements it by making exact
drift onset inspectable; it is not presented as real-world fraud evidence.

The download path deliberately stays on the standard library (`urllib` +
`hashlib`): the checksum gate is this project's trust boundary and should not
depend on a third-party HTTP client. `pandas.read_csv` only ever reads the
*already verified* local file, so nothing unverified reaches the parser.

## Research context

The production-project framing follows the end-to-end standard illustrated by
[Made With ML](https://github.com/GokuMohandas/Made-With-ML). FTRL-Proximal is
used because it is a strong, interpretable online baseline for sparse risk and
advertising systems. Drift evaluation is intentionally performed in event time
rather than by randomly splitting rows.
