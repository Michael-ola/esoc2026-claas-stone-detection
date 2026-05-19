# CLAAS Stone Detection ESoC 2026

Prototype pipeline for early stone detection in harvester headers using microphone and machine sensor data.

This repository contains solution work for the 2026 GC.OS European Summer of Code CLAAS challenge: **Embedded AI for Predictive Sensor Systems in Agriculture 4.0**.

## Problem overview

The challenge focuses on detecting stones entering the header of an agricultural harvester. The provided experiment used microphone recordings and machine sensor channels during harvesting runs. Metallic stones were used as proxy stone events because the metal detector provides a voltage spike that can be used as a reference event signal.

The long-term objective is to investigate whether the microphone signal can detect stone uptake early, potentially before the metal detector response, while keeping false detections low during normal operation.

## Current status

This repository currently implements:

- **Task 1: data loading**
- **Task 2: baseline early stone-detection model**

The current Task 2 pipeline is a complete, testable baseline rather than a final deployable model. It includes reference event extraction, live-style windowing, feature extraction, early-warning labels, a Random Forest classifier, grouped cross-validation, threshold sweeps, and false-alarm analysis.

## Expected data

The dataset is provided separately through the private ESoC CLAAS challenge repository and is not redistributed here.

This repository intentionally does **not** commit the `.mf4` or `.wav` files, in order to respect the confidentiality and redistribution restrictions of the challenge dataset.

By default, scripts expect the `.mf4` files to be available locally in:

```text
data/
```

Example local structure:

```text
data/
├── Messung_2025-05-09_08-59-34.mf4
├── Messung_2025-05-14_16-02-23.mf4
├── Messung_2025-05-20_16-30-26.mf4
├── Messung_2025-10-01_09-42-16.mf4
└── Messung_2025-10-01_17-18-12.mf4
```

The `data/` directory is ignored by Git, except for `data/.gitkeep`.

## Reviewer quickstart

A reviewer who already has access to the private ESoC CLAAS challenge repository can run this solution without editing the code.

### Option 1: Place the data locally in this repository

Copy or place the challenge `.mf4` files into this repository's ignored `data/` directory:

```text
data/
├── Messung_2025-05-09_08-59-34.mf4
├── Messung_2025-05-14_16-02-23.mf4
├── Messung_2025-05-20_16-30-26.mf4
├── Messung_2025-10-01_09-42-16.mf4
└── Messung_2025-10-01_17-18-12.mf4
```

Then run:

```bash
python scripts/inspect_data.py --raster 0.001
python scripts/audit_reference_events.py --raster 0.001
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1
```

### Option 2: Pass the original challenge data directory

If this repository and the original challenge repository are side by side, for example:

```text
projects/
├── esoc2026-challenge-claas/
│   └── data/
└── ESOC-CLAAS-2026/
    ├── scripts/
    ├── src/
    └── README.md
```

then run:

```bash
python scripts/inspect_data.py --data-dir ../esoc2026-challenge-claas/data --raster 0.001
python scripts/audit_reference_events.py --data-dir ../esoc2026-challenge-claas/data --raster 0.001
python scripts/run_baseline.py --data-dir ../esoc2026-challenge-claas/data --raster 0.001 --window-s 0.5 --hop-s 0.1
```

This avoids copying the dataset into this repository.

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

The project was developed and tested with Python 3.11.

## Project structure

```text
ESOC-CLAAS-2026/
├── data/                         # Local private challenge data, ignored by Git
├── scripts/
│   ├── audit_reference_events.py  # Audits reference voltage events
│   ├── inspect_data.py            # Inspects loaded MF4 files
│   ├── inspect_events.py          # Inspects candidate voltage events
│   └── run_baseline.py            # Runs the Task 2 Random Forest baseline
├── src/
│   └── claas_stone_detection/
│       ├── core/
│       │   ├── preprocessing.py   # Status/HeaderOn normalization
│       │   ├── schema.py          # Channel names and derived columns
│       │   └── validation.py      # DataFrame validation checks
│       ├── data/
│       │   └── io.py              # MF4 reading and dataset loading
│       ├── evaluation/
│       │   └── metrics.py         # Detection matching and evaluation metrics
│       ├── models/
│       │   └── baseline.py        # Random Forest baseline model
│       ├── reference/
│       │   ├── episodes.py        # Header-on episode extraction
│       │   ├── events.py          # Reference voltage event detection
│       │   └── labels.py          # Early-detection window labeling
│       └── streaming/
│           ├── features.py        # Window-level and temporal features
│           └── windowing.py       # Live-style sliding windows
├── tests/                         # Unit and integration tests
├── pyproject.toml                 # Project metadata and dependencies
└── README.md
```

## Running tests and linting

Run all tests:

```bash
python -m pytest
```

Run code quality checks:

```bash
ruff check .
```

The unit tests can run without the private dataset. Integration tests that require real `.mf4` files are skipped automatically if no data files are found in the local `data/` directory.

## Task 1: Data loading

The data loading layer reads CLAAS `.mf4` measurement files using `asammdf` and returns time-indexed `pandas.DataFrame` objects.

Example:

```python
from claas_stone_detection.data.io import read_dataset

dataset = read_dataset("data", raster=0.001)
print(dataset.keys())
```

The returned object is a dictionary:

```text
{
    "run_name": pandas.DataFrame,
    ...
}
```

Each DataFrame is indexed by time in seconds.

### Loaded DataFrame format

The raw channels are:

| Column | Description |
|---|---|
| `Sensor1` | Microphone/audio signal from the header |
| `VehicleSpeed` | Harvester speed |
| `CutLength` | Header cut length setting |
| `VoltageSignal` | Metal detector voltage signal |
| `Status` | Raw header status channel |

The loader also adds:

| Column | Description |
|---|---|
| `HeaderOn` | Boolean column derived from `Status` |

The raw `Status` channel is preserved. `HeaderOn` is added because the actual MF4 files expose status values such as `b"On"` and `b"Off"` rather than only numeric `1` and `0`.

### Inspecting the data

Run the inspection script using the default local `data/` directory:

```bash
python scripts/inspect_data.py
```

For faster inspection, use optional resampling:

```bash
python scripts/inspect_data.py --raster 0.001
```

Use a custom data directory:

```bash
python scripts/inspect_data.py --data-dir /path/to/challenge/data --raster 0.001
```

## Task 2: Early stone-detection baseline

The Task 2 pipeline builds a live-style early-warning model from the microphone and machine sensor signals.

The baseline flow is:

```text
MF4 files
→ loaded DataFrames
→ HeaderOn episodes
→ deduplicated reference voltage events
→ live-style sliding windows
→ window-level features
→ temporal delta features
→ early-warning labels
→ Random Forest baseline
→ GroupKFold evaluation by run
→ threshold sweep and false-alarm analysis
```

## Reference event detection

Reference events are inferred from the metal detector `VoltageSignal`.

The pipeline:

1. Extracts continuous `HeaderOn` operating episodes.
2. Extends each episode by a short grace period after shutdown.
3. Searches for high-voltage regions inside each extended episode.
4. Uses an episode-local quantile threshold.
5. Groups nearby high-voltage samples into candidate event regions.
6. Stores the voltage peak time, peak voltage, local threshold, and associated episode metadata.
7. Deduplicates repeated detections of the same physical voltage peak.

Deduplication is important because the same voltage peak can be detected through overlapping or extended episode regions. After deduplication, the current reference set contains 28 reference voltage events.

Audit reference events with:

```bash
python scripts/audit_reference_events.py --raster 0.001 --event-threshold-quantile 0.999 --min-ratio 1.10
```

The audit reports:

- Number of header-on episodes.
- Number of reference voltage events.
- Peak voltage.
- Local detection threshold.
- Peak-to-threshold ratio.
- Whether the event peak occurred inside the `HeaderOn` episode.
- Weakest reference events.

## Reference event format

Reference voltage events are represented by the `StoneEvent` dataclass.

Each `StoneEvent` stores:

- Measurement run name.
- Above-threshold event start time.
- Voltage peak time.
- Above-threshold event end time.
- Peak voltage.
- Local adaptive threshold.
- Associated header-on episode start and end time.
- Event source channel.

Event strength can be inspected using:

```text
peak_to_threshold_ratio = peak_voltage / threshold
```

The event extraction should be treated as a candidate/reference event layer, not as perfect ground truth. Conservative settings can focus on stronger voltage events, while more sensitive settings can include weaker candidate events for analysis.

## Live-style windowing

The baseline creates streaming-style windows over the signal instead of randomly sampling individual rows.

Default window settings:

```text
window_s = 0.5
hop_s = 0.1
window_region = header-on
```

By default, model input windows are created only during active `HeaderOn` operation. This avoids training and evaluating the model on post-header shutdown or grace-period audio.

The extended region can still be tested experimentally:

```bash
python scripts/run_baseline.py --window-region extended
```

Each window stores:

- Run name.
- Window start time.
- Window end time.
- Detection time.
- Integer start index.
- Integer end index.

The integer indexes are used for fast `iloc` slicing on large high-frequency time-series data.

## Feature extraction

The baseline extracts compact features from each live-style window.

Feature groups include:

- Time-domain audio statistics.
- RMS and peak amplitude.
- Crest factor.
- Zero-crossing rate.
- Frequency-domain band energies.
- Spectral centroid.
- Spectral bandwidth.
- High-frequency energy ratio.
- Vehicle speed statistics.
- Cut length statistics.
- Temporal delta features between consecutive windows.

Temporal delta features are computed within each measurement run only, so information does not leak across runs.

## Window labeling

The model is trained for early detection.

Labels are assigned as:

```text
1   positive early-warning window
0   normal negative window
-1  ignored ambiguous window
```

Default label settings:

```text
positive_horizon_s = 1.0
post_event_exclusion_s = 1.0
```

A window is positive if its detection time is before a reference event and within the positive horizon. Windows immediately after reference events are ignored to avoid shutdown or post-impact contamination.

## Baseline model

The current model is a Random Forest classifier trained on engineered sliding-window features.

Run the default baseline:

```bash
python scripts/run_baseline.py
```

Recommended explicit command:

```bash
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1
```

Current default baseline configuration:

```text
window_region = header-on
positive_horizon_s = 1.0
min_event_ratio = 1.0
score_threshold = 0.10
consensus = 1 of 1
```

The script performs grouped cross-validation by measurement run using `GroupKFold`. This prevents leakage from overlapping windows in the same run.

## Evaluation metrics

The baseline reports:

- True positive rate.
- Number of detected reference events.
- False detections per hour.
- Mean seconds between false detections.
- Average advance warning time.

The seconds-per-false-alarm metric is included because false detections per hour can hide the practical severity of the problem.

For example:

```text
307 false detections/hour ≈ one false alarm every 11.7 seconds
```

## Current baseline result

Using the current default configuration:

```bash
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1
```

the overall grouped cross-validation result at threshold `0.10` is:

```text
Reference events: 28
Detected events: 12/28
True positive rate: 0.429
False detections/hour: 307.441
Mean seconds between false detections: 11.710 s
Average advance time: 0.697 s
```

This result is not deployment-ready. It shows that the pipeline can detect some reference events early, but the false-alarm rate is still too high for real agricultural operation.

Operationally, one false alarm every 11.7 seconds would interrupt the harvester too frequently. This motivates stronger feature engineering, synthetic data generation, and more robust temporal models.

## Threshold sweep

The baseline runner prints a threshold sweep by default:

```text
0.05, 0.10, 0.15, 0.20, 0.30, 0.50
```

This exposes the trade-off between sensitivity and false alarms.

Lower thresholds detect more reference events but create many false alarms. Higher thresholds reduce false alarms but miss most reference events.

## Consensus alarm filtering

The evaluator supports k-of-n consensus alarm filtering. For example:

```bash
python scripts/run_baseline.py --consensus-k 2 --consensus-n 3
```

This means an alarm is emitted only if at least 2 of the last 3 windows exceed the score threshold.

Consensus filtering reduces isolated false alarms, but it can also reduce early detection sensitivity.

## Optional strong-reference filtering

Reference events can be filtered by peak-to-threshold ratio:

```bash
python scripts/run_baseline.py --min-event-ratio 1.10
```

This is useful for experiments with stronger reference events only. It is not enabled by default because the full deduplicated reference set is retained for the baseline.

## Example usage

Load one MF4 file:

```python
from claas_stone_detection.data.io import read_mf4

df = read_mf4("data/Messung_2025-05-09_08-59-34.mf4")
print(df.head())
```

Load all runs:

```python
from claas_stone_detection.data.io import read_dataset

dataset = read_dataset("data", raster=0.001)
print(dataset.keys())
```

Extract header-on episodes:

```python
from claas_stone_detection.reference.episodes import extract_header_on_episodes

episodes = extract_header_on_episodes(df)
print(episodes)
```

Detect voltage reference events:

```python
from claas_stone_detection.reference.events import detect_voltage_events

events = detect_voltage_events(df, run_name="example_run")
print(events)
```

Detect voltage reference events for all runs:

```python
from claas_stone_detection.reference.events import detect_voltage_events_in_dataset

events_by_run = detect_voltage_events_in_dataset(dataset)
print({run_name: len(events) for run_name, events in events_by_run.items()})
```

Run the baseline:

```bash
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1
```

Run the baseline with conservative alarm consensus:

```bash
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1 --consensus-k 2 --consensus-n 3
```

## Design notes

### Task 1 design

The data loading layer separates:

- Channel definitions in `core/schema.py`.
- Validation checks in `core/validation.py`.
- Status normalization in `core/preprocessing.py`.
- MF4 reading utilities in `data/io.py`.

This keeps data access independent from model development.

### Task 2 design

The Task 2 baseline separates:

- Episode extraction in `reference/episodes.py`.
- Reference event detection in `reference/events.py`.
- Window labeling in `reference/labels.py`.
- Streaming window generation in `streaming/windowing.py`.
- Feature extraction in `streaming/features.py`.
- Baseline model training in `models/baseline.py`.
- Evaluation metrics in `evaluation/metrics.py`.

The model does not directly know about MF4 files or voltage event detection. It consumes labeled feature tables, which keeps the pipeline modular and testable.

## Important limitations

The current baseline is intentionally simple and should be interpreted as a first engineering baseline, not a final deployable model.

Known limitations:

- The dataset contains only five real measurement runs.
- Reference labels are inferred from voltage events rather than manually annotated stone impacts.
- Some reference events are weak or borderline.
- Acoustic conditions vary strongly across runs.
- The Random Forest sees compact window features rather than raw temporal structure.
- False-alarm rates remain too high for deployment.

## Reproducibility notes

The baseline uses:

- Deterministic Random Forest random seed.
- Grouped cross-validation by run name.
- Explicit raster sampling.
- Unit tests for data loading, event detection, labeling, feature extraction, model training, and evaluation.

Run all checks with:

```bash
python -m pytest
ruff check .
```