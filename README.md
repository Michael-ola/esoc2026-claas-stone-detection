# CLAAS Stone Detection ESoC 2026

Prototype pipeline for early stone detection in harvester headers using microphone and machine sensor data.

This repository contains solution work for the 2026 GC.OS European Summer of Code CLAAS challenge: **Embedded AI for Predictive Sensor Systems in Agriculture 4.0**.

The project is organized around reusable engineering components rather than task-specific folders. Task 1 builds the shared data-loading interface. Task 2 reuses that interface for modeling and evaluation. Bonus 1 generates additional data in the same format. Bonus 2 reuses the Task 2 pipeline and exports a constrained Random Forest as a microcontroller artefact.

---

## Problem overview

The challenge focuses on detecting stones entering the header of an agricultural harvester. The provided experiment used microphone recordings and machine sensor channels during harvesting runs. Metallic stones were used as proxy stone events because the metal detector provides a voltage spike that can be used as a reference event signal.

The long-term goal is to investigate whether the microphone signal, together with operating context such as vehicle speed and cut length, can detect stone uptake early while keeping false detections low during normal operation.

---

## Current status

Implemented:

- **Task 1:** MF4 data loading into reusable `pandas.DataFrame` structures.
- **Task 2:** Random Forest early-detection baseline with grouped run-level validation.
- **Bonus 1:** Synthetic CLAAS-like data generation and Random Forest evaluation on 100 generated runs.
- **Bonus 2:** Microcontroller-oriented constrained Random Forest export artefacts.
- **2D CNN:** In progress. Spectrogram utilities and a compact CNN model scaffold exist, but the CNN training/evaluation workflow is not finalized.

---

## Expected data

The real challenge dataset is provided separately through the private ESoC CLAAS challenge repository and is not redistributed here.

This repository intentionally does **not** commit `.mf4`, `.wav`, or generated synthetic CSV datasets.

By default, real-data scripts expect the challenge `.mf4` files locally in:

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

The `data/` directory is ignored by Git except for `data/.gitkeep`.

Generated synthetic datasets are also ignored by Git:

```text
/synthetic_data/
/synthetic_data_test/
```

The source-code package `src/claas_stone_detection/synthetic_data/` is **not** ignored.

---

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

The project was developed with Python 3.11.

Run checks:

```bash
python -m pytest
ruff check .
```

---

## Project structure

```text
ESoC-CLAAS-2026/
├── artifacts/
│   └── mcu_random_forest/
│       ├── README.md
│       ├── claas_stone_rf_mcu.h
│       ├── claas_stone_rf_mcu.json
│       └── feature_order.txt
├── data/
│   └── .gitkeep
├── scripts/
│   ├── audit_reference_events.py
│   ├── export_mcu_random_forest.py
│   ├── generate_synthetic_dataset.py
│   ├── inspect_data.py
│   ├── inspect_events.py
│   ├── run_baseline.py
│   └── run_synthetic_baseline.py
├── src/
│   └── claas_stone_detection/
│       ├── core/
│       │   ├── preprocessing.py
│       │   ├── schema.py
│       │   └── validation.py
│       ├── data/
│       │   ├── io.py
│       │   └── synthetic_io.py
│       ├── edge/
│       │   └── random_forest_export.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── models/
│       │   ├── baseline.py
│       │   └── cnn2d.py
│       ├── pipelines/
│       │   └── baseline_pipeline.py
│       ├── reference/
│       │   ├── episodes.py
│       │   ├── events.py
│       │   └── labels.py
│       ├── spectrograms/
│       │   └── dataset.py
│       ├── streaming/
│       │   ├── features.py
│       │   └── windowing.py
│       └── synthetic_data/
│           └── generator.py
├── tests/
│   ├── test_baseline.py
│   ├── test_baseline_pipeline.py
│   ├── test_cnn2d.py
│   ├── test_episodes.py
│   ├── test_events.py
│   ├── test_features.py
│   ├── test_io.py
│   ├── test_labels.py
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   ├── test_random_forest_export.py
│   ├── test_schema.py
│   ├── test_spectrogram_dataset.py
│   ├── test_synthetic_generator.py
│   ├── test_synthetic_io.py
│   ├── test_validation.py
│   └── test_windowing.py
├── pyproject.toml
└── README.md
```

---

## Design principle: reuse instead of rebuilding

The repository intentionally separates concerns by responsibility, not by task number.

```text
Task 1
→ core/
→ data/io.py

Task 2
→ reference/
→ streaming/
→ models/baseline.py
→ evaluation/
→ pipelines/baseline_pipeline.py

Bonus 1
→ synthetic_data/generator.py
→ data/synthetic_io.py
→ the same Task 2 pipeline

Bonus 2
→ edge/random_forest_export.py
→ the same Task 2 feature/model pipeline
```

This prevents each challenge section from becoming a separate mini-project. Bonus 1 and Bonus 2 reuse the same data format, windowing, labeling, feature extraction, model training, and evaluation logic developed for Task 2.

---

## Loaded DataFrame format

Each real or synthetic run is represented as a time-indexed `pandas.DataFrame`.

Expected columns:

| Column | Description |
|---|---|
| `Sensor1` | Microphone/audio signal from the header |
| `VehicleSpeed` | Harvester speed |
| `CutLength` | Header cut length setting |
| `VoltageSignal` | Metal detector voltage signal or synthetic reference voltage |
| `Status` | Header status channel |
| `HeaderOn` | Boolean header-on state derived from `Status` or generated directly |

The common interface is:

```python
dict[str, pandas.DataFrame]
```

where each key is a run name and each value is one loaded run.

This interface is used by both real-data and synthetic-data pipelines.

---

## Task 1: data loading

The Task 1 loader reads CLAAS `.mf4` measurement files using `asammdf`.

Implemented:

- MF4 loading with configurable raster sampling.
- Required channel extraction.
- Time-indexed DataFrame output.
- Status normalization into `HeaderOn`.
- Schema and validation utilities.
- Tests for schema, preprocessing, validation, and MF4 loading.

Example:

```python
from claas_stone_detection.data.io import read_dataset

dataset = read_dataset("data", raster=0.001)
```

Inspect loaded data:

```bash
python scripts/inspect_data.py --raster 0.001
```

Use a custom data directory:

```bash
python scripts/inspect_data.py --data-dir /path/to/challenge/data --raster 0.001
```

---

## Task 2: Random Forest early-detection baseline

Task 2 implements a complete baseline for live-style early stone detection.

Implemented:

- Header-on episode extraction.
- Reference voltage event detection.
- Reference event deduplication.
- Early-detection window labeling.
- Sliding-window feature extraction.
- Temporal delta features.
- Random Forest baseline model.
- Grouped cross-validation by measurement run.
- Threshold sweep evaluation.
- Consensus alarm filtering.
- False-alarm interval reporting.
- Reference event audit tooling.

Run the default real-data baseline:

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

The script uses grouped cross-validation by measurement run. With five real runs, each fold trains on four complete runs and validates on one held-out run. This avoids leakage from overlapping sliding windows.

### Current real-data baseline result

Using the default Task 2 configuration:

```text
Reference events: 28
Detected events: 12/28
True positive rate: 0.429
False detections/hour: 307.441
Mean seconds between false detections: 11.710 s
Average advance time: 0.697 s
```

This is not deployment-ready. The seconds-per-false-detection value makes the limitation clear: one false alarm roughly every 11.7 seconds is operationally unacceptable. However, the result shows that the pipeline can detect some reference events early and provides a transparent baseline for improvement.

---

## Reference event detection

Reference events in the real data are inferred from the metal-detector `VoltageSignal`.

The pipeline:

1. Extracts continuous `HeaderOn` episodes.
2. Extends each episode by a short grace period after shutdown.
3. Searches for high-voltage regions inside the extended episode.
4. Groups nearby high-voltage samples into candidate events.
5. Deduplicates repeated detections of the same voltage peak.
6. Stores event metadata in the shared `StoneEvent` dataclass.

Audit reference events:

```bash
python scripts/audit_reference_events.py \
  --raster 0.001 \
  --event-threshold-quantile 0.999 \
  --min-ratio 1.10
```

The reference layer is treated as a proxy label source, not perfect physical ground truth. A model alarm without a matching voltage reference is counted as a false detection relative to the available reference signal, but some unmatched acoustic detections could still be physically meaningful.

---

## Windowing and labels

The baseline uses live-style sliding windows.

Default:

```text
window_s = 0.5
hop_s = 0.1
positive_horizon_s = 1.0
post_event_exclusion_s = 1.0
```

Labels:

| Label | Meaning |
|---:|---|
| `1` | Positive early-warning window |
| `0` | Normal negative window |
| `-1` | Ignored ambiguous window |

Positive labels are assigned to windows that occur before a reference event and within the early-warning horizon. Windows immediately after a reference event are ignored to avoid contamination from impact or shutdown artifacts.

---

## Feature extraction

The Random Forest baseline uses engineered features from each sliding window.

Feature groups include:

- Time-domain audio statistics.
- RMS and peak amplitude.
- Crest factor.
- Zero-crossing rate.
- Frequency-band energy.
- Spectral centroid.
- Spectral bandwidth.
- High-frequency energy ratio.
- Vehicle speed statistics.
- Cut length statistics.
- Run-local temporal delta features.

`VehicleSpeed` and `CutLength` are retained as operating-context features. They can help describe the acoustic background and may also correlate with stone-ingestion risk or event severity.

---

## Evaluation metrics

The baseline reports:

- True positive rate.
- Number of detected reference events.
- False detections per hour.
- Mean seconds between false detections.
- Average advance warning time.

The seconds-per-false-detection metric is reported because false detections per hour can sound abstract. For example:

```text
307 false detections/hour ≈ one false detection every 11.7 seconds
```

---

## Bonus 1: synthetic data generation

Bonus 1 generates CLAAS-like synthetic time-series runs in the same DataFrame format used by Task 1 and Task 2.

The generator creates:

- `Sensor1`
- `VehicleSpeed`
- `CutLength`
- `VoltageSignal`
- `Status`
- `HeaderOn`

Synthetic runs include:

- Header-on operating periods.
- Smooth vehicle-speed variation.
- Smooth cut-length variation.
- Background harvesting audio.
- Context-dependent stone-event probability.
- Pre-impact acoustic precursor.
- Main stone-like impact transient.
- Reference voltage spike after the acoustic event.

Generate a small smoke-test dataset:

```bash
python scripts/generate_synthetic_dataset.py \
  --output-dir synthetic_data_test \
  --n-runs 3 \
  --duration-s 20 \
  --sample-rate-hz 1000 \
  --event-rate-per-minute 30
```

Generate the main 100-run synthetic dataset:

```bash
python scripts/generate_synthetic_dataset.py \
  --output-dir synthetic_data \
  --n-runs 100 \
  --duration-s 120 \
  --sample-rate-hz 1000 \
  --event-rate-per-minute 2
```

The generated files are ignored by Git.

---

## Bonus 1: Random Forest on synthetic data

The synthetic Random Forest baseline reuses the same Task 2 pipeline. It only swaps the data source and reference-event source:

```text
real data:
MF4 files → read_dataset() → voltage-derived StoneEvent references

synthetic data:
CSV files → read_synthetic_dataset() → metadata-derived StoneEvent references
```

Run:

```bash
python scripts/run_synthetic_baseline.py --synthetic-dir synthetic_data
```

Current 100-run synthetic result after adding pre-impact acoustic precursors:

```text
Runs: 100
Windows: 108600
Reference events: 384

threshold=0.50
Detected events: 379/384
True positive rate: 0.987
False detections/hour: 9.578
Mean seconds between false detections: 375.866 s
Average advance time: 0.218 s
```

This demonstrates that the Task 2 pipeline scales to 100 generated runs and can learn a consistent injected pre-impact acoustic precursor. It should not be interpreted as proof that the real-world problem is solved, because the synthetic labels and event structure are controlled.

---

## 2D CNN status

2D CNN work is **in progress**.

Implemented so far:

- Reusable NumPy log-spectrogram extraction.
- Spectrogram window data structures.
- Compact `TinySpectrogramCNN` model scaffold.
- Tests that skip CNN execution when PyTorch is not installed.

Not finalized yet:

- CNN training script.
- CNN evaluation on synthetic spectrogram windows.
- Export or deployment path for the CNN.

The CNN path is intentionally separated from the Random Forest pipeline so that the project can keep a working, tested baseline while the neural model remains experimental.

---

## Bonus 2: microcontroller deployment artefacts

Bonus 2 assumes a 2 MB RAM automotive microcontroller and inference-only deployment.

The selected deployment model is a **constrained Random Forest**, not the CNN.

Reason:

- It reuses the existing Task 2 feature pipeline.
- It avoids Python, PyTorch, TensorFlow, or dynamic allocation at inference time.
- It compiles into plain C arrays and simple if/else tree traversal.
- It is easier to audit than a rushed neural-network deployment.
- It can be constrained with a small number of trees and limited tree depth.

Export MCU artefacts:

```bash
python scripts/export_mcu_random_forest.py \
  --synthetic-dir synthetic_data \
  --output-dir artifacts/mcu_random_forest \
  --max-runs 100 \
  --n-estimators 8 \
  --max-depth 6 \
  --min-samples-leaf 4
```

Generated artefacts:

```text
artifacts/mcu_random_forest/
├── README.md
├── claas_stone_rf_mcu.h
├── claas_stone_rf_mcu.json
└── feature_order.txt
```

Current export summary:

```text
Feature count: 36
Tree count: 8
Estimated model bytes: 11536
RAM limit: 2097152 bytes
2 MB fit check: PASS
```

The C header contains:

- Tree node arrays.
- Feature threshold comparisons.
- `claas_rf_predict_proba()`.
- `claas_rf_predict()`.

The JSON file contains the same exported model in a readable format. `feature_order.txt` records the exact feature vector order expected by firmware.

---

## Important limitations

The current project should be interpreted as an engineering prototype, not a finished production detector.

Known limitations:

- The real dataset contains only five measurement runs.
- Real reference labels are inferred from voltage events, not manual stone annotations.
- Some voltage reference events are weak or borderline.
- Acoustic conditions vary strongly across real runs.
- The Random Forest uses compact engineered features rather than full raw temporal structure.
- False-alarm rates on real data remain too high for deployment.
- Synthetic-data performance is much stronger because the event structure is controlled.
- CNN training/evaluation is still in progress.
- The Bonus 2 memory estimate covers exported model arrays, not the full firmware stack.

---

## Common commands

Run all tests:

```bash
python -m pytest
```

Run linting:

```bash
ruff check .
```

Run real-data baseline:

```bash
python scripts/run_baseline.py --raster 0.001 --window-s 0.5 --hop-s 0.1
```

Generate synthetic data:

```bash
python scripts/generate_synthetic_dataset.py \
  --output-dir synthetic_data \
  --n-runs 100 \
  --duration-s 120 \
  --sample-rate-hz 1000 \
  --event-rate-per-minute 2
```

Run synthetic Random Forest baseline:

```bash
python scripts/run_synthetic_baseline.py --synthetic-dir synthetic_data
```

Export Bonus 2 MCU artefacts:

```bash
python scripts/export_mcu_random_forest.py \
  --synthetic-dir synthetic_data \
  --output-dir artifacts/mcu_random_forest \
  --max-runs 100 \
  --n-estimators 8 \
  --max-depth 6 \
  --min-samples-leaf 4
```

---

## Reproducibility notes

The repository includes tests for:

- MF4 loading and validation.
- Header-on episode extraction.
- Voltage reference event detection.
- Reference event deduplication.
- Window labeling.
- Feature extraction.
- Random Forest baseline training.
- Evaluation metrics.
- Synthetic data generation and loading.
- Shared baseline pipeline reuse.
- Spectrogram utilities.
- MCU Random Forest export.

Run the full verification suite:

```bash
python -m pytest
ruff check .
```
