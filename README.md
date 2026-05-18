# CLAAS Stone Detection ESoC 2026

Prototype pipeline for early stone detection in harvester headers using microphone and machine sensor data.

This repository contains my solution work for the 2026 GC.OS European Summer of Code CLAAS challenge: **Embedded AI for Predictive Sensor Systems in Agriculture 4.0**.

## Problem overview

The challenge focuses on detecting stones entering the header of an agricultural harvester. The provided experiment used microphone recordings and machine sensor channels during harvesting runs. Metallic stones were used as proxy stone events because the metal detector provides a voltage spike that can be used as a reference event signal.

The long-term goal is to investigate whether the microphone signal can detect stone uptake early, potentially before the metal detector response, while keeping false detections low during normal operation.

## Current status

This repository currently implements:

- **Task 1: reading the data**
- **Task 2A: reference voltage event detection**

The data loading pipeline:

- reads CLAAS `.mf4` measurement files using `asammdf`
- extracts the required sensor channels
- converts each run into a time-indexed `pandas.DataFrame`
- normalizes the raw `Status` channel into a derived boolean `HeaderOn` column
- supports optional resampling through the `raster` argument
- includes tests for schema, preprocessing, validation, and MF4 loading

The reference event detection pipeline:

- extracts header-on operating episodes from `HeaderOn`
- extends each episode by a short grace period after shutdown
- searches for high `VoltageSignal` regions inside each extended episode
- groups nearby high-voltage samples into candidate event regions
- stores the peak time, peak voltage, local threshold, and related episode metadata

## Expected data

The dataset is provided separately through the private ESoC CLAAS challenge repository and is not redistributed here.

This repository intentionally does **not** commit the `.mf4` or `.wav` files, in order to respect the confidentiality and redistribution restrictions of the challenge dataset.

By default, scripts expect the `.mf4` files to be available locally in:

    data/

Example local structure:

    data/
    ├── Messung_2025-05-09_08-59-34.mf4
    ├── Messung_2025-05-14_16-02-23.mf4
    ├── Messung_2025-05-20_16-30-26.mf4
    ├── Messung_2025-10-01_09-42-16.mf4
    └── Messung_2025-10-01_17-18-12.mf4

The `data/` directory is ignored by Git, except for `data/.gitkeep`.

## Reviewer quickstart

A reviewer who already has access to the private ESoC CLAAS challenge repository can run this solution without editing the code.

There are two supported options.

### Option 1: Place the data locally in this repository

Copy or place the challenge `.mf4` files locally into this repository's ignored `data/` directory:

    data/
    ├── Messung_2025-05-09_08-59-34.mf4
    ├── Messung_2025-05-14_16-02-23.mf4
    ├── Messung_2025-05-20_16-30-26.mf4
    ├── Messung_2025-10-01_09-42-16.mf4
    └── Messung_2025-10-01_17-18-12.mf4

Then inspect the loaded data:

    python scripts/inspect_data.py --raster 0.001

To inspect candidate voltage reference events:

    python scripts/inspect_events.py --threshold-quantile 0.999

### Option 2: Pass the path to the original challenge data directory

If this repository and the original challenge repository are side by side, for example:

    projects/
    ├── esoc2026-challenge-claas/
    │   └── data/
    └── ESOC-CLAAS-2026/
        ├── scripts/
        ├── src/
        └── README.md

then run:

    python scripts/inspect_data.py --data-dir ../esoc2026-challenge-claas/data --raster 0.001

or:

    python scripts/inspect_events.py --data-dir ../esoc2026-challenge-claas/data --threshold-quantile 0.999

This avoids copying the dataset into this repository.

## Installation

Create and activate a virtual environment, then install the project in editable mode:

    python -m pip install -e ".[dev]"

## Project structure

    ESOC-CLAAS-2026/
    ├── data/                         # Local private data, ignored by Git
    ├── scripts/
    │   ├── inspect_data.py            # Inspect loaded MF4 files
    │   └── inspect_events.py          # Inspect candidate voltage events
    ├── src/
    │   └── claas_stone_detection/
    │       ├── schema.py              # Channel names and derived columns
    │       ├── io.py                  # MF4 loading utilities
    │       ├── preprocessing.py       # Status/HeaderOn normalization
    │       ├── validation.py          # DataFrame validation checks
    │       ├── episodes.py            # Header-on episode extraction
    │       └── events.py              # Voltage reference event detection
    ├── tests/                         # Unit and integration tests
    ├── pyproject.toml                 # Project metadata and dependencies
    └── README.md

## Inspecting the data

Run the inspection script using the default local `data/` directory:

    python scripts/inspect_data.py

For faster inspection, use optional resampling:

    python scripts/inspect_data.py --raster 0.001

You can also pass a custom data directory:

    python scripts/inspect_data.py --data-dir /path/to/challenge/data --raster 0.001

## Inspecting voltage reference events

The script `scripts/inspect_events.py` helps inspect candidate voltage reference events before model development.

Run with the default local `data/` directory:

    python scripts/inspect_events.py

Use a more conservative threshold quantile:

    python scripts/inspect_events.py --threshold-quantile 0.999

Use a custom data directory:

    python scripts/inspect_events.py --data-dir /path/to/challenge/data --threshold-quantile 0.999

The inspection script prints grouped candidate voltage events, including event start time, peak time, end time, peak voltage, and whether the event occurred during or shortly after a header-on episode.

## Loaded DataFrame format

Each MF4 file is loaded as a `pandas.DataFrame` indexed by time in seconds.

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

## Reference event format

Reference voltage events are represented by the `StoneEvent` dataclass.

Each `StoneEvent` stores the measurement run name, the above-threshold event region, the voltage peak time, the peak voltage, the local adaptive threshold, and the associated header-on episode.

The local threshold is stored because event detection uses an adaptive episode-level quantile threshold. This makes it possible to inspect event strength through:

    peak_to_threshold_ratio = peak_voltage / threshold

The event extraction should be treated as a candidate/reference event layer, not as perfect ground truth. Conservative settings can focus on stronger voltage events, while more sensitive settings can include weaker candidate events for further analysis.

## Example usage

Load one MF4 file:

    from claas_stone_detection.io import read_mf4

    df = read_mf4("data/Messung_2025-05-09_08-59-34.mf4")
    print(df.head())

Load all runs:

    from claas_stone_detection.io import read_dataset

    dataset = read_dataset("data", raster=0.001)
    print(dataset.keys())

Use a custom data directory:

    from claas_stone_detection.io import read_dataset

    dataset = read_dataset("../esoc2026-challenge-claas/data", raster=0.001)
    print(dataset.keys())

Extract header-on episodes:

    from claas_stone_detection.episodes import extract_header_on_episodes

    episodes = extract_header_on_episodes(df)
    print(episodes)

Detect voltage reference events:

    from claas_stone_detection.events import detect_voltage_events

    events = detect_voltage_events(df, run_name="example_run")
    print(events)

Detect voltage reference events for all runs:

    from claas_stone_detection.events import detect_voltage_events_in_dataset

    events_by_run = detect_voltage_events_in_dataset(dataset)
    print({run_name: len(events) for run_name, events in events_by_run.items()})

## Running tests

    python -m pytest

The unit tests can run without the private dataset. Integration tests that require real `.mf4` files are skipped automatically if no data files are found in the local `data/` directory.

Run code quality checks:

    ruff check .

## Notes on Task 1 design

The data loading layer intentionally separates:

- raw channel definitions in `schema.py`
- validation checks in `validation.py`
- status normalization in `preprocessing.py`
- MF4 reading utilities in `io.py`


## Notes on Task 2A design

The reference event detection layer intentionally separates header operation from voltage event detection:

- `episodes.py` extracts continuous header-on operating periods from `HeaderOn`
- each episode is extended by a short grace period after shutdown
- `events.py` searches for high `VoltageSignal` regions inside each extended episode
- nearby high-voltage samples are grouped into candidate event regions
- each event stores both the peak voltage and the local threshold used for detection

The grace period is important because strong voltage peaks may occur immediately after the header switches off. This likely reflects shutdown behavior following a metal detector trigger, so the peak should still be associated with the preceding header-on episode.