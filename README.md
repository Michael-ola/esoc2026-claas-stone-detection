# CLAAS Stone Detection ESoC 2026

Prototype pipeline for early stone detection in harvester headers using microphone and machine sensor data.

This repository contains my solution work for the 2026 GC.OS European Summer of Code CLAAS challenge: **Embedded AI for Predictive Sensor Systems in Agriculture 4.0**.

## Problem overview

The challenge focuses on detecting stones entering the header of an agricultural harvester. The provided experiment used microphone recordings and machine sensor channels during harvesting runs. Metallic stones were used as proxy stone events because the metal detector provides a voltage spike that can be used as a reference event signal.

The long-term goal is to investigate whether the microphone signal can detect stone uptake early, potentially before the metal detector response, while keeping false detections low during normal operation.

## Current status

This repository currently implements **Task 1: reading the data**.

The implemented data loading pipeline:

- reads CLAAS `.mf4` measurement files using `asammdf`
- extracts the required sensor channels
- converts each run into a time-indexed `pandas.DataFrame`
- normalizes the raw `Status` channel into a derived boolean `HeaderOn` column
- supports optional resampling through the `raster` argument
- includes tests for schema, preprocessing, validation, and MF4 loading

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

Then run:

    python scripts/inspect_data.py --raster 0.001

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

This avoids copying the dataset into this repository.

## Installation

Create and activate a virtual environment, then install the project in editable mode:

    python -m pip install -e ".[dev]"

## Inspecting the data

Run the inspection script using the default local `data/` directory:

    python scripts/inspect_data.py

For faster inspection, use optional resampling:

    python scripts/inspect_data.py --raster 0.001

You can also pass a custom data directory:

    python scripts/inspect_data.py --data-dir /path/to/challenge/data --raster 0.001

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

## Example usage

    from claas_stone_detection.io import read_dataset, read_mf4

    df = read_mf4("data/Messung_2025-05-09_08-59-34.mf4")
    print(df.head())

    dataset = read_dataset("data", raster=0.001)
    print(dataset.keys())

A custom data directory can also be used:

    from claas_stone_detection.io import read_dataset

    dataset = read_dataset("../esoc2026-challenge-claas/data", raster=0.001)
    print(dataset.keys())

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

This keeps the code modular and testable before moving to episode extraction, event labeling, and model development.