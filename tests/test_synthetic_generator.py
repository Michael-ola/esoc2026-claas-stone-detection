import numpy as np
import pytest

from claas_stone_detection.core.schema import DEFAULT_SCHEMA
from claas_stone_detection.synthetic_data.generator import (
    SyntheticRunConfig,
    enforce_minimum_event_spacing,
    generate_synthetic_dataset,
    generate_synthetic_run,
    make_preimpact_acoustic_precursor,
)


def test_generate_synthetic_run_has_expected_schema() -> None:
    schema = DEFAULT_SCHEMA
    config = SyntheticRunConfig(
        run_name="test_run",
        duration_s=10.0,
        sample_rate_hz=100.0,
        header_on_start_s=1.0,
        header_on_end_s=9.0,
        event_rate_per_minute=6.0,
        random_seed=7,
    )

    synthetic_run = generate_synthetic_run(config)
    frame = synthetic_run.frame

    expected_columns = {
        schema.sensor,
        schema.vehicle_speed,
        schema.cut_length,
        schema.voltage,
        schema.status,
        schema.header_on,
    }

    assert synthetic_run.run_name == "test_run"
    assert set(frame.columns) == expected_columns
    assert frame.index.name == "time_s"
    assert len(frame) == 1001
    assert frame[schema.header_on].dtype == bool


def test_generate_synthetic_run_is_reproducible_with_seed() -> None:
    config = SyntheticRunConfig(
        duration_s=8.0,
        sample_rate_hz=100.0,
        header_on_start_s=1.0,
        header_on_end_s=7.0,
        event_rate_per_minute=8.0,
        random_seed=11,
    )

    run_a = generate_synthetic_run(config)
    run_b = generate_synthetic_run(config)

    assert np.allclose(
        run_a.frame["Sensor1"].to_numpy(),
        run_b.frame["Sensor1"].to_numpy(),
    )
    assert np.allclose(
        run_a.frame["VoltageSignal"].to_numpy(),
        run_b.frame["VoltageSignal"].to_numpy(),
    )
    assert run_a.events == run_b.events


def test_generate_synthetic_run_injects_events_when_rate_is_high() -> None:
    config = SyntheticRunConfig(
        duration_s=30.0,
        sample_rate_hz=200.0,
        header_on_start_s=1.0,
        header_on_end_s=29.0,
        event_rate_per_minute=20.0,
        random_seed=21,
    )

    synthetic_run = generate_synthetic_run(config)

    assert len(synthetic_run.events) > 0
    assert synthetic_run.frame["VoltageSignal"].max() > 100.0


def test_generate_synthetic_dataset_returns_requested_number_of_runs() -> None:
    dataset = generate_synthetic_dataset(
        n_runs=4,
        base_seed=100,
        duration_s=5.0,
        sample_rate_hz=50.0,
    )

    assert len(dataset) == 4
    assert sorted(dataset.keys()) == [
        "synthetic_run_000",
        "synthetic_run_001",
        "synthetic_run_002",
        "synthetic_run_003",
    ]


def test_enforce_minimum_event_spacing_removes_close_events() -> None:
    result = enforce_minimum_event_spacing(
        event_times=[1.0, 2.0, 4.5, 8.0],
        min_spacing_s=3.0,
    )

    assert result == [1.0, 4.5, 8.0]


def test_generate_synthetic_dataset_rejects_invalid_run_count() -> None:
    with pytest.raises(ValueError, match="n_runs must be positive"):
        generate_synthetic_dataset(n_runs=0)


def test_generate_synthetic_run_rejects_invalid_config() -> None:
    config = SyntheticRunConfig(
        duration_s=10.0,
        sample_rate_hz=100.0,
        header_on_start_s=9.0,
        header_on_end_s=1.0,
    )

    with pytest.raises(ValueError, match="Header-on interval"):
        generate_synthetic_run(config)

def test_make_preimpact_acoustic_precursor_is_before_event() -> None:
    rng = np.random.default_rng(123)
    time = np.arange(0.0, 3.0, 0.001)

    precursor = make_preimpact_acoustic_precursor(
        time=time,
        event_time=2.0,
        amplitude=1.0,
        duration_s=0.5,
        rng=rng,
    )

    assert np.max(np.abs(precursor[time < 1.5])) == pytest.approx(0.0)
    assert np.max(np.abs(precursor[(time >= 1.5) & (time <= 2.0)])) > 0.0
    assert np.max(np.abs(precursor[time > 2.0])) == pytest.approx(0.0)


def test_make_preimpact_acoustic_precursor_rejects_invalid_duration() -> None:
    rng = np.random.default_rng(123)
    time = np.arange(0.0, 1.0, 0.001)

    with pytest.raises(ValueError, match="duration_s must be positive"):
        make_preimpact_acoustic_precursor(
            time=time,
            event_time=0.5,
            amplitude=1.0,
            duration_s=0.0,
            rng=rng,
        )

