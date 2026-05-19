from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema


@dataclass(frozen=True)
class SyntheticStoneEvent:
    """Ground-truth metadata for an injected synthetic stone event."""

    run_name: str
    event_time: float
    amplitude: float
    width_s: float
    vehicle_speed: float
    cut_length: float


@dataclass(frozen=True)
class SyntheticRun:
    """Synthetic CLAAS-like run and its injected event metadata."""

    run_name: str
    frame: pd.DataFrame
    events: list[SyntheticStoneEvent]


@dataclass(frozen=True)
class SyntheticRunConfig:
    """Configuration for one synthetic harvester run."""

    run_name: str = "synthetic_run"
    duration_s: float = 120.0
    sample_rate_hz: float = 1000.0
    header_on_start_s: float = 5.0
    header_on_end_s: float = 115.0
    base_vehicle_speed: float = 3.0
    base_cut_length: float = 12.0
    noise_std: float = 0.15
    event_rate_per_minute: float = 2.0
    random_seed: int | None = 42


def generate_synthetic_run(
    config: SyntheticRunConfig,
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> SyntheticRun:
    """Generate one CLAAS-like synthetic measurement run.

    The generated DataFrame follows the same column convention as the real MF4
    loading pipeline: Sensor1, VehicleSpeed, CutLength, VoltageSignal, Status,
    and HeaderOn. Synthetic stone-like events are injected as short broadband
    acoustic transients with matching voltage spikes.
    """
    validate_config(config)

    rng = np.random.default_rng(config.random_seed)
    time = make_time_index(
        duration_s=config.duration_s,
        sample_rate_hz=config.sample_rate_hz,
    )

    header_on = (
        (time >= config.header_on_start_s) & (time <= config.header_on_end_s)
    )

    vehicle_speed = generate_vehicle_speed(
        time=time,
        base_speed=config.base_vehicle_speed,
        rng=rng,
    )
    cut_length = generate_cut_length(
        time=time,
        base_cut_length=config.base_cut_length,
        rng=rng,
    )

    audio = generate_background_audio(
        time=time,
        header_on=header_on,
        vehicle_speed=vehicle_speed,
        cut_length=cut_length,
        noise_std=config.noise_std,
        rng=rng,
    )
    voltage = np.full_like(time, fill_value=35.0, dtype=float)

    event_times = sample_event_times(
        time=time,
        header_on=header_on,
        vehicle_speed=vehicle_speed,
        cut_length=cut_length,
        event_rate_per_minute=config.event_rate_per_minute,
        rng=rng,
    )

    events: list[SyntheticStoneEvent] = []

    for event_time in event_times:
        event_index = int(np.searchsorted(time, event_time))
        local_speed = float(vehicle_speed[event_index])
        local_cut_length = float(cut_length[event_index])

        amplitude = sample_event_amplitude(
            vehicle_speed=local_speed,
            cut_length=local_cut_length,
            rng=rng,
        )
        width_s = float(rng.uniform(0.015, 0.055))

        audio += make_preimpact_acoustic_precursor(
            time=time,
            event_time=event_time,
            amplitude=0.35 * amplitude,
            duration_s=float(rng.uniform(0.45, 0.90)),
            rng=rng,
        )
        audio += make_stone_audio_transient(
            time=time,
            event_time=event_time,
            amplitude=amplitude,
            width_s=width_s,
            rng=rng,
        )
        voltage += make_voltage_spike(
            time=time,
            event_time=event_time + float(rng.uniform(0.05, 0.25)),
            amplitude=amplitude * float(rng.uniform(700.0, 1200.0)),
            width_s=float(rng.uniform(0.025, 0.075)),
        )

        events.append(
            SyntheticStoneEvent(
                run_name=config.run_name,
                event_time=float(event_time),
                amplitude=float(amplitude),
                width_s=width_s,
                vehicle_speed=local_speed,
                cut_length=local_cut_length,
            )
        )

    status = np.where(header_on, "On", "Off")

    frame = pd.DataFrame(
        {
            schema.sensor: audio,
            schema.vehicle_speed: vehicle_speed,
            schema.cut_length: cut_length,
            schema.voltage: voltage,
            schema.status: status,
            schema.header_on: header_on,
        },
        index=pd.Index(time, name="time_s"),
    )

    return SyntheticRun(
        run_name=config.run_name,
        frame=frame,
        events=events,
    )


def generate_synthetic_dataset(
    n_runs: int,
    base_seed: int = 42,
    duration_s: float = 120.0,
    sample_rate_hz: float = 1000.0,
) -> dict[str, pd.DataFrame]:
    """Generate a dictionary of synthetic runs compatible with the baseline."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")

    dataset: dict[str, pd.DataFrame] = {}

    for run_index in range(n_runs):
        run_name = f"synthetic_run_{run_index:03d}"
        header_on_start_s = min(5.0, 0.1 * duration_s)
        header_on_end_s = max(
            header_on_start_s + 1.0 / sample_rate_hz,
            0.95 * duration_s,
        )

        config = SyntheticRunConfig(
            run_name=run_name,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            header_on_start_s=header_on_start_s,
            header_on_end_s=header_on_end_s,
            base_vehicle_speed=2.5 + 0.15 * (run_index % 7),
            base_cut_length=10.0 + float(run_index % 5),
            event_rate_per_minute=1.0 + 0.25 * float(run_index % 4),
            random_seed=base_seed + run_index,
        )
        synthetic_run = generate_synthetic_run(config)
        dataset[run_name] = synthetic_run.frame

    return dataset


def make_time_index(duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Create a uniformly sampled time index in seconds."""
    n_samples = int(round(duration_s * sample_rate_hz)) + 1
    return np.arange(n_samples, dtype=float) / sample_rate_hz


def generate_vehicle_speed(
    time: np.ndarray,
    base_speed: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth synthetic vehicle speed."""
    slow_variation = 0.25 * np.sin(2.0 * np.pi * time / 45.0)
    random_walk = np.cumsum(rng.normal(0.0, 0.0008, size=len(time)))
    speed = base_speed + slow_variation + random_walk
    return np.clip(speed, 0.2, None)


def generate_cut_length(
    time: np.ndarray,
    base_cut_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth synthetic cut length."""
    slow_variation = 0.8 * np.sin(2.0 * np.pi * time / 60.0 + 0.7)
    jitter = rng.normal(0.0, 0.05, size=len(time))
    cut_length = base_cut_length + slow_variation + jitter
    return np.clip(cut_length, 1.0, None)


def generate_background_audio(
    time: np.ndarray,
    header_on: np.ndarray,
    vehicle_speed: np.ndarray,
    cut_length: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic harvesting background audio."""
    engine = 0.25 * np.sin(2.0 * np.pi * 35.0 * time)
    header_component = 0.15 * np.sin(2.0 * np.pi * 90.0 * time)
    crop_noise_scale = 0.08 + 0.03 * vehicle_speed + 0.005 * cut_length
    crop_noise = crop_noise_scale * rng.normal(0.0, 1.0, size=len(time))

    audio = engine + header_on.astype(float) * (header_component + crop_noise)
    audio += rng.normal(0.0, noise_std, size=len(time))
    return audio.astype(float)


def sample_event_times(
    time: np.ndarray,
    header_on: np.ndarray,
    vehicle_speed: np.ndarray,
    cut_length: np.ndarray,
    event_rate_per_minute: float,
    rng: np.random.Generator,
) -> list[float]:
    """Sample event times using operating-context-dependent risk."""
    if event_rate_per_minute < 0:
        raise ValueError("event_rate_per_minute cannot be negative.")

    dt = float(np.median(np.diff(time)))
    base_probability = event_rate_per_minute / 60.0 * dt

    speed_factor = vehicle_speed / max(float(np.median(vehicle_speed)), 1e-6)
    cut_factor = np.clip(
        float(np.median(cut_length)) / np.clip(cut_length, 1e-6, None),
        0.5,
        2.0,
    )
    risk = base_probability * speed_factor * cut_factor * header_on.astype(float)

    event_mask = rng.random(len(time)) < risk
    raw_event_times = time[event_mask]

    return enforce_minimum_event_spacing(
        event_times=raw_event_times.tolist(),
        min_spacing_s=3.0,
    )


def enforce_minimum_event_spacing(
    event_times: list[float],
    min_spacing_s: float,
) -> list[float]:
    """Remove events that occur too close together."""
    if not event_times:
        return []

    kept = [float(event_times[0])]

    for event_time in event_times[1:]:
        if float(event_time) - kept[-1] >= min_spacing_s:
            kept.append(float(event_time))

    return kept


def sample_event_amplitude(
    vehicle_speed: float,
    cut_length: float,
    rng: np.random.Generator,
) -> float:
    """Sample event amplitude using operating context."""
    speed_component = 0.5 * vehicle_speed
    cut_component = 3.0 / max(cut_length, 1.0)
    random_component = float(rng.uniform(1.5, 3.5))
    return random_component + speed_component + cut_component


def make_preimpact_acoustic_precursor(
    time: np.ndarray,
    event_time: float,
    amplitude: float,
    duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a weak pre-impact acoustic precursor before a stone event.

    The precursor represents rubbing, rattling, or contact noise that can appear
    before the main impact and before the metal-detector voltage spike. This
    makes the synthetic task better aligned with early detection.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    start_time = event_time - duration_s
    active = (time >= start_time) & (time <= event_time)

    if not np.any(active):
        return np.zeros_like(time, dtype=float)

    local_time = np.clip((time - start_time) / duration_s, 0.0, 1.0)
    ramp = local_time**1.5
    envelope = active.astype(float) * ramp

    low_contact = np.sin(2.0 * np.pi * 95.0 * (time - start_time))
    mid_contact = np.sin(2.0 * np.pi * 240.0 * (time - start_time))
    broadband = rng.normal(0.0, 1.0, size=len(time))

    return amplitude * envelope * (
        0.35 * low_contact + 0.35 * mid_contact + 0.30 * broadband
    )


def make_stone_audio_transient(
    time: np.ndarray,
    event_time: float,
    amplitude: float,
    width_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a short broadband stone-like acoustic transient."""
    envelope = np.exp(-0.5 * ((time - event_time) / width_s) ** 2)
    carrier_1 = np.sin(2.0 * np.pi * 120.0 * (time - event_time))
    carrier_2 = np.sin(2.0 * np.pi * 350.0 * (time - event_time))
    broadband = rng.normal(0.0, 1.0, size=len(time))
    return amplitude * envelope * (0.4 * carrier_1 + 0.3 * carrier_2 + 0.3 * broadband)


def make_voltage_spike(
    time: np.ndarray,
    event_time: float,
    amplitude: float,
    width_s: float,
) -> np.ndarray:
    """Create a voltage reference spike associated with a synthetic event."""
    return amplitude * np.exp(-0.5 * ((time - event_time) / width_s) ** 2)


def validate_config(config: SyntheticRunConfig) -> None:
    """Validate synthetic run configuration."""
    if config.duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    if config.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    valid_header_interval = (
        0.0
        <= config.header_on_start_s
        < config.header_on_end_s
        <= config.duration_s
    )

    if not valid_header_interval:
        raise ValueError(
            "Header-on interval must satisfy "
            "0 <= header_on_start_s < header_on_end_s <= duration_s."
        )

    if config.noise_std < 0:
        raise ValueError("noise_std cannot be negative.")

    if config.event_rate_per_minute < 0:
        raise ValueError("event_rate_per_minute cannot be negative.")
