import argparse
from pathlib import Path

import pandas as pd

from claas_stone_detection.synthetic_data.generator import (
    SyntheticRunConfig,
    generate_synthetic_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CLAAS-like sensor runs for Bonus 1."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_data"),
        help="Directory where synthetic runs will be written.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of synthetic runs to generate.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=120.0,
        help="Duration of each synthetic run in seconds.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=1000.0,
        help="Synthetic sampling rate in Hz.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Each run uses base_seed + run_index.",
    )
    parser.add_argument(
        "--event-rate-per-minute",
        type=float,
        default=2.0,
        help="Approximate synthetic stone event rate during HeaderOn operation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_runs <= 0:
        raise ValueError("n_runs must be positive.")

    if args.duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    if args.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    print("\nSynthetic CLAAS CSV dataset generation")
    print("=" * 36)
    print(f"Output directory: {args.output_dir}")
    print(f"Runs: {args.n_runs}")
    print(f"Duration per run: {args.duration_s}s")
    print(f"Sample rate: {args.sample_rate_hz}Hz")
    print(f"Base seed: {args.base_seed}")

    for run_index in range(args.n_runs):
        run_name = f"synthetic_run_{run_index:03d}"
        header_on_start_s = min(5.0, 0.1 * args.duration_s)
        header_on_end_s = max(
            header_on_start_s + 1.0 / args.sample_rate_hz,
            0.95 * args.duration_s,
        )

        config = SyntheticRunConfig(
            run_name=run_name,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            header_on_start_s=header_on_start_s,
            header_on_end_s=header_on_end_s,
            base_vehicle_speed=2.5 + 0.15 * (run_index % 7),
            base_cut_length=10.0 + float(run_index % 5),
            event_rate_per_minute=args.event_rate_per_minute
            + 0.25 * float(run_index % 4),
            random_seed=args.base_seed + run_index,
        )

        synthetic_run = generate_synthetic_run(config)

        output_path = args.output_dir / f"{run_name}.csv"
        synthetic_run.frame.to_csv(output_path, index_label="time_s")

        for event in synthetic_run.events:
            metadata_rows.append(
                {
                    "run_name": event.run_name,
                    "event_time": event.event_time,
                    "amplitude": event.amplitude,
                    "width_s": event.width_s,
                    "vehicle_speed": event.vehicle_speed,
                    "cut_length": event.cut_length,
                    "run_file": output_path.name,
                }
            )

        print(
            f"{run_name}: rows={len(synthetic_run.frame)} | "
            f"events={len(synthetic_run.events)} | file={output_path}"
        )

    metadata = pd.DataFrame(
        metadata_rows,
        columns=[
            "run_name",
            "event_time",
            "amplitude",
            "width_s",
            "vehicle_speed",
            "cut_length",
            "run_file",
        ],
    )

    metadata_path = args.output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print("\nDone")
    print("-" * 4)
    print(f"Generated runs: {args.n_runs}")
    print(f"Injected events: {len(metadata)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
