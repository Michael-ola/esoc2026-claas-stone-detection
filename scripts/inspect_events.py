import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from claas_stone_detection.io import read_dataset
from claas_stone_detection.schema import DEFAULT_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect candidate voltage events in CLAAS MF4 files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing CLAAS .mf4 files. Defaults to ./data.",
    )
    parser.add_argument(
        "--raster",
        type=float,
        default=0.001,
        help="Sampling interval in seconds for inspection. Defaults to 0.001.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile used as adaptive threshold. Defaults to 0.995.",
    )
    parser.add_argument(
        "--min-gap-s",
        type=float,
        default=0.5,
        help="Minimum time gap used to separate voltage event groups.",
    )
    parser.add_argument(
        "--header-grace-s",
        type=float,
        default=2.0,
        help="Look-back window for checking recent HeaderOn activity.",
    )
    return parser.parse_args()


def group_candidate_regions(
    candidate_times: np.ndarray,
    min_gap_s: float,
) -> list[tuple[float, float]]:
    """Group candidate timestamps separated by less than min_gap_s."""
    if len(candidate_times) == 0:
        return []

    groups: list[tuple[float, float]] = []
    start_time = float(candidate_times[0])
    previous_time = float(candidate_times[0])

    for time_s in candidate_times[1:]:
        time_s = float(time_s)

        if time_s - previous_time > min_gap_s:
            groups.append((start_time, previous_time))
            start_time = time_s

        previous_time = time_s

    groups.append((start_time, previous_time))

    return groups


def header_was_recently_on(
    df: pd.DataFrame,
    peak_time: float,
    lookback_s: float,
) -> bool:
    schema = DEFAULT_SCHEMA
    context = df.loc[max(0.0, peak_time - lookback_s) : peak_time]
    return bool(context[schema.header_on].any())


def main() -> None:
    args = parse_args()
    schema = DEFAULT_SCHEMA

    dataset = read_dataset(args.data_dir, raster=args.raster)

    print("\nCLAAS grouped voltage event inspection")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Raster: {args.raster}")
    print(f"Threshold quantile: {args.threshold_quantile}")
    print(f"Min event gap: {args.min_gap_s}s")
    print(f"Header grace/lookback: {args.header_grace_s}s")
    print(f"Number of runs: {len(dataset)}")

    for run_name, df in dataset.items():
        voltage = df[schema.voltage]

        threshold = float(voltage.quantile(args.threshold_quantile))
        candidate_df = df[df[schema.voltage] > threshold]
        candidate_times = candidate_df.index.to_numpy(dtype=float)

        groups = group_candidate_regions(
            candidate_times=candidate_times,
            min_gap_s=args.min_gap_s,
        )

        print(f"\n{run_name}")
        print("-" * len(run_name))
        print(f"Duration: {df.index.max() - df.index.min():.2f}s")
        print(f"Rows: {len(df)}")
        print(f"Voltage min: {voltage.min():.3f}")
        print(f"Voltage median: {voltage.median():.3f}")
        print(f"Voltage max: {voltage.max():.3f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Candidate samples: {len(candidate_df)}")
        print(f"Candidate groups: {len(groups)}")

        for i, (start_time, end_time) in enumerate(groups, start=1):
            group_df = df.loc[start_time:end_time]
            peak_time = float(group_df[schema.voltage].idxmax())
            peak_voltage = float(group_df.loc[peak_time, schema.voltage])
            header_on_at_peak = bool(group_df.loc[peak_time, schema.header_on])
            recent_header_on = header_was_recently_on(
                df=df,
                peak_time=peak_time,
                lookback_s=args.header_grace_s,
            )

            print(
                f"  Event {i:02d}: "
                f"start={start_time:8.3f}s | "
                f"peak={peak_time:8.3f}s | "
                f"end={end_time:8.3f}s | "
                f"duration={end_time - start_time:6.3f}s | "
                f"peak_voltage={peak_voltage:10.3f} | "
                f"HeaderOn@peak={header_on_at_peak} | "
                f"HeaderOn_recent={recent_header_on}"
            )


if __name__ == "__main__":
    main()