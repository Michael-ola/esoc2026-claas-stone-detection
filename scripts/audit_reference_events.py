import argparse
from pathlib import Path

import pandas as pd

from claas_stone_detection.data.io import read_dataset
from claas_stone_detection.reference.episodes import extract_header_on_episodes
from claas_stone_detection.reference.events import StoneEvent, detect_voltage_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit reference voltage events used for early stone detection."
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
        help="Sampling interval in seconds used when loading MF4 files.",
    )
    parser.add_argument(
        "--event-threshold-quantile",
        type=float,
        default=0.999,
        help="Episode-level VoltageSignal quantile used for reference events.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help="Minimum peak-to-threshold ratio used for the filtered summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nCLAAS reference event audit")
    print("=" * 34)
    print(f"Data directory: {args.data_dir}")
    print(f"Raster: {args.raster}")
    print(f"Event threshold quantile: {args.event_threshold_quantile}")
    print(f"Filtered ratio threshold: {args.min_ratio}")

    dataset = read_dataset(args.data_dir, raster=args.raster)

    all_rows: list[dict[str, float | int | str | bool]] = []

    for run_name, df in dataset.items():
        episodes = extract_header_on_episodes(df)
        events = detect_voltage_events(
            df=df,
            episodes=episodes,
            threshold_quantile=args.event_threshold_quantile,
            run_name=run_name,
        )

        rows = events_to_rows(events)
        all_rows.extend(rows)

        print_run_summary(
            run_name=run_name,
            duration_s=float(df.index.max() - df.index.min()),
            n_episodes=len(episodes),
            rows=rows,
            min_ratio=args.min_ratio,
        )

    audit_table = pd.DataFrame(all_rows)

    print("\nOverall summary")
    print("-" * 15)

    if audit_table.empty:
        print("No reference events detected.")
        return

    print(f"Runs: {len(dataset)}")
    print(f"Reference events: {len(audit_table)}")
    print(
        "Peak-to-threshold ratio: "
        f"min={audit_table['peak_to_threshold_ratio'].min():.3f}, "
        f"median={audit_table['peak_to_threshold_ratio'].median():.3f}, "
        f"max={audit_table['peak_to_threshold_ratio'].max():.3f}"
    )
    print(
        "Events inside HeaderOn episode: "
        f"{int(audit_table['inside_header_on_episode'].sum())}/{len(audit_table)}"
    )
    print(
        f"Events with ratio >= {args.min_ratio:.2f}: "
        f"{int((audit_table['peak_to_threshold_ratio'] >= args.min_ratio).sum())}/"
        f"{len(audit_table)}"
    )

    print("\nWeakest reference events")
    print("-" * 24)
    print(
        audit_table.sort_values("peak_to_threshold_ratio")
        .head(10)[
            [
                "run_name",
                "peak_time",
                "peak_voltage",
                "threshold",
                "peak_to_threshold_ratio",
                "time_from_episode_end_to_peak",
                "inside_header_on_episode",
            ]
        ]
        .to_string(index=False)
    )


def events_to_rows(
    events: list[StoneEvent],
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []

    for event_index, event in enumerate(events, start=1):
        time_from_episode_end_to_peak = event.peak_time - event.episode_end_time
        inside_header_on_episode = event.peak_time <= event.episode_end_time

        rows.append(
            {
                "run_name": event.run_name,
                "event_index": event_index,
                "start_time": event.start_time,
                "peak_time": event.peak_time,
                "end_time": event.end_time,
                "duration_s": event.duration_s,
                "peak_voltage": event.peak_voltage,
                "threshold": event.threshold,
                "peak_to_threshold_ratio": event.peak_to_threshold_ratio,
                "episode_start_time": event.episode_start_time,
                "episode_end_time": event.episode_end_time,
                "time_from_episode_end_to_peak": time_from_episode_end_to_peak,
                "inside_header_on_episode": inside_header_on_episode,
            }
        )

    return rows


def print_run_summary(
    run_name: str,
    duration_s: float,
    n_episodes: int,
    rows: list[dict[str, float | int | str | bool]],
    min_ratio: float,
) -> None:
    print(f"\n{run_name}")
    print("-" * len(run_name))
    print(f"Duration: {duration_s:.2f}s")
    print(f"Header-on episodes: {n_episodes}")
    print(f"Reference events: {len(rows)}")

    if not rows:
        return

    table = pd.DataFrame(rows)
    strong_count = int((table["peak_to_threshold_ratio"] >= min_ratio).sum())
    inside_count = int(table["inside_header_on_episode"].sum())

    print(f"Events with ratio >= {min_ratio:.2f}: {strong_count}/{len(table)}")
    print(f"Events inside HeaderOn episode: {inside_count}/{len(table)}")
    print(
        "Ratio stats: "
        f"min={table['peak_to_threshold_ratio'].min():.3f}, "
        f"median={table['peak_to_threshold_ratio'].median():.3f}, "
        f"max={table['peak_to_threshold_ratio'].max():.3f}"
    )

    print("\nEvents:")
    for row in rows:
        print(
            f"  Event {int(row['event_index']):02d}: "
            f"peak={float(row['peak_time']):8.3f}s | "
            f"peak_voltage={float(row['peak_voltage']):10.3f} | "
            f"threshold={float(row['threshold']):10.3f} | "
            f"ratio={float(row['peak_to_threshold_ratio']):6.3f} | "
            f"episode_end_delta={float(row['time_from_episode_end_to_peak']):7.3f}s | "
            f"inside_header_on={bool(row['inside_header_on_episode'])}"
        )


if __name__ == "__main__":
    main()
