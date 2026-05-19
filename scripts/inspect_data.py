import argparse
from pathlib import Path

from claas_stone_detection.data.io import read_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect CLAAS MF4 measurement files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the CLAAS .mf4 files. Defaults to ./data.",
    )
    parser.add_argument(
        "--raster",
        type=float,
        default=None,
        help=(
            "Optional sampling interval in seconds for resampling. "
            "Example: --raster 0.001 for 1 ms sampling."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load all MF4 files and print a compact summary."""
    args = parse_args()

    dataset = read_dataset(args.data_dir, raster=args.raster)

    print("\nCLAAS MF4 dataset inspection")
    print("=" * 32)
    print(f"Data directory: {args.data_dir}")
    print(f"Raster: {args.raster}")
    print(f"Number of runs: {len(dataset)}")

    for run_name, df in dataset.items():
        print(f"\n{run_name}")
        print("-" * len(run_name))
        print(f"Shape: {df.shape}")
        print(f"Time range: {df.index.min():.6f}s to {df.index.max():.6f}s")
        print(f"Duration: {df.index.max() - df.index.min():.2f}s")
        print(f"Columns: {list(df.columns)}")
        print(f"HeaderOn values: {df['HeaderOn'].value_counts().to_dict()}")

        print("\nFirst rows:")
        print(df.head())


if __name__ == "__main__":
    main()