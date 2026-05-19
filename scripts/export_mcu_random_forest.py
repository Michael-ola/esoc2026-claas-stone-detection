import argparse
from pathlib import Path

from claas_stone_detection.data.synthetic_io import (
    metadata_to_stone_events,
    read_synthetic_dataset,
    read_synthetic_metadata,
)
from claas_stone_detection.edge.random_forest_export import (
    export_random_forest_for_mcu,
    write_c_header,
    write_deployment_note,
    write_export_json,
    write_feature_list,
)
from claas_stone_detection.models.baseline import train_random_forest_baseline
from claas_stone_detection.pipelines.baseline_pipeline import (
    build_labeled_feature_table,
)
from claas_stone_detection.reference.labels import IGNORE_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export constrained Random Forest artefacts for Bonus 2 "
            "MCU deployment."
        )
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("synthetic_data"),
        help="Synthetic dataset directory containing CSV runs and metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mcu_random_forest"),
        help="Directory where MCU artefacts will be written.",
    )
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--positive-horizon-s", type=float, default=1.0)
    parser.add_argument("--post-event-exclusion-s", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nBonus 2 MCU Random Forest export")
    print("=" * 34)
    print(f"Synthetic directory: {args.synthetic_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Trees: {args.n_estimators}")
    print(f"Max depth: {args.max_depth}")
    print(f"Min samples leaf: {args.min_samples_leaf}")

    dataset = read_synthetic_dataset(
        synthetic_dir=args.synthetic_dir,
        max_runs=args.max_runs,
    )
    metadata = read_synthetic_metadata(
        synthetic_dir=args.synthetic_dir,
        allowed_runs=set(dataset.keys()),
    )
    events_by_run = metadata_to_stone_events(metadata)

    labeled_table, _ = build_labeled_feature_table(
        dataset=dataset,
        events_by_run=events_by_run,
        window_s=args.window_s,
        hop_s=args.hop_s,
        positive_horizon_s=args.positive_horizon_s,
        post_event_exclusion_s=args.post_event_exclusion_s,
        window_region="header-on",
    )

    train_table = labeled_table[labeled_table["label"] != IGNORE_LABEL].copy()

    if train_table.empty:
        raise RuntimeError("No trainable windows were produced.")

    model = train_random_forest_baseline(
        labeled_table=train_table,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    export = export_random_forest_for_mcu(
        model=model.model,
        feature_names=model.feature_columns,
        model_name="claas_stone_rf_mcu",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_c_header(
        export=export,
        output_path=args.output_dir / "claas_stone_rf_mcu.h",
    )
    write_export_json(
        export=export,
        output_path=args.output_dir / "claas_stone_rf_mcu.json",
    )
    write_feature_list(
        feature_names=export.feature_names,
        output_path=args.output_dir / "feature_order.txt",
    )
    write_deployment_note(
        export=export,
        output_path=args.output_dir / "README.md",
    )

    print("\nExport complete")
    print("-" * 15)
    print(f"Feature count: {len(export.feature_names)}")
    print(f"Tree count: {len(export.trees)}")
    print(f"Estimated model bytes: {export.estimated_model_bytes}")
    print(f"Header: {args.output_dir / 'claas_stone_rf_mcu.h'}")
    print(f"JSON: {args.output_dir / 'claas_stone_rf_mcu.json'}")
    print(f"Feature order: {args.output_dir / 'feature_order.txt'}")
    print(f"Deployment note: {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
