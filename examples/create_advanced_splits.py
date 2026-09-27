"""Create reproducible building- or spatial-grouped experiment manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.splitting import assign_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/all_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/advanced_splits"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    parser.add_argument("--building-column", default="NID")
    parser.add_argument(
        "--spatial-group-column",
        default=None,
        help="Optional neighborhood/grid column. When supplied, whole spatial groups are held out.",
    )
    parser.add_argument("--label-column", default="final_label")
    parser.add_argument("--image-column", default="IMGAVAL")
    parser.add_argument("--floors-column", default="floors_no")
    parser.add_argument("--socioeconomic-column", default="socio_eco")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args()


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def prepare_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = frame.copy()
    data = data.replace([-999, "-999"], np.nan)
    data["label"] = data[args.label_column]
    data["has_image"] = as_boolean(data[args.image_column])
    data["has_tabular"] = data[
        [args.floors_column, args.socioeconomic_column]
    ].notna().all(axis=1)
    data["in_dataset_1"] = data["has_image"] & data["has_tabular"]
    data["in_dataset_2"] = data["has_image"]
    if args.socioeconomic_column in data:
        data[args.socioeconomic_column] = data[args.socioeconomic_column].replace(
            "Low-income zone", "Majority low-income zone"
        )
    return data[data["label"].notna()].copy()


def main() -> None:
    args = parse_args()
    data = prepare_frame(pd.read_csv(args.input), args)
    group_column = args.spatial_group_column or args.building_column
    if group_column not in data:
        raise KeyError(f"Group column '{group_column}' is not present in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        assigned, summary = assign_splits(
            data,
            seed=seed,
            group_column=group_column,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
        )
        stem = f"seed_{seed}_{'spatial' if args.spatial_group_column else 'building'}"
        assigned.to_csv(args.output_dir / f"{stem}.csv", index=False)
        class_counts = (
            assigned.groupby(["split", "label"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        metadata = {
            **asdict(summary),
            "input": str(args.input),
            "building_column": args.building_column,
            "spatial_group_column": args.spatial_group_column,
            "test_fraction": args.test_fraction,
            "validation_fraction": args.validation_fraction,
            "class_counts": class_counts,
            "test_set_policy": "created once from dataset 1; never modified by balancing",
        }
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

