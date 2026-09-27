"""Build path-based, building-level pickle files with optional multiple views."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], default="dataset1")
    parser.add_argument("--building-column", default="NID")
    parser.add_argument(
        "--image-columns",
        nargs="*",
        default=None,
        help="Columns containing filenames/paths. By default, NID + --extension is used.",
    )
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def resolve_paths(row: pd.Series, args: argparse.Namespace) -> list[str]:
    if args.image_columns:
        raw_paths = [row[column] for column in args.image_columns if pd.notna(row[column])]
    else:
        raw_paths = [f"{row[args.building_column]}{args.extension}"]
    resolved = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = args.images_dir / path
        if path.exists() or args.allow_missing_images:
            resolved.append(str(path.resolve()))
        else:
            raise FileNotFoundError(path)
    return resolved


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    membership_column = "in_dataset_1" if args.dataset == "dataset1" else "in_dataset_2"
    frame = frame[frame[membership_column].astype(bool) & frame["split"].notna()].copy()
    label_map = {label: index for index, label in enumerate(Config.labels)}
    frame["encoded_label"] = frame["label"].map(label_map)
    if frame["encoded_label"].isna().any():
        unknown = sorted(frame.loc[frame["encoded_label"].isna(), "label"].unique())
        raise ValueError(f"Unknown labels: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split = frame[frame["split"] == split_name]
        output: dict[str, list[dict]] = {}
        for building_id, rows in split.groupby(args.building_column, sort=True):
            labels = rows["encoded_label"].astype(int).unique()
            if len(labels) != 1:
                raise ValueError(f"Building {building_id} has inconsistent labels")
            image_paths: list[str] = []
            for _, row in rows.iterrows():
                image_paths.extend(resolve_paths(row, args))
            first = rows.iloc[0]
            output[str(building_id)] = [
                {
                    "image_paths": list(dict.fromkeys(image_paths)),
                    "floors_no": first.get("floors_no"),
                    "socio_eco": first.get("socio_eco"),
                    "label": int(labels[0]),
                }
            ]
        output_path = args.output_dir / f"{split_name}_{args.dataset}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(output, handle)
        print(f"Wrote {len(output)} buildings to {output_path}")


if __name__ == "__main__":
    main()

