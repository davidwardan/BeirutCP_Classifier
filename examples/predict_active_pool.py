"""Export fused and unimodal probabilities for an unlabeled building pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.advanced_data import MultiViewImageTabularDataset, build_transform
from src.advanced_models import AdaptiveOrdinalMultimodalClassifier
from src.data_preprocessor import DataPreprocessor
from src.ordinal_losses import ordinal_logits_to_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, default=None)
    parser.add_argument("--building-column", default="NID")
    parser.add_argument("--image-columns", nargs="*", default=None)
    parser.add_argument("--extension", default=".png")
    return parser.parse_args()


def resolve_image_paths(row: pd.Series, args: argparse.Namespace) -> list[str]:
    raw_paths = (
        [row[column] for column in args.image_columns if pd.notna(row[column])]
        if args.image_columns
        else [f"{row[args.building_column]}{args.extension}"]
    )
    paths = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = args.images_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(str(path.resolve()))
    return paths


def build_pool(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, list[dict]]:
    pool = {}
    for building_id, rows in frame.groupby(args.building_column, sort=True):
        image_paths = []
        for _, row in rows.iterrows():
            image_paths.extend(resolve_image_paths(row, args))
        first = rows.iloc[0]
        pool[str(building_id)] = [
            {
                "image_paths": list(dict.fromkeys(image_paths)),
                "floors_no": first.get("floors_no"),
                "socio_eco": first.get("socio_eco"),
                # Dataset collation requires a label; it is never used or exported.
                "label": 0,
            }
        ]
    return pool


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    pool = build_pool(pd.read_csv(args.pool), args)

    preprocessor = None
    if bool(config["model"]["use_tabular"]):
        constants_path = args.preprocessor or args.checkpoint.parent / "norm_constants.json"
        preprocessor = DataPreprocessor(["socio_eco"], ["floors_no"])
        preprocessor.load_constants(constants_path)
    dataset = MultiViewImageTabularDataset(
        pool,
        build_transform(int(config["data"]["image_size"]), training=False),
        preprocessor=preprocessor,
        max_views=int(config["data"]["max_views"]),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
    )
    model = AdaptiveOrdinalMultimodalClassifier(
        num_classes=int(config["model"]["num_classes"]),
        tabular_input_dim=int(checkpoint["tabular_input_dim"]),
        backbone=config["model"]["backbone"],
        embedding_dim=int(config["model"]["embedding_dim"]),
        fusion=config["model"]["fusion"],
        tabular_modality_dropout=0.0,
        pretrained=False,
        dino_repository=config["model"].get(
            "dino_repository", "facebookresearch/dinov2"
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    blend = float(config["evaluation"]["ordinal_probability_blend"])
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict unlabeled pool"):
            outputs = model(
                batch["images"].to(device),
                tabular=(
                    batch["tabular"].to(device)
                    if batch["tabular"].shape[1]
                    else None
                ),
                view_mask=batch["view_mask"].to(device),
                tabular_available=batch["tabular_available"].to(device),
            )
            categorical = outputs["logits"].softmax(dim=1)
            ordinal = ordinal_logits_to_probabilities(outputs["ordinal_logits"])
            fused = ((1.0 - blend) * categorical + blend * ordinal).cpu()
            image = outputs["image_logits"].softmax(dim=1).cpu()
            tabular = (
                outputs["tabular_logits"].softmax(dim=1).cpu()
                if outputs["tabular_logits"] is not None
                else None
            )
            for index, building_id in enumerate(batch["building_id"]):
                row = {"building_id": building_id}
                for class_index in range(fused.shape[1]):
                    row[f"prob_{class_index}"] = float(fused[index, class_index])
                    row[f"image_prob_{class_index}"] = float(image[index, class_index])
                    if tabular is not None:
                        row[f"tabular_prob_{class_index}"] = float(
                            tabular[index, class_index]
                        )
                row["view_attention"] = json.dumps(
                    outputs["view_attention"][index].cpu().tolist()
                )
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote probabilities for {len(rows)} buildings to {args.output}")


if __name__ == "__main__":
    main()

