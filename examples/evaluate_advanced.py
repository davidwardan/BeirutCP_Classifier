"""Evaluate an advanced checkpoint and export building-level probabilities."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.advanced_data import MultiViewImageTabularDataset, build_transform
from src.advanced_metrics import compute_metrics
from src.advanced_models import AdaptiveOrdinalMultimodalClassifier
from src.data_preprocessor import DataPreprocessor
from src.ordinal_losses import ordinal_logits_to_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tabular-condition",
        choices=["real", "masked", "shuffled"],
        default="real",
        help="Ablate tabular information without changing image inputs.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    return parser.parse_args()


def shuffled_tabular_copy(data: dict, seed: int) -> dict:
    """Return a copy with floors and socioeconomic values permuted by building."""
    output = copy.deepcopy(data)
    keys = list(output)
    source_keys = keys.copy()
    random.Random(seed).shuffle(source_keys)
    for target_key, source_key in zip(keys, source_keys):
        source = data[source_key][0]
        for record in output[target_key]:
            record["floors_no"] = source["floors_no"]
            record["socio_eco"] = source["socio_eco"]
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    with args.data.open("rb") as handle:
        data = pickle.load(handle)
    if args.tabular_condition != "real" and not bool(config["model"]["use_tabular"]):
        raise ValueError("Tabular ablations require a model trained with tabular inputs")
    if args.tabular_condition == "shuffled":
        data = shuffled_tabular_copy(data, args.shuffle_seed)

    preprocessor = None
    if bool(config["model"]["use_tabular"]):
        constants_path = args.preprocessor or args.checkpoint.parent / "norm_constants.json"
        preprocessor = DataPreprocessor(["socio_eco"], ["floors_no"])
        preprocessor.load_constants(constants_path)

    dataset = MultiViewImageTabularDataset(
        data,
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

    rows = []
    labels_all = []
    probabilities_all = []
    blend = float(config["evaluation"]["ordinal_probability_blend"])
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            tabular_available = batch["tabular_available"].to(device)
            if args.tabular_condition == "masked":
                tabular_available = torch.zeros_like(tabular_available)
            outputs = model(
                batch["images"].to(device),
                tabular=(
                    batch["tabular"].to(device)
                    if batch["tabular"].shape[1]
                    else None
                ),
                view_mask=batch["view_mask"].to(device),
                tabular_available=tabular_available,
            )
            categorical = outputs["logits"].softmax(dim=1)
            ordinal = ordinal_logits_to_probabilities(outputs["ordinal_logits"])
            probabilities = (1.0 - blend) * categorical + blend * ordinal
            image_probabilities = outputs["image_logits"].softmax(dim=1)
            tabular_probabilities = (
                outputs["tabular_logits"].softmax(dim=1)
                if outputs["tabular_logits"] is not None
                else None
            )
            labels = batch["label"].numpy()
            probabilities_np = probabilities.cpu().numpy()
            labels_all.append(labels)
            probabilities_all.append(probabilities_np)
            for index, building_id in enumerate(batch["building_id"]):
                row = {
                    "building_id": building_id,
                    "label": int(labels[index]),
                    "tabular_condition": args.tabular_condition,
                }
                for class_index in range(probabilities_np.shape[1]):
                    row[f"prob_{class_index}"] = float(probabilities_np[index, class_index])
                    row[f"image_prob_{class_index}"] = float(
                        image_probabilities[index, class_index].cpu()
                    )
                    if tabular_probabilities is not None:
                        row[f"tabular_prob_{class_index}"] = float(
                            tabular_probabilities[index, class_index].cpu()
                        )
                row["view_attention"] = json.dumps(
                    outputs["view_attention"][index].cpu().tolist()
                )
                rows.append(row)

    labels_array = np.concatenate(labels_all)
    probabilities_array = np.concatenate(probabilities_all)
    metrics = compute_metrics(probabilities_array, labels_array)
    predictions_frame = pd.DataFrame(rows)
    predictions_frame.to_csv(
        args.output_dir / f"predictions_{args.tabular_condition}.csv", index=False
    )
    metrics_text = json.dumps(metrics, indent=2, sort_keys=True)
    (args.output_dir / f"metrics_{args.tabular_condition}.json").write_text(
        metrics_text, encoding="utf-8"
    )
    if args.tabular_condition == "real":
        predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
        (args.output_dir / "metrics.json").write_text(
            metrics_text, encoding="utf-8"
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
