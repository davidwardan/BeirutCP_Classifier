"""Train the ordinal, adaptive-fusion, multi-view experiment on a GPU server."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.advanced_data import MultiViewImageTabularDataset, build_transform
from src.advanced_metrics import compute_metrics
from src.advanced_models import AdaptiveOrdinalMultimodalClassifier
from src.data_preprocessor import DataPreprocessor
from src.ordinal_losses import (
    CompositeOrdinalLoss,
    ordinal_logits_to_probabilities,
)
from src.vision_backbones import set_encoder_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/advanced_experiment.json"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def records_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame([record for records in data.values() for record in records])


def build_preprocessor(train_data: dict, output_dir: Path) -> DataPreprocessor:
    preprocessor = DataPreprocessor(
        categorical_features=["socio_eco"], continuous_features=["floors_no"]
    )
    preprocessor.fit(records_frame(train_data))
    preprocessor.save_constants(output_dir / "norm_constants.json")
    return preprocessor


def probabilities_from_outputs(
    outputs: dict[str, torch.Tensor | None], ordinal_blend: float
) -> torch.Tensor:
    categorical = outputs["logits"].softmax(dim=1)
    if ordinal_blend <= 0 or outputs.get("ordinal_logits") is None:
        return categorical
    ordinal = ordinal_logits_to_probabilities(outputs["ordinal_logits"])
    return (1.0 - ordinal_blend) * categorical + ordinal_blend * ordinal


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeOrdinalLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ordinal_blend: float,
) -> tuple[dict[str, float], float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    probabilities = []
    labels_all = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if training else "validate"):
            images = batch["images"].to(device)
            tabular = batch["tabular"].to(device)
            view_mask = batch["view_mask"].to(device)
            tabular_available = batch["tabular_available"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(
                images,
                tabular=tabular if tabular.shape[1] else None,
                view_mask=view_mask,
                tabular_available=tabular_available,
            )
            loss, _ = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            probabilities.append(
                probabilities_from_outputs(outputs, ordinal_blend).detach().cpu().numpy()
            )
            labels_all.append(labels.detach().cpu().numpy())

    probabilities_array = np.concatenate(probabilities)
    labels_array = np.concatenate(labels_all)
    return compute_metrics(probabilities_array, labels_array), float(np.mean(losses))


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data_dir or Path(config["data"]["directory"])
    output_dir = args.output_dir or Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    seed_everything(int(config["seed"]))

    dataset_name = config["data"]["dataset"]
    train_data = load_pickle(data_dir / f"train_{dataset_name}.pkl")
    validation_data = load_pickle(data_dir / f"val_{dataset_name}.pkl")
    use_tabular = bool(config["model"]["use_tabular"])
    preprocessor = build_preprocessor(train_data, output_dir) if use_tabular else None
    tabular_dim = (
        preprocessor.transform(records_frame(train_data).iloc[:1]).shape[1]
        if preprocessor is not None
        else 0
    )

    image_size = int(config["data"]["image_size"])
    max_views = int(config["data"]["max_views"])
    train_dataset = MultiViewImageTabularDataset(
        train_data,
        build_transform(image_size, training=True),
        preprocessor=preprocessor,
        max_views=max_views,
        training=True,
    )
    validation_dataset = MultiViewImageTabularDataset(
        validation_data,
        build_transform(image_size, training=False),
        preprocessor=preprocessor,
        max_views=max_views,
        training=False,
    )
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = AdaptiveOrdinalMultimodalClassifier(
        num_classes=int(config["model"]["num_classes"]),
        tabular_input_dim=tabular_dim,
        backbone=config["model"]["backbone"],
        embedding_dim=int(config["model"]["embedding_dim"]),
        fusion=config["model"]["fusion"],
        tabular_modality_dropout=float(config["model"]["tabular_modality_dropout"]),
        pretrained=bool(config["model"]["pretrained"]),
        dino_repository=config["model"].get(
            "dino_repository", "facebookresearch/dinov2"
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_config = config["loss"]
    criterion = CompositeOrdinalLoss(
        num_classes=int(config["model"]["num_classes"]),
        ce_weight=float(loss_config["ce_weight"]),
        emd_weight=float(loss_config["emd_weight"]),
        ordinal_weight=float(loss_config["ordinal_weight"]),
        image_aux_weight=float(loss_config["image_aux_weight"]),
        tabular_aux_weight=float(loss_config["tabular_aux_weight"]),
    ).to(device)

    freeze_epochs = int(config["training"]["freeze_encoder_epochs"])
    set_encoder_trainable(model.encoder, freeze_epochs == 0)
    encoder_parameters = list(model.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {
                "params": encoder_parameters,
                "lr": float(config["training"]["encoder_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(config["training"]["head_learning_rate"]),
            },
        ]
    )
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"]["warmup_epochs"])
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=float(config["training"]["minimum_learning_rate"]),
            ),
        ],
        milestones=[warmup_epochs],
    )

    best_score = -float("inf")
    patience = 0
    history = []
    ordinal_blend = float(config["evaluation"]["ordinal_probability_blend"])
    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            set_encoder_trainable(model.encoder, True)
        train_metrics, train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, ordinal_blend
        )
        validation_metrics, validation_loss = run_epoch(
            model, validation_loader, criterion, device, None, ordinal_blend
        )
        scheduler.step()
        score = validation_metrics["macro_f1"] - float(
            config["evaluation"]["mae_selection_penalty"]
        ) * validation_metrics["class_index_mae"]
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record))
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "tabular_input_dim": tabular_dim,
                    "epoch": epoch + 1,
                    "validation_metrics": validation_metrics,
                },
                output_dir / "best_model.pth",
            )
        else:
            patience += 1
            if patience >= int(config["training"]["early_stop_patience"]):
                break


if __name__ == "__main__":
    main()
