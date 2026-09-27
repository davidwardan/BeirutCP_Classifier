"""Train and evaluate ANN or classical tabular-only baselines on new splits."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.advanced_metrics import compute_metrics
from src.data_preprocessor import DataPreprocessor
from src.ordinal_losses import CompositeOrdinalLoss, ordinal_logits_to_probabilities
from src.tabular_models import TabularOrdinalClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def flatten_records(data: dict) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    building_ids = []
    records = []
    labels = []
    for building_id, building_records in data.items():
        unique_labels = {int(record["label"]) for record in building_records}
        if len(unique_labels) != 1:
            raise ValueError(f"Building {building_id} has inconsistent labels")
        building_ids.append(str(building_id))
        records.append(building_records[0])
        labels.append(unique_labels.pop())
    return building_ids, pd.DataFrame(records), np.asarray(labels, dtype=np.int64)


def aligned_probabilities(model: Any, features: np.ndarray, num_classes: int) -> np.ndarray:
    probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), num_classes), dtype=float)
    aligned[:, np.asarray(model.classes_, dtype=int)] = probabilities
    return aligned


def save_predictions(
    output_path: Path,
    building_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame({"building_id": building_ids, "label": labels})
    for class_index in range(probabilities.shape[1]):
        frame[f"prob_{class_index}"] = probabilities[:, class_index]
    frame.to_csv(output_path, index=False)


def neural_probabilities(
    model: TabularOrdinalClassifier,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    ordinal_blend: float,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    batches = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs = model(batch.to(device))
            categorical = outputs["logits"].softmax(dim=1)
            ordinal = ordinal_logits_to_probabilities(outputs["ordinal_logits"])
            batches.append(
                ((1.0 - ordinal_blend) * categorical + ordinal_blend * ordinal)
                .cpu()
                .numpy()
            )
    return np.concatenate(batches)


def train_ann(
    config: dict[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    output_dir: Path,
) -> TabularOrdinalClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularOrdinalClassifier(
        input_dim=train_features.shape[1],
        num_classes=int(config["model"]["num_classes"]),
        hidden_dims=config["tabular_model"].get("hidden_dims", [64, 32]),
    ).to(device)
    loss_config = config["loss"]
    criterion = CompositeOrdinalLoss(
        num_classes=int(config["model"]["num_classes"]),
        ce_weight=float(loss_config["ce_weight"]),
        emd_weight=float(loss_config["emd_weight"]),
        ordinal_weight=float(loss_config["ordinal_weight"]),
        image_aux_weight=0.0,
        tabular_aux_weight=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["head_learning_rate"])
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=(len(train_features) % batch_size == 1),
    )
    ordinal_blend = float(config["evaluation"]["ordinal_probability_blend"])
    best_score = -float("inf")
    patience = 0
    history = []
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss, _ = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probabilities = neural_probabilities(
            model,
            validation_features,
            int(config["training"]["batch_size"]),
            device,
            ordinal_blend,
        )
        validation_metrics = compute_metrics(
            validation_probabilities, validation_labels
        )
        score = validation_metrics["macro_f1"] - float(
            config["evaluation"]["mae_selection_penalty"]
        ) * validation_metrics["class_index_mae"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation": validation_metrics,
                "selection_score": score,
            }
        )
        print(json.dumps(history[-1]))
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
                    "input_dim": train_features.shape[1],
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(config["training"]["early_stop_patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    data_dir = Path(config["data"]["directory"])
    train_ids, train_frame, train_labels = flatten_records(
        load_pickle(data_dir / "train_dataset1.pkl")
    )
    validation_ids, validation_frame, validation_labels = flatten_records(
        load_pickle(data_dir / "val_dataset1.pkl")
    )
    test_ids, test_frame, test_labels = flatten_records(
        load_pickle(data_dir / "test_dataset1.pkl")
    )
    preprocessor = DataPreprocessor(["socio_eco"], ["floors_no"])
    preprocessor.fit(train_frame)
    preprocessor.save_constants(output_dir / "norm_constants.json")
    train_features = preprocessor.transform(train_frame)
    validation_features = preprocessor.transform(validation_frame)
    test_features = preprocessor.transform(test_frame)

    model_type = config["tabular_model"]["type"]
    num_classes = int(config["model"]["num_classes"])
    if model_type == "ann":
        model = train_ann(
            config,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            output_dir,
        )
        test_probabilities = neural_probabilities(
            model,
            test_features,
            int(config["training"]["batch_size"]),
            next(model.parameters()).device,
            float(config["evaluation"]["ordinal_probability_blend"]),
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(train_features, train_labels)
        test_probabilities = aligned_probabilities(model, test_features, num_classes)
        with (output_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(config["tabular_model"].get("n_estimators", 500)),
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(train_features, train_labels)
        test_probabilities = aligned_probabilities(model, test_features, num_classes)
        with (output_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    else:
        raise ValueError(f"Unsupported tabular model type: {model_type}")

    metrics = compute_metrics(test_probabilities, test_labels)
    save_predictions(
        output_dir / "predictions.csv", test_ids, test_labels, test_probabilities
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"model": model_type, "test": metrics}, indent=2))


if __name__ == "__main__":
    main()
