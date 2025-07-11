import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import random
import pandas as pd
from tqdm import tqdm

from config import Config
from src.hybrid_model import HybridSwinTabular
from src.data_loader import ImageTabularDataset
from src.data_preprocessor import DataPreprocessor


def main(seed: int = 42):
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Load config
    config = Config()
    os.makedirs(config.out_dir, exist_ok=True)

    # Load pickled dataset (with embedded images)
    with open(config.in_dir + "/train_dataset1.pkl", "rb") as f:
        train_data_dict = pickle.load(f)

    if config.val:
        with open(config.in_dir + "/val_dataset1.pkl", "rb") as f:
            val_data_dict = pickle.load(f)

    # Define transforms
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create and fit preprocessor using raw CSV (required for scaling/tabular prep)
    df = pd.read_csv("data/final_data.csv")
    df_train = df[
        (df["split"] == "train") & (df["in_dataset_1"] == True) & df["label"].notna()
    ]

    categorical_features = ["socio_eco"]
    continuous_features = ["floors_no"]

    preprocessor = DataPreprocessor(categorical_features, continuous_features)
    preprocessor.fit(df_train)
    preprocessor.save_constants("output/norm_constants.json")

    # Create PyTorch datasets/loaders
    train_dataset = ImageTabularDataset(
        train_data_dict, transform=train_transform, preprocessor=preprocessor
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val:
        val_dataset = ImageTabularDataset(
            val_data_dict, transform=val_transform, preprocessor=preprocessor
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabular_input_dim = preprocessor.transform(
        pd.DataFrame([train_data_dict[list(train_data_dict.keys())[0]][0]])
    ).shape[1]

    model = HybridSwinTabular(
        num_classes=config.num_classes, tabular_input_dim=tabular_input_dim
    ).to(device)
    optimizer_cls = {"adam": optim.Adam, "sgd": optim.SGD}.get(config.optimizer)
    if optimizer_cls is None:
        raise ValueError("Invalid optimizer")

    # Use different learning rates for swin, tabular_net, and fusion_mlp
    optimizer = optimizer_cls(
        [
            {"params": model.swin.parameters(), "lr": config.lr_max},
            {"params": model.tabular_net.parameters(), "lr": config.lr_max * 10},
            {"params": model.fusion_mlp.parameters(), "lr": config.lr_max * 10},
        ]
    )

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs - warmup_epochs, eta_min=config.lr_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Model ready to train...")
    best_val_loss = float("inf")
    patience_counter = 0

    pbar = tqdm(range(config.num_epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, tabular, labels in train_loader:
            inputs, tabular, labels = (
                inputs.to(device),
                tabular.to(device),
                labels.to(device).long(),
            )
            optimizer.zero_grad()
            outputs = model(inputs, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Validation
        if config.val:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, tabular, labels in val_loader:
                    inputs, tabular, labels = (
                        inputs.to(device),
                        tabular.to(device),
                        labels.to(device).long(),
                    )
                    outputs = model(inputs, tabular)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            pbar.set_postfix(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Acc": 100.0 * correct / total,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                }
            )

            if config.early_stop_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.out_dir, "Hybrid_best.pth"),
                    )
                    print("Best model saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stop_patience:
                        print("Early stopping triggered.")
                        break
        else:
            pbar.set_postfix(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Acc": 100.0 * correct / total,
                }
            )

    if not config.val:
        torch.save(model.state_dict(), os.path.join(config.out_dir, "Hybrid.pth"))


if __name__ == "__main__":
    main()
