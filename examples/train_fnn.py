import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import pandas as pd
from tqdm import tqdm

from config import Config
from src.data_preprocessor import DataPreprocessor
from src.data_loader import TabularDataset
from src.fnn_model import TabularFNN


def main(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    config = Config()
    os.makedirs(config.out_dir, exist_ok=True)

    # Load pickled tabular data
    with open(os.path.join(config.in_dir, "train_dataset1.pkl"), "rb") as f:
        train_data = pickle.load(f)

    if config.val:
        with open(os.path.join(config.in_dir, "val_dataset1.pkl"), "rb") as f:
            val_data = pickle.load(f)

    # Fit preprocessor from original CSV
    df = pd.read_csv("data/final_data.csv")
    df_train = df[
        (df["split"] == "train") & (df["in_dataset_1"] == True) & df["label"].notna()
    ]
    categorical = ["socio_eco"]
    continuous = ["floors_no"]

    preprocessor = DataPreprocessor(categorical, continuous)
    preprocessor.fit(df_train)
    preprocessor.save_constants("output/norm_constants_tabular.json")

    sample_record = train_data[list(train_data.keys())[0]][0]
    sample_tabular = {
        "floors_no": sample_record["floors_no"],
        "socio_eco": sample_record["socio_eco"],
    }
    input_dim = preprocessor.transform(pd.DataFrame([sample_tabular])).shape[1]

    # Datasets and loaders
    train_dataset = TabularDataset(train_data, preprocessor=preprocessor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val:
        val_dataset = TabularDataset(val_data, preprocessor=preprocessor)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularFNN(input_dim=input_dim, num_classes=config.num_classes).to(device)

    optimizer_cls = {"adam": optim.Adam, "sgd": optim.SGD}.get(config.optimizer)
    if optimizer_cls is None:
        raise ValueError("Invalid optimizer")

    optimizer = optimizer_cls(model.parameters(), lr=config.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr_min
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    pbar = tqdm(range(config.num_epochs))

    for epoch in pbar:
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = logits.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()

        scheduler.step()

        if config.val:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).long()
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
                    _, preds = logits.max(1)
                    val_total += y.size(0)
                    val_correct += preds.eq(y).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            pbar.set_postfix(
                {
                    "Train Loss": total_loss / len(train_loader),
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
                        model.state_dict(), os.path.join(config.out_dir, "FNN_best.pth")
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stop_patience:
                        print("Early stopping.")
                        break
        else:
            pbar.set_postfix(
                {
                    "Train Loss": total_loss / len(train_loader),
                    "Train Acc": 100.0 * correct / total,
                }
            )

    if not config.val:
        torch.save(model.state_dict(), os.path.join(config.out_dir, "FNN.pth"))


if __name__ == "__main__":
    main()
