import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.functional as F


# ------------------------------------------------------------------
# Label‑smoothing toward adjacent construction periods only
# ------------------------------------------------------------------
def _neighbour_kernel(num_classes: int, bandwidth: int = 1):
    idx = torch.arange(num_classes)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = ((diff > 0) & (diff <= bandwidth)).float()
    row_sums = mask.sum(dim=1, keepdim=True)
    # normalise each row so the neighbour probabilities sum to 1
    mask = torch.where(row_sums == 0, mask, mask / row_sums)
    return mask


def neighbour_smooth(
    y: torch.Tensor, num_classes: int, eps: float = 0.1, bandwidth: int = 1
):
    """
    Distribute `eps` of the probability mass uniformly to the `bandwidth`
    neighbouring classes on either side of the true label.
    y: (N,) integer class indices.
    Returns a (N, num_classes) tensor of soft labels.
    """
    K = _neighbour_kernel(num_classes, bandwidth).to(y.device)  # (C, C)
    one_hot = F.one_hot(y, num_classes).float()
    neigh_dist = K[y]  # (N, C)
    return one_hot * (1.0 - eps) + eps * neigh_dist


# ------------------------------------------------------------------


import random
import pandas as pd
from tqdm import tqdm

from config import Config
from src.swint_model import SwinTClassifier
from src.data_loader import ImageDataset
from src.data_preprocessor import DataPreprocessor


def main(seed: int = 42, soft_labels: bool = True):
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Load config
    config = Config()
    os.makedirs(config.out_dir, exist_ok=True)

    # Load pickled dataset (with embedded images)
    with open(config.in_dir + "/train_dataset2.pkl", "rb") as f:
        train_data_dict = pickle.load(f)

    if config.val:
        with open(config.in_dir + "/val_dataset2.pkl", "rb") as f:
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

    # Create PyTorch datasets/loaders
    train_dataset = ImageDataset(train_data_dict, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val:
        val_dataset = ImageDataset(val_data_dict, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)

    optimizer_cls = {"adam": optim.Adam, "sgd": optim.SGD}.get(config.optimizer)
    if optimizer_cls is None:
        raise ValueError("Invalid optimizer")

    optimizer = optimizer_cls(model.parameters(), lr=config.lr_max)

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
    if not soft_labels:
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

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            if not soft_labels:
                loss = criterion(outputs, labels)
            else:
                # Use neighbour smoothing for soft labels
                soft_labels = neighbour_smooth(
                    labels,
                    num_classes=config.num_classes,
                    eps=0.1,  # adjust if desired
                    bandwidth=1,
                )
                loss = F.cross_entropy(outputs, soft_labels)
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
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).long()
                    outputs = model(inputs)
                    if not soft_labels:
                        loss = criterion(outputs, labels)
                    else:
                        soft_labels = neighbour_smooth(
                            labels,
                            num_classes=config.num_classes,
                            eps=0.1,  # adjust if desired
                            bandwidth=1,
                        )
                        loss = F.cross_entropy(outputs, soft_labels)
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
                        os.path.join(config.out_dir, "SwinT_best.pth"),
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
        torch.save(model.state_dict(), os.path.join(config.out_dir, "SwinT.pth"))


if __name__ == "__main__":
    main()
