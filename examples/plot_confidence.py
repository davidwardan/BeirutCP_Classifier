#!/usr/bin/env python
# run_all_models.py
# ----------------------------------------------------------------------
"""
Full evaluation + per-sample visual comparison of
FNN ‖ Swin-T ‖ Hybrid Swin-T + Tabular.
"""
import os, random, pickle, logging
from typing import List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm
import pandas as pd

from config import Config
from src.data_loader import ImageTabularDataset
from src.data_preprocessor import DataPreprocessor
from src.swint_model import SwinTClassifier
from src.fnn_model import TabularFNN
from src.hybrid_model import HybridSwinTabular

# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ───────────────────────── helper functions ──────────────────────────
def get_one_sample_per_label(dataset: ImageTabularDataset, num_labels: int) -> Subset:
    """
    Return a torch Subset with exactly one random item for each label id.
    Assumes `dataset.targets` is a list/ndarray of integer class indices.
    """
    label_to_indices = {i: [] for i in range(num_labels)}
    for idx, tgt in enumerate(dataset.targets):
        label_to_indices[int(tgt)].append(idx)

    chosen = [random.choice(idxs) for idxs in label_to_indices.values()]
    return Subset(dataset, chosen)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    model_type: str,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    """
    Run full-set evaluation.
    `model_type` ∈ {'fnn', 'swint', 'hybrid'} to route inputs correctly.
    Returns (y_true, y_pred) lists for metrics.
    """
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for imgs, tabs, labels in tqdm(loader, desc=f"Testing {model_type.upper()}"):
        imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)

        if model_type == "fnn":
            outputs = model(tabs)
        elif model_type == "swint":
            outputs = model(imgs)
        else:  # hybrid
            outputs = model(imgs, tabs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    logger.info(
        f"[{model_type.upper()}]  Loss: {running_loss / len(loader):.4f} | "
        f"Acc: {100 * correct / total:.2f}%"
    )
    return y_true, y_pred


@torch.no_grad()
def show_sample_predictions(
    models: List[torch.nn.Module],
    model_names: List[str],
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
):
    """
    Build a figure with columns:
        image | FNN bars | SwinT bars | Hybrid bars
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    all_samples = list(loader)
    fig_all, axes_all = zip(
        *[
            plt.subplots(
                nrows=1,
                ncols=1 + len(models),
                figsize=(3 + 2.5 * len(models), 2.5),
                gridspec_kw={"width_ratios": [1] + [1.2] * len(models)},
            )
            for _ in range(len(all_samples))
        ]
    )
    for r, (img, tab, label) in enumerate(all_samples):
        axes = axes_all[r]
        if len(models) == 0:
            continue

        original_dataset = loader.dataset.dataset
        record_idx = loader.dataset.indices[r]
        record = original_dataset.entries[record_idx]
        cont_val = record["floors_no"]
        cat_val = record["socio_eco"]
        tabular_info = [f"{cont_val}", f"{cat_val}"]
        img, tab = img.to(device), tab.to(device)

        # leftmost cell = image
        disp_img = (img.squeeze() * std + mean).clamp(0, 1).permute(1, 2, 0)
        axes[0].imshow(disp_img)
        # axes[0].set_title(f"True: {class_names[label.item()]}", fontsize=10)
        axes[0].axis("off")
        # Display tabular info below the image, allowing wrapping if too long
        tab_str = "\n".join(tabular_info)
        # Use wrap=True and adjust bbox for better visibility if needed
        axes[0].text(
            0.5,
            -0.15,
            tab_str,
            transform=axes[0].transAxes,
            fontsize=10,
            ha="center",
            va="top",
            wrap=True,
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

        # predictions for each model
        for c, (model, mname) in enumerate(zip(models, model_names), start=1):
            if mname == "FNN":
                logits = model(tab)
            elif mname == "SwinT" or mname == "SwinT (dataset2)":
                logits = model(img)
            else:
                logits = model(img, tab)

            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

            ax = axes[c]
            bar_colors = [
                "forestgreen" if i == label.item() else "steelblue"
                for i in range(len(class_names))
            ]
            ax.barh(class_names, probs, color=bar_colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("prob.", fontsize=10)
            if r == 0:
                ax.set_title(f"{mname} Model", fontsize=10)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    import os

    os.makedirs("output/plots", exist_ok=True)
    for i, fig in enumerate(fig_all):
        fig.tight_layout()
        fig.savefig(f"output/plots/confidence_{i}.png", dpi=300)
        plt.close(fig)


# ─────────────────────────────── main ────────────────────────────────
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load test data dictionary ──────────────────────────────────
    pkl_path = os.path.join(
        cfg.in_dir, "test_dataset1.pkl"
    )  # contains images + tabular
    logger.info("Loading test data dictionary …")
    with open(pkl_path, "rb") as f:
        test_data_dict = pickle.load(f)
    logger.info("Loaded %s test samples.", sum(len(v) for v in test_data_dict.values()))

    # ── 2. Pre-processing for tabular part ────────────────────────────
    preproc = DataPreprocessor(
        categorical_features=["socio_eco"],
        continuous_features=["floors_no"],
    )
    preproc.load_constants(os.path.join(cfg.in_dir, "norm_constants.json"))

    # derive tabular dimension
    first_row = pd.DataFrame([test_data_dict[next(iter(test_data_dict))][0]])
    tab_dim = preproc.transform(first_row).shape[1]

    # ── 3. Transforms & Dataset / Loader ──────────────────────────────
    img_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = ImageTabularDataset(
        test_data_dict, transform=img_tfms, preprocessor=preproc
    )
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ── 4. Build & load the models ──────────────────────────────
    swint = SwinTClassifier(
        num_classes=cfg.num_classes,
        transfer_learning=(cfg.transfer_learning == 1),
    ).to(device)
    swint.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "swinT1_best.pth"), map_location=device
        )
    )
    swint.eval()

    # SwinT+ model
    swint_plus = SwinTClassifier(
        num_classes=cfg.num_classes,
        transfer_learning=(cfg.transfer_learning == 1),
    ).to(device)
    swint_plus.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "swinT_best.pth"), map_location=device
        )
    )
    swint_plus.eval()

    fnn = TabularFNN(
        num_classes=cfg.num_classes,
        input_dim=tab_dim,
    ).to(device)
    fnn.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "fnn_best.pth"), map_location=device
        )
    )
    fnn.eval()

    hybrid = HybridSwinTabular(
        num_classes=cfg.num_classes,
        tabular_input_dim=tab_dim,
    ).to(device)
    hybrid.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "Hybrid_best.pth"), map_location=device
        )
    )
    hybrid.eval()

    models = [swint, fnn, hybrid, swint_plus]
    model_names = ["SwinT", "FNN", "Hybrid", "SwinT (dataset2)"]

    # ── 6. Pick one random image per class & visualise predictions ───
    sample_ds = get_one_sample_per_label(test_ds, cfg.num_classes)
    sample_dl = DataLoader(sample_ds, batch_size=1, shuffle=False)

    show_sample_predictions(models, model_names, sample_dl, device, cfg.labels)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
