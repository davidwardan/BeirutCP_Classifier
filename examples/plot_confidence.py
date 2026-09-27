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

import matplotlib as mpl

# set plotting parameters
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "savefig.bbox": "tight",
        # PGF/LaTeX options for PGF export
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
        # LaTeX rendering
        "text.usetex": False,  # Set to True if you want full LaTeX rendering
        # high resolution
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

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
    Build one minimal row figure per sample with columns:
        image | model confidence bars...
    Saves each row as PNG and PDF for manuscript use.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    socio_to_index = {
        "low-income zone": 1,
        "majority low-income zone": 1,
        "approximately 50% low-income zone": 2,
        "approximately 50\\% low-income zone": 2,
        "minority low-income zone": 3,
        "not low-income zone": 4,
    }

    all_samples = list(loader)
    if not all_samples or len(models) == 0:
        return

    n_cols = 1 + len(models)
    os.makedirs("output/plots", exist_ok=True)

    for r, (img, tab, label) in enumerate(all_samples):
        fig, row_axes = plt.subplots(
            nrows=1,
            ncols=n_cols,
            figsize=(2.0 + 2.0 * len(models), 2.3),
            gridspec_kw={"width_ratios": [1.0] + [1.15] * len(models)},
            constrained_layout=True,
        )
        true_idx = label.item()

        original_dataset = loader.dataset.dataset
        record_idx = loader.dataset.indices[r]
        record = original_dataset.entries[record_idx]
        cont_val = record["floors_no"]
        cat_val = record["socio_eco"]
        if isinstance(cat_val, str):
            socio_idx = socio_to_index.get(cat_val.strip().lower(), cat_val)
        else:
            socio_idx = cat_val
        tabular_info = (
            f"Number of floors: {cont_val}\n"
            f"Socio-economic class: {socio_idx}"
        )
        img, tab = img.to(device), tab.to(device)

        # leftmost cell = image
        disp_img = (img.squeeze() * std + mean).clamp(0, 1).permute(1, 2, 0)
        row_axes[0].imshow(disp_img)
        row_axes[0].axis("off")
        row_axes[0].text(
            0.5,
            -0.1,
            tabular_info,
            transform=row_axes[0].transAxes,
            fontsize=7,
            ha="center",
            va="top",
            wrap=True,
        )
        row_axes[0].set_title(f"True: {class_names[true_idx]}", fontsize=8, pad=8)

        # predictions for each model
        for c, (model, mname) in enumerate(zip(models, model_names), start=1):
            if mname == "FNN":
                logits = model(tab)
            elif mname in {"SwinT", "SwinT1", "SwinT (D1)", "SwinT (D2)"}:
                logits = model(img)
            else:
                logits = model(img, tab)

            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred_idx = int(probs.argmax())

            ax = row_axes[c]
            bar_colors = ["#C2C8CF"] * len(class_names)
            # bar_colors[true_idx] = "#2E7D32"
            bar_colors[pred_idx] = "#1F77B4"

            ax.barh(class_names, probs, color=bar_colors, edgecolor="none")
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 1.0])
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            if c > 1:
                ax.set_yticklabels([])
            ax.set_xlabel("Prob.", fontsize=8)
            ax.set_title(mname, fontsize=9, pad=3)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("0.6")
            if pred_idx == true_idx:
                ax.set_facecolor("#D7FECC77")
            else:
                ax.set_facecolor("#FDE4DB7D")

        fig.savefig(f"output/plots/confidence_row_{r:02d}.png", dpi=400, bbox_inches="tight")
        fig.savefig(f"output/plots/confidence_row_{r:02d}.pdf", bbox_inches="tight")
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
            os.path.join(cfg.saved_model_dir, "swinT1_kernel10.pth"), map_location=device
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
            os.path.join(cfg.saved_model_dir, "swinT_kernel10.pth"), map_location=device
        )
    )
    swint_plus.eval()

    fnn = TabularFNN(
        num_classes=cfg.num_classes,
        input_dim=tab_dim,
    ).to(device)
    fnn.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "fnn_kernel10.pth"), map_location=device
        )
    )
    fnn.eval()

    hybrid = HybridSwinTabular(
        num_classes=cfg.num_classes,
        tabular_input_dim=tab_dim,
    ).to(device)
    hybrid.load_state_dict(
        torch.load(
            os.path.join(cfg.saved_model_dir, "Hybrid_kernel10.pth"), map_location=device
        )
    )
    hybrid.eval()

    models = [swint, swint_plus, fnn, hybrid]
    model_names = ["SwinT (D1)", "SwinT (D2)", "FNN", "Hybrid"]

    # ── 6. Pick one random image per class & visualise predictions ───
    sample_ds = get_one_sample_per_label(test_ds, cfg.num_classes)
    sample_dl = DataLoader(sample_ds, batch_size=1, shuffle=False)

    show_sample_predictions(models, model_names, sample_dl, device, cfg.labels)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
