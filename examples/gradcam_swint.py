"""
Compare Grad-CAM explainability on the same samples for:
  - SwinT (swinT1_kernel10.pth)
  - SwinT (Fused) (Hybrid_kernel10.pth, Swin image branch visualized)

Usage:
    python -m examples.gradcam_swint
"""

import os
import random
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms

from config import Config
from src.swint_model import SwinTClassifier
from src.hybrid_model import HybridSwinTabular
from src.data_preprocessor import DataPreprocessor
from src.utils import Utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class HybridImageOnlyWrapper(nn.Module):
    """Wrap HybridSwinTabular so Grad-CAM can call it with image input only."""

    def __init__(self, hybrid_model: HybridSwinTabular, tabular_vec: torch.Tensor):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.register_buffer("tabular_vec", tabular_vec.detach().clone())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        tab = self.tabular_vec
        if tab.ndim == 1:
            tab = tab.unsqueeze(0)
        if tab.shape[0] != batch_size:
            tab = tab.repeat(batch_size, 1)
        return self.hybrid_model(image, tab)


def sample_one_record_per_label(test_data_dict, num_classes):
    """Return one random full record per class index."""
    buckets = {idx: [] for idx in range(num_classes)}
    for sample_list in test_data_dict.values():
        for sample in sample_list:
            lbl = sample.get("label")
            img = sample.get("image")
            if lbl is not None and img is not None and lbl in buckets:
                buckets[lbl].append(sample)

    sampled = {}
    for class_idx, items in buckets.items():
        if items:
            sampled[class_idx] = random.choice(items)
    return sampled


def get_swin_target_layer(swin_model: nn.Module) -> nn.Module:
    """Pick a late Swin block norm layer for Grad-CAM."""
    try:
        return swin_model.features[-1][-1].norm2
    except Exception:
        return swin_model.features[-1]


def load_swint(weights_path: str, cfg: Config, device: torch.device) -> SwinTClassifier:
    model = SwinTClassifier(
        num_classes=cfg.num_classes,
        transfer_learning=(cfg.transfer_learning == 1),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_hybrid(
    weights_path: str, cfg: Config, tab_dim: int, device: torch.device
) -> HybridSwinTabular:
    model = HybridSwinTabular(
        num_classes=cfg.num_classes,
        tabular_input_dim=tab_dim,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def main(seed: int = 42) -> None:
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(cfg.in_dir, "test_dataset1.pkl")
    if not os.path.exists(data_path):
        logger.error(f"Missing test data file: {data_path}")
        return

    logger.info("Loading test data dictionary...")
    with open(data_path, "rb") as f:
        test_data_dict = pickle.load(f)
    logger.info("Loaded %s test samples.", sum(len(v) for v in test_data_dict.values()))

    preproc = DataPreprocessor(
        categorical_features=["socio_eco"],
        continuous_features=["floors_no"],
    )
    constants_path = os.path.join(cfg.in_dir, "norm_constants.json")
    preproc.load_constants(constants_path)

    first_record = test_data_dict[next(iter(test_data_dict))][0]
    tab_dim = preproc.transform(pd.DataFrame([first_record])).shape[1]

    swint_path = os.path.join(cfg.saved_model_dir, "swinT1_kernel10.pth")
    hybrid_path = os.path.join(cfg.saved_model_dir, "Hybrid_kernel10.pth")

    missing_paths = [p for p in [swint_path, hybrid_path] if not os.path.exists(p)]
    if missing_paths:
        logger.error("Missing model checkpoints:\n%s", "\n".join(missing_paths))
        return

    logger.info("Loading models...")
    swint = load_swint(swint_path, cfg, device)
    hybrid = load_hybrid(hybrid_path, cfg, tab_dim, device)

    swint_layer = get_swin_target_layer(swint.swin_transformer)
    hybrid_layer = get_swin_target_layer(hybrid.swin)

    pred_tfms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    samples = sample_one_record_per_label(test_data_dict, cfg.num_classes)
    if not samples:
        logger.error("Could not sample records by label.")
        return

    out_dir = os.path.join("output", "plots", "gradcam_comparison")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    for class_idx, record in samples.items():
        img_np = np.asarray(record["image"]).astype(np.uint8)
        tab_np = preproc.transform(pd.DataFrame([record]))[0].astype(np.float32)
        tab_tensor = torch.tensor(tab_np, device=device)

        input_tensor = pred_tfms(img_np).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_swint_idx = int(swint(input_tensor).argmax(dim=1).item())
            pred_h_idx = int(hybrid(input_tensor, tab_tensor.unsqueeze(0)).argmax(dim=1).item())

        hybrid_img_wrapper = HybridImageOnlyWrapper(hybrid, tab_tensor).to(device).eval()

        cam_swint = Utils.gradcam_explain_instance(
            swint,
            img_np,
            device,
            target_class=pred_swint_idx,
            target_layer_override=swint_layer,
            eigen_smooth=False,
            aug_smooth=False,
        )
        cam_h = Utils.gradcam_explain_instance(
            hybrid_img_wrapper,
            img_np,
            device,
            target_class=pred_h_idx,
            target_layer_override=hybrid_layer,
            eigen_smooth=False,
            aug_smooth=False,
        )

        true_label = cfg.labels[class_idx] if class_idx < len(cfg.labels) else str(class_idx)
        pred_swint_label = cfg.labels[pred_swint_idx] if pred_swint_idx < len(cfg.labels) else str(pred_swint_idx)
        pred_h_label = cfg.labels[pred_h_idx] if pred_h_idx < len(cfg.labels) else str(pred_h_idx)

        fig, axes = plt.subplots(
            1, 3, figsize=(7.4, 2.8), constrained_layout=True
        )

        axes[0].imshow(img_np)
        axes[0].set_title(f"Input\nTrue: {true_label}")
        axes[0].axis("off")

        axes[1].imshow(cam_swint)
        axes[1].set_title(f"SwinT\nPred: {pred_swint_label}")
        axes[1].axis("off")

        axes[2].imshow(cam_h)
        axes[2].set_title(f"SwinT (Fused)\nPred: {pred_h_label}")
        axes[2].axis("off")

        label_safe = true_label.replace("/", "_").replace(" ", "_")
        png_path = os.path.join(out_dir, f"gradcam_compare_{class_idx}_{label_safe}.png")
        pdf_path = os.path.join(out_dir, f"gradcam_compare_{class_idx}_{label_safe}.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s and %s", png_path, pdf_path)


if __name__ == "__main__":
    main()
