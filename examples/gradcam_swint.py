"""
Run Swin‑T classifier on the test set **and** generate Grad‑CAM heat‑maps
for one random image per class label.

Usage
-----
python -m examples.gradcam_swint
"""

import os
import random
import pickle
import logging

import numpy as np
import pathlib  # for potential future path checks
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from src.data_loader import ImageDataset
from src.swint_model import SwinTClassifier
from src.metrics import metrics
from src.utils import Utils  # contains `gradcam_explain_instance`

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: pick a random image path for each label
# ----------------------------------------------------------------------
def sample_one_image_per_label(test_data_dict, num_classes):
    """
    Build a mapping {class_idx: image_array} containing **one random sample**
    for every class index in ``0 … num_classes-1``.

    The input ``test_data_dict`` is expected to be a mapping like
    {nid: [ {..., "image": <np.ndarray>, "label": int, ...}, ... ], ... }.
    We iterate over every stored sample, bucket them by ``label``, and then
    pick one at random per label.
    """
    buckets = {idx: [] for idx in range(num_classes)}  # class_idx → list[image]
    for sample_list in test_data_dict.values():
        for sample in sample_list:
            lbl = sample.get("label")
            img = sample.get("image")
            if img is not None and lbl is not None and lbl in buckets:
                buckets[lbl].append(img)

    # Select exactly one random image per class (if available)
    samples = {}
    for class_idx, imgs in buckets.items():
        if imgs:
            samples[class_idx] = random.choice(imgs)
    return samples


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load the pickled test‑set description
    # ------------------------------------------------------------------
    logger.info("Loading test data dictionary…")
    try:
        with open(os.path.join(cfg.in_dir, "test_dataset2.pkl"), "rb") as f:
            test_data_dict = pickle.load(f)
    except FileNotFoundError as err:
        logger.error(f"Error loading test data dict: {err}")
        return
    logger.info(f"Loaded {sum(len(v) for v in test_data_dict.values())} test samples.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SwinTClassifier(
        num_classes=cfg.num_classes, transfer_learning=(cfg.transfer_learning == 1)
    ).to(device)

    # Load weights
    best_path = os.path.join(cfg.saved_model_dir, "swint_best.pth")
    fallback_path = os.path.join(cfg.saved_model_dir, "swint.pth")
    model_path = best_path if os.path.exists(best_path) else fallback_path
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------------
    # Grad‑CAM for one random image per label
    # ------------------------------------------------------------------
    logger.info("Generating Grad‑CAM visualisations…")
    samples = sample_one_image_per_label(test_data_dict, cfg.num_classes)
    if not samples:
        logger.warning(
            "No matching label → image pairs were found in the test dictionary; "
            "Grad‑CAM images cannot be generated."
        )
        return
    gradcam_dir = os.path.join(cfg.saved_model_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    # choose target layer for Grad‑CAM
    chosen_layer = model.swin_transformer.norm  # 14×14 map

    for class_idx, img_path in samples.items():
        # `img_path` is already an RGB numpy array straight from the pickle
        img_np = img_path  # keeping variable name consistent with previous logic

        # Run Grad‑CAM
        cam_img = Utils.gradcam_explain_instance(
            model,
            img_np,
            device,
            target_class=class_idx,
            target_layer_override=chosen_layer,
        )

        # Use human‑readable label if available, else the class index
        label_readable = (
            cfg.labels[class_idx]
            if hasattr(cfg, "labels") and len(cfg.labels) > class_idx
            else str(class_idx)
        )
        label_safe = str(label_readable).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(gradcam_dir, f"{label_safe}_gradcam.png")

        # Convert to uint8 if needed and save
        if cam_img.dtype != np.uint8:
            cam_img = cam_img.astype(np.uint8)
        from PIL import Image  # local import to avoid module removal earlier

        Image.fromarray(cam_img).save(out_path)

        plt.figure(figsize=(4, 4))
        plt.imshow(cam_img)
        plt.title(label_readable)
        plt.axis("off")
        plt.show()
        logger.info(f"Saved Grad‑CAM → {out_path}")


if __name__ == "__main__":
    main()
