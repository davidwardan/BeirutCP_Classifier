import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
from config import Config
from src.metrics import metrics

import logging
from torch.utils.data import DataLoader
import pickle
from src.data_loader import ImageDataset
from torchvision import transforms
from src.swint_model import SwinTClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def main():
    # Define configuration
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data dictionary for ImageDataset
    logger.info("Loading test data dictionary...")
    try:
        with open(os.path.join(config.in_dir, "test_dataset1.pkl"), "rb") as f:
            test_data_dict = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error loading test data dict: {e}")
        return

    logger.info(f"Loaded {sum(len(v) for v in test_data_dict.values())} test samples.")

    # Define transforms
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset and loader
    test_dataset = ImageDataset(test_data_dict, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = SwinTClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)

    # Load model weights
    weights_dir = config.saved_model_dir + "SwinT1_kernel10.pth"
    model_path = weights_dir
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Set up loss for integer-encoded targets
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    test_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # log classification metrics
    logger.info(
        f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
    )
    report = classification_report(y_true, y_pred, target_names=config.labels)
    print("Classification Report:\n", report)

    # print m-score
    m_score = metrics.get_mscore(np.array(y_pred), np.array(y_true))
    normalized_m_score = metrics.get_normscore(
        metrics.get_confusion_matrix(
            np.array(y_pred), np.array(y_true), normalized=True
        ),
        config.num_classes,
    )
    logger.info(f"m_score: {m_score:.4f}, Normalized m_score: {normalized_m_score:.4f}")

    # log confusion matrix
    cm = metrics.get_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
