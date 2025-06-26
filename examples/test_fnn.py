# examples/test_fnn.py
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import logging

from config import Config
from src.data_preprocessor import DataPreprocessor
from src.data_loader import TabularDataset
from src.fnn_model import TabularFNN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

from tqdm import tqdm
from src.metrics import metrics


def main():
    # Define configuration
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data dictionary for ImageTabularDataset
    logger.info("Loading test data dictionary...")
    try:
        with open(os.path.join(config.in_dir, "test_dataset1.pkl"), "rb") as f:
            test_data_dict = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error loading test data dict: {e}")
        return

    logger.info(f"Loaded {sum(len(v) for v in test_data_dict.values())} test samples.")

    # Load preprocessor
    constants_path = "output/norm_constants_tabular.json"
    preprocessor = DataPreprocessor(
        categorical_features=["socio_eco"],
        continuous_features=["floors_no"],
    )
    preprocessor.load_constants(constants_path)
    logger.info("Loaded preprocessing constants from %s", constants_path)

    # Create test dataset and loader
    test_dataset = TabularDataset(test_data_dict, preprocessor=preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Derive tabular-feature dimension on-the-fly
    sample_tab = preprocessor.transform(
        pd.DataFrame([test_data_dict[next(iter(test_data_dict))][0]])
    )
    tab_dim = sample_tab.shape[1]

    # Build FNN model
    model = TabularFNN(input_dim=tab_dim, num_classes=config.num_classes).to(device)

    # Load model weights
    model_path = (
        "weights/FNN_best.pth"
        if os.path.exists("weights/FNN_best.pth")
        else "weights/FNN.pth"
    )
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Restored weights from {model_path}")

    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for tabs, labels in tqdm(test_loader, desc="Testing"):
            tabs, labels = tabs.to(device), labels.to(device)
            outputs = model(tabs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    logger.info(
        f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
    )
    report = classification_report(y_true, y_pred, target_names=config.labels)
    print("Classification Report:\n", report)

    cm = metrics.get_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
