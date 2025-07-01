import os
import torch
import torch.nn as nn
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
from src.swint_model import ConvNextClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Define configuration
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data dictionary for ImageDataset
    logger.info("Loading test data dictionary...")
    try:
        with open(os.path.join(config.in_dir, "test_dataset2.pkl"), "rb") as f:
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

    model = ConvNextClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)

    # Load model weights
    weights_dir = config.saved_model_dir + "swint_new.pth"
    model_path = (
        weights_dir
        if os.path.exists(weights_dir)
        else config.saved_model_dir + "swint.pth"
    )
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
