import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import numpy as np
from src.utils import Processing
from config import Config
from src.metrics import metrics
import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from src.swint_model import SwinTClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        # Ensure label is an integer
        label = int(label)

        # Convert image to PIL and apply transformations
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Define configuration
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    logger.info("Loading test data...")
    try:
        x_test = Processing.load_data(config.in_dir + "/test/x_test.npy")
        y_test = Processing.load_data(config.in_dir + "/test/y_test.npy")
        x_test = Processing.norm_image(x_test)

        # If labels are one-hot encoded, convert them to integer indices
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)
        y_test = y_test.astype(np.int64)

    except FileNotFoundError as e:
        logger.error(f"Error loading test data: {e}")
        return

    logger.info(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # Define transforms
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create test dataset and loader
    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = SwinTClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)

    # Load model weights
    model_path = os.path.join(config.out_dir, "SwinT.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}.")
        return
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

    # Optionally, plot confusion matrix
    if config.plot_confusion_matrix:
        cm = metrics.confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=config.class_names
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


if __name__ == "__main__":
    main()
