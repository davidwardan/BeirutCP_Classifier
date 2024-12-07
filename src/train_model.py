import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import numpy as np
import random

from src.utils import Processing
from config import Config
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from src.swint_model import SwinTClassifier


# Custom dataset class
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


def main(seed: int = 42):
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Define configuration
    config = Config()

    # Load and preprocess data
    x_train = Processing.load_data(config.in_dir + "/train/x_train.npy")
    y_train = Processing.load_data(config.in_dir + "/train/y_train.npy")

    x_train = Processing.norm_image(x_train)

    # Ensure labels are integers (required for CrossEntropyLoss)
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")

    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
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

    if config.val == 1:
        x_val = Processing.load_data(config.in_dir + "/val/x_val.npy")
        y_val = Processing.load_data(config.in_dir + "/val/y_val.npy")

        if y_val.ndim > 1:
            y_val = np.argmax(y_val, axis=1)

        print(f"x_val shape: {x_val.shape} - y_val shape: {y_val.shape}")

    # Create datasets and loaders
    train_dataset = NumpyDataset(x_train, y_train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val == 1:
        val_dataset = NumpyDataset(x_val, y_val, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)

    # Define optimizer and loss function
    optimizer = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }.get(config.optimizer)

    if optimizer is None:
        raise ValueError("Invalid optimizer")

    optimizer = optimizer(model.parameters(), lr=config.lr_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # For regularization alpha=0.1

    # Training loop with early stopping
    print("Model ready to train...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

        if config.val == 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            print(
                f"Validation Loss: {val_loss:.4f}, Accuracy: {100 * val_correct / val_total:.2f}%"
            )

            # Early stopping logic
            if config.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), config.out_dir + "/SwinT_best.pth")
                    print("Best model saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print("Early stopping triggered.")
                        break
        else:
            # If no validation set is provided, no early stopping based on validation
            torch.save(model.state_dict(), config.out_dir + "/SwinT.pth")

    # If training completes without early stopping or config.val = 0, save final model
    if not config.val or patience_counter < config.patience:
        torch.save(model.state_dict(), config.out_dir + "/SwinT.pth")
        print("Final model saved.")


if __name__ == "__main__":
    main()
