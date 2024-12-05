import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize
import numpy as np
import pandas as pd
import random

from src.utils import processing
from src.SwinT import SwinTransformer
from config import Config


class SwinTClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, transfer_learning=True):
        super(SwinTClassifier, self).__init__()
        if transfer_learning:
            self.swin_transformer = SwinTransformer(
                "swin_base_224", pretrained=True, include_top=False
            )
        else:
            self.swin_transformer = SwinTransformer(
                "swin_base_224", pretrained=False, include_top=False
            )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            1024, 128
        )  # Update input size based on feature dimension from SwinT
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.out(x)
        return x


def main(seed: int = 42):
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Define configuration
    config = Config()

    # Load and preprocess data
    x_train = processing.load_data(config.in_dir + "/train/x_train.npy")
    y_train = processing.load_data(config.in_dir + "/train/y_train.npy")

    x_train = processing.norm_image(x_train)
    y_train = processing.to_categorical(y_train, config.num_classes)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")

    if config.val == 1:
        x_val = processing.load_data(config.in_dir + "/val/x_val.npy")
        y_val = processing.load_data(config.in_dir + "/val/y_val.npy")
        x_val = processing.norm_image(x_val)
        y_val = processing.to_categorical(y_val, config.num_classes)
        print(f"x_val shape: {x_val.shape} - y_val shape: {y_val.shape}")

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val == 1:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

    # Initialize model
    model = SwinTClassifier(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    )
    model.trainable = True

    # Define optimizer and loss function
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr_max)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr_max)
    else:
        raise ValueError("Invalid optimizer")

    criterion = nn.CrossEntropyLoss()

    # Train model
    print("Model ready to train...")
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

        if config.val == 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            print(
                f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * val_correct / val_total:.2f}%"
            )

    # Save model
    torch.save(model.state_dict(), config.out_dir + "/SwinT.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
