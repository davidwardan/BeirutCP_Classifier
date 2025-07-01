import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.entries = [record for records in data_dict.values() for record in records]
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        record = self.entries[idx]
        image = (record["image"]).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = int(record["label"])
        return image, label


class TabularDataset(Dataset):
    def __init__(self, data_dict, preprocessor=None):
        self.entries = []
        for nid, records in data_dict.items():
            for record in records:
                self.entries.append(record)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        record = self.entries[idx]
        df = pd.DataFrame([record])
        features = self.preprocessor.transform(df) if self.preprocessor else df.values
        label = int(record["label"])
        return torch.tensor(features[0], dtype=torch.float32), label


class ImageTabularDataset(Dataset):
    def __init__(self, data_dict, transform=None, preprocessor=None):
        self.entries = []
        for nid, records in data_dict.items():
            for record in records:
                self.entries.append(record)
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        record = self.entries[idx]

        # Get image (already preloaded in memory)
        image = (record["image"]).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get tabular data
        df = pd.DataFrame([record])
        features = self.preprocessor.transform(df) if self.preprocessor else df.values
        label = int(record["label"])

        return image, torch.tensor(features[0], dtype=torch.float32), label
