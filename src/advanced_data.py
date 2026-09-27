"""Datasets and transforms for multi-resolution, multi-view experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int, training: bool) -> transforms.Compose:
    """Reproduce the existing augmentation policy at a configurable resolution."""
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
            ]
        )
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


class MultiViewImageTabularDataset(Dataset):
    """Return a fixed-size set of views and optional tabular features per building.

    The input remains compatible with the repository's existing pickle format.
    A record can contain an in-memory ``image``, one ``image_path``, or a list of
    ``image_paths`` produced by ``examples/build_advanced_dataset.py``.
    """

    def __init__(
        self,
        data_dict: dict[str, list[dict[str, Any]]],
        transform: transforms.Compose,
        preprocessor: Any | None = None,
        max_views: int = 1,
        training: bool = False,
    ) -> None:
        if max_views < 1:
            raise ValueError("max_views must be at least 1")
        self.items = list(data_dict.items())
        self.transform = transform
        self.preprocessor = preprocessor
        self.max_views = max_views
        self.training = training
        self.targets = [self._label(records) for _, records in self.items]

    @staticmethod
    def _label(records: list[dict[str, Any]]) -> int:
        labels = {int(record["label"]) for record in records}
        if len(labels) != 1:
            raise ValueError(f"A building contains inconsistent labels: {labels}")
        return labels.pop()

    @staticmethod
    def _sources(records: list[dict[str, Any]]) -> list[Any]:
        sources: list[Any] = []
        for record in records:
            if "image_paths" in record:
                sources.extend(record["image_paths"])
            elif "image_path" in record:
                sources.append(record["image_path"])
            elif "image" in record:
                sources.append(record["image"])
        if not sources:
            raise ValueError("No image source found for a building")
        return sources

    @staticmethod
    def _open_image(source: Any) -> Image.Image:
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        return Image.fromarray(np.asarray(source).astype(np.uint8)).convert("RGB")

    def _choose_sources(self, sources: list[Any]) -> list[Any]:
        if len(sources) <= self.max_views:
            return sources
        if self.training:
            return random.sample(sources, self.max_views)
        indices = np.linspace(0, len(sources) - 1, self.max_views).round().astype(int)
        return [sources[index] for index in indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        building_id, records = self.items[index]
        chosen = self._choose_sources(self._sources(records))
        tensors = [self.transform(self._open_image(source)) for source in chosen]
        mask = torch.ones(self.max_views, dtype=torch.bool)
        while len(tensors) < self.max_views:
            tensors.append(torch.zeros_like(tensors[0]))
            mask[len(tensors) - 1] = False

        first = records[0]
        if self.preprocessor is not None:
            tabular_array = self.preprocessor.transform(pd.DataFrame([first]))[0]
            tabular = torch.tensor(tabular_array, dtype=torch.float32)
            tabular_available = torch.tensor(True)
        else:
            tabular = torch.empty(0, dtype=torch.float32)
            tabular_available = torch.tensor(False)

        return {
            "building_id": str(building_id),
            "images": torch.stack(tensors),
            "view_mask": mask,
            "tabular": tabular,
            "tabular_available": tabular_available,
            "label": torch.tensor(self._label(records), dtype=torch.long),
        }

