"""Tabular-only neural baselines for construction-period prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ordinal_losses import RankConsistentOrdinalHead


class TabularOrdinalClassifier(nn.Module):
    """Small MLP with nominal and rank-consistent ordinal output heads."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            current_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)
        self.ordinal_head = RankConsistentOrdinalHead(current_dim, num_classes)

    def forward(self, tabular: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.features(tabular)
        return {
            "logits": self.classifier(features),
            "ordinal_logits": self.ordinal_head(features),
            "image_logits": None,
            "tabular_logits": None,
            "embedding": features,
        }

