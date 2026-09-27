"""Ordinal heads, probability conversion, and losses for construction periods."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encode class ``y`` as the K-1 binary events ``y > k``."""
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def ordinal_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert monotone cumulative logits P(y > k) to K class probabilities."""
    cumulative = torch.sigmoid(logits)
    first = 1.0 - cumulative[:, :1]
    middle = cumulative[:, :-1] - cumulative[:, 1:]
    last = cumulative[:, -1:]
    probabilities = torch.cat((first, middle, last), dim=1)
    return probabilities.clamp_min(0.0) / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def squared_emd_loss(class_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Squared Earth Mover's Distance for ordered, single-label classes."""
    probabilities = class_logits.softmax(dim=1)
    targets = F.one_hot(labels, num_classes=class_logits.shape[1]).float()
    return (probabilities.cumsum(dim=1) - targets.cumsum(dim=1)).pow(2).mean()


def expected_class_index(probabilities: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(
        probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype
    )
    return (probabilities * indices.unsqueeze(0)).sum(dim=1)


class RankConsistentOrdinalHead(nn.Module):
    """A compact cumulative-link head with strictly ordered thresholds.

    The shared scalar score and monotonically increasing thresholds guarantee
    that P(y > k) is non-increasing as k advances through the periods.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.score = nn.Linear(input_dim, 1, bias=False)
        self.first_threshold = nn.Parameter(torch.tensor(-1.0))
        self.raw_increments = nn.Parameter(torch.zeros(num_classes - 2))

    def thresholds(self) -> torch.Tensor:
        if self.raw_increments.numel() == 0:
            return self.first_threshold.unsqueeze(0)
        increments = F.softplus(self.raw_increments) + 1e-4
        remaining = self.first_threshold + torch.cumsum(increments, dim=0)
        return torch.cat((self.first_threshold.unsqueeze(0), remaining))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.thresholds().unsqueeze(0)


class CompositeOrdinalLoss(nn.Module):
    """CE + EMD + cumulative ordinal loss with optional unimodal auxiliaries."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        emd_weight: float = 0.2,
        ordinal_weight: float = 0.2,
        image_aux_weight: float = 0.1,
        tabular_aux_weight: float = 0.05,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.emd_weight = emd_weight
        self.ordinal_weight = ordinal_weight
        self.image_aux_weight = image_aux_weight
        self.tabular_aux_weight = tabular_aux_weight
        self.register_buffer("class_weights", class_weights)

    def forward(
        self, outputs: Mapping[str, torch.Tensor | None], labels: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = outputs.get("logits")
        if logits is None:
            raise KeyError("Model outputs must contain 'logits'")

        parts: Dict[str, torch.Tensor] = {}
        parts["ce"] = F.cross_entropy(logits, labels, weight=self.class_weights)
        parts["emd"] = squared_emd_loss(logits, labels)

        ordinal_logits = outputs.get("ordinal_logits")
        if ordinal_logits is not None:
            parts["ordinal"] = F.binary_cross_entropy_with_logits(
                ordinal_logits, ordinal_targets(labels, self.num_classes)
            )
        else:
            parts["ordinal"] = logits.new_zeros(())

        image_logits = outputs.get("image_logits")
        parts["image_aux"] = (
            F.cross_entropy(image_logits, labels, weight=self.class_weights)
            if image_logits is not None
            else logits.new_zeros(())
        )
        tabular_logits = outputs.get("tabular_logits")
        if tabular_logits is not None:
            tabular_loss = F.cross_entropy(
                tabular_logits,
                labels,
                weight=self.class_weights,
                reduction="none",
            )
            tabular_available = outputs.get("tabular_available")
            if tabular_available is not None:
                mask = tabular_available.bool()
                parts["tabular_aux"] = (
                    tabular_loss[mask].mean() if mask.any() else logits.new_zeros(())
                )
            else:
                parts["tabular_aux"] = tabular_loss.mean()
        else:
            parts["tabular_aux"] = logits.new_zeros(())

        total = (
            self.ce_weight * parts["ce"]
            + self.emd_weight * parts["emd"]
            + self.ordinal_weight * parts["ordinal"]
            + self.image_aux_weight * parts["image_aux"]
            + self.tabular_aux_weight * parts["tabular_aux"]
        )
        parts["total"] = total
        return total, parts
