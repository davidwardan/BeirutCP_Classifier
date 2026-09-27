"""Vision encoders used by the advanced experiment pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        repository: str = "facebookresearch/dinov2",
    ) -> None:
        super().__init__()
        self.model = torch.hub.load(repository, model_name)
        self.output_dim = int(self.model.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return features


def build_vision_encoder(
    name: str,
    pretrained: bool = True,
    dino_repository: str = "facebookresearch/dinov2",
) -> tuple[nn.Module, int]:
    """Build an encoder and return it together with its feature dimension."""
    normalized_name = name.lower().replace("-", "_")

    if normalized_name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.swin_t(weights=weights)
        output_dim = int(encoder.head.in_features)
        encoder.head = nn.Identity()
        return encoder, output_dim

    if normalized_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.convnext_tiny(weights=weights)
        output_dim = int(encoder.classifier[-1].in_features)
        encoder.classifier[-1] = nn.Identity()
        return encoder, output_dim

    if normalized_name.startswith("dinov2_"):
        encoder = DinoV2Encoder(normalized_name, repository=dino_repository)
        return encoder, encoder.output_dim

    raise ValueError(
        f"Unsupported backbone '{name}'. Choose swin_t, convnext_tiny, or dinov2_vits14."
    )


def set_encoder_trainable(encoder: nn.Module, trainable: bool) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable

