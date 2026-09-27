"""Façade-aware ordinal image and image-tabular classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ordinal_losses import RankConsistentOrdinalHead
from src.vision_backbones import build_vision_encoder


class AttentionViewPool(nn.Module):
    """Aggregate one or more views into a building representation."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, max(32, feature_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(32, feature_dim // 4), 1),
        )

    def forward(
        self, features: torch.Tensor, view_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(features).squeeze(-1)
        if view_mask is not None:
            logits = logits.masked_fill(~view_mask.bool(), torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=1)
        pooled = (features * attention.unsqueeze(-1)).sum(dim=1)
        return pooled, attention


class AdaptiveOrdinalMultimodalClassifier(nn.Module):
    """Multi-view image encoder with adaptive tabular conditioning.

    ``fusion`` may be ``gated``, ``film``, or ``concat``. The forward method
    always returns a dictionary so the same trainer can apply fused, ordinal,
    and unimodal auxiliary losses.
    """

    def __init__(
        self,
        num_classes: int,
        tabular_input_dim: int = 0,
        backbone: str = "swin_t",
        embedding_dim: int = 256,
        fusion: str = "gated",
        tabular_modality_dropout: float = 0.15,
        pretrained: bool = True,
        dino_repository: str = "facebookresearch/dinov2",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tabular_input_dim = tabular_input_dim
        self.fusion = fusion
        self.tabular_modality_dropout = tabular_modality_dropout

        self.encoder, encoder_dim = build_vision_encoder(
            backbone, pretrained=pretrained, dino_repository=dino_repository
        )
        self.view_pool = AttentionViewPool(encoder_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.tabular_projection: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.film: nn.Module | None = None
        self.concat_fusion: nn.Module | None = None
        self.tabular_head: nn.Module | None = None

        if tabular_input_dim > 0:
            self.tabular_projection = nn.Sequential(
                nn.Linear(tabular_input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )
            if fusion == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()
                )
            elif fusion == "film":
                self.film = nn.Linear(embedding_dim, embedding_dim * 2)
            elif fusion == "concat":
                self.concat_fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                )
            else:
                raise ValueError("fusion must be one of: gated, film, concat")
            self.tabular_head = nn.Linear(embedding_dim, num_classes)

        self.fused_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.image_head = nn.Linear(embedding_dim, num_classes)
        self.ordinal_head = RankConsistentOrdinalHead(embedding_dim, num_classes)

    def _encode_views(
        self, images: torch.Tensor, view_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("images must have shape [B,C,H,W] or [B,V,C,H,W]")
        batch, views, channels, height, width = images.shape
        encoded = self.encoder(images.reshape(batch * views, channels, height, width))
        encoded = encoded.reshape(batch, views, -1)
        return self.view_pool(encoded, view_mask)

    def _apply_tabular_dropout(
        self, features: torch.Tensor, available: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        if available is None:
            available = torch.ones(batch, device=features.device, dtype=torch.bool)
        else:
            available = available.bool().reshape(batch)
        if self.training and self.tabular_modality_dropout > 0:
            retained = torch.rand(batch, device=features.device) >= self.tabular_modality_dropout
            available = available & retained
        return features * available.unsqueeze(1), available

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        tabular_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        raw_image, view_attention = self._encode_views(images, view_mask)
        image_features = self.image_projection(raw_image)
        fused = image_features
        tabular_features = None
        effective_tabular_mask = None

        if self.tabular_projection is not None and tabular is not None:
            tabular_features = self.tabular_projection(tabular)
            tabular_features, effective_tabular_mask = self._apply_tabular_dropout(
                tabular_features, tabular_available
            )
            if self.fusion == "gated":
                gate = self.gate(torch.cat((image_features, tabular_features), dim=1))
                fused = image_features + gate * tabular_features
            elif self.fusion == "film":
                gamma, beta = self.film(tabular_features).chunk(2, dim=1)
                fused = (1.0 + 0.1 * torch.tanh(gamma)) * image_features + beta
            else:
                fused = self.concat_fusion(
                    torch.cat((image_features, tabular_features), dim=1)
                )

        fused = self.fused_norm(fused)
        return {
            "logits": self.classifier(fused),
            "ordinal_logits": self.ordinal_head(fused),
            "image_logits": self.image_head(image_features),
            "tabular_logits": (
                self.tabular_head(tabular_features)
                if self.tabular_head is not None and tabular_features is not None
                else None
            ),
            "embedding": fused,
            "view_attention": view_attention,
            "tabular_available": effective_tabular_mask,
        }

