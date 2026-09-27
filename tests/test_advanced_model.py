import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from src.advanced_models import AdaptiveOrdinalMultimodalClassifier


class FakeEncoder(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        return torch.cat((pooled, pooled, pooled[:, :2]), dim=1)


class AdvancedModelTests(unittest.TestCase):
    @patch("src.advanced_models.build_vision_encoder")
    def test_multiview_gated_forward(self, build_encoder):
        build_encoder.return_value = (FakeEncoder(), 8)
        model = AdaptiveOrdinalMultimodalClassifier(
            num_classes=5,
            tabular_input_dim=5,
            embedding_dim=16,
            fusion="gated",
            pretrained=False,
        )
        outputs = model(
            torch.randn(3, 2, 3, 16, 16),
            tabular=torch.randn(3, 5),
            view_mask=torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool),
        )
        self.assertEqual(outputs["logits"].shape, (3, 5))
        self.assertEqual(outputs["ordinal_logits"].shape, (3, 4))
        self.assertEqual(outputs["view_attention"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()

