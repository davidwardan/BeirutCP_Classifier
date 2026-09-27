import unittest

import torch

from src.ordinal_losses import CompositeOrdinalLoss
from src.tabular_models import TabularOrdinalClassifier


class TabularModelTests(unittest.TestCase):
    def test_nominal_and_ordinal_heads_train_together(self):
        model = TabularOrdinalClassifier(input_dim=5, num_classes=5)
        model.train()
        outputs = model(torch.randn(8, 5))
        self.assertEqual(outputs["logits"].shape, (8, 5))
        self.assertEqual(outputs["ordinal_logits"].shape, (8, 4))
        criterion = CompositeOrdinalLoss(
            num_classes=5,
            ce_weight=1.0,
            emd_weight=0.2,
            ordinal_weight=0.2,
            image_aux_weight=0.0,
            tabular_aux_weight=0.0,
        )
        loss, parts = criterion(outputs, torch.arange(8) % 5)
        loss.backward()
        self.assertGreater(float(parts["total"]), 0.0)


if __name__ == "__main__":
    unittest.main()

