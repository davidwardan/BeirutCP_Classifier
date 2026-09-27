import unittest

import torch

from src.ordinal_losses import (
    RankConsistentOrdinalHead,
    ordinal_logits_to_probabilities,
    ordinal_targets,
    squared_emd_loss,
)


class OrdinalLossTests(unittest.TestCase):
    def test_ordinal_targets(self):
        labels = torch.tensor([0, 2, 4])
        encoded = ordinal_targets(labels, 5)
        self.assertTrue(torch.equal(encoded[0], torch.tensor([0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(encoded[1], torch.tensor([1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(encoded[2], torch.tensor([1.0, 1.0, 1.0, 1.0])))

    def test_probabilities_are_valid(self):
        head = RankConsistentOrdinalHead(8, 5)
        logits = head(torch.randn(7, 8))
        probabilities = ordinal_logits_to_probabilities(logits)
        self.assertEqual(probabilities.shape, (7, 5))
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(7)))

    def test_emd_prefers_nearby_error(self):
        labels = torch.tensor([2])
        near = torch.tensor([[-8.0, -8.0, -8.0, 8.0, -8.0]])
        far = torch.tensor([[8.0, -8.0, -8.0, -8.0, -8.0]])
        self.assertLess(squared_emd_loss(near, labels), squared_emd_loss(far, labels))


if __name__ == "__main__":
    unittest.main()

