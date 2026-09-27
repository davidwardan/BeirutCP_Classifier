import json
import unittest
from pathlib import Path


class ExperimentMatrixTests(unittest.TestCase):
    def test_controlled_matrix(self):
        matrix_path = Path(__file__).parents[1] / "configs" / "advanced_matrix.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        experiments = matrix["experiments"]
        names = [experiment["name"] for experiment in experiments]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(experiments), 14)
        image_experiments = [
            experiment for experiment in experiments if experiment["runner"] == "image"
        ]
        self.assertTrue(all(exp["data"]["image_size"] == 224 for exp in image_experiments))
        self.assertTrue(all(exp["data"]["max_views"] == 1 for exp in image_experiments))
        self.assertIn("swin_gated_ordinal_d1_224", names)
        self.assertIn("ann_ce_tabular_d1", names)
        self.assertIn("ann_ordinal_tabular_d1", names)


if __name__ == "__main__":
    unittest.main()

