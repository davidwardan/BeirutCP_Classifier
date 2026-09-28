import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from examples.summarize_advanced_results import (
    aggregate_results,
    collect_results,
    plot_confusion_matrices,
    plot_fusion_ablation,
    plot_primary_metrics,
)


class SummarizeAdvancedResultsTests(unittest.TestCase):
    def test_collects_image_and_tabular_metrics_without_duplicate_real_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_test = root / "seed_13" / "swin_gated_ordinal_d1_224" / "test"
            image_test.mkdir(parents=True)
            metrics = {
                "accuracy": 0.7,
                "balanced_accuracy": 0.65,
                "macro_f1": 0.6,
                "class_index_mae": 0.4,
                "quadratic_weighted_kappa": 0.75,
            }
            (image_test / "metrics_real.json").write_text(json.dumps(metrics))
            (image_test / "metrics.json").write_text(json.dumps(metrics))
            pd.DataFrame({"label": [0], "prob_0": [1.0]}).to_csv(
                image_test / "predictions_real.csv", index=False
            )
            tabular = root / "seed_13" / "ann_ce_tabular_d1"
            tabular.mkdir(parents=True)
            (tabular / "metrics.json").write_text(json.dumps(metrics))
            pd.DataFrame({"label": [0], "prob_0": [1.0]}).to_csv(
                tabular / "predictions.csv", index=False
            )

            summary, predictions = collect_results(root)
            aggregate = aggregate_results(summary)

            self.assertEqual(len(summary), 2)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(aggregate), 2)
            self.assertTrue((summary["condition"] == "real").all())

            report = root / "report"
            report.mkdir()
            plot_primary_metrics(aggregate, report)
            plot_fusion_ablation(aggregate, report)
            plot_confusion_matrices(predictions, report)
            self.assertTrue((report / "primary_metrics_comparison.png").exists())
            self.assertTrue((report / "confusion_matrices_real.pdf").exists())


if __name__ == "__main__":
    unittest.main()
