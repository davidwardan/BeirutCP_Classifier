import unittest

import pandas as pd

from src.splitting import assign_splits


class SplittingTests(unittest.TestCase):
    def test_groups_do_not_cross_splits(self):
        rows = []
        for label in range(5):
            for building in range(30):
                for view in range(2):
                    rows.append(
                        {
                            "building": f"{label}-{building}",
                            "label": str(label),
                            "in_dataset_1": True,
                            "in_dataset_2": True,
                            "view": view,
                        }
                    )
        assigned, summary = assign_splits(
            pd.DataFrame(rows), seed=13, group_column="building"
        )
        per_building = assigned.groupby("building")["split"].nunique()
        self.assertEqual(int(per_building.max()), 1)
        self.assertGreater(summary.test_groups, 0)
        self.assertGreater(summary.validation_groups, 0)

    def test_dataset2_only_rows_in_test_spatial_blocks_are_excluded(self):
        rows = []
        for block in range(25):
            for label in range(5):
                rows.append(
                    {
                        "building": f"common-{block}-{label}",
                        "block": f"block-{block}",
                        "label": str(label),
                        "in_dataset_1": True,
                        "in_dataset_2": True,
                    }
                )
                rows.append(
                    {
                        "building": f"extra-{block}-{label}",
                        "block": f"block-{block}",
                        "label": str(label),
                        "in_dataset_1": False,
                        "in_dataset_2": True,
                    }
                )
        assigned, _ = assign_splits(
            pd.DataFrame(rows), seed=13, group_column="block"
        )
        test_blocks = set(assigned.loc[assigned["split"] == "test", "block"])
        extras = assigned[
            assigned["block"].isin(test_blocks) & ~assigned["in_dataset_1"]
        ]
        self.assertTrue((extras["split"] == "heldout_extra").all())


if __name__ == "__main__":
    unittest.main()
