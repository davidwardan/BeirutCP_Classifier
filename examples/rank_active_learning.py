"""Rank unlabeled buildings by uncertainty and multimodal disagreement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entropy-weight", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--disagreement-weight", type=float, default=0.25)
    return parser.parse_args()


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -(probabilities * np.log(probabilities)).sum(axis=1) / np.log(
        probabilities.shape[1]
    )


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(left, 1e-12, 1.0)
    right = np.clip(right, 1e-12, 1.0)
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        (left * np.log(left / midpoint)).sum(axis=1)
        + (right * np.log(right / midpoint)).sum(axis=1)
    ) / np.log(2.0)


def columns_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    probability_columns = columns_with_prefix(frame, "prob_")
    probabilities = frame[probability_columns].to_numpy(float)
    entropy = normalized_entropy(probabilities)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin_uncertainty = 1.0 - (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )

    image_columns = columns_with_prefix(frame, "image_prob_")
    tabular_columns = columns_with_prefix(frame, "tabular_prob_")
    if image_columns and tabular_columns:
        disagreement = jensen_shannon(
            frame[image_columns].to_numpy(float),
            frame[tabular_columns].to_numpy(float),
        )
    else:
        disagreement = np.zeros(len(frame))

    frame["entropy_uncertainty"] = entropy
    frame["margin_uncertainty"] = margin_uncertainty
    frame["modality_disagreement"] = disagreement
    frame["active_learning_score"] = (
        args.entropy_weight * entropy
        + args.margin_weight * margin_uncertainty
        + args.disagreement_weight * disagreement
    )
    frame["predicted_class"] = probabilities.argmax(axis=1)
    frame = frame.sort_values("active_learning_score", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} ranked buildings to {args.output}")


if __name__ == "__main__":
    main()

