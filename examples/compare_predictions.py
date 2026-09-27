"""Paired building-level bootstrap comparison for two model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.advanced_metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame
        if column.startswith("prob_") and column.rsplit("_", 1)[1].isdigit()
    ]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def main() -> None:
    args = parse_args()
    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate)
    reference_probabilities = probability_columns(reference)
    candidate_probabilities = probability_columns(candidate)
    if len(reference_probabilities) != len(candidate_probabilities):
        raise ValueError("Models contain different numbers of probability columns")

    reference = reference[["building_id", "label", *reference_probabilities]].rename(
        columns={column: f"reference_{column}" for column in reference_probabilities}
    )
    candidate = candidate[["building_id", "label", *candidate_probabilities]].rename(
        columns={
            "label": "candidate_label",
            **{column: f"candidate_{column}" for column in candidate_probabilities},
        }
    )
    merged = reference.merge(candidate, on="building_id", validate="one_to_one")
    if not (merged["label"] == merged["candidate_label"]).all():
        raise ValueError("The two files disagree on one or more labels")

    labels = merged["label"].to_numpy(int)
    reference_array = merged[
        [f"reference_{column}" for column in reference_probabilities]
    ].to_numpy(float)
    candidate_array = merged[
        [f"candidate_{column}" for column in candidate_probabilities]
    ].to_numpy(float)
    reference_metrics = compute_metrics(reference_array, labels)
    candidate_metrics = compute_metrics(candidate_array, labels)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "class_index_mae",
        "quadratic_weighted_kappa",
    ]
    rng = np.random.default_rng(args.seed)
    differences = {metric: [] for metric in metric_names}
    for _ in range(args.replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        reference_sample = compute_metrics(reference_array[indices], labels[indices])
        candidate_sample = compute_metrics(candidate_array[indices], labels[indices])
        for metric in metric_names:
            differences[metric].append(
                candidate_sample[metric] - reference_sample[metric]
            )

    comparison = {
        "n_buildings": len(labels),
        "replicates": args.replicates,
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_reference": {},
    }
    for metric in metric_names:
        values = np.asarray(differences[metric])
        comparison["candidate_minus_reference"][metric] = {
            "observed": candidate_metrics[metric] - reference_metrics[metric],
            "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

