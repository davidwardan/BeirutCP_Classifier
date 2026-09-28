"""Aggregate advanced experiment metrics and create comparison figures."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import Config


PRIMARY_METRICS = [
    ("macro_f1", "Macro F1", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("class_index_mae", "Class-index MAE", False),
    ("quadratic_weighted_kappa", "Quadratic weighted kappa", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/advanced_matrix")
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Defaults to <output-root>/report.",
    )
    return parser.parse_args()


def display_name(name: str) -> str:
    replacements = {
        "swin_ce_image_d1_224": "Swin CE image (D1)",
        "swin_ordinal_image_d1_224": "Swin ordinal image (D1)",
        "swin_concat_ordinal_d1_224": "Swin concat (D1)",
        "swin_gated_ordinal_d1_224": "Swin gated (D1)",
        "swin_ordinal_image_d2_224": "Swin ordinal image (D2)",
        "ann_ce_tabular_d1": "ANN CE tabular (D1)",
        "ann_ordinal_tabular_d1": "ANN ordinal tabular (D1)",
        "logistic_regression_tabular_d1": "Logistic regression (D1)",
        "random_forest_tabular_d1": "Random forest (D1)",
    }
    return replacements.get(name, name.replace("_", " "))


def _seed_from_directory(path: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid seed directory: {path}")
    return int(match.group(1))


def collect_results(output_root: Path) -> tuple[pd.DataFrame, dict[tuple, Path]]:
    rows: list[dict] = []
    prediction_paths: dict[tuple[int, str, str], Path] = {}
    for seed_directory in sorted(output_root.glob("seed_*")):
        if not seed_directory.is_dir():
            continue
        seed = _seed_from_directory(seed_directory)
        for experiment_directory in sorted(seed_directory.iterdir()):
            if not experiment_directory.is_dir():
                continue
            experiment = experiment_directory.name
            conditioned_metrics = sorted(
                (experiment_directory / "test").glob("metrics_*.json")
            )
            metric_sources: list[tuple[str, Path, Path]] = []
            if conditioned_metrics:
                for metrics_path in conditioned_metrics:
                    condition = metrics_path.stem.removeprefix("metrics_")
                    predictions_path = metrics_path.with_name(
                        f"predictions_{condition}.csv"
                    )
                    metric_sources.append((condition, metrics_path, predictions_path))
            else:
                metrics_path = experiment_directory / "metrics.json"
                predictions_path = experiment_directory / "predictions.csv"
                if metrics_path.exists():
                    metric_sources.append(("real", metrics_path, predictions_path))

            for condition, metrics_path, predictions_path in metric_sources:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        "display_name": display_name(experiment),
                        "condition": condition,
                        **metrics,
                    }
                )
                if predictions_path.exists():
                    prediction_paths[(seed, experiment, condition)] = predictions_path

    if not rows:
        raise FileNotFoundError(
            f"No metrics.json or test/metrics_*.json files found under {output_root}"
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment", "condition", "seed"]
    ), prediction_paths


def aggregate_results(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in summary.columns
        if column not in {"seed", "experiment", "display_name", "condition"}
        and pd.api.types.is_numeric_dtype(summary[column])
    ]
    aggregate = summary.groupby(
        ["experiment", "display_name", "condition"], sort=True
    )[metric_columns].agg(["mean", "std", "count"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def _save_figure(figure: plt.Figure, report_dir: Path, stem: str) -> None:
    figure.savefig(report_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(report_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_primary_metrics(aggregate: pd.DataFrame, report_dir: Path) -> None:
    real = aggregate[aggregate["condition"] == "real"].copy()
    if real.empty:
        return
    names = real["display_name"].tolist()
    positions = np.arange(len(real))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (metric, title, higher_is_better) in zip(axes.flat, PRIMARY_METRICS):
        means = real[f"{metric}_mean"].to_numpy(float)
        errors = real[f"{metric}_std"].fillna(0.0).to_numpy(float)
        colors = ["#2f6f9f"] * len(real)
        if len(real):
            best = int(np.argmax(means) if higher_is_better else np.argmin(means))
            colors[best] = "#d97706"
        axis.bar(positions, means, yerr=errors, capsize=3, color=colors)
        axis.set_title(f"{title} (mean ± SD across seeds)")
        axis.set_xticks(positions, names, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, report_dir, "primary_metrics_comparison")


def plot_fusion_ablation(aggregate: pd.DataFrame, report_dir: Path) -> None:
    selected = aggregate[
        aggregate["experiment"].str.contains("swin_(?:concat|gated)", regex=True)
    ].copy()
    if selected.empty or selected["condition"].nunique() < 2:
        return
    experiments = selected["experiment"].drop_duplicates().tolist()
    conditions = [
        condition
        for condition in ("real", "masked", "shuffled")
        if condition in set(selected["condition"])
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    width = 0.8 / max(1, len(conditions))
    positions = np.arange(len(experiments))
    for condition_index, condition in enumerate(conditions):
        offsets = positions - 0.4 + width / 2 + condition_index * width
        for axis, metric, title in (
            (axes[0], "macro_f1", "Macro F1"),
            (axes[1], "class_index_mae", "Class-index MAE"),
        ):
            values = []
            errors = []
            for experiment in experiments:
                row = selected[
                    (selected["experiment"] == experiment)
                    & (selected["condition"] == condition)
                ]
                values.append(float(row[f"{metric}_mean"].iloc[0]))
                errors.append(float(row[f"{metric}_std"].fillna(0.0).iloc[0]))
            axis.bar(
                offsets,
                values,
                width=width,
                yerr=errors,
                capsize=3,
                label=condition,
            )
            axis.set_title(f"{title}: tabular ablation")
            axis.set_xticks(
                positions,
                [display_name(name) for name in experiments],
                rotation=25,
                ha="right",
            )
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(title="Condition")
    _save_figure(figure, report_dir, "fusion_ablation_comparison")


def plot_confusion_matrices(
    prediction_paths: dict[tuple[int, str, str], Path], report_dir: Path
) -> None:
    experiments = sorted(
        {experiment for _, experiment, condition in prediction_paths if condition == "real"}
    )
    if not experiments:
        return
    columns = min(3, len(experiments))
    rows = math.ceil(len(experiments) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.8 * columns, 4.2 * rows), squeeze=False
    )
    for axis, experiment in zip(axes.flat, experiments):
        labels_all = []
        predictions_all = []
        for key, predictions_path in prediction_paths.items():
            _, candidate, condition = key
            if candidate != experiment or condition != "real":
                continue
            frame = pd.read_csv(predictions_path)
            probability_columns = sorted(
                (column for column in frame if re.fullmatch(r"prob_\d+", column)),
                key=lambda column: int(column.split("_")[1]),
            )
            labels_all.extend(frame["label"].astype(int).tolist())
            predictions_all.extend(
                frame[probability_columns].to_numpy().argmax(axis=1).tolist()
            )
        matrix = confusion_matrix(
            labels_all,
            predictions_all,
            labels=np.arange(len(Config.labels)),
            normalize="true",
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else "black",
                )
        axis.set_title(display_name(experiment), fontsize=10)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(
            np.arange(len(Config.labels)),
            Config.labels,
            rotation=40,
            ha="right",
            fontsize=7,
        )
        axis.set_yticks(np.arange(len(Config.labels)), Config.labels, fontsize=7)
    for axis in axes.flat[len(experiments) :]:
        axis.axis("off")
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.55,
        label="Row-normalized proportion",
    )
    figure.suptitle("Test confusion matrices pooled across seeds", fontsize=14)
    _save_figure(figure, report_dir, "confusion_matrices_real")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or args.output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary, prediction_paths = collect_results(args.output_root)
    aggregate = aggregate_results(summary)
    summary.to_csv(report_dir / "metrics_by_seed.csv", index=False)
    aggregate.to_csv(report_dir / "metrics_mean_std.csv", index=False)
    plot_primary_metrics(aggregate, report_dir)
    plot_fusion_ablation(aggregate, report_dir)
    plot_confusion_matrices(prediction_paths, report_dir)
    print(f"Wrote comparative report to {report_dir}")


if __name__ == "__main__":
    main()
