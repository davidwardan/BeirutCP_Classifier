import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl

# set plotting parameters
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "savefig.bbox": "tight",
        # PGF/LaTeX options for PGF export
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
        # LaTeX rendering
        "text.usetex": False,  # Set to True if you want full LaTeX rendering
        # high resolution
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# set plotting style
plt.style.use("seaborn-colorblind")


def main(
    directory="data",
    period_order=["pre1935", "1935-1955", "1956-1971", "1972-1990", "post1990"],
    socio_order=[
        # "Low-income zone",
        "Majority low-income zone",
        "Approximately 50% low-income zone",
        "Minority low-income zone",
        "Not low-income zone",
    ],
    mask="in_dataset_1",
    split="train",
):

    df = pd.read_csv(f"{directory}/splitted_data.csv")

    # 1. Filter your DataFrame as before
    df_filtered = df[df[mask] == True].copy()
    if split is not None:
        df_filtered = df_filtered[df_filtered["split"] == split]
    df_filtered["socio_eco"] = df_filtered["socio_eco"].astype(str)
    df_filtered["label"] = df_filtered["label"].astype(str)

    # 2. Use manual socio-economic class order
    socio_class_order = socio_order
    custom_legend_labels = [
        "Majority low-income & low-income zones (1)",
        "Approximately 50% low-income zone (2)",
        "Minority low-income zone (3)",
        "Not low-income zone (4)",
    ]

    # 3. Generate a color palette automatically
    cmap = mpl.colormaps["tab10"]
    auto_colors = [cmap(i) for i in range(len(socio_class_order))]

    # 4. Build the counts matrix
    counts = []
    for se in socio_class_order:
        row = []
        for period in period_order:
            row.append(
                (
                    (df_filtered["label"] == period) & (df_filtered["socio_eco"] == se)
                ).sum()
            )
        counts.append(row)
    counts = np.array(counts)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(socio_class_order)
    x = np.arange(len(period_order))

    for i, se in enumerate(socio_class_order):
        ax.bar(
            x + i * width, counts[i], width, label=se, color=auto_colors[i], alpha=0.85
        )

    ax.set_xticks(x + width * (len(socio_class_order) - 1) / 2)
    ax.set_xticklabels(period_order, rotation=15)
    ax.set_xlabel("Construction Period")
    ax.set_ylabel("Count")

    # 6. Automatic legend
    handles = [
        Patch(facecolor=auto_colors[i], edgecolor="none", label=custom_legend_labels[i])
        for i in range(len(socio_class_order))
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=2,
        frameon=False,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("output/Figure2_train_final.png")
    plt.savefig("output/Figure2_train_final.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
