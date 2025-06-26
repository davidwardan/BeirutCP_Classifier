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
    mask="in_dataset_1",
    split=None,
):
    """
    Main function to execute the plotting script.
    """
    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/splitted_data.csv")

    # Filter and prepare the data
    df_filtered = df[df[mask] == True].copy()
    if split is not None:
        df_filtered = df_filtered[df_filtered["split"] == split]
    df_filtered["label"] = df_filtered["label"].astype(str)

    data = [
        df_filtered[df_filtered["label"] == p]["floors_no"].dropna()
        for p in period_order
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, patch_artist=True)

    # Rotate x-axis labels
    ax.set_xticklabels(period_order)
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.set_ylim(0, 55)
    ax.set_xlabel("Construction Period")
    ax.set_ylabel("Number of Floors")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("output/Figure3.png")
    plt.savefig("output/Figure3.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
