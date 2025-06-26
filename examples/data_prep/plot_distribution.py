import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
):
    """
    Main function to execute the plotting script.
    """
    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/splitted_data.csv")

    # Dataset membership
    mask_d1 = df["in_dataset_1"]
    mask_d2 = df["in_dataset_2"]

    # Value counts using masks
    vc2 = df[mask_d2]["label"].value_counts()
    vc1 = df[mask_d1]["label"].value_counts()

    # Align counts to your order
    counts1 = [vc1.get(lbl, 0) for lbl in period_order]
    counts2 = [vc2.get(lbl, 0) for lbl in period_order]

    x = np.arange(len(period_order))  # label locations
    width = 0.2  # width of the bars

    fig, ax = plt.subplots(figsize=(6, 4))

    rects1 = ax.bar(x - width / 2, counts1, width, label="Dataset no.1")
    rects2 = ax.bar(x + width / 2, counts2, width, label="Dataset no.2")

    # Add grid, labels, title, and legend
    ax.set_xlabel("Construction Period")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(period_order, rotation=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("output/Figure1.png")
    plt.savefig("output/Figure1.pdf")
    plt.show()


if __name__ == "__main__":
    main()
