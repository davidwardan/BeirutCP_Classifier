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
    mask="in_dataset_2",
):

    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/final_data.csv")
    df_ = df[df[mask] == True].copy()

    mask_original = df_[df_["is_augmented"] == False]
    mask_train = ((df_["split"] == "train") & (df_["is_augmented"] == False)) | (
        (df_["split"] == "test") & (df_["is_undersampled"] == True)
    )
    mask_train_final = df_["split"] == "train"
    mask_valid = df_["split"] == "val"
    mask_test = (df_["split"] == "test") & (df_["is_undersampled"] == False)
    mask_test_final = df_["split"] == "test"

    # Value counts using masks
    vc_original = mask_original["label"].value_counts()
    vc_train = df_[mask_train]["label"].value_counts()
    vc_train_final = df_[mask_train_final]["label"].value_counts()
    vc_valid = df_[mask_valid]["label"].value_counts()
    vc_test = df_[mask_test]["label"].value_counts()
    vc_test_final = df_[mask_test_final]["label"].value_counts()

    # Align counts to your order
    counts_original = [vc_original.get(lbl, 0) for lbl in period_order]
    counts_train = [vc_train.get(lbl, 0) for lbl in period_order]
    counts_train_final = [vc_train_final.get(lbl, 0) for lbl in period_order]
    counts_valid = [vc_valid.get(lbl, 0) for lbl in period_order]
    counts_test = [vc_test.get(lbl, 0) for lbl in period_order]
    counts_test_final = [vc_test_final.get(lbl, 0) for lbl in period_order]

    x = np.arange(len(period_order))  # label locations
    width = 0.15  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab20").colors
    rects0 = ax.bar(
        x - 2 * width, counts_original, width, label="Original", color=colors[7]
    )
    rects1 = ax.bar(x - width, counts_train, width, label="Train", color=colors[1])
    rects5 = ax.bar(x, counts_train_final, width, label="Train Final", color=colors[0])
    rects2 = ax.bar(x + width, counts_valid, width, label="Validation", color=colors[4])
    rects3 = ax.bar(x + 2 * width, counts_test, width, label="Test", color=colors[3])
    rects4 = ax.bar(
        x + 3 * width, counts_test_final, width, label="Test Final", color=colors[2]
    )

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  # Offset height down by changing the y offset to -3
                textcoords="offset points",
                ha="center",
                va="top",  # Change vertical alignment to 'top' since the text is below the bar
                fontsize=8,
                rotation=90,
            )

    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects5)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    # Add grid, labels, title, and legend
    ax.set_xlabel("Construction Period")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 2000)
    ax.set_xticks(x)
    ax.set_xticklabels(period_order, rotation=15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        frameon=False,
        # columnspacing=0.6,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("output/Figure5.png")
    plt.savefig("output/Figure5.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
