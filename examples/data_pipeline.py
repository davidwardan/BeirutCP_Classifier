# data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(directory):
    df = pd.read_csv(f"{directory}/all_data.csv")

    # Columns to keep
    columns_to_keep = [
        "NID",
        "Source",
        "floors_no",
        "socio_eco",
        "final_label",
        "IMGAVAL",
    ]

    # Filter the DataFrame to keep only the specified columns
    df = df[columns_to_keep].copy()

    # Set the index to 'NID'
    df.set_index("NID", inplace=True)

    # Replace -999 values with NaN
    df.replace([-999, "-999"], np.nan, inplace=True)

    # Rename columns
    df.rename(columns={"final_label": "label", "IMGAVAL": "has_image"}, inplace=True)

    # Convert to boolean
    df["has_image"] = df["has_image"].astype(bool)

    # Create new column indicating valid tabular data
    df["has_tabular"] = df[["floors_no", "socio_eco"]].notna().all(axis=1)

    # Remove rows with missing labels
    df = df[df["label"].notna()]

    # Normalize socio_eco categories
    df["socio_eco"] = df["socio_eco"].replace(
        "Low-income zone", "Majority low-income zone"
    )

    # Save to intermediate file
    df.to_csv(f"{directory}/prepared_data.csv")


def split_data(directory, splits=[0.7, 0.1, 0.2], labels_col="label"):
    df = pd.read_csv(f"{directory}/prepared_data.csv")

    df["in_dataset_1"] = (df["has_image"]) & (df["has_tabular"])
    df["in_dataset_2"] = df["has_image"]

    df["split"] = None
    random_state = 42

    df_test_candidates = df[df["in_dataset_1"]].copy()
    _, df_test = train_test_split(
        df_test_candidates,
        test_size=splits[2],
        stratify=df_test_candidates[labels_col],
        random_state=random_state,
    )
    df.loc[df_test.index, "split"] = "test"

    df_remaining = df.drop(index=df_test.index)

    df_train, df_val = train_test_split(
        df_remaining,
        test_size=splits[1] / (splits[0] + splits[1]),
        stratify=df_remaining[labels_col],
        random_state=random_state,
    )
    df.loc[df_train.index, "split"] = "train"
    df.loc[df_val.index, "split"] = "val"

    df.to_csv(f"{directory}/splitted_data.csv", index=False)


def augment_data(directory, to_augment=["pre1935", "1972-1990"]):
    df = pd.read_csv(f"{directory}/splitted_data.csv")
    df["is_augmented"] = False

    for period in to_augment:
        mask = (df["label"] == period) & (df["split"] == "train")
        df_augmented = df[mask].copy()
        df_augmented["NID"] = df_augmented["NID"].astype(str) + "_augment"
        df_augmented["is_augmented"] = True
        df = pd.concat([df, df_augmented], ignore_index=True)

    df.to_csv(f"{directory}/augmented_data.csv", index=False)


def undersample_data(directory, to_undersample=["1935-1955", "post1990"], to_move=400):
    df = pd.read_csv(f"{directory}/augmented_data.csv")
    df["is_undersampled"] = False

    for period in to_undersample:
        mask = (df["label"] == period) & (df["in_dataset_1"]) & (df["split"] == "train")
        df_undersampled = df[mask].copy()
        if len(df_undersampled) > to_move:
            df_undersampled = df_undersampled.sample(n=to_move, random_state=42)

        df.loc[df_undersampled.index, "split"] = "test"
        df.loc[df_undersampled.index, "is_undersampled"] = True

    df.to_csv(f"{directory}/final_data.csv", index=False)


def main():
    directory = "data"
    prepare_data(directory)
    split_data(directory)
    augment_data(directory)
    undersample_data(directory)


if __name__ == "__main__":
    main()
