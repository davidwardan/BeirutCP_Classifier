import pandas as pd
from sklearn.model_selection import train_test_split


def main(directory="data", splits=[0.7, 0.1, 0.2], labels_col="label"):
    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/prepared_data.csv")

    # create two columns to indicate dataset membership
    df["in_dataset_1"] = (df["has_image"]) & (df["has_tabular"])
    df["in_dataset_2"] = df["has_image"]

    # create a new column to indicate the split
    df["split"] = None

    # Ensure reproducibility
    random_state = 42

    # Get the unique labels
    labels = df[labels_col].unique()

    # Create a consistent test set from df that has images (dataset 2) and optionally tabular (dataset 1)
    df_test_candidates = df[df["in_dataset_1"]].copy()
    _, df_test = train_test_split(
        df_test_candidates,
        test_size=splits[2],
        stratify=df_test_candidates[labels_col],
        random_state=random_state,
    )

    # Mark the test split
    df.loc[df_test.index, "split"] = "test"

    # Remove test indices from further splitting
    df_remaining = df.drop(index=df_test.index)

    # Split the remaining into train and val
    df_train, df_val = train_test_split(
        df_remaining,
        test_size=splits[1] / (splits[0] + splits[1]),
        stratify=df_remaining[labels_col],
        random_state=random_state,
    )
    df.loc[df_train.index, "split"] = "train"
    df.loc[df_val.index, "split"] = "val"

    # Save the updated dataframe for inspection
    df.to_csv(f"{directory}/splitted_data.csv", index=False)


if __name__ == "__main__":
    main()
