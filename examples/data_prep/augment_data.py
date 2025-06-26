import pandas as pd


def main(directory="data", to_augment=["pre1935", "1972-1990"]):

    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/splitted_data.csv")

    # create a new column to indicate is_augmented
    df["is_augmented"] = False

    # Iterate over the periods to augment and duplicate the rows where split type is train and add to the NID _augment suffix
    for period in to_augment:
        mask = (df["label"] == period) & (df["split"] == "train")
        df_augmented = df[mask].copy()
        df_augmented["NID"] = df_augmented["NID"].astype(str) + "_augment"
        df_augmented["is_augmented"] = True
        df = pd.concat([df, df_augmented], ignore_index=True)
    # Save the updated dataframe for inspection
    df.to_csv(f"{directory}/augmented_data.csv", index=False)


if __name__ == "__main__":
    main()
