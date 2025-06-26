import pandas as pd


def main(directory="data", to_undersample=["1935-1955", "post1990"], to_move=400):
    """
    Main function to execute the undersampling script.
    """
    # Load the prepared dataset
    df = pd.read_csv(f"{directory}/augmented_data.csv")

    # create a new column to indicate is_undersampled
    df["is_undersampled"] = False

    # Iterate over the periods to undersample and move to_move random rows that satisfy the condition in dataset 1 and in dataset 2 split is train and then change split to test
    # and mark them as undersampled

    for period in to_undersample:
        mask = (df["label"] == period) & (df["in_dataset_1"]) & (df["split"] == "train")
        df_undersampled = df[mask].copy()

        # Randomly select rows to move
        if len(df_undersampled) > to_move:
            df_undersampled = df_undersampled.sample(n=to_move, random_state=42)

        # Update the split and mark as undersampled
        df.loc[df_undersampled.index, "split"] = "test"
        df.loc[df_undersampled.index, "is_undersampled"] = True

    # Save the updated dataframe for inspection
    df.to_csv(f"{directory}/final_data.csv", index=False)

if __name__ == "__main__":
    main()