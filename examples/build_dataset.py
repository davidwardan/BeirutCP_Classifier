import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from config import Config

load_image_as_array_failed_count = 0


def load_image_as_array(path, size=(224, 224)):
    global load_image_as_array_failed_count
    augmnet = False
    path = f"data/images/{path}.png"

    # WEB_RWFDWyQ9oKqWS6GUxdBETz_augment if path has augment suffix
    if path.endswith("_augment.png"):
        path = path[:-12] + ".png"
        augmnet = True
    try:
        if augmnet:
            # If augment is True we apply a vertical flip
            image = Image.open(path).convert("RGB").resize(size)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            image = Image.open(path).convert("RGB").resize(size)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        load_image_as_array_failed_count += 1
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    return np.array(image)


def main(data_csv, dataset_type):

    # read data from CSV
    df = pd.read_csv(data_csv)

    # keep data for the specified dataset type
    if dataset_type == "dataset1":
        df_filtered = df[df["in_dataset_1"] == True]
    elif dataset_type == "dataset2":
        df_filtered = df[df["in_dataset_2"] == True]
    else:
        raise ValueError("Invalid dataset type. Choose 'dataset1' or 'dataset2'.")

    # remove any rows with no label
    df_filtered = df_filtered[df_filtered["label"].notna()]

    # Encode labels
    config = Config()
    custom_order = config.labels
    label_map = {label: idx for idx, label in enumerate(custom_order)}
    df_filtered["label"] = df_filtered["label"].map(label_map)

    # split into dataframes for train, val and test
    splits = [
        df_filtered[df_filtered["split"] == "train"],
        df_filtered[df_filtered["split"] == "val"],
        df_filtered[df_filtered["split"] == "test"],
    ]

    for split in splits:
        # create a dictionnary that has NID as keys with needed information as values
        data_dict = {}
        for _, row in split.iterrows():
            nid = row["NID"]
            if nid not in data_dict:
                data_dict[nid] = []
            data_dict[nid].append(
                {
                    "image": load_image_as_array(nid),  # directly store the image array
                    "floors_no": row["floors_no"],
                    "socio_eco": row["socio_eco"],
                    "label": int(row["label"]),
                }
            )
        split_name = split["split"].iloc[0]
        print(
            f"[{split_name}] Failed to load {load_image_as_array_failed_count} images."
        )

        # pickle the data
        output_file = f"output/{split_name}_{dataset_type}.pkl"
        with open(output_file, "wb") as f:
            import pickle

            pickle.dump(data_dict, f)


if __name__ == "__main__":
    # Example usage
    data_csv = "data/final_data.csv"
    dataset_type = "dataset1"
    main(data_csv, dataset_type)
