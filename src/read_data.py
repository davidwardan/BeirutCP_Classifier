import os
import logging
from src.utils import processing, visualization
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the configuration
config = Config()


def main(in_dir: str, out_dir: str):
    """
    Preprocess data for training, validation, and testing.

    Args:
        in_dir (str): Input directory containing raw images.
        out_dir (str): Output directory to save processed data.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    data = []
    logging.info("Loading and zipping data...")
    # Load the data
    for label in config.labels:
        try:
            data += processing.zip_images(
                f"{in_dir}/{label}", config.labels.index(label)
            )
        except Exception as e:
            logging.error(f"Error loading data for label '{label}': {e}")
            continue

    # Split the data into training, validation, and test sets
    logging.info("Splitting data...")
    train_data, val_data, test_data = processing.split_data(data, 0.7, 0.1, 0.2)

    # Plot the distribution of the data
    visualization.plot_distribution(
        train_data, val_data, test_data, list(range(config.num_classes))
    )
    plt.savefig(os.path.join(out_dir, "data_distribution.png"))
    logging.info("Data distribution plot saved.")

    # Augmentation
    logging.info("Augmenting data...")
    try:
        data_pre1935 = processing.get_images_by_label(
            train_data, config.labels.index("pre1935")
        )
        data_1972_1990 = processing.get_images_by_label(
            train_data, config.labels.index("1972-1990")
        )

        data_pre1935_mirrored = processing.mirror_data(data_pre1935)
        data_1972_1990_mirrored = processing.mirror_data(data_1972_1990)
        data_1972_1990_cropped = processing.random_crop(
            data_1972_1990, 0.1 * config.image_size
        )

        # Add the augmented data to the training data
        train_data += (
            data_pre1935_mirrored + data_1972_1990_mirrored + data_1972_1990_cropped
        )
    except Exception as e:
        logging.error(f"Error during data augmentation: {e}")

    # Balance the data
    logging.info("Balancing data...")
    train_data, excess_data = processing.balance_data(train_data)

    # Add the excess data to the test data
    test_data += excess_data

    # Shuffle the data
    logging.info("Shuffling data...")
    train_data = processing.shuffle_data(train_data, config.random_seed)
    val_data = processing.shuffle_data(val_data, config.random_seed)
    test_data = processing.shuffle_data(test_data, config.random_seed)

    # Save the data
    try:
        logging.info("Saving processed data...")
        processing.save_to_pickle(train_data, f"{out_dir}/train_data.pkl")
        processing.save_to_pickle(val_data, f"{out_dir}/val_data.pkl")
        processing.save_to_pickle(test_data, f"{out_dir}/test_data.pkl")
        logging.info("Data saved successfully!")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


if __name__ == "__main__":
    main(in_dir=config.in_dir, out_dir=config.out_dir)
