from utils import processing, visualization
from config import Config

# Define the configuration
config = Config()


def main(in_dir: str, out_dir: str):
    data = []
    # Load the data
    for label in config.labels:
        data += processing.zip_images(f'{in_dir}/{label}', config.labels.index(label))

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = processing.split_data(data, 0.7, 0.1, 0.2)

    # Plot the distribution of the data
    visualization.plot_distribution(train_data, val_data, test_data, list(range(config.num_classes)))

    # Get all data that needs to be augmented
    data_pre1935 = processing.get_images_by_label(train_data, config.labels.index('pre1935'))
    data_1972_1990 = processing.get_images_by_label(train_data, config.labels.index('1972-1990'))

    # Augment the data
    data_pre1935_mirrored = processing.mirror_data(data_pre1935)
    data_1972_1990_mirrored = processing.mirror_data(data_1972_1990)
    data_1972_1990_cropped = processing.random_crop(data_1972_1990, 0.1 * config.image_size)  # 10% crop

    # Add the augmented data to the training data
    train_data += data_pre1935_mirrored
    train_data += data_1972_1990_mirrored
    train_data += data_1972_1990_cropped

    # Balance the data
    train_data, excess_data = processing.balance_data(train_data)

    # Add the excess data to the test data
    test_data += excess_data

    # shuffle the data
    train_data = processing.shuffle_data(train_data, 42)
    val_data = processing.shuffle_data(val_data, 42)
    test_data = processing.shuffle_data(test_data, 42)

    # Save the data
    processing.save_to_pickle(train_data, f'{out_dir}/train_data.pkl')
    processing.save_to_pickle(val_data, f'{out_dir}/val_data.pkl')
    processing.save_to_pickle(test_data, f'{out_dir}/test_data.pkl')

    print('Data saved successfully!')


if __name__ == "__main__":
    main(in_dir=config.in_dir, out_dir=config.out_dir)
