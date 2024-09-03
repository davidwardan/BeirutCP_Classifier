import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import os
import tqdm
from collections import Counter
import random

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap


class utils:
    def __init__(self, in_dir: str):
        self.in_dir = in_dir

    @staticmethod
    def Lime_explain_instance(model, image, num_samples: int, num_features: int):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image.astype('double'), model.predict,
                                                 top_labels=3, hide_color=0, num_samples=num_samples)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,
                                                    num_features=num_features, hide_rest=False)
        return mark_boundaries(temp / 2 + 0.5, mask)

    # TODO: Fix the issue of having to save the plot to a file and then read it back
    # TODO: Fix the issue with getting wrong top labels (the labels are not the same as the ones in the model)
    @staticmethod
    def shapley_explain_instance(model, image, labels: list = None, evals: int = 5000, top_labels: int = 3):
        masker = shap.maskers.Image("inpaint_ns", image[0].shape)  # other masker options: inpaint_telea
        explainer = shap.Explainer(model, masker, output_names=labels)
        shap_values = explainer(image, max_evals=evals, batch_size=100,
                                outputs=shap.Explanation.argsort.flip[:top_labels])
        # return shap.image_plot(shap_values)
        shap.image_plot(shap_values)

        # read the plot to variable
        plt.savefig('shap.png')
        img = plt.imread('shap.png')
        os.remove('shap.png')
        return img


class processing:

    def __init__(self):
        pass

    @staticmethod
    def load_data(file_path: str) -> np.ndarray:
        """
        This function loads data from a file.
        :param file_path: Path to the file.
        :return: Loaded data.
        """
        return np.load(file_path)

    @staticmethod
    def save_to_pickle(data: list, file_path: str):
        """
        This function saves data to a pickle file.
        :param data: Data to save.
        :param file_path: Path to save the pickle file.
        :return: None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_pickle(file_path: str):
        """
        This function loads data from a pickle file.
        :param file_path: Path to the pickle file.
        :return: Loaded data.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def norm_image(x: np.ndarray) -> np.ndarray:
        """
        Normalize an image to the range [0, 1]
        :param x:
        :return:
        """
        return x / 255.0

    @staticmethod
    def denorm_image(x: np.ndarray) -> np.ndarray:
        """
        Denormalize an image to the range [0, 255]
        :param x:
        :return:
        """
        return x * 255.0

    @staticmethod
    def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert an array of integers to a one-hot encoded array
        :param y:
        :param num_classes:
        :return:
        """
        return np.eye(num_classes)[y]

    @staticmethod
    def from_categorical(y: np.ndarray) -> np.ndarray:
        """
        Convert a one-hot encoded array to an array of integers
        :param y:
        :return:
        """
        return np.argmax(y, axis=1)

    @staticmethod
    def zip_images(directory: str, label: int):
        """
        This function reads images from a directory and returns them as a list of tuples (image, label).
        :param directory: The directory containing images
        :param label: The label to assign to all images in the directory
        :return: List of tuples where each tuple is (image, label)
        """
        data = []

        range_dir = os.listdir(directory)
        for file_name in tqdm.tqdm(range_dir):
            img_path = os.path.join(directory, file_name)
            img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
            img_array = np.array(img)  # Convert image to a numpy array
            data.append((img_array, label))  # Append tuple (image, label) to the list

        return data

    @staticmethod
    def unzip_images(data: list, base_directory: str):
        """
        Unzips a list of tuples (image, label) and saves the images into directories named after their labels.
        :param data: List of tuples (image, label)
        :param base_directory: Base directory where the images will be saved
        """
        for i, (img_array, label) in enumerate(data):
            label_directory = os.path.join(base_directory, str(label))

            # Create the label directory if it doesn't exist
            if not os.path.exists(label_directory):
                os.makedirs(label_directory)

            # Define the image path (e.g., "label_directory/image_0.png")
            img_path = os.path.join(label_directory, f"image_{i}.png")

            # Convert the numpy array back to an image and save it
            img = Image.fromarray(img_array)
            img.save(img_path)

    @staticmethod
    def split_data(data, train_size: float, val_size: float, test_size: float, random_state: int = None):
        """
        This function splits the dataset into training, validation, and test sets.

        :param data: List of tuples where each tuple is (image, label)
        :param train_size: Proportion of the dataset to include in the training set
        :param val_size: Proportion of the dataset to include in the validation set
        :param test_size: Proportion of the dataset to include in the test set
        :param random_state: Controls the shuffling applied to the data before applying the split
        :return: Tuple of (train_data, val_data, test_data) where each is a list of (image, label)
        """
        # Ensure the split sizes add up to 1
        assert np.isclose(train_size + val_size + test_size, 1.0), "Split sizes must add up to 1"

        # First split: Train + (Val + Test)
        train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_state)

        # Second split: Val + Test
        val_ratio = val_size / (val_size + test_size)  # Adjust val_size to be relative to the size of temp_data
        val_data, test_data = train_test_split(temp_data, train_size=val_ratio, random_state=random_state)

        return train_data, val_data, test_data

    @staticmethod
    def balance_data(train_data: list, random_state=None):
        """
        This function balances the data by randomly selecting samples from the majority class to match the number of
        samples in the minority class.
        :param train_data: List of tuples (image, label)
        :param random_state: int, random seed for reproducibility
        :return: List of tuples (image, label)
        with balanced classes and a list of tuples (image, label) with the excess data
        """

        # get all labels from the train data
        labels = [sample[1] for sample in train_data]

        # count the number of samples for each label
        counter = Counter(labels)

        # find the label with the fewest samples
        min_samples = min(counter.values())

        # create a list to store the balanced data
        balanced_data = []

        # create a list to store the excess data
        excess_data = []

        # get how many classes we have
        num_classes = len(set(labels))

        # for each class
        for i in range(num_classes):
            # get all the samples for that class
            samples = [sample for sample in train_data if sample[1] == i]

            # shuffle the samples
            random.seed(random_state)

            random.shuffle(samples)

            # add the first min_samples samples to the balanced data
            balanced_data += samples[:min_samples]

            # add the remaining samples to the excess data
            excess_data += samples[min_samples:]

        return balanced_data, excess_data

    @staticmethod
    def mirror_data(data: list):
        """
        This function mirrors the data horizontally.
        :param data: List of tuples (image, label)
        :return: List of tuples (image, label) with mirrored data
        """
        mirrored_data = []
        for image, label in data:
            mirrored_image = np.fliplr(image)
            mirrored_data.append((mirrored_image, label))
        return mirrored_data

    @staticmethod
    def random_crop(image: np.ndarray, crop_size: int):
        """
        This function randomly crops an image.
        :param image: Image to crop
        :param crop_size: Size of the crop (e.g., 24)
        :return: Cropped image
        """
        h, w, _ = image.shape
        x = np.random.randint(0, w - crop_size)
        y = np.random.randint(0, h - crop_size)
        return image[y:y + crop_size, x:x + crop_size]

    @staticmethod
    def shuffle_data(data: list, random_state=None):
        """
        This function shuffles the data.
        :param data: List of tuples (image, label)
        :param random_state: int, random seed for reproducibility
        :return: List of tuples (image, label) with shuffled data
        """
        random.seed(random_state)
        random.shuffle(data)
        return data

    @staticmethod
    def get_images_by_label(data: list, label: int):
        """
        This function returns all the images with a specific label.
        :param data: List of tuples (image, label)
        :param label: The label to search for
        :return: List of images with the specified label
        """
        return [image for image, l in data if l == label]


class visualization:

    def __init__(self):
        pass

    @staticmethod
    def plot_distribution(train_data, val_data, test_data, class_names):
        """
        Plots the distribution of data after splitting into training, validation, and test sets.
        :param train_data: List of tuples (image, label) for the training set
        :param val_data: List of tuples (image, label) for the validation set
        :param test_data: List of tuples (image, label) for the test set
        :param class_names: List of class names corresponding to the labels
        """
        # Count the number of samples for each label in each dataset
        train_labels = [label for _, label in train_data]
        val_labels = [label for _, label in val_data]
        test_labels = [label for _, label in test_data]

        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        test_counter = Counter(test_labels)

        # Prepare the data for plotting
        labels = sorted(set(train_labels + val_labels + test_labels))
        train_counts = [train_counter[label] for label in labels]
        val_counts = [val_counter[label] for label in labels]
        test_counts = [test_counter[label] for label in labels]

        # Plot the distribution
        x = range(len(labels))
        width = 0.25  # Width of the bars

        plt.figure(figsize=(10, 6))

        plt.bar(x, train_counts, width=width, label='Train', color='blue', align='center')
        plt.bar([p + width for p in x], val_counts, width=width, label='Validation', color='orange', align='center')
        plt.bar([p + width * 2 for p in x], test_counts, width=width, label='Test', color='green', align='center')

        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Data After Split')
        plt.xticks([p + width for p in x], [class_names[label] for label in labels])
        plt.legend()
        plt.show()

    # TODO: Set the size of the image in function of n and img_per_row (ensures that the images are not too small)
    @staticmethod
    def plot_n_images(images: np.ndarray, n: int, img_per_row: int, save_path: str):
        plt.figure(figsize=(20, 5))
        for i in range(n):
            plt.subplot(int(n / img_per_row), img_per_row, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
