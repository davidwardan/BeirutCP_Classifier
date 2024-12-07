import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import os
import tqdm
from collections import Counter
import random
import io

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from typing import List, Tuple


class Utils:
    def __init__(self, in_dir: str):
        self.in_dir = in_dir

    @staticmethod
    def lime_explain_instance(model, image, num_samples: int, num_features: int):
        """
        Generate LIME explanation for a given image.
        :param model: Model to explain.
        :param image: Input image as a NumPy array.
        :param num_samples: Number of samples for LIME explanation.
        :param num_features: Number of features to highlight in explanation.
        :return: Marked boundaries image showing explanation.
        """
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.astype("double"),
            model.predict,
            top_labels=3,
            hide_color=0,
            num_samples=num_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=num_features,
            hide_rest=False,
        )
        return mark_boundaries(temp / 2 + 0.5, mask)

    @staticmethod
    def shapley_explain_instance(
        model, image, labels: List[str] = None, evals: int = 5000, top_labels: int = 3
    ):
        """
        Generate SHAP explanation for a given image.
        :param model: Model to explain.
        :param image: Input image as a NumPy array.
        :param labels: List of class labels.
        :param evals: Number of evaluations for SHAP.
        :param top_labels: Number of top labels to explain.
        :return: SHAP explanation as an image.
        """
        masker = shap.maskers.Image("inpaint_ns", image[0].shape)
        explainer = shap.Explainer(model, masker, output_names=labels)
        shap_values = explainer(
            image,
            max_evals=evals,
            batch_size=100,
            outputs=shap.Explanation.argsort.flip[:top_labels],
        )

        plt.figure()
        shap.image_plot(shap_values, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        buf.close()
        return np.array(img)


class Processing:
    @staticmethod
    def load_data(file_path: str) -> np.ndarray:
        """Load data from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return np.load(file_path)

    @staticmethod
    def save_to_pickle(data: List, file_path: str):
        """Save data to a pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_pickle(file_path: str):
        """Load data from a pickle file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def norm_image(x: np.ndarray) -> np.ndarray:
        """Normalize an image to the range [0, 1]."""
        return x / 255.0

    @staticmethod
    def denorm_image(x: np.ndarray) -> np.ndarray:
        """Denormalize an image to the range [0, 255]."""
        return x * 255.0

    @staticmethod
    def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert an array of integers to a one-hot encoded array."""
        return np.eye(num_classes)[y]

    @staticmethod
    def from_categorical(y: np.ndarray) -> np.ndarray:
        """Convert a one-hot encoded array to an array of integers."""
        return np.argmax(y, axis=1)

    @staticmethod
    def zip_images(directory: str, label: int) -> List[Tuple[np.ndarray, int]]:
        """Read images from a directory and return as a list of (image, label) tuples."""
        data = []
        for file_name in tqdm.tqdm(
            os.listdir(directory), desc=f"Processing {directory}"
        ):
            img_path = os.path.join(directory, file_name)
            try:
                img = Image.open(img_path).convert("RGB")
                data.append((np.array(img), label))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return data


class Visualization:
    @staticmethod
    def plot_distribution(train_data, val_data, test_data, class_names):
        """Plot distribution of data after splitting into train, validation, and test sets."""
        train_labels = [label for _, label in train_data]
        val_labels = [label for _, label in val_data]
        test_labels = [label for _, label in test_data]

        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        test_counter = Counter(test_labels)

        labels = sorted(set(train_labels + val_labels + test_labels))
        train_counts = [train_counter[label] for label in labels]
        val_counts = [val_counter[label] for label in labels]
        test_counts = [test_counter[label] for label in labels]

        x = range(len(labels))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x, train_counts, width=width, label="Train", color="blue")
        plt.bar(
            [p + width for p in x],
            val_counts,
            width=width,
            label="Validation",
            color="orange",
        )
        plt.bar(
            [p + width * 2 for p in x],
            test_counts,
            width=width,
            label="Test",
            color="green",
        )

        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Data After Split")
        plt.xticks([p + width for p in x], [class_names[label] for label in labels])
        plt.legend()
        plt.show()

    @staticmethod
    def plot_n_images(images: np.ndarray, n: int, img_per_row: int, save_path: str):
        """Plot n images with a specified number of images per row."""
        rows = (n + img_per_row - 1) // img_per_row
        plt.figure(figsize=(img_per_row * 2, rows * 2))
        for i in range(n):
            plt.subplot(rows, img_per_row, i + 1)
            plt.imshow(images[i])
            plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight")
