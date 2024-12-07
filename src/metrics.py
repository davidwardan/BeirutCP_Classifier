import numpy as np
from sklearn.metrics import confusion_matrix


class metrics:

    @staticmethod
    def get_accuracy(output: np.ndarray, true: np.ndarray) -> float:
        """
        Calculate accuracy as the ratio of correct predictions to total samples.

        Args:
            output (np.ndarray): Model predictions (probabilities or logits).
            true (np.ndarray): Ground truth labels (one-hot or categorical).

        Returns:
            float: Accuracy value.
        """
        preds = np.argmax(output, axis=1)
        labels = np.argmax(true, axis=1)
        return np.mean(preds == labels)

    @staticmethod
    def get_confusion_matrix(
        output: np.ndarray, true: np.ndarray, normalized: bool = True
    ) -> np.ndarray:
        """
        Calculate the confusion matrix.

        Args:
            output (np.ndarray): Model predictions (probabilities or logits).
            true (np.ndarray): Ground truth labels (integer encoded).
            normalized (bool): If True, normalize the confusion matrix.

        Returns:
            np.ndarray: Confusion matrix.
        """
        return confusion_matrix(true, output, normalize="true" if normalized else None)

    @staticmethod
    def get_mscore(output: np.ndarray, true: np.ndarray) -> float:
        """
        Calculate the m_score, representing the average absolute difference
        between predictions and true labels.

        Args:
            output (np.ndarray): Model predictions (probabilities).
            true (np.ndarray): Ground truth labels (probabilities).

        Returns:
            float: m_score value.
        """
        return np.mean(np.abs(output - true))

    @staticmethod
    def get_normscore(cm_norm: np.ndarray, num_classes: int) -> float:
        """
        Calculate a normalized m_score based on the confusion matrix.

        Args:
            cm_norm (np.ndarray): Normalized confusion matrix.
            num_classes (int): Number of classes.

        Returns:
            float: Normalized m_score.
        """
        distance_matrix = np.abs(
            np.arange(num_classes)[:, None] - np.arange(num_classes)
        )
        weighted_score = np.multiply(cm_norm, distance_matrix)
        return np.sum(weighted_score) / num_classes

    @staticmethod
    def get_precision(output: np.ndarray, true: np.ndarray) -> float:
        """
        Calculate the average precision score.

        Args:
            output (np.ndarray): Model predictions (probabilities or logits).
            true (np.ndarray): Ground truth labels (one-hot or categorical).

        Returns:
            float: Precision value.
        """
        preds = np.argmax(output, axis=1)
        labels = np.argmax(true, axis=1)
        cm = confusion_matrix(labels, preds)
        precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-12)  # Avoid division by zero
        return np.nanmean(precision)  # Avoid NaN values with np.nanmean

    @staticmethod
    def get_recall(output: np.ndarray, true: np.ndarray) -> float:
        """
        Calculate the average recall score.

        Args:
            output (np.ndarray): Model predictions (probabilities or logits).
            true (np.ndarray): Ground truth labels (one-hot or categorical).

        Returns:
            float: Recall value.
        """
        preds = np.argmax(output, axis=1)
        labels = np.argmax(true, axis=1)
        cm = confusion_matrix(labels, preds)
        recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-12)  # Avoid division by zero
        return np.nanmean(recall)  # Avoid NaN values with np.nanmean
