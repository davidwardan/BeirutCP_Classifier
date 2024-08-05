import numpy as np
from sklearn.metrics import confusion_matrix


class metrics:

    @staticmethod
    # calculate accuracy
    def get_accuracy(output: np.ndarray, true: np.ndarray) -> float:
        array1 = np.array(output)
        array2 = np.array(true)

        array3 = np.argmax(array1, axis=1)
        array4 = np.argmax(array2, axis=1)

        accuracy = np.sum(array3 == array4) / len(array3)

        return accuracy

    @staticmethod
    # get confusion matrix
    def get_confusion_matrix(output: np.ndarray, true: np.ndarray, normalized: bool = True) -> np.ndarray:
        array1 = np.array(output)
        array2 = np.array(true)

        array3 = np.argmax(array1, axis=1)
        array4 = np.argmax(array2, axis=1)

        if normalized:
            cm = confusion_matrix(array4, array3, normalize='true')
        else:
            cm = confusion_matrix(array4, array3, normalize=None)

        return cm

    @staticmethod
    # calculate m_score
    def get_mscore(output: np.ndarray, true: np.ndarray) -> float:
        array1 = np.array(output)
        array2 = np.array(true)

        array3 = np.absolute(np.subtract(array1, array2))
        m_score = np.sum(array3) / len(array3)

        return m_score

    @staticmethod
    # calculate normalized m_score
    def get_normscore(cm_norm: np.ndarray, num_classes: int) -> float:
        matrix = np.abs(np.arange(num_classes)[:, None] - np.arange(num_classes))

        score = np.multiply(cm_norm, matrix)
        norm_m_score = np.sum(score) / 5
        return norm_m_score

    @staticmethod
    # calculate precision
    def get_precision(output: np.ndarray, true: np.ndarray) -> float:
        array1 = np.array(output)
        array2 = np.array(true)

        array3 = np.argmax(array1, axis=1)
        array4 = np.argmax(array2, axis=1)

        cm = confusion_matrix(array4, array3)
        precision = np.diag(cm) / np.sum(cm, axis=0)
        return np.average(precision)

    @staticmethod
    # calculate recall value
    def get_recall(output: np.ndarray, true: np.ndarray) -> float:
        array1 = np.array(output)
        array2 = np.array(true)

        array3 = np.argmax(array1, axis=1)
        array4 = np.argmax(array2, axis=1)

        cm = confusion_matrix(array4, array3)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        return np.average(recall)

